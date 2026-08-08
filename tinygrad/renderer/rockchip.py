from __future__ import annotations
# ruff: noqa: E702
import base64, math, os, struct
import numpy as np
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Iterable
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import Target, cdiv, cmod, floordiv, floormod
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 26
_HEADER = struct.Struct("<4sHHHHIIIIII")  # magic/version/target, scratch/gather counts, ops/constants, phase split, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBBBBiIIi"), struct.Struct("<IIi")
_FILL = struct.Struct("<BBHI")  # dst_kind, itemsize, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, flags, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg

class RKTarget(IntEnum): RK3588 = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True)
class RKScratch: size: int; alignment: int = 4096

@dataclass(frozen=True)
class RKGather:
  """Materialize an affine or fallback raw-lane index map."""
  src_index: int; dst_index: int; count: int; base: int = 0
  axes: tuple[tuple[int, int, int], ...] = ()  # dst divisor, range limit, source stride
  offsets: tuple[int, ...] = ()
  fill_bits: int = 0
  values: tuple[int, ...] = ()  # compile-time FP16 vector, no source argument
  partial: bool = False  # preserve lanes populated by another gather when the offset is negative
  dst_stride: int = 1  # destination stride in lanes; scalar FP16 reductions use 32 for 64-byte spacing
  dst_addend: int = 0  # destination offset in lanes
  dst_kind: RKBufferKind = RKBufferKind.SCRATCH
  itemsize: int = 2
  src_kind: RKBufferKind = RKBufferKind.ARG

@dataclass(frozen=True)
class RKMultiGather: gathers: tuple[RKGather, ...]

@dataclass(frozen=True)
class RKStatic: expr: UOp

RKLeaf = RKArg|RKStatic|RKGather|RKMultiGather|float|tuple[UOp, UOp, UOp|None, int]|None

@dataclass(frozen=True)
class RKFill: dst: RKArg; count: int; itemsize: int = 2

@dataclass(frozen=True)
class RKEWOp:
  """One contiguous FP16 DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); fill: RKFill|None = None; ew_ops: tuple[RKEWOp, ...] = ()
  mid_gathers: tuple[RKGather, ...] = (); gather_after: int = 0
  post_gathers: tuple[RKGather, ...] = ()

@dataclass(frozen=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.mid_gathers + image.post_gathers
  if image.mid_gathers and not 0 < image.gather_after < len(image.ew_ops): raise ValueError("invalid mid-gather split")
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(gathers),
                               len(image.ew_ops), len(image.constants), len(image.mid_gathers), len(image.post_gathers),
                               image.gather_after, int(image.fill is not None)))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in gathers:
    kind = 3 if g.partial else 2 if g.values else 1 if g.offsets else 0
    out += _GATHER.pack(g.dst_index, g.src_index, g.count, kind, len(g.axes), g.itemsize, int(g.dst_kind), int(g.src_kind),
                        g.base, g.fill_bits, g.dst_stride, g.dst_addend)
    if kind == 2: out += struct.pack(f"<{g.count}{'H' if g.itemsize == 2 else 'I'}", *g.values)
    elif kind in (1, 3): out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for op in image.ew_ops:
    if op.bool_output and not op.int32_output: raise ValueError("bool output requires INT32 conversion")
    op_flags = (int(op.submit_barrier) | int(op.compare)<<1 | int(op.stateful)<<2 | int(op.int32_output)<<3 |
                int(op.int32_input)<<4 | int(op.bool_output)<<5)
    out += _EWOP.pack(int(op.dst.kind), op_flags, op.dst.index,
                      int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index)
    out += _EWOP2.pack(op.count, op.ew_cfg) + struct.pack("<iii", op.dst.addend, op.lhs.addend, op.rhs.addend)
  if image.fill is not None: out += _FILL.pack(int(image.fill.dst.kind), image.fill.itemsize, image.fill.dst.index, image.fill.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nop, nconst, mid_count, post_count, gather_after, flags = _HEADER.unpack_from(blob)
  if (magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or mid_count+post_count > ngather or flags & ~1 or
      bool(mid_count) != (0 < gather_after < nop)): raise ValueError("invalid RKImage header")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_index, src_index, count, kind, naxes, itemsize, dst_kind, src_kind, base, fill_bits, dst_stride, dst_addend = \
      _GATHER.unpack_from(blob, off); off += _GATHER.size
    if (kind not in (0, 1, 2, 3) or (kind and naxes) or itemsize not in (2, 4) or dst_kind not in (0, 1) or src_kind not in (0, 1) or
        dst_stride < 1 or dst_addend < 0): raise ValueError("invalid RKGather")
    if kind == 2:
      values = struct.unpack_from(f"<{count}{'H' if itemsize == 2 else 'I'}", blob, off); off += itemsize*count
      gathers.append(RKGather(src_index, dst_index, count, fill_bits=fill_bits, values=values,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind)))
    elif kind in (1, 3):
      offsets = struct.unpack_from(f"<{count}i", blob, off); off += 4*count
      gathers.append(RKGather(src_index, dst_index, count, offsets=offsets, fill_bits=fill_bits, partial=kind == 3,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind)))
    else:
      axes = tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)); off += naxes*_GATHER_AXIS.size
      gathers.append(RKGather(src_index, dst_index, count, base, axes, fill_bits=fill_bits,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind)))
  ew_ops:list[RKEWOp] = []
  for _ in range(nop):
    dk, op_flags, di, lk, li, rk_, ri = _EWOP.unpack_from(blob, off); off += _EWOP.size
    if op_flags & ~0x3f or op_flags & 0x18 == 0x18 or op_flags & 0x20 and not op_flags & 0x08:
      raise ValueError("invalid RKEWOp flags")
    count, ew_cfg = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg,
                         bool(op_flags & 1), bool(op_flags & 2), bool(op_flags & 4), bool(op_flags & 8), bool(op_flags & 16),
                         bool(op_flags & 32)))
  fill = None
  if flags & 1:
    dst_kind, itemsize, dst_index, count = _FILL.unpack_from(blob, off); off += _FILL.size
    if itemsize not in (1, 2, 4, 8): raise ValueError("invalid RKFill item size")
    fill = RKFill(RKArg(RKBufferKind(dst_kind), dst_index), count, itemsize)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  pre_count = ngather-mid_count-post_count
  return RKImage(RKTarget(target), scratch, blob[off:], version, tuple(gathers[:pre_count]), fill, tuple(ew_ops),
                 tuple(gathers[pre_count:pre_count+mid_count]), gather_after, tuple(gathers[-post_count:] if post_count else ()))

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for reloc in stage.relocs:
    word = commands[reloc.word]
    value = (address(reloc.arg.kind, reloc.arg.index) + reloc.arg.addend) & 0xffffffff
    commands[reloc.word] = (word & ~0xffffffff0000) | (value << 16)
  return tuple(commands)

_DPU, _RDMA = 0x1001, 0x2001
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_EW_DATA_MODE_FP16 = 1 << 28
_EW_EDATA_SIZE_FP16 = 2 << 22
_EW_ALU_MIN = 1 << 16
_EW_ALU_ADD = 2 << 16
_EW_ALU_FDIV = 3 << 16
_EW_ALU_SUB = 4 << 16
_EW_ALU_ABS = 5 << 16
_EW_ALU_FLOOR = 7 << 16
_EW_ALU_CEIL = 8 << 16
_EW_RELUX_EN = 1 << 10
_EW_RELU_BYPASS = 1 << 9
_EW_OP_CVT_BYPASS = 1 << 8
_EW_LUT_BYPASS = 1 << 7
_EW_OP_SRC_DMA = 1 << 6
_EW_MUL_PRELU = 1 << 5
_EW_OP_TYPE_MUL = 1 << 2
_EW_CFG_COMMON = _EW_DATA_MODE_FP16 | _EW_EDATA_SIZE_FP16 | _EW_LUT_BYPASS | _EW_OP_SRC_DMA
_EW_CFG_RELU = _EW_CFG_COMMON
_EW_CFG_RELU6 = _EW_CFG_COMMON | _EW_RELUX_EN
_EW_CFG_MIN = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_MIN
_EW_CFG_ABS = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ABS
_EW_CFG_FLOOR = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_FLOOR
_EW_CFG_CEIL = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_CEIL
_EW_CFG_LEAKY_RELU = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_MUL_PRELU | _EW_OP_TYPE_MUL
_EW_STAGE_FP32_OUT = 1 << 29  # software tag consumed before writing EW_CFG
_DPU_DATA_FORMAT_FP16 = (2<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_OUT = (5<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT32_OUT = (4<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT32_IN = (2<<29)|(4<<26)|4
_BS_BN_BYPASS = 1|(1<<1)|(1<<4)|(1<<6)
_BS_OW_FP32_SCALAR = (1<<8)|(1<<5)|(1<<2)|(1<<1)
_BS_CFG_COMPARE = 0x40040
_BS_ALU_COMPARE = 0x33800000
_BS_MUL_COMPARE = 0x40000000
_BN_CFG_COMPARE = 0x40082
_BN_MUL_COMPARE = 0x7c000000
_BN_RELUX_COMPARE = 0x3f800000
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_FLOOR, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN, _NATIVE_POSITIVE_MASK,
 _NATIVE_RELU6, _NATIVE_SIGN) = ("rockchip_abs", "rockchip_ceil", "rockchip_floor", "rockchip_leaky_relu", "rockchip_mask_mul",
                                "rockchip_min", "rockchip_positive_mask", "rockchip_relu6", "rockchip_sign")
_EW_RELUX_CMP_RELU6 = struct.unpack("<I", struct.pack("<f", 6.0))[0]
_EW_CFG = {
  Ops.ADD: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ADD,
  Ops.SUB: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_SUB,
  Ops.MUL: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_OP_TYPE_MUL,
  Ops.MAX: _EW_CFG_COMMON | _EW_RELU_BYPASS,
  Ops.FDIV: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_ALU_FDIV,
}
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)

def _emit_fp32_out_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int) -> RKStage:
  """Emit one terminal scalar stage with FP16 inputs and an FP32 output."""
  if count != 1: raise ValueError(f"terminal EW fp32 count {count} out of range")
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP32_OUT),(_DPU,rk.REG_DPU_DST_SURF_STRIDE,1<<4),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,0),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0),
    (_DPU,rk.REG_DPU_BS_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),(_DPU,rk.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,0),(_DPU,rk.REG_DPU_WDMA_SIZE_1,0),(_DPU,rk.REG_DPU_BN_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0),(_DPU,rk.REG_DPU_EW_CFG,ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,0),(_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,4<<4),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, (2<<15)|(15<<11)|(2<<5)|(1<<3)|1))
  return RKStage(tuple(commands), tuple(relocs))

def _emit_stateful_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                         int32_output:bool=False, int32_input:bool=False) -> RKStage:
  """Emit a self-contained DPU EW stage, optionally consuming or producing native INT32."""
  if not (0 < count <= (4 if int32_output or int32_input else _MAX_EW_ELEMS_FP16)):
    raise ValueError(f"stateful EW count {count} out of range")
  lanes = 4 if int32_input else 8
  is_div = ew_cfg == _EW_CFG[Ops.FDIV]
  width = (count + lanes-1) // lanes - 1
  pipeline:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_BS_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),(_DPU,rk.REG_DPU_BS_OW_CFG,2),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,lanes-1),(_DPU,rk.REG_DPU_WDMA_SIZE_1,width),(_DPU,rk.REG_DPU_BN_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0))
  if compare: pipeline += ((_DPU,rk.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(_DPU,rk.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),
    (_DPU,rk.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_CFG,_BN_CFG_COMPARE),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE))
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_INT32_OUT if int32_output else
                                  _DPU_DATA_FORMAT_INT32_IN if int32_input else _DPU_DATA_FORMAT_FP16),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,((lanes-1)<<16)|(lanes-1))) + pipeline + (
    (_DPU,rk.REG_DPU_EW_CFG,_EW_CFG_COMMON|1 if compare else
                               (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,1 if int32_output or is_div else (1<<16)|1),(_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),
    (_DPU,rk.REG_DPU_SURFACE_ADD,4<<4),(_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,lanes-1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input else 2)<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  rdma_feature = (4<<15)|(15<<11)|(4<<5)|1 if int32_input else (2<<15)|(15<<11)|(2<<5)|(0 if is_div else 1<<3)|1
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature))
  return RKStage(tuple(commands), tuple(relocs))

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                  stateful:bool=False, int32_output:bool=False, int32_input:bool=False) -> RKStage:
  """Build one DPU EW command body without its PC-chain tail."""
  if ew_cfg & _EW_STAGE_FP32_OUT: return _emit_fp32_out_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_OUT)
  if compare or stateful or int32_output or int32_input:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg, compare, int32_output, int32_input)
  if not (0 < count <= _MAX_EW_ELEMS_FP16): raise ValueError(f"EW fp16 count {count} out of range")
  is_div = ew_cfg == _EW_CFG[Ops.FDIV]
  width = (count + 7) // 8 - 1
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP16),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7),
    (_DPU,rk.REG_DPU_EW_CFG,ew_cfg)) + (((_DPU,rk.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6),)
    if ew_cfg == _EW_CFG_RELU6 else ()) + (((_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,1<<6)) if is_div else ()) + (
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,1 if is_div else (1<<16)|1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, (2<<15)|(15<<11)|(2<<5)|(0 if is_div else 1<<3)|1))
  return RKStage(tuple(commands), tuple(relocs))

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM:
    if not u.src: return None
    u = u.src[0]
  return u

def _strip_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _local_load(u:UOp) -> UOp|None:
  u = _strip_cast(u)
  return u if u.op is Ops.LOAD and _root_param(u.src[0]) is None else None

def _eval_cast(value:int|float|bool, dtype:DType) -> int|float|bool:
  if dtype.scalar() is dtypes.bool: return bool(value)
  if dtype.scalar() in dtypes.ints: return int(value)
  if dtype.scalar() is dtypes.half: return struct.unpack("<e", struct.pack("<e", float(value)))[0]
  if dtype.scalar() is dtypes.float: return struct.unpack("<f", struct.pack("<f", float(value)))[0]
  return float(value)

def _eval_expr(u:UOp, env:dict[UOp, int], cache:dict[UOp, int|float|bool]) -> int|float|bool:
  if u in cache: return cache[u]
  if u.op is Ops.CONST: ret = _eval_cast(u.arg, u.dtype)
  elif u.op is Ops.RANGE: ret = env[u]
  elif u.op is Ops.CAST: ret = _eval_cast(_eval_expr(u.src[0], env, cache), u.dtype)
  elif u.op is Ops.WHERE:
    ret = _eval_cast(_eval_expr(u.src[1] if _eval_expr(u.src[0], env, cache) else u.src[2], env, cache), u.dtype)
  else:
    lhs = _eval_expr(u.src[0], env, cache)
    if u.op is Ops.RECIPROCAL: ret = _eval_cast(1.0 / float(lhs), u.dtype)
    elif u.op is Ops.TRUNC: ret = _eval_cast(int(lhs), u.dtype)
    else:
      rhs = _eval_expr(u.src[1], env, cache)
      if u.op is Ops.ADD: ret = _eval_cast(lhs + rhs, u.dtype)
      elif u.op is Ops.MUL: ret = _eval_cast(lhs * rhs, u.dtype)
      elif u.op is Ops.SUB: ret = _eval_cast(lhs - rhs, u.dtype)
      elif u.op is Ops.CDIV: ret = _eval_cast(cdiv(int(lhs), int(rhs)), u.dtype)
      elif u.op is Ops.CMOD: ret = _eval_cast(cmod(int(lhs), int(rhs)), u.dtype)
      elif u.op is Ops.FLOORDIV: ret = _eval_cast(floordiv(int(lhs), int(rhs)), u.dtype)
      elif u.op is Ops.FLOORMOD: ret = _eval_cast(floormod(int(lhs), int(rhs)), u.dtype)
      elif u.op is Ops.MAX: ret = _eval_cast(max(lhs, rhs), u.dtype)
      elif u.op is Ops.CMPLT: ret = lhs < rhs
      elif u.op is Ops.CMPNE: ret = lhs != rhs
      elif u.op is Ops.AND: ret = _eval_cast(int(lhs) & int(rhs), u.dtype)
      elif u.op is Ops.OR: ret = _eval_cast(int(lhs) | int(rhs), u.dtype)
      elif u.op is Ops.XOR: ret = _eval_cast(int(lhs) ^ int(rhs), u.dtype)
      else: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
  cache[u] = ret
  return ret

def _eval_int(u:UOp, env:dict[UOp, int], cache:dict[UOp, int|float|bool]|None=None) -> int:
  return int(_eval_expr(u, env, {} if cache is None else cache))

def _vector_cast(value, dtype:DType) -> np.ndarray:
  return np.asarray(value, dtype=np.dtype(dtype.scalar().fmt) if dtype.scalar().fmt is not None else None)

def _eval_vector(u:UOp, env:dict[UOp, np.ndarray], cache:dict[UOp, np.ndarray]) -> np.ndarray:
  if u in cache: return cache[u]
  if u.op is Ops.CONST: ret = _vector_cast(u.arg, u.dtype)
  elif u.op is Ops.RANGE: ret = env[u]
  elif u.op is Ops.CAST: ret = _vector_cast(_eval_vector(u.src[0], env, cache), u.dtype)
  elif u.op is Ops.WHERE:
    ret = _vector_cast(np.where(_eval_vector(u.src[0], env, cache), _eval_vector(u.src[1], env, cache),
                                _eval_vector(u.src[2], env, cache)), u.dtype)
  else:
    lhs = _eval_vector(u.src[0], env, cache)
    if u.op is Ops.RECIPROCAL: ret = _vector_cast(1.0 / lhs, u.dtype)
    elif u.op is Ops.TRUNC: ret = _vector_cast(np.trunc(lhs), u.dtype)
    else:
      rhs = _eval_vector(u.src[1], env, cache)
      if u.op is Ops.ADD: ret = _vector_cast(lhs + rhs, u.dtype)
      elif u.op is Ops.MUL: ret = _vector_cast(lhs * rhs, u.dtype)
      elif u.op is Ops.SUB: ret = _vector_cast(lhs - rhs, u.dtype)
      elif u.op in (Ops.CDIV, Ops.CMOD):
        with np.errstate(divide="ignore", invalid="ignore"):
          quotient = np.where(rhs != 0, np.trunc(lhs / rhs), 0)
        ret = _vector_cast(quotient if u.op is Ops.CDIV else lhs-quotient*rhs, u.dtype)
      elif u.op in (Ops.FLOORDIV, Ops.FLOORMOD):
        quotient = np.zeros(np.broadcast_shapes(lhs.shape, rhs.shape), dtype=np.result_type(lhs, rhs))
        np.floor_divide(lhs, rhs, out=quotient, where=rhs != 0)
        ret = _vector_cast(quotient if u.op is Ops.FLOORDIV else lhs-quotient*rhs, u.dtype)
      elif u.op is Ops.MAX: ret = _vector_cast(np.maximum(lhs, rhs), u.dtype)
      elif u.op is Ops.CMPLT: ret = lhs < rhs
      elif u.op is Ops.CMPNE: ret = lhs != rhs
      elif u.op is Ops.AND: ret = _vector_cast(np.bitwise_and(lhs, rhs), u.dtype)
      elif u.op is Ops.OR: ret = _vector_cast(np.bitwise_or(lhs, rhs), u.dtype)
      elif u.op is Ops.XOR: ret = _vector_cast(np.bitwise_xor(lhs, rhs), u.dtype)
      else: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
  cache[u] = ret
  return ret

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX}
def _is_static_expr(u:UOp, cache:dict[UOp, bool]|None=None) -> bool:
  if cache is not None and u in cache: return cache[u]
  ret = u.op in _STATIC_OPS and all(_is_static_expr(x, cache) for x in u.src)
  if cache is not None: cache[u] = ret
  return ret

def _index_ranges(index:UOp) -> list[UOp]: return [u for u in index.toposort() if u.op is Ops.RANGE]

RKOutput = tuple[UOp, UOp, int, UOp, UOp]
def _output_store(uops:list[UOp], dtype:DType|tuple[DType, ...], *, allow_local:bool=False, reject_reduce:bool=False) \
                  -> RKOutput|None:
  """Return the single statically-sized output store shared by specialized graph matchers."""
  stores = [u for u in uops if u.op is Ops.STORE]
  outputs = [(store, root) for store in stores if (root:=_root_param(store.src[0])) is not None]
  if (len(outputs) != 1 or not allow_local and len(stores) != 1 or reject_reduce and any(u.op is Ops.REDUCE for u in uops)):
    return None
  store, out_param = outputs[0]
  accepted = dtype if isinstance(dtype, tuple) else (dtype,)
  if out_param.dtype.scalar() not in accepted or out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  return store, out_param, int(out_param.src[0].arg), store.src[0].src[1], store.src[1]

def _iter_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  if not ranges: return [{}]
  order:list[UOp] = []
  seen:set[UOp] = set()
  def add(r:UOp) -> None:
    if r in seen: return
    for src in r.src[1:]:
      if src.op is Ops.RANGE: add(src)
    seen.add(r); order.append(r)
  for r in ranges: add(r)
  envs:list[dict[UOp, int]] = [{}]
  for r in order:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    envs = [{**env, r: i} for env in envs for i in range(int(r.src[0].arg))]
  return envs

def _loop_reduction_shape(store:UOp, out_param:UOp, nodes:list[UOp]) -> tuple[int, list[dict[UOp, int]], UOp, int]|None:
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  rows = int(out_param.src[0].arg)
  out_ranges = _index_ranges(store.src[0].src[1])
  reduce_ranges = [u for u in nodes if u.op is Ops.RANGE and u not in out_ranges]
  if rows <= 0 or len(reduce_ranges) != 1: return None
  reduce_range = reduce_ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) <= 0: return None
  try: envs = _iter_range_env(out_ranges)
  except RuntimeError: return None
  if len(envs) != rows or tuple(_eval_int(store.src[0].src[1], env) for env in envs) != tuple(range(rows)): return None
  return rows, envs, reduce_range, groups

@dataclass(frozen=True)
class RKLoopReduction:
  store:UOp; out:UOp; nodes:list[UOp]; rows:int; envs:list[dict[UOp, int]]; reduce_range:UOp; groups:int; update:UOp; post_scale:float

def _loop_reduction_match(uops:list[UOp]) -> RKLoopReduction|None:
  """Parse the output, accumulator update, shape, and optional final scale of a loop reduction."""
  if (output:=_output_store(uops, (dtypes.half, dtypes.float), allow_local=True)) is None: return None
  store, out, _, _, root = output
  nodes = list(root.toposort())
  if (shape:=_loop_reduction_shape(store, out, nodes)) is None: return None
  rows, envs, reduce_range, groups = shape
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1: return None
  if _local_load(root) is not None: post_scale = 1.0
  elif root.op is Ops.MUL and (load:=next((x for x in root.src if _local_load(x) is not None), None)) is not None and \
       (scale:=root.src[1 if root.src[0] is load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else: return None
  return RKLoopReduction(store, out, nodes, rows, envs, reduce_range, groups, _strip_cast(updates[0].src[1]), post_scale)

def _spaced_reduction_gathers(src_slot:int, dst_slot:int, rows:int, blocks:list[tuple[int, ...]]|tuple[tuple[int, ...], ...]) \
    -> tuple[RKGather, ...]:
  if rows != 1:
    return tuple(RKGather(src_slot, dst_slot, rows, offsets=block, dst_addend=i*32) for i,block in enumerate(blocks))
  offsets = tuple(block[0] for block in blocks)
  direct = offsets == tuple(range(len(blocks)))
  return (RKGather(src_slot, dst_slot, len(blocks), axes=((1, len(blocks), 1),) if direct else (),
                   offsets=() if direct else offsets, dst_stride=32),)

def _static_values(out_index:UOp, expr:UOp, count:int, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  values:list[int|None] = [None] * count
  for env in _iter_range_env(ranges):
    cache:dict[UOp, int|float|bool] = {}
    dst = _eval_int(out_index, env, cache)
    if not 0 <= dst < count: raise RuntimeError("RKPLAN_REJECT:static_index")
    value = encode(_eval_expr(expr, env, cache))
    if values[dst] not in (None, value): raise RuntimeError("RKPLAN_REJECT:static_index")
    values[dst] = value
  if any(x is None for x in values): raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(x for x in values if x is not None)

def _static_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  return _static_values(out_index, expr, count, lambda value:struct.unpack("<H", struct.pack("<e", float(value)))[0])

def _static_int_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  """Evaluate a compile-time integer expression in compact output order."""
  return _static_values(out_index, expr, count, int)

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op is Ops.RANGE: return 0, {u: 1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _affine_index(u.src[0]), _affine_index(u.src[1])
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {r: coeff*scale for r, coeff in affine[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  coeffs = lhs[1].copy()
  for r, coeff in rhs[1].items():
    if (merged:=coeffs.get(r, 0) + sign*coeff): coeffs[r] = merged
    elif r in coeffs: del coeffs[r]
  return lhs[0] + sign*rhs[0], coeffs

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(load_index) + ([] if gate is None else _index_ranges(gate))):
    raise RuntimeError("RKPLAN_REJECT:gather_index")
  envs = _iter_range_env(ranges)
  vector_env = {r: np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}
  cache:dict[UOp, np.ndarray] = {}
  out_affine = _affine_index(out_index)
  if out_affine is None: dst = np.broadcast_to(_eval_vector(out_index, vector_env, cache), len(envs)).astype(np.int64)
  else:
    dst = np.full(len(envs), out_affine[0], dtype=np.int64)
    for r,stride in out_affine[1].items(): dst += vector_env[r]*stride
  src = np.broadcast_to(_eval_vector(load_index, vector_env, cache), len(envs)).astype(np.int64)
  values = src if gate is None else np.where(np.broadcast_to(_eval_vector(gate, vector_env, cache), len(envs)), src, -1)
  if np.any((dst < 0) | (dst >= count)) or np.any(values < -1): raise RuntimeError("RKPLAN_REJECT:gather_index")
  offsets = np.full(count, -2, dtype=np.int64)
  offsets[dst] = values
  if np.any(offsets == -2): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(int(x) for x in offsets)

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0) -> RKGather:
  out_affine, load_affine = _affine_index(out_index), _affine_index(load_index)
  if gate is None and out_affine is not None and load_affine is not None and out_affine[0] == 0:
    expected = 1
    axes:list[tuple[int, int, int]] = []
    for r, dst_stride in sorted(out_affine[1].items(), key=lambda item: item[1]):
      if dst_stride != expected or r.src[0].op is not Ops.CONST: break
      limit = int(r.src[0].arg)
      if limit <= 0: break
      if (src_stride:=load_affine[1].get(r, 0)): axes.append((dst_stride, limit, src_stride))
      expected *= limit
    else:
      if expected == count and all(r in out_affine[1] for r in load_affine[1]):
        return RKGather(src_index, dst_index, count, load_affine[0], tuple(axes))
  return RKGather(src_index, dst_index, count, offsets=_gather_offsets(out_index, load_index, gate, count), fill_bits=fill_bits)

def _selection_gather(u:UOp, out_index:UOp, count:int, oslot:int, selection_cache:dict[UOp, tuple[bool, bool]]|None=None,
                      static_cache:dict[UOp, bool]|None=None) -> RKGather|RKMultiGather|None:
  """Collapse a static selection tree into raw offsets from one or more FP16 source buffers."""
  seen = selection_cache if selection_cache is not None else {}
  def selection_tree(x:UOp) -> tuple[bool, bool]:
    if x in seen: return seen[x]
    if x.op is Ops.CONST: ret = (x.dtype.scalar() is dtypes.half, False)
    elif x.op is Ops.CAST and x.dtype.scalar() is dtypes.half: ret = selection_tree(x.src[0])
    elif x.op is Ops.WHERE:
      lhs, rhs = selection_tree(x.src[1]), selection_tree(x.src[2]); ret = (lhs[0] and rhs[0], True)
    elif x.op is Ops.ADD:
      lhs, rhs = selection_tree(x.src[0]), selection_tree(x.src[1]); ret = (lhs[0] and rhs[0], lhs[1] or rhs[1])
    elif x.op is Ops.LOAD and x.src[0].op is Ops.INDEX:
      fallback = selection_tree(x.src[1]) if len(x.src) > 1 else (True, False)
      ret = (fallback[0], fallback[1] or len(x.src) > 2)
    else: ret = (False, False)
    seen[x] = ret
    return ret
  if selection_tree(u) != (True, True): return None
  out_ranges = _index_ranges(out_index)
  out_range_set, range_cache = set(out_ranges), {}
  def index_ranges(x:UOp) -> list[UOp]:
    if x not in range_cache: range_cache[x] = _index_ranges(x)
    return range_cache[x]
  envs = _iter_range_env(out_ranges)
  vector_env = {r: np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in out_ranges}
  eval_cache:dict[UOp, np.ndarray] = {}
  choices:dict[UOp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  empty = np.full(len(envs), -1, dtype=np.int64)
  def selected(x:UOp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x in choices: return choices[x]
    if x.op is Ops.CONST and x.dtype.scalar() is dtypes.half:
      bits = struct.unpack("<H", struct.pack("<e", float(x.arg)))[0]
      ret = empty, empty, np.full(len(envs), bits, dtype=np.int64)
    elif x.op is Ops.CAST and x.dtype.scalar() is dtypes.half: ret = selected(x.src[0])
    elif x.op is Ops.WHERE:
      if not _is_static_expr(x.src[0], static_cache): raise ValueError
      cond = np.broadcast_to(_eval_vector(x.src[0], vector_env, eval_cache), len(envs)).astype(bool)
      lhs, rhs = selected(x.src[1]), selected(x.src[2])
      ret = np.where(cond, lhs[0], rhs[0]), np.where(cond, lhs[1], rhs[1]), np.where(cond, lhs[2], rhs[2])
    elif x.op is Ops.ADD:
      lhs, rhs = selected(x.src[0]), selected(x.src[1])
      if np.any((lhs[0] >= 0) & (rhs[0] >= 0)) or np.any((lhs[2] >= 0) & (rhs[2] >= 0) & (lhs[2] != rhs[2])): raise ValueError
      use_lhs = lhs[0] >= 0
      ret = np.where(use_lhs, lhs[0], rhs[0]), np.where(use_lhs, lhs[1], rhs[1]), np.where(lhs[2] >= 0, lhs[2], rhs[2])
    else:
      if x.op is not Ops.LOAD or x.src[0].op is not Ops.INDEX or x.src[0].src[0].op is not Ops.PARAM: raise ValueError
      param, index, gate = x.src[0].src[0], x.src[0].src[1], x.src[2] if len(x.src) > 2 else None
      if (param.dtype.scalar() is not dtypes.half or param.arg.slot == oslot or param.src[0].op is not Ops.CONST or
          any(r not in out_range_set for r in index_ranges(index) + ([] if gate is None else index_ranges(gate)))): raise ValueError
      if gate is not None:
        if not _is_static_expr(gate, static_cache): raise ValueError
      enabled = np.ones(len(envs), dtype=bool) if gate is None else \
        np.broadcast_to(_eval_vector(gate, vector_env, eval_cache), len(envs)).astype(bool)
      fallback = selected(x.src[1]) if len(x.src) > 1 else (empty, empty, empty)
      offset = np.broadcast_to(_eval_vector(index, vector_env, eval_cache), len(envs)).astype(np.int64)
      ret = np.where(enabled, param.arg.slot, fallback[0]), np.where(enabled, offset, fallback[1]), np.where(enabled, -1, fallback[2])
    choices[x] = ret
    return ret
  slots, offsets, fills = [-2] * count, [-2] * count, [-2] * count
  try:
    dsts = np.broadcast_to(_eval_vector(out_index, vector_env, eval_cache), len(envs)).astype(np.int64)
    selected_slots, selected_offsets, selected_fills = selected(u)
    for dst,slot,offset,fill in zip(dsts, selected_slots, selected_offsets, selected_fills):
      if not 0 <= dst < count: raise ValueError
      if slots[dst] not in (-2, slot) or offsets[dst] not in (-2, offset) or fills[dst] not in (-2, fill): raise ValueError
      slots[dst], offsets[dst], fills[dst] = int(slot), int(offset), int(fill)
  except (RuntimeError, ValueError): return None
  sources = sorted(set(slot for slot in slots if slot >= 0))
  if not sources or any(offset == -2 for offset in offsets): return None
  fill_values = set(fill for slot,fill in zip(slots, fills) if slot < 0 and fill >= 0)
  if len(sources) == 1 and len(fill_values) <= 1:
    return RKGather(sources[0], 0, count, offsets=tuple(offsets), fill_bits=next(iter(fill_values), 0))
  plans:list[RKGather] = []
  if any(slot < 0 for slot in slots):
    plans.append(RKGather(sources[0], 0, count, values=tuple(fill if slot < 0 and fill >= 0 else 0 for slot,fill in zip(slots, fills))))
  plans.extend(RKGather(src, 0, count, offsets=tuple(offset if slot == src else -1 for slot,offset in zip(slots, offsets)), partial=True)
               for src in sources)
  return RKMultiGather(tuple(plans))

def _ew_leaf(u:UOp, out_index:UOp, count:int, oslot:int, static_cache:dict[UOp, bool]|None=None,
             selection_cache:dict[UOp, tuple[bool, bool]]|None=None) -> RKLeaf:
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.half: return float(u.arg)
  if u.dtype.scalar() is dtypes.half and _is_static_expr(u, static_cache): return RKStatic(u)
  if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half: return _ew_leaf(u.src[0], out_index, count, oslot, static_cache, selection_cache)
  if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
    param, index, gate = u.src[0].src[0], u.src[0].src[1], u.src[2] if len(u.src) > 2 else None
    if len(u.src) > 1 and u.src[1].op is not Ops.CONST:
      return _selection_gather(u, out_index, count, oslot, selection_cache, static_cache)
    fill_bits = struct.unpack("<H", struct.pack("<e", float(u.src[1].arg) if len(u.src) > 1 else 0.0))[0]
    if param.dtype.scalar() not in (dtypes.half, dtypes.float) or param.arg.slot == oslot or param.src[0].op is not Ops.CONST: return None
    if gate is None and index.key == out_index.key and int(param.src[0].arg) == count: return RKArg(RKBufferKind.ARG, param.arg.slot)
    return param, index, gate, fill_bits
  return _selection_gather(u, out_index, count, oslot, selection_cache, static_cache) if u.dtype.scalar() is dtypes.half else None

def _unsupported_ew_ops(uops:list[UOp], out_index:UOp, count:int, oslot:int, supported:dict) -> list[str]:
  bad:list[str] = []
  for i, u in enumerate(uops):
    if u.op in (Ops.CONST, Ops.PARAM, Ops.RANGE, Ops.END, Ops.SINK, Ops.STORE, Ops.INDEX): continue
    if u.op in (Ops.ADD, Ops.MUL) and u.dtype.scalar() is dtypes.int: continue
    if u.op in supported and u.dtype.scalar() is dtypes.half: continue
    if u.op is Ops.LOAD or (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half):
      if _ew_leaf(u, out_index, count, oslot) is None: bad.append(f"{i}:{u.op.name}")
      continue
    if u.op is Ops.CAST or u.op is Ops.REDUCE or u.op in GroupOp.ALU: bad.append(f"{i}:{u.op.name}")
  return bad

def _mul_reduction_terms(u:UOp) -> tuple[list[UOp], int]|None:
  """Flatten a half ADD tree containing products and optional bias terms."""
  if u.dtype.scalar() is not dtypes.half: return None
  if u.op is Ops.MUL: return [u], 0 if u.arg in (_NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL) else 1
  if u.op is not Ops.ADD: return [u], 0
  lhs, rhs = _mul_reduction_terms(u.src[0]), _mul_reduction_terms(u.src[1])
  return None if lhs is None or rhs is None else (lhs[0] + rhs[0], lhs[1] + rhs[1])

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is not Ops.MAX or u.dtype.scalar() is not dtypes.half: return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None

def _compensated_mul_sum(terms:list[UOp]) -> UOp:
  """Kahan sum built after symbolic simplification so compensation remains as DPU EW ops."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  total, correction = terms[0], zero
  for term in terms[1:]:
    adjusted = term.alu(Ops.ADD, correction.alu(Ops.MUL, neg_one))
    updated = total.alu(Ops.ADD, adjusted)
    correction = updated.alu(Ops.ADD, total.alu(Ops.MUL, neg_one)).alu(Ops.ADD, adjusted.alu(Ops.MUL, neg_one))
    total = updated
  return total

def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = x.alu(Ops.MUL, splitter)
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one)
  error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low)).alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  return total, lhs_error.alu(Ops.ADD, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  products, errors = tuple(x[0] for x in pairs), tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL)
  high, middle, low = products[0], zero, zero
  for part in products[1:] + errors:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = low.alu(Ops.ADD, error)
  middle = middle.alu(Ops.ADD, low)
  return high.alu(Ops.ADD, middle)

def _lower_dot_loop_reduction(loop:RKLoopReduction) -> RKImage|None:
  """Lower an FP16 dot loop as vector MUL terms followed by a balanced vector ADD tree."""
  if loop.out.dtype.scalar() is not dtypes.half or loop.rows > _MAX_EW_ELEMS_FP16 or loop.post_scale != 1.0: return None
  store, update, reduce_range, groups = loop.store, loop.update, loop.reduce_range, loop.groups
  if update.op is not Ops.ADD or (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None: return None
  product = _strip_cast(update.src[1 if update.src[0] is acc else 0])
  if product.op is not Ops.MUL or product.dtype.scalar() is not dtypes.half: return None
  for operand in product.src:
    operand = _strip_cast(operand)
    param = _root_param(operand.src[0]) if operand.op is Ops.LOAD and operand.src and operand.src[0].op is Ops.INDEX else None
    if param is None or operand.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST: return None
  terms = [product.substitute({reduce_range:reduce_range.const_like(r)}) for r in range(groups)]
  while len(terms) > 1:
    terms = [terms[i].alu(Ops.ADD, terms[i+1]) for i in range(0, len(terms)-1, 2)] + (terms[-1:] if len(terms) & 1 else [])
  return lower_ew([store.replace(src=(store.src[0], terms[0], *store.src[2:]))])

def _lower_centered_square_loop_reduction(loop:RKLoopReduction) -> RKImage|None:
  """Lower scalar SUM((x-center)^2)*scale using only aligned one-lane DPU EW stages."""
  if loop.groups < 2: return None
  out_param = loop.out
  rows, envs, reduce_range, groups, update, post_scale = \
    loop.rows, loop.envs, loop.reduce_range, loop.groups, loop.update, loop.post_scale
  fp32_out = out_param.dtype.scalar() is dtypes.float
  if update.op is not Ops.ADD: return None
  acc = next((x for x in update.src if _local_load(x) is not None), None)
  if acc is None: return None
  square = _strip_cast(update.src[1 if update.src[0] is acc else 0])
  if square.op is not Ops.MUL or square.src[0] is not square.src[1]: return None
  delta = _strip_cast(square.src[0])
  if delta.op is not Ops.ADD: return None

  direct = next((x for x in delta.src if _strip_cast(x).op is Ops.LOAD), None)
  negated = next((x for x in delta.src if x is not direct), None)
  if direct is None or negated is None: return None
  direct = _strip_cast(direct); negated = _strip_cast(negated)
  if negated.op is not Ops.MUL: return None
  center = next((_strip_cast(x) for x in negated.src if _strip_cast(x).op is Ops.LOAD), None)
  neg_one = next((x for x in negated.src if x.op is Ops.CONST and float(x.arg) == -1.0), None)
  if center is None or neg_one is None or direct.src[0].op is not Ops.INDEX or center.src[0].op is not Ops.INDEX: return None
  data_param, center_param = _root_param(direct.src[0]), _root_param(center.src[0])
  if (data_param is None or center_param is None or data_param.arg.slot == center_param.arg.slot or direct.dtype.scalar() is not dtypes.half or
      center.dtype.scalar() is not dtypes.half or data_param.src[0].op is not Ops.CONST or center_param.src[0].op is not Ops.CONST): return None
  try:
    data_blocks = tuple(tuple(_eval_int(direct.src[0].src[1], {**env, reduce_range:r}) for env in envs) for r in range(groups))
    center_offsets = tuple(_eval_int(center.src[0].src[1], env) for env in envs)
  except RuntimeError: return None
  if (int(data_param.src[0].arg) != rows*groups or sorted(offset for block in data_blocks for offset in block) != list(range(rows*groups)) or
      center_offsets != tuple(range(rows)) or int(center_param.src[0].arg) != rows): return None

  constants = () if post_scale == 1.0 else (post_scale,)
  center_arg = RKArg(RKBufferKind.ARG, center_param.arg.slot)
  def prepare(ops:list[RKEWOp], value:RKArg, _:dict[float, int]) -> None:
    ops.append(RKEWOp(value, value, center_arg, rows, _EW_CFG[Ops.SUB], stateful=not ops))
    ops.append(RKEWOp(value, value, value, rows, _EW_CFG[Ops.MUL]))
  return _reduction_image(out_param.arg.slot, rows, data_param.arg.slot, data_blocks, constants,
                          _EW_CFG[Ops.ADD], fp32_out, post_scale, prepare)

def _lower_scalar_loop_reduction(loop:RKLoopReduction) -> RKImage|None:
  """Turn a compact scalar register reduction into balanced FP16 DPU EW stages."""
  out_param, nodes = loop.out, loop.nodes
  rows, envs, reduce_range, groups, update, post_scale = \
    loop.rows, loop.envs, loop.reduce_range, loop.groups, loop.update, loop.post_scale
  fp32_out = out_param.dtype.scalar() is dtypes.float
  loads:list[tuple[UOp, UOp]] = []
  for u in nodes:
    if u.op is not Ops.LOAD or u.dtype.scalar() is not dtypes.half or not u.src or u.src[0].op is not Ops.INDEX: continue
    param = _root_param(u.src[0])
    if param is not None and param.arg.slot != out_param.arg.slot: loads.append((u, param))
  if not loads or len({param.key for _,param in loads}) != 1: return None
  in_param = loads[0][1]
  if in_param.src[0].op is not Ops.CONST: return None
  update_nodes = update.toposort()
  reduce_ops = {u.op for u in update_nodes if u.dtype.scalar() is dtypes.half and u.op in (Ops.ADD, Ops.MUL, Ops.MAX)}
  negate_inputs = reduce_ops == {Ops.MUL, Ops.MAX} and any(u.op is Ops.CONST and u.dtype.scalar() is dtypes.half and
                                                            float(u.arg) == -1.0 for u in update_nodes)
  if negate_inputs: reduce_op = Ops.MAX
  elif len(reduce_ops) == 1: reduce_op = next(iter(reduce_ops))
  else: return None
  if reduce_op not in _EW_CFG: return None
  try:
    blocks = [tuple(_eval_int(load.src[0].src[1], {**env, reduce_range:r}) for env in envs)
              for load,_ in loads for r in range(groups)]
  except RuntimeError: return None
  input_count = int(in_param.src[0].arg)
  if input_count != rows*len(blocks) or sorted(offset for block in blocks for offset in block) != list(range(input_count)): return None
  if input_count < 2: return None

  const_values = tuple(dict.fromkeys(x for x in ((-1.0,) if negate_inputs else ()) + ((post_scale,) if post_scale != 1.0 else ())))
  def prepare(ops:list[RKEWOp], value:RKArg, slots:dict[float, int]) -> None:
    ops.append(RKEWOp(value, value, RKArg(RKBufferKind.SCRATCH, slots[-1.0]), rows, _EW_CFG[Ops.MUL]))
  return _reduction_image(out_param.arg.slot, rows, in_param.arg.slot, blocks, const_values,
                          _EW_CFG[reduce_op], fp32_out, post_scale, prepare if negate_inputs else None)

def _flatten_binary(root:UOp, op:Ops) -> list[UOp]:
  return _flatten_binary(root.src[0], op)+_flatten_binary(root.src[1], op) if root.op is op else [root]

def _split_load_pairs(pairs:tuple[tuple[UOp, UOp], ...]) -> tuple[UOp, tuple[UOp, ...]]|None:
  """Split equality pairs into their one common lane and ordered candidate lanes."""
  if not pairs: return None
  common = set(pairs[0]).intersection(*(set(pair) for pair in pairs[1:]))
  if len(common) != 1: return None
  current = next(iter(common))
  candidates = tuple(next((x for x in pair if x is not current), None) for pair in pairs)
  if any(x is None for x in candidates): return None
  return current, tuple(x for x in candidates if x is not None)

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64
  return vector_bytes, vector_bytes//2, rows*vector_bytes//2

def _stripe_gathers(src_slot:int, dst_slot:int, count:int, rows:Iterable[Iterable[int]], vector_lanes:int, *,
                    values:bool=False, itemsize:int=2) -> tuple[RKGather, ...]:
  """Pack candidate or repeated-current rows into one aligned lane matrix."""
  return tuple(RKGather(src_slot, dst_slot, count, offsets=() if values else tuple(row), values=tuple(row) if values else (),
                        dst_addend=i*vector_lanes, itemsize=itemsize) for i,row in enumerate(rows))

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int) -> RKArg:
  """Append a balanced row reduction, making its first dependent stage self-contained."""
  first = True
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      ops.append(RKEWOp(active[i], active[i], active[i+1], count, cfg, submit_barrier=first, stateful=first))
      first = False; reduced.append(active[i])
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return active[0]

def _ew_eq_mask(ops:list[RKEWOp], arg:Callable[[int], RKArg], lhs:int, rhs:int, temps:tuple[int, int, int, int], one:int,
                lanes:int, barriers:tuple[bool, bool]=(False, True)) -> RKArg:
  """Append SUB, ABS, nonzero comparison, and inversion for an FP16 equality mask."""
  diff, magnitude, unequal, equal = temps
  ops.extend((RKEWOp(arg(diff), arg(lhs), arg(rhs), lanes, _EW_CFG[Ops.SUB], submit_barrier=barriers[0], stateful=barriers[0]),
              RKEWOp(arg(magnitude), arg(diff), arg(diff), lanes, _EW_CFG_ABS, submit_barrier=barriers[1], stateful=barriers[1]),
              RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), lanes, _EW_CFG[Ops.MAX], compare=True),
              RKEWOp(arg(equal), arg(one), arg(unequal), lanes, _EW_CFG[Ops.SUB], stateful=True)))
  return arg(equal)

def _reduce_arena(ops:list[RKEWOp], active:list[int], count:int, cfg:int, arena:Callable[[int], RKArg],
                  out:RKArg|None=None, fp32_out:bool=False) -> RKArg:
  """Append a balanced in-place arena reduction and optionally write its final stage directly to output."""
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs, final = active[i], active[i+1], len(active) == 2 and out is not None
      dst = out if final and out is not None else arena(lhs)
      ops.append(RKEWOp(dst, arena(lhs), arena(rhs), count,
                        cfg | (_EW_STAGE_FP32_OUT if fp32_out and final else 0)))
      reduced.append(lhs)
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return out if out is not None else arena(active[0])

def _reduction_image(out_slot:int, rows:int, source_slot:int, blocks:list[tuple[int, ...]]|tuple[tuple[int, ...], ...],
                     constants:tuple[float, ...], cfg:int, fp32_out:bool, post_scale:float,
                     prepare:Callable[[list[RKEWOp], RKArg, dict[float, int]], None]|None=None) -> RKImage:
  """Materialize row blocks, apply an optional lane transform, reduce them, and write the typed result."""
  const_slots, data_slot = {value:i for i,value in enumerate(constants)}, len(constants)
  gathers = _spaced_reduction_gathers(source_slot, data_slot, rows, blocks)
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, data_slot, offset)
  ops:list[RKEWOp] = []
  active = [i*64 for i in range(len(blocks))]
  if prepare is not None:
    for offset in active: prepare(ops, arena(offset), const_slots)
  out = RKArg(RKBufferKind.ARG, out_slot)
  reduced = _reduce_arena(ops, active, rows, cfg, arena, out if post_scale == 1.0 else None, fp32_out)
  if post_scale != 1.0:
    ops.append(RKEWOp(out, reduced, RKArg(RKBufferKind.SCRATCH, const_slots[post_scale]), rows,
                      _EW_CFG[Ops.MUL] | (_EW_STAGE_FP32_OUT if fp32_out else 0)))
  scratch = tuple(RKScratch(rows*2) for _ in constants) + (RKScratch(len(blocks)*64),)
  return RKImage(RKTarget.RK3588, scratch, b"".join(struct.pack("<e", value) for value in constants), gathers=gathers, ew_ops=tuple(ops))

def _lower_raw_int32_layout(output:RKOutput) -> RKImage|None:
  """Move an INT32 tensor through a static view or shrink without interpreting its values."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  if (value.op is not Ops.LOAD or value.dtype.scalar() is not dtypes.int or len(value.src) != 1 or value.src[0].op is not Ops.INDEX or
      value.src[0].src[0].op is not Ops.PARAM): return None
  source, source_index = value.src[0].src[:2]
  if source.src[0].op is not Ops.CONST: return None
  try: offsets = _gather_offsets(out_index, source_index, None, count)
  except RuntimeError: return None
  if len(set(offsets)) != count or any(not 0 <= offset < int(source.src[0].arg) for offset in offsets): return None
  gather = RKGather(source.arg.slot, out_param.arg.slot, count, offsets=offsets, dst_kind=RKBufferKind.ARG, itemsize=4)
  return RKImage(RKTarget.RK3588, gathers=(gather,))

def _lower_raw_fp16_bitcast(output:RKOutput) -> RKImage|None:
  """Pair adjacent FP16 lane representations into an INT32 output without numeric conversion."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  if value.op is not Ops.BITCAST or value.dtype.scalar() is not dtypes.int or len(value.src) != 1: return None
  packed = value.src[0]
  if packed.op is not Ops.ADD or packed.dtype.scalar() is not dtypes.uint: return None
  lanes:dict[int, UOp] = {}
  for term in packed.src:
    if (term.op is not Ops.SHL or len(term.src) != 2 or term.src[1].op is not Ops.CONST or
        (shift:=int(term.src[1].arg)) not in (0, 16)): return None
    cast = term.src[0]
    if (cast.op is not Ops.CAST or cast.dtype.scalar() is not dtypes.uint or len(cast.src) != 1 or
        cast.src[0].op is not Ops.BITCAST or cast.src[0].dtype.scalar() is not dtypes.ushort or len(cast.src[0].src) != 1): return None
    load = cast.src[0].src[0]
    if load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or len(load.src) != 1 or load.src[0].op is not Ops.INDEX: return None
    lanes[shift] = load
  if len(lanes) != 2: return None
  params = tuple(_root_param(lanes[shift].src[0]) for shift in (0, 16))
  if (any(param is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST for param in params) or
      params[0].arg.slot != params[1].arg.slot): return None  # type: ignore[union-attr]
  source = params[0]; assert source is not None
  try: low, high = (_gather_offsets(out_index, lanes[shift].src[0].src[1], None, count) for shift in (0, 16))
  except RuntimeError: return None
  source_count = int(source.src[0].arg)
  if any(offset < 0 or offset+1 >= source_count or offset & 1 or high[i] != offset+1 for i,offset in enumerate(low)): return None
  gather = RKGather(source.arg.slot, out_param.arg.slot, count, offsets=tuple(offset//2 for offset in low),
                    dst_kind=RKBufferKind.ARG, itemsize=4)
  return RKImage(RKTarget.RK3588, gathers=(gather,))

def _equality_pair(predicate:UOp) -> tuple[UOp, UOp]|None:
  """Recognize Tinygrad's boolean inversion of one CMPNE pair."""
  if predicate.op is not Ops.CMPNE: return None
  truths = [x for x in predicate.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg)]
  inequalities = [x for x in predicate.src if x.op is Ops.CMPNE]
  if len(truths) != 1 or len(inequalities) != 1: return None
  pair = inequalities[0].src
  return pair if len(pair) == 2 else None

def _load_equality(predicate:UOp) -> tuple[UOp, UOp]|None:
  """Recognize boolean equality between two loaded lanes."""
  if (pair:=_equality_pair(predicate)) is None: return None
  if len(pair) != 2 or any(x.op is not Ops.LOAD or x.src[0].op is not Ops.INDEX for x in pair): return None
  return pair[0], pair[1]

def _lower_occurrence_count(output:RKOutput) -> RKImage|None:
  """Lower stable-sort's unrolled prefix equality counts to DPU masks and ADD."""
  store, out_param, count, out_index, _ = output
  if count <= 0: return RKImage(RKTarget.RK3588)

  terms = _flatten_binary(store.src[1], Ops.ADD)
  parsed:list[tuple[tuple[UOp, UOp], UOp|None]] = []
  for term in terms:
    if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.int: return None
    predicate, gate = term.src[0], None
    if predicate.op is Ops.AND:
      equalities = [x for x in predicate.src if not _is_static_expr(x)]
      gates = [x for x in predicate.src if _is_static_expr(x)]
      if len(equalities) != 1 or len(gates) != 1: return None
      predicate, gate = equalities[0], gates[0]
    if (pair:=_load_equality(predicate)) is None or any(x.dtype.scalar() is not dtypes.half for x in pair): return None
    parsed.append((pair, gate))
  if not 2 <= len(parsed) <= 255: return None
  if (split:=_split_load_pairs(tuple(pair for pair,_ in parsed))) is None: return None
  current, candidate_loads = split
  loads = (*candidate_loads, current)
  params = tuple(_root_param(load.src[0]) for load in loads)
  if (any(param is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST for param in params) or
      len({param.arg.slot for param in params if param is not None}) != 1): return None
  source = params[0]; assert source is not None
  source_count = int(source.src[0].arg)

  try:
    candidate_offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in candidate_loads)
    current_offsets = _gather_offsets(out_index, current.src[0].src[1], None, count)
    valid_bits = tuple(_static_vector(out_index, gate, count) if gate is not None else (0x3c00,)*count for _,gate in parsed)
  except RuntimeError: return None
  if any(not 0 <= offset < source_count for offsets in (*candidate_offsets, current_offsets) for offset in offsets): return None
  for dst in range(count):
    valid_offsets = [offsets[dst] for offsets,bits in zip(candidate_offsets, valid_bits) if bits[dst] == 0x3c00]
    if not valid_offsets or len(valid_offsets) != len(set(valid_offsets)) or any(bits[dst] not in (0, 0x3c00) for bits in valid_bits): return None

  window = len(candidate_offsets)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one, candidates_slot, current_slot, valid_slot, diff, magnitude, unequal, equal, selected, int_tiles = range(11)
  gathers = _stripe_gathers(source.arg.slot, candidates_slot, count, candidate_offsets, vector_lanes)
  gathers += _stripe_gathers(source.arg.slot, current_slot, count, (current_offsets,)*window, vector_lanes)
  gathers += _stripe_gathers(source.arg.slot, valid_slot, count, valid_bits, vector_lanes, values=True)
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(((count+3)//4)*64))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = []
  equal_arg = _ew_eq_mask(ops, arg, candidates_slot, current_slot, (diff, magnitude, unequal, equal), one, matrix_lanes)
  ops.append(RKEWOp(arg(selected), equal_arg, arg(valid_slot), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  selected_arg = _reduce_rows(ops, [arg(selected, candidate*vector_bytes) for candidate in range(window)], count, _EW_CFG[Ops.ADD])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), selected_arg, arg(int_tiles), count,
                      _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _lower_sort_index_selection(output:RKOutput) -> RKImage|None:
  """Lower stable sort's value/count match and coordinate sum entirely to DPU EW."""
  store, out_param, count, out_index, _ = output
  if count <= 0: return RKImage(RKTarget.RK3588)

  terms = _flatten_binary(store.src[1], Ops.ADD)
  parsed:list[tuple[int, tuple[UOp, UOp], tuple[UOp, UOp]]] = []
  for term in terms:
    weight, cast = 1, term
    if term.op is Ops.MUL:
      constants = [x for x in term.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.int]
      casts = [x for x in term.src if x.op is Ops.CAST]
      if len(constants) != 1 or len(casts) != 1: return None
      weight, cast = int(constants[0].arg), casts[0]
    if weight <= 0 or cast.op is not Ops.CAST or cast.dtype.scalar() is not dtypes.int or cast.src[0].op is not Ops.AND: return None
    pairs = [_load_equality(x) for x in cast.src[0].src]
    if len(pairs) != 2 or any(pair is None for pair in pairs): return None
    typed = [pair for pair in pairs if pair is not None]
    half_pairs = [pair for pair in typed if all(x.dtype.scalar() is dtypes.half for x in pair)]
    int_pairs = [pair for pair in typed if all(x.dtype.scalar() is dtypes.int for x in pair)]
    if len(half_pairs) != 1 or len(int_pairs) != 1: return None
    parsed.append((weight, half_pairs[0], int_pairs[0]))
  if (not parsed or max(weight for weight,_,_ in parsed) > 255 or
      {weight for weight,_,_ in parsed} != set(range(1, max(weight for weight,_,_ in parsed)+1))): return None
  parsed.sort(key=lambda item: item[0])

  if (half_split:=_split_load_pairs(tuple(pair for _,pair,_ in parsed))) is None or \
     (int_split:=_split_load_pairs(tuple(pair for _,_,pair in parsed))) is None: return None
  half_current, half_candidates = half_split
  int_current, int_candidates = int_split
  all_loads = (*half_candidates, half_current, *int_candidates, int_current)
  params = tuple(_root_param(load.src[0]) for load in all_loads)
  if any(param is None or param.src[0].op is not Ops.CONST for param in params): return None
  concrete = tuple(param for param in params if param is not None)
  half_candidate_params = concrete[:len(half_candidates)]
  half_current_param = concrete[len(half_candidates)]
  int_candidate_start = len(half_candidates)+1
  int_candidate_params = concrete[int_candidate_start:int_candidate_start+len(int_candidates)]
  int_current_param = concrete[-1]
  if (len({x.arg.slot for x in half_candidate_params}) != 1 or len({x.arg.slot for x in int_candidate_params}) != 1 or
      any(x.dtype.scalar() is not dtypes.half for x in (*half_candidate_params, half_current_param)) or
      any(x.dtype.scalar() is not dtypes.int for x in (*int_candidate_params, int_current_param))): return None

  try:
    half_candidate_offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in half_candidates)
    half_current_offsets = _gather_offsets(out_index, half_current.src[0].src[1], None, count)
    int_candidate_offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in int_candidates)
    int_current_offsets = _gather_offsets(out_index, int_current.src[0].src[1], None, count)
  except RuntimeError: return None
  if half_candidate_offsets != int_candidate_offsets: return None
  maps_and_sizes = (*zip(half_candidate_offsets, half_candidate_params), (half_current_offsets, half_current_param),
                    *zip(int_candidate_offsets, int_candidate_params), (int_current_offsets, int_current_param))
  if any(any(not 0 <= offset < int(param.src[0].arg) for offset in offsets) for offsets,param in maps_and_sizes): return None

  rows = len(parsed)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, rows)
  (zero, one, raw_candidate_count, raw_current_count, candidate_count, current_count, convert_tiles,
   candidate_value, current_value, weights, value_diff, value_magnitude, value_unequal, value_equal,
   count_diff, count_magnitude, count_unequal, count_equal, selected, weighted, int_tiles) = range(21)
  weight_bits = tuple((struct.unpack("<H", struct.pack("<e", float(weight)))[0],)*count for weight,_,_ in parsed)
  gathers = _stripe_gathers(half_candidate_params[0].arg.slot, candidate_value, count, half_candidate_offsets, vector_lanes)
  gathers += _stripe_gathers(int_candidate_params[0].arg.slot, raw_candidate_count, count, int_candidate_offsets, vector_lanes, itemsize=4)
  gathers += _stripe_gathers(half_candidate_params[0].arg.slot, weights, count, weight_bits, vector_lanes, values=True)
  gathers += _stripe_gathers(half_current_param.arg.slot, current_value, count, (half_current_offsets,)*rows, vector_lanes)
  gathers += _stripe_gathers(int_current_param.arg.slot, raw_current_count, count, (int_current_offsets,)*rows, vector_lanes, itemsize=4)
  scratch_sizes = [matrix_lanes*2]*21
  scratch_sizes[raw_candidate_count] = scratch_sizes[raw_current_count] = matrix_lanes*4
  scratch_sizes[convert_tiles] = ((matrix_lanes+3)//4)*64
  scratch_sizes[int_tiles] = ((count+3)//4)*64
  scratch = tuple(RKScratch(_scratch_bytes(size//2) if i not in (raw_candidate_count, raw_current_count, convert_tiles, int_tiles)
                            else size) for i,size in enumerate(scratch_sizes))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(candidate_count), arg(raw_candidate_count), arg(convert_tiles), matrix_lanes, _EW_CFG[Ops.MAX], int32_input=True),
                       RKEWOp(arg(current_count), arg(raw_current_count), arg(convert_tiles), matrix_lanes, _EW_CFG[Ops.MAX], int32_input=True)]
  value_equal_arg = _ew_eq_mask(ops, arg, candidate_value, current_value, (value_diff, value_magnitude, value_unequal, value_equal),
                                one, matrix_lanes)
  count_equal_arg = _ew_eq_mask(ops, arg, candidate_count, current_count, (count_diff, count_magnitude, count_unequal, count_equal),
                                one, matrix_lanes, (True, False))
  ops.extend((RKEWOp(arg(selected), value_equal_arg, count_equal_arg, matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(arg(weighted), arg(selected), arg(weights), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)))
  selected_arg = _reduce_rows(ops, [arg(weighted, row*vector_bytes) for row in range(rows)], count, _EW_CFG[Ops.ADD])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), selected_arg, arg(int_tiles), count,
                      _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _lower_sort_compare(output:RKOutput) -> RKImage|None:
  """Lower one static bitonic compare/swap pass with DPU MAX and MIN."""
  _, out_param, count, out_index, value = output
  if value.op is not Ops.WHERE: return None
  condition = value.src[0]
  if count <= 0: return RKImage(RKTarget.RK3588)

  def maximum(root:UOp, negated:bool) -> tuple[UOp, UOp]|None:
    if root.op is not Ops.MAX or root.arg is not None: return None
    parsed = [_half_candidate(x) for x in root.src]
    if len(parsed) != 2 or any(x is None or x[1] != negated for x in parsed): return None
    candidates = [x for x in parsed if x is not None]
    return candidates[0][0], candidates[1][0]
  def extreme(root:UOp) -> tuple[bool, tuple[UOp, UOp]]|None:
    if root.op is Ops.MAX and root.arg == _NATIVE_MIN:
      parsed = [_half_candidate(x) for x in root.src]
      if len(parsed) == 2 and all(x is not None and not x[1] for x in parsed):
        candidates = [x for x in parsed if x is not None]
        return False, (candidates[0][0], candidates[1][0])
    if (pair:=maximum(root, False)) is not None: return True, pair
    if root.op is Ops.MUL and len(root.src) == 2:
      constants = [x for x in root.src if x.op is Ops.CONST and float(x.arg) == -1.0]
      inner = [x for x in root.src if x.op is Ops.MAX]
      if len(constants) == len(inner) == 1 and (pair:=maximum(inner[0], True)) is not None: return False, pair
    return None

  true_extreme, false_extreme = extreme(value.src[1]), extreme(value.src[2])
  if (true_extreme is None or false_extreme is None or true_extreme[0] == false_extreme[0] or
      set(true_extreme[1]) != set(false_extreme[1]) or not _is_static_expr(condition)): return None
  pair = true_extreme[1]
  plans:list[RKGather] = []
  try:
    for scratch_slot,load in enumerate(pair):
      if load.src[0].op is not Ops.INDEX or (source:=_root_param(load.src[0])) is None or source.dtype.scalar() is not dtypes.half or \
         source.src[0].op is not Ops.CONST: return None
      offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count))
      if any(not 0 <= offset < int(source.src[0].arg) for offset in offsets): return None
      plans.append(RKGather(source.arg.slot, scratch_slot, count, offsets=offsets))
    choose_max:list[bool|None] = [None]*count
    for env in _iter_range_env(_index_ranges(out_index)):
      cache:dict[UOp, int|float|bool] = {}
      dst = _eval_int(out_index, env, cache)
      selected_kind = true_extreme if bool(_eval_expr(condition, env, cache)) else false_extreme
      if not 0 <= dst < count or (choose_max[dst] is not None and choose_max[dst] != selected_kind[0]): return None
      choose_max[dst] = selected_kind[0]
  except RuntimeError: return None
  if any(x is None for x in choose_max): return None
  choices = [bool(x) for x in choose_max]
  scratch = tuple(RKScratch(_scratch_bytes(count)) for _ in range(4))
  ops = (RKEWOp(RKArg(RKBufferKind.SCRATCH, 2), RKArg(RKBufferKind.SCRATCH, 0), RKArg(RKBufferKind.SCRATCH, 1),
                count, _EW_CFG[Ops.MAX]),
         RKEWOp(RKArg(RKBufferKind.SCRATCH, 3), RKArg(RKBufferKind.SCRATCH, 0), RKArg(RKBufferKind.SCRATCH, 1),
                count, _EW_CFG_MIN))
  max_offsets = tuple(i if choose else -1 for i,choose in enumerate(choices))
  min_offsets = tuple(i if not choose else -1 for i,choose in enumerate(choices))
  post = (RKGather(2, out_param.arg.slot, count, offsets=max_offsets, dst_kind=RKBufferKind.ARG,
                   src_kind=RKBufferKind.SCRATCH),
          RKGather(3, out_param.arg.slot, count, offsets=min_offsets, partial=True, dst_kind=RKBufferKind.ARG,
                   src_kind=RKBufferKind.SCRATCH))
  return RKImage(RKTarget.RK3588, scratch, gathers=tuple(plans), ew_ops=ops, post_gathers=post)

def _cumulative_index_image(out_slot:int, count:int, candidate_plans:tuple[RKGather, ...],
                            extrema_plans:tuple[RKGather, ...], negated_candidates:bool, axis_coords:list[int],
                            first_tie:bool=False, negate_extrema:bool=False,
                            candidate_coords:tuple[tuple[int, ...], ...]|None=None, index_limit:int|None=None) -> RKImage:
  """Emit a matrix equality/select reduction shared by unrolled and loop cumulative indices."""
  window = len(candidate_plans)
  limit = window if index_limit is None else index_limit
  constants = (0.0, 1.0, float(limit)) if first_tie else (0.0, 1.0)
  zero, one, candidate_arena = 0, 1, len(constants)
  extrema_slots = tuple(range(candidate_arena+1, candidate_arena+1+len(extrema_plans)))
  first_temp = candidate_arena+1+len(extrema_plans)
  coordinate_arena, selected_arena, diff, magnitude, unequal, equal, int_tiles = range(first_temp, first_temp+7)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  def materialize(plans:tuple[RKGather, ...], scratch_slot:int) -> tuple[RKGather, ...]:
    return tuple(RKGather(plan.src_index, scratch_slot, count, plan.base, plan.axes, plan.offsets, plan.fill_bits,
                          values=plan.values, partial=plan.partial, dst_stride=plan.dst_stride,
                          dst_addend=i*vector_lanes, itemsize=plan.itemsize) for i,plan in enumerate(plans))
  gathers = materialize(candidate_plans, candidate_arena)
  for plan,scratch_slot in zip(extrema_plans, extrema_slots): gathers += materialize((plan,)*window, scratch_slot)
  coordinate_bits = tuple(tuple(struct.unpack("<H", struct.pack("<e", float(limit-(candidate_coords[candidate][dst]
                                if candidate_coords is not None else candidate) if first_tie else
                                candidate+1 if candidate <= axis_coords[dst] else 0)))[0]
                                for dst in range(count)) for candidate in range(window))
  gathers += tuple(RKGather(candidate_plans[0].src_index, coordinate_arena, count, values=bits, dst_addend=candidate*vector_lanes)
                   for candidate,bits in enumerate(coordinate_bits))
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(((count+3)//4)*64))
  def args(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ew_ops:list[RKEWOp] = []
  if negate_extrema:
    for slot in extrema_slots: ew_ops.append(RKEWOp(args(slot), args(zero), args(slot), matrix_lanes, _EW_CFG[Ops.SUB]))
  extrema = args(extrema_slots[0])
  for i,slot in enumerate(extrema_slots[1:]):
    ew_ops.append(RKEWOp(extrema, extrema, args(slot), matrix_lanes, _EW_CFG[Ops.MAX],
                         submit_barrier=negate_extrema and i == 0, stateful=negate_extrema and i == 0))
  if negated_candidates:
    ew_ops.append(RKEWOp(args(diff), args(zero), args(candidate_arena), matrix_lanes, _EW_CFG[Ops.SUB],
                          submit_barrier=bool(ew_ops), stateful=bool(ew_ops)))
  equal_arg = _ew_eq_mask(ew_ops, args, diff if negated_candidates else candidate_arena, extrema_slots[0],
                          (magnitude, magnitude, unequal, equal), one, matrix_lanes, (bool(ew_ops), True))
  ew_ops.extend((RKEWOp(args(diff), equal_arg, args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=True, stateful=True),
                 RKEWOp(args(selected_arena), equal_arg, args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=True, stateful=True)))
  selected = _reduce_rows(ew_ops, [args(selected_arena, candidate*vector_bytes) for candidate in range(window)], count, _EW_CFG[Ops.MAX])
  ew_ops.append(RKEWOp(args(diff), args(2), selected, count, _EW_CFG[Ops.SUB]) if first_tie else
                RKEWOp(args(diff), selected, args(one), count, _EW_CFG[Ops.SUB]))
  ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), args(diff), args(int_tiles), count,
                       _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<"+"e"*len(constants), *constants), gathers=gathers, ew_ops=tuple(ew_ops))

def _half_candidate(root:UOp) -> tuple[UOp, bool]|None:
  if root.op is Ops.LOAD: return (root, False)
  if root.op is Ops.MUL and len(root.src) == 2:
    loads = [x for x in root.src if x.op is Ops.LOAD]
    constants = [x for x in root.src if x.op is Ops.CONST and float(x.arg) == -1.0]
    if len(loads) == len(constants) == 1: return (loads[0], True)
  return None

def _param_load_groups(nodes:Iterable[UOp]) -> dict[int, list[UOp]]:
  """Group unique FP16 parameter loads by argument slot."""
  grouped:dict[int, list[UOp]] = {}
  for load in nodes:
    if load.op is Ops.LOAD and load.dtype.scalar() is dtypes.half and load.src[0].op is Ops.INDEX and load.src[0].src[0].op is Ops.PARAM:
      slot = load.src[0].src[0].arg.slot
      if load not in grouped.setdefault(slot, []): grouped[slot].append(load)
  return grouped

def _first_tie_selection(value:UOp, extrema:UOp, ordered_exprs:list[UOp]) -> bool:
  """Verify Tinygrad's descending-coordinate INT32 MAX encoding for the first equal candidate."""
  window, nodes = len(ordered_exprs), value.toposort()
  equal_casts:list[UOp] = []
  for expr in ordered_exprs:
    matches = [u for u in nodes if u.op is Ops.CMPNE and (u.src == (expr, extrema) or u.src == (extrema, expr))]
    if len(matches) != 1: return False
    inversions = [u for u in nodes if u.op is Ops.CMPNE and matches[0] in u.src and
                  any(x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg) for x in u.src)]
    casts = [u for u in nodes if u.op is Ops.CAST and u.dtype.scalar() is dtypes.int and
             len(inversions) == 1 and u.src == (inversions[0],)]
    if len(casts) != 1: return False
    equal_casts.append(casts[0])
  int_roots = [u for u in nodes if u.op is Ops.MAX and u.dtype.scalar() is dtypes.int]
  if not int_roots: return False
  selected = max(int_roots, key=lambda x:len(_flatten_binary(x, Ops.MAX)))
  terms = _flatten_binary(selected, Ops.MAX)
  if len(terms) != window: return False
  for candidate_index,cast in enumerate(equal_casts):
    weight = window-candidate_index
    if not any(term is cast if weight == 1 else term.op is Ops.MUL and cast in term.src and
               any(x.op is Ops.CONST and int(x.arg) == weight for x in term.src) for term in terms): return False
  return value.op is Ops.ADD and any(x.op is Ops.CONST and int(x.arg) == window for x in value.src) and \
    any(x.op is Ops.MUL and selected in x.src and any(y.op is Ops.CONST and int(y.arg) == -1 for y in x.src) for x in value.src)

def _fp16_gather_plan(value:UOp, out_index:UOp, count:int, out_slot:int) -> RKGather|None:
  """Turn one FP16 lane expression into the raw gather used by specialized reductions."""
  leaf = _ew_leaf(value, out_index, count, out_slot)
  if isinstance(leaf, RKArg):
    if leaf.kind is not RKBufferKind.ARG or leaf.addend: return None
    return RKGather(leaf.index, 0, count, axes=((1, count, 1),) if count > 1 else ())
  if isinstance(leaf, RKGather): plan = leaf
  elif isinstance(leaf, tuple) and len(leaf) == 4 and isinstance(leaf[0], UOp):
    param, index, gate, fill_bits = leaf
    plan = _gather_plan(param.arg.slot, 0, out_index, index, gate, count, fill_bits)
  else: return None
  return plan if (plan.dst_kind is RKBufferKind.SCRATCH and plan.src_kind is RKBufferKind.ARG and plan.dst_index == 0 and
                  plan.dst_stride == 1 and plan.dst_addend == 0 and plan.itemsize == 2 and not plan.values and not plan.partial) else None

def _plan_offsets(plan:RKGather) -> tuple[int, ...]:
  if plan.offsets: return plan.offsets
  return tuple(plan.base + sum((lane//divisor % limit)*stride for divisor,limit,stride in plan.axes) for lane in range(plan.count))

def _descending_index_root(value:UOp) -> tuple[int, UOp]|None:
  """Parse `limit - MAX(weighted equality candidates)`."""
  if value.op is not Ops.ADD or len(value.src) != 2: return None
  limits = [x for x in value.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.int and int(x.arg) > 0]
  negatives = [x for x in value.src if x.op is Ops.MUL and any(y.op is Ops.CONST and int(y.arg) == -1 for y in x.src)]
  if len(limits) != 1 or len(negatives) != 1: return None
  selected = next((x for x in negatives[0].src if x.op is not Ops.CONST), None)
  return (int(limits[0].arg), selected) if selected is not None else None

def _weighted_equality(term:UOp) -> tuple[tuple[UOp, UOp], UOp]|None:
  if term.op is not Ops.MUL or len(term.src) != 2: return None
  casts = [x for x in term.src if x.op is Ops.CAST and x.dtype.scalar() is dtypes.int and len(x.src) == 1]
  if len(casts) != 1 or (pair:=_equality_pair(casts[0].src[0])) is None or any(x.dtype.scalar() is not dtypes.half for x in pair): return None
  weight = term.src[1] if term.src[0] is casts[0] else term.src[0]
  return pair, weight

def _pool_index_image(out_slot:int, count:int, spatial_size:int, source_count:int, plans:tuple[RKGather, ...],
                      extrema_plan:RKGather, weights:tuple[tuple[int, ...], ...]) -> RKImage|None:
  """Validate static pool lanes and emit their descending-coordinate first-tie selection."""
  if (not 2 <= len(plans) <= 2048 or spatial_size > 2048 or source_count < spatial_size or source_count % spatial_size or
      count % (source_count//spatial_size) or len(weights) != len(plans) or
      len({plan.src_index for plan in plans}) != 1 or extrema_plan.src_index == plans[0].src_index or
      sorted(_plan_offsets(extrema_plan)) != list(range(count))): return None
  coordinates:list[tuple[int, ...]] = []
  valid_lanes = [0] * count
  min_bits = struct.unpack("<H", struct.pack("<e", float(dtypes.half.min)))[0]
  for plan,row_weights in zip(plans, weights):
    offsets = _plan_offsets(plan)
    if len(offsets) != count or len(row_weights) != count: return None
    row:list[int] = []
    for lane,(offset,weight) in enumerate(zip(offsets, row_weights)):
      if offset < 0:
        if offset != -1 or plan.fill_bits != min_bits or weight != dtypes.int.min: return None
        row.append(spatial_size)
      else:
        if offset >= source_count or weight != spatial_size-offset%spatial_size: return None
        row.append(offset%spatial_size); valid_lanes[lane] += 1
    coordinates.append(tuple(row))
  if any(valid < 1 for valid in valid_lanes): return None
  return _cumulative_index_image(out_slot, count, plans, (extrema_plan,), False, [0]*count,
                                 first_tie=True, candidate_coords=tuple(coordinates), index_limit=spatial_size)

def _lower_unrolled_pool_index(output:RKOutput) -> RKImage|None:
  """Select MaxPool's first spatial index with raw gathers and DPU equality masks."""
  _, out_param, count, out_index, value = output
  if count <= 0 or (root:=_descending_index_root(value)) is None: return None
  spatial_size, selected = root
  terms = _flatten_binary(selected, Ops.MAX)
  parsed = [_weighted_equality(term) for term in terms]
  if not 2 <= len(parsed) <= 2048 or any(item is None for item in parsed): return None
  concrete = [item for item in parsed if item is not None]
  pairs = tuple(pair for pair,_ in concrete)
  common = set(pairs[0]).intersection(*(set(pair) for pair in pairs[1:]))
  if len(common) != 1: return None
  extrema = next(iter(common))
  candidates = tuple(next((x for x in pair if x is not extrema), None) for pair in pairs)
  if any(x is None for x in candidates) or len(set(candidates)) != len(candidates): return None

  try:
    extrema_plan = _fp16_gather_plan(extrema, out_index, count, out_param.arg.slot)
    candidate_plans = tuple(_fp16_gather_plan(candidate, out_index, count, out_param.arg.slot)
                            for candidate in candidates if candidate is not None)
  except RuntimeError: return None
  if extrema_plan is None or len(candidate_plans) != len(candidates) or any(plan is None for plan in candidate_plans): return None
  plans = tuple(plan for plan in candidate_plans if plan is not None)
  params = [u for u in value.toposort() if u.op is Ops.PARAM and u.arg.slot in (plans[0].src_index, extrema_plan.src_index)]
  source_params = [u for u in params if u.arg.slot == plans[0].src_index]
  extrema_params = [u for u in params if u.arg.slot == extrema_plan.src_index]
  if (len(set(source_params)) != 1 or len(set(extrema_params)) != 1 or any(p.dtype.scalar() is not dtypes.half for p in params) or
      any(p.src[0].op is not Ops.CONST for p in params)): return None
  source_count = int(source_params[0].src[0].arg)
  if int(extrema_params[0].src[0].arg) != count: return None
  try:
    weights = tuple(_static_int_vector(out_index, weight, count) for _,weight in concrete)
  except (OverflowError, RuntimeError): return None
  return _pool_index_image(out_param.arg.slot, count, spatial_size, source_count, plans, extrema_plan, weights)

def _lower_loop_pool_index(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Lower the one-register loop used by a global MaxPool returned index."""
  _, out_param, count, out_index, value = output
  if count != 1 or _index_ranges(out_index) or (root:=_descending_index_root(value)) is None: return None
  spatial_size, selected = root
  if spatial_size > 2048 or selected.op is not Ops.LOAD: return None
  ranges = [u for u in uops if u.op is Ops.RANGE]
  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  if len(ranges) != 1 or ranges[0].src[0].op is not Ops.CONST or len(local_stores) != 2: return None
  reduce_range, window = ranges[0], int(ranges[0].src[0].arg)
  initial = next((u for u in local_stores if u.src[1].op is Ops.CONST and int(u.src[1].arg) == dtypes.int.min), None)
  update = next((u for u in local_stores if u.src[1].op is Ops.MAX and reduce_range in u.toposort()), None)
  if initial is None or update is None or not 2 <= window <= 2048: return None
  weighted = next((x for x in update.src[1].src if x.op is Ops.MUL), None)
  accumulator = [x for x in update.src[1].src if x.op is Ops.LOAD and _root_param(x.src[0]) is None]
  local_buffers = [{x for x in node.toposort() if x.op is Ops.BUFFER} for node in (initial.src[0], update.src[0], selected)]
  if (weighted is None or len(accumulator) != 1 or set(update.src[1].src) != {accumulator[0], weighted} or
      any(len(buffers) != 1 for buffers in local_buffers) or len(set.union(*local_buffers)) != 1 or
      (parsed:=_weighted_equality(weighted)) is None): return None
  pair, weight = parsed
  candidates = [x for x in pair if reduce_range in x.toposort()]
  extrema = [x for x in pair if reduce_range not in x.toposort()]
  if len(candidates) != 1 or len(extrema) != 1 or candidates[0].op is not Ops.LOAD or extrema[0].op is not Ops.LOAD: return None
  candidate_param, candidate_index = _root_param(candidates[0].src[0]), candidates[0].src[0].src[1]
  extrema_plan = _fp16_gather_plan(extrema[0], out_index, 1, out_param.arg.slot)
  extrema_params = [u for u in uops if u.op is Ops.PARAM and extrema_plan is not None and u.arg.slot == extrema_plan.src_index]
  if (candidate_param is None or candidate_param.src[0].op is not Ops.CONST or candidate_param.dtype.scalar() is not dtypes.half or
      extrema_plan is None or int(candidate_param.src[0].arg) != spatial_size or len(set(extrema_params)) != 1 or
      extrema_params[0].dtype.scalar() is not dtypes.half or extrema_params[0].src[0].op is not Ops.CONST or
      int(extrema_params[0].src[0].arg) != 1): return None
  try:
    offsets = tuple(_eval_int(candidate_index, {reduce_range:i}) for i in range(window))
    weights = tuple(_eval_int(weight, {reduce_range:i}) for i in range(window))
  except RuntimeError: return None
  if len(set(offsets)) != window: return None
  plans = tuple(RKGather(candidate_param.arg.slot, 0, 1, base=offset) for offset in offsets)
  return _pool_index_image(out_param.arg.slot, 1, spatial_size, spatial_size, plans, extrema_plan,
                           tuple((weight,) for weight in weights))

def _lower_unrolled_max_unpool(output:RKOutput) -> RKImage|None:
  """Scatter bounded MaxUnpool lanes through DPU INT32 equality and FP16 reduction."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  parsed:list[tuple[UOp, UOp, UOp]] = []
  for term in _flatten_binary(value, Ops.ADD):
    where = _strip_cast(term)
    if (where.op is not Ops.WHERE or _strip_cast(where.src[0]).op is not Ops.CMPNE or
        where.src[1].op is not Ops.CONST or float(where.src[1].arg) != 0.0): return None
    condition, selected_load = _strip_cast(where.src[0]), _strip_cast(where.src[2])
    index_loads = [x for x in condition.src if x.op is Ops.LOAD and x.dtype.scalar() is dtypes.int]
    coordinate_exprs = [x for x in condition.src if _is_static_expr(x)]
    if (len(index_loads) != 1 or len(coordinate_exprs) != 1 or selected_load.op is not Ops.LOAD or
        selected_load.dtype.scalar() is not dtypes.half or selected_load.src[0].op is not Ops.INDEX or
        index_loads[0].src[0].op is not Ops.INDEX or selected_load.src[0].src[1].key != index_loads[0].src[0].src[1].key): return None
    parsed.append((index_loads[0], selected_load, coordinate_exprs[0]))
  pooled = len(parsed)
  if not 2 <= pooled <= 255: return None
  index_params = [_root_param(index.src[0]) for index,_,_ in parsed]
  value_params = [_root_param(value.src[0]) for _,value,_ in parsed]
  if (any(param is None or param.src[0].op is not Ops.CONST for param in (*index_params,*value_params)) or
      len({param.arg.slot for param in index_params if param is not None}) != 1 or
      len({param.arg.slot for param in value_params if param is not None}) != 1): return None
  index_param, value_param = index_params[0], value_params[0]
  assert index_param is not None and value_param is not None
  source_count = int(index_param.src[0].arg)
  if (index_param.dtype.scalar() is not dtypes.int or value_param.dtype.scalar() is not dtypes.half or
      int(value_param.src[0].arg) != source_count or source_count % pooled): return None
  planes = source_count//pooled
  if not planes or count % planes: return None
  out_spatial = count//planes
  if out_spatial > 2048: return None
  try:
    rows = [(_gather_offsets(out_index, index.src[0].src[1], None, count),
             _gather_offsets(out_index, selected.src[0].src[1], None, count),
             _static_int_vector(out_index, coordinate, count)) for index,selected,coordinate in parsed]
  except RuntimeError: return None
  if any(indexes != values or coords != tuple(lane%out_spatial for lane in range(count)) for indexes,values,coords in rows): return None
  rows.sort(key=lambda row:row[0])
  if any(sorted(indexes[lane] for indexes,_,_ in rows) != list(range(lane//out_spatial*pooled, (lane//out_spatial+1)*pooled))
         for lane in range(count)): return None

  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, pooled)
  zero, one, compact_index, convert_tiles, half_index, coordinate_slot, values, diff, magnitude, unequal, equal, selected_slot = range(12)
  scratch_sizes = [matrix_lanes*2] * 12
  scratch_sizes[compact_index], scratch_sizes[convert_tiles] = source_count*2, ((source_count+3)//4)*64
  scratch = tuple(RKScratch(size) for size in scratch_sizes)
  coordinate_bits = tuple(struct.unpack("<H", struct.pack("<e", float(lane%out_spatial)))[0] for lane in range(count))
  gathers:tuple[RKGather, ...] = (); mid_gathers:tuple[RKGather, ...] = ()
  for row,(offsets,_,_) in enumerate(rows):
    gathers += (RKGather(value_param.arg.slot, values, count, offsets=offsets, dst_addend=row*vector_lanes),
                RKGather(index_param.arg.slot, coordinate_slot, count, values=coordinate_bits, dst_addend=row*vector_lanes))
    mid_gathers += (RKGather(compact_index, half_index, count, offsets=offsets, dst_addend=row*vector_lanes,
                             src_kind=RKBufferKind.SCRATCH),)
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(compact_index), RKArg(RKBufferKind.ARG, index_param.arg.slot), arg(convert_tiles), source_count,
                              _EW_CFG[Ops.MAX], int32_input=True)]
  equal_arg = _ew_eq_mask(ops, arg, half_index, coordinate_slot, (diff, magnitude, unequal, equal), one, matrix_lanes)
  ops.append(RKEWOp(arg(selected_slot), arg(values), equal_arg, matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  reduced = _reduce_rows(ops, [arg(selected_slot, row*vector_bytes) for row in range(pooled)], count, _EW_CFG[Ops.ADD])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), reduced, arg(zero), count, _EW_CFG[Ops.ADD]))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops),
                 mid_gathers=mid_gathers, gather_after=1)

@dataclass(frozen=True)
class RKArgMatch:
  source_slot:int; source_count:int; extrema:UOp; candidates:tuple[tuple[UOp, UOp, bool], ...]; extrema_plans:tuple[RKGather, ...]|None = None

def _lower_unrolled_arg_extrema(output:RKOutput) -> RKImage|None:
  """Share first-tie validation and gather packing across fused and split ArgMax/ArgMin graphs."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  nodes = value.toposort()

  def fused() -> RKArgMatch|None:
    roots:list[tuple[UOp, list[UOp], list[tuple[UOp, bool]]]] = []
    for root in nodes:
      if root.op is not Ops.MAX or root.dtype.scalar() is not dtypes.half: continue
      leaves = _flatten_binary(root, Ops.MAX); parsed_leaves = [_half_candidate(leaf) for leaf in leaves]
      if all(x is not None for x in parsed_leaves): roots.append((root, leaves, [x for x in parsed_leaves if x is not None]))
    if not roots: return None
    extrema, exprs, parsed = max(roots, key=lambda x:len(x[1]))
    loads = [load for load,_ in parsed]
    if not loads or loads[0].src[0].op is not Ops.INDEX or (source:=_root_param(loads[0].src[0])) is None or source.src[0].op is not Ops.CONST:
      return None
    return RKArgMatch(source.arg.slot, int(source.src[0].arg), extrema,
                      tuple((expr, load, negated) for expr,(load,negated) in zip(exprs, parsed)))

  def split() -> RKArgMatch|None:
    by_slot = _param_load_groups(nodes)
    extrema_groups = [(slot, group[0]) for slot,group in by_slot.items() if len(group) == 1 and
                      group[0].src[0].src[0].src[0].op is Ops.CONST and int(group[0].src[0].src[0].src[0].arg) == count]
    if len(extrema_groups) != 1 or len(by_slot) != 2: return None
    extrema_slot, extrema = extrema_groups[0]
    data_slot, loads = next((slot,group) for slot,group in by_slot.items() if slot != extrema_slot)
    data_param = loads[0].src[0].src[0]
    if data_param.src[0].op is not Ops.CONST: return None
    candidates:list[tuple[UOp, UOp, bool]] = []
    for cmp in [u for u in nodes if u.op is Ops.CMPNE and extrema in u.src]:
      expr = cmp.src[1] if cmp.src[0] is extrema else cmp.src[0]
      if (parsed:=_half_candidate(expr)) is not None: candidates.append((expr, *parsed))
    if {load for _,load,_ in candidates} != set(loads): return None
    try: extrema_offsets = tuple(_gather_offsets(out_index, extrema.src[0].src[1], None, count))
    except RuntimeError: return None
    if sorted(extrema_offsets) != list(range(count)): return None
    return RKArgMatch(data_slot, int(data_param.src[0].arg), extrema, tuple(candidates),
                      (RKGather(extrema_slot, 0, count, offsets=extrema_offsets),))

  if (match:=fused()) is None and (match:=split()) is None: return None
  signs, loads = {negated for _,_,negated in match.candidates}, [load for _,load,_ in match.candidates]
  if (len(match.candidates) < 2 or len(signs) != 1 or len(set(loads)) != len(loads) or
      any(load.src[0].op is not Ops.INDEX or (source:=_root_param(load.src[0])) is None or
          source.arg.slot != match.source_slot for load in loads)): return None
  negated, window = signs.pop(), len(loads)
  if match.source_count != count*window or window > 2048: return None
  try: offsets = [tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for load in loads]
  except RuntimeError: return None
  ordered = sorted(zip(offsets, (expr for expr,_,_ in match.candidates)), key=lambda x:x[0])
  if (sorted(offset for row,_ in ordered for offset in row) != list(range(match.source_count)) or
      not _first_tie_selection(value, match.extrema, [expr for _,expr in ordered])): return None
  plans = tuple(RKGather(match.source_slot, 0, count, offsets=row) for row,_ in ordered)
  return _cumulative_index_image(out_param.arg.slot, count, plans, plans if match.extrema_plans is None else match.extrema_plans,
                                 negated, [window-1]*count, first_tie=True, negate_extrema=negated and match.extrema_plans is None)

def _lower_loop_arg_extrema_index(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Lower the two-register-loop graph used by a padded global FP16 ArgMax/ArgMin."""
  _, out_param, count, out_index, value = output
  if count != 1 or _index_ranges(out_index): return None
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if len(ranges) != 2 or any(r.src[0].op is not Ops.CONST for r in ranges): return None
  first_range, second_range = ranges
  window = int(first_range.src[0].arg)
  if not 2 <= window <= 2048 or int(second_range.src[0].arg) != window: return None
  input_params = [u for u in uops if u.op is Ops.PARAM and u.dtype.scalar() is dtypes.half and
                  u.src[0].op is Ops.CONST and int(u.src[0].arg) == window]
  if len(input_params) != 1: return None
  source = input_params[0]
  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  initial_half = [u for u in local_stores if u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() is dtypes.half and
                  math.isinf(float(u.src[1].arg)) and float(u.src[1].arg) < 0]
  initial_int = [u for u in local_stores if u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() is dtypes.int and
                 int(u.src[1].arg) == -(1 << 31)]
  half_updates = [u for u in local_stores if u.src[1].op is Ops.MAX and u.src[1].dtype.scalar() is dtypes.half and
                  first_range in u.toposort()]
  int_updates = [u for u in local_stores if u.src[1].op is Ops.MAX and u.src[1].dtype.scalar() is dtypes.int and
                 second_range in u.toposort()]
  if not all(len(x) == 1 for x in (initial_half, initial_int, half_updates, int_updates)) or len(local_stores) != 4: return None

  def global_candidate(exprs:list[UOp], reduce_range:UOp) -> tuple[UOp, bool]|None:
    parsed = [(expr, candidate) for expr in exprs if (candidate:=_half_candidate(expr)) is not None and
              _root_param(candidate[0].src[0]) is source]
    if len(parsed) != 1: return None
    expr, (load, negated) = parsed[0]
    if load.src[0].op is not Ops.INDEX or load.src[0].src[1] is not reduce_range: return None
    try:
      if [_eval_int(load.src[0].src[1], {reduce_range:i}) for i in range(window)] != list(range(window)): return None
    except RuntimeError: return None
    return expr, negated

  half_candidate = global_candidate(list(half_updates[0].src[1].src), first_range)
  int_nodes = int_updates[0].src[1].toposort()
  comparisons = [u for u in int_nodes if u.op is Ops.CMPNE and u.dtype.scalar() is dtypes.bool]
  inner_cmps = [u for u in comparisons if any(_half_candidate(x) is not None for x in u.src)]
  if half_candidate is None or len(inner_cmps) != 1: return None
  cmp = inner_cmps[0]
  second_candidate = global_candidate(list(cmp.src), second_range)
  if second_candidate is None or second_candidate[1] != half_candidate[1]: return None
  extrema_operands = [x for x in cmp.src if x is not second_candidate[0]]
  if len(extrema_operands) != 1 or extrema_operands[0].op is not Ops.LOAD or _root_param(extrema_operands[0].src[0]) is not None:
    return None
  inversions = [u for u in comparisons if cmp in u.src and any(x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg) for x in u.src)]
  casts = [u for u in int_nodes if u.op is Ops.CAST and u.dtype.scalar() is dtypes.int and
           len(inversions) == 1 and u.src == (inversions[0],)]
  if len(casts) != 1: return None
  weighted = [u for u in int_nodes if u.op is Ops.MUL and u.dtype.scalar() is dtypes.int and casts[0] in u.src]
  if len(weighted) != 1: return None
  coordinate = weighted[0].src[1] if weighted[0].src[0] is casts[0] else weighted[0].src[0]
  try:
    if [_eval_int(coordinate, {second_range:i}) for i in range(window)] != list(range(window, 0, -1)): return None
  except RuntimeError: return None
  final_negative = [x for x in value.src if x.op is Ops.MUL and
                    any(y.op is Ops.LOAD and y.dtype.scalar() is dtypes.int and _root_param(y.src[0]) is None for y in x.src) and
                    any(y.op is Ops.CONST and int(y.arg) == -1 for y in x.src)]
  if (value.op is not Ops.ADD or len(final_negative) != 1 or
      not any(x.op is Ops.CONST and int(x.arg) == window for x in value.src)): return None
  candidate_plans = tuple(RKGather(source.arg.slot, 0, 1, base=i) for i in range(window))
  return _cumulative_index_image(out_param.arg.slot, 1, candidate_plans, candidate_plans,
                                 half_candidate[1], [window-1], first_tie=True, negate_extrema=half_candidate[1])

def _lower_cumulative_extrema_index(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Select unrolled cumulative MAX/MIN axis coordinates with DPU equality masks."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  nodes = value.toposort()
  if not any(u.op is Ops.CMPNE for u in nodes): return None
  by_slot = _param_load_groups(nodes)
  extrema_groups = [(slot, group[0]) for slot,group in by_slot.items() if len(group) == 1 and group[0].src[0].src[1].key == out_index.key]
  if len(extrema_groups) != 1 or len(by_slot) != 2: return None
  extrema_slot, _ = extrema_groups[0]
  data_slot, candidates = next((slot,group) for slot,group in by_slot.items() if slot != extrema_slot)
  def directly_negated(load:UOp) -> bool:
    return any(node.op is Ops.MUL and load in node.src and any(src.op is Ops.CONST and float(src.arg) == -1.0 for src in node.src)
               for node in nodes)
  candidate_signs = {directly_negated(load) for load in candidates}
  if len(candidate_signs) != 1: return None
  negated_candidates = candidate_signs.pop()
  params = [u for u in uops if u.op is Ops.PARAM and u.arg.slot in (data_slot, extrema_slot)]
  if len(params) != 2 or any(u.src[0].op is not Ops.CONST or int(u.src[0].arg) != count for u in params): return None
  try:
    candidate_offsets = [tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for load in candidates]
  except RuntimeError: return None
  if not candidate_offsets or any(any(offset < 0 for offset in offsets) for offsets in candidate_offsets): return None
  candidate_offsets.sort()
  window = len(candidate_offsets)
  if window > 2048 or count % window or any(len({offsets[dst] for offsets in candidate_offsets}) != window for dst in range(count)): return None
  axis_coords = [dst % window for dst in range(count)]
  current_offsets = [candidate_offsets[axis_coords[dst]][dst] for dst in range(count)]
  if sorted(current_offsets) != list(range(count)) or any(candidate_offsets[candidate][dst] >= candidate_offsets[candidate+1][dst]
                                                            for candidate in range(window-1) for dst in range(count)): return None

  extrema_plan = RKGather(extrema_slot, 0, count, 0, ((1, count, 1),))
  candidate_plans = tuple(RKGather(data_slot, 0, count, offsets=offsets) for offsets in candidate_offsets)
  return _cumulative_index_image(out_param.arg.slot, count, candidate_plans,
                                 (extrema_plan,), negated_candidates, axis_coords)

def _lower_cumulative_extrema_index_loop(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Lower Tinygrad's bounded-loop form used by the padded length-1022 scan."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_ranges = _index_ranges(out_index)
  if len(out_ranges) != 1 or out_ranges[0].src[0].op is not Ops.CONST or int(out_ranges[0].src[0].arg) != count: return None
  half_cmps = [u for u in uops if u.op is Ops.CMPNE and len(u.src) == 2 and all(x.dtype.scalar() is dtypes.half for x in u.src)]
  if len(half_cmps) != 1: return None
  lhs, rhs = half_cmps[0].src
  extrema_expr, candidate_expr = (lhs, rhs) if lhs.op is Ops.MAX else (rhs, lhs)
  if extrema_expr.op is not Ops.MAX: return None
  negated_candidates = False
  if candidate_expr.op is Ops.MUL:
    constants = [x for x in candidate_expr.src if x.op is Ops.CONST and float(x.arg) == -1.0]
    loads = [x for x in candidate_expr.src if x.op is Ops.LOAD]
    if len(constants) != 1 or len(loads) != 1: return None
    candidate_expr, negated_candidates = loads[0], True
  if candidate_expr.op is not Ops.LOAD or candidate_expr.src[0].op is not Ops.INDEX or candidate_expr.src[0].src[0].op is not Ops.PARAM: return None
  candidate_param, candidate_index = candidate_expr.src[0].src[:2]
  out_range = out_ranges[0]
  if (candidate_param.src[0].op is not Ops.CONST or int(candidate_param.src[0].arg) != count or candidate_index.op is not Ops.RANGE or
      candidate_index.src[0].op is not Ops.CONST or int(candidate_index.src[0].arg) != count or out_range not in candidate_index.src[1:]): return None
  reduce_range = candidate_index
  prefix_cmps = [u for u in uops if u.op is Ops.CMPLT and u.src == (out_range, reduce_range)]
  if len(prefix_cmps) != 1: return None
  def contains(root:UOp, *targets:UOp) -> bool:
    nodes = set(root.toposort()); return all(target in nodes for target in targets)
  gates = [u for u in uops if u.op is Ops.AND and contains(u, half_cmps[0], prefix_cmps[0])]
  if len(gates) != 1: return None
  index_maxes = [u for u in uops if u.op is Ops.MAX and u.dtype.scalar() is dtypes.int and contains(u, gates[0], reduce_range)]
  if len(index_maxes) != 1 or index_maxes[0] not in value.toposort(): return None
  extrema_loads = [u for u in extrema_expr.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.half and
                   u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM]
  if len(extrema_loads) != 2 or any(any(r is not out_range for r in _index_ranges(load.src[0].src[1])) for load in extrema_loads): return None
  try:
    candidate_values = [_eval_int(candidate_index, {reduce_range: candidate}) for candidate in range(count)]
    if sorted(candidate_values) != list(range(count)): return None
    candidate_plans = tuple(RKGather(candidate_param.arg.slot, 0, count, base=offset) for offset in candidate_values)
    extrema_plans = tuple(_gather_plan(load.src[0].src[0].arg.slot, 0, out_index, load.src[0].src[1], None, count)
                          for load in extrema_loads)
  except RuntimeError: return None
  for load,plan in zip(extrema_loads, extrema_plans):
    source_size = int(load.src[0].src[0].src[0].arg)
    if plan.offsets and any(not 0 <= offset < source_size for offset in plan.offsets): return None
  return _cumulative_index_image(out_param.arg.slot, count, candidate_plans,
                                 extrema_plans, negated_candidates, list(range(count)))

def _bool_reduction_image(out_slot:int, count:int, source_slot:int, offsets:tuple[tuple[int, ...], ...], op:Ops) -> RKImage:
  window = len(offsets)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one, candidates_slot, diff, magnitude, unequal, equal, int_tiles = range(8)
  gathers = _stripe_gathers(source_slot, candidates_slot, count, offsets, vector_lanes)
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(((count+3)//4)*64))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = []
  _ew_eq_mask(ops, arg, candidates_slot, zero, (diff, magnitude, unequal, equal), one, matrix_lanes)
  selected = _reduce_rows(ops, [arg(unequal, row*vector_bytes) for row in range(window)], count,
                          _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), selected, arg(int_tiles), count, _EW_CFG[Ops.MAX],
                    stateful=True, int32_output=True, bool_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _nonzero_load(term:UOp) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtypes.half and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and float(zero.arg) == 0.0]
  return candidates[0] if len(candidates) == 1 else None

def _lower_unrolled_bool_reduction(output:RKOutput) -> RKImage|None:
  """Reduce a canonical unrolled FP16 nonzero tree with balanced DPU EW MAX/MUL stages."""
  _, out_param, count, out_index, root = output
  root = _unwrap_condition(root)
  if root.op not in (Ops.OR, Ops.AND): return None
  loads = [_nonzero_load(term) for term in _flatten_binary(root, root.op)]
  if any(load is None for load in loads): return None
  concrete = [load for load in loads if load is not None]
  params = [_root_param(load.src[0]) for load in concrete]
  if (not concrete or any(param is None or param.src[0].op is not Ops.CONST for param in params) or
      len({param.arg.slot for param in params if param is not None}) != 1): return None
  source = params[0]; assert source is not None
  if count <= 0: return RKImage(RKTarget.RK3588)
  try: offsets = tuple(tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for load in concrete)
  except RuntimeError: return None
  source_count = int(source.src[0].arg)
  if source_count != count*len(concrete) or sorted(offset for row in offsets for offset in row) != list(range(source_count)): return None
  return _bool_reduction_image(out_param.arg.slot, count, source.arg.slot, offsets, root.op)

def _lower_loop_bool_reduction(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Lower the register-loop form of FP16 any/all through the same balanced DPU EW image."""
  store, out_param, _, _, root = output
  nodes = list(root.toposort())
  if (shape:=_loop_reduction_shape(store, out_param, nodes)) is None or _local_load(root) is None: return None
  rows, envs, reduce_range, groups = shape
  local_stores = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  updates = [u for u in local_stores if reduce_range in u.toposort()]
  if len(local_stores) != 2 or len(updates) != 1: return None
  update = _unwrap_condition(updates[0].src[1])
  if update.op not in (Ops.OR, Ops.AND): return None
  acc = next((x for x in update.src if _local_load(x) is not None), None)
  predicate = update.src[1 if update.src[0] is acc else 0] if acc is not None else None
  if predicate is None or (load:=_nonzero_load(predicate)) is None: return None
  source = _root_param(load.src[0])
  identity = update.op is Ops.AND
  initials = [u for u in local_stores if u is not updates[0] and u.src[1].op is Ops.CONST and
              u.src[1].dtype.scalar() is dtypes.bool and bool(u.src[1].arg) == identity]
  if len(initials) != 1 or source is None or source.src[0].op is not Ops.CONST: return None
  try:
    offsets = tuple(tuple(_eval_int(load.src[0].src[1], {**env, reduce_range:group}) for env in envs) for group in range(groups))
  except RuntimeError: return None
  source_count = int(source.src[0].arg)
  if source_count != rows*groups or sorted(offset for row in offsets for offset in row) != list(range(source_count)): return None
  return _bool_reduction_image(out_param.arg.slot, rows, source.arg.slot, offsets, update.op)

def _typed_int_image(output:RKOutput, value:UOp, bool_output:bool=False) -> RKImage:
  """Lower an exact FP16 integer expression and expose its DPU-converted INT32 lanes."""
  store, out_param, count, _, _ = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_slot = out_param.arg.slot
  half_param = out_param.replace(dtype=dtypes.half, arg=replace(out_param.arg, dtype=dtypes.half))
  half_index = store.src[0].replace(dtype=dtypes.half, src=(half_param, *store.src[0].src[1:]))
  image = lower_ew(list(store.replace(src=(half_index, value, *store.src[2:])).toposort()))
  terminal = [i for i,op in enumerate(image.ew_ops) if op.dst.kind is RKBufferKind.ARG and op.dst.index == out_slot]
  if terminal != [len(image.ew_ops)-1] or image.fill is not None or image.mid_gathers or image.post_gathers:
    raise RuntimeError("RKPLAN_REJECT:predicate_terminal")
  result_slot, tiles_slot = len(image.scratch), len(image.scratch)+1
  result, tiles = RKArg(RKBufferKind.SCRATCH, result_slot), RKArg(RKBufferKind.SCRATCH, tiles_slot)
  ops = (*image.ew_ops[:-1], replace(image.ew_ops[-1], dst=result),
         RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, tiles, count, _EW_CFG[Ops.MAX],
                stateful=True, int32_output=True, bool_output=bool_output))
  return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(((count+3)//4)*64)), ew_ops=ops)

def _ieee_comparison_mask(root:UOp) -> UOp|None:
  """Build an IEEE-correct FP16 comparison mask without evaluating tensor values on the host."""
  one = UOp.const(1.0, dtypes.half)
  def inverse(value:UOp) -> UOp: return one.alu(Ops.SUB, value)
  def numeric(value:UOp) -> UOp|None:
    value = _unwrap_condition(value)
    if value.dtype.scalar() not in (dtypes.half, dtypes.float): return None
    loads = [u for u in value.toposort() if u.op is Ops.LOAD]
    params = [_root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None for load in loads]
    if any(param is None or param.dtype.scalar() is not dtypes.half for param in params): return None
    return value if value.dtype.scalar() is dtypes.half else value.cast(dtypes.half)
  def classes(value:UOp) -> tuple[UOp, UOp, UOp, UOp]:
    high = _positive_mask(value.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
    low = _positive_mask(UOp.const(-65504.0, dtypes.half).alu(Ops.SUB, value))
    nan = _mask_mul(high, low)
    return nan, high.alu(Ops.SUB, nan), low.alu(Ops.SUB, nan), inverse(high.alu(Ops.MAX, low))
  def atom(op:Ops, lhs:UOp, rhs:UOp, invert:bool=False) -> UOp|None:
    if (left:=numeric(lhs)) is None or (right:=numeric(rhs)) is None: return None
    lhs_nan, lhs_pos, lhs_neg, lhs_finite = classes(left)
    rhs_nan, rhs_pos, rhs_neg, rhs_finite = classes(right)
    positive = _positive_mask(right.alu(Ops.SUB, left))
    if op is Ops.CMPLT:
      valid = inverse(lhs_nan.alu(Ops.MAX, rhs_nan))
      forced = _mask_mul(lhs_neg, inverse(rhs_neg)).alu(Ops.MAX, _mask_mul(rhs_pos, inverse(lhs_pos)))
      finite = _mask_mul(_mask_mul(lhs_finite, rhs_finite), positive)
      comparison = forced.alu(Ops.MAX, finite)
      return _mask_mul(valid, inverse(comparison) if invert else comparison)
    unequal = positive.alu(Ops.MAX, _positive_mask(left.alu(Ops.SUB, right)))
    finite_equal = _mask_mul(_mask_mul(lhs_finite, rhs_finite), inverse(unequal))
    equal = finite_equal.alu(Ops.MAX, _mask_mul(lhs_pos, rhs_pos)).alu(Ops.MAX, _mask_mul(lhs_neg, rhs_neg))
    return inverse(equal)
  def mask(value:UOp) -> UOp|None:
    value = _unwrap_condition(value)
    if value.op is Ops.CONST and value.dtype.scalar() is dtypes.bool: return UOp.const(float(bool(value.arg)), dtypes.half)
    if value.op is Ops.CMPNE:
      for expression, marker in (value.src, value.src[::-1]):
        marker = _unwrap_condition(marker)
        if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool:
          expression = _unwrap_condition(expression)
          if bool(marker.arg) and expression.op is Ops.CMPLT:
            return atom(Ops.CMPLT, expression.src[0], expression.src[1], invert=True)
          if (inner:=mask(expression)) is None: return None
          return inverse(inner) if bool(marker.arg) else inner
    if value.op in (Ops.CMPLT, Ops.CMPNE): return atom(value.op, value.src[0], value.src[1])
    if value.op in (Ops.OR, Ops.AND, Ops.XOR):
      lhs, rhs = mask(value.src[0]), mask(value.src[1])
      if lhs is None or rhs is None: return None
      if value.op is Ops.OR: return lhs.alu(Ops.MAX, rhs)
      if value.op is Ops.AND: return _mask_mul(lhs, rhs)
      delta = lhs.alu(Ops.SUB, rhs)
      return UOp(Ops.MAX, dtypes.half, src=(delta, delta), arg=_NATIVE_ABS)
    return None
  return mask(root)

def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if (load:=_nonzero_load(root)) is None: return None
  magnitude = UOp(Ops.MAX, dtypes.half, src=(load, load), arg=_NATIVE_ABS)
  return _positive_mask(magnitude)

def _lower_int_where(output:RKOutput) -> RKImage|None:
  """Select two exactly representable INT32 constants from an FP16 comparison and convert on DPU."""
  root = output[4]
  if root.op is not Ops.WHERE or root.dtype.scalar() is not dtypes.int: return None
  if (condition:=_ieee_comparison_mask(root.src[0])) is None: return None
  arms = root.src[1:]
  if any(arm.op is not Ops.CONST or arm.dtype.scalar() is not dtypes.int for arm in arms): return None
  yes, no = (int(arm.arg) for arm in arms)
  delta = yes-no
  try: exact = all(_eval_cast(value, dtypes.half) == value for value in (no, delta))
  except (OverflowError, struct.error): return None
  if not exact: return None
  return _typed_int_image(output, condition.alu(Ops.MUL, UOp.const(float(delta), dtypes.half)).alu(
    Ops.ADD, UOp.const(float(no), dtypes.half)))

def _lower_ieee_predicate(output:RKOutput) -> RKImage|None:
  """Classify FP16 NaN/infinity on DPU and expose the final 0/1 mask through the bool ABI."""
  root = output[4]

  def logical_not(u:UOp) -> UOp|None:
    if u.op is not Ops.CMPNE: return None
    for value, marker in (u.src, u.src[::-1]):
      if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg): return value
    return None

  def atom(u:UOp) -> tuple[UOp, str]|None:
    if u.op is Ops.CMPNE and u.src[0].key == u.src[1].key and u.src[0].op is Ops.LOAD and \
       u.src[0].dtype.scalar() is dtypes.half: return u.src[0], "nan"
    if (unequal:=logical_not(u)) is None or unequal.op is not Ops.CMPNE: return None
    for load, constant in (unequal.src, unequal.src[::-1]):
      if (load.op is Ops.LOAD and load.dtype.scalar() is dtypes.half and constant.op is Ops.CONST and
          math.isinf(value:=float(constant.arg))): return load, "positive_inf" if value > 0 else "negative_inf"
    return None

  def union(u:UOp) -> list[tuple[UOp, str]]|None:
    if (parsed:=atom(u)) is not None: return [parsed]
    if u.op is not Ops.OR: return None
    lhs, rhs = union(u.src[0]), union(u.src[1])
    return None if lhs is None or rhs is None else lhs+rhs

  matches, inverted = union(root), False
  if matches is None and (inner:=logical_not(root)) is not None: matches, inverted = union(inner), True
  if matches is None or len({load.key for load,_ in matches}) != 1: return None
  tags = frozenset(tag for _,tag in matches)
  if inverted:
    if tags != {"nan", "positive_inf", "negative_inf"}: return None
    kind = "finite"
  else:
    kinds = {frozenset(("nan",)):"nan", frozenset(("positive_inf",)):"positive_inf",
             frozenset(("negative_inf",)):"negative_inf",
             frozenset(("positive_inf", "negative_inf")):"inf"}
    if tags not in kinds: return None
    kind = kinds[tags]

  source = matches[0][0]
  positive_inf = _positive_mask(source.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
  negative_inf = _positive_mask(UOp.const(-65504.0, dtypes.half).alu(Ops.SUB, source))
  either, both = positive_inf.alu(Ops.MAX, negative_inf), _mask_mul(positive_inf, negative_inf)
  value = (both if kind == "nan" else UOp.const(1.0, dtypes.half).alu(Ops.SUB, either) if kind == "finite" else
           (positive_inf if kind == "positive_inf" else negative_inf if kind == "negative_inf" else either).alu(Ops.SUB, both))

  return _typed_int_image(output, value, bool_output=True)

def lower_ew(uops:list[UOp]) -> RKImage:
  if (bool_output:=_output_store(uops, dtypes.bool)) is not None:
    if bool_output[4].op is Ops.CONST:
      return RKImage(RKTarget.RK3588, constants=struct.pack("<?", bool(bool_output[4].arg)),
                     fill=RKFill(RKArg(RKBufferKind.ARG, bool_output[1].arg.slot), bool_output[2], 1))
    if (bool_reduction:=_lower_unrolled_bool_reduction(bool_output)) is not None: return bool_reduction
    if (predicate:=_lower_ieee_predicate(bool_output)) is not None: return predicate
    if (nonzero:=_fp16_nonzero_mask(bool_output[4])) is not None: return _typed_int_image(bool_output, nonzero, bool_output=True)
    if (comparison:=_ieee_comparison_mask(bool_output[4])) is not None: return _typed_int_image(bool_output, comparison, bool_output=True)
  if (bool_loop_output:=_output_store(uops, dtypes.bool, allow_local=True)) is not None and \
     (bool_loop_reduction:=_lower_loop_bool_reduction(uops, bool_loop_output)) is not None: return bool_loop_reduction
  if (half_output:=_output_store(uops, dtypes.half)) is not None:
    if (sort_compare:=_lower_sort_compare(half_output)) is not None: return sort_compare
    if (max_unpool:=_lower_unrolled_max_unpool(half_output)) is not None: return max_unpool
  int_output, int_loop_output = _output_store(uops, dtypes.int), _output_store(uops, dtypes.int, allow_local=True)
  if int_output is not None and not any(u.op is Ops.REDUCE for u in uops):
    if (occurrence_count:=_lower_occurrence_count(int_output)) is not None: return occurrence_count
    if (sort_index:=_lower_sort_index_selection(int_output)) is not None: return sort_index
  if int_loop_output is not None and (cumulative_loop:=_lower_cumulative_extrema_index_loop(uops, int_loop_output)) is not None:
    return cumulative_loop
  if int_output is not None:
    if (cumulative_index:=_lower_cumulative_extrema_index(uops, int_output)) is not None: return cumulative_index
    if (arg_extrema:=_lower_unrolled_arg_extrema(int_output)) is not None: return arg_extrema
    if (pool_index:=_lower_unrolled_pool_index(int_output)) is not None: return pool_index
    if (int_where:=_lower_int_where(int_output)) is not None: return int_where
    if (raw_bitcast:=_lower_raw_fp16_bitcast(int_output)) is not None: return raw_bitcast
  if int_loop_output is not None:
    if (loop_arg_extrema:=_lower_loop_arg_extrema_index(uops, int_loop_output)) is not None: return loop_arg_extrema
    if (loop_pool_index:=_lower_loop_pool_index(uops, int_loop_output)) is not None: return loop_pool_index
  if int_output is not None and (raw_int32:=_lower_raw_int32_layout(int_output)) is not None: return raw_int32
  if (loop:=_loop_reduction_match(uops)) is not None:
    if (dot_reduction:=_lower_dot_loop_reduction(loop)) is not None: return dot_reduction
    if (variance:=_lower_centered_square_loop_reduction(loop)) is not None: return variance
    if (loop_reduction:=_lower_scalar_loop_reduction(loop)) is not None: return loop_reduction
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar().fmt is None or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out_param = outs[0]; assert out_param is not None
  count, oslot, store = int(out_param.src[0].arg), out_param.arg.slot, stores[0]
  if store.src[0].op is not Ops.INDEX: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_index, out, val = store.src[0].src[1], RKArg(RKBufferKind.ARG, oslot), store.src[1]
  static_cache:dict[UOp, bool] = {}
  selection_cache:dict[UOp, tuple[bool, bool]] = {}
  leaf_cache:dict[UOp, RKLeaf] = {}
  def ew_leaf(u:UOp) -> RKLeaf:
    if u not in leaf_cache: leaf_cache[u] = _ew_leaf(u, out_index, count, oslot, static_cache, selection_cache)
    return leaf_cache[u]
  if val.op is Ops.CONST and val.dtype.scalar() is out_param.dtype.scalar():
    dtype, fmt = out_param.dtype.scalar(), out_param.dtype.scalar().fmt
    assert fmt is not None
    return RKImage(RKTarget.RK3588, constants=struct.pack("<"+fmt, val.arg), fill=RKFill(out, count, dtype.itemsize))
  if ew_leaf(val) is not None:
    val = val.alu(Ops.ADD, UOp.const(0.0, dtypes.half))
  mul_terms:list[UOp] = []
  def flatten_selection_product(x:UOp) -> None:
    if x.op is Ops.MUL and x.arg is None:
      flatten_selection_product(x.src[0]); flatten_selection_product(x.src[1])
    else: mul_terms.append(x)
  flatten_selection_product(val)
  mul_leaves = [ew_leaf(x) for x in mul_terms]
  sequential_product = len(mul_terms) > 2 and all(x is not None for x in mul_leaves) and \
    any(isinstance(x, (RKGather, RKMultiGather, tuple)) for x in mul_leaves)
  if sequential_product:
    val = mul_terms[0]
    for term in mul_terms[1:]: val = val.alu(Ops.MUL, term)
  supported = RockchipRenderer.code_for_op
  static_nodes:set[UOp] = set(out_index.toposort())
  for u in uops:
    leaf = ew_leaf(u)
    if isinstance(leaf, RKStatic): static_nodes.update(leaf.expr.toposort())
    elif isinstance(leaf, (RKGather, RKMultiGather)): static_nodes.update(u.toposort())
    elif isinstance(leaf, tuple):
      static_nodes.update(leaf[1].toposort())
      if leaf[2] is not None: static_nodes.update(leaf[2].toposort())
  if any(u not in static_nodes and (u.op is Ops.REDUCE or (u.op is Ops.CAST and u.dtype.scalar() is not dtypes.half) or
         (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and ew_leaf(u) is None) or
         (u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.float, dtypes.float32, dtypes.float64))) for u in uops):
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  if any(u not in static_nodes and u.op in GroupOp.ALU and u.op not in supported and u.dtype.scalar() is dtypes.half for u in uops):
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  relu_input = _relu_operand(val)
  reduction = relu_input if relu_input is not None else val
  if not sequential_product and (reduction_info:=_mul_reduction_terms(reduction)) is not None and \
     (reduction_info[1] or len(reduction_info[0]) > 8):
    terms = reduction_info[0]
    reduce_mode = os.getenv("ROCKCHIP_EW_REDUCE", "sequential").strip().lower()
    if reduce_mode == "kahan": reduced = _compensated_mul_sum(terms)
    elif reduce_mode == "twoproduct": reduced = _precise_mul_sum(terms)
    elif reduce_mode != "sequential": raise ValueError(f"invalid ROCKCHIP_EW_REDUCE={reduce_mode!r}")
    else: reduced = reduction
    val = reduced.alu(Ops.MAX, val.src[0] if val.src[1] is reduction else val.src[1]) if reduction is not val else reduced
  order:list[UOp] = []
  visited:dict[UOp, bool] = {}
  def visit(u:UOp) -> bool:
    if u in visited: return visited[u]
    if u.op in supported and u.dtype.scalar() is dtypes.half:
      visited[u] = all(visit(src) for src in u.src)
      if visited[u]: order.append(u)
    else: visited[u] = ew_leaf(u) is not None
    return visited[u]
  if not visit(val) or not order or order[-1] is not val:
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  mask_program = any(expr.op is Ops.MAX and expr.arg == _NATIVE_POSITIVE_MASK for expr in order)
  uses = {u: 0 for u in order}
  for node in order:
    for src in node.src:
      if src in uses: uses[src] += 1
  values:dict[UOp, RKArg] = {}
  leaves:dict[UOp, RKArg] = {}
  free:list[int] = []
  const_scratch:dict[bytes, int] = {}
  static_scratch:dict[tuple[int, ...], int] = {}
  gather_scratch:dict[tuple, int] = {}
  gathers:list[RKGather] = []
  for expr in order:
    for src in expr.src:
      if isinstance((leaf:=ew_leaf(src)), float):
        bits = struct.pack("<e", leaf)
        if bits not in const_scratch: const_scratch[bits] = len(const_scratch)
  if any((expr.op is Ops.MAX and expr.arg == _NATIVE_MIN) or (expr.op is Ops.SUB and expr.arg == _NATIVE_SIGN) for expr in order):
    zero_bits = struct.pack("<e", 0.0)
    if zero_bits not in const_scratch: const_scratch[zero_bits] = len(const_scratch)
  scratch_count = len(const_scratch)
  ew_ops:list[RKEWOp] = []
  def operand(u:UOp) -> RKArg:
    nonlocal scratch_count
    if u in values: return values[u]
    if u in leaves: return leaves[u]
    leaf = ew_leaf(u)
    assert leaf is not None
    if isinstance(leaf, float): ret = RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", leaf)])
    elif isinstance(leaf, RKStatic):
      static = _static_vector(out_index, leaf.expr, count)
      if static not in static_scratch:
        static_scratch[static] = scratch_count
        gathers.append(RKGather(0, scratch_count, count, values=static))
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, static_scratch[static])
    elif isinstance(leaf, RKArg): ret = leaf
    else:
      if isinstance(leaf, RKMultiGather): gather_plans = leaf.gathers
      elif isinstance(leaf, RKGather): gather_plans = (leaf,)
      else:
        param, index, gate, fill_bits = leaf
        gather_plans = (_gather_plan(param.arg.slot, 0, out_index, index, gate, count, fill_bits),)
      for plan in gather_plans:
        if plan.values: continue
        source = next((x for x in u.toposort() if x.op is Ops.PARAM and x.arg.slot == plan.src_index), None)
        if source is None or source.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:gather_index")
        if plan.offsets: low, high = min(plan.offsets, default=0), max(plan.offsets, default=-1)
        else:
          low = high = plan.base
          for _, limit, stride in plan.axes:
            if stride < 0: low += (limit-1)*stride
            else: high += (limit-1)*stride
        if low < -1 or high >= int(source.src[0].arg): raise RuntimeError("RKPLAN_REJECT:gather_index")
      key = tuple((plan.src_index, plan.count, plan.base, plan.axes, plan.offsets, plan.fill_bits, plan.values, plan.partial,
                   plan.dst_stride, plan.dst_addend)
                  for plan in gather_plans)
      if key not in gather_scratch:
        gather_scratch[key] = scratch_count
        gathers.extend(RKGather(plan.src_index, scratch_count, plan.count, plan.base, plan.axes, plan.offsets, plan.fill_bits,
                                values=plan.values, partial=plan.partial, dst_stride=plan.dst_stride, dst_addend=plan.dst_addend)
                       for plan in gather_plans)
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, gather_scratch[key])
    leaves[u] = ret
    return ret
  for expr in order:
    lhs, rhs = operand(expr.src[0]), operand(expr.src[1])
    if expr is val: dst = out
    elif not sequential_product and (reuse:=next((values[src] for src in expr.src if src in values and uses[src] == 1 and
                       values[src].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    if expr.op is Ops.MAX and expr.arg == _NATIVE_MIN:
      zero = RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", 0.0)])
      neg_lhs, neg_rhs, neg_max = (RKArg(RKBufferKind.SCRATCH, scratch_count+i) for i in range(3))
      scratch_count += 3
      ew_ops.extend((RKEWOp(neg_lhs, zero, lhs, count, _EW_CFG[Ops.SUB], stateful=mask_program),
                     RKEWOp(neg_rhs, zero, rhs, count, _EW_CFG[Ops.SUB], stateful=mask_program),
                     RKEWOp(neg_max, neg_lhs, neg_rhs, count, _EW_CFG[Ops.MAX], stateful=mask_program),
                     RKEWOp(dst, zero, neg_max, count, _EW_CFG[Ops.SUB], submit_barrier=sequential_product and expr is val,
                            stateful=mask_program)))
    elif expr.op is Ops.SUB and expr.arg == _NATIVE_SIGN:
      zero = RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", 0.0)])
      negative, negative_mask, positive_mask = (RKArg(RKBufferKind.SCRATCH, scratch_count+i) for i in range(3))
      scratch_count += 3
      ew_ops.extend((RKEWOp(negative, zero, lhs, count, _EW_CFG[Ops.SUB]),
                     RKEWOp(negative_mask, negative, negative, count, _EW_CFG[Ops.MAX], compare=True),
                     RKEWOp(positive_mask, lhs, lhs, count, _EW_CFG[Ops.MAX], compare=True),
                     RKEWOp(dst, positive_mask, negative_mask, count, _EW_CFG[Ops.SUB], stateful=True)))
    else:
      cfg = _EW_CFG_RELU6 if expr.op is Ops.MAX and expr.arg == _NATIVE_RELU6 else \
        _EW_CFG_ABS if expr.op is Ops.MAX and expr.arg == _NATIVE_ABS else \
        _EW_CFG_FLOOR if expr.op is Ops.MAX and expr.arg == _NATIVE_FLOOR else \
        _EW_CFG_CEIL if expr.op is Ops.MAX and expr.arg == _NATIVE_CEIL else \
        _EW_CFG_LEAKY_RELU if expr.op is Ops.MUL and expr.arg == _NATIVE_LEAKY_RELU else \
        _EW_CFG_RELU if _relu_operand(expr) is not None else _EW_CFG[expr.op]
      is_positive_mask = expr.op is Ops.MAX and expr.arg == _NATIVE_POSITIVE_MASK
      ew_ops.append(RKEWOp(dst, lhs, rhs, count, cfg, submit_barrier=sequential_product and expr is val,
                           compare=is_positive_mask, stateful=mask_program and not is_positive_mask))
    values[expr] = dst
    for dep in expr.src:
      if dep in uses:
        uses[dep] -= 1
        arg = values[dep]
        if uses[dep] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  constants = b""
  if const_scratch:
    by_slot = {slot: bits for bits, slot in const_scratch.items()}
    constants = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot) + 1))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(_scratch_bytes(count)) for _ in range(scratch_count)), constants,
                 gathers=tuple(gathers), ew_ops=tuple(ew_ops))

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

def _same_condition(lhs:UOp, rhs:UOp) -> bool:
  if lhs.key == rhs.key: return True
  if lhs.op is not Ops.AND or rhs.op is not Ops.AND or len(lhs.src) != 2 or len(rhs.src) != 2: return False
  return ((_same_condition(lhs.src[0], rhs.src[0]) and _same_condition(lhs.src[1], rhs.src[1])) or
          (_same_condition(lhs.src[0], rhs.src[1]) and _same_condition(lhs.src[1], rhs.src[0])))

def _opposite_condition(lhs:UOp, rhs:UOp) -> bool:
  def unwrap_not(x:UOp) -> UOp|None:
    if x.op is not Ops.CMPNE: return None
    if x.src[1].op is Ops.CONST and bool(x.src[1].arg): return x.src[0]
    if x.src[0].op is Ops.CONST and bool(x.src[0].arg): return x.src[1]
    return None
  lhs_not, rhs_not = unwrap_not(lhs), unwrap_not(rhs)
  return ((lhs_not is not None and _same_condition(lhs_not, rhs)) or
          (rhs_not is not None and _same_condition(lhs, rhs_not)))

def _fold_scaled_negative(x:UOp) -> UOp|None:
  """Map WHERE(base<0, base*scale, base) to native DPU EW PReLU."""
  gate, negative, base = x.src
  if (gate.op is not Ops.CMPLT or gate.src[0].key != base.key or gate.src[1].op is not Ops.CONST or
      float(gate.src[1].arg) != 0.0 or negative.op is not Ops.MUL): return None
  for value, factor in (negative.src, negative.src[::-1]):
    if value.key != base.key or factor.op is not Ops.CONST: continue
    scale = float(factor.arg)
    if 0.0 <= scale <= 1.0: return UOp(Ops.MUL, x.dtype, src=(base, factor), arg=_NATIVE_LEAKY_RELU)
  return None

def _fold_masked_mul(x:UOp) -> UOp|None:
  """Push a WHERE through MUL so inactive factors become identities before native DPU multiplication."""
  gate, yes, no = x.src
  if yes.op is Ops.WHERE and _same_condition(gate, yes.src[0]): return UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[1], no))
  if no.op is Ops.WHERE and _same_condition(gate, no.src[0]): return UOp(Ops.WHERE, x.dtype, src=(gate, yes, no.src[2]))
  one = UOp.const(1.0, dtypes.half)
  if yes.op is Ops.MUL and no.op is Ops.CONST:
    return UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[0], no)).alu(Ops.MUL, UOp(Ops.WHERE, x.dtype, src=(gate, yes.src[1], one)))
  if no.op is Ops.MUL and yes.op is Ops.CONST:
    return UOp(Ops.WHERE, x.dtype, src=(gate, yes, no.src[0])).alu(Ops.MUL, UOp(Ops.WHERE, x.dtype, src=(gate, one, no.src[1])))
  return None

def _const_operand(u:UOp, op:Ops, value:float|None=None) -> tuple[UOp, UOp]|None:
  if u.op is not op: return None
  for operand, const in (u.src, u.src[::-1]):
    if const.op is Ops.CONST and (value is None or float(const.arg) == value): return operand, const
  return None

def _native_min(lhs:UOp, rhs:UOp) -> UOp:
  return UOp(Ops.MAX, lhs.dtype, src=(lhs, rhs), arg=_NATIVE_MIN)

def _fold_ordered_where(x:UOp) -> UOp|None:
  """Turn ordered clamp WHEREs into native DPU EW MIN/MAX stages."""
  gate, yes, no = x.src
  if gate.op is Ops.OR and yes.op is Ops.CONST:
    for upper, lower in ((gate.src[0], gate.src[1]), (gate.src[1], gate.src[0])):
      if (upper.op is Ops.CMPLT and upper.src[0].key == yes.key and upper.src[1].op is Ops.MAX and
          lower.op is Ops.CMPLT and lower.src[0].key == no.key and lower.src[1].key == yes.key and
          {u.key for u in upper.src[1].src} == {no.key, yes.key}): return _native_min(upper.src[1], yes)
  if gate.op is not Ops.CMPLT: return None
  lhs, rhs = gate.src
  if yes.key == rhs.key and no.key == lhs.key: return lhs.alu(Ops.MAX, rhs)
  if yes.key == lhs.key and no.key == rhs.key: return _native_min(lhs, rhs)
  return None

def _unwrap_condition(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype.scalar() in (dtypes.bool, dtypes.half, dtypes.float): u = u.src[0]
  return u

def _positive_mask(u:UOp) -> UOp:
  return UOp(Ops.MAX, dtypes.half, src=(u, u), arg=_NATIVE_POSITIVE_MASK)

def _mask_mul(lhs:UOp, rhs:UOp) -> UOp:
  return UOp(Ops.MUL, dtypes.half, src=(lhs, rhs), arg=_NATIVE_MASK_MUL)

def _mask_expr(u:UOp) -> UOp|None:
  """Build an FP16 0/1 predicate using DPU positive-mask stages."""
  u = _unwrap_condition(u)
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    lhs, rhs = (_unwrap_condition(x) for x in u.src)
    for value, other in ((lhs, rhs), (rhs, lhs)):
      if value.op is Ops.CONST and value.dtype.scalar() is dtypes.bool:
        mask = _mask_expr(other)
        if mask is None: return None
        return UOp.const(1.0, dtypes.half).alu(Ops.SUB, mask) if bool(value.arg) else mask
    if lhs.dtype.scalar() not in (dtypes.half, dtypes.float) or rhs.dtype.scalar() not in (dtypes.half, dtypes.float): return None
    lhs, rhs = lhs.cast(dtypes.half), rhs.cast(dtypes.half)
    positive = _positive_mask(rhs.alu(Ops.SUB, lhs))
    if u.op is Ops.CMPLT: return positive
    return positive.alu(Ops.MAX, _positive_mask(lhs.alu(Ops.SUB, rhs)))
  if u.op in (Ops.OR, Ops.AND):
    mask_lhs, mask_rhs = (_mask_expr(x) for x in u.src)
    if mask_lhs is None or mask_rhs is None: return None
    return mask_lhs.alu(Ops.MAX, mask_rhs) if u.op is Ops.OR else _mask_mul(mask_lhs, mask_rhs)
  return None

def _fold_general_where(x:UOp) -> UOp|None:
  """Select FP16 arms with DPU masks, avoiding multiplication by nonfinite constants."""
  mask = _mask_expr(x.src[0])
  if mask is None: return None
  yes, no = (arm.cast(dtypes.half) for arm in x.src[1:])
  one = UOp.const(1.0, dtypes.half)
  inverse = one.alu(Ops.SUB, mask)
  gate = _unwrap_condition(x.src[0])
  if gate.op is Ops.CMPLT:
    lhs, rhs, yes_u, no_u = (_unwrap_condition(u) for u in (*gate.src, *x.src[1:]))
    if rhs.op is Ops.CONST and math.isfinite(float(rhs.arg)):
      threshold = float(rhs.arg)
      if yes_u.key == lhs.key and no_u.op is Ops.CONST and math.isfinite(float(no_u.arg)) and float(no_u.arg) != threshold:
        return _native_min(lhs.cast(dtypes.half), UOp.const(threshold, dtypes.half)).alu(
          Ops.ADD, _mask_mul(inverse, UOp.const(float(no_u.arg)-threshold, dtypes.half)))
      if no_u.key == lhs.key and yes_u.op is Ops.CONST and math.isfinite(float(yes_u.arg)) and float(yes_u.arg) != threshold:
        return lhs.cast(dtypes.half).alu(Ops.MAX, UOp.const(threshold, dtypes.half)).alu(
          Ops.ADD, _mask_mul(mask, UOp.const(float(yes_u.arg)-threshold, dtypes.half)))
  nonfinite = tuple(arm.op is Ops.CONST and not math.isfinite(float(arm.arg)) for arm in x.src[1:])
  if any(nonfinite):
    if any(arm.op is Ops.CONST and math.isnan(float(arm.arg)) for arm in x.src[1:]): return None
    if all(nonfinite): return yes if float(x.src[1].arg) == float(x.src[2].arg) else None
    inf_index = next(i for i,is_nonfinite in enumerate(nonfinite) if is_nonfinite)
    finite, denominator = (no, inverse) if inf_index == 0 else (yes, mask)
    sign = math.copysign(1.0, float(x.src[1+inf_index].arg))
    correction = UOp.const(sign, dtypes.half).alu(Ops.FDIV, denominator).alu(Ops.SUB, UOp.const(sign, dtypes.half))
    return finite.alu(Ops.ADD, correction)
  return _mask_mul(yes, mask).alu(Ops.ADD, _mask_mul(no, inverse))

def _fold_relu_cap(x:UOp) -> UOp|None:
  """Recognize relu(source)-relu(source-cap), the canonical ReLU6/clamp expansion."""
  def shifted(u:UOp) -> tuple[UOp, float]:
    return (term[0], float(term[1].arg)) if (term:=_const_operand(u, Ops.ADD)) is not None else (u, 0.0)
  for positive, negative in (x.src, x.src[::-1]):
    source, scaled = _relu_operand(positive), _const_operand(negative, Ops.MUL, -1.0)
    if source is None or scaled is None or (upper:=_relu_operand(scaled[0])) is None: continue
    source_base, source_shift = shifted(source)
    upper_base, upper_shift = shifted(upper)
    if source_base.key != upper_base.key or (cap:=source_shift-upper_shift) < 0.0: continue
    if cap == 6.0: return UOp(Ops.MAX, x.dtype, src=(source, UOp.const(0.0, dtypes.half)), arg=_NATIVE_RELU6)
    return _native_min(positive, UOp.const(cap, dtypes.half))
  return None

def _fold_abs(x:UOp) -> UOp|None:
  """Recognize tinygrad's signed-zero-aware ABS graph and select native DPU EW ABS."""
  for value, sign in (x.src, x.src[::-1]):
    if sign.op is not Ops.WHERE: continue
    nonzero, signed, zero = sign.src
    if (nonzero.op is not Ops.CMPNE or nonzero.src[0].key != value.key or nonzero.src[1].op is not Ops.CONST or
        float(nonzero.src[1].arg) != 0.0 or zero.op is not Ops.CONST or float(zero.arg) != 0.0 or signed.op is not Ops.WHERE): continue
    negative, minus_one, plus_one = signed.src
    if (negative.op is Ops.CMPLT and negative.src[0].key == value.key and negative.src[1].op is Ops.CONST and
        float(negative.src[1].arg) == 0.0 and minus_one.op is Ops.CONST and float(minus_one.arg) == -1.0 and
        plus_one.op is Ops.CONST and float(plus_one.arg) == 1.0):
      return UOp(Ops.MAX, x.dtype, src=(value, value), arg=_NATIVE_ABS)
  return None

def _fold_floor_ceil(x:UOp) -> UOp|None:
  """Recognize Tinygrad's TRUNC-based floor/ceil expansions and select the native DPU ALU."""
  condition, adjusted, truncated = x.src
  if (truncated.op is not Ops.TRUNC or len(truncated.src) != 1 or condition.op is not Ops.CMPLT or adjusted.op is not Ops.ADD or
      truncated not in adjusted.src): return None
  delta = next((float(u.arg) for u in adjusted.src if u.op is Ops.CONST), None)
  source = truncated.src[0]
  if delta == -1.0 and condition.src == (source, truncated): tag = _NATIVE_FLOOR
  elif delta == 1.0 and condition.src == (truncated, source): tag = _NATIVE_CEIL
  else: return None
  return UOp(Ops.MAX, x.dtype, src=(source, source), arg=tag)

def _fold_trunc(x:UOp) -> UOp:
  """Compose truncation from native floor/ceil without mask multiplication on infinities."""
  source, zero = x.src[0], UOp.const(0.0, dtypes.half)
  positive = source.alu(Ops.MAX, zero)
  negative = zero.alu(Ops.SUB, zero.alu(Ops.SUB, source).alu(Ops.MAX, zero))
  floor = UOp(Ops.MAX, x.dtype, src=(positive, positive), arg=_NATIVE_FLOOR)
  ceil = UOp(Ops.MAX, x.dtype, src=(negative, negative), arg=_NATIVE_CEIL)
  return floor.alu(Ops.ADD, ceil)

def _fold_sign(x:UOp) -> UOp|None:
  """Recognize WHERE(x!=0, WHERE(x<0, -1, 1), 0) before general WHERE lowering."""
  nonzero, signed, zero = x.src
  if (nonzero.op is not Ops.CMPNE or signed.op is not Ops.WHERE or zero.op is not Ops.CONST or float(zero.arg) != 0.0 or
      signed.src[0].op is not Ops.CMPLT or signed.src[1].op is not Ops.CONST or float(signed.src[1].arg) != -1.0 or
      signed.src[2].op is not Ops.CONST or float(signed.src[2].arg) != 1.0): return None
  source = next((u for u in nonzero.src if u.dtype.scalar() is dtypes.half and u.op is not Ops.CONST), None)
  if (source is None or not any(u.op is Ops.CONST and float(u.arg) == 0.0 for u in nonzero.src) or
      signed.src[0].src[0].key != source.key or signed.src[0].src[1].op is not Ops.CONST or
      float(signed.src[0].src[1].arg) != 0.0): return None
  return UOp(Ops.SUB, dtypes.half, src=(source, source), arg=_NATIVE_SIGN)

def _fold_minimum(x:UOp) -> UOp|None:
  """Recognize -max(-x,-y); native ALU-MIN mishandles infinities, so lowering expands it through SUB and MAX."""
  outer = _const_operand(x, Ops.MUL, -1.0)
  if outer is None or outer[0].op is not Ops.MAX: return None
  operands = [_const_operand(u, Ops.MUL, -1.0) for u in outer[0].src]
  if len(operands) != 2 or any(u is None for u in operands): return None
  lhs, rhs = (u for u in operands if u is not None)
  return _native_min(lhs[0], rhs[0])

def _fold_masked_load(gate:UOp, load:UOp, default:UOp) -> UOp|None:
  if len(load.src) <= 2 or load.src[1].op is not Ops.CONST: return None
  load_gate = load.src[2]
  same_default = float(load.src[1].arg) == float(default.arg)
  # If the outer condition implies the LOAD condition, the inner default is unreachable. This is the padded-pool form.
  outer_implies_inner = _same_condition(gate, load_gate) or (gate.op is Ops.AND and any(_same_condition(x, load_gate) for x in gate.src))
  if not same_default and not outer_implies_inner: return None
  return load.replace(src=(load.src[0], default, gate.alu(Ops.AND, load_gate) if same_default else gate))

def _fold_masked_max(gate:UOp, default:UOp, val:UOp, opposite:bool) -> UOp|None:
  if val.op is Ops.MAX:
    lhs = _fold_masked_max(gate, default, val.src[0], opposite)
    rhs = _fold_masked_max(gate, default, val.src[1], opposite)
    return None if lhs is None or rhs is None else val.replace(src=(lhs, rhs))
  if val.op is not Ops.LOAD or len(val.src) <= 2 or val.src[1].op is not Ops.CONST: return None
  def matches(condition:UOp) -> bool: return _opposite_condition(gate, condition) if opposite else _same_condition(gate, condition)
  condition_matches = matches(val.src[2]) or (val.src[2].op is Ops.AND and any(matches(x) for x in val.src[2].src))
  if condition_matches:
    return val.replace(src=(val.src[0], default, val.src[2]))
  return None

def _fp32_alu_to_fp16(x:UOp) -> UOp|None:
  return None if _is_static_expr(x) else x.src[0].cast(dtypes.half).alu(x.op, x.src[1].cast(dtypes.half))

def _fold_casted_relu(root:UOp) -> UOp|None:
  """Recover native half MAX from the float WHERE emitted for half ReLU inside a reduction."""
  if len(root.src) != 1 or (where:=root.src[0]).op is not Ops.WHERE or where.dtype.scalar() is not dtypes.float: return None
  gate, yes, no = where.src
  if (yes.op is not Ops.CAST or len(yes.src) != 1 or yes.src[0].dtype.scalar() is not dtypes.half or
      no.op is not Ops.CONST or float(no.arg) != 0.0): return None
  val = yes.src[0]
  if (gate.op is not Ops.CMPLT or gate.src[0].op is not Ops.CONST or float(gate.src[0].arg) != 0.0 or
      gate.src[1].key != val.key): return None
  return val.alu(Ops.MAX, UOp.const(0.0, dtypes.half))

def _fold_bool_to_half(predicate:UOp) -> UOp|None:
  """Materialize an embedded boolean predicate as an FP16 DPU 0/1 mask."""
  if (nonzero:=_fp16_nonzero_mask(predicate)) is not None: return nonzero
  return _ieee_comparison_mask(predicate)

def _replace_infinite_multiply(x:UOp) -> UOp|None:
  """DPU MUL maps finite infinity products to NaN; signed finite/zero FDIV has the required result."""
  for value, factor in (x.src, x.src[::-1]):
    if factor.op is not Ops.CONST or not math.isinf(scale:=float(factor.arg)): continue
    if scale < 0: value = value.alu(Ops.MUL, UOp.const(-1.0, dtypes.half))
    return value.alu(Ops.FDIV, UOp.const(0.0, dtypes.half))
  return None

def _preserve_infinite_division_sign(x:UOp) -> UOp|None:
  """RK3588 FDIV ignores the denominator sign for an infinite numerator; rebuild it with finite DPU intermediates."""
  numerator, denominator = x.src
  if numerator.op is not Ops.CONST or not math.isinf(value:=float(numerator.arg)): return None
  unit = UOp.const(-1.0 if value < 0 else 1.0, dtypes.half)
  return unit.alu(Ops.FDIV, denominator).alu(Ops.FDIV, UOp.const(0.0, dtypes.half))

_pm_fp32_to_fp16 = PatternMatcher([
  (UPat((Ops.ADD, Ops.MUL), dtypes.float, name="x"), _fp32_alu_to_fp16),
  (UPat(Ops.CAST, dtypes.half, name="root"), _fold_casted_relu),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.bool, name="predicate"),)), _fold_bool_to_half),
  (UPat(Ops.ADD, dtypes.half, name="x"), _fold_relu_cap),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_minimum),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
  (UPat(Ops.MUL, dtypes.half, name="x"), _replace_infinite_multiply),
  (UPat(Ops.FDIV, dtypes.half, name="x"), _preserve_infinite_division_sign),
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_masked_mul),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_ordered_where),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_scaled_negative),
  # Fold padding into the gather mask. This changes only host layout initialization; selected values still feed DPU EW.
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat(Ops.LOAD, dtypes.half, name="load"), UPat.cvar("default"))),
   lambda gate,load,default: _fold_masked_load(gate, load, default)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.var("val"), UPat.cvar("default"))),
   lambda gate,val,default: _fold_masked_max(gate, default, val, False)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.cvar("default"), UPat.var("val"))),
   lambda gate,default,val: _fold_masked_max(gate, default, val, True)),
  (UPat(Ops.WHERE, dtypes.half, name="x"), lambda x: x.src[1].alu(Ops.MAX, x.src[2])
   if x.src[0].op is Ops.CMPLT and x.src[0].src[0] is x.src[2] and x.src[0].src[1] is x.src[1] and
      x.src[2].op is Ops.CONST and float(x.src[2].arg) == 0.0 else None),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_general_where),
])
_pm_abs = PatternMatcher([(UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs)])
_pm_floor_ceil = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_floor_ceil)])
_pm_trunc = PatternMatcher([(UPat(Ops.TRUNC, dtypes.half, name="x"), _fold_trunc)])
_pm_sign = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_sign)])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  sink = graph_rewrite(sink, _pm_floor_ceil, name="rockchip floor/ceil")
  sink = graph_rewrite(sink, _pm_trunc, name="rockchip trunc")
  sink = graph_rewrite(sink, _pm_abs, name="rockchip abs")
  sink = graph_rewrite(sink, _pm_sign, name="rockchip sign")
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None, Ops.FDIV: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_ew(_fp16_rewrite(uops)))).decode()
