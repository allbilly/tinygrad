from __future__ import annotations
# ruff: noqa: E702
import base64, math, os, struct
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import Target, cdiv, cmod, floordiv, floormod
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 24
_HEADER = struct.Struct("<4sHHHHIII")  # magic, version, target, scratch, gathers, ops, constants, flags
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
  int32_output: bool = False; int32_input: bool = False

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); fill: RKFill|None = None; ew_ops: tuple[RKEWOp, ...] = ()
  post_gathers: tuple[RKGather, ...] = ()

@dataclass(frozen=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.post_gathers
  flags = int(image.fill is not None) | len(image.post_gathers)<<1
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(gathers),
                               len(image.ew_ops), len(image.constants), flags))
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
    op_flags = int(op.submit_barrier) | int(op.compare)<<1 | int(op.stateful)<<2 | int(op.int32_output)<<3 | int(op.int32_input)<<4
    out += _EWOP.pack(int(op.dst.kind), op_flags, op.dst.index,
                      int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index)
    out += _EWOP2.pack(op.count, op.ew_cfg) + struct.pack("<iii", op.dst.addend, op.lhs.addend, op.rhs.addend)
  if image.fill is not None: out += _FILL.pack(int(image.fill.dst.kind), image.fill.itemsize, image.fill.dst.index, image.fill.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nop, nconst, flags = _HEADER.unpack_from(blob)
  post_count = flags >> 1
  if magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or post_count > ngather:
    raise ValueError("invalid RKImage header")
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
    if op_flags & ~0x1f or op_flags & 0x18 == 0x18: raise ValueError("invalid RKEWOp flags")
    count, ew_cfg = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg,
                         bool(op_flags & 1), bool(op_flags & 2), bool(op_flags & 4), bool(op_flags & 8), bool(op_flags & 16)))
  fill = None
  if flags & 1:
    dst_kind, itemsize, dst_index, count = _FILL.unpack_from(blob, off); off += _FILL.size
    if itemsize not in (1, 2, 4, 8): raise ValueError("invalid RKFill item size")
    fill = RKFill(RKArg(RKBufferKind(dst_kind), dst_index), count, itemsize)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  split = len(gathers)-post_count
  return RKImage(RKTarget(target), scratch, blob[off:], version, tuple(gathers[:split]), fill, tuple(ew_ops), tuple(gathers[split:]))

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
_NATIVE_ABS, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN, _NATIVE_POSITIVE_MASK, _NATIVE_RELU6, _NATIVE_SIGN = \
  "rockchip_abs", "rockchip_leaky_relu", "rockchip_mask_mul", "rockchip_min", "rockchip_positive_mask", "rockchip_relu6", "rockchip_sign"
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
      else: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
  cache[u] = ret
  return ret

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.MAX}
def _is_static_expr(u:UOp, cache:dict[UOp, bool]|None=None) -> bool:
  if cache is not None and u in cache: return cache[u]
  ret = u.op in _STATIC_OPS and all(_is_static_expr(x, cache) for x in u.src)
  if cache is not None: cache[u] = ret
  return ret

def _index_ranges(index:UOp) -> list[UOp]: return [u for u in index.toposort() if u.op is Ops.RANGE]

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

def _static_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  values:list[int|None] = [None] * count
  for env in _iter_range_env(ranges):
    cache:dict[UOp, int|float|bool] = {}
    dst = _eval_int(out_index, env, cache)
    if not 0 <= dst < count: raise RuntimeError("RKPLAN_REJECT:static_index")
    bits = struct.unpack("<H", struct.pack("<e", float(_eval_expr(expr, env, cache))))[0]
    if values[dst] is not None and values[dst] != bits: raise RuntimeError("RKPLAN_REJECT:static_index")
    values[dst] = bits
  if any(x is None for x in values): raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(x for x in values if x is not None)

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

def _lower_dot_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Lower an FP16 dot loop as vector MUL terms followed by a balanced vector ADD tree."""
  global_stores = [(store, _root_param(store.src[0])) for store in uops if store.op is Ops.STORE]
  global_stores = [(store, param) for store,param in global_stores if param is not None]
  if len(global_stores) != 1: return None
  store, out_param = global_stores[0]
  assert out_param is not None
  if (out_param.dtype.scalar() is not dtypes.half or out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX or
      not 0 < int(out_param.src[0].arg) <= _MAX_EW_ELEMS_FP16): return None
  nodes = store.src[1].toposort()
  output_ranges = _index_ranges(store.src[0].src[1])
  reduce_ranges = [u for u in nodes if u.op is Ops.RANGE and u not in output_ranges]
  if len(reduce_ranges) != 1: return None
  reduce_range = reduce_ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) <= 0: return None
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1: return None
  update = _strip_cast(updates[0].src[1])
  if update.op is not Ops.ADD or (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None: return None
  product = _strip_cast(update.src[1 if update.src[0] is acc else 0])
  if product.op is not Ops.MUL or product.dtype.scalar() is not dtypes.half: return None
  for operand in product.src:
    operand = _strip_cast(operand)
    param = _root_param(operand.src[0]) if operand.op is Ops.LOAD and operand.src and operand.src[0].op is Ops.INDEX else None
    if param is None or operand.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST: return None
  if _local_load(store.src[1]) is None: return None
  terms = [product.substitute({reduce_range:reduce_range.const_like(r)}) for r in range(groups)]
  while len(terms) > 1:
    terms = [terms[i].alu(Ops.ADD, terms[i+1]) for i in range(0, len(terms)-1, 2)] + (terms[-1:] if len(terms) & 1 else [])
  return lower_ew([store.replace(src=(store.src[0], terms[0], *store.src[2:]))])

def _lower_centered_square_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Lower scalar SUM((x-center)^2)*scale using only aligned one-lane DPU EW stages."""
  global_stores = [(store, _root_param(store.src[0])) for store in uops if store.op is Ops.STORE]
  global_stores = [(store, param) for store,param in global_stores if param is not None]
  if len(global_stores) != 1: return None
  store, out_param = global_stores[0]
  assert out_param is not None
  nodes = store.src[1].toposort()
  ranges = [u for u in nodes if u.op is Ops.RANGE]
  fp32_out = out_param.dtype.scalar() is dtypes.float
  if (len(ranges) != 1 or out_param.dtype.scalar() not in (dtypes.half, dtypes.float) or out_param.src[0].op is not Ops.CONST or
      int(out_param.src[0].arg) != 1 or store.src[0].op is not Ops.INDEX): return None
  reduce_range = ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) < 2: return None

  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1 or (update:=_strip_cast(updates[0].src[1])).op is not Ops.ADD: return None
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
    data_offsets = tuple(_eval_int(direct.src[0].src[1], {reduce_range:r}) for r in range(groups))
    center_offsets = tuple(_eval_int(center.src[0].src[1], {reduce_range:r}) for r in range(groups))
  except RuntimeError: return None
  if (int(data_param.src[0].arg) != groups or sorted(data_offsets) != list(range(groups)) or len(set(center_offsets)) != 1 or
      not 0 <= center_offsets[0] < int(center_param.src[0].arg)): return None

  final_value = store.src[1]
  if _local_load(final_value) is not None: post_scale = 1.0
  elif final_value.op is Ops.MUL and (final_load:=next((x for x in final_value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=final_value.src[1 if final_value.src[0] is final_load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else: return None
  constants = () if post_scale == 1.0 else (post_scale,)
  data_slot = len(constants)
  gather = RKGather(data_param.arg.slot, data_slot, groups, axes=((1, groups, 1),) if data_offsets == tuple(range(groups)) else (),
                    offsets=() if data_offsets == tuple(range(groups)) else data_offsets, dst_stride=32)
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, data_slot, offset)
  center_arg = RKArg(RKBufferKind.ARG, center_param.arg.slot, center_offsets[0]*2)
  ew_ops:list[RKEWOp] = []
  active = [r*64 for r in range(groups)]
  for offset in active:
    value = arena(offset)
    ew_ops.append(RKEWOp(value, value, center_arg, 1, _EW_CFG[Ops.SUB], stateful=not ew_ops))
    ew_ops.append(RKEWOp(value, value, value, 1, _EW_CFG[Ops.MUL]))
  while len(active) > 1:
    next_active:list[int] = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs = active[i], active[i+1]
      final = len(active) == 2 and post_scale == 1.0
      dst = RKArg(RKBufferKind.ARG, out_param.arg.slot) if final else arena(lhs)
      ew_ops.append(RKEWOp(dst, arena(lhs), arena(rhs), 1, _EW_CFG[Ops.ADD] | (_EW_STAGE_FP32_OUT if fp32_out and final else 0)))
      next_active.append(lhs)
    if len(active) & 1: next_active.append(active[-1])
    active = next_active
  if post_scale != 1.0:
    ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), arena(active[0]), RKArg(RKBufferKind.SCRATCH, 0), 1,
                         _EW_CFG[Ops.MUL] | (_EW_STAGE_FP32_OUT if fp32_out else 0)))
  scratch = tuple(RKScratch(2) for _ in constants) + (RKScratch(groups*64),)
  return RKImage(RKTarget.RK3588, scratch, b"".join(struct.pack("<e", value) for value in constants), gathers=(gather,), ew_ops=tuple(ew_ops))

def _lower_scalar_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Turn a compact scalar register reduction into balanced FP16 DPU EW stages."""
  global_stores = [(store, _root_param(store.src[0])) for store in uops if store.op is Ops.STORE]
  global_stores = [(store, param) for store,param in global_stores if param is not None]
  if len(global_stores) != 1: return None
  store, out_param = global_stores[0]
  nodes = store.src[1].toposort()
  ranges = [u for u in nodes if u.op is Ops.RANGE]
  if len(ranges) != 1: return None
  assert out_param is not None
  fp32_out = out_param.dtype.scalar() is dtypes.float
  if (out_param.dtype.scalar() not in (dtypes.half, dtypes.float) or out_param.src[0].op is not Ops.CONST or int(out_param.src[0].arg) != 1 or
      store.src[0].op is not Ops.INDEX): return None
  reduce_range = ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) <= 0: return None
  loads:list[tuple[UOp, UOp]] = []
  for u in nodes:
    if u.op is not Ops.LOAD or u.dtype.scalar() is not dtypes.half or not u.src or u.src[0].op is not Ops.INDEX: continue
    param = _root_param(u.src[0])
    if param is not None and param.arg.slot != out_param.arg.slot: loads.append((u, param))
  if not loads or len({param.key for _,param in loads}) != 1: return None
  in_param = loads[0][1]
  if in_param.src[0].op is not Ops.CONST: return None
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1: return None
  final_value = store.src[1]
  if _local_load(final_value) is not None: post_scale = 1.0
  elif final_value.op is Ops.MUL and (final_load:=next((x for x in final_value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=final_value.src[1 if final_value.src[0] is final_load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else: return None
  reduce_ops = {u.op for u in updates[0].src[1].toposort()
                if u.dtype.scalar() is dtypes.half and u.op in (Ops.ADD, Ops.MUL, Ops.MAX)}
  negate_inputs = reduce_ops == {Ops.MUL, Ops.MAX} and any(u.op is Ops.CONST and u.dtype.scalar() is dtypes.half and
                                                            float(u.arg) == -1.0 for u in updates[0].src[1].toposort())
  if negate_inputs: reduce_op = Ops.MAX
  elif len(reduce_ops) == 1: reduce_op = next(iter(reduce_ops))
  else: return None
  if reduce_op not in _EW_CFG: return None
  gather_offsets = [tuple(_eval_int(load.src[0].src[1], {reduce_range:r}) for r in range(groups)) for load,_ in loads]
  input_count = int(in_param.src[0].arg)
  if input_count != groups*len(loads) or sorted(offset for offsets in gather_offsets for offset in offsets) != list(range(input_count)): return None
  if input_count < 2: return None

  const_values = tuple(dict.fromkeys(x for x in ((-1.0,) if negate_inputs else ()) + ((post_scale,) if post_scale != 1.0 else ())))
  const_slots = {value:i for i,value in enumerate(const_values)}
  data_slot = len(const_values)
  gathers = (RKGather(in_param.arg.slot, data_slot, input_count, 0, ((1, input_count, 1),), dst_stride=32),)
  cfg = _EW_CFG[reduce_op]
  ew_ops:list[RKEWOp] = []
  active = [i*64 for i in range(input_count)]
  if negate_inputs:
    neg_one = RKArg(RKBufferKind.SCRATCH, const_slots[-1.0])
    for offset in active:
      value = RKArg(RKBufferKind.SCRATCH, data_slot, offset)
      ew_ops.append(RKEWOp(value, value, neg_one, 1, _EW_CFG[Ops.MUL]))
  while len(active) > 1:
    next_active:list[int] = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs = active[i], active[i+1]
      final = len(active) == 2 and post_scale == 1.0
      dst = RKArg(RKBufferKind.ARG, out_param.arg.slot) if final else RKArg(RKBufferKind.SCRATCH, data_slot, lhs)
      ew_ops.append(RKEWOp(dst, RKArg(RKBufferKind.SCRATCH, data_slot, lhs), RKArg(RKBufferKind.SCRATCH, data_slot, rhs), 1,
                          cfg | (_EW_STAGE_FP32_OUT if fp32_out and final else 0)))
      next_active.append(lhs)
    if len(active) & 1: next_active.append(active[-1])
    active = next_active
  if post_scale != 1.0:
    ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), RKArg(RKBufferKind.SCRATCH, data_slot, active[0]),
                         RKArg(RKBufferKind.SCRATCH, const_slots[post_scale]), 1,
                         _EW_CFG[Ops.MUL] | (_EW_STAGE_FP32_OUT if fp32_out else 0)))
  constants = b"".join(struct.pack("<e", value) for value in const_values)
  scratch = tuple(RKScratch(2) for _ in const_values) + (RKScratch(input_count*64),)
  return RKImage(RKTarget.RK3588, scratch, constants, gathers=gathers, ew_ops=tuple(ew_ops))

def _lower_raw_int32_layout(uops:list[UOp]) -> RKImage|None:
  """Move an INT32 tensor through a static view or shrink without interpreting its values."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or out_param.dtype.scalar() is not dtypes.int: return None
  count, out_index, value = int(out_param.src[0].arg), stores[0].src[0].src[1], stores[0].src[1]
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

def _load_equality(predicate:UOp) -> tuple[UOp, UOp]|None:
  """Recognize boolean inversion of CMPNE between two loaded lanes."""
  if predicate.op is not Ops.CMPNE: return None
  truths = [x for x in predicate.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg)]
  inequalities = [x for x in predicate.src if x.op is Ops.CMPNE]
  if len(truths) != 1 or len(inequalities) != 1: return None
  pair = inequalities[0].src
  if len(pair) != 2 or any(x.op is not Ops.LOAD or x.src[0].op is not Ops.INDEX for x in pair): return None
  return pair[0], pair[1]

def _lower_occurrence_count(uops:list[UOp]) -> RKImage|None:
  """Lower stable-sort's unrolled prefix equality counts to DPU masks and ADD."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if (len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or
      out_param.dtype.scalar() is not dtypes.int or any(u.op is Ops.REDUCE for u in uops)): return None
  store = stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  count, out_index = int(out_param.src[0].arg), store.src[0].src[1]
  if count <= 0: return RKImage(RKTarget.RK3588)

  terms:list[UOp] = []
  def flatten(value:UOp) -> None:
    if value.op is Ops.ADD and value.dtype.scalar() is dtypes.int: flatten(value.src[0]); flatten(value.src[1])
    else: terms.append(value)
  flatten(store.src[1])
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
  common = set(parsed[0][0]).intersection(*(set(pair) for pair,_ in parsed[1:]))
  if len(common) != 1: return None
  current = next(iter(common))
  candidates = [next((x for x in pair if x is not current), None) for pair,_ in parsed]
  if any(x is None for x in candidates): return None
  candidate_loads = tuple(x for x in candidates if x is not None)
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
  vector_bytes = (count*2+63)&-64
  vector_lanes, matrix_lanes = vector_bytes//2, window*vector_bytes//2
  zero, one, candidates_slot, current_slot, valid_slot, diff, magnitude, unequal, equal, selected, int_tiles = range(11)
  gathers:tuple[RKGather, ...] = tuple(RKGather(source.arg.slot, candidates_slot, count, offsets=offsets,
                                                dst_addend=i*vector_lanes) for i,offsets in enumerate(candidate_offsets))
  gathers += tuple(RKGather(source.arg.slot, current_slot, count, offsets=current_offsets,
                            dst_addend=i*vector_lanes) for i in range(window))
  gathers += tuple(RKGather(source.arg.slot, valid_slot, count, values=bits,
                            dst_addend=i*vector_lanes) for i,bits in enumerate(valid_bits))
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(((count+3)//4)*64))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(diff), arg(candidates_slot), arg(current_slot), matrix_lanes, _EW_CFG[Ops.SUB]),
    RKEWOp(arg(magnitude), arg(diff), arg(diff), matrix_lanes, _EW_CFG_ABS, submit_barrier=True, stateful=True),
    RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(equal), arg(one), arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], stateful=True),
    RKEWOp(arg(selected), arg(equal), arg(valid_slot), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)]
  active_args = [arg(selected, candidate*vector_bytes) for candidate in range(window)]
  first = True
  while len(active_args) > 1:
    reduced:list[RKArg] = []
    for i in range(0, len(active_args)-1, 2):
      ops.append(RKEWOp(active_args[i], active_args[i], active_args[i+1], count, _EW_CFG[Ops.ADD], submit_barrier=first, stateful=first))
      first = False
      reduced.append(active_args[i])
    if len(active_args) & 1: reduced.append(active_args[-1])
    active_args = reduced
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), active_args[0], arg(int_tiles), count,
                      _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _lower_sort_index_selection(uops:list[UOp]) -> RKImage|None:
  """Lower stable sort's value/count match and coordinate sum entirely to DPU EW."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if (len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or
      out_param.dtype.scalar() is not dtypes.int or any(u.op is Ops.REDUCE for u in uops)): return None
  store = stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  count, out_index = int(out_param.src[0].arg), store.src[0].src[1]
  if count <= 0: return RKImage(RKTarget.RK3588)

  terms:list[UOp] = []
  def flatten(value:UOp) -> None:
    if value.op is Ops.ADD and value.dtype.scalar() is dtypes.int: flatten(value.src[0]); flatten(value.src[1])
    else: terms.append(value)
  flatten(store.src[1])
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

  def split_pairs(pairs:tuple[tuple[UOp, UOp], ...]) -> tuple[UOp, tuple[UOp, ...]]|None:
    common = set(pairs[0]).intersection(*(set(pair) for pair in pairs[1:]))
    if len(common) != 1: return None
    current = next(iter(common))
    candidates = tuple(next((x for x in pair if x is not current), None) for pair in pairs)
    return None if any(x is None for x in candidates) else (current, tuple(x for x in candidates if x is not None))

  if (half_split:=split_pairs(tuple(pair for _,pair,_ in parsed))) is None or \
     (int_split:=split_pairs(tuple(pair for _,_,pair in parsed))) is None: return None
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
  vector_bytes = (count*2+63)&-64
  vector_lanes, matrix_lanes = vector_bytes//2, rows*vector_bytes//2
  (zero, one, raw_candidate_count, raw_current_count, candidate_count, current_count, convert_tiles,
   candidate_value, current_value, weights, value_diff, value_magnitude, value_unequal, value_equal,
   count_diff, count_magnitude, count_unequal, count_equal, selected, weighted, int_tiles) = range(21)
  gathers:tuple[RKGather, ...] = ()
  for row,(weight, half_offsets, int_offsets) in enumerate(zip((x[0] for x in parsed), half_candidate_offsets, int_candidate_offsets)):
    addend = row*vector_lanes
    gathers += (RKGather(half_candidate_params[0].arg.slot, candidate_value, count, offsets=half_offsets, dst_addend=addend),
                RKGather(int_candidate_params[0].arg.slot, raw_candidate_count, count, offsets=int_offsets,
                         dst_addend=addend, itemsize=4),
                RKGather(half_candidate_params[0].arg.slot, weights, count,
                         values=(struct.unpack("<H", struct.pack("<e", float(weight)))[0],)*count, dst_addend=addend))
  gathers += tuple(RKGather(half_current_param.arg.slot, current_value, count, offsets=half_current_offsets,
                            dst_addend=row*vector_lanes) for row in range(rows))
  gathers += tuple(RKGather(int_current_param.arg.slot, raw_current_count, count, offsets=int_current_offsets,
                            dst_addend=row*vector_lanes, itemsize=4) for row in range(rows))
  scratch_sizes = [matrix_lanes*2]*21
  scratch_sizes[raw_candidate_count] = scratch_sizes[raw_current_count] = matrix_lanes*4
  scratch_sizes[convert_tiles] = ((matrix_lanes+3)//4)*64
  scratch_sizes[int_tiles] = ((count+3)//4)*64
  scratch = tuple(RKScratch(_scratch_bytes(size//2) if i not in (raw_candidate_count, raw_current_count, convert_tiles, int_tiles)
                            else size) for i,size in enumerate(scratch_sizes))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [
    RKEWOp(arg(candidate_count), arg(raw_candidate_count), arg(convert_tiles), matrix_lanes, _EW_CFG[Ops.MAX], int32_input=True),
    RKEWOp(arg(current_count), arg(raw_current_count), arg(convert_tiles), matrix_lanes, _EW_CFG[Ops.MAX], int32_input=True),
    RKEWOp(arg(value_diff), arg(candidate_value), arg(current_value), matrix_lanes, _EW_CFG[Ops.SUB]),
    RKEWOp(arg(value_magnitude), arg(value_diff), arg(value_diff), matrix_lanes, _EW_CFG_ABS, submit_barrier=True, stateful=True),
    RKEWOp(arg(value_unequal), arg(value_magnitude), arg(value_magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(value_equal), arg(one), arg(value_unequal), matrix_lanes, _EW_CFG[Ops.SUB], stateful=True),
    RKEWOp(arg(count_diff), arg(candidate_count), arg(current_count), matrix_lanes, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True),
    RKEWOp(arg(count_magnitude), arg(count_diff), arg(count_diff), matrix_lanes, _EW_CFG_ABS),
    RKEWOp(arg(count_unequal), arg(count_magnitude), arg(count_magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(count_equal), arg(one), arg(count_unequal), matrix_lanes, _EW_CFG[Ops.SUB], stateful=True),
    RKEWOp(arg(selected), arg(value_equal), arg(count_equal), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
    RKEWOp(arg(weighted), arg(selected), arg(weights), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)]
  active = [arg(weighted, row*vector_bytes) for row in range(rows)]
  first = True
  while len(active) > 1:
    reduced:list[RKArg] = []
    for i in range(0, len(active)-1, 2):
      ops.append(RKEWOp(active[i], active[i], active[i+1], count, _EW_CFG[Ops.ADD], submit_barrier=first, stateful=first))
      first = False
      reduced.append(active[i])
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), active[0], arg(int_tiles), count,
                      _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _lower_sort_compare(uops:list[UOp]) -> RKImage|None:
  """Lower one static bitonic compare/swap pass with DPU MAX and MIN."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or out_param.dtype.scalar() is not dtypes.half: return None
  store = stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX or store.src[1].op is not Ops.WHERE: return None
  count, out_index, condition = int(out_param.src[0].arg), store.src[0].src[1], store.src[1].src[0]
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

  true_extreme, false_extreme = extreme(store.src[1].src[1]), extreme(store.src[1].src[2])
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
                            first_tie:bool=False, negate_extrema:bool=False) -> RKImage:
  """Emit a matrix equality/select reduction shared by unrolled and loop cumulative indices."""
  window = len(candidate_plans)
  constants = (0.0, 1.0, float(window)) if first_tie else (0.0, 1.0)
  zero, one, candidate_arena = 0, 1, len(constants)
  extrema_slots = tuple(range(candidate_arena+1, candidate_arena+1+len(extrema_plans)))
  first_temp = candidate_arena+1+len(extrema_plans)
  coordinate_arena, selected_arena, diff, magnitude, unequal, equal, int_tiles = range(first_temp, first_temp+7)
  vector_bytes = (count*2+63)&-64
  vector_lanes, matrix_lanes = vector_bytes//2, window*vector_bytes//2
  def materialize(plans:tuple[RKGather, ...], scratch_slot:int) -> tuple[RKGather, ...]:
    return tuple(RKGather(plan.src_index, scratch_slot, count, plan.base, plan.axes, plan.offsets, plan.fill_bits,
                          values=plan.values, partial=plan.partial, dst_stride=plan.dst_stride,
                          dst_addend=i*vector_lanes, itemsize=plan.itemsize) for i,plan in enumerate(plans))
  gathers = materialize(candidate_plans, candidate_arena)
  for plan,scratch_slot in zip(extrema_plans, extrema_slots): gathers += materialize((plan,)*window, scratch_slot)
  coordinate_bits = tuple(tuple(struct.unpack("<H", struct.pack("<e", float(window-candidate if first_tie else
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
    ew_ops.extend((RKEWOp(args(diff), args(zero), args(candidate_arena), matrix_lanes, _EW_CFG[Ops.SUB],
                           submit_barrier=bool(ew_ops), stateful=bool(ew_ops)),
                   RKEWOp(args(magnitude), args(diff), extrema, matrix_lanes, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True)))
  else: ew_ops.append(RKEWOp(args(magnitude), args(candidate_arena), extrema, matrix_lanes, _EW_CFG[Ops.SUB],
                             submit_barrier=bool(ew_ops), stateful=bool(ew_ops)))
  ew_ops.extend((RKEWOp(args(magnitude), args(magnitude), args(magnitude), matrix_lanes, _EW_CFG_ABS,
                        submit_barrier=True, stateful=True),
                 RKEWOp(args(unequal), args(magnitude), args(magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True),
                 RKEWOp(args(equal), args(one), args(unequal), matrix_lanes, _EW_CFG[Ops.SUB], stateful=True),
                 RKEWOp(args(diff), args(equal), args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=True, stateful=True),
                 RKEWOp(args(selected_arena), args(equal), args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=True, stateful=True)))
  active = [args(selected_arena, candidate*vector_bytes) for candidate in range(window)]
  first_reduction = True
  while len(active) > 1:
    reduced:list[RKArg] = []
    for i in range(0, len(active)-1, 2):
      ew_ops.append(RKEWOp(active[i], active[i], active[i+1], count, _EW_CFG[Ops.MAX],
                           submit_barrier=first_reduction, stateful=first_reduction))
      first_reduction = False
      reduced.append(active[i])
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  ew_ops.append(RKEWOp(args(diff), args(2), active[0], count, _EW_CFG[Ops.SUB]) if first_tie else
                RKEWOp(args(diff), active[0], args(one), count, _EW_CFG[Ops.SUB]))
  ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), args(diff), args(int_tiles), count,
                       _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<"+"e"*len(constants), *constants), gathers=gathers, ew_ops=tuple(ew_ops))

def _flatten_binary(root:UOp, op:Ops) -> list[UOp]:
  return _flatten_binary(root.src[0], op)+_flatten_binary(root.src[1], op) if root.op is op else [root]

def _half_candidate(root:UOp) -> tuple[UOp, bool]|None:
  if root.op is Ops.LOAD: return (root, False)
  if root.op is Ops.MUL and len(root.src) == 2:
    loads = [x for x in root.src if x.op is Ops.LOAD]
    constants = [x for x in root.src if x.op is Ops.CONST and float(x.arg) == -1.0]
    if len(loads) == len(constants) == 1: return (loads[0], True)
  return None

def _first_tie_selection(value:UOp, extrema:UOp, ordered_exprs:list[UOp]) -> bool:
  """Verify Tinygrad's descending-coordinate INT32 MAX encoding for the first equal candidate."""
  window = len(ordered_exprs)
  equal_casts:list[UOp] = []
  for expr in ordered_exprs:
    matches = [u for u in value.toposort() if u.op is Ops.CMPNE and (u.src == (expr, extrema) or u.src == (extrema, expr))]
    if len(matches) != 1: return False
    inversions = [u for u in value.toposort() if u.op is Ops.CMPNE and matches[0] in u.src and
                  any(x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg) for x in u.src)]
    casts = [u for u in value.toposort() if u.op is Ops.CAST and u.dtype.scalar() is dtypes.int and
             len(inversions) == 1 and u.src == (inversions[0],)]
    if len(casts) != 1: return False
    equal_casts.append(casts[0])
  int_roots = [u for u in value.toposort() if u.op is Ops.MAX and u.dtype.scalar() is dtypes.int]
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

def _lower_arg_extrema_index(uops:list[UOp]) -> RKImage|None:
  """Lower fused FP16 ArgMax/ArgMin while preserving first-index tie semantics."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or out_param.dtype.scalar() is not dtypes.int: return None
  store = stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  count, out_index, value = int(out_param.src[0].arg), store.src[0].src[1], store.src[1]
  if count <= 0: return RKImage(RKTarget.RK3588)

  roots:list[tuple[UOp, list[UOp], list[tuple[UOp, bool]]]] = []
  for root in value.toposort():
    if root.op is not Ops.MAX or root.dtype.scalar() is not dtypes.half: continue
    leaves = _flatten_binary(root, Ops.MAX)
    maybe_parsed = [_half_candidate(leaf) for leaf in leaves]
    if all(x is not None for x in maybe_parsed): roots.append((root, leaves, [x for x in maybe_parsed if x is not None]))
  if not roots: return None
  extrema, candidate_exprs, parsed_candidates = max(roots, key=lambda x:len(x[1]))
  signs = {negated for _,negated in parsed_candidates}
  if len(parsed_candidates) < 2 or len(signs) != 1: return None
  negated_candidates = signs.pop()
  loads = [load for load,_ in parsed_candidates]
  if len(set(loads)) != len(loads) or any(load.src[0].op is not Ops.INDEX or load.src[0].src[0].op is not Ops.PARAM for load in loads): return None
  source = loads[0].src[0].src[0]
  if any(load.src[0].src[0] is not source for load in loads) or source.src[0].op is not Ops.CONST: return None
  window, source_count = len(loads), int(source.src[0].arg)
  if source_count != count*window or window > 2048: return None
  try: candidate_offsets = [tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for load in loads]
  except RuntimeError: return None
  ordered = sorted(zip(candidate_offsets, candidate_exprs), key=lambda x:x[0])
  if sorted(offset for offsets,_ in ordered for offset in offsets) != list(range(source_count)): return None
  if not _first_tie_selection(value, extrema, [expr for _,expr in ordered]): return None
  candidate_plans = tuple(RKGather(source.arg.slot, 0, count, offsets=offsets) for offsets,_ in ordered)
  return _cumulative_index_image(out_param.arg.slot, count, candidate_plans, candidate_plans,
                                 negated_candidates, [window-1]*count, first_tie=True, negate_extrema=negated_candidates)

def _lower_split_arg_extrema_index(uops:list[UOp]) -> RKImage|None:
  """Select an FP16 axis ArgMax/ArgMin coordinate against a separately materialized extreme."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or out_param.dtype.scalar() is not dtypes.int: return None
  store = stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  count, out_index, value = int(out_param.src[0].arg), store.src[0].src[1], store.src[1]
  if count <= 0: return RKImage(RKTarget.RK3588)
  loads = [u for u in value.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.half and
           u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM]
  by_slot:dict[int, list[UOp]] = {}
  for load in loads:
    slot = load.src[0].src[0].arg.slot
    if load not in by_slot.setdefault(slot, []): by_slot[slot].append(load)
  extrema_groups = [(slot, group[0]) for slot,group in by_slot.items() if len(group) == 1 and
                    group[0].src[0].src[0].src[0].op is Ops.CONST and int(group[0].src[0].src[0].src[0].arg) == count]
  if len(extrema_groups) != 1 or len(by_slot) != 2: return None
  extrema_slot, extrema = extrema_groups[0]
  data_slot, candidates = next((slot,group) for slot,group in by_slot.items() if slot != extrema_slot)
  data_param = candidates[0].src[0].src[0]
  if data_param.src[0].op is not Ops.CONST: return None
  source_count = int(data_param.src[0].arg)
  direct:list[tuple[UOp, UOp, bool]] = []
  for cmp in [u for u in value.toposort() if u.op is Ops.CMPNE and extrema in u.src]:
    expr = cmp.src[1] if cmp.src[0] is extrema else cmp.src[0]
    if (parsed:=_half_candidate(expr)) is not None: direct.append((expr, parsed[0], parsed[1]))
  signs = {negated for _,_,negated in direct}
  candidate_loads = {load for _,load,_ in direct}
  if not direct or len(signs) != 1 or len(candidate_loads) != len(direct) or candidate_loads != set(candidates): return None
  negated_candidates = signs.pop()
  window = len(direct)
  if source_count != count*window or window > 2048: return None
  try:
    candidate_offsets = [tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for _,load,_ in direct]
    extrema_offsets = tuple(_gather_offsets(out_index, extrema.src[0].src[1], None, count))
  except RuntimeError: return None
  if sorted(extrema_offsets) != list(range(count)): return None
  ordered = sorted(zip(candidate_offsets, (expr for expr,_,_ in direct)), key=lambda x:x[0])
  if sorted(offset for offsets,_ in ordered for offset in offsets) != list(range(source_count)): return None
  if not _first_tie_selection(value, extrema, [expr for _,expr in ordered]): return None
  candidate_plans = tuple(RKGather(data_slot, 0, count, offsets=offsets) for offsets,_ in ordered)
  return _cumulative_index_image(out_param.arg.slot, count, candidate_plans, (RKGather(extrema_slot, 0, count, offsets=extrema_offsets),),
                                 negated_candidates, [window-1]*count, first_tie=True)

def _lower_loop_arg_extrema_index(uops:list[UOp]) -> RKImage|None:
  """Lower the two-register-loop graph used by a padded global FP16 ArgMax/ArgMin."""
  final_stores = [(store, root) for store in uops if store.op is Ops.STORE and (root:=_root_param(store.src[0])) is not None]
  if len(final_stores) != 1: return None
  store, out_param = final_stores[0]
  if (out_param.dtype.scalar() is not dtypes.int or out_param.src[0].op is not Ops.CONST or int(out_param.src[0].arg) != 1 or
      store.src[0].op is not Ops.INDEX or _index_ranges(store.src[0].src[1])): return None
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
  comparisons = [u for u in int_updates[0].src[1].toposort() if u.op is Ops.CMPNE and u.dtype.scalar() is dtypes.bool]
  inner_cmps = [u for u in comparisons if any(_half_candidate(x) is not None for x in u.src)]
  if half_candidate is None or len(inner_cmps) != 1: return None
  cmp = inner_cmps[0]
  second_candidate = global_candidate(list(cmp.src), second_range)
  if second_candidate is None or second_candidate[1] != half_candidate[1]: return None
  extrema_operands = [x for x in cmp.src if x is not second_candidate[0]]
  if len(extrema_operands) != 1 or extrema_operands[0].op is not Ops.LOAD or _root_param(extrema_operands[0].src[0]) is not None:
    return None
  inversions = [u for u in comparisons if cmp in u.src and any(x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg) for x in u.src)]
  casts = [u for u in int_updates[0].src[1].toposort() if u.op is Ops.CAST and u.dtype.scalar() is dtypes.int and
           len(inversions) == 1 and u.src == (inversions[0],)]
  if len(casts) != 1: return None
  weighted = [u for u in int_updates[0].src[1].toposort() if u.op is Ops.MUL and u.dtype.scalar() is dtypes.int and casts[0] in u.src]
  if len(weighted) != 1: return None
  coordinate = weighted[0].src[1] if weighted[0].src[0] is casts[0] else weighted[0].src[0]
  try:
    if [_eval_int(coordinate, {second_range:i}) for i in range(window)] != list(range(window, 0, -1)): return None
  except RuntimeError: return None
  final_negative = [x for x in store.src[1].src if x.op is Ops.MUL and
                    any(y.op is Ops.LOAD and y.dtype.scalar() is dtypes.int and _root_param(y.src[0]) is None for y in x.src) and
                    any(y.op is Ops.CONST and int(y.arg) == -1 for y in x.src)]
  if (store.src[1].op is not Ops.ADD or len(final_negative) != 1 or
      not any(x.op is Ops.CONST and int(x.arg) == window for x in store.src[1].src)): return None
  candidate_plans = tuple(RKGather(source.arg.slot, 0, 1, base=i) for i in range(window))
  return _cumulative_index_image(out_param.arg.slot, 1, candidate_plans, candidate_plans,
                                 half_candidate[1], [window-1], first_tie=True, negate_extrema=half_candidate[1])

def _lower_cumulative_extrema_index(uops:list[UOp]) -> RKImage|None:
  """Select unrolled cumulative MAX/MIN axis coordinates with DPU equality masks."""
  stores = [u for u in uops if u.op is Ops.STORE]
  if len(stores) != 1 or (out_param:=_root_param(stores[0].src[0])) is None or out_param.dtype.scalar() is not dtypes.int: return None
  if out_param.src[0].op is not Ops.CONST or stores[0].src[0].op is not Ops.INDEX: return None
  count, out_index, value = int(out_param.src[0].arg), stores[0].src[0].src[1], stores[0].src[1]
  if count <= 0: return RKImage(RKTarget.RK3588)
  if not any(u.op is Ops.CMPNE for u in value.toposort()): return None
  loads = [u for u in value.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.half and
           u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM]
  by_slot:dict[int, list[UOp]] = {}
  for load in loads:
    slot = load.src[0].src[0].arg.slot
    if load not in by_slot.setdefault(slot, []): by_slot[slot].append(load)
  extrema_groups = [(slot, group[0]) for slot,group in by_slot.items() if len(group) == 1 and group[0].src[0].src[1].key == out_index.key]
  if len(extrema_groups) != 1 or len(by_slot) != 2: return None
  extrema_slot, _ = extrema_groups[0]
  data_slot, candidates = next((slot,group) for slot,group in by_slot.items() if slot != extrema_slot)
  def directly_negated(load:UOp) -> bool:
    return any(node.op is Ops.MUL and load in node.src and any(src.op is Ops.CONST and float(src.arg) == -1.0 for src in node.src)
               for node in value.toposort())
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

def _lower_cumulative_extrema_index_loop(uops:list[UOp]) -> RKImage|None:
  """Lower Tinygrad's bounded-loop form used by the padded length-1022 scan."""
  final_stores = [(store, root) for store in uops if store.op is Ops.STORE and (root:=_root_param(store.src[0])) is not None and
                  root.dtype.scalar() is dtypes.int]
  if len(final_stores) != 1: return None
  store, out_param = final_stores[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  count, out_index = int(out_param.src[0].arg), store.src[0].src[1]
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
  gates = [u for u in uops if u.op is Ops.AND and half_cmps[0] in u.toposort() and prefix_cmps[0] in u.toposort()]
  if len(gates) != 1: return None
  index_maxes = [u for u in uops if u.op is Ops.MAX and u.dtype.scalar() is dtypes.int and gates[0] in u.toposort() and
                 reduce_range in u.toposort()]
  if len(index_maxes) != 1 or index_maxes[0] not in store.src[1].toposort(): return None
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

def lower_ew(uops:list[UOp]) -> RKImage:
  if (sort_compare:=_lower_sort_compare(uops)) is not None: return sort_compare
  if (occurrence_count:=_lower_occurrence_count(uops)) is not None: return occurrence_count
  if (sort_index:=_lower_sort_index_selection(uops)) is not None: return sort_index
  if (cumulative_loop:=_lower_cumulative_extrema_index_loop(uops)) is not None: return cumulative_loop
  if (cumulative_index:=_lower_cumulative_extrema_index(uops)) is not None: return cumulative_index
  if (arg_extrema:=_lower_arg_extrema_index(uops)) is not None: return arg_extrema
  if (split_arg_extrema:=_lower_split_arg_extrema_index(uops)) is not None: return split_arg_extrema
  if (loop_arg_extrema:=_lower_loop_arg_extrema_index(uops)) is not None: return loop_arg_extrema
  if (raw_int32:=_lower_raw_int32_layout(uops)) is not None: return raw_int32
  if (dot_reduction:=_lower_dot_loop_reduction(uops)) is not None: return dot_reduction
  if (variance:=_lower_centered_square_loop_reduction(uops)) is not None: return variance
  if (loop_reduction:=_lower_scalar_loop_reduction(uops)) is not None: return loop_reduction
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
_pm_sign = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_sign)])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
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
