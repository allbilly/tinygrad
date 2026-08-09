from __future__ import annotations
# ruff: noqa: E702
import base64, math, os, struct
import numpy as np
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Iterable, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import Target, cdiv, cmod, floordiv, floormod, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 28
_HEADER = struct.Struct("<4sHHHHIIIIII")  # magic/version/target, scratch/gather counts, ops/constants, phase split, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBBBBiIIi"), struct.Struct("<IIi")
_FILL = struct.Struct("<BBHI")  # dst_kind, itemsize, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, flags, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg
_ITEM_FORMAT = {1:"B", 2:"H", 4:"I"}
_RKIMAGE_U16_MAX = (1 << 16) - 1

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
RKInt16Leaf = RKArg|RKStatic|RKGather|RKMultiGather|int|tuple[UOp, UOp, UOp|None, int]|None

@dataclass(frozen=True)
class RKFill: dst: RKArg; count: int; itemsize: int = 2

@dataclass(frozen=True)
class RKEWOp:
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False
  int16_output: bool = False; int16_input: bool = False

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
    if kind == 2: out += struct.pack(f"<{g.count}{_ITEM_FORMAT[g.itemsize]}", *g.values)
    elif kind in (1, 3): out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for op in image.ew_ops:
    if op.bool_output and not op.int32_output: raise ValueError("bool output requires INT32 conversion")
    int16_to_int32 = op.int16_input and op.int32_output and not op.int16_output and not op.int32_input
    if (op.int16_output or op.int16_input) and (op.int32_output or op.int32_input) and not int16_to_int32:
      raise ValueError("conflicting integer precision")
    op_flags = (int(op.submit_barrier) | int(op.compare)<<1 | int(op.stateful)<<2 | int(op.int32_output)<<3 |
                int(op.int32_input)<<4 | int(op.bool_output)<<5 | int(op.int16_output)<<6 | int(op.int16_input)<<7)
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
    if (kind not in (0, 1, 2, 3) or (kind and naxes) or itemsize not in _ITEM_FORMAT or dst_kind not in (0, 1) or src_kind not in (0, 1) or
        dst_stride < 1 or dst_addend < 0): raise ValueError("invalid RKGather")
    if kind == 2:
      values = struct.unpack_from(f"<{count}{_ITEM_FORMAT[itemsize]}", blob, off); off += itemsize*count
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
    int16_to_int32 = op_flags & 0x88 == 0x88 and not op_flags & 0x50
    if op_flags & ~0xff or op_flags & 0x20 and not op_flags & 0x08 or \
       op_flags & 0xc0 and op_flags & 0x18 and not int16_to_int32:
      raise ValueError("invalid RKEWOp flags")
    count, ew_cfg = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg,
                         bool(op_flags & 1), bool(op_flags & 2), bool(op_flags & 4), bool(op_flags & 8), bool(op_flags & 16),
                         bool(op_flags & 32), bool(op_flags & 64), bool(op_flags & 128)))
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
_EW_ELEMS_32BIT = 8*dtypes.half.itemsize//dtypes.float.itemsize
_FP16_EXACT_INTEGER = 1 << 11
_POOL_INDEX_DIGIT_BITS = 4
_POOL_INDEX_DIGIT_RADIX = 1 << _POOL_INDEX_DIGIT_BITS
_EW_DATA_MODE_FP16 = 1 << 28
_EW_EDATA_SIZE_FP16 = 2 << 22
_EW_ALU_MIN = 1 << 16
_EW_ALU_ADD = 2 << 16
_EW_ALU_FDIV = 3 << 16
_EW_ALU_SUB = 4 << 16
_EW_ALU_ABS = 5 << 16
_EW_ALU_NEG = 6 << 16
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
_EW_CFG_NEG = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_NEG
_EW_CFG_FLOOR = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_FLOOR
_EW_CFG_CEIL = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_CEIL
_EW_CFG_LEAKY_RELU = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_MUL_PRELU | _EW_OP_TYPE_MUL
_EW_STAGE_FP32_OUT = 1 << 29  # software tag consumed before writing EW_CFG
_DPU_DATA_FORMAT_FP16 = (2<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_OUT = (5<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT16_OUT = (1<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT16 = (1<<29)|(1<<26)|1
_DPU_DATA_FORMAT_INT16_TO_INT32 = (4<<29)|(1<<26)|1
_DPU_DATA_FORMAT_INT32_OUT = (4<<29)|(2<<26)|2
_DPU_DATA_FORMAT_INT32_IN = (2<<29)|(4<<26)|4
_DPU_DATA_FORMAT_INT32 = (4<<29)|(4<<26)|4
_BS_BN_BYPASS = 1|(1<<1)|(1<<4)|(1<<6)
_BS_OW_FP32_SCALAR = (1<<8)|(1<<5)|(1<<2)|(1<<1)
_BS_CFG_COMPARE = 0x40040
_BS_ALU_COMPARE = 0x33800000
_BS_MUL_COMPARE = 0x40000000
_BN_CFG_COMPARE = 0x40082
_BN_MUL_COMPARE = 0x7c000000
_BN_RELUX_COMPARE = 0x3f800000
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_COPYSIGN, _NATIVE_FLOOR, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN,
 _NATIVE_POSITIVE_MASK, _NATIVE_RELU6, _NATIVE_SIGN) = (
   "rockchip_abs", "rockchip_ceil", "rockchip_copysign", "rockchip_floor", "rockchip_leaky_relu", "rockchip_mask_mul",
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
def _reduction_stride(count:int) -> int: return round_up(count*2, 64)
def _int32_tiles_bytes(count:int) -> int: return cdiv(count, 4) * 64
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff
def _int16_low_bytes(source:RKArg, out_slot:int, count:int, stride:int=2) -> RKGather:
  return RKGather(source.index, out_slot, count, base=source.addend, axes=((1, count, stride),),
                  dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)

def _emit_stateful_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                         int32_output:bool=False, int32_input:bool=False,
                         int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False) -> RKStage:
  """Emit a self-contained DPU EW stage, optionally consuming or producing native integers."""
  native_int16, native_int32 = int16_input and int16_output, int32_input and int32_output
  int16_to_int32 = int16_input and int32_output and not int16_output and not int32_input
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if native_int32 else \
          _EW_ELEMS_32BIT if int32_output or int32_input or fp32_output else _MAX_EW_ELEMS_FP16
  if not 0 < count <= limit:
    raise ValueError(f"stateful EW count {count} out of range")
  lanes = 4 if int32_input else 8
  is_div = ew_cfg == _EW_CFG[Ops.FDIV]
  width = (count + lanes-1) // lanes - 1
  pipeline:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_BS_CFG,_BS_BN_BYPASS),(_DPU,rk.REG_DPU_BN_CFG,_BS_BN_BYPASS),
    (_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR if int16_to_int32 or fp32_output and count == 1 else 2),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,0 if fp32_output and count == 1 else 3 if fp32_output else lanes-1),
    (_DPU,rk.REG_DPU_WDMA_SIZE_1,width),(_DPU,rk.REG_DPU_BN_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0))
  if compare: pipeline += ((_DPU,rk.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(_DPU,rk.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),
    (_DPU,rk.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_CFG,_BN_CFG_COMPARE),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE))
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP32_OUT if fp32_output else
                                  _DPU_DATA_FORMAT_INT16 if native_int16 else _DPU_DATA_FORMAT_INT32 if native_int32 else
                                  _DPU_DATA_FORMAT_INT16_TO_INT32 if int16_to_int32 else
                                  _DPU_DATA_FORMAT_INT32_OUT if int32_output else _DPU_DATA_FORMAT_INT16_OUT if int16_output else
                                  _DPU_DATA_FORMAT_INT32_IN if int32_input else _DPU_DATA_FORMAT_FP16)) + \
    (((_DPU,rk.REG_DPU_DST_SURF_STRIDE,1<<4),) if int16_to_int32 or fp32_output else ()) + (
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0 if fp32_output and count == 1 else ((lanes-1)<<16)|(lanes-1))) + pipeline + (
    (_DPU,rk.REG_DPU_EW_CFG,_EW_CFG_COMMON|1 if compare else
                               (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else
                               ew_cfg & ~_EW_OP_CVT_BYPASS if native_int16 or int16_to_int32 else ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,0 if fp32_output else 1 if int32_output or int16_output or is_div else (1<<16)|1),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),
    (_DPU,rk.REG_DPU_SURFACE_ADD,(2 if native_int16 or int16_to_int32 else 4)<<4),(_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,lanes-1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input else 2)<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  rdma_feature = ((4 if int32_input else 1 if int16_input else 2)<<15)|(15<<11)|\
                 ((4 if int32_input else 1 if int16_input else 2)<<5)|(0 if is_div or int16_input else 1<<3)|1
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature))
  return RKStage(tuple(commands), tuple(relocs))

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                  stateful:bool=False, int32_output:bool=False, int32_input:bool=False,
                  int16_output:bool=False, int16_input:bool=False) -> RKStage:
  """Build one DPU EW command body without its PC-chain tail."""
  if ew_cfg & _EW_STAGE_FP32_OUT:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_OUT, fp32_output=True)
  if compare or stateful or int32_output or int32_input or int16_output or int16_input:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg, compare, int32_output, int32_input, int16_output, int16_input)
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
  elif u.op in (Ops.RANGE, Ops.SPECIAL): ret = env[u]
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
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
  elif u.op in (Ops.RANGE, Ops.SPECIAL): ret = env[u]
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
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

def _index_ranges(index:UOp) -> list[UOp]: return [u for u in index.toposort() if u.op in (Ops.RANGE, Ops.SPECIAL)]

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

def _local_add_loop(uops:list[UOp], out_index:UOp) -> tuple[UOp, int, UOp, list[UOp]]|None:
  """Parse one initialized local accumulator updated by `acc + term` over a constant range."""
  out_ranges = _index_ranges(out_index)
  reduce_ranges = [u for u in uops if u.op is Ops.RANGE and u not in out_ranges]
  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  if len(reduce_ranges) != 1 or len(local_stores) != 2: return None
  reduce_range = reduce_ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or (groups:=int(reduce_range.src[0].arg)) <= 0: return None
  updates = [store.src[1] for store in local_stores if store.src[1].op is Ops.ADD and reduce_range in store.src[1].toposort()]
  if len(updates) != 1: return None
  acc = next((x for x in updates[0].src if _local_load(x) is not None), None)
  term = next((x for x in updates[0].src if x is not acc), None)
  return None if acc is None or term is None else (reduce_range, groups, term, local_stores)

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

def _spaced_reduction_gathers(src_slot:int, dst_slot:int, rows:int, blocks:list[tuple[int, ...]]|tuple[tuple[int, ...], ...],
                               stride:int|None=None, fill_bits:int=0) -> tuple[RKGather, ...]:
  stride = _reduction_stride(rows) if stride is None else stride
  if stride < rows*2 or stride % 2: raise ValueError("invalid reduction stride")
  stride_lanes = stride//2
  if rows != 1:
    return tuple(RKGather(src_slot, dst_slot, rows, offsets=block, fill_bits=fill_bits,
                          dst_addend=i*stride_lanes) for i,block in enumerate(blocks))
  offsets = tuple(block[0] for block in blocks)
  direct = offsets == tuple(range(len(blocks)))
  return (RKGather(src_slot, dst_slot, len(blocks), axes=((1, len(blocks), 1),) if direct else (),
                   offsets=() if direct else offsets, fill_bits=fill_bits, dst_stride=stride_lanes),)

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
  return _static_values(out_index, expr, count, _fp16_bits)

def _static_int_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  """Evaluate a compile-time integer expression in compact output order."""
  return _static_values(out_index, expr, count, int)

def _static_int_vectors(out_index:UOp, exprs:tuple[UOp, ...], count:int) -> tuple[tuple[int, ...], ...]:
  """Vector-evaluate static integer rows with one shared index-expression cache."""
  ranges = _index_ranges(out_index)
  if any(r not in ranges for expr in exprs for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  envs = _iter_range_env(ranges)
  vector_env = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}
  cache:dict[UOp, np.ndarray] = {}
  dst = np.broadcast_to(_eval_vector(out_index, vector_env, cache), len(envs)).astype(np.int64)
  if len(envs) != count or np.any((dst < 0) | (dst >= count)) or not np.array_equal(np.sort(dst), np.arange(count)):
    return tuple(_static_int_vector(out_index, expr, count) for expr in exprs)
  order = np.argsort(dst)
  return tuple(tuple(int(x) for x in np.broadcast_to(_eval_vector(expr, vector_env, cache), len(envs))[order]) for expr in exprs)

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op in (Ops.RANGE, Ops.SPECIAL): return 0, {u: 1}
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

def _validate_gather_bounds(plan:RKGather, source_count:int) -> None:
  if plan.offsets: low, high = min(plan.offsets, default=0), max(plan.offsets, default=-1)
  else:
    low = high = plan.base
    for _,limit,stride in plan.axes:
      if stride < 0: low += (limit-1)*stride
      else: high += (limit-1)*stride
  if low < -1 or high >= source_count: raise RuntimeError("RKPLAN_REJECT:gather_index")

def _gather_cache_key(plans:Iterable[RKGather]) -> tuple:
  return tuple((p.src_index, p.count, p.base, p.axes, p.offsets, p.fill_bits, p.values, p.partial,
                p.dst_stride, p.dst_addend, p.itemsize, p.src_kind) for p in plans)

def _selection_gather(u:UOp, out_index:UOp, count:int, oslot:int, selection_cache:dict[UOp, tuple[bool, bool]]|None=None,
                      static_cache:dict[UOp, bool]|None=None, dtype:DType=dtypes.half) -> RKGather|RKMultiGather|None:
  """Collapse a static selection tree into raw offsets from one or more same-typed source buffers."""
  encode = _int16_bits if dtype is dtypes.int16 else _fp16_bits
  seen = selection_cache if selection_cache is not None else {}
  def selection_tree(x:UOp) -> tuple[bool, bool]:
    if x in seen: return seen[x]
    if x.op is Ops.CONST: ret = (x.dtype.scalar() is dtype, False)
    elif x.op is Ops.CAST and x.dtype.scalar() is dtype: ret = selection_tree(x.src[0])
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
    if x.op is Ops.CONST and x.dtype.scalar() is dtype:
      bits = encode(x.arg)
      ret = empty, empty, np.full(len(envs), bits, dtype=np.int64)
    elif x.op is Ops.CAST and x.dtype.scalar() is dtype: ret = selected(x.src[0])
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
      if (param.dtype.scalar() is not dtype or param.arg.slot == oslot or param.src[0].op is not Ops.CONST or
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
    fill_bits = _fp16_bits(u.src[1].arg if len(u.src) > 1 else 0)
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

def _loaded_equality_rows(out_index:UOp, count:int, pairs:tuple[tuple[UOp, UOp], ...], dtype:DType,
                          same_source:bool=False) -> tuple[int, int, tuple[tuple[int, ...], ...], tuple[int, ...]]|None:
  """Resolve one common load and its candidate loads into bounded striped-row sources."""
  if (split:=_split_load_pairs(pairs)) is None: return None
  current, candidates = split
  params = tuple(_root_param(load.src[0]) for load in (*candidates, current))
  if any(param is None or param.dtype.scalar() is not dtype or param.src[0].op is not Ops.CONST for param in params): return None
  concrete = tuple(param for param in params if param is not None)
  candidate_params, current_param = concrete[:-1], concrete[-1]
  if len({param.arg.slot for param in candidate_params}) != 1 or (same_source and candidate_params[0].arg.slot != current_param.arg.slot): return None
  try:
    candidate_offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in candidates)
    current_offsets = _gather_offsets(out_index, current.src[0].src[1], None, count)
  except RuntimeError: return None
  if (any(not 0 <= offset < int(param.src[0].arg) for offsets,param in zip(candidate_offsets, candidate_params) for offset in offsets) or
      any(not 0 <= offset < int(current_param.src[0].arg) for offset in current_offsets)): return None
  return candidate_params[0].arg.slot, current_param.arg.slot, candidate_offsets, current_offsets

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64
  return vector_bytes, vector_bytes//2, rows*vector_bytes//2

def _stripe_gathers(src_slot:int, dst_slot:int, count:int, rows:Iterable[Iterable[int]], vector_lanes:int, *,
                    values:bool=False, itemsize:int=2) -> tuple[RKGather, ...]:
  """Pack candidate or repeated-current rows into one aligned lane matrix."""
  return tuple(RKGather(src_slot, dst_slot, count, offsets=() if values else tuple(row), values=tuple(row) if values else (),
                        dst_addend=i*vector_lanes, itemsize=itemsize) for i,row in enumerate(rows))

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int, int16:bool=False, int32:bool=False) -> RKArg:
  """Append a balanced row reduction, making its first dependent stage self-contained."""
  if int16 and int32: raise ValueError("conflicting integer reduction precision")
  integer = int16 or int32
  first = True
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      ops.append(RKEWOp(active[i], active[i], active[i+1], count, cfg, submit_barrier=first and not integer,
                        stateful=first and not integer, int32_input=int32, int32_output=int32,
                        int16_input=int16, int16_output=int16))
      first = False; reduced.append(active[i])
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return active[0]

def _ew_ieee_positive_mask(ops:list[RKEWOp], arg:Callable[[int], RKArg], value:int,
                           constants:tuple[int, int, int], temps:tuple[int, ...], count:int) -> RKArg:
  """Append an IEEE-correct `0 < value` mask using only DPU EW stages."""
  one, maximum, negative_maximum = constants
  (high_delta, high, low_delta, low, nan, positive_inf, either_inf, finite,
   numeric_positive, finite_positive, result) = temps
  ops.extend((
    RKEWOp(arg(high_delta), arg(value), arg(maximum), count, _EW_CFG[Ops.SUB]),
    RKEWOp(arg(high), arg(high_delta), arg(high_delta), count, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(low_delta), arg(negative_maximum), arg(value), count, _EW_CFG[Ops.SUB], stateful=True),
    RKEWOp(arg(low), arg(low_delta), arg(low_delta), count, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(nan), arg(high), arg(low), count, _EW_CFG[Ops.MUL], stateful=True),
    RKEWOp(arg(positive_inf), arg(high), arg(nan), count, _EW_CFG[Ops.SUB]),
    RKEWOp(arg(either_inf), arg(high), arg(low), count, _EW_CFG[Ops.MAX]),
    RKEWOp(arg(finite), arg(one), arg(either_inf), count, _EW_CFG[Ops.SUB]),
    RKEWOp(arg(numeric_positive), arg(value), arg(value), count, _EW_CFG[Ops.MAX], compare=True),
    RKEWOp(arg(finite_positive), arg(finite), arg(numeric_positive), count, _EW_CFG[Ops.MUL], stateful=True),
    RKEWOp(arg(result), arg(positive_inf), arg(finite_positive), count, _EW_CFG[Ops.MAX])))
  return arg(result)

def _ew_eq_mask(ops:list[RKEWOp], arg:Callable[[int], RKArg], lhs:int, rhs:int, temps:tuple[int, int, int, int], one:int,
                lanes:int, barriers:tuple[bool, bool]=(False, True)) -> RKArg:
  """Append SUB, ABS, nonzero comparison, and inversion for an FP16 equality mask."""
  diff, magnitude, unequal, equal = temps
  ops.extend((RKEWOp(arg(diff), arg(lhs), arg(rhs), lanes, _EW_CFG[Ops.SUB], submit_barrier=barriers[0], stateful=barriers[0]),
              RKEWOp(arg(magnitude), arg(diff), arg(diff), lanes, _EW_CFG_ABS, submit_barrier=barriers[1], stateful=barriers[1]),
              RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), lanes, _EW_CFG[Ops.MAX], compare=True),
              RKEWOp(arg(equal), arg(one), arg(unequal), lanes, _EW_CFG[Ops.SUB], stateful=True)))
  return arg(equal)

def _ew_integer_eq_mask(ops:list[RKEWOp], arg:Callable[[int], RKArg], lhs:int, rhs:int,
                        temps:tuple[int, int, int, int], one:int, lanes:int) -> RKArg:
  """Compare exact nonnegative FP16 integers without the reset-heavy general comparison stage."""
  diff, magnitude, unequal, equal = temps
  ops.extend((RKEWOp(arg(diff), arg(lhs), arg(rhs), lanes, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True),
              RKEWOp(arg(magnitude), arg(diff), arg(diff), lanes, _EW_CFG_ABS, submit_barrier=True, stateful=True),
              RKEWOp(arg(unequal), arg(magnitude), arg(one), lanes, _EW_CFG_MIN, submit_barrier=True, stateful=True),
              RKEWOp(arg(equal), arg(one), arg(unequal), lanes, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True)))
  return arg(equal)

@dataclass(frozen=True)
class RKByteEquality:
  scratch_sizes:tuple[int, ...]; gathers:tuple[RKGather, ...]; mid_gathers:tuple[RKGather, ...]
  pre_ops:tuple[RKEWOp, ...]; mask_ops:tuple[RKEWOp, ...]; mask:RKArg
  vector_bytes:int; vector_lanes:int; matrix_lanes:int; tiles:int

RKIndexEquality = tuple[int, int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
RKCoordinateRows = tuple[tuple[int, ...], ...]
def _int32_equality_matrix(indices:tuple[RKIndexEquality, ...], count:int,
                           alternate_coordinates:tuple[tuple[RKCoordinateRows, ...], ...]|None=None) -> RKByteEquality|None:
  """Expand one or more exact dynamic INT32 byte equalities over aligned candidate rows."""
  rows = len(indices[0][3]) if indices else 0
  vector_bytes, vector_lanes, matrix_lanes = ((count*2, count, count) if rows == 1 else _stripe_layout(count, rows))
  coordinate_sets = tuple((index[3],)+alternates for index,alternates in zip(
    indices, alternate_coordinates if alternate_coordinates is not None else ((),)*len(indices)))
  if (not rows or any(len(offsets) != rows or len(coords) != rows or
                      any(len(row) != count for row in (*offsets, *coords)) for _,_,offsets,coords in indices) or
      len(coordinate_sets) != len(indices) or any(not sets or any(len(coords) != rows or
        any(len(row) != count for row in coords) for coords in sets) for sets in coordinate_sets) or
      matrix_lanes > _MAX_EW_ELEMS_FP16): return None
  one, next_slot = 0, 1
  layouts:list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
  for sets in coordinate_sets:
    raw = tuple(range(next_slot, next_slot+4)); half = tuple(range(next_slot+4, next_slot+8))
    matrix = tuple(range(next_slot+8, next_slot+12)); next_slot += 12
    coords = tuple(tuple(range(next_slot+i*4, next_slot+(i+1)*4)) for i in range(len(sets))); next_slot += len(sets)*4
    layouts.append((raw, half, matrix, coords))
  tiles = next_slot; next_slot += 1
  diff, magnitude, unequal, equal = range(next_slot, next_slot+4); next_slot += 4
  masks:list[tuple[int, ...]] = []
  for sets in coordinate_sets:
    masks.append(tuple(range(next_slot, next_slot+len(sets)))); next_slot += len(sets)
  scratch_sizes = [_scratch_bytes(matrix_lanes)] * next_slot
  scratch_sizes[tiles] = _int32_tiles_bytes(max(index_count for _,index_count,_,_ in indices))
  gathers:list[RKGather] = []
  mid:list[RKGather] = []
  pre:list[RKEWOp] = []
  def arg(slot:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot)
  for (index_slot,index_count,index_offset_rows,_),sets,(raw,half,matrix,coordinate_slots) in zip(indices, coordinate_sets, layouts):
    for slot in raw: scratch_sizes[slot] = index_count*4
    for slot in half: scratch_sizes[slot] = _scratch_bytes(index_count)
    gathers.extend(RKGather(index_slot, slot, index_count,
      offsets=tuple(lane*4+byte for lane in range(index_count)), dst_stride=4, itemsize=1) for byte,slot in enumerate(raw))
    for coordinate_matrix,coordinate_group in zip(sets, coordinate_slots):
      for byte,coordinate_slot in enumerate(coordinate_group):
        bits = tuple(tuple(_fp16_bits((value >> (byte*8)) & 0xff) for value in row) for row in coordinate_matrix)
        gathers.extend(_stripe_gathers(index_slot, coordinate_slot, count, bits, vector_lanes, values=True))
    mid.extend(RKGather(src, dst, count, offsets=index_offsets, dst_addend=row*vector_lanes, src_kind=RKBufferKind.SCRATCH)
               for src,dst in zip(half, matrix) for row,index_offsets in enumerate(index_offset_rows))
    pre.extend(RKEWOp(arg(dst), arg(src), arg(tiles), index_count, _EW_CFG[Ops.MAX], int32_input=True) for src,dst in zip(raw, half))
  ops:list[RKEWOp] = []
  axis_masks:list[RKArg] = []
  for (_,_,matrix,coordinate_slots),axis_slots in zip(layouts, masks):
    for coordinate_group,mask_slot in zip(coordinate_slots, axis_slots):
      byte_result = arg(one)
      for lhs,rhs in zip(matrix, coordinate_group):
        byte_equal = _ew_integer_eq_mask(ops, arg, lhs, rhs, (diff, magnitude, unequal, equal), one, matrix_lanes)
        ops.append(RKEWOp(arg(mask_slot), byte_result, byte_equal, matrix_lanes,
                          _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
        byte_result = arg(mask_slot)
    axis_mask = arg(axis_slots[0])
    for alternate_mask in axis_slots[1:]:
      ops.append(RKEWOp(axis_mask, axis_mask, arg(alternate_mask), matrix_lanes,
                        _EW_CFG[Ops.MAX], submit_barrier=True, stateful=True))
    axis_masks.append(axis_mask)
  final_mask:RKArg = axis_masks[0]
  for axis_mask in axis_masks[1:]:
    ops.append(RKEWOp(final_mask, final_mask, axis_mask, matrix_lanes,
                      _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  return RKByteEquality(tuple(scratch_sizes), tuple(gathers), tuple(mid), tuple(pre), tuple(ops), final_mask,
                        vector_bytes, vector_lanes, matrix_lanes, tiles)

def _reduce_arena(ops:list[RKEWOp], active:list[int], count:int, cfg:int, arena:Callable[[int], RKArg],
                  out:RKArg|None=None, fp32_out:bool=False, int16:bool=False) -> RKArg:
  """Append a balanced in-place arena reduction and optionally write its final stage directly to output."""
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs, final = active[i], active[i+1], len(active) == 2 and out is not None
      dst = out if final and out is not None else arena(lhs)
      ops.append(RKEWOp(dst, arena(lhs), arena(rhs), count,
                        cfg | (_EW_STAGE_FP32_OUT if fp32_out and final else 0),
                        int16_input=int16, int16_output=int16))
      reduced.append(lhs)
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return out if out is not None else arena(active[0])

def _reduction_image(out_slot:int, rows:int, source_slot:int, blocks:list[tuple[int, ...]]|tuple[tuple[int, ...], ...],
                     constants:tuple[float, ...], cfg:int, fp32_out:bool, post_scale:float,
                     prepare:Callable[[list[RKEWOp], RKArg, dict[float, int]], None]|None=None) -> RKImage:
  """Materialize row blocks, apply an optional lane transform, reduce them, and write the typed result."""
  const_slots, data_slot = {value:i for i,value in enumerate(constants)}, len(constants)
  stride = _reduction_stride(rows)
  gathers = _spaced_reduction_gathers(source_slot, data_slot, rows, blocks, stride)
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, data_slot, offset)
  ops:list[RKEWOp] = []
  active = [i*stride for i in range(len(blocks))]
  if prepare is not None:
    for offset in active: prepare(ops, arena(offset), const_slots)
  out = RKArg(RKBufferKind.ARG, out_slot)
  reduced = _reduce_arena(ops, active, rows, cfg, arena, out if post_scale == 1.0 else None, fp32_out)
  if post_scale != 1.0:
    ops.append(RKEWOp(out, reduced, RKArg(RKBufferKind.SCRATCH, const_slots[post_scale]), rows,
                      _EW_CFG[Ops.MUL] | (_EW_STAGE_FP32_OUT if fp32_out else 0)))
  scratch = tuple(RKScratch(rows*2) for _ in constants) + (RKScratch(len(blocks)*stride),)
  return RKImage(RKTarget.RK3588, scratch, b"".join(struct.pack("<e", value) for value in constants), gathers=gathers, ew_ops=tuple(ops))

def _int16_extrema_image(out_slot:int, rows:int, source_slot:int, blocks:tuple[tuple[int, ...], ...], minimum:bool) -> RKImage:
  """Materialize signed INT16 candidates and reduce them with native MIN/MAX."""
  stride = _reduction_stride(rows)
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, 0, offset)
  ops:list[RKEWOp] = []
  _reduce_arena(ops, [i*stride for i in range(len(blocks))], rows, _EW_CFG_MIN if minimum else _EW_CFG[Ops.MAX], arena,
                RKArg(RKBufferKind.ARG, out_slot), int16=True)
  identity = _int16_bits(32767 if minimum else -32768)
  return RKImage(RKTarget.RK3588, (RKScratch(len(blocks)*stride),),
                 gathers=_spaced_reduction_gathers(source_slot, 0, rows, blocks, stride, identity), ew_ops=tuple(ops))

def _int16_cumulative_candidate(expr:UOp, minimum:bool) -> UOp|None:
  """Recover one prefix-scan load from Tinygrad's native MAX or complemented-MAX form."""
  if not minimum: return expr if expr.op is Ops.LOAD else None
  if (candidate:=_int16_nonconst(expr, -1)) is not None: return candidate if candidate.op is Ops.LOAD else None
  if expr.op is not Ops.WHERE or not _int16_const(expr.src[1], -32768) or expr.src[2].op is not Ops.WHERE: return None
  nested = expr.src[2]
  if nested.src[0].key != expr.src[0].key or not _int16_const(nested.src[1], 0): return None
  candidate = _int16_nonconst(nested.src[2], -1)
  return candidate if candidate is not None and candidate.op is Ops.LOAD else None

def _lower_unrolled_int16_cumulative_extrema(uops:list[UOp]) -> RKImage|None:
  """Lower a fully unrolled one-dimensional signed INT16 prefix MIN/MAX."""
  if (output:=_output_store(uops, dtypes.int16)) is None: return None
  _, out_param, count, out_index, root = output
  minimum = (maximum:=_int16_nonconst(root, -1)) is not None
  maximum = maximum if minimum else root
  if maximum is None or maximum.op is not Ops.MAX: return None
  candidates = tuple(_int16_cumulative_candidate(expr, minimum) for expr in _flatten_binary(maximum, Ops.MAX))
  if any(candidate is None for candidate in candidates): return None
  loads = tuple(candidate for candidate in candidates if candidate is not None)
  params = {_root_param(load.src[0]) for load in loads if load.src and load.src[0].op is Ops.INDEX}
  if len(params) != 1 or (source:=next(iter(params))) is None or source.src[0].op is not Ops.CONST or int(source.src[0].arg) != count:
    return None
  try: blocks = tuple(_gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count) for load in loads)
  except RuntimeError: return None
  if len(blocks) != count or any(sorted(offset for block in blocks if (offset:=block[dst]) >= 0) != list(range(dst+1))
                                 for dst in range(count)): return None
  return _int16_extrema_image(out_param.arg.slot, count, source.arg.slot, blocks, minimum)

def _lower_int16_loop_extrema(uops:list[UOp]) -> RKImage|None:
  """Lower signed INT16 scalar or one-dimensional cumulative extrema loops."""
  if (output:=_output_store(uops, dtypes.int16, allow_local=True)) is None: return None
  store, out_param, _, _, root = output
  if (shape:=_loop_reduction_shape(store, out_param, uops)) is None: return None
  rows, envs, reduce_range, groups = shape
  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  updates = [u for u in local_stores if reduce_range in u.toposort()]
  initializers = [u for u in local_stores if reduce_range not in u.toposort()]
  if len(updates) != 1 or len(initializers) != 1 or groups < 2: return None
  initial = initializers[0].src[1]
  if not _int16_const(initial, -32768): return None

  update = updates[0].src[1]
  minimum = _local_load(root) is None
  final_value = _int16_nonconst(root, -1) if minimum else root
  if final_value is None or _local_load(final_value) is None: return None
  if update.op is not Ops.MAX: return None
  accumulator = next((x for x in update.src if _local_load(x) is not None), None)
  term = next((x for x in update.src if _local_load(x) is None), None)
  if accumulator is None or term is None: return None
  loads = [u for u in term.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int16 and
           u.src and u.src[0].op is Ops.INDEX and _root_param(u.src[0]) is not None]
  if len(loads) != 1: return None
  load = loads[0]
  if minimum and not any(_int16_nonconst(u, -1) is load for u in term.toposort()): return None
  source = _root_param(load.src[0])
  if source is None or source.arg.slot == out_param.arg.slot or source.src[0].op is not Ops.CONST: return None
  def candidate_offset(env:dict[UOp, int], lane:int) -> int:
    values = {**env, reduce_range:lane}
    return _eval_int(load.src[0].src[1], values) if len(load.src) < 3 or bool(_eval_expr(load.src[2], values, {})) else -1
  try:
    blocks = tuple(tuple(candidate_offset(env, lane) for env in envs) for lane in range(groups))
  except RuntimeError: return None
  input_count = int(source.src[0].arg)
  scalar = input_count == rows*groups and sorted(offset for block in blocks for offset in block if offset >= 0) == list(range(input_count))
  cumulative = input_count == rows == groups and all(sorted(value for block in blocks if (value:=block[dst]) >= 0) == list(range(dst+1))
                                                    for dst in range(rows))
  if not scalar and not cumulative: return None
  return _int16_extrema_image(out_param.arg.slot, rows, source.arg.slot, blocks, minimum)

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

def _lower_fp16_int32_cast(output:RKOutput) -> RKImage|None:
  """Truncate a direct FP16 load on DPU before the terminal INT32 conversion."""
  root = output[4]
  if (root.op is not Ops.CAST or root.dtype.scalar() is not dtypes.int or len(root.src) != 1 or
      root.src[0].op is not Ops.LOAD or root.src[0].dtype.scalar() is not dtypes.half): return None
  return _typed_int_image(output, _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=root.src)))

def _lower_fp16_fp32_cast(output:RKOutput) -> RKImage|None:
  """Widen a direct or statically gathered FP16 load with the DPU output converter."""
  _, out_param, count, out_index, root = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  if (root.op is not Ops.CAST or root.dtype.scalar() is not dtypes.float or len(root.src) != 1 or
      (load:=root.src[0]).op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or
      len(load.src) != 1 or load.src[0].op is not Ops.INDEX or
      (source:=_root_param(load.src[0])) is None or source.src[0].op is not Ops.CONST): return None
  try: offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
  except RuntimeError: return None
  if any(not 0 <= offset < int(source.src[0].arg) for offset in offsets): return None
  direct = count <= _EW_ELEMS_32BIT and offsets == tuple(range(count))
  values:tuple[RKArg, ...]; gathers:tuple[RKGather, ...]; scratch:tuple[RKScratch, ...]
  if direct:
    values, gathers, scratch = (RKArg(RKBufferKind.ARG, source.arg.slot),), (), ()
  else:
    groups = tuple(range(0, count, _EW_ELEMS_32BIT))
    values = tuple(RKArg(RKBufferKind.SCRATCH, 0, group//_EW_ELEMS_32BIT*16) for group in groups)
    gathers = tuple(RKGather(source.arg.slot, 0, min(_EW_ELEMS_32BIT, count-group),
                             offsets=offsets[group:group+_EW_ELEMS_32BIT],
                             dst_addend=group//_EW_ELEMS_32BIT*8) for group in groups)
    scratch = (RKScratch((count+_EW_ELEMS_32BIT-1)//_EW_ELEMS_32BIT*16),)
  ops = tuple(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot, group*16), value, value,
                     min(_EW_ELEMS_32BIT, count-group*_EW_ELEMS_32BIT),
                     _EW_CFG[Ops.MAX] | _EW_STAGE_FP32_OUT) for group,value in enumerate(values))
  return RKImage(RKTarget.RK3588, scratch, gathers=gathers, ew_ops=ops)

def _int16_byte_sum(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], source_slot:int,
                    operands:tuple[tuple[RKArg, ...], ...], count:int) -> tuple[RKArg, ...]:
  """Add four-byte operands exactly modulo 2**32 with native INT16 byte carries."""
  if len(operands) < 2 or any(len(operand) != 4 for operand in operands): raise ValueError("invalid byte sum")
  def arg(slot:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot)
  zero, one, byte_base = (scratch(count*2) for _ in range(3))
  thresholds = tuple(scratch(count*2) for _ in range(len(operands)-1))
  gathers.extend((RKGather(source_slot, zero, count, values=(0,)*count),
                  RKGather(source_slot, one, count, values=(1,)*count),
                  RKGather(source_slot, byte_base, count, values=(256,)*count)))
  gathers.extend(RKGather(source_slot, slot, count, values=(256*(level+1)-1,)*count)
                 for level,slot in enumerate(thresholds))
  carry, results = arg(zero), []
  int16 = dict(int16_input=True, int16_output=True)
  for byte in range(4):
    total = _reduce_rows(ops, [operand[byte] for operand in operands], count, _EW_CFG[Ops.ADD], int16=True)
    if byte:
      slot = scratch(count*2); ops.append(RKEWOp(arg(slot), total, carry, count, _EW_CFG[Ops.ADD], **int16)); total = arg(slot)
    bits:list[RKArg] = []
    for threshold in thresholds:
      delta, positive, bit = (scratch(count*2) for _ in range(3))
      ops.extend((RKEWOp(arg(delta), total, arg(threshold), count, _EW_CFG[Ops.SUB], **int16),
                  RKEWOp(arg(positive), arg(delta), arg(zero), count, _EW_CFG[Ops.MAX], **int16),
                  RKEWOp(arg(bit), arg(positive), arg(one), count, _EW_CFG_MIN, **int16)))
      bits.append(arg(bit))
    carry = _reduce_rows(ops, bits, count, _EW_CFG[Ops.ADD], int16=True)
    scaled, result = scratch(count*2), scratch(count*2)
    ops.extend((RKEWOp(arg(scaled), carry, arg(byte_base), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(arg(result), total, arg(scaled), count, _EW_CFG[Ops.SUB], **int16)))
    results.append(arg(result))
  return tuple(results)

def _lower_int32_byte_add(output:RKOutput) -> RKImage|None:
  """Add bounded arbitrary INT32 inputs exactly modulo 2**32 using native INT16 byte carries."""
  _, out_param, count, out_index, root = output
  terms = _flatten_binary(root, Ops.ADD)
  if not 1 <= count <= _MAX_EW_ELEMS_FP16 or not 2 <= len(terms) <= 8: return None
  inputs:list[tuple[UOp, tuple[int, ...]]] = []
  try:
    for term in terms:
      if term.op is not Ops.LOAD or term.dtype.scalar() is not dtypes.int or not term.src or term.src[0].op is not Ops.INDEX: return None
      param = _root_param(term.src[0])
      if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: return None
      offsets = _gather_offsets(out_index, term.src[0].src[1], term.src[2] if len(term.src) == 3 else None, count)
      if any(offset < -1 or offset >= int(param.src[0].arg) for offset in offsets): return None
      inputs.append((param, offsets))
  except RuntimeError: return None
  scratch_sizes:list[int] = []
  def scratch(size:int=count*2) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  constant_source = inputs[0][0].arg.slot
  operands:list[tuple[RKArg, ...]] = []
  for param,offsets in inputs:
    values = []
    for byte in range(4):
      slot = scratch()
      gathers.append(RKGather(param.arg.slot, slot, count,
        offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in offsets), dst_stride=2, itemsize=1))
      values.append(arg(slot))
    operands.append(tuple(values))
  results = _int16_byte_sum(ops, gathers, scratch, constant_source, tuple(operands), count)
  post = tuple(RKGather(result.index, out_param.arg.slot, count,
    offsets=tuple(result.addend+lane*2 for lane in range(count)), dst_stride=4, dst_addend=byte,
    dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1) for byte,result in enumerate(results))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers),
                 ew_ops=tuple(ops), post_gathers=post)

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

def _greater_half_load(predicate:UOp) -> tuple[UOp, float]|None:
  """Recognize a scalar-threshold forward predicate `constant < fp16_load`."""
  predicate = _unwrap_condition(predicate)
  if predicate.op is not Ops.CMPLT or len(predicate.src) != 2: return None
  threshold, load = predicate.src
  if (threshold.op is not Ops.CONST or threshold.dtype.scalar() is not dtypes.half or
      load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or not load.src or load.src[0].op is not Ops.INDEX): return None
  return load, float(threshold.arg)

def _positive_half_load(predicate:UOp) -> UOp|None:
  """Recognize the exact forward predicate `0 < fp16_load`."""
  parsed = _greater_half_load(predicate)
  return parsed[0] if parsed is not None and parsed[1] == 0.0 else None

def _prefix_load_rows(out_index:UOp, count:int, terms:list[UOp], dtype:DType,
                      parse:Callable[[UOp], tuple[UOp, UOp|None]|None], repeat:int=1) -> tuple[UOp, tuple[tuple[int, ...], ...]]|None:
  """Resolve one gated load per term and prove each output sees the exact source prefix."""
  source:UOp|None = None
  rows:list[tuple[int, ...]] = []
  try:
    for term in terms:
      if (parsed:=parse(term)) is None: return None
      load, valid_expr = parsed
      param = _root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None
      if (param is None or param.dtype.scalar() is not dtype or param.src[0].op is not Ops.CONST or
          source is not None and param.arg.slot != source.arg.slot): return None
      source = param
      valid = _static_int_vector(out_index, valid_expr, count) if valid_expr is not None else (1,)*count
      if any(bit not in (0, 1) for bit in valid): return None
      offsets = _gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count)
      rows.append(tuple(offset if bit else -1 for offset,bit in zip(offsets, valid)))
  except RuntimeError: return None
  if (source is None or int(source.src[0].arg)*repeat != count or
      any(sorted(offset for row in rows if (offset:=row[lane]) >= 0) != [i//repeat for i in range(lane+1)]
          for lane in range(count))): return None
  return source, tuple(rows)

def _fp16_prefix_image(out_slot:int, count:int, source_slot:int, rows:tuple[tuple[int, ...], ...],
                       mode:str, threshold:float) -> RKImage|None:
  """Emit a blocked FP16 predicate matrix reduction and exact INT32 prefix output."""
  if not rows or any(len(row) != count for row in rows) or mode not in ("greater", "nonzero"): return None
  block_lanes = (_MAX_EW_ELEMS_FP16//len(rows)//32)*32
  if block_lanes < 1: return None
  max_matrix_lanes = _stripe_layout(min(count, block_lanes), len(rows))[2]
  scratch_sizes = [_scratch_bytes(max_matrix_lanes)] * 4
  one, maximum, negative_maximum, threshold_slot = range(4)
  def scratch(count:int) -> int: scratch_sizes.append(_scratch_bytes(count)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  reduced_blocks:list[tuple[RKArg, int, int]] = []
  for start in range(0, count, block_lanes):
    block_count = min(block_lanes, count-start)
    vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(block_count, len(rows))
    values, shifted = scratch(matrix_lanes), scratch(matrix_lanes)
    temps = tuple(scratch(matrix_lanes) for _ in range(11))
    gathers.extend(_stripe_gathers(source_slot, values, block_count,
                                   (row[start:start+block_count] for row in rows), vector_lanes))
    if mode == "greater":
      ops.append(RKEWOp(arg(shifted), arg(values), arg(threshold_slot), matrix_lanes, _EW_CFG[Ops.SUB]))
      mask = _ew_ieee_positive_mask(ops, arg, shifted, (one, maximum, negative_maximum), temps, matrix_lanes)
    else:
      magnitude, mask_slot = temps[:2]
      ops.extend((RKEWOp(arg(magnitude), arg(values), arg(values), matrix_lanes, _EW_CFG_ABS),
                  RKEWOp(arg(mask_slot), arg(magnitude), arg(magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True)))
      mask = arg(mask_slot)
    reduced = _reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(len(rows))],
                           block_count, _EW_CFG[Ops.ADD])
    reduced_blocks.append((reduced, start, block_count))
  compact = scratch(count)
  scratch_sizes.append(_int32_tiles_bytes(count)); int_tiles = len(scratch_sizes)-1
  gather_after = len(ops)
  mid = tuple(RKGather(reduced.index, compact, block_count,
                       offsets=tuple(reduced.addend//2+lane for lane in range(block_count)), dst_addend=start,
                       src_kind=RKBufferKind.SCRATCH) for reduced,start,block_count in reduced_blocks)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), arg(compact), arg(int_tiles), count,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  constants = struct.pack("<eeee", 1.0, 65504.0, -65504.0, threshold)
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), constants,
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid, gather_after=gather_after)

def _lower_unrolled_fp16_prefix_count(output:RKOutput) -> RKImage|None:
  """Lower an unrolled FP16 positive/nonzero predicate sum to exact INT32 lanes."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  terms = _flatten_binary(root, Ops.ADD)
  parsed:dict[UOp, tuple[str, UOp, UOp, float]] = {}
  for term in terms:
    nodes = term.toposort()
    matches = [("greater", u, match[0], match[1]) for u in nodes if (match:=_greater_half_load(u)) is not None]
    matches += [("nonzero", u, load, 0.0) for u in nodes if (load:=_nonzero_load(u)) is not None]
    loads = [u for u in nodes if u.op is Ops.LOAD]
    if len(matches) != 1 or loads != [matches[0][2]]: return None
    parsed[term] = matches[0]
  modes = {mode for mode,_,_,_ in parsed.values()}
  if len(modes) != 1: return None
  mode = modes.pop()
  thresholds = {_fp16_bits(threshold) for _,_,_,threshold in parsed.values()}
  if len(thresholds) != 1: return None
  threshold = next(iter(parsed.values()))[3]
  params = tuple(_root_param(load.src[0]) for _,_,load,_ in parsed.values())
  if (any(param is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST for param in params) or
      len({param.arg.slot for param in params if param is not None}) != 1): return None
  source = next(param for param in params if param is not None)
  source_count = int(source.src[0].arg)
  rows:list[tuple[int, ...]] = []
  try:
    valid_rows = _static_int_vectors(out_index,
      tuple(term.substitute({parsed[term][1]:parsed[term][1].const_like(True)}) for term in terms), count)
    for term,valid in zip(terms, valid_rows):
      _, _, load, _ = parsed[term]
      offsets = _gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count)
      if any(bit not in (0, 1) or bit and not 0 <= offset < source_count for bit,offset in zip(valid, offsets)): return None
      rows.append(tuple(offset if bit else -1 for bit,offset in zip(valid, offsets)))
  except RuntimeError: return None
  return _fp16_prefix_image(out_param.arg.slot, count, source.arg.slot, tuple(rows), mode, threshold)

def _lower_unrolled_bool_prefix_count(output:RKOutput) -> RKImage|None:
  """Widen opaque bool bytes and emit their exact native-INT16 prefix as INT32."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  terms = _flatten_binary(root, Ops.ADD)
  if len(terms) != count: return None
  def parse(term:UOp) -> tuple[UOp, UOp|None]|None:
    if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.int or term.src[0].dtype.scalar() is not dtypes.bool: return None
    load = term.src[0]
    return (load, None) if load.op is Ops.LOAD and load.src[0].op is Ops.INDEX else None
  if (prefix:=_prefix_load_rows(out_index, count, terms, dtypes.bool, parse)) is None: return None
  source, rows = prefix
  vector_bytes, _, matrix_lanes = _stripe_layout(count, len(rows))
  gathers = tuple(RKGather(source.arg.slot, 0, count, offsets=row, dst_addend=i*vector_bytes, dst_stride=2, itemsize=1)
                  for i,row in enumerate(rows))
  ops:list[RKEWOp] = []
  total = _reduce_rows(ops, [RKArg(RKBufferKind.SCRATCH, 0, i*vector_bytes) for i in range(len(rows))], count,
                       _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), total, total, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, (RKScratch(_scratch_bytes(matrix_lanes)),), gathers=gathers, ew_ops=tuple(ops))

def _lower_unrolled_int32_prefix_count(output:RKOutput) -> RKImage|None:
  """Emit an exact prefix count for arbitrary external INT32 nonzero predicates."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  terms = _flatten_binary(root, Ops.ADD)
  if len(terms) != count: return None
  parsed:dict[UOp, tuple[UOp, UOp]] = {}
  for term in terms:
    matches = [(u, load) for u in term.toposort() if (load:=_int32_nonzero_load(u)) is not None]
    loads = [u for u in term.toposort() if u.op is Ops.LOAD]
    if len(matches) != 1 or loads != [matches[0][1]]: return None
    parsed[term] = matches[0]
  source = _root_param(next(iter(parsed.values()))[1].src[0])
  if source is None or source.src[0].op is not Ops.CONST: return None
  source_count = int(source.src[0].arg)
  repeat = count//source_count if source_count and count%source_count == 0 else 1
  if not 1 <= repeat <= 8: return None
  def parse(term:UOp) -> tuple[UOp, UOp]|None:
    predicate, load = parsed[term]
    return load, term.substitute({predicate:predicate.const_like(True)})
  if (prefix:=_prefix_load_rows(out_index, count, terms, dtypes.int, parse, repeat)) is None: return None
  source, rows = prefix
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, len(rows))
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  if (mask:=_native_int32_nonzero_mask(ops, gathers, scratch, source.arg.slot, rows, count, vector_lanes)) is None: return None
  reduced = _reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(len(rows))],
                         count, _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), reduced, reduced, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

def _lower_unrolled_int_occurrence_count(output:RKOutput) -> RKImage|None:
  """Count each requested coordinate in a bounded generated INT32 prefix vector."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  terms = _flatten_binary(root, Ops.ADD)
  if not terms: return None
  source:UOp|None = None
  offset_rows:list[tuple[int, ...]] = []
  coordinate_rows:list[tuple[int, ...]] = []
  try:
    for term in terms:
      comparisons = [u for u in term.toposort() if u.op is Ops.CMPNE and
                     any(x.op is Ops.LOAD and x.dtype.scalar() is dtypes.int for x in u.src)]
      loads = [u for u in term.toposort() if u.op is Ops.LOAD]
      if len(comparisons) != 1 or len(loads) != 1: return None
      comparison, load = comparisons[0], loads[0]
      coordinates = [x for x in comparison.src if x is not load and _is_static_expr(x)]
      if load not in comparison.src or len(coordinates) != 1 or load.dtype.scalar() is not dtypes.int or load.src[0].op is not Ops.INDEX:
        return None
      equal_value = _static_int_vector(out_index, term.substitute({comparison:comparison.const_like(False)}), count)
      unequal_value = _static_int_vector(out_index, term.substitute({comparison:comparison.const_like(True)}), count)
      if equal_value != (1,)*count or unequal_value != (0,)*count: return None
      param = _root_param(load.src[0])
      if (param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST or
          source is not None and param.arg.slot != source.arg.slot): return None
      source = param
      offset_rows.append(_gather_offsets(out_index, load.src[0].src[1], None, count))
      coordinate_rows.append(_static_int_vector(out_index, coordinates[0], count))
  except RuntimeError: return None
  if source is None or int(source.src[0].arg) != len(terms): return None
  if len(terms) > _FP16_EXACT_INTEGER: return None
  if any(sorted(row[lane] for row in offset_rows) != list(range(len(terms))) for lane in range(count)): return None
  expected = tuple(range(count))
  if any(row != expected for row in coordinate_rows): return None
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, len(terms))
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  one, compact, convert_tiles, candidates, coordinate_matrix, diff, magnitude, unequal, equal, int_tiles = range(10)
  scratch = [RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(10)]
  scratch[compact] = RKScratch(_scratch_bytes(len(terms)))
  scratch[convert_tiles] = RKScratch(_int32_tiles_bytes(len(terms)))
  scratch[int_tiles] = RKScratch(_int32_tiles_bytes(count))
  coordinate_bits = tuple(tuple(_fp16_bits(value) for value in row) for row in coordinate_rows)
  gathers = _stripe_gathers(source.arg.slot, coordinate_matrix, count, coordinate_bits, vector_lanes, values=True)
  mid = tuple(replace(gather, src_kind=RKBufferKind.SCRATCH) for gather in
              _stripe_gathers(compact, candidates, count, offset_rows, vector_lanes))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(compact), RKArg(RKBufferKind.ARG, source.arg.slot), arg(convert_tiles), len(terms),
                              _EW_CFG[Ops.MAX], int32_input=True)]
  equal_arg = _ew_integer_eq_mask(ops, arg, candidates, coordinate_matrix, (diff, magnitude, unequal, equal), one, matrix_lanes)
  reduced = _reduce_rows(ops, [RKArg(equal_arg.kind, equal_arg.index, equal_arg.addend+row*vector_bytes)
                               for row in range(len(terms))], count, _EW_CFG[Ops.ADD])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), reduced, arg(int_tiles), count,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(scratch), struct.pack("<e", 1.0), gathers=gathers, ew_ops=tuple(ops),
                 mid_gathers=mid, gather_after=1)

def _int32_sum_occurrence_image(out_slot:int, count:int, coordinate_values:tuple[int, ...],
                                row_sources:tuple[tuple[tuple[int, int]|None, ...], ...]) -> RKImage|None:
  """Emit a blocked exact histogram for rows formed by sums of opaque INT32 values."""
  rows = len(row_sources[0]) if row_sources else 0
  source_slots = {item[0] for operand in row_sources for item in operand if item is not None}
  if (not rows or len(coordinate_values) != count or not 2 <= len(row_sources) <= 8 or
      any(len(operand) != rows for operand in row_sources) or not source_slots): return None
  constant_source = min(source_slots)
  vector_bytes, vector_lanes, _ = _stripe_layout(count, 1)
  block_rows = max(1, _MAX_EW_ELEMS_FP16//vector_lanes)
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  partials:list[RKArg] = []
  int16 = dict(int16_input=True, int16_output=True)
  for start in range(0, rows, block_rows):
    op_start = len(ops)
    block_count = min(block_rows, rows-start)
    matrix_lanes = block_count*vector_lanes
    operands:list[tuple[RKArg, ...]] = []
    for operand in row_sources:
      byte_args = []
      for byte in range(4):
        slot = scratch(matrix_lanes*2)
        block = operand[start:start+block_count]
        for row,item in enumerate(block):
          if item is not None:
            gathers.append(RKGather(item[0], slot, count, base=item[1]*4+byte, dst_stride=2,
                                    dst_addend=row*vector_lanes*2, itemsize=1))
        if not any(item is not None for item in block):
          gathers.append(RKGather(constant_source, slot, matrix_lanes, values=(0,)*matrix_lanes))
        byte_args.append(arg(slot))
      operands.append(tuple(byte_args))
    summed = _int16_byte_sum(ops, gathers, scratch, constant_source, tuple(operands), matrix_lanes)
    one = scratch(matrix_lanes*2)
    gathers.append(RKGather(constant_source, one, matrix_lanes, values=(1,)*matrix_lanes))
    masks:list[RKArg] = []
    for byte,value in enumerate(summed):
      expected, diff, magnitude, unequal, equal = (scratch(matrix_lanes*2) for _ in range(5))
      values = tuple((coordinate >> (byte*8)) & 0xff for _ in range(block_count)
                     for coordinate in (*coordinate_values, *((0,)*(vector_lanes-count))))
      gathers.append(RKGather(constant_source, expected, matrix_lanes, values=values))
      ops.extend((RKEWOp(arg(diff), value, arg(expected), matrix_lanes, _EW_CFG[Ops.SUB], **int16),
                  RKEWOp(arg(magnitude), arg(diff), arg(diff), matrix_lanes, _EW_CFG_ABS, **int16),
                  RKEWOp(arg(unequal), arg(magnitude), arg(one), matrix_lanes, _EW_CFG_MIN, **int16),
                  RKEWOp(arg(equal), arg(one), arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], **int16)))
      masks.append(arg(equal))
    mask = masks[0]
    for byte_mask in masks[1:]:
      slot = scratch(matrix_lanes*2); ops.append(RKEWOp(arg(slot), mask, byte_mask, matrix_lanes, _EW_CFG[Ops.MUL], **int16)); mask = arg(slot)
    partials.append(_reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(block_count)],
                                 count, _EW_CFG[Ops.ADD], int16=True))
    if start and len(ops) > op_start: ops[op_start] = replace(ops[op_start], submit_barrier=True)
  result = _reduce_rows(ops, partials, count, _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, result, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

def _lower_loop_int32_occurrence_count(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Count exact arbitrary INT32 coordinate sums using native INT16 byte arithmetic."""
  _, out_param, count, out_index, _ = output
  out_ranges = _index_ranges(out_index)
  if not 1 <= count <= _FP16_EXACT_INTEGER or len(out_ranges) != 1 or (loop:=_local_add_loop(uops, out_index)) is None: return None
  reduce_range, rows, term, _ = loop
  if term.op is not Ops.WHERE or term.src[0].op is not Ops.CMPNE: return None
  if (term.src[1].op is not Ops.CONST or int(term.src[1].arg) != 0 or
      term.src[2].op is not Ops.CONST or int(term.src[2].arg) != 1): return None
  comparison = term.src[0]
  candidates = [x for x in comparison.src if reduce_range in x.toposort()]
  coordinates = [x for x in comparison.src if reduce_range not in x.toposort()]
  if len(candidates) != 1 or len(coordinates) != 1: return None
  try: coordinate_values = _static_int_vector(out_index, coordinates[0], count)
  except RuntimeError: return None
  if coordinate_values != tuple(range(count)): return None
  addends = _flatten_binary(candidates[0], Ops.ADD)
  if not 1 <= len(addends) <= 8: return None

  def source(x:UOp, lane:int) -> tuple[int, int]|None:
    if x.op is Ops.CONST:
      if int(x.arg) != 0: raise RuntimeError
      return None
    if x.op is Ops.WHERE:
      return source(x.src[1] if _eval_int(x.src[0], {reduce_range:lane}) else x.src[2], lane)
    if x.op is not Ops.LOAD or x.dtype.scalar() is not dtypes.int or not x.src or x.src[0].op is not Ops.INDEX: raise RuntimeError
    if len(x.src) == 3 and not _eval_int(x.src[2], {reduce_range:lane}): return source(x.src[1], lane)
    param = _root_param(x.src[0])
    if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: raise RuntimeError
    offset = _eval_int(x.src[0].src[1], {reduce_range:lane})
    if not 0 <= offset < int(param.src[0].arg): raise RuntimeError
    return param.arg.slot, offset

  try:
    operands = tuple(tuple(source(addend, lane) for lane in range(rows)) for addend in addends)
  except RuntimeError: return None
  row_sources = operands + tuple((None,)*rows for _ in range(max(0, 2-len(operands))))
  return _int32_sum_occurrence_image(out_param.arg.slot, count, coordinate_values, row_sources)

def _lower_unrolled_int32_sum_occurrence(output:RKOutput) -> RKImage|None:
  """Recognize devectorized exact INT32 sum-equality histogram rows."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  terms = _flatten_binary(root, Ops.ADD)
  if not 2 <= len(terms) <= 32767: return None

  def source(x:UOp) -> tuple[int, int]|None:
    if x.op is Ops.CONST:
      if int(x.arg) != 0: raise RuntimeError
      return None
    if x.op is Ops.WHERE:
      condition = _static_int_vector(out_index, x.src[0], count)
      if len(set(condition)) != 1: raise RuntimeError
      return source(x.src[1] if condition[0] else x.src[2])
    if x.op is not Ops.LOAD or x.dtype.scalar() is not dtypes.int or not x.src or x.src[0].op is not Ops.INDEX: raise RuntimeError
    if len(x.src) == 3:
      gate = _static_int_vector(out_index, x.src[2], count)
      if len(set(gate)) != 1: raise RuntimeError
      if not gate[0]: return source(x.src[1])
    param = _root_param(x.src[0])
    if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: raise RuntimeError
    offsets = _gather_offsets(out_index, x.src[0].src[1], None, count)
    if len(set(offsets)) != 1 or not 0 <= offsets[0] < int(param.src[0].arg): raise RuntimeError
    return param.arg.slot, offsets[0]

  rows:list[tuple[tuple[int, int]|None, ...]] = []
  coordinates:list[tuple[int, ...]] = []
  try:
    for term in terms:
      if term.op is not Ops.WHERE or term.src[0].op is not Ops.CMPNE: return None
      comparison = term.src[0]
      if (_static_int_vector(out_index, term.substitute({comparison:comparison.const_like(False)}), count) != (1,)*count or
          _static_int_vector(out_index, term.substitute({comparison:comparison.const_like(True)}), count) != (0,)*count): return None
      candidates = [x for x in comparison.src if any(u.op is Ops.LOAD for u in x.toposort())]
      coordinate = [x for x in comparison.src if not any(u.op is Ops.LOAD for u in x.toposort())]
      if len(candidates) != 1 or len(coordinate) != 1: return None
      addends = _flatten_binary(candidates[0], Ops.ADD)
      if not 1 <= len(addends) <= 8: return None
      rows.append(tuple(source(addend) for addend in addends))
      coordinates.append(_static_int_vector(out_index, coordinate[0], count))
  except RuntimeError: return None
  if any(values != tuple(range(count)) for values in coordinates): return None
  width = max(2, max(map(len, rows)))
  row_sources = tuple(tuple(row[operand] if operand < len(row) else None for row in rows) for operand in range(width))
  return _int32_sum_occurrence_image(out_param.arg.slot, count, coordinates[0], row_sources)

def _int32_index_selection_image(out_slot:int, count:int, index_slot:int, index_offsets:tuple[int, ...],
                                 candidate_values:tuple[tuple[int, ...], ...]) -> RKImage|None:
  """Select per-lane bounded INT16 values by exact external INT32 index equality."""
  rows = len(candidate_values)
  if not rows or any(len(values) != count or any(not -32768 <= value <= 32767 for value in values) for values in candidate_values): return None
  vector_bytes, vector_lanes, _ = _stripe_layout(count, 1)
  block_rows = max(1, _MAX_EW_ELEMS_FP16//vector_lanes)
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  partials:list[RKArg] = []
  for start in range(0, rows, block_rows):
    op_start = len(ops)
    block_count = min(block_rows, rows-start)
    matrix_lanes = block_count*vector_lanes
    coordinates = tuple((candidate,)*count for candidate in range(start, start+block_count))
    if (mask:=_native_int16_byte_mask(ops, gathers, scratch, index_slot, index_offsets,
                                     (coordinates,), count, vector_lanes)) is None: return None
    weight_slot, selected = scratch(matrix_lanes*2), scratch(matrix_lanes*2)
    values = tuple(value for row in candidate_values[start:start+block_count]
                   for value in (*row, *((0,)*(vector_lanes-count))))
    gathers.append(RKGather(index_slot, weight_slot, matrix_lanes, values=values))
    ops.append(RKEWOp(arg(selected), mask, arg(weight_slot), matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
    partials.append(_reduce_rows(ops, [arg(selected, row*vector_bytes) for row in range(block_count)],
                                 count, _EW_CFG[Ops.ADD], int16=True))
    if start and len(ops) > op_start: ops[op_start] = replace(ops[op_start], submit_barrier=True)
  result = _reduce_rows(ops, partials, count, _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, result, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

def _lower_bounded_int32_lookup(output:RKOutput) -> RKImage|None:
  """Evaluate a bounded static integer function selected by one arbitrary external INT32 index."""
  _, out_param, count, out_index, root = output
  loads = tuple({u.key:u for u in root.toposort() if u.op is Ops.LOAD}.values())
  if (not 1 <= count <= _FP16_EXACT_INTEGER or len(loads) != 1 or loads[0].dtype.scalar() is not dtypes.int or
      root.op is not Ops.WHERE or root.src[2].op is not Ops.CONST or int(root.src[2].arg) != 0): return None
  load = loads[0]
  if not load.src or load.src[0].op is not Ops.INDEX: return None
  param = _root_param(load.src[0])
  if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: return None
  gates = _flatten_binary(root.src[0], Ops.AND)
  limits = {int(u.src[1].arg) for u in gates if u.op is Ops.CMPLT and u.src[0].key == load.key and
            u.src[1].op is Ops.CONST and 0 < int(u.src[1].arg) <= 32767}
  nonnegative = [u for u in gates if u.op is Ops.CMPNE and any(x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool and bool(x.arg)
                 for x in u.src) and any(x.op is Ops.CMPLT and x.src[0].key == load.key and x.src[1].op is Ops.CONST and
                 int(x.src[1].arg) == 0 for x in u.src)]
  if len(gates) != 2 or len(limits) != 1 or len(nonnegative) != 1: return None
  limit = next(iter(limits))
  try:
    index_offsets = _gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count)
    candidate_values = _static_int_vectors(out_index,
      tuple(root.substitute({load:load.const_like(candidate)}) for candidate in range(limit)), count)
  except RuntimeError: return None
  if any(not 0 <= offset < int(param.src[0].arg) for offset in index_offsets): return None
  return _int32_index_selection_image(out_param.arg.slot, count, param.arg.slot, index_offsets, candidate_values)

def _lower_unrolled_int_prefix_sum(output:RKOutput) -> RKImage|None:
  """Lower the bounded histogram prefix emitted by fixed masked-select."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  normalized:tuple[UOp, int]|None = None
  if root.op is Ops.WHERE and len(root.src) == 3 and root.src[0].op is Ops.CMPLT:
    value, limit = root.src[0].src
    extents = [x for x in root.src[1].src if root.src[1].op is Ops.ADD and x.op is Ops.CONST and int(x.arg) > 0]
    values = [x for x in root.src[1].src if root.src[1].op is Ops.ADD and x.key == value.key]
    if (limit.op is not Ops.CONST or int(limit.arg) != 0 or root.src[2].key != value.key or
        len(extents) != 1 or len(values) != 1): return None
    normalized = value, int(extents[0].arg)
  value = normalized[0] if normalized is not None else root
  terms = _flatten_binary(value, Ops.ADD)
  if len(terms) != count: return None
  def parse(term:UOp) -> tuple[UOp, UOp|None]|None:
    loads = [u for u in term.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int]
    if len(loads) != 1 or term.key != loads[0].key: return None
    return loads[0], loads[0].src[2] if len(loads[0].src) == 3 else None
  if (prefix:=_prefix_load_rows(out_index, count, terms, dtypes.int, parse)) is None: return None
  source, rows = prefix
  if int(source.src[0].arg) != count: return None
  if normalized is not None and not 1 <= normalized[1] <= _FP16_EXACT_INTEGER: return None

  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  compact, convert_tiles, matrix, int_tiles = range(4)
  scratch = [RKScratch(_scratch_bytes(count)), RKScratch(_int32_tiles_bytes(count)),
             RKScratch(_scratch_bytes(matrix_lanes)), RKScratch(_int32_tiles_bytes(count))]
  gathers:list[RKGather] = []
  mid = _stripe_gathers(compact, matrix, count, rows, vector_lanes)
  mid = tuple(replace(gather, src_kind=RKBufferKind.SCRATCH) for gather in mid)
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(compact), RKArg(RKBufferKind.ARG, source.arg.slot), arg(convert_tiles), count,
                              _EW_CFG[Ops.MAX], int32_input=True)]
  reduced = _reduce_rows(ops, [arg(matrix, row*vector_bytes) for row in range(count)], count, _EW_CFG[Ops.ADD])
  result = reduced
  if normalized is not None:
    zero, extent, negative_delta, negative, correction, normalized_value = range(len(scratch), len(scratch)+6)
    scratch.extend(RKScratch(_scratch_bytes(count)) for _ in range(6))
    zero_bits, extent_bits = (_fp16_bits(value) for value in (0.0, normalized[1]))
    gathers.extend((RKGather(source.arg.slot, zero, count, values=(zero_bits,)*count),
                    RKGather(source.arg.slot, extent, count, values=(extent_bits,)*count)))
    ops.extend((RKEWOp(arg(negative_delta), arg(zero), reduced, count, _EW_CFG[Ops.SUB]),
                RKEWOp(arg(negative), arg(negative_delta), arg(negative_delta), count, _EW_CFG[Ops.MAX], compare=True),
                RKEWOp(arg(correction), arg(negative), arg(extent), count, _EW_CFG[Ops.MUL], stateful=True),
                RKEWOp(arg(normalized_value), reduced, arg(correction), count, _EW_CFG[Ops.ADD])))
    result = arg(normalized_value)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), result, arg(int_tiles), count,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(scratch), gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid, gather_after=1)

def _lower_loop_int_prefix_sum(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Normalize Tinygrad's local-register INT32 prefix loop into the proven unrolled prefix emitter."""
  store, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  out_ranges = _index_ranges(out_index)
  if len(out_ranges) != 1 or (loop:=_local_add_loop(uops, out_index)) is None: return None
  reduce_range, groups, term, local_stores = loop
  if groups != count or not any(s.src[1].op is Ops.CONST and int(s.src[1].arg) == 0 for s in local_stores): return None
  loads = [u for u in term.toposort() if u.op is Ops.LOAD and _root_param(u.src[0]) is not None]
  if len(loads) != 1: return None
  source = _root_param(loads[0].src[0])
  if source is None or source.dtype.scalar() is not dtypes.int or source.src[0].op is not Ops.CONST or int(source.src[0].arg) != count:
    return None
  terms = [term.substitute({reduce_range:reduce_range.const_like(lane)}) for lane in range(count)]
  value = terms[0]
  for term in terms[1:]: value = value.alu(Ops.ADD, term)
  result_loads = [u for u in root.toposort() if _local_load(u) is not None]
  def local_buffer(load:UOp) -> UOp:
    buf = load.src[0].src[0]
    while buf.op is Ops.AFTER: buf = buf.src[0]
    return buf
  if (not result_loads or len({local_buffer(u).key for u in result_loads}) != 1 or
      any(local_buffer(u).op is not Ops.BUFFER for u in result_loads)): return None
  result = root.substitute({u:value for u in result_loads})
  return _lower_unrolled_int_prefix_sum((store, out_param, count, out_index, result))

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
  if (rows:=_loaded_equality_rows(out_index, count, tuple(pair for pair,_ in parsed), dtypes.half, same_source=True)) is None: return None
  candidate_src, current_src, candidate_offsets, current_offsets = rows
  try:
    valid_bits = tuple(_static_vector(out_index, gate, count) if gate is not None else (0x3c00,)*count for _,gate in parsed)
  except RuntimeError: return None
  for dst in range(count):
    valid_offsets = [offsets[dst] for offsets,bits in zip(candidate_offsets, valid_bits) if bits[dst] == 0x3c00]
    if not valid_offsets or len(valid_offsets) != len(set(valid_offsets)) or any(bits[dst] not in (0, 0x3c00) for bits in valid_bits): return None

  window = len(candidate_offsets)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one, candidates_slot, current_slot, valid_slot, diff, magnitude, unequal, equal, selected, int_tiles = range(11)
  gathers = _stripe_gathers(candidate_src, candidates_slot, count, candidate_offsets, vector_lanes)
  gathers += _stripe_gathers(current_src, current_slot, count, (current_offsets,)*window, vector_lanes)
  gathers += _stripe_gathers(candidate_src, valid_slot, count, valid_bits, vector_lanes, values=True)
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(_int32_tiles_bytes(count)))
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

  half_rows = _loaded_equality_rows(out_index, count, tuple(pair for _,pair,_ in parsed), dtypes.half)
  int_rows = _loaded_equality_rows(out_index, count, tuple(pair for _,_,pair in parsed), dtypes.int)
  if half_rows is None or int_rows is None: return None
  half_candidate_src, half_current_src, half_candidate_offsets, half_current_offsets = half_rows
  int_candidate_src, int_current_src, int_candidate_offsets, int_current_offsets = int_rows
  if half_candidate_offsets != int_candidate_offsets: return None

  rows = len(parsed)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, rows)
  (zero, one, raw_candidate_count, raw_current_count, candidate_count, current_count, convert_tiles,
   candidate_value, current_value, weights, value_diff, value_magnitude, value_unequal, value_equal,
   count_diff, count_magnitude, count_unequal, count_equal, selected, weighted, int_tiles) = range(21)
  weight_bits = tuple((_fp16_bits(weight),)*count for weight,_,_ in parsed)
  gathers = _stripe_gathers(half_candidate_src, candidate_value, count, half_candidate_offsets, vector_lanes)
  gathers += _stripe_gathers(int_candidate_src, raw_candidate_count, count, int_candidate_offsets, vector_lanes, itemsize=4)
  gathers += _stripe_gathers(half_candidate_src, weights, count, weight_bits, vector_lanes, values=True)
  gathers += _stripe_gathers(half_current_src, current_value, count, (half_current_offsets,)*rows, vector_lanes)
  gathers += _stripe_gathers(int_current_src, raw_current_count, count, (int_current_offsets,)*rows, vector_lanes, itemsize=4)
  scratch_sizes = [matrix_lanes*2]*21
  scratch_sizes[raw_candidate_count] = scratch_sizes[raw_current_count] = matrix_lanes*4
  scratch_sizes[convert_tiles] = _int32_tiles_bytes(matrix_lanes)
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)
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
                            candidate_coords:tuple[tuple[int, ...], ...]|None=None, index_limit:int|None=None,
                            raw_weight:bool=False, native_int16:bool=False) -> RKImage:
  """Emit a matrix equality/select reduction shared by unrolled and loop cumulative indices."""
  window = len(candidate_plans)
  limit = window if index_limit is None else index_limit
  if native_int16: constants:tuple[int|float, ...] = (0, 1, limit) if first_tie else (0, 1)
  else: constants = (0.0, 1.0, float(limit)) if first_tie else (0.0, 1.0)
  complement = len(constants)
  if native_int16 and (negated_candidates or negate_extrema): constants += (-1,)
  zero, one, candidate_arena = 0, 1, len(constants)
  fused_extrema = native_int16 and extrema_plans is candidate_plans and window*window+2*window > _RKIMAGE_U16_MAX
  extrema_arena = candidate_arena+1
  extrema_slots = (candidate_arena+2,) if fused_extrema else tuple(range(candidate_arena+1, candidate_arena+1+len(extrema_plans)))
  first_temp = candidate_arena+3 if fused_extrema else candidate_arena+1+len(extrema_plans)
  coordinate_arena, selected_arena, diff, magnitude, unequal, equal, int_tiles = range(first_temp, first_temp+7)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  def materialize(plans:tuple[RKGather, ...], scratch_slot:int) -> tuple[RKGather, ...]:
    return tuple(RKGather(plan.src_index, scratch_slot, count, plan.base, plan.axes, plan.offsets, plan.fill_bits,
                          values=plan.values, partial=plan.partial, dst_stride=plan.dst_stride,
                          dst_addend=i*vector_lanes, itemsize=plan.itemsize) for i,plan in enumerate(plans))
  gathers = materialize(candidate_plans, candidate_arena)
  if fused_extrema: gathers += materialize(candidate_plans, extrema_arena)
  else:
    for plan,scratch_slot in zip(extrema_plans, extrema_slots): gathers += materialize((plan,)*window, scratch_slot)
  encode = _int16_bits if native_int16 else _fp16_bits
  coordinate_bits = tuple(tuple(encode(limit-(candidate_coords[candidate][dst]
                                if candidate_coords is not None else candidate) if first_tie else
                                candidate+1 if candidate <= axis_coords[dst] else 0)
                                for dst in range(count)) for candidate in range(window))
  gathers += tuple(RKGather(candidate_plans[0].src_index, coordinate_arena, count, values=bits, dst_addend=candidate*vector_lanes)
                   for candidate,bits in enumerate(coordinate_bits))
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(_int32_tiles_bytes(count)))
  def args(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ew_ops:list[RKEWOp] = []
  integer = dict(int16_input=True, int16_output=True) if native_int16 else {}
  mid_gathers:tuple[RKGather, ...] = ()
  gather_after = 0
  if fused_extrema:
    _reduce_rows(ew_ops, [args(extrema_arena, candidate*vector_bytes) for candidate in range(window)], count,
                 _EW_CFG_MIN if negated_candidates else _EW_CFG[Ops.MAX], int16=True)
    gather_after = len(ew_ops)
    offsets = tuple(range(count))
    mid_gathers = tuple(RKGather(extrema_arena, extrema_slots[0], count, offsets=offsets,
                                 dst_addend=candidate*vector_lanes, src_kind=RKBufferKind.SCRATCH) for candidate in range(window))
  if negate_extrema:
    for slot in extrema_slots:
      ew_ops.append(RKEWOp(args(slot), args(complement if native_int16 else zero), args(slot), matrix_lanes, _EW_CFG[Ops.SUB], **integer))
  extrema = args(extrema_slots[0])
  for i,slot in enumerate(extrema_slots[1:]):
    ew_ops.append(RKEWOp(extrema, extrema, args(slot), matrix_lanes, _EW_CFG[Ops.MAX],
                         submit_barrier=not native_int16 and negate_extrema and i == 0,
                         stateful=not native_int16 and negate_extrema and i == 0, **integer))
  if negated_candidates:
    ew_ops.append(RKEWOp(args(diff), args(complement if native_int16 else zero), args(candidate_arena), matrix_lanes, _EW_CFG[Ops.SUB],
                          submit_barrier=not native_int16 and bool(ew_ops), stateful=not native_int16 and bool(ew_ops), **integer))
  if native_int16:
    ew_ops.extend((RKEWOp(args(diff), args(diff if negated_candidates else candidate_arena), args(extrema_slots[0]), matrix_lanes,
                          _EW_CFG[Ops.SUB], **integer),
                   RKEWOp(args(magnitude), args(diff), args(diff), matrix_lanes, _EW_CFG_ABS, **integer),
                   RKEWOp(args(unequal), args(magnitude), args(one), matrix_lanes, _EW_CFG_MIN, **integer),
                   RKEWOp(args(equal), args(one), args(unequal), matrix_lanes, _EW_CFG[Ops.SUB], **integer)))
    equal_arg = args(equal)
  else:
    equal_arg = _ew_eq_mask(ew_ops, args, diff if negated_candidates else candidate_arena, extrema_slots[0],
                            (magnitude, magnitude, unequal, equal), one, matrix_lanes, (bool(ew_ops), True))
  ew_ops.extend((RKEWOp(args(diff), equal_arg, args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=not native_int16, stateful=not native_int16, **integer),
                 RKEWOp(args(selected_arena), equal_arg, args(coordinate_arena), matrix_lanes, _EW_CFG[Ops.MUL],
                        submit_barrier=not native_int16, stateful=not native_int16, **integer)))
  selected = _reduce_rows(ew_ops, [args(selected_arena, candidate*vector_bytes) for candidate in range(window)], count,
                          _EW_CFG[Ops.MAX], int16=native_int16)
  if first_tie:
    ew_ops.append(RKEWOp(args(diff), args(zero if raw_weight else 2), selected, count, _EW_CFG[Ops.SUB], **integer))
  else: ew_ops.append(RKEWOp(args(diff), selected, args(one), count, _EW_CFG[Ops.SUB], **integer))
  ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), args(diff), args(int_tiles), count,
                       _EW_CFG[Ops.MAX], stateful=True, int32_output=True, int16_input=native_int16))
  constant_data = struct.pack("<"+("h" if native_int16 else "e")*len(constants), *constants)
  return RKImage(RKTarget.RK3588, scratch, constant_data, gathers=gathers, ew_ops=tuple(ew_ops),
                 mid_gathers=mid_gathers, gather_after=gather_after)

def _half_candidate(root:UOp) -> tuple[UOp, bool]|None:
  if root.op is Ops.LOAD: return (root, False)
  if root.op is Ops.MUL and len(root.src) == 2:
    loads = [x for x in root.src if x.op is Ops.LOAD]
    constants = [x for x in root.src if x.op is Ops.CONST and float(x.arg) == -1.0]
    if len(loads) == len(constants) == 1: return (loads[0], True)
  return None

def _int16_candidate(root:UOp) -> tuple[UOp, bool]|None:
  if root.op is Ops.LOAD: return (root, False)
  candidate = _int16_nonconst(root, -1) if root.op is Ops.XOR else None
  return (candidate, True) if candidate is not None and candidate.op is Ops.LOAD else None

def _param_load_groups(nodes:Iterable[UOp], dtype:DType=dtypes.half) -> dict[int, list[UOp]]:
  """Group unique typed parameter loads by argument slot."""
  grouped:dict[int, list[UOp]] = {}
  for load in nodes:
    if load.op is Ops.LOAD and load.dtype.scalar() is dtype and load.src[0].op is Ops.INDEX and load.src[0].src[0].op is Ops.PARAM:
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

def _int16_leaf(u:UOp, out_index:UOp, count:int, out_slot:int, static_cache:dict[UOp, bool]|None=None) -> RKInt16Leaf:
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.int16: return int(u.arg)
  if u.dtype.scalar() is dtypes.int16 and _is_static_expr(u, static_cache): return RKStatic(u)
  if u.dtype.scalar() is dtypes.int16 and \
     (selection:=_selection_gather(u, out_index, count, out_slot, static_cache=static_cache, dtype=dtypes.int16)) is not None: return selection
  if u.op is not Ops.LOAD or u.dtype.scalar() is not dtypes.int16 or u.src[0].op is not Ops.INDEX or \
     (param:=_root_param(u.src[0])) is None or param.arg.slot == out_slot or param.src[0].op is not Ops.CONST: return None
  index, gate, fill = u.src[0].src[1], u.src[2] if len(u.src) > 2 else None, u.src[1] if len(u.src) > 1 else None
  if (fill is not None and fill.op is not Ops.CONST) or any(x.op is Ops.LOAD for x in index.toposort()) or \
     gate is not None and any(x.op is Ops.LOAD for x in gate.toposort()):
    return None
  return RKArg(RKBufferKind.ARG, param.arg.slot) if gate is None and index.key == out_index.key and int(param.src[0].arg) == count else \
    (param, index, gate, _int16_bits(0 if fill is None else fill.arg))

def _typed_gather_plan(value:UOp, out_index:UOp, count:int, out_slot:int, dtype:DType=dtypes.half) -> RKGather|None:
  """Turn one FP16 or INT16 lane expression into the raw gather used by specialized reductions."""
  if dtype is dtypes.half: leaf:RKLeaf|RKInt16Leaf = _ew_leaf(value, out_index, count, out_slot)
  elif dtype is dtypes.int16: leaf = _int16_leaf(value, out_index, count, out_slot)
  else: return None
  if isinstance(leaf, RKArg):
    if leaf.kind is not RKBufferKind.ARG or leaf.addend: return None
    return RKGather(leaf.index, 0, count, axes=((1, count, 1),) if count > 1 else ())
  if isinstance(leaf, RKGather): plan = leaf
  elif isinstance(leaf, RKMultiGather):
    if len(leaf.gathers) != 1: return None
    plan = leaf.gathers[0]
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

def _weighted_equality(term:UOp, dtype:DType=dtypes.half) -> tuple[tuple[UOp, UOp], UOp]|None:
  if term.op is not Ops.MUL or len(term.src) != 2: return None
  casts = [x for x in term.src if x.op is Ops.CAST and x.dtype.scalar() is dtypes.int and len(x.src) == 1]
  if len(casts) != 1 or (pair:=_equality_pair(casts[0].src[0])) is None or any(x.dtype.scalar() is not dtype for x in pair): return None
  weight = term.src[1] if term.src[0] is casts[0] else term.src[0]
  return pair, weight

def _wide_pool_index_image(out_slot:int, count:int, spatial_size:int, plans:tuple[RKGather, ...],
                           extrema_plan:RKGather, coordinates:tuple[tuple[int, ...], ...]) -> RKImage|None:
  """Select exact wide spatial indices as DPU-computed nibbles, then place their raw INT32 bytes."""
  window = len(plans)
  digit_mask = _POOL_INDEX_DIGIT_RADIX-1
  if (spatial_size > 1 << 16 or window*_POOL_INDEX_DIGIT_RADIX+digit_mask > _FP16_EXACT_INTEGER or
      len(coordinates) != window): return None

  priorities = [[0]*count for _ in range(window)]
  for lane in range(count):
    ordered = sorted((row[lane], candidate) for candidate,row in enumerate(coordinates) if row[lane] < spatial_size)
    if not ordered or len({coordinate for coordinate,_ in ordered}) != len(ordered): return None
    for rank,(_,candidate) in enumerate(ordered): priorities[candidate][lane] = len(ordered)-rank
  digits = max(1, ((spatial_size-1).bit_length()+_POOL_INDEX_DIGIT_BITS-1)//_POOL_INDEX_DIGIT_BITS)
  encoded = tuple(tuple(tuple(priority*_POOL_INDEX_DIGIT_RADIX + ((coordinate >> (digit*_POOL_INDEX_DIGIT_BITS)) & digit_mask)
                              if coordinate < spatial_size else 0 for coordinate,priority in zip(row,priority_row))
                              for row,priority_row in zip(coordinates, priorities)) for digit in range(digits))
  def half_bits(rows:Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(_fp16_bits(value) for value in row) for row in rows)

  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one, radix, candidates_slot, extrema_slot, priority_slot = range(6)
  encoded_slots = tuple(range(6, 6+digits)); next_slot = 6+digits
  diff, magnitude, unequal, equal, selected, priority_vector, scaled_priority = range(next_slot, next_slot+7); next_slot += 7
  digit_slots = tuple(range(next_slot, next_slot+digits)); next_slot += digits
  high_scaled = next_slot; next_slot += 1
  byte_slots = tuple(range(next_slot, next_slot+(digits+1)//2)); next_slot += len(byte_slots)
  int_tiles = next_slot; next_slot += 1
  int_slots = tuple(range(next_slot, next_slot+len(byte_slots))); next_slot += len(int_slots)

  gathers = tuple(replace(plan, dst_index=candidates_slot, dst_addend=candidate*vector_lanes)
                  for candidate,plan in enumerate(plans))
  gathers += tuple(replace(extrema_plan, dst_index=extrema_slot, dst_addend=candidate*vector_lanes)
                   for candidate in range(window))
  gathers += _stripe_gathers(plans[0].src_index, priority_slot, count, half_bits(priorities), vector_lanes, values=True)
  for slot,rows in zip(encoded_slots, encoded):
    gathers += _stripe_gathers(plans[0].src_index, slot, count, half_bits(rows), vector_lanes, values=True)

  scratch_sizes = [_scratch_bytes(matrix_lanes)] * next_slot
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)
  for slot in int_slots: scratch_sizes[slot] = max(count*4, 64)
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = []
  equal_arg = _ew_eq_mask(ops, arg, candidates_slot, extrema_slot, (diff, magnitude, unequal, equal), one, matrix_lanes)
  ops.append(RKEWOp(arg(selected), equal_arg, arg(priority_slot), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  priority = _reduce_rows(ops, [arg(selected, candidate*vector_bytes) for candidate in range(window)], count, _EW_CFG[Ops.MAX])
  ops.extend((RKEWOp(arg(priority_vector), priority, arg(zero), count, _EW_CFG[Ops.ADD]),
              RKEWOp(arg(scaled_priority), arg(priority_vector), arg(radix), count, _EW_CFG[Ops.MUL])))
  for encoded_slot,digit_slot in zip(encoded_slots, digit_slots):
    ops.append(RKEWOp(arg(selected), equal_arg, arg(encoded_slot), matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
    encoded_digit = _reduce_rows(ops, [arg(selected, candidate*vector_bytes) for candidate in range(window)], count, _EW_CFG[Ops.MAX])
    ops.append(RKEWOp(arg(digit_slot), encoded_digit, arg(scaled_priority), count, _EW_CFG[Ops.SUB]))
  for byte,byte_slot in enumerate(byte_slots):
    low, high = digit_slots[byte*2], digit_slots[byte*2+1] if byte*2+1 < digits else None
    if high is not None:
      ops.extend((RKEWOp(arg(high_scaled), arg(high), arg(radix), count, _EW_CFG[Ops.MUL]),
                  RKEWOp(arg(byte_slot), arg(low), arg(high_scaled), count, _EW_CFG[Ops.ADD])))
    else: ops.append(RKEWOp(arg(byte_slot), arg(low), arg(zero), count, _EW_CFG[Ops.ADD]))
  ops.extend(RKEWOp(arg(dst), arg(src), arg(int_tiles), count, _EW_CFG[Ops.MAX], int32_output=True)
             for src,dst in zip(byte_slots, int_slots))

  post_gathers:list[RKGather] = [RKGather(0, out_slot, count*4, values=(0,)*(count*4), dst_kind=RKBufferKind.ARG, itemsize=1)]
  byte_offsets = tuple(range(0, count*4, 4))
  post_gathers.extend(RKGather(slot, out_slot, count, offsets=byte_offsets, partial=True, dst_stride=4, dst_addend=byte,
                               dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
                      for byte,slot in enumerate(int_slots))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<eee", 0.0, 1.0,
                 float(_POOL_INDEX_DIGIT_RADIX)), gathers=gathers, ew_ops=tuple(ops), post_gathers=tuple(post_gathers))

def _raw_weight_pool_index_image(out_slot:int, count:int, plans:tuple[RKGather, ...], extrema_plan:RKGather,
                                 weights:tuple[tuple[int, ...], ...]) -> RKImage:
  """Select a negative pool weight through exact FP16 representation-byte equality."""
  window = len(plans)
  compact_lanes = window*count
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one = 0, 1
  raw_candidates = (2, 3); raw_extrema = (4, 5); convert_tiles = 6
  compact_candidates = (7, 8); compact_extrema = (9, 10)
  candidates = (11, 12); extrema = (13, 14); weight_slot = 15
  diff, magnitude, unequal, equal, combined, selected, negative, int_tiles = range(16, 24)
  scratch_sizes = [_scratch_bytes(matrix_lanes)]*24
  for slot in raw_candidates: scratch_sizes[slot] = compact_lanes*4
  for slot in raw_extrema: scratch_sizes[slot] = count*4
  for slot in compact_candidates: scratch_sizes[slot] = _scratch_bytes(compact_lanes)
  for slot in compact_extrema: scratch_sizes[slot] = _scratch_bytes(count)
  scratch_sizes[convert_tiles] = _int32_tiles_bytes(max(compact_lanes, count))
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)

  gathers:list[RKGather] = []
  for byte,(raw_candidate,raw_extreme) in enumerate(zip(raw_candidates, raw_extrema)):
    for row,plan in enumerate(plans):
      gathers.append(RKGather(plan.src_index, raw_candidate, count,
        offsets=tuple(offset*2+byte for offset in _plan_offsets(plan)), dst_stride=4,
        dst_addend=row*count*4, itemsize=1))
    gathers.append(RKGather(extrema_plan.src_index, raw_extreme, count,
      offsets=tuple(offset*2+byte for offset in _plan_offsets(extrema_plan)), dst_stride=4, itemsize=1))
  weight_bits = tuple(tuple(_fp16_bits(weight) for weight in row) for row in weights)
  gathers.extend(_stripe_gathers(plans[0].src_index, weight_slot, count, weight_bits, vector_lanes, values=True))

  mid_gathers:list[RKGather] = []
  for compact_candidate,compact_extreme,candidate,extreme in zip(compact_candidates, compact_extrema, candidates, extrema):
    for row in range(window):
      mid_gathers.extend((RKGather(compact_candidate, candidate, count, offsets=tuple(row*count+i for i in range(count)),
                                   dst_addend=row*vector_lanes, src_kind=RKBufferKind.SCRATCH),
                          RKGather(compact_extreme, extreme, count, offsets=tuple(range(count)),
                                   dst_addend=row*vector_lanes, src_kind=RKBufferKind.SCRATCH)))

  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(dst), arg(src), arg(convert_tiles), lanes, _EW_CFG[Ops.MAX], int32_input=True)
                       for src,dst,lanes in zip((*raw_candidates, *raw_extrema), (*compact_candidates, *compact_extrema),
                                                (compact_lanes, compact_lanes, count, count))]
  for byte,(candidate,extreme) in enumerate(zip(candidates, extrema)):
    byte_equal = _ew_integer_eq_mask(ops, arg, candidate, extreme, (diff, magnitude, unequal, equal), one, matrix_lanes)
    ops.append(RKEWOp(arg(combined), arg(combined) if byte else byte_equal, byte_equal if byte else arg(one), matrix_lanes,
                      _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  ops.append(RKEWOp(arg(selected), arg(combined), arg(weight_slot), matrix_lanes,
                    _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  selected_arg = _reduce_rows(ops, [arg(selected, row*vector_bytes) for row in range(window)], count, _EW_CFG[Ops.MAX])
  ops.extend((RKEWOp(arg(negative), arg(zero), selected_arg, count, _EW_CFG[Ops.SUB]),
              RKEWOp(RKArg(RKBufferKind.ARG, out_slot), arg(negative), arg(int_tiles), count,
                       _EW_CFG[Ops.MAX], stateful=True, int32_output=True)))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<ee", 0.0, 1.0),
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=tuple(mid_gathers), gather_after=4)

def _pool_index_image(out_slot:int, count:int, spatial_size:int, source_count:int, plans:tuple[RKGather, ...],
                      extrema_plan:RKGather, weights:tuple[tuple[int, ...], ...], raw_weight:bool=False,
                      native_int16:bool=False) -> RKImage|None:
  """Validate static pool lanes and emit their descending-coordinate first-tie selection."""
  if (not 2 <= len(plans) <= _FP16_EXACT_INTEGER or source_count < spatial_size or source_count % spatial_size or
      count % (source_count//spatial_size) or len(weights) != len(plans) or
      len({plan.src_index for plan in plans}) != 1 or extrema_plan.src_index == plans[0].src_index or
      sorted(_plan_offsets(extrema_plan)) != list(range(count))): return None
  coordinates:list[tuple[int, ...]] = []
  valid_lanes = [0] * count
  min_bits = _int16_bits(dtypes.int16.min) if native_int16 else _fp16_bits(dtypes.half.min)
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
  if native_int16 and (raw_weight or spatial_size > 32767): return None
  if raw_weight: return _raw_weight_pool_index_image(out_slot, count, plans, extrema_plan, weights)
  if spatial_size > _FP16_EXACT_INTEGER:
    return _wide_pool_index_image(out_slot, count, spatial_size, plans, extrema_plan, tuple(coordinates))
  return _cumulative_index_image(out_slot, count, plans, (extrema_plan,), False, [0]*count,
                                 first_tie=True, candidate_coords=tuple(coordinates), index_limit=spatial_size,
                                 raw_weight=raw_weight, native_int16=native_int16)

def _lower_unrolled_pool_index(output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Select MaxPool's first spatial index with raw gathers and DPU equality masks."""
  _, out_param, count, out_index, value = output
  if count <= 0: return None
  dtype = dtypes.int16 if native_int16 else dtypes.half
  raw_weight = False
  if (root:=_descending_index_root(value)) is not None: spatial_size, selected = root
  else:
    negatives = [x for x in value.src if value.op is Ops.MUL and x.op is Ops.CONST and int(x.arg) == -1]
    selected_values = [x for x in value.src if x not in negatives]
    if len(negatives) != 1 or len(selected_values) != 1: return None
    spatial_size, selected, raw_weight = 0, selected_values[0], True
  terms = _flatten_binary(selected, Ops.MAX)
  parsed = [_weighted_equality(term, dtype) for term in terms]
  if not 2 <= len(parsed) <= 2048 or any(item is None for item in parsed): return None
  concrete = [item for item in parsed if item is not None]
  pairs = tuple(pair for pair,_ in concrete)
  common = set(pairs[0]).intersection(*(set(pair) for pair in pairs[1:]))
  if len(common) != 1: return None
  extrema = next(iter(common))
  candidates = tuple(next((x for x in pair if x is not extrema), None) for pair in pairs)
  if any(x is None for x in candidates) or len(set(candidates)) != len(candidates): return None

  try:
    extrema_plan = _typed_gather_plan(extrema, out_index, count, out_param.arg.slot, dtype)
    candidate_plans = tuple(_typed_gather_plan(candidate, out_index, count, out_param.arg.slot, dtype)
                            for candidate in candidates if candidate is not None)
  except RuntimeError: return None
  if extrema_plan is None or len(candidate_plans) != len(candidates) or any(plan is None for plan in candidate_plans): return None
  plans = tuple(plan for plan in candidate_plans if plan is not None)
  params = [u for u in value.toposort() if u.op is Ops.PARAM and u.arg.slot in (plans[0].src_index, extrema_plan.src_index)]
  source_params = [u for u in params if u.arg.slot == plans[0].src_index]
  extrema_params = [u for u in params if u.arg.slot == extrema_plan.src_index]
  if (len(set(source_params)) != 1 or len(set(extrema_params)) != 1 or any(p.dtype.scalar() is not dtype for p in params) or
      any(p.src[0].op is not Ops.CONST for p in params)): return None
  source_count = int(source_params[0].src[0].arg)
  if raw_weight: spatial_size = source_count
  if int(extrema_params[0].src[0].arg) != count: return None
  try:
    weights = tuple(_static_int_vector(out_index, weight, count) for _,weight in concrete)
  except (OverflowError, RuntimeError): return None
  return _pool_index_image(out_param.arg.slot, count, spatial_size, source_count, plans, extrema_plan, weights,
                           raw_weight=raw_weight, native_int16=native_int16)

def _lower_loop_pool_index(uops:list[UOp], output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Lower the one-register loop used by a global MaxPool returned index."""
  _, out_param, count, out_index, value = output
  dtype = dtypes.int16 if native_int16 else dtypes.half
  if count != 1 or _index_ranges(out_index) or (root:=_descending_index_root(value)) is None: return None
  spatial_size, selected = root
  if selected.op is not Ops.LOAD: return None
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
      (parsed:=_weighted_equality(weighted, dtype)) is None): return None
  pair, weight = parsed
  candidates = [x for x in pair if reduce_range in x.toposort()]
  extrema = [x for x in pair if reduce_range not in x.toposort()]
  if len(candidates) != 1 or len(extrema) != 1 or candidates[0].op is not Ops.LOAD or extrema[0].op is not Ops.LOAD: return None
  candidate_param, candidate_index = _root_param(candidates[0].src[0]), candidates[0].src[0].src[1]
  extrema_plan = _typed_gather_plan(extrema[0], out_index, 1, out_param.arg.slot, dtype)
  extrema_params = [u for u in uops if u.op is Ops.PARAM and extrema_plan is not None and u.arg.slot == extrema_plan.src_index]
  if (candidate_param is None or candidate_param.src[0].op is not Ops.CONST or candidate_param.dtype.scalar() is not dtype or
      extrema_plan is None or int(candidate_param.src[0].arg) != spatial_size or len(set(extrema_params)) != 1 or
      extrema_params[0].dtype.scalar() is not dtype or extrema_params[0].src[0].op is not Ops.CONST or
      int(extrema_params[0].src[0].arg) != 1): return None
  try:
    offsets = tuple(_eval_int(candidate_index, {reduce_range:i}) for i in range(window))
    weights = tuple(_eval_int(weight, {reduce_range:i}) for i in range(window))
  except RuntimeError: return None
  if len(set(offsets)) != window: return None
  plans = tuple(RKGather(candidate_param.arg.slot, 0, 1, base=offset) for offset in offsets)
  return _pool_index_image(out_param.arg.slot, 1, spatial_size, spatial_size, plans, extrema_plan,
                           tuple((weight,) for weight in weights), native_int16=native_int16)

def _int16_max_unpool_image(out_slot:int, count:int, out_spatial:int, index_slot:int,
                            plans:tuple[RKGather, ...], index_offset:int=0) -> RKImage|None:
  """Scatter INT16 values through exact INT32 indices, then accumulate and write native INT32."""
  pooled = len(plans)
  if not pooled or out_spatial <= 0: return None
  vector_lanes = _stripe_layout(count, pooled)[1]
  matrix_lanes = pooled*vector_lanes
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(size); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  values, selected, converted = scratch(matrix_lanes*2), scratch(matrix_lanes*2), scratch(matrix_lanes*4)
  gathers:list[RKGather] = [replace(plan, dst_index=values, dst_addend=row*vector_lanes) for row,plan in enumerate(plans)]
  ops:list[RKEWOp] = []
  coordinates = tuple(tuple(lane%out_spatial-index_offset for lane in range(count)) for _ in range(pooled))
  offsets = tuple(_plan_offsets(plan) for plan in plans)
  if (mask:=_native_int16_byte_mask(ops, gathers, scratch, index_slot, offsets, (coordinates,), count, vector_lanes)) is None: return None
  ops.extend((RKEWOp(arg(selected), arg(values), mask, matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True),
              RKEWOp(arg(converted), arg(selected), arg(selected), matrix_lanes, _EW_CFG[Ops.MAX], int16_input=True, int32_output=True)))
  reduced = _reduce_rows(ops, [arg(converted, row*vector_lanes*4) for row in range(pooled)], count, _EW_CFG[Ops.ADD], int32=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, reduced, count,
                    _EW_CFG[Ops.MAX], int32_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

def _single_max_unpool_image(out_slot:int, count:int, out_spatial:int, source_count:int, index_slot:int,
                             plan:RKGather, index_offset:int) -> RKImage|None:
  """Select one pooled value by masking its exact two-byte FP16 representation on DPU."""
  if out_spatial > _FP16_EXACT_INTEGER or any(not 0 <= offset < source_count for offset in _plan_offsets(plan)): return None
  (zero, one, compact_index, raw_low, raw_high, convert_tiles, half_low, half_high, index_matrix, coordinate,
   value_low, value_high, diff, magnitude, unequal, equal, selected_low, selected_high, int_tiles, int_low, int_high) = range(21)
  scratch_sizes = [_scratch_bytes(count)]*21
  for slot in (compact_index, half_low, half_high): scratch_sizes[slot] = _scratch_bytes(source_count)
  for slot in (raw_low, raw_high): scratch_sizes[slot] = source_count*4
  scratch_sizes[convert_tiles] = scratch_sizes[int_tiles] = _int32_tiles_bytes(max(source_count, count))
  scratch_sizes[int_low] = scratch_sizes[int_high] = max(count*4, 64)

  byte_offsets = tuple(lane*2 for lane in range(source_count))
  coordinate_bits = tuple(_fp16_bits(lane%out_spatial-index_offset) for lane in range(count))
  gathers = (RKGather(plan.src_index, raw_low, source_count, offsets=byte_offsets, dst_stride=4, itemsize=1),
             RKGather(plan.src_index, raw_high, source_count, offsets=tuple(offset+1 for offset in byte_offsets), dst_stride=4, itemsize=1),
             RKGather(index_slot, coordinate, count, values=coordinate_bits))
  mid_gathers = (replace(plan, src_index=compact_index, dst_index=index_matrix, src_kind=RKBufferKind.SCRATCH),
                 replace(plan, src_index=half_low, dst_index=value_low, src_kind=RKBufferKind.SCRATCH),
                 replace(plan, src_index=half_high, dst_index=value_high, src_kind=RKBufferKind.SCRATCH))
  def arg(slot:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot)
  ops:list[RKEWOp] = [RKEWOp(arg(compact_index), RKArg(RKBufferKind.ARG, index_slot), arg(convert_tiles), source_count,
                              _EW_CFG[Ops.MAX], int32_input=True),
                       RKEWOp(arg(half_low), arg(raw_low), arg(convert_tiles), source_count, _EW_CFG[Ops.MAX], int32_input=True),
                       RKEWOp(arg(half_high), arg(raw_high), arg(convert_tiles), source_count, _EW_CFG[Ops.MAX], int32_input=True)]
  equal_arg = _ew_eq_mask(ops, arg, index_matrix, coordinate, (diff, magnitude, unequal, equal), one, count)
  ops.extend((RKEWOp(arg(selected_low), arg(value_low), equal_arg, count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(arg(selected_high), arg(value_high), equal_arg, count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(arg(int_low), arg(selected_low), arg(int_tiles), count, _EW_CFG[Ops.MAX], int32_output=True),
              RKEWOp(arg(int_high), arg(selected_high), arg(int_tiles), count, _EW_CFG[Ops.MAX], int32_output=True)))
  native_offsets = tuple(range(0, count*4, 4))
  post_gathers = tuple(RKGather(slot, out_slot, count, offsets=native_offsets, dst_stride=2, dst_addend=byte,
                                dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
                       for byte,slot in enumerate((int_low, int_high)))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<ee", 0.0, 1.0),
                 gathers=gathers, ew_ops=tuple(ops), mid_gathers=mid_gathers, post_gathers=post_gathers, gather_after=3)

def _max_unpool_image(out_slot:int, count:int, out_spatial:int, source_count:int, index_slot:int,
                      plans:tuple[RKGather, ...], index_offset:int=0) -> RKImage|None:
  """Emit exact dynamic-index comparison, FP16 selection, and candidate reduction."""
  pooled = len(plans)
  if pooled == 1: return _single_max_unpool_image(out_slot, count, out_spatial, source_count, index_slot, plans[0], index_offset)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, pooled)
  if out_spatial > _FP16_EXACT_INTEGER:
    if index_offset: return None
    index_bytes = max(1, ((out_spatial-1).bit_length()+7)//8)
    if index_bytes > 4: return None
    zero, one = 0, 1
    raw_bytes = tuple(range(2, 2+index_bytes)); next_slot = 2+index_bytes
    convert_tiles = next_slot; next_slot += 1
    half_bytes = tuple(range(next_slot, next_slot+index_bytes)); next_slot += index_bytes
    index_matrices = tuple(range(next_slot, next_slot+index_bytes)); next_slot += index_bytes
    coordinate_matrices = tuple(range(next_slot, next_slot+index_bytes)); next_slot += index_bytes
    wide_values = next_slot; next_slot += 1
    diff, magnitude, unequal, equal, combined, selected_slot = range(next_slot, next_slot+6); next_slot += 6
    scratch_sizes = [_scratch_bytes(matrix_lanes)] * next_slot
    for slot in raw_bytes: scratch_sizes[slot] = source_count*4
    scratch_sizes[convert_tiles] = _int32_tiles_bytes(source_count)
    for slot in half_bytes: scratch_sizes[slot] = _scratch_bytes(source_count)

    wide_gathers:tuple[RKGather, ...] = tuple(RKGather(index_slot, slot, source_count,
      offsets=tuple(lane*4+byte for lane in range(source_count)), dst_stride=4, itemsize=1) for byte,slot in enumerate(raw_bytes))
    wide_gathers += tuple(replace(plan, dst_index=wide_values, dst_addend=row*vector_lanes) for row,plan in enumerate(plans))
    wide_coordinate_bits = tuple(tuple(_fp16_bits((lane%out_spatial >> (byte*8)) & 0xff)
                                       for lane in range(count)) for byte in range(index_bytes))
    for slot,bits in zip(coordinate_matrices, wide_coordinate_bits):
      wide_gathers += _stripe_gathers(index_slot, slot, count, (bits,)*pooled, vector_lanes, values=True)
    wide_mid_gathers = tuple(replace(plan, src_index=half_slot, dst_index=matrix_slot, dst_addend=row*vector_lanes,
                                     src_kind=RKBufferKind.SCRATCH)
                             for half_slot,matrix_slot in zip(half_bytes, index_matrices) for row,plan in enumerate(plans))
    def wide_arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
    wide_ops:list[RKEWOp] = [RKEWOp(wide_arg(dst), wide_arg(src), wide_arg(convert_tiles), source_count,
                                    _EW_CFG[Ops.MAX], int32_input=True) for src,dst in zip(raw_bytes, half_bytes)]
    gather_after = len(wide_ops)
    for byte,(index_matrix,coordinate_matrix) in enumerate(zip(index_matrices, coordinate_matrices)):
      byte_equal = _ew_integer_eq_mask(wide_ops, wide_arg, index_matrix, coordinate_matrix,
                                       (diff, magnitude, unequal, equal), one, matrix_lanes)
      wide_ops.append(RKEWOp(wide_arg(combined), wide_arg(combined) if byte else byte_equal,
                             byte_equal if byte else wide_arg(one), matrix_lanes,
                             _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
    wide_ops.append(RKEWOp(wide_arg(selected_slot), wide_arg(wide_values), wide_arg(combined), matrix_lanes,
                           _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
    reduced = _reduce_rows(wide_ops, [wide_arg(selected_slot, row*vector_bytes) for row in range(pooled)], count, _EW_CFG[Ops.ADD])
    wide_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, wide_arg(zero), count, _EW_CFG[Ops.ADD]))
    return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<ee", 0.0, 1.0),
                   gathers=wide_gathers, ew_ops=tuple(wide_ops), mid_gathers=wide_mid_gathers, gather_after=gather_after)

  zero, one, compact_index, convert_tiles, half_index, coordinate_slot, values, diff, magnitude, unequal, equal, selected_slot = range(12)
  scratch_sizes = [matrix_lanes*2] * 12
  scratch_sizes[compact_index], scratch_sizes[convert_tiles] = source_count*2, _int32_tiles_bytes(source_count)
  scratch = tuple(RKScratch(size) for size in scratch_sizes)
  coordinate_bits = tuple(_fp16_bits(lane%out_spatial-index_offset) for lane in range(count))
  gathers:tuple[RKGather, ...] = (); mid_gathers:tuple[RKGather, ...] = ()
  for row,plan in enumerate(plans):
    gathers += (replace(plan, dst_index=values, dst_addend=row*vector_lanes),
                RKGather(index_slot, coordinate_slot, count, values=coordinate_bits, dst_addend=row*vector_lanes))
    mid_gathers += (replace(plan, src_index=compact_index, dst_index=half_index, dst_addend=row*vector_lanes,
                            src_kind=RKBufferKind.SCRATCH),)
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(compact_index), RKArg(RKBufferKind.ARG, index_slot), arg(convert_tiles), source_count,
                              _EW_CFG[Ops.MAX], int32_input=True)]
  equal_arg = _ew_eq_mask(ops, arg, half_index, coordinate_slot, (diff, magnitude, unequal, equal), one, matrix_lanes)
  ops.append(RKEWOp(arg(selected_slot), arg(values), equal_arg, matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  reduced = _reduce_rows(ops, [arg(selected_slot, row*vector_bytes) for row in range(pooled)], count, _EW_CFG[Ops.ADD])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, arg(zero), count, _EW_CFG[Ops.ADD]))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops),
                 mid_gathers=mid_gathers, gather_after=1)

RKDynamicIndex = tuple[int, int, tuple[int, ...]]
def _raw_16bit_gathers(src_slot:int, raw_slots:tuple[int, int], count:int, offset_rows:tuple[tuple[int, ...], ...],
                       vector_lanes:int) -> tuple[RKGather, ...]:
  return tuple(RKGather(src_slot, slot, count, offsets=tuple(offset*2+byte for offset in offsets), dst_stride=4,
                        dst_addend=row*vector_lanes*4, itemsize=1)
               for byte,slot in enumerate(raw_slots) for row,offsets in enumerate(offset_rows))

def _append_byte_conversions(ops:list[RKEWOp], arg:Callable[[int], RKArg], raw:tuple[int, ...], half:tuple[int, ...],
                             tiles:int, count:int) -> None:
  ops.extend(RKEWOp(arg(dst), arg(src), arg(tiles), count, _EW_CFG[Ops.MAX], int32_input=True) for src,dst in zip(raw, half))

def _raw_16bit_output(ops:list[RKEWOp], arg:Callable[[int], RKArg], values:tuple[RKArg, RKArg], count:int, tiles:int,
                      int_slots:tuple[int, int], out_slot:int, int16_output:bool=False) -> tuple[RKGather, ...]:
  ops.extend(RKEWOp(arg(dst), value, value if int16_output else arg(tiles), count, _EW_CFG[Ops.MAX],
                    submit_barrier=int16_output, int32_output=not int16_output, int16_output=int16_output)
             for value,dst in zip(values, int_slots))
  itemsize = 2 if int16_output else 4
  offsets = tuple(range(0, count*itemsize, itemsize))
  return tuple(RKGather(slot, out_slot, count, offsets=offsets, dst_stride=2, dst_addend=byte,
                        dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
               for byte,slot in enumerate(int_slots))

def _lower_exact_fp16_copysign(output:RKOutput) -> RKImage|None:
  """Replace only the raw FP16 sign bit using exact byte conversion and DPU masks."""
  _, out_param, count, out_index, value = output
  if value.op is not Ops.ADD or value.arg != _NATIVE_COPYSIGN or len(value.src) != 2: return None
  magnitude, sign = value.src
  if any(x.op is not Ops.LOAD or x.dtype.scalar() is not dtypes.half or len(x.src) != 1 or x.src[0].op is not Ops.INDEX
         for x in (magnitude, sign)): return None
  params = tuple(_root_param(x.src[0]) for x in (magnitude, sign))
  if any(param is None or param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST for param in params): return None
  concrete = tuple(param for param in params if param is not None)
  try: magnitude_offsets, sign_offsets = (_gather_offsets(out_index, x.src[0].src[1], None, count) for x in (magnitude, sign))
  except RuntimeError: return None
  if any(not 0 <= offset < int(param.src[0].arg) for offsets,param in zip((magnitude_offsets, sign_offsets), concrete)
         for offset in offsets): return None

  threshold, sign_value, raw_bytes, half_bytes, convert_tiles, diff, sign_masks, weights, stripped, int_tiles, int_bytes = range(11)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, 3)
  high_lanes, output_lanes = 2*vector_lanes, 2*vector_lanes
  scratch_sizes = [_scratch_bytes(matrix_lanes)] * 11
  scratch_sizes[raw_bytes] = matrix_lanes*4
  scratch_sizes[convert_tiles] = _int32_tiles_bytes(matrix_lanes)
  scratch_sizes[int_tiles] = _int32_tiles_bytes(output_lanes)
  scratch_sizes[int_bytes] = output_lanes*4
  magnitude_slot, sign_slot = concrete[0].arg.slot, concrete[1].arg.slot
  gathers = (RKGather(magnitude_slot, raw_bytes, count, offsets=tuple(offset*2 for offset in magnitude_offsets), dst_stride=4,
                      itemsize=1),
             RKGather(magnitude_slot, raw_bytes, count, offsets=tuple(offset*2+1 for offset in magnitude_offsets), dst_stride=4,
                      dst_addend=vector_lanes*4, itemsize=1),
             RKGather(sign_slot, raw_bytes, count, offsets=tuple(offset*2+1 for offset in sign_offsets), dst_stride=4,
                      dst_addend=2*vector_lanes*4, itemsize=1))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = [RKEWOp(arg(half_bytes), arg(raw_bytes), arg(convert_tiles), matrix_lanes, _EW_CFG[Ops.MAX], int32_input=True),
         RKEWOp(arg(diff), arg(half_bytes, vector_bytes), arg(threshold), high_lanes, _EW_CFG[Ops.SUB]),
         RKEWOp(arg(sign_masks), arg(diff), arg(diff), high_lanes, _EW_CFG[Ops.MAX], compare=True),
         RKEWOp(arg(weights), arg(sign_masks), arg(sign_value), high_lanes, _EW_CFG[Ops.MUL], stateful=True),
         RKEWOp(arg(stripped), arg(half_bytes, vector_bytes), arg(weights), vector_lanes, _EW_CFG[Ops.SUB], stateful=True),
         RKEWOp(arg(half_bytes, vector_bytes), arg(stripped), arg(weights, vector_bytes), vector_lanes, _EW_CFG[Ops.ADD], stateful=True),
         RKEWOp(arg(int_bytes), arg(half_bytes), arg(int_tiles), output_lanes, _EW_CFG[Ops.MAX], int32_output=True)]
  offsets = tuple(range(0, count*4, 4))
  post = (RKGather(int_bytes, out_param.arg.slot, count, offsets=offsets, dst_stride=2, dst_kind=RKBufferKind.ARG,
                   src_kind=RKBufferKind.SCRATCH, itemsize=1),
          RKGather(int_bytes, out_param.arg.slot, count, offsets=tuple(vector_lanes*4+offset for offset in offsets), dst_stride=2,
                   dst_addend=1, dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<ee", 127.0, 128.0),
                 gathers=gathers, ew_ops=tuple(ops), post_gathers=post)

def _lower_bounded_exact_fp16_copysign(uops:list[UOp]) -> RKImage|None:
  """Keep the reset-heavy raw-bit path within one page of four-lane conversion tiles."""
  if (output:=_output_store(uops, dtypes.half)) is None: return None
  matrix_lanes = _stripe_layout(output[2], 3)[2]
  if _int32_tiles_bytes(matrix_lanes) > os.sysconf("SC_PAGESIZE"): return None
  return None if (tagged:=_fold_copysign(output[4])) is None else _lower_exact_fp16_copysign((*output[:4], tagged))

def _dynamic_16bit_gather_image(out_slot:int, count:int, indices:tuple[RKDynamicIndex, ...], plans:tuple[RKGather, ...],
                                coordinates:tuple[tuple[int, ...], ...],
                                alternate_coordinates:tuple[tuple[tuple[int, ...], ...], ...]|None=None,
                                gate:RKDynamicIndex|None=None) -> RKImage|None:
  """Select raw FP16 or INT16 bytes with exact native integer masks, sharing repeated trailing lanes."""
  if (not plans or len(coordinates) != len(indices) or any(len(axis) != len(plans) for axis in coordinates) or
      alternate_coordinates is not None and (len(alternate_coordinates) != len(indices) or
        any(any(len(axis) != len(plans) for axis in alternatives) for alternatives in alternate_coordinates)) or
      len({plan.src_index for plan in plans}) != 1): return None
  plan_offsets = tuple(_plan_offsets(plan) for plan in plans)
  repeat = next((lanes for lanes in range(min(8, count), 0, -1) if count%lanes == 0 and
                 all(all(row[start:start+lanes] == (row[start],)*lanes for start in range(0, count, lanes))
                     for row in (*(offsets for _,_,offsets in indices), *((gate[2],) if gate else ())))), 1)
  group_count = count//repeat
  grouped_indices = tuple((slot, index_count, offsets[::repeat]) for slot,index_count,offsets in indices)
  grouped_gate = None if gate is None else (gate[0], gate[1], gate[2][::repeat])
  vector_lanes = _stripe_layout(group_count, 2)[1] if len(plans) > 1 else group_count
  block_rows = max(1, _MAX_EW_ELEMS_FP16//vector_lanes)
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(size); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  block_values:list[list[tuple[RKArg, RKArg]]] = []
  for start in range(0, len(plans), block_rows):
    stop = min(len(plans), start+block_rows)
    rows, matrix_lanes = stop-start, (stop-start)*vector_lanes
    axis_masks:list[RKArg] = []
    for axis,((index_slot,_,index_offsets),candidate_values) in enumerate(zip(grouped_indices, coordinates)):
      alternatives = (candidate_values,) + (() if alternate_coordinates is None else alternate_coordinates[axis])
      coordinate_sets = tuple(tuple((value,)*group_count for value in alternative[start:stop]) for alternative in alternatives)
      if (mask:=_native_int16_byte_mask(ops, gathers, scratch, index_slot, index_offsets,
                                        coordinate_sets, group_count, vector_lanes)) is None: return None
      axis_masks.append(mask)
    combined_mask = axis_masks[0]
    for axis_mask in axis_masks[1:]:
      dst = scratch(matrix_lanes*2); ops.append(RKEWOp(arg(dst), combined_mask, axis_mask, matrix_lanes,
        _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)); combined_mask = arg(dst)
    matrix_value = tuple(tuple(scratch(matrix_lanes*2) for _ in range(2)) for _ in range(repeat))
    selected = tuple(tuple(scratch(matrix_lanes*2) for _ in range(2)) for _ in range(repeat))
    for candidate,row in enumerate(plan_offsets[start:stop]):
      for channel in range(repeat):
        for byte,slot in enumerate(matrix_value[channel]):
          gathers.append(RKGather(plans[0].src_index, slot, group_count,
            offsets=tuple(offset*2+byte for offset in row[channel::repeat]), dst_stride=2,
            dst_addend=candidate*vector_lanes*2, itemsize=1))
    block_result:list[tuple[RKArg, RKArg]] = []
    for channel in range(repeat):
      ops.extend(RKEWOp(arg(dst), arg(src), combined_mask, matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)
                 for src,dst in zip(matrix_value[channel], selected[channel]))
      block_result.append(tuple(_reduce_rows(ops, [arg(slot, row*vector_lanes*2) for row in range(rows)], group_count,
        _EW_CFG[Ops.ADD], int16=True) for slot in selected[channel]))  # type: ignore[arg-type]
    block_values.append(block_result)
  if not block_values: return None

  results:list[tuple[RKArg, RKArg]] = []
  for channel in range(repeat):
    channel_values:list[RKArg] = []
    for byte in range(2):
      value = block_values[0][channel][byte]
      for block in block_values[1:]:
        dst = scratch(group_count*2); ops.append(RKEWOp(arg(dst), value, block[channel][byte], group_count,
          _EW_CFG[Ops.ADD], int16_input=True, int16_output=True))
        value = arg(dst)
      channel_values.append(value)
    results.append((channel_values[0], channel_values[1]))
  if grouped_gate is not None:
    gate_slot = scratch(group_count*2)
    gathers.append(RKGather(grouped_gate[0], gate_slot, group_count, offsets=grouped_gate[2], dst_stride=2, itemsize=1))
    for channel,pair in enumerate(results):
      masked = (scratch(group_count*2), scratch(group_count*2))
      ops.extend(RKEWOp(arg(dst), value, arg(gate_slot), group_count, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)
                 for value,dst in zip(pair, masked))
      results[channel] = (arg(masked[0]), arg(masked[1]))
  byte_offsets = tuple(range(0, group_count*2, 2))
  post_gathers = tuple(RKGather(value.index, out_slot, group_count, offsets=tuple(value.addend+offset for offset in byte_offsets),
    dst_stride=repeat*2, dst_addend=channel*2+byte, dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
    for channel,pair in enumerate(results) for byte,value in enumerate(pair))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers),
                 ew_ops=tuple(ops), post_gathers=post_gathers)

def _dynamic_16bit_scatter_image(out_slot:int, count:int, index_slot:int, index_count:int,
                                 index_offset_rows:tuple[tuple[int, ...], ...], coordinate_rows:tuple[tuple[int, ...], ...],
                                 source_slot:int, source_offset_rows:tuple[tuple[int, ...], ...],
                                 base_slot:int, base_offsets:tuple[int, ...]) -> RKImage|None:
  """Apply bounded last-wins Scatter through exact INT32 equality and raw 16-bit selection."""
  rows = len(index_offset_rows)
  if (not rows or len(coordinate_rows) != rows or len(source_offset_rows) != rows or len(base_offsets) != count or
      any(len(row) != count for row in (*index_offset_rows, *coordinate_rows, *source_offset_rows))): return None
  if (equality:=_int32_equality_matrix(((index_slot, index_count, index_offset_rows, coordinate_rows),), count)) is None: return None
  next_slot = len(equality.scratch_sizes)
  raw_source = (next_slot, next_slot+1); half_source = (next_slot+2, next_slot+3)
  raw_base = (next_slot+4, next_slot+5); half_base = (next_slot+6, next_slot+7)
  effective, remaining, not_equal = range(next_slot+8, next_slot+11)
  selected = (next_slot+11, next_slot+12); result = (next_slot+13, next_slot+14)
  int_tiles, int_low, int_high = range(next_slot+15, next_slot+18)
  scratch_sizes = list(equality.scratch_sizes) + [_scratch_bytes(equality.matrix_lanes)] * 18
  for slot in raw_source: scratch_sizes[slot] = equality.matrix_lanes*4
  for slot in raw_base: scratch_sizes[slot] = count*4
  for slot in half_base: scratch_sizes[slot] = _scratch_bytes(count)
  for slot in (remaining, not_equal, *result): scratch_sizes[slot] = _scratch_bytes(count)
  scratch_sizes[equality.tiles] = _int32_tiles_bytes(max(equality.matrix_lanes, count, index_count))
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)
  scratch_sizes[int_low] = scratch_sizes[int_high] = count*4

  gathers = list(equality.gathers + _raw_16bit_gathers(source_slot, raw_source, count, source_offset_rows, equality.vector_lanes) +
                 _raw_16bit_gathers(base_slot, raw_base, count, (base_offsets,), count))

  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = list(equality.pre_ops)
  _append_byte_conversions(ops, arg, raw_source, half_source, equality.tiles, equality.matrix_lanes)
  _append_byte_conversions(ops, arg, raw_base, half_base, equality.tiles, count)
  gather_after = len(ops)
  ops.extend(equality.mask_ops)
  remaining_arg = arg(0)
  for row in range(rows-1, -1, -1):
    equal = RKArg(equality.mask.kind, equality.mask.index, equality.mask.addend+row*equality.vector_bytes)
    ops.extend((RKEWOp(arg(effective, row*equality.vector_bytes), equal, remaining_arg, count, _EW_CFG[Ops.MUL],
                       submit_barrier=True, stateful=True),
                RKEWOp(arg(not_equal), arg(0), equal, count, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True),
                RKEWOp(arg(remaining), remaining_arg, arg(not_equal), count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)))
    remaining_arg = arg(remaining)
  ops.extend(RKEWOp(arg(dst), arg(src), arg(effective), equality.matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)
             for src,dst in zip(half_source, selected))
  reduced = tuple(_reduce_rows(ops, [arg(slot, row*equality.vector_bytes) for row in range(rows)], count, _EW_CFG[Ops.ADD])
                  for slot in selected)
  ops.extend(RKEWOp(arg(dst), arg(src), remaining_arg, count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)
             for src,dst in zip(half_base, result))
  ops.extend(RKEWOp(arg(dst), candidate, arg(dst), count, _EW_CFG[Ops.ADD], submit_barrier=True, stateful=True)
             for candidate,dst in zip(reduced, result))
  post_gathers = _raw_16bit_output(ops, arg, (arg(result[0]), arg(result[1])), count, int_tiles,
                                   (int_low, int_high), out_slot)
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<e", 1.0),
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=equality.mid_gathers, gather_after=gather_after,
                 post_gathers=post_gathers)

def _lower_dynamic_scalar_scatter_reduce(output:RKOutput) -> RKImage|None:
  """Lower bounded scalar Scatter add/multiply through exact INT32 equality and DPU EW reduction."""
  _, out_param, count, out_index, value = output
  if count <= 0 or value.op not in (Ops.ADD, Ops.MUL): return None
  reduce_op, neutral = value.op, 0.0 if value.op is Ops.ADD else 1.0
  terms = _flatten_binary(value, reduce_op)
  bases = [term for term in terms if (root:=_strip_cast(term)).op is Ops.LOAD and root.dtype.scalar() is dtypes.half]
  updates = [term for term in terms if term not in bases]
  if len(bases) != 1 or not updates: return None
  base = _strip_cast(bases[0])
  base_param = _root_param(base.src[0]) if base.src and base.src[0].op is Ops.INDEX else None
  if base_param is None or base_param.dtype.scalar() is not dtypes.half or base_param.src[0].op is not Ops.CONST: return None

  neutral_bits = _fp16_bits(neutral)
  index_rows:list[tuple[int, ...]] = []
  coordinate_rows:list[tuple[int, ...]] = []
  valid_rows:list[tuple[int, ...]] = []
  scalar_bits:int|None = None
  index_param:UOp|None = None
  try:
    for term in updates:
      root = _strip_cast(term)
      int_loads = tuple({u.key:u for u in root.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int and
                         u.src and u.src[0].op is Ops.INDEX}.values())
      if len(int_loads) != 1: return None
      load = int_loads[0]
      direct = [(comparison, coordinate) for comparison in root.toposort() if comparison.op is Ops.CMPNE
                for dynamic,coordinate in (comparison.src, comparison.src[::-1])
                if dynamic is load and _is_static_expr(coordinate)]
      if len(direct) != 1: return None
      comparison, coordinate = direct[0]
      equal = root.substitute({comparison:comparison.const_like(False)})
      unequal = root.substitute({comparison:comparison.const_like(True)})
      if not _is_static_expr(equal) or not _is_static_expr(unequal): return None
      equal_bits, unequal_bits = _static_vector(out_index, equal, count), _static_vector(out_index, unequal, count)
      if any(bits != neutral_bits for bits in unequal_bits): return None
      nonneutral = {bits for bits in equal_bits if bits != neutral_bits}
      if len(nonneutral) != 1 or scalar_bits not in (None, next(iter(nonneutral))): return None
      scalar_bits = next(iter(nonneutral))
      valid_rows.append(tuple(_fp16_bits(bits != neutral_bits) for bits in equal_bits))
      coordinate_rows.append(_static_int_vector(out_index, coordinate, count))
      index_rows.append(_gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count))
      param = _root_param(load.src[0])
      if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST or \
         index_param is not None and param.arg.slot != index_param.arg.slot: return None
      index_param = param
  except (OverflowError, RuntimeError, struct.error): return None
  if scalar_bits is None or index_param is None: return None
  index_count = int(index_param.src[0].arg)
  if any(offset >= index_count for row in index_rows for offset in row): return None
  if (equality:=_int32_equality_matrix(((index_param.arg.slot, index_count, tuple(index_rows), tuple(coordinate_rows)),), count)) is None:
    return None
  try: base_plan = _gather_plan(base_param.arg.slot, 0, out_index, base.src[0].src[1], base.src[2] if len(base.src) == 3 else None, count)
  except RuntimeError: return None
  if any(offset >= int(base_param.src[0].arg) for offset in base_plan.offsets): return None

  scratch_sizes = list(equality.scratch_sizes)
  def scratch(lanes:int) -> int:
    scratch_sizes.append(_scratch_bytes(lanes)); return len(scratch_sizes)-1
  valid, effective = scratch(equality.matrix_lanes), scratch(equality.matrix_lanes)
  base_slot = scratch(count)
  gathers = list(equality.gathers)
  gathers.append(replace(base_plan, dst_index=base_slot))
  gathers.extend(_stripe_gathers(index_param.arg.slot, valid, count, valid_rows, equality.vector_lanes, values=True))
  def constant(bits:int, lanes:int) -> int:
    slot = scratch(lanes)
    gathers.append(RKGather(index_param.arg.slot, slot, lanes, values=(bits,)*lanes))
    return slot
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = list(equality.pre_ops) + list(equality.mask_ops)
  ops.append(RKEWOp(arg(effective), equality.mask, arg(valid), equality.matrix_lanes,
                    _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  scalar = struct.unpack("<e", struct.pack("<H", scalar_bits))[0]
  if scalar == 0.0: return None  # signed-zero selection needs raw representation handling
  out = RKArg(RKBufferKind.ARG, out_param.arg.slot)

  if reduce_op is Ops.MUL and math.isfinite(scalar):
    scalar_matrix = constant(scalar_bits, equality.matrix_lanes)
    selected, remaining, factors = scratch(equality.matrix_lanes), scratch(equality.matrix_lanes), scratch(equality.matrix_lanes)
    ops.extend((RKEWOp(arg(selected), arg(effective), arg(scalar_matrix), equality.matrix_lanes, _EW_CFG[Ops.MUL]),
                RKEWOp(arg(remaining), arg(0), arg(effective), equality.matrix_lanes, _EW_CFG[Ops.SUB]),
                RKEWOp(arg(factors), arg(remaining), arg(selected), equality.matrix_lanes, _EW_CFG[Ops.ADD])))
    factor = _reduce_rows(ops, [arg(factors, row*equality.vector_bytes) for row in range(len(updates))], count, _EW_CFG[Ops.MUL])
    ops.append(RKEWOp(out, arg(base_slot), factor, count, _EW_CFG[Ops.MUL]))
  else:
    hits = _reduce_rows(ops, [arg(effective, row*equality.vector_bytes) for row in range(len(updates))], count, _EW_CFG[Ops.ADD])
    remaining_slot:int|None = None
    if math.isfinite(scalar):
      scalar_slot, contribution = constant(scalar_bits, count), scratch(count)
      ops.append(RKEWOp(arg(contribution), hits, arg(scalar_slot), count, _EW_CFG[Ops.MUL]))
      result = arg(contribution)
    else:
      hit, maximum, doubled, special = scratch(count), constant(0x7bff, count), constant(0x4000, count), scratch(count)
      ops.extend((RKEWOp(arg(hit), hits, arg(0), count, _EW_CFG_MIN),
                  RKEWOp(arg(special), arg(hit), arg(maximum), count, _EW_CFG[Ops.MUL]),
                  RKEWOp(arg(special), arg(special), arg(doubled), count, _EW_CFG[Ops.MUL])))
      if math.isnan(scalar):
        remaining_slot = scratch(count)
        ops.extend((RKEWOp(arg(remaining_slot), arg(0), arg(hit), count, _EW_CFG[Ops.SUB]),
                    RKEWOp(arg(special), arg(special), arg(remaining_slot), count, _EW_CFG[Ops.MUL])))
      elif scalar < 0:
        zero = constant(0, count)
        ops.append(RKEWOp(arg(special), arg(zero), arg(special), count, _EW_CFG[Ops.SUB]))
      result = arg(special)
    if reduce_op is Ops.MUL:
      remaining, factor_slot = remaining_slot if remaining_slot is not None else scratch(count), scratch(count)
      if remaining_slot is None: ops.append(RKEWOp(arg(remaining), arg(0), arg(hit), count, _EW_CFG[Ops.SUB]))
      ops.extend((RKEWOp(arg(factor_slot), arg(remaining), result, count, _EW_CFG[Ops.ADD]),
                  RKEWOp(out, arg(base_slot), arg(factor_slot), count, _EW_CFG[Ops.MUL])))
    else: ops.append(RKEWOp(out, arg(base_slot), result, count, _EW_CFG[Ops.ADD]))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<e", 1.0),
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=equality.mid_gathers,
                 gather_after=len(equality.pre_ops))

def _negative_normalized_index(root:UOp) -> tuple[UOp, int]|None:
  """Recognize `WHERE(index < 0, index + extent, index)` exactly."""
  if root.op is not Ops.WHERE or root.src[0].op is not Ops.CMPLT or root.src[0].src[1].op is not Ops.CONST or \
     int(root.src[0].src[1].arg) != 0: return None
  index = root.src[0].src[0]
  if index.op is not Ops.LOAD or index.dtype.scalar() is not dtypes.int or root.src[2].key != index.key or root.src[1].op is not Ops.ADD:
    return None
  indexes = [x for x in root.src[1].src if x.key == index.key]
  extents = [x for x in root.src[1].src if x.op is Ops.CONST and int(x.arg) > 0]
  return (index, int(extents[0].arg)) if len(indexes) == len(extents) == 1 else None

def _bounded_index_gate(gate:UOp, bounded:UOp, limit:int) -> bool:
  """Require the canonical nonnegative and strict-upper-bound checks within a conjunction."""
  if gate.op is not Ops.AND: return False
  nodes = gate.toposort()
  upper = any(u.op is Ops.CMPLT and u.src[0].key == bounded.key and u.src[1].op is Ops.CONST and
              int(u.src[1].arg) == limit for u in nodes)
  lower = any(u.op is Ops.CMPNE and any(marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and
              comparison.op is Ops.CMPLT and comparison.src[0].key == bounded.key and comparison.src[1].op is Ops.CONST and
              int(comparison.src[1].arg) == 0 for comparison,marker in (u.src, u.src[::-1])) for u in nodes)
  return lower and upper

def _native_int16_byte_mask(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], index_slot:int,
                            index_offsets:tuple[int, ...]|tuple[tuple[int, ...], ...], coordinate_sets:tuple[RKCoordinateRows, ...],
                            count:int, vector_lanes:int) -> RKArg|None:
  """Compare arbitrary INT32 values exactly as four unsigned bytes using native INT16 DPU EW."""
  rows = len(coordinate_sets[0]) if coordinate_sets else 0
  if not rows or any(len(group) != rows or any(len(row) != count for row in group) for group in coordinate_sets): return None
  matrix_lanes = rows*vector_lanes
  if index_offsets and isinstance(index_offsets[0], int): offset_rows = (index_offsets,)*rows
  else: offset_rows = typing_cast(tuple[tuple[int, ...], ...], index_offsets)
  if len(offset_rows) != rows or any(len(offsets) != count for offsets in offset_rows): return None
  one, diff, magnitude, unequal = (scratch(matrix_lanes*2) for _ in range(4))
  gathers.append(RKGather(index_slot, one, matrix_lanes, values=(1,)*matrix_lanes, itemsize=2))
  def arg(slot:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot)
  masks:list[RKArg] = []
  for coordinates in coordinate_sets:
    byte_masks:list[RKArg] = []
    for byte in range(4):
      dynamic, static, equal = (scratch(matrix_lanes*2) for _ in range(3))
      gathers.extend(RKGather(index_slot, dynamic, count, offsets=tuple(offset*4+byte for offset in offsets),
        dst_stride=2, dst_addend=row*vector_lanes*2, itemsize=1) for row,offsets in enumerate(offset_rows))
      values = tuple((value >> (byte*8)) & 0xff for row in coordinates for value in (*row, *((0,)*(vector_lanes-count))))
      gathers.append(RKGather(index_slot, static, matrix_lanes, values=values, itemsize=2))
      ops.extend((RKEWOp(arg(diff), arg(dynamic), arg(static), matrix_lanes, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True),
                  RKEWOp(arg(magnitude), arg(diff), arg(diff), matrix_lanes, _EW_CFG_ABS, int16_input=True, int16_output=True),
                  RKEWOp(arg(unequal), arg(magnitude), arg(one), matrix_lanes, _EW_CFG_MIN, int16_input=True, int16_output=True),
                  RKEWOp(arg(equal), arg(one), arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True)))
      byte_masks.append(arg(equal))
    mask = byte_masks[0]
    for byte_mask in byte_masks[1:]:
      dst = scratch(matrix_lanes*2); ops.append(RKEWOp(arg(dst), mask, byte_mask, matrix_lanes,
        _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)); mask = arg(dst)
    masks.append(mask)
  mask = masks[0]
  for alternate in masks[1:]:
    dst = scratch(matrix_lanes*2); ops.append(RKEWOp(arg(dst), mask, alternate, matrix_lanes,
      _EW_CFG[Ops.MAX], int16_input=True, int16_output=True)); mask = arg(dst)
  return mask

def _native_int32_nonzero_mask(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], source_slot:int,
                               rows:tuple[tuple[int, ...], ...], count:int, vector_lanes:int) -> RKArg|None:
  """Test four opaque INT32 bytes for nonzero using unsigned-byte INT16 MIN/MAX stages."""
  if not rows or any(len(row) != count for row in rows): return None
  matrix_lanes = len(rows)*vector_lanes
  one = scratch(matrix_lanes*2)
  byte_masks = tuple(scratch(matrix_lanes*2) for _ in range(4))
  gathers.append(RKGather(source_slot, one, matrix_lanes, values=(1,)*matrix_lanes, itemsize=2))
  for byte,slot in enumerate(byte_masks):
    gathers.extend(RKGather(source_slot, slot, count,
      offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in row), dst_addend=i*vector_lanes*2,
      dst_stride=2, itemsize=1) for i,row in enumerate(rows))
  def arg(slot:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot)
  for slot in byte_masks:
    ops.append(RKEWOp(arg(slot), arg(slot), arg(one), matrix_lanes, _EW_CFG_MIN, int16_input=True, int16_output=True))
  mask = arg(byte_masks[0])
  for slot in byte_masks[1:]:
    ops.append(RKEWOp(mask, mask, arg(slot), matrix_lanes, _EW_CFG[Ops.MAX], int16_input=True, int16_output=True))
  return mask

def _lower_int32_bounds_mask(output:RKOutput) -> RKImage|None:
  """Lower canonical positive-only or negative-normalized INT32 index bounds to one exact bool mask."""
  _, out_param, count, out_index, root = output
  if count <= 0: return None
  normalized = tuple((node, *parsed) for node in root.toposort() if (parsed:=_negative_normalized_index(node)) is not None)
  normalized_by_load = {load.key:(bounded, extent) for bounded,load,extent in normalized}
  loads = tuple({u.key:u for u in root.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}.values())
  if not loads or len(normalized_by_load) != len(normalized): return None
  specs:list[tuple[UOp, UOp, int, bool, UOp]] = []
  for load in loads:
    if load.key in normalized_by_load:
      bounded, extent = normalized_by_load[load.key]; wrapped = True
    else:
      limits = {int(u.src[1].arg) for u in root.toposort() if u.op is Ops.CMPLT and u.src[0].key == load.key and
                u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0}
      if len(limits) != 1: return None
      bounded, extent, wrapped = load, next(iter(limits)), False
    gates = [node for node in root.toposort() if _bounded_index_gate(node, bounded, extent) and
             {u.key for u in node.toposort() if u.op is Ops.LOAD} == {load.key}]
    if len(gates) != 1: return None
    specs.append((bounded, load, extent, wrapped, gates[0]))
  terms = [term for *_,term in specs]
  actual_leaves, expected_leaves = _flatten_binary(root, Ops.AND), tuple(x for term in terms for x in _flatten_binary(term, Ops.AND))
  if len(actual_leaves) != len(expected_leaves) or \
     {x.key for x in actual_leaves} != {x.key for x in expected_leaves}: return None
  axes:list[tuple[UOp, int, UOp, tuple[int, ...], int, bool]] = []
  for bounded,load,extent,wrapped,term in specs:
    param = _root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None
    if (not _bounded_index_gate(term, bounded, extent) or param is None or param.dtype.scalar() is not dtypes.int or
        param.src[0].op is not Ops.CONST or {u.key for u in term.toposort() if u.op is Ops.LOAD} != {load.key}): return None
    try: offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
    except RuntimeError: return None
    index_count = int(param.src[0].arg)
    if any(not 0 <= offset < index_count for offset in offsets): return None
    axes.append((param, index_count, load, offsets, extent, wrapped))
  if {u.key for u in root.toposort() if u.op is Ops.LOAD} != {load.key for _,_,load,_,_,_ in axes}: return None

  layouts = tuple((*_stripe_layout(count, extent), extent) for *_,extent,_ in axes)
  if any(matrix_lanes > _MAX_EW_ELEMS_FP16 for _,_,matrix_lanes,_ in layouts): return None
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(size); return len(scratch_sizes)-1
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  valid_axes:list[RKArg] = []
  for (param,_,_,offsets,extent,wrapped),(_,vector_lanes,_,_) in zip(axes, layouts):
    positive = tuple((coordinate,)*count for coordinate in range(extent))
    negative = tuple((coordinate,)*count for coordinate in range(-extent, 0))
    if (mask:=_native_int16_byte_mask(ops, gathers, scratch, param.arg.slot, offsets,
                                      (positive, negative) if wrapped else (positive,), count, vector_lanes)) is None: return None
    valid_axes.append(_reduce_rows(ops, [RKArg(mask.kind, mask.index, mask.addend+row*vector_lanes*2)
                                        for row in range(extent)], count, _EW_CFG[Ops.MAX], int16=True))
  result = valid_axes[0]
  for valid in valid_axes[1:]:
    dst = scratch(count*2); ops.append(RKEWOp(arg(dst), result, valid, count, _EW_CFG[Ops.MUL],
      int16_input=True, int16_output=True)); result = arg(dst)
  post = (_int16_low_bytes(result, out_param.arg.slot, count),)
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes),
                 gathers=tuple(gathers), ew_ops=tuple(ops), post_gathers=post)

def _lower_normalized_linear_index(output:RKOutput) -> RKImage|None:
  """Lower a validity-gated sum of negative-normalized INT32 coordinates without a Cartesian candidate matrix."""
  _, out_param, count, out_index, root = output
  if count <= 0 or root.op is not Ops.WHERE or root.src[0].op is not Ops.CMPLT or root.src[0].src[1].op is not Ops.CONST or \
     int(root.src[0].src[1].arg) != 0 or root.src[2].key != root.src[0].src[0].key or root.src[1].op is not Ops.ADD:
    return None
  gated = root.src[2]
  extents = [x for x in root.src[1].src if x.op is Ops.CONST and int(x.arg) > 0]
  wrapped_values = [x for x in root.src[1].src if x.key == gated.key]
  if len(extents) != 1 or len(wrapped_values) != 1 or gated.op is not Ops.WHERE or gated.src[2].op is not Ops.CONST or \
     int(gated.src[2].arg) != 0: return None
  gate, linear, source_extent = gated.src[0], gated.src[1], int(extents[0].arg)
  if gate.op is not Ops.LOAD or gate.dtype.scalar() is not dtypes.bool or len(gate.src) != 1 or gate.src[0].op is not Ops.INDEX:
    return None
  gate_param = _root_param(gate.src[0])
  if gate_param is None or gate_param.dtype.scalar() is not dtypes.bool or gate_param.src[0].op is not Ops.CONST: return None

  axes:list[tuple[UOp, int, tuple[int, ...], int, int, bool]] = []
  for term in _flatten_binary(linear, Ops.ADD):
    value, coefficient = term, 1
    if term.op is Ops.MUL:
      constants = [x for x in term.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.int]
      dynamic = [x for x in term.src if x not in constants]
      if len(constants) != len(dynamic) != 0 or len(constants) != 1: return None
      value, coefficient = dynamic[0], int(constants[0].arg)
    if coefficient <= 0: return None
    if (normalized:=_negative_normalized_index(value)) is not None: load, extent, wrapped = *normalized, True
    elif value.op is Ops.LOAD and value.dtype.scalar() is dtypes.int and source_extent%coefficient == 0:
      load, extent, wrapped = value, source_extent//coefficient, False
    else: return None
    param = _root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None
    if param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST: return None
    try: offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
    except RuntimeError: return None
    index_count = int(param.src[0].arg)
    if any(not 0 <= offset < index_count for offset in offsets): return None
    axes.append((param, index_count, offsets, extent, coefficient, wrapped))
  coefficients = {coefficient for *_,coefficient,_ in axes}
  refined:list[tuple[UOp, int, tuple[int, ...], int, int, bool]] = []
  for param,index_count,offsets,extent,coefficient,wrapped in axes:
    upper_stride = min((other for other in coefficients if other > coefficient), default=source_extent)
    if not wrapped:
      if upper_stride%coefficient: return None
      extent = upper_stride//coefficient
    refined.append((param,index_count,offsets,extent,coefficient,wrapped))
  axes = refined
  try: gate_offsets = _gather_offsets(out_index, gate.src[0].src[1], None, count)
  except RuntimeError: return None
  if (not axes or any(not 0 <= offset < int(gate_param.src[0].arg) for offset in gate_offsets) or
      len({param.arg.slot for param,_,_,_,_,_ in axes}) != len(axes) or
      {u.key for u in root.toposort() if u.op is Ops.LOAD} != {gate.key, *(u.key for u in linear.toposort() if u.op is Ops.LOAD)}):
    return None
  if source_extent >= 1<<15: return None

  layouts = tuple((*_stripe_layout(count, extent), extent) for *_,extent,_,_ in axes)
  if any(matrix_lanes > _MAX_EW_ELEMS_FP16 for _,_,matrix_lanes,_ in layouts): return None
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(size); return len(scratch_sizes)-1
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  contributions:list[RKArg] = []
  for (param,_,offsets,extent,coefficient,wrapped),(_,vector_lanes,matrix_lanes,_) in zip(axes, layouts):
    positive = tuple((coordinate,)*count for coordinate in range(extent))
    negative = tuple((coordinate,)*count for coordinate in range(-extent, 0))
    if (mask:=_native_int16_byte_mask(ops, gathers, scratch, param.arg.slot, offsets,
                                      (positive, negative) if wrapped else (positive,), count, vector_lanes)) is None: return None
    weights, selected = scratch(matrix_lanes*2), scratch(matrix_lanes*2)
    values = tuple(value for row in range(extent) for value in (row*coefficient,)*count+(0,)*(vector_lanes-count))
    gathers.append(RKGather(param.arg.slot, weights, matrix_lanes, values=values, itemsize=2))
    ops.append(RKEWOp(arg(selected), mask, arg(weights), matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
    contributions.append(_reduce_rows(ops, [arg(selected, row*vector_lanes*2) for row in range(extent)], count,
                                      _EW_CFG[Ops.ADD], int16=True))
  result = contributions[0]
  for contribution in contributions[1:]:
    dst = scratch(count*2); ops.append(RKEWOp(arg(dst), result, contribution, count, _EW_CFG[Ops.ADD],
      int16_input=True, int16_output=True)); result = arg(dst)
  gate_slot, gated_slot = scratch(count*2), scratch(count*2)
  gathers.append(RKGather(gate_param.arg.slot, gate_slot, count, offsets=gate_offsets, dst_stride=2, itemsize=1))
  ops.append(RKEWOp(arg(gated_slot), result, arg(gate_slot), count, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
  result = arg(gated_slot)
  post = (RKGather(0, out_param.arg.slot, count*4, values=(0,)*(count*4), dst_kind=RKBufferKind.ARG, itemsize=1),
          RKGather(result.index, out_param.arg.slot, count, offsets=tuple(result.addend+i*2 for i in range(count)), partial=True, dst_stride=4,
                   dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1),
          RKGather(result.index, out_param.arg.slot, count, offsets=tuple(result.addend+i*2+1 for i in range(count)), partial=True,
                   dst_stride=4, dst_addend=1,
                   dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes),
                 gathers=tuple(gathers), ew_ops=tuple(ops), post_gathers=post)

def _full_predicate_count(expr:UOp, out_index:UOp, count:int, dtype:DType, predicate:Callable[[UOp], UOp|None],
                          max_scale:int=1) -> tuple[UOp, int]|None:
  """Prove an unrolled sum covers every typed source predicate uniformly, possibly repeated or scaled."""
  terms, source, offsets, scales = _flatten_binary(expr, Ops.ADD), None, [], []
  try:
    for term in terms:
      scale = 1
      if max_scale > 1 and term.op is Ops.MUL:
        constants = [u for u in term.src if u.op is Ops.CONST and u.dtype.scalar() is dtypes.int]
        values = [u for u in term.src if u not in constants]
        if len(constants) != 1 or len(values) != 1: return None
        scale, term = int(constants[0].arg), values[0]
      if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.int or len(term.src) != 1 or \
         (load:=predicate(term.src[0])) is None or [u for u in term.toposort() if u.op is Ops.LOAD] != [load]: return None
      param = _root_param(load.src[0])
      if (param is None or param.dtype.scalar() is not dtype or param.src[0].op is not Ops.CONST or
          source is not None and param.arg.slot != source.arg.slot): return None
      source = param
      row = _gather_offsets(out_index, load.src[0].src[1], None, count)
      if len(set(row)) != 1: return None
      offsets.append(row[0]); scales.append(scale)
  except RuntimeError: return None
  if source is None or not scales or len(set(scales)) != 1: return None
  source_count = int(source.src[0].arg)
  if source_count <= 0 or len(terms)%source_count or any(offsets.count(i) != len(terms)//source_count for i in range(source_count)): return None
  effective_scale = scales[0]*(len(terms)//source_count)
  return (source, effective_scale) if 1 <= effective_scale <= max_scale else None

def _full_fp16_greater_count(expr:UOp, out_index:UOp, count:int, max_scale:int=1) -> tuple[UOp, int, float]|None:
  """Prove a full unrolled count uses one uniform `constant < fp16_load` predicate."""
  thresholds:list[float] = []
  def predicate(u:UOp) -> UOp|None:
    if (parsed:=_greater_half_load(u)) is None: return None
    thresholds.append(parsed[1])
    return parsed[0]
  if (info:=_full_predicate_count(expr, out_index, count, dtypes.half, predicate, max_scale)) is None or not thresholds: return None
  if len({_fp16_bits(threshold) for threshold in thresholds}) != 1: return None
  return *info, thresholds[0]

def _full_bool_count(expr:UOp, out_index:UOp, count:int) -> tuple[UOp, int]|None:
  """Prove a scalar bool-load sum covers one complete external mask."""
  def predicate(u:UOp) -> UOp|None:
    return u if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.bool and u.src[0].op is Ops.INDEX else None
  return _full_predicate_count(expr, out_index, count, dtypes.bool, predicate)

def _lower_unrolled_fp16_predicate_total(output:RKOutput) -> RKImage|None:
  """Count one bounded full-source FP16 threshold predicate entirely on DPU EW."""
  _, out_param, count, out_index, root = output
  if count != 1 or (info:=_full_fp16_greater_count(root, out_index, count, 8)) is None: return None
  source, scale, threshold = info
  source_count = int(source.src[0].arg)
  if not 1 <= scale <= 8 or not 1 <= source_count*scale <= _FP16_EXACT_INTEGER: return None
  one, maximum, negative_maximum, threshold_slot, scale_slot, shifted = range(6)
  temps = tuple(range(6, 17)); spaced, int_tiles = 17, 18
  scratch = tuple(RKScratch(source_count*64 if slot == spaced else _int32_tiles_bytes(1) if slot == int_tiles else
                            _scratch_bytes(source_count)) for slot in range(int_tiles+1))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = [RKEWOp(arg(shifted), RKArg(RKBufferKind.ARG, source.arg.slot), arg(threshold_slot), source_count, _EW_CFG[Ops.SUB])]
  mask = _ew_ieee_positive_mask(ops, arg, shifted, (one, maximum, negative_maximum), temps, source_count)
  gather_after = len(ops)
  mid = (RKGather(mask.index, spaced, source_count, offsets=tuple(mask.addend//2+lane for lane in range(source_count)),
                  dst_stride=32, src_kind=RKBufferKind.SCRATCH),)
  total = _reduce_rows(ops, [arg(spaced, lane*64) for lane in range(source_count)], 1, _EW_CFG[Ops.ADD])
  if scale != 1: ops.append(RKEWOp(total, total, arg(scale_slot), 1, _EW_CFG[Ops.MUL]))
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), total, arg(int_tiles), 1,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  constants = struct.pack("<eeeee", 1.0, 65504.0, -65504.0, threshold, scale)
  return RKImage(RKTarget.RK3588, scratch, constants, ew_ops=tuple(ops), mid_gathers=mid, gather_after=gather_after)

def _lower_loop_fp16_predicate_total(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Normalize a scalar local-register predicate reduction into the verified unrolled total emitter."""
  store, out_param, count, out_index, _ = output
  if count != 1 or (loop:=_local_add_loop(uops, out_index)) is None: return None
  reduce_range, groups, term, _ = loop
  terms = [term.substitute({reduce_range:reduce_range.const_like(lane)}) for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value.alu(Ops.ADD, term)
  return _lower_unrolled_fp16_predicate_total((store, out_param, count, out_index, value))

def _lower_loop_fp16_prefix_count(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Normalize a local-register FP16 predicate scan into the blocked prefix emitter."""
  store, out_param, count, out_index, _ = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or (loop:=_local_add_loop(uops, out_index)) is None: return None
  reduce_range, groups, term, _ = loop
  terms = [term.substitute({reduce_range:reduce_range.const_like(lane)}) for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value.alu(Ops.ADD, term)
  return _lower_unrolled_fp16_prefix_count((store, out_param, count, out_index, value))

RKFixedNonzero = tuple[UOp, int, UOp, tuple[int, ...], tuple[tuple[int, ...], ...], int]
def _fixed_nonzero_plan(output:RKOutput, dtype:DType, predicate:Callable[[UOp], UOp|None],
                        encodable:Callable[[int], bool]) -> RKFixedNonzero|None:
  """Prove the common fixed-size Nonzero count, compact-index, coordinate, and fill graph."""
  _, _, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or root.op is not Ops.WHERE or len(root.src) != 3: return None
  fill = root.src[2]
  if fill.op is not Ops.CONST or fill.dtype.scalar() is not dtypes.int: return None
  fill_value = int(fill.arg)
  totals = [(u.src[1], info) for u in root.toposort() if u.op is Ops.CMPLT and u.src[0].key == out_index.key and
            (info:=_full_predicate_count(u.src[1], out_index, count, dtype, predicate, 8)) is not None]
  if len(totals) != 1: return None
  total_expr, (source, rank) = totals[0]
  coordinate_count = int(source.src[0].arg)*rank
  if not 1 <= coordinate_count <= _FP16_EXACT_INTEGER: return None
  source_loads = {u.key for u in total_expr.toposort() if u.op is Ops.LOAD}
  index_loads = [u for u in root.toposort() if u.op is Ops.LOAD and u.key not in source_loads]
  if len(index_loads) != 1 or {u.key for u in root.toposort() if u.op is Ops.LOAD} != source_loads|{index_loads[0].key}: return None
  index_load = index_loads[0]
  index_param = _root_param(index_load.src[0]) if index_load.src and index_load.src[0].op is Ops.INDEX else None
  if index_param is None or index_param.dtype.scalar() is not dtypes.int or index_param.src[0].op is not Ops.CONST or \
     int(index_param.src[0].arg) != count: return None
  try:
    index_offsets = _gather_offsets(out_index, index_load.src[0].src[1], None, count)
    total_const = total_expr.const_like(count)
    coordinate_rows = tuple(_static_int_vector(out_index, root.substitute(
      {total_expr:total_const, index_load:index_load.const_like(i)}), count) for i in range(coordinate_count))
    first = coordinate_rows[0]
    for selected_count in range(count+1):
      got = _static_int_vector(out_index, root.substitute(
        {total_expr:total_expr.const_like(selected_count), index_load:index_load.const_like(0)}), count)
      if got != tuple(first[lane] if lane < selected_count else fill_value for lane in range(count)): return None
    if index_offsets != tuple(range(count)) or not encodable(fill_value) or \
       any(not encodable(value) for row in coordinate_rows for value in row): return None
  except (RuntimeError, OverflowError, struct.error): return None
  return source, rank, index_param, index_offsets, coordinate_rows, fill_value

def _lower_fixed_fp16_nonzero(output:RKOutput) -> RKImage|None:
  """Select fixed nonzero coordinates through DPU count, exact INT32 equality, and static coordinate rows."""
  _, out_param, count, _, _ = output
  if (plan:=_fixed_nonzero_plan(output, dtypes.half, _nonzero_load,
                                lambda value:_eval_cast(value, dtypes.half) == value)) is None: return None
  source, rank, index_param, index_offsets, coordinate_rows, fill_value = plan
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows)

  equality_inputs = ((index_param.arg.slot, count, (index_offsets,)*coordinate_count,
                      tuple((candidate,)*count for candidate in range(coordinate_count))),)
  if (equality:=_int32_equality_matrix(equality_inputs, count)) is None: return None
  base = len(equality.scratch_sizes)
  (source_values, magnitude, nonzero, rank_value, total_vector, output_coordinate, coordinate_matrix, selected,
   valid_delta, valid, one, guarded, remaining, fill_slot, fill_part, result, int_tiles) = range(base, base+17)
  _, source_vector_lanes, source_matrix_lanes = _stripe_layout(1, source_count)
  scratch_sizes = [*equality.scratch_sizes, *([_scratch_bytes(max(count, equality.matrix_lanes))]*17)]
  scratch_sizes[source_values] = _scratch_bytes(source_matrix_lanes)
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)
  coordinate_bits = tuple(tuple(_fp16_bits(value) for value in row) for row in coordinate_rows)
  gathers = list(equality.gathers + _stripe_gathers(source.arg.slot, source_values, 1,
                 ((lane,) for lane in range(source_count)), source_vector_lanes))
  gathers += list(_stripe_gathers(source.arg.slot, coordinate_matrix, count, coordinate_bits, equality.vector_lanes, values=True))
  gathers.extend((RKGather(source.arg.slot, rank_value, 1, values=(_fp16_bits(rank),)),
                  RKGather(source.arg.slot, output_coordinate, count, values=tuple(_fp16_bits(i) for i in range(count))),
                  RKGather(source.arg.slot, one, count, values=(_fp16_bits(1),)*count),
                  RKGather(source.arg.slot, fill_slot, count, values=(_fp16_bits(fill_value),)*count)))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = list(equality.pre_ops)
  ops.extend((RKEWOp(arg(magnitude), arg(source_values), arg(source_values), source_matrix_lanes, _EW_CFG_ABS),
              RKEWOp(arg(nonzero), arg(magnitude), arg(magnitude), source_matrix_lanes, _EW_CFG[Ops.MAX], compare=True)))
  total = _reduce_rows(ops, [arg(nonzero, row*64) for row in range(source_count)], 1, _EW_CFG[Ops.ADD])
  if rank != 1:
    ops.append(RKEWOp(total, total, arg(rank_value), 1, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  gather_after = len(ops)
  mid = equality.mid_gathers + (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count,
                                         src_kind=RKBufferKind.SCRATCH),)
  ops.extend(equality.mask_ops)
  ops.append(RKEWOp(arg(selected), equality.mask, arg(coordinate_matrix), equality.matrix_lanes,
                    _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True))
  selected_value = _reduce_rows(ops, [arg(selected, row*equality.vector_bytes) for row in range(coordinate_count)],
                                count, _EW_CFG[Ops.ADD])
  ops.extend((RKEWOp(arg(valid_delta), arg(total_vector), arg(output_coordinate), count, _EW_CFG[Ops.SUB]),
              RKEWOp(arg(valid), arg(valid_delta), arg(valid_delta), count, _EW_CFG[Ops.MAX], compare=True),
              RKEWOp(arg(guarded), selected_value, arg(valid), count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(arg(remaining), arg(one), arg(valid), count, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True),
              RKEWOp(arg(fill_part), arg(fill_slot), arg(remaining), count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(arg(result), arg(guarded), arg(fill_part), count, _EW_CFG[Ops.ADD], submit_barrier=True, stateful=True),
              RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), arg(result), arg(int_tiles), count,
                     _EW_CFG[Ops.MAX], stateful=True, int32_output=True)))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<e", 1.0),
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid, gather_after=gather_after)

def _lower_fixed_int32_nonzero(output:RKOutput) -> RKImage|None:
  """Select fixed coordinates of arbitrary INT32 nonzeros through native INT16 byte masks."""
  _, out_param, count, _, _ = output
  if (plan:=_fixed_nonzero_plan(output, dtypes.int, _int32_nonzero_load,
                                lambda value:-32768 <= value <= 32767)) is None: return None
  source, rank, index_param, index_offsets, coordinate_rows, fill_value = plan
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, coordinate_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  _, source_vector_lanes, _ = _stripe_layout(1, source_count)
  source_rows = tuple((lane,) for lane in range(source_count))
  if (source_mask:=_native_int32_nonzero_mask(ops, gathers, scratch, source.arg.slot, source_rows, 1,
                                              source_vector_lanes)) is None: return None
  total = _reduce_rows(ops, [replace(source_mask, addend=source_mask.addend+row*64) for row in range(source_count)],
                       1, _EW_CFG[Ops.ADD], int16=True)
  if rank != 1:
    rank_slot = scratch(2)
    gathers.append(RKGather(source.arg.slot, rank_slot, 1, values=(_int16_bits(rank),)))
    ops.append(RKEWOp(total, total, arg(rank_slot), 1, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
  candidates = tuple((candidate,)*count for candidate in range(coordinate_count))
  if (equal:=_native_int16_byte_mask(ops, gathers, scratch, index_param.arg.slot, index_offsets,
                                     (candidates,), count, vector_lanes)) is None: return None
  gather_after = len(ops)
  total_vector, output_coordinate, zero, one, coordinate_matrix = (scratch(matrix_lanes*2) for _ in range(5))
  mid = (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count, src_kind=RKBufferKind.SCRATCH),)
  gathers.extend(_stripe_gathers(source.arg.slot, coordinate_matrix, count,
    tuple(tuple(_int16_bits(value) for value in row) for row in coordinate_rows), vector_lanes, values=True))
  gathers.extend((RKGather(source.arg.slot, output_coordinate, count, values=tuple(range(count))),
                  RKGather(source.arg.slot, zero, matrix_lanes, values=(0,)*matrix_lanes),
                  RKGather(source.arg.slot, one, matrix_lanes, values=(1,)*matrix_lanes)))
  fill_slot = scratch(count*2)
  gathers.append(RKGather(source.arg.slot, fill_slot, count, values=(_int16_bits(fill_value),)*count))
  valid_delta, positive, valid, remaining = (scratch(count*2) for _ in range(4))
  selected, guarded, fill_part, result = (scratch(matrix_lanes*2) for _ in range(4))
  int16 = dict(int16_input=True, int16_output=True)
  ops.extend((RKEWOp(arg(valid_delta), arg(total_vector), arg(output_coordinate), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(arg(positive), arg(valid_delta), arg(zero), count, _EW_CFG[Ops.MAX], **int16),
              RKEWOp(arg(valid), arg(positive), arg(one), count, _EW_CFG_MIN, **int16),
              RKEWOp(arg(remaining), arg(one), arg(valid), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(arg(selected), equal, arg(coordinate_matrix), matrix_lanes, _EW_CFG[Ops.MUL], **int16)))
  selected_value = _reduce_rows(ops, [arg(selected, row*vector_bytes) for row in range(coordinate_count)],
                                count, _EW_CFG[Ops.ADD], int16=True)
  ops.extend((RKEWOp(arg(guarded), selected_value, arg(valid), count, _EW_CFG[Ops.MUL], **int16),
              RKEWOp(arg(fill_part), arg(fill_slot), arg(remaining), count, _EW_CFG[Ops.MUL], **int16),
              RKEWOp(arg(result), arg(guarded), arg(fill_part), count, _EW_CFG[Ops.ADD], **int16),
              RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), arg(result), arg(result), count, _EW_CFG[Ops.MAX],
                     int16_input=True, int32_output=True)))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=mid, gather_after=gather_after)

def _lower_fixed_fp16_masked_select(output:RKOutput) -> RKImage|None:
  """Lower a bounded fixed-size `x.masked_select(x > 0)` gather/fill without host value inspection."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER: return None
  folded = root.op is Ops.LOAD
  count_info:tuple[UOp, int, float]|None = None
  if root.op is Ops.WHERE and len(root.src) == 3: condition, selected, fill = root.src
  elif root.op is Ops.LOAD and len(root.src) == 3:
    selected, fill = root, root.src[1]
    conditions = [(u, info) for u in root.src[2].toposort() if u.op is Ops.CMPLT and u.src[0].key == out_index.key and
                  (info:=_full_fp16_greater_count(u.src[1], out_index, count)) is not None]
    if len(conditions) != 1: return None
    condition, count_info = conditions[0]
  else: return None
  if (condition.op is not Ops.CMPLT or condition.src[0].key != out_index.key or
      fill.op is not Ops.CONST or fill.dtype.scalar() is not dtypes.half or selected.op is not Ops.LOAD or
      selected.dtype.scalar() is not dtypes.half or len(selected.src) != 3 or selected.src[0].op is not Ops.INDEX or
      selected.src[1].op is not Ops.CONST or (selected.src[1].key != fill.key if folded else float(selected.src[1].arg) != 0.0)):
    return None
  if count_info is None and (count_info:=_full_fp16_greater_count(condition.src[1], out_index, count)) is None: return None
  assert count_info is not None
  source, _, threshold = count_info
  source_count = int(source.src[0].arg)
  if not 1 <= source_count <= _FP16_EXACT_INTEGER: return None
  data_param, data_index, gate = _root_param(selected.src[0]), selected.src[0].src[1], selected.src[2]
  index_loads = [u for u in data_index.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int]
  if (data_param is None or data_param.arg.slot != source.arg.slot or data_param.dtype.scalar() is not dtypes.half or
      data_param.src[0].op is not Ops.CONST or len(index_loads) != 1 or data_index.key != index_loads[0].key): return None
  index_load = index_loads[0]
  index_param = _root_param(index_load.src[0]) if index_load.src and index_load.src[0].op is Ops.INDEX else None
  gate_int_loads = {u.key for u in gate.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}
  gate_half_params = [_root_param(u.src[0]) for u in gate.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.half]
  if (index_param is None or index_param.dtype.scalar() is not dtypes.int or index_param.src[0].op is not Ops.CONST or
      int(index_param.src[0].arg) != count or gate_int_loads != {index_load.key} or
      any(param is None or param.arg.slot != source.arg.slot for param in gate_half_params) or
      not _bounded_index_gate(gate, index_load, source_count)): return None
  try: index_offsets = _gather_offsets(out_index, index_load.src[0].src[1], None, count)
  except RuntimeError: return None
  if index_offsets != tuple(range(count)): return None

  coordinates = tuple(range(source_count))
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, source_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  one, compact_index, convert_tiles, index_matrix, coordinate_matrix, diff, magnitude, unequal, equal = range(9)
  raw_source = (9, 10); half_source = (11, 12)
  source_values, maximum, negative_maximum = 13, 14, 15
  positive_temps = tuple(range(16, 27))
  total_vector, output_coordinate, valid_delta, valid = range(27, 31)
  selected_bytes, guarded_bytes = (31, 32), (33, 34)
  remaining, fill_bytes, fill_parts, result = 35, (36, 37), (38, 39), (40, 41)
  int_tiles, int_bytes = 42, (43, 44)
  value_matrix = (45, 46)
  threshold_slot = 47
  scratch_sizes = [_scratch_bytes(matrix_lanes)] * 48
  scratch_sizes[compact_index] = _scratch_bytes(count)
  scratch_sizes[convert_tiles] = _int32_tiles_bytes(max(count, source_count))
  for slot in raw_source: scratch_sizes[slot] = source_count*4
  for slot in half_source: scratch_sizes[slot] = _scratch_bytes(source_count)
  scratch_sizes[int_tiles] = _int32_tiles_bytes(count)
  for slot in int_bytes: scratch_sizes[slot] = count*4

  source_rows = tuple((coordinate,)*count for coordinate in coordinates)
  coordinate_rows = tuple((_fp16_bits(coordinate),)*count for coordinate in coordinates)
  gathers = list(_stripe_gathers(source.arg.slot, coordinate_matrix, count, coordinate_rows, vector_lanes, values=True) +
                 _raw_16bit_gathers(source.arg.slot, raw_source, source_count, (tuple(range(source_count)),), source_count))
  gathers.extend(_stripe_gathers(source.arg.slot, source_values, 1, ((lane,) for lane in range(source_count)), vector_lanes))
  max_bits, negmax_bits = _fp16_bits(65504), _fp16_bits(-65504)
  gathers.extend((RKGather(source.arg.slot, maximum, matrix_lanes, values=(max_bits,)*matrix_lanes),
                  RKGather(source.arg.slot, negative_maximum, matrix_lanes, values=(negmax_bits,)*matrix_lanes),
                  RKGather(source.arg.slot, threshold_slot, matrix_lanes, values=(_fp16_bits(threshold),)*matrix_lanes)))
  coordinate_bits = tuple(_fp16_bits(lane) for lane in range(count))
  gathers.append(RKGather(source.arg.slot, output_coordinate, count, values=coordinate_bits))
  fill_bits = _fp16_bits(fill.arg)
  for byte,slot in enumerate(fill_bytes):
    byte_bits = _fp16_bits((fill_bits >> (byte*8)) & 0xff)
    gathers.append(RKGather(source.arg.slot, slot, count, values=(byte_bits,)*count))

  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = [RKEWOp(arg(compact_index), RKArg(RKBufferKind.ARG, index_param.arg.slot), arg(convert_tiles), count,
                              _EW_CFG[Ops.MAX], int32_input=True)]
  _append_byte_conversions(ops, arg, raw_source, half_source, convert_tiles, source_count)
  ops.append(RKEWOp(arg(source_values), arg(source_values), arg(threshold_slot), matrix_lanes, _EW_CFG[Ops.SUB]))
  positive = _ew_ieee_positive_mask(ops, arg, source_values, (0, maximum, negative_maximum), positive_temps, matrix_lanes)
  total = _reduce_rows(ops, [RKArg(positive.kind, positive.index, positive.addend+lane*vector_bytes)
                             for lane in range(source_count)],
                       1, _EW_CFG[Ops.ADD])
  gather_after = len(ops)
  mid_gathers = tuple(replace(gather, src_kind=RKBufferKind.SCRATCH) for gather in
                      (_stripe_gathers(compact_index, index_matrix, count, (index_offsets,)*source_count, vector_lanes) +
                       _stripe_gathers(half_source[0], value_matrix[0], count, source_rows, vector_lanes) +
                       _stripe_gathers(half_source[1], value_matrix[1], count, source_rows, vector_lanes))) + \
                (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count, src_kind=RKBufferKind.SCRATCH),)
  equal_arg = _ew_integer_eq_mask(ops, arg, index_matrix, coordinate_matrix, (diff, magnitude, unequal, equal), one, matrix_lanes)
  ops.extend((RKEWOp(arg(valid_delta), arg(total_vector), arg(output_coordinate), count, _EW_CFG[Ops.SUB]),
              RKEWOp(arg(valid), arg(valid_delta), arg(valid_delta), count, _EW_CFG[Ops.MAX], compare=True)))
  ops.extend(RKEWOp(arg(dst), arg(src), equal_arg, matrix_lanes, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)
             for src,dst in zip(value_matrix, selected_bytes))
  selected_reduced = tuple(_reduce_rows(ops, [arg(slot, row*vector_bytes) for row in range(source_count)],
                                        count, _EW_CFG[Ops.ADD]) for slot in selected_bytes)
  ops.extend(RKEWOp(arg(dst), value, arg(valid), count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)
             for value,dst in zip(selected_reduced, guarded_bytes))
  ops.append(RKEWOp(arg(remaining), arg(0), arg(valid), count, _EW_CFG[Ops.SUB], submit_barrier=True, stateful=True))
  ops.extend(RKEWOp(arg(dst), arg(src), arg(remaining), count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True)
             for src,dst in zip(fill_bytes, fill_parts))
  ops.extend(RKEWOp(arg(dst), arg(selected_slot), arg(fill_slot), count, _EW_CFG[Ops.ADD], submit_barrier=True, stateful=True)
             for selected_slot,fill_slot,dst in zip(guarded_bytes, fill_parts, result))
  post_gathers = _raw_16bit_output(ops, arg, (arg(result[0]), arg(result[1])), count, int_tiles, int_bytes,
                                   out_param.arg.slot)
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), struct.pack("<e", 1.0),
                 gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid_gathers, gather_after=gather_after,
                 post_gathers=post_gathers)

def _lower_fixed_integer_masked_select(output:RKOutput, dtype:DType=dtypes.int) -> RKImage|None:
  """Select arbitrary integer values by exact byte equality under a complete external bool count."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or root.op is not Ops.WHERE or len(root.src) != 3: return None
  condition, selected, fill = root.src
  if (condition.op is not Ops.CMPLT or condition.src[0].key != out_index.key or
      fill.op is not Ops.CONST or fill.dtype.scalar() is not dtype or
      selected.op is not Ops.LOAD or selected.dtype.scalar() is not dtype or len(selected.src) != 3 or
      selected.src[0].op is not Ops.INDEX or selected.src[1].op is not Ops.CONST or int(selected.src[1].arg) != 0): return None
  if (mask_info:=_full_bool_count(condition.src[1], out_index, count)) is None or mask_info[1] != 1: return None
  mask_param = mask_info[0]
  source_count = int(mask_param.src[0].arg)
  data_param, data_index, gate = _root_param(selected.src[0]), selected.src[0].src[1], selected.src[2]
  index_loads = [u for u in data_index.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int]
  if (data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or
      int(data_param.src[0].arg) != source_count or len(index_loads) != 1 or data_index.key != index_loads[0].key): return None
  index_load = index_loads[0]
  index_param = _root_param(index_load.src[0]) if index_load.src and index_load.src[0].op is Ops.INDEX else None
  if (index_param is None or index_param.dtype.scalar() is not dtypes.int or index_param.src[0].op is not Ops.CONST or
      int(index_param.src[0].arg) != count or not _bounded_index_gate(gate, index_load, source_count) or
      {u.key for u in gate.toposort() if u.op is Ops.LOAD} != {index_load.key}): return None
  try: index_offsets = _gather_offsets(out_index, index_load.src[0].src[1], None, count)
  except RuntimeError: return None
  if index_offsets != tuple(range(count)): return None
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, source_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(size, 64)); return len(scratch_sizes)-1
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  bool_values = scratch(source_count*64)
  gathers.extend(RKGather(mask_param.arg.slot, bool_values, 1, offsets=(lane,), dst_addend=lane*64, dst_stride=2, itemsize=1)
                 for lane in range(source_count))
  total = _reduce_rows(ops, [arg(bool_values, lane*64) for lane in range(source_count)], 1, _EW_CFG[Ops.ADD], int16=True)
  coordinates = tuple((coordinate,)*count for coordinate in range(source_count))
  if (equal:=_native_int16_byte_mask(ops, gathers, scratch, index_param.arg.slot, index_offsets,
                                     (coordinates,), count, vector_lanes)) is None: return None
  gather_after = len(ops)
  total_vector, output_coordinate, zero, one = (scratch(count*2) for _ in range(4))
  mid = (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count, src_kind=RKBufferKind.SCRATCH),)
  gathers.extend((RKGather(mask_param.arg.slot, output_coordinate, count, values=tuple(range(count)), itemsize=2),
                  RKGather(mask_param.arg.slot, zero, count, values=(0,)*count, itemsize=2),
                  RKGather(mask_param.arg.slot, one, count, values=(1,)*count, itemsize=2)))
  itemsize = dtype.itemsize
  raw_values = tuple(scratch(matrix_lanes*2) for _ in range(itemsize))
  for byte,slot in enumerate(raw_values):
    gathers.extend(RKGather(data_param.arg.slot, slot, count, offsets=(coordinate*itemsize+byte,)*count,
                           dst_addend=coordinate*vector_bytes, dst_stride=2, itemsize=1)
                   for coordinate in range(source_count))
  fill_bits = int(fill.arg) & ((1 << (itemsize*8))-1)
  fill_slots = tuple(scratch(count*2) for _ in range(itemsize))
  for byte,slot in enumerate(fill_slots):
    gathers.append(RKGather(mask_param.arg.slot, slot, count, values=((fill_bits >> (byte*8)) & 0xff,)*count, itemsize=2))

  valid_delta, positive, valid, remaining = (scratch(count*2) for _ in range(4))
  int16 = dict(int16_input=True, int16_output=True)
  ops.extend((RKEWOp(arg(valid_delta), arg(total_vector), arg(output_coordinate), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(arg(positive), arg(valid_delta), arg(zero), count, _EW_CFG[Ops.MAX], **int16),
              RKEWOp(arg(valid), arg(positive), arg(one), count, _EW_CFG_MIN, **int16),
              RKEWOp(arg(remaining), arg(one), arg(valid), count, _EW_CFG[Ops.SUB], **int16)))
  results:list[RKArg] = []
  for value,fill_slot in zip(raw_values, fill_slots):
    selected_matrix, guarded, fill_part, result = (scratch(matrix_lanes*2), scratch(count*2), scratch(count*2), scratch(count*2))
    ops.append(RKEWOp(arg(selected_matrix), arg(value), equal, matrix_lanes, _EW_CFG[Ops.MUL], **int16))
    selected_byte = _reduce_rows(ops, [arg(selected_matrix, row*vector_bytes) for row in range(source_count)], count,
                                 _EW_CFG[Ops.ADD], int16=True)
    ops.extend((RKEWOp(arg(guarded), selected_byte, arg(valid), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(arg(fill_part), arg(fill_slot), arg(remaining), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(arg(result), arg(guarded), arg(fill_part), count, _EW_CFG[Ops.ADD], **int16)))
    results.append(arg(result))
  post = tuple(RKGather(value.index, out_param.arg.slot, count,
                        offsets=tuple(value.addend+lane*2 for lane in range(count)), dst_stride=itemsize, dst_addend=byte,
                        dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
               for byte,value in enumerate(results))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=mid, gather_after=gather_after, post_gathers=post)

def _lower_dynamic_16bit_gather(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
  """Recognize a bounds-masked 16-bit load addressed by one dynamic INT32 index."""
  _, out_param, count, out_index, load = output
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or
      load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or int(load.src[1].arg) != 0): return None
  data_param, data_index, gate = _root_param(load.src[0]), load.src[0].src[1], load.src[2]
  index_loads = [u for u in data_index.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int]
  if (data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or
      len(index_loads) != 1): return None
  index_load = index_loads[0]
  index_param = _root_param(index_load.src[0]) if index_load.src and index_load.src[0].op is Ops.INDEX else None
  bool_loads = [u for u in gate.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.bool]
  if (index_param is None or index_param.dtype.scalar() is not dtypes.int or index_param.src[0].op is not Ops.CONST or
      len(bool_loads) > 1 or {u.key for u in gate.toposort() if u.op is Ops.LOAD} != {index_load.key, *(u.key for u in bool_loads)} or
      gate.op is not Ops.AND): return None

  bounded = data_index if data_index.op is Ops.WHERE else index_load
  limits = [int(u.src[1].arg) for u in gate.toposort() if u.op is Ops.CMPLT and u.src[0].key == bounded.key and
            u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0]
  if len(set(limits)) != 1 or not _bounded_index_gate(gate, bounded, limit:=limits[0]): return None
  coordinates = tuple(range(limit))
  if bounded is not index_load:
    if (normalized:=_negative_normalized_index(bounded)) is None or normalized[0].key != index_load.key or normalized[1] != limit: return None
    coordinates += tuple(range(-limit, 0))
  try:
    index_offsets = _gather_offsets(out_index, index_load.src[0].src[1], None, count)
    plans = tuple(_gather_plan(data_param.arg.slot, 0, out_index,
      data_index.substitute({index_load:index_load.const_like(candidate)}), None, count) for candidate in coordinates)
  except RuntimeError: return None
  data_count, index_count = int(data_param.src[0].arg), int(index_param.src[0].arg)
  if (any(not 0 <= offset < index_count for offset in index_offsets) or
      any(not 0 <= offset < data_count for plan in plans for offset in _plan_offsets(plan))): return None
  external_gate:RKDynamicIndex|None = None
  if bool_loads:
    bool_load = bool_loads[0]
    bool_param = _root_param(bool_load.src[0]) if len(bool_load.src) == 1 and bool_load.src[0].op is Ops.INDEX else None
    if bool_load not in _flatten_binary(gate, Ops.AND) or bool_param is None or bool_param.dtype.scalar() is not dtypes.bool or \
       bool_param.src[0].op is not Ops.CONST: return None
    try: bool_offsets = _gather_offsets(out_index, bool_load.src[0].src[1], None, count)
    except RuntimeError: return None
    bool_count = int(bool_param.src[0].arg)
    if any(not 0 <= offset < bool_count for offset in bool_offsets): return None
    external_gate = (bool_param.arg.slot, bool_count, bool_offsets)
  return _dynamic_16bit_gather_image(out_param.arg.slot, count, ((index_param.arg.slot, index_count, index_offsets),), plans,
                                     (coordinates,), gate=external_gate)

def _lower_unrolled_multi_16bit_fancy_index(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
  """Collapse Tinygrad's unrolled multi-index selection into compact exact INT32 equality rows."""
  _, out_param, count, out_index, root = output
  terms = _flatten_binary(root, Ops.ADD)
  if count <= 0 or len(terms) < 2: return None
  normalized = tuple((node, *parsed) for node in root.toposort() if (parsed:=_negative_normalized_index(node)) is not None)
  normalized_by_load = {load.key:(node, extent) for node,load,extent in normalized}
  dynamic_loads = tuple({u.key:u for u in root.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}.values())
  if not dynamic_loads or len(normalized_by_load) != len(normalized): return None
  direct_domains:dict[UOp, set[int]] = {load:set() for load in dynamic_loads if load.key not in normalized_by_load}
  for predicate in root.toposort():
    if (pair:=_equality_pair(predicate)) is None: continue
    for value,constant in (pair, pair[::-1]):
      if value in direct_domains and constant.op is Ops.CONST and constant.dtype.scalar() is dtypes.int:
        direct_domains[value].add(int(constant.arg))
  axes:list[tuple[UOp, UOp, int, bool]] = []
  for load in dynamic_loads:
    if load.key in normalized_by_load: node, extent, wrapped = *normalized_by_load[load.key], True
    else:
      domain = direct_domains[load]
      if not domain or domain != set(range(max(domain)+1)): return None
      node, extent, wrapped = load, max(domain)+1, False
    axes.append((node, load, extent, wrapped))
  axis_roots = {node.key:(axis, extent) for axis,(node,_,extent,_) in enumerate(axes)}

  candidates:dict[tuple[int, ...], UOp] = {}
  for term in terms:
    if term.op is not Ops.WHERE or term.dtype.scalar() is not dtype: return None
    loads = [arm for arm in term.src[1:] if arm.op is Ops.LOAD and arm.dtype.scalar() is dtype and
             len(arm.src) == 1 and arm.src[0].op is Ops.INDEX]
    zeros = [arm for arm in term.src[1:] if arm.op is Ops.CONST and arm.dtype.scalar() is dtype and int(arm.arg) == 0]
    if len(loads) != 1 or len(zeros) != 1 or term.src[1] is not loads[0]: return None
    candidate_coordinates:list[int|None] = [None]*len(axes)
    predicates = _flatten_binary(term.src[0], Ops.AND)
    if len(predicates) != len(axes): return None
    for predicate in predicates:
      if (pair:=_equality_pair(predicate)) is None: return None
      matches = [(axis_roots[value.key], constant) for value,constant in (pair, pair[::-1])
                 if value.key in axis_roots and constant.op is Ops.CONST and constant.dtype.scalar() is dtypes.int]
      if len(matches) != 1: return None
      (axis,extent),constant = matches[0]; coordinate = int(constant.arg)
      if candidate_coordinates[axis] is not None or not 0 <= coordinate < extent: return None
      candidate_coordinates[axis] = coordinate
    if any(coordinate is None for coordinate in candidate_coordinates): return None
    key = tuple(coordinate for coordinate in candidate_coordinates if coordinate is not None)
    if key in candidates: return None
    candidates[key] = loads[0]

  combinations:tuple[tuple[int, ...], ...] = ((),)
  for _,_,extent,_ in axes: combinations = tuple(prefix+(value,) for prefix in combinations for value in range(extent))
  if set(candidates) != set(combinations): return None
  candidate_loads = tuple(candidates[values] for values in combinations)
  params = tuple(_root_param(load.src[0]) for load in candidate_loads)
  index_params = tuple(_root_param(load.src[0]) for load in dynamic_loads)
  if (any(param is None or param.dtype.scalar() is not dtype or
      param.src[0].op is not Ops.CONST for param in params) or len({param.arg.slot for param in params if param is not None}) != 1 or
      any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in index_params)):
    return None
  data_param = next(param for param in params if param is not None)
  try:
    plans = tuple(_gather_plan(data_param.arg.slot, 0, out_index, load.src[0].src[1], None, count) for load in candidate_loads)
  except RuntimeError: return None
  concrete_indices = tuple(param for param in index_params if param is not None)
  try: offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in dynamic_loads)
  except RuntimeError: return None
  data_count = int(data_param.src[0].arg)
  index_counts = tuple(int(param.src[0].arg) for param in concrete_indices)
  if (any(not 0 <= offset < size for row,size in zip(offsets, index_counts) for offset in row) or
      any(not 0 <= offset < data_count for plan in plans for offset in _plan_offsets(plan))): return None
  indices = tuple((param.arg.slot, size, row) for param,size,row in zip(concrete_indices, index_counts, offsets))
  coordinates = tuple(tuple(values[axis] for values in combinations) for axis in range(len(axes)))
  alternates = tuple((tuple(value-extent for value in axis),) if wrapped else ()
                     for axis,(_,_,extent,wrapped) in zip(coordinates, axes))
  return _dynamic_16bit_gather_image(out_param.arg.slot, count, indices, plans, coordinates, alternates)

def _lower_multi_16bit_fancy_index(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
  """Lower a bounded 16-bit load addressed by positive-only or negative-normalized INT32 tensors."""
  _, out_param, count, out_index, load = output
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or
      load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or int(load.src[1].arg) != 0): return None
  data_param, data_index, gate = _root_param(load.src[0]), load.src[0].src[1], load.src[2]
  normalized = tuple((u, *parsed) for u in data_index.toposort() if (parsed:=_negative_normalized_index(u)) is not None)
  normalized_by_load = {load.key:(root, extent) for root,load,extent in normalized}
  if len(normalized_by_load) != len(normalized): return None
  loads = tuple({u.key:u for u in data_index.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}.values())
  axes:list[tuple[UOp, UOp, int, bool]] = []
  for dynamic in loads:
    if dynamic.key in normalized_by_load:
      root, extent = normalized_by_load[dynamic.key]; wrapped = True
    else:
      limits = {int(u.src[1].arg) for u in gate.toposort() if u.op is Ops.CMPLT and u.src[0].key == dynamic.key and
                u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0}
      if len(limits) != 1: return None
      root, extent, wrapped = dynamic, next(iter(limits)), False
    if not _bounded_index_gate(gate, root, extent): return None
    axes.append((root, dynamic, extent, wrapped))
  if (data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or
      not axes or len({load.key for load in loads}) != len(loads) or
      {u.key for u in gate.toposort() if u.op is Ops.LOAD} != {u.key for u in loads} or
      any(root.key not in {u.key for u in data_index.toposort()} for root,_,_,_ in axes)): return None
  params = tuple(_root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None for load in loads)
  if any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in params): return None
  concrete = tuple(param for param in params if param is not None)
  options = tuple(tuple(range(extent)) for _,_,extent,_ in axes)
  combinations:tuple[tuple[int, ...], ...] = ((),)
  for axis in options:
    combinations = tuple(prefix+(value,) for prefix in combinations for value in axis)
  try:
    offsets = tuple(_gather_offsets(out_index, load.src[0].src[1], None, count) for load in loads)
    plans = tuple(_gather_plan(data_param.arg.slot, 0, out_index, data_index.substitute(mapping), gate.substitute(mapping), count)
                  for values in combinations
                  for mapping in ({load:load.const_like(value) for load,value in zip(loads, values)},))
  except RuntimeError: return None
  data_count = int(data_param.src[0].arg)
  index_counts = tuple(int(param.src[0].arg) for param in concrete)
  if (any(not 0 <= offset < index_count for axis_offsets,index_count in zip(offsets, index_counts) for offset in axis_offsets) or
      any(not 0 <= offset < data_count for plan in plans for offset in _plan_offsets(plan))): return None
  indices = tuple((param.arg.slot, index_count, axis_offsets)
                  for param,index_count,axis_offsets in zip(concrete, index_counts, offsets))
  coordinates = tuple(tuple(values[axis] for values in combinations) for axis in range(len(loads)))
  alternates = tuple((tuple(value-extent for value in axis),) if wrapped else ()
                     for axis,(_,_,extent,wrapped) in zip(coordinates, axes))
  return _dynamic_16bit_gather_image(out_param.arg.slot, count, indices, plans, coordinates, alternates)

def _lower_dynamic_16bit_scatter(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
  """Lower bounded last-wins Scatter for 16-bit values and one external INT32 index buffer."""
  _, out_param, count, out_index, value = output
  if count <= 0 or value.op is not Ops.WHERE: return None
  int_loads = tuple({u.key:u for u in value.toposort() if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int and
                     len(u.src) == 1 and u.src[0].op is Ops.INDEX}.values())
  if not 1 <= len(int_loads) <= 8: return None
  direct:dict[UOp, tuple[UOp, UOp]] = {}
  for comparison in (u for u in value.toposort() if u.op is Ops.CMPNE and len(u.src) == 2):
    for load,coordinate in (comparison.src, comparison.src[::-1]):
      if load in int_loads and _is_static_expr(coordinate): direct[comparison] = load, coordinate
  if len(direct) != len(int_loads) or {load for load,_ in direct.values()} != set(int_loads): return None
  entries:list[tuple[tuple[int, ...], UOp, UOp, tuple[int, ...]]] = []
  try:
    for comparison,(load,coordinate) in direct.items():
      offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
      entries.append((offsets, comparison, load, _static_int_vector(out_index, coordinate, count)))
  except RuntimeError: return None
  entries.sort(key=lambda entry:entry[0])
  params = tuple(_root_param(load.src[0]) for _,_,load,_ in entries)
  if (any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in params) or
      len({param.arg.slot for param in params if param is not None}) != 1): return None
  index_param = next(param for param in params if param is not None)
  index_count = int(index_param.src[0].arg)
  index_offset_rows = tuple(offsets for offsets,_,_,_ in entries)
  if (sorted({offset for row in index_offset_rows for offset in row}) != list(range(index_count)) or
      any(any(lhs[lane] >= rhs[lane] for lhs,rhs in zip(index_offset_rows, index_offset_rows[1:])) for lane in range(count))): return None
  comparison_candidates = {comparison:candidate for candidate,(_,comparison,_,_) in enumerate(entries)}

  def predicate(root:UOp, matches:int) -> bool:
    if root in comparison_candidates: return not bool(matches & (1 << comparison_candidates[root]))
    if root.op is Ops.CONST and root.dtype.scalar() is dtypes.bool: return bool(root.arg)
    if root.op not in (Ops.CMPNE, Ops.AND, Ops.OR) or len(root.src) != 2: raise RuntimeError("RKPLAN_REJECT:scatter_predicate")
    lhs, rhs = predicate(root.src[0], matches), predicate(root.src[1], matches)
    return lhs != rhs if root.op is Ops.CMPNE else lhs and rhs if root.op is Ops.AND else lhs or rhs

  def selected(root:UOp, matches:int) -> UOp:
    if root.op is Ops.LOAD and root.dtype.scalar() is dtype and len(root.src) == 1 and root.src[0].op is Ops.INDEX: return root
    if root.op is not Ops.WHERE or len(root.src) != 3: raise RuntimeError("RKPLAN_REJECT:scatter_selection")
    return selected(root.src[1] if predicate(root.src[0], matches) else root.src[2], matches)

  try:
    base = selected(value, 0)
    sources = tuple(selected(value, 1 << candidate) for candidate in range(len(entries)))
    for matches in range(1 << len(entries)):
      expected = base if matches == 0 else sources[matches.bit_length()-1]
      if selected(value, matches).key != expected.key: return None
  except RuntimeError: return None
  base_param = _root_param(base.src[0])
  source_params = tuple(_root_param(source.src[0]) for source in sources)
  if (base_param is None or base_param.dtype.scalar() is not dtype or base_param.src[0].op is not Ops.CONST or
      any(param is None or param.dtype.scalar() is not dtype or param.src[0].op is not Ops.CONST for param in source_params) or
      len({param.arg.slot for param in source_params if param is not None}) != 1): return None
  source_param = next(param for param in source_params if param is not None)
  try:
    base_offsets = _gather_offsets(out_index, base.src[0].src[1], None, count)
    source_offset_rows = tuple(_gather_offsets(out_index, source.src[0].src[1], None, count) for source in sources)
  except RuntimeError: return None
  base_count, source_count = int(base_param.src[0].arg), int(source_param.src[0].arg)
  if (any(not 0 <= offset < base_count for offset in base_offsets) or
      any(not 0 <= offset < source_count for offsets in source_offset_rows for offset in offsets)): return None
  coordinate_rows = tuple(coordinates for _,_,_,coordinates in entries)
  return _dynamic_16bit_scatter_image(out_param.arg.slot, count, index_param.arg.slot, index_count,
                                      index_offset_rows, coordinate_rows, source_param.arg.slot, source_offset_rows,
                                      base_param.arg.slot, base_offsets)

def _affine_load_offset(root:UOp, load:UOp) -> int|None:
  """Return the constant in an integer `load + constant` expression."""
  root = _strip_cast(root)
  if root is load: return 0
  if root.op is Ops.ADD and len(root.src) == 2:
    for expression,constant in ((root.src[0], root.src[1]), (root.src[1], root.src[0])):
      if constant.op is Ops.CONST and (offset:=_affine_load_offset(expression, load)) is not None: return offset+int(constant.arg)
  if root.op is Ops.SUB and len(root.src) == 2 and root.src[1].op is Ops.CONST and \
     (offset:=_affine_load_offset(root.src[0], load)) is not None: return offset-int(root.src[1].arg)
  return None

def _lower_unrolled_max_unpool(output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Scatter bounded unrolled MaxUnpool lanes through exact DPU index equality."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  value_dtype = dtypes.int16 if native_int16 else dtypes.half
  parsed:list[tuple[UOp, UOp, UOp, int]] = []
  for term in _flatten_binary(value, Ops.ADD):
    where = _strip_cast(term)
    if (where.op is not Ops.WHERE or _strip_cast(where.src[0]).op is not Ops.CMPNE or
        where.src[1].op is not Ops.CONST or float(where.src[1].arg) != 0.0): return None
    condition, selected_load = _strip_cast(where.src[0]), _strip_cast(where.src[2])
    index_loads = [x for x in condition.toposort() if x.op is Ops.LOAD and x.dtype.scalar() is dtypes.int]
    coordinate_exprs = [x for x in condition.src if _is_static_expr(x)]
    if (len(index_loads) != 1 or len(coordinate_exprs) != 1 or selected_load.op is not Ops.LOAD or
        selected_load.dtype.scalar() is not value_dtype or selected_load.src[0].op is not Ops.INDEX or
        index_loads[0].src[0].op is not Ops.INDEX or selected_load.src[0].src[1].key != index_loads[0].src[0].src[1].key): return None
    dynamic = next((x for x in condition.src if index_loads[0] in x.toposort()), None)
    if dynamic is None or (index_offset:=_affine_load_offset(dynamic, index_loads[0])) is None: return None
    parsed.append((index_loads[0], selected_load, coordinate_exprs[0], index_offset))
  pooled = len(parsed)
  if not 1 <= pooled <= 255 or len({offset for *_,offset in parsed}) != 1: return None
  index_params = [_root_param(index.src[0]) for index,_,_,_ in parsed]
  value_params = [_root_param(value.src[0]) for _,value,_,_ in parsed]
  if (any(param is None or param.src[0].op is not Ops.CONST for param in (*index_params,*value_params)) or
      len({param.arg.slot for param in index_params if param is not None}) != 1 or
      len({param.arg.slot for param in value_params if param is not None}) != 1): return None
  index_param, value_param = index_params[0], value_params[0]
  assert index_param is not None and value_param is not None
  source_count = int(index_param.src[0].arg)
  if (index_param.dtype.scalar() is not dtypes.int or value_param.dtype.scalar() is not value_dtype or
      int(value_param.src[0].arg) != source_count or source_count % pooled): return None
  planes = source_count//pooled
  if not planes or count % planes: return None
  out_spatial = count//planes
  try:
    rows = [(_gather_offsets(out_index, index.src[0].src[1], None, count),
             _gather_offsets(out_index, selected.src[0].src[1], None, count),
             _static_int_vector(out_index, coordinate, count)) for index,selected,coordinate,_ in parsed]
  except RuntimeError: return None
  if any(indexes != values or coords != tuple(lane%out_spatial for lane in range(count)) for indexes,values,coords in rows): return None
  rows.sort(key=lambda row:row[0])
  if any(sorted(indexes[lane] for indexes,_,_ in rows) != list(range(lane//out_spatial*pooled, (lane//out_spatial+1)*pooled))
         for lane in range(count)): return None
  plans = tuple(RKGather(value_param.arg.slot, 0, count, offsets=indexes) for indexes,_,_ in rows)
  return (_int16_max_unpool_image(out_param.arg.slot, count, out_spatial, index_param.arg.slot, plans, parsed[0][3]) if native_int16 else
          _max_unpool_image(out_param.arg.slot, count, out_spatial, source_count, index_param.arg.slot, plans, parsed[0][3]))

def _lower_loop_max_unpool(uops:list[UOp], output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Recognize MaxUnpool's plane/coordinate/candidate loop and emit the shared exact scatter image."""
  store, out_param, count, out_index, value = output
  value_dtype = dtypes.int16 if native_int16 else dtypes.half
  ranges = [u for u in uops if u.op is Ops.RANGE]
  reduce_ranges = [u for u in ranges if u.arg[1] is AxisType.REDUCE]
  weak_ranges = [u for u in ranges if u.arg[1] is AxisType.WEAK]
  if count <= 0 or len(reduce_ranges) != 1 or len(weak_ranges) != 2: return None
  reduce_range = reduce_ranges[0]
  if reduce_range.src[0].op is not Ops.CONST or not 2 <= (pooled:=int(reduce_range.src[0].arg)) <= 255: return None

  wheres = [u for u in uops if u.op is Ops.WHERE]
  if len(wheres) != 1: return None
  where = wheres[0]
  if (where.src[0].op is not Ops.CMPNE or where.src[1].op is not Ops.CONST or float(where.src[1].arg) != 0.0 or
      (selected:=_strip_cast(where.src[2])).op is not Ops.LOAD or selected.dtype.scalar() is not value_dtype): return None
  index_loads = [x for x in where.src[0].src if x.op is Ops.LOAD and x.dtype.scalar() is dtypes.int]
  coordinate_exprs = [x for x in where.src[0].src if _is_static_expr(x)]
  if len(index_loads) != 1 or len(coordinate_exprs) != 1: return None
  index_load, coordinate_expr = index_loads[0], coordinate_exprs[0]
  if (index_load.src[0].op is not Ops.INDEX or selected.src[0].op is not Ops.INDEX or
      index_load.src[0].src[1].key != selected.src[0].src[1].key): return None
  index_param, value_param = _root_param(index_load.src[0]), _root_param(selected.src[0])
  if (index_param is None or value_param is None or index_param.src[0].op is not Ops.CONST or value_param.src[0].op is not Ops.CONST or
      index_param.dtype.scalar() is not dtypes.int or value_param.dtype.scalar() is not value_dtype): return None
  source_count = int(index_param.src[0].arg)
  if int(value_param.src[0].arg) != source_count or source_count % pooled: return None
  planes = source_count//pooled
  if not planes or count % planes: return None
  out_spatial = count//planes
  plane_ranges = [r for r in weak_ranges if r.src[0].op is Ops.CONST and int(r.src[0].arg) == planes]
  coordinate_ranges = [r for r in weak_ranges if r.src[0].op is Ops.CONST and int(r.src[0].arg) == out_spatial]
  if len(plane_ranges) != 1 or len(coordinate_ranges) != 1 or plane_ranges[0] is coordinate_ranges[0]: return None
  plane_range, coordinate_range = plane_ranges[0], coordinate_ranges[0]

  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  initial = [u for u in local_stores if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0]
  updates = [u for u in local_stores if where in u.src[1].toposort()]
  out_load = _local_load(value)
  if len(local_stores) != 2 or len(initial) != 1 or len(updates) != 1 or out_load is None: return None
  update_terms = _flatten_binary(updates[0].src[1], Ops.ADD)
  accumulators = [term for term in update_terms if _local_load(term) is not None]
  if len(update_terms) != 2 or len(accumulators) != 1: return None
  buffers = [{x for x in node.toposort() if x.op is Ops.BUFFER} for node in (initial[0].src[0], updates[0].src[0], accumulators[0], out_load)]
  if any(len(group) != 1 for group in buffers) or len(set.union(*buffers)) != 1: return None

  try:
    if tuple(_eval_int(out_index, {plane_range:p, coordinate_range:c}) for p in range(planes)
             for c in range(out_spatial)) != tuple(range(count)): return None
    if any(_eval_int(coordinate_expr, {plane_range:p, coordinate_range:c, reduce_range:r}) != c
           for p in range(planes) for c in range(out_spatial) for r in (0, pooled-1)): return None
    source_index = index_load.src[0].src[1]
    if any(_eval_int(source_index, {plane_range:p, coordinate_range:c, reduce_range:r}) != p*pooled+r
           for p in range(planes) for c in (0, out_spatial-1) for r in range(pooled)): return None
  except RuntimeError: return None
  plans = tuple(RKGather(value_param.arg.slot, 0, count, base=candidate, axes=((out_spatial, planes, pooled),))
                for candidate in range(pooled))
  return (_int16_max_unpool_image(out_param.arg.slot, count, out_spatial, index_param.arg.slot, plans) if native_int16 else
          _max_unpool_image(out_param.arg.slot, count, out_spatial, source_count, index_param.arg.slot, plans))

@dataclass(frozen=True)
class RKArgMatch:
  source_slot:int; source_count:int; extrema:UOp; candidates:tuple[tuple[UOp, UOp, bool], ...]; extrema_plans:tuple[RKGather, ...]|None = None

def _lower_unrolled_arg_extrema(output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Share first-tie validation and gather packing across fused and split ArgMax/ArgMin graphs."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  nodes = value.toposort()
  dtype, candidate = (dtypes.int16, _int16_candidate) if native_int16 else (dtypes.half, _half_candidate)

  def fused() -> RKArgMatch|None:
    roots:list[tuple[UOp, list[UOp], list[tuple[UOp, bool]]]] = []
    for root in nodes:
      if root.op is not Ops.MAX or root.dtype.scalar() is not dtype: continue
      leaves = _flatten_binary(root, Ops.MAX); parsed_leaves = [candidate(leaf) for leaf in leaves]
      if all(x is not None for x in parsed_leaves): roots.append((root, leaves, [x for x in parsed_leaves if x is not None]))
    if not roots: return None
    extrema, exprs, parsed = max(roots, key=lambda x:len(x[1]))
    loads = [load for load,_ in parsed]
    if not loads or loads[0].src[0].op is not Ops.INDEX or (source:=_root_param(loads[0].src[0])) is None or source.src[0].op is not Ops.CONST:
      return None
    return RKArgMatch(source.arg.slot, int(source.src[0].arg), extrema,
                      tuple((expr, load, negated) for expr,(load,negated) in zip(exprs, parsed)))

  def split() -> RKArgMatch|None:
    by_slot = _param_load_groups(nodes, dtype)
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
      if (parsed:=candidate(expr)) is not None: candidates.append((expr, *parsed))
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
                                 negated, [window-1]*count, first_tie=True, negate_extrema=negated and match.extrema_plans is None,
                                 native_int16=native_int16)

def _lower_loop_arg_extrema_index(uops:list[UOp], output:RKOutput, native_int16:bool=False) -> RKImage|None:
  """Lower the two-register-loop graph used by a padded global ArgMax/ArgMin."""
  _, out_param, count, out_index, value = output
  if count != 1 or _index_ranges(out_index): return None
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if len(ranges) != 2 or any(r.src[0].op is not Ops.CONST for r in ranges): return None
  first_range, second_range = ranges
  window = int(first_range.src[0].arg)
  if not 2 <= window <= 2048 or int(second_range.src[0].arg) != window: return None
  dtype, candidate = (dtypes.int16, _int16_candidate) if native_int16 else (dtypes.half, _half_candidate)
  input_params = [u for u in uops if u.op is Ops.PARAM and u.dtype.scalar() is dtype and
                  u.src[0].op is Ops.CONST and int(u.src[0].arg) == window]
  if len(input_params) != 1: return None
  source = input_params[0]
  local_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  initial_value = [u for u in local_stores if u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() is dtype and
                   (int(u.src[1].arg) == -32768 if native_int16 else math.isinf(float(u.src[1].arg)) and float(u.src[1].arg) < 0)]
  initial_int = [u for u in local_stores if u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() is dtypes.int and
                 int(u.src[1].arg) == -(1 << 31)]
  value_updates = [u for u in local_stores if u.src[1].op is Ops.MAX and u.src[1].dtype.scalar() is dtype and
                   first_range in u.toposort()]
  int_updates = [u for u in local_stores if u.src[1].op is Ops.MAX and u.src[1].dtype.scalar() is dtypes.int and
                 second_range in u.toposort()]
  if not all(len(x) == 1 for x in (initial_value, initial_int, value_updates, int_updates)) or len(local_stores) != 4: return None

  def global_candidate(exprs:list[UOp], reduce_range:UOp) -> tuple[UOp, bool]|None:
    parsed = [(expr, match) for expr in exprs if (match:=candidate(expr)) is not None and _root_param(match[0].src[0]) is source]
    if len(parsed) != 1: return None
    expr, (load, negated) = parsed[0]
    if load.src[0].op is not Ops.INDEX or load.src[0].src[1] is not reduce_range: return None
    try:
      if [_eval_int(load.src[0].src[1], {reduce_range:i}) for i in range(window)] != list(range(window)): return None
    except RuntimeError: return None
    return expr, negated

  value_candidate = global_candidate(list(value_updates[0].src[1].src), first_range)
  int_nodes = int_updates[0].src[1].toposort()
  comparisons = [u for u in int_nodes if u.op is Ops.CMPNE and u.dtype.scalar() is dtypes.bool]
  inner_cmps = [u for u in comparisons if any(candidate(x) is not None for x in u.src)]
  if value_candidate is None or len(inner_cmps) != 1: return None
  cmp = inner_cmps[0]
  second_candidate = global_candidate(list(cmp.src), second_range)
  if second_candidate is None or second_candidate[1] != value_candidate[1]: return None
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
                                 value_candidate[1], [window-1], first_tie=True, negate_extrema=value_candidate[1],
                                 native_int16=native_int16)

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
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(_int32_tiles_bytes(count)))
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops:list[RKEWOp] = []
  _ew_eq_mask(ops, arg, candidates_slot, zero, (diff, magnitude, unequal, equal), one, matrix_lanes)
  selected = _reduce_rows(ops, [arg(unequal, row*vector_bytes) for row in range(window)], count,
                          _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), selected, arg(int_tiles), count, _EW_CFG[Ops.MAX],
                    stateful=True, int32_output=True, bool_output=True))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

def _block_bool_reduction_ops(buf:RKArg, count:int, groups:int, op:Ops, int16:bool=False) -> list[RKEWOp]:
  """Reduce each contiguous block in place with one DPU EW tree."""
  ops:list[RKEWOp] = []
  first = True
  for lane in range(count):
    width, base = groups, lane*groups*2
    while width > 1:
      left, pairs = (width+1)//2, width//2
      ops.append(RKEWOp(replace(buf, addend=base), replace(buf, addend=base), replace(buf, addend=base+left*2), pairs,
                        _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL], submit_barrier=first and not int16,
                        stateful=first and not int16, int16_input=int16, int16_output=int16))
      first, width = False, left
  return ops

def _contiguous_bool_reduction_image(out_slot:int, count:int, source_slot:int, groups:int, op:Ops) -> RKImage:
  """Reduce contiguous FP16 blocks without materializing their transposed offset matrix."""
  source_count = count*groups
  zero, diff, magnitude, unequal, packed, int_tiles = range(6)
  def arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
  ops = [RKEWOp(arg(diff), RKArg(RKBufferKind.ARG, source_slot), arg(zero), source_count, _EW_CFG[Ops.SUB]),
         RKEWOp(arg(magnitude), arg(diff), arg(diff), source_count, _EW_CFG_ABS, submit_barrier=True, stateful=True),
         RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), source_count, _EW_CFG[Ops.MAX], compare=True)]
  ops.extend(_block_bool_reduction_ops(arg(unequal), count, groups, op))
  gather_after = len(ops)
  mid = (RKGather(unequal, packed, count, offsets=tuple(lane*groups for lane in range(count)),
                  src_kind=RKBufferKind.SCRATCH),)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), arg(packed), arg(int_tiles), count, _EW_CFG[Ops.MAX],
                    stateful=True, int32_output=True, bool_output=True))
  full, output = _scratch_bytes(source_count), _scratch_bytes(count)
  scratch = (RKScratch(full), RKScratch(full), RKScratch(full), RKScratch(full), RKScratch(output),
             RKScratch(_int32_tiles_bytes(count)))
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<e", 0.0), ew_ops=tuple(ops), mid_gathers=mid, gather_after=gather_after)

def _stored_bool_reduction_image(out_slot:int, count:int, source_slot:int, offsets:tuple[tuple[int, ...], ...], op:Ops) -> RKImage:
  """Place opaque bool bytes into zeroed INT16 lanes and reduce them with native integer EW."""
  vector_bytes, _, matrix_lanes = _stripe_layout(count, len(offsets))
  gathers = tuple(RKGather(source_slot, 0, count, offsets=row, dst_addend=i*vector_bytes, dst_stride=2, itemsize=1)
                  for i,row in enumerate(offsets))
  ops:list[RKEWOp] = []
  selected = _reduce_rows(ops, [RKArg(RKBufferKind.SCRATCH, 0, i*vector_bytes) for i in range(len(offsets))], count,
                          _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL], int16=True)
  post = (_int16_low_bytes(selected, out_slot, count),)
  return RKImage(RKTarget.RK3588, (RKScratch(_scratch_bytes(matrix_lanes)),), gathers=gathers,
                 ew_ops=tuple(ops), post_gathers=post)

def _contiguous_stored_bool_reduction_image(out_slot:int, count:int, source_slot:int, groups:int, op:Ops) -> RKImage:
  """Reduce contiguous opaque bool blocks after widening their bytes into native INT16 lanes."""
  source_count = count*groups
  ops = _block_bool_reduction_ops(RKArg(RKBufferKind.SCRATCH, 0), count, groups, op, int16=True)
  gathers = (RKGather(source_slot, 0, source_count, dst_stride=2, itemsize=1),)
  post = (_int16_low_bytes(RKArg(RKBufferKind.SCRATCH, 0), out_slot, count, groups*2),)
  return RKImage(RKTarget.RK3588, (RKScratch(_scratch_bytes(source_count)),), gathers=gathers,
                 ew_ops=tuple(ops), post_gathers=post)

def _nonzero_load(term:UOp) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtypes.half and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and float(zero.arg) == 0.0]
  return candidates[0] if len(candidates) == 1 else None

def _int32_nonzero_load(term:UOp) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtypes.int and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and int(zero.arg) == 0]
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

def _lower_stored_bool_reduction(output:RKOutput) -> RKImage|None:
  """Reduce an unrolled tree loaded from one canonical external bool buffer."""
  _, out_param, count, out_index, root = output
  root = _unwrap_condition(root)
  if root.op not in (Ops.OR, Ops.AND): return None
  loads = [_unwrap_condition(term) for term in _flatten_binary(root, root.op)]
  if any(load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.bool or load.src[0].op is not Ops.INDEX for load in loads): return None
  params = [_root_param(load.src[0]) for load in loads]
  if (not loads or any(param is None or param.dtype.scalar() is not dtypes.bool or param.src[0].op is not Ops.CONST for param in params) or
      len({param.arg.slot for param in params if param is not None}) != 1): return None
  source = params[0]; assert source is not None
  if count <= 0: return RKImage(RKTarget.RK3588)
  try: offsets = tuple(tuple(_gather_offsets(out_index, load.src[0].src[1], None, count)) for load in loads)
  except RuntimeError: return None
  source_count = int(source.src[0].arg)
  if source_count != count*len(loads) or sorted(offset for row in offsets for offset in row) != list(range(source_count)): return None
  return _stored_bool_reduction_image(out_param.arg.slot, count, source.arg.slot, offsets, root.op)

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

def _lower_grouped_bool_reduction(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Lower grouped FP16 or stored-bool any/all after proving launch coordinates and full source coverage."""
  store, out_param, count, out_index, root = output
  if _local_load(root) is None or len(store.src) not in (2, 3) or count <= 0: return None
  nodes = list(root.toposort())
  local_stores = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None]
  updates = [(u, _strip_cast(u.src[1])) for u in local_stores if _strip_cast(u.src[1]).op in (Ops.OR, Ops.AND)]
  if len(local_stores) != 5 or len(updates) != 2 or len({value.op for _,value in updates}) != 1: return None
  op = updates[0][1].op
  half_updates = [(u, value, load) for u,value in updates for src in value.src if (load:=_nonzero_load(src)) is not None]
  bool_updates = [(u, value, src) for u,value in updates for src in value.src if
                  src.op is Ops.LOAD and src.dtype.scalar() is dtypes.bool and _root_param(src.src[0]) is not None]
  if len(half_updates)+len(bool_updates) != 1: return None
  stored_bool = not half_updates
  _, _, load = (bool_updates if stored_bool else half_updates)[0]
  source = _root_param(load.src[0])
  source_dtype = dtypes.bool if stored_bool else dtypes.half
  if source is None or source.dtype.scalar() is not source_dtype or source.src[0].op is not Ops.CONST: return None
  identity = op is Ops.AND
  initials = [u for u in local_stores if u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() is dtypes.bool and
              bool(u.src[1].arg) == identity]
  bridges = [u for u in local_stores if _local_load(u.src[1]) is not None]
  external_loads = [u for u in nodes if u.op is Ops.LOAD and _root_param(u.src[0]) is not None]
  if len(initials) != 2 or len(bridges) != 1 or external_loads != [load]: return None

  source_index = load.src[0].src[1]
  specials = [u for u in nodes if u.op is Ops.SPECIAL]
  reduce_ranges = [u for u in source_index.toposort() if u.op is Ops.RANGE]
  if not specials or len(reduce_ranges) != 1 or any(u.src[0].op is not Ops.CONST for u in (*specials, *reduce_ranges)): return None
  reduce_range = reduce_ranges[0]
  extents = tuple(int(u.src[0].arg) for u in specials) + (int(reduce_range.src[0].arg),)
  if any(extent <= 0 for extent in extents): return None
  shape = tuple(extents)
  env:dict[UOp, np.ndarray] = {}
  for axis,(u,extent) in enumerate(zip((*specials, reduce_range), extents)):
    env[u] = np.arange(extent, dtype=np.int64).reshape((1,)*axis+(extent,)+(1,)*(len(shape)-axis-1))
  try:
    source_offsets = np.broadcast_to(_eval_vector(source_index, env, {}), shape).astype(np.int64, copy=False)
    output_offsets = np.broadcast_to(_eval_vector(out_index, env, {}), shape).astype(np.int64, copy=False)
    if len(store.src) == 3:
      special_shape = shape[:-1]
      special_env = {u:env[u][..., 0] for u in specials}
      output_gate = np.broadcast_to(_eval_vector(store.src[2], special_env, {}), special_shape).astype(bool, copy=False)
      stored_offsets = np.broadcast_to(_eval_vector(out_index, special_env, {}), special_shape).astype(np.int64, copy=False)
  except (KeyError, RuntimeError, ValueError): return None
  source_count = int(source.src[0].arg)
  if (source_offsets.size != source_count or np.any((source_offsets < 0) | (source_offsets >= source_count)) or
      not np.array_equal(np.sort(source_offsets, axis=None), np.arange(source_count, dtype=np.int64)) or
      np.any((output_offsets < 0) | (output_offsets >= count))): return None
  flat_source, flat_output = source_offsets.reshape(-1), output_offsets.reshape(-1)
  groups = source_count//count
  if groups*count != source_count: return None
  order = np.argsort(flat_output, kind="stable")
  if not np.array_equal(flat_output[order], np.repeat(np.arange(count, dtype=np.int64), groups)): return None
  matrix = flat_source[order].reshape(count, groups)
  if len(store.src) == 3 and (np.any((stored_offsets < 0) | (stored_offsets >= count)) or
                              any(np.count_nonzero(output_gate & (stored_offsets == lane)) != 1 for lane in range(count))): return None
  if np.array_equal(matrix, np.arange(source_count, dtype=np.int64).reshape(count, groups)):
    return (_contiguous_stored_bool_reduction_image if stored_bool else _contiguous_bool_reduction_image)(
      out_param.arg.slot, count, source.arg.slot, groups, op)
  offsets = tuple(tuple(int(value) for value in matrix[:, group]) for group in range(groups))
  return (_stored_bool_reduction_image if stored_bool else _bool_reduction_image)(out_param.arg.slot, count, source.arg.slot, offsets, op)

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
  return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(_int32_tiles_bytes(count))), ew_ops=ops)

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

def _lower_one_hot(output:RKOutput) -> RKImage|None:
  """Compare dynamic INT32 indices with static class coordinates byte-wise on DPU EW."""
  _, out_param, count, out_index, root = output
  if (root.op is not Ops.WHERE or root.dtype.scalar() is not dtypes.int or root.src[0].op is not Ops.CMPNE or
      any(arm.op is not Ops.CONST or arm.dtype.scalar() is not dtypes.int for arm in root.src[1:]) or
      tuple(int(arm.arg) for arm in root.src[1:]) != (0, 1)): return None
  loads = [x for x in root.src[0].src if x.op is Ops.LOAD and x.dtype.scalar() is dtypes.int and
           len(x.src) == 1 and x.src[0].op is Ops.INDEX]
  coordinates = [x for x in root.src[0].src if _is_static_expr(x)]
  if len(loads) != 1 or len(coordinates) != 1: return None
  load, coordinate = loads[0], coordinates[0]
  source = _root_param(load.src[0])
  if source is None or source.dtype.scalar() is not dtypes.int or source.src[0].op is not Ops.CONST: return None
  source_count = int(source.src[0].arg)
  if count <= 0: return RKImage(RKTarget.RK3588)
  if source_count <= 0: return None
  try:
    offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
    coordinate_values = _static_int_vector(out_index, coordinate, count)
  except RuntimeError: return None
  if (any(not 0 <= offset < source_count for offset in offsets) or
      any(not -(1<<31) <= value < 1<<31 for value in coordinate_values)): return None
  if (equality:=_int32_equality_matrix(((source.arg.slot, source_count, (offsets,), (coordinate_values,)),), count)) is None: return None
  int_tiles = len(equality.scratch_sizes)
  scratch = tuple(RKScratch(size) for size in equality.scratch_sizes) + (RKScratch(_int32_tiles_bytes(count)),)
  terminal = RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), equality.mask,
                    RKArg(RKBufferKind.SCRATCH, int_tiles), count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True)
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<e", 1.0), gathers=equality.gathers,
                 ew_ops=equality.pre_ops+equality.mask_ops+(terminal,), mid_gathers=equality.mid_gathers, gather_after=4)

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

def _int16_const(u:UOp, value:int) -> bool:
  return u.op is Ops.CONST and u.dtype.scalar() in (dtypes.int16, dtypes.weakint) and int(u.arg) == value

def _int16_nonconst(u:UOp, value:int) -> UOp|None:
  if len(u.src) != 2: return None
  if _int16_const(u.src[0], value): return u.src[1]
  if _int16_const(u.src[1], value): return u.src[0]
  return None

def _fold_native_int16_sign(x:UOp) -> UOp|None:
  """Recover WHERE(x!=0, WHERE(x<0, -1, 1), 0) as the exact clamp min(max(x,-1),1)."""
  if (not _int16_const(x.src[2], 0) or x.src[0].op is not Ops.CMPNE or
      (candidate:=_int16_nonconst(x.src[0], 0)) is None or x.src[1].op is not Ops.WHERE): return None
  condition, negative, positive = x.src[1].src
  if (condition.op is not Ops.CMPLT or condition.src[0].key != candidate.key or not _int16_const(condition.src[1], 0) or
      not _int16_const(negative, -1) or not _int16_const(positive, 1)): return None
  lower = candidate.alu(Ops.MAX, UOp.const(-1, dtypes.int16))
  return UOp(Ops.MAX, x.dtype, src=(lower, UOp.const(1, dtypes.int16)), arg=_NATIVE_MIN)

def _int16_relu_operand(x:UOp) -> UOp|None:
  if x.op is not Ops.WHERE or x.src[0].op is not Ops.CMPLT or not _int16_const(x.src[2], 0): return None
  lhs, rhs = x.src[0].src
  return rhs if _int16_const(lhs, 0) and x.src[1].key == rhs.key else None

def _fold_native_int16_relu6(x:UOp) -> UOp|None:
  """Recover relu(x)-relu(x-6) as the exact clamp min(max(x,0),6)."""
  for positive, negative in (x.src, x.src[::-1]):
    source, scaled = _int16_relu_operand(positive), _int16_nonconst(negative, -1)
    if source is None or scaled is None or (shifted:=_int16_relu_operand(scaled)) is None: continue
    if shifted.op is Ops.ADD and (base:=_int16_nonconst(shifted, -6)) is not None and base.key == source.key:
      lower = source.alu(Ops.MAX, UOp.const(0, dtypes.int16))
      return UOp(Ops.MAX, x.dtype, src=(lower, UOp.const(6, dtypes.int16)), arg=_NATIVE_MIN)
  return None

def _fold_native_int16_leaky_relu(x:UOp) -> UOp|None:
  """Recover integral leaky ReLU as x + min(x,0)*(slope-1), preserving saturating INT16 semantics."""
  gate, negative, positive = x.src
  if gate.op is not Ops.CMPLT or not _int16_const(gate.src[1], 0) or positive.key != gate.src[0].key or negative.op is not Ops.MUL:
    return None
  for factor, value in (negative.src, negative.src[::-1]):
    if (value.key != positive.key or factor.op is not Ops.CONST or factor.dtype.scalar() not in (dtypes.int16, dtypes.weakint) or
        not 2 <= (slope:=int(factor.arg)) <= 32767): continue
    below = UOp(Ops.MAX, x.dtype, src=(positive, UOp.const(0, dtypes.int16)), arg=_NATIVE_MIN)
    correction = below if slope == 2 else below.alu(Ops.MUL, UOp.const(slope-1, dtypes.int16))
    return positive.alu(Ops.ADD, correction)
  return None

def _fold_native_int16(x:UOp) -> UOp|None:
  """Recover native integer operations from Tinygrad's portable decompositions."""
  if x.op is Ops.XOR and (maximum:=_int16_nonconst(x, -1)) is not None and maximum.op is Ops.MAX:
    lhs, rhs = (_int16_nonconst(src, -1) for src in maximum.src)
    if lhs is not None and rhs is not None: return UOp(Ops.MAX, x.dtype, src=(lhs, rhs), arg=_NATIVE_MIN)
  if x.op is Ops.MUL:
    for source, sign in (x.src, x.src[::-1]):
      if _int16_const(sign, -1): return UOp(Ops.NEG, x.dtype, src=(source,))
      if (sign.op is not Ops.WHERE or len(sign.src) != 3 or not _int16_const(sign.src[2], 0) or
          sign.src[0].op is not Ops.CMPNE or source not in sign.src[0].src or
          not any(_int16_const(u, 0) for u in sign.src[0].src)): continue
      inner = sign.src[1]
      if (inner.op is Ops.WHERE and inner.src[0].op is Ops.CMPLT and inner.src[0].src[0].key == source.key and
          _int16_const(inner.src[0].src[1], 0) and _int16_const(inner.src[1], -1) and _int16_const(inner.src[2], 1)):
        return UOp(Ops.MAX, x.dtype, src=(source, source), arg=_NATIVE_ABS)
  if x.op is Ops.ADD:
    for lhs,rhs in (x.src, x.src[::-1]):
      if rhs.op is Ops.NEG: return UOp(Ops.SUB, x.dtype, src=(lhs, rhs.src[0]))
  if x.op is Ops.WHERE and x.src[0].op is Ops.CMPLT:
    lhs, rhs = x.src[0].src
    if x.src[1].key == lhs.key and x.src[2].key == rhs.key: return UOp(Ops.MAX, x.dtype, src=(lhs, rhs), arg=_NATIVE_MIN)
    if x.src[1].key == rhs.key and x.src[2].key == lhs.key: return lhs.alu(Ops.MAX, rhs)
  if x.op is Ops.WHERE and (mask:=_native_int16_comparison(x.src[0])) is not None:
    selected = mask.alu(Ops.MUL, x.src[1])
    rejected = UOp.const(1, dtypes.int16).alu(Ops.SUB, mask).alu(Ops.MUL, x.src[2])
    return selected.alu(Ops.ADD, rejected)
  return None

def _native_int16_comparison(root:UOp) -> UOp|None:
  """Express signed INT16 comparisons and their boolean compositions as saturating integer ALU masks."""
  if root.op in (Ops.CMPLT, Ops.CMPNE) and all(src.dtype.scalar() is dtypes.int16 for src in root.src):
    lhs, rhs = root.src
    delta = rhs.alu(Ops.SUB, lhs) if root.op is Ops.CMPLT else lhs.alu(Ops.SUB, rhs)
    magnitude = delta.alu(Ops.MAX, UOp.const(0, dtypes.int16)) if root.op is Ops.CMPLT else \
                UOp(Ops.MAX, dtypes.int16, src=(delta, delta), arg=_NATIVE_ABS)
    return UOp(Ops.MAX, dtypes.int16, src=(magnitude, UOp.const(1, dtypes.int16)), arg=_NATIVE_MIN)
  if root.op in (Ops.AND, Ops.OR, Ops.XOR):
    left_mask, right_mask = (_native_int16_comparison(src) for src in root.src)
    if left_mask is None or right_mask is None: return None
    if root.op is Ops.AND: return left_mask.alu(Ops.MUL, right_mask)
    if root.op is Ops.OR: return left_mask.alu(Ops.MAX, right_mask)
    delta = left_mask.alu(Ops.SUB, right_mask)
    return UOp(Ops.MAX, dtypes.int16, src=(delta, delta), arg=_NATIVE_ABS)
  if root.op is Ops.CMPNE:
    for value, marker in (root.src, root.src[::-1]):
      if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg):
        if (mask:=_native_int16_comparison(value)) is not None: return UOp.const(1, dtypes.int16).alu(Ops.SUB, mask)
  return None

_pm_native_int16_hard_activation = PatternMatcher([
  (UPat(Ops.WHERE, dtype=dtypes.int16, name="x"), _fold_native_int16_sign),
  (UPat(Ops.WHERE, dtype=dtypes.int16, name="x"), _fold_native_int16_leaky_relu),
  (UPat(Ops.ADD, dtype=dtypes.int16, name="x"), _fold_native_int16_relu6),
])
_pm_native_int16 = PatternMatcher([(UPat(GroupOp.ALU, dtype=dtypes.int16, name="x"), _fold_native_int16)])

def _fold_native_int16_complemented_min(root:UOp) -> UOp:
  """Recover MAX(~x...) temporary extrema as `~MIN(x...)` for ArgMin index selection."""
  leaves = _flatten_binary(root, Ops.MAX) if root.op is Ops.MAX else []
  candidates = tuple(_int16_nonconst(leaf, -1) for leaf in leaves)
  if len(candidates) < 2 or any(candidate is None for candidate in candidates): return root
  values = tuple(candidate for candidate in candidates if candidate is not None)
  minimum = values[0]
  for value in values[1:]: minimum = UOp(Ops.MAX, dtypes.int16, src=(minimum, value), arg=_NATIVE_MIN)
  return UOp.const(-1, dtypes.int16).alu(Ops.SUB, minimum)

def _lower_native_int16_ew(uops:list[UOp]) -> RKImage|None:
  """Lower signed INT16 arithmetic and exact comparison masks to the DPU integer EW pipeline."""
  int32_output = bool_output = False
  if (output:=_output_store(uops, dtypes.int16)) is None:
    output = _output_store(uops, dtypes.int)
    if output is not None and output[4].op is Ops.CAST and output[4].src[0].dtype.scalar() is dtypes.int16:
      output, int32_output = (*output[:4], output[4].src[0]), True
    else:
      output = _output_store(uops, dtypes.bool)
      if output is None or (comparison:=_native_int16_comparison(output[4])) is None: return None
      output, bool_output = (*output[:4], comparison), True
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  value = _fold_native_int16_complemented_min(value)
  value = graph_rewrite(value, _pm_native_int16_hard_activation, name="rockchip native int16 hard activation")
  value = graph_rewrite(value, _pm_native_int16, name="rockchip native int16")
  supported = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.NEG}
  static_cache:dict[UOp, bool] = {}
  leaf_cache:dict[UOp, RKInt16Leaf] = {}
  def leaf(u:UOp) -> RKInt16Leaf:
    if u not in leaf_cache: leaf_cache[u] = _int16_leaf(u, out_index, count, out_param.arg.slot, static_cache)
    return leaf_cache[u]
  if leaf(value) is not None: value = value.alu(Ops.ADD, UOp.const(0, dtypes.int16))
  order:list[UOp] = []
  visited:dict[UOp, bool] = {}
  def visit(u:UOp) -> bool:
    if u in visited: return visited[u]
    if u.op in supported and u.dtype.scalar() is dtypes.int16:
      visited[u] = len(u.src) == (1 if u.op is Ops.NEG else 2) and all(visit(src) for src in u.src)
      if visited[u]: order.append(u)
    else: visited[u] = leaf(u) is not None
    return visited[u]
  if not visit(value) or not order or order[-1] is not value: return None

  uses = {u:sum(u in expr.src for expr in order) for u in order}
  values:dict[UOp, RKArg] = {}
  leaves:dict[UOp, RKArg] = {}
  free:list[int] = []
  constants:dict[bytes, int] = {}
  static_slots:dict[tuple[int, ...], int] = {}
  gather_slots:dict[tuple, int] = {}
  gathers:list[RKGather] = []
  for expr in order:
    for src in expr.src:
      if isinstance((parsed:=leaf(src)), int):
        bits = struct.pack("<H", _int16_bits(parsed))
        if bits not in constants: constants[bits] = len(constants)
  scratch_count = len(constants)
  out = RKArg(RKBufferKind.SCRATCH, scratch_count) if bool_output else RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if bool_output: scratch_count += 1
  def operand(u:UOp) -> RKArg:
    nonlocal scratch_count
    if u in values: return values[u]
    if u in leaves: return leaves[u]
    parsed = leaf(u); assert parsed is not None
    if isinstance(parsed, int): ret = RKArg(RKBufferKind.SCRATCH, constants[struct.pack("<H", _int16_bits(parsed))])
    elif isinstance(parsed, RKStatic):
      vector = _static_values(out_index, parsed.expr, count, _int16_bits)
      if vector not in static_slots:
        static_slots[vector] = scratch_count
        gathers.append(RKGather(0, scratch_count, count, values=vector))
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, static_slots[vector])
    elif isinstance(parsed, RKArg): ret = parsed
    else:
      if isinstance(parsed, RKMultiGather): plans = parsed.gathers
      elif isinstance(parsed, RKGather): plans = (parsed,)
      else:
        assert isinstance(parsed, tuple)
        param, index, gate, fill_bits = parsed
        plans = (_gather_plan(param.arg.slot, 0, out_index, index, gate, count, fill_bits),)
      for plan in plans:
        if plan.values: continue
        source = next((x for x in u.toposort() if x.op is Ops.PARAM and x.arg.slot == plan.src_index), None)
        if source is None or source.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:gather_index")
        _validate_gather_bounds(plan, int(source.src[0].arg))
      key = _gather_cache_key(plans)
      if key not in gather_slots:
        gather_slots[key] = scratch_count
        gathers.extend(replace(p, dst_index=scratch_count, itemsize=2) for p in plans)
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, gather_slots[key])
    leaves[u] = ret
    return ret
  ew_ops:list[RKEWOp] = []
  for expr in order:
    lhs = operand(expr.src[0]); rhs = lhs if expr.op is Ops.NEG else operand(expr.src[1])
    if expr is value: dst = out
    elif (reuse:=next((values[src] for src in expr.src if src in values and uses[src] == 1 and
                       values[src].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    cfg = _EW_CFG_MIN if expr.op is Ops.MAX and expr.arg == _NATIVE_MIN else \
      _EW_CFG_ABS if expr.op is Ops.MAX and expr.arg == _NATIVE_ABS else _EW_CFG_NEG if expr.op is Ops.NEG else _EW_CFG[expr.op]
    wide = int32_output and expr is value
    ew_ops.append(RKEWOp(dst, lhs, rhs, count, cfg, int16_input=True, int16_output=not wide, int32_output=wide))
    values[expr] = dst
    for dep in expr.src:
      if dep in uses:
        uses[dep] -= 1
        if uses[dep] == 0 and values[dep].kind is RKBufferKind.SCRATCH and values[dep] != dst: free.append(values[dep].index)
  constant_data = b""
  if constants:
    by_slot = {slot:bits for bits,slot in constants.items()}
    constant_data = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot)+1))
  post = (_int16_low_bytes(out, out_param.arg.slot, count),) if bool_output else ()
  return RKImage(RKTarget.RK3588, tuple(RKScratch(_scratch_bytes(count)) for _ in range(scratch_count)), constant_data,
                 gathers=tuple(gathers), ew_ops=tuple(ew_ops), post_gathers=post)

def lower_ew(uops:list[UOp]) -> RKImage:
  if (int16_cumulative:=_lower_unrolled_int16_cumulative_extrema(uops)) is not None: return int16_cumulative
  if (int16_ew:=_lower_native_int16_ew(uops)) is not None: return int16_ew
  if (int16_extrema:=_lower_int16_loop_extrema(uops)) is not None: return int16_extrema
  if (int16_output:=_output_store(uops, dtypes.int16)) is not None:
    if (fixed_select:=_lower_fixed_integer_masked_select(int16_output, dtypes.int16)) is not None: return fixed_select
    if (unrolled_fancy:=_lower_unrolled_multi_16bit_fancy_index(int16_output, dtypes.int16)) is not None: return unrolled_fancy
    if (dynamic_gather:=_lower_dynamic_16bit_gather(int16_output, dtypes.int16)) is not None: return dynamic_gather
    if (fancy_index:=_lower_multi_16bit_fancy_index(int16_output, dtypes.int16)) is not None: return fancy_index
    if (scatter:=_lower_dynamic_16bit_scatter(int16_output, dtypes.int16)) is not None: return scatter
  if (float_output:=_output_store(uops, dtypes.float)) is not None and \
     (fp16_cast:=_lower_fp16_fp32_cast(float_output)) is not None: return fp16_cast
  if (bool_output:=_output_store(uops, dtypes.bool)) is not None:
    if bool_output[4].op is Ops.CONST:
      return RKImage(RKTarget.RK3588, constants=struct.pack("<?", bool(bool_output[4].arg)),
                     fill=RKFill(RKArg(RKBufferKind.ARG, bool_output[1].arg.slot), bool_output[2], 1))
    if (bounds_mask:=_lower_int32_bounds_mask(bool_output)) is not None: return bounds_mask
    if (stored_bool_reduction:=_lower_stored_bool_reduction(bool_output)) is not None: return stored_bool_reduction
    if (bool_reduction:=_lower_unrolled_bool_reduction(bool_output)) is not None: return bool_reduction
    if (predicate:=_lower_ieee_predicate(bool_output)) is not None: return predicate
    if (nonzero:=_fp16_nonzero_mask(bool_output[4])) is not None: return _typed_int_image(bool_output, nonzero, bool_output=True)
    if (comparison:=_ieee_comparison_mask(bool_output[4])) is not None: return _typed_int_image(bool_output, comparison, bool_output=True)
  if (bool_loop_output:=_output_store(uops, dtypes.bool, allow_local=True)) is not None:
    if (bool_loop_reduction:=_lower_loop_bool_reduction(uops, bool_loop_output)) is not None: return bool_loop_reduction
    if (grouped_bool_reduction:=_lower_grouped_bool_reduction(uops, bool_loop_output)) is not None: return grouped_bool_reduction
  if (half_output:=_output_store(uops, dtypes.half)) is not None:
    if (sort_compare:=_lower_sort_compare(half_output)) is not None: return sort_compare
    if (max_unpool:=_lower_unrolled_max_unpool(half_output)) is not None: return max_unpool
    if (fixed_masked_select:=_lower_fixed_fp16_masked_select(half_output)) is not None: return fixed_masked_select
    if (unrolled_fancy:=_lower_unrolled_multi_16bit_fancy_index(half_output)) is not None: return unrolled_fancy
    if (dynamic_gather:=_lower_dynamic_16bit_gather(half_output)) is not None: return dynamic_gather
    if (fancy_index:=_lower_multi_16bit_fancy_index(half_output)) is not None: return fancy_index
    if (scatter_reduce:=_lower_dynamic_scalar_scatter_reduce(half_output)) is not None: return scatter_reduce
    if (scatter:=_lower_dynamic_16bit_scatter(half_output)) is not None: return scatter
  if (half_loop_output:=_output_store(uops, dtypes.half, allow_local=True)) is not None and \
     (loop_max_unpool:=_lower_loop_max_unpool(uops, half_loop_output)) is not None: return loop_max_unpool
  int_output, int_loop_output = _output_store(uops, dtypes.int), _output_store(uops, dtypes.int, allow_local=True)
  if int_output is not None and not any(u.op is Ops.REDUCE for u in uops):
    if (linear_index:=_lower_normalized_linear_index(int_output)) is not None: return linear_index
    if (fp16_cast:=_lower_fp16_int32_cast(int_output)) is not None: return fp16_cast
    if (bounded_lookup:=_lower_bounded_int32_lookup(int_output)) is not None: return bounded_lookup
    if (byte_add:=_lower_int32_byte_add(int_output)) is not None: return byte_add
    if (int32_prefix:=_lower_unrolled_int32_prefix_count(int_output)) is not None: return int32_prefix
    if (bool_prefix:=_lower_unrolled_bool_prefix_count(int_output)) is not None: return bool_prefix
    if (fixed_int_select:=_lower_fixed_integer_masked_select(int_output)) is not None: return fixed_int_select
    if (predicate_total:=_lower_unrolled_fp16_predicate_total(int_output)) is not None: return predicate_total
    if (positive_prefix:=_lower_unrolled_fp16_prefix_count(int_output)) is not None: return positive_prefix
    if (fixed_int_nonzero:=_lower_fixed_int32_nonzero(int_output)) is not None: return fixed_int_nonzero
    if (fixed_nonzero:=_lower_fixed_fp16_nonzero(int_output)) is not None: return fixed_nonzero
    if (sum_occurrence:=_lower_unrolled_int32_sum_occurrence(int_output)) is not None: return sum_occurrence
    if (int_occurrence:=_lower_unrolled_int_occurrence_count(int_output)) is not None: return int_occurrence
    if (int_prefix:=_lower_unrolled_int_prefix_sum(int_output)) is not None: return int_prefix
    if (occurrence_count:=_lower_occurrence_count(int_output)) is not None: return occurrence_count
    if (sort_index:=_lower_sort_index_selection(int_output)) is not None: return sort_index
  if int_loop_output is not None:
    if (loop_predicate_total:=_lower_loop_fp16_predicate_total(uops, int_loop_output)) is not None: return loop_predicate_total
    if (loop_fp16_prefix:=_lower_loop_fp16_prefix_count(uops, int_loop_output)) is not None: return loop_fp16_prefix
    if (loop_occurrence:=_lower_loop_int32_occurrence_count(uops, int_loop_output)) is not None: return loop_occurrence
    if (loop_prefix:=_lower_loop_int_prefix_sum(uops, int_loop_output)) is not None: return loop_prefix
    if (cumulative_loop:=_lower_cumulative_extrema_index_loop(uops, int_loop_output)) is not None: return cumulative_loop
  if int_output is not None:
    if (cumulative_index:=_lower_cumulative_extrema_index(uops, int_output)) is not None: return cumulative_index
    if (int16_arg_extrema:=_lower_unrolled_arg_extrema(int_output, native_int16=True)) is not None: return int16_arg_extrema
    if (arg_extrema:=_lower_unrolled_arg_extrema(int_output)) is not None: return arg_extrema
    if (int16_pool_index:=_lower_unrolled_pool_index(int_output, native_int16=True)) is not None: return int16_pool_index
    if (pool_index:=_lower_unrolled_pool_index(int_output)) is not None: return pool_index
    if (int16_max_unpool:=_lower_unrolled_max_unpool(int_output, native_int16=True)) is not None: return int16_max_unpool
    if (one_hot:=_lower_one_hot(int_output)) is not None: return one_hot
    if (int_where:=_lower_int_where(int_output)) is not None: return int_where
    if (raw_bitcast:=_lower_raw_fp16_bitcast(int_output)) is not None: return raw_bitcast
  if int_loop_output is not None:
    if (int16_loop_arg_extrema:=_lower_loop_arg_extrema_index(uops, int_loop_output, native_int16=True)) is not None:
      return int16_loop_arg_extrema
    if (loop_arg_extrema:=_lower_loop_arg_extrema_index(uops, int_loop_output)) is not None: return loop_arg_extrema
    if (int16_loop_pool_index:=_lower_loop_pool_index(uops, int_loop_output, native_int16=True)) is not None: return int16_loop_pool_index
    if (loop_pool_index:=_lower_loop_pool_index(uops, int_loop_output)) is not None: return loop_pool_index
    if (int16_loop_max_unpool:=_lower_loop_max_unpool(uops, int_loop_output, native_int16=True)) is not None: return int16_loop_max_unpool
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
        _validate_gather_bounds(plan, int(source.src[0].arg))
      key = _gather_cache_key(gather_plans)
      if key not in gather_scratch:
        gather_scratch[key] = scratch_count
        gathers.extend(replace(plan, dst_index=scratch_count) for plan in gather_plans)
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

def _fold_copysign(x:UOp) -> UOp|None:
  """Recognize Tinygrad's signed-zero-aware copysign graph before numeric sign folding."""
  if x.op is not Ops.WHERE or len(x.src) != 3: return None
  condition, negative, positive = x.src
  if condition.op is not Ops.OR or negative.op is not Ops.MUL or positive.op is not Ops.MUL: return None
  negated = _const_operand(negative, Ops.MUL, -1.0)
  absolute = _fold_abs(positive)
  if negated is None or negated[0].key != positive.key or absolute is None: return None
  def lt_zero(root:UOp) -> UOp|None:
    return root.src[0] if root.op is Ops.CMPLT and root.src[1].op is Ops.CONST and float(root.src[1].arg) == 0.0 else None
  for direct, reciprocal in (condition.src, condition.src[::-1]):
    sign, inverse = lt_zero(direct), lt_zero(reciprocal)
    if (sign is not None and inverse is not None and inverse.op is Ops.FDIV and inverse.src[0].op is Ops.CONST and
        float(inverse.src[0].arg) == 1.0 and inverse.src[1].key == sign.key):
      return UOp(Ops.ADD, x.dtype, src=(absolute.src[0], sign), arg=_NATIVE_COPYSIGN)
  return None

def _fold_floor_ceil(x:UOp) -> UOp|None:
  """Recognize Tinygrad's TRUNC-based floor/ceil expansions and select the native DPU ALU."""
  if x.op is not Ops.WHERE or len(x.src) != 3: return None
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

def _fold_round(x:UOp) -> UOp|None:
  """Recognize Tinygrad's round-to-even graph and compose it from native FLOOR/TRUNC and DPU masks."""
  if x.op is not Ops.WHERE or len(x.src) != 3: return None
  gate, yes, no = x.src
  floor, ceil = _fold_floor_ceil(yes), _fold_floor_ceil(no)
  if floor is None or ceil is None or floor.arg != _NATIVE_FLOOR or ceil.arg != _NATIVE_CEIL or gate.op is not Ops.CMPNE: return None
  floor_shift, ceil_shift = _const_operand(floor.src[0], Ops.ADD, 0.5), _const_operand(ceil.src[0], Ops.ADD, -0.5)
  source, ceil_source = (None, None) if floor_shift is None or ceil_shift is None else (floor_shift[0], ceil_shift[0])
  if source is None or ceil_source is None or source.key != ceil_source.key: return None
  positive = next((u for u in gate.src if u.op is Ops.CMPLT and u.src[1].key == source.key and
                   u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0), None)
  parity = next((u for u in gate.src if u is not positive), None)
  if positive is None or parity is None or parity.op is not Ops.CMPNE: return None
  unequal = next((u for u in parity.src if u.op is Ops.CMPNE), None)
  truth = next((u for u in parity.src if u.op is Ops.CONST and u.dtype.scalar() is dtypes.bool), None)
  if unequal is None or truth is None or not bool(truth.arg): return None
  truncated_half = next((u for u in unequal.src if u.op is Ops.TRUNC), None)
  half_value = next((u for u in unequal.src if u is not truncated_half), None)
  truncated = next((u for u in half_value.src if u.op is Ops.TRUNC), None) if half_value is not None and half_value.op is Ops.MUL else None
  scale = next((u for u in half_value.src if u.op is Ops.CONST), None) if half_value is not None and half_value.op is Ops.MUL else None
  if (truncated_half is None or half_value is None or truncated_half.src != (half_value,) or truncated is None or
      truncated.src != (source,) or scale is None or float(scale.arg) != 0.5): return None
  one, half = (UOp.const(v, dtypes.half) for v in (1.0, 0.5))
  def native(value:UOp, tag:str) -> UOp: return UOp(Ops.MAX, dtypes.half, src=(value, value), arg=tag)
  source_floor = native(source, _NATIVE_FLOOR)
  tie_delta = source.alu(Ops.SUB, source_floor).alu(Ops.SUB, half)
  tie = one.alu(Ops.SUB, _positive_mask(native(tie_delta, _NATIVE_ABS)))
  greater = _positive_mask(tie_delta)
  floor_half = source_floor.alu(Ops.MUL, half)
  parity_delta = floor_half.alu(Ops.SUB, _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(floor_half,))))
  odd = _positive_mask(native(parity_delta, _NATIVE_ABS))
  increment = greater.alu(Ops.MAX, _mask_mul(tie, odd))
  return source_floor.alu(Ops.ADD, increment)

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
_pm_round = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_round)])
_pm_floor_ceil = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_floor_ceil)])
_pm_trunc = PatternMatcher([(UPat(Ops.TRUNC, dtypes.half, name="x"), _fold_trunc)])
_pm_sign = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_sign)])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  sink = graph_rewrite(sink, _pm_round, name="rockchip round")
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
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_bounded_exact_fp16_copysign(uops)
    return base64.b64encode(encode_image(image if image is not None else lower_ew(_fp16_rewrite(uops)))).decode()

class RockchipBoolRenderer(RockchipRenderer):
  """Expose one 16-lane local bool tile that the renderer consumes as grouped DPU reduction work."""
  has_local, shared_max = True, 16
