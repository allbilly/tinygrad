from __future__ import annotations
# ruff: noqa: E702
import base64, heapq, math, os, struct
import numpy as np
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Iterable, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import Target, cdiv, ceildiv, cmod, floordiv, floormod, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 36
_HEADER = struct.Struct("<4sHHHHHIII")  # magic/version, scratch/gather/host counts, ops/constants, mid-gather count
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<I"), struct.Struct("<HHIBBBBBiIIii"), struct.Struct("<IIi")
_HOST_ADDRESS = struct.Struct("<BBBBBHHHIIIIIiiiiii")
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, flags, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg
_ITEM_FORMAT = {1:"B", 2:"H", 4:"I"}
_RKIMAGE_U16_MAX = (1 << 16) - 1

class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1
class RKLayout(IntEnum): FP16 = 0; INT16 = 1; BOOL_MASK = 2; INT32 = 3; BOOL_INT16 = 4; INT_FP16 = 5
class RKExecutionClass(IntEnum): NATIVE = 0; HOST_ADDRESS = 1

@dataclass(frozen=True, slots=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True, slots=True)
class RKValue:
  """Physical ABI. BOOL_MASK and bounded exact INT_FP16 values occupy FP16 scratch lanes."""
  arg: RKArg; dtype: DType; count: int; layout: RKLayout

@dataclass(frozen=True, slots=True)
class RKScratch: size: int

@dataclass(frozen=True, slots=True)
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
  after: int = -1  # EW-op split; -1 schedules the gather after the final stage

@dataclass(frozen=True, slots=True)
class RKHostAddress:
  """Host-calculated raw-lane movement. It never owns numeric or reduction semantics."""
  src: RKArg; index: RKArg; dst: RKArg; count: int; src_count: int; dst_count: int
  itemsize: int = 2; index_itemsize: int = 4; fill_bits: int = 0
  index_limit: int = 0; base: int = 0; index_scale: int = 1; lane_stride: int = 0

@dataclass(frozen=True, slots=True)
class RKEWOp:
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False
  int16_output: bool = False; int16_input: bool = False

@dataclass(frozen=True, slots=True)
class RKImage:
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""
  gathers: tuple[RKGather, ...] = (); ew_ops: tuple[RKEWOp, ...] = ()
  mid_gathers: tuple[RKGather, ...] = ()
  host_gathers: tuple[RKHostAddress, ...] = (); host_scatters: tuple[RKHostAddress, ...] = ()

  @property
  def execution_class(self) -> RKExecutionClass:
    return RKExecutionClass.HOST_ADDRESS if self.host_gathers or self.host_scatters else RKExecutionClass.NATIVE

def _hoist_leading_vector_materialization(image:RKImage) -> RKImage:
  """Compose a leading vector copy and its lane gathers so scalar execution starts in one NPU chain."""
  if len(image.ew_ops) < 2 or not image.mid_gathers: return image
  lead = image.ew_ops[0]
  if (lead.ew_cfg != _EW_CFG[Ops.MAX] or lead.lhs != lead.rhs or lead.lhs.kind is not RKBufferKind.SCRATCH or
      lead.dst.kind is not RKBufferKind.SCRATCH or lead.lhs.addend or lead.dst.addend or
      any((lead.compare, lead.stateful, lead.int32_output, lead.int32_input, lead.bool_output,
           lead.int16_output, lead.int16_input))): return image
  producers = [g for g in image.gathers if g.dst_kind is lead.lhs.kind and g.dst_index == lead.lhs.index]
  if len(producers) != 1: return image
  producer = producers[0]
  if (producer.values or producer.partial or producer.fill_bits or producer.count != lead.count or producer.dst_addend or
      producer.dst_stride != 1 or producer.itemsize != 2 or producer.src_kind is not RKBufferKind.ARG): return image
  moved = [g for g in image.mid_gathers if g.after == 1]
  if not moved or any(g.src_kind is not lead.dst.kind or g.src_index != lead.dst.index or g.itemsize != 2 for g in moved): return image
  written:set[tuple[int, int]] = set()
  for op in image.ew_ops[1:]:
    if any(arg.kind is lead.dst.kind and arg.index == lead.dst.index and (arg.addend, op.count) not in written for arg in (op.lhs, op.rhs)):
      return image
    if op.dst.kind is lead.dst.kind and op.dst.index == lead.dst.index: written.add((op.dst.addend, op.count))
  source_offsets = _plan_offsets(producer)
  indices = tuple(_plan_offsets(g) for g in moved)
  if any(index < 0 or index >= len(source_offsets) for row in indices for index in row): return image
  direct = tuple(replace(g, src_kind=producer.src_kind, src_index=producer.src_index, base=0, axes=(),
                         offsets=tuple(source_offsets[index] for index in row), after=-1) for g,row in zip(moved, indices))
  remaining = tuple(replace(g, after=max(0, g.after-1)) for g in image.mid_gathers if g.after != 1)
  return replace(image, gathers=image.gathers+direct, ew_ops=image.ew_ops[1:], mid_gathers=remaining)

def _reuse_linear_scratch(image:RKImage, constant_slots:dict[bytes, int]) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  ends, order = [-1] * len(image.scratch), list[tuple[int, int]]()
  def touch(arg:RKArg, event:int) -> None:
    if arg.kind is not RKBufferKind.SCRATCH: return
    if not 0 <= arg.index < len(ends): raise ValueError("invalid virtual scratch slot")
    if ends[arg.index] < 0: order.append((event, arg.index))
    ends[arg.index] = event
  def touch_gather(gather:RKGather, event:int) -> None:
    if not gather.values: touch(RKArg(gather.src_kind, gather.src_index), event)
    touch(RKArg(gather.dst_kind, gather.dst_index), event)
  def touch_host(host:RKHostAddress, event:int) -> None:
    touch(host.src, event); touch(host.index, event); touch(host.dst, event)
  for slot in constant_slots.values(): touch(RKArg(RKBufferKind.SCRATCH, slot), 0)
  event = 1
  for gather in image.gathers: touch_gather(gather, event); event += 1
  for host in image.host_gathers: touch_host(host, event); event += 1
  mid_by_point:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers:
    mid_by_point.setdefault(gather.after, []).append(gather)
  for index,op in enumerate(image.ew_ops):
    for gather in mid_by_point.get(index, ()): touch_gather(gather, event); event += 1
    touch(op.lhs, event); touch(op.rhs, event); touch(op.dst, event)
    event += 1
  for gather in mid_by_point.get(len(image.ew_ops), ()): touch_gather(gather, event); event += 1
  for host in image.host_scatters: touch_host(host, event); event += 1
  if not order: return replace(image, scratch=(), constants=b"")
  # Mid-program gathers may populate one logical slot in several partial phases. The runtime clears a
  # destination once per physical slot, so these stateful materialization slots must not alias.
  pinned = {gather.dst_index for gather in image.mid_gathers if gather.dst_kind is RKBufferKind.SCRATCH}
  intervals = ((start, ends[slot], slot) for start,slot in order)
  remap:dict[int, int] = {}
  physical:list[RKScratch] = []
  active:list[tuple[int, int]] = []
  available:list[int] = []
  for start,end,slot in intervals:
    while active and active[0][0] < start:
      _,target = heapq.heappop(active)
      heapq.heappush(available, target)
    spec = image.scratch[slot]
    if slot not in pinned and available:
      target = heapq.heappop(available)
      physical[target] = RKScratch(max(physical[target].size, spec.size))
    else:
      target = len(physical)
      physical.append(spec)
    if slot not in pinned: heapq.heappush(active, (end, target))
    remap[slot] = target
  physical_args = tuple(RKArg(RKBufferKind.SCRATCH, slot) for slot in range(len(physical)))
  def remap_arg(arg:RKArg) -> RKArg:
    if arg.kind is not RKBufferKind.SCRATCH: return arg
    return physical_args[remap[arg.index]] if not arg.addend else RKArg(arg.kind, remap[arg.index], arg.addend)
  def remap_gather(gather:RKGather) -> RKGather:
    return replace(gather,
    src_index=remap[gather.src_index] if not gather.values and gather.src_kind is RKBufferKind.SCRATCH else gather.src_index,
    dst_index=remap[gather.dst_index] if gather.dst_kind is RKBufferKind.SCRATCH else gather.dst_index)
  def remap_host(host:RKHostAddress) -> RKHostAddress:
    return replace(host, src=remap_arg(host.src), index=remap_arg(host.index), dst=remap_arg(host.dst))
  gathers = tuple(remap_gather(gather) for gather in image.gathers)
  ew_ops = tuple(RKEWOp(remap_arg(op.dst), remap_arg(op.lhs), remap_arg(op.rhs), op.count, op.ew_cfg, op.submit_barrier,
    op.compare, op.stateful, op.int32_output, op.int32_input, op.bool_output, op.int16_output, op.int16_input) for op in image.ew_ops)
  by_slot:dict[int, bytes] = {}
  for bits,slot in constant_slots.items():
    target = remap[slot]
    if target in by_slot and by_slot[target] != bits: raise ValueError("overlapping scratch constants")
    by_slot[target] = bits
  constants = b"" if not by_slot else b"".join(by_slot.get(slot, b"\0\0") for slot in range(max(by_slot)+1))
  return replace(image, scratch=tuple(physical), constants=constants, gathers=gathers, ew_ops=ew_ops,
    mid_gathers=tuple(remap_gather(gather) for gather in image.mid_gathers),
    host_gathers=tuple(remap_host(host) for host in image.host_gathers),
    host_scatters=tuple(remap_host(host) for host in image.host_scatters))

@dataclass(frozen=True, slots=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True, slots=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.mid_gathers
  if any(not 0 <= g.after <= len(image.ew_ops) for g in image.mid_gathers):
    raise ValueError("invalid mid-gather split")
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, RKIMAGE_VERSION, len(image.scratch), len(gathers),
                               len(image.host_gathers), len(image.host_scatters),
                               len(image.ew_ops), len(image.constants), len(image.mid_gathers)))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size)
  for g in gathers:
    kind = 3 if g.partial else 2 if g.values else 1 if g.offsets else 0
    out += _GATHER.pack(g.dst_index, g.src_index, g.count, kind, len(g.axes), g.itemsize, int(g.dst_kind), int(g.src_kind),
                        g.base, g.fill_bits, g.dst_stride, g.dst_addend, g.after)
    if kind == 2: out += struct.pack(f"<{g.count}{_ITEM_FORMAT[g.itemsize]}", *g.values)
    elif kind in (1, 3): out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for host in image.host_gathers + image.host_scatters:
    if host.itemsize not in _ITEM_FORMAT or host.index_itemsize not in (2, 4): raise ValueError("invalid RKHostAddress item size")
    out += _HOST_ADDRESS.pack(int(host.src.kind), int(host.index.kind), int(host.dst.kind), host.itemsize, host.index_itemsize,
      host.src.index, host.index.index, host.dst.index, host.count, host.src_count, host.dst_count, host.fill_bits,
      host.index_limit, host.src.addend, host.index.addend, host.dst.addend,
      host.base, host.index_scale, host.lane_stride)
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
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, nscratch, ngather, nhost_gather, nhost_scatter, nop, nconst, mid_count = \
    _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or mid_count > ngather:
    raise ValueError("invalid RKImage header")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_index, src_index, count, kind, naxes, itemsize, dst_kind, src_kind, base, fill_bits, dst_stride, dst_addend, after = \
      _GATHER.unpack_from(blob, off); off += _GATHER.size
    if (kind not in (0, 1, 2, 3) or (kind and naxes) or itemsize not in _ITEM_FORMAT or dst_kind not in (0, 1) or src_kind not in (0, 1) or
        dst_stride < 1 or dst_addend < 0): raise ValueError("invalid RKGather")
    if kind == 2:
      values = struct.unpack_from(f"<{count}{_ITEM_FORMAT[itemsize]}", blob, off); off += itemsize*count
      gathers.append(RKGather(src_index, dst_index, count, fill_bits=fill_bits, values=values,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind), after=after))
    elif kind in (1, 3):
      offsets = struct.unpack_from(f"<{count}i", blob, off); off += 4*count
      gathers.append(RKGather(src_index, dst_index, count, offsets=offsets, fill_bits=fill_bits, partial=kind == 3,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind), after=after))
    else:
      axes = tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)); off += naxes*_GATHER_AXIS.size
      gathers.append(RKGather(src_index, dst_index, count, base, axes, fill_bits=fill_bits,
                              dst_stride=dst_stride, dst_addend=dst_addend, dst_kind=RKBufferKind(dst_kind), itemsize=itemsize,
                              src_kind=RKBufferKind(src_kind), after=after))
  host_addresses:list[RKHostAddress] = []
  for _ in range(nhost_gather+nhost_scatter):
    src_kind, index_kind, dst_kind, itemsize, index_itemsize, src_index, index_index, dst_index, count, src_count, dst_count, \
      fill_bits, index_limit, src_addend, index_addend, dst_addend, base, index_scale, lane_stride = \
      _HOST_ADDRESS.unpack_from(blob, off)
    off += _HOST_ADDRESS.size
    if (src_kind not in (0, 1) or index_kind not in (0, 1) or dst_kind not in (0, 1) or itemsize not in _ITEM_FORMAT or
        index_itemsize not in (2, 4) or min(count, src_count, dst_count, index_limit) < 0):
      raise ValueError("invalid RKHostAddress")
    host_addresses.append(RKHostAddress(RKArg(RKBufferKind(src_kind), src_index, src_addend),
      RKArg(RKBufferKind(index_kind), index_index, index_addend), RKArg(RKBufferKind(dst_kind), dst_index, dst_addend),
      count, src_count, dst_count, itemsize, index_itemsize, fill_bits, index_limit, base, index_scale, lane_stride))
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
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  pre_count = ngather-mid_count
  if any(not 0 <= gather.after <= nop for gather in gathers[pre_count:]): raise ValueError("invalid mid-gather split")
  return RKImage(scratch, blob[off:], tuple(gathers[:pre_count]), tuple(ew_ops),
                 tuple(gathers[pre_count:]), tuple(host_addresses[:nhost_gather]), tuple(host_addresses[nhost_gather:]))

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for reloc in stage.relocs:
    word = commands[reloc.word]
    value = (address(reloc.arg.kind, reloc.arg.index) + reloc.arg.addend) & 0xffffffff
    commands[reloc.word] = (word & ~0xffffffff0000) | (value << 16)
  return tuple(commands)

_DPU, _RDMA = 0x1001, 0x2001
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_MAX_MAPPED_DOT_SCRATCH_BYTES = 352 << 20
_MAX_GENERIC_UNROLL = 1 << 14
_MIN_GENERIC_PRODUCT_RESIDUAL_TERMS = 64
_MAX_GENERIC_EXPANDED_NODES = 1 << 20
_MAX_OPTIONAL_RECIPE_NODES = 4096
_MAX_STATIC_LOCAL_STEPS = 1 << 20
_MAX_STATIC_RANGE_ENVS = 1 << 18
_EW_ELEMS_32BIT = 8*dtypes.half.itemsize//dtypes.float.itemsize
_FP16_EXACT_INTEGER = 1 << 11
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
_EW_CFG_RELU6 = _EW_CFG_COMMON | _EW_RELUX_EN
_EW_CFG_MIN = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_MIN
_EW_CFG_ABS = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ABS
_EW_CFG_NEG = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_NEG
_EW_CFG_FLOOR = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_FLOOR
_EW_CFG_CEIL = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_CEIL
_EW_CFG_LEAKY_RELU = _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_MUL_PRELU | _EW_OP_TYPE_MUL
_EW_STAGE_FP32_OUT = 1 << 29  # software tags consumed before writing EW_CFG
_EW_STAGE_FP32_IN = 1 << 30
_DPU_DATA_FORMAT_FP16 = (2<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_OUT = (5<<29)|(2<<26)|2
_DPU_DATA_FORMAT_FP32_IN = (2<<29)|(5<<26)|2
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
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_FLOOR, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN,
 _NATIVE_POSITIVE_MASK, _NATIVE_PRECISE_ADD, _NATIVE_RAW_MIN, _NATIVE_RELU6, _NATIVE_SIGN) = (
   "rockchip_abs", "rockchip_ceil", "rockchip_floor", "rockchip_leaky_relu", "rockchip_mask_mul",
   "rockchip_min", "rockchip_positive_mask", "rockchip_precise_add", "rockchip_raw_min", "rockchip_relu6", "rockchip_sign")
_EW_RELUX_CMP_RELU6 = struct.unpack("<I", struct.pack("<f", 6.0))[0]
_EW_CFG = {
  Ops.ADD: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_ADD,
  Ops.SUB: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_ALU_SUB,
  Ops.MUL: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_OP_TYPE_MUL,
  Ops.MAX: _EW_CFG_COMMON | _EW_RELU_BYPASS,
  Ops.FDIV: _EW_CFG_COMMON | _EW_RELU_BYPASS | _EW_OP_CVT_BYPASS | _EW_ALU_FDIV,
}
_INT16_EW = dict(int16_input=True, int16_output=True)
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)
def _scratch_arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)
def _reduction_stride(count:int) -> int: return round_up(count*2, 64)
def _int32_tiles_bytes(count:int) -> int: return ceildiv(count, 4) * 64
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff
def _int16_low_bytes(source:RKArg, out_slot:int, count:int, stride:int=2) -> RKGather:
  return RKGather(source.index, out_slot, count, base=source.addend, axes=((1, count, stride),),
                  dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)

def _finish_ew_stage(regs:tuple[tuple[int, int, int], ...], dst:RKArg, lhs:RKArg, rhs:RKArg, rdma_feature:int) -> RKStage:
  commands = [_cmd(*x) for x in regs]
  bindings = ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
              (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs))
  relocs = tuple(RKReloc(len(commands)+i, arg) for i,(_,_,arg) in enumerate(bindings))
  commands.extend(_cmd(target, reg, 0) for target,reg,_ in bindings)
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature))
  return RKStage(tuple(commands), relocs)

def _emit_stateful_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                         int32_output:bool=False, int32_input:bool=False,
                         int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False, fp32_input:bool=False) -> RKStage:
  """Emit a self-contained DPU EW stage, optionally consuming or producing native integers."""
  native_int16, native_int32 = int16_input and int16_output, int32_input and int32_output
  int16_to_int32 = int16_input and int32_output and not int16_output and not int32_input
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if native_int32 else \
          _EW_ELEMS_32BIT if int32_output or int32_input or fp32_output or fp32_input else _MAX_EW_ELEMS_FP16
  if not 0 < count <= limit:
    raise ValueError(f"stateful EW count {count} out of range")
  lanes = 4 if int32_input or fp32_input else 8
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
    (_DPU,rk.REG_DPU_DATA_FORMAT,_DPU_DATA_FORMAT_FP32_OUT if fp32_output else _DPU_DATA_FORMAT_FP32_IN if fp32_input else
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
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input or fp32_input else 2)<<2)))
  rdma_precision = 5 if fp32_input else 4 if int32_input else 1 if int16_input else 2
  rdma_feature = (rdma_precision<<15)|(15<<11)|(rdma_precision<<5)|(0 if is_div or int16_input or fp32_input else 1<<3)|1
  return _finish_ew_stage(regs, dst, lhs, rhs, rdma_feature)

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                  stateful:bool=False, int32_output:bool=False, int32_input:bool=False,
                  int16_output:bool=False, int16_input:bool=False) -> RKStage:
  """Build one DPU EW command body without its PC-chain tail."""
  if ew_cfg & _EW_STAGE_FP32_OUT:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_OUT, fp32_output=True)
  if ew_cfg & _EW_STAGE_FP32_IN:
    return _emit_stateful_stage(dst, lhs, rhs, count, ew_cfg & ~_EW_STAGE_FP32_IN, fp32_input=True)
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
  return _finish_ew_stage(regs, dst, lhs, rhs, (2<<15)|(15<<11)|(2<<5)|(0 if is_div else 1<<3)|1)

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
  if dtype.scalar() is dtypes.half:
    try: return struct.unpack("<e", struct.pack("<e", float(value)))[0]
    except OverflowError: return math.copysign(math.inf, float(value))
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

def _eval_int(u:UOp, env:dict[UOp, int]) -> int: return int(_eval_expr(u, env, {}))

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

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX}
def _is_static_expr(u:UOp) -> bool:
  return u.op in _STATIC_OPS and all(_is_static_expr(x) for x in u.src)

def _index_ranges(index:UOp) -> list[UOp]:
  """Ranges used as index values, excluding AFTER/END ordering dependencies attached to a RANGE."""
  ranges:list[UOp] = []
  def walk(u:UOp) -> None:
    if u.op in (Ops.RANGE, Ops.SPECIAL):
      if u not in ranges: ranges.append(u)
      return
    for src in u.src: walk(src)
  walk(index)
  return ranges

RKOutput = tuple[UOp, UOp, int, UOp, UOp]
def _output_store(uops:list[UOp], dtype:DType|tuple[DType, ...], *, allow_local:bool=False) \
                  -> RKOutput|None:
  """Return the single statically-sized output store shared by specialized graph matchers."""
  stores = [u for u in uops if u.op is Ops.STORE]
  outputs = [(store, root) for store in stores if (root:=_root_param(store.src[0])) is not None]
  if len(outputs) != 1 or not allow_local and len(stores) != 1:
    return None
  store, out_param = outputs[0]
  accepted = dtype if isinstance(dtype, tuple) else (dtype,)
  if out_param.dtype.scalar() not in accepted or out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None
  return store, out_param, int(out_param.src[0].arg), store.src[0].src[1], store.src[1]

def _iter_range_env(ranges:list[UOp], max_envs:int=_MAX_STATIC_RANGE_ENVS) -> list[dict[UOp, int]]:
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
    bound = int(r.src[0].arg)
    if bound < 0 or bound and len(envs) > max_envs//bound: raise RuntimeError("RKPLAN_REJECT:static_index_budget")
    envs = [{**env, r: i} for env in envs for i in range(bound)]
  return envs

def _iter_selected_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  """Enumerate only the selected structural axes, preserving dependent output axes as vector lanes."""
  envs:list[dict[UOp, int]] = [{}]
  for r in ranges:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    envs = [{**env, r:i} for env in envs for i in range(int(r.src[0].arg))]
  return envs

def _static_vector_env(out_index:UOp, exprs:tuple[UOp, ...]) -> tuple[list[dict[UOp, int]], dict[UOp, np.ndarray], dict[UOp, np.ndarray]]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for expr in exprs for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  envs = _iter_range_env(ranges)
  return envs, {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}, {}

def _static_values(out_index:UOp, expr:UOp, count:int, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
  envs, vector_env, cache = _static_vector_env(out_index, (expr,))
  dst_lanes = np.broadcast_to(_eval_vector(out_index, vector_env, cache), len(envs)).astype(np.int64)
  expr_lanes = np.broadcast_to(_eval_vector(expr, vector_env, cache), len(envs))
  values:list[int|None] = [None] * count
  for dst,raw in zip(dst_lanes, expr_lanes):
    if not 0 <= dst < count: raise RuntimeError("RKPLAN_REJECT:static_index")
    value = encode(raw.item())
    if values[dst] not in (None, value): raise RuntimeError("RKPLAN_REJECT:static_index")
    values[dst] = value
  if any(x is None for x in values): raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(x for x in values if x is not None)

def _static_int_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  """Evaluate a compile-time integer expression in compact output order."""
  return _static_values(out_index, expr, count, int)

def _static_int_vectors(out_index:UOp, exprs:tuple[UOp, ...], count:int) -> tuple[tuple[int, ...], ...]:
  """Vector-evaluate static integer rows with one shared index-expression cache."""
  envs, vector_env, cache = _static_vector_env(out_index, exprs)
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

def _divided_affine_index(u:UOp) -> tuple[int, dict[tuple[UOp, int], int]]|None:
  """Represent static address arithmetic as a sum of scaled `range//divisor` terms."""
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() in (dtypes.int, dtypes.uint):
    return _divided_affine_index(u.src[0])
  if u.op in (Ops.RANGE, Ops.SPECIAL): return 0, {(u, 1):1}
  if (u.op is Ops.CDIV and len(u.src) == 2 and u.src[0].op in (Ops.RANGE, Ops.SPECIAL) and
      u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0):
    return 0, {(u.src[0], int(u.src[1].arg)):1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _divided_affine_index(u.src[0]), _divided_affine_index(u.src[1])
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, divided = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return divided[0]*scale, {term:coefficient*scale for term,coefficient in divided[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  terms = lhs[1].copy()
  for term,coefficient in rhs[1].items():
    if (merged:=terms.get(term, 0)+sign*coefficient): terms[term] = merged
    elif term in terms: del terms[term]
  return lhs[0]+sign*rhs[0], terms

class RKStaticIndexEvaluator:
  """Share one compile-time output RANGE materialization across related static gather plans."""
  def __init__(self, out_index:UOp, count:int):
    self.out_index, self.count, self.ranges = out_index, count, _index_ranges(out_index)
    self._vectors:tuple[dict[UOp, np.ndarray], np.ndarray]|None = None

  def _prepare(self) -> tuple[dict[UOp, np.ndarray], np.ndarray]:
    if self._vectors is not None: return self._vectors
    envs = _iter_range_env(self.ranges)
    vector_env = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in self.ranges}
    if (out_affine:=_affine_index(self.out_index)) is None:
      dst = np.broadcast_to(_eval_vector(self.out_index, vector_env, {}), len(envs)).astype(np.int64)
    else:
      dst = np.full(len(envs), out_affine[0], dtype=np.int64)
      for r,stride in out_affine[1].items(): dst += vector_env[r]*stride
    if np.any((dst < 0) | (dst >= self.count)): raise RuntimeError("RKPLAN_REJECT:gather_index")
    self._vectors = vector_env, dst
    return self._vectors

  def offsets(self, load_index:UOp, gate:UOp|None) -> tuple[int, ...]:
    if any(r not in self.ranges for r in _index_ranges(load_index) + ([] if gate is None else _index_ranges(gate))):
      raise RuntimeError("RKPLAN_REJECT:gather_index")
    vector_env, dst = self._prepare()
    cache:dict[UOp, np.ndarray] = {}
    src = np.broadcast_to(_eval_vector(load_index, vector_env, cache), len(dst)).astype(np.int64)
    values = src if gate is None else np.where(np.broadcast_to(_eval_vector(gate, vector_env, cache), len(dst)), src, -1)
    if np.any(values < -1): raise RuntimeError("RKPLAN_REJECT:gather_index")
    offsets = np.full(self.count, -2, dtype=np.int64)
    offsets[dst] = values
    if np.any(offsets == -2): raise RuntimeError("RKPLAN_REJECT:gather_index")
    return tuple(int(x) for x in offsets)

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> tuple[int, ...]:
  return RKStaticIndexEvaluator(out_index, count).offsets(load_index, gate)

def _contiguous_output(out_index:UOp, count:int) -> bool:
  """Prove that an affine output index covers every destination lane once."""
  ranges, affine = _index_ranges(out_index), _affine_index(out_index)
  if affine is None or affine[0] != 0 or set(affine[1]) != set(ranges): return False
  extent = 1
  for r,stride in sorted(affine[1].items(), key=lambda item:item[1]):
    if stride != extent or not r.src or r.src[0].op is not Ops.CONST or (limit:=int(r.src[0].arg)) <= 0: return False
    extent *= limit
  return extent == count

def _typed_load_offsets(load:UOp, dtype:DType, out_index:UOp, count:int, allow_fill:bool=False) -> tuple[UOp, tuple[int, ...]]|None:
  """Resolve one typed global load to bounded static offsets."""
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX: return None
  param = _root_param(load.src[0])
  if param is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None
  try: offsets = _gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count)
  except RuntimeError: return None
  if any(offset < (-1 if allow_fill else 0) or offset >= int(param.src[0].arg) for offset in offsets): return None
  return param, offsets

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0,
                 index_evaluator:RKStaticIndexEvaluator|None=None) -> RKGather:
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
  if gate is None and out_affine is not None and out_affine[0] == 0 and \
     (load_divided:=_divided_affine_index(load_index)) is not None:
    expected = 1
    for r,dst_stride in sorted(out_affine[1].items(), key=lambda item:item[1]):
      if dst_stride != expected or r.src[0].op is not Ops.CONST or (limit:=int(r.src[0].arg)) <= 0: break
      expected *= limit
    else:
      if expected == count and all(r in out_affine[1] and divisor <= int(r.src[0].arg)
                                   for r,divisor in load_divided[1]):
        divided_axes = tuple((out_affine[1][r]*divisor, (int(r.src[0].arg)+divisor-1)//divisor, stride)
                             for (r,divisor),stride in load_divided[1].items() if stride)
        return RKGather(src_index, dst_index, count, load_divided[0], divided_axes)
  offsets = (index_evaluator or RKStaticIndexEvaluator(out_index, count)).offsets(load_index, gate)
  return RKGather(src_index, dst_index, count, offsets=offsets, fill_bits=fill_bits)

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

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is not Ops.MAX or u.dtype.scalar() is not dtypes.half: return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None


def _precise_add(lhs:UOp, rhs:UOp) -> UOp: return UOp(Ops.ADD, lhs.dtype, src=(lhs, rhs), arg=_NATIVE_PRECISE_ADD)
def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return _precise_add(lhs, UOp(Ops.MUL, dtypes.half, src=(rhs, neg_one)))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = UOp(Ops.MUL, dtypes.half, src=(x, splitter))
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(UOp(Ops.MUL, dtypes.half, src=(lhs_high, rhs_high)), term, neg_one)
  error = _precise_add(_precise_add(error, UOp(Ops.MUL, dtypes.half, src=(lhs_high, rhs_low))), UOp(Ops.MUL, dtypes.half, src=(lhs_low, rhs_high)))
  return term, _precise_add(error, UOp(Ops.MUL, dtypes.half, src=(lhs_low, rhs_low)))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = _precise_add(lhs, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  return total, _precise_add(lhs_error, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_add_parts(terms:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 addition residuals as a high lane plus a low correction lane."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  high, middle, low = terms[0], zero, zero
  for part in terms[1:]:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = _precise_add(low, error)
  return high, _precise_add(middle, low)

def _precise_sum_parts(terms:list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 product and addition residuals as a high lane plus a low correction lane."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  return _precise_add_parts(tuple(x[0] for x in pairs) + tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL))

def _tag_precise_adds(root:UOp) -> UOp:
  """Mark additions inside an already compensated physical recipe so they are not expanded again."""
  tagged:dict[UOp, UOp] = {}
  for node in root.toposort():
    tagged[node] = node.replace(src=tuple(tagged[src] for src in node.src),
      arg=_NATIVE_PRECISE_ADD if node.op is Ops.ADD and node.arg is None else node.arg)
  return tagged[root]

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  high, middle = _precise_sum_parts(list(_tag_precise_adds(UOp(Ops.SINK, src=tuple(terms))).src))
  return _precise_add(high, middle)

def _finish_mapped_add_reduction(mapped:RKImage, out_slot:int, rows:int, groups:int, post_scale:float,
                                 op_barriers:bool=False, compensated_limit:int=_reduction_stride(1)//2, kahan:bool=False) -> RKImage|None:
  """Retarget a vector map image into scratch, then append a row-wise ADD reduction."""
  if not mapped.ew_ops or any(gather.after == len(mapped.ew_ops) for gather in mapped.mid_gathers): return None
  lanes = rows*groups
  scratch_shift, value_slot = 1, len(mapped.scratch)+1
  def remap_arg(arg:RKArg) -> RKArg:
    if arg.kind is RKBufferKind.SCRATCH: return replace(arg, index=arg.index+scratch_shift)
    return replace(arg, kind=RKBufferKind.SCRATCH, index=value_slot) if arg.index == out_slot else arg
  def remap_gather(gather:RKGather) -> RKGather:
    src = remap_arg(RKArg(gather.src_kind, gather.src_index))
    dst = remap_arg(RKArg(gather.dst_kind, gather.dst_index))
    return replace(gather, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  pre_ops = tuple(replace(op, dst=remap_arg(op.dst), lhs=remap_arg(op.lhs), rhs=remap_arg(op.rhs)) for op in mapped.ew_ops)
  gathers = tuple(remap_gather(gather) for gather in mapped.gathers)
  host_gathers = tuple(replace(host, src=remap_arg(host.src), index=remap_arg(host.index), dst=remap_arg(host.dst))
                       for host in mapped.host_gathers)
  host_scatters = tuple(replace(host, src=remap_arg(host.src), index=remap_arg(host.index), dst=remap_arg(host.dst))
                        for host in mapped.host_scatters)
  ops = list(pre_ops)
  outer, inner = rows, groups
  stride, arena_slot = _reduction_stride(outer), value_slot+1
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, arena_slot, offset)
  scaled_slot = arena_slot+1
  mid = tuple(RKGather(value_slot, arena_slot, rows, base=group*rows, axes=((1, rows, 1),),
                       dst_addend=group*(stride//2), src_kind=RKBufferKind.SCRATCH) for group in range(groups))
  destination = RKArg(RKBufferKind.ARG, out_slot) if post_scale == 1.0 else RKArg(RKBufferKind.SCRATCH, scaled_slot)
  if kahan:
    def temporary(index:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, value_slot, index*stride)
    reduced = _kahan_add(ops, [group*stride for group in range(inner)], outer, arena, temporary, destination,
                         op_barriers=op_barriers)
  elif 2 <= inner <= compensated_limit:
    def temporary(index:int) -> RKArg: return RKArg(RKBufferKind.SCRATCH, value_slot, index*stride)
    reduced = _compensated_add(ops, [group*stride for group in range(inner)], outer, arena, temporary, destination,
                               op_barriers=op_barriers)
  else:
    reduced = _reduce_arena(ops, [group*stride for group in range(inner)], outer, _EW_CFG[Ops.ADD], arena, destination,
                            op_barriers=op_barriers)
  if post_scale != 1.0:
    ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, RKArg(RKBufferKind.SCRATCH, 0), rows, _EW_CFG[Ops.MUL],
                      submit_barrier=op_barriers, stateful=op_barriers))
  scratch = (RKScratch(_scratch_bytes(lanes)), *mapped.scratch, RKScratch(_scratch_bytes(lanes)), RKScratch(inner*stride))
  if post_scale != 1.0: scratch += (RKScratch(_scratch_bytes(rows)),)
  mapped_mid = tuple(remap_gather(gather) for gather in mapped.mid_gathers)
  mid = mapped_mid+tuple(replace(gather, after=len(pre_ops)) for gather in mid)
  return RKImage(scratch, struct.pack("<e", post_scale)+mapped.constants,
                 gathers=gathers, ew_ops=tuple(ops), mid_gathers=mid,
                 host_gathers=host_gathers, host_scatters=host_scatters)


def _append_inplace_image(first:RKImage, second:RKImage) -> RKImage|None:
  """Append an in-place EW image, scheduling its input materialization after the first image completes."""
  if not second.ew_ops or second.host_gathers or second.host_scatters or \
     any(gather.after == len(first.ew_ops) for gather in first.mid_gathers): return None
  first_constants, second_constants = len(first.constants)//2, len(second.constants)//2
  def first_arg(arg:RKArg) -> RKArg:
    return replace(arg, index=arg.index+second_constants) \
      if arg.kind is RKBufferKind.SCRATCH and arg.index >= first_constants else arg
  def second_arg(arg:RKArg) -> RKArg:
    if arg.kind is not RKBufferKind.SCRATCH: return arg
    return replace(arg, index=first_constants+arg.index if arg.index < second_constants else len(first.scratch)+arg.index)
  def first_gather(gather:RKGather) -> RKGather:
    src, dst = first_arg(RKArg(gather.src_kind, gather.src_index)), first_arg(RKArg(gather.dst_kind, gather.dst_index))
    return replace(gather, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  first_ops = tuple(replace(op, dst=first_arg(op.dst), lhs=first_arg(op.lhs), rhs=first_arg(op.rhs)) for op in first.ew_ops)
  second_ops = [replace(op, dst=second_arg(op.dst), lhs=second_arg(op.lhs), rhs=second_arg(op.rhs)) for op in second.ew_ops]
  second_ops[0] = replace(second_ops[0], submit_barrier=True, stateful=True)
  def second_gather(gather:RKGather, after:int) -> RKGather:
    src, dst = second_arg(RKArg(gather.src_kind, gather.src_index)), second_arg(RKArg(gather.dst_kind, gather.dst_index))
    return replace(gather, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index, after=after)
  split = len(first_ops)
  second_mid = tuple(second_gather(gather, split) for gather in second.gathers)+tuple(
    second_gather(gather, split+gather.after) for gather in second.mid_gathers)
  scratch = (first.scratch[:first_constants] + second.scratch[:second_constants] + first.scratch[first_constants:] +
             second.scratch[second_constants:])
  return RKImage(scratch, first.constants+second.constants,
                 gathers=tuple(first_gather(gather) for gather in first.gathers), ew_ops=first_ops+tuple(second_ops),
                 mid_gathers=tuple(first_gather(gather) for gather in first.mid_gathers)+second_mid)

def _lower_mapped_add_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Evaluate one fused FP16 map over the whole reduction domain, then reduce its materialized lanes."""
  if (output:=_output_store(uops, dtypes.half, allow_local=True)) is None: return None
  store, out, rows, out_index, root = output
  nodes, out_ranges = list(root.toposort()), _index_ranges(out_index)
  reduce_ranges = [u for u in nodes if u.op is Ops.RANGE and u not in out_ranges]
  if not reduce_ranges or any(r.src[0].op is not Ops.CONST or int(r.src[0].arg) <= 0 for r in reduce_ranges): return None
  try: envs = _iter_range_env(out_ranges)
  except RuntimeError: return None
  if len(envs) != rows or tuple(_eval_int(out_index, env) for env in envs) != tuple(range(rows)): return None
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and any(r in u.toposort() for r in reduce_ranges)]
  if len(updates) != 1: return None
  value, post_root, post_local = root, None, None
  if _local_load(value) is not None: post_scale = 1.0
  elif value.op is Ops.MUL and (load:=next((x for x in value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=value.src[1 if value.src[0] is load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else:
    # Keep the reduction structural, then feed its materialized result through the ordinary UOp executor.  Select the
    # highest CAST boundary around the local accumulator so physical FP16 output is not reinterpreted as local FP32.
    local_refs = [u for u in root.toposort() if _local_load(u) is not None]
    if not local_refs: return None
    post_local = next((u for u in local_refs if u.dtype.scalar() is out.dtype.scalar()), local_refs[-1])
    post_root, post_scale = root, 1.0
  if not math.isfinite(post_scale): return None
  update = _strip_cast(updates[0].src[1])
  if update.op is not Ops.ADD or (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None:
    return None
  term = update.src[1 if update.src[0] is acc else 0]
  groups, flat = 1, UOp.const(0, out_index.dtype)
  for reduce_range in reduce_ranges[::-1]:
    flat = flat.alu(Ops.ADD, reduce_range.alu(Ops.MUL, reduce_range.const_like(groups)))
    groups *= int(reduce_range.src[0].arg)
  lanes, out_slot = rows*groups, out.arg.slot
  linear_index = flat.alu(Ops.MUL, flat.const_like(rows)).alu(Ops.ADD, out_index)
  fake_out = out.replace(src=(out.src[0].const_like(lanes),))
  fake_index = store.src[0].replace(src=(fake_out, linear_index))
  fake_store = store.replace(src=(fake_index, term, *store.src[2:]))
  map_uops = _fp16_rewrite(list(UOp(Ops.SINK, src=(fake_store,)).toposort()))
  mapped = _lower_uop_program(map_uops, vectorize_reductions=False, recipes_ready=True)
  if mapped is None: return None
  reduced = _finish_mapped_add_reduction(mapped, out_slot, rows, groups, post_scale)
  if reduced is None or post_root is None or post_local is None: return reduced
  fake_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out_slot)
  fake_source = UOp.param(fake_slot, dtypes.half, (rows,))
  range_substitutions = {axis:axis.replace(src=(axis.src[0],)) for axis in _index_ranges(out_index) if len(axis.src) > 1}
  post_index = store.src[0].substitute(range_substitutions) if range_substitutions else store.src[0]
  fake_index = post_index.replace(src=(fake_source, *post_index.src[1:]))
  post_load = UOp(Ops.LOAD, dtypes.half, src=(fake_index,))
  if post_local.dtype.scalar() is not dtypes.half: post_load = post_load.cast(post_local.dtype.scalar())
  post_value = post_root.substitute({post_local:post_load, **range_substitutions})
  post_store = store.replace(src=(post_index, post_value, *store.src[2:]))
  post_uops = _fp16_rewrite(list(UOp(Ops.SINK, src=(post_store,)).toposort()))
  post = _lower_uop_program(post_uops, vectorize_reductions=False, recipes_ready=True)
  if post is None: return None
  def alias(arg:RKArg) -> RKArg:
    return replace(arg, index=out_slot) if arg.kind is RKBufferKind.ARG and arg.index == fake_slot else arg
  def alias_gather(gather:RKGather) -> RKGather:
    src, dst = alias(RKArg(gather.src_kind, gather.src_index)), alias(RKArg(gather.dst_kind, gather.dst_index))
    return replace(gather, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  post = replace(post, gathers=tuple(alias_gather(gather) for gather in post.gathers),
    ew_ops=tuple(replace(op, dst=alias(op.dst), lhs=alias(op.lhs), rhs=alias(op.rhs)) for op in post.ew_ops),
    mid_gathers=tuple(alias_gather(gather) for gather in post.mid_gathers))
  appended = _append_inplace_image(reduced, post)
  return appended


def _lower_vectorized_mul_add_reduction(uops:list[UOp]) -> RKImage|None:
  """Execute repeated FP16 MUL UOps with product residuals, then compensate their physical ADD reduction."""
  if (output:=_output_store(uops, dtypes.half)) is None: return None
  _, out, rows, out_index, root = output
  bias:UOp|None = None
  summed, post_scale, relu = root, 1.0, False
  if root.op is Ops.WHERE and len(root.src) == 3 and root.src[0].op is Ops.CMPLT and \
     len(root.src[0].src) == 2 and root.src[0].src[0].op is Ops.CONST and float(root.src[0].src[0].arg) == 0.0 and \
     root.src[0].src[1].key == root.src[1].key and root.src[2].op is Ops.CONST and float(root.src[2].arg) == 0.0:
    summed, relu = root.src[1], True
  if summed.op is Ops.ADD:
    for dot,candidate in (summed.src, summed.src[::-1]):
      candidate = _strip_cast(candidate)
      if (candidate.op is Ops.LOAD and candidate.dtype.scalar() is dtypes.half) or \
         (candidate.dtype.scalar() in (dtypes.half, dtypes.float) and _is_static_expr(candidate)):
        summed, bias = dot, candidate; break
  while True:
    if summed.op is Ops.CAST: summed = summed.src[0]; continue
    value, scale = next(((a, b) for a,b in (summed.src, summed.src[::-1]) if b.op is Ops.CONST), (None, None)) \
      if summed.op is Ops.MUL else (None, None)
    if value is None or scale is None: break
    post_scale *= float(scale.arg); summed = value
  if summed.op is not Ops.ADD or not math.isfinite(post_scale): return None
  terms = tuple(_strip_cast(term) for term in _flatten_binary(summed, Ops.ADD))
  groups, lanes = len(terms), rows*len(terms)
  chunk_lanes = _MAX_EW_ELEMS_FP16*(_reduction_stride(1)//2)
  if groups != 8 and (groups < _MIN_GENERIC_PRODUCT_RESIDUAL_TERMS or groups < 256 and rows <= _MAX_EW_ELEMS_FP16) or \
     8*_scratch_bytes(lanes)+_scratch_bytes(min(chunk_lanes, lanes)) > _MAX_MAPPED_DOT_SCRATCH_BYTES: return None

  index_evaluator = RKStaticIndexEvaluator(out_index, rows)
  parsed:list[tuple[tuple[UOp, RKGather], tuple[UOp, RKGather]]] = []
  for term in terms:
    if term.op is not Ops.MUL or term.arg is not None: return None
    operands:list[tuple[UOp, RKGather]] = []
    for src in term.src:
      load = _strip_cast(src)
      if (load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or not load.src or load.src[0].op is not Ops.INDEX or
          (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtypes.half or
          not param.src or param.src[0].op is not Ops.CONST or len(load.src) > 1 and load.src[1].op is not Ops.CONST): return None
      gate = load.src[2] if len(load.src) > 2 else None
      fill_bits = _fp16_bits(load.src[1].arg if len(load.src) > 1 else 0)
      try:
        plan = _gather_plan(param.arg.slot, 0, out_index, load.src[0].src[1], gate, rows, fill_bits, index_evaluator)
        _validate_gather_bounds(plan, int(param.src[0].arg))
      except RuntimeError: return None
      operands.append((param, plan))
    parsed.append((operands[0], operands[1]))
  slots = (parsed[0][0][0].arg.slot, parsed[0][1][0].arg.slot)
  normalized:list[tuple[tuple[UOp, RKGather], tuple[UOp, RKGather]]] = []
  for pair in parsed:
    pair_slots = (pair[0][0].arg.slot, pair[1][0].arg.slot)
    if pair_slots == slots: normalized.append(pair)
    elif pair_slots[::-1] == slots: normalized.append((pair[1], pair[0]))
    else: return None
  gathers = tuple(replace(operand[1], dst_index=side+1, dst_addend=group*rows)
                  for group,pair in enumerate(normalized) for side,operand in enumerate(pair))
  mapped_ops:list[RKEWOp] = []
  splitter = RKArg(RKBufferKind.SCRATCH, 0)
  for start in range(0, lanes, chunk_lanes):
    count, offset = min(chunk_lanes, lanes-start), start*2
    lhs, rhs = RKArg(RKBufferKind.SCRATCH, 1, offset), RKArg(RKBufferKind.SCRATCH, 2, offset)
    lhs_high, rhs_high = RKArg(RKBufferKind.SCRATCH, 3, offset), RKArg(RKBufferKind.SCRATCH, 4, offset)
    product = RKArg(RKBufferKind.ARG, out.arg.slot, offset)
    error = RKArg(RKBufferKind.ARG, out.arg.slot, lanes*2+offset)
    stages = ((product, lhs, rhs, Ops.MUL),
              (lhs_high, lhs, splitter, Ops.MUL), (rhs_high, lhs_high, lhs, Ops.SUB),
              (lhs_high, lhs_high, rhs_high, Ops.SUB), (lhs, lhs, lhs_high, Ops.SUB),
              (rhs_high, rhs, splitter, Ops.MUL), (error, rhs_high, rhs, Ops.SUB),
              (rhs_high, rhs_high, error, Ops.SUB), (rhs, rhs, rhs_high, Ops.SUB),
              (error, lhs_high, rhs_high, Ops.MUL), (error, error, product, Ops.SUB),
              (lhs_high, lhs_high, rhs, Ops.MUL), (error, error, lhs_high, Ops.ADD),
              (rhs_high, lhs, rhs_high, Ops.MUL), (error, error, rhs_high, Ops.ADD),
              (lhs, lhs, rhs, Ops.MUL), (error, error, lhs, Ops.ADD))
    mapped_ops.extend(RKEWOp(dst, left, right, count, _EW_CFG[op], submit_barrier=i == 0 and bool(start), stateful=True)
                      for i,(dst,left,right,op) in enumerate(stages))
  mapped = RKImage(
    (RKScratch(_scratch_bytes(min(chunk_lanes, lanes))), *(RKScratch(_scratch_bytes(lanes)) for _ in range(4))),
    struct.pack("<e", 65.0), gathers=gathers, ew_ops=tuple(mapped_ops))
  finished = _finish_mapped_add_reduction(mapped, out.arg.slot, rows, groups*2, post_scale,
                                           op_barriers=True, compensated_limit=groups*2, kahan=groups == 8)
  if finished is None: return None
  if bias is not None:
    if bias.op is Ops.LOAD:
      bias_param = _root_param(bias.src[0]) if bias.src and bias.src[0].op is Ops.INDEX else None
      if bias_param is None or bias_param.src[0].op is not Ops.CONST: return None
      try: bias_offsets = index_evaluator.offsets(bias.src[0].src[1], bias.src[2] if len(bias.src) > 2 else None)
      except RuntimeError: return None
      if any(not 0 <= offset < int(bias_param.src[0].arg) for offset in bias_offsets): return None
      if int(bias_param.src[0].arg) == rows and bias_offsets == tuple(range(rows)):
        bias_arg = RKArg(RKBufferKind.ARG, bias_param.arg.slot)
      else:
        bias_arg = RKArg(RKBufferKind.SCRATCH, len(finished.scratch))
        finished = replace(finished, scratch=(*finished.scratch, RKScratch(_scratch_bytes(rows))),
                           gathers=(*finished.gathers, RKGather(bias_param.arg.slot, bias_arg.index, rows, offsets=bias_offsets)))
    else:
      try: values = _static_values(out_index, bias, rows, _fp16_bits)
      except RuntimeError: return None
      bias_arg = RKArg(RKBufferKind.SCRATCH, len(finished.scratch))
      finished = replace(finished, scratch=(*finished.scratch, RKScratch(_scratch_bytes(rows))),
                         gathers=(*finished.gathers, RKGather(0, bias_arg.index, rows, values=values)))
    add_bias = RKEWOp(RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.ARG, out.arg.slot),
                      bias_arg, rows, _EW_CFG[Ops.ADD], submit_barrier=True, stateful=True)
    finished = replace(finished, ew_ops=(*finished.ew_ops, add_bias))
  if relu:
    relu_image = RKImage((RKScratch(_scratch_bytes(rows)),), struct.pack("<e", 0.0), ew_ops=(
      RKEWOp(RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.SCRATCH, 0),
             rows, _EW_CFG[Ops.MAX]),))
    if (finished:=_append_inplace_image(finished, relu_image)) is None: return None
  return finished

def _flatten_binary(root:UOp, op:Ops, *, plain:bool=False) -> list[UOp]:
  leaves, stack = [], [root]
  while stack:
    node = stack.pop()
    if node.op is op and (not plain or node.arg is None): stack.extend(reversed(node.src))
    else: leaves.append(node)
  return leaves

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64
  return vector_bytes, vector_bytes//2, rows*vector_bytes//2

def _stripe_gathers(src_slot:int, dst_slot:int, count:int, rows:Iterable[Iterable[int]], vector_lanes:int, *,
                    values:bool=False, itemsize:int=2) -> tuple[RKGather, ...]:
  """Pack candidate or repeated-current rows into one aligned lane matrix."""
  return tuple(RKGather(src_slot, dst_slot, count, offsets=() if values else tuple(row), values=tuple(row) if values else (),
                        dst_addend=i*vector_lanes, itemsize=itemsize) for i,row in enumerate(rows))

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int, int16:bool=False) -> RKArg:
  """Append a balanced row reduction, making its first dependent stage self-contained."""
  first = True
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      ops.append(RKEWOp(active[i], active[i], active[i+1], count, cfg, submit_barrier=first and not int16,
                        stateful=first and not int16,
                        int16_input=int16, int16_output=int16))
      first = False; reduced.append(active[i])
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return active[0]


def _ew_eq_mask(ops:list[RKEWOp], arg:Callable[[int], RKArg], lhs:int, rhs:int, temps:tuple[int, int, int, int], one:int,
                lanes:int) -> None:
  """Append SUB, ABS, nonzero comparison, and inversion for an FP16 equality mask."""
  diff, magnitude, unequal, equal = temps
  ops.extend((RKEWOp(arg(diff), arg(lhs), arg(rhs), lanes, _EW_CFG[Ops.SUB]),
              RKEWOp(arg(magnitude), arg(diff), arg(diff), lanes, _EW_CFG_ABS, submit_barrier=True, stateful=True),
              RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), lanes, _EW_CFG[Ops.MAX], compare=True),
              RKEWOp(arg(equal), arg(one), arg(unequal), lanes, _EW_CFG[Ops.SUB], stateful=True)))


def _ew_native_int16_eq_mask(ops:list[RKEWOp], allocate:Callable[[], RKArg], lhs:RKArg, rhs:RKArg,
                             one:RKArg, lanes:int) -> RKArg:
  """Compare native INT16 lanes whose subtraction is proven not to overflow."""
  diff, magnitude, unequal, equal = (allocate() for _ in range(4))
  ops.extend((RKEWOp(diff, lhs, rhs, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(magnitude, diff, diff, lanes, _EW_CFG_ABS, **_INT16_EW),
              RKEWOp(unequal, magnitude, one, lanes, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(equal, one, unequal, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
  return equal

def _fp16_high_and_nan(ops:list[RKEWOp], allocate:Callable[[], RKArg], high:RKArg, low:RKArg,
                       zero:RKArg, one:RKArg, const123:RKArg, const124:RKArg, const127:RKArg, const128:RKArg,
                       lanes:int) -> tuple[RKArg, RKArg]:
  """Canonicalize signed zero's FP16 high byte and classify NaNs with native INT16 byte arithmetic."""
  sign_delta, sign_positive, sign, sign_scale, magnitude = (allocate() for _ in range(5))
  ops.extend((RKEWOp(sign_delta, high, const127, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(sign_positive, sign_delta, zero, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(sign, sign_positive, one, lanes, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(sign_scale, sign, const128, lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(magnitude, high, sign_scale, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
  high_zero = _ew_native_int16_eq_mask(ops, allocate, magnitude, zero, one, lanes)
  low_zero = _ew_native_int16_eq_mask(ops, allocate, low, zero, one, lanes)
  zero_value, zero_sign, canonical = (allocate() for _ in range(3))
  exponent_delta, exponent_positive, exponent_all = (allocate() for _ in range(3))
  mantissa_delta, mantissa_positive, mantissa_high, mantissa_low, mantissa, nan = (allocate() for _ in range(6))
  ops.extend((RKEWOp(zero_value, high_zero, low_zero, lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(zero_sign, sign_scale, zero_value, lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(canonical, high, zero_sign, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(exponent_delta, magnitude, const123, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(exponent_positive, exponent_delta, zero, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(exponent_all, exponent_positive, one, lanes, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(mantissa_delta, magnitude, const124, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(mantissa_positive, mantissa_delta, zero, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(mantissa_high, mantissa_positive, one, lanes, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(mantissa_low, low, one, lanes, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(mantissa, mantissa_high, mantissa_low, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(nan, exponent_all, mantissa, lanes, _EW_CFG[Ops.MUL], **_INT16_EW)))
  return canonical, nan


RKCoordinateRows = tuple[tuple[int, ...], ...]

def _reduce_arena(ops:list[RKEWOp], active:list[int], count:int, cfg:int, arena:Callable[[int], RKArg],
                  out:RKArg|None=None, op_barriers:bool=False) -> RKArg:
  """Append a balanced in-place arena reduction and optionally write its final stage directly to output."""
  while len(active) > 1:
    reduced = []
    for i in range(0, len(active)-1, 2):
      lhs, rhs, final = active[i], active[i+1], len(active) == 2 and out is not None
      dst = out if final and out is not None else arena(lhs)
      ops.append(RKEWOp(dst, arena(lhs), arena(rhs), count, cfg,
                        submit_barrier=op_barriers and bool(ops), stateful=op_barriers))
      reduced.append(lhs)
    if len(active) & 1: reduced.append(active[-1])
    active = reduced
  return out if out is not None else arena(active[0])

def _compensated_add(ops:list[RKEWOp], active:list[int], count:int, arena:Callable[[int], RKArg],
                     temporary:Callable[[int], RKArg], out:RKArg, op_barriers:bool=False) -> RKArg:
  """Reduce aligned FP16 vectors with TwoSum residuals retained in consumed arena lanes."""
  errors:list[int] = []
  while len(active) > 1:
    reduced:list[int] = []
    for lhs,rhs in zip(active[::2], active[1::2]):
      total, delta = temporary(0), temporary(1)
      ops.extend(RKEWOp(dst, lhs_arg, rhs_arg, count, cfg, submit_barrier=op_barriers, stateful=op_barriers)
                 for dst,lhs_arg,rhs_arg,cfg in ((total, arena(lhs), arena(rhs), _EW_CFG[Ops.ADD]),
                   (delta, total, arena(lhs), _EW_CFG[Ops.SUB]), (arena(rhs), arena(rhs), delta, _EW_CFG[Ops.SUB]),
                   (delta, total, delta, _EW_CFG[Ops.SUB]), (arena(lhs), arena(lhs), delta, _EW_CFG[Ops.SUB]),
                   (arena(rhs), arena(lhs), arena(rhs), _EW_CFG[Ops.ADD]), (arena(lhs), total, total, _EW_CFG[Ops.MAX])))
      reduced.append(lhs); errors.append(rhs)
    if len(active)&1: reduced.append(active[-1])
    active = reduced
  residual = _reduce_arena(ops, errors, count, _EW_CFG[Ops.ADD], arena, op_barriers=op_barriers)
  ops.append(RKEWOp(out, arena(active[0]), residual, count, _EW_CFG[Ops.ADD],
                    submit_barrier=op_barriers, stateful=op_barriers))
  return out

def _kahan_add(ops:list[RKEWOp], active:list[int], count:int, arena:Callable[[int], RKArg],
               temporary:Callable[[int], RKArg], out:RKArg, op_barriers:bool=False) -> RKArg:
  """Accumulate physical lanes literally with Kahan correction for mixed-sign FP16 sums."""
  if not active: raise ValueError("empty Kahan reduction")
  total, correction, adjusted, updated = (temporary(i) for i in range(4))
  def emit(dst:RKArg, lhs:RKArg, rhs:RKArg, op:Ops) -> None:
    ops.append(RKEWOp(dst, lhs, rhs, count, _EW_CFG[op], submit_barrier=op_barriers, stateful=op_barriers))
  emit(total, arena(active[0]), arena(active[0]), Ops.MAX)
  emit(correction, arena(active[0]), arena(active[0]), Ops.SUB)
  for offset in active[1:]:
    emit(adjusted, arena(offset), correction, Ops.SUB)
    emit(updated, total, adjusted, Ops.ADD)
    emit(correction, updated, total, Ops.SUB)
    emit(correction, correction, adjusted, Ops.SUB)
    emit(total, updated, updated, Ops.MAX)
  emit(out, total, total, Ops.MAX)
  return out

def _int32_division_root(root:UOp) -> tuple[str, UOp, UOp]|None:
  """Recognize truncating quotient/remainder and Tinygrad's canonical floor corrections."""
  if root.op is Ops.CDIV and root.dtype.scalar() is dtypes.int: return "trunc", root.src[0], root.src[1]
  if root.op is Ops.CMOD and root.dtype.scalar() is dtypes.int: return "cmod", root.src[0], root.src[1]
  if root.op is not Ops.ADD or root.dtype.scalar() is not dtypes.int: return None
  divisions = [u for u in root.toposort() if u.op is Ops.CDIV and u.dtype.scalar() is dtypes.int]
  remainders = [u for u in root.toposort() if u.op is Ops.CMOD and u.dtype.scalar() is dtypes.int]
  def negative_operand(predicate:UOp) -> UOp|None:
    if predicate.op is not Ops.CMPLT or len(predicate.src) != 2: return None
    constants = [x for x in predicate.src if x.op is Ops.CONST and int(x.arg) == 0]
    return next((x for x in predicate.src if x.op is not Ops.CONST), None) if len(constants) == 1 else None
  def valid_sign_diff(predicate:UOp, operands:tuple[UOp, UOp]) -> bool:
    dynamic = [x for x in operands if x.op is not Ops.CONST]
    negative_constants = sum(x.op is Ops.CONST and int(x.arg) < 0 for x in operands)
    if predicate.op is Ops.CMPLT:
      return not negative_constants and len(dynamic) == 1 and (operand:=negative_operand(predicate)) is not None and operand.key == dynamic[0].key
    if predicate.op is not Ops.CMPNE: return False
    comparisons = [negative_operand(x) for x in predicate.src if x.op is Ops.CMPLT]
    bools = [x for x in predicate.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.bool]
    if len(dynamic) == 2:
      return len(comparisons) == 2 and {x.key for x in comparisons if x is not None} == {x.key for x in dynamic}
    return len(dynamic) == negative_constants == len(comparisons) == len(bools) == 1 and comparisons[0] is not None and \
      comparisons[0].key == dynamic[0].key and bool(bools[0].arg)
  if len(remainders) != 1: return None
  remainder = remainders[0]
  if len(divisions) == 1:
    division = divisions[0]
    if len(division.src) != 2: return None
    division_operands = (division.src[0], division.src[1])
    if tuple(x.key for x in division.src) != tuple(x.key for x in remainder.src): return None
    correction = next((x for x in root.src if x is not division), None)
    if correction is None or correction.op is not Ops.MUL: return None
    constants = [x for x in correction.src if x.op is Ops.CONST and x.dtype.scalar() is dtypes.int and int(x.arg) == -1]
    predicates = [x.src[0] for x in correction.src if x.op is Ops.CAST and x.dtype.scalar() is dtypes.int and len(x.src) == 1]
    if len(constants) != 1 or len(predicates) != 1 or predicates[0].op is not Ops.AND: return None
    terms = predicates[0].src
    nonzero = next((x for x in terms if x.op is Ops.CMPNE and any(y.key == remainder.key for y in x.src)), None)
    sign_diff = next((x for x in terms if x is not nonzero), None)
    return ("floor", *division_operands) if nonzero is not None and sign_diff is not None and \
      valid_sign_diff(sign_diff, division_operands) else None
  if len(remainder.src) != 2: return None
  remainder_operands = (remainder.src[0], remainder.src[1])
  correction = next((x for x in root.src if x is not remainder), None)
  if correction is None or correction.op is not Ops.WHERE or len(correction.src) != 3 or correction.src[0].op is not Ops.AND: return None
  condition, selected, zero = correction.src
  if selected.key != remainder.src[1].key or zero.op is not Ops.CONST or int(zero.arg) != 0: return None
  nonzero = next((x for x in condition.src if x.op is Ops.CMPNE and any(y.key == remainder.key for y in x.src)), None)
  sign_diff = next((x for x in condition.src if x is not nonzero), None)
  return ("floormod", *remainder_operands) if nonzero is not None and sign_diff is not None and \
    valid_sign_diff(sign_diff, remainder_operands) else None

def _lower_int32_division(output:RKOutput) -> RKImage|None:
  """Divide signed INT32 exactly with a byte-restoring divider on native INT16 EW."""
  _, out_param, count, out_index, root = output
  if not 1 <= count <= _MAX_EW_ELEMS_FP16 or (parsed_root:=_int32_division_root(root)) is None: return None
  mode, lhs, rhs = parsed_root
  operands:list[tuple[UOp|None, tuple[int, ...]|int]] = []
  for term in (lhs, rhs):
    if term.op is Ops.CONST and term.dtype.scalar() is dtypes.int:
      operands.append((None, int(term.arg)&0xffffffff))
    elif (parsed:=_typed_load_offsets(term, dtypes.int, out_index, count, allow_fill=True)) is not None:
      operands.append(parsed)
    else: return None
  sources = [param for param,_ in operands if param is not None]
  if not sources: return None

  stride, rows = _reduction_stride(count), 0
  def allocate() -> RKArg:
    nonlocal rows
    value = RKArg(RKBufferKind.SCRATCH, 0, rows*stride); rows += 1
    return value
  gathers:list[RKGather] = []
  raw_operands:list[tuple[RKArg, ...]] = []
  for param,spec in operands:
    raw = tuple(allocate() for _ in range(4)); raw_operands.append(raw)
    if param is None:
      value = typing_cast(int, spec)
      for byte,dst in enumerate(raw):
        gathers.append(RKGather(sources[0].arg.slot, 0, count, values=((value >> (byte*8))&0xff,)*count,
                                dst_addend=dst.addend//2, itemsize=2))
    else:
      offsets = typing_cast(tuple[int, ...], spec)
      for byte,dst in enumerate(raw):
        gathers.append(RKGather(param.arg.slot, 0, count,
          offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in offsets), dst_stride=2,
          dst_addend=dst.addend, itemsize=1))
  constants:dict[int, RKArg] = {}
  for constant_value in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256):
    constants[constant_value] = dst = allocate()
    gathers.append(RKGather(sources[0].arg.slot, 0, count, values=(constant_value,)*count,
                            dst_addend=dst.addend//2, itemsize=2))
  ops:list[RKEWOp] = []

  def clamp_one(value:RKArg) -> RKArg:
    positive, result = allocate(), allocate()
    ops.extend((RKEWOp(positive, value, constants[0], count, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(result, positive, constants[1], count, _EW_CFG_MIN, **_INT16_EW)))
    return result

  def positive_over(value:RKArg, threshold:int) -> RKArg:
    delta = allocate()
    ops.append(RKEWOp(delta, value, constants[threshold], count, _EW_CFG[Ops.SUB], **_INT16_EW))
    return clamp_one(delta)

  def xor_bit(lhs_bit:RKArg, rhs_bit:RKArg) -> RKArg:
    result = allocate()
    ops.extend((RKEWOp(result, lhs_bit, rhs_bit, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(result, result, result, count, _EW_CFG_ABS, **_INT16_EW)))
    return result

  def twos_complement(raw:tuple[RKArg, ...], sign:RKArg) -> tuple[RKArg, ...]:
    """Conditionally negate four unsigned byte lanes, carrying in base 256."""
    carry, result = sign, []
    for byte in raw:
      doubled, invert_delta, selected, total = allocate(), allocate(), allocate(), allocate()
      ops.extend((RKEWOp(doubled, byte, byte, count, _EW_CFG[Ops.ADD], **_INT16_EW),
                  RKEWOp(invert_delta, constants[255], doubled, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(invert_delta, invert_delta, sign, count, _EW_CFG[Ops.MUL], **_INT16_EW),
                  RKEWOp(selected, byte, invert_delta, count, _EW_CFG[Ops.ADD], **_INT16_EW),
                  RKEWOp(total, selected, carry, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
      carry = positive_over(total, 255)
      scaled, value = allocate(), allocate()
      ops.extend((RKEWOp(scaled, carry, constants[256], count, _EW_CFG[Ops.MUL], **_INT16_EW),
                  RKEWOp(value, total, scaled, count, _EW_CFG[Ops.SUB], **_INT16_EW)))
      result.append(value)
    return tuple(result)

  signs, magnitudes = [], []
  for raw in raw_operands:
    sign = positive_over(raw[3], 127)
    signs.append(sign); magnitudes.append(twos_complement(raw, sign))
  numerator, denominator = magnitudes
  denominator_bits = [clamp_one(byte) for byte in denominator]
  denominator_nonzero = _reduce_rows(ops, denominator_bits, count, _EW_CFG[Ops.MAX], int16=True)
  numerator_bits = tuple(bit for byte in numerator for bit in _int16_byte_bits(ops, allocate, constants, byte, count))

  remainder = [constants[0]]*4
  quotient = [constants[0]]*4
  for bit_index in range(31, -1, -1):
    shifted:list[RKArg] = []
    incoming = numerator_bits[bit_index]
    for byte_arg in remainder:
      carry = positive_over(byte_arg, 127)
      doubled, scaled, wrapped, out_value = allocate(), allocate(), allocate(), allocate()
      ops.extend((RKEWOp(doubled, byte_arg, byte_arg, count, _EW_CFG[Ops.ADD], **_INT16_EW),
                  RKEWOp(scaled, carry, constants[256], count, _EW_CFG[Ops.MUL], **_INT16_EW),
                  RKEWOp(wrapped, doubled, scaled, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(out_value, wrapped, incoming, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
      shifted.append(out_value); incoming = carry
    remainder = shifted

    greater, equal = constants[0], constants[1]
    for left,right in zip(reversed(remainder), reversed(denominator)):
      diff, positive, candidate = allocate(), allocate(), allocate()
      ops.extend((RKEWOp(diff, left, right, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(positive, diff, constants[0], count, _EW_CFG[Ops.MAX], **_INT16_EW),
                  RKEWOp(positive, positive, constants[1], count, _EW_CFG_MIN, **_INT16_EW),
                  RKEWOp(candidate, equal, positive, count, _EW_CFG_MIN, **_INT16_EW)))
      next_greater = allocate()
      ops.append(RKEWOp(next_greater, greater, candidate, count, _EW_CFG[Ops.MAX], **_INT16_EW)); greater = next_greater
      magnitude, unequal, byte_equal, next_equal = allocate(), allocate(), allocate(), allocate()
      ops.extend((RKEWOp(magnitude, diff, diff, count, _EW_CFG_ABS, **_INT16_EW),
                  RKEWOp(unequal, magnitude, constants[1], count, _EW_CFG_MIN, **_INT16_EW),
                  RKEWOp(byte_equal, constants[1], unequal, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(next_equal, equal, byte_equal, count, _EW_CFG_MIN, **_INT16_EW)))
      equal = next_equal
    ge = allocate()
    ops.extend((RKEWOp(ge, greater, equal, count, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(ge, ge, denominator_nonzero, count, _EW_CFG_MIN, **_INT16_EW)))

    borrow, reduced = constants[0], []
    for left,right in zip(remainder, denominator):
      masked, partial, delta = allocate(), allocate(), allocate()
      ops.extend((RKEWOp(masked, right, ge, count, _EW_CFG[Ops.MUL], **_INT16_EW),
                  RKEWOp(partial, left, masked, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(delta, partial, borrow, count, _EW_CFG[Ops.SUB], **_INT16_EW)))
      negative = allocate()
      ops.append(RKEWOp(negative, constants[0], delta, count, _EW_CFG[Ops.SUB], **_INT16_EW))
      borrow = clamp_one(negative)
      scaled, out_value = allocate(), allocate()
      ops.extend((RKEWOp(scaled, borrow, constants[256], count, _EW_CFG[Ops.MUL], **_INT16_EW),
                  RKEWOp(out_value, delta, scaled, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
      reduced.append(out_value)
    remainder = reduced
    byte_index, weight = bit_index >> 3, 1 << (bit_index&7)
    weighted, out_value = allocate(), allocate()
    ops.extend((RKEWOp(weighted, ge, constants[weight], count, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(out_value, quotient[byte_index], weighted, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
    quotient[byte_index] = out_value

  quotient_sign = xor_bit(signs[0], signs[1])
  if mode in ("cmod", "floormod"):
    result_magnitude = tuple(remainder)
    result_sign = signs[0]
    if mode == "floormod":
      remainder_nonzero = _reduce_rows(ops, [clamp_one(byte) for byte in remainder], count, _EW_CFG[Ops.MAX], int16=True)
      correction = allocate()
      ops.append(RKEWOp(correction, quotient_sign, remainder_nonzero, count, _EW_CFG_MIN, **_INT16_EW))
      corrected = []
      for rem,denom in zip(remainder, denominator):
        doubled, delta, selected, out_value = allocate(), allocate(), allocate(), allocate()
        ops.extend((RKEWOp(doubled, rem, rem, count, _EW_CFG[Ops.ADD], **_INT16_EW),
                    RKEWOp(delta, denom, doubled, count, _EW_CFG[Ops.SUB], **_INT16_EW),
                    RKEWOp(selected, delta, correction, count, _EW_CFG[Ops.MUL], **_INT16_EW),
                    RKEWOp(out_value, rem, selected, count, _EW_CFG[Ops.ADD], **_INT16_EW)))
        corrected.append(out_value)
      result_magnitude, result_sign = tuple(corrected), signs[1]
    result = twos_complement(result_magnitude, result_sign)
  else:
    if mode == "floor":
      remainder_nonzero = _reduce_rows(ops, [clamp_one(byte) for byte in remainder], count, _EW_CFG[Ops.MAX], int16=True)
      correction = allocate()
      ops.append(RKEWOp(correction, quotient_sign, remainder_nonzero, count, _EW_CFG_MIN, **_INT16_EW))
      carry, corrected = correction, []
      for byte_arg in quotient:
        total = allocate(); ops.append(RKEWOp(total, byte_arg, carry, count, _EW_CFG[Ops.ADD], **_INT16_EW))
        carry = positive_over(total, 255)
        scaled, out_value = allocate(), allocate()
        ops.extend((RKEWOp(scaled, carry, constants[256], count, _EW_CFG[Ops.MUL], **_INT16_EW),
                    RKEWOp(out_value, total, scaled, count, _EW_CFG[Ops.SUB], **_INT16_EW)))
        corrected.append(out_value)
      quotient = corrected
    result = twos_complement(tuple(quotient), quotient_sign)
  post = tuple(RKGather(value.index, out_param.arg.slot, count,
    offsets=tuple(value.addend+lane*2 for lane in range(count)), dst_stride=4, dst_addend=byte,
    dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1) for byte,value in enumerate(result))
  return RKImage((RKScratch(rows*stride),), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=tuple(replace(gather, after=len(ops)) for gather in post))

def _int16_byte_bits(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     value:RKArg, lanes:int) -> tuple[RKArg, ...]:
  """Split unsigned byte lanes into eight exact native INT16 0/1 planes."""
  result:list[RKArg|None] = [None]*8
  remainder = value
  for bit in range(7, 0, -1):
    delta, positive, flag, scaled, next_remainder = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, remainder, constants[(1<<bit)-1], lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(flag, positive, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(scaled, flag, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(next_remainder, remainder, scaled, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
    result[bit], remainder = flag, next_remainder
  result[0] = remainder
  return typing_cast(tuple[RKArg, ...], tuple(result))

def _lower_int32_byte_logic(output:RKOutput) -> RKImage|None:
  """Evaluate exact INT32 AND/OR/XOR by decomposing opaque bytes into native INT16 bit lanes."""
  _, out_param, count, out_index, root = output
  if root.op not in (Ops.AND, Ops.OR, Ops.XOR) or root.dtype.scalar() is not dtypes.int or len(root.src) != 2 or \
     not 1 <= count*4 <= _MAX_EW_ELEMS_FP16: return None
  operands:list[tuple[UOp|None, tuple[int, ...]|int]] = []
  for term in root.src:
    if term.op is Ops.CONST and term.dtype.scalar() is dtypes.int: operands.append((None, int(term.arg)&0xffffffff)); continue
    if (parsed:=_typed_load_offsets(term, dtypes.int, out_index, count, allow_fill=True)) is None: return None
    operands.append(parsed)
  sources = [param for param,_ in operands if param is not None]
  if not sources: return None
  lanes, vector_bytes, rows = count*4, _reduction_stride(count*4), 0
  def allocate() -> RKArg:
    nonlocal rows
    value = RKArg(RKBufferKind.SCRATCH, 0, rows*vector_bytes); rows += 1
    return value
  gathers:list[RKGather] = []
  values:list[RKArg] = []
  for param,spec in operands:
    value = allocate(); values.append(value)
    if param is None:
      constant = typing_cast(int, spec)
      byte_values = tuple((constant >> (byte*8))&0xff for _ in range(count) for byte in range(4))
      gathers.append(RKGather(sources[0].arg.slot, 0, lanes, values=byte_values, dst_addend=value.addend//2, itemsize=2))
    else:
      offsets = typing_cast(tuple[int, ...], spec)
      byte_offsets = tuple(offset*4+byte if offset >= 0 else -1 for offset in offsets for byte in range(4))
      gathers.append(RKGather(param.arg.slot, 0, lanes, offsets=byte_offsets, dst_stride=2,
                              dst_addend=value.addend, itemsize=1))
  constants:dict[int, RKArg] = {}
  for constant_value in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128):
    constants[constant_value] = slot = allocate()
    gathers.append(RKGather(sources[0].arg.slot, 0, lanes, values=(constant_value,)*lanes,
                            dst_addend=slot.addend//2, itemsize=2))
  ops:list[RKEWOp] = []
  lhs, rhs = _int16_byte_bits(ops, allocate, constants, values[0], lanes), \
             _int16_byte_bits(ops, allocate, constants, values[1], lanes)
  weighted:list[RKArg] = []
  for bit,(left,right) in enumerate(zip(lhs, rhs)):
    combined = allocate()
    if root.op is Ops.XOR:
      ops.extend((RKEWOp(combined, left, right, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(combined, combined, combined, lanes, _EW_CFG_ABS, **_INT16_EW)))
    else: ops.append(RKEWOp(combined, left, right, lanes, _EW_CFG_MIN if root.op is Ops.AND else _EW_CFG[Ops.MAX], **_INT16_EW))
    if bit: ops.append(RKEWOp(combined, combined, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
    weighted.append(combined)
  result = _reduce_rows(ops, weighted, lanes, _EW_CFG[Ops.ADD], int16=True)
  return RKImage((RKScratch(rows*vector_bytes),), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=(replace(_int16_low_bytes(result, out_param.arg.slot, lanes), after=len(ops)),))

def _lower_int32_shift(output:RKOutput) -> RKImage|None:
  """Evaluate exact 32-bit shifts with a five-stage native INT16 barrel shifter."""
  _, out_param, count, out_index, root = output
  if root.op is Ops.CAST and root.dtype.scalar() is dtypes.int and len(root.src) == 1: root = root.src[0]
  if root.op not in (Ops.SHL, Ops.SHR) or root.dtype.scalar() not in (dtypes.int, dtypes.uint) or len(root.src) != 2: return None
  value, shift = root.src
  dtype = root.dtype.scalar()
  if (parsed_value:=_typed_load_offsets(value, dtype, out_index, count, allow_fill=True)) is None: return None
  source, value_offsets = parsed_value
  shift_spec:tuple[UOp, tuple[int, ...]]|int
  if shift.op is Ops.CONST and shift.dtype.scalar() in (dtypes.int, dtypes.uint): shift_spec = int(shift.arg)&0xff
  elif shift.dtype.scalar() in (dtypes.int, dtypes.uint) and \
       (parsed_shift:=_typed_load_offsets(shift, shift.dtype.scalar(), out_index, count, allow_fill=True)) is not None:
    shift_spec = parsed_shift
  else: return None

  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, 32)
  pre_lanes = vector_lanes*5
  if count < 1 or matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  pre_rows = 0
  def pre_allocate() -> RKArg:
    nonlocal pre_rows
    value = RKArg(RKBufferKind.SCRATCH, 0, pre_rows*pre_lanes*2); pre_rows += 1
    return value
  raw = pre_allocate()
  gathers:list[RKGather] = []
  for byte in range(4):
    gathers.append(RKGather(source.arg.slot, 0, count,
      offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in value_offsets),
      dst_stride=2, dst_addend=raw.addend+byte*vector_bytes, itemsize=1))
  if isinstance(shift_spec, int):
    gathers.append(RKGather(source.arg.slot, 0, count, values=(shift_spec,)*count,
                           dst_addend=raw.addend//2+4*vector_lanes, itemsize=2))
  else:
    shift_source, shift_offsets = shift_spec
    gathers.append(RKGather(shift_source.arg.slot, 0, count,
      offsets=tuple(offset*4 if offset >= 0 else -1 for offset in shift_offsets),
      dst_stride=2, dst_addend=raw.addend+4*vector_bytes, itemsize=1))
  constants:dict[int, RKArg] = {}
  for constant_value in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128):
    constants[constant_value] = slot = pre_allocate()
    gathers.append(RKGather(source.arg.slot, 0, pre_lanes, values=(constant_value,)*pre_lanes,
                            dst_addend=slot.addend//2, itemsize=2))
  pre_ops:list[RKEWOp] = []
  planes = _int16_byte_bits(pre_ops, pre_allocate, constants, raw, pre_lanes)

  post_rows = 0
  def post_allocate() -> RKArg:
    nonlocal post_rows
    value = RKArg(RKBufferKind.SCRATCH, 1, post_rows*matrix_lanes*2); post_rows += 1
    return value
  bits, masks, sign, zero, weights = post_allocate(), tuple(post_allocate() for _ in range(5)), \
                                     post_allocate(), post_allocate(), post_allocate()
  mid:list[RKGather] = []
  for absolute_bit in range(32):
    plane, byte = absolute_bit&7, absolute_bit>>3
    mid.append(RKGather(planes[plane].index, bits.index, count,
      base=planes[plane].addend//2+byte*vector_lanes, axes=((1, count, 1),),
      dst_addend=bits.addend//2+absolute_bit*vector_lanes, src_kind=RKBufferKind.SCRATCH))
  for bit,mask in enumerate(masks):
    mid.append(RKGather(planes[bit].index, mask.index, matrix_lanes,
      base=planes[bit].addend//2+4*vector_lanes, axes=((1, vector_lanes, 1),),
      dst_addend=mask.addend//2, src_kind=RKBufferKind.SCRATCH))
  mid.append(RKGather(planes[7].index, sign.index, matrix_lanes,
    base=planes[7].addend//2+3*vector_lanes, axes=((1, vector_lanes, 1),),
    dst_addend=sign.addend//2, src_kind=RKBufferKind.SCRATCH))
  weight_values = tuple(1 << (row&7) if lane < count else 0 for row in range(32) for lane in range(vector_lanes))
  mid.append(RKGather(source.arg.slot, weights.index, matrix_lanes, values=weight_values,
                      dst_addend=weights.addend//2, dst_kind=RKBufferKind.SCRATCH))

  ops = list(pre_ops)
  current = bits
  for bit,amount in enumerate((1, 2, 4, 8, 16)):
    temp, result = post_allocate(), post_allocate()
    if root.op is Ops.SHL:
      normal_rows, normal_dst, shifted_src = 32-amount, amount, 0
      boundary_rows, boundary_dst = amount, 0
    else:
      normal_rows, normal_dst, shifted_src = 32-amount, 0, amount
      boundary_rows, boundary_dst = amount, 32-amount
    normal_count, normal_addend = normal_rows*vector_lanes, normal_dst*vector_bytes
    ops.extend((RKEWOp(RKArg(temp.kind, temp.index, temp.addend+normal_addend),
                          RKArg(current.kind, current.index, current.addend+shifted_src*vector_bytes),
                          RKArg(current.kind, current.index, current.addend+normal_addend), normal_count, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(RKArg(temp.kind, temp.index, temp.addend+normal_addend),
                          RKArg(temp.kind, temp.index, temp.addend+normal_addend),
                          RKArg(masks[bit].kind, masks[bit].index, masks[bit].addend+normal_addend),
                          normal_count, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(RKArg(result.kind, result.index, result.addend+normal_addend),
                          RKArg(current.kind, current.index, current.addend+normal_addend),
                          RKArg(temp.kind, temp.index, temp.addend+normal_addend), normal_count, _EW_CFG[Ops.ADD], **_INT16_EW)))
    boundary_addend = boundary_dst*vector_bytes
    fill = sign if root.op is Ops.SHR and dtype is dtypes.int else zero
    ops.extend((RKEWOp(RKArg(temp.kind, temp.index, temp.addend+boundary_addend),
                          RKArg(fill.kind, fill.index, fill.addend+boundary_addend),
                          RKArg(current.kind, current.index, current.addend+boundary_addend),
                          boundary_rows*vector_lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(RKArg(temp.kind, temp.index, temp.addend+boundary_addend),
                          RKArg(temp.kind, temp.index, temp.addend+boundary_addend),
                          RKArg(masks[bit].kind, masks[bit].index, masks[bit].addend+boundary_addend),
                          boundary_rows*vector_lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(RKArg(result.kind, result.index, result.addend+boundary_addend),
                          RKArg(current.kind, current.index, current.addend+boundary_addend),
                          RKArg(temp.kind, temp.index, temp.addend+boundary_addend),
                          boundary_rows*vector_lanes, _EW_CFG[Ops.ADD], **_INT16_EW)))
    current = result
  weighted = post_allocate()
  ops.append(RKEWOp(weighted, current, weights, matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
  byte_results = tuple(_reduce_rows(ops,
    [RKArg(weighted.kind, weighted.index, weighted.addend+(byte*8+bit)*vector_bytes) for bit in range(8)],
    vector_lanes, _EW_CFG[Ops.ADD], int16=True) for byte in range(4))
  post = tuple(RKGather(result.index, out_param.arg.slot, count,
    base=result.addend, axes=((1, count, 2),), dst_stride=4, dst_addend=byte,
    dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1) for byte,result in enumerate(byte_results))
  return RKImage((RKScratch(pre_rows*pre_lanes*2), RKScratch(post_rows*matrix_lanes*2)),
                 gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=tuple(replace(gather, after=len(pre_ops)) for gather in mid)+
                             tuple(replace(gather, after=len(ops)) for gather in post))

def _lower_raw_fp16_bitcast(output:RKOutput) -> RKImage|None:
  """Pair adjacent FP16 lane representations into an INT32 output without numeric conversion."""
  _, out_param, count, out_index, value = output
  if count <= 0: return RKImage()
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
  return RKImage(gathers=(gather,))


def _int32_index_selection_image(out_slot:int, count:int, index_slot:int, index_offsets:tuple[int, ...],
                                 candidate_values:tuple[tuple[int, ...], ...]) -> RKImage|None:
  """Select per-lane bounded INT16 values by exact external INT32 index equality."""
  rows = len(candidate_values)
  if not rows or any(len(values) != count or any(not -32768 <= value <= 32767 for value in values) for values in candidate_values): return None
  vector_bytes, vector_lanes, _ = _stripe_layout(count, 1)
  block_rows = max(1, _MAX_EW_ELEMS_FP16//vector_lanes)
  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
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
    ops.append(RKEWOp(_scratch_arg(selected), mask, _scratch_arg(weight_slot), matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
    partials.append(_reduce_rows(ops, [_scratch_arg(selected, row*vector_bytes) for row in range(block_count)],
                                 count, _EW_CFG[Ops.ADD], int16=True))
    if start and len(ops) > op_start: ops[op_start] = replace(ops[op_start], submit_barrier=True)
  result = _reduce_rows(ops, partials, count, _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, result, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

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


def _int32_less_mask(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     lhs_components:list[RKArg], rhs_components:list[RKArg], lanes:int) -> RKArg:
  """Compare signed INT32 lanes represented as high-to-low widened bytes."""
  def biased_sign(value:RKArg) -> RKArg:
    delta, positive, high, scaled, biased = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, value, constants[127], lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(high, positive, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(scaled, high, constants[256], lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(biased, value, constants[128], lanes, _EW_CFG[Ops.ADD], **_INT16_EW),
                RKEWOp(biased, biased, scaled, lanes, _EW_CFG[Ops.SUB], **_INT16_EW)))
    return biased
  lhs_components[0], rhs_components[0] = biased_sign(lhs_components[0]), biased_sign(rhs_components[0])
  less, equal = constants[0], constants[1]
  for lhs,rhs in zip(lhs_components, rhs_components):
    maximum, lhs_delta, rhs_delta, lhs_less, rhs_less, unequal, same, selected, next_less, next_equal = (allocate() for _ in range(10))
    ops.extend((RKEWOp(maximum, lhs, rhs, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(lhs_delta, maximum, lhs, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(rhs_delta, maximum, rhs, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(lhs_less, lhs_delta, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(rhs_less, rhs_delta, constants[1], lanes, _EW_CFG_MIN, **_INT16_EW),
                RKEWOp(unequal, lhs_less, rhs_less, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(same, constants[1], unequal, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                RKEWOp(selected, equal, lhs_less, lanes, _EW_CFG[Ops.MUL], **_INT16_EW),
                RKEWOp(next_less, less, selected, lanes, _EW_CFG[Ops.MAX], **_INT16_EW),
                RKEWOp(next_equal, equal, same, lanes, _EW_CFG[Ops.MUL], **_INT16_EW)))
    less, equal = next_less, next_equal
  return less

def _plan_offsets(plan:RKGather) -> tuple[int, ...]:
  if plan.offsets: return plan.offsets
  return tuple(plan.base + sum((lane//divisor % limit)*stride for divisor,limit,stride in plan.axes) for lane in range(plan.count))

RKDynamicIndex = tuple[int, int, tuple[int, ...]]
def _dynamic_raw_gather_image(out_slot:int, count:int, indices:tuple[RKDynamicIndex, ...], plans:tuple[RKGather, ...],
                              coordinates:tuple[tuple[int, ...], ...],
                              alternate_coordinates:tuple[tuple[tuple[int, ...], ...], ...]|None=None,
                              gate:RKDynamicIndex|None=None, itemsize:int=2) -> RKImage|None:
  """Select raw typed bytes with exact native integer masks, sharing repeated trailing lanes."""
  if (itemsize not in (1, 2, 4) or not plans or len(coordinates) != len(indices) or
      any(len(axis) != len(plans) for axis in coordinates) or
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
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  block_values:list[list[tuple[RKArg, ...]]] = []
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
      dst = scratch(matrix_lanes*2); ops.append(RKEWOp(_scratch_arg(dst), combined_mask, axis_mask, matrix_lanes,
        _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)); combined_mask = _scratch_arg(dst)
    matrix_value = tuple(tuple(scratch(matrix_lanes*2) for _ in range(itemsize)) for _ in range(repeat))
    selected = tuple(tuple(scratch(matrix_lanes*2) for _ in range(itemsize)) for _ in range(repeat))
    for candidate,row in enumerate(plan_offsets[start:stop]):
      for channel in range(repeat):
        for byte,slot in enumerate(matrix_value[channel]):
          gathers.append(RKGather(plans[0].src_index, slot, group_count,
            offsets=tuple(offset*itemsize+byte for offset in row[channel::repeat]), dst_stride=2,
            dst_addend=candidate*vector_lanes*2, itemsize=1))
    block_result:list[tuple[RKArg, ...]] = []
    for channel in range(repeat):
      ops.extend(RKEWOp(_scratch_arg(dst), _scratch_arg(src), combined_mask, matrix_lanes, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)
                 for src,dst in zip(matrix_value[channel], selected[channel]))
      block_result.append(tuple(_reduce_rows(ops, [_scratch_arg(slot, row*vector_lanes*2) for row in range(rows)], group_count,
        _EW_CFG[Ops.ADD], int16=True) for slot in selected[channel]))  # type: ignore[arg-type]
    block_values.append(block_result)
  if not block_values: return None

  results:list[tuple[RKArg, ...]] = []
  for channel in range(repeat):
    channel_values:list[RKArg] = []
    for byte in range(itemsize):
      value = block_values[0][channel][byte]
      for block in block_values[1:]:
        dst = scratch(group_count*2); ops.append(RKEWOp(_scratch_arg(dst), value, block[channel][byte], group_count,
          _EW_CFG[Ops.ADD], int16_input=True, int16_output=True))
        value = _scratch_arg(dst)
      channel_values.append(value)
    results.append(tuple(channel_values))
  if grouped_gate is not None:
    gate_slot = scratch(group_count*2)
    gathers.append(RKGather(grouped_gate[0], gate_slot, group_count, offsets=grouped_gate[2], dst_stride=2, itemsize=1))
    for channel,pair in enumerate(results):
      masked = tuple(scratch(group_count*2) for _ in range(itemsize))
      ops.extend(RKEWOp(_scratch_arg(dst), value, _scratch_arg(gate_slot), group_count, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)
                 for value,dst in zip(pair, masked))
      results[channel] = tuple(_scratch_arg(slot) for slot in masked)
  byte_offsets = tuple(range(0, group_count*2, 2))
  terminal = tuple(RKGather(value.index, out_slot, group_count, offsets=tuple(value.addend+offset for offset in byte_offsets),
    dst_stride=repeat*itemsize, dst_addend=channel*itemsize+byte,
    dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
    for channel,pair in enumerate(results) for byte,value in enumerate(pair))
  return RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=tuple(replace(gather, after=len(ops)) for gather in terminal))

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
  masks:list[RKArg] = []
  for coordinates in coordinate_sets:
    byte_masks:list[RKArg] = []
    for byte in range(4):
      dynamic, static, equal = (scratch(matrix_lanes*2) for _ in range(3))
      gathers.extend(RKGather(index_slot, dynamic, count, offsets=tuple(offset*4+byte for offset in offsets),
        dst_stride=2, dst_addend=row*vector_lanes*2, itemsize=1) for row,offsets in enumerate(offset_rows))
      values = tuple((value >> (byte*8)) & 0xff for row in coordinates for value in (*row, *((0,)*(vector_lanes-count))))
      gathers.append(RKGather(index_slot, static, matrix_lanes, values=values, itemsize=2))
      ops.extend((RKEWOp(_scratch_arg(diff), _scratch_arg(dynamic), _scratch_arg(static), matrix_lanes,
                          _EW_CFG[Ops.SUB], **_INT16_EW),
                  RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), matrix_lanes, _EW_CFG_ABS, **_INT16_EW),
                  RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, **_INT16_EW),
                  RKEWOp(_scratch_arg(equal), _scratch_arg(one), _scratch_arg(unequal), matrix_lanes,
                          _EW_CFG[Ops.SUB], **_INT16_EW)))
      byte_masks.append(_scratch_arg(equal))
    mask = byte_masks[0]
    for byte_mask in byte_masks[1:]:
      dst = scratch(matrix_lanes*2); ops.append(RKEWOp(_scratch_arg(dst), mask, byte_mask, matrix_lanes,
        _EW_CFG[Ops.MUL], int16_input=True, int16_output=True)); mask = _scratch_arg(dst)
    masks.append(mask)
  mask = masks[0]
  for alternate in masks[1:]:
    dst = scratch(matrix_lanes*2); ops.append(RKEWOp(_scratch_arg(dst), mask, alternate, matrix_lanes,
      _EW_CFG[Ops.MAX], int16_input=True, int16_output=True)); mask = _scratch_arg(dst)
  return mask

def _native_integer_nonzero_mask(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], source_slot:int,
                                 rows:tuple[tuple[int, ...], ...], count:int, vector_lanes:int, itemsize:int=4) -> RKArg|None:
  """Test opaque integer bytes for nonzero using unsigned-byte INT16 MIN/MAX stages."""
  if not rows or any(len(row) != count for row in rows): return None
  matrix_lanes = len(rows)*vector_lanes
  one = scratch(matrix_lanes*2)
  byte_masks = tuple(scratch(matrix_lanes*2) for _ in range(itemsize))
  gathers.append(RKGather(source_slot, one, matrix_lanes, values=(1,)*matrix_lanes, itemsize=2))
  for byte,slot in enumerate(byte_masks):
    gathers.extend(RKGather(source_slot, slot, count,
      offsets=tuple(offset*itemsize+byte if offset >= 0 else -1 for offset in row), dst_addend=i*vector_lanes*2,
      dst_stride=2, itemsize=1) for i,row in enumerate(rows))
  for slot in byte_masks:
    ops.append(RKEWOp(_scratch_arg(slot), _scratch_arg(slot), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, int16_input=True, int16_output=True))
  mask = _scratch_arg(byte_masks[0])
  for slot in byte_masks[1:]:
    ops.append(RKEWOp(mask, mask, _scratch_arg(slot), matrix_lanes, _EW_CFG[Ops.MAX], int16_input=True, int16_output=True))
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
    dst = scratch(count*2); ops.append(RKEWOp(_scratch_arg(dst), result, valid, count, _EW_CFG[Ops.MUL],
      int16_input=True, int16_output=True)); result = _scratch_arg(dst)
  return RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=(replace(_int16_low_bytes(result, out_param.arg.slot, count), after=len(ops)),))

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


RKBoundedPredicateCoordinates = tuple[UOp, int, UOp, tuple[int, ...], tuple[tuple[int, ...], ...], int]

def _bounded_predicate_coordinate_plan(output:RKOutput, dtype:DType, predicate:Callable[[UOp], UOp|None],
                                       encodable:Callable[[int], bool]) -> RKBoundedPredicateCoordinates|None:
  """Prove a bounded predicate count plus dynamic-rank coordinate selection and fill program."""
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

def _lower_bounded_integer_predicate_coordinates(output:RKOutput, dtype:DType) -> RKImage|None:
  """Execute bounded integer predicate coordinates through native INT16 byte masks."""
  _, out_param, count, _, _ = output
  if (plan:=_bounded_predicate_coordinate_plan(output, dtype, lambda u:_nonzero_load(u, dtype),
                                               lambda value:-32768 <= value <= 32767)) is None: return None
  source, rank, index_param, index_offsets, coordinate_rows, fill_value = plan
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, coordinate_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  scratch_sizes:list[int] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(64, size)); return len(scratch_sizes)-1
  gathers:list[RKGather] = []
  ops:list[RKEWOp] = []
  _, source_vector_lanes, _ = _stripe_layout(1, source_count)
  source_rows = tuple((lane,) for lane in range(source_count))
  if (source_mask:=_native_integer_nonzero_mask(ops, gathers, scratch, source.arg.slot, source_rows, 1,
                                                source_vector_lanes, dtype.itemsize)) is None: return None
  total = _reduce_rows(ops, [replace(source_mask, addend=source_mask.addend+row*64) for row in range(source_count)],
                       1, _EW_CFG[Ops.ADD], int16=True)
  if rank != 1:
    rank_slot = scratch(2)
    gathers.append(RKGather(source.arg.slot, rank_slot, 1, values=(_int16_bits(rank),)))
    ops.append(RKEWOp(total, total, _scratch_arg(rank_slot), 1, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
  candidates = tuple((candidate,)*count for candidate in range(coordinate_count))
  if (equal:=_native_int16_byte_mask(ops, gathers, scratch, index_param.arg.slot, index_offsets,
                                     (candidates,), count, vector_lanes)) is None: return None
  gather_after = len(ops)
  total_vector, output_coordinate, zero, one, coordinate_matrix = (scratch(matrix_lanes*2) for _ in range(5))
  mid = (RKGather(total.index, total_vector, count, offsets=(total.addend//2,)*count,
                  src_kind=RKBufferKind.SCRATCH, after=gather_after),)
  gathers.extend(_stripe_gathers(source.arg.slot, coordinate_matrix, count,
    tuple(tuple(_int16_bits(value) for value in row) for row in coordinate_rows), vector_lanes, values=True))
  gathers.extend((RKGather(source.arg.slot, output_coordinate, count, values=tuple(range(count))),
                  RKGather(source.arg.slot, zero, matrix_lanes, values=(0,)*matrix_lanes),
                  RKGather(source.arg.slot, one, matrix_lanes, values=(1,)*matrix_lanes)))
  fill_slot = scratch(count*2)
  gathers.append(RKGather(source.arg.slot, fill_slot, count, values=(_int16_bits(fill_value),)*count))
  valid_delta, positive, valid, remaining = (scratch(count*2) for _ in range(4))
  selected, guarded, fill_part, result = (scratch(matrix_lanes*2) for _ in range(4))
  ops.extend((RKEWOp(_scratch_arg(valid_delta), _scratch_arg(total_vector), _scratch_arg(output_coordinate), count, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(_scratch_arg(positive), _scratch_arg(valid_delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(_scratch_arg(valid), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN, **_INT16_EW),
              RKEWOp(_scratch_arg(remaining), _scratch_arg(one), _scratch_arg(valid), count, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(_scratch_arg(selected), equal, _scratch_arg(coordinate_matrix), matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW)))
  selected_value = _reduce_rows(ops, [_scratch_arg(selected, row*vector_bytes) for row in range(coordinate_count)],
                                count, _EW_CFG[Ops.ADD], int16=True)
  ops.extend((RKEWOp(_scratch_arg(guarded), selected_value, _scratch_arg(valid), count, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(_scratch_arg(fill_part), _scratch_arg(fill_slot), _scratch_arg(remaining), count, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(_scratch_arg(result), _scratch_arg(guarded), _scratch_arg(fill_part), count, _EW_CFG[Ops.ADD], **_INT16_EW),
              RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), _scratch_arg(result), _scratch_arg(result), count, _EW_CFG[Ops.MAX],
                     int16_input=True, int32_output=True)))
  return RKImage(tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid)


def _lower_direct_dynamic_typed_load(output:RKOutput, dtype:DType) -> RKImage|None:
  """Materialize a bounds-masked typed LOAD addressed by one dynamic INT32 index."""
  _, out_param, count, out_index, load = output
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or
      load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or load.src[1].arg != 0): return None
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
  return _dynamic_raw_gather_image(out_param.arg.slot, count, ((index_param.arg.slot, index_count, index_offsets),), plans,
                                   (coordinates,), gate=external_gate, itemsize=dtype.itemsize)

def _lower_dynamic_multi_index_typed_load(output:RKOutput, dtype:DType) -> RKImage|None:
  """Materialize one typed LOAD addressed by positive or negative-normalized dynamic INT32 axes."""
  _, out_param, count, out_index, load = output
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or
      load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or load.src[1].arg != 0): return None
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
  return _dynamic_raw_gather_image(out_param.arg.slot, count, indices, plans, coordinates, alternates)

def _bool_reduction_image(out_slot:int, count:int, source_slot:int, offsets:tuple[tuple[int, ...], ...], op:Ops) -> RKImage:
  window = len(offsets)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, window)
  zero, one, candidates_slot, diff, magnitude, unequal, equal, int_tiles = range(8)
  gathers = _stripe_gathers(source_slot, candidates_slot, count, offsets, vector_lanes)
  scratch = (*(RKScratch(_scratch_bytes(matrix_lanes)) for _ in range(int_tiles)), RKScratch(_int32_tiles_bytes(count)))
  ops:list[RKEWOp] = []
  _ew_eq_mask(ops, _scratch_arg, candidates_slot, zero, (diff, magnitude, unequal, equal), one, matrix_lanes)
  selected = _reduce_rows(ops, [_scratch_arg(unequal, row*vector_bytes) for row in range(window)], count,
                          _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL])
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), selected, _scratch_arg(int_tiles), count, _EW_CFG[Ops.MAX],
                    stateful=True, int32_output=True, bool_output=True))
  return RKImage(scratch, struct.pack("<ee", 0.0, 1.0), gathers=gathers, ew_ops=tuple(ops))

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
  ops = [RKEWOp(_scratch_arg(diff), RKArg(RKBufferKind.ARG, source_slot), _scratch_arg(zero), source_count, _EW_CFG[Ops.SUB]),
         RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), source_count, _EW_CFG_ABS, submit_barrier=True, stateful=True),
         RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(magnitude), source_count, _EW_CFG[Ops.MAX], compare=True)]
  ops.extend(_block_bool_reduction_ops(_scratch_arg(unequal), count, groups, op))
  gather_after = len(ops)
  mid = (RKGather(unequal, packed, count, offsets=tuple(lane*groups for lane in range(count)),
                  src_kind=RKBufferKind.SCRATCH, after=gather_after),)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), _scratch_arg(packed), _scratch_arg(int_tiles), count, _EW_CFG[Ops.MAX],
                    stateful=True, int32_output=True, bool_output=True))
  full, output = _scratch_bytes(source_count), _scratch_bytes(count)
  scratch = (RKScratch(full), RKScratch(full), RKScratch(full), RKScratch(full), RKScratch(output),
             RKScratch(_int32_tiles_bytes(count)))
  return RKImage(scratch, struct.pack("<e", 0.0), ew_ops=tuple(ops), mid_gathers=mid)

def _stored_bool_reduction_image(out_slot:int, count:int, source_slot:int, offsets:tuple[tuple[int, ...], ...], op:Ops) -> RKImage:
  """Place opaque bool bytes into zeroed INT16 lanes and reduce them with native integer EW."""
  vector_bytes, _, matrix_lanes = _stripe_layout(count, len(offsets))
  gathers = tuple(RKGather(source_slot, 0, count, offsets=row, dst_addend=i*vector_bytes, dst_stride=2, itemsize=1)
                  for i,row in enumerate(offsets))
  ops:list[RKEWOp] = []
  selected = _reduce_rows(ops, [RKArg(RKBufferKind.SCRATCH, 0, i*vector_bytes) for i in range(len(offsets))], count,
                          _EW_CFG[Ops.MAX if op is Ops.OR else Ops.MUL], int16=True)
  return RKImage((RKScratch(_scratch_bytes(matrix_lanes)),), gathers=gathers, ew_ops=tuple(ops),
                 mid_gathers=(replace(_int16_low_bytes(selected, out_slot, count), after=len(ops)),))


def _contiguous_stored_bool_reduction_image(out_slot:int, count:int, source_slot:int, groups:int, op:Ops) -> RKImage:
  """Reduce contiguous opaque bool blocks after widening their bytes into native INT16 lanes."""
  source_count = count*groups
  ops = _block_bool_reduction_ops(RKArg(RKBufferKind.SCRATCH, 0), count, groups, op, int16=True)
  gathers = (RKGather(source_slot, 0, source_count, dst_stride=2, itemsize=1),)
  return RKImage((RKScratch(_scratch_bytes(source_count)),), gathers=gathers, ew_ops=tuple(ops),
                 mid_gathers=(replace(_int16_low_bytes(RKArg(RKBufferKind.SCRATCH, 0), out_slot, count, groups*2), after=len(ops)),))

def _nonzero_load(term:UOp, dtype:DType=dtypes.half) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtype and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and zero.arg == 0]
  return candidates[0] if len(candidates) == 1 else None


def _lower_grouped_bool_reduction(output:RKOutput) -> RKImage|None:
  """Lower grouped FP16 or stored-bool any/all after proving launch coordinates and full source coverage."""
  store, out_param, count, out_index, root = output
  if _local_load(root) is None or len(store.src) not in (2, 3) or count <= 0: return None
  nodes = root.toposort()
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

def _isclose_match(root:UOp) -> tuple[UOp, UOp, bool, bool]|None:
  """Recover isclose's original operands, equal_nan mode, and its exact-equality FP16 tolerance range."""
  nodes = root.toposort()
  inverted = {inner.key for u in nodes if u.op is Ops.CMPNE and len(u.src) == 2 for inner,marker in (u.src, u.src[::-1])
              if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg)}
  pairs = [u for u in nodes if u.op is Ops.CMPNE and len(u.src) == 2 and u.src[0].dtype.scalar() is dtypes.half and
           u.src[1].dtype.scalar() is dtypes.half and u.src[0].key != u.src[1].key and
           u.key in inverted and not any(x.op is Ops.CONST and math.isinf(float(x.arg)) for x in u.src)]
  self_nan = [u for u in nodes if u.op is Ops.CMPNE and len(u.src) == 2 and u.src[0].key == u.src[1].key and
              u.src[0].dtype.scalar() is dtypes.half]
  self_values = tuple({u.src[0].key:u.src[0] for u in self_nan}.values())
  operands = pairs[0].src if len(pairs) == 1 else (self_values[0], self_values[0]) if len(self_values) == 1 else ()
  infinities = {float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.half, dtypes.float) and
                math.isinf(float(u.arg))}
  if root.op is not Ops.OR or len(operands) != 2 or not self_nan or infinities != {-math.inf, math.inf}: return None
  equal_nan = any(u.op is Ops.AND and len(u.src) == 2 and all(x in self_nan for x in u.src) for u in nodes) or \
              any(x in self_nan for x in root.src)
  finite_constants = [abs(float(u.arg)) for u in nodes if u.op is Ops.CONST and
                      u.dtype.scalar() in (dtypes.half, dtypes.float) and math.isfinite(float(u.arg))]
  exact = any(_fp16_bits(value) == _fp16_bits(1e-5) for value in finite_constants) and \
          not any(1e-4 < value < .1 for value in finite_constants)
  return operands[0], operands[1], equal_nan, exact

def _ieee_comparison_mask(root:UOp) -> UOp|None:
  """Build an IEEE-correct FP16 comparison mask without evaluating tensor values on the host."""
  one = UOp.const(1.0, dtypes.half)
  def inverse(value:UOp) -> UOp: return one.alu(Ops.SUB, value)
  def numeric(value:UOp) -> UOp|None:
    original, value = value, _unwrap_condition(value)
    if value.dtype.scalar() not in (dtypes.half, dtypes.float) and original.dtype.scalar() in (dtypes.half, dtypes.float): value = original
    if value.dtype.scalar() not in (dtypes.half, dtypes.float): return None
    loads = [u for u in value.toposort() if u.op is Ops.LOAD]
    params = [_root_param(load.src[0]) if load.src and load.src[0].op is Ops.INDEX else None for load in loads]
    if any(param is None or param.dtype.scalar() is not dtypes.half for param in params): return None
    return value if value.dtype.scalar() is dtypes.half else value.cast(dtypes.half)
  def classes(value:UOp) -> tuple[UOp, UOp, UOp, UOp]:
    high = _positive_mask(value.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
    negated = UOp(Ops.NEG, dtypes.half, src=(value,))
    low = _positive_mask(negated.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
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
  result = mask(root)
  if result is None: return None
  # The FP16 tolerance product may underflow even when both realized operands are bitwise equal. Tinygrad's isclose
  # graph always carries its original lhs!=rhs test beside two self-NaN tests and the explicit infinity constants.
  # OR exact IEEE equality back into that graph; NaN remains unequal unless the original equal_nan branch accepts it.
  if (isclose:=_isclose_match(root)) is not None and (unequal:=atom(Ops.CMPNE, isclose[0], isclose[1])) is not None:
    result = _mask_mul(result, _mask_mul(classes(isclose[0])[3], classes(isclose[1])[3]))
    exact = inverse(unequal)
    if isclose[2]:
      lhs_nan, rhs_nan = classes(isclose[0])[0], classes(isclose[1])[0]
      exact = exact.alu(Ops.MAX, _mask_mul(lhs_nan, rhs_nan))
    if isclose[3]: return exact
    result = result.alu(Ops.MAX, exact)
  return result

def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if root.op is Ops.CAST and root.dtype.scalar() is dtypes.bool and len(root.src) == 1 and root.src[0].dtype.scalar() is dtypes.half:
    root = root.src[0] != UOp.const(0.0, dtypes.half)
  if (load:=_nonzero_load(root)) is None: return None
  magnitude = UOp(Ops.MAX, dtypes.half, src=(load, load), arg=_NATIVE_ABS)
  return _positive_mask(magnitude)

def _exact_int_range(root:UOp) -> tuple[int, int]|None:
  """Conservatively bound an integer UOp before choosing its exact physical scratch layout."""
  cache:dict[UOp, tuple[int, int]|None] = {}
  def bounds(u:UOp) -> tuple[int, int]|None:
    if u in cache: return cache[u]
    result:tuple[int, int]|None = None
    if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.int, dtypes.weakint): result = (int(u.arg), int(u.arg))
    elif u.op is Ops.RANGE and u.src[0].op is Ops.CONST: result = (0, max(0, int(u.src[0].arg)-1))
    elif u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.bool: result = (0, 1)
    elif u.op is Ops.WHERE and len(u.src) == 3:
      yes, no = bounds(u.src[1]), bounds(u.src[2])
      if yes is not None and no is not None: result = (min(yes[0], no[0]), max(yes[1], no[1]))
    elif u.op is Ops.XOR and len(u.src) == 2:
      for marker, source in (u.src, u.src[::-1]):
        if marker.op is Ops.CONST and int(marker.arg) == -1 and (source_bounds:=bounds(source)) is not None:
          result = (-1-source_bounds[1], -1-source_bounds[0])
    elif u.op is Ops.CMOD and len(u.src) == 2:
      left, right = bounds(u.src[0]), bounds(u.src[1])
      if right is not None and right[0] == right[1] and right[0] != 0:
        extent = abs(right[0])-1
        result = (-extent, extent) if left is None else \
                 (0, extent) if left[0] >= 0 else (-extent, 0) if left[1] <= 0 else (-extent, extent)
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2:
      left, right = bounds(u.src[0]), bounds(u.src[1])
      if left is not None and right is not None:
        if u.op is Ops.ADD: result = (left[0]+right[0], left[1]+right[1])
        elif u.op is Ops.SUB: result = (left[0]-right[1], left[1]-right[0])
        elif u.op is Ops.MUL:
          products = tuple(a*b for a in left for b in right)
          result = (min(products), max(products))
        else: result = (max(left[0], right[0]), max(left[1], right[1]))
    cache[u] = result
    return result
  return bounds(root)

def _int_fp16_expr(u:UOp) -> UOp:
  """Represent an integer UOp whose values are exactly carried in FP16 lanes as a half-valued recipe."""
  if u.dtype.scalar() is not dtypes.int: raise _RKGenericReject
  if u.op is Ops.CONST: return UOp.const(float(int(u.arg)), dtypes.half)
  if u.op is Ops.CAST and len(u.src) == 1:
    if u.src[0].dtype.scalar() is dtypes.half: return _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=u.src))
    if u.src[0].dtype.scalar() is dtypes.bool: return u.src[0].cast(dtypes.half)
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2:
    return UOp(u.op, dtypes.half, src=tuple(_int_fp16_expr(src) for src in u.src), arg=u.arg)
  if u.op is Ops.WHERE and len(u.src) == 3:
    return UOp(Ops.WHERE, dtypes.half, src=(u.src[0], _int_fp16_expr(u.src[1]), _int_fp16_expr(u.src[2])), arg=u.arg)
  if u.op is Ops.CMOD and len(u.src) == 2:
    lhs, rhs = (_int_fp16_expr(src) for src in u.src)
    quotient = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(lhs.alu(Ops.FDIV, rhs),)))
    return lhs.alu(Ops.SUB, quotient.alu(Ops.MUL, rhs))
  raise _RKGenericReject(f"INT_FP16 recipe {u.op.name}")

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

class _RKGenericReject(Exception): pass

def _runtime_index(u:UOp) -> tuple[UOp, UOp, UOp, int]|None:
  """Return the index LOAD, its parameter, lane-address expression, and raw index width."""
  u = _strip_cast(u)
  if (u.op is not Ops.LOAD or len(u.src) != 1 or u.src[0].op is not Ops.INDEX or
      (param:=_root_param(u.src[0])) is None or param.src[0].op is not Ops.CONST): return None
  dtype = param.dtype.scalar()
  if dtype not in (dtypes.int, dtypes.int16): return None
  return u, param, u.src[0].src[1], dtype.itemsize

def _has_runtime_address(root:UOp) -> bool:
  """True when a value LOAD obtains its address or gate from another runtime LOAD."""
  for load in root.toposort():
    if load.op is not Ops.LOAD or not load.src or load.src[0].op is not Ops.INDEX: continue
    address_nodes = load.src[0].src[1].toposort()
    gate_nodes = load.src[2].toposort() if len(load.src) > 2 else ()
    if any(_runtime_index(node) is not None for node in (*address_nodes, *gate_nodes)): return True
  return False

def _runtime_affine_index(u:UOp, out_index:UOp, count:int) -> tuple[UOp, UOp, int, int, int, int, int]|None:
  """Resolve `static_lane_base + runtime_index * scale` without reading the runtime index on the renderer."""
  loads = [node for node in u.toposort() if _runtime_index(node) is not None]
  if len(loads) != 1 or (info:=_runtime_index(loads[0])) is None: return None
  load, param, lane_index, itemsize = info
  try:
    lane_offsets = _static_int_vector(out_index, lane_index, count)
    zero = _static_int_vector(out_index, u.substitute({load:load.const_like(0)}), count)
    one = _static_int_vector(out_index, u.substitute({load:load.const_like(1)}), count)
  except RuntimeError: return None
  lane_offset = lane_offsets[0]
  if lane_offsets != tuple(lane_offset+lane for lane in range(count)) or \
     not 0 <= lane_offset <= int(param.src[0].arg)-count: return None
  scales = tuple(a-b for a,b in zip(one, zero))
  if len(set(scales)) != 1: return None
  lane_stride = zero[1]-zero[0] if count > 1 else 0
  if zero != tuple(zero[0]+lane*lane_stride for lane in range(count)): return None
  return load, param, itemsize, lane_offset, zero[0], scales[0], lane_stride

def _fp32_expr_to_half(u:UOp) -> UOp:
  """Represent a float ADD/MUL expression with a three-half expansion at its FP16 storage boundary."""
  if u.dtype.scalar() is dtypes.half: return u
  if u.dtype.scalar() is not dtypes.float: raise _RKGenericReject
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.half: return u.src[0]
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() in (dtypes.int, dtypes.int16, dtypes.bool):
    return u.src[0].cast(dtypes.half)
  if u.op is Ops.LOAD: return u.cast(dtypes.half)
  if u.op is Ops.CONST: return UOp.const(float(u.arg), dtypes.half)
  if _is_static_expr(u): return u.cast(dtypes.half)
  if u.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN) and len(u.src) == 1:
    return UOp(u.op, dtypes.half, src=(_fp32_expr_to_half(u.src[0]),), arg=u.arg)
  if u.op is Ops.MUL and len(u.src) == 2:
    return UOp(Ops.MUL, dtypes.half, src=tuple(_fp32_expr_to_half(src) for src in u.src))
  if u.op in (Ops.SUB, Ops.MAX) and len(u.src) == 2:
    return UOp(u.op, dtypes.half, src=tuple(_fp32_expr_to_half(src) for src in u.src), arg=u.arg)
  if u.op is Ops.NEG and len(u.src) == 1:
    return UOp(Ops.NEG, dtypes.half, src=(_fp32_expr_to_half(u.src[0]),))
  if u.op is Ops.ADD:
    return _precise_mul_sum(_fp32_add_terms(u))
  raise _RKGenericReject

def _nested_fp32_storage_cast(x:UOp) -> UOp|None:
  try: return _fp32_expr_to_half(x)
  except _RKGenericReject: return None

_pm_half_storage_algebra = PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.float, name="x"),)), _nested_fp32_storage_cast),
  (UPat(Ops.FDIV, dtypes.half, src=(UPat.var("x"), UPat.var("y"))),
   lambda x,y:x.alu(Ops.MUL, UOp(Ops.RECIPROCAL, dtypes.half, src=(y,)))),
])

def _canonical_half_storage(source:UOp) -> UOp:
  """Commit one FP32 storage expression, then reuse Tinygrad's ordinary algebra on its now-identical half values."""
  converted = _fp32_expr_to_half(source)
  if len(source.toposort()) > 64: return converted
  simplified = graph_rewrite(converted, _pm_half_storage_algebra+sym,
                             name="rockchip half storage algebra")
  return _tag_precise_adds(graph_rewrite(simplified, pm_commit_weak, name="rockchip commit storage constants"))

def _fp32_add_terms(u:UOp) -> list[UOp]:
  return [_fp32_expr_to_half(term) for term in _flatten_binary(u, Ops.ADD)]

def _fp32_add_has_product_terms(u:UOp) -> bool:
  """Whether a floating ADD tree contains a direct floating or cast-half product term."""
  return any((term.op is Ops.MUL and term.dtype.scalar() is dtypes.float) or
             (term.op is Ops.CAST and len(term.src) == 1 and term.src[0].op is Ops.MUL and
              term.src[0].dtype.scalar() is dtypes.half) for term in _flatten_binary(u, Ops.ADD))

def _fp32_ratio_to_half(u:UOp) -> UOp|None:
  """Divide two FP32 ADD boundaries while retaining their high/low half expansions through FDIV."""
  if u.op is not Ops.FDIV or u.dtype.scalar() is not dtypes.half or len(u.src) != 2: return None
  sums:list[UOp] = []
  for source in u.src:
    if source.op is not Ops.CAST or source.dtype.scalar() is not dtypes.half or len(source.src) != 1 or \
       source.src[0].dtype.scalar() is not dtypes.float or source.src[0].op is not Ops.ADD: return None
    sums.append(source.src[0])
  numerator_high,numerator_low = _precise_sum_parts(_fp32_add_terms(sums[0]))
  denominator_high,denominator_low = _precise_sum_parts(_fp32_add_terms(sums[1]))
  numerator = numerator_high.alu(Ops.ADD, numerator_low)
  denominator = denominator_high.alu(Ops.ADD, denominator_low)
  quotient = numerator.alu(Ops.FDIV, denominator)
  neg_one = UOp.const(-1.0, dtypes.half)
  residual = _sub_half(numerator_high, quotient.alu(Ops.MUL, denominator_high), neg_one).alu(Ops.ADD,
    _sub_half(numerator_low, quotient.alu(Ops.MUL, denominator_low), neg_one))
  root = quotient.alu(Ops.ADD, residual.alu(Ops.FDIV, denominator))
  return _tag_precise_adds(root)

def _accurate_add_recipe(u:UOp) -> UOp:
  terms:list[UOp] = []
  for x in _flatten_binary(u, Ops.ADD, plain=True):
    if x.op is Ops.CAST and x.dtype.scalar() is dtypes.half and len(x.src) == 1 and x.src[0].dtype.scalar() is dtypes.float and \
       x.src[0].op is Ops.ADD:
      terms.extend(_fp32_add_terms(x.src[0]))
    else: terms.append(x)
  if sum(term.op is Ops.MUL and term.arg is None for term in terms) < 2: raise _RKGenericReject
  if any(any(node.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN) for node in term.toposort()) for term in terms):
    raise _RKGenericReject
  return _precise_mul_sum(terms)

class RKContext:
  """Typed physical lowering context. UOps remain the only semantic IR."""
  def __init__(self, output:RKOutput, nodes:dict[UOp, None], *, accurate_adds:bool=True, static_load_offsets:dict[UOp, tuple[int, ...]]|None=None):
    _, self.out_param, self.count, self.out_index, self.root = output
    self.out = RKArg(RKBufferKind.ARG, self.out_param.arg.slot)
    self.values:dict[UOp, RKValue] = {}
    self.scratch:list[RKScratch] = []
    self.constants:dict[bytes, int] = {}
    self.materialized_slots:dict[tuple, RKValue] = {}
    self.static_load_offsets = {} if static_load_offsets is None else static_load_offsets
    self.int32_components:dict[RKArg, tuple[RKValue, ...]] = {}
    self.raw_bytes:dict[RKArg, tuple[RKValue, RKValue]] = {}
    self.int16_masks:dict[RKArg, int] = {}
    self.fp16_components:dict[RKArg, tuple[RKValue, RKArg, RKArg]] = {}
    self.fp16_ordered:dict[RKArg, tuple[RKArg, RKArg]] = {}
    self.gathers:list[RKGather] = []
    self.host_gathers:list[RKHostAddress] = []
    self.mid_gathers:list[RKGather] = []
    self.ew_ops:list[RKEWOp] = []
    self.mask_program = any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in nodes)
    int_range = _exact_int_range(self.root) if self.root.dtype.scalar() is dtypes.int else None
    packed_bool_load = any(node.op is Ops.LOAD and node.dtype.scalar() is dtypes.bool and _root_param(node.src[0]) is not None for node in nodes)
    native_bool = any(node.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in node.src) for node in nodes)
    embedded_half_int = any(node.op is Ops.CAST and node.dtype.scalar() is dtypes.int and len(node.src) == 1 and
                            node.src[0].dtype.scalar() in (dtypes.half, dtypes.bool) for node in nodes)
    dynamic_int_load = any(node.op is Ops.LOAD and node.dtype.scalar() is dtypes.int and node.src and
                           _root_param(node.src[0]) is not None for node in nodes)
    self.int_layout = (RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int and dynamic_int_load else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and (packed_bool_load or native_bool) and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT_FP16 if self.root.dtype.scalar() is dtypes.int and embedded_half_int else
                       RKLayout.INT_FP16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -2048 <= int_range[0] <= int_range[1] <= 2048 else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int else
                       RKLayout.INT32 if dynamic_int_load else
                       RKLayout.INT_FP16 if embedded_half_int else None)
    self.accurate_adds = accurate_adds
    self.nodes = nodes
    self.static_nodes:set[UOp] = set()
    for node in nodes:
      if node.op in _STATIC_OPS and all(src in self.static_nodes for src in node.src): self.static_nodes.add(node)

  def _scratch(self, dtype:DType, layout:RKLayout, size:int|None=None) -> RKValue:
    slot = len(self.scratch)
    self.scratch.append(RKScratch(self.count*4 if size is None and layout is RKLayout.INT32 else
                                  _scratch_bytes(self.count) if size is None else size))
    return RKValue(RKArg(RKBufferKind.SCRATCH, slot), dtype, self.count, layout)

  def _int16_arg(self) -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg

  def _dst(self, u:UOp, dtype:DType, layout:RKLayout) -> RKValue:
    if (u is self.root and self.out_param.dtype.scalar() is dtype and
        (dtype is dtypes.half and layout is RKLayout.FP16 or dtype is dtypes.int16 and layout is RKLayout.INT16 or
         dtype is dtypes.int and layout is RKLayout.INT32)):
      return RKValue(self.out, dtype, self.count, layout)
    return self._scratch(dtype, layout)

  def _materialized_slot(self, key:tuple, dtype:DType, layout:RKLayout, plan:RKGather, size:int|None=None) -> RKValue:
    if key not in self.materialized_slots:
      value = self._scratch(dtype, layout, size)
      self.gathers.append(replace(plan, dst_index=value.arg.index))
      self.materialized_slots[key] = value
    return RKValue(self.materialized_slots[key].arg, dtype, self.count, layout)

  def _static_slot(self, dtype:DType, layout:RKLayout, vector:tuple[int, ...]) -> RKValue:
    return self._materialized_slot(("static", layout, vector), dtype, layout,
      RKGather(0, 0, self.count, values=vector, itemsize=4 if layout is RKLayout.INT32 else 2))

  def _gather_slot(self, dtype:DType, layout:RKLayout, plan:RKGather, size:int) -> RKValue:
    return self._materialized_slot(("gather", layout, _gather_cache_key((plan,))), dtype, layout, plan, size)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> RKValue:
    dtype = dtype_hint or u.dtype.scalar()
    if dtype is dtypes.int and self.int_layout is RKLayout.INT32:
      return self._static_slot(dtype, self.int_layout, (int(u.arg) & 0xffffffff,) * self.count)
    if dtype in (dtypes.half, dtypes.float): bits, layout = struct.pack("<e", float(u.arg)), RKLayout.FP16
    elif dtype in (dtypes.int16, dtypes.uchar): bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), RKLayout.INT16
    elif dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16:
      bits, layout = struct.pack("<e", float(u.arg)), self.int_layout
    elif dtype is dtypes.int and self.int_layout is RKLayout.INT16:
      bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), self.int_layout
    elif dtype is dtypes.bool: bits, layout = struct.pack("<e", float(bool(u.arg))), RKLayout.BOOL_MASK
    else: raise _RKGenericReject(f"constant {dtype}")
    if bits in self.constants:
      slot = self.constants[bits]
      return RKValue(RKArg(RKBufferKind.SCRATCH, slot), dtype, self.count, layout)
    value = self._scratch(dtype, layout)
    self.constants[bits] = value.arg.index
    return value

  def _operand(self, u:UOp, dtype:DType) -> RKValue:
    return self._constant(u, dtype) if u.op is Ops.CONST and \
      (u.dtype.scalar() in dtypes.weaks or dtype is dtypes.half and u.dtype.scalar() is dtypes.float) else self.lower(u)

  def _static(self, u:UOp, bool_layout:RKLayout=RKLayout.BOOL_MASK) -> RKValue:
    dtype = u.dtype.scalar()
    if not _index_ranges(u):
      scalar = _eval_expr(u, {}, {})
      if dtype is dtypes.bool and bool_layout is RKLayout.BOOL_INT16:
        value = self._constant(UOp.const(int(bool(scalar)), dtypes.int16))
        return RKValue(value.arg, dtype, self.count, bool_layout)
      return self._constant(UOp.const(scalar, dtype))
    if dtype is dtypes.half: vector, layout = _static_values(self.out_index, u, self.count, _fp16_bits), RKLayout.FP16
    elif dtype is dtypes.int16: vector, layout = _static_values(self.out_index, u, self.count, _int16_bits), RKLayout.INT16
    elif dtype is dtypes.int:
      values = _static_values(self.out_index, u, self.count, int)
      if self.int_layout is RKLayout.INT_FP16 and all(-2048 <= value <= 2048 for value in values):
        vector, layout = tuple(_fp16_bits(float(value)) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT16 and all(-32768 <= value <= 32767 for value in values):
        vector, layout = tuple(_int16_bits(value) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT32:
        vector, layout = tuple(value & 0xffffffff for value in values), self.int_layout
      else: raise _RKGenericReject
    elif dtype is dtypes.bool:
      if bool_layout is RKLayout.BOOL_INT16: vector, layout = _static_values(self.out_index, u, self.count, lambda x:int(bool(x))), bool_layout
      else: vector, layout = _static_values(self.out_index, u, self.count, lambda x:_fp16_bits(float(bool(x)))), RKLayout.BOOL_MASK
    else: raise _RKGenericReject
    return self._static_slot(dtype, layout, vector)

  def _load(self, u:UOp, fill_override:int|None=None) -> RKValue:
    dtype = u.dtype.scalar()
    if dtype not in (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool) or not u.src or u.src[0].op is not Ops.INDEX or \
       (param:=_root_param(u.src[0])) is None or param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST:
      raise _RKGenericReject
    index, gate, default = u.src[0].src[1], u.src[2] if len(u.src) > 2 else None, u.src[1] if len(u.src) > 1 else None
    index_nodes, gate_nodes = index.toposort(), () if gate is None else gate.toposort()
    runtime_address = any(x.op is Ops.LOAD for x in (*index_nodes, *gate_nodes))
    if default is not None and default.op is not Ops.CONST:
      if dtype not in (dtypes.half, dtypes.int16, dtypes.int) or gate is None or runtime_address:
        raise _RKGenericReject
      expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
      itemsize = 4 if expected is RKLayout.INT32 else 2
      schedule = len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)
      fallback = self.lower(default)
      if fallback.layout is not expected or fallback.count != self.count or \
         schedule != (len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)):
        raise _RKGenericReject
      plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      value = self._scratch(dtype, expected, self.count*itemsize)
      self.gathers.append(RKGather(fallback.arg.index, value.arg.index, self.count,
        base=fallback.arg.addend//itemsize, axes=((1, self.count, 1),), src_kind=fallback.arg.kind, itemsize=itemsize))
      self.gathers.append(replace(plan, dst_index=value.arg.index, partial=True, itemsize=itemsize))
      return value
    if dtype is dtypes.float:
      if runtime_address: raise _RKGenericReject
      fill_bits = struct.unpack("<I", struct.pack("<f", float(0 if default is None else default.arg)))[0]
      plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count, fill_bits)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      groups = tuple(range(0, self.count, _EW_ELEMS_32BIT))
      raw_key = ("fp32_raw", _gather_cache_key((replace(plan, itemsize=4),)))
      raw = self._materialized_slot(raw_key, dtype, RKLayout.FP16, replace(plan, itemsize=4), len(groups)*16)
      aligned = self._scratch(dtypes.half, RKLayout.FP16, len(groups)*16)
      zero = self._scratch(dtype, RKLayout.FP16, 16)
      self.gathers.append(RKGather(0, zero.arg.index, _EW_ELEMS_32BIT, values=(0,)*_EW_ELEMS_32BIT, itemsize=4))
      for group,start in enumerate(groups):
        lanes = min(_EW_ELEMS_32BIT, self.count-start)
        source = replace(raw.arg, addend=group*16)
        self.ew_ops.append(RKEWOp(replace(aligned.arg, addend=group*16), source, zero.arg, lanes,
                                  _EW_CFG[Ops.ADD] | _EW_STAGE_FP32_IN, stateful=True))
      compact = self._scratch(dtypes.half, RKLayout.FP16, self.count*2)
      self.mid_gathers.append(RKGather(aligned.arg.index, compact.arg.index, self.count,
        offsets=tuple((lane//_EW_ELEMS_32BIT)*8+lane%_EW_ELEMS_32BIT for lane in range(self.count)),
        src_kind=RKBufferKind.SCRATCH, after=len(self.ew_ops)))
      return RKValue(compact.arg, dtype, self.count, RKLayout.FP16)
    if dtype is dtypes.bool:
      if runtime_address: raise _RKGenericReject
      offsets = _gather_offsets(self.out_index, index, gate, self.count)
      plan = RKGather(param.arg.slot, 0, self.count, offsets=offsets, fill_bits=int(bool(default.arg)) if default is not None else 0,
                      dst_stride=2, itemsize=1)
      _validate_gather_bounds(plan, int(param.src[0].arg))
      return self._gather_slot(dtype, RKLayout.BOOL_INT16, plan, self.count*2)
    layout = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
    itemsize = 4 if layout is RKLayout.INT32 else 2
    fill_bits = fill_override if fill_override is not None else _fp16_bits(0 if default is None else default.arg) if dtype is dtypes.half else \
      _int16_bits(0 if default is None else default.arg) if dtype is dtypes.int16 else int(0 if default is None else default.arg) & 0xffffffff
    if u not in self.static_load_offsets and runtime_address:
      if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1": raise _RKGenericReject
      runtime_index = _runtime_affine_index(index, self.out_index, self.count)
      runtime_loads = {node.key:node for node in (*index_nodes, *gate_nodes)
                       if _runtime_index(node) is not None}
      if runtime_index is not None:
        runtime_load, index_param, index_itemsize, index_offset, base, index_scale, lane_stride = runtime_index
      elif len(runtime_loads) == 1 and (runtime_info:=_runtime_index(next(iter(runtime_loads.values())))) is not None:
        runtime_load, index_param, index_lane, index_itemsize = runtime_info
        try: index_lanes = _static_int_vector(self.out_index, index_lane, self.count)
        except RuntimeError: raise _RKGenericReject from None
        index_offset = index_lanes[0]
        if index_lanes != tuple(index_offset+lane for lane in range(self.count)) or \
           not 0 <= index_offset <= int(index_param.src[0].arg)-self.count: raise _RKGenericReject
        base = index_scale = lane_stride = 0
      else: raise _RKGenericReject
      index_limit = int(param.src[0].arg)
      if gate is not None:
        limits = [int(node.src[1].arg) for node in gate_nodes if node.op is Ops.CMPLT and
                  node.src[0].key == runtime_load.key and node.src[1].op is Ops.CONST and int(node.src[1].arg) > 0]
        if len(set(limits)) != 1 or not _bounded_index_gate(gate, runtime_load, limits[0]) or \
           {node.key for node in gate_nodes if node.op is Ops.LOAD} != {runtime_load.key}: raise _RKGenericReject
        index_limit = limits[0]
      source = RKArg(RKBufferKind.ARG, param.arg.slot)
      source_count = int(param.src[0].arg)
      if runtime_index is None:
        if gate is None or index_limit <= 0 or self.count*index_limit > _MAX_STATIC_RANGE_ENVS: raise _RKGenericReject
        try:
          candidates = tuple(_static_int_vector(self.out_index, index.substitute({runtime_load:runtime_load.const_like(candidate)}), self.count)
                             for candidate in range(index_limit))
        except RuntimeError: raise _RKGenericReject from None
        offsets = tuple(candidates[candidate][lane] for lane in range(self.count) for candidate in range(index_limit))
        plan = RKGather(param.arg.slot, 0, len(offsets), offsets=offsets, itemsize=itemsize)
        _validate_gather_bounds(plan, source_count)
        source, source_count = self._gather_slot(dtype, layout, plan, len(offsets)*itemsize).arg, len(offsets)
        base, index_scale, lane_stride = 0, 1, index_limit
      value = self._scratch(dtype, layout, self.count*itemsize)
      self.host_gathers.append(RKHostAddress(source,
        RKArg(RKBufferKind.ARG, index_param.arg.slot, index_offset*index_itemsize), value.arg,
        self.count, source_count, self.count,
        itemsize=itemsize, index_itemsize=index_itemsize, fill_bits=fill_bits, index_limit=index_limit,
        base=base, index_scale=index_scale, lane_stride=lane_stride))
      return value
    if gate is None and index.key == self.out_index.key and int(param.src[0].arg) == self.count:
      return RKValue(RKArg(RKBufferKind.ARG, param.arg.slot), dtype, self.count, layout)
    plan = RKGather(param.arg.slot, 0, self.count, offsets=self.static_load_offsets[u], fill_bits=fill_bits) if u in self.static_load_offsets else \
      _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count, fill_bits)
    _validate_gather_bounds(plan, int(param.src[0].arg))
    return self._gather_slot(dtype, layout, replace(plan, itemsize=itemsize), self.count*itemsize)

  def _emit(self, dst:RKValue, lhs:RKValue, rhs:RKValue, cfg:int, *, compare:bool=False) -> RKValue:
    integer16, integer32 = dst.layout in (RKLayout.INT16, RKLayout.BOOL_INT16), dst.layout is RKLayout.INT32
    if integer16 and (lhs.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16) or
                      rhs.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16)): raise _RKGenericReject
    if integer32 and (lhs.layout is not RKLayout.INT32 or rhs.layout is not RKLayout.INT32): raise _RKGenericReject
    if not integer16 and not integer32 and lhs.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16) or \
       not integer16 and not integer32 and rhs.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16): raise _RKGenericReject
    barrier = not integer16 and not integer32 and cfg in (_EW_CFG_FLOOR, _EW_CFG[Ops.FDIV])
    self.ew_ops.append(RKEWOp(dst.arg, lhs.arg, rhs.arg, self.count, cfg, submit_barrier=barrier,
      compare=compare, stateful=integer32 or not integer16 and (self.mask_program and not compare or barrier),
      int16_output=integer16, int16_input=integer16, int32_output=integer32, int32_input=integer32))
    self.mask_program |= compare
    return dst

  def _masked_where(self, u:UOp, dtype:DType, layout:RKLayout, selector:RKValue, yes:RKValue, no:RKValue, one:RKValue) -> RKValue:
    selected_yes, inverse, selected_no = (self._scratch(one.dtype, one.layout) for _ in range(3))
    self._emit(selected_yes, selector, yes, _EW_CFG[Ops.MUL])
    self._emit(inverse, one, selector, _EW_CFG[Ops.SUB])
    self._emit(selected_no, inverse, no, _EW_CFG[Ops.MUL])
    return self._emit(self._dst(u, dtype, layout), selected_yes, selected_no, _EW_CFG[Ops.ADD])

  def _native_min(self, u:UOp, lhs:RKValue, rhs:RKValue) -> RKValue:
    zero = self.lower(UOp.const(0.0, dtypes.half))
    neg_lhs, neg_rhs = (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(2))
    self._emit(neg_lhs, zero, lhs, _EW_CFG[Ops.SUB])
    self._emit(neg_rhs, zero, rhs, _EW_CFG[Ops.SUB])
    self._emit(neg_lhs, neg_lhs, neg_rhs, _EW_CFG[Ops.MAX])
    dst = self._dst(u, dtypes.half, RKLayout.FP16) if u is self.root else neg_lhs
    return self._emit(dst, zero, neg_lhs, _EW_CFG[Ops.SUB])

  def _raw_bytes(self, value:RKValue) -> tuple[RKValue, RKValue]:
    if value.layout not in (RKLayout.FP16, RKLayout.INT16): raise _RKGenericReject
    if value.arg in self.raw_bytes: return self.raw_bytes[value.arg]
    low, high = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(2))
    split_after = len(self.ew_ops)
    for byte,part in enumerate((low, high)):
      self.mid_gathers.append(RKGather(value.arg.index, part.arg.index, self.count,
        base=value.arg.addend+byte, axes=((1, self.count, 2),), dst_stride=2,
        src_kind=value.arg.kind, itemsize=1, after=split_after))
    self.raw_bytes[value.arg] = low, high
    return self.raw_bytes[value.arg]

  def _pack_int16_bytes(self, u:UOp, low:RKValue, high:RKValue, mask:int|None=None) -> RKValue:
    result = self._dst(u, dtypes.int16, RKLayout.INT16)
    pack_after = len(self.ew_ops)
    for byte,source in enumerate((low, high)):
      self.mid_gathers.append(RKGather(source.arg.index, result.arg.index, self.count,
        base=source.arg.addend, axes=((1, self.count, 2),), dst_stride=2, dst_addend=byte,
        dst_kind=result.arg.kind, src_kind=source.arg.kind, itemsize=1, after=pack_after))
    self.raw_bytes[result.arg] = low, high
    if mask is not None: self.int16_masks[result.arg] = mask
    return result

  def _alu(self, u:UOp) -> RKValue:
    if u.op is Ops.RECIPROCAL:
      src = self.lower(u.src[0]); one = self.lower(UOp.const(1.0, dtypes.half))
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), one, src, _EW_CFG[Ops.FDIV])
    if u.op is Ops.NEG:
      src = self.lower(u.src[0])
      dst = self._dst(u, u.dtype.scalar(), src.layout)
      return self._emit(dst, src, src, _EW_CFG_NEG)
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.ADD and u.arg is None and (recipe:=_fold_relu_cap(u)) is not None:
      return self.lower(recipe)
    if u.op is Ops.FDIV and (recipe:=_preserve_infinite_division_sign(u)) is not None:
      return self.lower(recipe)
    dtype = u.dtype.scalar()
    int_range = _exact_int_range(u) if dtype is dtypes.int else None
    bounded = (self.int_layout is RKLayout.INT_FP16 and int_range is not None and -2048 <= int_range[0] <= int_range[1] <= 2048 or
               self.int_layout is RKLayout.INT16 and int_range is not None and -32768 <= int_range[0] <= int_range[1] <= 32767 or
               self.int_layout is RKLayout.INT32)
    expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if bounded else None
    if expected is None: raise _RKGenericReject(f"alu {u.op.name} {dtype} bounds={int_range}")
    def operand(src:UOp) -> RKValue:
      if (u.op is Ops.MAX and dtype is dtypes.half and src.op is Ops.CONST and math.isinf(float(src.arg)) and float(src.arg) < 0):
        return self._constant(UOp.const(-65504.0, dtypes.half))
      if (u.op is Ops.MAX and dtype is dtypes.half and src.op is Ops.LOAD and len(src.src) > 2 and src.src[1].op is Ops.CONST and
          math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0 and _is_static_expr(src.src[2])):
        return self._load(src, _fp16_bits(-65504.0))
      return self._operand(src, dtype)
    lhs, rhs = operand(u.src[0]), operand(u.src[1])
    compatible = (RKLayout.FP16, RKLayout.BOOL_MASK) if expected is RKLayout.FP16 else (expected,)
    if lhs.layout not in compatible or rhs.layout not in compatible: raise _RKGenericReject
    if u.op is Ops.SUB and u.arg == _NATIVE_SIGN:
      if expected is not RKLayout.FP16 or lhs.layout is not RKLayout.FP16: raise _RKGenericReject
      zero = self._constant(UOp.const(0.0, dtypes.half))
      negative = self._scratch(dtypes.half, RKLayout.FP16)
      negative_mask, positive_mask = (self._scratch(dtypes.bool, RKLayout.BOOL_MASK) for _ in range(2))
      self._emit(negative, zero, lhs, _EW_CFG[Ops.SUB])
      self._emit(negative_mask, negative, negative, _EW_CFG[Ops.MAX], compare=True)
      self._emit(positive_mask, lhs, lhs, _EW_CFG[Ops.MAX], compare=True)
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), positive_mask, negative_mask, _EW_CFG[Ops.SUB])
    if u.op is Ops.MAX and u.arg == _NATIVE_MIN:
      if expected is RKLayout.FP16: return self._native_min(u, lhs, rhs)
      dst = self._dst(u, dtype, RKLayout.INT16)
      return self._emit(dst, lhs, rhs, _EW_CFG_MIN)
    if u.op is Ops.MAX and u.arg == _NATIVE_RAW_MIN:
      dst = self._dst(u, dtype, expected)
      return self._emit(dst, lhs, rhs, _EW_CFG_MIN)
    cfg = _EW_CFG_ABS if u.op is Ops.MAX and u.arg == _NATIVE_ABS else \
      _EW_CFG_FLOOR if u.op is Ops.MAX and u.arg == _NATIVE_FLOOR else \
      _EW_CFG_CEIL if u.op is Ops.MAX and u.arg == _NATIVE_CEIL else \
      _EW_CFG_RELU6 if u.op is Ops.MAX and u.arg == _NATIVE_RELU6 else \
      _EW_CFG_LEAKY_RELU if u.op is Ops.MUL and u.arg == _NATIVE_LEAKY_RELU else _EW_CFG[u.op]
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    layout = RKLayout.BOOL_MASK if compare else expected
    out_dtype = dtypes.bool if compare else dtype
    dst = self._dst(u, out_dtype, layout)
    return self._emit(dst, lhs, rhs, cfg, compare=compare)

  def _coerce_bool(self, value:RKValue, layout:RKLayout) -> RKValue:
    if value.layout is layout: return value
    if value.layout is not RKLayout.BOOL_MASK or layout is not RKLayout.BOOL_INT16: raise _RKGenericReject
    converted = self._scratch(dtypes.bool, RKLayout.BOOL_INT16)
    self.ew_ops.append(RKEWOp(converted.arg, value.arg, value.arg, self.count, _EW_CFG[Ops.MAX], submit_barrier=True,
                             stateful=True, int16_output=True))
    return converted

  def _bool_binary(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.CMPNE:
      for expression,marker in (u.src, u.src[::-1]):
        if (marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and
            expression.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in expression.src)):
          less = self.lower(expression)
          if less.layout is not RKLayout.BOOL_INT16: raise _RKGenericReject
          operands = tuple(self._operand(src, dtypes.half) for src in expression.src)
          nan = tuple(self._fp16_component_values(value)[2] for value in operands)
          one = self._constant(UOp.const(1, dtypes.int16))
          inverse, either_nan, numeric = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(3))
          self._emit(inverse, one, less, _EW_CFG[Ops.SUB])
          self._emit(either_nan, RKValue(nan[0], dtypes.int16, self.count, RKLayout.INT16),
                     RKValue(nan[1], dtypes.int16, self.count, RKLayout.INT16), _EW_CFG[Ops.MAX])
          self._emit(numeric, one, either_nan, _EW_CFG[Ops.SUB])
          return self._emit(self._dst(u, dtypes.bool, RKLayout.BOOL_INT16), inverse, numeric, _EW_CFG[Ops.MUL])
    values = [self.lower(src) if not (src.op is Ops.CONST and src.dtype.scalar() is dtypes.bool) else None for src in u.src]
    preferred = (RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in values) else
                 RKLayout.BOOL_MASK)
    lhs, rhs = (self._static(src, preferred) if value is None else self._coerce_bool(value, preferred)
                for src,value in zip(u.src, values))
    dst = self._dst(u, dtypes.bool, preferred)
    if u.op is Ops.AND: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MUL])
    if u.op is Ops.OR: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MAX])
    data_dtype, data_layout = (dtypes.int16, RKLayout.INT16) if preferred is RKLayout.BOOL_INT16 else (dtypes.half, RKLayout.FP16)
    delta = self._scratch(data_dtype, data_layout)
    self._emit(delta, lhs, rhs, _EW_CFG[Ops.SUB])
    if u.op in (Ops.XOR, Ops.CMPNE): return self._emit(dst, delta, delta, _EW_CFG_ABS)
    if u.op is Ops.CMPEQ:
      raw_one = self._constant(UOp.const(1, dtypes.int16)) if preferred is RKLayout.BOOL_INT16 else self.lower(UOp.const(True, dtypes.bool))
      one = RKValue(raw_one.arg, dtypes.bool, self.count, preferred)
      unequal = self._scratch(dtypes.bool, preferred)
      self._emit(unequal, delta, delta, _EW_CFG_ABS)
      return self._emit(dst, one, unequal, _EW_CFG[Ops.SUB])
    raise _RKGenericReject

  def _integer_bitwise(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.AND:
      for marker,source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or (mask:=int(marker.arg)&0xffff) not in (0x7fff, 0x8000): continue
        value = self.lower(source)
        if value.layout is not RKLayout.INT16: raise _RKGenericReject
        low, high = self._raw_bytes(value)
        zero, one, const127, const128 = (self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128))
        delta, positive, sign, sign_scale = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(4))
        self._emit(delta, high, const127, _EW_CFG[Ops.SUB])
        self._emit(positive, delta, zero, _EW_CFG[Ops.MAX])
        self._emit(sign, positive, one, _EW_CFG_MIN)
        self._emit(sign_scale, sign, const128, _EW_CFG[Ops.MUL])
        if mask == 0x7fff:
          masked_high = self._scratch(dtypes.int16, RKLayout.INT16)
          self._emit(masked_high, high, sign_scale, _EW_CFG[Ops.SUB])
          return self._pack_int16_bytes(u, low, masked_high, mask)
        return self._pack_int16_bytes(u, zero, sign_scale, mask)
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.OR:
      values = tuple(self.lower(source) for source in u.src)
      masks = tuple(self.int16_masks.get(value.arg) for value in values)
      if all(mask is not None for mask in masks) and typing_cast(int, masks[0]) & typing_cast(int, masks[1]) == 0:
        parts = tuple(self._raw_bytes(value) for value in values)
        low, high = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(2))
        self._emit(low, parts[0][0], parts[1][0], _EW_CFG[Ops.ADD])
        self._emit(high, parts[0][1], parts[1][1], _EW_CFG[Ops.ADD])
        return self._pack_int16_bytes(u, low, high, typing_cast(int, masks[0]) | typing_cast(int, masks[1]))
    if u.op is Ops.XOR:
      for marker, source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or int(marker.arg) != -1: continue
        dtype = u.dtype.scalar()
        layout = RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if dtype is dtypes.int else None
        if layout is None: raise _RKGenericReject
        rhs = self.lower(source)
        if rhs.layout is not layout: raise _RKGenericReject
        if layout is RKLayout.INT32:
          components = self._int32_bytes(rhs)
          const255 = self._constant(UOp.const(255, dtypes.int16))
          inverted = tuple(self._emit(self._scratch(dtypes.int16, RKLayout.INT16), const255, component, _EW_CFG[Ops.SUB])
                           for component in components)
          result = self._scratch(dtypes.int, RKLayout.INT32)
          pack_after = len(self.ew_ops)
          for byte,component in enumerate(inverted):
            self.mid_gathers.append(RKGather(component.arg.index, result.arg.index, self.count,
              base=component.arg.addend, axes=((1, self.count, 2),), dst_stride=4, dst_addend=byte,
              src_kind=component.arg.kind, itemsize=1, after=pack_after))
          self.int32_components[result.arg] = inverted
          return result
        lhs = self._constant(UOp.const(-1, dtype))
        return self._emit(self._dst(u, dtype, layout), lhs, rhs, _EW_CFG[Ops.SUB])
    if u.dtype.scalar() is not dtypes.int16 or u.op not in (Ops.AND, Ops.OR, Ops.XOR): raise _RKGenericReject
    values = tuple(self.lower(source) for source in u.src)
    if any(value.layout is not RKLayout.INT16 for value in values): raise _RKGenericReject
    lanes = self.count*2
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16, _scratch_bytes(lanes)).arg
    raw:list[RKArg] = []
    for value in values:
      expanded = allocate(); raw.append(expanded)
      self.mid_gathers.append(RKGather(value.arg.index, expanded.index, lanes, base=value.arg.addend,
        axes=((1, lanes, 1),), dst_stride=2, src_kind=value.arg.kind, itemsize=1, after=len(self.ew_ops)))
    constants:dict[int, RKArg] = {}
    for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128):
      constants[number] = constant = allocate()
      self.gathers.append(RKGather(self.out_param.arg.slot, constant.index, lanes, values=(number,)*lanes))
    lhs_bits, rhs_bits = (_int16_byte_bits(self.ew_ops, allocate, constants, value, lanes) for value in raw)
    weighted:list[RKArg] = []
    for bit,(left,right) in enumerate(zip(lhs_bits, rhs_bits)):
      combined = allocate()
      if u.op is Ops.XOR:
        self.ew_ops.extend((RKEWOp(combined, left, right, lanes, _EW_CFG[Ops.SUB], **_INT16_EW),
                            RKEWOp(combined, combined, combined, lanes, _EW_CFG_ABS, **_INT16_EW)))
      else:
        self.ew_ops.append(RKEWOp(combined, left, right, lanes,
          _EW_CFG_MIN if u.op is Ops.AND else _EW_CFG[Ops.MAX], **_INT16_EW))
      if bit: self.ew_ops.append(RKEWOp(combined, combined, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
      weighted.append(combined)
    combined = _reduce_rows(self.ew_ops, weighted, lanes, _EW_CFG[Ops.ADD], int16=True)
    result = self._dst(u, dtypes.int16, RKLayout.INT16)
    self.mid_gathers.append(RKGather(combined.index, result.arg.index, lanes, base=combined.addend,
      axes=((1, lanes, 2),), dst_stride=1, dst_kind=result.arg.kind,
      src_kind=RKBufferKind.SCRATCH, itemsize=1, after=len(self.ew_ops)))
    return result

  def _compare(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src): return self._bool_binary(u)
    if u.op in (Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_equality(u)
    if u.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_less(u)
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      sources = tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src)
      bounds = tuple(_exact_int_range(src) for src in sources)
      if self.int_layout is RKLayout.INT_FP16 or self.int_layout is not RKLayout.INT32 and all(
        bound is not None and -2048 <= bound[0] <= bound[1] <= 2048 for bound in bounds
      ):
        recipe = UOp(u.op, dtypes.bool, src=tuple(_int_fp16_expr(src) for src in sources), arg=u.arg)
        value = self.lower(recipe)
        if value.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
        return value
      return self._int32_compare(u.replace(src=sources))
    if all(src.dtype.scalar() is dtypes.int16 for src in u.src):
      if (int16_recipe:=_native_int16_comparison(u)) is None: raise _RKGenericReject
      value = self.lower(int16_recipe)
      if value.layout is not RKLayout.INT16: raise _RKGenericReject
      return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_INT16)
    predicate = UOp(Ops.CMPNE, src=u.src) if u.op is Ops.CMPEQ else u
    if (ieee_recipe:=_ieee_comparison_mask(predicate)) is None: raise _RKGenericReject
    if u.op is Ops.CMPEQ: ieee_recipe = UOp.const(1.0, dtypes.half).alu(Ops.SUB, ieee_recipe)
    return self._ieee_bool(ieee_recipe)

  def _ieee_bool(self, recipe:UOp) -> RKValue:
    value = self.lower(recipe)
    if value.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK): raise _RKGenericReject
    return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_MASK)

  def _int32_bytes(self, source:RKValue) -> tuple[RKValue, ...]:
    if source.layout is not RKLayout.INT32: raise _RKGenericReject
    if source.arg in self.int32_components: return self.int32_components[source.arg]
    copied = self._scratch(dtypes.int, RKLayout.INT32)
    self._emit(copied, source, source, _EW_CFG[Ops.MAX])
    after = len(self.ew_ops)
    components = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(4))
    for byte,component in enumerate(components):
      self.mid_gathers.append(RKGather(copied.arg.index, component.arg.index, self.count,
        base=copied.arg.addend+byte, axes=((1, self.count, 4),), dst_stride=2,
        src_kind=RKBufferKind.SCRATCH, itemsize=1, after=after))
    self.int32_components[source.arg] = components
    return components

  def _int32_compare(self, u:UOp) -> RKValue:
    def operand(src:UOp) -> RKValue:
      value = self._static(src) if src in self.static_nodes else self.lower(src)
      if value.layout is not RKLayout.INT32: raise _RKGenericReject
      return value
    lhs, rhs = (operand(src) for src in u.src)
    lhs_bytes, rhs_bytes = self._int32_bytes(lhs), self._int32_bytes(rhs)
    constants = {value:self._constant(UOp.const(value, dtypes.int16)).arg for value in (0, 1, 127, 128, 256)}
    if u.op is Ops.CMPLT:
      mask = _int32_less_mask(self.ew_ops, self._int16_arg, constants, [value.arg for value in lhs_bytes[::-1]],
                              [value.arg for value in rhs_bytes[::-1]], self.count)
    else:
      equal = constants[1]
      for left,right in zip(lhs_bytes, rhs_bytes):
        byte_equal = _ew_native_int16_eq_mask(self.ew_ops, self._int16_arg, left.arg, right.arg, constants[1], self.count)
        selected = self._int16_arg()
        self.ew_ops.append(RKEWOp(selected, equal, byte_equal, self.count, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
        equal = selected
      if u.op is Ops.CMPEQ: mask = equal
      elif u.op is Ops.CMPNE:
        mask = self._int16_arg()
        self.ew_ops.append(RKEWOp(mask, constants[1], equal, self.count, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True))
      else: raise _RKGenericReject
    return RKValue(mask, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_equality(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 equality through raw bytes and native INT16 arithmetic, without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    lhs_low,lhs_high,lhs_nan = self._fp16_component_values(values[0])
    rhs_low,rhs_high,rhs_nan = self._fp16_component_values(values[1])
    low_equal = _ew_native_int16_eq_mask(self.ew_ops, self._int16_arg, lhs_low.arg, rhs_low.arg, constants[1].arg, self.count)
    high_equal = _ew_native_int16_eq_mask(self.ew_ops, self._int16_arg, lhs_high, rhs_high, constants[1].arg, self.count)
    either_nan, numeric, bits_equal, equal = (self._int16_arg() for _ in range(4))
    self.ew_ops.extend((RKEWOp(either_nan, lhs_nan, rhs_nan, self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
                        RKEWOp(numeric, constants[1].arg, either_nan, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
                        RKEWOp(bits_equal, low_equal, high_equal, self.count, _EW_CFG[Ops.MUL], **_INT16_EW),
                        RKEWOp(equal, bits_equal, numeric, self.count, _EW_CFG[Ops.MUL], **_INT16_EW)))
    if u.op is Ops.CMPNE:
      unequal = self._int16_arg()
      self.ew_ops.append(RKEWOp(unequal, constants[1].arg, equal, self.count, _EW_CFG[Ops.SUB], **_INT16_EW))
      equal = unequal
    return RKValue(equal, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_component_values(self, value:RKValue) -> tuple[RKValue, RKArg, RKArg]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if value.arg in self.fp16_components: return self.fp16_components[value.arg]
    low, high = self._raw_bytes(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 123, 124, 127, 128)}
    clean_high,nan = _fp16_high_and_nan(self.ew_ops, self._int16_arg, high.arg, low.arg,
      constants[0].arg, constants[1].arg, constants[123].arg, constants[124].arg,
      constants[127].arg, constants[128].arg, self.count)
    self.fp16_components[value.arg] = low, clean_high, nan
    return self.fp16_components[value.arg]

  def _fp16_ordered_values(self, value:RKValue) -> tuple[RKArg, RKArg]:
    """Map a classified FP16 lane to two unsigned bytes whose lexical order is IEEE numeric order."""
    if value.arg in self.fp16_ordered: return self.fp16_ordered[value.arg]
    low, clean_high, _ = self._fp16_component_values(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128, 255)}
    sign_delta, sign_positive, sign = (self._int16_arg() for _ in range(3))
    positive_high, negative_high, high_delta, high_selected, ordered_high = (self._int16_arg() for _ in range(5))
    negative_low, low_delta, low_selected, ordered_low = (self._int16_arg() for _ in range(4))
    self.ew_ops.extend((
      RKEWOp(sign_delta, clean_high, constants[127].arg, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
      RKEWOp(sign_positive, sign_delta, constants[0].arg, self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
      RKEWOp(sign, sign_positive, constants[1].arg, self.count, _EW_CFG_MIN, **_INT16_EW),
      RKEWOp(positive_high, clean_high, constants[128].arg, self.count, _EW_CFG[Ops.ADD], **_INT16_EW),
      RKEWOp(negative_high, constants[255].arg, clean_high, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
      RKEWOp(high_delta, negative_high, positive_high, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
      RKEWOp(high_selected, sign, high_delta, self.count, _EW_CFG[Ops.MUL], **_INT16_EW),
      RKEWOp(ordered_high, positive_high, high_selected, self.count, _EW_CFG[Ops.ADD], **_INT16_EW),
      RKEWOp(negative_low, constants[255].arg, low.arg, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
      RKEWOp(low_delta, negative_low, low.arg, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
      RKEWOp(low_selected, sign, low_delta, self.count, _EW_CFG[Ops.MUL], **_INT16_EW),
      RKEWOp(ordered_low, low.arg, low_selected, self.count, _EW_CFG[Ops.ADD], **_INT16_EW)))
    self.fp16_ordered[value.arg] = ordered_high, ordered_low
    return self.fp16_ordered[value.arg]

  def _fp16_less(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 less-than as an ordered raw-byte comparison without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    ordered = tuple(self._fp16_ordered_values(value) for value in values)
    nan = tuple(self._fp16_component_values(value)[2] for value in values)
    less, equal = constants[0].arg, constants[1].arg
    for lhs,rhs in zip(ordered[0], ordered[1]):
      maximum, lhs_delta, rhs_delta, lhs_less, rhs_less, unequal, same, selected, next_less, next_equal = \
        (self._int16_arg() for _ in range(10))
      self.ew_ops.extend((RKEWOp(maximum, lhs, rhs, self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
        RKEWOp(lhs_delta, maximum, lhs, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
        RKEWOp(rhs_delta, maximum, rhs, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
        RKEWOp(lhs_less, lhs_delta, constants[1].arg, self.count, _EW_CFG_MIN, **_INT16_EW),
        RKEWOp(rhs_less, rhs_delta, constants[1].arg, self.count, _EW_CFG_MIN, **_INT16_EW),
        RKEWOp(unequal, lhs_less, rhs_less, self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
        RKEWOp(same, constants[1].arg, unequal, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
        RKEWOp(selected, equal, lhs_less, self.count, _EW_CFG[Ops.MUL], **_INT16_EW),
        RKEWOp(next_less, less, selected, self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
        RKEWOp(next_equal, equal, same, self.count, _EW_CFG[Ops.MUL], **_INT16_EW)))
      less, equal = next_less, next_equal
    either_nan, numeric, result = (self._int16_arg() for _ in range(3))
    self.ew_ops.extend((RKEWOp(either_nan, nan[0], nan[1], self.count, _EW_CFG[Ops.MAX], **_INT16_EW),
                       RKEWOp(numeric, constants[1].arg, either_nan, self.count, _EW_CFG[Ops.SUB], **_INT16_EW),
                       RKEWOp(result, less, numeric, self.count, _EW_CFG[Ops.MUL], **_INT16_EW)))
    return RKValue(result, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _raw_where(self, u:UOp, selector:RKValue, yes:RKValue, no:RKValue) -> RKValue:
    """Select arbitrary FP16 bit patterns with DPU INT16 byte arithmetic and raw layout gathers."""
    if selector.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16) or yes.layout is not no.layout or \
       yes.layout not in (RKLayout.FP16, RKLayout.INT_FP16, RKLayout.INT32): raise _RKGenericReject
    if selector.layout is RKLayout.BOOL_MASK:
      mask = self._scratch(dtypes.int16, RKLayout.BOOL_INT16)
      self.ew_ops.append(RKEWOp(mask.arg, selector.arg, selector.arg, self.count, _EW_CFG[Ops.MAX], submit_barrier=True,
                               stateful=True, int16_output=True))
    else: mask = RKValue(selector.arg, dtypes.int16, self.count, RKLayout.INT16)
    split_after = len(self.ew_ops)
    itemsize = 4 if yes.layout is RKLayout.INT32 else 2
    yes_bytes = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(itemsize))
    no_bytes = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(itemsize))
    for source,parts in ((yes, yes_bytes), (no, no_bytes)):
      for byte,part in enumerate(parts):
        self.mid_gathers.append(RKGather(source.arg.index, part.arg.index, self.count,
          base=source.arg.addend+byte, axes=((1, self.count, itemsize),), dst_stride=2,
          src_kind=source.arg.kind, itemsize=1, after=split_after))
    selected_bytes:list[RKValue] = []
    for yes_byte,no_byte in zip(yes_bytes, no_bytes):
      delta, selected, result = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(3))
      self._emit(delta, yes_byte, no_byte, _EW_CFG[Ops.SUB])
      self._emit(selected, RKValue(mask.arg, dtypes.int16, self.count, RKLayout.INT16), delta, _EW_CFG[Ops.MUL])
      selected_bytes.append(self._emit(result, no_byte, selected, _EW_CFG[Ops.ADD]))
    pack_after = len(self.ew_ops)
    result = self._scratch(u.dtype.scalar(), yes.layout)
    for byte,source in enumerate(selected_bytes):
      self.mid_gathers.append(RKGather(source.arg.index, result.arg.index, self.count,
        base=source.arg.addend, axes=((1, self.count, 2),), dst_stride=itemsize, dst_addend=byte,
        src_kind=RKBufferKind.SCRATCH, itemsize=1, after=pack_after))
    return result

  def _where(self, u:UOp) -> RKValue:
    if len(u.src) != 3: raise _RKGenericReject
    if u.dtype.scalar() is dtypes.bool:
      dynamic = [None if src in self.static_nodes else self.lower(src) for src in u.src]
      preferred = (RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in dynamic) else
                   RKLayout.BOOL_MASK)
      values = [self._static(src, preferred) if value is None else self._coerce_bool(value, preferred)
                for src,value in zip(u.src, dynamic)]
      selector, yes, no = values
      data_dtype = dtypes.int16 if preferred is RKLayout.BOOL_INT16 else dtypes.half
      one = self._constant(UOp.const(1, data_dtype))
      return self._masked_where(u, dtypes.bool, preferred, selector, yes, no, one)
    if u is self.root and u.dtype.scalar() is dtypes.int and all(
      arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.int for arm in u.src[1:]
    ):
      yes_int, no_int = (int(arm.arg) for arm in u.src[1:])
      try: exact = all(_eval_cast(value, dtypes.half) == value for value in (no_int, yes_int-no_int))
      except (OverflowError, struct.error): exact = False
      if not exact: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
      if selector.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
      arithmetic_dtype, arithmetic_layout = ((dtypes.int16, RKLayout.INT16) if selector.layout is RKLayout.BOOL_INT16 else
                                              (dtypes.half, RKLayout.FP16))
      delta, baseline = (self._constant(UOp.const(value, arithmetic_dtype)) for value in (yes_int-no_int, no_int))
      selected, result = (self._scratch(arithmetic_dtype, arithmetic_layout) for _ in range(2))
      self._emit(selected, selector, delta, _EW_CFG[Ops.MUL])
      self._emit(result, baseline, selected, _EW_CFG[Ops.ADD])
      if arithmetic_layout is RKLayout.INT16: return self._widen_int16(u, result)
      tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
      value = RKValue(self.out, dtypes.int, self.count, RKLayout.INT32)
      self.ew_ops.append(RKEWOp(value.arg, result.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
      return value
    if u is self.root and u.dtype.scalar() in (dtypes.half, dtypes.int16) and _is_static_expr(u.src[0]):
      dtype = u.dtype.scalar()
      routes:dict[UOp, list[bool]] = {}
      def route(node:UOp, active:tuple[bool, ...]) -> None:
        if node.op is Ops.WHERE and _is_static_expr(node.src[0]):
          selector = tuple(bool(x) for x in _static_values(self.out_index, node.src[0], self.count, lambda x:int(bool(x))))
          route(node.src[1], tuple(live and take for live,take in zip(active, selector)))
          route(node.src[2], tuple(live and not take for live,take in zip(active, selector)))
          return
        mask = routes.setdefault(node, [False]*self.count)
        for i,live in enumerate(active): mask[i] |= live
      route(u, (True,)*self.count)
      def exact_operand(src:UOp) -> RKValue:
        if (src.op is Ops.LOAD and dtype is dtypes.half and len(src.src) > 2 and src.src[1].op is Ops.CONST and
            math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0.0 and
            (param:=_root_param(src.src[0])) is not None and param.src[0].op is Ops.CONST and int(param.src[0].arg) < self.count):
          return self._load(src, _fp16_bits(-65504.0))
        return self.lower(src)
      expected = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16
      itemsize = dtype.itemsize
      for partial,(leaf,mask) in enumerate(routes.items()):
        value = exact_operand(leaf)
        if value.layout is not expected: raise _RKGenericReject
        offsets = tuple(value.arg.addend//itemsize+i if take else -1 for i,take in enumerate(mask))
        self.mid_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count, offsets=offsets,
          partial=bool(partial), dst_kind=RKBufferKind.ARG, src_kind=value.arg.kind, itemsize=itemsize))
      return RKValue(self.out, dtype, self.count, expected)
    condition_uop = _strip_cast(u.src[0])
    for fold in (_fold_where_abs, _fold_ordered_where, _fold_threshold_where):
      if (recipe:=fold(u)) is not None: return self.lower(recipe)
    if (u.src[1].op is Ops.EXP2 and u.src[2].op is Ops.CONST and float(u.src[2].arg) == 1.0 and
        len(u.src[1].src) == 1 and (scaled:=u.src[1].src[0]).op is Ops.MUL):
      infinite = next((factor for factor in scaled.src if factor.op is Ops.CONST and
                       math.isinf(float(factor.arg))), None)
      source = next((factor for factor in scaled.src if factor is not infinite), None)
      if (infinite is not None and source is not None and condition_uop.op is Ops.CMPNE and
          any(term.key == source.key for term in condition_uop.src) and
          any(term.op is Ops.CONST and float(term.arg) == 0.0 for term in condition_uop.src)):
        zero_u, one_u = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
        condition_value = self.lower(u.src[0])
        positive = self.lower(UOp(Ops.CMPLT, dtypes.bool, src=(zero_u, source)))
        negative_mask = self.lower(UOp(Ops.CMPLT, dtypes.bool, src=(source, zero_u)))
        if any(value.layout is not RKLayout.BOOL_MASK for value in (condition_value, positive, negative_mask)): raise _RKGenericReject
        one, zero = self._constant(one_u), self._constant(zero_u)
        signed = self._scratch(dtypes.bool, RKLayout.BOOL_MASK)
        self._emit(signed, positive, negative_mask, _EW_CFG[Ops.MAX])
        unordered = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(unordered, condition_value, signed, _EW_CFG[Ops.SUB])
        finite_zero = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(finite_zero, one, condition_value, _EW_CFG[Ops.SUB])
        overflow = negative_mask if float(infinite.arg) < 0.0 else positive
        overflow_denominator, overflow_quotient, overflow_correction = \
          (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(3))
        self._emit(overflow_denominator, one, overflow, _EW_CFG[Ops.SUB])
        self._emit(overflow_quotient, one, overflow_denominator, _EW_CFG[Ops.FDIV])
        self._emit(overflow_correction, overflow_quotient, one, _EW_CFG[Ops.SUB])
        unordered_denominator, unordered_correction = (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(2))
        self._emit(unordered_denominator, one, unordered, _EW_CFG[Ops.SUB])
        self._emit(unordered_correction, zero, unordered_denominator, _EW_CFG[Ops.FDIV])
        finite = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(finite, finite_zero, overflow_correction, _EW_CFG[Ops.ADD])
        return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, unordered_correction, _EW_CFG[Ops.ADD])
    nonfinite = [i for i,arm in enumerate(u.src[1:]) if arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.half and
                 not math.isfinite(float(arm.arg))]
    if len(nonfinite) == 1:
      inf_index, finite_u = nonfinite[0], u.src[2-nonfinite[0]]
      finite = self.lower(finite_u)
      if finite.layout is not RKLayout.FP16: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
      if selector.layout is RKLayout.BOOL_INT16:
        yes, no = (self.lower(src) for src in u.src[1:])
        return self._raw_where(u, selector, yes, no)
      if selector.layout is not RKLayout.BOOL_MASK: raise _RKGenericReject
      one = self._constant(UOp.const(1.0, dtypes.half))
      if inf_index == 0:
        denominator = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(denominator, one, selector, _EW_CFG[Ops.SUB])
      else: denominator = selector
      if math.isnan(float(u.src[1+inf_index].arg)):
        zero = self._constant(UOp.const(0.0, dtypes.half))
        correction = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(correction, zero, denominator, _EW_CFG[Ops.FDIV])
        return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, correction, _EW_CFG[Ops.ADD])
      sign = self._constant(UOp.const(math.copysign(1.0, float(u.src[1+inf_index].arg)), dtypes.half))
      quotient, correction = self._scratch(dtypes.half, RKLayout.FP16), self._scratch(dtypes.half, RKLayout.FP16)
      self._emit(quotient, sign, denominator, _EW_CFG[Ops.FDIV])
      self._emit(correction, quotient, sign, _EW_CFG[Ops.SUB])
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, correction, _EW_CFG[Ops.ADD])
    yes, no = (self.lower(src) for src in u.src[1:])
    if yes.layout is not no.layout or yes.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT_FP16, RKLayout.INT32):
      raise _RKGenericReject
    mask_layout = RKLayout.BOOL_INT16 if yes.layout in (RKLayout.INT16, RKLayout.INT32) else RKLayout.BOOL_MASK
    selector = self._static(u.src[0], mask_layout) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
    if selector.layout is not mask_layout and not (yes.layout in (RKLayout.FP16, RKLayout.INT_FP16, RKLayout.INT32) and
                                                    selector.layout in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16)):
      raise _RKGenericReject
    dtype = dtypes.half if yes.layout is RKLayout.FP16 else dtypes.int16 if yes.layout is RKLayout.INT16 else dtypes.int
    if dtype is dtypes.int16:
      one = self._constant(UOp.const(1, dtypes.int16))
      return self._masked_where(u, dtype, yes.layout, selector, yes, no, one)
    return self._raw_where(u, selector, yes, no)

  def _widen_int16(self, u:UOp, source:RKValue) -> RKValue:
    if source.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16) or \
       (u is not self.root and self.int_layout is not RKLayout.INT32) or \
       (u is self.root and self.out_param.dtype.scalar() is not dtypes.int):
      raise _RKGenericReject
    zero = self._constant(UOp.const(0, dtypes.int16))
    value = RKValue(self.out if u is self.root else self._scratch(dtypes.int, RKLayout.INT32).arg,
                    dtypes.int, self.count, RKLayout.INT32)
    self.ew_ops.append(RKEWOp(value.arg, source.arg, zero.arg, self.count, _EW_CFG[Ops.ADD], int16_input=True, int32_output=True))
    return value

  def _widen_exact_int(self, source:RKValue) -> None:
    if self.out_param.dtype.scalar() is not dtypes.int or source.layout is not RKLayout.INT_FP16: raise _RKGenericReject
    tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
    self.ew_ops.append(RKEWOp(self.out, source.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True))

  def _narrow_int32(self, source:RKValue) -> RKValue:
    if source.layout is not RKLayout.INT32: raise _RKGenericReject
    value = self._scratch(dtypes.half, RKLayout.FP16)
    tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
    self.ew_ops.append(RKEWOp(value.arg, source.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX], int32_input=True))
    return value

  def _math(self, u:UOp) -> RKValue:
    if len(u.src) != 1 or u.dtype.scalar() is not dtypes.half: raise _RKGenericReject
    if u.op is Ops.SQRT:
      if (recipe:=_dpu_sqrt(u.src[0])) is None: raise _RKGenericReject
    elif u.op is Ops.EXP2: recipe = _dpu_exp2(u.src[0])
    elif u.op is Ops.LOG2: recipe = _dpu_log2(u.src[0])
    elif u.op is Ops.SIN: recipe = _dpu_sin(u.src[0])
    else: raise _RKGenericReject
    recipe = _tag_precise_adds(recipe)
    value = self.lower(recipe)
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if u is self.root and value.arg != self.out:
      value = self._emit(RKValue(self.out, dtypes.half, self.count, RKLayout.FP16), value, value, _EW_CFG[Ops.MAX])
    return value

  def lower(self, u:UOp) -> RKValue:
    if u in self.values: return self.values[u]
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif (dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool) and u in self.static_nodes and
          not any(isinstance(node.arg, str) and node.arg.startswith("rockchip_") for node in u.toposort())):
      value = self._static(u)
    elif u.op is Ops.LOAD: value = self._load(u)
    elif u.op is Ops.BITCAST and len(u.src) == 1:
      source = self.lower(u.src[0])
      if dtype is dtypes.int16 and u.src[0].dtype.scalar() is dtypes.half and source.layout is RKLayout.FP16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.half and u.src[0].dtype.scalar() is dtypes.int16 and source.layout is RKLayout.INT16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      else: raise _RKGenericReject(f"bitcast {u.src[0].dtype.scalar()}->{dtype}")
      if u is self.root and value.arg != self.out:
        self.mid_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count,
          base=value.arg.addend//2, axes=((1, self.count, 1),), dst_kind=RKBufferKind.ARG,
          src_kind=value.arg.kind, itemsize=2))
        value = RKValue(self.out, dtype, self.count, value.layout)
    elif u.op is Ops.CAST and len(u.src) == 1:
      source_dtype = u.src[0].dtype.scalar()
      int_range = _exact_int_range(u.src[0]) if source_dtype is dtypes.int else None
      if dtype is dtypes.half and source_dtype is dtypes.float:
        if u.src[0].op is Ops.LOAD: source = self._load(u.src[0])
        else:
          recipe = _fp32_expr_to_half(u.src[0])
          source = self.lower(recipe)
      elif dtype is dtypes.half and source_dtype is dtypes.int and int_range is not None and \
           -_FP16_EXACT_INTEGER <= int_range[0] <= int_range[1] <= _FP16_EXACT_INTEGER:
        recipe = _int_fp16_expr(u.src[0])
        source = self.lower(recipe)
      else: source = self.lower(u.src[0])
      if dtype is dtypes.uchar and source_dtype is dtypes.half and source.layout is RKLayout.FP16:
        cast_source = u.src[0]
        if (relu:=_relu_operand(cast_source)) is not None: cast_source = relu.alu(Ops.MAX, UOp.const(0.0, dtypes.half))
        truncated = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(cast_source,)))
        quotient = _native_floor(truncated.alu(Ops.MUL, UOp.const(1.0/256.0, dtypes.half)))
        recipe = truncated.alu(Ops.SUB, quotient.alu(Ops.MUL, UOp.const(256.0, dtypes.half)))
        converted, value = self.lower(recipe), self._scratch(dtype, RKLayout.INT16)
        self.ew_ops.append(RKEWOp(value.arg, converted.arg, converted.arg, self.count, _EW_CFG[Ops.MAX],
                                  submit_barrier=True, stateful=True, int16_output=True))
      elif dtype is dtypes.bool and source_dtype is dtypes.half and source.layout is RKLayout.FP16:
        magnitude = UOp(Ops.MAX, dtypes.half, src=(u.src[0], u.src[0]), arg=_NATIVE_ABS)
        recipe = _positive_mask(magnitude)
        value = self.lower(recipe)
      elif dtype is dtypes.half and source.layout is RKLayout.INT32:
        value = self._narrow_int32(source)
      elif dtype is dtypes.float and source_dtype is dtypes.int and source.layout is RKLayout.INT32:
        value = self._narrow_int32(source)
      elif dtype is dtypes.float and source_dtype is dtypes.bool and source.layout is RKLayout.BOOL_INT16:
        recipe = u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half))
        value = self.lower(recipe)
      elif dtype is dtypes.half and source.layout is RKLayout.BOOL_INT16:
        recipe = u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half))
        value = self.lower(recipe)
      elif dtype is dtypes.half and source.layout in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16):
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      elif dtype is dtypes.float and source_dtype is dtypes.half and source.layout is RKLayout.FP16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      elif dtype is dtypes.int16 and source.layout in (RKLayout.INT16, RKLayout.BOOL_INT16):
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.int and source.layout is RKLayout.BOOL_INT16 and self.int_layout is RKLayout.INT16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.int and source.layout is RKLayout.BOOL_INT16 and self.int_layout is RKLayout.INT_FP16:
        recipe = u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half))
        converted = self.lower(recipe)
        if converted.layout is not RKLayout.FP16: raise _RKGenericReject
        value = RKValue(converted.arg, dtype, self.count, RKLayout.INT_FP16)
      elif dtype is dtypes.int and source.layout in (RKLayout.FP16, RKLayout.BOOL_MASK):
        if self.int_layout is RKLayout.INT_FP16:
          if source.layout is RKLayout.BOOL_MASK: value = RKValue(source.arg, dtype, self.count, self.int_layout)
          else:
            recipe = _int_fp16_expr(u)
            converted = self.lower(recipe)
            value = RKValue(converted.arg, dtype, self.count, self.int_layout)
        elif self.int_layout is RKLayout.INT16:
          value = self._scratch(dtype, self.int_layout)
          self.ew_ops.append(RKEWOp(value.arg, source.arg, source.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int16_output=True))
        else: raise _RKGenericReject
      elif dtype is dtypes.int: value = self._widen_int16(u, source)
      else: raise _RKGenericReject(f"cast {source.layout.name}->{dtype}")
    elif u.op is Ops.ADD and dtype is dtypes.half and u.arg is None and self.accurate_adds:
      try: value = self.lower(_accurate_add_recipe(u))
      except _RKGenericReject: value = self._alu(u)
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL): value = self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in u.src): value = self._compare(u)
    elif dtype is dtypes.bool and u.op in (Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ) and \
         all(src.dtype.scalar() is dtypes.bool for src in u.src): value = self._bool_binary(u)
    elif (dtype is dtypes.bool and u.op in (Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ) and
          (ieee_recipe:=_ieee_comparison_mask(u)) is not None): value = self._ieee_bool(ieee_recipe)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype is dtypes.bool: value = self._bool_binary(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype in (dtypes.int16, dtypes.int): value = self._integer_bitwise(u)
    elif u.op is Ops.CMOD and dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16:
      recipe = _int_fp16_expr(u)
      converted = self.lower(recipe)
      value = RKValue(converted.arg, dtype, self.count, self.int_layout)
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN): value = self._math(u)
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    self.values[u] = value
    return value

  def finish(self) -> RKImage:
    nodes = self.nodes
    if len(nodes) > 800 and not any(node.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.WHERE) for node in nodes):
      for node in nodes:
        if node.dtype.scalar() in (dtypes.half, dtypes.int16, dtypes.bool) and node.op in (Ops.CONST, Ops.LOAD, Ops.CAST, *GroupOp.ALU):
          self.lower(node)
    result = self.lower(self.root)
    dtype = self.out_param.dtype.scalar()
    if dtype is dtypes.half and result.layout in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16):
      if result.arg != self.out: self._emit(RKValue(self.out, dtype, self.count, RKLayout.FP16), result, result, _EW_CFG[Ops.MAX])
    elif dtype is dtypes.int16 and result.layout is RKLayout.INT16:
      if result.arg != self.out: self._emit(RKValue(self.out, dtype, self.count, RKLayout.INT16), result, result, _EW_CFG[Ops.MAX])
    elif dtype is dtypes.uchar and result.layout is RKLayout.INT16:
      self.mid_gathers.append(_int16_low_bytes(result.arg, self.out_param.arg.slot, self.count))
    elif dtype is dtypes.bool and result.layout is RKLayout.BOOL_MASK:
      tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
      self.ew_ops.append(RKEWOp(self.out, result.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX],
        stateful=True, int32_output=True, bool_output=True))
    elif dtype is dtypes.bool and result.layout is RKLayout.BOOL_INT16:
      self.mid_gathers.append(_int16_low_bytes(result.arg, self.out_param.arg.slot, self.count))
    elif dtype is dtypes.int and result.layout is RKLayout.INT32:
      if result.arg != self.out:
        self.mid_gathers.append(RKGather(result.arg.index, self.out_param.arg.slot, self.count,
          base=result.arg.addend//4, axes=((1, self.count, 1),), dst_kind=RKBufferKind.ARG,
          src_kind=result.arg.kind, itemsize=4))
    elif dtype is dtypes.int and result.layout is RKLayout.INT_FP16: self._widen_exact_int(result)
    elif dtype is dtypes.int and result.layout is RKLayout.INT16: self._widen_int16(self.root, result)
    elif dtype is dtypes.float and result.layout is RKLayout.FP16:
      groups = tuple(range(0, self.count, _EW_ELEMS_32BIT))
      aligned = self._scratch(dtypes.half, RKLayout.FP16, len(groups)*16)
      split = len(self.ew_ops)
      for group,start in enumerate(groups):
        lanes = min(_EW_ELEMS_32BIT, self.count-start)
        self.mid_gathers.append(RKGather(result.arg.index, aligned.arg.index, lanes,
          offsets=tuple(result.arg.addend//2+lane for lane in range(start, start+lanes)), dst_addend=group*8,
          src_kind=result.arg.kind, after=split))
        source = replace(aligned.arg, addend=group*16)
        self.ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG, self.out_param.arg.slot, start*4), source, source, lanes,
                                  _EW_CFG[Ops.MAX] | _EW_STAGE_FP32_OUT))
    else: raise _RKGenericReject
    constants = b""
    if self.constants:
      by_slot = {slot:bits for bits,slot in self.constants.items()}
      constants = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot)+1))
    image = RKImage(tuple(self.scratch), constants, gathers=tuple(self.gathers), ew_ops=tuple(self.ew_ops),
                    mid_gathers=tuple(replace(gather, after=len(self.ew_ops)) if gather.after < 0 else gather
                                      for gather in self.mid_gathers),
                    host_gathers=tuple(self.host_gathers))
    return _reuse_linear_scratch(image, self.constants)

def _structural_reduce(reduce_op:Ops, dtype:DType, terms:list[UOp]) -> UOp:
  if reduce_op is Ops.ADD and dtype.scalar() is dtypes.half:
    nonzero = [term for term in terms if not (term.op is Ops.CONST and float(term.arg) == 0.0)]
    if nonzero and all(term.op is Ops.MUL and term.dtype.scalar() is dtypes.half for term in nonzero):
      return _precise_mul_sum(nonzero)
  while len(terms) > 1:
    terms = [UOp(reduce_op, dtype, src=(terms[i], terms[i+1])) for i in range(0, len(terms)-1, 2)] + \
      (terms[-1:] if len(terms) & 1 else [])
  return terms[0]

def _expand_math_uops(root:UOp, nodes:dict[UOp, None], *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  bounded_recipes = len(nodes) <= _MAX_OPTIONAL_RECIPE_NODES
  if not bounded_recipes and not any(u.op in (Ops.WHERE, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.TRUNC) or
    u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and u.src[0].dtype.scalar() is dtypes.float for u in nodes): return root
  composite_math = _fold_inverse_hyperbolic(root) if bounded_recipes else None
  if composite_math is None and bounded_recipes: composite_math = _fold_atan(root)
  cache:dict[UOp, UOp] = {}
  exact_static_selection = root.op is Ops.WHERE and _is_static_expr(root.src[0]) and not any(
    node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half, dtypes.float) and not math.isfinite(float(node.arg))
    for node in nodes)
  def physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp:
    placeholders = {source:UOp.param(-index-1, source.dtype, ()) for index,source in enumerate(opaque)}
    rewritten = _fp16_rewrite(list(UOp(Ops.SINK, src=(recipe.substitute(placeholders),)).toposort()))
    if not rewritten or rewritten[-1].op is not Ops.SINK or len(rewritten[-1].src) != 1: raise _RKGenericReject
    tagged_root = _tag_precise_adds(rewritten[-1].src[0])
    return tagged_root.substitute({placeholder:source for source,placeholder in placeholders.items()})
  if composite_math is not None: root = physical_recipe(composite_math)
  def rewrite(u:UOp) -> UOp:
    if u in cache: return cache[u]
    if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.float:
      mapped = rewrite(physical_recipe(_dpu_sin(u.src[0].src[0]), (u.src[0].src[0],))) if u.src[0].op is Ops.SIN else \
        _canonical_half_storage(u.src[0])
      cache[u] = mapped
      return mapped
    if accurate_adds and bounded_recipes and u.op is Ops.ADD and u.dtype.scalar() is dtypes.half and u.arg is None:
      try:
        mapped = _accurate_add_recipe(u)
        cache[u] = mapped
        return mapped
      except _RKGenericReject: pass
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.WHERE and (absolute:=_fold_where_abs(mapped)) is not None: mapped = rewrite(absolute)
    if exact_static_selection and mapped.op is Ops.MUL and (minimum:=_fold_minimum(mapped)) is not None:
      mapped = minimum.replace(arg=_NATIVE_RAW_MIN)
    if mapped.op is Ops.SQRT:
      if (recipe:=_dpu_sqrt(mapped.src[0])) is None: raise _RKGenericReject
      mapped = rewrite(physical_recipe(recipe, (mapped.src[0],)))
    elif mapped.op is Ops.EXP2: mapped = rewrite(physical_recipe(_dpu_exp2(mapped.src[0]), (mapped.src[0],)))
    elif mapped.op is Ops.LOG2:
      if mapped.src[0].op is Ops.WHERE: raise _RKGenericReject
      mapped = rewrite(physical_recipe(_dpu_log2(mapped.src[0]), (mapped.src[0],)))
    elif mapped.op is Ops.SIN: mapped = rewrite(physical_recipe(_dpu_sin(mapped.src[0]), (mapped.src[0],)))
    elif mapped.op is Ops.TRUNC and mapped.dtype.scalar() is dtypes.half and not _is_static_expr(mapped):
      mapped = rewrite(_fold_trunc(mapped))
    cache[u] = mapped
    return mapped
  return rewrite(root)

_pm_fp32_sin_storage = PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat(Ops.SIN, dtypes.float),)),
   lambda root:_expand_math_uops(root, root.toposort())),
])

def _finite_max_neutrals(root:UOp, nodes:dict[UOp, None]) -> tuple[UOp, dict[UOp, None]]:
  """Canonicalize finite physical neutrals for FP selectors and exact INT32 MAX arithmetic."""
  int_min = any(u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg) == dtypes.int.min for u in nodes)
  if root.op is not Ops.MAX and not int_min: return root,nodes
  finite_selectors = root.op is Ops.MAX
  cache:dict[tuple[UOp, bool], UOp] = {}
  stack:list[tuple[UOp, bool, bool]] = [(root, False, False)]
  while stack:
    u,under_max,ready = stack.pop()
    key = (u, under_max)
    if key in cache: continue
    active = under_max or u.op is Ops.MAX and u.dtype.scalar() is dtypes.int
    if active and u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg) == dtypes.int.min:
      cache[key] = u.const_like(-2048)
    elif ready:
      src = tuple(cache[(source, active)] for source in u.src)
      if (finite_selectors and u.op is Ops.WHERE and src[1].op is Ops.CONST and
          src[1].dtype.scalar() in (dtypes.half, dtypes.float) and math.isinf(float(src[1].arg)) and float(src[1].arg) < 0.0):
        src = (src[0], src[1].const_like(-65504.0), src[2])
      cache[key] = u.replace(src=src)
    else:
      stack.append((u, under_max, True))
      stack.extend((src, active, False) for src in reversed(u.src))
  return cache[(root, False)], cache[(root, False)].toposort()

def _substitute_static_ranges(root:UOp, replacements:dict[UOp, UOp]) -> UOp:
  cache:dict[UOp, UOp] = {}
  for u in root.toposort(): cache[u] = replacements[u] if u in replacements else u.replace(src=tuple(cache[src] for src in u.src))
  return cache[root]

def _unroll_static_reduces(root:UOp) -> UOp:
  """Interpret static REDUCE/RANGE structure into ordinary semantic UOps."""
  cache:dict[UOp, UOp] = {}
  def rewrite(u:UOp) -> UOp:
    if u in cache: return cache[u]
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.REDUCE:
      reduce_op = mapped.arg[0] if isinstance(mapped.arg, tuple) else mapped.arg
      ranges = list(mapped.src[1:])
      if reduce_op not in (Ops.ADD, Ops.MAX, Ops.MUL) or not ranges or any(
        r.op is not Ops.RANGE or r.src[0].op is not Ops.CONST for r in ranges): raise _RKGenericReject
      iterations = math.prod(int(r.src[0].arg) for r in ranges)
      if iterations > _MAX_GENERIC_UNROLL or iterations*len(mapped.src[0].toposort()) > _MAX_GENERIC_EXPANDED_NODES:
        raise _RKGenericReject
      envs = _iter_selected_range_env(ranges)
      if not envs: raise _RKGenericReject
      terms = [_substitute_static_ranges(mapped.src[0], {r:r.const_like(env[r]) for r in ranges}) for env in envs]
      mapped = _structural_reduce(reduce_op, u.dtype, terms)
    cache[u] = mapped
    return mapped
  return rewrite(root)

def _local_buffer(u:UOp) -> UOp|None:
  u = _strip_cast(u)
  if u.op in (Ops.LOAD, Ops.STORE): u = u.src[0]
  if u.op is Ops.INDEX: u = u.src[0]
  while u.op is Ops.AFTER: u = u.src[0]
  return u if u.op is Ops.BUFFER else None

def _unroll_static_local(uops:list[UOp], output:RKOutput, root:UOp) -> UOp:
  """Execute one static local accumulator for ADD/MAX/MUL without recovering a tensor operation."""
  local_loads = [load for u in root.toposort() if (load:=_local_load(u)) is not None]
  buffers = {_local_buffer(load) for load in local_loads}
  if not local_loads: return root
  if None in buffers: raise _RKGenericReject
  typed_buffers = typing_cast(set[UOp], buffers)
  if len(typed_buffers) > 1:
    definitions = _static_local_defs(uops, typed_buffers)
    expanded:dict[UOp, UOp] = {}
    active:set[UOp] = set()
    def expand_dependencies(expr:UOp, owner:UOp) -> UOp:
      substitutions = {load:expand_buffer(buffer) for node in expr.toposort() if (load:=_local_load(node)) is not None and
                       (buffer:=_local_buffer(load)) is not None and buffer is not owner}
      return _substitute_static_ranges(expr, substitutions)
    def expand_buffer(buffer:UOp) -> UOp:
      if buffer in expanded: return expanded[buffer]
      if buffer in active: raise _RKGenericReject
      active.add(buffer)
      definition = definitions[buffer]
      if not definition.loops or any(loop.src[0].op is not Ops.CONST or not 0 <= int(loop.src[0].arg) <= _MAX_GENERIC_UNROLL
                                     for loop in definition.loops):
        raise _RKGenericReject
      terms = [expand_dependencies(definition.initial, buffer)]
      for env in _iter_selected_range_env(list(definition.loops)):
        term = _substitute_static_ranges(definition.term, {loop:loop.const_like(env[loop]) for loop in definition.loops})
        terms.append(expand_dependencies(term, buffer))
      expanded[buffer] = _structural_reduce(definition.update_op, buffer.dtype, terms)
      active.remove(buffer)
      if len(expanded[buffer].toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
      return expanded[buffer]
    substitutions = {load:expand_buffer(typing_cast(UOp, _local_buffer(load))) for load in local_loads}
    return _substitute_static_ranges(root, substitutions)
  buffer = next(iter(typed_buffers))
  stores = [u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]
  out_ranges = set(_index_ranges(output[3]))
  updates:list[tuple[UOp, UOp, list[UOp]]] = []
  initializers:list[UOp] = []
  for store in stores:
    value = _strip_cast(store.src[1])
    accumulator = next((src for src in value.src if _local_load(src) is not None and _local_buffer(src) is buffer), None) \
      if value.op in (Ops.ADD, Ops.MAX, Ops.MUL) else None
    if accumulator is None:
      if not any(r not in out_ranges for r in _index_ranges(value)): initializers.append(store.src[1])
      continue
    term = value.src[1 if value.src[0] is accumulator else 0]
    ranges = [r for r in _index_ranges(term) if r not in out_ranges]
    updates.append((value, term, ranges))
  if len(initializers) != 1 or len(updates) != 1: raise _RKGenericReject
  update, term, ranges = updates[0]
  if not ranges or any(r.src[0].op is not Ops.CONST for r in ranges): raise _RKGenericReject
  if any(node.op is Ops.WHERE and node.dtype.scalar() is dtypes.float for node in term.toposort()): raise _RKGenericReject
  iterations = math.prod(int(r.src[0].arg) for r in ranges)
  if iterations > _MAX_GENERIC_UNROLL or iterations*len(term.toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
  terms = [initializers[0]]
  for env in _iter_selected_range_env(ranges):
    terms.append(_substitute_static_ranges(term, {r:r.const_like(env[r]) for r in ranges}))
  reduced = _structural_reduce(update.op, update.dtype, terms)
  substitutions = {load:reduced for load in local_loads if _local_buffer(load) is buffer}
  return _substitute_static_ranges(root, substitutions)

@dataclass(frozen=True, slots=True)
class _RKStaticLocalDef:
  initial:UOp; update_op:Ops; term:UOp; loops:tuple[UOp, ...]

def _static_local_defs(uops:list[UOp], buffers:set[UOp]) -> dict[UOp, _RKStaticLocalDef]:
  """Parse scalar local accumulators without assigning tensor-operation meaning to their loops."""
  definitions:dict[UOp, _RKStaticLocalDef] = {}
  for buffer in buffers:
    if buffer.src[0].op is not Ops.CONST or int(buffer.src[0].arg) != 1: raise _RKGenericReject
    stores = [u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]
    initializers:list[UOp] = []
    updates:list[tuple[Ops, UOp, tuple[UOp, ...]]] = []
    for store in stores:
      value = _strip_cast(store.src[1])
      accumulator = [(i, _local_load(src)) for i,src in enumerate(value.src)] if value.op in (Ops.ADD, Ops.MAX, Ops.MUL) else []
      accumulator = [(i, load) for i,load in accumulator if load is not None and _local_buffer(load) is buffer]
      if len(accumulator) == 1:
        term = value.src[1-accumulator[0][0]]
        if any(_local_buffer(load) is buffer for node in term.toposort() if (load:=_local_load(node)) is not None):
          raise _RKGenericReject
        base = store.src[0].src[0] if store.src and store.src[0].op is Ops.INDEX else None
        loops = tuple(src for src in base.src[1:] if src.op is Ops.RANGE) if base is not None and base.op is Ops.AFTER else ()
        if not loops: raise _RKGenericReject
        updates.append((value.op, term, loops))
      elif not any(_local_buffer(load) is buffer for node in value.toposort() if (load:=_local_load(node)) is not None):
        initializers.append(store.src[1])
    if len(initializers) != 1 or len(updates) != 1: raise _RKGenericReject
    definitions[buffer] = _RKStaticLocalDef(initializers[0], *updates[0])
  return definitions

def _lower_multi_scalar_local_reductions(uops:list[UOp]) -> RKImage|None:
  """Materialize independent scalar local ADD programs, then execute their shared output UOps."""
  if (output:=_output_store(uops, dtypes.half, allow_local=True)) is None: return None
  store, out, count, _, root = output
  def semantic_local_loads(expr:UOp) -> list[UOp]:
    if expr.op in (Ops.RANGE, Ops.SPECIAL): return []
    if expr.op is Ops.LOAD and _local_buffer(expr) is not None: return [expr]
    return [load for src in expr.src for load in semantic_local_loads(src)]
  local_loads = list(dict.fromkeys(semantic_local_loads(root)))
  buffers = list(dict.fromkeys(buffer for load in local_loads if (buffer:=_local_buffer(load)) is not None))
  if not 1 < len(buffers) <= count: return None
  try: definitions = _static_local_defs(uops, set(buffers))
  except _RKGenericReject: return None
  if any(definition.update_op is not Ops.ADD or definition.initial.op is not Ops.CONST or
         float(definition.initial.arg) != 0.0 or not definition.loops or
         any(loop.src[0].op is not Ops.CONST or int(loop.src[0].arg) <= 0 for loop in definition.loops)
         for definition in definitions.values()): return None
  if any(semantic_local_loads(definition.term) for definition in definitions.values()):
    return None

  next_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  staged:RKImage|None = None
  sources:dict[UOp, UOp] = {}

  for buffer in buffers:
    definition, groups = definitions[buffer], math.prod(int(loop.src[0].arg) for loop in definitions[buffer].loops)
    if groups > _MAX_STATIC_RANGE_ENVS: return None
    flat = UOp.const(0, dtypes.int)
    stride = 1
    for loop in reversed(definition.loops):
      flat = flat.alu(Ops.ADD, loop.alu(Ops.MUL, loop.const_like(stride)))
      stride *= int(loop.src[0].arg)
    fake_slot, next_slot = next_slot, next_slot+1
    fake = UOp.param(fake_slot, dtypes.half, (groups,))
    map_store = fake.index(flat).store(definition.term)
    mapped = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(map_store,)).toposort())),
                                vectorize_reductions=False, recipes_ready=True)
    if mapped is None or (reduced:=_finish_mapped_add_reduction(mapped, fake_slot, 1, groups, 1.0)) is None:
      return None
    staged = reduced if staged is None else _append_inplace_image(staged, reduced)
    if staged is None: return None
    sources[buffer] = fake

  substitutions:dict[UOp, UOp] = {}
  for load in local_loads:
    buffer = _local_buffer(load)
    if buffer is None: return None
    fake = sources[buffer]
    replacement = fake.index(0).load()
    substitutions[load] = replacement.cast(load.dtype.scalar()) if load.dtype.scalar() is not dtypes.half else replacement
  post_root = root.substitute(substitutions)
  range_substitutions = {axis:axis.replace(src=(axis.src[0],)) for axis in _index_ranges(store.src[0].src[1]) if len(axis.src) > 1}
  if range_substitutions: post_root = post_root.substitute(range_substitutions)
  post_index = store.src[0].substitute(range_substitutions) if range_substitutions else store.src[0]
  post_store = store.replace(src=(post_index, post_root, *store.src[2:]))
  post = _lower_uop_program(_fp16_rewrite(list(UOp(Ops.SINK, src=(post_store,)).toposort())),
                            vectorize_reductions=False, recipes_ready=True)
  if post is None or staged is None: return None
  appended = _append_inplace_image(staged, post)
  if appended is None: return None
  scratch_base = len(appended.scratch)
  slot_to_scratch = {fake.arg.slot:scratch_base+i for i,fake in enumerate(sources.values())}
  def arg(value:RKArg) -> RKArg:
    return RKArg(RKBufferKind.SCRATCH, slot_to_scratch[value.index], value.addend) \
      if value.kind is RKBufferKind.ARG and value.index in slot_to_scratch else value
  def gather(value:RKGather) -> RKGather:
    src, dst = arg(RKArg(value.src_kind, value.src_index)), arg(RKArg(value.dst_kind, value.dst_index))
    return replace(value, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  return replace(appended, scratch=appended.scratch+tuple(RKScratch(64) for _ in sources),
    gathers=tuple(gather(x) for x in appended.gathers),
    ew_ops=tuple(replace(op, dst=arg(op.dst), lhs=arg(op.lhs), rhs=arg(op.rhs)) for op in appended.ew_ops),
    mid_gathers=tuple(gather(x) for x in appended.mid_gathers))

def _static_local_load_offsets(uops:list[UOp], output:RKOutput, root:UOp) -> dict[UOp, tuple[int, ...]]:
  """Execute bounded constant/control local programs used only to materialize global addresses."""
  local_loads = {load for node in root.toposort() if (load:=_local_load(node)) is not None}
  if not local_loads: return {}
  global_loads = [node for node in root.toposort() if node.op is Ops.LOAD and node.src and node.src[0].op is Ops.INDEX and
                  _root_param(node.src[0]) is not None and any(_local_load(x) is not None for expr in
                  (node.src[0].src[1], node.src[2] if len(node.src) > 2 else UOp.const(True, dtypes.bool)) for x in expr.toposort())]
  if not global_loads: return {}
  covered = {load for node in global_loads for expr in (node.src[0].src[1], node.src[2] if len(node.src) > 2 else UOp.const(True, dtypes.bool))
             for x in expr.toposort() if (load:=_local_load(x)) is not None}
  if covered != local_loads: return {}
  buffers = {_local_buffer(load) for load in local_loads}
  if None in buffers: raise _RKGenericReject
  definitions = _static_local_defs(uops, typing_cast(set[UOp], buffers))
  free_cache:dict[UOp, set[UOp]] = {}
  visiting:set[UOp] = set()
  def semantic_ranges(expr:UOp, owner:UOp) -> set[UOp]:
    if expr.op is Ops.RANGE: return {expr}
    if (load:=_local_load(expr)) is not None:
      buffer = _local_buffer(load)
      if buffer is owner: return set()
      if buffer is None: raise _RKGenericReject
      return free_ranges(buffer)
    return set().union(*(semantic_ranges(src, owner) for src in expr.src)) if expr.src else set()
  def free_ranges(buffer:UOp) -> set[UOp]:
    if buffer in free_cache: return free_cache[buffer]
    if buffer in visiting: raise _RKGenericReject
    visiting.add(buffer)
    definition = definitions[buffer]
    result = (semantic_ranges(definition.initial, buffer) | semantic_ranges(definition.term, buffer)) - set(definition.loops)
    visiting.remove(buffer); free_cache[buffer] = result
    return result
  for buffer in definitions: free_ranges(buffer)

  Value = int|float|bool|np.ndarray
  budget = [_MAX_STATIC_LOCAL_STEPS]
  local_cache:dict[tuple[UOp, tuple[tuple[UOp, int|float|bool], ...]], Value] = {}
  active:set[UOp] = set()
  def cast_value(value:Value, dtype:DType) -> Value:
    return _vector_cast(value, dtype) if isinstance(value, np.ndarray) else _eval_cast(value, dtype)
  def binary(op:Ops, dtype:DType, lhs:Value, rhs:Value) -> Value:
    if op is Ops.ADD: value = lhs + rhs
    elif op is Ops.MUL: value = lhs * rhs
    elif op is Ops.SUB: value = lhs - rhs
    elif op is Ops.MAX: value = np.maximum(lhs, rhs) if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray) else max(lhs, rhs)
    elif op is Ops.CMPLT: return lhs < rhs
    elif op is Ops.CMPNE: return lhs != rhs
    elif op is Ops.AND: value = np.bitwise_and(lhs, rhs) if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray) else int(lhs) & int(rhs)
    elif op is Ops.OR: value = np.bitwise_or(lhs, rhs) if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray) else int(lhs) | int(rhs)
    elif op is Ops.XOR: value = np.bitwise_xor(lhs, rhs) if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray) else int(lhs) ^ int(rhs)
    elif op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
      if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
        left, right = np.asarray(lhs), np.asarray(rhs)
        with np.errstate(divide="ignore", invalid="ignore"):
          quotient = np.where(right != 0, np.trunc(left/right) if op in (Ops.CDIV, Ops.CMOD) else np.floor_divide(left, right), 0)
        value = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else left-quotient*right
      else:
        value = cdiv(int(lhs), int(rhs)) if op is Ops.CDIV else cmod(int(lhs), int(rhs)) if op is Ops.CMOD else \
                floordiv(int(lhs), int(rhs)) if op is Ops.FLOORDIV else floormod(int(lhs), int(rhs))
    else: raise _RKGenericReject
    return cast_value(value, dtype)
  def eval_buffer(buffer:UOp, env:dict[UOp, Value]) -> Value:
    key_items:list[tuple[UOp, int|float|bool]] = []
    cacheable = True
    for r in sorted(free_cache[buffer], key=lambda x:x.key):
      if r not in env: raise _RKGenericReject
      if isinstance(env[r], np.ndarray): cacheable = False; break
      key_items.append((r, typing_cast(int|float|bool, env[r])))
    key = (buffer, tuple(key_items))
    if cacheable and key in local_cache: return local_cache[key]
    if buffer in active: raise _RKGenericReject
    definition = definitions[buffer]
    if not definition.loops or any(loop.src[0].op is not Ops.CONST or not 0 <= int(loop.src[0].arg) <= _MAX_GENERIC_UNROLL
                                   for loop in definition.loops):
      raise _RKGenericReject
    active.add(buffer)
    accumulator = evaluate(definition.initial, env)
    for loop_env in _iter_selected_range_env(list(definition.loops)):
      budget[0] -= 1
      if budget[0] < 0: raise _RKGenericReject
      term = evaluate(definition.term, {**env, **loop_env})
      accumulator = binary(definition.update_op, buffer.dtype, accumulator, term)
    active.remove(buffer)
    if cacheable: local_cache[key] = accumulator
    return accumulator
  def evaluate(expr:UOp, env:dict[UOp, Value]) -> Value:
    if expr.op is Ops.CONST: return cast_value(expr.arg, expr.dtype)
    if expr.op in (Ops.RANGE, Ops.SPECIAL):
      if expr not in env: raise _RKGenericReject
      return env[expr]
    if expr.op is Ops.AFTER: return evaluate(expr.src[0], env)
    if expr.op is Ops.CAST: return cast_value(evaluate(expr.src[0], env), expr.dtype)
    if expr.op is Ops.NEG: return cast_value(-evaluate(expr.src[0], env), expr.dtype)
    if expr.op is Ops.WHERE:
      condition, yes, no = (evaluate(src, env) for src in expr.src)
      return cast_value(np.where(condition, yes, no) if isinstance(condition, np.ndarray) else yes if condition else no, expr.dtype)
    if expr.op is Ops.LOAD:
      if (buffer:=_local_buffer(expr)) is None: raise _RKGenericReject
      index = evaluate(expr.src[0].src[1], env)
      if np.any(np.asarray(index) != 0): raise _RKGenericReject
      return eval_buffer(buffer, env)
    if len(expr.src) != 2: raise _RKGenericReject
    return binary(expr.op, expr.dtype, evaluate(expr.src[0], env), evaluate(expr.src[1], env))

  ranges = _index_ranges(output[3])
  envs = _iter_range_env(ranges)
  if len(envs) != output[2]: raise _RKGenericReject
  vector_env:dict[UOp, Value] = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}
  destinations = np.broadcast_to(np.asarray(evaluate(output[3], vector_env), dtype=np.int64), len(envs))
  if np.any((destinations < 0) | (destinations >= output[2])) or not np.array_equal(np.sort(destinations), np.arange(output[2])):
    raise _RKGenericReject
  order = np.argsort(destinations)
  offsets:dict[UOp, tuple[int, ...]] = {}
  for load in global_loads:
    index = np.broadcast_to(np.asarray(evaluate(load.src[0].src[1], vector_env), dtype=np.int64), len(envs))
    if len(load.src) > 2:
      gate = np.broadcast_to(np.asarray(evaluate(load.src[2], vector_env), dtype=np.bool_), len(envs))
      index = np.where(gate, index, -1)
    offsets[load] = tuple(int(value) for value in index[order])
  return offsets

def _lower_vectorized_scalar_local_extrema(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Vectorize two dependent scalar local MAX accumulators without assigning meaning to their tensor source."""
  _, out_param, count, _, root = output
  if count != 1 or out_param.dtype.scalar() is not dtypes.int: return None
  buffers = {buffer for u in uops if u.op is Ops.BUFFER and (buffer:=_local_buffer(u)) is not None}
  try: definitions = _static_local_defs(uops, buffers)
  except _RKGenericReject: return None
  value_buffers = [buffer for buffer,definition in definitions.items() if buffer.dtype.scalar() is dtypes.half and
                   definition.update_op is Ops.MAX]
  index_buffers = [buffer for buffer,definition in definitions.items() if buffer.dtype.scalar() is dtypes.int and
                   definition.update_op is Ops.MAX]
  if len(definitions) != 2 or len(value_buffers) != 1 or len(index_buffers) != 1: return None
  value_buffer, index_buffer = value_buffers[0], index_buffers[0]
  value_def, index_def = definitions[value_buffer], definitions[index_buffer]
  if (value_def.initial.op is not Ops.CONST or not math.isinf(float(value_def.initial.arg)) or float(value_def.initial.arg) >= 0 or
      index_def.initial.op is not Ops.CONST or int(index_def.initial.arg) != dtypes.int.min or
      len(value_def.loops) != len(index_def.loops) or not value_def.loops): return None
  extents = tuple(int(loop.src[0].arg) for loop in value_def.loops if loop.src[0].op is Ops.CONST)
  if len(extents) != len(value_def.loops) or any(not 1 <= extent <= min(32767, _MAX_EW_ELEMS_FP16) for extent in extents): return None
  total = math.prod(extents)
  if not 2 <= total <= min(32767, _MAX_EW_ELEMS_FP16): return None
  if tuple(int(loop.src[0].arg) for loop in index_def.loops) != extents: return None

  weighted_terms = _flatten_binary(index_def.term, Ops.MUL)
  casts = [term for term in weighted_terms if term.op is Ops.CAST and term.dtype.scalar() is dtypes.int and
           term.src[0].dtype.scalar() is dtypes.bool]
  if len(casts) != 1 or len(weighted_terms) != 2: return None
  coordinate = weighted_terms[1 if weighted_terms[0] is casts[0] else 0]
  predicate = casts[0].src[0]
  inverted = False
  if predicate.op is Ops.CMPNE:
    for inner,marker in (predicate.src, predicate.src[::-1]):
      if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg):
        predicate, inverted = inner, True
        break
  if not inverted or predicate.op is not Ops.CMPNE or len(predicate.src) != 2 or \
     any(src.dtype.scalar() is not dtypes.half for src in predicate.src): return None
  current = next((src for src in predicate.src if (load:=_local_load(src)) is not None and
                  _local_buffer(load) is value_buffer), None)
  candidate = next((src for src in predicate.src if src is not current), None)
  if current is None or candidate is None: return None
  loop_map = dict(zip(index_def.loops, value_def.loops))
  mapped_candidate = _substitute_static_ranges(candidate, loop_map)
  if _strip_cast(mapped_candidate).key != _strip_cast(value_def.term).key: return None
  try:
    index_envs = _iter_selected_range_env(list(index_def.loops))
    coordinates = tuple(_eval_int(coordinate, env) for env in index_envs)
  except RuntimeError: return None
  if len(coordinates) != total or any(not 0 <= value <= 32767 for value in coordinates): return None

  def semantic_local_loads(expr:UOp) -> list[UOp]:
    if (load:=_local_load(expr)) is not None: return [load]
    return [load for src in expr.src for load in semantic_local_loads(src)]
  final_loads = list(dict.fromkeys(load for load in semantic_local_loads(root) if _local_buffer(load) is index_buffer))
  if len(final_loads) != 1: return None
  final_load = final_loads[0]
  try: mapped_outputs = tuple(_eval_int(root.substitute({final_load:final_load.const_like(value)}), {}) for value in coordinates)
  except RuntimeError: return None
  if len(coordinates) < 2: return None
  second = next((i for i in range(1, len(coordinates)) if coordinates[i] != coordinates[0]), None)
  if second is None: return None
  coordinate_delta, output_delta = coordinates[second]-coordinates[0], mapped_outputs[second]-mapped_outputs[0]
  if output_delta % coordinate_delta: return None
  slope = output_delta//coordinate_delta
  baseline = mapped_outputs[0]-slope*coordinates[0]
  if any(result != baseline+slope*value for value,result in zip(coordinates, mapped_outputs)) or \
     not all(-32768 <= value <= 32767 for value in (*mapped_outputs, slope, baseline)): return None

  global_slots = [u.arg.slot for u in uops if u.op is Ops.PARAM and u.arg is not None]
  fake_slot = max(global_slots, default=out_param.arg.slot)+1
  fake_out = UOp.param(fake_slot, dtypes.half, (total,))
  linear = UOp.const(0, dtypes.int)
  for axis,loop in enumerate(value_def.loops):
    stride = math.prod(extents[axis+1:])
    linear = linear.alu(Ops.ADD, loop.alu(Ops.MUL, UOp.const(stride, dtypes.int)))
  fake_store = fake_out.index(linear).store(value_def.term).end(*value_def.loops)
  child = _lower_uop_program(list(fake_store.sink().toposort()), vectorize_reductions=False)
  if child is None or child.host_gathers or child.host_scatters: return None

  scratch = list(child.scratch)
  def allocate(lanes:int=total) -> RKArg:
    scratch.append(RKScratch(_scratch_bytes(lanes)))
    return RKArg(RKBufferKind.SCRATCH, len(scratch)-1)
  values = allocate()
  def map_arg(arg:RKArg) -> RKArg:
    return RKArg(RKBufferKind.SCRATCH, values.index, arg.addend) if arg.kind is RKBufferKind.ARG and arg.index == fake_slot else arg
  def map_gather(gather:RKGather, *, after:int|None=None) -> RKGather:
    src, dst = map_arg(RKArg(gather.src_kind, gather.src_index)), map_arg(RKArg(gather.dst_kind, gather.dst_index))
    return replace(gather, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index,
                   after=gather.after if after is None else after)
  ops = [replace(op, dst=map_arg(op.dst), lhs=map_arg(op.lhs), rhs=map_arg(op.rhs)) for op in child.ew_ops]
  gathers = [map_gather(gather) for gather in child.gathers]
  mid = [map_gather(gather) for gather in child.mid_gathers]

  scalar_stride = _reduction_stride(1)
  reduction_values = allocate(total*scalar_stride//2)
  mid.append(RKGather(values.index, reduction_values.index, total, axes=((1, total, 1),), dst_stride=scalar_stride//2,
                      src_kind=RKBufferKind.SCRATCH, after=len(ops)))
  best = _reduce_rows(ops, [RKArg(reduction_values.kind, reduction_values.index, lane*scalar_stride) for lane in range(total)],
                      1, _EW_CFG[Ops.MAX])
  equality_after = len(ops)
  best_values = allocate()
  mid.append(RKGather(best.index, best_values.index, total, offsets=(best.addend//2,)*total,
                      src_kind=RKBufferKind.SCRATCH, after=equality_after))
  raw = tuple((allocate(), allocate()) for _ in range(2))
  for source,parts in ((values, raw[0]), (best_values, raw[1])):
    for byte,part in enumerate(parts):
      mid.append(RKGather(source.index, part.index, total, base=source.addend+byte, axes=((1, total, 2),), dst_stride=2,
                          src_kind=RKBufferKind.SCRATCH, itemsize=1, after=equality_after))
  constants = {number:allocate() for number in (0, 1, 123, 124, 127, 128)}
  for number,dst in constants.items(): gathers.append(RKGather(out_param.arg.slot, dst.index, total, values=(number,)*total))
  def alloc() -> RKArg: return allocate()
  lhs_high,lhs_nan = _fp16_high_and_nan(ops, alloc, raw[0][1], raw[0][0], constants[0], constants[1],
    constants[123], constants[124], constants[127], constants[128], total)
  rhs_high,rhs_nan = _fp16_high_and_nan(ops, alloc, raw[1][1], raw[1][0], constants[0], constants[1],
    constants[123], constants[124], constants[127], constants[128], total)
  low_equal = _ew_native_int16_eq_mask(ops, alloc, raw[0][0], raw[1][0], constants[1], total)
  high_equal = _ew_native_int16_eq_mask(ops, alloc, lhs_high, rhs_high, constants[1], total)
  either_nan, numeric, bits_equal, equal = (alloc() for _ in range(4))
  ops.extend((RKEWOp(either_nan, lhs_nan, rhs_nan, total, _EW_CFG[Ops.MAX], **_INT16_EW),
              RKEWOp(numeric, constants[1], either_nan, total, _EW_CFG[Ops.SUB], **_INT16_EW),
              RKEWOp(bits_equal, low_equal, high_equal, total, _EW_CFG[Ops.MUL], **_INT16_EW),
              RKEWOp(equal, bits_equal, numeric, total, _EW_CFG[Ops.MUL], **_INT16_EW)))
  coordinate_values, weighted = allocate(), allocate()
  gathers.append(RKGather(out_param.arg.slot, coordinate_values.index, total, values=coordinates))
  ops.append(RKEWOp(weighted, equal, coordinate_values, total, _EW_CFG[Ops.MUL], **_INT16_EW))
  weighted_spaced = allocate(total*scalar_stride//2)
  mid.append(RKGather(weighted.index, weighted_spaced.index, total, axes=((1, total, 1),), dst_stride=scalar_stride//2,
                      src_kind=RKBufferKind.SCRATCH, after=len(ops)))
  selected = _reduce_rows(ops, [RKArg(weighted_spaced.kind, weighted_spaced.index, lane*scalar_stride) for lane in range(total)],
                          1, _EW_CFG[Ops.MAX], int16=True)
  result = selected
  if slope != 1:
    scale = allocate(1); gathers.append(RKGather(out_param.arg.slot, scale.index, 1, values=(_int16_bits(slope),)))
    scaled = allocate(1); ops.append(RKEWOp(scaled, result, scale, 1, _EW_CFG[Ops.MUL], **_INT16_EW)); result = scaled
  if baseline:
    offset = allocate(1); gathers.append(RKGather(out_param.arg.slot, offset.index, 1, values=(_int16_bits(baseline),)))
    translated = allocate(1); ops.append(RKEWOp(translated, result, offset, 1, _EW_CFG[Ops.ADD], **_INT16_EW)); result = translated
  zero = allocate(1); gathers.append(RKGather(out_param.arg.slot, zero.index, 1, values=(0,)))
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), result, zero, 1, _EW_CFG[Ops.ADD],
                    int16_input=True, int32_output=True))
  image = RKImage(tuple(scratch), child.constants, gathers=tuple(gathers), ew_ops=tuple(ops),
                  mid_gathers=tuple(mid))
  return image if all(len(items) <= _RKIMAGE_U16_MAX for items in
                      (image.scratch, image.gathers, image.ew_ops, image.mid_gathers)) else None

def _lower_host_scatter(uops:list[UOp]) -> RKImage|None:
  """Lower a direct dynamic STORE as raw last-writer host address materialization."""
  if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1" or \
     (output:=_output_store(uops, (dtypes.half, dtypes.int16))) is None or len(output[0].src) != 2: return None
  _, out_param, out_count, dynamic_index, value = output
  if (index_info:=_runtime_index(dynamic_index)) is None: return None
  _, index_param, lane_index, index_itemsize = index_info
  value = _strip_cast(value)
  if (value.op is not Ops.LOAD or len(value.src) != 1 or value.src[0].op is not Ops.INDEX or
      (source:=_root_param(value.src[0])) is None or source.src[0].op is not Ops.CONST or
      value.src[0].src[1].key != lane_index.key or source.dtype.scalar() is not out_param.dtype.scalar()): return None
  count = int(index_param.src[0].arg)
  if int(source.src[0].arg) < count: return None
  address = RKHostAddress(RKArg(RKBufferKind.ARG, source.arg.slot), RKArg(RKBufferKind.ARG, index_param.arg.slot),
    RKArg(RKBufferKind.ARG, out_param.arg.slot), count, int(source.src[0].arg), out_count,
    itemsize=out_param.dtype.scalar().itemsize, index_itemsize=index_itemsize)
  return RKImage(host_scatters=(address,))

def _lower_uop_program(uops:list[UOp], *, vectorize_reductions:bool=True, recipes_ready:bool=False) -> RKImage|None:
  """Lower a composable typed UOp program; return None for the legacy correctness oracle."""
  output_stores = [u for u in uops if u.op is Ops.STORE and _root_param(u.src[0]) is not None]
  if len(output_stores) > 1:
    combined:RKImage|None = None
    for store in output_stores:
      child = _lower_uop_program(list(UOp(Ops.SINK, src=(store,)).toposort()),
                                 vectorize_reductions=vectorize_reductions, recipes_ready=recipes_ready)
      if child is None: return None
      combined = child if combined is None else _append_inplace_image(combined, child)
      if combined is None: return None
    return combined
  if vectorize_reductions and (multi_local:=_lower_multi_scalar_local_reductions(uops)) is not None: return multi_local
  if (scatter:=_lower_host_scatter(uops)) is not None: return scatter
  if (int_output:=_output_store(uops, dtypes.int)) is not None:
    if (raw_bitcast:=_lower_raw_fp16_bitcast(int_output)) is not None: return raw_bitcast
    if (division:=_lower_int32_division(int_output)) is not None: return division
  if (bool_loop_output:=_output_store(uops, dtypes.bool, allow_local=True)) is not None and \
     (grouped_bool_reduction:=_lower_grouped_bool_reduction(bool_loop_output)) is not None: return grouped_bool_reduction
  for dtype in (dtypes.half, dtypes.int16, dtypes.int):
    if (direct_load:=_output_store(uops, dtype)) is None: continue
    if (image:=_lower_direct_dynamic_typed_load(direct_load, dtype)) is not None: return image
    if (image:=_lower_dynamic_multi_index_typed_load(direct_load, dtype)) is not None: return image
    root = direct_load[4]
    if root.op is Ops.WHERE and root.src[1].op is Ops.LOAD and root.src[2].op is Ops.CONST and \
       (folded_load:=_fold_masked_load(root.src[0], root.src[1], root.src[2], allow_additional_gate_loads=True)) is not None and \
       (image:=_lower_direct_dynamic_typed_load((*direct_load[:4], folded_load), dtype)) is not None: return image
  if vectorize_reductions and (local_output:=_output_store(uops, dtypes.int, allow_local=True)) is not None and \
     (local_extrema:=_lower_vectorized_scalar_local_extrema(uops, local_output)) is not None: return local_extrema
  storage_uops:list[UOp]|None = None
  storage_product_adds = False
  if any(u.dtype.scalar() is dtypes.float for u in uops):
    sink = next((u for u in uops if u.op is Ops.SINK), None)
    if sink is not None:
      storage_sink = sink
      if (storage_output:=_output_store(uops, dtypes.half, allow_local=True)) is not None:
        storage_root = storage_output[4]
        root_storage = storage_root.op is Ops.CAST and len(storage_root.src) == 1 and storage_root.dtype.scalar() is dtypes.half and \
          storage_root.src[0].dtype.scalar() is dtypes.float
        storage_product_adds = any(boundary.op is Ops.CAST and boundary.dtype.scalar() is dtypes.half and len(boundary.src) == 1 and
          boundary.src[0].dtype.scalar() is dtypes.float and _fp32_add_has_product_terms(boundary.src[0]) and
          (boundary is not storage_root or len(boundary.src[0].toposort()) > 64) for boundary in storage_root.toposort())
        # A later half FDIV/WHERE/etc. can own several independent FP32 reduction boundaries. Commit each CAST
        # before the bottom-up generic rewrite erases the semantic FP32 ADD tree, including pure denominator sums.
        if (ratio:=_fp32_ratio_to_half(storage_root)) is not None: storage_sink = storage_sink.substitute({storage_root:ratio})
        else:
          nested_storage:dict[UOp, UOp] = {}
          for boundary in storage_root.toposort():
            if boundary is storage_root or boundary.op is not Ops.CAST or boundary.dtype.scalar() is not dtypes.half or \
               len(boundary.src) != 1 or boundary.src[0].dtype.scalar() is not dtypes.float or boundary.src[0].op is not Ops.ADD: continue
            try: nested_storage[boundary] = _canonical_half_storage(boundary.src[0])
            except _RKGenericReject: pass
          if nested_storage: storage_sink = storage_sink.substitute(nested_storage)
        if root_storage:
          try:
            source = storage_root.src[0]
            if not _has_runtime_address(source):
              converted = _expand_math_uops(storage_root, storage_root.toposort()) if source.op is Ops.SIN else _canonical_half_storage(source)
              storage_sink = sink.substitute({storage_root:converted})
          except _RKGenericReject: pass
      storage_sink = graph_rewrite(storage_sink, _pm_fp32_sin_storage, name="rockchip fp32 sin storage")
      storage_uops = list(graph_rewrite(storage_sink, _pm_generic_storage_precision,
                                        name="rockchip generic storage precision").toposort())
  if vectorize_reductions and (mul_add:=_lower_vectorized_mul_add_reduction(uops)) is not None: return mul_add
  mapped_loop = _lower_mapped_add_loop_reduction(uops) if vectorize_reductions else None
  if mapped_loop is not None: return mapped_loop
  if storage_uops is not None: uops = storage_uops
  if (output:=_output_store(uops, (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool, dtypes.uchar), allow_local=True)) is None or \
     len(output[0].src) != 2:
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage()
  if output[1].dtype.scalar() is dtypes.bool:
    if (bounds_mask:=_lower_int32_bounds_mask(output)) is not None: return bounds_mask
  if output[1].dtype.scalar() is dtypes.int:
    if (bitwise:=_lower_int32_byte_logic(output)) is not None: return bitwise
    if (shift:=_lower_int32_shift(output)) is not None: return shift
    for source_dtype in (dtypes.int16, dtypes.int):
      if (coordinates:=_lower_bounded_integer_predicate_coordinates(output, source_dtype)) is not None: return coordinates
    if (lookup:=_lower_bounded_int32_lookup(output)) is not None: return lookup
  try:
    if (not _contiguous_output(output[3], output[2]) and
        _static_int_vector(output[3], output[3], output[2]) != tuple(range(output[2]))): return None
    reduced = _unroll_static_reduces(output[4]) if any(u.op is Ops.REDUCE for u in uops) else output[4]
    static_load_offsets = _static_local_load_offsets(uops, output, reduced)
    local_root = reduced if static_load_offsets else _unroll_static_local(uops, output, reduced)
    root = graph_rewrite(local_root, _pm_masked_loads, name="rockchip masked load materialization")
    root_nodes = root.toposort()
    root,root_nodes = _finite_max_neutrals(root, root_nodes)
    defer_nodes = storage_uops if storage_uops is not None else root_nodes
    defer_math = len(defer_nodes) > 256
    if not recipes_ready and not defer_math and \
       (expanded:=_expand_math_uops(root, root_nodes, accurate_adds=storage_uops is None or storage_product_adds)) is not root:
      root,root_nodes = expanded,expanded.toposort()
    expanded_nodes = root_nodes
    if len(expanded_nodes) > _MAX_GENERIC_EXPANDED_NODES:
      if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(expanded_nodes)}")
      return None
    if root is not output[4]:
      output = (*output[:4], root)
    image = RKContext(output, expanded_nodes, accurate_adds=(not recipes_ready and (storage_uops is None or storage_product_adds) and
                                             len(expanded_nodes) <= _MAX_OPTIONAL_RECIPE_NODES and
                                             not _has_runtime_address(output[4])),
                      static_load_offsets=static_load_offsets).finish()
    image_u16_counts = (len(image.scratch), len(image.gathers)+len(image.mid_gathers),
                        len(image.host_gathers), len(image.host_scatters))
    if any(count > _RKIMAGE_U16_MAX for count in image_u16_counts):
      if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1":
        raise _RKGenericReject("image u16 counts " + repr(image_u16_counts) + f", ew_ops={len(image.ew_ops)}")
      return None
    return image
  except (_RKGenericReject, RuntimeError, ValueError, KeyError):
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise
    return None


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

def _finite_positive_mask(u:UOp) -> UOp:
  """Map finite binary16 values to `u > 0` without the stateful DPU compare path."""
  magnitude = u.alu(Ops.MAX, UOp.const(0.0, dtypes.half))
  for _ in range(2): magnitude = magnitude.alu(Ops.MUL, UOp.const(256.0, dtypes.half))
  return _native_min(magnitude, UOp.const(1.0, dtypes.half))

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

def _fold_threshold_where(x:UOp) -> UOp|None:
  """Select a compared value or finite constant without multiplying an inactive nonfinite value."""
  gate = _unwrap_condition(x.src[0])
  if gate.op is not Ops.CMPLT or gate.src[1].op is not Ops.CONST or not math.isfinite(float(gate.src[1].arg)): return None
  lhs, threshold = gate.src[0], float(gate.src[1].arg)
  yes, no = (_unwrap_condition(u) for u in x.src[1:])
  if (mask:=_mask_expr(x.src[0])) is None: return None
  inverse = UOp.const(1.0, dtypes.half).alu(Ops.SUB, mask)
  if yes.key == lhs.key and no.op is Ops.CONST and math.isfinite(float(no.arg)) and float(no.arg) != threshold:
    return _native_min(lhs.cast(dtypes.half), UOp.const(threshold, dtypes.half)).alu(
      Ops.ADD, _mask_mul(inverse, UOp.const(float(no.arg)-threshold, dtypes.half)))
  if no.key == lhs.key and yes.op is Ops.CONST and math.isfinite(float(yes.arg)) and float(yes.arg) != threshold:
    return lhs.cast(dtypes.half).alu(Ops.MAX, UOp.const(threshold, dtypes.half)).alu(
      Ops.ADD, _mask_mul(mask, UOp.const(float(yes.arg)-threshold, dtypes.half)))
  return None

def _fold_general_where(x:UOp) -> UOp|None:
  """Select FP16 arms with DPU masks, avoiding multiplication by nonfinite constants."""
  if (threshold:=_fold_threshold_where(x)) is not None: return threshold
  mask = _mask_expr(x.src[0])
  if mask is None: return None
  yes, no = (arm.cast(dtypes.half) for arm in x.src[1:])
  one = UOp.const(1.0, dtypes.half)
  inverse = one.alu(Ops.SUB, mask)
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
  def relu(u:UOp) -> UOp|None:
    if (source:=_relu_operand(u)) is not None: return source
    return _relu_operand(folded) if u.op is Ops.WHERE and (folded:=_fold_ordered_where(u)) is not None else None
  def shifted(u:UOp) -> tuple[UOp, float]:
    return (term[0], float(term[1].arg)) if (term:=_const_operand(u, Ops.ADD)) is not None else (u, 0.0)
  for positive, negative in (x.src, x.src[::-1]):
    source, scaled = relu(positive), _const_operand(negative, Ops.MUL, -1.0)
    if source is None or scaled is None or (upper:=relu(scaled[0])) is None: continue
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

def _fold_where_abs(x:UOp) -> UOp|None:
  """Recognize `WHERE(x < 0, -x, x)` before an unselected infinity can contaminate a mask blend."""
  if x.op is not Ops.WHERE or len(x.src) != 3 or x.dtype.scalar() is not dtypes.half: return None
  condition = _strip_cast(x.src[0])
  negative = _strip_cast(x.src[1])
  source = condition.src[0] if condition.op is Ops.CMPLT else None
  negated = source is not None and negative.op is Ops.NEG and len(negative.src) == 1 and negative.src[0].key == source.key
  if source is not None and (scaled:=_const_operand(negative, Ops.MUL, -1.0)) is not None:
    negated |= scaled[0].key == source.key
  if (source is not None and source.op is Ops.FDIV and negative.op is Ops.FDIV and
      source.src[1].key == negative.src[1].key and source.src[0].op is Ops.CONST and negative.src[0].op is Ops.CONST):
    negated |= float(source.src[0].arg) == -float(negative.src[0].arg)
  if (condition.op is not Ops.CMPLT or condition.src[1].op is not Ops.CONST or float(condition.src[1].arg) != 0.0 or
      x.src[2].key != condition.src[0].key or not negated): return None
  return UOp(Ops.MAX, x.dtype, src=(condition.src[0], condition.src[0]), arg=_NATIVE_ABS)

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

def _fold_masked_load(gate:UOp, load:UOp, default:UOp, allow_additional_gate_loads:bool=False) -> UOp|None:
  if len(load.src) <= 2 or load.src[1].op is not Ops.CONST: return None
  load_gate = load.src[2]
  if not allow_additional_gate_loads and \
     {u.key for u in gate.toposort() if u.op is Ops.LOAD} - {u.key for u in load_gate.toposort() if u.op is Ops.LOAD}: return None
  same_default = float(load.src[1].arg) == float(default.arg)
  # If the outer condition implies the LOAD condition, the inner default is unreachable. This is the padded-pool form.
  outer_implies_inner = _same_condition(gate, load_gate) or (gate.op is Ops.AND and any(_same_condition(x, load_gate) for x in gate.src))
  if not same_default and not outer_implies_inner: return None
  return load.replace(src=(load.src[0], default, gate.alu(Ops.AND, load_gate) if same_default else gate))

_pm_masked_materialization = PatternMatcher([
  (UPat(Ops.WHERE, src=(UPat.var("gate"), UPat(Ops.LOAD, name="load"), UPat.cvar("default"))),
   lambda gate,load,default: _fold_masked_load(gate, load, default)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.var("val"), UPat.cvar("default"))),
   lambda gate,val,default: _fold_masked_max(gate, default, val, False)),
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat.cvar("default"), UPat.var("val"))),
   lambda gate,default,val: _fold_masked_max(gate, default, val, True)),
])
_pm_masked_loads = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_masked_mul)]) + _pm_masked_materialization

def _fold_masked_max(gate:UOp, default:UOp, val:UOp, opposite:bool) -> UOp|None:
  if val.op is Ops.MAX:
    lhs = _fold_masked_max(gate, default, val.src[0], opposite)
    rhs = _fold_masked_max(gate, default, val.src[1], opposite)
    return None if lhs is None or rhs is None else val.replace(src=(lhs, rhs))
  if val.op is Ops.MUL:
    for source,factor in (val.src, val.src[::-1]):
      if factor.op is not Ops.CONST or not math.isfinite(scale:=float(factor.arg)) or scale == 0.0 or math.isnan(float(default.arg)):
        continue
      folded = _fold_masked_max(gate, default.const_like(float(default.arg)/scale), source, opposite)
      if folded is not None: return val.replace(src=(folded, factor))
  if val.op is not Ops.LOAD or len(val.src) <= 2 or val.src[1].op is not Ops.CONST: return None
  def matches(condition:UOp) -> bool: return _opposite_condition(gate, condition) if opposite else _same_condition(gate, condition)
  condition_matches = matches(val.src[2]) or (val.src[2].op is Ops.AND and any(matches(x) for x in val.src[2].src))
  if condition_matches:
    return val.replace(src=(val.src[0], default, val.src[2]))
  return None

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

_pm_storage_common = PatternMatcher([
  (UPat((Ops.ADD, Ops.MUL), dtypes.float, name="x"),
   lambda x:None if _is_static_expr(x) else x.src[0].cast(dtypes.half).alu(x.op, x.src[1].cast(dtypes.half))),
  (UPat(Ops.CAST, dtypes.half, name="root"), _fold_casted_relu),
  (UPat(Ops.ADD, dtypes.half, name="x"), _fold_relu_cap),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_minimum),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
  (UPat(Ops.MUL, dtypes.half, name="x"), _replace_infinite_multiply),
  (UPat(Ops.FDIV, dtypes.half, name="x"), _preserve_infinite_division_sign),
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
])
_pm_fp32_to_fp16 = _pm_storage_common + PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.bool, name="predicate"),)),
   lambda predicate:nonzero if (nonzero:=_fp16_nonzero_mask(predicate)) is not None else _ieee_comparison_mask(predicate)),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_masked_mul),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_ordered_where),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_scaled_negative),
]) + _pm_masked_materialization + PatternMatcher([
  # Fold padding into the gather mask. This changes only host layout initialization; selected values still feed DPU EW.
  (UPat(Ops.WHERE, dtypes.half, name="x"), lambda x: x.src[1].alu(Ops.MAX, x.src[2])
   if x.src[0].op is Ops.CMPLT and x.src[0].src[0] is x.src[2] and x.src[0].src[1] is x.src[1] and
      x.src[2].op is Ops.CONST and float(x.src[2].arg) == 0.0 else None),
  (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_general_where),
])
_pm_generic_storage_precision = PatternMatcher([(UPat(Ops.WHERE, dtypes.float, name="x"),
  lambda x:None if _is_static_expr(x) else
  UOp(Ops.WHERE, dtypes.half, src=(x.src[0], x.src[1].cast(dtypes.half), x.src[2].cast(dtypes.half)), arg=x.arg))]) + _pm_storage_common
_pm_abs = PatternMatcher([(UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
                          (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_where_abs)])
_pm_round = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_round)])
_pm_floor_ceil = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_floor_ceil)])
_pm_trunc = PatternMatcher([(UPat(Ops.TRUNC, dtypes.half, name="x"), _fold_trunc)])
_pm_sign = PatternMatcher([(UPat(Ops.WHERE, dtypes.half, name="x"), _fold_sign)])

def _unit_ratio_source(root:UOp) -> UOp|None:
  """Match Tinygrad's x/sqrt(1+x*x) normalization."""
  candidates = ((root.src[0], root.src[1]),) if root.op is Ops.FDIV else tuple((source, inverse.src[0])
    for source,inverse in (root.src, root.src[::-1]) if inverse.op is Ops.RECIPROCAL and len(inverse.src) == 1)
  for source, denominator in candidates:
    if (denominator.op is not Ops.SQRT or len(denominator.src) != 1 or
        (radicand:=denominator.src[0]).op is not Ops.ADD): continue
    square = next((u for u in radicand.src if u.op is Ops.MUL and len(u.src) == 2 and
                   u.src[0].key == source.key and u.src[1].key == source.key), None)
    unit = next((u for u in radicand.src if u.op is Ops.CONST and float(u.arg) == 1.0), None)
    if square is not None and unit is not None: return source
  return None

def _fold_atan(root:UOp) -> UOp|None:
  """Replace Tinygrad's asin-based atan with a compact range-reduced DPU polynomial."""
  nodes = root.toposort()
  sources:dict[bytes, UOp] = {}
  for u in nodes:
    if u.op in (Ops.MUL, Ops.FDIV) and (source:=_unit_ratio_source(u)) is not None: sources[source.key] = source
  constants = {float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.half, dtypes.float)}
  if (len(sources) != 1 or not any(abs(v-math.pi/2) < 1e-12 for v in constants) or
      not any(abs(v-1.570796305) < 1e-10 for v in constants) or
      not any(abs(v+0.0012624911) < 1e-8 for v in constants)): return None
  source = next(iter(sources.values())).cast(dtypes.half)
  one = UOp.const(1.0, dtypes.half)
  magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS)
  reduced = _native_min(magnitude, one.alu(Ops.FDIV, magnitude))
  angle = reduced.alu(Ops.MUL, UOp.const(math.pi/4, dtypes.half).alu(Ops.ADD,
    one.alu(Ops.SUB, reduced).alu(Ops.MUL, UOp.const(0.2447, dtypes.half).alu(
      Ops.ADD, reduced.alu(Ops.MUL, UOp.const(0.0663, dtypes.half))))))
  large = _finite_positive_mask(magnitude.alu(Ops.SUB, one))
  reflected = UOp.const(math.pi/2, dtypes.half).alu(Ops.SUB, angle)
  selected = angle.alu(Ops.ADD, large.alu(Ops.MUL, reflected.alu(Ops.SUB, angle)))
  sign = source.alu(Ops.FDIV, magnitude.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)))
  return selected.alu(Ops.MUL, sign)

_pm_atan = PatternMatcher([(UPat(Ops.MUL, (dtypes.half, dtypes.float), name="root"), _fold_atan)])

def _hyperbolic_log_source(root:UOp) -> tuple[UOp, int]|None:
  """Match log(x + sqrt(x*x +/- 1)) after natural log expands to LOG2 times ln(2)."""
  if root.op is not Ops.MUL: return None
  for logarithm, scale in (root.src, root.src[::-1]):
    if (scale.op is not Ops.CONST or abs(float(scale.arg)-math.log(2)) > 1e-12 or logarithm.op is not Ops.LOG2 or
        len(logarithm.src) != 1 or (argument:=logarithm.src[0]).op is not Ops.ADD): continue
    for source, radical in (argument.src, argument.src[::-1]):
      if radical.op is not Ops.SQRT or len(radical.src) != 1 or (radicand:=radical.src[0]).op is not Ops.ADD: continue
      square = next((u for u in radicand.src if u.op is Ops.MUL and len(u.src) == 2 and
                     u.src[0].key == source.key and u.src[1].key == source.key), None)
      offset = next((u for u in radicand.src if u.op is Ops.CONST and float(u.arg) in (-1.0, 1.0)), None)
      if square is not None and offset is not None: return source, int(float(offset.arg))
  return None

def _poly_horner(source:UOp, coefficients:tuple[float, ...]) -> UOp:
  result = UOp.const(coefficients[-1], dtypes.half)
  for coefficient in reversed(coefficients[:-1]):
    result = result.alu(Ops.MUL, source).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  return result

def _hyperbolic_tail(magnitude:UOp, coefficients:tuple[float, ...]) -> UOp:
  """Approximate log(2*x) plus an inverse-even-power correction for large positive x."""
  one, ln2 = UOp.const(1.0, dtypes.half), UOp.const(math.log(2), dtypes.half)
  inverse = one.alu(Ops.FDIV, magnitude); inverse_square = inverse.alu(Ops.MUL, inverse)
  correction = inverse_square.alu(Ops.MUL, _poly_horner(inverse_square, coefficients))
  return magnitude.alu(Ops.LOG2).alu(Ops.MUL, ln2).alu(Ops.ADD, ln2).alu(Ops.ADD, correction)

def _fold_inverse_hyperbolic(root:UOp) -> UOp|None:
  """Stabilize Tinygrad's FP16 asinh/acosh expansions without LUT or CMAC."""
  if (matched:=_hyperbolic_log_source(root)) is None: return None
  source, offset = matched; source = source.cast(dtypes.half)
  one, ln2 = UOp.const(1.0, dtypes.half), UOp.const(math.log(2), dtypes.half)
  if offset == 1:
    magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS)
    bounded = _native_min(magnitude, UOp.const(1.5, dtypes.half))
    square = bounded.alu(Ops.MUL, bounded)
    small = bounded.alu(Ops.MUL, _poly_horner(square,
      (0.99989513, -0.16376462, 0.06135906, -0.01879756, 0.00268578)))
    safe = magnitude.alu(Ops.MAX, UOp.const(1.5, dtypes.half))
    large = _hyperbolic_tail(safe, (0.25, -3/32, 5/96))
    gate = _finite_positive_mask(magnitude.alu(Ops.SUB, UOp.const(1.5, dtypes.half)))
    selected = small.alu(Ops.ADD, gate.alu(Ops.MUL, large.alu(Ops.SUB, small)))
    sign = source.alu(Ops.FDIV, magnitude.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)))
    return selected.alu(Ops.MUL, sign)
  bounded = _native_min(source, UOp.const(2.0, dtypes.half)).alu(Ops.MAX, UOp.const(-2.0, dtypes.half))
  square = bounded.alu(Ops.MUL, bounded)
  small = bounded.alu(Ops.ADD, square.alu(Ops.SUB, one).sqrt()).alu(Ops.LOG2).alu(Ops.MUL, ln2)
  safe = source.alu(Ops.MAX, UOp.const(2.0, dtypes.half))
  large = _hyperbolic_tail(safe, (-0.25, -3/32, -5/96))
  gate = _finite_positive_mask(source.alu(Ops.SUB, UOp.const(2.0, dtypes.half)))
  return small.alu(Ops.ADD, gate.alu(Ops.MUL, large.alu(Ops.SUB, small)))

_pm_inverse_hyperbolic = PatternMatcher([
  (UPat(Ops.MUL, (dtypes.half, dtypes.float), name="root"), _fold_inverse_hyperbolic),
])

def _fold_alt_sigmoid_gradient(root:UOp) -> UOp|None:
  """Recover the stable sigmoid derivative from exp(x)/(1+exp(x)) differentiation."""
  if (root.op is not Ops.CAST or root.dtype.scalar() is not dtypes.half or len(root.src) != 1 or
      (body:=root.src[0]).op is not Ops.MUL or body.dtype.scalar() is not dtypes.float): return None
  exponential = next((u for u in body.src if u.op is Ops.EXP2), None)
  correction = next((u for u in body.src if u is not exponential), None)
  if exponential is None or correction is None or (scaled:=exponential.src[0]).op is not Ops.MUL: return None
  factor = next((u for u in scaled.src if u.op is Ops.CONST and abs(float(u.arg)-1/math.log(2)) < 1e-12), None)
  source = next((_strip_cast(u) for u in scaled.src if u is not factor), None)
  nodes = correction.toposort()
  if (factor is None or source is None or source.dtype.scalar() is not dtypes.half or
      sum(u.op is Ops.FDIV for u in nodes) != 3 or
      not any(u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and u.src[0].key == exponential.key for u in nodes) or
      not all(any(u.op is Ops.CONST and float(u.arg) == value for u in nodes) for value in (-1.0, 1.0))): return None
  one = UOp.const(1.0, dtypes.half)
  denominator = one.alu(Ops.ADD, source.alu(Ops.MUL, UOp.const(-1/math.log(2), dtypes.half)).alu(Ops.EXP2))
  sigmoid = one.alu(Ops.FDIV, denominator)
  return sigmoid.alu(Ops.MUL, one.alu(Ops.SUB, sigmoid))

_pm_alt_sigmoid_gradient = PatternMatcher([
  (UPat(Ops.CAST, dtypes.half, name="root"), _fold_alt_sigmoid_gradient),
])

def _dpu_sqrt(source:UOp, reciprocal:bool=False) -> UOp|None:
  """Approximate FP16 sqrt/rsqrt with range-independent Babylonian iterations on DPU EW."""
  if any(_local_load(u) is not None for u in source.toposort()): return None
  source = source.cast(dtypes.half)
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  negative = _positive_mask(zero.alu(Ops.SUB, source))
  finite = _native_min(source.alu(Ops.MAX, zero), UOp.const(65504.0, dtypes.half))
  safe = finite.alu(Ops.MAX, UOp.const(2**-24, dtypes.half))
  estimate = safe.alu(Ops.MAX, one)
  for _ in range(14): estimate = estimate.alu(Ops.ADD, safe.alu(Ops.FDIV, estimate)).alu(Ops.MUL, UOp.const(0.5, dtypes.half))
  valid = one.alu(Ops.SUB, negative)
  invalid_factor = valid.alu(Ops.FDIV, valid)
  result = estimate.alu(Ops.FDIV, source) if reciprocal else source.alu(Ops.FDIV, estimate)
  return result.alu(Ops.ADD, invalid_factor.alu(Ops.SUB, one))

def _native_floor(source:UOp) -> UOp:
  return UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_FLOOR)

def _dpu_periodic_reduce(source:UOp, reciprocal_period:float, split:tuple[float, ...], half_period:float) -> tuple[UOp, UOp, UOp]:
  """Reduce a finite FP16 angle with split constants so large products do not erase the residual."""
  one = UOp.const(1.0, dtypes.half)
  bounded = _native_min(source.cast(dtypes.half).alu(Ops.MAX, UOp.const(-10000.0, dtypes.half)),
                        UOp.const(10000.0, dtypes.half))
  quotient = bounded.alu(Ops.MUL, UOp.const(reciprocal_period, dtypes.half))
  magnitude = UOp(Ops.MAX, dtypes.half, src=(quotient, quotient), arg=_NATIVE_ABS)
  multiple = _native_floor(magnitude.alu(Ops.ADD, UOp.const(0.5, dtypes.half))).alu(
    Ops.MUL, _positive_mask(quotient).alu(Ops.MUL, UOp.const(2.0, dtypes.half)).alu(Ops.SUB, one))
  reduced = bounded
  for coefficient in split: reduced = reduced.alu(Ops.SUB, multiple.alu(Ops.MUL, UOp.const(coefficient, dtypes.half)))
  # The rounded FP16 quotient can be a few periods off at large magnitudes. Normalize the small residual instead.
  for _ in range(3):
    correction = _positive_mask(reduced.alu(Ops.SUB, UOp.const(half_period, dtypes.half))).alu(
      Ops.SUB, _positive_mask(UOp.const(-half_period, dtypes.half).alu(Ops.SUB, reduced)))
    multiple = multiple.alu(Ops.ADD, correction)
    for coefficient in split: reduced = reduced.alu(Ops.SUB, correction.alu(Ops.MUL, UOp.const(coefficient, dtypes.half)))
  return bounded, multiple, reduced

def _dpu_periodic_reduce_parts(source:UOp, reciprocal_period:float, split:tuple[float, ...], half_period:float) -> tuple[UOp, UOp]:
  """Return a periodic reduction as an FP16 high lane plus its arithmetic residual."""
  bounded, multiple, _ = _dpu_periodic_reduce(source, reciprocal_period, split, half_period)
  return _precise_add_parts([bounded, *(multiple.alu(Ops.MUL, UOp.const(-coefficient, dtypes.half)) for coefficient in split)])

def _dpu_reflected_angle(reduced:UOp, one:UOp) -> tuple[UOp, UOp, UOp]:
  magnitude = UOp(Ops.MAX, dtypes.half, src=(reduced, reduced), arg=_NATIVE_ABS)
  reflected = _positive_mask(magnitude.alu(Ops.SUB, UOp.const(math.pi/2, dtypes.half)))
  pi_minus = UOp.const(3.0, dtypes.half).alu(Ops.SUB, magnitude).alu(Ops.ADD, UOp.const(0.140625, dtypes.half)).alu(
    Ops.ADD, UOp.const(math.pi-3.140625, dtypes.half))
  angle = _mask_mul(magnitude, one.alu(Ops.SUB, reflected)).alu(Ops.ADD, _mask_mul(pi_minus, reflected))
  return angle, angle.alu(Ops.MUL, angle), reflected

def _dpu_sin(source:UOp) -> UOp:
  """Approximate FP16 SIN without LUTs using Cody-Waite reduction and an odd polynomial."""
  one = UOp.const(1.0, dtypes.half)
  period_split = (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125)
  if source.dtype.scalar() is dtypes.float:
    terms:list[UOp] = []
    residuals:list[UOp] = []
    for u in _flatten_binary(source, Ops.ADD):
      if u.op is Ops.CONST:
        high = struct.unpack("<e", struct.pack("<e", float(u.arg)))[0]
        terms.append(UOp.const(high, dtypes.half))
        if (low:=float(u.arg)-high) != 0.0: residuals.append(UOp.const(low, dtypes.half))
      else: terms.append(_fp32_expr_to_half(u))
    reduced_parts = [_dpu_periodic_reduce_parts(term, 1/(2*math.pi), period_split, math.pi) for term in terms]
    reduced_terms = [part[0] for part in reduced_parts]
    residuals.extend(part[1] for part in reduced_parts)
    reduced, addition_residual = _precise_sum_parts(reduced_terms)
    residuals.append(addition_residual)
    for _ in range(3):
      correction = _positive_mask(reduced.alu(Ops.SUB, UOp.const(math.pi, dtypes.half))).alu(
        Ops.SUB, _positive_mask(UOp.const(-math.pi, dtypes.half).alu(Ops.SUB, reduced)))
      reduced, normalization_residual = _precise_add_parts(
        [reduced, *(correction.alu(Ops.MUL, UOp.const(-coefficient, dtypes.half)) for coefficient in period_split)])
      residuals.append(normalization_residual)
    invalid = terms[0].alu(Ops.MUL, UOp.const(0.0, dtypes.half))
    for term in terms[1:]: invalid = invalid.alu(Ops.ADD, term.alu(Ops.MUL, UOp.const(0.0, dtypes.half)))
  else:
    source = source.cast(dtypes.half)
    _, _, reduced = _dpu_periodic_reduce(source, 1/(2*math.pi), period_split, math.pi)
    invalid = source.alu(Ops.MUL, UOp.const(0.0, dtypes.half))
  angle, square, reflected = _dpu_reflected_angle(reduced, one)
  polynomial = UOp.const(1/362880, dtypes.half)
  for coefficient in (-1/5040, 1/120, -1/6, 1.0):
    polynomial = polynomial.alu(Ops.MUL, square).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  sign = one.alu(Ops.SUB, _positive_mask(UOp.const(0.0, dtypes.half).alu(Ops.SUB, reduced)).alu(
    Ops.MUL, UOp.const(2.0, dtypes.half)))
  result = angle.alu(Ops.MUL, polynomial).alu(Ops.MUL, sign)
  if source.dtype.scalar() is dtypes.float and residuals:
    residual = residuals[0]
    for term in residuals[1:]: residual = residual.alu(Ops.ADD, term)
    cosine = _poly_horner(square, (1.0, -1/2, 1/24, -1/720, 1/40320)).alu(
      Ops.MUL, one.alu(Ops.SUB, reflected.alu(Ops.MUL, UOp.const(2.0, dtypes.half))))
    result = result.alu(Ops.ADD, residual.alu(Ops.MUL, cosine))
  return result.alu(Ops.ADD, invalid)

def _dpu_cos(source:UOp) -> UOp:
  """Approximate FP16 COS after reducing the original angle, preserving large-input phase."""
  source = source.cast(dtypes.half)
  one = UOp.const(1.0, dtypes.half)
  _, _, reduced = _dpu_periodic_reduce(source, 1/(2*math.pi), (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125), math.pi)
  _, square, reflected = _dpu_reflected_angle(reduced, one)
  polynomial = _poly_horner(square, (1.0, -1/2, 1/24, -1/720, 1/40320))
  sign = one.alu(Ops.SUB, reflected.alu(Ops.MUL, UOp.const(2.0, dtypes.half)))
  return polynomial.alu(Ops.MUL, sign).alu(Ops.ADD, source.alu(Ops.MUL, UOp.const(0.0, dtypes.half)))

def _dpu_tan_magnitude(angle:UOp, pole_magnitude:UOp) -> UOp:
  """Evaluate positive tangent magnitude directly or through reciprocal pole distance."""
  one = UOp.const(1.0, dtypes.half)
  near_pole = _positive_mask(angle.alu(Ops.SUB, UOp.const(0.75, dtypes.half)))
  local = _mask_mul(angle, one.alu(Ops.SUB, near_pole)).alu(Ops.ADD, _mask_mul(pole_magnitude, near_pole))
  square = local.alu(Ops.MUL, local)
  polynomial = UOp.const(1382/155925, dtypes.half)
  for coefficient in (62/2835, 17/315, 2/15, 1/3, 1.0):
    polynomial = polynomial.alu(Ops.MUL, square).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  tangent = local.alu(Ops.MUL, polynomial)
  safe_tangent = tangent.alu(Ops.ADD, one.alu(Ops.SUB, near_pole))
  return _mask_mul(tangent, one.alu(Ops.SUB, near_pole)).alu(Ops.ADD, near_pole.alu(Ops.FDIV, safe_tangent))

def _dpu_tan(source:UOp) -> UOp:
  """Approximate FP16 TAN with precise near-pole and large-angle reductions."""
  source = source.cast(dtypes.half)
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  bounded, multiple, reduced = _dpu_periodic_reduce(source, 1/math.pi,
    (2.0, 1.0, 0.125, 0.015625, math.pi-3.140625), math.pi/2)
  magnitude = UOp(Ops.MAX, dtypes.half, src=(reduced, reduced), arg=_NATIVE_ABS)
  reduced_sign = one.alu(Ops.SUB, _positive_mask(zero.alu(Ops.SUB, reduced)).alu(Ops.MUL, UOp.const(2.0, dtypes.half)))
  pole_index = multiple.alu(Ops.ADD, reduced_sign.alu(Ops.MUL, UOp.const(0.5, dtypes.half)))
  distance = bounded
  for coefficient in (3.0, 0.140625, math.pi-3.140625):
    distance = distance.alu(Ops.SUB, pole_index.alu(Ops.MUL, UOp.const(coefficient, dtypes.half)))
  pole_magnitude = UOp(Ops.MAX, dtypes.half, src=(distance, distance), arg=_NATIVE_ABS)
  small = _dpu_tan_magnitude(magnitude, pole_magnitude).alu(Ops.MUL, reduced_sign)

  _, _, broad = _dpu_periodic_reduce(source, 1/(2*math.pi),
    (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125), math.pi)
  angle, _, reflected = _dpu_reflected_angle(broad, one)
  broad_pole = UOp.const(1.5703125, dtypes.half).alu(Ops.SUB, angle).alu(
    Ops.ADD, UOp.const(math.pi/2-1.5703125, dtypes.half))
  broad_sign = one.alu(Ops.SUB, _positive_mask(zero.alu(Ops.SUB, broad)).alu(Ops.MUL, UOp.const(2.0, dtypes.half))).alu(
    Ops.MUL, one.alu(Ops.SUB, reflected.alu(Ops.MUL, UOp.const(2.0, dtypes.half))))
  large = _dpu_tan_magnitude(angle, broad_pole).alu(Ops.MUL, broad_sign)
  use_large = _finite_positive_mask(UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS).alu(
    Ops.SUB, UOp.const(8.0, dtypes.half)))
  result = small.alu(Ops.ADD, use_large.alu(Ops.MUL, large.alu(Ops.SUB, small)))
  return result.alu(Ops.ADD, source.alu(Ops.MUL, zero))

def _dpu_pow2_integer(exponent:UOp) -> UOp:
  """Build `2**exponent` for the FP16 exponent range with exact native DPU arithmetic."""
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  shifted = _native_min(exponent.alu(Ops.ADD, UOp.const(24.0, dtypes.half)).alu(Ops.MAX, zero), UOp.const(39.0, dtypes.half))
  scale, quotient = UOp.const(2**-24, dtypes.half), shifted
  for factor,repeats in ((2.0, 1), (4.0, 1), (16.0, 1), (256.0, 1), (256.0, 2), (256.0, 4)):
    half = _native_floor(quotient.alu(Ops.MUL, UOp.const(0.5, dtypes.half)))
    bit = quotient.alu(Ops.SUB, half.alu(Ops.MUL, UOp.const(2.0, dtypes.half)))
    multiplier = one.alu(Ops.ADD, bit.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
    for _ in range(repeats): scale = scale.alu(Ops.MUL, multiplier)
    quotient = half
  return scale

def _dpu_exp2_bounded(bounded:UOp) -> UOp:
  integer = _native_floor(bounded)
  fraction = bounded.alu(Ops.SUB, integer)
  polynomial = UOp.const(0.0013333558, dtypes.half)
  for coefficient in (0.0096181291, 0.0555041087, 0.2402265069, 0.6931471806, 1.0):
    polynomial = polynomial.alu(Ops.MUL, fraction).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  return polynomial.alu(Ops.MUL, _dpu_pow2_integer(integer))

def _dpu_exp2(source:UOp) -> UOp:
  """Approximate FP16 EXP2 without LUTs using native FLOOR, Horner arithmetic, and exact exponent scaling."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  one = UOp.const(1.0, dtypes.half)
  bounded = _native_min(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(15.9921875, dtypes.half))
  result = _dpu_exp2_bounded(bounded)
  below = mask_fn(UOp.const(-24.0, dtypes.half).alu(Ops.SUB, source))
  above = mask_fn(source.alu(Ops.SUB, UOp.const(15.9921875, dtypes.half)))
  finite = _mask_mul(result, one.alu(Ops.SUB, below))
  return finite.alu(Ops.ADD, one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one))

def _dpu_exp2_nonpositive(source:UOp) -> UOp:
  """Approximate EXP2 for a known nonpositive finite-or-negative-infinite input without domain comparisons."""
  source = source.cast(dtypes.half)
  bounded = _native_min(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(0.0, dtypes.half))
  return _dpu_exp2_bounded(bounded)

def _fold_masked_exp2(x:UOp) -> UOp|None:
  """Move a static `-inf` padding mask outside EXP2 so cumulative exponentials remain compact."""
  exponent = x.src[0]
  scaled, factor = next(((value, const) for value,const in (exponent.src, exponent.src[::-1]) if const.op is Ops.CONST), (None, None)) \
    if exponent.op is Ops.MUL else (exponent, UOp.const(1.0, exponent.dtype))
  if scaled is None or factor is None or scaled.op is not Ops.WHERE: return None
  gate, yes, no = scaled.src
  padded = tuple(arm.op is Ops.CONST and math.isinf(float(arm.arg)) and float(arm.arg) < 0 for arm in (yes, no))
  if padded.count(True) != 1 or not _is_static_expr(gate): return None
  value, mask = (no, UOp.const(1.0, dtypes.half).alu(Ops.SUB, gate.cast(dtypes.half))) if padded[0] else (yes, gate.cast(dtypes.half))
  return _mask_mul(_dpu_exp2_nonpositive(value.cast(dtypes.half).alu(Ops.MUL, factor.cast(dtypes.half))), mask)

def _dpu_log2(source:UOp) -> UOp:
  """Approximate FP16 LOG2 without LUTs using threshold exponent extraction and an atanh polynomial."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  mantissa = _native_min(source.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)), UOp.const(65504.0, dtypes.half))
  exponent = zero
  for factor,shift in ((256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0)):
    predecessor = struct.unpack("<e", struct.pack("<H", _fp16_bits(factor)-1))[0]
    mask = _finite_positive_mask(mantissa.alu(Ops.SUB, UOp.const(predecessor, dtypes.half)))
    divisor = one.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
    mantissa = mantissa.alu(Ops.FDIV, divisor)
    exponent = exponent.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(shift, dtypes.half)))
  for factor,shift in ((256.0, 8.0), (256.0, 8.0), (256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0)):
    mask = _finite_positive_mask(UOp.const(2.0/factor, dtypes.half).alu(Ops.SUB, mantissa))
    multiplier = one.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
    mantissa = mantissa.alu(Ops.MUL, multiplier)
    exponent = exponent.alu(Ops.SUB, mask.alu(Ops.MUL, UOp.const(shift, dtypes.half)))
  z = mantissa.alu(Ops.SUB, one).alu(Ops.FDIV, mantissa.alu(Ops.ADD, one))
  z2 = z.alu(Ops.MUL, z)
  polynomial = UOp.const(1.0/9.0, dtypes.half)
  for coefficient in (1.0/7.0, 1.0/5.0, 1.0/3.0, 1.0):
    polynomial = polynomial.alu(Ops.MUL, z2).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  result = exponent.alu(Ops.ADD, z.alu(Ops.MUL, polynomial).alu(Ops.MUL, UOp.const(2.0/math.log(2.0), dtypes.half)))
  nonzero = mask_fn(source).alu(Ops.MAX, mask_fn(zero.alu(Ops.SUB, source)))
  zero_correction = UOp.const(-1.0, dtypes.half).alu(Ops.FDIV, nonzero).alu(Ops.ADD, one)
  valid = one.alu(Ops.SUB, mask_fn(zero.alu(Ops.SUB, source)))
  negative_correction = valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one)
  above = mask_fn(source.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
  inf_correction = one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one)
  return result.alu(Ops.ADD, zero_correction).alu(Ops.ADD, negative_correction).alu(Ops.ADD, inf_correction)

_pm_rsqrt = PatternMatcher([(UPat(Ops.RECIPROCAL, (dtypes.half, dtypes.float),
  src=(UPat(Ops.SQRT, (dtypes.half, dtypes.float), src=(UPat.var("source"),)),)), lambda source:_dpu_sqrt(source, True))])
_pm_sqrt = PatternMatcher([(UPat(Ops.SQRT, (dtypes.half, dtypes.float), src=(UPat.var("source"),)), lambda source:_dpu_sqrt(source))])
_pm_exp2 = PatternMatcher([(UPat(Ops.EXP2, (dtypes.half, dtypes.float), src=(UPat.var("source"),)), lambda source:_dpu_exp2(source))])
_pm_masked_exp2 = PatternMatcher([(UPat(Ops.EXP2, (dtypes.half, dtypes.float), name="x"), _fold_masked_exp2)])
_pm_log2 = PatternMatcher([(UPat(Ops.LOG2, (dtypes.half, dtypes.float), src=(UPat.var("source"),)), lambda source:_dpu_log2(source))])
_pm_sin = PatternMatcher([(UPat(Ops.SIN, (dtypes.half, dtypes.float), src=(UPat.var("source"),)), lambda source:_dpu_sin(source))])
def _cos_source(x:UOp) -> UOp|None:
  """Recover x from Tinygrad's casted cos(x) = sin(pi/2-x)."""
  x = _strip_cast(x)
  if x.op is not Ops.SIN or len(x.src) != 1 or (phase:=_const_operand(_strip_cast(x.src[0]), Ops.ADD, math.pi/2)) is None: return None
  negative = _const_operand(phase[0], Ops.MUL, -1.0)
  return _strip_cast(negative[0]) if negative is not None else None
_pm_cos = PatternMatcher([(UPat(Ops.SIN, (dtypes.half, dtypes.float), name="x"),
  lambda x:_dpu_cos(source) if (source:=_cos_source(x)) is not None else None)])
_pm_tan = PatternMatcher([(UPat(Ops.FDIV, (dtypes.half, dtypes.float), name="x"),
  lambda x:_dpu_tan(source) if len(x.src) == 2 and (numerator:=_strip_cast(x.src[0])).op is Ops.SIN and
  (cosine:=_cos_source(x.src[1])) is not None and (source:=_strip_cast(numerator.src[0])).key == cosine.key else None)])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  sink = graph_rewrite(sink, _pm_alt_sigmoid_gradient, name="rockchip alternate sigmoid gradient")
  sink = graph_rewrite(sink, _pm_inverse_hyperbolic, name="rockchip inverse hyperbolic")
  sink = graph_rewrite(sink, _pm_atan, name="rockchip atan")
  sink = graph_rewrite(sink, _pm_tan, name="rockchip tan")
  sink = graph_rewrite(sink, _pm_cos, name="rockchip cos")
  sink = graph_rewrite(sink, _pm_sin, name="rockchip sin")
  sink = graph_rewrite(sink, _pm_masked_exp2, name="rockchip masked exp2")
  sink = graph_rewrite(sink, _pm_exp2, name="rockchip exp2")
  sink = graph_rewrite(sink, _pm_log2, name="rockchip log2")
  sink = graph_rewrite(sink, _pm_rsqrt, name="rockchip rsqrt")
  sink = graph_rewrite(sink, _pm_sqrt, name="rockchip sqrt")
  sink = graph_rewrite(sink, _pm_round, name="rockchip round")
  sink = graph_rewrite(sink, _pm_floor_ceil, name="rockchip floor/ceil")
  sink = graph_rewrite(sink, _pm_trunc, name="rockchip trunc")
  sink = graph_rewrite(sink, _pm_abs, name="rockchip abs")
  sink = graph_rewrite(sink, _pm_sign, name="rockchip sign")
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4, shared_max = True, False, False, 16
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None,
                 Ops.FDIV: lambda: None, Ops.SQRT: lambda: None, Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_uop_program(uops)
    if image is None: image = _lower_uop_program(_fp16_rewrite(uops), recipes_ready=True)
    if image is None: raise RuntimeError("RKPLAN_REJECT:generic_uops " + repr([(i, u.op.name, str(u.dtype)) for i,u in enumerate(uops)]))
    return base64.b64encode(encode_image(_hoist_leading_vector_materialization(image))).decode()
