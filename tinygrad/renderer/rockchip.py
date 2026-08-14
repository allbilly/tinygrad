from __future__ import annotations
# ruff: noqa: E702
import base64, functools, heapq, math, os, struct
import numpy as np
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Iterable, Mapping, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes, float_to_fp16
from tinygrad.helpers import Target, ceildiv, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, UPat, PatternMatcher, exec_alu, graph_rewrite, python_alu
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 31
_HEADER = struct.Struct("<4sHHHHHHIIIIII")  # magic/version/target, scratch/gather/host counts, ops/constants, phase split, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBBBBiIIii"), struct.Struct("<IIi")
_HOST_ADDRESS = struct.Struct("<BBBBBHHHIIIIIIiiiiii")
_FILL = struct.Struct("<BBHI")  # dst_kind, itemsize, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, flags, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg
_ITEM_FORMAT = {1:"B", 2:"H", 4:"I"}
_RKIMAGE_U16_MAX = (1 << 16) - 1

class RKTarget(IntEnum): RK3588 = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1
class RKLayout(IntEnum): FP16 = 0; INT16 = 1; BOOL_MASK = 2; INT32 = 3; BOOL_INT16 = 4; INT_FP16 = 5
class RKExecutionClass(IntEnum): NATIVE = 0; HOST_ADDRESS = 1

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True)
class RKValue:
  """Physical ABI. BOOL_MASK and bounded exact INT_FP16 values occupy FP16 scratch lanes."""
  arg: RKArg; dtype: DType; count: int; layout: RKLayout

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
  after: int = -1  # EW-op split for a mid-program gather; -1 uses RKImage.gather_after

@dataclass(frozen=True)
class RKHostAddress:
  """Host-calculated raw-lane movement. It never owns numeric or reduction semantics."""
  src: RKArg; index: RKArg; dst: RKArg; count: int; src_count: int; dst_count: int
  itemsize: int = 2; index_itemsize: int = 4; fill_bits: int = 0; normalize_negative: bool = False
  index_limit: int = 0; base: int = 0; index_scale: int = 1; lane_stride: int = 0

@dataclass(frozen=True)
class RKMultiGather: gathers: tuple[RKGather, ...]

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
  host_gathers: tuple[RKHostAddress, ...] = (); host_scatters: tuple[RKHostAddress, ...] = ()

  @property
  def execution_class(self) -> RKExecutionClass:
    return RKExecutionClass.HOST_ADDRESS if self.host_gathers or self.host_scatters else RKExecutionClass.NATIVE

def _map_image_args(image:RKImage, fn:Callable[[RKArg], RKArg]) -> RKImage:
  def gather(value:RKGather) -> RKGather:
    src, dst = fn(RKArg(value.src_kind, value.src_index)), fn(RKArg(value.dst_kind, value.dst_index))
    return replace(value, src_kind=src.kind, src_index=src.index, dst_kind=dst.kind, dst_index=dst.index)
  def host(value:RKHostAddress) -> RKHostAddress: return replace(value, src=fn(value.src), index=fn(value.index), dst=fn(value.dst))
  return replace(image, gathers=tuple(map(gather, image.gathers)), mid_gathers=tuple(map(gather, image.mid_gathers)),
    post_gathers=tuple(map(gather, image.post_gathers)), ew_ops=tuple(replace(op, dst=fn(op.dst), lhs=fn(op.lhs), rhs=fn(op.rhs))
    for op in image.ew_ops), fill=replace(image.fill, dst=fn(image.fill.dst)) if image.fill is not None else None,
    host_gathers=tuple(map(host, image.host_gathers)), host_scatters=tuple(map(host, image.host_scatters)))

def _alias_image_args(image:RKImage, aliases:dict[int, RKArg]) -> RKImage:
  return _map_image_args(image, lambda arg:replace(aliases[arg.index], addend=aliases[arg.index].addend+arg.addend)
    if arg.kind is RKBufferKind.ARG and arg.index in aliases else arg)

def _reuse_linear_scratch(image:RKImage, constant_slots:dict[bytes, int]) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  events:dict[int, tuple[int, int]] = {}
  def touch(arg:RKArg, event:int) -> None:
    if arg.kind is RKBufferKind.SCRATCH: events[arg.index] = (events.get(arg.index, (event, event))[0], event)
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
    mid_by_point.setdefault(gather.after if gather.after >= 0 else image.gather_after, []).append(gather)
  for index,op in enumerate(image.ew_ops):
    for gather in mid_by_point.get(index, ()): touch_gather(gather, event); event += 1
    touch(op.lhs, event); touch(op.rhs, event); touch(op.dst, event)
    event += 1
  for gather in mid_by_point.get(len(image.ew_ops), ()): touch_gather(gather, event); event += 1
  for gather in image.post_gathers: touch_gather(gather, event); event += 1
  for host in image.host_scatters: touch_host(host, event); event += 1
  if image.fill is not None: touch(image.fill.dst, event)
  if not events: return replace(image, scratch=(), constants=b"")
  if any(not 0 <= slot < len(image.scratch) for slot in events): raise ValueError("invalid virtual scratch slot")
  # Mid-program gathers may populate one logical slot in several partial phases. The runtime clears a
  # destination once per physical slot, so these stateful materialization slots must not alias.
  pinned = {gather.dst_index for gather in image.mid_gathers if gather.dst_kind is RKBufferKind.SCRATCH}
  intervals = sorted(((points[0], points[1], slot) for slot,points in events.items()), key=lambda item:(item[0], item[2]))
  remap:dict[int, int] = {}
  physical:list[RKScratch] = []
  physical_reusable:list[bool] = []
  active:list[tuple[int, int]] = []
  available:list[int] = []
  for start,end,slot in intervals:
    while active and active[0][0] < start:
      _,target = heapq.heappop(active)
      if physical_reusable[target]: heapq.heappush(available, target)
    spec = image.scratch[slot]
    if slot not in pinned and available:
      target = heapq.heappop(available)
      physical[target] = RKScratch(max(physical[target].size, spec.size), max(physical[target].alignment, spec.alignment))
    else:
      target = len(physical)
      physical.append(spec); physical_reusable.append(slot not in pinned)
    if physical_reusable[target]: heapq.heappush(active, (end, target))
    remap[slot] = target
  def remap_arg(arg:RKArg) -> RKArg:
    return RKArg(arg.kind, remap[arg.index], arg.addend) if arg.kind is RKBufferKind.SCRATCH else arg
  def remap_ew(op:RKEWOp) -> RKEWOp:
    return RKEWOp(remap_arg(op.dst), remap_arg(op.lhs), remap_arg(op.rhs), op.count, op.ew_cfg, op.submit_barrier,
      op.compare, op.stateful, op.int32_output, op.int32_input, op.bool_output, op.int16_output, op.int16_input)
  def remap_gather(gather:RKGather) -> RKGather:
    return replace(gather,
    src_index=remap[gather.src_index] if not gather.values and gather.src_kind is RKBufferKind.SCRATCH else gather.src_index,
    dst_index=remap[gather.dst_index] if gather.dst_kind is RKBufferKind.SCRATCH else gather.dst_index)
  def remap_host(host:RKHostAddress) -> RKHostAddress:
    return replace(host, src=remap_arg(host.src), index=remap_arg(host.index), dst=remap_arg(host.dst))
  gathers = tuple(remap_gather(gather) for gather in image.gathers)
  ew_ops = tuple(map(remap_ew, image.ew_ops))
  by_slot:dict[int, bytes] = {}
  for bits,slot in constant_slots.items():
    target = remap[slot]
    if target in by_slot and by_slot[target] != bits: raise ValueError("overlapping scratch constants")
    by_slot[target] = bits
  constants = b"" if not by_slot else b"".join(by_slot.get(slot, b"\0\0") for slot in range(max(by_slot)+1))
  return replace(image, scratch=tuple(physical), constants=constants, gathers=gathers, ew_ops=ew_ops,
    mid_gathers=tuple(remap_gather(gather) for gather in image.mid_gathers),
    post_gathers=tuple(remap_gather(gather) for gather in image.post_gathers),
    host_gathers=tuple(remap_host(host) for host in image.host_gathers),
    host_scatters=tuple(remap_host(host) for host in image.host_scatters),
    fill=None if image.fill is None else replace(image.fill, dst=remap_arg(image.fill.dst)))

@dataclass(frozen=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.mid_gathers + image.post_gathers
  if image.mid_gathers and any(not 0 <= (g.after if g.after >= 0 else image.gather_after) <= len(image.ew_ops) for g in image.mid_gathers):
    raise ValueError("invalid mid-gather split")
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(gathers),
                               len(image.host_gathers), len(image.host_scatters),
                               len(image.ew_ops), len(image.constants), len(image.mid_gathers), len(image.post_gathers),
                               image.gather_after, int(image.fill is not None)))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
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
      int(host.normalize_negative), host.index_limit, host.src.addend, host.index.addend, host.dst.addend,
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
  if image.fill is not None: out += _FILL.pack(int(image.fill.dst.kind), image.fill.itemsize, image.fill.dst.index, image.fill.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nhost_gather, nhost_scatter, nop, nconst, mid_count, post_count, gather_after, flags = \
    _HEADER.unpack_from(blob)
  if (magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or mid_count+post_count > ngather or flags & ~1 or
      (mid_count and not 0 <= gather_after < nop) or (not mid_count and gather_after != 0)): raise ValueError("invalid RKImage header")
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
      fill_bits, host_flags, index_limit, src_addend, index_addend, dst_addend, base, index_scale, lane_stride = \
      _HOST_ADDRESS.unpack_from(blob, off)
    off += _HOST_ADDRESS.size
    if (src_kind not in (0, 1) or index_kind not in (0, 1) or dst_kind not in (0, 1) or itemsize not in _ITEM_FORMAT or
        index_itemsize not in (2, 4) or host_flags & ~1 or min(count, src_count, dst_count, index_limit) < 0):
      raise ValueError("invalid RKHostAddress")
    host_addresses.append(RKHostAddress(RKArg(RKBufferKind(src_kind), src_index, src_addend),
      RKArg(RKBufferKind(index_kind), index_index, index_addend), RKArg(RKBufferKind(dst_kind), dst_index, dst_addend),
      count, src_count, dst_count, itemsize, index_itemsize, fill_bits, bool(host_flags & 1),
      index_limit, base, index_scale, lane_stride))
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
                 tuple(gathers[pre_count:pre_count+mid_count]), gather_after, tuple(gathers[-post_count:] if post_count else ()),
                 tuple(host_addresses[:nhost_gather]), tuple(host_addresses[nhost_gather:]))

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
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_COPYSIGN, _NATIVE_FLOOR, _NATIVE_LEAKY_RELU, _NATIVE_MASK_MUL, _NATIVE_MIN,
 _NATIVE_POSITIVE_MASK, _NATIVE_PRECISE_ADD, _NATIVE_RAW_MIN, _NATIVE_RELU6, _NATIVE_SIGN) = (
   "rockchip_abs", "rockchip_ceil", "rockchip_copysign", "rockchip_floor", "rockchip_leaky_relu", "rockchip_mask_mul",
   "rockchip_min", "rockchip_positive_mask", "rockchip_precise_add", "rockchip_raw_min", "rockchip_relu6", "rockchip_sign")
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
def _int32_tiles_bytes(count:int) -> int: return ceildiv(count, 4) * 64
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff
def _int16_low_bytes(source:RKArg, out_slot:int, count:int, stride:int=2) -> RKGather:
  return RKGather(source.index, out_slot, count, base=source.addend, axes=((1, count, stride),),
                  dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)

@functools.lru_cache(maxsize=256)
def _stateful_stage_template(count:int, ew_cfg:int, compare:bool=False, int32_output:bool=False, int32_input:bool=False,
                             int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False, fp32_input:bool=False) \
                             -> tuple[tuple[int, ...], tuple[int, ...]]:
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
  commands = [_cmd(*x) for x in regs]
  relocs:list[int] = []
  for target,reg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR),
                     (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR)):
    relocs.append(len(commands)); commands.append(_cmd(target, reg, 0))
  rdma_precision = 5 if fp32_input else 4 if int32_input else 1 if int16_input else 2
  rdma_feature = (rdma_precision<<15)|(15<<11)|(rdma_precision<<5)|(0 if is_div or int16_input or fp32_input else 1<<3)|1
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature))
  return tuple(commands), tuple(relocs)

def _emit_stateful_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                         int32_output:bool=False, int32_input:bool=False,
                         int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False, fp32_input:bool=False) -> RKStage:
  commands, words = _stateful_stage_template(count, ew_cfg, compare, int32_output, int32_input,
    int16_output, int16_input, fp32_output, fp32_input)
  return RKStage(commands, tuple(RKReloc(word, arg) for word,arg in zip(words, (dst, lhs, rhs))))

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

def _semantic_loads(u:UOp, cache:dict[UOp, tuple[UOp, ...]]|None=None) -> tuple[UOp, ...]:
  if cache is not None and u in cache: return cache[u]
  sources = () if u.op in (Ops.RANGE, Ops.SPECIAL) else u.src[:1] if u.op is Ops.AFTER else u.src
  result = (u,) if u.op is Ops.LOAD else tuple(dict.fromkeys(y for x in sources for y in _semantic_loads(x, cache)))
  if cache is not None: cache[u] = result
  return result

def _semantic_local_loads(u:UOp, cache:dict[UOp, tuple[UOp, ...]]|None=None) -> tuple[UOp, ...]:
  if cache is not None and u in cache: return cache[u]
  sources = () if u.op in (Ops.RANGE, Ops.SPECIAL) else u.src[:1] if u.op is Ops.AFTER else u.src
  result = (load,) if (load:=_local_load(u)) is not None else tuple(dict.fromkeys(y for x in sources for y in _semantic_local_loads(x, cache)))
  if cache is not None: cache[u] = result
  return result

def _static_cast(value, dtype:DType, vector:bool=False):
  if vector or isinstance(value, np.ndarray): return np.asarray(value, dtype=np.dtype(dtype.scalar().fmt) if dtype.scalar().fmt is not None else None)
  if dtype.scalar() is dtypes.bool: return bool(value)
  if dtype.scalar() in dtypes.ints: return int(value)
  if dtype.scalar() is dtypes.half: return float_to_fp16(value)
  if dtype.scalar() is dtypes.float: return struct.unpack("<f", struct.pack("<f", float(value)))[0]
  return float(value)

_STATIC_SCALAR_ALU = {Ops.ADD, Ops.MUL, Ops.SUB, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.MAX,
                      Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR}
def _static_alu(op:Ops, dtype:DType, values:tuple, vector:bool):
  if vector:
    lhs = values[0]
    if op in (Ops.CDIV, Ops.CMOD):
      rhs = values[1]
      with np.errstate(divide="ignore", invalid="ignore"): quotient = np.where(rhs != 0, np.trunc(lhs / rhs), 0)
      value = quotient if op is Ops.CDIV else lhs-quotient*rhs
    elif op in (Ops.FLOORDIV, Ops.FLOORMOD):
      rhs = values[1]
      quotient = np.zeros(np.broadcast_shapes(lhs.shape, rhs.shape), dtype=np.result_type(lhs, rhs))
      np.floor_divide(lhs, rhs, out=quotient, where=rhs != 0)
      value = quotient if op is Ops.FLOORDIV else lhs-quotient*rhs
    elif op is Ops.MAX: value = np.where(lhs < values[1], values[1], lhs)
    elif op is Ops.WHERE: value = np.where(*values)
    elif op is Ops.TRUNC: value = np.vectorize(int, otypes=[np.int64])(lhs)
    elif op is Ops.RECIPROCAL: value = 1.0 / lhs
    else:
      try: value = python_alu[op](*values)
      except KeyError: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {op.name}")
  else:
    if op not in _STATIC_SCALAR_ALU: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {op.name}")
    operands = tuple(int(x) for x in values) if op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.AND, Ops.OR, Ops.XOR) else values
    value = exec_alu(op, dtype, operands, truncate_output=False)
  return value if not vector and op in (Ops.CMPLT, Ops.CMPNE) else _static_cast(value, dtype, vector)

def _eval_expr(u:UOp, env:dict[UOp, int], cache:dict[UOp, int|float|bool]) -> int|float|bool:
  if u in cache: return cache[u]
  if u.op is Ops.CONST: ret = _static_cast(u.arg, u.dtype)
  elif u.op in (Ops.RANGE, Ops.SPECIAL): ret = env[u]
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
  elif u.op is Ops.CAST: ret = _static_cast(_eval_expr(u.src[0], env, cache), u.dtype)
  elif u.op is Ops.WHERE:
    ret = _static_cast(_eval_expr(u.src[1] if _eval_expr(u.src[0], env, cache) else u.src[2], env, cache), u.dtype)
  else:
    lhs = _eval_expr(u.src[0], env, cache)
    if u.op is Ops.RECIPROCAL: ret = _static_cast(1.0 / float(lhs), u.dtype)
    elif u.op is Ops.TRUNC: ret = _static_cast(int(lhs), u.dtype)
    else: ret = _static_alu(u.op, u.dtype, (lhs, _eval_expr(u.src[1], env, cache)), False)
  cache[u] = ret
  return ret

def _eval_int(u:UOp, env:dict[UOp, int], cache:dict[UOp, int|float|bool]|None=None) -> int:
  return int(_eval_expr(u, env, {} if cache is None else cache))

def _eval_vector(u:UOp, env:Mapping[UOp, np.ndarray|int], cache:dict[UOp, np.ndarray],
                 load:Callable[[UOp], np.ndarray]|None=None) -> np.ndarray:
  if u in cache: return cache[u]
  if u.op is Ops.CONST: ret = _static_cast(u.arg, u.dtype, True)
  elif u.op in (Ops.RANGE, Ops.SPECIAL): ret = _static_cast(env[u], u.dtype, True)
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
  elif u.op is Ops.AFTER: ret = _eval_vector(u.src[0], env, cache, load)
  elif u.op is Ops.LOAD and load is not None: ret = load(u)
  elif u.op is Ops.CAST: ret = _static_cast(_eval_vector(u.src[0], env, cache, load), u.dtype, True)
  else:
    values = tuple(_eval_vector(x, env, cache, load) for x in u.src)
    ret = _static_alu(u.op, u.dtype, values, True)
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

def _iter_range_env(ranges:list[UOp], max_envs:int|None=_MAX_STATIC_RANGE_ENVS, dependencies:bool=True) -> list[dict[UOp, int]]:
  if not ranges: return [{}]
  order:list[UOp] = []
  if dependencies:
    seen:set[UOp] = set()
    def add(r:UOp) -> None:
      if r in seen: return
      for src in r.src[1:]:
        if src.op is Ops.RANGE: add(src)
      seen.add(r); order.append(r)
    for r in ranges: add(r)
  else: order = ranges
  envs:list[dict[UOp, int]] = [{}]
  for r in order:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    bound = int(r.src[0].arg)
    if max_envs is not None and (bound < 0 or bound and len(envs) > max_envs//bound): raise RuntimeError("RKPLAN_REJECT:static_index_budget")
    envs = [{**env, r: i} for env in envs for i in range(bound)]
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
  post_sqrt:bool = False; post_reciprocal:bool = False; post_cuberoot:bool = False

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

def _unrolled_local_add(uops:list[UOp], out_index:UOp, initial:tuple[DType, int]|None=None) -> UOp|None:
  """Expand one statically bounded local ADD loop after optionally validating its identity."""
  if (loop:=_local_add_loop(uops, out_index)) is None: return None
  reduce_range, groups, term, local_stores = loop
  initializers = [local for local in local_stores if reduce_range not in local.toposort()]
  if initial is not None and (len(initializers) != 1 or initializers[0].src[1].op is not Ops.CONST or
     initializers[0].src[1].dtype.scalar() is not initial[0] or int(initializers[0].src[1].arg) != initial[1]): return None
  terms = [term.substitute({reduce_range:reduce_range.const_like(lane)}) for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value.alu(Ops.ADD, term)
  return value

def _loop_reduction_match(uops:list[UOp]) -> RKLoopReduction|None:
  """Parse the output, accumulator update, shape, and optional final scale of a loop reduction."""
  if (output:=_output_store(uops, (dtypes.half, dtypes.float), allow_local=True)) is None: return None
  store, out, _, _, root = output
  nodes = list(root.toposort())
  if (shape:=_loop_reduction_shape(store, out, nodes)) is None: return None
  rows, envs, reduce_range, groups = shape
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and reduce_range in u.toposort()]
  if len(updates) != 1: return None
  value_root = root
  if root.op is Ops.MAX:
    unclamped, epsilon = next(((value, const) for value,const in (root.src, root.src[::-1]) if const.op is Ops.CONST), (None, None))
    if unclamped is None or epsilon is None or _fp16_bits(float(epsilon.arg)) != 0: return None
    value_root = unclamped
  post_sqrt = value_root.op is Ops.SQRT and len(value_root.src) == 1
  post_reciprocal = value_root.op is Ops.FDIV and len(value_root.src) == 2 and value_root.src[0].op is Ops.CONST and \
                    float(value_root.src[0].arg) == 1.0 and _local_load(value_root.src[1]) is not None
  local_values = {load.key:load for node in value_root.toposort() if (load:=_local_load(node)) is not None}
  post_cuberoot = not post_sqrt and not post_reciprocal and len(local_values) == 1 and any(
    node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half, dtypes.float) and abs(float(node.arg)-1/3) < 1e-6
    for node in value_root.toposort())
  value = (value_root.src[0] if post_sqrt else value_root.src[1] if post_reciprocal else
           next(iter(local_values.values())) if post_cuberoot else value_root)
  if _local_load(value) is not None: post_scale = 1.0
  elif value.op is Ops.MUL and (load:=next((x for x in value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=value.src[1 if value.src[0] is load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else: return None
  return RKLoopReduction(store, out, nodes, rows, envs, reduce_range, groups, _strip_cast(updates[0].src[1]),
                         post_scale, post_sqrt, post_reciprocal, post_cuberoot)

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

@functools.lru_cache(maxsize=8)
def _static_vector_env(out_index:UOp, count:int) -> tuple[tuple[UOp, ...], dict[UOp, np.ndarray], np.ndarray]:
  ranges = tuple(_index_ranges(out_index))
  envs = _iter_range_env(list(ranges))
  vector_env = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}
  dst_lanes = np.broadcast_to(_eval_vector(out_index, vector_env, {}), len(envs)).astype(np.int64)
  return ranges, vector_env, dst_lanes

def _static_values(out_index:UOp, expr:UOp, count:int, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
  ranges, vector_env, dst_lanes = _static_vector_env(out_index, count)
  if any(r not in ranges for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  expr_lanes = np.broadcast_to(_eval_vector(expr, vector_env, {}), len(dst_lanes))
  encoded:np.ndarray
  if encode is _fp16_bits:
    fp_values = np.asarray(expr_lanes, dtype=np.float64)
    if np.any(np.isfinite(fp_values) & (np.abs(fp_values) >= 65520)): raise OverflowError("float too large to pack with e format")
    encoded = fp_values.astype(np.float16).view(np.uint16)
  elif encode is _int16_bits: encoded = np.asarray(expr_lanes).astype(np.int64) & 0xffff
  elif encode is int: encoded = np.asarray(expr_lanes).astype(np.int64)
  else: encoded = np.fromiter((encode(raw.item()) for raw in expr_lanes), dtype=np.int64, count=len(expr_lanes))
  if np.any((dst_lanes < 0) | (dst_lanes >= count)): raise RuntimeError("RKPLAN_REJECT:static_index")
  order = np.argsort(dst_lanes); dst, values = dst_lanes[order], encoded[order]
  starts = np.empty(len(dst), dtype=np.bool_); starts[:1] = True; starts[1:] = dst[1:] != dst[:-1]
  if not np.array_equal(dst[starts], np.arange(count)) or np.any(values[1:][~starts[1:]] != values[:-1][~starts[1:]]):
    raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(int(x) for x in values[starts])

def _static_int_vector(out_index:UOp, expr:UOp, count:int) -> tuple[int, ...]:
  """Evaluate a compile-time integer expression in compact output order."""
  return _static_values(out_index, expr, count, int)

def _static_int_vectors(out_index:UOp, exprs:tuple[UOp, ...], count:int) -> tuple[tuple[int, ...], ...]:
  """Vector-evaluate static integer rows with one shared index-expression cache."""
  ranges, vector_env, dst = _static_vector_env(out_index, count)
  if any(r not in ranges for expr in exprs for r in _index_ranges(expr)): raise RuntimeError("RKPLAN_REJECT:static_index")
  cache:dict[UOp, np.ndarray] = {}
  if len(dst) != count or np.any((dst < 0) | (dst >= count)) or not np.array_equal(np.sort(dst), np.arange(count)):
    return tuple(_static_int_vector(out_index, expr, count) for expr in exprs)
  order = np.argsort(dst)
  return tuple(tuple(int(x) for x in np.broadcast_to(_eval_vector(expr, vector_env, cache), len(dst))[order]) for expr in exprs)

_LinearTerm = UOp|tuple[UOp, int]
def _linear_index(u:UOp, divided:bool=False) -> tuple[int, dict[_LinearTerm, int]]|None:
  """Represent static address arithmetic as a sum of scaled RANGE or RANGE//constant terms."""
  if u.op is Ops.CONST: return int(u.arg), {}
  if divided and u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() in (dtypes.int, dtypes.uint):
    return _linear_index(u.src[0], divided)
  term:_LinearTerm|None = (u, 1) if divided and u.op in (Ops.RANGE, Ops.SPECIAL) else u if u.op in (Ops.RANGE, Ops.SPECIAL) else None
  if divided and u.op is Ops.CDIV and len(u.src) == 2 and u.src[0].op in (Ops.RANGE, Ops.SPECIAL) and \
     u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0: term = (u.src[0], int(u.src[1].arg))
  if term is not None: return 0, {term:1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _linear_index(u.src[0], divided), _linear_index(u.src[1], divided)
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {key:coefficient*scale for key,coefficient in affine[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  terms = lhs[1].copy()
  for key,coefficient in rhs[1].items():
    if (merged:=terms.get(key, 0)+sign*coefficient): terms[key] = merged
    elif key in terms: del terms[key]
  return lhs[0]+sign*rhs[0], terms

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  return typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(u))

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

def _contiguous_output_samples(out_index:UOp, count:int) -> list[dict[UOp, int]]|None:
  """Prove a contiguous affine output index and return bounded range samples for large matcher validation."""
  ranges, affine = _index_ranges(out_index), _affine_index(out_index)
  if affine is None or affine[0] != 0 or set(affine[1]) != set(ranges): return None
  extent = 1
  for r,stride in sorted(affine[1].items(), key=lambda item:item[1]):
    if stride != extent or not r.src or r.src[0].op is not Ops.CONST or (limit:=int(r.src[0].arg)) <= 0: return None
    extent *= limit
  if extent != count: return None
  envs:list[dict[UOp, int]] = [{}]
  for r in ranges:
    limit = int(r.src[0].arg)
    samples = tuple(dict.fromkeys((0, min(1, limit-1), limit//2, limit-1)))
    envs = [{**env, r:value} for env in envs for value in samples]
  return envs

def _typed_load_offsets(load:UOp, dtype:DType, out_index:UOp, count:int, allow_fill:bool=False) -> tuple[UOp, tuple[int, ...]]|None:
  """Resolve one typed global load to bounded static offsets."""
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX: return None
  param = _root_param(load.src[0])
  if param is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None
  try: offsets = _gather_offsets(out_index, load.src[0].src[1], load.src[2] if len(load.src) == 3 else None, count)
  except RuntimeError: return None
  if any(offset < (-1 if allow_fill else 0) or offset >= int(param.src[0].arg) for offset in offsets): return None
  return param, offsets

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
  if gate is None and out_affine is not None and out_affine[0] == 0 and \
     (load_divided:=typing_cast(tuple[int, dict[tuple[UOp, int], int]]|None, _linear_index(load_index, True))) is not None:
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

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is not Ops.MAX or u.dtype.scalar() is not dtypes.half: return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None


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

def _precise_add_parts(terms:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 addition residuals as a high lane plus a low correction lane."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  high, middle, low = terms[0], zero, zero
  for part in terms[1:]:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = low.alu(Ops.ADD, error)
  middle = middle.alu(Ops.ADD, low)
  return high, middle

def _precise_sum_parts(terms:list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 product and addition residuals as a high lane plus a low correction lane."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  products, errors = tuple(x[0] for x in pairs), tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL)
  return _precise_add_parts(products + errors)

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  high, middle = _precise_sum_parts(terms)
  root = high.alu(Ops.ADD, middle)
  cache:dict[UOp, UOp] = {}
  for u in root.toposort():
    tagged = u.replace(src=tuple(cache[src] for src in u.src))
    if tagged.op is Ops.ADD: tagged = tagged.replace(arg=_NATIVE_PRECISE_ADD)
    cache[u] = tagged
  return cache[root]

def _lower_composed_uops(uops:list[UOp], *, recipes_ready:bool=False) -> RKImage:
  image = _lower_uop_program(uops, vectorize_reductions=False, recipes_ready=recipes_ready)
  if image is None: raise RuntimeError("RKPLAN_REJECT:composed_uops")
  return image

def _lower_dot_loop_reduction(loop:RKLoopReduction) -> RKImage|None:
  """Lower an FP16 dot loop as vector MUL terms followed by a balanced vector ADD tree."""
  if (loop.out.dtype.scalar() is not dtypes.half or loop.post_scale != 1.0 or
      loop.post_sqrt or loop.post_reciprocal or loop.post_cuberoot): return None
  store, update, reduce_range, groups = loop.store, loop.update, loop.reduce_range, loop.groups
  if update.op is not Ops.ADD or (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None: return None
  product = _strip_cast(update.src[1 if update.src[0] is acc else 0])
  if product.op is not Ops.MUL or product.arg is not None or product.dtype.scalar() is not dtypes.half: return None
  for operand in product.src:
    operand = _strip_cast(operand)
    param = _root_param(operand.src[0]) if operand.op is Ops.LOAD and operand.src and operand.src[0].op is Ops.INDEX else None
    if param is None or operand.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST: return None
  terms = [product.substitute({reduce_range:reduce_range.const_like(r)}) for r in range(groups)]
  if groups >= _MIN_GENERIC_PRODUCT_RESIDUAL_TERMS:
    try: return _lower_composed_uops([store.replace(src=(store.src[0], _precise_mul_sum(terms), *store.src[2:]))])
    except RuntimeError: pass
  summed = terms[0]
  for term in terms[1:]: summed = summed.alu(Ops.ADD, term)
  precise_store = store.replace(src=(store.src[0], summed, *store.src[2:]))
  if (precise:=_lower_vectorized_mul_add_reduction(list(UOp(Ops.SINK, src=(precise_store,)).toposort()))) is not None: return precise
  # Materialize bounded dot domains so the arena reduction preserves a real balanced tree instead of a rewritten ADD chain.
  lanes, out_index = loop.rows*groups, store.src[0].src[1]
  if lanes <= _MAX_EW_ELEMS_FP16*(_reduction_stride(1)//2):
    linear_index = reduce_range.alu(Ops.MUL, reduce_range.const_like(loop.rows)).alu(Ops.ADD, out_index)
    fake_out = loop.out.replace(src=(loop.out.src[0].const_like(lanes),))
    fake_index = store.src[0].replace(src=(fake_out, linear_index))
    fake_store = store.replace(src=(fake_index, product, *store.src[2:]))
    try: mapped = _lower_composed_uops(_fp16_rewrite(list(UOp(Ops.SINK, src=(fake_store,)).toposort())), recipes_ready=True)
    except RuntimeError: mapped = None
    if mapped is not None and (finished:=_finish_mapped_add_reduction(mapped, loop.out.arg.slot, loop.rows, groups, 1.0)) is not None:
      return finished
  while len(terms) > 1:
    terms = [terms[i].alu(Ops.ADD, terms[i+1]) for i in range(0, len(terms)-1, 2)] + (terms[-1:] if len(terms) & 1 else [])
  return _lower_composed_uops([store.replace(src=(store.src[0], terms[0], *store.src[2:]))])

def _lower_scalar_loop_reduction(loop:RKLoopReduction) -> RKImage|None:
  """Turn a compact scalar register reduction into balanced FP16 DPU EW stages."""
  if loop.post_reciprocal or loop.post_cuberoot: return None
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
  accumulator = next((x for x in update.src if _local_load(x) is not None), None)
  term = next((x for x in update.src if x is not accumulator), None)
  if not negate_inputs and (term is None or _strip_cast(term).op is not Ops.LOAD): return None
  if negate_inputs: reduce_op = Ops.MAX
  elif update.op in (Ops.ADD, Ops.MUL, Ops.MAX) and reduce_ops in (set(), {update.op}): reduce_op = update.op
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
  def prepare(ops:list[RKEWOp], value:RKArg, slots:dict[float, int], _scratch:Callable[[], RKArg]) -> None:
    ops.append(RKEWOp(value, value, RKArg(RKBufferKind.SCRATCH, slots[-1.0]), rows, _EW_CFG[Ops.MUL]))
  return _reduction_image(out_param.arg.slot, rows, in_param.arg.slot, blocks, const_values,
                          _EW_CFG[reduce_op], fp32_out, post_scale, prepare if negate_inputs else None)

def _finish_mapped_add_reduction(mapped:RKImage, out_slot:int, rows:int, groups:int, post_scale:float,
                                 op_barriers:bool=False, compensated_limit:int=_reduction_stride(1)//2, kahan:bool=False) -> RKImage|None:
  """Retarget a vector map image into scratch, then append a row-wise ADD reduction."""
  if mapped.fill is not None or mapped.post_gathers or not mapped.ew_ops: return None
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
  gather_after = len(pre_ops)
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
  return RKImage(RKTarget.RK3588, scratch, struct.pack("<e", post_scale)+mapped.constants,
                 gathers=gathers, ew_ops=tuple(ops), mid_gathers=mid,
                 gather_after=mapped.gather_after if mapped_mid else gather_after,
                 host_gathers=host_gathers, host_scatters=host_scatters)

def _append_mapped_product_residual(mapped:RKImage, out_slot:int, out_index:UOp, product:UOp, lanes:int) -> RKImage|None:
  """Materialize one direct LOAD operand and retain the FP16 two-product residual of a mapped MUL vector."""
  if product.op is not Ops.MUL or product.arg is not None or mapped.fill is not None or mapped.post_gathers or mapped.host_scatters: return None
  direct, expression = next(((_strip_cast(load), other) for load,other in (product.src, product.src[::-1])
    if _strip_cast(load).op is Ops.LOAD), (None, None))
  if (direct is None or expression is None or direct.dtype.scalar() is not dtypes.half or not direct.src or
      direct.src[0].op is not Ops.INDEX or (param:=_root_param(direct.src[0])) is None or
      param.dtype.scalar() is not dtypes.half or param.src[0].op is not Ops.CONST or
      len(direct.src) > 1 and direct.src[1].op is not Ops.CONST): return None
  base = len(mapped.scratch)
  lhs, rhs, lhs_high, rhs_high, splitter = (RKArg(RKBufferKind.SCRATCH, base+i) for i in range(5))
  def remap(arg:RKArg) -> RKArg:
    return RKArg(RKBufferKind.SCRATCH, lhs.index, arg.addend) \
      if arg.kind is RKBufferKind.ARG and arg.index == out_slot else arg
  mapped = _map_image_args(mapped, remap)
  gate = direct.src[2] if len(direct.src) > 2 else None
  fill_bits = _fp16_bits(direct.src[1].arg if len(direct.src) > 1 else 0)
  try:
    rhs_gather = _gather_plan(param.arg.slot, rhs.index, out_index, direct.src[0].src[1], gate, lanes, fill_bits)
    _validate_gather_bounds(rhs_gather, int(param.src[0].arg))
  except RuntimeError: return None
  product_arg, error_arg = RKArg(RKBufferKind.ARG, out_slot), RKArg(RKBufferKind.ARG, out_slot, lanes*2)
  stages = ((product_arg, lhs, rhs, Ops.MUL),
            (lhs_high, lhs, splitter, Ops.MUL), (rhs_high, lhs_high, lhs, Ops.SUB),
            (lhs_high, lhs_high, rhs_high, Ops.SUB), (lhs, lhs, lhs_high, Ops.SUB),
            (rhs_high, rhs, splitter, Ops.MUL), (error_arg, rhs_high, rhs, Ops.SUB),
            (rhs_high, rhs_high, error_arg, Ops.SUB), (rhs, rhs, rhs_high, Ops.SUB),
            (error_arg, lhs_high, rhs_high, Ops.MUL), (error_arg, error_arg, product_arg, Ops.SUB),
            (lhs_high, lhs_high, rhs, Ops.MUL), (error_arg, error_arg, lhs_high, Ops.ADD),
            (rhs_high, lhs, rhs_high, Ops.MUL), (error_arg, error_arg, rhs_high, Ops.ADD),
            (lhs, lhs, rhs, Ops.MUL), (error_arg, error_arg, lhs, Ops.ADD))
  ops = mapped.ew_ops+tuple(
    RKEWOp(dst, left, right, lanes, _EW_CFG[op], submit_barrier=i == 0, stateful=True)
    for i,(dst,left,right,op) in enumerate(stages))
  return replace(mapped, scratch=mapped.scratch+tuple(RKScratch(_scratch_bytes(lanes)) for _ in range(5)),
    gathers=mapped.gathers+(rhs_gather,
      RKGather(out_slot, splitter.index, lanes, values=(_fp16_bits(65.0),)*lanes, dst_kind=RKBufferKind.SCRATCH)),
    ew_ops=ops)

def _append_inplace_image(first:RKImage, second:RKImage) -> RKImage|None:
  """Append an in-place EW image, scheduling its input materialization after the first image completes."""
  if first.post_gathers or second.fill is not None or not second.ew_ops or second.host_gathers or second.host_scatters: return None
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
    second_gather(gather, split+(gather.after if gather.after >= 0 else second.gather_after)) for gather in second.mid_gathers)
  scratch = (first.scratch[:first_constants] + second.scratch[:second_constants] + first.scratch[first_constants:] +
             second.scratch[second_constants:])
  return RKImage(RKTarget.RK3588, scratch, first.constants+second.constants,
                 gathers=tuple(first_gather(gather) for gather in first.gathers), ew_ops=first_ops+tuple(second_ops),
                 mid_gathers=tuple(first_gather(gather) for gather in first.mid_gathers)+second_mid,
                 gather_after=first.gather_after, post_gathers=tuple(second_gather(gather, -1) for gather in second.post_gathers))

def _lower_mapped_add_loop_reduction(uops:list[UOp]) -> RKImage|None:
  """Evaluate one fused FP16 map over the whole reduction domain, then reduce its materialized lanes."""
  def reject(_reason:str) -> RKImage|None: return None
  if (output:=_output_store(uops, dtypes.half, allow_local=True)) is None: return reject("output")
  store, out, rows, out_index, root = output
  nodes, out_ranges = list(root.toposort()), _index_ranges(out_index)
  reduce_ranges = [u for u in nodes if u.op is Ops.RANGE and u not in out_ranges]
  if not reduce_ranges or any(r.src[0].op is not Ops.CONST or int(r.src[0].arg) <= 0 for r in reduce_ranges): return reject("ranges")
  try: envs = _iter_range_env(out_ranges)
  except RuntimeError: return reject("envs")
  if len(envs) != rows or tuple(_eval_int(out_index, env) for env in envs) != tuple(range(rows)): return reject("index")
  updates = [u for u in nodes if u.op is Ops.STORE and _root_param(u.src[0]) is None and any(r in u.toposort() for r in reduce_ranges)]
  if len(updates) != 1: return reject(f"updates:{len(updates)}")
  value, post_root, post_local = root, None, None
  if _local_load(value) is not None: post_scale = 1.0
  elif value.op is Ops.MUL and (load:=next((x for x in value.src if _local_load(x) is not None), None)) is not None and \
       (scale:=value.src[1 if value.src[0] is load else 0]).op is Ops.CONST: post_scale = float(scale.arg)
  else:
    # Keep the reduction structural, then feed its materialized result through the ordinary UOp executor.  Select the
    # highest CAST boundary around the local accumulator so physical FP16 output is not reinterpreted as local FP32.
    local_refs = [u for u in root.toposort() if _local_load(u) is not None]
    if not local_refs: return reject("post_local")
    post_local = next((u for u in local_refs if u.dtype.scalar() is out.dtype.scalar()), local_refs[-1])
    post_root, post_scale = root, 1.0
  if not math.isfinite(post_scale): return reject("scale")
  update = _strip_cast(updates[0].src[1])
  if update.op is not Ops.ADD or (acc:=next((x for x in update.src if _local_load(x) is not None), None)) is None:
    return reject(f"update:{update.op.name}")
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
  if mapped is None: return reject("map")
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
  if post is None: return reject("post")
  post = _alias_image_args(post, {fake_slot:RKArg(RKBufferKind.ARG, out_slot)})
  appended = _append_inplace_image(reduced, post)
  return appended if appended is not None else reject("append")

def _lower_vectorized_unrolled_add_reduction(uops:list[UOp]) -> RKImage|None:
  """Execute repeated ADD terms once as vector UOps, then physically reduce their materialized lanes."""
  def reject(_reason:str) -> RKImage|None: return None
  if (output:=_output_store(uops, dtypes.half)) is None: return reject("output")
  store, out, count, out_index, root = output
  legacy_shape = root.op is Ops.CAST and root.src[0].op is Ops.MUL
  summed, post_scale = root, 1.0
  while True:
    if summed.op is Ops.CAST: summed = summed.src[0]; continue
    value, scale = next(((a, b) for a,b in (summed.src, summed.src[::-1]) if b.op is Ops.CONST), (None, None)) \
      if summed.op is Ops.MUL else (None, None)
    if value is None or scale is None: break
    post_scale *= float(scale.arg); summed = value
  if summed.op is not Ops.ADD or not math.isfinite(post_scale): return reject("root")
  terms = tuple(_strip_cast(term) for term in _flatten_binary(summed, Ops.ADD))
  if len(terms) < 2 or count*len(terms) > _MAX_STATIC_RANGE_ENVS: return reject(f"term_count:{count}:{len(terms)}")
  mapped_math = any(u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN) for u in terms[0].toposort())
  if count > 1 and not mapped_math: return reject(f"nonmath_rows:{count}")
  out_ranges = _index_ranges(out_index)
  try: out_envs = _iter_range_env(out_ranges)
  except RuntimeError: return reject("output_env")
  if len(out_envs) != count or tuple(_eval_int(out_index, env) for env in out_envs) != tuple(range(count)): return reject("output_index")
  vector_env = {r:np.fromiter((env[r] for env in out_envs), dtype=np.int64, count=count) for r in out_ranges}
  def input_leaf(u:UOp) -> tuple[UOp, UOp]|None:
    index = u if u.op is Ops.INDEX else u.src[0] if u.op is Ops.LOAD and u.src and u.src[0].op is Ops.INDEX else None
    param = _root_param(index) if index is not None else None
    return (index, param) if index is not None and param is not None and \
      param.dtype.scalar() in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool) else None
  input_slots = {parsed[1].arg.slot for term in terms for u in term.toposort()
                 for parsed in (input_leaf(u),) if parsed is not None}
  if not legacy_shape and len(input_slots) < 2: return reject(f"input_slots:{len(input_slots)}")
  supported_ops = (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.RECIPROCAL, Ops.NEG,
                   Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.AND, Ops.OR, Ops.XOR, Ops.WHERE,
                   Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.LOAD, Ops.INDEX, Ops.PARAM, Ops.CONST, Ops.CAST, Ops.RANGE)
  if any(u.op not in supported_ops for u in terms[0].toposort()):
    return reject("unsupported:"+",".join(sorted({u.op.name for u in terms[0].toposort() if u.op not in supported_ops})))
  signature_cache:dict[UOp, tuple] = {}
  def signature(u:UOp) -> tuple:
    if u in signature_cache: return signature_cache[u]
    if (leaf:=input_leaf(u)) is not None: return ("input", leaf[1].arg.slot, u.op, u.dtype.scalar())
    signature_cache[u] = ret = (u.op, u.dtype.scalar(), u.arg, tuple(signature(src) for src in u.src))
    return ret
  template, template_signature = terms[0], signature(terms[0])
  for term in terms[1:]:
    if signature(term) == template_signature: continue
    return reject(f"signature:{len(terms)}")
  loaded_indices = {u.src[0] for u in template.toposort() if u.op is Ops.LOAD and u.src and u.src[0].op is Ops.INDEX}
  leaves = [u for u in template.toposort() if input_leaf(u) is not None and u not in loaded_indices]
  if not leaves: return reject("leaves")
  counterparts:dict[UOp, list[UOp]] = {leaf:[] for leaf in leaves}
  def pair(lhs:UOp, rhs:UOp, found:dict[UOp, UOp]) -> bool:
    if lhs in counterparts:
      if lhs in found and found[lhs].key != rhs.key: return False
      found[lhs] = rhs; return True
    return all(pair(a, b, found) for a,b in zip(lhs.src, rhs.src))
  for term in terms:
    found:dict[UOp, UOp] = {}
    if not pair(template, term, found) or set(found) != set(leaves): return reject("pair")
    for leaf in leaves: counterparts[leaf].append(found[leaf])
  first = input_leaf(leaves[0]); assert first is not None
  first_index = first[0]
  axis = max((r.arg[0] for r in _index_ranges(out_index) if isinstance(r.arg, tuple)), default=-1)+1
  lane = UOp(Ops.RANGE, first_index.src[1].dtype, src=(first_index.src[1].const_like(len(terms)),), arg=(axis, AxisType.LOOP))
  substitutions:dict[UOp, UOp] = {}
  static_fallbacks:list[tuple[int, UOp, tuple[int, ...]]] = []
  next_fake_slot = 1+max((u.arg.slot for u in uops if u.op is Ops.PARAM), default=out.arg.slot)
  for leaf in leaves:
    paired = counterparts[leaf]
    parsed = [input_leaf(node) for node in paired]
    if any(item is None for item in parsed): return reject("leaf_parse")
    concrete = [item for item in parsed if item is not None]
    params = [item[1] for item in concrete]
    if len({param.key for param in params}) != 1: return reject("leaf_param")
    runtime_loads = [[node for node in item[0].src[1].toposort() if _runtime_index(node) is not None] for item in concrete]
    if any(runtime_loads): return reject("runtime_address_literal")
    try:
      eval_cache:dict[UOp, np.ndarray] = {}
      base_values = np.broadcast_to(_eval_vector(concrete[0][0].src[1], vector_env, eval_cache), count).astype(np.int64)
      offsets_list:list[int] = []
      absolute_offsets:list[int] = []
      for item in concrete:
        item_offsets = np.broadcast_to(_eval_vector(item[0].src[1], vector_env, eval_cache), count).astype(np.int64)
        absolute_offsets.extend(int(value) for value in item_offsets)
        delta = np.unique(item_offsets-base_values)
        if len(delta) != 1: raise RuntimeError("nonuniform repeated index")
        offsets_list.append(int(delta[0]))
      offsets = tuple(offsets_list)
    except (RuntimeError, ValueError): return reject("leaf_offset")
    run = next((i for i,value in enumerate(offsets) if value != offsets[0]), len(offsets))
    period = next((size for size in range(2, len(offsets)) if len(offsets)%size == 0 and
                   offsets == offsets[:size]*(len(offsets)//size)), len(offsets))
    if period < len(offsets):
      blocks, index_lane = offsets[:period], lane.alu(Ops.FLOORMOD, lane.const_like(period))
    elif 1 < run < len(offsets) and len(offsets)%run == 0 and all(len(set(offsets[i:i+run])) == 1 for i in range(0, len(offsets), run)):
      blocks, index_lane = offsets[::run], lane.alu(Ops.FLOORDIV, lane.const_like(run))
    else: blocks, index_lane = offsets, lane
    stride = blocks[1]-blocks[0]
    if blocks == tuple(blocks[0]+i*stride for i in range(len(blocks))):
      mapped_index = concrete[0][0].src[1].alu(Ops.ADD,
        index_lane.alu(Ops.MUL, lane.const_like(stride)).alu(Ops.ADD, lane.const_like(blocks[0])))
      source_param = params[0]
    else:
      if any(not 0 <= offset < int(params[0].src[0].arg) for offset in absolute_offsets): return reject("nonaffine_bounds")
      source_param = UOp.param(next_fake_slot, params[0].dtype, (count*len(terms),))
      static_fallbacks.append((next_fake_slot, params[0], tuple(absolute_offsets)))
      next_fake_slot += 1
      mapped_index = lane.alu(Ops.MUL, lane.const_like(count)).alu(Ops.ADD, out_index)
    index = concrete[0][0].replace(src=(source_param, mapped_index, *concrete[0][0].src[2:]))
    substitutions[leaf] = index if leaf.op is Ops.INDEX else leaf.replace(src=(index, *leaf.src[1:]))
  vector = template.substitute(substitutions)
  fake_out = out.replace(src=(out.src[0].const_like(count*len(terms)),))
  fake_index = store.src[0].replace(src=(fake_out, lane.alu(Ops.MUL, lane.const_like(count)).alu(Ops.ADD, out_index)))
  precise_product = count == 1 and 64 <= len(terms) <= 512 and abs(post_scale) >= 0.5 and vector.op is Ops.MUL and \
    any(_strip_cast(src).op is Ops.LOAD for src in vector.src)
  mapped_root = next((other for load,other in (vector.src, vector.src[::-1]) if _strip_cast(load).op is Ops.LOAD), vector) \
    if precise_product else vector
  def lower_map(root:UOp) -> RKImage|None:
    map_uops = list(UOp(Ops.SINK, src=(store.replace(src=(fake_index, root, *store.src[2:])),)).toposort())
    return _lower_uop_program(_fp16_rewrite(map_uops), vectorize_reductions=False, recipes_ready=True)
  mapped = lower_map(mapped_root)
  if mapped is None: return reject("mapped")
  mapped_groups = len(terms)
  if precise_product:
    if (residual:=_append_mapped_product_residual(mapped, out.arg.slot, fake_index.src[1], vector, count*len(terms))) is None:
      if (mapped:=lower_map(vector)) is None: return reject("mapped_product")
    else: mapped, mapped_groups = residual, len(terms)*2
  if static_fallbacks:
    fallback_slots = {fake_slot:len(mapped.scratch)+i for i,(fake_slot,_,_) in enumerate(static_fallbacks)}
    def remap_arg(arg:RKArg) -> RKArg:
      return RKArg(RKBufferKind.SCRATCH, fallback_slots[arg.index], arg.addend) \
        if arg.kind is RKBufferKind.ARG and arg.index in fallback_slots else arg
    fallback_gathers = tuple(RKGather(param.arg.slot, fallback_slots[fake_slot], len(offsets), offsets=offsets,
                                     itemsize=param.dtype.scalar().itemsize)
                             for fake_slot,param,offsets in static_fallbacks)
    mapped = _map_image_args(mapped, remap_arg)
    mapped = replace(mapped, scratch=mapped.scratch+tuple(
      RKScratch(max(64, len(offsets)*param.dtype.scalar().itemsize)) for _,param,offsets in static_fallbacks),
      gathers=fallback_gathers+mapped.gathers)
  finished = _finish_mapped_add_reduction(mapped, out.arg.slot, count, mapped_groups, post_scale,
                                           op_barriers=precise_product, kahan=precise_product)
  if finished is not None and count > 1 and mapped_math and finished.gather_after < len(finished.ew_ops):
    ops = list(finished.ew_ops)
    ops[finished.gather_after] = replace(ops[finished.gather_after], stateful=True)
    finished = replace(finished, ew_ops=tuple(ops))
  return finished if finished is not None else reject("finish")


def _lower_vectorized_mul_add_reduction(uops:list[UOp]) -> RKImage|None:
  """Execute repeated FP16 MUL UOps with product residuals, then compensate their physical ADD reduction."""
  if (output:=_output_store(uops, dtypes.half)) is None: return None
  store, out, rows, out_index, root = output
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
        plan = _gather_plan(param.arg.slot, 0, out_index, load.src[0].src[1], gate, rows, fill_bits)
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
  mapped = RKImage(RKTarget.RK3588,
    (RKScratch(_scratch_bytes(min(chunk_lanes, lanes))), *(RKScratch(_scratch_bytes(lanes)) for _ in range(4))),
    struct.pack("<e", 65.0), gathers=gathers, ew_ops=tuple(mapped_ops))
  finished = _finish_mapped_add_reduction(mapped, out.arg.slot, rows, groups*2, post_scale,
                                           op_barriers=True, compensated_limit=groups*2, kahan=groups == 8)
  if finished is None: return None
  if bias is not None:
    if bias.op is Ops.LOAD:
      bias_param = _root_param(bias.src[0]) if bias.src and bias.src[0].op is Ops.INDEX else None
      if bias_param is None or bias_param.src[0].op is not Ops.CONST: return None
      try: bias_offsets = _gather_offsets(out_index, bias.src[0].src[1], bias.src[2] if len(bias.src) > 2 else None, rows)
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
    relu_image = RKImage(RKTarget.RK3588, (RKScratch(_scratch_bytes(rows)),), struct.pack("<e", 0.0), ew_ops=(
      RKEWOp(RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.ARG, out.arg.slot), RKArg(RKBufferKind.SCRATCH, 0),
             rows, _EW_CFG[Ops.MAX]),))
    if (finished:=_append_inplace_image(finished, relu_image)) is None: return None
  return finished

def _iter_binary(root:UOp, op:Ops, dtype:DType|None=None) -> Iterable[UOp]:
  stack = [root]
  while stack:
    node = stack.pop()
    if node.op is op and (dtype is None or node.dtype.scalar() is dtype): stack.extend(reversed(node.src))
    else: yield node

def _flatten_binary(root:UOp, op:Ops) -> list[UOp]: return list(_iter_binary(root, op))

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64
  return vector_bytes, vector_bytes//2, rows*vector_bytes//2

def _stripe_gathers(src_slot:int, dst_slot:int, count:int, rows:Iterable[Iterable[int]], vector_lanes:int, *,
                    values:bool=False, itemsize:int=2) -> tuple[RKGather, ...]:
  """Pack candidate or repeated-current rows into one aligned lane matrix."""
  return tuple(RKGather(src_slot, dst_slot, count, offsets=() if values else tuple(row), values=tuple(row) if values else (),
                        dst_addend=i*vector_lanes, itemsize=itemsize) for i,row in enumerate(rows))

def _scratch_arg(slot:int, addend:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slot, addend)

def _physical_lists(minimum:int=0) -> tuple[list[int], Callable[[int], int], list[RKGather], list[RKEWOp]]:
  scratch_sizes:list[int] = []; gathers:list[RKGather] = []; ops:list[RKEWOp] = []
  def scratch(size:int) -> int: scratch_sizes.append(max(minimum, size)); return len(scratch_sizes)-1
  return scratch_sizes, scratch, gathers, ops

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
                lanes:int, barriers:tuple[bool, bool]=(False, True)) -> None:
  """Append SUB, ABS, nonzero comparison, and inversion for an FP16 equality mask."""
  diff, magnitude, unequal, equal = temps
  ops.extend((RKEWOp(arg(diff), arg(lhs), arg(rhs), lanes, _EW_CFG[Ops.SUB], submit_barrier=barriers[0], stateful=barriers[0]),
              RKEWOp(arg(magnitude), arg(diff), arg(diff), lanes, _EW_CFG_ABS, submit_barrier=barriers[1], stateful=barriers[1]),
              RKEWOp(arg(unequal), arg(magnitude), arg(magnitude), lanes, _EW_CFG[Ops.MAX], compare=True),
              RKEWOp(arg(equal), arg(one), arg(unequal), lanes, _EW_CFG[Ops.SUB], stateful=True)))


def _ew_native_int16_eq_mask(ops:list[RKEWOp], allocate:Callable[[], RKArg], lhs:RKArg, rhs:RKArg,
                             one:RKArg, lanes:int) -> RKArg:
  """Compare native INT16 lanes whose subtraction is proven not to overflow."""
  diff, magnitude, unequal, equal = (allocate() for _ in range(4))
  integer = dict(int16_input=True, int16_output=True)
  ops.extend((RKEWOp(diff, lhs, rhs, lanes, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(magnitude, diff, diff, lanes, _EW_CFG_ABS, **integer),
              RKEWOp(unequal, magnitude, one, lanes, _EW_CFG_MIN, **integer),
              RKEWOp(equal, one, unequal, lanes, _EW_CFG[Ops.SUB], **integer)))
  return equal

def _fp16_high_and_nan(ops:list[RKEWOp], allocate:Callable[[], RKArg], high:RKArg, low:RKArg,
                       zero:RKArg, one:RKArg, const123:RKArg, const124:RKArg, const127:RKArg, const128:RKArg,
                       lanes:int) -> tuple[RKArg, RKArg]:
  """Canonicalize signed zero's FP16 high byte and classify NaNs with native INT16 byte arithmetic."""
  integer = dict(int16_input=True, int16_output=True)
  sign_delta, sign_positive, sign, sign_scale, magnitude = (allocate() for _ in range(5))
  ops.extend((RKEWOp(sign_delta, high, const127, lanes, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(sign_positive, sign_delta, zero, lanes, _EW_CFG[Ops.MAX], **integer),
              RKEWOp(sign, sign_positive, one, lanes, _EW_CFG_MIN, **integer),
              RKEWOp(sign_scale, sign, const128, lanes, _EW_CFG[Ops.MUL], **integer),
              RKEWOp(magnitude, high, sign_scale, lanes, _EW_CFG[Ops.SUB], **integer)))
  high_zero = _ew_native_int16_eq_mask(ops, allocate, magnitude, zero, one, lanes)
  low_zero = _ew_native_int16_eq_mask(ops, allocate, low, zero, one, lanes)
  zero_value, zero_sign, canonical = (allocate() for _ in range(3))
  exponent_delta, exponent_positive, exponent_all = (allocate() for _ in range(3))
  mantissa_delta, mantissa_positive, mantissa_high, mantissa_low, mantissa, nan = (allocate() for _ in range(6))
  ops.extend((RKEWOp(zero_value, high_zero, low_zero, lanes, _EW_CFG[Ops.MUL], **integer),
              RKEWOp(zero_sign, sign_scale, zero_value, lanes, _EW_CFG[Ops.MUL], **integer),
              RKEWOp(canonical, high, zero_sign, lanes, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(exponent_delta, magnitude, const123, lanes, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(exponent_positive, exponent_delta, zero, lanes, _EW_CFG[Ops.MAX], **integer),
              RKEWOp(exponent_all, exponent_positive, one, lanes, _EW_CFG_MIN, **integer),
              RKEWOp(mantissa_delta, magnitude, const124, lanes, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(mantissa_positive, mantissa_delta, zero, lanes, _EW_CFG[Ops.MAX], **integer),
              RKEWOp(mantissa_high, mantissa_positive, one, lanes, _EW_CFG_MIN, **integer),
              RKEWOp(mantissa_low, low, one, lanes, _EW_CFG_MIN, **integer),
              RKEWOp(mantissa, mantissa_high, mantissa_low, lanes, _EW_CFG[Ops.MAX], **integer),
              RKEWOp(nan, exponent_all, mantissa, lanes, _EW_CFG[Ops.MUL], **integer)))
  return canonical, nan


RKIndexEquality = tuple[int, int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
RKCoordinateRows = tuple[tuple[int, ...], ...]

def _reduce_arena(ops:list[RKEWOp], active:list[int], count:int, cfg:int, arena:Callable[[int], RKArg],
                  out:RKArg|None=None, fp32_out:bool=False, int16:bool=False, level_barriers:bool=False,
                  op_barriers:bool=False) -> RKArg:
  """Append a balanced in-place arena reduction and optionally write its final stage directly to output."""
  while len(active) > 1:
    reduced, first = [], True
    for i in range(0, len(active)-1, 2):
      lhs, rhs, final = active[i], active[i+1], len(active) == 2 and out is not None
      dst = out if final and out is not None else arena(lhs)
      ops.append(RKEWOp(dst, arena(lhs), arena(rhs), count,
                        cfg | (_EW_STAGE_FP32_OUT if fp32_out and final else 0),
                        int16_input=int16, int16_output=int16,
                        submit_barrier=(op_barriers or level_barriers and first) and bool(ops),
                        stateful=op_barriers or level_barriers and first))
      first = False
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

def _append_dpu_sqrt_ops(ops:list[RKEWOp], source:RKArg, out:RKArg, count:int, slots:dict[float, int],
                         scratch:Callable[[], RKArg]) -> None:
  """Append a nonnegative Babylonian sqrt after variance reduction."""
  def const(value:float) -> RKArg: return RKArg(RKBufferKind.SCRATCH, slots[value])
  zero, one, maximum, minimum, half = (const(value) for value in (0.0, 1.0, 65504.0, 2**-24, 0.5))
  first = True
  def emit(dst:RKArg, lhs:RKArg, rhs:RKArg, cfg:int) -> None:
    nonlocal first
    ops.append(RKEWOp(dst, lhs, rhs, count, cfg, submit_barrier=first, stateful=True)); first = False
  lower = scratch(); emit(lower, source, minimum, _EW_CFG[Ops.MAX])
  neg_lower, neg_maximum, neg_max = scratch(), scratch(), scratch()
  emit(neg_lower, zero, lower, _EW_CFG[Ops.SUB]); emit(neg_maximum, zero, maximum, _EW_CFG[Ops.SUB])
  emit(neg_max, neg_lower, neg_maximum, _EW_CFG[Ops.MAX]); emit(lower, zero, neg_max, _EW_CFG[Ops.SUB])
  estimate, quotient, summed = scratch(), scratch(), scratch()
  emit(estimate, lower, one, _EW_CFG[Ops.MAX])
  for _ in range(14):
    emit(quotient, lower, estimate, _EW_CFG[Ops.FDIV]); emit(summed, estimate, quotient, _EW_CFG[Ops.ADD])
    emit(estimate, summed, half, _EW_CFG[Ops.MUL])
  emit(out, source, estimate, _EW_CFG[Ops.FDIV])

def _append_dpu_cuberoot_ops(ops:list[RKEWOp], source:RKArg, out:RKArg, count:int, slots:dict[float, int],
                             scratch:Callable[[], RKArg]) -> None:
  """Append a nonnegative FP16 cube root and finish as x/estimate² so zero stays exact."""
  one, third = (RKArg(RKBufferKind.SCRATCH, slots[value]) for value in (1.0, 1/3))
  estimate, square, quotient, doubled, summed = (scratch() for _ in range(5))
  ops.append(RKEWOp(estimate, source, one, count, _EW_CFG[Ops.MAX], submit_barrier=True, stateful=True))
  for _ in range(10):
    ops.extend((RKEWOp(square, estimate, estimate, count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
                RKEWOp(quotient, source, square, count, _EW_CFG[Ops.FDIV], stateful=True),
                RKEWOp(doubled, estimate, estimate, count, _EW_CFG[Ops.ADD], stateful=True),
                RKEWOp(summed, doubled, quotient, count, _EW_CFG[Ops.ADD], stateful=True),
                RKEWOp(estimate, summed, third, count, _EW_CFG[Ops.MUL], stateful=True)))
  ops.extend((RKEWOp(square, estimate, estimate, count, _EW_CFG[Ops.MUL], submit_barrier=True, stateful=True),
              RKEWOp(out, source, square, count, _EW_CFG[Ops.FDIV], stateful=True)))

def _reduction_image(out_slot:int, rows:int, source_slot:int, blocks:list[tuple[int, ...]]|tuple[tuple[int, ...], ...],
                     constants:tuple[float, ...], cfg:int, fp32_out:bool, post_scale:float,
                     prepare:Callable[[list[RKEWOp], RKArg, dict[float, int], Callable[[], RKArg]], None]|None=None,
                     post_sqrt:bool=False, post_reciprocal:bool=False, post_cuberoot:bool=False,
                     prepare_whole:bool=False, fill_bits:int=0) -> RKImage:
  """Materialize row blocks, apply an optional lane transform, reduce them, and write the typed result."""
  if post_sqrt: constants = tuple(dict.fromkeys((*constants, 0.0, 1.0, 65504.0, 2**-24, 0.5)))
  elif post_reciprocal: constants = tuple(dict.fromkeys((*constants, 1.0)))
  elif post_cuberoot: constants = tuple(dict.fromkeys((*constants, 1.0, 1/3)))
  const_slots, data_slot = {value:i for i,value in enumerate(constants)}, len(constants)
  stride = _reduction_stride(rows)
  gathers = _spaced_reduction_gathers(source_slot, data_slot, rows, blocks, stride, fill_bits)
  def arena(offset:int=0) -> RKArg: return RKArg(RKBufferKind.SCRATCH, data_slot, offset)
  extra:list[RKScratch] = []
  def scratch() -> RKArg:
    extra.append(RKScratch(len(blocks)*stride if prepare_whole else rows*2))
    return RKArg(RKBufferKind.SCRATCH, data_slot+len(extra))
  ops:list[RKEWOp] = []
  active = [i*stride for i in range(len(blocks))]
  if prepare is not None:
    if prepare_whole: prepare(ops, arena(), const_slots, scratch)
    else:
      for offset in active: prepare(ops, arena(offset), const_slots, scratch)
  out = RKArg(RKBufferKind.ARG, out_slot)
  reduced = _reduce_arena(ops, active, rows, cfg, arena,
                          out if post_scale == 1.0 and not post_sqrt and not post_reciprocal and not post_cuberoot else None,
                          fp32_out, level_barriers=post_reciprocal or post_cuberoot)
  if post_scale != 1.0:
    scaled = scratch() if post_sqrt else out
    scale_value, scale_cfg = (0.0, _EW_CFG[Ops.FDIV]) if math.isinf(post_scale) else (post_scale, _EW_CFG[Ops.MUL])
    ops.append(RKEWOp(scaled, reduced, RKArg(RKBufferKind.SCRATCH, const_slots[scale_value]), rows,
                      scale_cfg | (_EW_STAGE_FP32_OUT if fp32_out else 0)))
    reduced = scaled
  if post_sqrt: _append_dpu_sqrt_ops(ops, reduced, out, rows, const_slots, scratch)
  elif post_reciprocal:
    ops.append(RKEWOp(out, RKArg(RKBufferKind.SCRATCH, const_slots[1.0]), reduced, rows, _EW_CFG[Ops.FDIV],
                      submit_barrier=True, stateful=True))
  elif post_cuberoot: _append_dpu_cuberoot_ops(ops, reduced, out, rows, const_slots, scratch)
  scratch_buffers = tuple(RKScratch(rows*2) for _ in constants) + (RKScratch(len(blocks)*stride), *extra)
  return RKImage(RKTarget.RK3588, scratch_buffers, b"".join(struct.pack("<e", value) for value in constants),
                 gathers=gathers, ew_ops=tuple(ops))

def _lower_fp16_int32_cast(output:RKOutput) -> RKImage|None:
  """Truncate a direct FP16 load on DPU before the terminal INT32 conversion."""
  root = output[4]
  if (root.op is not Ops.CAST or root.dtype.scalar() is not dtypes.int or len(root.src) != 1 or
      root.src[0].op is not Ops.LOAD or root.src[0].dtype.scalar() is not dtypes.half): return None
  return _typed_half_image(output, _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=root.src)), True)

def _lower_fp16_uint8_cast(output:RKOutput) -> RKImage|None:
  """Truncate FP16 modulo 256 on DPU, convert to INT16, then expose each low byte."""
  root = output[4]
  zero = UOp.const(0.0, dtypes.half)
  if root.op is Ops.CAST and root.dtype.scalar() is dtypes.uchar and len(root.src) == 1 and \
     root.src[0].dtype.scalar() is dtypes.half: source = root.src[0]
  elif (root.op is Ops.WHERE and root.dtype.scalar() is dtypes.uchar and len(root.src) == 3 and
        root.src[0].op is Ops.CMPLT and root.src[0].src[0].op is Ops.CONST and float(root.src[0].src[0].arg) == 0.0 and
        root.src[1].op is Ops.CAST and root.src[1].dtype.scalar() is dtypes.uchar and len(root.src[1].src) == 1 and
        root.src[1].src[0].dtype.scalar() is dtypes.half and root.src[0].src[1].key == root.src[1].src[0].key and
        root.src[2].op is Ops.CONST and int(root.src[2].arg) == 0):
    source = root.src[1].src[0].alu(Ops.MAX, zero)
  else: return None
  if (relu:=_relu_operand(source)) is not None: source = relu.alu(Ops.MAX, zero)
  truncated = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(source,)))
  quotient = _native_floor(truncated.alu(Ops.MUL, UOp.const(1.0/256.0, dtypes.half)))
  remainder = truncated.alu(Ops.SUB, quotient.alu(Ops.MUL, UOp.const(256.0, dtypes.half)))
  return _typed_half_image(output, remainder, False)

def _lower_integer_fp32_cast(output:RKOutput) -> RKImage|None:
  """Compose the DPU INT32-to-FP16 and FP16-to-FP32 converters for integer and boolean inputs."""
  _, out_param, count, out_index, root = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  if (root.op is not Ops.CAST or root.dtype.scalar() is not dtypes.float or len(root.src) != 1 or
      (load:=root.src[0]).op is not Ops.LOAD or load.dtype.scalar() not in (dtypes.int, dtypes.bool) or
      len(load.src) != 1 or load.src[0].op is not Ops.INDEX or
      (source:=_root_param(load.src[0])) is None or source.src[0].op is not Ops.CONST): return None
  try: offsets = _gather_offsets(out_index, load.src[0].src[1], None, count)
  except RuntimeError: return None
  if any(not 0 <= offset < int(source.src[0].arg) for offset in offsets): return None
  fp16_slot, tiles_slot = 0, 1
  groups = tuple(range(0, count, _EW_ELEMS_32BIT))
  scratch_sizes = [max(64, len(groups)*16), _int32_tiles_bytes(_EW_ELEMS_32BIT)]
  gathers:tuple[RKGather, ...] = ()
  input_arg = RKArg(RKBufferKind.ARG, source.arg.slot)
  if load.dtype.scalar() is dtypes.bool or offsets != tuple(range(count)):
    raw_slot = len(scratch_sizes)
    scratch_sizes.append(max(64, count*4))
    gathers = (RKGather(source.arg.slot, raw_slot, count, offsets=offsets,
                        dst_stride=4 if load.dtype.scalar() is dtypes.bool else 1,
                        itemsize=1 if load.dtype.scalar() is dtypes.bool else 4),)
    input_arg = RKArg(RKBufferKind.SCRATCH, raw_slot)
  ops:list[RKEWOp] = []
  for group,start in enumerate(groups):
    lanes = min(_EW_ELEMS_32BIT, count-start)
    ops.append(RKEWOp(RKArg(RKBufferKind.SCRATCH, fp16_slot, group*16), replace(input_arg, addend=start*4),
                      RKArg(RKBufferKind.SCRATCH, tiles_slot), lanes, _EW_CFG[Ops.MAX], int32_input=True))
  for group,start in enumerate(groups):
    lanes = min(_EW_ELEMS_32BIT, count-start)
    value = RKArg(RKBufferKind.SCRATCH, fp16_slot, group*16)
    ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot, group*16), value, value, lanes,
                        _EW_CFG[Ops.MAX] | _EW_STAGE_FP32_OUT))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=gathers, ew_ops=tuple(ops))

def _int16_byte_sum(ops:list[RKEWOp], gathers:list[RKGather], scratch:Callable[[int], int], source_slot:int,
                    operands:tuple[tuple[RKArg, ...], ...], count:int) -> tuple[RKArg, ...]:
  """Add four-byte operands exactly modulo 2**32 with native INT16 byte carries."""
  if len(operands) < 2 or any(len(operand) != 4 for operand in operands): raise ValueError("invalid byte sum")
  zero, one, byte_base = (scratch(count*2) for _ in range(3))
  thresholds = tuple(scratch(count*2) for _ in range(len(operands)-1))
  gathers.extend((RKGather(source_slot, zero, count, values=(0,)*count),
                  RKGather(source_slot, one, count, values=(1,)*count),
                  RKGather(source_slot, byte_base, count, values=(256,)*count)))
  gathers.extend(RKGather(source_slot, slot, count, values=(256*(level+1)-1,)*count)
                 for level,slot in enumerate(thresholds))
  carry, results = _scratch_arg(zero), []
  int16 = dict(int16_input=True, int16_output=True)
  for byte in range(4):
    total = _reduce_rows(ops, [operand[byte] for operand in operands], count, _EW_CFG[Ops.ADD], int16=True)
    if byte:
      slot = scratch(count*2); ops.append(RKEWOp(_scratch_arg(slot), total, carry, count, _EW_CFG[Ops.ADD], **int16)); total = _scratch_arg(slot)
    bits:list[RKArg] = []
    for threshold in thresholds:
      delta, positive, bit = (scratch(count*2) for _ in range(3))
      ops.extend((RKEWOp(_scratch_arg(delta), total, _scratch_arg(threshold), count, _EW_CFG[Ops.SUB], **int16),
                  RKEWOp(_scratch_arg(positive), _scratch_arg(delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX], **int16),
                  RKEWOp(_scratch_arg(bit), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN, **int16)))
      bits.append(_scratch_arg(bit))
    carry = _reduce_rows(ops, bits, count, _EW_CFG[Ops.ADD], int16=True)
    scaled, result = scratch(count*2), scratch(count*2)
    ops.extend((RKEWOp(_scratch_arg(scaled), carry, _scratch_arg(byte_base), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(_scratch_arg(result), total, _scratch_arg(scaled), count, _EW_CFG[Ops.SUB], **int16)))
    results.append(_scratch_arg(result))
  return tuple(results)

def _int16_byte_bits(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     value:RKArg, lanes:int) -> tuple[RKArg, ...]:
  """Split unsigned byte lanes into eight exact native INT16 0/1 planes."""
  integer = dict(int16_input=True, int16_output=True)
  result:list[RKArg|None] = [None]*8
  remainder = value
  for bit in range(7, 0, -1):
    delta, positive, flag, scaled, next_remainder = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, remainder, constants[(1<<bit)-1], lanes, _EW_CFG[Ops.SUB], **integer),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **integer),
                RKEWOp(flag, positive, constants[1], lanes, _EW_CFG_MIN, **integer),
                RKEWOp(scaled, flag, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **integer),
                RKEWOp(next_remainder, remainder, scaled, lanes, _EW_CFG[Ops.SUB], **integer)))
    result[bit], remainder = flag, next_remainder
  result[0] = remainder
  return typing_cast(tuple[RKArg, ...], tuple(result))

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

def _greater_half_load(predicate:UOp) -> tuple[UOp, float]|None:
  """Recognize a scalar-threshold forward predicate `constant < fp16_load`."""
  predicate = _unwrap_condition(predicate)
  if predicate.op is not Ops.CMPLT or len(predicate.src) != 2: return None
  threshold, load = predicate.src
  if (threshold.op is not Ops.CONST or threshold.dtype.scalar() is not dtypes.half or
      load.op is not Ops.LOAD or load.dtype.scalar() is not dtypes.half or not load.src or load.src[0].op is not Ops.INDEX): return None
  return load, float(threshold.arg)

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
      ops.append(RKEWOp(_scratch_arg(shifted), _scratch_arg(values), _scratch_arg(threshold_slot), matrix_lanes, _EW_CFG[Ops.SUB]))
      mask = _ew_ieee_positive_mask(ops, _scratch_arg, shifted, (one, maximum, negative_maximum), temps, matrix_lanes)
    else:
      magnitude, mask_slot = temps[:2]
      ops.extend((RKEWOp(_scratch_arg(magnitude), _scratch_arg(values), _scratch_arg(values), matrix_lanes, _EW_CFG_ABS),
                  RKEWOp(_scratch_arg(mask_slot), _scratch_arg(magnitude), _scratch_arg(magnitude), matrix_lanes, _EW_CFG[Ops.MAX], compare=True)))
      mask = _scratch_arg(mask_slot)
    reduced = _reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(len(rows))],
                           block_count, _EW_CFG[Ops.ADD])
    reduced_blocks.append((reduced, start, block_count))
  compact = scratch(count)
  scratch_sizes.append(_int32_tiles_bytes(count)); int_tiles = len(scratch_sizes)-1
  gather_after = len(ops)
  mid = tuple(RKGather(reduced.index, compact, block_count,
                       offsets=tuple(reduced.addend//2+lane for lane in range(block_count)), dst_addend=start,
                       src_kind=RKBufferKind.SCRATCH) for reduced,start,block_count in reduced_blocks)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), _scratch_arg(compact), _scratch_arg(int_tiles), count,
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
  scratch_sizes, scratch, gathers, ops = _physical_lists(64)
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
        byte_args.append(_scratch_arg(slot))
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
      ops.extend((RKEWOp(_scratch_arg(diff), value, _scratch_arg(expected), matrix_lanes, _EW_CFG[Ops.SUB], **int16),
                  RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), matrix_lanes, _EW_CFG_ABS, **int16),
                  RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, **int16),
                  RKEWOp(_scratch_arg(equal), _scratch_arg(one), _scratch_arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], **int16)))
      masks.append(_scratch_arg(equal))
    mask = masks[0]
    for byte_mask in masks[1:]:
      slot = scratch(matrix_lanes*2)
      ops.append(RKEWOp(_scratch_arg(slot), mask, byte_mask, matrix_lanes, _EW_CFG[Ops.MUL], **int16)); mask = _scratch_arg(slot)
    partials.append(_reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(block_count)],
                                 count, _EW_CFG[Ops.ADD], int16=True))
    if start and len(ops) > op_start: ops[op_start] = replace(ops[op_start], submit_barrier=True)
  result = _reduce_rows(ops, partials, count, _EW_CFG[Ops.ADD], int16=True)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, result, count, _EW_CFG[Ops.MAX],
                    int16_input=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops))

def _lower_loop_int32_equality_add(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Vectorize a local ADD loop of exact INT32 equality predicates with native byte arithmetic."""
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
  ops:list[RKEWOp] = [RKEWOp(_scratch_arg(compact), RKArg(RKBufferKind.ARG, source.arg.slot), _scratch_arg(convert_tiles), count,
                              _EW_CFG[Ops.MAX], int32_input=True)]
  reduced = _reduce_rows(ops, [_scratch_arg(matrix, row*vector_bytes) for row in range(count)], count, _EW_CFG[Ops.ADD])
  result = reduced
  if normalized is not None:
    zero, one, extent, negative_delta, positive, negative, correction, normalized_value = range(len(scratch), len(scratch)+8)
    scratch.extend(RKScratch(_scratch_bytes(count)) for _ in range(8))
    zero_bits, one_bits, extent_bits = (_fp16_bits(value) for value in (0.0, 1.0, normalized[1]))
    gathers.extend((RKGather(source.arg.slot, zero, count, values=(zero_bits,)*count),
                    RKGather(source.arg.slot, one, count, values=(one_bits,)*count),
                    RKGather(source.arg.slot, extent, count, values=(extent_bits,)*count)))
    ops.extend((RKEWOp(_scratch_arg(negative_delta), _scratch_arg(zero), reduced, count, _EW_CFG[Ops.SUB]),
                RKEWOp(_scratch_arg(positive), _scratch_arg(negative_delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX]),
                RKEWOp(_scratch_arg(negative), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN),
                RKEWOp(_scratch_arg(correction), _scratch_arg(negative), _scratch_arg(extent), count, _EW_CFG[Ops.MUL], stateful=True),
                RKEWOp(_scratch_arg(normalized_value), reduced, _scratch_arg(correction), count, _EW_CFG[Ops.ADD])))
    result = _scratch_arg(normalized_value)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), result, _scratch_arg(int_tiles), count,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  return RKImage(RKTarget.RK3588, tuple(scratch), gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=mid, gather_after=1)

def _lower_loop_int32_prefix_add(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Vectorize a local-register INT32 prefix ADD loop into physical reduction stages."""
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

RKDynamicEquality = tuple[int, int, tuple[tuple[int, ...], ...], tuple[int, ...]]
def _int32_less_mask(ops:list[RKEWOp], allocate:Callable[[], RKArg], constants:dict[int, RKArg],
                     lhs_components:list[RKArg], rhs_components:list[RKArg], lanes:int) -> RKArg:
  """Compare signed INT32 lanes represented as high-to-low widened bytes."""
  integer = dict(int16_input=True, int16_output=True)
  def biased_sign(value:RKArg) -> RKArg:
    delta, positive, high, scaled, biased = (allocate() for _ in range(5))
    ops.extend((RKEWOp(delta, value, constants[127], lanes, _EW_CFG[Ops.SUB], **integer),
                RKEWOp(positive, delta, constants[0], lanes, _EW_CFG[Ops.MAX], **integer),
                RKEWOp(high, positive, constants[1], lanes, _EW_CFG_MIN, **integer),
                RKEWOp(scaled, high, constants[256], lanes, _EW_CFG[Ops.MUL], **integer),
                RKEWOp(biased, value, constants[128], lanes, _EW_CFG[Ops.ADD], **integer),
                RKEWOp(biased, biased, scaled, lanes, _EW_CFG[Ops.SUB], **integer)))
    return biased
  lhs_components[0], rhs_components[0] = biased_sign(lhs_components[0]), biased_sign(rhs_components[0])
  return _ordered_byte_less(ops, allocate, constants[0], constants[1], lhs_components, rhs_components, lanes)

def _ordered_byte_less(ops:list[RKEWOp], allocate:Callable[[], RKArg], zero:RKArg, one:RKArg,
                       lhs_components:Iterable[RKArg], rhs_components:Iterable[RKArg], lanes:int) -> RKArg:
  integer = dict(int16_input=True, int16_output=True)
  less, equal = zero, one
  for lhs,rhs in zip(lhs_components, rhs_components):
    maximum, lhs_delta, rhs_delta, lhs_less, rhs_less, unequal, same, selected, next_less, next_equal = (allocate() for _ in range(10))
    ops.extend((RKEWOp(maximum, lhs, rhs, lanes, _EW_CFG[Ops.MAX], **integer),
                RKEWOp(lhs_delta, maximum, lhs, lanes, _EW_CFG[Ops.SUB], **integer),
                RKEWOp(rhs_delta, maximum, rhs, lanes, _EW_CFG[Ops.SUB], **integer),
                RKEWOp(lhs_less, lhs_delta, one, lanes, _EW_CFG_MIN, **integer),
                RKEWOp(rhs_less, rhs_delta, one, lanes, _EW_CFG_MIN, **integer),
                RKEWOp(unequal, lhs_less, rhs_less, lanes, _EW_CFG[Ops.MAX], **integer),
                RKEWOp(same, one, unequal, lanes, _EW_CFG[Ops.SUB], **integer),
                RKEWOp(selected, equal, lhs_less, lanes, _EW_CFG[Ops.MUL], **integer),
                RKEWOp(next_less, less, selected, lanes, _EW_CFG[Ops.MAX], **integer),
                RKEWOp(next_equal, equal, same, lanes, _EW_CFG[Ops.MUL], **integer)))
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
  scratch_sizes, scratch, gathers, ops = _physical_lists()
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
  post_gathers = tuple(RKGather(value.index, out_slot, group_count, offsets=tuple(value.addend+offset for offset in byte_offsets),
    dst_stride=repeat*itemsize, dst_addend=channel*itemsize+byte,
    dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
    for channel,pair in enumerate(results) for byte,value in enumerate(pair))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers),
                 ew_ops=tuple(ops), post_gathers=post_gathers)

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
  integer = dict(int16_input=True, int16_output=True)
  masks:list[RKArg] = []
  for coordinates in coordinate_sets:
    byte_masks:list[RKArg] = []
    for byte in range(4):
      dynamic, static, equal = (scratch(matrix_lanes*2) for _ in range(3))
      gathers.extend(RKGather(index_slot, dynamic, count, offsets=tuple(offset*4+byte for offset in offsets),
        dst_stride=2, dst_addend=row*vector_lanes*2, itemsize=1) for row,offsets in enumerate(offset_rows))
      values = tuple((value >> (byte*8)) & 0xff for row in coordinates for value in (*row, *((0,)*(vector_lanes-count))))
      gathers.append(RKGather(index_slot, static, matrix_lanes, values=values, itemsize=2))
      ops.extend((RKEWOp(_scratch_arg(diff), _scratch_arg(dynamic), _scratch_arg(static), matrix_lanes, _EW_CFG[Ops.SUB], **integer),
                  RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), matrix_lanes, _EW_CFG_ABS, **integer),
                  RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(one), matrix_lanes, _EW_CFG_MIN, **integer),
                  RKEWOp(_scratch_arg(equal), _scratch_arg(one), _scratch_arg(unequal), matrix_lanes, _EW_CFG[Ops.SUB], **integer)))
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
  scratch_sizes, scratch, gathers, ops = _physical_lists()
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
  post = (_int16_low_bytes(result, out_param.arg.slot, count),)
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
  ops = [RKEWOp(_scratch_arg(shifted), RKArg(RKBufferKind.ARG, source.arg.slot), _scratch_arg(threshold_slot), source_count, _EW_CFG[Ops.SUB])]
  mask = _ew_ieee_positive_mask(ops, _scratch_arg, shifted, (one, maximum, negative_maximum), temps, source_count)
  gather_after = len(ops)
  mid = (RKGather(mask.index, spaced, source_count, offsets=tuple(mask.addend//2+lane for lane in range(source_count)),
                  dst_stride=32, src_kind=RKBufferKind.SCRATCH),)
  total = _reduce_rows(ops, [_scratch_arg(spaced, lane*64) for lane in range(source_count)], 1, _EW_CFG[Ops.ADD])
  if scale != 1: ops.append(RKEWOp(total, total, _scratch_arg(scale_slot), 1, _EW_CFG[Ops.MUL]))
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), total, _scratch_arg(int_tiles), 1,
                    _EW_CFG[Ops.MAX], stateful=True, int32_output=True))
  constants = struct.pack("<eeeee", 1.0, 65504.0, -65504.0, threshold, scale)
  return RKImage(RKTarget.RK3588, scratch, constants, ew_ops=tuple(ops), mid_gathers=mid, gather_after=gather_after)

def _lower_loop_fp16_predicate_total(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Normalize a scalar local-register predicate reduction into the verified unrolled total emitter."""
  store, out_param, count, out_index, _ = output
  if count != 1 or (value:=_unrolled_local_add(uops, out_index)) is None: return None
  return _lower_unrolled_fp16_predicate_total((store, out_param, count, out_index, value))

def _lower_loop_fp16_prefix_count(uops:list[UOp], output:RKOutput) -> RKImage|None:
  """Normalize a local-register FP16 predicate scan into the blocked prefix emitter."""
  store, out_param, count, out_index, _ = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or (value:=_unrolled_local_add(uops, out_index)) is None: return None
  return _lower_unrolled_fp16_prefix_count((store, out_param, count, out_index, value))

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

def _lower_bounded_integer_predicate_coordinates(output:RKOutput, dtype:DType=dtypes.int) -> RKImage|None:
  """Execute bounded integer predicate coordinates through native INT16 byte masks."""
  _, out_param, count, _, _ = output
  if (plan:=_bounded_predicate_coordinate_plan(output, dtype, lambda u:_integer_nonzero_load(u, dtype),
                                               lambda value:-32768 <= value <= 32767)) is None: return None
  source, rank, index_param, index_offsets, coordinate_rows, fill_value = plan
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows)
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, coordinate_count)
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  scratch_sizes, scratch, gathers, ops = _physical_lists(64)
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
  ops.extend((RKEWOp(_scratch_arg(valid_delta), _scratch_arg(total_vector), _scratch_arg(output_coordinate), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(_scratch_arg(positive), _scratch_arg(valid_delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX], **int16),
              RKEWOp(_scratch_arg(valid), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN, **int16),
              RKEWOp(_scratch_arg(remaining), _scratch_arg(one), _scratch_arg(valid), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(_scratch_arg(selected), equal, _scratch_arg(coordinate_matrix), matrix_lanes, _EW_CFG[Ops.MUL], **int16)))
  selected_value = _reduce_rows(ops, [_scratch_arg(selected, row*vector_bytes) for row in range(coordinate_count)],
                                count, _EW_CFG[Ops.ADD], int16=True)
  ops.extend((RKEWOp(_scratch_arg(guarded), selected_value, _scratch_arg(valid), count, _EW_CFG[Ops.MUL], **int16),
              RKEWOp(_scratch_arg(fill_part), _scratch_arg(fill_slot), _scratch_arg(remaining), count, _EW_CFG[Ops.MUL], **int16),
              RKEWOp(_scratch_arg(result), _scratch_arg(guarded), _scratch_arg(fill_part), count, _EW_CFG[Ops.ADD], **int16),
              RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), _scratch_arg(result), _scratch_arg(result), count, _EW_CFG[Ops.MAX],
                     int16_input=True, int32_output=True)))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=mid, gather_after=gather_after)

def _lower_dynamic_load_with_bool_total_gate(output:RKOutput, dtype:DType=dtypes.int) -> RKImage|None:
  """Select raw typed values for `lane < sum(bool)` under an exact dynamic INT32 address."""
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

  scratch_sizes, scratch, gathers, ops = _physical_lists(64)
  bool_values = scratch(source_count*64)
  gathers.extend(RKGather(mask_param.arg.slot, bool_values, 1, offsets=(lane,), dst_addend=lane*64, dst_stride=2, itemsize=1)
                 for lane in range(source_count))
  total = _reduce_rows(ops, [_scratch_arg(bool_values, lane*64) for lane in range(source_count)], 1, _EW_CFG[Ops.ADD], int16=True)
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
  ops.extend((RKEWOp(_scratch_arg(valid_delta), _scratch_arg(total_vector), _scratch_arg(output_coordinate), count, _EW_CFG[Ops.SUB], **int16),
              RKEWOp(_scratch_arg(positive), _scratch_arg(valid_delta), _scratch_arg(zero), count, _EW_CFG[Ops.MAX], **int16),
              RKEWOp(_scratch_arg(valid), _scratch_arg(positive), _scratch_arg(one), count, _EW_CFG_MIN, **int16),
              RKEWOp(_scratch_arg(remaining), _scratch_arg(one), _scratch_arg(valid), count, _EW_CFG[Ops.SUB], **int16)))
  results:list[RKArg] = []
  for value,fill_slot in zip(raw_values, fill_slots):
    selected_matrix, guarded, fill_part, result = (scratch(matrix_lanes*2), scratch(count*2), scratch(count*2), scratch(count*2))
    ops.append(RKEWOp(_scratch_arg(selected_matrix), _scratch_arg(value), equal, matrix_lanes, _EW_CFG[Ops.MUL], **int16))
    selected_byte = _reduce_rows(ops, [_scratch_arg(selected_matrix, row*vector_bytes) for row in range(source_count)], count,
                                 _EW_CFG[Ops.ADD], int16=True)
    ops.extend((RKEWOp(_scratch_arg(guarded), selected_byte, _scratch_arg(valid), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(_scratch_arg(fill_part), _scratch_arg(fill_slot), _scratch_arg(remaining), count, _EW_CFG[Ops.MUL], **int16),
                RKEWOp(_scratch_arg(result), _scratch_arg(guarded), _scratch_arg(fill_part), count, _EW_CFG[Ops.ADD], **int16)))
    results.append(_scratch_arg(result))
  post = tuple(RKGather(value.index, out_param.arg.slot, count,
                        offsets=tuple(value.addend+lane*2 for lane in range(count)), dst_stride=itemsize, dst_addend=byte,
                        dst_kind=RKBufferKind.ARG, src_kind=RKBufferKind.SCRATCH, itemsize=1)
               for byte,value in enumerate(results))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 mid_gathers=mid, gather_after=gather_after, post_gathers=post)

def _lower_direct_dynamic_typed_load(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
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

def _lower_dynamic_multi_index_typed_load(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
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
  ops = [RKEWOp(_scratch_arg(diff), RKArg(RKBufferKind.ARG, source_slot), _scratch_arg(zero), source_count, _EW_CFG[Ops.SUB]),
         RKEWOp(_scratch_arg(magnitude), _scratch_arg(diff), _scratch_arg(diff), source_count, _EW_CFG_ABS, submit_barrier=True, stateful=True),
         RKEWOp(_scratch_arg(unequal), _scratch_arg(magnitude), _scratch_arg(magnitude), source_count, _EW_CFG[Ops.MAX], compare=True)]
  ops.extend(_block_bool_reduction_ops(_scratch_arg(unequal), count, groups, op))
  gather_after = len(ops)
  mid = (RKGather(unequal, packed, count, offsets=tuple(lane*groups for lane in range(count)),
                  src_kind=RKBufferKind.SCRATCH),)
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), _scratch_arg(packed), _scratch_arg(int_tiles), count, _EW_CFG[Ops.MAX],
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

def _integer_predicate_reduction_image(out_slot:int, count:int, source_slot:int, offsets:tuple[tuple[int, ...], ...],
                                       op:Ops, itemsize:int) -> RKImage|None:
  """Reduce exact integer nonzero masks with native INT16 ADD/MAX/MUL."""
  vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(count, len(offsets))
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None
  scratch_sizes, scratch, gathers, ops = _physical_lists(64)
  if (mask:=_native_integer_nonzero_mask(ops, gathers, scratch, source_slot, offsets,
                                         count, vector_lanes, itemsize)) is None: return None
  reduced = _reduce_rows(ops, [replace(mask, addend=mask.addend+row*vector_bytes) for row in range(len(offsets))],
                         count, _EW_CFG[{Ops.OR:Ops.MAX, Ops.AND:Ops.MUL}.get(op, op)], int16=True)
  if op is Ops.ADD:
    ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_slot), reduced, reduced, count, _EW_CFG[Ops.MAX],
                      int16_input=True, int32_output=True))
    post:tuple[RKGather, ...] = ()
  else: post = (_int16_low_bytes(reduced, out_slot, count),)
  return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in scratch_sizes), gathers=tuple(gathers), ew_ops=tuple(ops),
                 post_gathers=post)

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

def _integer_nonzero_load(term:UOp, dtype:DType=dtypes.int) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [load for load,zero in (term.src, term.src[::-1]) if load.op is Ops.LOAD and load.dtype.scalar() is dtype and
                load.src[0].op is Ops.INDEX and zero.op is Ops.CONST and int(zero.arg) == 0]
  return candidates[0] if len(candidates) == 1 else None

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
  if predicate is None or (load:=_nonzero_load(predicate)) is None and \
     (load:=_integer_nonzero_load(predicate, dtypes.int16)) is None: return None
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
  return (_integer_predicate_reduction_image(out_param.arg.slot, rows, source.arg.slot, offsets, update.op, 2)
          if load.dtype.scalar() is dtypes.int16 else _bool_reduction_image(out_param.arg.slot, rows, source.arg.slot, offsets, update.op))

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

def _typed_half_image(output:RKOutput, value:UOp, int32:bool, bool_output:bool=False) -> RKImage:
  """Lower an exact FP16 expression through the requested native integer output ABI."""
  store, out_param, count, _, _ = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_slot = out_param.arg.slot
  half_param = out_param.replace(dtype=dtypes.half, arg=replace(out_param.arg, dtype=dtypes.half))
  half_index = store.src[0].replace(dtype=dtypes.half, src=(half_param, *store.src[0].src[1:]))
  replacement = store.replace(src=(half_index, value, *store.src[2:]))
  image = _lower_composed_uops(list(replacement.toposort())) if int32 else \
    _lower_composed_uops(_fp16_rewrite(list(UOp(Ops.SINK, src=(replacement,)).toposort())), recipes_ready=True)
  terminal = [i for i,op in enumerate(image.ew_ops) if op.dst.kind is RKBufferKind.ARG and op.dst.index == out_slot]
  if terminal != [len(image.ew_ops)-1] or image.fill is not None or image.mid_gathers or image.post_gathers:
    raise RuntimeError("RKPLAN_REJECT:predicate_terminal" if int32 else "RKPLAN_REJECT:uint8_terminal")
  result_slot, auxiliary_slot = len(image.scratch), len(image.scratch)+1
  result = RKArg(RKBufferKind.SCRATCH, result_slot)
  if int32:
    auxiliary = RKArg(RKBufferKind.SCRATCH, auxiliary_slot)
    ops = (*image.ew_ops[:-1], replace(image.ew_ops[-1], dst=result),
           RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, auxiliary, count, _EW_CFG[Ops.MAX],
                  stateful=True, int32_output=True, bool_output=bool_output))
    return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(_int32_tiles_bytes(count))), ew_ops=ops)
  int_result = RKArg(RKBufferKind.SCRATCH, auxiliary_slot)
  ops = (*image.ew_ops[:-1], replace(image.ew_ops[-1], dst=result),
         RKEWOp(int_result, result, result, count, _EW_CFG[Ops.MAX], submit_barrier=True, stateful=True, int16_output=True))
  return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(_scratch_bytes(count))), ew_ops=ops,
                 post_gathers=(_int16_low_bytes(int_result, out_slot, count),))

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
  return result

def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if root.op is Ops.CAST and root.dtype.scalar() is dtypes.bool and len(root.src) == 1 and root.src[0].dtype.scalar() is dtypes.half:
    root = root.src[0] != UOp.const(0.0, dtypes.half)
  if (load:=_nonzero_load(root)) is None: return None
  magnitude = UOp(Ops.MAX, dtypes.half, src=(load, load), arg=_NATIVE_ABS)
  return _positive_mask(magnitude)

def _exact_int_range(root:UOp, cache:dict[UOp, tuple[int, int]|None]|None=None) -> tuple[int, int]|None:
  """Conservatively bound an integer UOp before choosing its exact physical scratch layout."""
  if cache is None: cache = {}
  if root in cache: return cache[root]
  if root.dtype.scalar() not in (dtypes.int, dtypes.weakint): valid = False
  elif root.op is Ops.CONST or root.op is Ops.RANGE and len(root.src) == 1 and root.src[0].op is Ops.CONST: valid = True
  elif root.op is Ops.CAST and len(root.src) == 1: valid = root.src[0].dtype.scalar() is dtypes.bool
  elif root.op is Ops.WHERE and len(root.src) == 3: valid = all(_exact_int_range(src, cache) is not None for src in root.src[1:])
  elif root.op is Ops.XOR and len(root.src) == 2:
    valid = any(marker.op is Ops.CONST and marker.arg == -1 and _exact_int_range(source, cache) is not None
                for marker,source in (root.src, root.src[::-1]))
  elif root.op is Ops.CMOD and len(root.src) == 2:
    right = _exact_int_range(root.src[1], cache)
    valid = right is not None and right[0] == right[1] != 0
  else: valid = root.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(root.src) == 2 and \
                all(_exact_int_range(src, cache) is not None for src in root.src)
  result = None
  if valid:
    low, high, dtype = int(root.vmin), int(root.vmax), root.dtype.scalar()
    if dtype.min <= low <= high <= dtype.max: result = (0, max(0, high)) if root.op is Ops.RANGE else (low, high)
  cache[root] = result
  return result

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
  return graph_rewrite(simplified, pm_commit_weak, name="rockchip commit storage constants")

def _fp32_add_terms(u:UOp) -> list[UOp]: return [_fp32_expr_to_half(x) for x in _iter_binary(u, Ops.ADD, dtypes.float)]

def _fp32_add_has_product_terms(u:UOp) -> bool:
  """Whether a floating ADD tree contains a direct floating or cast-half product term."""
  return any((term.op is Ops.MUL and term.dtype.scalar() is dtypes.float) or
             (term.op is Ops.CAST and len(term.src) == 1 and term.src[0].op is Ops.MUL and
              term.src[0].dtype.scalar() is dtypes.half) for term in _iter_binary(u, Ops.ADD, dtypes.float))

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
  cache:dict[UOp, UOp] = {}
  for node in root.toposort():
    tagged = node.replace(src=tuple(cache[src] for src in node.src))
    if tagged.op is Ops.ADD: tagged = tagged.replace(arg=_NATIVE_PRECISE_ADD)
    cache[node] = tagged
  return cache[root]

def _accurate_add_recipe(u:UOp) -> UOp:
  terms:list[UOp] = []
  def flatten(x:UOp) -> None:
    if x.op is Ops.ADD and x.dtype.scalar() is dtypes.half and x.arg is None:
      flatten(x.src[0]); flatten(x.src[1])
    elif x.op is Ops.CAST and x.dtype.scalar() is dtypes.half and len(x.src) == 1 and x.src[0].dtype.scalar() is dtypes.float and \
         x.src[0].op is Ops.ADD:
      terms.extend(_fp32_add_terms(x.src[0]))
    else: terms.append(x)
  flatten(u)
  if sum(term.op is Ops.MUL and term.arg is None for term in terms) < 2: raise _RKGenericReject
  if any(any(node.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN) for node in term.toposort()) for term in terms):
    raise _RKGenericReject
  return _precise_mul_sum(terms)

class RKContext:
  """Typed physical lowering context. UOps remain the only semantic IR."""
  def __init__(self, output:RKOutput, *, accurate_adds:bool=True):
    self.store, self.out_param, self.count, self.out_index, self.root = output
    self.out = RKArg(RKBufferKind.ARG, self.out_param.arg.slot)
    self.values:dict[UOp, RKValue] = {}
    self.scratch:list[RKScratch] = []
    self.constants:dict[bytes, int] = {}
    self.static_slots:dict[tuple[RKLayout, tuple[int, ...]], RKValue] = {}
    self.gather_slots:dict[tuple, RKValue] = {}
    self.raw_components:dict[RKArg, tuple[RKValue, ...]] = {}
    self.int32_divmod:dict[tuple[UOp, UOp], tuple[tuple[RKArg, ...], tuple[RKArg, ...], RKArg, RKArg]] = {}
    self.int16_masks:dict[RKArg, int] = {}
    self.fp16_components:dict[RKArg, tuple[RKValue, RKArg, RKArg]] = {}
    self.fp16_ordered:dict[RKArg, tuple[RKArg, RKArg]] = {}
    self.gathers:list[RKGather] = []
    self.host_gathers:list[RKHostAddress] = []
    self.mid_gathers:list[RKGather] = []
    self.post_gathers:list[RKGather] = []
    self.ew_ops:list[RKEWOp] = []
    self.mask_program = any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in self.root.toposort())
    nodes = self.root.toposort()
    self.int_ranges:dict[UOp, tuple[int, int]|None] = {}
    int_range = _exact_int_range(self.root, self.int_ranges) if self.root.dtype.scalar() is dtypes.int else None
    packed_bool_load = any(node.op is Ops.LOAD and node.dtype.scalar() is dtypes.bool and _root_param(node.src[0]) is not None for node in nodes)
    embedded_half_int = any(node.op is Ops.CAST and node.dtype.scalar() is dtypes.int and len(node.src) == 1 and
                            node.src[0].dtype.scalar() in (dtypes.half, dtypes.bool) for node in nodes)
    dynamic_int_load = any(node.op is Ops.LOAD and node.dtype.scalar() in (dtypes.int, dtypes.uint) and node.src and
                           _root_param(node.src[0]) is not None for node in nodes)
    self.int_layout = (RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int and dynamic_int_load else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and packed_bool_load and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT_FP16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -2048 <= int_range[0] <= int_range[1] <= 2048 else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int else
                       RKLayout.INT32 if dynamic_int_load else
                       RKLayout.INT_FP16 if embedded_half_int else None)
    self.accurate_adds = accurate_adds
    self.static_nodes:set[UOp] = set()
    for node in self.root.toposort():
      if node.op in _STATIC_OPS and all(src in self.static_nodes for src in node.src): self.static_nodes.add(node)

  def _scratch(self, dtype:DType, layout:RKLayout, size:int|None=None) -> RKValue:
    slot = len(self.scratch)
    self.scratch.append(RKScratch(self.count*4 if size is None and layout is RKLayout.INT32 else
                                  _scratch_bytes(self.count) if size is None else size))
    return RKValue(RKArg(RKBufferKind.SCRATCH, slot), dtype, self.count, layout)

  def _dst(self, u:UOp, dtype:DType, layout:RKLayout) -> RKValue:
    if (u is self.root and self.out_param.dtype.scalar() is dtype and
        (dtype is dtypes.half and layout is RKLayout.FP16 or dtype is dtypes.int16 and layout is RKLayout.INT16 or
         dtype is dtypes.int and layout is RKLayout.INT32)):
      return RKValue(self.out, dtype, self.count, layout)
    return self._scratch(dtype, layout)

  def _alu_dst(self, u:UOp, dtype:DType, layout:RKLayout, operands:tuple[tuple[UOp, RKValue], ...]) -> RKValue:
    del operands
    if (u is self.root and self.out_param.dtype.scalar() is dtype and
        (dtype is dtypes.half and layout is RKLayout.FP16 or dtype is dtypes.int16 and layout is RKLayout.INT16 or
         dtype is dtypes.int and layout is RKLayout.INT32)):
      return RKValue(self.out, dtype, self.count, layout)
    return self._scratch(dtype, layout)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> RKValue:
    dtype = dtype_hint or u.dtype.scalar()
    if dtype is dtypes.uint or dtype is dtypes.int and self.int_layout is RKLayout.INT32:
      vector = (int(u.arg) & 0xffffffff,) * self.count
      key = (RKLayout.INT32, vector)
      if key not in self.static_slots:
        value = self._scratch(dtype, RKLayout.INT32)
        self.gathers.append(RKGather(0, value.arg.index, self.count, values=vector, itemsize=4))
        self.static_slots[key] = value
      cached = self.static_slots[key]
      return RKValue(cached.arg, dtype, self.count, RKLayout.INT32)
    if dtype in (dtypes.half, dtypes.float): bits, layout = struct.pack("<e", float(u.arg)), RKLayout.FP16
    elif dtype is dtypes.int16: bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), RKLayout.INT16
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
    elif dtype in (dtypes.int, dtypes.uint):
      values = _static_values(self.out_index, u, self.count, int)
      if dtype is dtypes.uint: vector, layout = tuple(value & 0xffffffff for value in values), RKLayout.INT32
      elif self.int_layout is RKLayout.INT_FP16 and all(-2048 <= value <= 2048 for value in values):
        vector, layout = tuple(_fp16_bits(float(value)) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT16 and all(-32768 <= value <= 32767 for value in values):
        vector, layout = tuple(_int16_bits(value) for value in values), self.int_layout
      elif self.int_layout is RKLayout.INT32:
        vector, layout = tuple(value & 0xffffffff for value in values), self.int_layout
      else: raise _RKGenericReject
    elif dtype is dtypes.bool:
      if bool_layout is RKLayout.BOOL_INT16: vector, layout = _static_values(self.out_index, u, self.count, int), bool_layout
      else: vector, layout = _static_values(self.out_index, u, self.count, _fp16_bits), RKLayout.BOOL_MASK
    else: raise _RKGenericReject
    key = (layout, vector)
    if key not in self.static_slots:
      value = self._scratch(dtype, layout)
      self.gathers.append(RKGather(0, value.arg.index, self.count, values=vector, itemsize=4 if layout is RKLayout.INT32 else 2))
      self.static_slots[key] = value
    cached = self.static_slots[key]
    return RKValue(cached.arg, dtype, self.count, layout)

  def _static_int32(self, u:UOp) -> RKValue:
    vector = tuple(value & 0xffffffff for value in _static_values(self.out_index, u, self.count, int))
    key = (RKLayout.INT32, vector)
    if key not in self.static_slots:
      value = self._scratch(dtypes.int, RKLayout.INT32)
      self.gathers.append(RKGather(0, value.arg.index, self.count, values=vector, itemsize=4))
      self.static_slots[key] = value
    return self.static_slots[key]

  def _load(self, u:UOp, fill_override:int|None=None) -> RKValue:
    dtype = u.dtype.scalar()
    if dtype not in (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) or not u.src or u.src[0].op is not Ops.INDEX or \
       (param:=_root_param(u.src[0])) is None or param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST:
      raise _RKGenericReject
    index = u.src[0].src[1]
    gate = u.src[2] if len(u.src) > 2 else None
    default = u.src[1] if len(u.src) > 1 else None
    index_loads, gate_loads = _semantic_loads(index), () if gate is None else _semantic_loads(gate)
    runtime_address = bool(index_loads or gate_loads)
    if default is not None and default.op is not Ops.CONST:
      if dtype not in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint) or gate is None or runtime_address:
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
      if raw_key not in self.gather_slots:
        raw = self._scratch(dtype, RKLayout.FP16, len(groups)*16)
        self.gathers.append(replace(plan, dst_index=raw.arg.index, itemsize=4))
        self.gather_slots[raw_key] = raw
      raw = self.gather_slots[raw_key]
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
      key = (RKLayout.BOOL_INT16, _gather_cache_key((plan,)))
      if key not in self.gather_slots:
        value = self._scratch(dtype, RKLayout.BOOL_INT16, self.count*2)
        self.gathers.append(replace(plan, dst_index=value.arg.index))
        self.gather_slots[key] = value
      return self.gather_slots[key]
    layout = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
    itemsize = 4 if layout is RKLayout.INT32 else 2
    if runtime_address:
      if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1": raise _RKGenericReject
      runtime_index = _runtime_affine_index(index, self.out_index, self.count)
      runtime_loads = {node.key:node for node in (*index_loads, *gate_loads) if _runtime_index(node) is not None}
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
        limits = [int(node.src[1].arg) for node in gate.toposort() if node.op is Ops.CMPLT and
                  node.src[0].key == runtime_load.key and node.src[1].op is Ops.CONST and int(node.src[1].arg) > 0]
        if len(set(limits)) != 1 or not _bounded_index_gate(gate, runtime_load, limits[0]) or \
           {node.key for node in gate.toposort() if node.op is Ops.LOAD} != {runtime_load.key}: raise _RKGenericReject
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
        key = (layout, _gather_cache_key((plan,)))
        if key not in self.gather_slots:
          matrix = self._scratch(dtype, layout, len(offsets)*itemsize)
          self.gathers.append(replace(plan, dst_index=matrix.arg.index))
          self.gather_slots[key] = matrix
        source, source_count = self.gather_slots[key].arg, len(offsets)
        base, index_scale, lane_stride = 0, 1, index_limit
      value = self._scratch(dtype, layout, self.count*itemsize)
      fill_bits = fill_override if fill_override is not None else _fp16_bits(0 if default is None else default.arg) if dtype is dtypes.half else \
        _int16_bits(0 if default is None else default.arg) if dtype is dtypes.int16 else int(0 if default is None else default.arg) & 0xffffffff
      self.host_gathers.append(RKHostAddress(source,
        RKArg(RKBufferKind.ARG, index_param.arg.slot, index_offset*index_itemsize), value.arg,
        self.count, source_count, self.count,
        itemsize=itemsize, index_itemsize=index_itemsize, fill_bits=fill_bits, index_limit=index_limit,
        base=base, index_scale=index_scale, lane_stride=lane_stride))
      return value
    if gate is None and index.key == self.out_index.key and int(param.src[0].arg) == self.count:
      return RKValue(RKArg(RKBufferKind.ARG, param.arg.slot), dtype, self.count, layout)
    fill_bits = fill_override if fill_override is not None else _fp16_bits(0 if default is None else default.arg) if dtype is dtypes.half else \
      _int16_bits(0 if default is None else default.arg) if dtype is dtypes.int16 else int(0 if default is None else default.arg) & 0xffffffff
    plan = _gather_plan(param.arg.slot, 0, self.out_index, index, gate, self.count, fill_bits)
    _validate_gather_bounds(plan, int(param.src[0].arg))
    key = (layout, _gather_cache_key((plan,)))
    if key not in self.gather_slots:
      value = self._scratch(dtype, layout, self.count*itemsize)
      self.gathers.append(replace(plan, dst_index=value.arg.index, itemsize=itemsize))
      self.gather_slots[key] = value
    return self.gather_slots[key]

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

  def _native_min(self, u:UOp, lhs:RKValue, rhs:RKValue) -> RKValue:
    zero = self.lower(UOp.const(0.0, dtypes.half))
    neg_lhs, neg_rhs = (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(2))
    self._emit(neg_lhs, zero, lhs, _EW_CFG[Ops.SUB])
    self._emit(neg_rhs, zero, rhs, _EW_CFG[Ops.SUB])
    self._emit(neg_lhs, neg_lhs, neg_rhs, _EW_CFG[Ops.MAX])
    dst = self._dst(u, dtypes.half, RKLayout.FP16) if u is self.root else neg_lhs
    return self._emit(dst, zero, neg_lhs, _EW_CFG[Ops.SUB])

  def _raw_parts(self, value:RKValue) -> tuple[RKValue, ...]:
    if value.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT32): raise _RKGenericReject
    if value.arg in self.raw_components: return self.raw_components[value.arg]
    itemsize, source = (4 if value.layout is RKLayout.INT32 else 2), value
    if itemsize == 4:
      source = self._scratch(dtypes.int, RKLayout.INT32)
      self._emit(source, value, value, _EW_CFG[Ops.MAX])
    parts = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(itemsize))
    split_after = len(self.ew_ops)
    for byte,part in enumerate(parts):
      self.mid_gathers.append(RKGather(source.arg.index, part.arg.index, self.count,
        base=source.arg.addend+byte, axes=((1, self.count, itemsize),), dst_stride=2,
        src_kind=source.arg.kind, itemsize=1, after=split_after))
    self.raw_components[value.arg] = parts
    return parts

  def _pack_raw(self, u:UOp, components:Iterable[RKValue|RKArg], layout:RKLayout, mask:int|None=None) -> RKValue:
    parts = tuple(part if isinstance(part, RKValue) else RKValue(part, dtypes.int16, self.count, RKLayout.INT16) for part in components)
    itemsize = 4 if layout is RKLayout.INT32 else 2
    if len(parts) != itemsize: raise _RKGenericReject
    result = self._dst(u, u.dtype.scalar(), layout)
    pack_after = len(self.ew_ops)
    for byte,source in enumerate(parts):
      self.mid_gathers.append(RKGather(source.arg.index, result.arg.index, self.count,
        base=source.arg.addend, axes=((1, self.count, 2),), dst_stride=itemsize, dst_addend=byte,
        dst_kind=result.arg.kind, src_kind=source.arg.kind, itemsize=1, after=pack_after))
    self.raw_components[result.arg] = parts
    if mask is not None: self.int16_masks[result.arg] = mask
    return result

  def _alu(self, u:UOp) -> RKValue:
    if u.op is Ops.RECIPROCAL:
      src = self.lower(u.src[0]); one = self.lower(UOp.const(1.0, dtypes.half))
      return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), one, src, _EW_CFG[Ops.FDIV])
    if u.op is Ops.NEG:
      src = self.lower(u.src[0])
      dst = self._alu_dst(u, u.dtype.scalar(), src.layout, ((u.src[0], src),))
      return self._emit(dst, src, src, _EW_CFG_NEG)
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.ADD and (recipe:=_fold_relu_cap(u)) is not None:
      return self.lower(recipe)
    if u.op is Ops.FDIV and (recipe:=_preserve_infinite_division_sign(u)) is not None:
      return self.lower(recipe)
    dtype = u.dtype.scalar()
    int_range = _exact_int_range(u, self.int_ranges) if dtype is dtypes.int else None
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
      dst = self._alu_dst(u, dtype, RKLayout.INT16, ((u.src[0], lhs), (u.src[1], rhs)))
      return self._emit(dst, lhs, rhs, _EW_CFG_MIN)
    if u.op is Ops.MAX and u.arg == _NATIVE_RAW_MIN:
      dst = self._alu_dst(u, dtype, expected, ((u.src[0], lhs), (u.src[1], rhs)))
      return self._emit(dst, lhs, rhs, _EW_CFG_MIN)
    cfg = _EW_CFG_ABS if u.op is Ops.MAX and u.arg == _NATIVE_ABS else \
      _EW_CFG_FLOOR if u.op is Ops.MAX and u.arg == _NATIVE_FLOOR else \
      _EW_CFG_CEIL if u.op is Ops.MAX and u.arg == _NATIVE_CEIL else \
      _EW_CFG_RELU6 if u.op is Ops.MAX and u.arg == _NATIVE_RELU6 else \
      _EW_CFG_LEAKY_RELU if u.op is Ops.MUL and u.arg == _NATIVE_LEAKY_RELU else _EW_CFG[u.op]
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    layout = RKLayout.BOOL_MASK if compare else expected
    out_dtype = dtypes.bool if compare else dtype
    dst = self._alu_dst(u, out_dtype, layout, ((u.src[0], lhs), (u.src[1], rhs)))
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
    if preferred not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
    for i,(src,value) in enumerate(zip(u.src, values)):
      if value is None:
        raw = self._constant(UOp.const(int(bool(src.arg)), dtypes.int16)) if preferred is RKLayout.BOOL_INT16 else self._constant(src)
        values[i] = RKValue(raw.arg, dtypes.bool, self.count, preferred)
      else: values[i] = self._coerce_bool(value, preferred)
    lhs, rhs = values
    assert lhs is not None and rhs is not None
    if lhs.layout is not preferred or rhs.layout is not preferred: raise _RKGenericReject
    dst = self._dst(u, dtypes.bool, preferred)
    if u.op is Ops.AND: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MUL])
    if u.op is Ops.OR: return self._emit(dst, lhs, rhs, _EW_CFG[Ops.MAX])
    if u.op in (Ops.XOR, Ops.CMPNE):
      for source,one,other in ((u.src[0], lhs, rhs), (u.src[1], rhs, lhs)):
        if source.op is Ops.CONST and source.dtype.scalar() is dtypes.bool and bool(source.arg): return self._emit(dst, one, other, _EW_CFG[Ops.SUB])
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
        low, high = self._raw_parts(value)
        zero, one, const127, const128 = (self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128))
        delta, positive, sign, sign_scale = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(4))
        self._emit(delta, high, const127, _EW_CFG[Ops.SUB])
        self._emit(positive, delta, zero, _EW_CFG[Ops.MAX])
        self._emit(sign, positive, one, _EW_CFG_MIN)
        self._emit(sign_scale, sign, const128, _EW_CFG[Ops.MUL])
        if mask == 0x7fff:
          masked_high = self._scratch(dtypes.int16, RKLayout.INT16)
          self._emit(masked_high, high, sign_scale, _EW_CFG[Ops.SUB])
          return self._pack_raw(u, (low, masked_high), RKLayout.INT16, mask)
        return self._pack_raw(u, (zero, sign_scale), RKLayout.INT16, mask)
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.OR:
      values = tuple(self.lower(source) for source in u.src)
      masks = tuple(self.int16_masks.get(value.arg) for value in values)
      if all(mask is not None for mask in masks) and typing_cast(int, masks[0]) & typing_cast(int, masks[1]) == 0:
        parts = tuple(self._raw_parts(value) for value in values)
        low, high = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(2))
        self._emit(low, parts[0][0], parts[1][0], _EW_CFG[Ops.ADD])
        self._emit(high, parts[0][1], parts[1][1], _EW_CFG[Ops.ADD])
        return self._pack_raw(u, (low, high), RKLayout.INT16, typing_cast(int, masks[0]) | typing_cast(int, masks[1]))
    if u.op is Ops.XOR:
      for marker, source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or int(marker.arg) != -1: continue
        dtype = u.dtype.scalar()
        layout = RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if dtype is dtypes.int else None
        if layout is None: raise _RKGenericReject
        if layout is RKLayout.INT32 and u is self.root and 1 <= self.count*4 <= _MAX_EW_ELEMS_FP16 and \
           (parsed:=_typed_load_offsets(source, dtypes.int, self.out_index, self.count, allow_fill=True)) is not None:
          param, offsets = parsed
          lanes, stride, slot = self.count*4, _reduction_stride(self.count*4), len(self.scratch)
          self.scratch.append(RKScratch(stride*3))
          raw_arg, constant_arg, inverted_arg = (RKArg(RKBufferKind.SCRATCH, slot, row*stride) for row in range(3))
          self.gathers.extend((RKGather(param.arg.slot, raw_arg.index, lanes,
            offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in offsets for byte in range(4)), dst_stride=2, itemsize=1),
            RKGather(param.arg.slot, constant_arg.index, lanes, values=(255,)*lanes, dst_addend=constant_arg.addend//2)))
          self.ew_ops.append(RKEWOp(inverted_arg, constant_arg, raw_arg, lanes, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True))
          self.post_gathers.append(_int16_low_bytes(inverted_arg, self.out_param.arg.slot, lanes))
          return RKValue(self.out, dtype, self.count, layout)
        rhs = self.lower(source)
        if rhs.layout is not layout: raise _RKGenericReject
        if layout is RKLayout.INT32:
          components = self._raw_parts(rhs)
          const255 = self._constant(UOp.const(255, dtypes.int16))
          inverted = tuple(self._emit(self._scratch(dtypes.int16, RKLayout.INT16), const255, component, _EW_CFG[Ops.SUB])
                           for component in components)
          return self._pack_raw(u, inverted, RKLayout.INT32)
        lhs = self._constant(UOp.const(-1, dtype))
        return self._emit(self._dst(u, dtype, layout), lhs, rhs, _EW_CFG[Ops.SUB])
    dtype = u.dtype.scalar()
    layout = RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32 if dtype is dtypes.int and self.int_layout is RKLayout.INT32 else None
    if layout is None or u.op not in (Ops.AND, Ops.OR, Ops.XOR): raise _RKGenericReject
    values = tuple(self.lower(source) for source in u.src)
    if any(value.layout is not layout for value in values): raise _RKGenericReject
    lanes = self.count*(4 if layout is RKLayout.INT32 else 2)
    if not 1 <= lanes <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
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
    integer = dict(int16_input=True, int16_output=True)
    weighted:list[RKArg] = []
    for bit,(left,right) in enumerate(zip(lhs_bits, rhs_bits)):
      combined = allocate()
      if u.op is Ops.XOR:
        self.ew_ops.extend((RKEWOp(combined, left, right, lanes, _EW_CFG[Ops.SUB], **integer),
                            RKEWOp(combined, combined, combined, lanes, _EW_CFG_ABS, **integer)))
      else:
        self.ew_ops.append(RKEWOp(combined, left, right, lanes,
          _EW_CFG_MIN if u.op is Ops.AND else _EW_CFG[Ops.MAX], **integer))
      if bit: self.ew_ops.append(RKEWOp(combined, combined, constants[1<<bit], lanes, _EW_CFG[Ops.MUL], **integer))
      weighted.append(combined)
    combined = _reduce_rows(self.ew_ops, weighted, lanes, _EW_CFG[Ops.ADD], int16=True)
    result = self._dst(u, dtype, layout)
    self.mid_gathers.append(RKGather(combined.index, result.arg.index, lanes, base=combined.addend,
      axes=((1, lanes, 2),), dst_stride=1, dst_kind=result.arg.kind,
      src_kind=RKBufferKind.SCRATCH, itemsize=1, after=len(self.ew_ops)))
    return result

  def _int32_shift(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or u.dtype.scalar() not in (dtypes.int, dtypes.uint) or \
       u.src[1].dtype.scalar() not in (dtypes.int, dtypes.uint) or self.int_layout is not RKLayout.INT32:
      raise _RKGenericReject
    value = self.lower(u.src[0])
    if value.layout is not RKLayout.INT32: raise _RKGenericReject
    vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(self.count, 32)
    pre_lanes = vector_lanes*5
    if self.count < 1 or matrix_lanes > _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    pre_arena, pre_row = self._scratch(dtypes.int16, RKLayout.INT16, 51*pre_lanes*2).arg, 0
    def pre_allocate() -> RKArg:
      nonlocal pre_row
      result = replace(pre_arena, addend=pre_row*pre_lanes*2); pre_row += 1
      return result
    raw, value_parts = pre_allocate(), self._raw_parts(value)
    shift = None if u.src[1].op is Ops.CONST else self.lower(u.src[1])
    if shift is not None and shift.layout is not RKLayout.INT32: raise _RKGenericReject
    shift_part = None if shift is None else self._raw_parts(shift)[0]
    raw_after = len(self.ew_ops)
    for byte,part in enumerate(value_parts):
      self.mid_gathers.append(RKGather(part.arg.index, raw.index, self.count, base=part.arg.addend//2,
        axes=((1, self.count, 1),), dst_addend=byte*vector_lanes,
        src_kind=part.arg.kind, after=raw_after))
    if shift_part is None:
      self.mid_gathers.append(RKGather(self.out_param.arg.slot, raw.index, self.count,
        values=(int(u.src[1].arg)&0xff,)*self.count, dst_addend=4*vector_lanes, after=raw_after))
    else:
      self.mid_gathers.append(RKGather(shift_part.arg.index, raw.index, self.count, base=shift_part.arg.addend//2,
        axes=((1, self.count, 1),), dst_addend=4*vector_lanes,
        src_kind=shift_part.arg.kind, after=raw_after))
    constants = {number:pre_allocate() for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)}
    for number,dst in constants.items():
      self.gathers.append(RKGather(self.out_param.arg.slot, dst.index, pre_lanes,
        values=(number,)*pre_lanes, dst_addend=dst.addend//2))
    planes = _int16_byte_bits(self.ew_ops, pre_allocate, constants, raw, pre_lanes)

    post_arena, post_row = self._scratch(dtypes.int16, RKLayout.INT16, 20*matrix_lanes*2).arg, 0
    def post_allocate() -> RKArg:
      nonlocal post_row
      result = replace(post_arena, addend=post_row*matrix_lanes*2); post_row += 1
      return result
    bits, masks, sign, zero, weights = post_allocate(), tuple(post_allocate() for _ in range(5)), \
                                       post_allocate(), post_allocate(), post_allocate()
    post_after = len(self.ew_ops)
    for absolute_bit in range(32):
      plane,byte = absolute_bit&7, absolute_bit>>3
      self.mid_gathers.append(RKGather(planes[plane].index, bits.index, self.count,
        base=planes[plane].addend//2+byte*vector_lanes, axes=((1, self.count, 1),),
        dst_addend=absolute_bit*vector_lanes, src_kind=RKBufferKind.SCRATCH, after=post_after))
    for bit,mask in enumerate(masks):
      self.mid_gathers.append(RKGather(planes[bit].index, mask.index, matrix_lanes,
        base=planes[bit].addend//2+4*vector_lanes, axes=((1, vector_lanes, 1),),
        dst_addend=mask.addend//2, src_kind=RKBufferKind.SCRATCH, after=post_after))
    self.mid_gathers.append(RKGather(planes[7].index, sign.index, matrix_lanes,
      base=planes[7].addend//2+3*vector_lanes, axes=((1, vector_lanes, 1),),
      dst_addend=sign.addend//2, src_kind=RKBufferKind.SCRATCH, after=post_after))
    self.mid_gathers.append(RKGather(self.out_param.arg.slot, weights.index, matrix_lanes,
      values=tuple(1 << (row&7) if lane < self.count else 0 for row in range(32) for lane in range(vector_lanes)),
      dst_addend=weights.addend//2, after=post_after))
    integer = dict(int16_input=True, int16_output=True)
    current = bits
    for bit,amount in enumerate((1, 2, 4, 8, 16)):
      temp, result = post_allocate(), post_allocate()
      normal_rows,normal_dst,shifted_src = (32-amount, amount, 0) if u.op is Ops.SHL else (32-amount, 0, amount)
      boundary_rows,boundary_dst = (amount, 0) if u.op is Ops.SHL else (amount, 32-amount)
      for rows,dst_row,src,fill in ((normal_rows, normal_dst, current, shifted_src),
                                    (boundary_rows, boundary_dst, sign if u.op is Ops.SHR and u.dtype.scalar() is dtypes.int else zero, 0)):
        addend = dst_row*vector_bytes
        dst,old,selected = (replace(arg, addend=arg.addend+addend) for arg in (temp, current, result))
        source = replace(src, addend=src.addend+(fill*vector_bytes if src is current else addend))
        mask = replace(masks[bit], addend=masks[bit].addend+addend)
        count = rows*vector_lanes
        self.ew_ops.extend((RKEWOp(dst, source, old, count, _EW_CFG[Ops.SUB], **integer),
                            RKEWOp(dst, dst, mask, count, _EW_CFG[Ops.MUL], **integer),
                            RKEWOp(selected, old, dst, count, _EW_CFG[Ops.ADD], **integer)))
      current = result
    weighted = post_allocate()
    self.ew_ops.append(RKEWOp(weighted, current, weights, matrix_lanes, _EW_CFG[Ops.MUL], **integer))
    byte_results = tuple(_reduce_rows(self.ew_ops,
      [replace(weighted, addend=weighted.addend+(byte*8+bit)*vector_bytes) for bit in range(8)],
      vector_lanes, _EW_CFG[Ops.ADD], int16=True) for byte in range(4))
    return self._pack_raw(u, byte_results, RKLayout.INT32)

  def _compare(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src): return self._bool_binary(u)
    if u.op in (Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_equality(u)
    if u.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_less(u)
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      sources = tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src)
      bounds = tuple(_exact_int_range(src, self.int_ranges) for src in sources)
      if self.int_layout is RKLayout.INT_FP16 or self.int_layout is not RKLayout.INT32 and all(
        bound is not None and -2048 <= bound[0] <= bound[1] <= 2048 for bound in bounds
      ):
        value = self.lower(UOp(u.op, dtypes.bool, src=tuple(_int_fp16_expr(src) for src in sources), arg=u.arg))
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
    value = self.lower(ieee_recipe)
    if value.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK): raise _RKGenericReject
    return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_MASK)

  def _ieee_bool(self, recipe:UOp) -> RKValue:
    value = self.lower(recipe)
    if value.layout not in (RKLayout.FP16, RKLayout.BOOL_MASK): raise _RKGenericReject
    return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_MASK)

  def _i16(self, lhs:RKArg, rhs:RKArg, cfg:int) -> RKArg:
    dst = self._scratch(dtypes.int16, RKLayout.INT16).arg
    self.ew_ops.append(RKEWOp(dst, lhs, rhs, self.count, cfg, int16_input=True, int16_output=True))
    return dst

  def _i16_const(self, value:int) -> RKArg: return self._constant(UOp.const(value, dtypes.int16)).arg

  def _i16_clamp_one(self, value:RKArg) -> RKArg:
    return self._i16(self._i16(value, self._i16_const(0), _EW_CFG[Ops.MAX]), self._i16_const(1), _EW_CFG_MIN)

  def _i16_positive_over(self, value:RKArg, threshold:int) -> RKArg:
    return self._i16_clamp_one(self._i16(value, self._i16_const(threshold), _EW_CFG[Ops.SUB]))

  def _i16_xor(self, lhs:RKArg, rhs:RKArg) -> RKArg:
    delta = self._i16(lhs, rhs, _EW_CFG[Ops.SUB])
    return self._i16(delta, delta, _EW_CFG_ABS)

  def _i16_twos_complement(self, raw:tuple[RKArg, ...], sign:RKArg) -> tuple[RKArg, ...]:
    carry, result = sign, []
    for byte in raw:
      doubled = self._i16(byte, byte, _EW_CFG[Ops.ADD])
      inverted = self._i16(self._i16(self._i16_const(255), doubled, _EW_CFG[Ops.SUB]), sign, _EW_CFG[Ops.MUL])
      total = self._i16(self._i16(byte, inverted, _EW_CFG[Ops.ADD]), carry, _EW_CFG[Ops.ADD])
      carry = self._i16_positive_over(total, 255)
      result.append(self._i16(total, self._i16(carry, self._i16_const(256), _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB]))
    return tuple(result)

  def _int32_divmod(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or self.int_layout is not RKLayout.INT32 or not 1 <= self.count <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    key = u.src
    if key not in self.int32_divmod:
      values = tuple(self._operand(src, dtypes.int) for src in key)
      if any(value.layout is not RKLayout.INT32 for value in values): raise _RKGenericReject
      raw = tuple(tuple(part.arg for part in self._raw_parts(value)) for value in values)
      constants = {value:self._i16_const(value) for value in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256)}
      signs = tuple(self._i16_positive_over(value[3], 127) for value in raw)
      numerator, denominator = (self._i16_twos_complement(value, sign) for value,sign in zip(raw, signs))
      denominator_nonzero = _reduce_rows(self.ew_ops, [self._i16_clamp_one(value) for value in denominator],
                                         self.count, _EW_CFG[Ops.MAX], int16=True)
      numerator_bits = tuple(bit for byte in numerator for bit in
                             _int16_byte_bits(self.ew_ops, lambda:self._scratch(dtypes.int16, RKLayout.INT16).arg,
                                              constants, byte, self.count))
      remainder, quotient = [constants[0]]*4, [constants[0]]*4
      for bit_index in range(31, -1, -1):
        shifted:list[RKArg] = []
        incoming = numerator_bits[bit_index]
        for byte in remainder:
          carry = self._i16_positive_over(byte, 127)
          doubled = self._i16(byte, byte, _EW_CFG[Ops.ADD])
          wrapped = self._i16(doubled, self._i16(carry, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
          shifted.append(self._i16(wrapped, incoming, _EW_CFG[Ops.ADD])); incoming = carry
        remainder = shifted
        greater, equal = constants[0], constants[1]
        for left,right in zip(reversed(remainder), reversed(denominator)):
          diff = self._i16(left, right, _EW_CFG[Ops.SUB])
          positive = self._i16(self._i16(diff, constants[0], _EW_CFG[Ops.MAX]), constants[1], _EW_CFG_MIN)
          greater = self._i16(greater, self._i16(equal, positive, _EW_CFG_MIN), _EW_CFG[Ops.MAX])
          magnitude = self._i16(diff, diff, _EW_CFG_ABS)
          byte_equal = self._i16(constants[1], self._i16(magnitude, constants[1], _EW_CFG_MIN), _EW_CFG[Ops.SUB])
          equal = self._i16(equal, byte_equal, _EW_CFG_MIN)
        ge = self._i16(self._i16(greater, equal, _EW_CFG[Ops.MAX]), denominator_nonzero, _EW_CFG_MIN)
        borrow, reduced = constants[0], []
        for left,right in zip(remainder, denominator):
          partial = self._i16(left, self._i16(right, ge, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
          delta = self._i16(partial, borrow, _EW_CFG[Ops.SUB])
          borrow = self._i16_clamp_one(self._i16(constants[0], delta, _EW_CFG[Ops.SUB]))
          reduced.append(self._i16(delta, self._i16(borrow, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD]))
        remainder = reduced
        byte_index, weight = bit_index >> 3, 1 << (bit_index&7)
        quotient[byte_index] = self._i16(quotient[byte_index], self._i16(ge, constants[weight], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
      self.int32_divmod[key] = tuple(quotient), tuple(remainder), signs[0], self._i16_xor(signs[0], signs[1])
    quotient_raw, remainder_raw, remainder_sign, quotient_sign = self.int32_divmod[key]
    return self._pack_raw(u, self._i16_twos_complement(
      quotient_raw if u.op is Ops.CDIV else remainder_raw, quotient_sign if u.op is Ops.CDIV else remainder_sign), RKLayout.INT32)

  def _int32_compare(self, u:UOp) -> RKValue:
    def operand(src:UOp) -> RKValue:
      value = self._static_int32(src) if src in self.static_nodes else self.lower(src)
      if value.layout is not RKLayout.INT32: raise _RKGenericReject
      return value
    lhs, rhs = (operand(src) for src in u.src)
    lhs_bytes, rhs_bytes = self._raw_parts(lhs), self._raw_parts(rhs)
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    constants = {value:self._constant(UOp.const(value, dtypes.int16)).arg for value in (0, 1, 127, 128, 256)}
    if u.op is Ops.CMPLT:
      mask = _int32_less_mask(self.ew_ops, allocate, constants, [value.arg for value in lhs_bytes[::-1]],
                              [value.arg for value in rhs_bytes[::-1]], self.count)
    else:
      equal = constants[1]
      for left,right in zip(lhs_bytes, rhs_bytes):
        byte_equal = _ew_native_int16_eq_mask(self.ew_ops, allocate, left.arg, right.arg, constants[1], self.count)
        selected = allocate()
        self.ew_ops.append(RKEWOp(selected, equal, byte_equal, self.count, _EW_CFG[Ops.MUL], int16_input=True, int16_output=True))
        equal = selected
      if u.op is Ops.CMPEQ: mask = equal
      elif u.op is Ops.CMPNE:
        mask = allocate()
        self.ew_ops.append(RKEWOp(mask, constants[1], equal, self.count, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True))
      else: raise _RKGenericReject
    return RKValue(mask, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_equality(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 equality through raw bytes and native INT16 arithmetic, without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    lhs_low,lhs_high,lhs_nan = self._fp16_component_values(values[0])
    rhs_low,rhs_high,rhs_nan = self._fp16_component_values(values[1])
    low_equal = _ew_native_int16_eq_mask(self.ew_ops, allocate, lhs_low.arg, rhs_low.arg, constants[1].arg, self.count)
    high_equal = _ew_native_int16_eq_mask(self.ew_ops, allocate, lhs_high, rhs_high, constants[1].arg, self.count)
    either_nan, numeric, bits_equal, equal = (allocate() for _ in range(4))
    integer = dict(int16_input=True, int16_output=True)
    self.ew_ops.extend((RKEWOp(either_nan, lhs_nan, rhs_nan, self.count, _EW_CFG[Ops.MAX], **integer),
                        RKEWOp(numeric, constants[1].arg, either_nan, self.count, _EW_CFG[Ops.SUB], **integer),
                        RKEWOp(bits_equal, low_equal, high_equal, self.count, _EW_CFG[Ops.MUL], **integer),
                        RKEWOp(equal, bits_equal, numeric, self.count, _EW_CFG[Ops.MUL], **integer)))
    if u.op is Ops.CMPNE:
      unequal = allocate()
      self.ew_ops.append(RKEWOp(unequal, constants[1].arg, equal, self.count, _EW_CFG[Ops.SUB], **integer))
      equal = unequal
    return RKValue(equal, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_component_values(self, value:RKValue) -> tuple[RKValue, RKArg, RKArg]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if value.arg in self.fp16_components: return self.fp16_components[value.arg]
    low, high = self._raw_parts(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 123, 124, 127, 128)}
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    clean_high,nan = _fp16_high_and_nan(self.ew_ops, allocate, high.arg, low.arg,
      constants[0].arg, constants[1].arg, constants[123].arg, constants[124].arg,
      constants[127].arg, constants[128].arg, self.count)
    self.fp16_components[value.arg] = low, clean_high, nan
    return self.fp16_components[value.arg]

  def _fp16_ordered_values(self, value:RKValue) -> tuple[RKArg, RKArg]:
    """Map a classified FP16 lane to two unsigned bytes whose lexical order is IEEE numeric order."""
    if value.arg in self.fp16_ordered: return self.fp16_ordered[value.arg]
    low, clean_high, _ = self._fp16_component_values(value)
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128, 255)}
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    integer = dict(int16_input=True, int16_output=True)
    sign_delta, sign_positive, sign = (allocate() for _ in range(3))
    positive_high, negative_high, high_delta, high_selected, ordered_high = (allocate() for _ in range(5))
    negative_low, low_delta, low_selected, ordered_low = (allocate() for _ in range(4))
    self.ew_ops.extend((
      RKEWOp(sign_delta, clean_high, constants[127].arg, self.count, _EW_CFG[Ops.SUB], **integer),
      RKEWOp(sign_positive, sign_delta, constants[0].arg, self.count, _EW_CFG[Ops.MAX], **integer),
      RKEWOp(sign, sign_positive, constants[1].arg, self.count, _EW_CFG_MIN, **integer),
      RKEWOp(positive_high, clean_high, constants[128].arg, self.count, _EW_CFG[Ops.ADD], **integer),
      RKEWOp(negative_high, constants[255].arg, clean_high, self.count, _EW_CFG[Ops.SUB], **integer),
      RKEWOp(high_delta, negative_high, positive_high, self.count, _EW_CFG[Ops.SUB], **integer),
      RKEWOp(high_selected, sign, high_delta, self.count, _EW_CFG[Ops.MUL], **integer),
      RKEWOp(ordered_high, positive_high, high_selected, self.count, _EW_CFG[Ops.ADD], **integer),
      RKEWOp(negative_low, constants[255].arg, low.arg, self.count, _EW_CFG[Ops.SUB], **integer),
      RKEWOp(low_delta, negative_low, low.arg, self.count, _EW_CFG[Ops.SUB], **integer),
      RKEWOp(low_selected, sign, low_delta, self.count, _EW_CFG[Ops.MUL], **integer),
      RKEWOp(ordered_low, low.arg, low_selected, self.count, _EW_CFG[Ops.ADD], **integer)))
    self.fp16_ordered[value.arg] = ordered_high, ordered_low
    return self.fp16_ordered[value.arg]

  def _fp16_less(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 less-than as an ordered raw-byte comparison without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    constants = {number:self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1)}
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    integer = dict(int16_input=True, int16_output=True)
    ordered = tuple(self._fp16_ordered_values(value) for value in values)
    nan = tuple(self._fp16_component_values(value)[2] for value in values)
    less = _ordered_byte_less(self.ew_ops, allocate, constants[0].arg, constants[1].arg, ordered[0], ordered[1], self.count)
    either_nan, numeric, result = (allocate() for _ in range(3))
    self.ew_ops.extend((RKEWOp(either_nan, nan[0], nan[1], self.count, _EW_CFG[Ops.MAX], **integer),
                       RKEWOp(numeric, constants[1].arg, either_nan, self.count, _EW_CFG[Ops.SUB], **integer),
                       RKEWOp(result, less, numeric, self.count, _EW_CFG[Ops.MUL], **integer)))
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
      if preferred not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
      values = [self._static(src, preferred) if value is None else self._coerce_bool(value, preferred)
                for src,value in zip(u.src, dynamic)]
      selector, yes, no = values
      data_dtype, data_layout = ((dtypes.int16, RKLayout.INT16) if preferred is RKLayout.BOOL_INT16 else
                                 (dtypes.half, RKLayout.FP16))
      one = self._constant(UOp.const(1, data_dtype))
      selected_yes, inverse, selected_no = (self._scratch(data_dtype, data_layout) for _ in range(3))
      self._emit(selected_yes, selector, yes, _EW_CFG[Ops.MUL])
      self._emit(inverse, one, selector, _EW_CFG[Ops.SUB])
      self._emit(selected_no, inverse, no, _EW_CFG[Ops.MUL])
      result = self._emit(self._dst(u, dtypes.bool, preferred), selected_yes, selected_no, _EW_CFG[Ops.ADD])
      return RKValue(result.arg, dtypes.bool, self.count, preferred)
    if u is self.root and u.dtype.scalar() is dtypes.int and all(
      arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.int for arm in u.src[1:]
    ):
      yes_int, no_int = (int(arm.arg) for arm in u.src[1:])
      try: exact = all(_static_cast(value, dtypes.half) == value for value in (no_int, yes_int-no_int))
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
          selector = tuple(bool(x) for x in _static_values(self.out_index, node.src[0], self.count, int))
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
        self.post_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count, offsets=offsets,
          partial=bool(partial), dst_kind=RKBufferKind.ARG, src_kind=value.arg.kind, itemsize=itemsize))
      return RKValue(self.out, dtype, self.count, expected)
    if (recipe:=_fold_where_abs(u)) is not None:
      return self.lower(recipe)
    if (recipe:=_fold_ordered_where(u)) is not None:
      return self.lower(recipe)
    if (recipe:=_fold_threshold_where(u)) is not None:
      return self.lower(recipe)
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
      if math.isnan(float(u.src[1+inf_index].arg)):
        zero, one = self._constant(UOp.const(0.0, dtypes.half)), self._constant(UOp.const(1.0, dtypes.half))
        if inf_index == 0:
          denominator = self._scratch(dtypes.half, RKLayout.FP16)
          self._emit(denominator, one, selector, _EW_CFG[Ops.SUB])
        else: denominator = selector
        correction = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(correction, zero, denominator, _EW_CFG[Ops.FDIV])
        return self._emit(self._dst(u, dtypes.half, RKLayout.FP16), finite, correction, _EW_CFG[Ops.ADD])
      one, sign = self._constant(UOp.const(1.0, dtypes.half)), self._constant(
        UOp.const(math.copysign(1.0, float(u.src[1+inf_index].arg)), dtypes.half))
      if inf_index == 0:
        denominator = self._scratch(dtypes.half, RKLayout.FP16)
        self._emit(denominator, one, selector, _EW_CFG[Ops.SUB])
      else: denominator = selector
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
      selected_yes, inverse, selected_no = (self._scratch(dtype, yes.layout) for _ in range(3))
      self._emit(selected_yes, selector, yes, _EW_CFG[Ops.MUL])
      self._emit(inverse, one, selector, _EW_CFG[Ops.SUB])
      self._emit(selected_no, inverse, no, _EW_CFG[Ops.MUL])
      return self._emit(self._dst(u, dtype, yes.layout), selected_yes, selected_no, _EW_CFG[Ops.ADD])
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
    tagged:dict[UOp, UOp] = {}
    for node in recipe.toposort():
      tagged[node] = node.replace(src=tuple(tagged[src] for src in node.src),
        arg=_NATIVE_PRECISE_ADD if node.op is Ops.ADD and node.arg is None else node.arg)
    recipe = tagged[recipe]
    value = self.lower(recipe)
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if u is self.root and value.arg != self.out:
      value = self._emit(RKValue(self.out, dtypes.half, self.count, RKLayout.FP16), value, value, _EW_CFG[Ops.MAX])
    return value

  def lower(self, u:UOp) -> RKValue:
    if u in self.values: return self.values[u]
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif (dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) and u in self.static_nodes and
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
        self.post_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count,
          base=value.arg.addend//2, axes=((1, self.count, 1),), dst_kind=RKBufferKind.ARG,
          src_kind=value.arg.kind, itemsize=2))
        value = RKValue(self.out, dtype, self.count, value.layout)
    elif u.op is Ops.CAST and len(u.src) == 1:
      source_dtype = u.src[0].dtype.scalar()
      int_range = _exact_int_range(u.src[0], self.int_ranges) if source_dtype is dtypes.int else None
      if dtype is dtypes.half and source_dtype is dtypes.float:
        source = self._load(u.src[0]) if u.src[0].op is Ops.LOAD else self.lower(_fp32_expr_to_half(u.src[0]))
      elif dtype is dtypes.half and source_dtype is dtypes.int and int_range is not None and \
           -_FP16_EXACT_INTEGER <= int_range[0] <= int_range[1] <= _FP16_EXACT_INTEGER:
        source = self.lower(_int_fp16_expr(u.src[0]))
      else: source = self.lower(u.src[0])
      if dtype is dtypes.half and source.layout is RKLayout.INT32:
        value = self._narrow_int32(source)
      elif source.layout is RKLayout.BOOL_INT16 and (dtype is dtypes.half or
                                                     dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16):
        value = self.lower(u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half)))
        if dtype is dtypes.int:
          if value.layout is not RKLayout.FP16: raise _RKGenericReject
          value = RKValue(value.arg, dtype, self.count, RKLayout.INT_FP16)
      elif (source.layout is RKLayout.FP16 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.half) or
            dtype is dtypes.half and source.layout in (RKLayout.BOOL_MASK, RKLayout.INT_FP16)):
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      elif (dtype is dtypes.int16 and source.layout in (RKLayout.INT16, RKLayout.BOOL_INT16) or
            dtype is dtypes.int and source.layout is RKLayout.BOOL_INT16 and self.int_layout is RKLayout.INT16):
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.int and source.layout in (RKLayout.FP16, RKLayout.BOOL_MASK):
        if self.int_layout is RKLayout.INT_FP16:
          if source.layout is RKLayout.BOOL_MASK: value = RKValue(source.arg, dtype, self.count, self.int_layout)
          else: value = RKValue(self.lower(_int_fp16_expr(u)).arg, dtype, self.count, self.int_layout)
        elif self.int_layout is RKLayout.INT16:
          value = self._scratch(dtype, self.int_layout)
          self.ew_ops.append(RKEWOp(value.arg, source.arg, source.arg, self.count, _EW_CFG[Ops.MAX], stateful=True, int16_output=True))
        else: raise _RKGenericReject
      elif dtype is dtypes.int and source_dtype is dtypes.uint and source.layout is RKLayout.INT32:
        value = RKValue(source.arg, dtype, self.count, source.layout)
      elif dtype is dtypes.int: value = self._widen_int16(u, source)
      else: raise _RKGenericReject(f"cast {source.layout.name}->{dtype}")
    elif u.op is Ops.ADD and dtype is dtypes.half and u.arg is None and self.accurate_adds:
      try: value = self.lower(_accurate_add_recipe(u))
      except _RKGenericReject: value = self._alu(u)
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL): value = self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype is dtypes.bool:
      if all(src.dtype.scalar() is dtypes.bool for src in u.src): value = self._bool_binary(u)
      elif (ieee_recipe:=_ieee_comparison_mask(u)) is not None: value = self._ieee_bool(ieee_recipe)
      else: value = self._bool_binary(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype in (dtypes.int16, dtypes.int): value = self._integer_bitwise(u)
    elif u.op in (Ops.SHL, Ops.SHR) and dtype in (dtypes.int, dtypes.uint): value = self._int32_shift(u)
    elif u.op in (Ops.CDIV, Ops.CMOD) and dtype is dtypes.int:
      if self.int_layout is RKLayout.INT_FP16: value = RKValue(self.lower(_int_fp16_expr(u)).arg, dtype, self.count, self.int_layout)
      else: value = self._int32_divmod(u)
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN): value = self._math(u)
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    self.values[u] = value
    return value

  def finish(self) -> RKImage:
    nodes = self.root.toposort()
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
    elif dtype is dtypes.bool and result.layout is RKLayout.BOOL_MASK:
      tiles = self._scratch(dtypes.int, RKLayout.INT32, _int32_tiles_bytes(self.count))
      self.ew_ops.append(RKEWOp(self.out, result.arg, tiles.arg, self.count, _EW_CFG[Ops.MAX],
        stateful=True, int32_output=True, bool_output=True))
    elif dtype is dtypes.bool and result.layout is RKLayout.BOOL_INT16:
      self.post_gathers.append(_int16_low_bytes(result.arg, self.out_param.arg.slot, self.count))
    elif dtype is dtypes.int and result.layout is RKLayout.INT32:
      if result.arg != self.out:
        self.post_gathers.append(RKGather(result.arg.index, self.out_param.arg.slot, self.count,
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
    gather_after = min((g.after for g in self.mid_gathers if g.after >= 0), default=0)
    image = RKImage(RKTarget.RK3588, tuple(self.scratch), constants, gathers=tuple(self.gathers), ew_ops=tuple(self.ew_ops),
                    mid_gathers=tuple(self.mid_gathers), gather_after=gather_after, post_gathers=tuple(self.post_gathers),
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

def _expand_math_uops(root:UOp, *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  bounded_recipes = len(root.toposort()) <= _MAX_OPTIONAL_RECIPE_NODES
  composite_math = _fold_inverse_hyperbolic(root) if bounded_recipes else None
  if composite_math is None and bounded_recipes: composite_math = _fold_atan(root)
  cache:dict[UOp, UOp] = {}
  exact_static_selection = root.op is Ops.WHERE and _is_static_expr(root.src[0]) and not any(
    node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half, dtypes.float) and not math.isfinite(float(node.arg))
    for node in root.toposort())
  def physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp:
    placeholders = {source:UOp.param(-index-1, source.dtype, ()) for index,source in enumerate(opaque)}
    rewritten = _fp16_rewrite(list(UOp(Ops.SINK, src=(recipe.substitute(placeholders),)).toposort()))
    if not rewritten or rewritten[-1].op is not Ops.SINK or len(rewritten[-1].src) != 1: raise _RKGenericReject
    tagged:dict[UOp, UOp] = {}
    def tag_adds(u:UOp) -> UOp:
      if u in tagged: return tagged[u]
      mapped = u.replace(src=tuple(tag_adds(src) for src in u.src),
                         arg=_NATIVE_PRECISE_ADD if u.op is Ops.ADD and u.arg is None else u.arg)
      tagged[u] = mapped
      return mapped
    tagged_root = tag_adds(rewritten[-1].src[0])
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
   lambda root:_expand_math_uops(root)),
])

def _finite_max_neutral_selectors(root:UOp) -> UOp:
  """Use the canonical finite FP16 MAX neutral for selected negative-infinity padding."""
  if root.op is not Ops.MAX: return root
  cache:dict[UOp, UOp] = {}
  for node in root.toposort():
    src = tuple(cache[x] for x in node.src)
    if (node.op is Ops.WHERE and src[1].op is Ops.CONST and src[1].dtype.scalar() in (dtypes.half, dtypes.float) and
        math.isinf(float(src[1].arg)) and float(src[1].arg) < 0.0): src = (src[0], src[1].const_like(-65504.0), src[2])
    cache[node] = node.replace(src=src)
  return cache[root]

def _finite_int_max_neutrals(root:UOp) -> UOp:
  """Canonicalize INT32_MIN only while it acts as a structural MAX neutral in exact scratch arithmetic."""
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
      cache[key] = u.replace(src=tuple(cache[(src, active)] for src in u.src))
    else:
      stack.append((u, under_max, True))
      stack.extend((src, active, False) for src in reversed(u.src))
  return cache[(root, False)]

def _substitute_static_ranges(root:UOp, replacements:dict[UOp, UOp]) -> UOp:
  cache:dict[UOp, UOp] = {}
  def rewrite(u:UOp) -> UOp:
    if u in replacements: return replacements[u]
    if u in cache: return cache[u]
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    cache[u] = mapped
    return mapped
  return rewrite(root)

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
      envs = _iter_range_env(ranges, None, False)
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
  local_cache:dict[UOp, tuple[UOp, ...]] = {}
  local_loads = list(_semantic_local_loads(root, local_cache))
  buffers = {_local_buffer(load) for load in local_loads}
  if not local_loads: return root
  if None in buffers: raise _RKGenericReject
  typed_buffers = typing_cast(set[UOp], buffers)
  definitions:dict[UOp, _RKStaticLocalDef] = {}
  if len(typed_buffers) == 1:
    discovered, pending = set(typed_buffers), list(typed_buffers)
    try:
      while pending:
        buffer = pending.pop()
        definitions.update(_static_local_defs(uops, {buffer}))
        for expr in (definitions[buffer].initial, definitions[buffer].term):
          for load in _semantic_local_loads(expr, local_cache):
            dependency = _local_buffer(load)
            if dependency is None: raise _RKGenericReject
            if dependency not in discovered: discovered.add(dependency); pending.append(dependency)
      typed_buffers = discovered
    except _RKGenericReject: definitions = {}
  if len(typed_buffers) > 1:
    if not definitions: definitions = _static_local_defs(uops, typed_buffers)
    expanded:dict[tuple[UOp, tuple[tuple[UOp, int], ...]], UOp] = {}
    active:set[tuple[UOp, tuple[tuple[UOp, int], ...]]] = set()
    budget = [_MAX_STATIC_LOCAL_STEPS]
    def expand_dependencies(expr:UOp, owner:UOp, env:dict[UOp, int]) -> UOp:
      substitutions = {axis:axis.const_like(value) for axis,value in env.items()}
      substitutions.update({load:expand_buffer(buffer, env) for load in _semantic_local_loads(expr, local_cache)
                            if (buffer:=_local_buffer(load)) is not None and buffer is not owner})
      return _substitute_static_ranges(expr, substitutions)
    def expand_buffer(buffer:UOp, env:dict[UOp, int]) -> UOp:
      key = buffer, tuple(sorted(env.items(), key=lambda item:item[0].key))
      if key in expanded: return expanded[key]
      if key in active: raise _RKGenericReject
      active.add(key)
      if buffer not in definitions: definitions.update(_static_local_defs(uops, {buffer}))
      definition = definitions[buffer]
      if not definition.loops or any(loop.src[0].op is not Ops.CONST or not 0 <= int(loop.src[0].arg) <= _MAX_GENERIC_UNROLL
                                     for loop in definition.loops):
        raise _RKGenericReject
      iterations = math.prod(int(loop.src[0].arg) for loop in definition.loops)
      if iterations > _MAX_GENERIC_UNROLL or iterations > budget[0]: raise _RKGenericReject
      budget[0] -= iterations
      accumulator = expand_dependencies(definition.initial, buffer, env)
      for loop_env in _iter_range_env(list(definition.loops), None, False):
        term = expand_dependencies(definition.term, buffer, {**env, **loop_env})
        accumulator = UOp(definition.update_op, buffer.dtype, src=(accumulator, term))
      expanded[key] = accumulator
      active.remove(key)
      if len(expanded[key].toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
      return expanded[key]
    substitutions = {load:expand_buffer(typing_cast(UOp, _local_buffer(load)), {}) for load in local_loads}
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
  reduced = initializers[0]
  for env in _iter_range_env(ranges, None, False):
    reduced = UOp(update.op, update.dtype, src=(reduced, _substitute_static_ranges(term, {r:r.const_like(env[r]) for r in ranges})))
  substitutions = {load:reduced for load in local_loads if _local_buffer(load) is buffer}
  return root.substitute(substitutions)

@dataclass(frozen=True)
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
  """Materialize independent scalar FP32 local ADD programs, then execute their shared output UOps."""
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
  if any(buffer.dtype.scalar() is not dtypes.float or definition.update_op is not Ops.ADD or
         definition.initial.op is not Ops.CONST or float(definition.initial.arg) != 0.0 or not definition.loops or
         any(loop.src[0].op is not Ops.CONST or int(loop.src[0].arg) <= 0 for loop in definition.loops)
         for buffer,definition in definitions.items()): return None
  if any(semantic_local_loads(definition.term) for definition in definitions.values()): return None
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
    if mapped is None or (reduced:=_finish_mapped_add_reduction(mapped, fake_slot, 1, groups, 1.0)) is None: return None
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
  if post is None or staged is None or (appended:=_append_inplace_image(staged, post)) is None: return None
  scratch_base = len(appended.scratch)
  slot_to_scratch = {fake.arg.slot:scratch_base+i for i,fake in enumerate(sources.values())}
  aliases = {slot:RKArg(RKBufferKind.SCRATCH, target) for slot,target in slot_to_scratch.items()}
  return replace(_alias_image_args(appended, aliases), scratch=appended.scratch+tuple(RKScratch(64) for _ in sources))

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
    index_envs = _iter_range_env(list(index_def.loops), None, False)
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
  if child is None or child.fill is not None or child.host_gathers or child.host_scatters: return None

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
  mid.extend(map_gather(gather, after=len(ops)) for gather in child.post_gathers)

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
  integer = dict(int16_input=True, int16_output=True)
  either_nan, numeric, bits_equal, equal = (alloc() for _ in range(4))
  ops.extend((RKEWOp(either_nan, lhs_nan, rhs_nan, total, _EW_CFG[Ops.MAX], **integer),
              RKEWOp(numeric, constants[1], either_nan, total, _EW_CFG[Ops.SUB], **integer),
              RKEWOp(bits_equal, low_equal, high_equal, total, _EW_CFG[Ops.MUL], **integer),
              RKEWOp(equal, bits_equal, numeric, total, _EW_CFG[Ops.MUL], **integer)))
  coordinate_values, weighted = allocate(), allocate()
  gathers.append(RKGather(out_param.arg.slot, coordinate_values.index, total, values=coordinates))
  ops.append(RKEWOp(weighted, equal, coordinate_values, total, _EW_CFG[Ops.MUL], **integer))
  weighted_spaced = allocate(total*scalar_stride//2)
  mid.append(RKGather(weighted.index, weighted_spaced.index, total, axes=((1, total, 1),), dst_stride=scalar_stride//2,
                      src_kind=RKBufferKind.SCRATCH, after=len(ops)))
  selected = _reduce_rows(ops, [RKArg(weighted_spaced.kind, weighted_spaced.index, lane*scalar_stride) for lane in range(total)],
                          1, _EW_CFG[Ops.MAX], int16=True)
  result = selected
  if slope != 1:
    scale = allocate(1); gathers.append(RKGather(out_param.arg.slot, scale.index, 1, values=(_int16_bits(slope),)))
    scaled = allocate(1); ops.append(RKEWOp(scaled, result, scale, 1, _EW_CFG[Ops.MUL], **integer)); result = scaled
  if baseline:
    offset = allocate(1); gathers.append(RKGather(out_param.arg.slot, offset.index, 1, values=(_int16_bits(baseline),)))
    translated = allocate(1); ops.append(RKEWOp(translated, result, offset, 1, _EW_CFG[Ops.ADD], **integer)); result = translated
  zero = allocate(1); gathers.append(RKGather(out_param.arg.slot, zero.index, 1, values=(0,)))
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), result, zero, 1, _EW_CFG[Ops.ADD],
                    int16_input=True, int32_output=True))
  image = RKImage(RKTarget.RK3588, tuple(scratch), child.constants, gathers=tuple(gathers), ew_ops=tuple(ops),
                  mid_gathers=tuple(mid), gather_after=child.gather_after)
  return image if all(len(items) <= _RKIMAGE_U16_MAX for items in
                      (image.scratch, image.gathers, image.ew_ops, image.mid_gathers)) else None

def _lower_host_scatter(uops:list[UOp]) -> RKImage|None:
  """Lower a direct dynamic STORE as raw last-writer host address materialization."""
  if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1" or \
     (output:=_output_store(uops, (dtypes.half, dtypes.int16))) is None or len(output[0].src) != 2: return None
  store, out_param, out_count, dynamic_index, value = output
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
  return RKImage(RKTarget.RK3588, host_scatters=(address,))

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
  if (float_output:=_output_store(uops, dtypes.float)) is not None and \
     (integer_cast:=_lower_integer_fp32_cast(float_output)) is not None: return integer_cast
  if (int_output:=_output_store(uops, dtypes.int)) is not None:
    if (raw_bitcast:=_lower_raw_fp16_bitcast(int_output)) is not None: return raw_bitcast
    if (fp16_cast:=_lower_fp16_int32_cast(int_output)) is not None: return fp16_cast
  if (bool_output:=_output_store(uops, dtypes.bool)) is not None and \
     (nonzero:=_fp16_nonzero_mask(bool_output[4])) is not None: return _typed_half_image(bool_output, nonzero, True, bool_output=True)
  if (byte_output:=_output_store(uops, dtypes.uchar)) is not None and \
     (fp16_byte_cast:=_lower_fp16_uint8_cast(byte_output)) is not None: return fp16_byte_cast
  if (bool_loop_output:=_output_store(uops, dtypes.bool, allow_local=True)) is not None:
    if (bool_loop_reduction:=_lower_loop_bool_reduction(uops, bool_loop_output)) is not None: return bool_loop_reduction
    if (grouped_bool_reduction:=_lower_grouped_bool_reduction(uops, bool_loop_output)) is not None: return grouped_bool_reduction
  for dtype in (dtypes.half, dtypes.int16, dtypes.int):
    if (direct_load:=_output_store(uops, dtype)) is None: continue
    if (image:=_lower_direct_dynamic_typed_load(direct_load, dtype)) is not None: return image
    if (image:=_lower_dynamic_multi_index_typed_load(direct_load, dtype)) is not None: return image
    root = direct_load[4]
    if root.op is Ops.WHERE and root.src[1].op is Ops.LOAD and root.src[2].op is Ops.CONST and \
       (folded_load:=_fold_masked_load(root.src[0], root.src[1], root.src[2])) is not None and \
       (image:=_lower_direct_dynamic_typed_load((*direct_load[:4], folded_load), dtype)) is not None: return image
  for dtype in (dtypes.int16, dtypes.int):
    if ((gated_load:=_output_store(uops, dtype)) is not None and
        (image:=_lower_dynamic_load_with_bool_total_gate(gated_load, dtype)) is not None): return image
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
              converted = _expand_math_uops(storage_root) if source.op is Ops.SIN else _canonical_half_storage(source)
              storage_sink = sink.substitute({storage_root:converted})
          except _RKGenericReject: pass
      storage_sink = graph_rewrite(storage_sink, _pm_fp32_sin_storage, name="rockchip fp32 sin storage")
      storage_uops = list(graph_rewrite(storage_sink, _pm_generic_storage_precision,
                                        name="rockchip generic storage precision").toposort())
  if vectorize_reductions and (mul_add:=_lower_vectorized_mul_add_reduction(uops)) is not None: return mul_add
  if vectorize_reductions and (scalar_loop:=_loop_reduction_match(uops)) is not None:
    if (dot_reduction:=_lower_dot_loop_reduction(scalar_loop)) is not None: return dot_reduction
    if not (scalar_loop.post_sqrt or scalar_loop.post_reciprocal or scalar_loop.post_cuberoot) and \
       (scalar_reduction:=_lower_scalar_loop_reduction(scalar_loop)) is not None:
      return scalar_reduction
  mapped_loop = _lower_mapped_add_loop_reduction(uops) if vectorize_reductions else None
  if mapped_loop is not None: return mapped_loop
  if vectorize_reductions and (reduction:=_lower_vectorized_unrolled_add_reduction(uops)) is not None: return reduction
  if storage_uops is not None: uops = storage_uops
  if (output:=_output_store(uops, (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool), allow_local=True)) is None or \
     len(output[0].src) != 2:
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage(RKTarget.RK3588)
  if output[1].dtype.scalar() is dtypes.bool:
    if (bounds_mask:=_lower_int32_bounds_mask(output)) is not None: return bounds_mask
  if output[1].dtype.scalar() is dtypes.int:
    if (predicate_total:=_lower_unrolled_fp16_predicate_total(output)) is not None: return predicate_total
    if (predicate_prefix:=_lower_unrolled_fp16_prefix_count(output)) is not None: return predicate_prefix
    for source_dtype in (dtypes.int16, dtypes.int):
      if (coordinates:=_lower_bounded_integer_predicate_coordinates(output, source_dtype)) is not None: return coordinates
    if (prefix_sum:=_lower_unrolled_int_prefix_sum(output)) is not None: return prefix_sum
    if (predicate_total:=_lower_loop_fp16_predicate_total(uops, output)) is not None: return predicate_total
    if (predicate_prefix:=_lower_loop_fp16_prefix_count(uops, output)) is not None: return predicate_prefix
    if (equality_add:=_lower_loop_int32_equality_add(uops, output)) is not None: return equality_add
    if (prefix_add:=_lower_loop_int32_prefix_add(uops, output)) is not None: return prefix_add
  try:
    if (_contiguous_output_samples(output[3], output[2]) is None and
        _static_int_vector(output[3], output[3], output[2]) != tuple(range(output[2]))): return None
    reduced = _unroll_static_reduces(output[4]) if any(u.op is Ops.REDUCE for u in uops) else output[4]
    local_root = _unroll_static_local(uops, output, reduced)
    root = _finite_int_max_neutrals(_finite_max_neutral_selectors(local_root))
    root_nodes = root.toposort()
    defer_nodes = storage_uops if storage_uops is not None else root_nodes
    defer_math = len(defer_nodes) > 256
    if not recipes_ready and not defer_math:
      root = _expand_math_uops(root, accurate_adds=storage_uops is None or storage_product_adds)
    expanded_nodes = root.toposort()
    if len(expanded_nodes) > _MAX_GENERIC_EXPANDED_NODES:
      if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(expanded_nodes)}")
      return None
    if root is not output[4]:
      store = output[0].replace(src=(output[0].src[0], root))
      uops = list(UOp(Ops.SINK, src=(store,)).toposort())
      if (output:=_output_store(uops, (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool))) is None: return None
    image = RKContext(output, accurate_adds=(not recipes_ready and (storage_uops is None or storage_product_adds) and
                                             len(expanded_nodes) <= _MAX_OPTIONAL_RECIPE_NODES and
                                             not _has_runtime_address(output[4]))).finish()
    image_u16_counts = (len(image.scratch), len(image.gathers)+len(image.mid_gathers)+len(image.post_gathers),
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

def _fold_trunc(x:UOp) -> UOp:
  """Compose truncation from native floor/ceil without mask multiplication on infinities."""
  source, zero = x.src[0], UOp.const(0.0, dtypes.half)
  positive = source.alu(Ops.MAX, zero)
  negative = zero.alu(Ops.SUB, zero.alu(Ops.SUB, source).alu(Ops.MAX, zero))
  floor = UOp(Ops.MAX, x.dtype, src=(positive, positive), arg=_NATIVE_FLOOR)
  ceil = UOp(Ops.MAX, x.dtype, src=(negative, negative), arg=_NATIVE_CEIL)
  return floor.alu(Ops.ADD, ceil)

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

def _fp32_where_to_fp16(x:UOp) -> UOp|None:
  """Choose the backend's FP16 physical arithmetic representation for a dynamic float WHERE."""
  return None if _is_static_expr(x) else \
    UOp(Ops.WHERE, dtypes.half, src=(x.src[0], x.src[1].cast(dtypes.half), x.src[2].cast(dtypes.half)), arg=x.arg)

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
])
_pm_generic_storage_precision = PatternMatcher([
  (UPat(Ops.WHERE, dtypes.float, name="x"), _fp32_where_to_fp16),
  (UPat((Ops.ADD, Ops.MUL), dtypes.float, name="x"), _fp32_alu_to_fp16),
  (UPat(Ops.ADD, dtypes.half, name="x"), _fold_relu_cap),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_minimum),
  (UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
  (UPat(Ops.MUL, dtypes.half, name="x"), _replace_infinite_multiply),
  (UPat(Ops.FDIV, dtypes.half, name="x"), _preserve_infinite_division_sign),
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
])
_pm_abs = PatternMatcher([(UPat(Ops.MUL, dtypes.half, name="x"), _fold_abs),
                          (UPat(Ops.WHERE, dtypes.half, name="x"), _fold_where_abs)])
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

def _dpu_sin(source:UOp) -> UOp:
  """Approximate FP16 SIN without LUTs using Cody-Waite reduction and an odd polynomial."""
  one = UOp.const(1.0, dtypes.half)
  period_split = (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125)
  if source.dtype.scalar() is dtypes.float:
    terms:list[UOp] = []
    residuals:list[UOp] = []
    def flatten(u:UOp) -> None:
      if u.op is Ops.ADD and u.dtype.scalar() is dtypes.float: flatten(u.src[0]); flatten(u.src[1])
      elif u.op is Ops.CONST:
        high = struct.unpack("<e", struct.pack("<e", float(u.arg)))[0]
        terms.append(UOp.const(high, dtypes.half))
        if (low:=float(u.arg)-high) != 0.0: residuals.append(UOp.const(low, dtypes.half))
      else: terms.append(_fp32_expr_to_half(u))
    flatten(source)
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
  magnitude = UOp(Ops.MAX, dtypes.half, src=(reduced, reduced), arg=_NATIVE_ABS)
  reflected = _positive_mask(magnitude.alu(Ops.SUB, UOp.const(math.pi/2, dtypes.half)))
  pi_minus = UOp.const(3.0, dtypes.half).alu(Ops.SUB, magnitude).alu(Ops.ADD, UOp.const(0.140625, dtypes.half)).alu(
    Ops.ADD, UOp.const(math.pi-3.140625, dtypes.half))
  angle = _mask_mul(magnitude, one.alu(Ops.SUB, reflected)).alu(Ops.ADD, _mask_mul(pi_minus, reflected))
  square = angle.alu(Ops.MUL, angle)
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

def _dpu_exp2(source:UOp) -> UOp:
  """Approximate FP16 EXP2 without LUTs using native FLOOR, Horner arithmetic, and exact exponent scaling."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  one = UOp.const(1.0, dtypes.half)
  bounded = _native_min(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(15.9921875, dtypes.half))
  integer = _native_floor(bounded)
  fraction = bounded.alu(Ops.SUB, integer)
  polynomial = UOp.const(0.0013333558, dtypes.half)
  for coefficient in (0.0096181291, 0.0555041087, 0.2402265069, 0.6931471806, 1.0):
    polynomial = polynomial.alu(Ops.MUL, fraction).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  result = polynomial.alu(Ops.MUL, _dpu_pow2_integer(integer))
  below = mask_fn(UOp.const(-24.0, dtypes.half).alu(Ops.SUB, source))
  above = mask_fn(source.alu(Ops.SUB, UOp.const(15.9921875, dtypes.half)))
  finite = _mask_mul(result, one.alu(Ops.SUB, below))
  return finite.alu(Ops.ADD, one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one))

def _dpu_exp2_nonpositive(source:UOp) -> UOp:
  """Approximate EXP2 for a known nonpositive finite-or-negative-infinite input without domain comparisons."""
  source = source.cast(dtypes.half)
  bounded = _native_min(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(0.0, dtypes.half))
  integer = _native_floor(bounded)
  fraction = bounded.alu(Ops.SUB, integer)
  polynomial = UOp.const(0.0013333558, dtypes.half)
  for coefficient in (0.0096181291, 0.0555041087, 0.2402265069, 0.6931471806, 1.0):
    polynomial = polynomial.alu(Ops.MUL, fraction).alu(Ops.ADD, UOp.const(coefficient, dtypes.half))
  return polynomial.alu(Ops.MUL, _dpu_pow2_integer(integer))

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

def _fp16_predecessor(value:float) -> float:
  """Return the previous positive binary16 value for inclusive threshold masks."""
  return struct.unpack("<e", struct.pack("<H", _fp16_bits(value)-1))[0]

def _dpu_log2(source:UOp) -> UOp:
  """Approximate FP16 LOG2 without LUTs using threshold exponent extraction and an atanh polynomial."""
  source = source.cast(dtypes.half)
  mask_fn = _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask
  zero, one = UOp.const(0.0, dtypes.half), UOp.const(1.0, dtypes.half)
  mantissa = _native_min(source.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)), UOp.const(65504.0, dtypes.half))
  exponent = zero
  for factor,shift in ((256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0)):
    mask = _finite_positive_mask(mantissa.alu(Ops.SUB, UOp.const(_fp16_predecessor(factor), dtypes.half)))
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
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  sink = graph_rewrite(sink, _pm_inverse_hyperbolic, name="rockchip inverse hyperbolic")
  sink = graph_rewrite(sink, _pm_atan, name="rockchip atan")
  sink = graph_rewrite(sink, _pm_sin, name="rockchip sin")
  sink = graph_rewrite(sink, _pm_masked_exp2, name="rockchip masked exp2")
  sink = graph_rewrite(sink, _pm_exp2, name="rockchip exp2")
  sink = graph_rewrite(sink, _pm_log2, name="rockchip log2")
  sink = graph_rewrite(sink, _pm_rsqrt, name="rockchip rsqrt")
  sink = graph_rewrite(sink, _pm_sqrt, name="rockchip sqrt")
  sink = graph_rewrite(sink, _pm_abs, name="rockchip abs")
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None,
                 Ops.FDIV: lambda: None, Ops.SQRT: lambda: None, Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_uop_program(uops)
    if image is None: image = _lower_uop_program(_fp16_rewrite(uops), recipes_ready=True)
    if image is None: raise RuntimeError("RKPLAN_REJECT:generic_uops " + repr([(i, u.op.name, str(u.dtype)) for i,u in enumerate(uops)]))
    return base64.b64encode(encode_image(image)).decode()

class RockchipBoolRenderer(RockchipRenderer):
  """Expose one 16-lane local bool tile that the renderer consumes as grouped DPU reduction work."""
  has_local, shared_max = True, 16
