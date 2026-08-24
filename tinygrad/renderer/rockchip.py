from __future__ import annotations
# ruff: noqa: E702
import base64, functools, heapq, itertools, math, os, struct
import numpy as np
from dataclasses import astuple, dataclass, replace
from enum import IntEnum
from typing import Any, Callable, Iterable, Mapping, NamedTuple, cast, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes, float_to_fp16
from tinygrad.helpers import ceildiv, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, UPat, PatternMatcher, exec_alu, graph_rewrite, python_alu
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 31
_HEADER = struct.Struct("<4sHHHHHHIIIIII")  # magic/version/target, scratch/gather/host counts, ops/constants, phase split, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBBBBiIIii"), struct.Struct("<IIi")
_HOST_ADDRESS, _EWOP, _CMAC = struct.Struct("<BBBBBHHHIIIIIIiiiiii"), struct.Struct("<BBHIIIIIIiii"), struct.Struct("<BBBBHHHiiiIII")
_ITEM_FORMAT, _RKIMAGE_U16_MAX = {1:"B", 2:"H", 4:"I"}, (1 << 16) - 1

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
  # Axes are (destination divisor, range limit, source stride); offsets provide the non-affine fallback.
  axes: tuple[tuple[int, int, int], ...] = (); offsets: tuple[int, ...] = (); fill_bits: int = 0
  # Compile-time values have no source argument; partial gathers preserve lanes populated by another gather.
  values: tuple[int, ...] = (); partial: bool = False
  # Scalar FP16 reductions use a destination stride of 32 for 64-byte spacing.
  dst_stride: int = 1; dst_addend: int = 0
  # A negative phase split uses RKImage.gather_after.
  dst_kind: RKBufferKind = RKBufferKind.SCRATCH; itemsize: int = 2; src_kind: RKBufferKind = RKBufferKind.ARG; after: int = -1

@dataclass(frozen=True)
class RKHostAddress:
  """Host-calculated raw-lane movement. It never owns numeric or reduction semantics."""
  src: RKArg; index: RKArg; dst: RKArg; count: int; src_count: int; dst_count: int
  itemsize: int = 2; index_itemsize: int = 4; fill_bits: int = 0; index_limit: int = 0; base: int = 0; index_scale: int = 1; lane_stride: int = 0

@dataclass(frozen=True)
class RKEWOp:
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False; int16_output: bool = False; int16_input: bool = False

@dataclass(frozen=True)
class RKCMAC:
  """One fixed FP16 matrix contraction with an optional terminal BS ReLU; gathers own only its physical packing."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; m: int; n: int; k: int; out_fp16: bool = True; relu: bool = False

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); ew_ops: tuple[RKEWOp, ...] = ()
  mid_gathers: tuple[RKGather, ...] = (); gather_after: int = 0; post_gathers: tuple[RKGather, ...] = ()
  host_gathers: tuple[RKHostAddress, ...] = (); host_scatters: tuple[RKHostAddress, ...] = ()
  cmac: RKCMAC|None = None

  @property
  def execution_class(self) -> RKExecutionClass: return RKExecutionClass(bool(self.host_gathers or self.host_scatters))

def _map_image_args(image:RKImage, fn:Callable[[RKArg], RKArg], *, map_value_src:bool=True) -> RKImage:
  def gather(v:RKGather) -> RKGather:
    src = fn(RKArg(v.src_kind,v.src_index)) if map_value_src or not v.values else RKArg(v.src_kind,v.src_index)
    return replace(v,src_kind=src.kind,src_index=src.index,dst_kind=(dst:=fn(RKArg(v.dst_kind,v.dst_index))).kind,dst_index=dst.index)
  def host(value:RKHostAddress) -> RKHostAddress: return replace(value, src=fn(value.src), index=fn(value.index), dst=fn(value.dst))
  return replace(image, gathers=tuple(map(gather, image.gathers)), mid_gathers=tuple(map(gather, image.mid_gathers)),
    post_gathers=tuple(map(gather, image.post_gathers)), ew_ops=tuple(replace(op, dst=fn(op.dst), lhs=fn(op.lhs), rhs=fn(op.rhs))
    for op in image.ew_ops), host_gathers=tuple(map(host, image.host_gathers)), host_scatters=tuple(map(host, image.host_scatters)),
    cmac=None if image.cmac is None else replace(image.cmac, dst=fn(image.cmac.dst), lhs=fn(image.cmac.lhs), rhs=fn(image.cmac.rhs)))

def _alias_image_args(image:RKImage, aliases:dict[int, RKArg]) -> RKImage:
  return _map_image_args(image, lambda arg:replace(aliases[arg.index], addend=aliases[arg.index].addend+arg.addend) if
                         arg.kind is RKBufferKind.ARG and arg.index in aliases else arg)

def _reuse_linear_scratch(image:RKImage, constant_slots:dict[bytes, int]) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  def gather_args(gather:RKGather) -> tuple[RKArg, ...]:
    return (() if gather.values else (RKArg(gather.src_kind, gather.src_index),))+(RKArg(gather.dst_kind, gather.dst_index),)
  mid_by_point:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers: mid_by_point.setdefault(gather.after if gather.after >= 0 else image.gather_after, []).append(gather)
  schedule = [tuple(RKArg(RKBufferKind.SCRATCH, slot) for slot in constant_slots.values())] + [gather_args(gather) for gather in image.gathers]
  schedule += [(host.src, host.index, host.dst) for host in image.host_gathers]
  for index,op in enumerate(image.ew_ops): schedule += [gather_args(gather) for gather in mid_by_point.get(index, ())] + [(op.lhs, op.rhs, op.dst)]
  schedule += [gather_args(gather) for gather in mid_by_point.get(len(image.ew_ops), ())] + \
    ([] if image.cmac is None else [(image.cmac.lhs, image.cmac.rhs, image.cmac.dst)]) + [gather_args(gather) for gather in image.post_gathers]
  schedule += [(host.src, host.index, host.dst) for host in image.host_scatters]
  events:dict[int, tuple[int, int]] = {}
  for event,args in enumerate(schedule):
    events.update((arg.index, (events.get(arg.index, (event, event))[0], event))
                  for arg in args if arg.kind is RKBufferKind.SCRATCH)
  if any(not 0 <= slot < len(image.scratch) for slot in events): raise ValueError("invalid virtual scratch slot")
  # Mid-program gathers may populate one logical slot in several partial phases. The runtime clears a
  # destination once per physical slot, so these stateful materialization slots must not alias.
  pinned = {gather.dst_index for gather in image.mid_gathers if gather.dst_kind is RKBufferKind.SCRATCH}
  remap, physical, active, available = typing_cast(dict[int, int], {}), typing_cast(list[RKScratch], []), \
    typing_cast(list[tuple[int, int]], []), typing_cast(list[int], [])
  for start,end,slot in sorted(((points[0], points[1], slot) for slot,points in events.items()), key=lambda item:(item[0], item[2])):
    while active and active[0][0] < start:
      heapq.heappush(available, heapq.heappop(active)[1])
    spec, target = image.scratch[slot], heapq.heappop(available) if slot not in pinned and available else len(physical)
    if target == len(physical):
      physical.append(spec)
    else:
      physical[target] = RKScratch(max(physical[target].size, spec.size), max(physical[target].alignment, spec.alignment))
    if slot not in pinned: heapq.heappush(active, (end, target))
    remap[slot] = target
  image = _map_image_args(image, lambda arg: replace(arg,index=remap[arg.index]) if arg.kind is RKBufferKind.SCRATCH else arg, map_value_src=False)
  by_slot:dict[int, bytes] = {}
  for bits,slot in constant_slots.items():
    if by_slot.setdefault(remap[slot], bits) != bits: raise ValueError("overlapping scratch constants")
  constants = b"" if not by_slot else b"".join(by_slot.get(slot, b"\0\0") for slot in range(max(by_slot)+1))
  return replace(image, scratch=tuple(physical), constants=constants)

class RKStage(NamedTuple): commands: tuple[int, ...]; relocs: tuple[tuple[int, RKArg], ...]

def encode_image(image:RKImage) -> bytes:
  gathers = image.gathers + image.mid_gathers + image.post_gathers
  if image.cmac is not None and (image.ew_ops or image.mid_gathers or image.host_gathers or image.host_scatters):
    raise ValueError("CMAC must be one fixed native stage")
  if image.cmac is not None: _validate_cmac(image.cmac, image.scratch)
  if image.mid_gathers and any(not 0 <= (g.after if g.after >= 0 else image.gather_after) <= len(image.ew_ops) for g in image.mid_gathers):
    raise ValueError("invalid mid-gather split")
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(gathers),
                               len(image.host_gathers), len(image.host_scatters),
                               len(image.ew_ops), len(image.constants), len(image.mid_gathers), len(image.post_gathers),
                               image.gather_after, int(image.cmac is not None)))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in gathers:
    out += _GATHER.pack(g.dst_index, g.src_index, g.count, kind := 3 if g.partial else 2 if g.values else 1 if g.offsets else 0,
      len(g.axes), g.itemsize, int(g.dst_kind), int(g.src_kind),
      g.base, g.fill_bits, g.dst_stride, g.dst_addend, g.after) + (struct.pack(f"<{g.count}{_ITEM_FORMAT[g.itemsize]}", *g.values) if kind == 2 else
      struct.pack(f"<{g.count}i", *g.offsets) if kind in (1, 3) else b"".join(_GATHER_AXIS.pack(*axis) for axis in g.axes))
  for host in image.host_gathers + image.host_scatters:
    if host.itemsize not in _ITEM_FORMAT or host.index_itemsize not in (2, 4): raise ValueError("invalid RKHostAddress item size")
    args = (host.src, host.index, host.dst)
    out += _HOST_ADDRESS.pack(*(int(arg.kind) for arg in args), host.itemsize, host.index_itemsize,
      *(arg.index for arg in args), host.count, host.src_count, host.dst_count, host.fill_bits, 0, host.index_limit,
      *(arg.addend for arg in args), host.base, host.index_scale, host.lane_stride)
  for op in image.ew_ops:
    if op.bool_output and not op.int32_output: raise ValueError("bool output requires INT32 conversion")
    int16_to_int32 = op.int16_input and op.int32_output and not op.int16_output and not op.int32_input
    if (op.int16_output or op.int16_input) and (op.int32_output or op.int32_input) and not int16_to_int32:
      raise ValueError("conflicting integer precision")
    op_flags = (int(op.submit_barrier) | int(op.compare)<<1 | int(op.stateful)<<2 | int(op.int32_output)<<3 |
                int(op.int32_input)<<4 | int(op.bool_output)<<5 | int(op.int16_output)<<6 | int(op.int16_input)<<7)
    out += _EWOP.pack(int(op.dst.kind), op_flags, op.dst.index, int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index,
                      op.count, op.ew_cfg, op.dst.addend, op.lhs.addend, op.rhs.addend)
  if (cmac:=image.cmac) is not None:
    out += _CMAC.pack(int(cmac.dst.kind), int(cmac.lhs.kind), int(cmac.rhs.kind), int(cmac.out_fp16)|int(cmac.relu)<<1,
                      cmac.dst.index, cmac.lhs.index, cmac.rhs.index, cmac.dst.addend, cmac.lhs.addend, cmac.rhs.addend, cmac.m, cmac.n, cmac.k)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nhost_gather, nhost_scatter, nop, nconst, mid_count, post_count, gather_after, flags = \
    _HEADER.unpack_from(blob)
  if (magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or mid_count+post_count > ngather or flags & ~1 or flags and (nop or mid_count or nhost_gather or nhost_scatter) or  # noqa: E501
      (mid_count and not 0 <= gather_after < nop) or (not mid_count and gather_after != 0)): raise ValueError("invalid RKImage header")
  off,scratch=_HEADER.size+nscratch*_SCRATCH.size, tuple(RKScratch(*_SCRATCH.unpack_from(blob,_HEADER.size+i*_SCRATCH.size))for i in range(nscratch))
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_index, src_index, count, kind, naxes, itemsize, dst_kind, src_kind, base, fill_bits, dst_stride, dst_addend, after = \
      _GATHER.unpack_from(blob, off); off += _GATHER.size
    if (kind not in (0, 1, 2, 3) or (kind and naxes) or itemsize not in _ITEM_FORMAT or dst_kind not in (0, 1) or src_kind not in (0, 1) or
        dst_stride < 1 or dst_addend < 0): raise ValueError("invalid RKGather")
    axes, offsets, values = (((), (), struct.unpack_from(f"<{count}{_ITEM_FORMAT[itemsize]}", blob, off)) if kind == 2 else
      ((), struct.unpack_from(f"<{count}i", blob, off), ()) if kind in (1, 3) else
      (tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)), (), ()))
    off += itemsize*count if kind == 2 else 4*count if kind in (1, 3) else naxes*_GATHER_AXIS.size
    gathers.append(RKGather(src_index, dst_index, count, base if kind == 0 else 0, axes, offsets, fill_bits, values, kind == 3,
      dst_stride, dst_addend, RKBufferKind(dst_kind), itemsize, RKBufferKind(src_kind), after))
  host_addresses, ew_ops = typing_cast(list[RKHostAddress], []), typing_cast(list[RKEWOp], [])
  for _ in range(nhost_gather+nhost_scatter):
    src_kind, index_kind, dst_kind, itemsize, index_itemsize, src_index, index_index, dst_index, count, src_count, dst_count, \
      fill_bits, host_flags, index_limit, src_addend, index_addend, dst_addend, base, index_scale, lane_stride = _HOST_ADDRESS.unpack_from(blob, off); off += _HOST_ADDRESS.size  # noqa: E501
    if (src_kind not in (0, 1) or index_kind not in (0, 1) or dst_kind not in (0, 1) or itemsize not in _ITEM_FORMAT or
        index_itemsize not in (2, 4) or host_flags or min(count, src_count, dst_count, index_limit) < 0):
      raise ValueError("invalid RKHostAddress")
    host_addresses.append(RKHostAddress(RKArg(RKBufferKind(src_kind), src_index, src_addend),
      RKArg(RKBufferKind(index_kind), index_index, index_addend), RKArg(RKBufferKind(dst_kind), dst_index, dst_addend),
      count, src_count, dst_count, itemsize, index_itemsize, fill_bits, index_limit, base, index_scale, lane_stride))
  for _ in range(nop):
    dk, op_flags, di, lk, li, rk_, ri, count, ew_cfg, da, la, ra = _EWOP.unpack_from(blob, off); off += _EWOP.size
    if op_flags & ~0xff or op_flags & 0x20 and not op_flags & 0x08 or \
       op_flags & 0xc0 and op_flags & 0x18 and not (op_flags & 0x88 == 0x88 and not op_flags & 0x50):
      raise ValueError("invalid RKEWOp flags")
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
      RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg, *(bool(op_flags & 1<<bit) for bit in range(8))))
  cmac = None
  if flags:
    dk,lk,rk_,fp16,di,li,ri,da,la,ra,m,n,k = _CMAC.unpack_from(blob, off); off += _CMAC.size
    if max(dk,lk,rk_) > 1 or fp16 > 3 or min(da,la,ra) < 0 or min(m,n,k) <= 0: raise ValueError("invalid RKCMAC")
    cmac = RKCMAC(RKArg(RKBufferKind(dk),di,da), RKArg(RKBufferKind(lk),li,la), RKArg(RKBufferKind(rk_),ri,ra),m,n,k,bool(fp16&1),bool(fp16&2))
    _validate_cmac(cmac, scratch)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  return RKImage(RKTarget(target), scratch, blob[off:], version, tuple(gathers[:ngather-mid_count-post_count]), tuple(ew_ops),
                 tuple(gathers[ngather-mid_count-post_count:ngather-post_count]), gather_after, tuple(gathers[-post_count:] if post_count else ()),
                 tuple(host_addresses[:nhost_gather]), tuple(host_addresses[nhost_gather:]), cmac)

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for word_index,arg in stage.relocs:
    commands[word_index] = (commands[word_index] & ~0xffffffff0000) | (((address(arg.kind, arg.index) + arg.addend) & 0xffffffff) << 16)
  return tuple(commands)

# Admission and exact-carrier bounds.
(_DPU, _RDMA, _MAX_EW_ELEMS_FP16, _MAX_GENERIC_UNROLL, _MAX_GENERIC_EXPANDED_NODES, _MAX_OPTIONAL_RECIPE_NODES, _MAX_STATIC_LOCAL_STEPS, _MAX_STATIC_RANGE_ENVS, _MAX_DYNAMIC_SELECTOR_CELLS, _EW_ELEMS_32BIT, _FP16_EXACT_INTEGER) = (  # noqa: E501
  0x1001, 0x2001, 64000, 1 << 14, 1 << 20, 4096, 1 << 20, 1 << 18, 1 << 22, 8*dtypes.half.itemsize//dtypes.float.itemsize, 1 << 11)
# Native EW register fields.
_EW_RELU_BYPASS, _EW_OP_CVT_BYPASS = 1 << 9, 1 << 8
_EW_CFG_COMMON = (1 << 28) | (2 << 22) | (1 << 7) | (1 << 6)
(_EW_CFG_RELU6, _EW_CFG_MIN, _EW_CFG_ABS, _EW_CFG_NEG, _EW_CFG_FLOOR, _EW_CFG_CEIL) = tuple(
  _EW_CFG_COMMON|flags for flags in (1<<10, _EW_RELU_BYPASS|(1<<16), _EW_RELU_BYPASS|(5<<16),
  _EW_RELU_BYPASS|(6<<16), _EW_RELU_BYPASS|(7<<16), _EW_RELU_BYPASS|(8<<16)))
# Software stage tags and DPU data-format registers.
_EW_STAGE_FP32_OUT, _EW_STAGE_FP32_IN = 1 << 29, 1 << 30
_DPU_DATA_FORMATS = ((5<<29)|(2<<26)|2, (2<<29)|(5<<26)|2, (1<<29)|(1<<26)|1, (4<<29)|(4<<26)|4,
  (4<<29)|(1<<26)|1, (4<<29)|(2<<26)|2, (1<<29)|(2<<26)|2, (2<<29)|(4<<26)|4, (2<<29)|(2<<26)|2)
# Batch-size and batch-normalization registers used by compare stages.
(_BS_BN_BYPASS, _BS_OW_FP32_SCALAR, _BS_CFG_COMPARE, _BS_ALU_COMPARE, _BS_MUL_COMPARE, _BN_CFG_COMPARE, _BN_MUL_COMPARE,
 _BN_RELUX_COMPARE) = (1|(1<<1)|(1<<4)|(1<<6), (1<<8)|(1<<5)|(1<<2)|(1<<1), 0x40040, 0x33800000, 0x40000000, 0x40082, 0x7c000000, 0x3f800000)
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_FLOOR, _NATIVE_MASK_MUL, _NATIVE_MIN, _NATIVE_POSITIVE_MASK, _NATIVE_PRECISE_ADD,
 _NATIVE_RELU6, _NATIVE_SIGN) = tuple("rockchip_"+name for name in "abs ceil floor mask_mul min positive_mask precise_add relu6 sign".split())
_EW_RELUX_CMP_RELU6, _INT16_EW = struct.unpack("<I", struct.pack("<f", 6.0))[0], dict(int16_input=True, int16_output=True)
_EW_CFG = {op:_EW_CFG_COMMON|_EW_RELU_BYPASS|flags for op,flags in ((Ops.ADD,2<<16), (Ops.SUB,4<<16), (Ops.MUL,_EW_OP_CVT_BYPASS|1<<2), (Ops.MAX,0), (Ops.FDIV,_EW_OP_CVT_BYPASS|3<<16))}  # noqa: E501
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff

def _cmac_layout(n:int, k:int) -> tuple[int, int, int]: aligned_k,align_out=max(32,round_up(k,32)),max(32,round_up(n,32)); align_in=max(aligned_k,align_out); return align_in,align_out,align_in if align_in != aligned_k else k  # noqa: E501

def _validate_cmac(op:RKCMAC, scratch:tuple[RKScratch, ...]|None=None) -> None:
  ai,ao,_ = _cmac_layout(op.n,op.k)
  if not 0 < op.m <= 0x7ff or ai > 0xffff or ao > 0x3fff or op.m*ai*2 > 10*32768 or ai > 12*32 and op.m != 1: raise ValueError("CMAC shape out of range")  # noqa: E501
  args,needs,alignments = (op.lhs,op.rhs,op.dst),(op.m*ai*2,ao*ai*2,op.m*ao*4),(2,2,2 if op.out_fp16 else 4)
  if any(arg.kind is not RKBufferKind.SCRATCH or arg.addend < 0 or arg.addend%alignment for arg,alignment in zip(args,alignments)): raise ValueError("CMAC requires aligned scratch buffers")  # noqa: E501
  if scratch is not None and any(not 0 <= arg.index < len(scratch) or arg.addend+need > scratch[arg.index].size for arg,need in zip(args,needs)): raise ValueError("CMAC exceeds scratch buffer")  # noqa: E501

def emit_cmac_stage(op:RKCMAC) -> RKStage:
  """Emit the 45-qword GEMM body; terminal BS ReLU preserves the runtime-owned four-qword PC tail."""
  C,O,D,ai,ao,ek = 0x201,0x801,_DPU,*_cmac_layout(op.n, op.k)
  _validate_cmac(op)
  row_bytes = ai*2; grains = max(80, (ceildiv(2*32768, row_bytes)+1)&~1); banks = min(11, max(1, ceildiv(op.m*row_bytes, 32768)))
  line_stride, notch, (precision,size_e) = 4*min(ceildiv(ek,32),13), 8*min(ao//32,13)-1, (2,1) if op.out_fp16 else (5,3)
  regs = ((D,rk.REG_DPU_S_POINTER,0xe), (C,rk.REG_CNA_CONV_CON1,(2<<4)|(2<<7)|(1<<29)),
    (C,rk.REG_CNA_CONV_CON2,grains<<4), (C,rk.REG_CNA_CONV_CON3,9), (C,rk.REG_CNA_DATA_SIZE0,(1<<16)|op.m),
    (C,rk.REG_CNA_DATA_SIZE1,((ai-1)<<16)|ai), (C,rk.REG_CNA_DATA_SIZE2,1), (C,rk.REG_CNA_DATA_SIZE3,op.m),
    (C,rk.REG_CNA_WEIGHT_SIZE0,row_bytes*ao), (C,rk.REG_CNA_WEIGHT_SIZE1,row_bytes),
    (C,rk.REG_CNA_WEIGHT_SIZE2,(1<<24)|(1<<16)|ao), (C,rk.REG_CNA_CBUF_CON0,((12-banks)<<4)|banks),
    (C,rk.REG_CNA_CBUF_CON1,ceildiv(ai,32)), (C,rk.REG_CNA_CVT_CON0,11), (C,rk.REG_CNA_CVT_CON1,1<<16),
    (C,rk.REG_CNA_CVT_CON2,1<<16), (C,rk.REG_CNA_CVT_CON3,1<<16), (C,rk.REG_CNA_CVT_CON4,1<<16),
    (C,rk.REG_CNA_FEATURE_DATA_ADDR,0), (C,rk.REG_CNA_DMA_CON0,(15<<16)|15), (C,rk.REG_CNA_DMA_CON1,line_stride),
    (C,rk.REG_CNA_DMA_CON2,0), (C,rk.REG_CNA_FC_DATA_SIZE0,(1<<16)|op.m), (C,rk.REG_CNA_FC_DATA_SIZE1,ai),
    (C,rk.REG_CNA_DCOMP_ADDR0,0), (O,rk.REG_CORE_MISC_CFG,(2<<8)|1), (O,rk.REG_CORE_DATAOUT_SIZE_0,(op.m-1)<<16),
    (O,rk.REG_CORE_DATAOUT_SIZE_1,ao-1), (O,rk.REG_CORE_RESERVED_3030,0), (D,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)),
    (D,rk.REG_DPU_DATA_FORMAT,(precision<<29)|(2<<26)|2), (D,rk.REG_DPU_DST_BASE_ADDR,0), (D,rk.REG_DPU_DST_SURF_STRIDE,1<<4),
    (D,rk.REG_DPU_DATA_CUBE_WIDTH,0), (D,rk.REG_DPU_DATA_CUBE_HEIGHT,op.m-1),
    (D,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,(notch<<16)|notch), (D,rk.REG_DPU_DATA_CUBE_CHANNEL,((ao-1)<<16)|(ao-1)),
    (D,rk.REG_DPU_BS_CFG,0x12 if op.relu else 0x53), (D,rk.REG_DPU_BS_OW_CFG,(size_e<<8)|(size_e<<5)|(size_e<<2)|2),
    (D,rk.REG_DPU_WDMA_SIZE_0,ao-1), (D,rk.REG_DPU_WDMA_SIZE_1,(op.m-1)<<16), (D,rk.REG_DPU_BN_CFG,0x53),
    (D,rk.REG_DPU_EW_CFG,0x383), (D,rk.REG_DPU_OUT_CVT_SCALE,(1<<16)|1 if op.out_fp16 else 0), (D,rk.REG_DPU_SURFACE_ADD,4<<4))
  return RKStage(tuple(_cmd(*reg) for reg in regs), ((18,op.lhs),(24,op.rhs),(31,op.dst)))

def _raw_gather(source:RKArg, out_slot:int, count:int, stride:int=2, itemsize:int=1,
                src_kind:RKBufferKind=RKBufferKind.SCRATCH, dst_stride:int=1, dst_addend:int=0, offsets:tuple[int, ...]=()) -> RKGather:
  return RKGather(source.index, out_slot, count, base=source.addend, axes=() if offsets else ((1, count, stride),), offsets=offsets,
                  dst_kind=RKBufferKind.ARG, dst_stride=dst_stride, dst_addend=dst_addend, src_kind=src_kind, itemsize=itemsize)

@functools.lru_cache(maxsize=256)
def _stage_template(count:int, ew_cfg:int, compare:bool=False, stateful:bool=False, int32_output:bool=False, int32_input:bool=False,
                    int16_output:bool=False, int16_input:bool=False, fp32_output:bool=False, fp32_input:bool=False) \
                    -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Emit one DPU EW register template, sharing its physical prefix and RDMA tail across every precision."""
  D, R = _DPU, rk
  special, native_int16, native_int32 = (compare or stateful or int32_output or int32_input or int16_output or int16_input or
    fp32_output or fp32_input), int16_input and int16_output, int32_input and int32_output
  int16_to_int32 = int16_input and int32_output and not int16_output and not int32_input
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if native_int32 else _EW_ELEMS_32BIT if \
    int32_output or int32_input or fp32_output or fp32_input else _MAX_EW_ELEMS_FP16
  if not 0 < count <= limit:
    raise ValueError(f"{'stateful EW' if special else 'EW fp16'} count {count} out of range")
  lanes, is_div = (4 if int32_input or fp32_input else 8), ew_cfg == _EW_CFG[Ops.FDIV]
  width, data_format = (count + lanes-1) // lanes - 1, next(format_ for flag,format_ in zip(
    (fp32_output, fp32_input, native_int16, native_int32, int16_to_int32, int32_output, int16_output, int32_input, True), _DPU_DATA_FORMATS) if flag)
  regs:tuple[tuple[int, int, int], ...] = ((D,R.REG_DPU_S_POINTER,0xe),(D,R.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (D,R.REG_DPU_DATA_FORMAT,data_format)) + (((D,R.REG_DPU_DST_SURF_STRIDE,1<<4),) if int16_to_int32 or fp32_output else ()) + (
    (D,R.REG_DPU_DATA_CUBE_WIDTH,width),(D,R.REG_DPU_DATA_CUBE_HEIGHT,0),(D,R.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (D,R.REG_DPU_DATA_CUBE_CHANNEL,0 if fp32_output and count == 1 else ((lanes-1)<<16)|(lanes-1)))
  if special:
    pipeline:tuple[tuple[int, int, int], ...] = ((D,R.REG_DPU_BS_CFG,_BS_BN_BYPASS),(D,R.REG_DPU_BN_CFG,_BS_BN_BYPASS),
      (D,R.REG_DPU_BS_ALU_CFG,0),(D,R.REG_DPU_BS_MUL_CFG,0),
      (D,R.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR if int16_to_int32 or fp32_output and count == 1 else 2),
      (D,R.REG_DPU_WDMA_SIZE_0,0 if fp32_output and count == 1 else 3 if fp32_output else lanes-1),(D,R.REG_DPU_WDMA_SIZE_1,width),
      (D,R.REG_DPU_BN_MUL_CFG,0),(D,R.REG_DPU_BN_RELUX_CMP_VALUE,0))
    if compare: pipeline += ((D,R.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(D,R.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),
      (D,R.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),(D,R.REG_DPU_BN_CFG,_BN_CFG_COMPARE),(D,R.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),
      (D,R.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE))
    ew = _EW_CFG_COMMON|1 if compare else (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else \
      ew_cfg & ~_EW_OP_CVT_BYPASS if native_int16 or int16_to_int32 else ew_cfg
    regs += pipeline + ((D,R.REG_DPU_EW_CFG,ew),(D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),
      (D,R.REG_DPU_OUT_CVT_SCALE,0 if fp32_output else 1 if int32_output or int16_output or is_div else (1<<16)|1),
      (D,R.REG_DPU_OUT_CVT_SHIFT,0),(D,R.REG_DPU_SURFACE_ADD,(2 if native_int16 or int16_to_int32 else 4)<<4))
  else:
    regs += ((D,R.REG_DPU_EW_CFG,ew_cfg),) + (((D,R.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6),) if ew_cfg == _EW_CFG_RELU6 else ()) + (
      ((D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),(D,R.REG_DPU_OUT_CVT_SHIFT,0),
       (D,R.REG_DPU_SURFACE_ADD,1<<6)) if is_div else ()) + ((D,R.REG_DPU_OUT_CVT_SCALE,1 if is_div else (1<<16)|1),)
  regs += ((_RDMA,R.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,lanes-1),
    (_RDMA,R.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input or fp32_input else 2)<<2)))
  rdma_precision = 5 if fp32_input else 4 if int32_input else 1 if int16_input else 2
  rdma_feature = (rdma_precision<<15)|(15<<11)|(rdma_precision<<5)|(0 if is_div or int16_input or fp32_input else 1<<3)|1
  bindings = ((_DPU, R.REG_DPU_DST_BASE_ADDR), (_RDMA, R.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR), (_RDMA, R.REG_DPU_RDMA_RDMA_EW_BASE_ADDR))
  commands = tuple(_cmd(*reg) for reg in regs)+tuple(_cmd(target, reg, 0) for target,reg in bindings)
  return commands+(_cmd(_RDMA, R.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature),), tuple(range(len(regs), len(commands)))

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int, compare:bool=False,
                  stateful:bool=False, int32_output:bool=False, int32_input:bool=False,
                  int16_output:bool=False, int16_input:bool=False) -> RKStage:
  """Build one DPU EW command body without its PC-chain tail."""
  fp32_output, fp32_input = bool(ew_cfg & _EW_STAGE_FP32_OUT), bool(ew_cfg & _EW_STAGE_FP32_IN)
  ew_cfg &= ~(_EW_STAGE_FP32_OUT|_EW_STAGE_FP32_IN)
  commands, words = _stage_template(count, ew_cfg, compare, stateful, int32_output, int32_input, int16_output, int16_input, fp32_output, fp32_input)
  return RKStage(commands, tuple(zip(words, (dst, lhs, rhs))))

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM and u.src: u = u.src[0]
  return u if u.op is Ops.PARAM else None

def _strip_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _typed_cast_source(u:UOp, dtype:DType, source:DType) -> UOp|None:
  return u.src[0] if u.op is Ops.CAST and u.dtype.scalar() is dtype and len(u.src) == 1 and u.src[0].dtype.scalar() is source else None

def _inverted_condition(u:UOp, typed:bool=False) -> UOp|None: return None if u.op is not Ops.CMPNE else next(
  (v for v,m in (u.src, u.src[::-1]) if m.op is Ops.CONST and bool(m.arg) and (not typed or m.dtype.scalar() is dtypes.bool)), None)

def _local_load(u:UOp) -> UOp|None: return u if (u:=_strip_cast(u)).op is Ops.LOAD and _root_param(u.src[0]) is None else None

@functools.lru_cache(maxsize=4096)
def _semantic_loads(u:UOp, local:bool=False) -> tuple[UOp, ...]:
  sources = () if u.op in (Ops.RANGE, Ops.SPECIAL) else u.src[:1] if u.op is Ops.AFTER else u.src
  load = _local_load(u) if local else u if u.op is Ops.LOAD else None
  return (load,) if load is not None else tuple(dict.fromkeys(y for x in sources for y in _semantic_loads(x, local)))

def _static_cast(value, dtype:DType, vector:bool=False):
  if vector or isinstance(value, np.ndarray): return np.asarray(value, dtype=np.dtype(dtype.scalar().fmt) if dtype.scalar().fmt is not None else None)
  return bool(value) if (scalar:=dtype.scalar()) is dtypes.bool else int(value) if scalar in dtypes.ints else float_to_fp16(value) if scalar is \
    dtypes.half else struct.unpack("<f", struct.pack("<f", float(value)))[0] if scalar is dtypes.float else float(value)

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX}
_STATIC_INT_ALU = {Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.AND, Ops.OR, Ops.XOR}
_STATIC_SCALAR_ALU = _STATIC_OPS - {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST, Ops.WHERE} | _STATIC_INT_ALU

def _eval_expr(u:UOp, env:Mapping[UOp, int|float|bool|np.ndarray], cache:dict[UOp, int|float|bool|np.ndarray], vector:bool=False) -> int|float|bool|np.ndarray:  # noqa: E501
  if u in cache: return cache[u]
  if u.op is Ops.CONST: return cache.setdefault(u, _static_cast(u.arg, u.dtype, vector))
  elif u.op in (Ops.RANGE, Ops.SPECIAL): return cache.setdefault(u, _static_cast(env[u], u.dtype, True) if vector else env[u])
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
  elif u.op is Ops.AFTER or u.op is Ops.CAST: return cache.setdefault(u, _eval_expr(u.src[0], env, cache, vector) if u.op is Ops.AFTER else
    _static_cast(_eval_expr(u.src[0], env, cache, vector), u.dtype, vector))
  elif u.op is Ops.WHERE and not vector:
    return cache.setdefault(u, _static_cast(_eval_expr(u.src[1] if _eval_expr(u.src[0], env, cache) else u.src[2], env, cache), u.dtype))
  values = tuple(_eval_expr(src, env, cache, vector) for src in u.src)
  if vector:
    if u.op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
      floor = u.op in (Ops.FLOORDIV, Ops.FLOORMOD)
      with np.errstate(divide="ignore", invalid="ignore"): quotient = np.where(values[1] != 0,
        np.floor_divide(values[0], values[1]) if floor else np.trunc(values[0] / values[1]), 0)
      value = quotient if u.op in (Ops.CDIV, Ops.FLOORDIV) else values[0]-quotient*values[1]
    elif u.op in (Ops.MAX, Ops.WHERE, Ops.TRUNC, Ops.RECIPROCAL): value = np.where(*values) if u.op is Ops.WHERE else \
      np.where(values[0] < values[1], values[1], values[0]) if u.op is Ops.MAX else \
      np.vectorize(int, otypes=[np.int64])(values[0]) if u.op is Ops.TRUNC else 1.0 / values[0]
    else:
      try: value = python_alu[u.op](*values)
      except KeyError: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
  else:
    if u.op not in _STATIC_SCALAR_ALU: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
    value = (1.0 / float(values[0]) if u.op is Ops.RECIPROCAL else int(values[0])) if u.op in (Ops.RECIPROCAL, Ops.TRUNC) else \
      exec_alu(u.op, u.dtype, tuple(int(x) for x in values) if u.op in _STATIC_INT_ALU else values, truncate_output=False)
  return cache.setdefault(u, _static_cast(value, u.dtype, vector))

def _is_static_expr(u:UOp) -> bool: return u.op in _STATIC_OPS and all(_is_static_expr(x) for x in u.src)

def _index_ranges(index:UOp) -> list[UOp]:
  """Ranges used as index values, excluding AFTER/END ordering dependencies attached to a RANGE."""
  return [index] if index.op in (Ops.RANGE, Ops.SPECIAL) else list(dict.fromkeys(r for src in index.src for r in _index_ranges(src)))

RKOutput = tuple[UOp, UOp, int, UOp, UOp]
def _outs(uops:list[UOp]) -> tuple[RKOutput|None, RKOutput|None, list[UOp]]:
  """Return the single statically-sized output store shared by specialized graph matchers."""
  stores = [u for u in uops if u.op is Ops.STORE]
  outputs = [(store, root) for store in stores if (root:=_root_param(store.src[0])) is not None]
  if len(outputs) != 1: return None, None, [store for store,_ in outputs]
  store, out_param = outputs[0]
  if out_param.src[0].op is not Ops.CONST or store.src[0].op is not Ops.INDEX: return None, None, []
  output = store, out_param, int(out_param.src[0].arg), store.src[0].src[1], store.src[1]
  return (output if len(stores) == 1 else None), output, [store]

def _output_store(uops,dtype,*,allow_local=False)->RKOutput|None: return _admit(_outs(uops)[allow_local], dtype)

def _admit(o,d,v=True)->RKOutput|None: return o if v and o is not None and o[1].dtype.scalar() in (d if isinstance(d,tuple) else (d,)) else None
def _try(o,d,f,*a,v=True)->RKImage|None: return None if not v or (o:=_admit(o,d)) is None else f(o,*a)

def _iter_range_env(ranges:list[UOp], max_envs:int|None=_MAX_STATIC_RANGE_ENVS, dependencies:bool=True) -> list[dict[UOp, int]]:
  order, ranges = (tuple(ranges), []) if dependencies else (ranges, ranges)
  if dependencies:
    def add(r:UOp) -> None:
      if r in ranges: return
      for src in r.src[1:]:
        if src.op is Ops.RANGE: add(src)
      ranges.append(r)
    for r in order: add(r)
  envs:list[dict[UOp, int]] = [{}]
  for r in ranges:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    bound = int(r.src[0].arg)
    if max_envs is not None and (bound < 0 or bound and len(envs) > max_envs//bound): raise RuntimeError("RKPLAN_REJECT:static_index_budget")
    envs = [{**env, r: i} for env in envs for i in range(bound)]
  return envs

@functools.lru_cache(maxsize=8)
def _static_vector_env(out_index:UOp, *expressions:UOp, reject:str="static_index") -> tuple[dict[UOp, np.ndarray], np.ndarray]:
  ranges = tuple(_index_ranges(out_index)); envs = _iter_range_env(list(ranges))
  vector_env = {r:np.fromiter((env[r] for env in envs), dtype=np.int64, count=len(envs)) for r in ranges}
  if any(r not in ranges for expression in expressions for r in _index_ranges(expression)): raise RuntimeError(f"RKPLAN_REJECT:{reject}")
  return vector_env, np.broadcast_to(_eval_expr(out_index, vector_env, {}, True), len(envs)).astype(np.int64)

def _static_values(out_index:UOp, expr:UOp, count:int, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
  vector_env, dst_lanes = _static_vector_env(out_index, expr)
  expr_lanes = np.broadcast_to(_eval_expr(expr, vector_env, {}, True), len(dst_lanes))
  if encode is _fp16_bits:
    fp_values = np.asarray(expr_lanes, dtype=np.float64)
    if np.any(np.isfinite(fp_values) & (np.abs(fp_values) >= 65520)): raise OverflowError("float too large to pack with e format")
    encoded:np.ndarray = fp_values.astype(np.float16).view(np.uint16)
  elif encode in (_int16_bits, int): encoded = np.asarray(expr_lanes).astype(np.int64) & (0xffff if encode is _int16_bits else -1)
  else: encoded = np.fromiter((encode(value.item()) for value in expr_lanes), dtype=np.int64, count=len(expr_lanes))
  order = np.argsort(dst_lanes); dst, values = dst_lanes[order], encoded[order]
  starts = np.empty(len(dst), dtype=np.bool_); starts[:1] = True; starts[1:] = dst[1:] != dst[:-1]
  if not np.array_equal(dst[starts], np.arange(count)) or np.any(values[1:][~starts[1:]] != values[:-1][~starts[1:]]):
    raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(values[starts].tolist())

def _linear_index(u:UOp, divided:bool=False) -> tuple[int, dict[UOp|tuple[UOp, int], int]]|None:
  """Represent static address arithmetic as a sum of scaled RANGE or RANGE//constant terms."""
  if u.op is Ops.CONST: return int(u.arg), {}
  if divided and u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() in (dtypes.int, dtypes.uint): return _linear_index(u.src[0], divided)
  term:UOp|tuple[UOp, int]|None = ((u, 1) if divided else u) if u.op in (Ops.RANGE, Ops.SPECIAL) else None
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
  sign, terms = -1 if u.op is Ops.SUB else 1, lhs[1].copy()
  for key,coefficient in rhs[1].items(): terms[key] = terms.get(key, 0)+sign*coefficient
  return lhs[0]+sign*rhs[0], {key:coefficient for key,coefficient in terms.items() if coefficient}

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> tuple[int, ...]:
  vector_env, dst = _static_vector_env(out_index, load_index, *((gate,) if gate is not None else ()), reject="gather_index")
  cache:dict[UOp, int|float|bool|np.ndarray] = {}; src = np.broadcast_to(_eval_expr(load_index, vector_env, cache, True), len(dst)).astype(np.int64)
  values = src if gate is None else np.where(active:=np.broadcast_to(_eval_expr(gate, vector_env, cache, True), len(dst)), src, -1)
  if np.any((src < 0) & (gate is None or active)): raise RuntimeError("RKPLAN_REJECT:gather_index")
  if np.any((dst < 0) | (dst >= count)): raise RuntimeError("RKPLAN_REJECT:gather_index")
  offsets = np.full(count, -2, dtype=np.int64); offsets[dst] = values
  if np.any(offsets == -2): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(int(x) for x in offsets)

def _affine_output_axes(affine:tuple[int, dict[UOp, int]], count:int) -> tuple[tuple[UOp, int, int], ...]|None:
  ordered = tuple(sorted(affine[1].items(), key=lambda item:item[1]))
  limits = tuple(int(r.src[0].arg) if r.src and r.src[0].op is Ops.CONST else 0 for r,_ in ordered)
  return tuple((r, stride, limit) for (r,stride),limit in zip(ordered, limits)) if all(limit > 0 and stride == math.prod(limits[:i])
    for i,((_,stride),limit) in enumerate(zip(ordered, limits))) and math.prod(limits) == count else None

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0) -> RKGather:
  out_affine, load_affine = tuple(typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(index)) for index in (out_index, load_index))
  if gate is None and out_affine is not None and out_affine[0] == 0 and (output_axes:=_affine_output_axes(out_affine, count)) is not None:
    if load_affine is not None and all(r in out_affine[1] for r in load_affine[1]):
      return RKGather(src_index,dst_index,count,load_affine[0],tuple((d,l,load_affine[1][r]) for r,d,l in output_axes if load_affine[1].get(r,0)))
    if (load_divided:=typing_cast(tuple[int, dict[tuple[UOp, int], int]]|None, _linear_index(load_index, True))) is not None and \
       all(r in out_affine[1] and divisor <= int(r.src[0].arg) for r,divisor in load_divided[1]):
      return RKGather(src_index, dst_index, count, load_divided[0],
                      tuple((out_affine[1][r]*divisor, (int(r.src[0].arg)+divisor-1)//divisor, stride)
                            for (r,divisor),stride in load_divided[1].items() if stride))
  return RKGather(src_index, dst_index, count, offsets=_gather_offsets(out_index, load_index, gate, count), fill_bits=fill_bits)

def _validate_gather_bounds(plan:RKGather, source_count:int) -> None:
  deltas = tuple((limit-1)*stride for _,limit,stride in plan.axes) if not plan.offsets else ()
  low,high=(min(plan.offsets,default=0),max(plan.offsets,default=-1)) if plan.offsets else (plan.base+sum(min(delta,0) for delta in deltas),plan.base+sum(max(delta,0) for delta in deltas))  # noqa: E501
  if low < (0 if not plan.offsets else -1) or high >= source_count: raise RuntimeError("RKPLAN_REJECT:gather_index")

class RKTypedLoadPlan(NamedTuple):
  """Typed source metadata shared by static-offset and physical-gather consumers."""
  param:UOp; gather:RKGather

def _typed_load_plan(load:UOp, dtype:DType, out_index:UOp, count:int, *, fill_bits:int|None=None, require_offsets:bool=False) -> RKTypedLoadPlan|None:  # noqa: E501
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX: return None
  if (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None  # noqa: E501
  gate = load.src[2] if len(load.src) > 2 else None
  fill_bits = fill_bits if fill_bits is not None else _fp16_bits(load.src[1].arg if len(load.src) > 1 else 0) if dtype is dtypes.half else 0
  try:
    _validate_gather_bounds(gather:=_gather_plan(param.arg.slot,0,out_index,load.src[0].src[1],gate,count,fill_bits),int(param.src[0].arg)); gather=replace(gather,base=0,axes=(),offsets=_gather_offsets(out_index,load.src[0].src[1],gate,count)) if require_offsets else gather  # noqa: E501
  except RuntimeError: return None
  return RKTypedLoadPlan(param, gather)

def _gather_cache_key(plans:Iterable[RKGather]) -> tuple: return tuple(v[0:1]+v[2:11]+v[12:14] for v in map(astuple, plans))

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is Ops.WHERE and (folded:=_fold_ordered_where(u)) is not None: u = folded
  if u.op is not Ops.MAX or u.arg is not None or u.dtype.scalar() not in (dtypes.half,dtypes.float): return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None


def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = x.alu(Ops.MUL, splitter)
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low, rhs_high, rhs_low = (*_split_half(term.src[0], neg_one, splitter), *_split_half(term.src[1], neg_one, splitter))
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one)
  error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low)).alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  return total, _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one).alu(Ops.ADD, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_add_parts(terms:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 addition residuals as a high lane plus a low correction lane."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  high, middle, low = terms[0], zero, zero
  for part in terms[1:]:
    high, error = _two_sum(high, part, neg_one)
    middle, error = _two_sum(middle, error, neg_one)
    low = low.alu(Ops.ADD, error)
  return high, middle.alu(Ops.ADD, low)

def _precise_sum_parts(terms:list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 product and addition residuals as a high lane plus a low correction lane."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  pairs = tuple(_two_product(term, neg_one, splitter) if term.op is Ops.MUL else (term, zero) for term in terms)
  return _precise_add_parts(tuple(x[0] for x in pairs) + tuple(x[1] for x,term in zip(pairs, terms) if term.op is Ops.MUL))

def _tag_precise_adds(root:UOp) -> UOp:
  """Mark physical ADDs so the generic accuracy pass does not expand an already compensated recipe."""
  tagged:dict[UOp, UOp] = {}
  return root.topovisit(lambda node: node.replace(src=tuple(tagged[src] for src in node.src),
    arg=_NATIVE_PRECISE_ADD if node.op is Ops.ADD else node.arg), tagged)

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  high, middle = _precise_sum_parts(terms); return _tag_precise_adds(high.alu(Ops.ADD, middle))

def _ew_ops(stages:Iterable[tuple[RKArg, RKArg, RKArg, Ops|int]], count:int, **flags) -> tuple[RKEWOp, ...]:
  return tuple(RKEWOp(dst, lhs, rhs, count, cfg if not isinstance(cfg, Ops) else _EW_CFG[cfg],
                      **flags) for dst,lhs,rhs,cfg in stages)

def _append_inplace_image(first:RKImage, second:RKImage) -> RKImage|None:
  """Append an in-place EW image, scheduling its input materialization after the first image completes."""
  if first.post_gathers or not second.ew_ops or second.host_gathers or second.host_scatters: return None
  fc,sc,kind,fs = len(first.constants)//2, len(second.constants)//2, RKBufferKind.SCRATCH, len(first.scratch)
  first, second = _map_image_args(first, lambda arg: replace(arg, index=arg.index+sc) if arg.kind is kind and arg.index >= fc else arg), \
                   _map_image_args(second, lambda arg: replace(arg,index=fc+arg.index if arg.index<sc else fs+arg.index) if arg.kind is kind else arg)
  second_ops = (replace(second.ew_ops[0], submit_barrier=True, stateful=True), *second.ew_ops[1:])
  second_mid = tuple(replace(gather, after=len(first.ew_ops)) for gather in second.gathers)+tuple(
    replace(gather, after=len(first.ew_ops)+(gather.after if gather.after >= 0 else second.gather_after)) for gather in second.mid_gathers)
  scratch = first.scratch[:fc] + second.scratch[:sc] + first.scratch[fc:] + second.scratch[sc:]
  return RKImage(RKTarget.RK3588, scratch, first.constants+second.constants,
                 gathers=first.gathers, ew_ops=first.ew_ops+second_ops, mid_gathers=first.mid_gathers+second_mid,
                 gather_after=first.gather_after, post_gathers=tuple(replace(gather, after=-1) for gather in second.post_gathers))

def _iter_binary(root:UOp, op:Ops, dtype:DType|None=None, plain:bool=False) -> Iterable[UOp]:
  stack = [root]
  while stack:
    node = stack.pop()
    if node.op is op and (dtype is None or node.dtype.scalar() is dtype) and (not plain or node.arg is None): stack.extend(reversed(node.src))
    else: yield node

def _lower_cmac_reduction(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Factor a real weighted sum/dot/matmul and pre-round linear terms into one fixed CMAC stage."""
  _,out,rows,out_index,root = output
  if rows <= 0 or out.dtype.scalar() not in (dtypes.half,dtypes.float): return None
  relu_root = _relu_operand(root)
  if relu_root is None and (fp32_root:=_typed_cast_source(root,dtypes.half,dtypes.float)) is not None: relu_root = _relu_operand(fp32_root)
  if relu_root is not None: root = relu_root
  graph = root.toposort(); local_loads = _semantic_loads(root, local=True) if any(u.op is Ops.BUFFER for u in graph) else (); local_add = bool(local_loads) and all(load.dtype.scalar() is dtypes.float for load in local_loads); additive = local_add or (_strip_cast(root).op is Ops.ADD and _strip_cast(root).dtype.scalar() is dtypes.float) or any(u.op is Ops.REDUCE and (u.arg is Ops.ADD or isinstance(u.arg, tuple) and u.arg and u.arg[0] is Ops.ADD) for u in graph)  # noqa: E501
  try: root = _unroll_static_reduces(_unroll_static_local(uops, root) if local_loads else root, precise=False)
  except (_RKGenericReject, RuntimeError, ValueError): return None
  scale,exact_scale = 1.0,True
  while (pair:=_const_operand(root:=_strip_cast(root), Ops.MUL)) is not None: root,factor=pair[0],float(pair[1].arg); scale*=factor; exact_scale=exact_scale and factor > 0.0 and math.frexp(factor)[0] == 0.5 and float_to_fp16(scale) == scale  # noqa: E501
  terms = tuple(_strip_cast(term) for term in _iter_binary(root, Ops.ADD)) if root.op is Ops.ADD else \
    (_strip_cast(root),) if additive else ()
  terms = tuple(term for term in terms if not (term.op is Ops.CONST and float(term.arg) == 0.0)); groups = len(terms)
  if local_loads and not local_add or groups < (1 if additive else 4) or groups > _MAX_GENERIC_UNROLL: return None
  # Keep oversized affine contractions compact; reject nonaffine terms before materializing output offsets.
  parsed:list[tuple[tuple[RKTypedLoadPlan, ...], float]] = []; cells = 0
  for term in terms:
    factors = tuple(map(_strip_cast, _iter_binary(_strip_cast(term), Ops.MUL, plain=True))); constants = tuple(node for node in factors if node.op is Ops.CONST); loads = tuple(node for node in factors if node.op is not Ops.CONST)  # noqa: E501
    if len(constants) > 2 or len(constants) > 1 and term.dtype.scalar() is not dtypes.float or not exact_scale or any(float_to_fp16(float(node.arg)) != float(node.arg) for node in constants) or len(loads) > 2 or any(len(load.src) > 1 and (load.src[1].op is not Ops.CONST or float(load.src[1].arg) != 0.0 or math.copysign(1.0,float(load.src[1].arg)) < 0.0) for load in loads) or not math.isfinite(weight:=scale*math.prod(float(node.arg) for node in constants)) or len(loads) < 2 and float_to_fp16(weight) != weight or len(loads) == 2 and weight != 1.0 or rows*len(loads) > _MAX_DYNAMIC_SELECTOR_CELLS-cells and ((affine:=typing_cast(tuple[int, dict[UOp, int]]|None,_linear_index(out_index))) is None or affine[0] != 0 or _affine_output_axes(affine,rows) is None or any(load.op is not Ops.LOAD or len(load.src) > 2 or not load.src or load.src[0].op is not Ops.INDEX or (load_affine:=_linear_index(load.src[0].src[1])) is None or any(axis not in affine[1] for axis in load_affine[1]) for load in loads)): return None  # noqa: E501
    if None in (plans:=typing_cast(tuple[RKTypedLoadPlan, ...], tuple(_typed_load_plan(load,dtypes.half,out_index,rows) for load in loads))): return None  # noqa: E501
    parsed.append((plans,weight)); cells += rows*len(loads)
  # A is row-stable and B is column-stable; explicit offset tables allow source-stride differences and broadcasts.
  def value(plan:RKTypedLoadPlan, lane:int) -> int: gather=plan.gather; return gather.offsets[lane] if gather.offsets else gather.base+sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes)  # noqa: E501
  def align(m:int, n:int, lanes:tuple[int, ...]=()) -> tuple[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float], ...]:
    aligned:list[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float]] = []
    for operands,weight in parsed:
      def lane(i:int) -> int: return lanes[i] if lanes else i
      if not operands: aligned.append((None,None,weight)); continue
      row,col = zip(*(((not lanes and not plan.gather.offsets and all(limit == 1 or divisor%n == 0 for divisor,limit,_ in plan.gather.axes),not lanes and not plan.gather.offsets and all(limit == 1 or n%divisor == 0 and n//divisor%limit == 0 for divisor,limit,_ in plan.gather.axes)) if cells > _MAX_DYNAMIC_SELECTOR_CELLS else (all(value(plan,lane(i*n+j)) == value(plan,lane(i*n)) for i in range(m) for j in range(n)),all(value(plan,lane(i*n+j)) == value(plan,lane(j)) for i in range(m) for j in range(n)))) for plan in operands))  # noqa: E501
      order = ((0,None) if row[0] else (None,0) if col[0] else None) if len(operands) == 1 else (0,1) if row[0] and col[1] else (1,0) if row[1] and col[0] else None  # noqa: E501
      if order is None: return ()
      aligned.append((None if order[0] is None else operands[order[0]],None if order[1] is None else operands[order[1]],weight))
    return tuple(aligned)
  candidates = [(diagonal,not any(len(operands) == 2 for operands,_ in parsed) and n != 1,bool(lanes),m*ai+ao*ai+2*m*ao,-n,ai,ao,m,n,lanes,outputs,normalized) for diagonal,m,n,lanes,outputs,normalized in itertools.chain(  # noqa: E501
    ((False,m,n,lanes,outputs,align(m,n,lanes)) for affine in (_linear_index(out_index),) for m,n,lanes,outputs in itertools.chain(((rows//n,n,(),()) for n in range(1,rows+1) if rows%n == 0),  # noqa: E501
      ((limit,rows//limit,tuple(high*stride*limit+row*stride+low for row in range(limit) for high in range(rows//stride//limit) for low in range(stride)),tuple((i//stride%limit)*(rows//limit)+i//(stride*limit)*stride+i%stride for i in range(rows))) for _,stride,limit in ((_affine_output_axes(typing_cast(tuple[int, dict[UOp, int]], affine),rows) if affine is not None else None) or ())))),  # noqa: E501
    ((True,rows,rows,(),tuple(i*rows+i for i in range(rows)),tuple((operands[0] if operands else None,operands[1] if len(operands) == 2 else None,weight) for operands,weight in parsed)),))  # noqa: E501
    for ai,ao,_ in (_cmac_layout(n,groups),) if m <= 0x7ff and ai <= 0xffff and ao <= 0x3fff and m*ai*2 <= 10*32768 and (m == 1 or ai <= 12*32) and normalized]  # noqa: E501
  if not candidates: return None
  diagonal,_,_,_,_,ai,ao,m,n,lanes,outputs,normalized = min(candidates, key=lambda item:item[:5]); fp16 = out.dtype.scalar() is dtypes.half
  a_cells = tuple(((source.param.arg.slot,value(source,lanes[row if diagonal else row*n] if lanes else row if diagonal else row*n)) if (source:=normalized[k][0]) is not None else (None,_fp16_bits(1.0 if normalized[k][1] is None else normalized[k][2]))) if k < groups else (None,0) for row in range(m) for k in range(ai))  # noqa: E501
  b_cells = tuple(((source.param.arg.slot,value(source,lanes[ob*16+ni] if lanes else ob*16+ni)) if (source:=normalized[k][1]) is not None else (None,_fp16_bits(normalized[k][2]))) if ob*16+ni < n and (k:=ib*32+ki) < groups else (None,0) for ob in range(ao//16) for ib in range(ai//32) for ni in range(16) for ki in range(32))  # noqa: E501
  gathers = tuple(gather for dst,cells in enumerate((a_cells,b_cells)) for sources,values in ((tuple(dict.fromkeys(source for source,_ in cells if source is not None)),tuple(value if source is None else 0 for source,value in cells)),) for seeded in (not sources or any(values),) for gather in (((RKGather(out.arg.slot,dst,len(cells),values=values),) if seeded else ()) + tuple(RKGather(source,dst,len(cells),offsets=tuple(value if owner == source else -1 for owner,value in cells),partial=seeded or bool(i)) for i,source in enumerate(sources))))  # noqa: E501
  if sum(gather.count for gather in gathers)+rows > _MAX_DYNAMIC_SELECTOR_CELLS: return None
  output_offsets = tuple(row*ao*(2 if fp16 else 1)+(col//16*32+col%16 if fp16 else col) for i in (outputs or range(rows)) for row,col in (divmod(i,n),))  # noqa: E501
  return RKImage(RKTarget.RK3588,(RKScratch(m*ai*2),RKScratch(ao*ai*2),RKScratch(m*ao*4)),gathers=gathers,post_gathers=(RKGather(2,out.arg.slot,rows,offsets=output_offsets,dst_kind=RKBufferKind.ARG,itemsize=2 if fp16 else 4,  # noqa: E501
    src_kind=RKBufferKind.SCRATCH),),cmac=RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),m,n,groups,fp16,relu_root is not None))  # noqa: E501

def _stripe_layout(count:int, rows:int) -> tuple[int, int, int]:
  vector_bytes = (count*2+63)&-64; return vector_bytes, vector_bytes//2, rows*vector_bytes//2

class _RKBuilder:
  """Allocate physical scratch and append native stages without owning semantic lowering."""
  def __init__(self, minimum:int=0):
    self.minimum = minimum; self.sizes,self.gathers,self.ops = typing_cast(tuple[list[int], list[RKGather], list[RKEWOp]], ([], [], []))
  def scratch(self, size:int, addend:int=0) -> RKArg:
    self.sizes.append(max(self.minimum, size)); return RKArg(RKBufferKind.SCRATCH, len(self.sizes)-1, addend)
  def constant(self, source:int, count:int, value:int, itemsize:int=2, dst:RKArg|None=None) -> RKArg:
    if dst is None: dst = self.scratch(count*itemsize)
    self.gathers.append(RKGather(source, dst.index, count, values=(value,)*count, itemsize=itemsize)); return dst
  def i16(self, lhs:RKArg, rhs:RKArg, count:int, cfg:int, dst:RKArg|None=None) -> RKArg:
    if dst is None: dst = self.scratch(count*2)
    self.ops.append(RKEWOp(dst, lhs, rhs, count, cfg, **_INT16_EW)); return dst
  def image(self, constants:bytes=b"", **kwargs) -> RKImage:
    return RKImage(RKTarget.RK3588, tuple(RKScratch(size) for size in self.sizes), constants,
                   gathers=tuple(self.gathers), ew_ops=tuple(self.ops), **kwargs)

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int, int16:bool=False) -> RKArg:
  """Append a balanced row reduction, making its first dependent stage self-contained."""
  first = not int16
  while len(active) > 1:
    for lhs,rhs in zip(active[::2], active[1::2]):
      ops.append(RKEWOp(lhs, lhs, rhs, count, cfg, submit_barrier=first, stateful=first, int16_input=int16, int16_output=int16)); first = False  # noqa: E501
    active = active[::2]
  return active[0]

def _masked_rows(builder:_RKBuilder, value:RKArg, mask:RKArg, rows:int, lanes:int, count:int) -> RKArg:
  selected = builder.i16(value, mask, rows*lanes, _EW_CFG[Ops.MUL])
  return _reduce_rows(builder.ops, [replace(selected, addend=row*lanes*2) for row in range(rows)], count, _EW_CFG[Ops.ADD], int16=True)

def _lower_fp16_uint8_cast(output:RKOutput) -> RKImage|None:
  """Truncate FP16 modulo 256 on DPU, convert to INT16, then expose each low byte."""
  root, zero = output[4], UOp.const(0.0, dtypes.half); source = _typed_cast_source(root, dtypes.uchar, dtypes.half)
  if source is None and root.op is Ops.WHERE and root.dtype.scalar() is dtypes.uchar and len(root.src) == 3:
    condition, positive, fallback = root.src
    source = _typed_cast_source(positive, dtypes.uchar, dtypes.half)
    if source is None or condition.op is not Ops.CMPLT or condition.src[0].op is not Ops.CONST or float(condition.src[0].arg) != 0.0 or \
       condition.src[1].key != source.key or fallback.op is not Ops.CONST or int(fallback.arg) != 0: return None
    source = source.alu(Ops.MAX, zero)
  if source is None: return None
  if (relu:=_relu_operand(source)) is not None: source = relu.alu(Ops.MAX, zero)
  truncated = _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(source,)))
  quotient = UOp(Ops.MAX, dtypes.half, src=((floor_input:=truncated.alu(Ops.MUL, UOp.const(1.0/256.0, dtypes.half))), floor_input), arg=_NATIVE_FLOOR)
  return _typed_half_image(output, truncated.alu(Ops.SUB, quotient.alu(Ops.MUL, UOp.const(256.0, dtypes.half))), False)

def _int16_byte_bits(ops:list[RKEWOp], alloc:Callable[[], RKArg], const:dict[int, RKArg], value:RKArg, lanes:int, weighted=False)->tuple[RKArg, ...]:
  """Split unsigned byte lanes into exact native planes, optionally retaining each bit's scale."""
  result, remainder = typing_cast(list[RKArg|None], [None]*8), value
  for bit in range(7, 0, -1):
    delta, positive, flag, scaled, next_remainder = (alloc() for _ in range(5))
    ops.extend(_ew_ops(((delta, remainder, const[(1<<bit)-1], Ops.SUB),
      (positive, delta, const[0], Ops.MAX), (flag, positive, const[1], _EW_CFG_MIN),
      (scaled, flag, const[1<<bit], Ops.MUL), (next_remainder, remainder, scaled, Ops.SUB)), lanes, **_INT16_EW))
    result[bit], remainder = scaled if weighted else flag, next_remainder
  result[0] = remainder
  return typing_cast(tuple[RKArg, ...], tuple(result))

def _ordered_byte_less(ops:list[RKEWOp], allocate:Callable[[], RKArg], zero:RKArg, one:RKArg,
                       lhs_components:Iterable[RKArg], rhs_components:Iterable[RKArg], lanes:int) -> RKArg:
  less, equal = zero, one
  for lhs,rhs in zip(lhs_components, rhs_components):
    maximum, lhs_delta, rhs_delta, lhs_less, rhs_less, unequal, same, selected, next_less, next_equal = (allocate() for _ in range(10))
    ops.extend(_ew_ops(((maximum, lhs, rhs, Ops.MAX), (lhs_delta, maximum, lhs, Ops.SUB),
      (rhs_delta, maximum, rhs, Ops.SUB), (lhs_less, lhs_delta, one, _EW_CFG_MIN),
      (rhs_less, rhs_delta, one, _EW_CFG_MIN), (unequal, lhs_less, rhs_less, Ops.MAX),
      (same, one, unequal, Ops.SUB), (selected, equal, lhs_less, Ops.MUL),
      (next_less, less, selected, Ops.MAX), (next_equal, equal, same, Ops.MUL)), lanes, **_INT16_EW))
    less, equal = next_less, next_equal
  return less

def _candidate_gather(src:int, dst:int, rows:tuple[tuple[int, ...], ...], lanes:int) -> RKGather:
  """Gather candidate rows compactly when their active rectangle is affine."""
  n, base = len(rows[0]), rows[0][0]; dx, dy = (rows[0][1]-base if n > 1 else 0), (rows[1][0]-base if len(rows) > 1 else 0)
  return RKGather(src, dst, len(rows)*lanes, base, ((1, n, dx), (lanes, len(rows), dy)), dst_stride=2, itemsize=1) if lanes%n == 0 and \
    all(value == base+r*dy+c*dx for r,row in enumerate(rows) for c,value in enumerate(row)) else \
    RKGather(src, dst, len(rows)*lanes, offsets=tuple(x for row in rows for x in (*row, *(-1,)*(lanes-n))), dst_stride=2, itemsize=1)

def _bounded_index_gate(gate:UOp, bounded:UOp, limit:int|None=None) -> int|None:
  """Return the exact positive bound proved by the canonical conjunction, or None."""
  if gate.op is not Ops.AND: return None
  nodes = gate.toposort()
  limits = {int(u.src[1].arg) for u in nodes if u.op is Ops.CMPLT and u.src[0].key == bounded.key and
            u.src[1].op is Ops.CONST and int(u.src[1].arg) > 0}
  if limit is None: limit = next(iter(limits)) if len(limits) == 1 else None
  if limit not in limits: return None
  return limit if any((comparison:=_inverted_condition(u, typed=True)) is not None and comparison.op is Ops.CMPLT and
    comparison.src[0].key == bounded.key and comparison.src[1].op is Ops.CONST and int(comparison.src[1].arg) == 0 for u in nodes) else None

def _native_int16_byte_mask(builder:_RKBuilder, index_slot:int, index_offsets:tuple[int, ...]|tuple[tuple[int, ...], ...],
                            coordinate_sets:tuple[tuple[tuple[int, ...], ...], ...], count:int, vector_lanes:int) -> RKArg|None:
  """Compare arbitrary INT32 values exactly as four unsigned bytes using native INT16 DPU EW."""
  rows = len(coordinate_sets[0]) if coordinate_sets else 0
  if not rows or any(len(group) != rows or any(len(row) != count for row in group) for group in coordinate_sets): return None
  matrix_lanes = rows*vector_lanes
  offset_rows = (index_offsets,)*rows if index_offsets and isinstance(index_offsets[0], int) else cast(tuple[tuple[int, ...], ...], index_offsets)
  if len(offset_rows) != rows or any(len(offsets) != count for offsets in offset_rows): return None
  one, diff, magnitude, unequal = builder.constant(index_slot, matrix_lanes, 1), *(builder.scratch(matrix_lanes*2) for _ in range(3))
  masks:list[RKArg] = []
  for coordinates in coordinate_sets:
    byte_masks:list[RKArg] = []
    for byte in range(4):
      dynamic, static, equal = (builder.scratch(matrix_lanes*2) for _ in range(3))
      builder.gathers.append(_candidate_gather(index_slot, dynamic.index,
        tuple(tuple(offset*4+byte for offset in offsets) for offsets in offset_rows), vector_lanes))
      builder.gathers.append(RKGather(index_slot, static.index, matrix_lanes,
        values=tuple((value >> (byte*8)) & 0xff for row in coordinates for value in (*row, *((0,)*(vector_lanes-count)))), itemsize=2))
      for lhs,rhs,cfg,dst in ((dynamic, static, _EW_CFG[Ops.SUB], diff), (diff, diff, _EW_CFG_ABS, magnitude),
                              (magnitude, one, _EW_CFG_MIN, unequal)): builder.i16(lhs, rhs, matrix_lanes, cfg, dst)
      byte_masks.append(builder.i16(one, unequal, matrix_lanes, _EW_CFG[Ops.SUB], equal))
    masks.append(_reduce_byte_masks(builder, byte_masks, matrix_lanes, _EW_CFG[Ops.MUL]))
  return _reduce_byte_masks(builder, masks, matrix_lanes, _EW_CFG[Ops.MAX])

def _reduce_byte_masks(builder:_RKBuilder, masks:list[RKArg]|tuple[RKArg, ...], count:int, cfg:int, *, in_place:bool=False) -> RKArg:
  return functools.reduce(lambda result,value: builder.i16(result, value, count, cfg, result if in_place else None), masks[1:], masks[0])

def _prefix_valid(builder:_RKBuilder, total:RKArg, coordinate:RKArg, zero:RKArg, one:RKArg, count:int) -> tuple[RKArg, RKArg]:
  delta, positive, valid, remaining = (builder.scratch(count*2) for _ in range(4))
  for lhs,rhs,cfg,dst in ((total, coordinate, _EW_CFG[Ops.SUB], delta), (delta, zero, _EW_CFG[Ops.MAX], positive),
                          (positive, one, _EW_CFG_MIN, valid), (one, valid, _EW_CFG[Ops.SUB], remaining)):
    builder.i16(lhs, rhs, count, cfg, dst)
  return valid, remaining

def _full_predicate_count(expr:UOp, out_index:UOp, count:int, dtype:DType, predicate:Callable[[UOp], UOp|None],
                          max_scale:int=1) -> tuple[UOp, int]|None:
  """Prove an unrolled sum covers every typed source predicate uniformly, possibly repeated or scaled."""
  terms, source, offsets, scales = list(_iter_binary(expr, Ops.ADD)), None, [], []
  try:
    for term in terms:
      scale = 1
      if max_scale > 1 and term.op is Ops.MUL:
        constants = [u for u in term.src if u.op is Ops.CONST and u.dtype.scalar() is dtypes.int]
        if len(constants) != 1 or len(term.src)-len(constants) != 1: return None
        scale, term = int(constants[0].arg), next(u for u in term.src if u not in constants)
      if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.int or len(term.src) != 1 or \
         (load:=predicate(term.src[0])) is None or [u for u in term.toposort() if u.op is Ops.LOAD] != [load]: return None
      if (parsed:=_typed_load_plan(load, dtype, out_index, count, require_offsets=True)) is None or len(set(parsed.gather.offsets)) != 1: return None
      if source is not None and parsed.param.arg.slot != source.arg.slot: return None
      source = parsed.param
      offsets.append(parsed.gather.offsets[0])
      scales.append(scale)
  except RuntimeError: return None
  if source is None or not scales or len(set(scales)) != 1: return None
  if (source_count:=int(source.src[0].arg)) <= 0 or len(terms)%source_count or any(offsets.count(i) != len(terms)//source_count for i in range(source_count)): return None  # noqa: E501
  return (source, scales[0]*(len(terms)//source_count)) if 1 <= scales[0]*(len(terms)//source_count) <= max_scale else None

def _lower_bounded_integer_predicate_coordinates(output:RKOutput, dtype:DType=dtypes.int) -> RKImage|None:
  """Prove and execute bounded integer predicate coordinates through native INT16 byte masks."""
  _, out, count, out_index, root = output
  if not 1 <= count <= _FP16_EXACT_INTEGER or root.op is not Ops.WHERE or len(root.src) != 3: return None
  if (fill:=root.src[2]).op is not Ops.CONST or fill.dtype.scalar() is not dtypes.int: return None
  fill_value = int(fill.arg)
  totals = [(u.src[1], info) for u in root.toposort() if u.op is Ops.CMPLT and u.src[0].key == out_index.key and
            (info:=_full_predicate_count(u.src[1], out_index, count, dtype, lambda x:_nonzero_load(x, dtype), 8)) is not None]
  if len(totals) != 1: return None
  total_expr, (source, rank) = totals[0]
  if (coordinate_count:=int(source.src[0].arg)*rank) < 1 or coordinate_count > _FP16_EXACT_INTEGER: return None
  source_loads = {u.key for u in total_expr.toposort() if u.op is Ops.LOAD}
  index_loads = [u for u in root.toposort() if u.op is Ops.LOAD and u.key not in source_loads]
  if len(index_loads) != 1 or {u.key for u in root.toposort() if u.op is Ops.LOAD} != source_loads|{index_loads[0].key}: return None
  index_load = index_loads[0]
  if (index_plan:=_typed_load_plan(index_load, dtypes.int, out_index, count, require_offsets=True)) is None or \
      int(index_plan.param.src[0].arg) != count: return None
  try:
    coordinate_rows = tuple(_static_values(out_index, root.substitute(
      {total_expr:total_expr.const_like(count), index_load:index_load.const_like(i)}), count, int) for i in range(coordinate_count))
    for selected_count in range(count+1):
      got = _static_values(out_index, root.substitute({total_expr:total_expr.const_like(selected_count), index_load:index_load.const_like(0)}), count, int)  # noqa: E501
      if got != tuple(coordinate_rows[0][lane] if lane < selected_count else fill_value for lane in range(count)): return None
    if index_plan.gather.offsets != tuple(range(count)) or not -32768 <= fill_value <= 32767 or \
       any(not -32768 <= value <= 32767 for row in coordinate_rows for value in row): return None
  except (RuntimeError, OverflowError, struct.error): return None
  index_param, index_offsets = index_plan.param, index_plan.gather.offsets
  source_count, coordinate_count = int(source.src[0].arg), len(coordinate_rows); _, vector_lanes, matrix_lanes = _stripe_layout(count, coordinate_count)  # noqa: E501
  if matrix_lanes > _MAX_EW_ELEMS_FP16: return None

  builder, source_vector_lanes = _RKBuilder(64), _stripe_layout(1, source_count)[1]; source_matrix_lanes = source_count*source_vector_lanes
  one, byte_masks = builder.constant(source.arg.slot, source_matrix_lanes, 1), tuple(builder.scratch(source_matrix_lanes*2) for _ in range(dtype.itemsize))  # noqa: E501
  for byte,value in enumerate(byte_masks):
    builder.gathers.extend(RKGather(source.arg.slot, value.index, 1,
      offsets=(i*dtype.itemsize+byte,), dst_addend=i*source_vector_lanes*2, dst_stride=2, itemsize=1) for i in range(source_count))
  for value in byte_masks: builder.i16(value, one, source_matrix_lanes, _EW_CFG_MIN, value)
  source_mask = _reduce_byte_masks(builder, byte_masks, source_matrix_lanes, _EW_CFG[Ops.MAX], in_place=True)
  total = _reduce_rows(builder.ops, [replace(source_mask, addend=source_mask.addend+row*64) for row in range(source_count)], 1, _EW_CFG[Ops.ADD], int16=True)  # noqa: E501
  if rank != 1: builder.i16(total, builder.constant(source.arg.slot, 1, _int16_bits(rank)), 1, _EW_CFG[Ops.MUL], total)
  candidates = tuple((candidate,)*count for candidate in range(coordinate_count))
  if (equal:=_native_int16_byte_mask(builder, index_param.arg.slot, index_offsets, (candidates,), count, vector_lanes)) is None: return None
  gather_after, total_vector, output_coordinate, zero, one, coordinate_matrix = len(builder.ops),*(builder.scratch(matrix_lanes*2) for _ in range(5))
  mid = (RKGather(total.index, total_vector.index, count, offsets=(total.addend//2,)*count, src_kind=RKBufferKind.SCRATCH),)
  builder.gathers.extend(RKGather(source.arg.slot, coordinate_matrix.index, count,
    values=tuple(_int16_bits(value) for value in row), dst_addend=i*vector_lanes) for i,row in enumerate(coordinate_rows))
  builder.gathers.extend((RKGather(source.arg.slot, output_coordinate.index, count, values=tuple(range(count))),
                          RKGather(source.arg.slot, zero.index, matrix_lanes, values=(0,)*matrix_lanes),
                          RKGather(source.arg.slot, one.index, matrix_lanes, values=(1,)*matrix_lanes)))
  fill_value_arg = builder.constant(source.arg.slot, count, _int16_bits(fill_value))
  valid, remaining, reduced = (*_prefix_valid(builder, total_vector, output_coordinate, zero, one, count),
                               _masked_rows(builder, coordinate_matrix, equal, coordinate_count, vector_lanes, count))
  guarded, fill_part, result = (builder.scratch(matrix_lanes*2) for _ in range(3))
  for lhs,rhs,cfg,dst in ((reduced, valid, _EW_CFG[Ops.MUL], guarded), (fill_value_arg, remaining, _EW_CFG[Ops.MUL], fill_part),
                          (guarded, fill_part, _EW_CFG[Ops.ADD], result)): builder.i16(lhs, rhs, count, cfg, dst)
  builder.ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out.arg.slot), result, result, count, _EW_CFG[Ops.MAX], int16_input=True, int32_output=True))
  return builder.image(mid_gathers=mid, gather_after=gather_after)

def _lower_dynamic_typed_load(output:RKOutput, dtype:DType=dtypes.half) -> RKImage|None:
  """Parse and materialize one bounded dynamic typed LOAD as exact candidate-major raw bytes."""
  _, out_param, count, out_index, root = output
  total_gate:tuple[UOp, UOp]|None = None
  load = root
  if root.op is Ops.WHERE and len(root.src) == 3:
    condition, load, fill = root.src
    if dtype is dtypes.half:
      if load.op is not Ops.LOAD or fill.op is not Ops.CONST or len(load.src) <= 2 or load.src[1].op is not Ops.CONST: return None
      load_gate, same_default = load.src[2], float(load.src[1].arg) == float(fill.arg)
      if not same_default and not (_same_condition(condition, load_gate) or
         condition.op is Ops.AND and any(_same_condition(x, load_gate) for x in condition.src)): return None
      load = load.replace(src=(load.src[0], fill, condition.alu(Ops.AND, load_gate) if same_default else condition))
    else:
      if (dtype not in (dtypes.int16, dtypes.int) or not 1 <= count <= _FP16_EXACT_INTEGER or condition.op is not Ops.CMPLT or
          _strip_cast(condition.src[0]).key != out_index.key or fill.op is not Ops.CONST or fill.dtype.scalar() is not dtype or
          (mask_info:=_full_predicate_count(condition.src[1], out_index, count, dtypes.bool, lambda u:u if u.op is Ops.LOAD and
            u.dtype.scalar() is dtypes.bool and u.src[0].op is Ops.INDEX else None)) is None or mask_info[1] != 1): return None
      total_gate = mask_info[0], fill
  if (count <= 0 or load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or len(load.src) != 3 or load.src[0].op is not Ops.INDEX or
      load.src[1].op is not Ops.CONST or load.src[1].arg != 0): return None
  data_param, data_index, gate, gate_nodes = _root_param(load.src[0]), load.src[0].src[1], load.src[2], load.src[2].toposort()
  bool_loads = tuple(u for u in gate_nodes if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.bool)
  address_nodes = data_index.toposort()
  normalized = tuple((u, index, int(addition[1].arg)) for u in address_nodes
    if u.op is Ops.WHERE and u.src[0].op is Ops.CMPLT and u.src[0].src[1].op is Ops.CONST and int(u.src[0].src[1].arg) == 0
    and (index:=u.src[0].src[0]).op is Ops.LOAD and index.dtype.scalar() is dtypes.int and u.src[2].key == index.key
    and (addition:=_const_operand(u.src[1], Ops.ADD)) is not None and addition[0].key == index.key and int(addition[1].arg) > 0)
  normalized_by_load = {dynamic.key:(root, extent) for root,dynamic,extent in normalized}
  dynamic_loads = tuple({u.key:u for u in address_nodes if u.op is Ops.LOAD and u.dtype.scalar() is dtypes.int}.values())
  if not dynamic_loads or len(normalized_by_load) != len(normalized): return None
  dynamic_axes:list[tuple[UOp, UOp, int, bool]] = []
  for dynamic in dynamic_loads:
    axis, extent, wrapped = (*normalized_by_load[dynamic.key], True) if dynamic.key in normalized_by_load else \
      (dynamic, _bounded_index_gate(gate, dynamic), False)
    if extent is None or wrapped and _bounded_index_gate(gate, axis, extent) is None: return None
    dynamic_axes.append((axis, dynamic, extent, wrapped))
  axes = tuple(dynamic_axes)
  loads = tuple(axis[1] for axis in axes); params = typing_cast(tuple[UOp, ...], tuple(_root_param(u.src[0]) if u.src and u.src[0].op is Ops.INDEX else None for u in loads))  # noqa: E501
  if (data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or len(bool_loads) > 1 or
      any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in params) or
      {u.key for u in gate_nodes if u.op is Ops.LOAD} != {u.key for u in (*loads, *bool_loads)}): return None
  bool_load = bool_loads[0] if bool_loads else None
  bool_param = _root_param(bool_load.src[0]) if bool_load is not None and len(bool_load.src) == 1 and bool_load.src[0].op is Ops.INDEX else None
  if bool_load is not None and (bool_load not in _iter_binary(gate, Ops.AND) or bool_param is None or bool_param.dtype.scalar() is not
                                dtypes.bool or bool_param.src[0].op is not Ops.CONST): return None
  if total_gate is not None and (bool_load is not None or len(axes) != 1 or data_index.key != loads[0].key or axes[0][3]): return None
  if any(not 0 <= param.arg.slot <= _RKIMAGE_U16_MAX for param in (out_param, data_param, *params,
         *((bool_param,) if bool_param is not None else ()), *((total_gate[0],) if total_gate is not None else ()))): return None
  candidates, data_count = math.prod(extent for _,_,extent,_ in axes), int(data_param.src[0].arg)
  if total_gate is not None and (data_count != int(total_gate[0].src[0].arg) or candidates != data_count or int(params[0].src[0].arg) != count):
    return None
  alternative_count, external_count, pre_repeat = sum(1+wrapped for *_,wrapped in axes), int(bool_load is not None), min(8, count)
  total_scratch, total_gathers = (0, 0) if total_gate is None else (5+3*dtype.itemsize, data_count+3+dtype.itemsize)
  if (pre_vl:=_stripe_layout(count, 2)[1] if candidates > 1 else count) > _MAX_EW_ELEMS_FP16: return None
  pre_blocks = ceildiv(candidates, _MAX_EW_ELEMS_FP16//pre_vl)
  if candidates > min(_MAX_STATIC_RANGE_ENVS, _MAX_DYNAMIC_SELECTOR_CELLS//count) or any(size > _RKIMAGE_U16_MAX for size in (
     pre_blocks*(3*len(axes)+16*alternative_count+2*pre_repeat*dtype.itemsize)+external_count+total_scratch,
     pre_blocks*(len(axes)+8*alternative_count+pre_repeat*dtype.itemsize)+external_count+pre_repeat*dtype.itemsize+total_gathers)): return None
  combinations = tuple(itertools.product(*(range(extent) for _,_,extent,_ in axes)))
  external:tuple[int, tuple[int, ...]]|None = None
  if bool_load is not None and bool_param is not None:
    try: bool_offsets = _gather_offsets(out_index, bool_load.src[0].src[1], None, count)
    except RuntimeError: return None
    if any(not 0 <= offset < int(bool_param.src[0].arg) for offset in bool_offsets): return None
    external = bool_param.arg.slot, bool_offsets
  try:
    index_offsets = tuple(_gather_offsets(out_index, dynamic.src[0].src[1], None, count) for dynamic in loads)
    plans = tuple(_gather_plan(data_param.arg.slot, 0, out_index, data_index.substitute(mapping), gate.substitute(mapping |
      ({bool_load:bool_load.const_like(True)} if bool_load is not None else {})), count) for values in combinations
      for mapping in ({dynamic:dynamic.const_like(value) for dynamic,value in zip(loads, values)},))
    plan_offsets = tuple(plan.offsets or tuple(plan.base+sum((lane//d%n)*s for d,n,s in plan.axes) for lane in range(plan.count)) for plan in plans)
  except RuntimeError: return None
  if (any(not 0 <= offset < int(param.src[0].arg) or offset*4+3 > dtypes.int.max for offsets,param in zip(index_offsets, params)
          for offset in offsets) or any(not 0 <= offset < data_count or offset*dtype.itemsize+dtype.itemsize-1 > dtypes.int.max
          for row in plan_offsets for offset in row) or external is not None and any(offset > dtypes.int.max for offset in external[1])): return None
  if total_gate is not None and index_offsets[0] != tuple(range(count)): return None
  repeat = next((width for width in range(min(8, count), 0, -1) if count%width == 0 and all(all(
    row[start:start+width] == (row[start],)*width for start in range(0, count, width)) for row in (*index_offsets,
    *((external[1],) if external else ())))), 1)
  group_count, grouped_indices = count//repeat, tuple((param.arg.slot, offsets[::repeat]) for param,offsets in zip(params, index_offsets))
  grouped_gate = None if external is None else (external[0], external[1][::repeat])
  if (vector_lanes:=_stripe_layout(group_count, 2)[1] if candidates > 1 else group_count) > _MAX_EW_ELEMS_FP16: return None
  block_count = ceildiv(candidates, block_rows:=_MAX_EW_ELEMS_FP16//vector_lanes)
  if any(size > _RKIMAGE_U16_MAX for size in (block_count*(3*len(axes)+16*alternative_count+2*repeat*dtype.itemsize)+external_count+total_scratch,
     block_count*(len(axes)+8*alternative_count+repeat*dtype.itemsize)+external_count+repeat*dtype.itemsize+total_gathers)): return None
  builder, blocks = _RKBuilder(), []
  for start,stop in ((start,min(candidates,start+block_rows)) for start in range(0,candidates,block_rows)):
    rows, matrix_lanes, axis_masks = stop-start, (stop-start)*vector_lanes, typing_cast(list[RKArg], [])
    for axis_index,((index_slot, offsets), values) in enumerate(zip(grouped_indices, zip(*combinations))):
      alternatives = (values, *((tuple(value-axes[axis_index][2] for value in values),) if axes[axis_index][3] else ()))
      coordinate_sets = tuple(tuple((value,)*group_count for value in alternative[start:stop]) for alternative in alternatives)
      if (mask:=_native_int16_byte_mask(builder, index_slot, offsets, coordinate_sets, group_count, vector_lanes)) is None: return None
      axis_masks.append(mask)
    mask = _reduce_rows(builder.ops, axis_masks, matrix_lanes, _EW_CFG[Ops.MUL], int16=True)
    block:list[list[RKArg]] = []
    for channel in range(repeat):
      result:list[RKArg] = []
      for byte in range(dtype.itemsize):
        raw, raw_rows = builder.scratch(matrix_lanes*2), tuple(tuple(x*dtype.itemsize+byte for x in row[channel::repeat]) for row in plan_offsets[start:stop])  # noqa: E501
        builder.gathers.append(_candidate_gather(data_param.arg.slot, raw.index, raw_rows, vector_lanes))
        result.append(_masked_rows(builder, raw, mask, rows, vector_lanes, group_count))
      block.append(result)
    blocks.append(block)
  results = blocks[0]
  for block in blocks[1:]:
    for result,other in zip(results, block):
      for value,part in zip(result, other): builder.i16(value, part, group_count, _EW_CFG[Ops.ADD], value)
  if grouped_gate is not None:
    gate_value = builder.scratch(group_count*2)
    builder.gathers.append(RKGather(grouped_gate[0], gate_value.index, group_count, offsets=grouped_gate[1], dst_stride=2, itemsize=1))
    for result in results:
      for value in result: builder.i16(value, gate_value, group_count, _EW_CFG[Ops.MUL], value)
  mid, gather_after = typing_cast(tuple[RKGather, ...], ()), 0
  if total_gate is not None:
    mask_param, fill = total_gate
    bool_values = builder.scratch(data_count*64)
    builder.gathers.extend(RKGather(mask_param.arg.slot, bool_values.index, 1, offsets=(lane,), dst_addend=lane*64, dst_stride=2, itemsize=1)
                           for lane in range(data_count))
    total = _reduce_rows(builder.ops, [replace(bool_values, addend=lane*64) for lane in range(data_count)], 1, _EW_CFG[Ops.ADD], int16=True)
    gather_after, total_vector, coordinate, zero, one = len(builder.ops), *(builder.scratch(count*2) for _ in range(4))
    mid = (RKGather(total.index, total_vector.index, count, offsets=(total.addend//2,)*count, src_kind=RKBufferKind.SCRATCH),)
    builder.gathers.append(RKGather(mask_param.arg.slot, coordinate.index, count, values=tuple(range(count))))
    for dst,constant in ((zero,0),(one,1)): builder.constant(mask_param.arg.slot, count, constant, dst=dst)
    valid, _ = _prefix_valid(builder, total_vector, coordinate, zero, one, count)
    fill_bits = int(fill.arg) & ((1 << (dtype.itemsize*8))-1)
    for byte,value in enumerate(results[0]):
      fill_value = builder.constant(mask_param.arg.slot, count, (fill_bits >> (byte*8)) & 0xff)
      selected = builder.i16(builder.i16(value, fill_value, count, _EW_CFG[Ops.SUB]), valid, count, _EW_CFG[Ops.MUL])
      builder.i16(fill_value, selected, count, _EW_CFG[Ops.ADD], value)
  terminal = tuple(_raw_gather(RKArg(RKBufferKind.SCRATCH, value.index, value.addend), out_param.arg.slot, group_count,
    dst_stride=repeat*dtype.itemsize, dst_addend=channel*dtype.itemsize+byte, offsets=tuple(range(value.addend, value.addend+group_count*2, 2)))
    for channel,result in enumerate(results) for byte,value in enumerate(result))
  image = builder.image(mid_gathers=mid, gather_after=gather_after, post_gathers=terminal)
  gathers = image.gathers+image.mid_gathers+image.post_gathers
  if any(size > _RKIMAGE_U16_MAX for size in (len(image.scratch), len(gathers))) or \
     any(not 0 <= index <= _RKIMAGE_U16_MAX for gather in gathers for index in (gather.src_index, gather.dst_index)) or \
     any(not 0 <= arg.index <= _RKIMAGE_U16_MAX for op in image.ew_ops for arg in (op.dst, op.lhs, op.rhs)): return None
  return image

def _nonzero_load(term:UOp, dtype:DType=dtypes.half) -> UOp|None:
  term = _unwrap_condition(term)
  if term.op is not Ops.CMPNE: return None
  candidates = [loaded for load,zero in (term.src, term.src[::-1]) if
                (loaded:=load if load.op is Ops.LOAD else load.load()).dtype.scalar() is dtype and
                loaded.src[0].op is Ops.INDEX and zero.op is Ops.CONST and zero.arg == 0]
  return candidates[0] if len(candidates) == 1 else None

def _typed_half_image(output:RKOutput, value:UOp, int32:bool, bool_output:bool=False) -> RKImage:
  """Lower an exact FP16 expression through the requested native integer output ABI."""
  store, out_param, count, _, _ = output
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_slot, replacement = out_param.arg.slot, store.replace(src=(store.src[0].replace(dtype=dtypes.half, src=(out_param.replace(dtype=dtypes.half, arg=replace(out_param.arg, dtype=dtypes.half)), *store.src[0].src[1:])), value, *store.src[2:]))  # noqa: E501
  image = _lower_uop_program(list(replacement.toposort()), vectorize_reductions=False) if int32 else \
    _lower_uop_program(list(UOp(Ops.SINK, src=(replacement,)).toposort()), vectorize_reductions=False, recipes_ready=True)
  if image is None: raise RuntimeError("RKPLAN_REJECT:composed_uops")
  terminal = [i for i,op in enumerate(image.ew_ops) if op.dst.kind is RKBufferKind.ARG and op.dst.index == out_slot]
  if terminal != [len(image.ew_ops)-1] or image.mid_gathers or image.post_gathers: raise RuntimeError("RKPLAN_REJECT:predicate_terminal" if int32 else "RKPLAN_REJECT:uint8_terminal")  # noqa: E501
  result_slot, auxiliary_slot = len(image.scratch), len(image.scratch)+1; prefix = (*image.ew_ops[:-1], replace(image.ew_ops[-1], dst=(result:=RKArg(RKBufferKind.SCRATCH, result_slot))))  # noqa: E501
  if int32:
    return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(ceildiv(count, 4)*64)), ew_ops=(*prefix,
      RKEWOp(RKArg(RKBufferKind.ARG, out_slot), result, RKArg(RKBufferKind.SCRATCH, auxiliary_slot), count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True, bool_output=bool_output)))  # noqa: E501
  return replace(image, scratch=(*image.scratch, RKScratch(_scratch_bytes(count)), RKScratch(_scratch_bytes(count))), ew_ops=(*prefix,
    RKEWOp(int_result:=RKArg(RKBufferKind.SCRATCH, auxiliary_slot), result, result, count, _EW_CFG[Ops.MAX], submit_barrier=True, stateful=True, int16_output=True)), post_gathers=(_raw_gather(int_result, out_slot, count),))  # noqa: E501

def _half_backed_value(value:UOp) -> UOp|None:
  """Normalize a half-backed numeric expression for the exact raw FP16 comparator."""
  original, value = value, _unwrap_condition(value)
  if value.op is Ops.INDEX: value = value.load()
  if value.op is Ops.CONST and value.dtype.scalar() is dtypes.weakfloat: value = UOp.const(float(value.arg), dtypes.half)
  if value.dtype.scalar() not in (dtypes.half, dtypes.float) and original.dtype.scalar() in (dtypes.half, dtypes.float): value = original
  valid = value.dtype.scalar() in (dtypes.half, dtypes.float) and not any(not load.src or load.src[0].op is not Ops.INDEX or
    (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtypes.half for load in value.toposort() if load.op is Ops.LOAD)
  return (value if value.dtype.scalar() is dtypes.half else value.cast(dtypes.half)) if valid else None

def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if (source:=_typed_cast_source(root, dtypes.bool, dtypes.half)) is not None: root = source != UOp.const(0.0, dtypes.half)
  if (load:=_nonzero_load(root)) is None: return None
  return _positive_mask(UOp(Ops.MAX, dtypes.half, src=(load, load), arg=_NATIVE_ABS))

@functools.lru_cache(maxsize=4096)
def _exact_int_range(root:UOp) -> tuple[int, int]|None:
  """Conservatively bound an integer UOp before choosing its exact physical scratch layout."""
  dtype = root.dtype.scalar(); valid = dtype in (dtypes.int, dtypes.weakint) and (root.op is Ops.CONST or
    root.op is Ops.RANGE and len(root.src) == 1 and root.src[0].op is Ops.CONST or
    root.op is Ops.CAST and len(root.src) == 1 and root.src[0].dtype.scalar() is dtypes.bool or
    root.op is Ops.WHERE and len(root.src) == 3 and all(_exact_int_range(src) is not None for src in root.src[1:]) or
    root.op is Ops.XOR and len(root.src) == 2 and any(marker.op is Ops.CONST and marker.arg == -1 and
      _exact_int_range(source) is not None for marker,source in (root.src, root.src[::-1])) or
    root.op is Ops.CMOD and len(root.src) == 2 and (right:=_exact_int_range(root.src[1])) is not None and right[0] == right[1] != 0 or
    root.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(root.src) == 2 and
      all(_exact_int_range(src) is not None for src in root.src))
  bounds = (int(root.vmin), int(root.vmax))
  return ((0,max(0,bounds[1])) if root.op is Ops.RANGE else bounds) if valid and dtype.min <= bounds[0] <= bounds[1] <= dtype.max else None

def _int_fp16_expr(u:UOp) -> UOp:
  """Represent an integer UOp whose values are exactly carried in FP16 lanes as a half-valued recipe."""
  if u.dtype.scalar() is not dtypes.int: raise _RKGenericReject
  if u.op is Ops.CONST: return UOp.const(float(int(u.arg)), dtypes.half)
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() in (dtypes.half, dtypes.bool):
    return _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=u.src)) if u.src[0].dtype.scalar() is dtypes.half else u.src[0].cast(dtypes.half)
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX) and len(u.src) == 2:
    return UOp(u.op, dtypes.half, src=tuple(_int_fp16_expr(src) for src in u.src), arg=u.arg)
  if u.op is Ops.WHERE and len(u.src) == 3:
    return UOp(Ops.WHERE, dtypes.half, src=(u.src[0], _int_fp16_expr(u.src[1]), _int_fp16_expr(u.src[2])), arg=u.arg)
  if u.op is Ops.CMOD and len(u.src) == 2:
    lhs, rhs = (_int_fp16_expr(src) for src in u.src)
    return lhs.alu(Ops.SUB, _fold_trunc(UOp(Ops.TRUNC, dtypes.half, src=(lhs.alu(Ops.FDIV, rhs),))).alu(Ops.MUL, rhs))
  raise _RKGenericReject(f"INT_FP16 recipe {u.op.name}")

class _RKGenericReject(Exception): pass

def _runtime_index(u:UOp) -> tuple[UOp, UOp, UOp, int]|None:
  """Return the index LOAD, its parameter, lane-address expression, and raw index width."""
  if ((u:=_strip_cast(u)).op is not Ops.LOAD or len(u.src) != 1 or u.src[0].op is not Ops.INDEX or (param:=_root_param(u.src[0])) is None or
      param.src[0].op is not Ops.CONST or param.dtype.scalar() not in (dtypes.int, dtypes.int16)): return None
  return u, param, u.src[0].src[1], param.dtype.scalar().itemsize

def _runtime_lane_offset(info:tuple[UOp, UOp, UOp, int], out_index:UOp, count:int) -> int|None:
  try: lane_offsets = _static_values(out_index, info[2], count, int)
  except RuntimeError: return None
  return lane_offsets[0] if lane_offsets == tuple(lane_offsets[0]+lane for lane in range(count)) and \
    0 <= lane_offsets[0] <= int(info[1].src[0].arg)-count else None

def _has_runtime_address(root:UOp) -> bool:
  """True when a value LOAD obtains its address or gate from another runtime LOAD."""
  return any(_runtime_index(node) is not None for load in root.toposort()
             if load.op is Ops.LOAD and load.src and load.src[0].op is Ops.INDEX
             for node in (*load.src[0].src[1].toposort(), *(load.src[2].toposort() if len(load.src) > 2 else ())))

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
  if ((u.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN, Ops.NEG) and len(u.src) == 1) or
      (u.op in (Ops.MUL, Ops.SUB, Ops.MAX) and len(u.src) == 2)):
    return UOp(u.op, dtypes.half, src=tuple(_fp32_expr_to_half(src) for src in u.src), arg=u.arg if u.op not in (Ops.MUL, Ops.NEG) else None)
  if u.op is Ops.ADD:
    return _precise_mul_sum(_fp32_add_terms(u))
  raise _RKGenericReject

def _nested_fp32_storage_cast(x:UOp) -> UOp|None:
  try: return _fp32_expr_to_half(x)
  except _RKGenericReject: return None

_pm_half_storage_algebra = PatternMatcher([(UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.float, name="x"),)), _nested_fp32_storage_cast),
  (UPat(Ops.FDIV, dtypes.half, src=(UPat.var("x"), UPat.var("y"))), lambda x,y:x.alu(Ops.MUL, UOp(Ops.RECIPROCAL, dtypes.half, src=(y,))))])

def _canonical_half_storage(source:UOp) -> UOp:
  """Commit one FP32 storage expression, then reuse Tinygrad's ordinary algebra on its now-identical half values."""
  converted = _fp32_expr_to_half(source)
  if len(source.toposort()) > 64: return converted
  simplified = graph_rewrite(converted, _pm_half_storage_algebra+sym, name="rockchip half storage algebra")
  return graph_rewrite(simplified, pm_commit_weak, name="rockchip commit storage constants")

def _fp32_add_terms(u:UOp) -> list[UOp]: return [_fp32_expr_to_half(x) for x in _iter_binary(u, Ops.ADD, dtypes.float)]

def _fp32_ratio_to_half(u:UOp) -> UOp|None:
  """Divide two FP32 ADD boundaries while retaining their high/low half expansions through FDIV."""
  if u.op is not Ops.FDIV or u.dtype.scalar() is not dtypes.half or len(u.src) != 2: return None
  sums:list[UOp] = []
  for boundary in u.src:
    source = _typed_cast_source(boundary, dtypes.half, dtypes.float)
    if source is None or source.op is not Ops.ADD: return None
    sums.append(source)
  numerator_high,numerator_low = _precise_sum_parts(_fp32_add_terms(sums[0]))
  denominator_high,denominator_low = _precise_sum_parts(_fp32_add_terms(sums[1]))
  numerator, denominator = numerator_high.alu(Ops.ADD, numerator_low), denominator_high.alu(Ops.ADD, denominator_low)
  quotient, neg_one = numerator.alu(Ops.FDIV, denominator), UOp.const(-1.0, dtypes.half)
  residual = _sub_half(numerator_high, quotient.alu(Ops.MUL, denominator_high), neg_one).alu(Ops.ADD,
    _sub_half(numerator_low, quotient.alu(Ops.MUL, denominator_low), neg_one))
  return _tag_precise_adds(quotient.alu(Ops.ADD, residual.alu(Ops.FDIV, denominator)))

def _accurate_add_recipe(u:UOp) -> UOp:
  terms = [part for x in _iter_binary(u, Ops.ADD, plain=True) for part in
           next((_fp32_add_terms(source) for source in (_typed_cast_source(x, dtypes.half, dtypes.float),)
                 if source is not None and source.op is Ops.ADD), (x,))]
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
    self.materialized_slots:dict[tuple, int] = {}
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
    self.mask_program, nodes = any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in self.root.toposort()), self.root.toposort()
    int_range = _exact_int_range(self.root) if self.root.dtype.scalar() is dtypes.int else None
    packed_bool_load = any(node.op is Ops.LOAD and node.dtype.scalar() is dtypes.bool and _root_param(node.src[0]) is not None for node in nodes)
    embedded_half_int = any(node.op is Ops.CAST and node.dtype.scalar() is dtypes.int and len(node.src) == 1 and
                            node.src[0].dtype.scalar() in (dtypes.half, dtypes.bool) for node in nodes)
    dynamic_int_load = any(node.op is Ops.LOAD and node.dtype.scalar() in (dtypes.int, dtypes.uint) and node.src and
                           _root_param(node.src[0]) is not None for node in nodes)
    self.int_layout = (RKLayout.INT32 if self.root.dtype.scalar() is dtypes.int and dynamic_int_load else
                       RKLayout.INT16 if self.root.dtype.scalar() is dtypes.int and packed_bool_load and int_range is not None and
                       -32768 <= int_range[0] <= int_range[1] <= 32767 else
                       RKLayout.INT_FP16 if self.root.dtype.scalar() is dtypes.int and self.root.op is Ops.CAST and embedded_half_int else
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

  def _scratch(self, dtype:DType, layout:RKLayout, size:int|None=None, u:UOp|None=None) -> RKValue:
    if u is self.root and self.out_param.dtype.scalar() is dtype and \
        (dtype is dtypes.half and layout is RKLayout.FP16 or dtype is dtypes.int16 and layout is RKLayout.INT16 or
         dtype is dtypes.int and layout is RKLayout.INT32): return RKValue(self.out, dtype, self.count, layout)
    self.scratch.append(RKScratch(self.count*4 if size is None and layout is RKLayout.INT32 else
                                  _scratch_bytes(self.count) if size is None else size))
    return RKValue(RKArg(RKBufferKind.SCRATCH, len(self.scratch)-1), dtype, self.count, layout)

  def _slot(self, cache:dict, source:bytes|RKGather|tuple, dtype:DType, layout:RKLayout,
            size:int|None=None, key:tuple|None=None) -> RKValue:
    if isinstance(source, tuple):
      plan = RKGather(0, 0, self.count, values=source, itemsize=4 if layout is RKLayout.INT32 else 2)
      cache_key:bytes|tuple = ("static", layout, source)
    elif isinstance(source, RKGather):
      plan, cache_key = source, ("gather", layout, _gather_cache_key((source,))) if key is None else ("gather", key)
    else:
      plan, cache_key = None, source
    if cache_key not in cache:
      value = self._scratch(dtype, layout, size)
      if plan is not None: self.gathers.append(replace(plan, dst_index=value.arg.index))
      cache[cache_key] = value.arg.index
    return RKValue(RKArg(RKBufferKind.SCRATCH, cache[cache_key]), dtype, self.count, layout)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> RKValue:
    dtype = dtype_hint or u.dtype.scalar()
    if dtype is dtypes.uint or dtype is dtypes.int and self.int_layout is RKLayout.INT32:
      return self._slot(self.materialized_slots, (int(u.arg) & 0xffffffff,) * self.count, dtype, RKLayout.INT32)
    if dtype in (dtypes.half, dtypes.float) or dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16:
      bits, layout = struct.pack("<e", float(u.arg)), RKLayout.FP16 if dtype in (dtypes.half, dtypes.float) else RKLayout.INT_FP16
    elif dtype is dtypes.int16 or dtype is dtypes.int and self.int_layout is RKLayout.INT16:
      bits, layout = struct.pack("<H", _int16_bits(int(u.arg))), RKLayout.INT16
    elif dtype is dtypes.bool: bits, layout = struct.pack("<e", float(bool(u.arg))), RKLayout.BOOL_MASK
    else: raise _RKGenericReject(f"constant {dtype}")
    return self._slot(self.constants, bits, dtype, layout)

  def _operand(self, u:UOp, dtype:DType) -> RKValue:
    return self._constant(u, dtype) if u.op is Ops.CONST and \
      (u.dtype.scalar() in dtypes.weaks or dtype is dtypes.half and u.dtype.scalar() is dtypes.float) else self.lower(u)

  def _static(self, u:UOp, bool_layout:RKLayout=RKLayout.BOOL_MASK) -> RKValue:
    dtype = u.dtype.scalar()
    if not _index_ranges(u):
      scalar = typing_cast(int|float|bool, _eval_expr(u, {}, {}))
      if dtype is dtypes.bool and bool_layout is RKLayout.BOOL_INT16:
        return replace(self._constant(UOp.const(int(bool(scalar)), dtypes.int16)), dtype=dtype, layout=bool_layout)
      return self._constant(UOp.const(scalar, dtype))
    encoders = {dtypes.half: (_fp16_bits, RKLayout.FP16), dtypes.int16: (_int16_bits, RKLayout.INT16),
                dtypes.bool: (int, bool_layout) if bool_layout is RKLayout.BOOL_INT16 else (_fp16_bits, RKLayout.BOOL_MASK)}
    if dtype in encoders:
      encode, layout = encoders[dtype]
      return self._slot(self.materialized_slots, _static_values(self.out_index, u, self.count, encode), dtype, layout)
    elif dtype in (dtypes.int, dtypes.uint):
      values = _static_values(self.out_index, u, self.count, int)
      if dtype is dtypes.uint: return self._slot(self.materialized_slots, tuple(value & 0xffffffff for value in values), dtype, RKLayout.INT32)
      elif self.int_layout is RKLayout.INT_FP16 and all(-2048 <= value <= 2048 for value in values):
        return self._slot(self.materialized_slots, tuple(_fp16_bits(float(value)) for value in values), dtype, self.int_layout)
      elif self.int_layout is RKLayout.INT16 and all(-32768 <= value <= 32767 for value in values):
        return self._slot(self.materialized_slots, tuple(_int16_bits(value) for value in values), dtype, self.int_layout)
      elif self.int_layout is RKLayout.INT32:
        return self._slot(self.materialized_slots, tuple(value & 0xffffffff for value in values), dtype, self.int_layout)
      else: raise _RKGenericReject
    else: raise _RKGenericReject

  def _load(self, u:UOp, fill_override:int|None=None) -> RKValue:
    dtype = u.dtype.scalar()
    if dtype not in (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) or not u.src or u.src[0].op is not Ops.INDEX or \
       (param:=_root_param(u.src[0])) is None or param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST:
      raise _RKGenericReject
    index, gate = u.src[0].src[1], u.src[2] if len(u.src) > 2 else None; default = u.src[1] if len(u.src) > 1 else None
    address_loads = _semantic_loads(index)+(() if gate is None else _semantic_loads(gate))
    layout = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32
    if default is not None and default.op is not Ops.CONST:
      if dtype not in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint) or gate is None or address_loads: raise _RKGenericReject
      schedule, fallback = (len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)), self.lower(default)
      if fallback.layout is not layout or fallback.count != self.count or schedule != (len(self.ew_ops), len(self.mid_gathers), len(self.host_gathers)): raise _RKGenericReject  # noqa: E501
      if (typed_plan:=_typed_load_plan(u,dtype,self.out_index,self.count,fill_bits=0)) is None: raise _RKGenericReject
      value = self._scratch(dtype,layout,self.count*dtype.itemsize)
      self.gathers.extend((RKGather(fallback.arg.index,value.arg.index,self.count,base=fallback.arg.addend//dtype.itemsize,
        axes=((1,self.count,1),),src_kind=fallback.arg.kind,itemsize=dtype.itemsize),
        replace(typed_plan.gather,dst_index=value.arg.index,partial=True,itemsize=dtype.itemsize)))
      return value
    if dtype in (dtypes.float, dtypes.bool) and address_loads: raise _RKGenericReject
    fill_bits = struct.unpack("<I", struct.pack("<f", float(0 if default is None else default.arg)))[0] if dtype is dtypes.float else \
      fill_override if fill_override is not None else _fp16_bits(0 if default is None else default.arg) if dtype is dtypes.half else \
      _int16_bits(0 if default is None else default.arg) if dtype is dtypes.int16 else int(0 if default is None else default.arg) & 0xffffffff
    if address_loads:
      if os.getenv("ROCKCHIP_HOST_GATHER","1") != "1": raise _RKGenericReject
      # Resolve an affine or table-addressed runtime index without reading it on the renderer.
      infos = tuple(info for node in index.toposort() if (info:=_runtime_index(node)) is not None); affine = False; index_offset:int|None = None
      if len(infos) == 1 and (lane_offset:=_runtime_lane_offset(infos[0],self.out_index,self.count)) is not None:
        load,runtime_param,_,index_itemsize = infos[0]
        try: zero_indices,one_indices = (_static_values(self.out_index,index.substitute({load:load.const_like(value)}),self.count,int) for value in (0,1))  # noqa: E501
        except RuntimeError: pass
        else:
          lane_stride = zero_indices[1]-zero_indices[0] if self.count > 1 else 0
          if len({a-b for a,b in zip(one_indices,zero_indices)}) == 1 and zero_indices == tuple(zero_indices[0]+lane*lane_stride for lane in range(self.count)):  # noqa: E501
            runtime_load,index_param,index_offset,base,index_scale,affine = load,runtime_param,lane_offset,zero_indices[0],one_indices[0]-zero_indices[0],True  # noqa: E501
      if not affine:
        runtime_loads = {info[0].key:info for node in address_loads if (info:=_runtime_index(node)) is not None}
        if len(runtime_loads) != 1: raise _RKGenericReject
        runtime_load,index_param,_,index_itemsize = table_info = next(iter(runtime_loads.values())); base = index_scale = lane_stride = 0; index_offset = _runtime_lane_offset(table_info,self.out_index,self.count)  # noqa: E501
      if index_offset is None: raise _RKGenericReject
      index_limit = _bounded_index_gate(gate,runtime_load) if gate is not None else int(param.src[0].arg)
      if index_limit is None or gate is not None and {node.key for node in gate.toposort() if node.op is Ops.LOAD} != {runtime_load.key}: raise _RKGenericReject  # noqa: E501
      source,source_count = RKArg(RKBufferKind.ARG,param.arg.slot),int(param.src[0].arg)
      if not affine:
        if gate is None or index_limit <= 0 or self.count*index_limit > _MAX_STATIC_RANGE_ENVS: raise _RKGenericReject
        try: candidates = tuple(_static_values(self.out_index,index.substitute({runtime_load:runtime_load.const_like(candidate)}),self.count,int)
                                for candidate in range(index_limit))
        except RuntimeError: raise _RKGenericReject from None
        offsets = tuple(candidates[candidate][lane] for lane in range(self.count) for candidate in range(index_limit))
        plan = RKGather(param.arg.slot,0,len(offsets),offsets=offsets,itemsize=dtype.itemsize); _validate_gather_bounds(plan,source_count)
        source,source_count = self._slot(self.materialized_slots,plan,dtype,layout,len(offsets)*dtype.itemsize).arg,len(offsets)
        base,index_scale,lane_stride = 0,1,index_limit
      value = self._scratch(dtype,layout,self.count*dtype.itemsize)
      self.host_gathers.append(RKHostAddress(source,RKArg(RKBufferKind.ARG,index_param.arg.slot,index_offset*index_itemsize),value.arg,
        self.count,source_count,self.count,dtype.itemsize,index_itemsize,fill_bits,index_limit,base,index_scale,lane_stride))
      return value
    if (typed_plan:=_typed_load_plan(u, dtype, self.out_index, self.count, fill_bits=fill_bits,
                                     require_offsets=dtype is dtypes.bool)) is None: raise _RKGenericReject
    if dtype is dtypes.float:
      groups = tuple(range(0,self.count,_EW_ELEMS_32BIT))
      raw = self._slot(self.materialized_slots,replace(typed_plan.gather,itemsize=4),dtypes.float,RKLayout.FP16,len(groups)*16,
        ("fp32_raw",_gather_cache_key((replace(typed_plan.gather,itemsize=4),))))
      aligned,zero = self._scratch(dtypes.half,RKLayout.FP16,len(groups)*16),self._scratch(dtypes.float,RKLayout.FP16,16)
      self.gathers.append(RKGather(0,zero.arg.index,_EW_ELEMS_32BIT,values=(0,)*_EW_ELEMS_32BIT,itemsize=4))
      for group,start in enumerate(groups): self.ew_ops.append(RKEWOp(replace(aligned.arg,addend=group*16),replace(raw.arg,addend=group*16),
        zero.arg,min(_EW_ELEMS_32BIT,self.count-start),_EW_CFG[Ops.ADD]|_EW_STAGE_FP32_IN,stateful=True))
      compact = self._scratch(dtypes.half,RKLayout.FP16,self.count*2)
      self.mid_gathers.append(RKGather(aligned.arg.index,compact.arg.index,self.count,
        offsets=tuple((lane//_EW_ELEMS_32BIT)*8+lane%_EW_ELEMS_32BIT for lane in range(self.count)),
        src_kind=RKBufferKind.SCRATCH,after=len(self.ew_ops)))
      return RKValue(compact.arg,dtypes.float,self.count,RKLayout.FP16)
    if dtype is dtypes.bool:
      return self._slot(self.materialized_slots,replace(typed_plan.gather,fill_bits=int(bool(default.arg)) if default is not None else 0,dst_stride=2,itemsize=1),dtype,RKLayout.BOOL_INT16,self.count*2)  # noqa: E501
    if gate is None and u.src[0].src[1].key == self.out_index.key and int(typed_plan.param.src[0].arg) == self.count:
      return RKValue(RKArg(RKBufferKind.ARG,typed_plan.param.arg.slot),dtype,self.count,layout)
    return self._slot(self.materialized_slots,replace(typed_plan.gather,itemsize=dtype.itemsize),dtype,layout,self.count*dtype.itemsize)

  def _emit(self, dst:RKValue, lhs:RKValue, rhs:RKValue, cfg:int, *, compare:bool=False) -> RKValue:
    integer16, integer32 = dst.layout in (RKLayout.INT16, RKLayout.BOOL_INT16), dst.layout is RKLayout.INT32
    allowed = (RKLayout.INT16, RKLayout.BOOL_INT16) if integer16 else (RKLayout.INT32,) if integer32 else \
      (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16)
    if lhs.layout not in allowed or rhs.layout not in allowed: raise _RKGenericReject
    barrier = not integer16 and not integer32 and cfg in (_EW_CFG_FLOOR, _EW_CFG[Ops.FDIV])
    self.ew_ops.append(RKEWOp(dst.arg, lhs.arg, rhs.arg, self.count, cfg, submit_barrier=barrier,
      compare=compare, stateful=integer32 or not integer16 and (self.mask_program and not compare or barrier),
      int16_output=integer16, int16_input=integer16, int32_output=integer32, int32_input=integer32))
    self.mask_program |= compare; return dst

  def _byte_gather(self, source:RKArg, dest:RKArg, count:int, *, base:int=0, source_stride:int=1,
                   source_limit:int|None=None, dst_stride:int=1, dst_addend:int=0, itemsize:int=2, after:int=-1) -> RKArg:
    self.mid_gathers.append(RKGather(source.index, dest.index, count, base=base,
      axes=((1, count if source_limit is None else source_limit, source_stride),), dst_stride=dst_stride, dst_addend=dst_addend,
      dst_kind=dest.kind, src_kind=source.kind, itemsize=itemsize, after=after)); return dest

  def _raw(self, source:RKValue|Iterable[RKValue|RKArg], layout:RKLayout|None=None, *, u:UOp|None=None,
           mask:int|None=None, dst:RKValue|None=None, cache:bool=True, copy_wide:bool=True) -> Any:
    if isinstance(source, RKValue):
      value = source
      if value.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT_FP16, RKLayout.INT32): raise _RKGenericReject
      if cache and value.arg in self.raw_components: return self.raw_components[value.arg]
      itemsize, source = (4 if value.layout is RKLayout.INT32 else 2), value
      if itemsize == 4 and copy_wide:
        source = self._scratch(dtypes.int, RKLayout.INT32)
        self._emit(source, value, value, _EW_CFG[Ops.MAX])
      parts = tuple(self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(itemsize))
      for byte,part in enumerate(parts): self._byte_gather(source.arg, part.arg, self.count, base=source.arg.addend+byte, source_stride=itemsize,
                                                           dst_stride=2, itemsize=1, after=len(self.ew_ops))
      if cache: self.raw_components[value.arg] = parts
      return parts
    if layout is None or u is None: raise _RKGenericReject
    parts = tuple(part if isinstance(part, RKValue) else RKValue(part, dtypes.int16, self.count, RKLayout.INT16) for part in source)
    itemsize = 4 if layout is RKLayout.INT32 else 2
    if len(parts) != itemsize: raise _RKGenericReject
    result = self._scratch(u.dtype.scalar(), layout, u=u) if dst is None else dst
    for byte,part in enumerate(parts): self._byte_gather(part.arg, result.arg, self.count, base=part.arg.addend, source_stride=2,
                                                         dst_stride=itemsize, dst_addend=byte, itemsize=1, after=len(self.ew_ops))
    if cache: self.raw_components[result.arg] = parts
    if mask is not None: self.int16_masks[result.arg] = mask
    return result

  def _alu(self, u:UOp) -> RKValue:
    if u.op in (Ops.RECIPROCAL, Ops.NEG):
      src = self.lower(u.src[0])
      if u.op is Ops.RECIPROCAL:
        one = self.lower(UOp.const(1.0, dtypes.half))
        return self._emit(self._scratch(dtypes.half, RKLayout.FP16, u=u), one, src, _EW_CFG[Ops.FDIV])
      return self._emit(self._scratch(u.dtype.scalar(), src.layout, u=u), src, src, _EW_CFG_NEG)
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.ADD and (recipe:=_fold_relu_cap(u)) is not None:
      return self.lower(recipe)
    if u.op is Ops.FDIV and (recipe:=_preserve_infinite_division_sign(u)) is not None:
      return self.lower(recipe)
    dtype = u.dtype.scalar()
    int_range = _exact_int_range(u) if dtype is dtypes.int else None
    bounded = self.int_layout is RKLayout.INT32 or self.int_layout is RKLayout.INT_FP16 and int_range is not None and \
      -2048 <= int_range[0] <= int_range[1] <= 2048 or self.int_layout is RKLayout.INT16 and int_range is not None and \
      -32768 <= int_range[0] <= int_range[1] <= 32767
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
      zero, negative = self._constant(UOp.const(0.0, dtypes.half)), self._scratch(dtypes.half, RKLayout.FP16)
      negative_mask, positive_mask = (self._scratch(dtypes.bool, RKLayout.BOOL_MASK) for _ in range(2))
      self._emit(negative, zero, lhs, _EW_CFG[Ops.SUB])
      self._emit(negative_mask, negative, negative, _EW_CFG[Ops.MAX], compare=True)
      self._emit(positive_mask, lhs, lhs, _EW_CFG[Ops.MAX], compare=True)
      return self._emit(self._scratch(dtypes.half, RKLayout.FP16, u=u), positive_mask, negative_mask, _EW_CFG[Ops.SUB])
    if u.op is Ops.MAX and u.arg == _NATIVE_MIN:
      if expected is not RKLayout.FP16: return self._emit(self._scratch(dtype, RKLayout.INT16, u=u), lhs, rhs, _EW_CFG_MIN)
      zero = self.lower(UOp.const(0.0, dtypes.half))
      neg_lhs, neg_rhs = (self._scratch(dtypes.half, RKLayout.FP16) for _ in range(2))
      self._emit(neg_lhs, zero, lhs, _EW_CFG[Ops.SUB])
      self._emit(neg_rhs, zero, rhs, _EW_CFG[Ops.SUB])
      self._emit(neg_lhs, neg_lhs, neg_rhs, _EW_CFG[Ops.MAX])
      return self._emit(self._scratch(dtypes.half, RKLayout.FP16, u=u) if u is self.root else neg_lhs, zero, neg_lhs, _EW_CFG[Ops.SUB])
    cfg = _EW_CFG_ABS if u.op is Ops.MAX and u.arg == _NATIVE_ABS else _EW_CFG_FLOOR if u.op is Ops.MAX and u.arg == _NATIVE_FLOOR else \
      _EW_CFG_CEIL if u.op is Ops.MAX and u.arg == _NATIVE_CEIL else _EW_CFG_RELU6 if u.op is Ops.MAX and u.arg == _NATIVE_RELU6 else _EW_CFG[u.op]
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    layout, out_dtype = (RKLayout.BOOL_MASK, dtypes.bool) if compare else (expected, dtype)
    return self._emit(self._scratch(out_dtype, layout, u=u), lhs, rhs, cfg, compare=compare)

  def _coerce_bool(self, value:RKValue, layout:RKLayout) -> RKValue:
    if value.layout is layout: return value
    if (value.layout, layout) != (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
    result = self._scratch(dtypes.bool, RKLayout.BOOL_INT16)
    self.ew_ops.append(RKEWOp(result.arg, value.arg, value.arg, self.count, _EW_CFG[Ops.MAX], submit_barrier=True, stateful=True, int16_output=True)); return result  # noqa: E501

  def _bool_binary(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.CMPNE:
      for expression,marker in (u.src, u.src[::-1]):
        if marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and expression.op is Ops.CMPLT:
          sources = tuple(_half_backed_value(src) for src in expression.src)
          if any(src is None for src in sources): continue
          less = self.lower(expression)
          if less.layout is not RKLayout.BOOL_INT16: raise _RKGenericReject
          operands = tuple(self._operand(src, dtypes.half) for src in typing_cast(tuple[UOp, UOp], sources))
          nan = tuple(self._fp16_component_values(value)[2] for value in operands)
          one_arg = self._constant(UOp.const(1, dtypes.int16)).arg
          inverse = self._i16(one_arg, less.arg, _EW_CFG[Ops.SUB])
          numeric = self._i16(one_arg, self._i16(nan[0], nan[1], _EW_CFG[Ops.MAX]), _EW_CFG[Ops.SUB])
          result = self._i16(inverse, numeric, _EW_CFG[Ops.MUL], self._scratch(dtypes.bool, RKLayout.BOOL_INT16, u=u).arg)
          return RKValue(result, dtypes.bool, self.count, RKLayout.BOOL_INT16)
    values = [self.lower(src) if not (src.op is Ops.CONST and src.dtype.scalar() is dtypes.bool) else None for src in u.src]
    preferred = RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in values) else RKLayout.BOOL_MASK
    lhs, rhs = (self._static(src, preferred) if value is None else self._coerce_bool(value, preferred) for src,value in zip(u.src, values))
    dst = self._scratch(dtypes.bool, preferred, u=u)
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
      one, unequal = RKValue(raw_one.arg, dtypes.bool, self.count, preferred), self._scratch(dtypes.bool, preferred)
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
        low, high = self._raw(value)
        zero, _, _, const128 = (self._constant(UOp.const(number, dtypes.int16)) for number in (0, 1, 127, 128))
        sign = self._i16_clamp_one(self._i16(high.arg, self._constant(UOp.const(127, dtypes.int16)).arg, _EW_CFG[Ops.SUB]))
        sign_scale = self._i16(sign, const128.arg, _EW_CFG[Ops.MUL])
        if mask == 0x7fff:
          return self._raw((low, self._i16(high.arg, sign_scale, _EW_CFG[Ops.SUB])), RKLayout.INT16, u=u, mask=mask)
        return self._raw((zero, sign_scale), RKLayout.INT16, u=u, mask=mask)
    if u.dtype.scalar() is dtypes.int16 and u.op is Ops.OR:
      values = tuple(self.lower(source) for source in u.src)
      masks = tuple(self.int16_masks.get(value.arg) for value in values)
      if all(mask is not None for mask in masks) and typing_cast(int, masks[0]) & typing_cast(int, masks[1]) == 0:
        summed_parts = tuple(self._i16(lhs.arg, rhs.arg, _EW_CFG[Ops.ADD]) for lhs,rhs in zip(*(tuple(self._raw(value) for value in values))))
        return self._raw(summed_parts, RKLayout.INT16, u=u, mask=typing_cast(int, masks[0]) | typing_cast(int, masks[1]))
    if u.op is Ops.XOR:
      for marker, source in (u.src, u.src[::-1]):
        if marker.op is not Ops.CONST or int(marker.arg) != -1: continue
        dtype = u.dtype.scalar()
        layout = RKLayout.INT16 if dtype is dtypes.int16 else self.int_layout if dtype is dtypes.int else None
        if layout is None: raise _RKGenericReject
        if layout is RKLayout.INT32 and u is self.root and 1 <= self.count*4 <= _MAX_EW_ELEMS_FP16 and \
           (parsed:=_typed_load_plan(source, dtypes.int, self.out_index, self.count, require_offsets=True)) is not None:
          param, offsets = parsed.param, parsed.gather.offsets
          lanes, stride, slot = self.count*4, round_up(self.count*4*2, 64), len(self.scratch)
          self.scratch.append(RKScratch(stride*3))
          raw_arg, constant_arg, inverted_arg = (RKArg(RKBufferKind.SCRATCH, slot, row*stride) for row in range(3))
          self.gathers.extend((RKGather(param.arg.slot, raw_arg.index, lanes,
            offsets=tuple(offset*4+byte if offset >= 0 else -1 for offset in offsets for byte in range(4)), dst_stride=2, itemsize=1),
            RKGather(param.arg.slot, constant_arg.index, lanes, values=(255,)*lanes, dst_addend=constant_arg.addend//2)))
          self.ew_ops.append(RKEWOp(inverted_arg, constant_arg, raw_arg, lanes, _EW_CFG[Ops.SUB], int16_input=True, int16_output=True))
          self.post_gathers.append(_raw_gather(inverted_arg, self.out_param.arg.slot, lanes))
          return RKValue(self.out, dtype, self.count, layout)
        rhs = self.lower(source)
        if rhs.layout is not layout: raise _RKGenericReject
        if layout is RKLayout.INT32:
          components, const255 = self._raw(rhs), self._constant(UOp.const(255, dtypes.int16))
          return self._raw(tuple(self._i16(const255.arg, component.arg, _EW_CFG[Ops.SUB]) for component in components), RKLayout.INT32, u=u)
        lhs = self._constant(UOp.const(-1, dtype))
        return self._emit(self._scratch(dtype, layout, u=u), lhs, rhs, _EW_CFG[Ops.SUB])
    dtype = u.dtype.scalar()
    layout = RKLayout.INT16 if dtype is dtypes.int16 else RKLayout.INT32 if dtype is dtypes.int and self.int_layout is RKLayout.INT32 else None
    if layout is None or u.op not in (Ops.AND, Ops.OR, Ops.XOR): raise _RKGenericReject
    values = tuple(self.lower(source) for source in u.src)
    if any(value.layout is not layout for value in values): raise _RKGenericReject
    lanes = self.count*(4 if layout is RKLayout.INT32 else 2)
    if not 1 <= lanes <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16, _scratch_bytes(lanes)).arg
    raw = tuple(self._byte_gather(v.arg, allocate(), lanes, base=v.arg.addend, dst_stride=2, itemsize=1, after=len(self.ew_ops)) for v in values)
    constants = {number:allocate() for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)}; self.gathers.extend(
      RKGather(self.out_param.arg.slot, constant.index, lanes, values=(number,)*lanes) for number,constant in constants.items())
    lhs_bits, rhs_bits = (_int16_byte_bits(self.ew_ops, allocate, constants, value, lanes, weighted=True) for value in raw)
    weighted = [allocate() for _ in lhs_bits]
    for combined,left,right in zip(weighted, lhs_bits, rhs_bits):
      self.ew_ops.extend(_ew_ops(((combined, left, right, Ops.SUB), (combined, combined, combined, _EW_CFG_ABS)) if u.op is Ops.XOR else
        ((combined, left, right, _EW_CFG_MIN if u.op is Ops.AND else _EW_CFG[Ops.MAX]),), lanes, **_INT16_EW))
    combined, result = _reduce_rows(self.ew_ops, weighted, lanes, _EW_CFG[Ops.ADD], int16=True), self._scratch(dtype, layout, u=u)
    self._byte_gather(combined, result.arg, lanes, base=combined.addend, source_stride=2, itemsize=1, after=len(self.ew_ops))
    return result

  def _int32_shift(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or u.dtype.scalar() not in (dtypes.int, dtypes.uint) or \
       u.src[1].dtype.scalar() not in (dtypes.int, dtypes.uint) or self.int_layout is not RKLayout.INT32:
      raise _RKGenericReject
    if (value:=self.lower(u.src[0])).layout is not RKLayout.INT32: raise _RKGenericReject
    vector_bytes, vector_lanes, matrix_lanes = _stripe_layout(self.count, 32)
    pre_lanes = vector_lanes*5
    if self.count < 1 or matrix_lanes > _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    pre_arena = self._scratch(dtypes.int16, RKLayout.INT16, 51*pre_lanes*2).arg
    pre_allocate = iter(replace(pre_arena, addend=row*pre_lanes*2) for row in range(51)).__next__
    raw, value_parts, shift = pre_allocate(), self._raw(value), None if u.src[1].op is Ops.CONST else self.lower(u.src[1])
    if shift is not None and shift.layout is not RKLayout.INT32: raise _RKGenericReject
    shift_part = None if shift is None else self._raw(shift)[0]
    for byte,part in enumerate(value_parts): self._byte_gather(part.arg, raw, self.count, base=part.arg.addend//2,
                                                               dst_addend=byte*vector_lanes, after=len(self.ew_ops))
    if shift_part is None:
      self.mid_gathers.append(RKGather(self.out_param.arg.slot, raw.index, self.count,
        values=(int(u.src[1].arg)&0xff,)*self.count, dst_addend=4*vector_lanes, after=len(self.ew_ops)))
    else: self._byte_gather(shift_part.arg, raw, self.count, base=shift_part.arg.addend//2,
                            dst_addend=4*vector_lanes, after=len(self.ew_ops))
    constants = {number:pre_allocate() for number in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)}
    self.gathers.extend(RKGather(self.out_param.arg.slot, dst.index, pre_lanes, values=(number,)*pre_lanes, dst_addend=dst.addend//2)
                        for number,dst in constants.items())
    planes = _int16_byte_bits(self.ew_ops, pre_allocate, constants, raw, pre_lanes)

    post_arena = self._scratch(dtypes.int16, RKLayout.INT16, 20*matrix_lanes*2).arg
    post_allocate = iter(replace(post_arena, addend=row*matrix_lanes*2) for row in range(20)).__next__
    bits, masks, sign, zero, weights = post_allocate(), tuple(post_allocate() for _ in range(5)), post_allocate(), post_allocate(), post_allocate()
    def gather_plane(plane:RKArg, destination:RKArg, count:int, byte:int, source_limit:int|None=None,
                     dst_addend:int|None=None) -> None:
      self._byte_gather(plane, destination, count, base=plane.addend//2+byte*vector_lanes, source_limit=source_limit,
                        dst_addend=destination.addend//2 if dst_addend is None else dst_addend, after=len(self.ew_ops))
    for absolute_bit in range(32):
      gather_plane(planes[absolute_bit&7], bits, self.count, absolute_bit>>3, dst_addend=absolute_bit*vector_lanes)
    for bit,mask in enumerate(masks): gather_plane(planes[bit], mask, matrix_lanes, 4, vector_lanes)
    gather_plane(planes[7], sign, matrix_lanes, 3, vector_lanes)
    self.mid_gathers.append(RKGather(self.out_param.arg.slot, weights.index, matrix_lanes,
      values=tuple(1 << (row&7) if lane < self.count else 0 for row in range(32) for lane in range(vector_lanes)),
      dst_addend=weights.addend//2, after=len(self.ew_ops)))
    current = bits
    for bit,amount in enumerate((1, 2, 4, 8, 16)):
      temp, result = post_allocate(), post_allocate()
      normal_rows,normal_dst,shifted_src,boundary_rows,boundary_dst = (32-amount, amount, 0, amount, 0) if u.op is Ops.SHL else \
        (32-amount, 0, amount, amount, 32-amount)
      for rows,dst_row,src,fill in ((normal_rows, normal_dst, current, shifted_src),
                                    (boundary_rows, boundary_dst, sign if u.op is Ops.SHR and u.dtype.scalar() is dtypes.int else zero, 0)):
        dst,old,selected = (replace(arg, addend=arg.addend+dst_row*vector_bytes) for arg in (temp, current, result))
        source = replace(src, addend=src.addend+(fill*vector_bytes if src is current else dst_row*vector_bytes))
        mask = replace(masks[bit], addend=masks[bit].addend+dst_row*vector_bytes)
        self.ew_ops.extend(_ew_ops(((dst, source, old, Ops.SUB), (dst, dst, mask, Ops.MUL),
          (selected, old, dst, Ops.ADD)), rows*vector_lanes, **_INT16_EW))
      current = result
    weighted = post_allocate()
    self.ew_ops.append(RKEWOp(weighted, current, weights, matrix_lanes, _EW_CFG[Ops.MUL], **_INT16_EW))
    return self._raw(tuple(_reduce_rows(self.ew_ops, [replace(weighted, addend=weighted.addend+(byte*8+bit)*vector_bytes) for bit in range(8)],
      vector_lanes, _EW_CFG[Ops.ADD], int16=True) for byte in range(4)), RKLayout.INT32, u=u)

  def _compare(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src): return self._bool_binary(u)
    if u.op is Ops.CMPNE and any(src.op is Ops.INDEX for src in u.src) and (nonzero:=_fp16_nonzero_mask(u)) is not None:
      value = self.lower(nonzero)
      if value.layout is not RKLayout.BOOL_MASK: raise _RKGenericReject
      return RKValue(value.arg, dtypes.bool, self.count, value.layout)
    if u.op in (Ops.CMPNE, Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_equality(u)
    if u.op is Ops.CMPLT and all(src.dtype.scalar() is dtypes.half for src in u.src): return self._fp16_less(u)
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      sources = tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src)
      bounds = tuple(_exact_int_range(src) for src in sources)
      if self.int_layout is RKLayout.INT_FP16 or self.int_layout is not RKLayout.INT32 and all(bound is not None and
        -2048 <= bound[0] <= bound[1] <= 2048 for bound in bounds):
        value = self.lower(UOp(u.op, dtypes.bool, src=tuple(_int_fp16_expr(src) for src in sources), arg=u.arg))
        if value.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
        return value
      return self._int32_compare(u.replace(src=sources))
    if all(src.dtype.scalar() is dtypes.int16 for src in u.src):
      if u.op not in (Ops.CMPLT, Ops.CMPNE): raise _RKGenericReject
      lhs, rhs = u.src
      delta = rhs.alu(Ops.SUB, lhs) if u.op is Ops.CMPLT else lhs.alu(Ops.SUB, rhs)
      magnitude = delta.alu(Ops.MAX, UOp.const(0, dtypes.int16)) if u.op is Ops.CMPLT else \
        UOp(Ops.MAX, dtypes.int16, src=(delta, delta), arg=_NATIVE_ABS)
      value = self.lower(UOp(Ops.MAX, dtypes.int16, src=(magnitude, UOp.const(1, dtypes.int16)), arg=_NATIVE_MIN))
      if value.layout is not RKLayout.INT16: raise _RKGenericReject
      return RKValue(value.arg, dtypes.bool, self.count, RKLayout.BOOL_INT16)
    sources = tuple(_half_backed_value(src) for src in u.src)
    if u.op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) or any(src is None for src in sources): raise _RKGenericReject
    return (self._fp16_less if u.op is Ops.CMPLT else self._fp16_equality)(u.replace(src=typing_cast(tuple[UOp, UOp], sources)))

  def _i16(self, lhs:RKArg, rhs:RKArg, cfg:int, dst:RKArg|None=None) -> RKArg:
    result = RKValue(dst or self._scratch(dtypes.int16, RKLayout.INT16).arg, dtypes.int16, self.count, RKLayout.INT16)
    return self._emit(result, RKValue(lhs, dtypes.int16, self.count, RKLayout.INT16), RKValue(rhs, dtypes.int16, self.count, RKLayout.INT16), cfg).arg

  def _i16_equal(self, lhs:RKArg, rhs:RKArg, diff:RKArg|None=None) -> RKArg:
    magnitude = self._i16(diff:=self._i16(lhs, rhs, _EW_CFG[Ops.SUB]) if diff is None else diff, diff, _EW_CFG_ABS)
    return self._i16(self._i16_const(1), self._i16(magnitude, self._i16_const(1), _EW_CFG_MIN), _EW_CFG[Ops.SUB])

  def _i16_const(self, value:int) -> RKArg: return self._constant(UOp.const(value, dtypes.int16)).arg

  def _i16_clamp_one(self, value:RKArg) -> RKArg:
    return self._i16(self._i16(value, self._i16_const(0), _EW_CFG[Ops.MAX]), self._i16_const(1), _EW_CFG_MIN)

  def _i16_twos_complement(self, raw:tuple[RKArg, ...], sign:RKArg) -> tuple[RKArg, ...]:
    carry, result = sign, []
    for byte in raw:
      inverted = self._i16(
        self._i16(self._i16_const(255), self._i16(byte, byte, _EW_CFG[Ops.ADD]), _EW_CFG[Ops.SUB]),
        sign, _EW_CFG[Ops.MUL])
      total = self._i16(self._i16(byte, inverted, _EW_CFG[Ops.ADD]), carry, _EW_CFG[Ops.ADD])
      carry = self._i16_clamp_one(self._i16(total, self._i16_const(255), _EW_CFG[Ops.SUB]))
      result.append(self._i16(total, self._i16(carry, self._i16_const(256), _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB]))
    return tuple(result)

  def _int32_divmod(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or self.int_layout is not RKLayout.INT32 or not 1 <= self.count <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    key = u.src
    if key not in self.int32_divmod:
      values = tuple(self._operand(src, dtypes.int) for src in key)
      if any(value.layout is not RKLayout.INT32 for value in values): raise _RKGenericReject
      raw = tuple(tuple(part.arg for part in self._raw(value)) for value in values)
      constants = {value:self._i16_const(value) for value in
                   (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256)}
      signs = tuple(self._i16_clamp_one(self._i16(value[3], self._i16_const(127), _EW_CFG[Ops.SUB])) for value in raw)
      numerator, denominator = (self._i16_twos_complement(value, sign) for value,sign in zip(raw, signs))
      denominator_nonzero = _reduce_rows(self.ew_ops, [self._i16_clamp_one(value) for value in denominator], self.count, _EW_CFG[Ops.MAX], int16=True)
      numerator_bits = tuple(itertools.chain.from_iterable(_int16_byte_bits(self.ew_ops, lambda:self._scratch(dtypes.int16, RKLayout.INT16).arg,
                                                                            constants, byte, self.count) for byte in numerator))
      remainder, quotient = [constants[0]]*4, [constants[0]]*4
      for bit_index in range(31, -1, -1):
        shifted:list[RKArg] = []
        incoming = numerator_bits[bit_index]
        for byte in remainder:
          carry = self._i16_clamp_one(self._i16(byte, self._constant(UOp.const(127, dtypes.int16)).arg, _EW_CFG[Ops.SUB]))
          wrapped = self._i16(self._i16(byte, byte, _EW_CFG[Ops.ADD]), self._i16(carry, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
          shifted.append(self._i16(wrapped, incoming, _EW_CFG[Ops.ADD])); incoming = carry
        remainder, greater, equal = shifted, constants[0], constants[1]
        for left,right in zip(reversed(remainder), reversed(denominator)):
          diff = self._i16(left, right, _EW_CFG[Ops.SUB])
          positive = self._i16(self._i16(diff, constants[0], _EW_CFG[Ops.MAX]), constants[1], _EW_CFG_MIN)
          greater = self._i16(greater, self._i16(equal, positive, _EW_CFG_MIN), _EW_CFG[Ops.MAX])
          equal = self._i16(equal, self._i16_equal(left, right, diff), _EW_CFG_MIN)
        ge = self._i16(self._i16(greater, equal, _EW_CFG[Ops.MAX]), denominator_nonzero, _EW_CFG_MIN)
        borrow, reduced = constants[0], []
        for left,right in zip(remainder, denominator):
          delta = self._i16(self._i16(left, self._i16(right, ge, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB]), borrow, _EW_CFG[Ops.SUB])
          borrow = self._i16_clamp_one(self._i16(constants[0], delta, _EW_CFG[Ops.SUB]))
          reduced.append(self._i16(delta, self._i16(borrow, constants[256], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD]))
        remainder, byte_index, weight = reduced, bit_index >> 3, 1 << (bit_index&7)
        quotient[byte_index] = self._i16(quotient[byte_index], self._i16(ge, constants[weight], _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
      sign_delta = self._i16(signs[0], signs[1], _EW_CFG[Ops.SUB])
      self.int32_divmod[key] = tuple(quotient), tuple(remainder), signs[0], self._i16(sign_delta, sign_delta, _EW_CFG_ABS)
    quotient_raw, remainder_raw, remainder_sign, quotient_sign = self.int32_divmod[key]
    packed_raw, sign = (quotient_raw, quotient_sign) if u.op is Ops.CDIV else (remainder_raw, remainder_sign)
    return self._raw(self._i16_twos_complement(packed_raw, sign), RKLayout.INT32, u=u)

  def _int32_compare(self, u:UOp) -> RKValue:
    def operand(src:UOp) -> RKValue:
      value = self._slot(self.materialized_slots, tuple(x & 0xffffffff for x in _static_values(self.out_index, src, self.count, int)),
                         dtypes.int, RKLayout.INT32) if src in self.static_nodes else self.lower(src)
      if value.layout is not RKLayout.INT32: raise _RKGenericReject
      return value
    lhs, rhs = (operand(src) for src in u.src)
    lhs_bytes, rhs_bytes = self._raw(lhs), self._raw(rhs)
    def allocate() -> RKArg: return self._scratch(dtypes.int16, RKLayout.INT16).arg
    constants = {value:self._constant(UOp.const(value, dtypes.int16)).arg for value in (0, 1, 127, 128, 256)}
    if u.op is Ops.CMPLT:
      lhs_components, rhs_components = [value.arg for value in lhs_bytes[::-1]], [value.arg for value in rhs_bytes[::-1]]
      def biased_sign(value:RKArg) -> RKArg:
        delta, positive, high, scaled, biased = (allocate() for _ in range(5))
        self.ew_ops.extend(_ew_ops(((delta, value, constants[127], Ops.SUB), (positive, delta, constants[0], Ops.MAX),
          (high, positive, constants[1], _EW_CFG_MIN), (scaled, high, constants[256], Ops.MUL),
          (biased, value, constants[128], Ops.ADD), (biased, biased, scaled, Ops.SUB)), self.count, **_INT16_EW))
        return biased
      lhs_components[0], rhs_components[0] = biased_sign(lhs_components[0]), biased_sign(rhs_components[0])
      mask = _ordered_byte_less(self.ew_ops, allocate, constants[0], constants[1], lhs_components, rhs_components, self.count)
    else:
      equal = constants[1]
      for left,right in zip(lhs_bytes, rhs_bytes):
        equal = self._i16(equal, self._i16_equal(left.arg, right.arg), _EW_CFG[Ops.MUL])
      if u.op is Ops.CMPEQ: mask = equal
      elif u.op is Ops.CMPNE: mask = self._i16(constants[1], equal, _EW_CFG[Ops.SUB])
      else: raise _RKGenericReject
    return RKValue(mask, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_equality(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 equality through raw bytes and native INT16 arithmetic, without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    _, one = (self._constant(UOp.const(number, dtypes.int16)).arg for number in (0, 1))
    lhs_low,lhs_high,lhs_nan,rhs_low,rhs_high,rhs_nan = (*self._fp16_component_values(values[0]), *self._fp16_component_values(values[1]))
    low_equal, high_equal = self._i16_equal(lhs_low.arg, rhs_low.arg), self._i16_equal(lhs_high, rhs_high)
    numeric = self._i16(one, self._i16(lhs_nan, rhs_nan, _EW_CFG[Ops.MAX]), _EW_CFG[Ops.SUB])
    equal = self._i16(self._i16(low_equal, high_equal, _EW_CFG[Ops.MUL]), numeric, _EW_CFG[Ops.MUL])
    if u.op is Ops.CMPNE: equal = self._i16(one, equal, _EW_CFG[Ops.SUB])
    return RKValue(equal, dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _fp16_component_values(self, value:RKValue) -> tuple[RKValue, RKArg, RKArg]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    if value.arg in self.fp16_components: return self.fp16_components[value.arg]
    (low, high), (zero, one, const123, const124, const127, const128) = self._raw(value), tuple(map(self._i16_const,(0,1,123,124,127,128)))
    sign_scale = self._i16(self._i16_clamp_one(self._i16(high.arg, const127, _EW_CFG[Ops.SUB])), const128, _EW_CFG[Ops.MUL])
    magnitude = self._i16(high.arg, sign_scale, _EW_CFG[Ops.SUB])
    high_zero, low_zero = self._i16_equal(magnitude, zero), self._i16_equal(low.arg, zero)
    clean_high = self._i16(high.arg, self._i16(sign_scale, self._i16(high_zero, low_zero, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.MUL]), _EW_CFG[Ops.SUB])
    exponent = self._i16_clamp_one(self._i16(magnitude, const123, _EW_CFG[Ops.SUB]))
    mantissa = self._i16(self._i16_clamp_one(self._i16(magnitude, const124, _EW_CFG[Ops.SUB])),
      self._i16(low.arg, one, _EW_CFG_MIN), _EW_CFG[Ops.MAX])
    return self.fp16_components.setdefault(value.arg, (low, clean_high, self._i16(exponent, mantissa, _EW_CFG[Ops.MUL])))

  def _fp16_ordered_values(self, value:RKValue) -> tuple[RKArg, RKArg]:
    """Map a classified FP16 lane to two unsigned bytes whose lexical order is IEEE numeric order."""
    if value.arg in self.fp16_ordered: return self.fp16_ordered[value.arg]
    (low,clean_high,_),(_,_,const127,const128,const255) = self._fp16_component_values(value), (*map(self._i16_const,(0,1,127,128,255)),)
    sign, positive_high = self._i16_clamp_one(self._i16(clean_high, const127, _EW_CFG[Ops.SUB])), self._i16(clean_high, const128, _EW_CFG[Ops.ADD])
    high_delta = self._i16(self._i16(const255, clean_high, _EW_CFG[Ops.SUB]), positive_high, _EW_CFG[Ops.SUB])
    ordered_high = self._i16(positive_high, self._i16(sign, high_delta, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])
    low_delta = self._i16(self._i16(const255, low.arg, _EW_CFG[Ops.SUB]), low.arg, _EW_CFG[Ops.SUB])
    return self.fp16_ordered.setdefault(value.arg, (ordered_high, self._i16(low.arg, self._i16(sign, low_delta, _EW_CFG[Ops.MUL]), _EW_CFG[Ops.ADD])))

  def _fp16_less(self, u:UOp) -> RKValue:
    """Evaluate IEEE FP16 less-than as an ordered raw-byte comparison without reset-heavy compare stages."""
    values = tuple(self._operand(src, dtypes.half) for src in u.src)
    if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
    zero, one = (self._constant(UOp.const(number, dtypes.int16)).arg for number in (0, 1))
    ordered, nan = tuple(self._fp16_ordered_values(value) for value in values), tuple(self._fp16_component_values(value)[2] for value in values)
    less = _ordered_byte_less(self.ew_ops, lambda: self._scratch(dtypes.int16, RKLayout.INT16).arg, zero, one, ordered[0], ordered[1], self.count)
    numeric = self._i16(one, self._i16(nan[0], nan[1], _EW_CFG[Ops.MAX]), _EW_CFG[Ops.SUB])
    return RKValue(self._i16(less, numeric, _EW_CFG[Ops.MUL]), dtypes.bool, self.count, RKLayout.BOOL_INT16)

  def _masked_where(self, u:UOp, dtype:DType, layout:RKLayout, selector:RKValue,
                    yes:RKValue, no:RKValue, one:RKValue) -> RKValue:
    selected_yes, inverse, selected_no = (self._scratch(one.dtype, one.layout) for _ in range(3))
    for dst,lhs,rhs,cfg in ((selected_yes, selector, yes, _EW_CFG[Ops.MUL]), (inverse, one, selector, _EW_CFG[Ops.SUB]),
                            (selected_no, inverse, no, _EW_CFG[Ops.MUL])): self._emit(dst, lhs, rhs, cfg)
    return self._emit(self._scratch(dtype, layout, u=u), selected_yes, selected_no, _EW_CFG[Ops.ADD])

  def _raw_where(self, u:UOp) -> RKValue:
    """Select typed values exactly, keeping nonfinite arms lazy until the selector layout is known."""
    specials = [i for i,arm in enumerate(u.src[1:]) if arm.op is Ops.CONST and arm.dtype is dtypes.half and not math.isfinite(float(arm.arg))]
    selector:RKValue|None = None
    if len(specials) == 1:
      special, finite = specials[0], self.lower(u.src[2-specials[0]])
      if finite.layout is not RKLayout.FP16: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
      if selector.layout is RKLayout.BOOL_MASK:
        one, denominator = self._constant(UOp.const(1.0, dtypes.half)), selector
        if special == 0:
          denominator = self._scratch(dtypes.half, RKLayout.FP16)
          self._emit(denominator, one, selector, _EW_CFG[Ops.SUB])
        if math.isnan(float(u.src[1+special].arg)):
          zero, correction = self._constant(UOp.const(0.0, dtypes.half)), self._scratch(dtypes.half, RKLayout.FP16)
          self._emit(correction, zero, denominator, _EW_CFG[Ops.FDIV])
        else:
          sign = self._constant(UOp.const(math.copysign(1.0, float(u.src[1+special].arg)), dtypes.half))
          quotient, correction = self._scratch(dtypes.half, RKLayout.FP16), self._scratch(dtypes.half, RKLayout.FP16)
          self._emit(quotient, sign, denominator, _EW_CFG[Ops.FDIV])
          self._emit(correction, quotient, sign, _EW_CFG[Ops.SUB])
        return self._emit(self._scratch(dtypes.half, RKLayout.FP16, u=u), finite, correction, _EW_CFG[Ops.ADD])
    yes, no = (self.lower(src) for src in u.src[1:])
    if yes.layout is not no.layout or yes.layout not in (RKLayout.FP16, RKLayout.INT16, RKLayout.INT_FP16, RKLayout.INT32):
      raise _RKGenericReject
    mask_layout = RKLayout.BOOL_INT16 if yes.layout in (RKLayout.INT16, RKLayout.INT32) else RKLayout.BOOL_MASK
    if selector is None: selector = self._static(u.src[0], mask_layout) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
    allowed_masks = (mask_layout,) if yes.layout is RKLayout.INT16 else (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16)
    if selector.layout not in allowed_masks: raise _RKGenericReject
    if yes.layout is RKLayout.INT16:
      return self._masked_where(u, dtypes.int16, yes.layout, selector, yes, no, self._constant(UOp.const(1, dtypes.int16)))
    mask, (yes_bytes,no_bytes) = self._coerce_bool(selector, RKLayout.BOOL_INT16), (self._raw(x, cache=False, copy_wide=False) for x in (yes,no))
    selected_bytes:list[RKValue] = []
    for yes_byte,no_byte in zip(yes_bytes, no_bytes):
      delta, selected, result = (self._scratch(dtypes.int16, RKLayout.INT16) for _ in range(3))
      for dst,lhs,rhs,cfg in ((delta, yes_byte, no_byte, _EW_CFG[Ops.SUB]), (selected, mask, delta, _EW_CFG[Ops.MUL])): self._emit(dst, lhs, rhs, cfg)
      selected_bytes.append(self._emit(result, no_byte, selected, _EW_CFG[Ops.ADD]))
    return self._raw(selected_bytes, yes.layout, u=u, dst=self._scratch(u.dtype.scalar(), yes.layout), cache=False)

  def _where(self, u:UOp) -> RKValue:
    if len(u.src) != 3: raise _RKGenericReject
    if u.dtype.scalar() is dtypes.bool:
      dynamic = [None if src in self.static_nodes else self.lower(src) for src in u.src]
      preferred = RKLayout.BOOL_INT16 if any(value is not None and value.layout is RKLayout.BOOL_INT16 for value in dynamic) else RKLayout.BOOL_MASK
      selector, yes, no = [self._static(src, preferred) if value is None else self._coerce_bool(value, preferred) for src,value in zip(u.src,dynamic)]
      return self._masked_where(u, dtypes.bool, preferred, selector, yes, no,
                               self._constant(UOp.const(1, dtypes.int16 if preferred is RKLayout.BOOL_INT16 else dtypes.half)))
    if u is self.root and u.dtype.scalar() is dtypes.int and all(
      arm.op is Ops.CONST and arm.dtype.scalar() is dtypes.int for arm in u.src[1:]
    ):
      yes_int, no_int = (int(arm.arg) for arm in u.src[1:])
      try: exact = all(_static_cast(value, dtypes.half) == value for value in (no_int, yes_int-no_int))
      except (OverflowError, struct.error): exact = False
      if not exact: raise _RKGenericReject
      selector = self._static(u.src[0], RKLayout.BOOL_MASK) if _is_static_expr(u.src[0]) else self.lower(u.src[0])
      if selector.layout not in (RKLayout.BOOL_MASK, RKLayout.BOOL_INT16): raise _RKGenericReject
      data_dtype, data_layout = ((dtypes.int16, RKLayout.INT16) if selector.layout is RKLayout.BOOL_INT16 else (dtypes.half, RKLayout.FP16))
      delta, baseline = (self._constant(UOp.const(value, data_dtype)) for value in (yes_int-no_int, no_int))
      selected, result = (self._scratch(data_dtype, data_layout) for _ in range(2))
      self._emit(selected, selector, delta, _EW_CFG[Ops.MUL])
      self._emit(result, baseline, selected, _EW_CFG[Ops.ADD])
      if data_layout is RKLayout.INT16: return self._widen_int16(u, result)
      return RKValue(result.arg, dtypes.int, self.count, RKLayout.INT_FP16)
    if u is self.root and u.dtype.scalar() in (dtypes.half, dtypes.int16) and _is_static_expr(u.src[0]):
      dtype = u.dtype.scalar()
      routes:dict[UOp, list[bool]] = {}
      def route(node:UOp, active:tuple[bool, ...]) -> None:
        if node.op is Ops.WHERE and _is_static_expr(node.src[0]):
          selector = tuple(bool(x) for x in _static_values(self.out_index, node.src[0], self.count, int))
          for child,take in zip(node.src[1:], (selector, tuple(not x for x in selector))):
            route(child, tuple(live and pick for live,pick in zip(active, take)))
        else: routes[node] = [old or live for old,live in zip(routes.get(node, [False]*self.count), active)]
      route(u, (True,)*self.count)
      def exact_operand(src:UOp) -> RKValue:
        return self._load(src, _fp16_bits(-65504.0)) if (src.op is Ops.LOAD and dtype is dtypes.half and len(src.src) > 2 and
          src.src[1].op is Ops.CONST and math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0.0 and
          (param:=_root_param(src.src[0])) is not None and param.src[0].op is Ops.CONST and int(param.src[0].arg) < self.count) else self.lower(src)
      expected, itemsize = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16, dtype.itemsize
      for partial,(leaf,mask) in enumerate(routes.items()):
        value = exact_operand(leaf)
        if value.layout is not expected: raise _RKGenericReject
        offsets = tuple(value.arg.addend//itemsize+i if take else -1 for i,take in enumerate(mask))
        self.post_gathers.append(RKGather(value.arg.index, self.out_param.arg.slot, self.count, offsets=offsets,
          partial=bool(partial), dst_kind=RKBufferKind.ARG, src_kind=value.arg.kind, itemsize=itemsize))
      return RKValue(self.out, dtype, self.count, expected)
    for fold in (_fold_where_abs, _fold_ordered_where, _fold_threshold_where):
      if (recipe:=fold(u)) is not None: return self.lower(recipe)
    return self._raw_where(u)

  def _widen_int16(self, u:UOp, source:RKValue) -> RKValue:
    if source.layout not in (RKLayout.INT16, RKLayout.BOOL_INT16) or (u is not self.root and self.int_layout is not RKLayout.INT32) or \
       (u is self.root and self.out_param.dtype.scalar() is not dtypes.int):
      raise _RKGenericReject
    zero, value = self._i16_const(0), self._scratch(dtypes.int,RKLayout.INT32,u=u)
    self.ew_ops.append(RKEWOp(value.arg, source.arg, zero, self.count, _EW_CFG[Ops.ADD], int16_input=True, int32_output=True)); return value

  def lower(self, u:UOp) -> RKValue:
    if u in self.values: return self.values[u]
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif (dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool) and u in self.static_nodes and
          not any(isinstance(node.arg, str) and node.arg.startswith("rockchip_") for node in u.toposort())):
      value = self._static(u)
    elif u.op is Ops.INDEX: value = self.lower(u.load())
    elif u.op is Ops.LOAD: value = self._load(u)
    elif u.op is Ops.BITCAST and len(u.src) == 1:
      source = self.lower(u.src[0])
      if dtype is dtypes.int16 and u.src[0].dtype.scalar() is dtypes.half and source.layout is RKLayout.FP16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.half and u.src[0].dtype.scalar() is dtypes.int16 and source.layout is RKLayout.INT16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      else: raise _RKGenericReject(f"bitcast {u.src[0].dtype.scalar()}->{dtype}")
      if u is self.root and value.arg != self.out:
        self.post_gathers.append(_raw_gather(replace(value.arg, addend=value.arg.addend//2), self.out_param.arg.slot, self.count,
                                             stride=1, itemsize=2, src_kind=value.arg.kind))
        value = RKValue(self.out, dtype, self.count, value.layout)
    elif u.op is Ops.CAST and len(u.src) == 1:
      source_dtype = u.src[0].dtype.scalar()
      int_range = _exact_int_range(u.src[0]) if source_dtype is dtypes.int else None
      source = self._load(u.src[0]) if dtype is dtypes.half and source_dtype is dtypes.float and u.src[0].op is Ops.LOAD else \
        self.lower(_fp32_expr_to_half(u.src[0])) if dtype is dtypes.half and source_dtype is dtypes.float else \
        self.lower(_int_fp16_expr(u.src[0])) if dtype is dtypes.half and source_dtype is dtypes.int and int_range is not None and \
        -_FP16_EXACT_INTEGER <= int_range[0] <= int_range[1] <= _FP16_EXACT_INTEGER else self.lower(u.src[0])
      if source.layout is RKLayout.INT32 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.int):
        value, tile = self._scratch(dtypes.half, RKLayout.FP16), self._scratch(dtypes.int, RKLayout.INT32, (self.count+3)//4<<6).arg
        self.ew_ops.append(RKEWOp(value.arg, source.arg, tile, self.count, _EW_CFG[Ops.MAX], int32_input=True))
      elif source.layout is RKLayout.BOOL_INT16 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.bool or
                                                     dtype is dtypes.int and self.int_layout is RKLayout.INT_FP16):
        value = self.lower(u.src[0].where(UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half)))
        if dtype is dtypes.int and value.layout is not RKLayout.FP16: raise _RKGenericReject
        if dtype is dtypes.int: value = RKValue(value.arg, dtype, self.count, RKLayout.INT_FP16)
      elif (source.layout is RKLayout.FP16 and (dtype is dtypes.half or dtype is dtypes.float and source_dtype is dtypes.half) or
            dtype is dtypes.half and source.layout in (RKLayout.BOOL_MASK, RKLayout.INT_FP16) or
            dtype is dtypes.int16 and source.layout in (RKLayout.INT16, RKLayout.BOOL_INT16) or
            dtype is dtypes.int and source.layout is RKLayout.BOOL_INT16 and self.int_layout is RKLayout.INT16):
        value = RKValue(source.arg, dtype, self.count,
                        RKLayout.FP16 if source.layout in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16) else RKLayout.INT16)
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
    elif dtype is dtypes.bool and u.op in (Ops.MUL, Ops.MAX):
      value = self._bool_binary(UOp(Ops.AND if u.op is Ops.MUL else Ops.OR, dtype, src=u.src))
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL): value = self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype in (dtypes.bool, dtypes.int16, dtypes.int):
      value = self._bool_binary(u) if dtype is dtypes.bool else self._integer_bitwise(u)
    elif u.op in (Ops.SHL, Ops.SHR) and dtype in (dtypes.int, dtypes.uint): value = self._int32_shift(u)
    elif u.op in (Ops.CDIV, Ops.CMOD) and dtype is dtypes.int:
      if self.int_layout is RKLayout.INT_FP16: value = RKValue(self.lower(_int_fp16_expr(u)).arg, dtype, self.count, self.int_layout)
      else: value = self._int32_divmod(u)
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN) and len(u.src) == 1 and dtype is dtypes.half and \
         (recipe:=_DPU_MATH[u.op](u.src[0])) is not None:
      value = self.lower(_tag_precise_adds(recipe))
      if value.layout is not RKLayout.FP16: raise _RKGenericReject
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    return self.values.setdefault(u, value)

  def finish(self) -> RKImage:
    nodes = self.root.toposort()
    if len(nodes) > 800 and not any(node.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.WHERE) for node in nodes):
      for node in nodes:
        if node.dtype.scalar() in (dtypes.half, dtypes.int16, dtypes.bool) and node.op in (Ops.CONST, Ops.LOAD, Ops.CAST, *GroupOp.ALU):
          self.lower(node)
    result, dtype = self.lower(self.root), self.out_param.dtype.scalar()
    if (dtype is dtypes.half and result.layout in (RKLayout.FP16, RKLayout.BOOL_MASK, RKLayout.INT_FP16) or
        dtype is dtypes.int16 and result.layout is RKLayout.INT16):
      layout = RKLayout.FP16 if dtype is dtypes.half else RKLayout.INT16
      if result.arg != self.out: self._emit(RKValue(self.out, dtype, self.count, layout), result, result, _EW_CFG[Ops.MAX])
    elif ((dtype is dtypes.bool and result.layout is RKLayout.BOOL_MASK or
           dtype is dtypes.int and result.layout is RKLayout.INT_FP16) and
          (tile:=self._scratch(dtypes.int, RKLayout.INT32, (self.count+3)//4<<6).arg)):
      self.ew_ops.append(RKEWOp(self.out, result.arg, tile, self.count, _EW_CFG[Ops.MAX], stateful=True, int32_output=True,
                                bool_output=dtype is dtypes.bool))
    elif (dtype is dtypes.bool and result.layout is RKLayout.BOOL_INT16 or dtype is dtypes.int and result.layout is RKLayout.INT32):
      if dtype is dtypes.bool or result.arg != self.out:
        source = result.arg if dtype is dtypes.bool else replace(result.arg, addend=result.arg.addend//4)
        self.post_gathers.append(_raw_gather(source, self.out_param.arg.slot, self.count,
          stride=1 if dtype is dtypes.int else 2, itemsize=4 if dtype is dtypes.int else 1, src_kind=source.kind))
    elif dtype is dtypes.int and result.layout is RKLayout.INT16: self._widen_int16(self.root, result)
    elif dtype is dtypes.float and result.layout is RKLayout.FP16:
      groups = tuple(range(0, self.count, _EW_ELEMS_32BIT))
      aligned, split = self._scratch(dtypes.half, RKLayout.FP16, len(groups)*16), len(self.ew_ops)
      for group,start in enumerate(groups):
        lanes = min(_EW_ELEMS_32BIT, self.count-start)
        self.mid_gathers.append(RKGather(result.arg.index, aligned.arg.index, lanes,
          offsets=tuple(result.arg.addend//2+lane for lane in range(start, start+lanes)), dst_addend=group*8,
          src_kind=result.arg.kind, after=split))
        source = replace(aligned.arg, addend=group*16)
        self.ew_ops.append(RKEWOp(replace(self.out, addend=start*4), source, source, lanes, _EW_CFG[Ops.MAX] | _EW_STAGE_FP32_OUT))
    else: raise _RKGenericReject
    constants = b"" if not self.constants else b"".join(
      {slot:bits for bits,slot in self.constants.items()}.get(i, b"\0\0") for i in range(max(self.constants.values())+1))
    image = RKImage(RKTarget.RK3588, tuple(self.scratch), constants, RKIMAGE_VERSION, tuple(self.gathers), tuple(self.ew_ops),
                    tuple(self.mid_gathers), min((g.after for g in self.mid_gathers if g.after >= 0), default=0),
                    tuple(self.post_gathers), tuple(self.host_gathers))
    return _reuse_linear_scratch(image, self.constants)

def _expand_math_uops(root:UOp, *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  bounded_recipes = len(root.toposort()) <= _MAX_OPTIONAL_RECIPE_NODES
  composite_math = _fold_inverse_hyperbolic(root) if bounded_recipes else None
  if composite_math is None and bounded_recipes: composite_math = _fold_atan(root)
  cache:dict[UOp, UOp] = {}
  def physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp:
    placeholders = {source:UOp.param(-index-1, source.dtype, ()) for index,source in enumerate(opaque)}
    return _tag_precise_adds(recipe.substitute(placeholders)).substitute({placeholder:source for source,placeholder in placeholders.items()})
  if composite_math is not None: root = physical_recipe(composite_math)
  def rewrite(u:UOp) -> UOp:
    if u in cache: return cache[u]
    if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.float:
      return cache.setdefault(u, rewrite(physical_recipe(_dpu_sin(u.src[0].src[0]), (u.src[0].src[0],)))
                              if u.src[0].op is Ops.SIN else _canonical_half_storage(u.src[0]))
    if accurate_adds and bounded_recipes and u.op is Ops.ADD and u.dtype.scalar() is dtypes.half and u.arg is None:
      try:
        return cache.setdefault(u, _accurate_add_recipe(u))
      except _RKGenericReject: pass
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.WHERE and (absolute:=_fold_where_abs(mapped)) is not None: mapped = rewrite(absolute)
    if mapped.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN):
      if mapped.op is Ops.LOG2 and mapped.src[0].op is Ops.WHERE: raise _RKGenericReject
      if (recipe:=_DPU_MATH[mapped.op](mapped.src[0])) is None: raise _RKGenericReject
      mapped = rewrite(physical_recipe(recipe, (mapped.src[0],)))
    elif mapped.op is Ops.TRUNC and mapped.dtype.scalar() is dtypes.half and not _is_static_expr(mapped):
      mapped = rewrite(_fold_trunc(mapped))
    return cache.setdefault(u, mapped)
  return rewrite(root)

def _finite_int_max_neutrals(root:UOp) -> UOp:
  """Canonicalize finite physical neutrals for FP selectors and exact INT32 MAX arithmetic."""
  cache:dict[tuple[UOp, bool], UOp] = {}
  stack:list[tuple[UOp, bool, bool]] = [(root, False, False)]
  while stack:
    u,under_max,ready = stack.pop(); key,active = (u,under_max), under_max or u.op is Ops.MAX and u.dtype.scalar() is dtypes.int
    if key in cache: continue
    if not ready:
      stack.append((u, under_max, True))
      stack.extend((src, active, False) for src in reversed(u.src))
      continue
    src = tuple(cache[(source,active)] for source in u.src)
    if root.op is Ops.MAX and u.op is Ops.WHERE and src[1].op is Ops.CONST and src[1].dtype.scalar() in (dtypes.half,dtypes.float) and math.isinf(float(src[1].arg)) and float(src[1].arg) < 0.0: src = (src[0],src[1].const_like(-65504.0),src[2])  # noqa: E501
    cache[key] = u.const_like(-2048) if active and u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg) == dtypes.int.min else u.replace(src=src)  # noqa: E501
  return cache[(root, False)]

def _unroll_static_reduces(root:UOp, precise:bool=True) -> UOp:
  """Interpret static REDUCE/RANGE structure into ordinary semantic UOps; CMAC leaves product terms factorable."""
  cache:dict[UOp, UOp] = {}
  for u in root.toposort():
    mapped = u.replace(src=tuple(cache[src] for src in u.src))
    if mapped.op is Ops.REDUCE:
      reduce_op, ranges = mapped.arg[0] if isinstance(mapped.arg, tuple) else mapped.arg, list(mapped.src[1:])
      if reduce_op not in (Ops.ADD, Ops.MAX, Ops.MUL) or not ranges or any(r.op is not Ops.RANGE for r in ranges): raise _RKGenericReject
      envs = _iter_range_env(ranges, _MAX_GENERIC_UNROLL, False)
      if not envs or len(envs)*len(mapped.src[0].toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
      terms = [mapped.src[0].substitute({r:r.const_like(env[r]) for r in ranges}, walk=True) for env in envs]
      nonzero = [term for term in terms if not (term.op is Ops.CONST and float(term.arg) == 0.0)] \
        if reduce_op is Ops.ADD and u.dtype.scalar() is dtypes.half else []
      if precise and nonzero and all(term.op is Ops.MUL and term.dtype.scalar() is dtypes.half for term in nonzero): mapped = _precise_mul_sum(nonzero)  # noqa: E501
      else:
        while len(terms) > 1:
          terms = [UOp(reduce_op, u.dtype, src=(terms[i], terms[i+1])) for i in range(0, len(terms)-1, 2)] + (terms[-1:] if len(terms) & 1 else [])
        mapped = terms[0]
    cache[u] = mapped
  return cache[root]

def _local_buffer(u:UOp) -> UOp|None:
  u = _strip_cast(u)
  while u.op in (Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.AFTER): u = u.src[0]
  return u if u.op is Ops.BUFFER else None

def _unroll_static_local(uops:list[UOp], root:UOp) -> UOp:
  """Execute ordered static local accumulators without recovering a tensor operation."""
  local_loads, definitions = _semantic_loads(root, True), typing_cast(dict[UOp, _RKStaticLocalDef], {})
  expanded:dict[tuple[UOp, tuple[tuple[UOp, int], ...]], UOp] = {}
  active:set[tuple[UOp, tuple[tuple[UOp, int], ...]]] = set()
  budget = [_MAX_STATIC_LOCAL_STEPS]
  def expand_dependencies(expr:UOp, owner:UOp, env:dict[UOp, int]) -> UOp:
    substitutions = {axis:axis.const_like(value) for axis,value in env.items()}
    substitutions.update({load:expand_load(load, env) for load in _semantic_loads(expr, True)
                          if _local_buffer(load) is not owner})
    return expr.substitute(substitutions, walk=True)
  def expand_load(load:UOp, env:dict[UOp, int]) -> UOp:
    buffer = _local_buffer(load)
    if buffer is None or buffer.src[0].op is not Ops.CONST: raise _RKGenericReject
    stores = [u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]
    if len(stores) == 1 and stores[0].src[0].op is Ops.INDEX and load.src[0].op is Ops.INDEX:
      store_index, load_index = stores[0].src[0].src[1], load.src[0].src[1]
      axes, load_axes = _index_ranges(store_index), _index_ranges(load_index)
      if buffer.dtype.scalar() is not dtypes.bool: raise _RKGenericReject
      if len(axes) != 1 or store_index.key != axes[0].key or axes[0].op is not Ops.SPECIAL or \
         axes[0].src[0].op is not Ops.CONST or (store_extent:=int(axes[0].src[0].arg)) <= 0 or \
         store_extent > int(buffer.src[0].arg) or len(load_axes) != 1 or load_index.key != load_axes[0].key or \
         load_axes[0].op not in (Ops.RANGE, Ops.SPECIAL) or load_axes[0].src[0].op is not Ops.CONST or \
         (load_extent:=int(load_axes[0].src[0].arg)) <= 0 or load_extent > int(buffer.src[0].arg): raise _RKGenericReject
      updates = {node.op for node in root.toposort() if node.dtype.scalar() is dtypes.bool and node.op in (Ops.AND, Ops.OR) and any(
        _local_buffer(local) is buffer for local in _semantic_loads(node, True))}
      if len(updates) != 1: raise _RKGenericReject
      expanded = expand_dependencies(stores[0].src[1], buffer, env).substitute({axes[0]:load_index}, walk=True)
      if store_extent < int(buffer.src[0].arg):
        expanded = (load_index < UOp.const(store_extent, load_index.dtype)).where(expanded,UOp.const(updates.pop() is Ops.AND, buffer.dtype.scalar()))
      return expanded.substitute({axis:axis.const_like(value) for axis,value in env.items()}, walk=True) if env else expanded
    return expand_buffer(buffer, env)
  def expand_buffer(buffer:UOp, env:dict[UOp, int]) -> UOp:
    key = buffer, tuple(sorted(env.items(), key=lambda item:item[0].key))
    if key in expanded: return expanded[key]
    if key in active: raise _RKGenericReject
    active.add(key)
    definition = definitions.setdefault(buffer, _static_local_defs(uops, {buffer})[buffer])
    if not definition.loops or any(loop.src[0].op is not Ops.CONST or not 0 <= int(loop.src[0].arg) <= _MAX_GENERIC_UNROLL
                                   for loop in definition.loops): raise _RKGenericReject
    if (iterations:=math.prod(int(loop.src[0].arg) for loop in definition.loops)) > min(_MAX_GENERIC_UNROLL, budget[0]): raise _RKGenericReject
    budget[0] -= iterations
    accumulator = expand_dependencies(definition.initial, buffer, env)
    for loop_env in _iter_range_env(list(definition.loops), None, False):
      accumulator = UOp(definition.update_op, definition.term.dtype, src=(accumulator, expand_dependencies(definition.term, buffer, env|loop_env)))
    active.remove(key)
    if len(accumulator.toposort()) > _MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
    return expanded.setdefault(key, accumulator)
  return root.substitute({load:expand_load(load, {}) for load in local_loads}, walk=True)

class _RKStaticLocalDef(NamedTuple): initial:UOp; update_op:Ops; term:UOp; loops:tuple[UOp, ...]

def _static_local_defs(uops:list[UOp], buffers:set[UOp]) -> dict[UOp, _RKStaticLocalDef]:
  """Parse scalar local accumulators without assigning tensor-operation meaning to their loops."""
  definitions:dict[UOp, _RKStaticLocalDef] = {}
  for buffer in buffers:
    if buffer.src[0].op is not Ops.CONST or int(buffer.src[0].arg) != 1: raise _RKGenericReject
    initializers:list[UOp] = []
    updates:list[tuple[Ops, UOp, tuple[UOp, ...]]] = []
    for store in (u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer):
      value = _strip_cast(store.src[1])
      accumulator = [(i, load) for i,src in enumerate(value.src) if value.op in (Ops.ADD, Ops.MAX, Ops.MUL, Ops.AND, Ops.OR) and \
                     (load:=_local_load(src)) is not None and _local_buffer(load) is buffer]  # noqa: E501
      if len(accumulator) == 1:
        term = value.src[1-accumulator[0][0]]
        if any(_local_buffer(load) is buffer for node in term.toposort() if (load:=_local_load(node)) is not None):
          raise _RKGenericReject
        loops = tuple(src for src in store.src[0].src[0].src[1:] if src.op is Ops.RANGE) if store.src and store.src[0].op is Ops.INDEX and \
          store.src[0].src[0].op is Ops.AFTER else tuple(r for r in _index_ranges(term) if r.arg[1] is AxisType.REDUCE)  # noqa: E501
        updates.append((value.op, term, loops))
      elif not any(_local_buffer(load) is buffer for node in value.toposort() if (load:=_local_load(node)) is not None):
        initializers.append(store.src[1])
    if len(initializers) != 1 or len(updates) != 1: raise _RKGenericReject
    definitions[buffer] = _RKStaticLocalDef(initializers[0], *updates[0])
  return definitions

def _lower_vectorized_scalar_local_extrema(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Vectorize two dependent scalar local MAX accumulators without assigning meaning to their tensor source."""
  _, out_param, count, _, root = output
  if count != 1 or out_param.dtype.scalar() is not dtypes.int: return None
  try: definitions = _static_local_defs(uops, {buffer for u in uops if u.op is Ops.BUFFER and (buffer:=_local_buffer(u)) is not None})
  except _RKGenericReject: return None
  limit = min(32767, _MAX_EW_ELEMS_FP16)
  descriptors = {dtype:(buffer, definition, tuple(int(loop.src[0].arg) for loop in definition.loops))
    for dtype,initial in ((dtypes.half,-math.inf),(dtypes.int,dtypes.int.min)) for buffer,definition in definitions.items()
    if buffer.dtype.scalar() is dtype and definition.update_op is Ops.MAX and definition.initial.op is Ops.CONST and
    float(definition.initial.arg) == initial and definition.loops and all(loop.src[0].op is Ops.CONST and 1 <= int(loop.src[0].arg) <= limit for loop in definition.loops)}  # noqa: E501
  if (len(definitions) != 2 or set(descriptors) != {dtypes.half, dtypes.int} or
      descriptors[dtypes.half][2] != descriptors[dtypes.int][2]): return None
  (value_buffer,value_def,extents), (index_buffer,index_def,_) = descriptors[dtypes.half], descriptors[dtypes.int]
  if not 2 <= (total:=math.prod(extents)) <= limit: return None
  weighted_terms = list(_iter_binary(index_def.term, Ops.MUL))
  casts = [term for term in weighted_terms if term.op is Ops.CAST and term.dtype.scalar() is dtypes.int and term.src[0].dtype.scalar() is dtypes.bool]  # noqa: E501
  if len(casts) != 1 or len(weighted_terms) != 2: return None
  coordinate, predicate = weighted_terms[1 if weighted_terms[0] is casts[0] else 0], _inverted_condition(casts[0].src[0], True)
  if predicate is None or predicate.op is not Ops.CMPNE or len(predicate.src) != 2 or \
     any(src.dtype.scalar() is not dtypes.half for src in predicate.src): return None
  if ((current:=next((src for src in predicate.src if (load:=_local_load(src)) is not None and _local_buffer(load) is value_buffer), None)) is None or
      (candidate:=next((src for src in predicate.src if src is not current), None)) is None): return None
  if _strip_cast(candidate.substitute(dict(zip(index_def.loops, value_def.loops)), walk=True)).key != _strip_cast(value_def.term).key: return None  # noqa: E501
  try: coordinates = tuple(int(_eval_expr(coordinate, env, {})) for env in _iter_range_env(list(index_def.loops), None, False))
  except RuntimeError: return None
  if len(coordinates) != total or any(not 0 <= value <= 32767 for value in coordinates): return None
  if len(final_loads:=[load for load in _semantic_loads(root, local=True) if _local_buffer(load) is index_buffer]) != 1: return None
  try: mapped_outputs = tuple(int(_eval_expr(root.substitute({final_loads[0]:final_loads[0].const_like(value)}), {}, {})) for value in coordinates)
  except RuntimeError: return None
  if (second:=next((i for i in range(1, len(coordinates)) if coordinates[i] != coordinates[0]), None)) is None: return None
  coordinate_delta, output_delta = coordinates[second]-coordinates[0], mapped_outputs[second]-mapped_outputs[0]
  if (slope:=output_delta//coordinate_delta)*coordinate_delta != output_delta: return None
  baseline = mapped_outputs[0]-slope*coordinates[0]
  if any(result != baseline+slope*value for value,result in zip(coordinates, mapped_outputs)) or not all(-32768 <= value <= 32767 for value in (*mapped_outputs, slope, baseline)): return None  # noqa: E501
  flat = UOp.const(0, dtypes.int)
  for index,axis in reversed(tuple(enumerate(value_def.loops))):
    flat = flat.alu(Ops.ADD, axis.alu(Ops.MUL, axis.const_like(math.prod(extents[index+1:]))))
  fake_param = UOp.param(1+max((u.arg.slot for u in uops if u.op is Ops.PARAM and u.arg is not None), default=out_param.arg.slot), dtypes.half, (total,))  # noqa: E501
  child_store = fake_param.index(flat).store(value_def.term).end(*value_def.loops)
  child = _lower_uop_program(list(child_store.sink().toposort()), vectorize_reductions=False)
  if child is None or child.host_gathers or child.host_scatters: return None
  target = RKArg(RKBufferKind.SCRATCH, len(child.scratch)); child = replace(_alias_image_args(child, {fake_param.arg.slot:target}), scratch=child.scratch+(RKScratch(_scratch_bytes(total)),))  # noqa: E501
  scratch = list(child.scratch); values = RKArg(RKBufferKind.SCRATCH, len(scratch)-1)
  def allocate(lanes:int=total) -> RKArg: scratch.append(RKScratch(_scratch_bytes(lanes))); return RKArg(RKBufferKind.SCRATCH, len(scratch)-1)
  ops, gathers, mid = list(child.ew_ops), list(child.gathers), list(child.mid_gathers); mid.extend(replace(gather, after=len(ops)) for gather in child.post_gathers)  # noqa: E501
  spaced = allocate(total*32)
  mid.append(RKGather(values.index, spaced.index, total, axes=((1,total,1),), dst_stride=32, src_kind=RKBufferKind.SCRATCH, after=len(ops)))  # noqa: E501
  best = _reduce_rows(ops, [replace(spaced, addend=lane*64) for lane in range(total)], 1, _EW_CFG[Ops.MAX])
  best_values = allocate(); mid.append(RKGather(best.index, best_values.index, total, offsets=(best.addend//2,)*total, src_kind=RKBufferKind.SCRATCH, after=len(ops)))  # noqa: E501
  fake_out, fake_values, fake_best, fake_coordinates = range(fake_param.arg.slot, fake_param.arg.slot+4)
  lane = UOp.range(total, max((u.arg[0] for u in (*value_def.loops, *index_def.loops) if isinstance(u.arg, tuple)), default=-1)+1)
  lhs, rhs = UOp.param(fake_values, dtypes.half, (total,)).index(lane).load(), UOp.param(fake_best, dtypes.half, (total,)).index(lane).load()
  equal = UOp(Ops.CMPEQ, dtypes.bool, src=(lhs, rhs)).cast(dtypes.int16)*UOp.param(fake_coordinates, dtypes.int16, (total,)).index(lane).load()
  selected = UOp.param(fake_out, dtypes.int16, (total,)).index(lane).store(equal).end(lane)
  if (selected_image:=_lower_uop_program(list(selected.sink().toposort()), vectorize_reductions=False)) is None: return None
  coordinate_arg = allocate(); gathers.append(RKGather(out_param.arg.slot, coordinate_arg.index, total, values=coordinates))
  prefix = RKImage(RKTarget.RK3588, tuple(scratch), child.constants, gathers=tuple(gathers), ew_ops=tuple(ops),
                   mid_gathers=tuple(mid), gather_after=child.gather_after)
  if (combined:=_append_inplace_image(prefix, selected_image)) is None: return None
  def retained(arg:RKArg) -> RKArg: return replace(arg, index=arg.index+len(selected_image.constants)//2) \
    if arg.kind is RKBufferKind.SCRATCH and arg.index >= len(prefix.constants)//2 else arg
  weighted = RKArg(RKBufferKind.SCRATCH, len(combined.scratch))
  combined = _alias_image_args(combined, {fake_values:retained(values), fake_best:retained(best_values),
                                          fake_coordinates:retained(coordinate_arg), fake_out:weighted})
  scratch, gathers, ops, mid = list(combined.scratch), list(combined.gathers), list(combined.ew_ops), list(combined.mid_gathers)
  scratch.append(RKScratch(_scratch_bytes(total)))
  mid.append(RKGather(weighted.index, retained(spaced).index, total, axes=((1,total,1),), dst_stride=32, src_kind=RKBufferKind.SCRATCH, after=len(ops)))  # noqa: E501
  result = _reduce_rows(ops, [replace(retained(spaced), addend=lane*64) for lane in range(total)], 1, _EW_CFG[Ops.MAX], int16=True)
  for value,op in ((v,o) for v,o,n in ((slope,Ops.MUL,1), (baseline,Ops.ADD,0)) if v != n):
    source, previous, result = allocate(1), result, allocate(1)
    gathers.append(RKGather(out_param.arg.slot, source.index, 1, values=(_int16_bits(value),)))
    ops.append(RKEWOp(result, previous, source, 1, _EW_CFG[op], **_INT16_EW))
  zero = allocate(1); gathers.append(RKGather(out_param.arg.slot, zero.index, 1, values=(0,)))
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG, out_param.arg.slot), result, zero, 1, _EW_CFG[Ops.ADD], int16_input=True, int32_output=True))
  image = RKImage(RKTarget.RK3588, tuple(scratch), combined.constants, gathers=tuple(gathers), ew_ops=tuple(ops), mid_gathers=tuple(mid), gather_after=child.gather_after)  # noqa: E501
  return image if all(len(items) <= _RKIMAGE_U16_MAX for items in (image.scratch, image.gathers, image.ew_ops, image.mid_gathers)) else None

def _lower_host_scatter(output:RKOutput) -> RKImage|None:
  """Lower a direct dynamic STORE as raw last-writer host address materialization."""
  if os.getenv("ROCKCHIP_HOST_GATHER", "1") != "1" or len(output[0].src) != 2: return None
  _, out_param, out_count, dynamic_index, value = output
  if (index_info:=_runtime_index(dynamic_index)) is None: return None
  (_, index_param, lane_index, index_itemsize), value = index_info, _strip_cast(value)
  if (value.op is not Ops.LOAD or len(value.src) != 1 or value.src[0].op is not Ops.INDEX or
      (source:=_root_param(value.src[0])) is None or source.src[0].op is not Ops.CONST or
      value.src[0].src[1].key != lane_index.key or source.dtype.scalar() is not out_param.dtype.scalar()): return None
  count = int(index_param.src[0].arg)
  if int(source.src[0].arg) < count: return None
  address = RKHostAddress(RKArg(RKBufferKind.ARG, source.arg.slot), RKArg(RKBufferKind.ARG, index_param.arg.slot),
    RKArg(RKBufferKind.ARG, out_param.arg.slot), count, int(source.src[0].arg), out_count,
    itemsize=out_param.dtype.scalar().itemsize, index_itemsize=index_itemsize)
  return RKImage(RKTarget.RK3588, host_scatters=(address,))

def _same_condition(a:UOp, b:UOp) -> bool: return a.key == b.key or a.op is b.op is Ops.AND and len(a.src) == len(b.src) == 2 and any(
  all(_same_condition(x, y) for x,y in zip(a.src, order)) for order in (b.src, b.src[::-1]))

def _lower_uop_program(uops:list[UOp], *, vectorize_reductions:bool=True, recipes_ready:bool=False) -> RKImage|None:
  """Lower a composable typed UOp program; return None for the legacy correctness oracle."""
  if any(u.op is Ops.PARAM and not 0 <= u.arg.slot <= _RKIMAGE_U16_MAX for u in uops): return None
  accepted = (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool, dtypes.uchar)
  strict_output, local_output, output_stores = _outs(uops)
  if len(output_stores) > 1:
    lower_store = functools.partial(_lower_uop_program, vectorize_reductions=vectorize_reductions, recipes_ready=recipes_ready)
    if (combined:=lower_store(list(UOp(Ops.SINK, src=(output_stores[0],)).toposort()))) is None: return None
    for store in output_stores[1:]:
      if (child:=lower_store(list(UOp(Ops.SINK, src=(store,)).toposort()))) is None: return None
      if (combined:=_append_inplace_image(combined, child)) is None: return None
    return combined
  strict_output, local_output = (_admit(output, accepted) for output in (strict_output, local_output))
  if (cmac:=_try(local_output, (dtypes.half,dtypes.float), _lower_cmac_reduction, uops, v=vectorize_reductions)) is not None: return cmac
  if (scatter:=_try(strict_output, (dtypes.half, dtypes.int16), _lower_host_scatter)) is not None: return scatter
  if (image:=_try(strict_output, dtypes.uchar, _lower_fp16_uint8_cast)) is not None: return image
  if (bool_output:=_admit(strict_output, dtypes.bool)) is not None and \
     (nonzero:=_fp16_nonzero_mask(bool_output[4])) is not None: return _typed_half_image(bool_output, nonzero, True, bool_output=True)
  for dtype in (dtypes.half, dtypes.int16, dtypes.int):
    if (image:=_try(strict_output, dtype, _lower_dynamic_typed_load, dtype)) is not None: return image
  if (extrema:=_try(local_output, dtypes.int, _lower_vectorized_scalar_local_extrema, uops, v=vectorize_reductions)) is not None: return extrema
  storage_uops, storage_product_adds = typing_cast(list[UOp]|None, None), False
  if any(u.dtype.scalar() is dtypes.float for u in uops):
    if (sink:=next((u for u in uops if u.op is Ops.SINK), None)) is not None:
      storage_sink = sink
      if (storage_output:=_admit(local_output, dtypes.half)) is not None:
        storage_root, root_storage = storage_output[4], _typed_cast_source(storage_output[4], dtypes.half, dtypes.float) is not None
        storage_product_adds = any(boundary.op is Ops.CAST and boundary.dtype.scalar() is dtypes.half and len(boundary.src)==1 and
          boundary.src[0].dtype.scalar() is dtypes.float and any(term.op is Ops.MUL and term.dtype.scalar() is dtypes.float or
            (source:=_typed_cast_source(term, term.dtype.scalar(), dtypes.half)) is not None and source.op is Ops.MUL
            for term in _iter_binary(boundary.src[0], Ops.ADD, dtypes.float)) and
          (boundary is not storage_root or len(boundary.src[0].toposort())>64) for boundary in storage_root.toposort())
        # A later half FDIV/WHERE/etc. can own several independent FP32 reduction boundaries. Commit each CAST
        # before the bottom-up generic rewrite erases the semantic FP32 ADD tree, including pure denominator sums.
        if (ratio:=_fp32_ratio_to_half(storage_root)) is not None: storage_sink = storage_sink.substitute({storage_root:ratio})
        else:
          nested_storage:dict[UOp, UOp] = {}
          for boundary in storage_root.toposort():
            if boundary is storage_root or (source:=_typed_cast_source(boundary, dtypes.half, dtypes.float)) is None or \
               source.op is not Ops.ADD: continue
            try: nested_storage[boundary] = _canonical_half_storage(source)
            except _RKGenericReject: pass
          if nested_storage: storage_sink = storage_sink.substitute(nested_storage)
        if root_storage:
          try:
            source = storage_root.src[0]
            if not _has_runtime_address(source):
              converted = _expand_math_uops(storage_root) if source.op is Ops.SIN else _canonical_half_storage(source)
              storage_sink = sink.substitute({storage_root:converted})
          except _RKGenericReject: pass
      storage_sink = storage_sink.substitute({u:_expand_math_uops(u) for u in storage_sink.toposort()
        if u.op is Ops.CAST and u.dtype is dtypes.half and len(u.src) == 1 and u.src[0].op is Ops.SIN and u.src[0].dtype is dtypes.float})
      storage_uops = list(graph_rewrite(storage_sink, _pm_storage_common, name="rockchip generic storage precision").toposort())
  if (output:=local_output if storage_uops is None else _output_store(uops:=storage_uops, accepted, allow_local=True)) is None or \
     output[1].dtype.scalar() is dtypes.uchar or len(output[0].src) != 2:
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage(RKTarget.RK3588)
  if output[2] > _MAX_EW_ELEMS_FP16 and any(_local_load(node) is not None for node in output[4].toposort()): return None
  if output[1].dtype.scalar() is dtypes.int and (coordinates:=next((x for dtype in (dtypes.int16,dtypes.int)
      if (x:=_lower_bounded_integer_predicate_coordinates(output, dtype)) is not None), None)) is not None: return coordinates
  try:
    if not ((affine:=typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(output[3]))) is not None and affine[0] == 0 and
            set(affine[1]) == set(_index_ranges(output[3])) and _affine_output_axes(affine, output[2]) is not None) and \
       _static_values(output[3], output[3], output[2], int) != tuple(range(output[2])): return None
    root=_finite_int_max_neutrals(_unroll_static_local(uops,_unroll_static_reduces(output[4]) if Ops.REDUCE in (u.op for u in uops) else output[4]))
    if not recipes_ready and len(storage_uops if storage_uops is not None else root.toposort()) <= 256:
      root = _expand_math_uops(root, accurate_adds=storage_uops is None or storage_product_adds)
    n = root.toposort()
    if len(n) > _MAX_GENERIC_EXPANDED_NODES and os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(n)}")
    if len(n) > _MAX_GENERIC_EXPANDED_NODES: return None
    if root is not output[4]: output = (output[0].replace(src=(output[0].src[0], root)), *output[1:4], root)
    image = RKContext(output, accurate_adds=not recipes_ready and (storage_uops is None or storage_product_adds) and
                      len(n) <= _MAX_OPTIONAL_RECIPE_NODES and not _has_runtime_address(output[4])).finish()
    counts = (len(image.scratch), len(image.gathers)+len(image.mid_gathers)+len(image.post_gathers),
              len(image.host_gathers), len(image.host_scatters))
    if any(count > _RKIMAGE_U16_MAX for count in counts) and os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1":
      raise _RKGenericReject("image u16 counts " + repr(counts) + f", ew_ops={len(image.ew_ops)}")
    if any(count > _RKIMAGE_U16_MAX for count in counts): return None
    return image
  except (_RKGenericReject, RuntimeError, ValueError, KeyError):
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise
    return None


class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

def _const_operand(u:UOp, op:Ops, value:float|None=None) -> tuple[UOp, UOp]|None: return None if u.op is not op else next(
  ((a, b) for a,b in (u.src, u.src[::-1]) if b.op is Ops.CONST and (value is None or float(b.arg) == value)), None)

def _positive_mask(u:UOp) -> UOp: return UOp(Ops.MAX, dtypes.half, src=(u, u), arg=_NATIVE_POSITIVE_MASK)

def _mask_mul(a:UOp, b:UOp) -> UOp: return a.alu(Ops.MUL, b, arg=_NATIVE_MASK_MUL)

def _half(value:float) -> UOp: return UOp.const(value, dtypes.half)

def _native_min(*values:UOp, dtype:DType|None=None) -> UOp: return UOp(Ops.MAX, dtype or values[0].dtype, src=(values[0],values[1]), arg=_NATIVE_MIN)

def _native_same(value:UOp, arg:str) -> UOp: return UOp(Ops.MAX, value.dtype, src=(value,value), arg=arg)

def _fold_ordered_where(x:UOp) -> UOp|None:
  """Turn ordered clamp WHEREs into native DPU EW MIN/MAX stages."""
  gate, yes, no = x.src
  if gate.op is Ops.OR and yes.op is Ops.CONST:
    for upper, lower in ((gate.src[0], gate.src[1]), (gate.src[1], gate.src[0])):
      if (upper.op is Ops.CMPLT and upper.src[0].key == yes.key and upper.src[1].op is Ops.MAX and
          lower.op is Ops.CMPLT and lower.src[0].key == no.key and lower.src[1].key == yes.key and
          {u.key for u in upper.src[1].src} == {no.key, yes.key}): return UOp(Ops.MAX, upper.src[1].dtype, src=(upper.src[1], yes), arg=_NATIVE_MIN)
  if gate.op is not Ops.CMPLT: return None
  lhs, rhs = gate.src
  if yes.key == rhs.key and no.key == lhs.key: return lhs.alu(Ops.MAX, rhs)
  if yes.key == lhs.key and no.key == rhs.key: return UOp(Ops.MAX, lhs.dtype, src=(lhs, rhs), arg=_NATIVE_MIN)
  return None

def _unwrap_condition(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype.scalar() in (dtypes.bool, dtypes.half, dtypes.float): u = u.src[0]
  return u

def _finite_positive_mask(u:UOp) -> UOp:
  """Map finite binary16 values to `u > 0` without the stateful DPU compare path."""
  magnitude = u.alu(Ops.MAX, UOp.const(0.0, dtypes.half)).alu(Ops.MUL, UOp.const(256.0, dtypes.half)).alu(Ops.MUL, UOp.const(256.0, dtypes.half))
  return UOp(Ops.MAX, magnitude.dtype, src=(magnitude, UOp.const(1.0, dtypes.half)), arg=_NATIVE_MIN)

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
  lhs, threshold, yes, no = gate.src[0], float(gate.src[1].arg), *(_unwrap_condition(u) for u in x.src[1:])
  if (mask:=_mask_expr(x.src[0])) is None: return None
  if yes.key == lhs.key and no.op is Ops.CONST and math.isfinite(float(no.arg)) and float(no.arg) != threshold:
    return _native_min(lhs.cast(dtypes.half), _half(threshold), dtype=lhs.dtype).alu(
      Ops.ADD, _mask_mul(UOp.const(1.0, dtypes.half).alu(Ops.SUB, mask), _half(float(no.arg)-threshold)))
  if no.key == lhs.key and yes.op is Ops.CONST and math.isfinite(float(yes.arg)) and float(yes.arg) != threshold:
    return lhs.cast(dtypes.half).alu(Ops.MAX, _half(threshold)).alu(
      Ops.ADD, _mask_mul(mask, _half(float(yes.arg)-threshold)))
  return None

def _fold_relu_cap(x:UOp) -> UOp|None:
  """Recognize relu(source)-relu(source-cap), the canonical ReLU6/clamp expansion."""
  for positive, negative in (x.src, x.src[::-1]):
    source, scaled = _relu_operand(positive), _const_operand(negative, Ops.MUL, -1.0)
    if source is None or scaled is None or (upper:=_relu_operand(scaled[0])) is None: continue
    source_base, source_shift = (source, 0.0) if (term:=_const_operand(source, Ops.ADD)) is None else (term[0], float(term[1].arg))
    upper_base, upper_shift = (upper, 0.0) if (term:=_const_operand(upper, Ops.ADD)) is None else (term[0], float(term[1].arg))
    if source_base.key != upper_base.key or (cap:=source_shift-upper_shift) < 0.0: continue
    if cap == 6.0: return UOp(Ops.MAX, x.dtype, src=(source, UOp.const(0.0, dtypes.half)), arg=_NATIVE_RELU6)
    return UOp(Ops.MAX, positive.dtype, src=(positive, UOp.const(cap, dtypes.half)), arg=_NATIVE_MIN)
  return None

def _fold_where_abs(x:UOp) -> UOp|None:
  """Recognize `WHERE(x < 0, -x, x)` before an unselected infinity can contaminate a mask blend."""
  if x.op is not Ops.WHERE or len(x.src) != 3 or x.dtype.scalar() is not dtypes.half: return None
  condition, negative = _strip_cast(x.src[0]), _strip_cast(x.src[1])
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
  negative = zero.alu(Ops.SUB, zero.alu(Ops.SUB, source).alu(Ops.MAX, zero))
  return _native_same(source.alu(Ops.MAX, zero), _NATIVE_FLOOR).alu(Ops.ADD, _native_same(negative, _NATIVE_CEIL))

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
  return UOp.const(-1.0 if value < 0 else 1.0, dtypes.half).alu(Ops.FDIV, denominator).alu(Ops.FDIV, UOp.const(0.0, dtypes.half))

_pm_storage_common = PatternMatcher([(UPat((Ops.WHERE, Ops.ADD, Ops.MUL), dtypes.float, name="x"), lambda x:None if _is_static_expr(x) else
  UOp(Ops.WHERE, dtypes.half, src=(x.src[0], x.src[1].cast(dtypes.half), x.src[2].cast(dtypes.half)), arg=x.arg) if x.op is Ops.WHERE else
  x.src[0].cast(dtypes.half).alu(x.op, x.src[1].cast(dtypes.half))),
  *[(UPat(op, dtypes.half, name="x"), callback) for op,callback in (
    (Ops.ADD, _fold_relu_cap), (Ops.MUL, _replace_infinite_multiply), (Ops.FDIV, _preserve_infinite_division_sign))],
  (UPat(Ops.CAST, dtypes.half, name="root", src=(UPat.cvar("c"),)), lambda root,c: root.const_like(c.arg)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),])
def _square_offset(radicand:UOp, source:UOp, values:tuple[float, ...]) -> float|None:
  if radicand.op is not Ops.ADD or not any(u.op is Ops.MUL and len(u.src) == 2 and all(x.key == source.key for x in u.src)
                                             for u in radicand.src): return None
  return next((float(u.arg) for u in radicand.src if u.op is Ops.CONST and float(u.arg) in values), None)

def _fold_atan(root:UOp) -> UOp|None:
  """Replace Tinygrad's asin-based atan with a compact range-reduced DPU polynomial."""
  nodes = root.toposort()
  sources:dict[bytes, UOp] = {}
  for u in nodes:
    # Match Tinygrad's x/sqrt(1+x*x) normalization.
    candidates = ((u.src[0],u.src[1]),) if u.op is Ops.FDIV else tuple((source,inverse.src[0]) for source,inverse in
      (u.src,u.src[::-1]) if inverse.op is Ops.RECIPROCAL and len(inverse.src) == 1) if u.op is Ops.MUL else ()
    if (source:=next((source for source,denominator in candidates if denominator.op is Ops.SQRT and len(denominator.src) == 1 and
                     _square_offset(denominator.src[0],source,(1.0,)) is not None),None)) is not None: sources[source.key] = source
  constants = {float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype.scalar() in (dtypes.half, dtypes.float)}
  if len(sources) != 1 or any(not any(abs(v-target) < tolerance for v in constants) for target,tolerance in
                              ((math.pi/2, 1e-12), (1.570796305, 1e-10), (-0.0012624911, 1e-8))): return None
  source, one = next(iter(sources.values())).cast(dtypes.half), UOp.const(1.0, dtypes.half)
  magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS)
  reduced = UOp(Ops.MAX, magnitude.dtype, src=(magnitude, one.alu(Ops.FDIV, magnitude)), arg=_NATIVE_MIN)
  tail = one.alu(Ops.SUB, reduced).alu(Ops.MUL, _half(0.2447).alu(Ops.ADD, reduced.alu(Ops.MUL, _half(0.0663))))
  angle = reduced.alu(Ops.MUL, _half(math.pi/4).alu(Ops.ADD, tail))
  large, reflected = _finite_positive_mask(magnitude.alu(Ops.SUB, one)), UOp.const(math.pi/2, dtypes.half).alu(Ops.SUB, angle)
  selected = angle.alu(Ops.ADD, large.alu(Ops.MUL, reflected.alu(Ops.SUB, angle)))
  return selected.alu(Ops.MUL, source.alu(Ops.FDIV, magnitude.alu(Ops.MAX, UOp.const(2**-24, dtypes.half))))

def _poly_horner(source:UOp, coefficients:tuple[float, ...]) -> UOp:
  return functools.reduce(lambda r,c: r.alu(Ops.MUL, source).alu(Ops.ADD, source.const_like(c)),
                          coefficients[-2::-1], source.const_like(coefficients[-1]))

def _fold_inverse_hyperbolic(root:UOp) -> UOp|None:
  """Stabilize Tinygrad's FP16 asinh/acosh expansions without LUT or CMAC."""
  if root.op is not Ops.MUL: return None
  # Match log(x + sqrt(x*x +/- 1)) after natural log expands to LOG2 times ln(2).
  matched = next(((source,int(offset)) for logarithm,scale in (root.src,root.src[::-1]) if scale.op is Ops.CONST and
    not abs(float(scale.arg)-math.log(2)) > 1e-12 and logarithm.op is Ops.LOG2 and len(logarithm.src) == 1 and
    (argument:=logarithm.src[0]).op is Ops.ADD for source,radical in (argument.src,argument.src[::-1]) if
    radical.op is Ops.SQRT and len(radical.src) == 1 and (offset:=_square_offset(radical.src[0],source,(-1.0,1.0))) is not None),None)
  if matched is None: return None
  source, offset = matched; source = source.cast(dtypes.half); one, ln2 = UOp.const(1.0, dtypes.half), UOp.const(math.log(2), dtypes.half)
  asinh, threshold = offset == 1, _half(1.5 if offset == 1 else 2.0); magnitude = UOp(Ops.MAX, dtypes.half, src=(source, source), arg=_NATIVE_ABS) if asinh else source  # noqa: E501
  bounded = _native_min(magnitude, threshold) if asinh else _native_min(source, threshold).alu(Ops.MAX, _half(-2.0)); square = bounded.alu(Ops.MUL, bounded)  # noqa: E501
  small = bounded.alu(Ops.MUL, _poly_horner(square, (0.99989513, -0.16376462, 0.06135906, -0.01879756, 0.00268578))) if asinh else bounded.alu(Ops.ADD, square.alu(Ops.SUB, one).sqrt()).alu(Ops.LOG2).alu(Ops.MUL, ln2)  # noqa: E501
  safe = (magnitude if asinh else source).alu(Ops.MAX,threshold)
  # Approximate log(2*x) plus an inverse-even-power correction for large positive x.
  inverse = one.alu(Ops.FDIV,safe); inverse_square = inverse.alu(Ops.MUL,inverse)
  correction = inverse_square.alu(Ops.MUL,_poly_horner(inverse_square,(0.25,-3/32,5/96) if asinh else (-0.25,-3/32,-5/96)))
  large = safe.alu(Ops.LOG2).alu(Ops.MUL,ln2).alu(Ops.ADD,ln2).alu(Ops.ADD,correction)
  selected = small.alu(Ops.ADD,_finite_positive_mask((magnitude if asinh else source).alu(Ops.SUB,threshold)).alu(Ops.MUL,large.alu(Ops.SUB,small)))  # noqa: E501
  return selected.alu(Ops.MUL,source.alu(Ops.FDIV,magnitude.alu(Ops.MAX,_half(2**-24)))) if asinh else selected

def _dpu_math_base(source:UOp) -> tuple[UOp, UOp, UOp, Callable[[UOp], UOp]]:
  source, zero, one = source.cast(dtypes.half), _half(0.0), _half(1.0)
  return source, zero, one, _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask

def _dpu_sqrt(source:UOp) -> UOp|None:
  """Approximate FP16 sqrt with range-independent Babylonian iterations on DPU EW."""
  if any(_local_load(u) is not None for u in source.toposort()): return None
  source, zero, one, _ = _dpu_math_base(source); finite = UOp(Ops.MAX, source.dtype, src=(source.alu(Ops.MAX, zero), UOp.const(65504.0, dtypes.half)), arg=_NATIVE_MIN)  # noqa: E501
  safe = finite.alu(Ops.MAX, UOp.const(2**-24, dtypes.half))
  estimate = safe.alu(Ops.MAX, one)
  for _ in range(14): estimate = estimate.alu(Ops.ADD, safe.alu(Ops.FDIV, estimate)).alu(Ops.MUL, UOp.const(0.5, dtypes.half))
  valid = one.alu(Ops.SUB, _positive_mask(zero.alu(Ops.SUB, source))); return source.alu(Ops.FDIV, estimate).alu(Ops.ADD, valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one))  # noqa: E501

def _dpu_periodic_reduce(source:UOp, reciprocal_period:float, split:tuple[float, ...], half_period:float) -> tuple[UOp, UOp, UOp]:
  """Reduce a finite FP16 angle with split constants so large products do not erase the residual."""
  half, one = dtypes.half, UOp.const(1.0, dtypes.half); bounded = UOp(Ops.MAX, half, src=(source.cast(half).alu(Ops.MAX, UOp.const(-10000.0, half)), UOp.const(10000.0, half)), arg=_NATIVE_MIN)  # noqa: E501
  quotient = bounded.alu(Ops.MUL, UOp.const(reciprocal_period, half)); magnitude = UOp(Ops.MAX, half, src=(quotient, quotient), arg=_NATIVE_ABS)
  multiple = _native_same(magnitude.alu(Ops.ADD, _half(0.5)), _NATIVE_FLOOR).alu(
    Ops.MUL, _positive_mask(quotient).alu(Ops.MUL, _half(2.0)).alu(Ops.SUB, one))
  reduced = functools.reduce(lambda value,coefficient: value.alu(Ops.SUB, multiple.alu(Ops.MUL, UOp.const(coefficient, dtypes.half))), split, bounded)
  # The rounded FP16 quotient can be a few periods off at large magnitudes. Normalize the small residual instead.
  for _ in range(3):
    correction = _positive_mask(reduced.alu(Ops.SUB,_half(half_period))).alu(Ops.SUB,_positive_mask(_half(-half_period).alu(Ops.SUB,reduced))); multiple = multiple.alu(Ops.ADD, correction)  # noqa: E501
    for coefficient in split: reduced = reduced.alu(Ops.SUB, correction.alu(Ops.MUL, UOp.const(coefficient, half)))
  return bounded, multiple, reduced

def _dpu_sin(source:UOp) -> UOp:
  """Approximate FP16 SIN without LUTs using Cody-Waite reduction and an odd polynomial."""
  half, one = dtypes.half, UOp.const(1.0, dtypes.half); period_split = (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125)
  if source.dtype.scalar() is dtypes.float:
    terms:list[UOp] = []; residuals:list[UOp] = []
    for u in _iter_binary(source, Ops.ADD, dtypes.float):
      if u.op is Ops.CONST:
        high = struct.unpack("<e", struct.pack("<e", float(u.arg)))[0]; terms.append(UOp.const(high, half))
        if (low:=float(u.arg)-high) != 0.0: residuals.append(UOp.const(low, half))
      else: terms.append(_fp32_expr_to_half(u))
    reduced_parts = [_precise_add_parts([bounded, *(multiple.alu(Ops.MUL, UOp.const(-coefficient, half)) for coefficient in period_split)])
      for term in terms for bounded, multiple, _ in (_dpu_periodic_reduce(term, 1/(2*math.pi), period_split, math.pi),)]
    residuals.extend(part[1] for part in reduced_parts); reduced, addition_residual = _precise_sum_parts([part[0] for part in reduced_parts])
    residuals.append(addition_residual)
    for _ in range(3):
      correction = _positive_mask(reduced.alu(Ops.SUB,_half(math.pi))).alu(Ops.SUB,_positive_mask(_half(-math.pi).alu(Ops.SUB,reduced))); reduced, normalization_residual = _precise_add_parts([reduced, *(correction.alu(Ops.MUL, UOp.const(-coefficient, half)) for coefficient in period_split)])  # noqa: E501
      residuals.append(normalization_residual)
    invalid=functools.reduce(lambda v,t:v.alu(Ops.ADD,t.alu(Ops.MUL,UOp.const(0.0,half))),terms[1:],terms[0].alu(Ops.MUL,UOp.const(0.0,half)))
  else:
    source = source.cast(half); _, _, reduced = _dpu_periodic_reduce(source, 1/(2*math.pi), period_split, math.pi)
    invalid = source.alu(Ops.MUL, UOp.const(0.0, half))
  magnitude = UOp(Ops.MAX, half, src=(reduced, reduced), arg=_NATIVE_ABS); reflected = _positive_mask(magnitude.alu(Ops.SUB, UOp.const(math.pi/2, half)))  # noqa: E501
  pi_minus = _half(3.0).alu(Ops.SUB, magnitude).alu(Ops.ADD, _half(0.140625)).alu(Ops.ADD, _half(math.pi-3.140625)); angle = _mask_mul(magnitude, one.alu(Ops.SUB, reflected)).alu(Ops.ADD, _mask_mul(pi_minus, reflected))  # noqa: E501
  square = angle.alu(Ops.MUL, angle); sign = one.alu(Ops.SUB, _positive_mask(UOp.const(0.0, half).alu(Ops.SUB, reduced)).alu(Ops.MUL, UOp.const(2.0, half)))  # noqa: E501
  result = angle.alu(Ops.MUL, _poly_horner(square, (1.0, -1/6, 1/120, -1/5040, 1/362880))).alu(Ops.MUL, sign)
  if source.dtype.scalar() is dtypes.float and residuals:
    residual = functools.reduce(lambda value,term: value.alu(Ops.ADD, term), residuals[1:], residuals[0])
    cosine = _poly_horner(square, (1.0, -1/2, 1/24, -1/720, 1/40320)).alu(Ops.MUL, one.alu(Ops.SUB, reflected.alu(Ops.MUL, _half(2.0)))); result = result.alu(Ops.ADD, residual.alu(Ops.MUL, cosine))  # noqa: E501
  return result.alu(Ops.ADD, invalid)

def _dpu_exp2(source:UOp) -> UOp:
  """Approximate FP16 EXP2 without LUTs using native FLOOR, Horner arithmetic, and exact exponent scaling."""
  source, zero, one, mask_fn = _dpu_math_base(source)
  bounded = UOp(Ops.MAX, source.dtype, src=(source.alu(Ops.MAX, UOp.const(-24.0, dtypes.half)), UOp.const(15.9921875, dtypes.half)), arg=_NATIVE_MIN)
  integer = UOp(Ops.MAX, dtypes.half, src=(bounded, bounded), arg=_NATIVE_FLOOR)
  # Build `2**exponent` for the FP16 exponent range with exact native DPU arithmetic.
  scale,quotient = UOp.const(2**-24,dtypes.half),_native_min(integer.alu(Ops.ADD,_half(24.0)).alu(Ops.MAX,zero),_half(39.0))
  for factor,repeats in ((2.0,1),(4.0,1),(16.0,1),(256.0,1),(256.0,2),(256.0,4)):
    halved = UOp(Ops.MAX,dtypes.half,src=((half_floor:=quotient.alu(Ops.MUL,UOp.const(0.5,dtypes.half))),half_floor),arg=_NATIVE_FLOOR)
    bit = quotient.alu(Ops.SUB,halved.alu(Ops.MUL,UOp.const(2.0,dtypes.half)))
    for _ in range(repeats): scale = scale.alu(Ops.MUL,one.alu(Ops.ADD,bit.alu(Ops.MUL,UOp.const(factor-1.0,dtypes.half))))
    quotient = halved
  result = _poly_horner(bounded.alu(Ops.SUB,integer),(1,0.6931471806,0.2402265069,0.0555041087,0.0096181291,0.0013333558)).alu(Ops.MUL,scale)
  below, above = mask_fn(UOp.const(-24.0, dtypes.half).alu(Ops.SUB, source)), mask_fn(source.alu(Ops.SUB, UOp.const(15.9921875, dtypes.half)))
  finite = UOp(Ops.MUL, dtypes.half, src=(result, one.alu(Ops.SUB, below)), arg=_NATIVE_MASK_MUL)
  return finite.alu(Ops.ADD, one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one))

def _dpu_log2(source:UOp) -> UOp:
  """Approximate FP16 LOG2 without LUTs using threshold exponent extraction and an atanh polynomial."""
  source, zero, one, mask_fn = _dpu_math_base(source)
  mantissa = UOp(Ops.MAX, source.dtype, src=(source.alu(Ops.MAX, UOp.const(2**-24, dtypes.half)), UOp.const(65504.0, dtypes.half)), arg=_NATIVE_MIN)
  exponent = zero
  for upper,steps in ((True, ((256.0, 8.0), (16.0, 4.0), (4.0, 2.0), (2.0, 1.0))),
                      (False, ((256.0, 8.0),)*3+((16.0, 4.0), (4.0, 2.0), (2.0, 1.0)))):
    for factor,shift in steps:
      threshold = UOp.const(struct.unpack("<e", struct.pack("<H", _fp16_bits(factor)-1))[0] if upper else 2.0/factor, dtypes.half)
      mask = _finite_positive_mask(mantissa.alu(Ops.SUB, threshold) if upper else threshold.alu(Ops.SUB, mantissa))
      multiplier = one.alu(Ops.ADD, mask.alu(Ops.MUL, UOp.const(factor-1.0, dtypes.half)))
      mantissa = mantissa.alu(Ops.FDIV if upper else Ops.MUL, multiplier)
      exponent = exponent.alu(Ops.ADD if upper else Ops.SUB, mask.alu(Ops.MUL, UOp.const(shift, dtypes.half)))
  z = mantissa.alu(Ops.SUB, one).alu(Ops.FDIV, mantissa.alu(Ops.ADD, one))
  result = exponent.alu(Ops.ADD, z.alu(Ops.MUL, _poly_horner(z.alu(Ops.MUL, z), (1, 1/3, 1/5, 1/7, 1/9))).alu(Ops.MUL, _half(2/math.log(2))))
  nonzero = mask_fn(source).alu(Ops.MAX, mask_fn(zero.alu(Ops.SUB, source)))
  zero_correction, valid = UOp.const(-1.0, dtypes.half).alu(Ops.FDIV, nonzero).alu(Ops.ADD, one), one.alu(Ops.SUB, mask_fn(zero.alu(Ops.SUB, source)))
  negative_correction, above = valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one), mask_fn(source.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
  inf_correction = one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one)
  return result.alu(Ops.ADD, zero_correction).alu(Ops.ADD, negative_correction).alu(Ops.ADD, inf_correction)

_DPU_MATH = {Ops.SQRT:_dpu_sqrt, Ops.EXP2:_dpu_exp2, Ops.LOG2:_dpu_log2, Ops.SIN:_dpu_sin}
_pm_exp2_fallback = PatternMatcher([(UPat(Ops.EXP2, (dtypes.half, dtypes.float), src=(UPat.var("source"),)), lambda source:_dpu_exp2(source))])

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None,
                 Ops.FDIV: lambda: None, Ops.SQRT: lambda: None, Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_uop_program(uops)
    if image is None:
      sink = graph_rewrite(next(u for u in uops if u.op is Ops.SINK), _pm_exp2_fallback, name="rockchip exp2 fallback")
      image = _lower_uop_program(list(graph_rewrite(sink, _pm_storage_common, name="rockchip fallback storage").toposort()), recipes_ready=True)
    if image is None: raise RuntimeError("RKPLAN_REJECT:generic_uops " + repr([(i, u.op.name, str(u.dtype)) for i,u in enumerate(uops)]))
    return base64.b64encode(encode_image(image)).decode()

class RockchipBoolRenderer(RockchipRenderer):
  """Expose one 16-lane local bool tile that the renderer consumes as grouped DPU reduction work."""
  has_local, shared_max = True, 16
