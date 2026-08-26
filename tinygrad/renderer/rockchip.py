from __future__ import annotations
# ruff: noqa: E702
import base64, functools, heapq, io, itertools, marshal, math, os, struct, zlib
import numpy as np
from enum import IntEnum
from typing import Any, Callable, Iterable, Mapping, NamedTuple, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes, float_to_fp16
from tinygrad.helpers import ceildiv, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import AxisType, GroupOp, Ops, UOp, UPat, PatternMatcher, exec_alu, graph_rewrite
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak

RKIMAGE_MAGIC, RKIMAGE_VERSION, _RKIMAGE_U16_MAX = b"RKIM", 32, (1 << 16) - 1

class RKTarget(IntEnum): RK3588 = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1
class RKLayout(IntEnum):
  FP16 = 0; INT16 = 1; INT32 = 2
  def itemsize(self) -> int: return 4 if self is RKLayout.INT32 else 2
class RKExecutionClass(IntEnum): NATIVE = 0; HOST_ADDRESS = 1

class RKArg(NamedTuple): kind: RKBufferKind; index: int; addend: int = 0  # type: ignore[assignment]

class RKValue(NamedTuple):
  """Typed value in one physical FP16, INT16, or INT32 carrier."""
  arg: RKArg; dtype: DType; count: int; layout: RKLayout  # type: ignore[assignment]

class RKScratch(NamedTuple): size: int; alignment: int = 4096

class RKGather(NamedTuple):
  """Materialize an affine or fallback raw-lane index map."""
  src_index: int; dst_index: int; count: int; base: int = 0  # type: ignore[assignment]
  # Axes are (destination divisor, range limit, source stride); offsets provide the non-affine fallback.
  axes: tuple[tuple[int, int, int], ...] = (); offsets: tuple[int, ...] = (); fill_bits: int = 0
  # Compile-time values have no source argument; partial gathers preserve lanes populated by another gather.
  values: tuple[int, ...] = (); partial: bool = False
  # Scalar FP16 reductions use a destination stride of 32 for 64-byte spacing.
  dst_stride: int = 1; dst_addend: int = 0
  # A negative phase split uses RKImage.gather_after.
  dst_kind: RKBufferKind = RKBufferKind.SCRATCH; itemsize: int = 2; src_kind: RKBufferKind = RKBufferKind.ARG; after: int = -1

class RKHostAddress(NamedTuple):
  """Host-calculated raw-lane movement. It never owns numeric or reduction semantics."""
  src: RKArg; index: RKArg; dst: RKArg; count: int; src_count: int; dst_count: int  # type: ignore[assignment]
  itemsize: int = 2; index_itemsize: int = 4; fill_bits: int = 0; index_limit: int = 0; base: int = 0; index_scale: int = 1; lane_stride: int = 0

class RKEWOp(NamedTuple):
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int  # type: ignore[assignment]
  submit_barrier: bool = False; compare: bool = False; stateful: bool = False
  int32_output: bool = False; int32_input: bool = False; bool_output: bool = False; int16_output: bool = False; int16_input: bool = False

class RKCMAC(NamedTuple):
  """One fixed FP16 matrix contraction with an optional terminal BS ReLU; gathers own only its physical packing."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; m: int; n: int; k: int; out_fp16: bool = True; relu: bool = False

class RKImage(NamedTuple):
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
    return v._replace(src_kind=src.kind,src_index=src.index,dst_kind=(dst:=fn(RKArg(v.dst_kind,v.dst_index))).kind,dst_index=dst.index)
  def host(value:RKHostAddress) -> RKHostAddress: return value._replace(src=fn(value.src), index=fn(value.index), dst=fn(value.dst))
  return image._replace(gathers=tuple(map(gather, image.gathers)), mid_gathers=tuple(map(gather, image.mid_gathers)),
    post_gathers=tuple(map(gather, image.post_gathers)), ew_ops=tuple(op._replace(dst=fn(op.dst), lhs=fn(op.lhs), rhs=fn(op.rhs))
    for op in image.ew_ops), host_gathers=tuple(map(host, image.host_gathers)), host_scatters=tuple(map(host, image.host_scatters)),
    cmac=None if image.cmac is None else image.cmac._replace(dst=fn(image.cmac.dst), lhs=fn(image.cmac.lhs), rhs=fn(image.cmac.rhs)))

def _alias_image_args(image:RKImage, aliases:dict[int, RKArg]) -> RKImage:
  return _map_image_args(image, lambda arg:aliases[arg.index]._replace(addend=aliases[arg.index].addend+arg.addend) if
                         arg.kind is RKBufferKind.ARG and arg.index in aliases else arg)

def _reuse_linear_scratch(image:RKImage, constant_slots:dict[bytes, int]) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  def gather_args(gather:RKGather) -> tuple[RKArg, ...]:
    return (() if gather.values else (RKArg(gather.src_kind, gather.src_index),))+(RKArg(gather.dst_kind, gather.dst_index),)
  mid_by_point:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers: mid_by_point.setdefault(gather.after if gather.after >= 0 else image.gather_after, []).append(gather)
  schedule = [tuple(RKArg(RKBufferKind.SCRATCH, slot) for slot in constant_slots.values())] + [gather_args(gather) for gather in image.gathers]
  schedule += [(host.src, host.index, host.dst) for host in image.host_gathers] + ([] if image.cmac is None else [(image.cmac.lhs, image.cmac.rhs, image.cmac.dst)])  # noqa: E501
  for index,op in enumerate(image.ew_ops): schedule += [gather_args(gather) for gather in mid_by_point.get(index, ())] + [(op.lhs, op.rhs, op.dst)]
  schedule += [gather_args(gather) for gather in mid_by_point.get(len(image.ew_ops), ())] + [gather_args(gather) for gather in image.post_gathers]  # noqa: E501
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
  image = _map_image_args(image, lambda arg: arg._replace(index=remap[arg.index]) if arg.kind is RKBufferKind.SCRATCH else arg, map_value_src=False)
  by_slot:dict[int, bytes] = {}
  for bits,slot in constant_slots.items():
    if by_slot.setdefault(remap[slot], bits) != bits: raise ValueError("overlapping scratch constants")
  constants = b"" if not by_slot else b"".join(by_slot.get(slot, b"\0\0") for slot in range(max(by_slot)+1))
  return image._replace(scratch=tuple(physical), constants=constants)

class RKStage(NamedTuple): commands: tuple[int, ...]; relocs: tuple[tuple[int, RKArg], ...]

def _plain_image(value): return tuple(map(_plain_image,value)) if isinstance(value,tuple) else int(value) if isinstance(value,IntEnum) else value
def _fits(values:Iterable[int], bits:int=32, signed:bool=False) -> bool: return all(isinstance(x,int) and -(1<<(bits-1)) <= x < 1<<(bits-1) if signed else isinstance(x,int) and 0 <= x < 1<<bits for x in values)  # noqa: E501

def _validate_image(image:RKImage) -> None:
  gathers, hosts = image.gathers+image.mid_gathers+image.post_gathers, image.host_gathers+image.host_scatters
  if image.version != RKIMAGE_VERSION or image.target is not RKTarget.RK3588 or not isinstance(image.constants,bytes) or not _fits((len(image.ew_ops),len(image.constants),len(image.mid_gathers),len(image.post_gathers),image.gather_after)) or any(len(x) > _RKIMAGE_U16_MAX for x in (image.scratch,gathers,image.host_gathers,image.host_scatters)): raise ValueError("invalid RKImage header")  # noqa: E501
  if image.cmac is not None and hosts: raise ValueError("CMAC cannot mix with host addressing")
  if image.cmac is not None: _validate_cmac(image.cmac, image.scratch)
  if any(not _fits((s.size,s.alignment)) for s in image.scratch): raise ValueError("invalid RKScratch")
  if image.mid_gathers and (not 0 <= image.gather_after < len(image.ew_ops) or any(not 0 <= (g.after if g.after >= 0 else image.gather_after) <= len(image.ew_ops) for g in image.mid_gathers)): raise ValueError("invalid mid-gather split")  # noqa: E501
  if not image.mid_gathers and image.gather_after != 0: raise ValueError("invalid mid-gather split")
  if any(g.itemsize not in (1,2,4) or not _fits((g.src_index,g.dst_index),16) or not _fits((g.count,g.fill_bits,g.dst_stride)) or not _fits((g.base,g.dst_addend,g.after),signed=True) or g.dst_stride < 1 or g.dst_addend < 0 or len(g.axes) > 255 or bool(g.values)+bool(g.offsets)+bool(g.axes) > 1 or g.partial and not g.offsets or g.values and (len(g.values) != g.count or not _fits(g.values,g.itemsize*8)) or g.offsets and (len(g.offsets) != g.count or not _fits(g.offsets,signed=True)) or any(not _fits(axis[:2]) or not _fits(axis[2:],signed=True) for axis in g.axes) for g in gathers): raise ValueError("invalid RKGather")  # noqa: E501
  if any(h.itemsize not in (1,2,4) or h.index_itemsize not in (2,4) or not _fits((h.src.index,h.index.index,h.dst.index),16) or not _fits((h.count,h.src_count,h.dst_count,h.fill_bits,h.index_limit)) or not _fits((h.src.addend,h.index.addend,h.dst.addend,h.base,h.index_scale,h.lane_stride),signed=True) for h in hosts): raise ValueError("invalid RKHostAddress")  # noqa: E501
  for op in image.ew_ops:
    int16_to_int32 = op.int16_input and op.int32_output and not op.int16_output and not op.int32_input
    if not _fits((op.dst.index,),16) or not _fits((op.lhs.index,op.rhs.index,op.count,op.ew_cfg)) or not _fits((op.dst.addend,op.lhs.addend,op.rhs.addend),signed=True) or op.bool_output and not op.int32_output or (op.int16_output or op.int16_input) and (op.int32_output or op.int32_input) and not int16_to_int32: raise ValueError("invalid RKEWOp flags")  # noqa: E501

def encode_image(image:RKImage) -> bytes:
  _validate_image(image); return RKIMAGE_MAGIC+struct.pack("<H", image.version)+zlib.compress(marshal.dumps(_plain_image(image), 4), 1)

def decode_image(blob:bytes) -> RKImage:
  def arg(x): return RKArg(RKBufferKind(x[0]), *x[1:])
  def gather(x): return RKGather(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],RKBufferKind(x[11]),x[12],RKBufferKind(x[13]),x[14])  # noqa: E501
  def host(x): return RKHostAddress(*(arg(v) for v in x[:3]), *x[3:])
  def ew(x): return RKEWOp(*(arg(v) for v in x[:3]), *x[3:])
  def cmac(x): return RKCMAC(*(arg(v) for v in x[:3]), *x[3:])
  try:
    if blob[:4] != RKIMAGE_MAGIC or struct.unpack_from("<H", blob, 4)[0] != RKIMAGE_VERSION: raise ValueError
    codec=zlib.decompressobj(); payload=codec.decompress(blob[6:])
    if codec.unused_data or not codec.eof: raise ValueError
    stream=io.BytesIO(payload); values=marshal.load(stream)
    if stream.tell() != len(payload): raise ValueError
    target,scratch,constants,version,gathers,ops,mid,gather_after,post,host_gathers,host_scatters,contract=values
    image=RKImage(RKTarget(target),tuple(RKScratch(*x) for x in scratch),constants,version,tuple(map(gather,gathers)),tuple(map(ew,ops)),
      tuple(map(gather,mid)),gather_after,tuple(map(gather,post)),tuple(map(host,host_gathers)),tuple(map(host,host_scatters)),None if contract is None else cmac(contract))  # noqa: E501
    _validate_image(image); return image
  except (EOFError, TypeError, ValueError, IndexError, KeyError, struct.error, zlib.error): raise ValueError("invalid RKImage") from None

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
  if not 0 < op.m <= 0x7ff or ai > 13*32 or ao > 0x3fff or op.m*ai*2 > 10*32768 or ai > 12*32 and op.m != 1: raise ValueError("CMAC shape out of range")  # noqa: E501
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
  special, native_int16, native_int32, c = (compare or stateful or int32_output or int32_input or int16_output or int16_input or
    fp32_output or fp32_input), int16_input and int16_output, int32_input and int32_output, compare
  int16_to_int32 = int16_input and int32_output and not int16_output and not int32_input
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if native_int32 else _EW_ELEMS_32BIT if \
    int32_output or int32_input or fp32_output or fp32_input else _MAX_EW_ELEMS_FP16
  if not 0 < count <= limit: raise ValueError(f"{'stateful EW' if special else 'EW fp16'} count {count} out of range")
  lanes, is_div = (4 if int32_input or fp32_input else 8), ew_cfg == _EW_CFG[Ops.FDIV]
  width, data_format = (count + lanes-1) // lanes - 1, next(format_ for flag,format_ in zip(
    (fp32_output, fp32_input, native_int16, native_int32, int16_to_int32, int32_output, int16_output, int32_input, True), _DPU_DATA_FORMATS) if flag)
  regs:tuple[tuple[int, int, int], ...] = ((D,R.REG_DPU_S_POINTER,0xe),(D,R.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (D,R.REG_DPU_DATA_FORMAT,data_format)) + (((D,R.REG_DPU_DST_SURF_STRIDE,1<<4),) if int16_to_int32 or fp32_output else ()) + (
    (D,R.REG_DPU_DATA_CUBE_WIDTH,width),(D,R.REG_DPU_DATA_CUBE_HEIGHT,0),(D,R.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (D,R.REG_DPU_DATA_CUBE_CHANNEL,0 if fp32_output and count == 1 else ((lanes-1)<<16)|(lanes-1)))
  if special:
    # Keep the compare phase lazy so invalid ordinary EW configs do not inspect it.
    pipeline = (((D,R.REG_DPU_BS_CFG,_BS_BN_BYPASS),(D,R.REG_DPU_BN_CFG,_BS_BN_BYPASS),(D,R.REG_DPU_BS_ALU_CFG,0),(D,R.REG_DPU_BS_MUL_CFG,0),
      (D,R.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR if int16_to_int32 or fp32_output and count == 1 else 2),
      (D,R.REG_DPU_WDMA_SIZE_0,0 if fp32_output and count == 1 else 3 if fp32_output else lanes-1),(D,R.REG_DPU_WDMA_SIZE_1,width),
      (D,R.REG_DPU_BN_MUL_CFG,0),(D,R.REG_DPU_BN_RELUX_CMP_VALUE,0))
      + (((D,R.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(D,R.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),(D,R.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),
      (D,R.REG_DPU_BN_CFG,_BN_CFG_COMPARE),(D,R.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),(D,R.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE)) if c else ())
      + ((D,R.REG_DPU_EW_CFG,_EW_CFG_COMMON|1 if compare else (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else \
      ew_cfg & ~_EW_OP_CVT_BYPASS if native_int16 or int16_to_int32 else ew_cfg),
      (D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),
      (D,R.REG_DPU_OUT_CVT_SCALE,0 if fp32_output else 1 if int32_output or int16_output or is_div else (1<<16)|1),
      (D,R.REG_DPU_OUT_CVT_SHIFT,0),(D,R.REG_DPU_SURFACE_ADD,(2 if native_int16 or int16_to_int32 else 4)<<4)))
  else:
    pipeline = ((D,R.REG_DPU_EW_CFG,ew_cfg),) + (((D,R.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6),) if ew_cfg == _EW_CFG_RELU6 else ()) + (
      ((D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),(D,R.REG_DPU_OUT_CVT_SHIFT,0),
       (D,R.REG_DPU_SURFACE_ADD,1<<6)) if is_div else ()) + ((D,R.REG_DPU_OUT_CVT_SCALE,1 if is_div else (1<<16)|1),)
  regs += pipeline + ((_RDMA,R.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
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
_STATIC_SCALAR_ALU = _STATIC_OPS - {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST} | \
  {Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.AND, Ops.OR, Ops.XOR}

def _eval_expr(u:UOp, env:Mapping[UOp, int|float|bool|np.ndarray], cache:dict[UOp, int|float|bool|np.ndarray], vector:bool=False) -> int|float|bool|np.ndarray:  # noqa: E501
  if u in cache: return cache[u]
  if u.op is Ops.CONST: value = _static_cast(u.arg, u.dtype, vector)
  elif u.op in (Ops.RANGE, Ops.SPECIAL): value = _static_cast(env[u], u.dtype, True) if vector else env[u]
  elif u.op is Ops.PARAM: raise RuntimeError("RKPLAN_REJECT:dynamic_static_expr")
  elif u.op is Ops.AFTER: value = _eval_expr(u.src[0], env, cache, vector)
  else:
    values = tuple(_eval_expr(src, env, cache, vector) for src in u.src)
    if u.op is Ops.CAST: value = values[0]
    elif not vector and u.op not in _STATIC_SCALAR_ALU: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
    else:
      try: value = np.frompyfunc(lambda *x:exec_alu(u.op,u.dtype,x,False),len(values),1)(*values) if vector else exec_alu(u.op,u.dtype,values,False)
      except KeyError: raise RuntimeError(f"RKPLAN_REJECT:unsupported_static {u.op.name}")
    value = _static_cast(value, u.dtype, vector)
  return cache.setdefault(u, value)

@functools.lru_cache(maxsize=4096)
def _is_static_expr(u:UOp) -> bool: return u.op in (Ops.RANGE,Ops.SPECIAL) or u.op in _STATIC_OPS and all(_is_static_expr(x) for x in u.src)

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

def _admit(o,d,v=True)->RKOutput|None: return o if v and o is not None and o[1].dtype.scalar() in (d if isinstance(d,tuple) else (d,)) else None
def _try(o,d,f,*a,v=True)->RKImage|None: return None if not v or (o:=_admit(o,d)) is None else f(o,*a)

def _iter_range_env(ranges:list[UOp], max_envs:int|None=_MAX_STATIC_RANGE_ENVS, dependencies:bool=True) -> list[dict[UOp, int]]:
  if dependencies:
    def dependency_order(r:UOp)->tuple[UOp,...]: return (*itertools.chain.from_iterable(dependency_order(src) for src in r.src[1:] if src.op is Ops.RANGE),r)  # noqa: E501
    ranges=list(dict.fromkeys(node for root in ranges for node in dependency_order(root)))
  if any(r.src[0].op is not Ops.CONST for r in ranges): raise RuntimeError("RKPLAN_REJECT:unsupported_index")
  bounds=tuple(int(r.src[0].arg) for r in ranges)
  if any(bound<0 for bound in bounds) or max_envs is not None and math.prod(bounds)>max_envs: raise RuntimeError("RKPLAN_REJECT:static_index_budget")  # noqa: E501
  return [dict(zip(ranges, values)) for values in itertools.product(*(range(bound) for bound in bounds))]

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
  else: encoded = np.asarray(expr_lanes).astype(np.int64) & (0xffff if encode is _int16_bits else -1) if encode in (_int16_bits,int) else \
    np.fromiter((encode(value.item()) for value in expr_lanes),dtype=np.int64,count=len(expr_lanes))
  dst,values=dst_lanes[order:=np.argsort(dst_lanes)],encoded[order]
  starts=np.r_[True,dst[1:]!=dst[:-1]]
  if not np.array_equal(dst[starts], np.arange(count)) or np.any(values[1:][~starts[1:]] != values[:-1][~starts[1:]]):
    raise RuntimeError("RKPLAN_REJECT:static_index")
  return tuple(values[starts].tolist())

def _linear_index(u:UOp, divided:bool=False) -> tuple[int, dict[UOp|tuple[UOp, int], int]]|None:
  """Represent static address arithmetic as a sum of scaled RANGE or RANGE//constant terms."""
  if divided and u.op is Ops.CAST and len(u.src) == 1 and u.dtype.scalar() in (dtypes.int,dtypes.uint): u=u.src[0]
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op in (Ops.RANGE, Ops.SPECIAL): return 0, {((u, 1) if divided else u):1}
  if divided and u.op is Ops.CDIV and len(u.src)==2 and u.src[0].op in (Ops.RANGE,Ops.SPECIAL) and u.src[1].op is Ops.CONST and int(u.src[1].arg)>0: return 0,{(u.src[0],int(u.src[1].arg)):1}  # noqa: E501
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _linear_index(u.src[0], divided), _linear_index(u.src[1], divided)
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {key:coefficient*scale for key,coefficient in affine[1].items()}
  sign=-1 if u.op is Ops.SUB else 1
  return lhs[0]+sign*rhs[0],{key:value for key in lhs[1].keys()|rhs[1].keys() if (value:=lhs[1].get(key,0)+sign*rhs[1].get(key,0))}  # noqa: E501

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> tuple[int, ...]:
  vector_env, dst = _static_vector_env(out_index, load_index, *((gate,) if gate is not None else ()), reject="gather_index")
  src=np.broadcast_to(_eval_expr(load_index,vector_env,cache:=typing_cast(dict[UOp,int|float|bool|np.ndarray],{}),vector=True),len(dst)).astype(np.int64)  # noqa: E501
  values = src if gate is None else np.where(active:=np.broadcast_to(_eval_expr(gate, vector_env, cache, True), len(dst)), src, -1)
  if np.any((src<0)&(gate is None or active)) or np.any((dst<0)|(dst>=count)): raise RuntimeError("RKPLAN_REJECT:gather_index")
  offsets = np.full(count, -2, dtype=np.int64); offsets[dst] = values
  if np.any(offsets == -2): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(int(x) for x in offsets)

def _affine_output_axes(affine:tuple[int, dict[UOp, int]], count:int) -> tuple[tuple[UOp, int, int], ...]|None:
  ordered = tuple(sorted(affine[1].items(), key=lambda item:item[1]))
  limits = tuple(int(r.src[0].arg) if r.src and r.src[0].op is Ops.CONST else 0 for r,_ in ordered)
  valid=all(limit>0 and stride==math.prod(limits[:i]) for i,((_,stride),limit) in enumerate(zip(ordered,limits)))
  return tuple((r,stride,limit) for (r,stride),limit in zip(ordered,limits)) if valid and math.prod(limits)==count else None

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0) -> RKGather:
  if gate is None and (out_affine:=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index))) is not None and out_affine[0]==0 and (output_axes:=_affine_output_axes(out_affine,count)) is not None:  # noqa: E501
    if (load_divided:=typing_cast(tuple[int, dict[tuple[UOp, int], int]]|None, _linear_index(load_index, True))) is not None and \
       all(r in out_affine[1] and divisor <= int(r.src[0].arg) for r,divisor in load_divided[1]):
      # Preserve the ordinary affine axis order and object graph; true divided plans retain expression order.
      return RKGather(src_index,dst_index,count,load_divided[0],tuple((d,l,load_affine[1][r]) for r,d,l in output_axes if load_affine[1].get(r,0)) if (load_affine:=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(load_index))) is not None else  # noqa: E501
        tuple((out_affine[1][r]*divisor,(int(r.src[0].arg)+divisor-1)//divisor,stride) for (r,divisor),stride in load_divided[1].items() if stride))  # noqa: E501
  return RKGather(src_index, dst_index, count, offsets=_gather_offsets(out_index, load_index, gate, count), fill_bits=fill_bits)

def _validate_gather_bounds(plan:RKGather, source_count:int) -> None:
  low,high=(min(plan.offsets,default=0),max(plan.offsets,default=-1)) if plan.offsets else tuple(plan.base+sum(fn((limit-1)*stride,0) for _,limit,stride in plan.axes) for fn in (min,max))  # noqa: E501
  if low < (0 if not plan.offsets else -1) or high >= source_count: raise RuntimeError("RKPLAN_REJECT:gather_index")

class RKTypedLoadPlan(NamedTuple):
  """Typed source metadata shared by static-offset and physical-gather consumers."""
  param:UOp; gather:RKGather

def _typed_load_plan(load:UOp, dtype:DType, out_index:UOp, count:int, *, fill_bits:int|None=None, require_offsets:bool=False) -> RKTypedLoadPlan|None:  # noqa: E501
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX: return None
  if (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None
  gate,fill_bits=load.src[2] if len(load.src)>2 else None,fill_bits if fill_bits is not None else _fp16_bits(load.src[1].arg if len(load.src)>1 else 0) if dtype is dtypes.half else 0  # noqa: E501
  try:
    gather=_gather_plan(param.arg.slot,0,out_index,load.src[0].src[1],gate,count,fill_bits)
    _validate_gather_bounds(gather,int(param.src[0].arg)); gather=gather._replace(base=0,axes=(),offsets=_gather_offsets(out_index,load.src[0].src[1],gate,count)) if require_offsets else gather  # noqa: E501
  except RuntimeError: return None
  return RKTypedLoadPlan(param, gather)

def _gather_cache_key(plans:Iterable[RKGather]) -> tuple: return tuple(v[0:1]+v[2:11]+v[12:14] for v in plans)

def _relu_operand(u:UOp) -> UOp|None:
  if u.op is Ops.WHERE and (folded:=_fold_ordered_where(u)) is not None: u = folded
  if u.op is not Ops.MAX or u.arg is not None or u.dtype.scalar() not in (dtypes.half,dtypes.float): return None
  if u.src[0].op is Ops.CONST and float(u.src[0].arg) == 0.0: return u.src[1]
  if u.src[1].op is Ops.CONST and float(u.src[1].arg) == 0.0: return u.src[0]
  return None


def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = x.alu(Ops.MUL, splitter)
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one); return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low, rhs_high, rhs_low = (*_split_half(term.src[0], neg_one, splitter), *_split_half(term.src[1], neg_one, splitter))
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one); error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low)).alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))  # noqa: E501
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  return total, _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one).alu(Ops.ADD, _sub_half(rhs, rhs_virtual, neg_one))

def _precise_add_parts(terms:tuple[UOp, ...]|list[UOp]) -> tuple[UOp, UOp]:
  """Recover FP16 addition residuals as a high lane plus a low correction lane."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  high, middle, low = terms[0], zero, zero
  for part in terms[1:]: high,error=_two_sum(high,part,neg_one); middle,error=_two_sum(middle,error,neg_one); low=low.alu(Ops.ADD,error)  # noqa: E501
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

def _physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp: return _tag_precise_adds(recipe.substitute(placeholders:={source:UOp.param(-index-1,source.dtype,()) for index,source in enumerate(opaque)})).substitute({placeholder:source for source,placeholder in placeholders.items()})  # noqa: E501

def _kahan_mul_sum(terms:list[UOp]) -> UOp:
  """Accumulate composite products and their TwoProduct residuals in their proven physical order."""
  neg_one,splitter=UOp.const(-1.0,dtypes.half),UOp.const(65.0,dtypes.half); pairs=tuple(_two_product(term,neg_one,splitter) for term in terms); active=tuple(x[0] for x in pairs)+tuple(x[1] for x in pairs); total,correction=active[0],UOp.const(0.0,dtypes.half)  # noqa: E501
  for value in active[1:]: adjusted=value.alu(Ops.SUB,correction); updated=total.alu(Ops.ADD,adjusted); correction=updated.alu(Ops.SUB,total).alu(Ops.SUB,adjusted); total=updated  # noqa: E501
  return _tag_precise_adds(total)

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  return _kahan_mul_sum(terms) if all(term.op is Ops.MUL and term.arg is None and term.dtype.scalar() is dtypes.half and any(_strip_cast(source).op is Ops.LOAD for source in term.src) for term in terms) and (len(terms) == 8 and all(all(_strip_cast(source).op is Ops.LOAD for source in term.src) for term in terms) or 64 <= len(terms) <= 512 and any(any(_strip_cast(source).op is not Ops.LOAD for source in term.src) for term in terms)) else _tag_precise_adds((parts:=_precise_sum_parts(terms))[0].alu(Ops.ADD,parts[1]))  # noqa: E501

def _append_inplace_image(first:RKImage, second:RKImage) -> RKImage|None:
  """Append an in-place EW image, scheduling its input materialization after the first image completes."""
  if not second.ew_ops or second.cmac is not None or second.host_gathers or second.host_scatters or \
     first.post_gathers and first.cmac is None: return None
  fc,sc,kind,fs = len(first.constants)//2, len(second.constants)//2, RKBufferKind.SCRATCH, len(first.scratch)
  first, second = _map_image_args(first, lambda arg: arg._replace(index=arg.index+sc) if arg.kind is kind and arg.index >= fc else arg), \
                   _map_image_args(second, lambda arg: arg._replace(index=fc+arg.index if arg.index<sc else fs+arg.index) if arg.kind is kind else arg)  # noqa: E501
  cmac_mid=tuple(gather._replace(after=0) for gather in first.post_gathers) if first.cmac is not None else ()
  second_ops=(second.ew_ops[0]._replace(submit_barrier=True,stateful=True),*second.ew_ops[1:]); second_mid=tuple(gather._replace(after=len(first.ew_ops)) for gather in second.gathers)+tuple(  # noqa: E501
    gather._replace(after=len(first.ew_ops)+(gather.after if gather.after >= 0 else second.gather_after)) for gather in second.mid_gathers)
  scratch=first.scratch[:fc]+second.scratch[:sc]+first.scratch[fc:]+second.scratch[sc:]; return RKImage(RKTarget.RK3588,scratch,first.constants+second.constants,  # noqa: E501
                 gathers=first.gathers, ew_ops=first.ew_ops+second_ops, mid_gathers=first.mid_gathers+cmac_mid+second_mid,
                 gather_after=first.gather_after, post_gathers=tuple(gather._replace(after=-1) for gather in second.post_gathers),cmac=first.cmac)

def _lower_cmac_storage_epilogue(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Commit one output-shaped FP32 contraction to HALF on CMAC before its ordinary HALF epilogue."""
  store,out,count,index,root=output
  fake_slot=1+max((u.arg.slot for u in uops if u.op is Ops.PARAM),default=out.arg.slot); fake=UOp.param(fake_slot,dtypes.half,(count,))
  for boundary in (u for u in root.toposort() if u is not root and _typed_cast_source(u,dtypes.half,dtypes.float) is not None):
    source=typing_cast(UOp,_typed_cast_source(boundary,dtypes.half,dtypes.float)); terms=tuple(_strip_cast(term) for term in _iter_binary(source,Ops.ADD)) if source.op is Ops.ADD else ()  # noqa: E501
    if len(terms) == 8 and all(term.op is Ops.MUL and term.arg is None and all(src.dtype.scalar() is dtypes.half and _strip_cast(src).op is Ops.LOAD for src in term.src) for term in terms): continue  # noqa: E501
    if (prefix:=_lower_cmac_reduction((store.replace(src=(store.src[0],boundary)),out,count,index,boundary),uops)) is None: continue
    suffix_store=store.replace(src=(store.src[0],root.substitute({boundary:fake.index(index).load()})))
    if any(_root_param(load.src[0]) is out for load in _semantic_loads(suffix_store)): continue
    suffix=_lower_uop_program(list(UOp(Ops.SINK,src=(suffix_store,)).toposort()),vectorize_reductions=False)
    if suffix is not None:
      target=RKArg(RKBufferKind.SCRATCH,len(suffix.scratch)); suffix=_alias_image_args(suffix,{fake_slot:target})._replace(scratch=suffix.scratch+(RKScratch(_scratch_bytes(count)),),gathers=(RKGather(out.arg.slot,target.index,count,axes=((1,count,1),)),*suffix.gathers))  # noqa: E501
    if suffix is not None and (combined:=_append_inplace_image(prefix,suffix)) is not None: return combined
  return None

def _iter_binary(root:UOp, op:Ops, dtype:DType|None=None, plain:bool=False) -> Iterable[UOp]:
  stack = [root]
  while stack:
    node = stack.pop()
    if node.op is op and (dtype is None or node.dtype.scalar() is dtype) and (not plain or node.arg is None): stack.extend(reversed(node.src))
    else: yield node

class _RKCMACShape(NamedTuple):
  diagonal:bool; m:int; n:int; lanes:tuple[int, ...]; outputs:tuple[int, ...]
  terms:tuple[tuple[RKTypedLoadPlan|None, RKTypedLoadPlan|None, float], ...]; ai:int; ao:int

def _lower_cmac_reduction(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Factor a real weighted sum/dot/matmul and pre-round linear terms into one fixed CMAC stage."""
  _,out,rows,out_index,root=output
  if rows <= 0 or out.dtype.scalar() not in (dtypes.half,dtypes.float): return None
  relu_root = _relu_operand(root)
  if relu_root is None and (fp32_root:=_typed_cast_source(root,dtypes.half,dtypes.float)) is not None: relu_root = _relu_operand(fp32_root)
  if relu_root is not None: root = relu_root
  graph=root.toposort(); local_loads=_semantic_loads(root,local=True) if any(u.op is Ops.BUFFER for u in graph) else ()
  local_add=bool(local_loads) and all(load.dtype.scalar() is dtypes.float for load in local_loads)
  additive=local_add or (_strip_cast(root).op is Ops.ADD and _strip_cast(root).dtype.scalar() is dtypes.float) or any(u.op is Ops.REDUCE and (u.arg is Ops.ADD or isinstance(u.arg,tuple) and u.arg and u.arg[0] is Ops.ADD) for u in graph)  # noqa: E501
  try: root = _unroll_static_reduces(_unroll_static_local(uops, root) if local_loads else root, precise=False)
  except (_RKGenericReject, RuntimeError, ValueError): return None
  scale,exact_scale = 1.0,True
  while (pair:=_const_operand(root:=_strip_cast(root),Ops.MUL)) is not None: root,factor=pair[0],float(pair[1].arg); scale*=factor; exact_scale=exact_scale and factor>0.0 and math.frexp(factor)[0]==0.5 and float_to_fp16(scale)==scale  # noqa: E501
  terms=tuple(_strip_cast(term) for term in _iter_binary(root,Ops.ADD)) if root.op is Ops.ADD else (_strip_cast(root),) if additive else ()
  terms=tuple(term for term in terms if not (term.op is Ops.CONST and float(term.arg)==0.0)); groups=len(terms)
  if local_loads and not local_add or groups < (1 if additive else 4) or groups > _MAX_GENERIC_UNROLL: return None
  # Admit each linear term directly into its typed, bounded gather proof; no parsed/aligned mirror records survive this owner.
  parsed:list[tuple[tuple[RKTypedLoadPlan,...],float]]=[]; cells=0
  for term in terms:
    factors=tuple(map(_strip_cast,_iter_binary(_strip_cast(term),Ops.MUL,plain=True)))
    constants=tuple(node for node in factors if node.op is Ops.CONST); loads=tuple(node for node in factors if node.op is not Ops.CONST)
    weight=scale*math.prod(float(node.arg) for node in constants)
    oversized=cells+rows*len(loads)>_MAX_DYNAMIC_SELECTOR_CELLS
    affine=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index)) if oversized else None
    invalid_factor=len(constants)>2 or len(constants)>1 and term.dtype.scalar() is not dtypes.float or len(loads)>2 or not exact_scale or any(float_to_fp16(float(node.arg))!=float(node.arg) for node in constants)  # noqa: E501
    invalid_weight=not math.isfinite(weight) or len(loads)<2 and float_to_fp16(weight)!=weight or len(loads)==2 and weight!=1.0
    invalid_load=any(len(load.src)>1 and (load.src[1].op is not Ops.CONST or float(load.src[1].arg)!=0.0 or math.copysign(1.0,float(load.src[1].arg))<0.0) for load in loads)  # noqa: E501
    invalid_large=oversized and (affine is None or affine[0]!=0 or _affine_output_axes(affine,rows) is None or any(load.op is not Ops.LOAD or len(load.src)>2 or not load.src or load.src[0].op is not Ops.INDEX or (load_affine:=_linear_index(load.src[0].src[1])) is None or any(axis not in affine[1] for axis in load_affine[1]) for load in loads))  # noqa: E501
    if invalid_factor or invalid_weight or invalid_load or invalid_large: return None
    plans=tuple(_typed_load_plan(load,dtypes.half,out_index,rows) for load in loads)
    if any(plan is None for plan in plans): return None
    parsed.append((typing_cast(tuple[RKTypedLoadPlan,...],plans),weight)); cells+=rows*len(loads)
  def offset(plan:RKTypedLoadPlan,lane:int)->int:
    gather=plan.gather
    return gather.offsets[lane] if gather.offsets else gather.base+sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes)  # noqa: E501
  # Align a physical view only when every operand is invariant across its assigned CMAC row or column.
  def align(m:int,n:int,lanes:tuple[int,...]=())->tuple[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float],...]:
    aligned:list[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float]]=[]
    for operands,weight in parsed:
      if not operands: aligned.append((None,None,weight)); continue
      def lane(i:int)->int: return lanes[i] if lanes else i
      row,col=zip(*(((not lanes and not plan.gather.offsets and all(limit==1 or divisor%n==0 for divisor,limit,_ in plan.gather.axes),not lanes and not plan.gather.offsets and all(limit==1 or n%divisor==0 and n//divisor%limit==0 for divisor,limit,_ in plan.gather.axes)) if cells>_MAX_DYNAMIC_SELECTOR_CELLS else (all(offset(plan,lane(i*n+j))==offset(plan,lane(i*n)) for i in range(m) for j in range(n)),all(offset(plan,lane(i*n+j))==offset(plan,lane(j)) for i in range(m) for j in range(n)))) for plan in operands))  # noqa: E501
      single=(0,None) if row[0] else (None,0) if col[0] else None
      order=single if len(operands)==1 else (0,1) if row[0] and col[1] else (1,0) if row[1] and col[0] else None
      if order is None: return ()
      aligned.append((None if order[0] is None else operands[order[0]],None if order[1] is None else operands[order[1]],weight))
    return tuple(aligned)
  # Enumerate ordinary, affine-permuted, and diagonal views, then retain the established resource score.
  views:list[tuple[bool,int,int,tuple[int,...],tuple[int,...],tuple[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float],...]]]=[(False,rows//n,n,(),(),align(rows//n,n)) for n in range(1,rows+1) if rows%n==0]  # noqa: E501
  for _,stride,limit in (_affine_output_axes(affine,rows) if (affine:=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index))) is not None else None) or ():  # noqa: E501
    m,n=limit,rows//limit
    lanes=tuple(high*stride*limit+row*stride+low for row in range(limit) for high in range(rows//stride//limit) for low in range(stride))
    outputs=tuple((i//stride%limit)*n+i//(stride*limit)*stride+i%stride for i in range(rows))
    views.append((False,m,n,lanes,outputs,align(m,n,lanes)))
  diagonal=tuple((operands[0] if operands else None,operands[1] if len(operands)==2 else None,weight) for operands,weight in parsed)
  views.append((True,rows,rows,(),tuple(i*rows+i for i in range(rows)),diagonal))
  candidates=[_RKCMACShape(diagonal,m,n,lanes,outputs,normalized,ai,ao) for diagonal,m,n,lanes,outputs,normalized in views for ai,ao,_ in (_cmac_layout(n,groups),) if m<=0x7ff and ai<=13*32 and ao<=0x3fff and m*ai*2<=10*32768 and (m==1 or ai<=12*32) and normalized and (out.dtype.scalar() is not dtypes.float or m!=1 or n!=1 or any(lhs is None or rhs is not None or weight!=1.0 for lhs,rhs,weight in normalized))]  # noqa: E501
  if not candidates: return None
  has_pair=any(len(operands)==2 for operands,_ in parsed)
  shape=min(candidates,key=lambda item:(item.diagonal,not has_pair and item.n!=1,bool(item.lanes),item.m*item.ai+item.ao*item.ai+2*item.m*item.ao,-item.n))  # noqa: E501
  lhs0,rhs0,_=shape.terms[0]
  # Dense affine surfaces remain symbolic; irregular and weighted surfaces are materialized as bounded raw gathers.
  dense=not shape.diagonal and not shape.lanes and not shape.outputs and shape.ai==groups and shape.ao==shape.n and lhs0 is not None and rhs0 is not None and all(lhs is not None and rhs is not None and weight==1.0 and lhs.param is lhs0.param and rhs.param is rhs0.param and not lhs.gather.offsets and lhs.gather.base==lhs0.gather.base+k and lhs.gather.axes==((shape.n,shape.m,groups),) and not rhs.gather.offsets and rhs.gather.base==rhs0.gather.base+k*shape.n and rhs.gather.axes==((1,shape.n,1),) for k,(lhs,rhs,weight) in enumerate(shape.terms))  # noqa: E501
  fp16=out.dtype.scalar() is dtypes.half
  if dense and lhs0 is not None and rhs0 is not None:
    gathers=[RKGather(lhs0.param.arg.slot,0,shape.m*groups,base=lhs0.gather.base,axes=((1,shape.m*groups,1),)),RKGather(rhs0.param.arg.slot,1,shape.n*groups,base=rhs0.gather.base,axes=((groups*16,shape.n//16,16),(512,groups//32,32*shape.n),(32,16,1),(1,32,shape.n)))]  # noqa: E501
  else:
    a=tuple(((source.param.arg.slot,offset(source,shape.lanes[row if shape.diagonal else row*shape.n] if shape.lanes else row if shape.diagonal else row*shape.n)) if (source:=shape.terms[k][0]) is not None else (None,_fp16_bits(1.0 if shape.terms[k][1] is None else shape.terms[k][2]))) if k<groups else (None,0) for row in range(shape.m) for k in range(shape.ai))  # noqa: E501
    b=tuple(((source.param.arg.slot,offset(source,shape.lanes[ob*16+ni] if shape.lanes else ob*16+ni)) if (source:=shape.terms[k][1]) is not None else (None,_fp16_bits(shape.terms[k][2]))) if ob*16+ni<shape.n and (k:=ib*32+ki)<groups else (None,0) for ob in range(shape.ao//16) for ib in range(shape.ai//32) for ni in range(16) for ki in range(32))  # noqa: E501
    gathers=[]
    for dst,packed in enumerate((a,b)):
      sources=tuple(dict.fromkeys(source for source,_ in packed if source is not None))
      values=tuple(value if source is None else 0 for source,value in packed); seeded=not sources or any(values)
      if seeded: gathers.append(RKGather(out.arg.slot,dst,len(packed),values=values))
      gathers.extend(RKGather(source,dst,len(packed),offsets=tuple(value if owner==source else -1 for owner,value in packed),
        partial=seeded or bool(i)) for i,source in enumerate(sources))
  if sum(gather.count for gather in gathers)+rows>_MAX_DYNAMIC_SELECTOR_CELLS: return None
  output_axes=((shape.n,shape.m,shape.ao*2),(16,shape.n//16,32),(1,16,1)) if dense and fp16 else ((1,rows,1),) if dense else ()
  output_offsets=() if dense else tuple(row*shape.ao*(2 if fp16 else 1)+(col//16*32+col%16 if fp16 else col) for i in (shape.outputs or range(rows)) for row,col in (divmod(i,shape.n),))  # noqa: E501
  return RKImage(RKTarget.RK3588,(RKScratch(shape.m*shape.ai*2),RKScratch(shape.ao*shape.ai*2),RKScratch(shape.m*shape.ao*4)),gathers=tuple(gathers),post_gathers=(RKGather(2,out.arg.slot,rows,axes=output_axes,offsets=output_offsets,dst_kind=RKBufferKind.ARG,itemsize=2 if fp16 else 4,src_kind=RKBufferKind.SCRATCH),),cmac=RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),shape.m,shape.n,groups,fp16,relu_root is not None))  # noqa: E501

def _reduce_rows(ops:list[RKEWOp], active:list[RKArg], count:int, cfg:int, int16:bool=False) -> RKArg:
  """Append a balanced row reduction, making its first dependent stage self-contained."""
  first = not int16
  while len(active) > 1:
    for lhs,rhs in zip(active[::2], active[1::2]):
      ops.append(RKEWOp(lhs, lhs, rhs, count, cfg, submit_barrier=first, stateful=first, int16_input=int16, int16_output=int16)); first = False  # noqa: E501
    active = active[::2]
  return active[0]

def _i16_min(lhs:UOp, rhs:UOp) -> UOp: return UOp(Ops.MAX,dtypes.int16,src=(lhs,rhs),arg=_NATIVE_MIN)
def _i16_abs(value:UOp) -> UOp: return UOp(Ops.MAX,dtypes.int16,src=(value,value),arg=_NATIVE_ABS)
def _i16_bit(value:UOp) -> UOp: return _i16_min(value.alu(Ops.MAX,value.const_like(0)),value.const_like(1))
def _i16_equal(lhs:UOp, rhs:UOp) -> UOp: return lhs.const_like(1).alu(Ops.SUB,_i16_min(_i16_abs(lhs.alu(Ops.SUB,rhs)),lhs.const_like(1)))
def _sign_bias(value:UOp) -> UOp: return value.alu(Ops.ADD,value.const_like(128)).alu(Ops.SUB,_i16_bit(value.alu(Ops.SUB,value.const_like(127))).alu(Ops.MUL,value.const_like(256)))  # noqa: E501

def _i16_compare(op:Ops, lhs:UOp, rhs:UOp) -> UOp:
  delta=(rhs if op is Ops.CMPLT else lhs).alu(Ops.SUB,lhs if op is Ops.CMPLT else rhs)
  result=_i16_bit(delta if op is Ops.CMPLT else _i16_abs(delta))
  return result.const_like(1).alu(Ops.SUB,result) if op is Ops.CMPEQ else result

def _i16_select(selector:UOp, yes:UOp, no:UOp) -> UOp:
  one=selector.const_like(1)
  return selector.alu(Ops.MUL,yes).alu(Ops.ADD,one.alu(Ops.SUB,selector).alu(Ops.MUL,no))

def _byte_bits(value:UOp) -> tuple[UOp, ...]:
  """Split one unsigned byte expression into exact least-significant-first INT16 bit planes."""
  result, remainder = typing_cast(list[UOp|None],[None]*8),value
  for bit in range(7,0,-1):
    result[bit]=flag=_i16_bit(remainder.alu(Ops.SUB,value.const_like((1<<bit)-1)))
    remainder=remainder.alu(Ops.SUB,flag.alu(Ops.MUL,value.const_like(1<<bit)))
  result[0]=remainder; return typing_cast(tuple[UOp,...],tuple(result))

def _ordered_bits(lhs:Iterable[UOp], rhs:Iterable[UOp]) -> UOp:
  """Compare equal-width unsigned components from most to least significant."""
  left=tuple(lhs); less,equal=left[0].const_like(0),left[0].const_like(1)
  for a,b in zip(left,rhs):
    maximum=a.alu(Ops.MAX,b); a_less,b_less=(_i16_bit(maximum.alu(Ops.SUB,value)) for value in (a,b))
    less=less.alu(Ops.MAX,equal.alu(Ops.MUL,a_less)); equal=equal.alu(Ops.MUL,equal.const_like(1).alu(Ops.SUB,a_less.alu(Ops.MAX,b_less)))
  return less

def _twos_complement(raw:Iterable[UOp], sign:UOp) -> tuple[UOp, ...]:
  carry,result=sign,[]
  for byte in raw:
    inverted=byte.const_like(255).alu(Ops.SUB,byte.alu(Ops.MUL,byte.const_like(2))).alu(Ops.MUL,sign)
    total=byte.alu(Ops.ADD,inverted).alu(Ops.ADD,carry); carry=_i16_bit(total.alu(Ops.SUB,total.const_like(255)))
    result.append(total.alu(Ops.SUB,carry.alu(Ops.MUL,total.const_like(256))))
  return tuple(result)

def _lower_raw_fp16_bitcast(output:RKOutput) -> RKImage|None:
  """Pair adjacent FP16 lane representations into an INT32 output without numeric conversion."""
  _,out,n,index,value=output; packed=value.src[0] if value.op is Ops.BITCAST and value.dtype is dtypes.int and len(value.src)==1 else None
  if n <= 0 or packed is None or packed.op is not Ops.ADD or packed.dtype.scalar() is not dtypes.uint: return None
  lanes:dict[int,RKTypedLoadPlan|None]={int(term.src[1].arg):_typed_load_plan(bitcast.src[0],dtypes.half,index,n,require_offsets=True) for term in packed.src if term.op is Ops.SHL and len(term.src)==2 and term.src[1].op is Ops.CONST and int(term.src[1].arg) in (0,16) for bitcast in (_typed_cast_source(term.src[0],dtypes.uint,dtypes.ushort),) if bitcast is not None and bitcast.op is Ops.BITCAST and len(bitcast.src)==1 and len(bitcast.src[0].src)==1}  # noqa: E501
  if len(packed.src)!=2 or set(lanes)!={0,16} or (low:=lanes[0]) is None or (high:=lanes[16]) is None or low.param.arg!=high.param.arg or any(a&1 or b!=a+1 for a,b in zip(low.gather.offsets,high.gather.offsets)): return None  # noqa: E501
  return RKImage(RKTarget.RK3588,gathers=(_raw_gather(RKArg(RKBufferKind.ARG,low.param.arg.slot),out.arg.slot,n,itemsize=4,src_kind=RKBufferKind.ARG)._replace(axes=(),offsets=tuple(offset//2 for offset in low.gather.offsets)),))  # noqa: E501

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

def _dynamic_load_recipe(load:UOp, out_index:UOp, count:int) -> UOp|None:
  """Express one bounded runtime-address LOAD as exact candidate-selection semantics."""
  dtype = load.dtype.scalar()
  if count <= 0 or dtype not in (dtypes.half,dtypes.int16,dtypes.int) or load.op is not Ops.LOAD or len(load.src) != 3 or \
     load.src[0].op is not Ops.INDEX or load.src[1].op is not Ops.CONST or load.src[1].arg != 0: return None
  data_param,data_index,gate = _root_param(load.src[0]),load.src[0].src[1],load.src[2]
  address_nodes = data_index.toposort()
  normalized = tuple((node,index,int(addition[1].arg)) for node in address_nodes if node.op is Ops.WHERE and
    node.src[0].op is Ops.CMPLT and node.src[0].src[1].op is Ops.CONST and int(node.src[0].src[1].arg) == 0 and
    (index:=node.src[0].src[0]).op is Ops.LOAD and index.dtype.scalar() is dtypes.int and node.src[2].key == index.key and
    (addition:=_const_operand(node.src[1],Ops.ADD)) is not None and addition[0].key == index.key and int(addition[1].arg) > 0)
  normalized_by_load = {dynamic.key:(base,extent) for base,dynamic,extent in normalized}
  dynamic_loads = tuple({node.key:node for node in address_nodes if node.op is Ops.LOAD and node.dtype.scalar() is dtypes.int}.values())
  if data_param is None or data_param.dtype.scalar() is not dtype or data_param.src[0].op is not Ops.CONST or \
     not dynamic_loads or len(normalized_by_load) != len(normalized): return None
  axes:list[tuple[UOp, UOp, int, bool]] = []
  for dynamic in dynamic_loads:
    axis,extent,wrapped = (*normalized_by_load[dynamic.key],True) if dynamic.key in normalized_by_load else \
      (dynamic,_bounded_index_gate(gate,dynamic),False)
    if extent is None or wrapped and _bounded_index_gate(gate,axis,extent) is None: return None
    axes.append((axis,dynamic,extent,wrapped))
  gate_loads = tuple(node for node in gate.toposort() if node.op is Ops.LOAD)
  bool_loads = tuple(node for node in gate_loads if node.dtype.scalar() is dtypes.bool)
  raw_params = tuple(_root_param(dynamic.src[0]) if dynamic.src and dynamic.src[0].op is Ops.INDEX else None for dynamic in dynamic_loads)
  invalid_params = any(param is None or param.dtype.scalar() is not dtypes.int or param.src[0].op is not Ops.CONST for param in raw_params)
  if len(bool_loads) > 1 or invalid_params or \
     {node.key for node in gate_loads} != {node.key for node in (*dynamic_loads,*bool_loads)}: return None
  params = typing_cast(tuple[UOp, ...],raw_params); bool_load = bool_loads[0] if bool_loads else None
  if bool_load is not None:
    bool_param = _root_param(bool_load.src[0]) if len(bool_load.src) == 1 and bool_load.src[0].op is Ops.INDEX else None
    if bool_load not in _iter_binary(gate,Ops.AND) or bool_param is None or bool_param.dtype.scalar() is not dtypes.bool or bool_param.src[0].op is not Ops.CONST: return None  # noqa: E501
    try: bool_offsets = _gather_offsets(out_index,bool_load.src[0].src[1],None,count)
    except RuntimeError: return None
    if any(not 0 <= offset < int(bool_param.src[0].arg) for offset in bool_offsets): return None
  candidates = math.prod(extent for _,_,extent,_ in axes)
  if candidates > min(_MAX_STATIC_RANGE_ENVS,_MAX_DYNAMIC_SELECTOR_CELLS//count): return None
  combinations = tuple(itertools.product(*(range(extent) for _,_,extent,_ in axes)))
  mappings = tuple({dynamic:dynamic.const_like(value) for (_,dynamic,_,_),value in zip(axes,values)} for values in combinations)
  static_gate = {} if bool_load is None else {bool_load:bool_load.const_like(True)}
  candidate_loads = tuple(load.substitute(mapping|static_gate) for mapping in mappings)
  try:
    index_offsets = tuple(_gather_offsets(out_index,dynamic.src[0].src[1],None,count) for dynamic in dynamic_loads)
    plans = tuple(_typed_load_plan(candidate,dtype,out_index,count,require_offsets=True) for candidate in candidate_loads)
  except RuntimeError: return None
  if any(plan is None for plan in plans) or any(not 0 <= offset < int(param.src[0].arg) or offset*4+3 > dtypes.int.max for offsets,param in zip(index_offsets,params) for offset in offsets) or any(offset >= 0 and offset*dtype.itemsize+dtype.itemsize-1 > dtypes.int.max for plan in plans if plan is not None for offset in plan.gather.offsets): return None  # noqa: E501
  selected = UOp.const(0,dtype)
  for values,candidate in zip(combinations,candidate_loads):
    masks = [functools.reduce(lambda x,y:x|y,(UOp(Ops.CMPEQ,dtypes.bool,src=(dynamic,dynamic.const_like(alternative)))
      for alternative in ((value,value-extent) if wrapped else (value,)))) for (_,dynamic,extent,wrapped),value in zip(axes,values)]
    mask = functools.reduce(lambda x,y:x&y,masks)
    selected = (mask if bool_load is None else mask&bool_load).where(candidate,selected)
  return selected

def _fp16_nonzero_mask(root:UOp) -> UOp|None:
  """Recognize a direct FP16-to-bool cast; ABS then positivity is exact for zero, infinity, and NaN."""
  if (source:=_typed_cast_source(root, dtypes.bool, dtypes.half)) is not None: root = source != UOp.const(0.0, dtypes.half)
  if (root:=_unwrap_condition(root)).op is not Ops.CMPNE: return None
  loads=[loaded for value,zero in (root.src,root.src[::-1]) if (loaded:=value if value.op is Ops.LOAD else value.load()).dtype.scalar() is dtypes.half and loaded.src[0].op is Ops.INDEX and zero.op is Ops.CONST and zero.arg==0]  # noqa: E501
  return _positive_mask(UOp(Ops.MAX,dtypes.half,src=(loads[0],loads[0]),arg=_NATIVE_ABS)) if len(loads)==1 else None

def _half_backed_value(value:UOp) -> UOp|None:
  """Normalize a half-backed numeric expression for the exact raw FP16 comparator."""
  original, value = value, _unwrap_condition(value)
  if value.op is Ops.INDEX: value = value.load()
  if value.op is Ops.CONST and value.dtype.scalar() is dtypes.weakfloat: value = UOp.const(float(value.arg), dtypes.half)
  if value.dtype.scalar() not in (dtypes.half, dtypes.float) and original.dtype.scalar() in (dtypes.half, dtypes.float): value = original
  valid = value.dtype.scalar() in (dtypes.half, dtypes.float) and not any(not load.src or load.src[0].op is not Ops.INDEX or
    (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtypes.half for load in value.toposort() if load.op is Ops.LOAD)
  return (value if value.dtype.scalar() is dtypes.half else value.cast(dtypes.half)) if valid else None

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

def _half_int_expr(u:UOp) -> UOp|None:
  if u.op is Ops.CONST: return UOp.const(float(u.arg),dtypes.half)
  if (source:=_typed_cast_source(u,dtypes.int,dtypes.half)) is not None: return _fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source,)))
  if (source:=_typed_cast_source(u,dtypes.int,dtypes.bool)) is not None: return source.cast(dtypes.half)
  if u.op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.CMOD) and len(u.src) == 2:
    mapped=tuple(_half_int_expr(src) for src in u.src)
    if any(src is None for src in mapped): return None
    lhs,rhs=typing_cast(tuple[UOp,UOp],mapped)
    if u.op is Ops.CMOD: return lhs.alu(Ops.SUB,_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(lhs.alu(Ops.FDIV,rhs),))).alu(Ops.MUL,rhs))  # noqa: E501
    return u.replace(dtype=dtypes.half,src=(lhs,rhs))
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  condition=u.src[0]
  if condition.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.int for src in condition.src):
    compared=tuple(_half_int_expr(src) for src in condition.src)
    if any(src is None for src in compared): return None
    condition=condition.replace(src=typing_cast(tuple[UOp,...],compared))
  arms=tuple(_half_int_expr(src) for src in u.src[1:])
  return None if any(src is None for src in arms) else UOp(Ops.WHERE,dtypes.half,src=(condition,*typing_cast(tuple[UOp,...],arms)))

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

class _RKRuntimeAddress(NamedTuple):
  """A proved affine or table-backed runtime load address."""
  load:UOp; param:UOp; itemsize:int; offset:int; base:int; scale:int; lane_stride:int; affine:bool

def _runtime_load_address(index:UOp, out_index:UOp, count:int, loads:tuple[UOp, ...]) -> _RKRuntimeAddress|None:
  """Resolve one runtime index without reading its values on the renderer."""
  infos = tuple(info for node in index.toposort() if (info:=_runtime_index(node)) is not None)
  if len(infos) == 1 and (lane_offset:=_runtime_lane_offset(infos[0], out_index, count)) is not None:
    load,param,_,itemsize = infos[0]
    try: zero,one = (_static_values(out_index,index.substitute({load:load.const_like(value)}),count,int) for value in (0,1))
    except RuntimeError: pass
    else:
      lane_stride = zero[1]-zero[0] if count > 1 else 0
      if len({a-b for a,b in zip(one,zero)}) == 1 and zero == tuple(zero[0]+lane*lane_stride for lane in range(count)):
        return _RKRuntimeAddress(load,param,itemsize,lane_offset,zero[0],one[0]-zero[0],lane_stride,True)
  runtime_loads = {info[0].key:info for node in loads if (info:=_runtime_index(node)) is not None}
  if len(runtime_loads) != 1: return None
  load,param,_,itemsize = info = next(iter(runtime_loads.values()))
  if (lane_offset:=_runtime_lane_offset(info,out_index,count)) is None: return None
  return _RKRuntimeAddress(load,param,itemsize,lane_offset,0,0,0,False)

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
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() in (dtypes.int, dtypes.int16, dtypes.bool): return u.src[0].cast(dtypes.half)
  if u.op is Ops.LOAD: return u.cast(dtypes.half)
  if u.op is Ops.CONST: return UOp.const(float(u.arg), dtypes.half)
  if _is_static_expr(u): return u.cast(dtypes.half)
  if ((u.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN, Ops.NEG) and len(u.src) == 1) or
      (u.op in (Ops.MUL, Ops.SUB, Ops.MAX) and len(u.src) == 2)):
    return UOp(u.op, dtypes.half, src=tuple(_fp32_expr_to_half(src) for src in u.src), arg=u.arg if u.op not in (Ops.MUL, Ops.NEG) else None)
  if u.op is Ops.ADD:
    # Apply static nonfinite masks after the compensated finite sum: TwoSum arithmetic on infinity produces NaN.
    terms=_fp32_add_terms(u); masks=[term for term in terms if _is_static_expr(term) and any(node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half,dtypes.float) and not math.isfinite(float(node.arg)) for node in term.toposort())]; return functools.reduce(lambda value,mask:value.alu(Ops.ADD,mask),masks,_precise_mul_sum([term for term in terms if term not in masks]))  # noqa: E501
  raise _RKGenericReject

def _nested_fp32_storage_cast(x:UOp) -> UOp|None:
  try: return _fp32_expr_to_half(x)
  except _RKGenericReject: return None

_pm_half_storage_algebra = PatternMatcher([(UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.float, name="x"),)), _nested_fp32_storage_cast),
  (UPat(Ops.FDIV, dtypes.half, src=(UPat.var("x"), UPat.var("y"))), lambda x,y:x.alu(Ops.MUL, UOp(Ops.RECIPROCAL, dtypes.half, src=(y,))))])

def _canonical_half_storage(source:UOp) -> UOp:
  """Commit one FP32 storage expression, then reuse Tinygrad's ordinary algebra on its now-identical half values."""
  converted = _fp32_expr_to_half(source)
  return converted if len(source.toposort()) > 64 else graph_rewrite(graph_rewrite(converted,_pm_half_storage_algebra+sym,name="rockchip half storage algebra"),pm_commit_weak,name="rockchip commit storage constants")  # noqa: E501

def _fp32_add_terms(u:UOp) -> list[UOp]: return [_fp32_expr_to_half(x) for x in _iter_binary(u, Ops.ADD, dtypes.float)]

def _fp32_ratio_to_half(u:UOp) -> UOp|None:
  """Divide two FP32 ADD boundaries while retaining their high/low half expansions through FDIV."""
  if u.op is not Ops.FDIV or u.dtype.scalar() is not dtypes.half or len(u.src) != 2: return None
  sources=tuple(_typed_cast_source(boundary,dtypes.half,dtypes.float) for boundary in u.src)
  if any(source is None or source.op is not Ops.ADD for source in sources): return None
  sums=typing_cast(tuple[UOp,UOp],sources)
  numerator_high,numerator_low=_precise_sum_parts(_fp32_add_terms(sums[0])); denominator_high,denominator_low=_precise_sum_parts(_fp32_add_terms(sums[1]))  # noqa: E501
  numerator, denominator = numerator_high.alu(Ops.ADD, numerator_low), denominator_high.alu(Ops.ADD, denominator_low)
  quotient, neg_one = numerator.alu(Ops.FDIV, denominator), UOp.const(-1.0, dtypes.half)
  residual = _sub_half(numerator_high, quotient.alu(Ops.MUL, denominator_high), neg_one).alu(Ops.ADD,
    _sub_half(numerator_low, quotient.alu(Ops.MUL, denominator_low), neg_one))
  return _tag_precise_adds(quotient.alu(Ops.ADD, residual.alu(Ops.FDIV, denominator)))

def _accurate_add_recipe(u:UOp, pure:bool=False) -> UOp|None:
  terms=[part for x in _iter_binary(u,Ops.ADD,plain=True) for part in next((_fp32_add_terms(source)
    for source in (_typed_cast_source(x,dtypes.half,dtypes.float),) if source is not None and source.op is Ops.ADD),(x,))]
  if sum(term.op is Ops.MUL and term.arg is None for term in terms) < 2 or any(any(node.op in (Ops.EXP2,Ops.LOG2,Ops.SQRT,Ops.SIN) for node in term.toposort()) for term in terms) or pure and any(not (term.op is Ops.MUL and term.dtype.scalar() is dtypes.half or term.op is Ops.CONST and float(term.arg) == 0.0) for term in terms): return None  # noqa: E501
  return _precise_mul_sum([term for term in terms if term.op is not Ops.CONST or float(term.arg) != 0.0])

class RKContext:
  """Typed physical lowering context. UOps remain the only semantic IR."""
  def __init__(self, output:RKOutput):
    self.store,self.out_param,self.count,self.out_index,self.root=output; self.out=RKArg(RKBufferKind.ARG,self.out_param.arg.slot)
    # Initialize per-context state before checking the root-derived layout.
    self.values:dict[UOp,RKValue]={}; self.scratch:list[RKScratch]=[]; self.constants:dict[bytes,int]={}; self.materialized_slots:dict[tuple,int]={}  # noqa: E501
    self.raw_components:dict[RKArg,tuple[RKValue,...]]={}
    self.recipe_owners:dict[UOp,UOp]={}
    self.gathers:list[RKGather]=[]; self.host_gathers:list[RKHostAddress]=[]; self.mid_gathers:list[RKGather]=[]; self.post_gathers:list[RKGather]=[]  # noqa: E501
    self.ew_ops:list[RKEWOp] = []
    nodes=self.root.toposort(); self.semantic_nodes=set(nodes); self.mask_program=any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in nodes)  # noqa: E501
    int_range=_exact_int_range(self.root) if self.root.dtype.scalar() is dtypes.int else None
    dynamic_int_load=any(node.op is Ops.LOAD and node.dtype.scalar() in (dtypes.int,dtypes.uint) and node.src and _root_param(node.src[0]) is not None for node in nodes)  # noqa: E501
    cmod_support=tuple(_half_int_expr(node) is not None for node in nodes if node.op is Ops.CMOD)
    wide_int=dynamic_int_load or any(node.op is Ops.CDIV for node in nodes) or False in cmod_support
    narrow_int=not wide_int and (self.root.dtype.scalar() is dtypes.int and int_range is not None and \
      -32768 <= int_range[0] <= int_range[1] <= 32767 or self.root.dtype.scalar() is not dtypes.int and bool(cmod_support))
    self.int_layout = RKLayout.INT16 if narrow_int else RKLayout.INT32

  def _layout(self, dtype:DType) -> RKLayout:
    if (layout:={dtypes.half:RKLayout.FP16, dtypes.float:RKLayout.FP16, dtypes.int16:RKLayout.INT16,
      dtypes.uchar:RKLayout.INT16, dtypes.bool:RKLayout.INT16, dtypes.uint:RKLayout.INT32}.get(
        dtype,self.int_layout if dtype is dtypes.int else None)) is None: raise _RKGenericReject(f"layout {dtype}")
    return layout

  def _scratch(self, dtype:DType, layout:RKLayout, size:int|None=None, u:UOp|None=None) -> RKValue:
    output_layout = {dtypes.half:RKLayout.FP16,dtypes.int16:RKLayout.INT16,dtypes.int:RKLayout.INT32}.get(dtype)
    if (u is self.root or u is not None and self.recipe_owners.get(u) is self.root) and self.out_param.dtype.scalar() is dtype and layout is output_layout: return RKValue(self.out,dtype,self.count,layout)  # noqa: E501
    self.scratch.append(RKScratch(size if size is not None else self.count*layout.itemsize() if layout is RKLayout.INT32 else _scratch_bytes(self.count)))  # noqa: E501
    return RKValue(RKArg(RKBufferKind.SCRATCH, len(self.scratch)-1), dtype, self.count, layout)

  def _slot(self, cache:dict, source:bytes|RKGather|tuple, dtype:DType, layout:RKLayout,
            size:int|None=None, key:tuple|None=None) -> RKValue:
    if isinstance(source, RKGather):
      plan, cache_key = source, typing_cast(bytes|tuple,("gather",layout,_gather_cache_key((source,))) if key is None else ("gather",key))
    elif isinstance(source, tuple):
      plan = RKGather(0, 0, self.count, values=source, itemsize=layout.itemsize())
      cache_key = ("static", layout, source)
    else:
      plan, cache_key = None, source
    if cache_key not in cache:
      value = self._scratch(dtype, layout, size)
      if plan is not None: self.gathers.append(plan._replace(dst_index=value.arg.index))
      cache[cache_key] = value.arg.index
    return RKValue(RKArg(RKBufferKind.SCRATCH, cache[cache_key]), dtype, self.count, layout)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> RKValue:
    dtype, layout = dtype_hint or u.dtype.scalar(), self._layout(dtype_hint or u.dtype.scalar())
    if layout is RKLayout.INT32: return self._slot(self.materialized_slots,(int(u.arg)&0xffffffff,)*self.count,dtype,layout)
    bits = struct.pack("<e",float(u.arg)) if layout is RKLayout.FP16 else struct.pack("<H",_int16_bits(u.arg))
    return self._slot(self.constants, bits, dtype, layout)

  def _operand(self, u:UOp, dtype:DType) -> RKValue:
    return self._constant(u, dtype) if u.op is Ops.CONST and \
      (u.dtype.scalar() in dtypes.weaks or dtype is dtypes.half and u.dtype.scalar() is dtypes.float) else self.lower(u)

  def _static(self, u:UOp) -> RKValue:
    dtype, layout = u.dtype.scalar(), self._layout(u.dtype.scalar())
    if not _index_ranges(u): return self._constant(UOp.const(typing_cast(int|float|bool,_eval_expr(u,{},{})),dtype))
    values = _static_values(self.out_index,u,self.count,_fp16_bits if layout is RKLayout.FP16 else int)
    if dtype is dtypes.int and layout is RKLayout.INT16 and any(not -32768 <= value <= 32767 for value in values): raise _RKGenericReject
    encoded = values if layout is RKLayout.FP16 else tuple(value&0xffffffff if layout is RKLayout.INT32 else _int16_bits(value) for value in values)
    return self._slot(self.materialized_slots,encoded,dtype,layout)

  def _masked_load_default(self, u:UOp, dtype:DType, layout:RKLayout, gate:UOp|None, default:UOp,
                           runtime_address:bool) -> RKValue:
    """Overlay a static masked load on a separately materialized default."""
    if dtype not in (dtypes.half,dtypes.int16,dtypes.int,dtypes.uint) or gate is None or runtime_address: raise _RKGenericReject
    schedule, fallback = (len(self.ew_ops),len(self.mid_gathers),len(self.host_gathers)),self.lower(default)
    if fallback.layout is not layout or fallback.count != self.count or \
       schedule != (len(self.ew_ops),len(self.mid_gathers),len(self.host_gathers)): raise _RKGenericReject
    if (plan:=_typed_load_plan(u,dtype,self.out_index,self.count,fill_bits=0)) is None: raise _RKGenericReject
    value = self._scratch(dtype,layout,self.count*dtype.itemsize)
    self.gathers.extend((RKGather(fallback.arg.index,value.arg.index,self.count,base=fallback.arg.addend//dtype.itemsize,
      axes=((1,self.count,1),),src_kind=fallback.arg.kind,itemsize=dtype.itemsize),
      plan.gather._replace(dst_index=value.arg.index,partial=True,itemsize=dtype.itemsize)))
    return value

  def _host_address_load(self, param:UOp, index:UOp, gate:UOp|None, address_loads:tuple[UOp, ...],
                         dtype:DType, layout:RKLayout, fill_bits:int) -> RKValue:
    """Materialize a proved dynamic address as explicit raw host movement."""
    if os.getenv("ROCKCHIP_HOST_GATHER","1") != "1" or \
       (address:=_runtime_load_address(index,self.out_index,self.count,address_loads)) is None: raise _RKGenericReject
    index_limit = _bounded_index_gate(gate,address.load) if gate is not None else int(param.src[0].arg)
    if index_limit is None or gate is not None and \
       {node.key for node in gate.toposort() if node.op is Ops.LOAD} != {address.load.key}: raise _RKGenericReject
    source,source_count = RKArg(RKBufferKind.ARG,param.arg.slot),int(param.src[0].arg)
    base,index_scale,lane_stride = address.base,address.scale,address.lane_stride
    if not address.affine:
      if gate is None or index_limit <= 0 or self.count*index_limit > _MAX_STATIC_RANGE_ENVS: raise _RKGenericReject
      try: candidates = tuple(_static_values(self.out_index,index.substitute({address.load:address.load.const_like(candidate)}),
        self.count,int) for candidate in range(index_limit))
      except RuntimeError: raise _RKGenericReject from None
      offsets = tuple(candidates[candidate][lane] for lane in range(self.count) for candidate in range(index_limit))
      plan = RKGather(param.arg.slot,0,len(offsets),offsets=offsets,itemsize=dtype.itemsize)
      _validate_gather_bounds(plan,source_count)
      source,source_count = self._slot(self.materialized_slots,plan,dtype,layout,len(offsets)*dtype.itemsize).arg,len(offsets)
      base,index_scale,lane_stride = 0,1,index_limit
    value = self._scratch(dtype,layout,self.count*dtype.itemsize)
    self.host_gathers.append(RKHostAddress(source,
      RKArg(RKBufferKind.ARG,address.param.arg.slot,address.offset*address.itemsize),value.arg,
      self.count,source_count,self.count,dtype.itemsize,address.itemsize,fill_bits,index_limit,base,index_scale,lane_stride))
    return value

  def _fp32_load(self, plan:RKTypedLoadPlan) -> RKValue:
    """Convert gathered FP32 storage through the NPU's aligned FP16 input ABI."""
    groups = tuple(range(0,self.count,_EW_ELEMS_32BIT))
    raw = self._slot(self.materialized_slots,plan.gather._replace(itemsize=4),dtypes.float,RKLayout.FP16,len(groups)*16,
      ("fp32_raw",_gather_cache_key((plan.gather._replace(itemsize=4),))))
    aligned,zero = self._scratch(dtypes.half,RKLayout.FP16,len(groups)*16),self._scratch(dtypes.float,RKLayout.FP16,16)
    self.gathers.append(RKGather(0,zero.arg.index,_EW_ELEMS_32BIT,values=(0,)*_EW_ELEMS_32BIT,itemsize=4))
    for group,start in enumerate(groups): self.ew_ops.append(RKEWOp(aligned.arg._replace(addend=group*16),
      raw.arg._replace(addend=group*16),zero.arg,min(_EW_ELEMS_32BIT,self.count-start),
      _EW_CFG[Ops.ADD]|_EW_STAGE_FP32_IN,stateful=True))
    compact = self._scratch(dtypes.half,RKLayout.FP16,self.count*2)
    self.mid_gathers.append(RKGather(aligned.arg.index,compact.arg.index,self.count,
      offsets=tuple((lane//_EW_ELEMS_32BIT)*8+lane%_EW_ELEMS_32BIT for lane in range(self.count)),
      src_kind=RKBufferKind.SCRATCH,after=len(self.ew_ops)))
    return RKValue(compact.arg,dtypes.float,self.count,RKLayout.FP16)

  def _load(self, u:UOp, fill_override:int|None=None) -> RKValue:
    dtype,layout = u.dtype.scalar(),self._layout(u.dtype.scalar())
    if not u.src or u.src[0].op is not Ops.INDEX or (param:=_root_param(u.src[0])) is None or \
       param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST: raise _RKGenericReject
    index,gate = u.src[0].src[1],u.src[2] if len(u.src) > 2 else None
    default = u.src[1] if len(u.src) > 1 else None
    address_loads = _semantic_loads(index)+(() if gate is None else _semantic_loads(gate))
    if default is not None and default.op is not Ops.CONST:
      return self._masked_load_default(u,dtype,layout,gate,default,bool(address_loads))
    if dtype in (dtypes.float,dtypes.bool) and address_loads: raise _RKGenericReject
    fill = 0 if default is None else default.arg
    if dtype is dtypes.float: fill_bits = struct.unpack("<I",struct.pack("<f",float(fill)))[0]
    elif fill_override is not None: fill_bits = fill_override
    elif dtype is dtypes.half: fill_bits = _fp16_bits(fill)
    elif dtype is dtypes.int16: fill_bits = _int16_bits(fill)
    else: fill_bits = int(fill) & 0xffffffff
    if address_loads:
      if (u is self.root or self.root.op is Ops.WHERE) and \
         (recipe:=_dynamic_load_recipe(u,self.out_index,self.count)) is not None: return self.lower(recipe)
      return self._host_address_load(param,index,gate,address_loads,dtype,layout,fill_bits)
    if (plan:=_typed_load_plan(u,dtype,self.out_index,self.count,fill_bits=fill_bits,
                              require_offsets=dtype is dtypes.bool)) is None: raise _RKGenericReject
    if dtype is dtypes.float: return self._fp32_load(plan)
    if dtype is dtypes.bool:
      return self._slot(self.materialized_slots,plan.gather._replace(
        fill_bits=int(bool(default.arg)) if default is not None else 0,dst_stride=2,itemsize=1),
        dtype,RKLayout.INT16,self.count*2)
    if gate is None and index.key == self.out_index.key and int(plan.param.src[0].arg) == self.count:
      return RKValue(RKArg(RKBufferKind.ARG,plan.param.arg.slot),dtype,self.count,layout)
    return self._slot(self.materialized_slots,plan.gather._replace(itemsize=dtype.itemsize),
      dtype,layout,self.count*dtype.itemsize)

  def _emit(self, dst:RKValue, lhs:RKValue, rhs:RKValue, cfg:int, *, compare:bool=False) -> RKValue:
    integer16, integer32 = dst.layout is RKLayout.INT16, dst.layout is RKLayout.INT32
    if lhs.layout is not dst.layout or rhs.layout is not dst.layout: raise _RKGenericReject
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
           dst:RKValue|None=None, cache:bool=True, copy_wide:bool=True) -> Any:
    if isinstance(source, RKValue):
      value = source
      if cache and value.arg in self.raw_components: return self.raw_components[value.arg]
      itemsize, source = value.layout.itemsize(), value
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
    itemsize = layout.itemsize()
    if len(parts) != itemsize: raise _RKGenericReject
    result = self._scratch(u.dtype.scalar(), layout, u=u) if dst is None else dst
    for byte,part in enumerate(parts): self._byte_gather(part.arg, result.arg, self.count, base=part.arg.addend, source_stride=2,
                                                         dst_stride=itemsize, dst_addend=byte, itemsize=1, after=len(self.ew_ops))
    if cache: self.raw_components[result.arg] = parts
    return result

  def _physical(self, value:RKValue|RKArg|UOp) -> UOp:
    """Bind an existing physical carrier into an ordinary semantic UOp recipe."""
    if isinstance(value,UOp): return value
    arg,layout=(value.arg,value.layout) if isinstance(value,RKValue) else (value,RKLayout.INT16)
    dtype={RKLayout.FP16:dtypes.half,RKLayout.INT16:dtypes.int16,RKLayout.INT32:dtypes.int}[layout]
    node=UOp(Ops.NOOP,dtype,src=(UOp.const(0,dtype),),arg=arg)
    self.values.setdefault(node,RKValue(arg,dtype,self.count,layout)); return node

  def _lower_recipe(self, owner:UOp, recipe:UOp, dtype:DType|None=None) -> RKValue:
    self.recipe_owners[recipe]=owner; value=self.lower(recipe)
    return value if dtype is None else value._replace(dtype=dtype)

  def _bitplanes(self, value:RKValue) -> tuple[UOp, ...]:
    return tuple(itertools.chain.from_iterable(_byte_bits(self._physical(part)) for part in self._raw(value,copy_wide=False)))

  def _pack_bits(self, bits:Iterable[UOp], layout:RKLayout, u:UOp) -> RKValue:
    planes=tuple(bits)
    if len(planes)!=layout.itemsize()*8: raise _RKGenericReject
    raw=tuple(functools.reduce(lambda x,y:x.alu(Ops.ADD,y),
      (planes[byte*8+bit].alu(Ops.MUL,planes[byte*8+bit].const_like(1<<bit)) for bit in range(1,8)),planes[byte*8])
      for byte in range(layout.itemsize()))
    return self._raw(tuple(self.lower(part) for part in raw),layout,u=u)

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
    dtype, int_range = u.dtype.scalar(), _exact_int_range(u) if u.dtype.scalar() is dtypes.int else None
    bounded = self.int_layout is RKLayout.INT32 or self.int_layout is RKLayout.INT16 and int_range is not None and \
      -32768 <= int_range[0] <= int_range[1] <= 32767
    if dtype is dtypes.int and not bounded: raise _RKGenericReject(f"alu {u.op.name} {dtype} bounds={int_range}")
    expected = self._layout(dtype)
    def operand(src:UOp) -> RKValue:
      if (u.op is Ops.MAX and dtype is dtypes.half and src.op is Ops.CONST and math.isinf(float(src.arg)) and float(src.arg) < 0):
        return self._constant(UOp.const(-65504.0, dtypes.half))
      if (u.op is Ops.MAX and dtype is dtypes.half and src.op is Ops.LOAD and len(src.src) > 2 and src.src[1].op is Ops.CONST and
          math.isinf(float(src.src[1].arg)) and float(src.src[1].arg) < 0 and _is_static_expr(src.src[2])):
        return self._load(src, _fp16_bits(-65504.0))
      return self._operand(src, dtype)
    lhs, rhs = operand(u.src[0]), operand(u.src[1])
    if lhs.layout is not expected or rhs.layout is not expected: raise _RKGenericReject
    left,right=self._physical(lhs),self._physical(rhs)
    if u.op is Ops.SUB and u.arg == _NATIVE_SIGN:
      if expected is not RKLayout.FP16: raise _RKGenericReject
      zero=left.const_like(0.0)
      return self._lower_recipe(u,_positive_mask(left).alu(Ops.SUB,_positive_mask(zero.alu(Ops.SUB,left))))
    if u.op is Ops.MAX and u.arg == _NATIVE_MIN:
      if expected is not RKLayout.FP16: return self._emit(self._scratch(dtype,RKLayout.INT16,u=u),lhs,rhs,_EW_CFG_MIN)
      zero=left.const_like(0.0)
      return self._lower_recipe(u,zero.alu(Ops.SUB,zero.alu(Ops.SUB,left).alu(Ops.MAX,zero.alu(Ops.SUB,right))))
    cfg = _EW_CFG_ABS if u.op is Ops.MAX and u.arg == _NATIVE_ABS else _EW_CFG_FLOOR if u.op is Ops.MAX and u.arg == _NATIVE_FLOOR else \
      _EW_CFG_CEIL if u.op is Ops.MAX and u.arg == _NATIVE_CEIL else _EW_CFG_RELU6 if u.op is Ops.MAX and u.arg == _NATIVE_RELU6 else _EW_CFG[u.op]
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    return self._emit(self._scratch(dtype,expected,u=u),lhs,rhs,cfg,compare=compare)

  def _convert(self, u:UOp|None, source:RKValue, dtype:DType, layout:RKLayout) -> RKValue:
    """Cross one physical carrier boundary using the native DPU conversion stage."""
    if source.layout is layout: return source._replace(dtype=dtype)
    pair, flags, cfg = (source.layout,layout), {}, _EW_CFG[Ops.MAX]
    result=None if pair==(RKLayout.INT16,RKLayout.INT32) else self._scratch(dtype,layout,u=None if pair==(RKLayout.INT32,RKLayout.FP16) else u)  # noqa: E501
    if pair == (RKLayout.FP16,RKLayout.INT16): rhs,flags=source.arg,dict(stateful=True,int16_output=True,submit_barrier=dtype in (dtypes.bool,dtypes.uchar))  # noqa: E501
    elif pair == (RKLayout.FP16,RKLayout.INT32): rhs,flags=self._scratch(dtypes.int,RKLayout.INT32,(self.count+3)//4<<6).arg,dict(stateful=True,int32_output=True)  # noqa: E501
    elif pair == (RKLayout.INT32,RKLayout.FP16): rhs,flags=self._scratch(dtypes.int,RKLayout.INT32,(self.count+3)//4<<6).arg,dict(int32_input=True)  # noqa: E501
    elif pair == (RKLayout.INT16,RKLayout.INT32): rhs,flags,cfg=self._constant(UOp.const(0,dtypes.int16)).arg,dict(int16_input=True,int32_output=True),_EW_CFG[Ops.ADD]  # noqa: E501
    else: raise _RKGenericReject(f"convert {source.layout.name}->{layout.name}")
    if result is None: result=self._scratch(dtype,layout,u=u)
    self.ew_ops.append(RKEWOp(result.arg,source.arg,rhs,self.count,cfg,**flags)); return result

  def _bool_binary(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    ordered=next(((expression,typing_cast(tuple[UOp,UOp],sources)) for expression,marker in (u.src,u.src[::-1])
      if u.op is Ops.CMPNE and marker.op is Ops.CONST and marker.dtype.scalar() is dtypes.bool and bool(marker.arg) and expression.op is Ops.CMPLT
      for sources in (tuple(_half_backed_value(src) for src in expression.src),) if all(src is not None for src in sources)),None)
    if ordered is not None:
      expression,sources=ordered; less=self.lower(expression)
      if less.layout is not RKLayout.INT16: raise _RKGenericReject
      nan=tuple(self._fp16_component_values(self._operand(src,dtypes.half))[2] for src in sources); one=self._physical(less).const_like(1)
      return self._lower_recipe(u,one.alu(Ops.SUB,self._physical(less)).alu(Ops.MUL,
        one.alu(Ops.SUB,nan[0].alu(Ops.MAX,nan[1]))),dtypes.bool)
    values=tuple(self.lower(src) for src in u.src)
    if any(value.dtype is not dtypes.bool or value.layout is not RKLayout.INT16 for value in values): raise _RKGenericReject
    lhs,rhs=(self._physical(value) for value in values)
    if u.op in (Ops.AND,Ops.OR): result=lhs.alu(Ops.MUL if u.op is Ops.AND else Ops.MAX,rhs)
    if u.op in (Ops.XOR, Ops.CMPNE):
      complement=next((other for source,other in zip(u.src,(rhs,lhs)) if source.op is Ops.CONST and bool(source.arg)),None)
      result=lhs.const_like(1).alu(Ops.SUB,complement) if complement is not None else _i16_compare(Ops.CMPNE,lhs,rhs)
    elif u.op is Ops.CMPEQ: result=_i16_compare(Ops.CMPEQ,lhs,rhs)
    elif u.op not in (Ops.AND,Ops.OR): raise _RKGenericReject
    return self._lower_recipe(u,result,dtypes.bool)

  def _integer_bitwise(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    dtype,layout=u.dtype.scalar(),self._layout(u.dtype.scalar())
    if dtype not in (dtypes.int16,dtypes.int) or u.op not in (Ops.AND,Ops.OR,Ops.XOR): raise _RKGenericReject
    if u.op is Ops.XOR and (pair:=next(((source,marker) for source,marker in (u.src,u.src[::-1])
      if marker.op is Ops.CONST and int(marker.arg)==-1),None)) is not None:
      value=self.lower(pair[0])
      if value.layout is RKLayout.INT16: return self._lower_recipe(u,UOp.const(-1,dtypes.int16).alu(Ops.SUB,self._physical(value)))
      inverted=tuple(self.lower(self._physical(component).const_like(255).alu(Ops.SUB,self._physical(component))) for component in self._raw(value))  # noqa: E501
      return self._raw(inverted,RKLayout.INT32,u=u)
    masked=tuple(_const_operand(term,Ops.AND) for term in u.src) if dtype is dtypes.int16 and u.op is Ops.OR else ()
    if len(masked)==2 and all(pair is not None for pair in masked) and {int(typing_cast(tuple[UOp,UOp],pair)[1].arg)&0xffff for pair in masked}=={0x7fff,0x8000}:  # noqa: E501
      sources={int(typing_cast(tuple[UOp,UOp],pair)[1].arg)&0xffff:typing_cast(tuple[UOp,UOp],pair)[0] for pair in masked}
      (low,high),(_,sign_high)=(self._raw(self.lower(sources[mask])) for mask in (0x7fff,0x8000))
      hi,shi=self._physical(high),self._physical(sign_high)
      magnitude_sign=_i16_bit(hi.alu(Ops.SUB,hi.const_like(127))).alu(Ops.MUL,hi.const_like(128))
      sign=_i16_bit(shi.alu(Ops.SUB,shi.const_like(127))).alu(Ops.MUL,shi.const_like(128))
      return self._raw(tuple(self.lower(part) for part in (self._physical(low),hi.alu(Ops.SUB,magnitude_sign).alu(Ops.ADD,sign))),layout,u=u)  # noqa: E501
    values = tuple(self.lower(source) for source in u.src)
    if any(value.layout is not layout for value in values): raise _RKGenericReject
    if not 1 <= self.count*layout.itemsize() <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    lhs_bits,rhs_bits=(self._bitplanes(value) for value in values)
    combined=tuple(left.alu(Ops.MUL,right) if u.op is Ops.AND else left.alu(Ops.MAX,right) if u.op is Ops.OR else
                   _i16_abs(left.alu(Ops.SUB,right))
                   for left,right in zip(lhs_bits,rhs_bits))
    return self._pack_bits(combined,layout,u)

  def _int32_shift(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or u.dtype.scalar() not in (dtypes.int, dtypes.uint) or \
       u.src[1].dtype.scalar() not in (dtypes.int, dtypes.uint) or self.int_layout is not RKLayout.INT32:
      raise _RKGenericReject
    if (value:=self.lower(u.src[0])).layout is not RKLayout.INT32: raise _RKGenericReject
    if self.count<1 or 16*((self.count*2+63)&-64)>_MAX_EW_ELEMS_FP16: raise _RKGenericReject
    current=self._bitplanes(value); signed=u.op is Ops.SHR and u.dtype.scalar() is dtypes.int
    masks=() if u.src[1].op is Ops.CONST else self._bitplanes(self.lower(u.src[1]))[:5]
    for bit,amount in enumerate((1,2,4,8,16)):
      if not masks and not (int(u.src[1].arg)&amount): continue
      fill=current[31] if signed else current[0].const_like(0)
      shifted=tuple(current[index-amount] if u.op is Ops.SHL and index>=amount else
        current[index+amount] if u.op is Ops.SHR and index+amount<32 else fill for index in range(32))
      current=shifted if not masks else tuple(old.alu(Ops.ADD,masks[bit].alu(Ops.MUL,new.alu(Ops.SUB,old)))
        for old,new in zip(current,shifted))
    return self._pack_bits(current,RKLayout.INT32,u)

  def _compare(self, u:UOp) -> RKValue:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src): return self._bool_binary(u)
    if u.op is Ops.CMPNE and (u is self.root or any(src.op is Ops.INDEX for src in u.src)) and (nonzero:=_fp16_nonzero_mask(u)) is not None:  # noqa: E501
      value = self.lower(nonzero)
      if value.layout is not RKLayout.FP16: raise _RKGenericReject
      return self._convert(u,value,dtypes.bool,RKLayout.INT16)
    nan:tuple[UOp,...]=()
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      int_sources = typing_cast(tuple[UOp, UOp], tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src))  # noqa: E501
      half_sources=tuple(_half_int_expr(src) for src in int_sources)
      if self.int_layout is RKLayout.INT16 and all(src is not None for src in half_sources): return self.lower(u.replace(src=typing_cast(tuple[UOp,...],half_sources)))  # noqa: E501
      if self.int_layout is RKLayout.INT16: return self._lower_recipe(u,_i16_compare(u.op,*(self._physical(self._operand(src,dtypes.int)) for src in int_sources)),dtypes.bool)  # noqa: E501
      values=tuple(self._operand(src,dtypes.int) for src in int_sources)
      if any(value.layout is not RKLayout.INT32 for value in values): raise _RKGenericReject
      components=tuple(tuple(self._physical(part) for part in self._raw(value)) for value in values)
      if u.op is Ops.CMPLT: components=tuple((_sign_bias(parts[3]),*parts[2::-1]) for parts in components)
    elif all(src.dtype.scalar() is dtypes.int16 for src in u.src): return self._lower_recipe(u,_i16_compare(u.op,*(self._physical(self._operand(src,dtypes.int16)) for src in u.src)),dtypes.bool)  # noqa: E501
    else:
      half_sources = u.src if all(src.dtype.scalar() is dtypes.half for src in u.src) else tuple(_half_backed_value(src) for src in u.src)
      if u.op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) or any(src is None for src in half_sources): raise _RKGenericReject
      values=tuple(self._operand(src,dtypes.half) for src in typing_cast(tuple[UOp,UOp],half_sources))
      if any(value.layout is not RKLayout.FP16 for value in values): raise _RKGenericReject
      nan=tuple(self._fp16_component_values(value)[2] for value in values)
      components=tuple(self._fp16_ordered_values(value) if u.op is Ops.CMPLT else
        (self._physical(parts[0]),parts[1]) for value in values for parts in (self._fp16_component_values(value),))
    result=_ordered_bits(*components) if u.op is Ops.CMPLT else functools.reduce(
      lambda equal,pair:equal.alu(Ops.MUL,_i16_equal(*pair)),zip(*components),components[0][0].const_like(1))
    if nan: result=result.alu(Ops.MUL,result.const_like(1).alu(Ops.SUB,nan[0].alu(Ops.MAX,nan[1])))
    if u.op is Ops.CMPNE: result=result.const_like(1).alu(Ops.SUB,result)
    return self.lower(result)._replace(dtype=dtypes.bool)

  def _int32_divmod(self, u:UOp) -> RKValue:
    if len(u.src) != 2 or not 1 <= self.count <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    values = tuple(self._operand(src, dtypes.int) for src in u.src)
    if any(value.layout is not RKLayout.INT32 for value in values): raise _RKGenericReject
    raw=tuple(tuple(self._physical(part) for part in self._raw(value)) for value in values)
    signs=tuple(_i16_bit(value[3].alu(Ops.SUB,value[3].const_like(127))) for value in raw)
    numerator,denominator=(_twos_complement(value,sign) for value,sign in zip(raw,signs))
    denominator_nonzero=functools.reduce(lambda x,y:x.alu(Ops.MAX,y),map(_i16_bit,denominator))
    numerator_bits=tuple(itertools.chain.from_iterable(map(_byte_bits,numerator)))
    zero=numerator[0].const_like(0); remainder,quotient=[zero]*4,[zero]*4
    for bit_index in range(31, -1, -1):
      shifted,incoming=[],numerator_bits[bit_index]
      for byte in remainder:
        carry=_i16_bit(byte.alu(Ops.SUB,byte.const_like(127)))
        wrapped=byte.alu(Ops.ADD,byte).alu(Ops.SUB,carry.alu(Ops.MUL,byte.const_like(256)))
        shifted.append(wrapped.alu(Ops.ADD,incoming)); incoming=carry
      remainder=shifted
      ge=denominator_nonzero.alu(Ops.MUL,denominator_nonzero.const_like(1).alu(Ops.SUB,
        _ordered_bits(reversed(remainder),reversed(denominator)))); borrow,reduced=zero,[]
      for left,right in zip(remainder, denominator):
        delta=left.alu(Ops.SUB,right.alu(Ops.MUL,ge)).alu(Ops.SUB,borrow)
        borrow=_i16_bit(zero.alu(Ops.SUB,delta)); reduced.append(delta.alu(Ops.ADD,borrow.alu(Ops.MUL,zero.const_like(256))))
      remainder=reduced; byte_index,weight=bit_index>>3,1<<(bit_index&7)
      quotient[byte_index]=quotient[byte_index].alu(Ops.ADD,ge.alu(Ops.MUL,zero.const_like(weight)))
    quotient_raw, remainder_raw, remainder_sign, quotient_sign = tuple(quotient),tuple(remainder),signs[0],_i16_abs(signs[0].alu(Ops.SUB,signs[1]))
    packed_raw, sign = (quotient_raw, quotient_sign) if u.op is Ops.CDIV else (remainder_raw, remainder_sign)
    return self._raw(tuple(self.lower(value) for value in _twos_complement(packed_raw,sign)),RKLayout.INT32,u=u)

  def _fp16_component_values(self, value:RKValue) -> tuple[RKValue, UOp, UOp]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    low,high=self._raw(value); lo,hi=self._physical(low),self._physical(high)
    sign_scale=_i16_bit(hi.alu(Ops.SUB,hi.const_like(127))).alu(Ops.MUL,hi.const_like(128)); magnitude=hi.alu(Ops.SUB,sign_scale)
    clean=hi.alu(Ops.SUB,sign_scale.alu(Ops.MUL,_i16_equal(magnitude,magnitude.const_like(0)).alu(Ops.MUL,_i16_equal(lo,lo.const_like(0)))))
    exponent=_i16_bit(magnitude.alu(Ops.SUB,magnitude.const_like(123)))
    mantissa=_i16_bit(magnitude.alu(Ops.SUB,magnitude.const_like(124))).alu(Ops.MAX,_i16_min(lo,lo.const_like(1)))
    return low,clean,exponent.alu(Ops.MUL,mantissa)

  def _fp16_ordered_values(self, value:RKValue) -> tuple[UOp, UOp]:
    """Map a classified FP16 lane to two unsigned bytes whose lexical order is IEEE numeric order."""
    low,hi,_=self._fp16_component_values(value); lo=self._physical(low)
    sign=_i16_bit(hi.alu(Ops.SUB,hi.const_like(127))); positive=hi.alu(Ops.ADD,hi.const_like(128))
    high_delta=hi.const_like(255).alu(Ops.SUB,hi).alu(Ops.SUB,positive); low_delta=lo.const_like(255).alu(Ops.SUB,lo).alu(Ops.SUB,lo)
    return positive.alu(Ops.ADD,sign.alu(Ops.MUL,high_delta)),lo.alu(Ops.ADD,sign.alu(Ops.MUL,low_delta))

  def _threshold_where(self, u:UOp) -> RKValue|None:
    """Build a cheap FP16 0/1 predicate for a finite-threshold mask while excluding unordered lanes.
    Select a compared value or finite constant without multiplying an inactive nonfinite value."""
    gate = _unwrap_condition(u.src[0])
    if gate.op is not Ops.CMPLT or gate.src[1].op is not Ops.CONST or not math.isfinite(float(gate.src[1].arg)) or any(src.dtype.scalar() not in (dtypes.half,dtypes.float) for src in gate.src): return None  # noqa: E501
    lhs, yes, no = gate.src[0], *(_unwrap_condition(src) for src in u.src[1:])
    if not (yes.key == lhs.key and no.op is Ops.CONST and math.isfinite(float(no.arg)) and float(no.arg) != float(gate.src[1].arg) or no.key == lhs.key and yes.op is Ops.CONST and math.isfinite(float(yes.arg)) and float(yes.arg) != float(gate.src[1].arg)): return None  # noqa: E501
    value = self.lower(lhs.cast(dtypes.half))
    if value.layout is not RKLayout.FP16: raise _RKGenericReject
    selector = self.lower(_positive_mask(gate.src[1].cast(dtypes.half).alu(Ops.SUB, lhs.cast(dtypes.half))))
    if selector.layout is not RKLayout.FP16 or selector.dtype is not dtypes.half: raise _RKGenericReject
    nan = self._fp16_component_values(value)[2]
    selector_uop=self._physical(self._convert(None,selector,dtypes.bool,RKLayout.INT16)); nan_uop=self._physical(nan)
    return self._raw_where(u,self.lower(selector_uop.alu(Ops.MUL,nan_uop.const_like(1).alu(Ops.SUB,nan_uop)))._replace(dtype=dtypes.bool))

  def _raw_where(self, u:UOp, selector:RKValue|None=None) -> RKValue:
    """Select typed values through one canonical INT16 mask, preserving nonfinite arms as raw bytes."""
    yes, no = (self.lower(src) for src in u.src[1:])
    if yes.layout is not no.layout or not isinstance(yes.layout, RKLayout): raise _RKGenericReject
    if selector is None: selector = self.lower(u.src[0])
    if selector.dtype is not dtypes.bool or selector.layout is not RKLayout.INT16: raise _RKGenericReject
    if yes.layout is RKLayout.INT16:
      return self._lower_recipe(u,_i16_select(*(self._physical(value) for value in (selector,yes,no))),u.dtype.scalar())
    mask=self._physical(selector); yes_bytes,no_bytes=(self._raw(x,cache=False,copy_wide=False) for x in (yes,no))
    selected_bytes=[self.lower((n:=self._physical(no_byte)).alu(Ops.ADD,mask.alu(Ops.MUL,self._physical(yes_byte).alu(Ops.SUB,n)))) for yes_byte,no_byte in zip(yes_bytes,no_bytes)]  # noqa: E501
    return self._raw(selected_bytes, yes.layout, u=u, dst=self._scratch(u.dtype.scalar(), yes.layout), cache=False)

  def _where(self, u:UOp) -> RKValue:
    if len(u.src) != 3: raise _RKGenericReject
    if u is self.root and u.dtype.scalar() is dtypes.uchar and (source:=_typed_cast_source(u.src[1],dtypes.uchar,dtypes.half)) is not None and (condition:=u.src[0]).op is Ops.CMPLT and condition.src[0].op is Ops.CONST and float(condition.src[0].arg)==0.0 and condition.src[1].key==source.key and u.src[2].op is Ops.CONST and int(u.src[2].arg)==0:  # noqa: E501
      return self.lower(source.alu(Ops.MAX,UOp.const(0.0,dtypes.half)).cast(dtypes.uchar))
    if u is self.root and u.dtype.scalar() is dtypes.int and all(arm.op is Ops.CONST and -32768<=int(arm.arg)<=32767 for arm in u.src[1:]):  # noqa: E501
      selector=self.lower(u.src[0]); yes,no=(self._constant(UOp.const(int(arm.arg),dtypes.int16)) for arm in u.src[1:])
      return self._convert(u,self._lower_recipe(u,_i16_select(*(self._physical(value) for value in (selector,yes,no))),u.dtype.scalar()),dtypes.int,RKLayout.INT32)  # noqa: E501
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
    if (threshold:=self._threshold_where(u)) is not None: return threshold
    for fold in (_fold_where_abs, _fold_ordered_where):
      if (recipe:=fold(u)) is not None: return self.lower(recipe)
    return self._raw_where(u)

  def _cast(self, u:UOp) -> RKValue:
    dtype, source_dtype = u.dtype.scalar(), u.src[0].dtype.scalar()
    if dtype in (dtypes.bool,dtypes.uchar) and source_dtype is not dtypes.half or \
       dtype is dtypes.float and source_dtype not in (dtypes.half,dtypes.int,dtypes.bool) or \
       dtype is dtypes.int and source_dtype is dtypes.float: raise _RKGenericReject(f"cast {source_dtype}->{dtype}")
    source_u = u.src[0]
    if dtype is dtypes.uchar:
      if (relu:=_relu_operand(source_u)) is not None: source_u=relu.alu(Ops.MAX,UOp.const(0.0,dtypes.half))
      truncated=_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source_u,)))
      source_u=truncated.alu(Ops.SUB,_native_same(truncated.alu(Ops.MUL,UOp.const(1.0/256.0,dtypes.half)),_NATIVE_FLOOR).alu(
        Ops.MUL,UOp.const(256.0,dtypes.half)))
    elif dtype is dtypes.bool: source_u=_positive_mask(UOp(Ops.MAX,dtypes.half,src=(source_u,source_u),arg=_NATIVE_ABS))
    elif dtype is dtypes.int and source_dtype is dtypes.half: source_u=_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source_u,)))
    elif source_dtype is dtypes.bool and dtype in (dtypes.half,dtypes.float):
      source_u=source_u.where(UOp.const(1.0,dtypes.half),UOp.const(0.0,dtypes.half))
    if dtype is dtypes.half and source_dtype is dtypes.float:
      source=self._load(u.src[0]) if u.src[0].op is Ops.LOAD else self.lower(_fp32_expr_to_half(u.src[0]))
    elif dtype is dtypes.half and source_dtype is dtypes.int and (recipe:=_half_int_expr(u.src[0])) is not None: source=self.lower(recipe)
    else: source=self.lower(source_u)
    if source_dtype is dtypes.int and source.layout is RKLayout.INT16 and dtype in (dtypes.half,dtypes.float):
      source=self._convert(u.src[0],source,dtypes.int,RKLayout.INT32)
    if dtype in (dtypes.int16,dtypes.uint) and source.layout is not self._layout(dtype): raise _RKGenericReject
    return self._convert(u,source,dtype,self._layout(dtype))

  def _finish_value(self, result:RKValue, dtype:DType) -> None:
    if dtype is dtypes.int and result.layout is RKLayout.INT16: result=self._convert(self.root,result,dtype,RKLayout.INT32)
    expected=RKLayout.INT32 if dtype is dtypes.int else self._layout(dtype)
    if result.layout is not expected: raise _RKGenericReject
    if dtype is dtypes.float:
      groups=tuple(range(0,self.count,_EW_ELEMS_32BIT)); aligned,split=self._scratch(dtypes.half,RKLayout.FP16,len(groups)*16),len(self.ew_ops)
      for group,start in enumerate(groups):
        lanes=min(_EW_ELEMS_32BIT,self.count-start); self.mid_gathers.append(RKGather(result.arg.index,aligned.arg.index,lanes,
          offsets=tuple(result.arg.addend//2+lane for lane in range(start,start+lanes)),dst_addend=group*8,src_kind=result.arg.kind,after=split))  # noqa: E501
        source=aligned.arg._replace(addend=group*16); self.ew_ops.append(RKEWOp(self.out._replace(addend=start*4),source,source,lanes,
          _EW_CFG[Ops.MAX]|_EW_STAGE_FP32_OUT))
      return
    if result.arg == self.out: return
    if dtype in (dtypes.half,dtypes.int16): self._emit(RKValue(self.out,dtype,self.count,expected),result,result,_EW_CFG[Ops.MAX]); return
    source=result.arg._replace(addend=result.arg.addend//4) if dtype is dtypes.int else result.arg
    self.post_gathers.append(_raw_gather(source,self.out_param.arg.slot,self.count,stride=1 if dtype is dtypes.int else 2,
      itemsize=dtype.itemsize,src_kind=source.kind))

  def lower(self, u:UOp) -> RKValue:
    if u in self.values: return self.values[u]
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif (dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool, dtypes.uchar) and u in self.semantic_nodes and _is_static_expr(u) and  # noqa: E501
          not any(isinstance(node.arg, str) and node.arg.startswith("rockchip_") for node in u.toposort())):
      value = self._static(u)
    elif u.op in (Ops.INDEX, Ops.LOAD): value = self.lower(u.load()) if u.op is Ops.INDEX else self._load(u)
    elif u.op is Ops.BITCAST and len(u.src) == 1:
      source = self.lower(u.src[0])
      if dtype is dtypes.int16 and u.src[0].dtype.scalar() is dtypes.half and source.layout is RKLayout.FP16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.INT16)
      elif dtype is dtypes.half and u.src[0].dtype.scalar() is dtypes.int16 and source.layout is RKLayout.INT16:
        value = RKValue(source.arg, dtype, self.count, RKLayout.FP16)
      else: raise _RKGenericReject(f"bitcast {u.src[0].dtype.scalar()}->{dtype}")
      if u is self.root and value.arg != self.out:
        self.post_gathers.append(_raw_gather(value.arg._replace(addend=value.arg.addend//2), self.out_param.arg.slot, self.count,
                                             stride=1, itemsize=2, src_kind=value.arg.kind))
        value = RKValue(self.out, dtype, self.count, value.layout)
    elif u.op is Ops.CAST and len(u.src) == 1: value=self._cast(u)
    elif dtype is dtypes.bool and u.op in (Ops.MUL, Ops.MAX):
      value = self._bool_binary(UOp(Ops.AND if u.op is Ops.MUL else Ops.OR, dtype, src=u.src))
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL): value = self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND, Ops.OR, Ops.XOR) and dtype in (dtypes.bool, dtypes.int16, dtypes.int):
      value = self._bool_binary(u) if dtype is dtypes.bool else self._integer_bitwise(u)
    elif u.op in (Ops.SHL, Ops.SHR) and dtype in (dtypes.int, dtypes.uint): value = self._int32_shift(u)
    elif u.op is Ops.CMOD and dtype is dtypes.int and self.int_layout is RKLayout.INT16 and (recipe:=_half_int_expr(u)) is not None: value=self.lower(recipe.cast(dtypes.int))  # noqa: E501
    elif u.op in (Ops.CDIV,Ops.CMOD) and dtype is dtypes.int and (u.op is not Ops.CMOD or self.int_layout is not RKLayout.INT16): value=self._int32_divmod(u)  # noqa: E501
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN) and len(u.src) == 1 and dtype is dtypes.half and \
         (recipe:=_DPU_MATH[u.op](u.src[0])) is not None:
      value = self.lower(_physical_recipe(recipe,(u.src[0],)))
      if value.layout is not RKLayout.FP16: raise _RKGenericReject
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    return self.values.setdefault(u, value)

  def finish(self) -> RKImage:
    nodes=self.root.toposort(); predicated=any(node.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ,Ops.WHERE) and not _is_static_expr(node) for node in nodes); blocked:set[UOp]=set()  # noqa: E501
    # A dynamic predicate taints only its consumers; independent FP16 arithmetic can form one physical prelude.
    if len(nodes)>800:
      for node in nodes:
        if node.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ,Ops.WHERE) and not _is_static_expr(node) or any(src in blocked for src in node.src): blocked.add(node)  # noqa: E501
        # A maximal compensated ADD owns its prefixes; eagerly lowering each prefix only creates unused physical copies.
        elif ((not predicated and node.dtype.scalar() in (dtypes.half,dtypes.int16,dtypes.bool,dtypes.uchar) and node.op in (Ops.CONST,Ops.LOAD,Ops.CAST,*GroupOp.ALU)) or (node.dtype.scalar() is dtypes.half and node.op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.FDIV,Ops.NEG,Ops.RECIPROCAL) and all(load.dtype.scalar() is dtypes.half and _typed_load_plan(load,dtypes.half,self.out_index,self.count) is not None for load in _semantic_loads(node)))): self.lower(node)  # noqa: E501
    result, dtype = self.lower(self.root), self.out_param.dtype.scalar(); self._finish_value(result,dtype)
    constants = b"" if not self.constants else b"".join(
      {slot:bits for bits,slot in self.constants.items()}.get(i, b"\0\0") for i in range(max(self.constants.values())+1))
    image = RKImage(RKTarget.RK3588, tuple(self.scratch), constants, RKIMAGE_VERSION, tuple(self.gathers), tuple(self.ew_ops),
                    tuple(self.mid_gathers), min((g.after for g in self.mid_gathers if g.after >= 0), default=0),
                    tuple(self.post_gathers), tuple(self.host_gathers))
    return _reuse_linear_scratch(image, self.constants)

def _expand_math_uops(root:UOp, *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  if (ratio:=_fp32_ratio_to_half(root)) is not None: return ratio
  bounded_recipes = len(root.toposort()) <= _MAX_OPTIONAL_RECIPE_NODES
  if bounded_recipes and (composite_math:=next((recipe for fold in (_fold_inverse_hyperbolic,_fold_atan) if (recipe:=fold(root)) is not None),None)) is not None: root=_physical_recipe(composite_math)  # noqa: E501
  @functools.cache
  def rewrite(u:UOp) -> UOp:
    if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.float and not _has_runtime_address(u.src[0]):  # noqa: E501
      return (rewrite(_physical_recipe(_dpu_sin(u.src[0].src[0]),(u.src[0].src[0],)))
        if u.src[0].op is Ops.SIN else _canonical_half_storage(u.src[0]))
    if accurate_adds and bounded_recipes and u.op is Ops.ADD and u.dtype.scalar() is dtypes.half and u.arg is None and (recipe:=_accurate_add_recipe(u)) is not None: return recipe  # noqa: E501
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.op is Ops.WHERE and (absolute:=_fold_where_abs(mapped)) is not None: mapped = rewrite(absolute)
    if mapped.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN):
      if mapped.op is Ops.LOG2 and mapped.src[0].op is Ops.WHERE: raise _RKGenericReject
      if (recipe:=_DPU_MATH[mapped.op](mapped.src[0])) is None: raise _RKGenericReject
      mapped = rewrite(_physical_recipe(recipe, (mapped.src[0],)))
    elif mapped.op is Ops.TRUNC and mapped.dtype.scalar() is dtypes.half and not _is_static_expr(mapped):
      mapped = rewrite(_fold_trunc(mapped))
    return mapped
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
    if (mapped:=u.replace(src=tuple(cache[src] for src in u.src))).op is Ops.REDUCE:
      reduce_op, ranges = mapped.arg[0] if isinstance(mapped.arg, tuple) else mapped.arg, list(mapped.src[1:])
      if reduce_op not in (Ops.ADD, Ops.MAX, Ops.MUL) or not ranges or any(r.op is not Ops.RANGE for r in ranges): raise _RKGenericReject
      if not (envs:=_iter_range_env(ranges,_MAX_GENERIC_UNROLL,False)) or \
         len(envs)*len(mapped.src[0].toposort())>_MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
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
    substitutions={axis:axis.const_like(value) for axis,value in env.items()} | {
      load:expand_load(load,env) for load in _semantic_loads(expr,True) if _local_buffer(load) is not owner}
    return expr.substitute(substitutions, walk=True)
  def expand_load(load:UOp, env:dict[UOp, int]) -> UOp:
    if (buffer:=_local_buffer(load)) is None or buffer.src[0].op is not Ops.CONST: raise _RKGenericReject
    if len(stores:=[u for u in uops if u.op is Ops.STORE and _local_buffer(u) is buffer]) == 1 and \
       stores[0].src[0].op is Ops.INDEX and load.src[0].op is Ops.INDEX:
      store_index, load_index = stores[0].src[0].src[1], load.src[0].src[1]
      axes, load_axes = _index_ranges(store_index), _index_ranges(load_index)
      if buffer.dtype.scalar() is not dtypes.bool: raise _RKGenericReject
      if len(axes) != 1 or store_index.key != axes[0].key or axes[0].op is not Ops.SPECIAL or \
         axes[0].src[0].op is not Ops.CONST or (store_extent:=int(axes[0].src[0].arg)) <= 0 or \
         store_extent > int(buffer.src[0].arg) or len(load_axes) != 1 or load_index.key != load_axes[0].key or \
         load_axes[0].op not in (Ops.RANGE, Ops.SPECIAL) or load_axes[0].src[0].op is not Ops.CONST or \
         (load_extent:=int(load_axes[0].src[0].arg)) <= 0 or load_extent > int(buffer.src[0].arg): raise _RKGenericReject
      if len(updates:={node.op for node in root.toposort() if node.dtype.scalar() is dtypes.bool and node.op in (Ops.AND,Ops.OR) and any(
        _local_buffer(local) is buffer for local in _semantic_loads(node,True))}) != 1: raise _RKGenericReject
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
    if not (definition:=definitions.setdefault(buffer,_static_local_defs(uops,{buffer})[buffer])).loops or any(
       not (loop.src[0].op is Ops.CONST and 0<=int(loop.src[0].arg)<=_MAX_GENERIC_UNROLL) for loop in definition.loops): raise _RKGenericReject
    if (iterations:=math.prod(int(loop.src[0].arg) for loop in definition.loops)) > min(_MAX_GENERIC_UNROLL, budget[0]): raise _RKGenericReject
    budget[0] -= iterations
    terms=[expand_dependencies(definition.initial,buffer,env),*(expand_dependencies(definition.term,buffer,env|loop_env) for loop_env in _iter_range_env(list(definition.loops),None,False))]  # noqa: E501
    while definition.update_op is Ops.ADD and buffer.dtype.scalar() is dtypes.float and len(terms)>1: terms=[UOp(Ops.ADD,definition.term.dtype,src=(terms[i],terms[i+1])) for i in range(0,len(terms)-1,2)]+(terms[-1:] if len(terms)&1 else [])  # noqa: E501
    accumulator=terms[0] if definition.update_op is Ops.ADD and buffer.dtype.scalar() is dtypes.float else functools.reduce(lambda value,term:UOp(definition.update_op,definition.term.dtype,src=(value,term)),terms[1:],terms[0])  # noqa: E501
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

def _match_scalar_extrema(output:RKOutput, uops:list[UOp]) -> tuple[UOp,int,_RKStaticLocalDef,tuple[int,...],int,int]|None:
  """Recognize paired scalar MAX accumulators as a value vector and affine output coordinate."""
  _,out,count,_,root=output
  if count != 1 or out.dtype.scalar() is not dtypes.int: return None
  try:
    definitions=_static_local_defs(uops,{buffer for u in uops if u.op is Ops.BUFFER and (buffer:=_local_buffer(u)) is not None}); limit=min(32767,_MAX_EW_ELEMS_FP16)  # noqa: E501
    descriptors={dtype:(buffer,definition,tuple(int(loop.src[0].arg) for loop in definition.loops)) for dtype,initial in ((dtypes.half,-math.inf),(dtypes.int,dtypes.int.min)) for buffer,definition in definitions.items()  # noqa: E501
      if buffer.dtype.scalar() is dtype and definition.update_op is Ops.MAX and definition.initial.op is Ops.CONST and float(definition.initial.arg)==initial and definition.loops and all(loop.src[0].op is Ops.CONST and 1<=int(loop.src[0].arg)<=limit for loop in definition.loops)}  # noqa: E501
    (value_buffer,value_def,extents),(index_buffer,index_def,index_extents)=descriptors[dtypes.half],descriptors[dtypes.int]
    if len(definitions)!=2 or extents!=index_extents or not 2<=(total:=math.prod(extents))<=limit: return None
    terms=tuple(_iter_binary(index_def.term,Ops.MUL)); cast=next(term for term in terms if term.op is Ops.CAST and term.dtype.scalar() is dtypes.int and term.src[0].dtype.scalar() is dtypes.bool)  # noqa: E501
    if len(terms)!=2: return None
    coordinate=next(term for term in terms if term is not cast); predicate=_inverted_condition(cast.src[0],True)
    if predicate is None or predicate.op is not Ops.CMPNE or len(predicate.src)!=2 or any(src.dtype.scalar() is not dtypes.half for src in predicate.src): return None  # noqa: E501
    current=next(src for src in predicate.src if (load:=_local_load(src)) is not None and _local_buffer(load) is value_buffer); candidate=next(src for src in predicate.src if src is not current)  # noqa: E501
    if _strip_cast(candidate.substitute(dict(zip(index_def.loops,value_def.loops)),walk=True)).key != _strip_cast(value_def.term).key: return None  # noqa: E501
    coordinates=tuple(int(_eval_expr(coordinate,env,{})) for env in _iter_range_env(list(index_def.loops),None,False))
    if len(coordinates)!=total or any(not 0<=value<=32767 for value in coordinates): return None
    if len(final_loads:=tuple(load for load in _semantic_loads(root,local=True) if _local_buffer(load) is index_buffer))!=1: return None
    mapped=tuple(int(_eval_expr(root.substitute({final_loads[0]:final_loads[0].const_like(value)}),{},{})) for value in coordinates)
    second=next(i for i in range(1,total) if coordinates[i]!=coordinates[0]); dx,dy=coordinates[second]-coordinates[0],mapped[second]-mapped[0]; slope=dy//dx; baseline=mapped[0]-slope*coordinates[0]  # noqa: E501
    if slope*dx!=dy: return None
    if any(result!=baseline+slope*value for value,result in zip(coordinates,mapped)) or not all(-32768<=value<=32767 for value in (*mapped,slope,baseline)): return None  # noqa: E501
    return out,total,value_def,coordinates,slope,baseline
  except (_RKGenericReject,RuntimeError,KeyError,StopIteration,ZeroDivisionError): return None

def _lower_scalar_extrema(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Lower a matched scalar extrema pair without recursively compiling its typed comparison phase."""
  if (plan:=_match_scalar_extrema(output,uops)) is None: return None
  out,total,value_def,coordinates,slope,baseline=plan
  flat=functools.reduce(lambda value,item:value.alu(Ops.ADD,item[1].alu(Ops.MUL,item[1].const_like(math.prod(tuple(int(loop.src[0].arg) for loop in value_def.loops[item[0]+1:]))))),reversed(tuple(enumerate(value_def.loops))),UOp.const(0,dtypes.int))  # noqa: E501
  fake_slot=1+max((u.arg.slot for u in uops if u.op is Ops.PARAM and u.arg is not None),default=out.arg.slot); fake=UOp.param(fake_slot,dtypes.half,(total,)); child_store=fake.index(flat).store(value_def.term).end(*value_def.loops)  # noqa: E501
  if (child:=_lower_uop_program(list(child_store.sink().toposort()),vectorize_reductions=False)) is None or child.host_gathers or child.host_scatters: return None  # noqa: E501
  target=RKArg(RKBufferKind.SCRATCH,len(child.scratch)); child=_alias_image_args(child,{fake_slot:target})._replace(scratch=child.scratch+(RKScratch(_scratch_bytes(total)),))  # noqa: E501
  lane=UOp.range(total,max((u.arg[0] for u in uops if u.op is Ops.RANGE and isinstance(u.arg,tuple)),default=-1)+1); loads=tuple(UOp.param(fake_slot+i,dtype,(total,)).index(lane).load() for i,dtype in enumerate((dtypes.half,dtypes.half,dtypes.int16),1))  # noqa: E501
  weighted=UOp(Ops.CMPEQ,dtypes.bool,src=loads[:2]).cast(dtypes.int16)*loads[2]; fake_out=UOp.param(fake_slot+4,dtypes.int16,(total,)); context=RKContext((fake_out.index(lane).store(weighted),fake_out,total,lane,weighted))  # noqa: E501
  # Continue the typed comparison in the child image's physical state; this keeps one allocator and one dependency schedule.
  context.scratch,context.gathers,context.ew_ops,context.mid_gathers=list(child.scratch),list(child.gathers),list(child.ew_ops),list(child.mid_gathers)
  context.constants={child.constants[i:i+2]:i//2 for i in range(0,len(child.constants),2)}; context.mid_gathers.extend(gather._replace(after=len(context.ew_ops)) for gather in child.post_gathers)  # noqa: E501
  def allocate(lanes:int=total)->RKArg: return context._scratch(dtypes.int16,RKLayout.INT16,_scratch_bytes(lanes)).arg
  values,spaced=target,allocate(total*32); context.mid_gathers.append(RKGather(values.index,spaced.index,total,axes=((1,total,1),),dst_stride=32,src_kind=RKBufferKind.SCRATCH,after=len(context.ew_ops)))  # noqa: E501
  best=_reduce_rows(context.ew_ops,[spaced._replace(addend=i*64) for i in range(total)],1,_EW_CFG[Ops.MAX]); best_values,coordinate_arg=allocate(),allocate(); context.mid_gathers.append(RKGather(best.index,best_values.index,total,offsets=(best.addend//2,)*total,src_kind=RKBufferKind.SCRATCH,after=len(context.ew_ops)))  # noqa: E501
  context.gathers.append(RKGather(out.arg.slot,coordinate_arg.index,total,values=coordinates)); context.values.update(dict(zip(loads,(RKValue(values,dtypes.half,total,RKLayout.FP16),RKValue(best_values,dtypes.half,total,RKLayout.FP16),RKValue(coordinate_arg,dtypes.int16,total,RKLayout.INT16)))))  # noqa: E501
  split=len(context.ew_ops); context.out=allocate(); selected=context.lower(weighted).arg; context.ew_ops[split]=context.ew_ops[split]._replace(submit_barrier=True,stateful=True)  # noqa: E501
  context.mid_gathers.append(RKGather(selected.index,spaced.index,total,axes=((1,total,1),),dst_stride=32,src_kind=RKBufferKind.SCRATCH,after=len(context.ew_ops))); result=_reduce_rows(context.ew_ops,[spaced._replace(addend=i*64) for i in range(total)],1,_EW_CFG[Ops.MAX],int16=True)  # noqa: E501
  for value,op in ((v,o) for v,o,n in ((slope,Ops.MUL,1),(baseline,Ops.ADD,0)) if v!=n):
    source,previous,result=allocate(1),result,allocate(1); context.gathers.append(RKGather(out.arg.slot,source.index,1,values=(_int16_bits(value),))); context.ew_ops.append(RKEWOp(result,previous,source,1,_EW_CFG[op],**_INT16_EW))  # noqa: E501
  zero=allocate(1); context.gathers.append(RKGather(out.arg.slot,zero.index,1,values=(0,))); context.ew_ops.append(RKEWOp(RKArg(RKBufferKind.ARG,out.arg.slot),result,zero,1,_EW_CFG[Ops.ADD],int16_input=True,int32_output=True))  # noqa: E501
  constants=b"" if not context.constants else b"".join({slot:bits for bits,slot in context.constants.items()}.get(i,b"\0\0") for i in range(max(context.constants.values())+1)); image=_reuse_linear_scratch(RKImage(RKTarget.RK3588,tuple(context.scratch),constants,gathers=tuple(context.gathers),ew_ops=tuple(context.ew_ops),mid_gathers=tuple(context.mid_gathers),gather_after=child.gather_after),context.constants)  # noqa: E501
  return image if all(len(items)<=_RKIMAGE_U16_MAX for items in (image.scratch,image.gathers,image.ew_ops,image.mid_gathers)) else None

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
  return RKImage(RKTarget.RK3588, host_scatters=(RKHostAddress(RKArg(RKBufferKind.ARG, source.arg.slot), RKArg(RKBufferKind.ARG, index_param.arg.slot),  # noqa: E501
    RKArg(RKBufferKind.ARG, out_param.arg.slot), count, int(source.src[0].arg), out_count,
    itemsize=out_param.dtype.scalar().itemsize, index_itemsize=index_itemsize),))

def _lower_uop_program(uops:list[UOp], *, vectorize_reductions:bool=True, recipes_ready:bool=False) -> RKImage|None:
  """Lower a composable typed UOp program; return None for the legacy correctness oracle."""
  if any(u.op is Ops.PARAM and not 0 <= u.arg.slot <= _RKIMAGE_U16_MAX for u in uops): return None
  accepted = (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool, dtypes.uchar)
  strict_output, local_output, output_stores = _outs(uops)
  if len(output_stores) > 1:
    lower_store = functools.partial(_lower_uop_program, vectorize_reductions=vectorize_reductions, recipes_ready=recipes_ready)
    if (combined:=lower_store(list(UOp(Ops.SINK, src=(output_stores[0],)).toposort()))) is None: return None
    for store in output_stores[1:]:
      if (child:=lower_store(list(UOp(Ops.SINK,src=(store,)).toposort()))) is None or \
         (combined:=_append_inplace_image(combined,child)) is None: return None
    return combined
  strict_output, local_output = (_admit(output, accepted) for output in (strict_output, local_output))
  if (cmac:=_try(local_output, (dtypes.half,dtypes.float), _lower_cmac_reduction, uops, v=vectorize_reductions)) is not None: return cmac
  if (mixed:=_try(strict_output,dtypes.half,_lower_cmac_storage_epilogue,uops,v=vectorize_reductions)) is not None: return mixed
  if (scatter:=_try(strict_output, (dtypes.half, dtypes.int16), _lower_host_scatter)) is not None: return scatter
  if (image:=_try(strict_output,dtypes.int,_lower_raw_fp16_bitcast)) is not None: return image
  if (extrema:=_try(local_output,dtypes.int,_lower_scalar_extrema,uops,v=vectorize_reductions)) is not None: return extrema
  storage_uops, storage_product_adds = None, False
  if any(u.dtype.scalar() is dtypes.float for u in uops) and (sink:=next((u for u in uops if u.op is Ops.SINK),None)) is not None:
    try:
      if (storage_output:=_admit(local_output,dtypes.half)) is not None:
        storage_root=storage_output[4]; storage_product_adds=any((boundary is not storage_root or len(boundary.src[0].toposort())>64) and _accurate_add_recipe(boundary) is not None for boundary in storage_root.toposort() if _typed_cast_source(boundary,dtypes.half,dtypes.float) is not None)  # noqa: E501
        sink=sink.substitute({storage_root:_expand_math_uops(storage_root,accurate_adds=False)})
    except _RKGenericReject: pass
    storage_uops=list(graph_rewrite(sink,_pm_storage_common,name="rockchip generic storage precision").toposort())
  if (output:=local_output if storage_uops is None else _admit(_outs(uops:=storage_uops)[1],accepted)) is None or len(output[0].src) != 2:  # noqa: E501
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage(RKTarget.RK3588)
  if output[2] > _MAX_EW_ELEMS_FP16 and any(_local_load(node) is not None for node in output[4].toposort()): return None
  try:
    if not ((affine:=typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(output[3]))) is not None and affine[0] == 0 and
            set(affine[1]) == set(_index_ranges(output[3])) and _affine_output_axes(affine, output[2]) is not None) and \
       _static_values(output[3], output[3], output[2], int) != tuple(range(output[2])): return None
    root=_finite_int_max_neutrals(_unroll_static_local(uops,_unroll_static_reduces(output[4]) if Ops.REDUCE in (u.op for u in uops) else output[4]))
    if not recipes_ready:
      root = _expand_math_uops(root,accurate_adds=storage_uops is None or storage_product_adds) if len(root.toposort()) <= 256 else recipe if (base:=_strip_cast(root)).dtype.scalar() is dtypes.half and (recipe:=_accurate_add_recipe(base,pure=True)) is not None else root  # noqa: E501
    if len(n:=root.toposort()) > _MAX_GENERIC_EXPANDED_NODES and os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(n)}")  # noqa: E501
    if len(n) > _MAX_GENERIC_EXPANDED_NODES: return None
    if root is not output[4]: output = (output[0].replace(src=(output[0].src[0], root)), *output[1:4], root)
    image = RKContext(output).finish()
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
