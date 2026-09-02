from __future__ import annotations
# ruff: noqa: E702
import base64, functools, heapq, io, itertools, math, os, pickle, struct, zlib
import numpy as np
from enum import IntEnum
from typing import Any, Callable, Iterable, Mapping, NamedTuple, cast as typing_cast
from tinygrad.device import Compiler
from tinygrad.dtype import DType, dtypes, float_to_fp16
from tinygrad.helpers import ceildiv, polyN, round_up
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite, identity_element, python_alu
from tinygrad.uop.symbolic import sym
from tinygrad.uop.weak import pm_commit_weak, pm_lower_index_dtype

RKIMAGE_MAGIC, RKIMAGE_VERSION, _RKIMAGE_U16_MAX = b"RKIM", 36, (1 << 16) - 1

class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1

class RKEWMode(IntEnum):
  HALF_TO_FLOAT=0; FLOAT_TO_HALF=1; INT16=2; INT32=3; INT16_TO_INT32=4; HALF_TO_INT32=5
  HALF_TO_INT16=6; INT32_TO_HALF=7; HALF=8; BOUNDED=9; STATEFUL=9; COMPARE=10

class RKArg(NamedTuple): kind: RKBufferKind; index: int; addend: int = 0  # type: ignore[assignment]

class RKGather(NamedTuple):
  """Materialize an affine or fallback raw-lane index map."""
  src: RKArg|None; dst: RKArg; count: int; base: int = 0  # type: ignore[assignment]
  # Axes are (destination divisor, range limit, source stride); offsets provide the non-affine fallback.
  axes: tuple[tuple[int, int, int], ...] = (); offsets: tuple[int, ...] = (); fill_bits: int = 0
  # Compile-time values have no source argument; partial gathers preserve lanes populated by another gather.
  values: tuple[int, ...] = (); partial: bool = False
  # Mapped reductions use a destination stride of 8 for 16-byte DPU atom alignment.
  dst_stride: int = 1; dst_addend: int = 0
  itemsize: int = 2
  # A runtime index makes this raw movement host-addressed; numeric semantics remain on the NPU.
  index: RKArg|None = None; index_itemsize: int = 4  # type: ignore[assignment]

class RKEWOp(NamedTuple):
  """One contiguous DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int  # type: ignore[assignment]
  submit_barrier: bool = False; mode: RKEWMode = RKEWMode.HALF

class RKCMAC(NamedTuple):
  """One fixed FP16 matrix contraction with an optional terminal BS ReLU; gathers own only its physical packing."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; m: int; n: int; k: int; out_fp16: bool = True; relu: bool = False

class RKImage(NamedTuple): scratch: tuple[int, ...] = (); program: tuple[RKGather|RKEWOp|RKCMAC, ...] = ()

def _op_args(op:RKGather|RKEWOp|RKCMAC) -> tuple[RKArg, ...]: return (() if op.src is None else (op.src,))+(() if op.index is None else (op.index,))+(op.dst,) if isinstance(op,RKGather) else (op.dst,op.lhs,op.rhs)  # noqa: E501

def _map_image_args(image:RKImage, fn:Callable[[RKArg], RKArg]) -> RKImage:
  return image._replace(program=tuple(op._replace(src=None if op.src is None else fn(op.src),dst=fn(op.dst),index=None if op.index is None else fn(op.index)) if isinstance(op,RKGather) else op._replace(dst=fn(op.dst),lhs=fn(op.lhs),rhs=fn(op.rhs)) for op in image.program))  # noqa: E501

def _alias_image_args(image:RKImage, aliases:dict[int, RKArg]) -> RKImage:
  return _map_image_args(image, lambda arg:aliases[arg.index]._replace(addend=aliases[arg.index].addend+arg.addend) if
                         arg.kind is RKBufferKind.ARG and arg.index in aliases else arg)

def _reuse_linear_scratch(image:RKImage) -> RKImage:
  """Color virtual scratch lifetimes across the complete physical execution schedule."""
  prelude:list[RKGather|RKEWOp|RKCMAC] = []; body:list[RKGather|RKEWOp|RKCMAC] = []; ready:set[int] = set(); written:set[int] = set()
  for op in image.program:
    deps=tuple(arg for arg in ((op.src,op.index) if isinstance(op,RKGather) else ()) if arg is not None); preload=isinstance(op,RKGather) and op.dst.kind is RKBufferKind.SCRATCH and all(arg.index in ready if arg.kind is RKBufferKind.SCRATCH else arg.index not in written for arg in deps) and (not op.partial or op.dst.index in ready)  # noqa: E501
    (prelude if preload else body).append(op); ready.add(op.dst.index) if preload else None
    if not preload and op.dst.kind is RKBufferKind.ARG: written.add(op.dst.index)
    if isinstance(op,(RKEWOp,RKCMAC)) and op.dst.kind is RKBufferKind.SCRATCH: ready.discard(op.dst.index)
  image=image._replace(program=tuple(prelude+body))
  events:dict[int, tuple[int, int]] = {}
  for event,args in enumerate(map(_op_args,image.program)):
    events.update((arg.index, (events.get(arg.index, (event, event))[0], event))
                  for arg in args if arg.kind is RKBufferKind.SCRATCH)
  if any(not 0 <= slot < len(image.scratch) for slot in events): raise ValueError("invalid virtual scratch slot")
  remap:dict[int,int]={}; physical:list[int]=[]; active:list[tuple[int,int]]=[]
  for start,end,slot in sorted(((points[0], points[1], slot) for slot,points in events.items()), key=lambda item:(item[0], item[2])):
    spec,target=image.scratch[slot],heapq.heappop(active)[1] if active and active[0][0]<start else len(physical)
    if target == len(physical): physical.append(0)
    physical[target] = max(physical[target], spec)
    heapq.heappush(active, (end, target))
    remap[slot] = target
  return _map_image_args(image,lambda arg:arg._replace(index=remap[arg.index]) if arg.kind is RKBufferKind.SCRATCH else arg)._replace(scratch=tuple(physical))  # noqa: E501

def _fits(values:Iterable[int], bits:int=32, signed:bool=False) -> bool: return (data:=np.asarray(values)).dtype.kind in "biu" and bool(np.all((-(1<<(bits-1)) if signed else 0)<=data)&np.all(data<(1<<(bits-1) if signed else 1<<bits))) if isinstance(values,tuple) and len(values)>1024 else all(isinstance(x,int) and -(1<<(bits-1)) <= x < 1<<(bits-1) if signed else isinstance(x,int) and 0 <= x < 1<<bits for x in values)  # noqa: E501

def _validate_image(image:RKImage) -> None:
  gathers=tuple(op for op in image.program if isinstance(op,RKGather)); hosts=tuple(op for op in gathers if op.index is not None); static=tuple(op for op in gathers if op.index is None); ew_ops=tuple(op for op in image.program if isinstance(op,RKEWOp)); cmacs=tuple(op for op in image.program if isinstance(op,RKCMAC))  # noqa: E501
  if len(image.scratch)>_RKIMAGE_U16_MAX or any(type(op) not in (RKGather,RKEWOp,RKCMAC) for op in image.program) or any(not _fits((size,)) for size in image.scratch): raise ValueError("invalid RKImage header")  # noqa: E501
  if len(cmacs)>1 or cmacs and hosts: raise ValueError("invalid CMAC schedule")
  if cmacs: _validate_cmac(cmacs[0],image.scratch)
  if any(g.itemsize not in (1,2,4) or (g.src is None) != bool(g.values) or not _fits((g.count,g.fill_bits,g.dst_stride)) or not _fits((g.base,g.dst_addend),signed=True) or g.dst_stride < 1 or g.dst_addend < 0 or len(g.axes)>255 or bool(g.values)+bool(g.offsets)+bool(g.axes)>1 or g.values and (len(g.values) not in (1,g.count) or not _fits(g.values,g.itemsize*8)) or g.offsets and (len(g.offsets)!=g.count or not _fits(g.offsets,signed=True)) or any(not _fits(axis[:2]) or not _fits(axis[2:],signed=True) for axis in g.axes) for g in static): raise ValueError("invalid RKGather")  # noqa: E501
  if any(h.src is None or h.index is None or h.dst.kind is not RKBufferKind.SCRATCH or h.values or h.offsets or h.axes or h.partial or h.base or h.dst_stride!=1 or h.dst_addend or h.itemsize not in (1,2,4) or h.index_itemsize not in (2,4) or not _fits((h.count,h.fill_bits)) or not _fits((h.src.addend,h.index.addend,h.dst.addend),signed=True) for h in hosts): raise ValueError("invalid runtime RKGather")  # noqa: E501
  if any(not _fits((arg.index,),16) for op in image.program for arg in _op_args(op)): raise ValueError("invalid RKArg")
  if any(op.mode==RKEWMode.HALF_TO_FLOAT and nxt.mode!=RKEWMode.HALF_TO_FLOAT or op.mode in (RKEWMode.HALF_TO_INT32,RKEWMode.INT16_TO_INT32) and op.dst.kind is RKBufferKind.ARG for op,nxt in zip(ew_ops,ew_ops[1:])): raise ValueError("invalid RKEWOp sequence")  # noqa: E501
  if any(not _fits((op.count,op.ew_cfg,op.mode)) or op.mode >= len(RKEWMode) or not _fits((op.dst.addend,op.lhs.addend,op.rhs.addend),signed=True) or op.mode in (RKEWMode.HALF_TO_INT32,RKEWMode.INT32_TO_HALF) and (op.count>4 or op.dst!=op.lhs or op.lhs!=op.rhs or op.dst.kind is not RKBufferKind.SCRATCH) for op in ew_ops): raise ValueError("invalid RKEWOp flags")  # noqa: E501

def encode_image(image:RKImage, *, validate:bool=True) -> bytes:
  _validate_image(image) if validate else None; return RKIMAGE_MAGIC+struct.pack("<H",RKIMAGE_VERSION)+zlib.compress(pickle.dumps(image,5),1)

def decode_image(blob:bytes) -> RKImage:
  try:
    if blob[:4] != RKIMAGE_MAGIC or struct.unpack_from("<H", blob, 4)[0] != RKIMAGE_VERSION: raise ValueError
    codec=zlib.decompressobj(); payload=codec.decompress(blob[6:])
    stream=io.BytesIO(payload)
    if codec.unused_data or not codec.eof or type(image:=pickle.load(stream)) is not RKImage or stream.tell()!=len(payload): raise ValueError
    _validate_image(image); return image
  except Exception: raise ValueError("invalid RKImage") from None

# Admission and exact-carrier bounds.
(_DPU, _RDMA, _MAX_EW_ELEMS_FP16, _MAX_GENERIC_UNROLL, _MAX_GENERIC_EXPANDED_NODES, _MAX_OPTIONAL_RECIPE_NODES, _MAX_STATIC_RANGE_ENVS, _MAX_DYNAMIC_SELECTOR_CELLS, _EW_ELEMS_32BIT, _FP16_EXACT_INTEGER) = (  # noqa: E501
  0x1001, 0x2001, 64000, 1 << 14, 1 << 20, 4096, 1 << 20, 1 << 22, 8*dtypes.half.itemsize//dtypes.float.itemsize, 1 << 11)
# Native EW register fields.
_EW_RELU_BYPASS, _EW_OP_CVT_BYPASS = 1 << 9, 1 << 8
_EW_CFG_COMMON = (1 << 28) | (2 << 22) | (1 << 7) | (1 << 6)
(_EW_CFG_RELU6, _EW_CFG_MIN, _EW_CFG_ABS, _EW_CFG_NEG, _EW_CFG_FLOOR, _EW_CFG_CEIL) = tuple(
  _EW_CFG_COMMON|flags for flags in (1<<10, _EW_RELU_BYPASS|(1<<16), _EW_RELU_BYPASS|(5<<16),
  _EW_RELU_BYPASS|(6<<16), _EW_RELU_BYPASS|(7<<16), _EW_RELU_BYPASS|(8<<16)))
# DPU data-format registers, indexed by RKEWMode.
_DPU_DATA_FORMATS = ((5<<29)|(2<<26)|2, (2<<29)|(5<<26)|2, (1<<29)|(1<<26)|1, (4<<29)|(4<<26)|4,
  (4<<29)|(1<<26)|1, (4<<29)|(2<<26)|2, (1<<29)|(2<<26)|2, (2<<29)|(4<<26)|4)+(((2<<29)|(2<<26)|2),)*3
# Batch-size and batch-normalization registers used by compare stages.
(_BS_BN_BYPASS, _BS_OW_FP32_SCALAR, _BS_CFG_COMPARE, _BS_ALU_COMPARE, _BS_MUL_COMPARE, _BN_CFG_COMPARE, _BN_MUL_COMPARE,
 _BN_RELUX_COMPARE) = (1|(1<<1)|(1<<4)|(1<<6), (1<<8)|(1<<5)|(1<<2)|(1<<1), 0x40040, 0x33800000, 0x40000000, 0x40082, 0x7c000000, 0x3f800000)
(_NATIVE_ABS, _NATIVE_CEIL, _NATIVE_FLOOR, _NATIVE_MASK_MUL, _NATIVE_MIN, _NATIVE_POSITIVE_MASK, _NATIVE_PRECISE_ADD,
 _NATIVE_RELU6, _NATIVE_SIGN) = tuple("rockchip_"+name for name in "abs ceil floor mask_mul min positive_mask precise_add relu6 sign".split())
_EW_RELUX_CMP_RELU6 = struct.unpack("<I", struct.pack("<f", 6.0))[0]
_EW_CFG = {op:_EW_CFG_COMMON|_EW_RELU_BYPASS|flags for op,flags in ((Ops.ADD,2<<16), (Ops.SUB,4<<16), (Ops.MUL,_EW_OP_CVT_BYPASS|1<<2), (Ops.MAX,0), (Ops.FDIV,_EW_OP_CVT_BYPASS|3<<16))}  # noqa: E501
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)
def _fp16_bits(value:float|int) -> int: return struct.unpack("<H", struct.pack("<e", float(value)))[0]
def _int16_bits(value:int|float|bool) -> int: return int(value) & 0xffff

def _cmac_layout(n:int, k:int) -> tuple[int, int, int]: aligned_k,align_out=max(32,round_up(k,32)),max(32,round_up(n,32)); align_in=max(aligned_k,align_out); return align_in,align_out,align_in if align_in != aligned_k else k  # noqa: E501

def _validate_cmac(op:RKCMAC, scratch:tuple[int, ...]|None=None) -> None:
  ai,ao,_ = _cmac_layout(op.n,op.k)
  if not 0 < op.m <= 0x7ff or ai > 13*32 or ao > 0x3fff or op.m*ai*2 > 10*32768 or ai > 12*32 and op.m != 1: raise ValueError("CMAC shape out of range")  # noqa: E501
  args,needs,alignments = (op.lhs,op.rhs,op.dst),(op.m*ai*2,ao*ai*2,op.m*ao*4),(2,2,2 if op.out_fp16 else 4)
  if any(arg.kind is not RKBufferKind.SCRATCH or arg.addend < 0 or arg.addend%alignment for arg,alignment in zip(args,alignments)): raise ValueError("CMAC requires aligned scratch buffers")  # noqa: E501
  if scratch is not None and any(not 0 <= arg.index < len(scratch) or arg.addend+need > scratch[arg.index] for arg,need in zip(args,needs)): raise ValueError("CMAC exceeds scratch buffer")  # noqa: E501

def emit_cmac_stage(op:RKCMAC, address:Callable[[RKArg],int]) -> tuple[int, ...]:
  """Emit the 45-qword GEMM body; terminal BS ReLU preserves the runtime-owned four-qword PC tail."""
  C,O,D,ai,ao,ek = 0x201,0x801,_DPU,*_cmac_layout(op.n, op.k)
  row_bytes = ai*2; grains = max(80, (ceildiv(2*32768, row_bytes)+1)&~1); banks = min(11, max(1, ceildiv(op.m*row_bytes, 32768)))
  line_stride, notch, (precision,size_e) = 4*min(ceildiv(ek,32),13), 8*min(ao//32,13)-1, (2,1) if op.out_fp16 else (5,3)
  regs = ((D,rk.REG_DPU_S_POINTER,0xe), (C,rk.REG_CNA_CONV_CON1,(2<<4)|(2<<7)|(1<<29)),
    (C,rk.REG_CNA_CONV_CON2,grains<<4), (C,rk.REG_CNA_CONV_CON3,9), (C,rk.REG_CNA_DATA_SIZE0,(1<<16)|op.m),
    (C,rk.REG_CNA_DATA_SIZE1,((ai-1)<<16)|ai), (C,rk.REG_CNA_DATA_SIZE2,1), (C,rk.REG_CNA_DATA_SIZE3,op.m),
    (C,rk.REG_CNA_WEIGHT_SIZE0,row_bytes*ao), (C,rk.REG_CNA_WEIGHT_SIZE1,row_bytes),
    (C,rk.REG_CNA_WEIGHT_SIZE2,(1<<24)|(1<<16)|ao), (C,rk.REG_CNA_CBUF_CON0,((12-banks)<<4)|banks),
    (C,rk.REG_CNA_CBUF_CON1,ceildiv(ai,32)), (C,rk.REG_CNA_CVT_CON0,11), (C,rk.REG_CNA_CVT_CON1,1<<16),
    (C,rk.REG_CNA_CVT_CON2,1<<16), (C,rk.REG_CNA_CVT_CON3,1<<16), (C,rk.REG_CNA_CVT_CON4,1<<16),
    (C,rk.REG_CNA_FEATURE_DATA_ADDR,address(op.lhs)), (C,rk.REG_CNA_DMA_CON0,(15<<16)|15), (C,rk.REG_CNA_DMA_CON1,line_stride),
    (C,rk.REG_CNA_DMA_CON2,0), (C,rk.REG_CNA_FC_DATA_SIZE0,(1<<16)|op.m), (C,rk.REG_CNA_FC_DATA_SIZE1,ai),
    (C,rk.REG_CNA_DCOMP_ADDR0,address(op.rhs)), (O,rk.REG_CORE_MISC_CFG,(2<<8)|1), (O,rk.REG_CORE_DATAOUT_SIZE_0,(op.m-1)<<16),
    (O,rk.REG_CORE_DATAOUT_SIZE_1,ao-1), (O,rk.REG_CORE_RESERVED_3030,0), (D,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)),
    (D,rk.REG_DPU_DATA_FORMAT,(precision<<29)|(2<<26)|2), (D,rk.REG_DPU_DST_BASE_ADDR,address(op.dst)), (D,rk.REG_DPU_DST_SURF_STRIDE,1<<4),
    (D,rk.REG_DPU_DATA_CUBE_WIDTH,0), (D,rk.REG_DPU_DATA_CUBE_HEIGHT,op.m-1),
    (D,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,(notch<<16)|notch), (D,rk.REG_DPU_DATA_CUBE_CHANNEL,((ao-1)<<16)|(ao-1)),
    (D,rk.REG_DPU_BS_CFG,0x12 if op.relu else 0x53), (D,rk.REG_DPU_BS_OW_CFG,(size_e<<8)|(size_e<<5)|(size_e<<2)|2),
    (D,rk.REG_DPU_WDMA_SIZE_0,ao-1), (D,rk.REG_DPU_WDMA_SIZE_1,(op.m-1)<<16), (D,rk.REG_DPU_BN_CFG,0x53),
    (D,rk.REG_DPU_EW_CFG,0x383), (D,rk.REG_DPU_OUT_CVT_SCALE,(1<<16)|1 if op.out_fp16 else 0), (D,rk.REG_DPU_SURFACE_ADD,4<<4))
  return tuple(_cmd(*reg) for reg in regs)

def _raw_gather(source:RKArg, out_slot:int, count:int, stride:int=2, itemsize:int=1, dst_stride:int=1, dst_addend:int=0, offsets:tuple[int, ...]=()) -> RKGather:  # noqa: E501
  return RKGather(source,RKArg(RKBufferKind.ARG,out_slot),count,axes=() if offsets else ((1,count,stride),),offsets=offsets,
                  dst_stride=dst_stride,dst_addend=dst_addend,itemsize=itemsize)

@functools.lru_cache(maxsize=256)
def _stage_template(count:int, ew_cfg:int, mode:RKEWMode=RKEWMode.HALF) -> tuple[tuple[int, ...], int]:
  """Emit either a self-initializing DPU EW body or a lean FP16 continuation body."""
  D, R = _DPU, rk
  native_int16,int16_to_int32,fp32_output,fp32_input = (mode == x for x in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_FLOAT,RKEWMode.FLOAT_TO_HALF))  # noqa: E501
  int32_output,int32_input = mode in (RKEWMode.INT32,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT32), mode in (RKEWMode.INT32,RKEWMode.INT32_TO_HALF)  # noqa: E501
  special,compare = mode != RKEWMode.HALF, mode == RKEWMode.COMPARE
  limit = 8 if int16_to_int32 else _MAX_EW_ELEMS_FP16//2 if mode==RKEWMode.INT32 else _EW_ELEMS_32BIT if int32_output or int32_input or fp32_output or fp32_input else _MAX_EW_ELEMS_FP16  # noqa: E501
  if not 0 < count <= limit: raise ValueError(f"{'initialized EW' if special else 'EW fp16'} count {count} out of range")
  lanes, is_div = (4 if int32_input or fp32_input else 8), ew_cfg == _EW_CFG[Ops.FDIV]
  width, data_format = (count + lanes-1) // lanes - 1, _DPU_DATA_FORMATS[mode]
  regs:tuple[tuple[int, int, int], ...] = ((D,R.REG_DPU_S_POINTER,0xe),(D,R.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (D,R.REG_DPU_DATA_FORMAT,data_format)) + (((D,R.REG_DPU_DST_SURF_STRIDE,1<<4),) if int16_to_int32 or fp32_output else ()) + (
    (D,R.REG_DPU_DATA_CUBE_WIDTH,width),(D,R.REG_DPU_DATA_CUBE_HEIGHT,0),(D,R.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (D,R.REG_DPU_DATA_CUBE_CHANNEL,0 if fp32_output and count == 1 else ((lanes-1)<<16)|(lanes-1)))
  if special:
    pipeline = (((D,R.REG_DPU_BS_CFG,_BS_BN_BYPASS),(D,R.REG_DPU_BN_CFG,_BS_BN_BYPASS),(D,R.REG_DPU_BS_ALU_CFG,0),(D,R.REG_DPU_BS_MUL_CFG,0),
      (D,R.REG_DPU_BS_OW_CFG,_BS_OW_FP32_SCALAR if int16_to_int32 or fp32_output and count == 1 else 2),
      (D,R.REG_DPU_WDMA_SIZE_0,0 if fp32_output and count == 1 else 3 if fp32_output else lanes-1),(D,R.REG_DPU_WDMA_SIZE_1,width),
      (D,R.REG_DPU_BN_MUL_CFG,0),(D,R.REG_DPU_BN_RELUX_CMP_VALUE,0))
      + (((D,R.REG_DPU_BS_CFG,_BS_CFG_COMPARE),(D,R.REG_DPU_BS_ALU_CFG,_BS_ALU_COMPARE),(D,R.REG_DPU_BS_MUL_CFG,_BS_MUL_COMPARE),
      (D,R.REG_DPU_BN_CFG,_BN_CFG_COMPARE),(D,R.REG_DPU_BN_MUL_CFG,_BN_MUL_COMPARE),
      (D,R.REG_DPU_BN_RELUX_CMP_VALUE,_BN_RELUX_COMPARE)) if compare else ())
      + (((D,R.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6),) if ew_cfg == _EW_CFG_RELU6 else ())
      + ((D,R.REG_DPU_EW_CFG,_EW_CFG_COMMON|1 if compare else (ew_cfg & ~(3<<22)) | (3<<22) | _EW_OP_CVT_BYPASS if int32_input else \
      ew_cfg & ~_EW_OP_CVT_BYPASS if native_int16 or int16_to_int32 else ew_cfg),
      (D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),
      (D,R.REG_DPU_OUT_CVT_SCALE,0 if fp32_output else 1 if int32_output or mode in (RKEWMode.INT16,RKEWMode.HALF_TO_INT16) or is_div else (1<<16)|1),  # noqa: E501
      (D,R.REG_DPU_OUT_CVT_SHIFT,0),(D,R.REG_DPU_SURFACE_ADD,(2 if native_int16 or int16_to_int32 else 4)<<4)))
  else:
    pipeline = ((D,R.REG_DPU_EW_CFG,ew_cfg),) + (((D,R.REG_DPU_EW_RELUX_CMP_VALUE,_EW_RELUX_CMP_RELU6),) if ew_cfg == _EW_CFG_RELU6 else ()) + (
      ((D,R.REG_DPU_EW_CVT_SCALE_VALUE,1),(D,R.REG_DPU_OUT_CVT_OFFSET,0),(D,R.REG_DPU_OUT_CVT_SHIFT,0),
       (D,R.REG_DPU_SURFACE_ADD,1<<6)) if is_div else ()) + ((D,R.REG_DPU_OUT_CVT_SCALE,1 if is_div else (1<<16)|1),)
  regs += pipeline + ((_RDMA,R.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,R.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,lanes-1),
    (_RDMA,R.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|((3 if int32_input or fp32_input else 2)<<2)))
  rdma_precision = 5 if fp32_input else 4 if int32_input else 1 if mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32) else 2
  rdma_feature = (rdma_precision<<15)|(15<<11)|(rdma_precision<<5)|(0 if is_div or mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32) or fp32_input else 1<<3)|1  # noqa: E501
  return tuple(_cmd(*reg) for reg in regs), rdma_feature

def emit_ew_stage(op:RKEWOp, address:Callable[[RKArg],int]) -> tuple[int, ...]:
  """Build one DPU EW command body without its PC-chain tail."""
  commands,feature = _stage_template(op.count,op.ew_cfg,op.mode)
  return commands+tuple(_cmd(target,reg,address(arg)) for target,reg,arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,op.dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,op.lhs),(_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,op.rhs)))+(_cmd(_RDMA,rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,feature),)  # noqa: E501

def _root_param(u:UOp) -> UOp|None: return root if (root:=u.buf_uop).op is Ops.PARAM else None

def _strip_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _typed_cast_source(u:UOp, dtype:DType, source:DType) -> UOp|None:
  return u.src[0] if u.op is Ops.CAST and u.dtype.scalar() is dtype and len(u.src) == 1 and u.src[0].dtype.scalar() is source else None

@functools.lru_cache(maxsize=65536)
def _semantic_loads(u:UOp) -> tuple[UOp, ...]:
  sources = () if u.op in (Ops.RANGE, Ops.SPECIAL) else u.src[:1] if u.op is Ops.AFTER else u.src
  return (u,) if u.op is Ops.LOAD else tuple(dict.fromkeys(y for x in sources for y in _semantic_loads(x)))

_STATIC_OPS = {Ops.CONST, Ops.RANGE, Ops.SPECIAL, Ops.CAST, Ops.ADD, Ops.MUL, Ops.SUB, Ops.RECIPROCAL, Ops.TRUNC, Ops.WHERE,
               Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD}

@functools.lru_cache(maxsize=4096)
def _static_ranges(u:UOp) -> tuple[UOp, ...]|None:
  """Return first-seen value ranges for a static expression, hiding RANGE ordering dependencies."""
  if u.op in (Ops.RANGE,Ops.SPECIAL) or u.op not in _STATIC_OPS: return (u,) if u.op in (Ops.RANGE,Ops.SPECIAL) else None
  sources=tuple(_static_ranges(source) for source in u.src)
  return None if any(x is None for x in sources) else tuple(dict.fromkeys(itertools.chain.from_iterable(typing_cast(tuple[tuple[UOp,...],...],sources))))  # noqa: E501

def _is_static_expr(u:UOp) -> bool: return _static_ranges(u) is not None
def _index_ranges(u:UOp) -> list[UOp]: return list(_static_ranges(u) or ())

def _eval_static(u:UOp, env:Mapping[UOp, int|float|bool|np.ndarray], cache:dict[UOp,np.ndarray]|None=None) -> np.ndarray:
  """Evaluate one static UOp graph with NumPy scalar or lane semantics."""
  cache={} if cache is None else cache; cache.update({node:np.asarray(value,dtype=np.dtype(node.dtype.scalar().fmt) if node.dtype.scalar().fmt is not None else None) for node,value in env.items()})  # noqa: E501
  for node in u.toposort(gate=lambda item:item not in cache):
    if node.op in (Ops.CONST,Ops.RANGE,Ops.SPECIAL): value=node.arg if node.op is Ops.CONST else env[node]
    else:
      values=tuple(cache[source] for source in node.src)
      if node.op is Ops.CAST: value=values[0]
      elif node.op in (Ops.WHERE,Ops.MAX): value=np.where(*values) if node.op is Ops.WHERE else np.where(values[1]>values[0],values[1],values[0])
      elif node.op in (Ops.CDIV,Ops.CMOD,Ops.FLOORDIV,Ops.FLOORMOD):
        quotient=np.where(values[1],np.abs(values[0])//np.abs(np.where(values[1],values[1],1))*np.where(values[0]*values[1]<0,-1,1),0) if node.op in (Ops.CDIV,Ops.CMOD) else np.where(values[1],values[0]//np.where(values[1],values[1],1),0); value=values[0]-quotient*values[1] if node.op in (Ops.CMOD,Ops.FLOORMOD) else quotient  # noqa: E501
      elif node.op in (Ops.RECIPROCAL,Ops.TRUNC): value=np.divide(1,values[0]) if node.op is Ops.RECIPROCAL else np.where((truncated:=np.trunc(values[0]))==0,0,truncated)  # noqa: E501
      else: value=python_alu[node.op](*values)
    cache[node]=np.asarray(value,dtype=np.dtype(node.dtype.scalar().fmt) if node.dtype.scalar().fmt is not None else None)
  return cache[u]

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

@functools.lru_cache(maxsize=8)
def _static_lanes(index:UOp|tuple[UOp,...], *roots:UOp, limit:int=_MAX_STATIC_RANGE_ENVS, dependencies:bool=True) -> tuple[np.ndarray,...]:
  """Enumerate one bounded static lane space and evaluate all requested roots in it."""
  ranges,roots=(_static_ranges(index) or (),(index,*roots)) if isinstance(index,UOp) else (index,roots)
  axes=tuple(dict.fromkeys(node for root in ranges for node in root.toposort() if node.op in (Ops.RANGE,Ops.SPECIAL))) if dependencies else ranges  # noqa: E501
  bounds=tuple(int(r.src[0].arg) if r.src and r.src[0].op is Ops.CONST else -1 for r in axes); count=math.prod(bounds)
  if any(bound<0 for bound in bounds) or count>limit: raise RuntimeError("RKPLAN_REJECT:static_index_budget")
  if any((used:=_static_ranges(root)) is None or any(r not in ranges for r in used) for root in roots): raise RuntimeError("RKPLAN_REJECT:static_index")  # noqa: E501
  env=dict(zip(axes,np.indices(bounds,dtype=np.int64).reshape(len(axes),count))) if axes else {}; cache:dict[UOp,np.ndarray]={}
  return tuple(np.broadcast_to(_eval_static(root,env,cache),count) for root in roots)

def _static_values(out_index:UOp, expr:UOp, count:int, encode:Callable[[int|float|bool], int]) -> tuple[int, ...]:
  dst_lanes,expr_lanes=_static_lanes(out_index,expr)
  if encode is _fp16_bits:
    if np.any(np.isfinite(fp_values:=np.asarray(expr_lanes,dtype=np.float64)) & (np.abs(fp_values)>=65520)): raise OverflowError("float too large to pack with e format")  # noqa: E501
    encoded:np.ndarray=fp_values.astype(np.float16).view(np.uint16)
  else: encoded=np.asarray(expr_lanes).astype(np.int64)&(0xffff if encode is _int16_bits else -1) if encode in (_int16_bits,int) else np.fromiter((encode(value.item()) for value in expr_lanes),dtype=np.int64,count=len(expr_lanes))  # noqa: E501
  dst,values=dst_lanes[order:=np.argsort(dst_lanes)],encoded[order]; starts=np.r_[True,dst[1:]!=dst[:-1]]
  if not np.array_equal(dst[starts],np.arange(count)) or np.any(values[1:][~starts[1:]]!=values[:-1][~starts[1:]]): raise RuntimeError("RKPLAN_REJECT:static_index")  # noqa: E501
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
  dst,src,*mask=_static_lanes(out_index,load_index,*((gate,) if gate is not None else ()))
  dst,src=dst.astype(np.int64),src.astype(np.int64); values=src if gate is None else np.where(active:=mask[0],src,-1)
  if np.any((src<0)&(gate is None or active)) or np.any(dst<0) or np.any(dst>=count) or not np.all(np.bincount(dst,minlength=count)): raise RuntimeError("RKPLAN_REJECT:gather_index")  # noqa: E501
  offsets=np.full(count,-1,dtype=np.int64); offsets[dst]=values
  return tuple(offsets.tolist())

def _affine_output_axes(affine:tuple[int, dict[UOp, int]], count:int) -> tuple[tuple[UOp, int, int], ...]|None:
  ordered = tuple(sorted(affine[1].items(), key=lambda item:item[1]))
  limits = tuple(int(r.src[0].arg) if r.src and r.src[0].op is Ops.CONST else 0 for r,_ in ordered)
  return tuple((r,stride,limit) for (r,stride),limit in zip(ordered,limits)) if all(limit>0 and stride==math.prod(limits[:i]) for i,((_,stride),limit) in enumerate(zip(ordered,limits))) and math.prod(limits)==count else None  # noqa: E501

def _gather_plan(src_index:int, dst_index:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int, fill_bits:int=0) -> RKGather:
  if gate is None and (out_affine:=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index))) is not None and out_affine[0]==0 and (output_axes:=_affine_output_axes(out_affine,count)) is not None:  # noqa: E501
    if (load_divided:=typing_cast(tuple[int, dict[tuple[UOp, int], int]]|None, _linear_index(load_index, True))) is not None and \
       all(r in out_affine[1] and divisor <= int(r.src[0].arg) for r,divisor in load_divided[1]):
      # Preserve the ordinary affine axis order and object graph; true divided plans retain expression order.
      return RKGather(RKArg(RKBufferKind.ARG,src_index),RKArg(RKBufferKind.SCRATCH,dst_index),count,load_divided[0],tuple((d,l,load_affine[1][r]) for r,d,l in output_axes if load_affine[1].get(r,0)) if (load_affine:=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(load_index))) is not None else  # noqa: E501
        tuple((out_affine[1][r]*divisor,(int(r.src[0].arg)+divisor-1)//divisor,stride) for (r,divisor),stride in load_divided[1].items() if stride))  # noqa: E501
  return RKGather(RKArg(RKBufferKind.ARG,src_index),RKArg(RKBufferKind.SCRATCH,dst_index),count,offsets=_gather_offsets(out_index,load_index,gate,count),fill_bits=fill_bits)  # noqa: E501

def _validate_gather_bounds(plan:RKGather, source_count:int) -> None:
  low,high=(min(plan.offsets,default=0),max(plan.offsets,default=-1)) if plan.offsets else tuple(plan.base+sum(fn((limit-1)*stride,0) for _,limit,stride in plan.axes) for fn in (min,max))  # noqa: E501
  if low < (0 if not plan.offsets else -1) or high >= source_count: raise RuntimeError("RKPLAN_REJECT:gather_index")

class RKTypedLoadPlan(NamedTuple):
  """Typed source metadata shared by static-offset and physical-gather consumers."""
  param:UOp; gather:RKGather

def _typed_load_plan(load:UOp, dtype:DType, out_index:UOp, count:int, *, fill_bits:int|None=None, require_offsets:bool=False) -> RKTypedLoadPlan|None:  # noqa: E501
  if load.op is not Ops.LOAD or load.dtype.scalar() is not dtype or not load.src or load.src[0].op is not Ops.INDEX or len(load.src)>1 and load.src[1].op is not Ops.CONST and fill_bits is None: return None  # noqa: E501
  if (param:=_root_param(load.src[0])) is None or param.dtype.scalar() is not dtype or not param.src or param.src[0].op is not Ops.CONST: return None
  gate,fill_bits=load.src[2] if len(load.src)>2 else None,fill_bits if fill_bits is not None else _fp16_bits(load.src[1].arg if len(load.src)>1 else 0) if dtype is dtypes.half else 0  # noqa: E501
  try:
    gather=_gather_plan(param.arg.slot,0,out_index,load.src[0].src[1],gate,count,fill_bits)
    _validate_gather_bounds(gather,int(param.src[0].arg)); gather=gather._replace(base=0,axes=(),offsets=_gather_offsets(out_index,load.src[0].src[1],gate,count)) if require_offsets else gather  # noqa: E501
  except RuntimeError: return None
  return RKTypedLoadPlan(param, gather)

def _gather_cache_key(plans:Iterable[RKGather]) -> tuple: return tuple(v[0:1]+v[2:] for v in plans)

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
    arg=_NATIVE_PRECISE_ADD if node.op is Ops.ADD and node.dtype.scalar() is dtypes.half else node.arg), tagged)

def _physical_recipe(recipe:UOp, opaque:tuple[UOp, ...]=()) -> UOp: return _tag_precise_adds(recipe.substitute(placeholders:={source:UOp.param(-index-1,source.dtype,()) for index,source in enumerate(opaque)})).substitute({placeholder:source for source,placeholder in placeholders.items()})  # noqa: E501

def _kahan_sum(terms:tuple[UOp,...]|list[UOp]) -> UOp:
  """Accumulate physical HALF terms while retaining the running rounding correction."""
  if any(node.op is Ops.CONST and node.dtype.scalar() in (dtypes.half,dtypes.float,dtypes.weakfloat) and not math.isfinite(float(node.arg)) for term in terms for node in term.toposort()): return _fold_static_terms(Ops.ADD,terms[0].dtype,list(terms),False)  # noqa: E501
  total,correction=terms[0],UOp.const(0.0,dtypes.half)
  for value in terms[1:]: adjusted=value.alu(Ops.SUB,correction); updated=total.alu(Ops.ADD,adjusted); correction=updated.alu(Ops.SUB,total).alu(Ops.SUB,adjusted); total=updated  # noqa: E501
  return _tag_precise_adds(total)

def _kahan_mul_sum(terms:list[UOp]) -> UOp:
  """Accumulate composite products and their TwoProduct residuals in their proven physical order."""
  neg_one,splitter=UOp.const(-1.0,dtypes.half),UOp.const(65.0,dtypes.half); pairs=tuple(_two_product(term,neg_one,splitter) for term in terms)
  return _kahan_sum(tuple(x[0] for x in pairs)+tuple(x[1] for x in pairs))

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a three-half expansion using only DPU EW ops."""
  return _kahan_mul_sum(terms) if all(term.op is Ops.MUL and term.arg is None and term.dtype.scalar() is dtypes.half and any(_strip_cast(source).op is Ops.LOAD for source in term.src) for term in terms) and (len(terms) == 8 and all(all(_strip_cast(source).op is Ops.LOAD for source in term.src) for term in terms) or 64 <= len(terms) <= 512 and any(any(_strip_cast(source).op is not Ops.LOAD for source in term.src) for term in terms)) else _tag_precise_adds((parts:=_precise_sum_parts(terms))[0].alu(Ops.ADD,parts[1]))  # noqa: E501

def _append_inplace_image(first:RKImage, second:RKImage, *, link:tuple[int,RKArg,int]|None=None, chain:bool=False) -> RKImage|None:
  """Append an in-place EW image, linking a produced value through a safe barrier or uninterrupted chain."""
  second_ew=[i for i,op in enumerate(second.program) if isinstance(op,RKEWOp)]
  if not second_ew or any(isinstance(op,RKCMAC) or isinstance(op,RKGather) and op.index is not None for op in second.program): return None
  chained=chain and any(isinstance(op,RKEWOp) for op in first.program) and all(isinstance(second.program[i],RKEWOp) for i in range(second_ew[0],second_ew[-1]+1)) and not any(isinstance(op,RKGather) and op.src is not None and op.src.kind is RKBufferKind.ARG and link is not None and op.src.index==link[0] for op in second.program[:second_ew[0]])  # noqa: E501
  fs=len(first.scratch); second=_map_image_args(second,lambda arg:arg._replace(index=fs+arg.index) if arg.kind is RKBufferKind.SCRATCH else arg)
  if link is not None:
    source_slot,source,count=link; target=source
    if source.kind is RKBufferKind.ARG and not chained:
      target=RKArg(RKBufferKind.SCRATCH,fs+len(second.scratch)); second=second._replace(scratch=second.scratch+(_scratch_bytes(count),),program=(RKGather(source,target,count,axes=((1,count,1),)),)+second.program)  # noqa: E501
    second=_alias_image_args(second,{source_slot:target})
  program=list(second.program); first_ew=next(i for i,op in enumerate(program) if isinstance(op,RKEWOp)); op=typing_cast(RKEWOp,program[first_ew])
  program[first_ew]=op._replace(submit_barrier=not chained)
  if not chained: return RKImage(first.scratch+second.scratch,program=first.program+tuple(program))
  split=next(i for i,op in enumerate(first.program) if isinstance(op,RKEWOp))
  return RKImage(first.scratch+second.scratch,program=first.program[:split]+tuple(program[:first_ew])+first.program[split:]+tuple(program[first_ew:]))  # noqa: E501

def _lower_output_selector(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Compile the two independent surfaces of an outermost static output selector."""
  store,out,count,index,root=output
  if root.op is not Ops.WHERE or len(store.src)!=2 or out.dtype.scalar() is not dtypes.half or not _is_static_expr(root.src[0]): return None  # noqa: E501
  selectors=_index_ranges(root.src[0])
  if count<=0 or count%2 or len(selectors)!=1 or not selectors[0].src or selectors[0].src[0].op is not Ops.CONST or int(selectors[0].src[0].arg)!=2: return None  # noqa: E501
  selector,stride=selectors[0],count//2
  try: choices=tuple(bool(_eval_static(root.src[0],{selector:value})) for value in range(2))
  except (RuntimeError,ValueError,KeyError): return None
  if set(choices)!={False,True}: return None
  slot=1+max((u.arg.slot for u in uops if u.op is Ops.PARAM),default=out.arg.slot); images=[]
  for value,choice in enumerate(choices):
    replacement={selector:selector.const_like(value)}; branch=graph_rewrite(root.src[1 if choice else 2].substitute(replacement),sym); branch_index=graph_rewrite(index.substitute(replacement).alu(Ops.SUB,index.const_like(value*stride)),sym)  # noqa: E501
    fake=UOp.param(slot,out.dtype,(stride,)); branch_store=fake.index(branch_index).store(branch)
    if (image:=_lower_uop_program(list(branch_store.sink().toposort()))) is None: return None
    target=RKArg(RKBufferKind.SCRATCH,len(image.scratch)); image=_alias_image_args(image,{slot:target}); commit=RKGather(target,RKArg(RKBufferKind.ARG,out.arg.slot),stride,  # noqa: E501
      axes=((1,stride,1),),dst_addend=value*stride)
    images.append(image._replace(scratch=image.scratch+(_scratch_bytes(stride),),program=image.program+(commit,)))
  if sum(any(isinstance(op,RKCMAC) for op in image.program) for image in images)>1: return None
  return _append_inplace_image(*typing_cast(tuple[RKImage,RKImage],tuple(sorted(images,key=lambda image:any(isinstance(op,RKCMAC) for op in image.program)))))  # noqa: E501

def _lower_cmac_storage_epilogue(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Commit one output-shaped FP32 contraction to HALF on CMAC before its ordinary HALF epilogue."""
  store,out,count,index,root=output
  fake_slot=1+max((u.arg.slot for u in uops if u.op is Ops.PARAM),default=out.arg.slot); fake=UOp.param(fake_slot,dtypes.half,(count,))
  for boundary in (u for u in root.toposort() if u is not root and _typed_cast_source(u,dtypes.half,dtypes.float) is not None):
    source=typing_cast(UOp,_typed_cast_source(boundary,dtypes.half,dtypes.float)); terms=tuple(_strip_cast(term) for term in _iter_binary(source,Ops.ADD)) if source.op is Ops.ADD else ()  # noqa: E501
    if any(node.op is Ops.REDUCE and isinstance(node.arg,tuple) and node.arg[0] is Ops.ADD and all(axis.src and axis.src[0].op is Ops.CONST for axis in node.src[1:]) and math.prod(int(axis.src[0].arg) for axis in node.src[1:])==8 for node in boundary.toposort()) or len(terms) == 8 and all(term.op is Ops.MUL and term.arg is None and all(src.dtype.scalar() is dtypes.half and _strip_cast(src).op is Ops.LOAD for src in term.src) for term in terms): continue  # noqa: E501
    if (prefix:=_lower_reduction((store.replace(src=(store.src[0],boundary)),out,count,index,boundary),uops)) is None: continue
    suffix_store=store.replace(src=(store.src[0],root.substitute({boundary:fake.index(index).load()})))
    suffix=None if any(_root_param(load.src[0]) is out for load in _semantic_loads(suffix_store)) else _lower_uop_program(list(UOp(Ops.SINK,src=(suffix_store,)).toposort()),vectorize_reductions=False)  # noqa: E501
    if suffix is not None and (combined:=_append_inplace_image(prefix,suffix,link=(fake_slot,RKArg(RKBufferKind.ARG,out.arg.slot),count),
      chain=not any(isinstance(op,RKCMAC) for op in prefix.program))) is not None: return combined  # noqa: E501
  return None

def _iter_binary(root:UOp, op:Ops, dtype:DType|None=None, plain:bool=False) -> Iterable[UOp]:
  stack = [root]
  while stack:
    node = stack.pop()
    if node.op is op and (dtype is None or node.dtype.scalar() is dtype) and (not plain or node.arg is None): stack.extend(reversed(node.src))
    else: yield node

def _gate_zero_term(term:UOp) -> UOp:
  """Move a static zero-select into its load so padded products keep a linear physical carrier."""
  term=_strip_cast(term)
  if term.op is not Ops.WHERE or not _is_static_expr(term.src[0]) or term.src[2].op is not Ops.CONST or float(term.src[2].arg)!=0.0: return term  # noqa: E501
  if (load:=next(iter(_semantic_loads(term.src[1])),None)) is None or (default:=load.src[1] if len(load.src)>1 else load.const_like(0.0)).op is not Ops.CONST or float(default.arg)!=0.0: return term  # noqa: E501
  gate=term.src[0] if len(load.src)<3 else load.src[2].alu(Ops.AND,term.src[0])
  return term.src[1].substitute({load:load.replace(src=(load.src[0],default,gate,*load.src[3:]))})

def _lower_cmac_reduce(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Keep CMAC responsible for separable sums/products; mapped reduction owns every other bounded shape."""
  _,out,rows,out_index,root=output
  if rows<=0 or out.dtype.scalar() not in (dtypes.half,dtypes.float) or any(node.op is Ops.REDUCE and isinstance(node.arg,tuple) and node.arg[0] is Ops.ADD and all(axis.src and axis.src[0].op is Ops.CONST for axis in node.src[1:]) and math.prod(int(axis.src[0].arg) for axis in node.src[1:])>13*32 for node in root.toposort()): return None  # noqa: E501
  relu_root=_relu_operand(root)
  if relu_root is None and (fp32_root:=_typed_cast_source(root,dtypes.half,dtypes.float)) is not None: relu_root=_relu_operand(fp32_root)
  root=_strip_cast(relu_root if relu_root is not None else root); additive=root.op is Ops.ADD and root.dtype.scalar() is dtypes.float or any(node.op is Ops.REDUCE and isinstance(node.arg,tuple) and node.arg[0] is Ops.ADD for node in root.toposort())  # noqa: E501
  try: root = _unroll_static_reduces(root, precise=False)
  except (_RKGenericReject, RuntimeError, ValueError): return None
  scale,exact_scale=1.0,True
  while (pair:=_const_operand(root:=_strip_cast(root),Ops.MUL)) is not None: root,factor=pair[0],float(pair[1].arg); scale*=factor; exact_scale=exact_scale and factor>0.0 and math.frexp(factor)[0]==0.5 and float_to_fp16(scale)==scale  # noqa: E501
  terms=tuple(_gate_zero_term(term) for term in _iter_binary(root,Ops.ADD)) if root.op is Ops.ADD else (_gate_zero_term(root),) if additive else (); terms=tuple(term for term in terms if not (term.op is Ops.CONST and float(term.arg)==0.0)); groups=len(terms)  # noqa: E501
  if groups < (1 if additive else 4) or groups>13*32: return None
  parsed:list[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float]]=[]
  for term in terms:
    factors=tuple(map(_strip_cast,_iter_binary(_strip_cast(term),Ops.MUL,plain=True))); constants=tuple(node for node in factors if node.op is Ops.CONST); loads=tuple(node for node in factors if node.op is not Ops.CONST); weight=scale*math.prod(float(node.arg) for node in constants)  # noqa: E501
    plans=tuple(_typed_load_plan(load,dtypes.half,out_index,rows) for load in loads)
    if len(constants)>2 or len(constants)>1 and term.dtype.scalar() is not dtypes.float or len(loads)>2 or not exact_scale or any(float_to_fp16(float(node.arg))!=float(node.arg) for node in constants) or not math.isfinite(weight) or len(loads)<2 and float_to_fp16(weight)!=weight or len(loads)==2 and weight!=1.0 or out.dtype.scalar() is dtypes.float and rows==1 and len(loads)==1 and weight==1.0 or any(len(load.src)>1 and (load.src[1].op is not Ops.CONST or float(load.src[1].arg)!=0.0 or math.copysign(1.0,float(load.src[1].arg))<0.0) for load in loads) or any(plan is None for plan in plans): return None  # noqa: E501
    valid=typing_cast(tuple[RKTypedLoadPlan,...],plans); parsed.append((valid[0] if valid else None,valid[1] if len(valid)>1 else None,weight))
  load_pairs=tuple(typing_cast(tuple[RKTypedLoadPlan,RKTypedLoadPlan],pair[:2]) for pair in parsed) if all(pair[0] is not None and pair[1] is not None for pair in parsed) else ()  # noqa: E501
  for n in range(32,rows+1,32) if groups%32==0 else ():
    if rows%n: continue
    m=rows//n
    if n>0x3fff or m>0x7ff or m*groups*2>10*32768 or m!=1 and groups>12*32 or m*groups+n*groups+rows>_MAX_DYNAMIC_SELECTOR_CELLS: continue  # noqa: E501
    for lhs0,rhs0 in (load_pairs[0],load_pairs[0][::-1]) if load_pairs else ():
      if not all(any(lhs.param is lhs0.param and rhs.param is rhs0.param and not lhs.gather.offsets and lhs.gather.base==lhs0.gather.base+k and lhs.gather.axes==((n,m,groups),) and not rhs.gather.offsets and rhs.gather.base==rhs0.gather.base+k*n and rhs.gather.axes==((1,n,1),) for lhs,rhs in (pair,pair[::-1])) for k,pair in enumerate(load_pairs)): continue  # noqa: E501
      lhs=RKGather(RKArg(RKBufferKind.ARG,lhs0.param.arg.slot),RKArg(RKBufferKind.SCRATCH,0),m*groups,base=lhs0.gather.base,axes=((1,m*groups,1),))  # noqa: E501
      rhs=RKGather(RKArg(RKBufferKind.ARG,rhs0.param.arg.slot),RKArg(RKBufferKind.SCRATCH,1),n*groups,base=rhs0.gather.base,axes=((groups*16,n//16,16),(512,groups//32,32*n),(32,16,1),(1,32,n)))  # noqa: E501
      fp16=out.dtype.scalar() is dtypes.half; cmac=RKCMAC(RKArg(RKBufferKind.SCRATCH,2),lhs.dst,rhs.dst,m,n,groups,fp16,relu_root is not None)
      commit=RKGather(cmac.dst,RKArg(RKBufferKind.ARG,out.arg.slot),rows,axes=((n,m,n*2),(16,n//16,32),(1,16,1)) if fp16 else ((1,rows,1),),itemsize=2 if fp16 else 4)  # noqa: E501
      return RKImage((m*groups*2,n*groups*2,m*n*4),program=(lhs,rhs,cmac,commit))
  patterns:dict[tuple,np.ndarray]={}
  def plan_offsets(plan:RKTypedLoadPlan) -> np.ndarray:
    if (key:=(plan.gather.axes,plan.gather.offsets)) not in patterns: patterns[key]=np.asarray(plan.gather.offsets,dtype=np.int64) if plan.gather.offsets else sum((np.arange(rows,dtype=np.int64)//divisor%limit*stride for divisor,limit,stride in plan.gather.axes),np.zeros(rows,dtype=np.int64))  # noqa: E501
    return patterns[key]
  def align(m:int,n:int,lanes:tuple[int,...]) -> tuple[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float],...]:
    aligned:list[tuple[RKTypedLoadPlan|None,RKTypedLoadPlan|None,float]]=[]; logical=np.asarray(lanes or range(rows),dtype=np.int64).reshape(m,n)  # noqa: E501
    for lhs,rhs,weight in parsed:
      plans=typing_cast(tuple[RKTypedLoadPlan,...],tuple(plan for plan in (lhs,rhs) if plan is not None))
      if not plans: aligned.append((None,None,weight)); continue
      row,col=zip(*((bool(np.all((grid:=plan_offsets(plan)[logical])==grid[:,:1])),bool(np.all(grid==grid[:1,:]))) for plan in plans))
      order=(0,None) if len(plans)==1 and row[0] else (None,0) if len(plans)==1 and col[0] else (0,1) if len(plans)==2 and row[0] and col[1] else (1,0) if len(plans)==2 and row[1] and col[0] else None  # noqa: E501
      if order is None: return ()
      aligned.append((None if order[0] is None else plans[order[0]],None if order[1] is None else plans[order[1]],weight))
    return tuple(aligned)
  out_affine=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index)); output_axes=(_affine_output_axes(out_affine,rows) if out_affine is not None else None) or (); views:list[tuple[int,int,tuple[int,...],tuple[int,...]]]=[(rows,1,(),()),(1,rows,(),())]+[(m,n,lanes,tuple(map(int,np.argsort(lanes)))) for lhs,rhs in ((load_pairs[0],load_pairs[0][::-1]) if load_pairs else ()) for left,right in ((plan_offsets(lhs),plan_offsets(rhs)),) for m,n in ((len(np.unique(left)),len(np.unique(right))),) for lanes in (tuple(map(int,np.lexsort((right,left)))),) if rows>_MAX_GENERIC_UNROLL and m*n==rows]  # noqa: E501
  for _,stride,limit in output_axes:
    m,n=limit,rows//limit; lanes=tuple(high*stride*limit+row*stride+low for row in range(limit) for high in range(rows//stride//limit) for low in range(stride)); outputs=tuple((i//stride%limit)*n+i//(stride*limit)*stride+i%stride for i in range(rows)); views.append((m,n,() if lanes==tuple(range(rows)) else lanes,() if outputs==tuple(range(rows)) else outputs))  # noqa: E501
  candidates=[(m,n,lanes,outputs,aligned,ai,ao) for m,n,lanes,outputs in views for aligned in (align(m,n,lanes),) for ai,ao,_ in (_cmac_layout(n,groups),) if aligned and m<=0x7ff and ai<=13*32 and ao<=0x3fff and m*ai*2<=10*32768 and (m==1 or ai<=12*32)]  # noqa: E501
  diagonal=not candidates; m,n,lanes,outputs,shape_terms,ai,ao=min(candidates,key=lambda shape:(shape[0]==1 and rows>1,shape[0]*shape[5]+shape[6]*shape[5]+2*shape[0]*shape[6])) if candidates else (rows,rows,(),(),tuple(parsed),*_cmac_layout(rows,groups)[:2])  # noqa: E501
  if m>0x7ff or ai>13*32 or ao>0x3fff or m*ai*2>10*32768 or m!=1 and ai>12*32: return None
  packed_a=tuple(((plan.param.arg.slot,int(plan_offsets(plan)[row if diagonal else lanes[row*n] if lanes else row*n])+(0 if plan.gather.offsets else plan.gather.base)) if (plan:=shape_terms[k][0]) is not None else (None,_fp16_bits(1.0 if shape_terms[k][1] is None else shape_terms[k][2]))) if k<groups else (None,0) for row in range(m) for k in range(ai))  # noqa: E501
  packed_b=tuple(((plan.param.arg.slot,int(plan_offsets(plan)[col if diagonal else lanes[col] if lanes else col])+(0 if plan.gather.offsets else plan.gather.base)) if (plan:=shape_terms[k][1]) is not None else (None,_fp16_bits(shape_terms[k][2]))) if col<n and (k:=ib*32+ki)<groups else (None,0) for ob in range(ao//16) for ib in range(ai//32) for ni in range(16) for ki in range(32) for col in (ob*16+ni,))  # noqa: E501
  gathers=[]
  for dst,packed in enumerate((packed_a,packed_b)):
    sources=tuple(dict.fromkeys(owner for owner,_ in packed if owner is not None)); values=tuple(value if owner is None else 0 for owner,value in packed); seeded=not sources or any(values)  # noqa: E501
    if seeded: gathers.append(RKGather(None,RKArg(RKBufferKind.SCRATCH,dst),len(packed),values=(values[0],) if len(set(values))==1 else values))
    gathers.extend(RKGather(RKArg(RKBufferKind.ARG,source),RKArg(RKBufferKind.SCRATCH,dst),len(packed),offsets=tuple(value if owner==source else -1 for owner,value in packed),partial=seeded or bool(i)) for i,source in enumerate(sources))  # noqa: E501
  if sum(gather.count for gather in gathers)+rows>_MAX_DYNAMIC_SELECTOR_CELLS: return None
  fp16=out.dtype.scalar() is dtypes.half; cmac=RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),m,n,groups,fp16,relu_root is not None)  # noqa: E501
  output_offsets=tuple(row*ao*(2 if fp16 else 1)+(col//16*32+col%16 if fp16 else col) for position in (tuple(i*rows+i for i in range(rows)) if diagonal else outputs or tuple(range(rows))) for row,col in (divmod(position,n),))  # noqa: E501
  commit=RKGather(cmac.dst,RKArg(RKBufferKind.ARG,out.arg.slot),rows,offsets=output_offsets,itemsize=2 if fp16 else 4)
  return RKImage((m*ai*2,ao*ai*2,m*ao*4),program=(*gathers,cmac,commit))

def _lower_one_hot_gather(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Collapse a canonical complete one-hot ADD reduction to one NPU-addressed raw gather."""
  store,out,rows,_,root=output; reductions=tuple(node for node in root.toposort() if node.op is Ops.REDUCE); value=reductions[0] if len(reductions)==1 else None  # noqa: E501
  if rows<1 or rows>_RKIMAGE_U16_MAX or out.dtype.scalar() is not dtypes.half or value is None or not isinstance(value.arg,tuple) or value.arg[0] is not Ops.ADD: return None  # noqa: E501
  ranges=tuple(value.src[1:]); body=_strip_cast(value.src[0]); source=body.src[1].load() if body.op is Ops.WHERE and body.src[1].op is Ops.INDEX else body.src[1] if body.op is Ops.WHERE else body  # noqa: E501
  if not ranges or any(axis.op not in (Ops.RANGE,Ops.SPECIAL) or not axis.src or axis.src[0].op is not Ops.CONST for axis in ranges) or body.op is not Ops.WHERE or body.src[2].op is not Ops.CONST or float(body.src[2].arg)!=0.0 or source.op is not Ops.LOAD or source.dtype.scalar() is not dtypes.half or len(source.src)!=1 or source.src[0].op is not Ops.INDEX or (param:=_root_param(source.src[0])) is None or param.src[0].op is not Ops.CONST: return None  # noqa: E501
  selections:dict[UOp,UOp]={}
  for predicate in _iter_binary(body.src[0],Ops.AND):
    if predicate.op is not Ops.CMPNE or len(tuple(node for node in predicate.src if node.op is Ops.CONST and node.dtype.scalar() is dtypes.bool and bool(node.arg)))!=1 or len(unequal:=tuple(node for node in predicate.src if node.op is Ops.CMPNE))!=1 or len(unequal[0].src)!=2: return None  # noqa: E501
    if len(matches:=tuple((axis,other) for axis in ranges for candidate,other in (unequal[0].src,unequal[0].src[::-1]) if _strip_cast(candidate).key==axis.key and other.dtype.scalar() is dtypes.int and not any(item in ranges for item in other.toposort())))!=1 or matches[0][0] in selections or not _semantic_loads(matches[0][1]): return None  # noqa: E501
    selections[matches[0][0]]=matches[0][1]
  if set(selections)!=set(ranges): return None
  valid=functools.reduce(lambda x,y:x.alu(Ops.AND,y),tuple(selected.alu(Ops.CMPLT,selected.const_like(0)).alu(Ops.CMPNE,UOp.const(True,dtypes.bool)).alu(Ops.AND,selected.alu(Ops.CMPLT,selected.const_like(int(axis.src[0].arg)))) for axis,selected in selections.items())); index=graph_rewrite(source.src[0].src[1].substitute(selections,walk=True),sym); direct=source.src[0].replace(src=(source.src[0].src[0],index)).load(body.src[2],valid); replacement=root.substitute({value:direct})  # noqa: E501
  return _lower_uop_program(list(store.replace(src=(store.src[0],replacement)).sink().toposort()),vectorize_reductions=False)

def _lower_int_equality_count(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Count exact INT32 candidates equal to each bounded output lane without repeating candidate arithmetic."""
  _,out,count,out_index,root=output
  if out.dtype.scalar() is not dtypes.int or not 1<=count<=_FP16_EXACT_INTEGER or root.op is not Ops.REDUCE or not isinstance(root.arg,tuple) or root.arg[0] is not Ops.ADD or len(root.src)!=2: return None  # noqa: E501
  body,axis=root.src
  if not axis.src or axis.src[0].op is not Ops.CONST or not 1<=(candidates:=int(axis.src[0].arg))<=_FP16_EXACT_INTEGER or body.op is not Ops.WHERE: return None  # noqa: E501
  condition,zero,one=body.src
  if condition.op is not Ops.CMPNE or zero.op is not Ops.CONST or int(zero.arg)!=0 or one.op is not Ops.CONST or int(one.arg)!=1: return None
  pair=next(((candidate,lane) for candidate,lane in (condition.src,condition.src[::-1]) if _strip_cast(lane).key==out_index.key),None)
  ranges=() if pair is None else tuple(node for node in pair[0].toposort() if node.op in (Ops.RANGE,Ops.SPECIAL))
  if pair is None or pair[0].dtype.scalar() is not dtypes.int or set(ranges)!={axis}: return None
  candidate=pair[0]; lane=UOp.range(candidates,1+max((node.arg[0] for node in uops if node.op is Ops.RANGE and isinstance(node.arg,tuple)),default=-1),dtype=dtypes.int)  # noqa: E501
  slot=1+max((node.arg.slot for node in uops if node.op is Ops.PARAM),default=out.arg.slot); fake=UOp.param(slot,dtypes.int,(candidates,))
  try:
    if _static_values(out_index,out_index,count,int)!=tuple(range(count)): return None
    candidate_image=_lower_uop_program(list(fake.index(lane).store(candidate.substitute({axis:lane},walk=True)).end(lane).sink().toposort()))
  except (RuntimeError,ValueError,OverflowError): return None
  if candidate_image is None: return None
  size=1<<(candidates-1).bit_length(); target=RKArg(RKBufferKind.SCRATCH,len(candidate_image.scratch))
  candidate_image=_alias_image_args(candidate_image._replace(scratch=candidate_image.scratch+(size*4,)),{slot:target}); split=next((index for index,op in enumerate(candidate_image.program) if not isinstance(op,RKGather)),len(candidate_image.program))  # noqa: E501
  if any(not isinstance(op,RKGather) for op in candidate_image.program[:split]) or any(not isinstance(op,RKEWOp) for op in candidate_image.program[split:]): return None  # noqa: E501
  scratch=list(candidate_image.scratch); candidate_preloads=list(typing_cast(tuple[RKGather,...],candidate_image.program[:split])); preloads:list[RKGather]=[]; ops:list[RKEWOp]=list(typing_cast(tuple[RKEWOp,...],candidate_image.program[split:])); cut=len(ops)  # noqa: E501
  def alloc(amount:int,addend:int=0) -> RKArg:
    scratch.append(max(64,amount)); return RKArg(RKBufferKind.SCRATCH,len(scratch)-1,addend)
  def emit(lhs:RKArg,rhs:RKArg,lanes:int,cfg:int,dst:RKArg|None=None) -> RKArg:
    if dst is None: dst=alloc(lanes*2)
    ops.append(RKEWOp(dst,lhs,rhs,lanes,cfg,mode=RKEWMode.INT16)); return dst
  block=math.ceil(count/32)*32; limit=1<<((_MAX_EW_ELEMS_FP16//block).bit_length()-1); accumulator=None
  for start in range(0,candidates,limit):
    rows=min(limit,candidates-start); chunk_size=1<<(rows-1).bit_length(); lanes=chunk_size*block
    unit,match=alloc(lanes*2),alloc(lanes*2)
    preloads.extend((RKGather(None,unit,lanes,values=(1,)),RKGather(None,match,lanes,values=tuple(1 if index//block<rows else 0 for index in range(lanes)))))  # noqa: E501
    for byte in range(4):
      dynamic,static,difference,equal=(alloc(lanes*2) for _ in range(4))
      preloads.extend((RKGather(target,dynamic,lanes,base=start*4+byte,axes=((block,chunk_size,4),),dst_stride=2,itemsize=1),RKGather(None,static,lanes,values=tuple(((index%block)>>(byte*8))&0xff if index%block<count else 0 for index in range(lanes)))))  # noqa: E501
      emit(dynamic,static,lanes,_EW_CFG[Ops.SUB],difference); emit(difference,difference,lanes,_EW_CFG_ABS,difference); emit(difference,unit,lanes,_EW_CFG_MIN,difference); emit(unit,difference,lanes,_EW_CFG[Ops.SUB],equal); emit(match,equal,lanes,_EW_CFG[Ops.MUL],match)  # noqa: E501
    active=chunk_size//2
    while active: emit(match,match._replace(addend=active*block*2),active*block,_EW_CFG[Ops.ADD],match); active//=2
    accumulator=match if accumulator is None else emit(accumulator,match,count,_EW_CFG[Ops.ADD],accumulator)
  if accumulator is None: return None
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG,out.arg.slot),accumulator,accumulator,count,_EW_CFG[Ops.MAX],mode=RKEWMode.INT16_TO_INT32))
  image=_reuse_linear_scratch(RKImage(tuple(scratch),tuple(candidate_preloads)+tuple(ops[:cut])+tuple(preloads)+tuple(ops[cut:])))
  try: _validate_image(image)
  except ValueError: return None
  return image

def _lower_bounded_int_lookup(output:RKOutput) -> RKImage|None:
  """Select a static INT16-valued row by an exact, range-gated runtime INT32 index."""
  _,out,count,out_index,root=output
  if out.dtype.scalar() is not dtypes.int or not 1<=count<=_FP16_EXACT_INTEGER or root.op is not Ops.WHERE or len(root.src)!=3 or root.src[2].op is not Ops.CONST or int(root.src[2].arg)!=0: return None  # noqa: E501
  gate,value=root.src[:2]
  if gate.op is not Ops.AND or len(gate.src)!=2: return None
  upper=next((term for term in gate.src if term.op is Ops.CMPLT and term.src[1].op is Ops.CONST and 0<int(term.src[1].arg)<=_FP16_EXACT_INTEGER),None)
  if upper is None: return None
  source,limit=upper.src[0],int(upper.src[1].arg); nonnegative=next((term for term in gate.src if term is not upper),None)
  if nonnegative is None or nonnegative.op is not Ops.CMPNE or not any(mark.op is Ops.CONST and mark.dtype.scalar() is dtypes.bool and bool(mark.arg) for mark in nonnegative.src): return None  # noqa: E501
  negative=next((term for term in nonnegative.src if term.op is Ops.CMPLT),None)
  if negative is None or negative.src[0].key!=source.key or negative.src[1].op is not Ops.CONST or int(negative.src[1].arg)!=0: return None
  if source.op is not Ops.LOAD or tuple(node for node in root.toposort() if node.op is Ops.LOAD)!=(source,): return None
  try:
    plan=_typed_load_plan(source,dtypes.int,out_index,count,require_offsets=True)
    if plan is None or plan.gather.src is None or plan.gather.offsets!=tuple(range(count)): return None
    rows=tuple(_static_values(out_index,value.substitute({source:source.const_like(candidate)},walk=True),count,int) for candidate in range(limit))
  except (RuntimeError,ValueError,OverflowError): return None
  if any(not -32768<=item<=32767 for row in rows for item in row): return None
  scratch:list[int]=[]; preloads:list[RKGather]=[]; ops:list[RKEWOp]=[]
  def alloc(amount:int,addend:int=0) -> RKArg:
    scratch.append(max(64,amount)); return RKArg(RKBufferKind.SCRATCH,len(scratch)-1,addend)
  def emit(lhs:RKArg,rhs:RKArg,lanes:int,cfg:int,dst:RKArg|None=None) -> RKArg:
    if dst is None: dst=alloc(lanes*2)
    ops.append(RKEWOp(dst,lhs,rhs,lanes,cfg,mode=RKEWMode.INT16)); return dst
  block=math.ceil(count/32)*32; chunk_limit=1<<((_MAX_EW_ELEMS_FP16//block).bit_length()-1); accumulator=None
  for start in range(0,limit,chunk_limit):
    active_rows=min(chunk_limit,limit-start); size=1<<(active_rows-1).bit_length(); lanes=size*block
    unit,match,values=alloc(lanes*2),alloc(lanes*2),alloc(lanes*2)
    preloads.extend((RKGather(None,unit,lanes,values=(1,)),RKGather(None,match,lanes,values=tuple(1 if index//block<active_rows else 0 for index in range(lanes))),RKGather(None,values,lanes,values=tuple(rows[start+index//block][index%block] if index//block<active_rows and index%block<count else 0 for index in range(lanes)))))  # noqa: E501
    for byte in range(4):
      dynamic,static,difference,equal=(alloc(lanes*2) for _ in range(4))
      preloads.extend((RKGather(plan.gather.src,dynamic,lanes,offsets=tuple((index%block)*4+byte if index%block<count else -1 for index in range(lanes)),dst_stride=2,itemsize=1),RKGather(None,static,lanes,values=tuple(((start+index//block)>>(byte*8))&0xff if index//block<active_rows else 0 for index in range(lanes)))))  # noqa: E501
      emit(dynamic,static,lanes,_EW_CFG[Ops.SUB],difference); emit(difference,difference,lanes,_EW_CFG_ABS,difference); emit(difference,unit,lanes,_EW_CFG_MIN,difference); emit(unit,difference,lanes,_EW_CFG[Ops.SUB],equal); emit(match,equal,lanes,_EW_CFG[Ops.MUL],match)  # noqa: E501
    emit(values,match,lanes,_EW_CFG[Ops.MUL],values); active=size//2
    while active: emit(values,values._replace(addend=active*block*2),active*block,_EW_CFG[Ops.ADD],values); active//=2
    accumulator=values if accumulator is None else emit(accumulator,values,count,_EW_CFG[Ops.ADD],accumulator)
  if accumulator is None: return None
  ops.append(RKEWOp(RKArg(RKBufferKind.ARG,out.arg.slot),accumulator,accumulator,count,_EW_CFG[Ops.MAX],mode=RKEWMode.INT16_TO_INT32))
  image=_reuse_linear_scratch(RKImage(tuple(scratch),tuple(preloads)+tuple(ops)))
  try: _validate_image(image)
  except ValueError: return None
  return image

def _lower_reduction(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Prefer one dynamic gather or contraction, then map and reduce every remaining bounded reduction on the DPU."""
  return _lower_int_equality_count(output,uops) or _lower_bounded_int_lookup(output) or _lower_one_hot_gather(output,uops) or _lower_cmac_reduce(output,uops) or _lower_mapped_reduce(output,uops)  # noqa: E501

def _reduce_mapped_rows(ops:list[RKEWOp], scratch:list[int], gathers:list[RKGather], source:RKArg, lanes:int, cfg:int, rows:int=1, int16:bool=False, kahan:bool=False, pairwise:bool=False, barrier:bool=True) -> RKArg:  # noqa: E501
  """Reduce one mapped surface through atom-aligned carriers; bit reversal retains the balanced tree order."""
  block=8 if rows==1 else round_up(rows,8); groups=lanes if rows==1 else lanes//block
  if pairwise:
    size=1<<(groups-1).bit_length(); high,target,low,next_low,temporary,virtual=(RKArg(RKBufferKind.SCRATCH,len(scratch)+i) for i in range(6)); scratch.extend((_scratch_bytes(size*block),)*6); bits=size.bit_length()-1; offsets=tuple(index if (index:=int(f"{lane:0{bits}b}"[::-1],2))<groups else -1 for lane in range(size)); offsets=offsets if rows==1 else tuple(index*block+row if index>=0 else -1 for index in offsets for row in range(block)); gathers.extend((RKGather(source,high,len(offsets),offsets=offsets,fill_bits=0,dst_stride=block if rows==1 else 1),RKGather(None,low,size*block,values=(0,))))  # noqa: E501
    while size>1: size//=2; count=size*block; right,low_right=high._replace(addend=count*2),low._replace(addend=count*2); ops.extend((RKEWOp(target,high,right,count,cfg,submit_barrier=barrier,mode=RKEWMode.STATEFUL if barrier else RKEWMode.HALF),RKEWOp(temporary,target,high,count,_EW_CFG[Ops.SUB]),RKEWOp(virtual,target,temporary,count,_EW_CFG[Ops.SUB]),RKEWOp(virtual,high,virtual,count,_EW_CFG[Ops.SUB]),RKEWOp(temporary,right,temporary,count,_EW_CFG[Ops.SUB]),RKEWOp(next_low,virtual,temporary,count,cfg),RKEWOp(temporary,low,low_right,count,cfg),RKEWOp(next_low,next_low,temporary,count,cfg),RKEWOp(virtual,target,next_low,count,cfg),RKEWOp(temporary,virtual,target,count,_EW_CFG[Ops.SUB]),RKEWOp(next_low,next_low,temporary,count,_EW_CFG[Ops.SUB]))); high,target,virtual=virtual,high,target; low,next_low=next_low,low  # noqa: E501
    ops.append(RKEWOp(target,high,low,block if rows==1 else rows,cfg)); return target
  if kahan: arena=RKArg(RKBufferKind.SCRATCH,len(scratch)); total,updated,correction,adjusted=(RKArg(RKBufferKind.SCRATCH,len(scratch)+i+1) for i in range(4)); scratch.extend((_scratch_bytes(groups*block),*(_scratch_bytes(rows),)*4)); gathers.extend((RKGather(source,arena,lanes,axes=((1,lanes,1),),dst_stride=block if rows==1 else 1),RKGather(None,correction,rows,values=(0,)))); ops.append(RKEWOp(total,arena,arena,rows,_EW_CFG[Ops.MAX],submit_barrier=barrier,mode=RKEWMode.STATEFUL))  # noqa: E501
  for index in range(1,groups) if kahan else ():
    value=arena._replace(addend=index*block*2); ops.append(RKEWOp(adjusted,value,correction,rows,_EW_CFG[Ops.SUB])); ops.append(RKEWOp(updated,total,adjusted,rows,_EW_CFG[Ops.ADD])); ops.append(RKEWOp(correction,updated,total,rows,_EW_CFG[Ops.SUB])); ops.append(RKEWOp(correction,correction,adjusted,rows,_EW_CFG[Ops.SUB])); total,updated=updated,total  # noqa: E501
  if kahan: return total
  size=1<<(groups-1).bit_length(); current,target=(RKArg(RKBufferKind.SCRATCH,len(scratch)+i) for i in range(2)); scratch.extend((_scratch_bytes(size*block),)*2); bits=size.bit_length()-1; offsets=tuple(index if (index:=int(f"{lane:0{bits}b}"[::-1],2))<groups else -1 for lane in range(size)); neutral=0 if cfg==_EW_CFG[Ops.ADD] else (1 if int16 else _fp16_bits(1)) if cfg==_EW_CFG[Ops.MUL] else _int16_bits(dtypes.int16.min) if int16 else _fp16_bits(-math.inf); offsets=offsets if rows==1 else tuple(index*block+row if index>=0 else -1 for index in offsets for row in range(block)); gathers.append(RKGather(source,current,len(offsets),offsets=offsets,fill_bits=neutral,dst_stride=block if rows==1 else 1)); first=barrier and not int16  # noqa: E501
  while size>1: size//=2; count=size*block; ops.append(RKEWOp(target,current,current._replace(addend=current.addend+count*2),count,cfg,submit_barrier=first,mode=RKEWMode.INT16 if int16 else RKEWMode.STATEFUL if first else RKEWMode.HALF)); first=False; current,target=target,current  # noqa: E501
  return current

def _reduce_boolean_rows(ops:list[RKEWOp], scratch:list[int], gathers:list[RKGather], source:RKArg, lanes:int, cfg:int, rows:int) -> RKArg:
  """Reduce each dense boolean row in place using exact INT16 carriers."""
  groups=lanes//rows; block=round_up(rows,8); size=1<<(groups-1).bit_length(); current,target=(RKArg(RKBufferKind.SCRATCH,len(scratch)+i) for i in range(2)); scratch.extend((_scratch_bytes(size*block),)*2)  # noqa: E501
  bits=size.bit_length()-1; offsets=tuple(row*groups+index if (index:=int(f"{group:0{bits}b}"[::-1],2))<groups and row<rows else -1 for group in range(size) for row in range(block)); gathers.append(RKGather(source,current,len(offsets),offsets=offsets,fill_bits=1 if cfg==_EW_CFG[Ops.MUL] else 0))  # noqa: E501
  while size>1: size//=2; count=size*block; ops.append(RKEWOp(target,current,current._replace(addend=current.addend+count*2),count,cfg,mode=RKEWMode.INT16)); current,target=target,current  # noqa: E501
  return current

def _lower_mapped_reduce(output:RKOutput, uops:list[UOp]) -> RKImage|None:
  """Render one canonical mapped reduction, reduce it physically, then compile its dependent scalar suffix."""
  store,out,rows,out_index,root=output
  if rows<1 or out.dtype.scalar() not in (dtypes.half,dtypes.int,dtypes.bool): return None
  reductions=tuple(node for node in root.toposort() if node.op is Ops.REDUCE and isinstance(node.arg,tuple) and node.arg[0] in (Ops.ADD,Ops.MAX,Ops.MUL)); nested={child for value in reductions for child in value.src[0].toposort() if child is not value and child.op is Ops.REDUCE}  # noqa: E501
  if not (outer:=tuple(value for value in reductions if value not in nested)): return None
  value=outer[0]; body=value.src[0]; ranges=list(value.src[1:])
  while (inner:=_strip_cast(body)).op is Ops.REDUCE and isinstance(inner.arg,tuple) and inner.arg[0] is value.arg[0]:
    body=inner.src[0]; ranges.extend(inner.src[1:])
  if out.dtype.scalar() in (dtypes.int,dtypes.half) and any(node.op is Ops.REDUCE for node in body.toposort()):
    leaves=tuple(reduction for reduction in reductions if not any(node.op is Ops.REDUCE for node in reduction.src[0].toposort()))
    if not leaves or out.dtype.scalar() is dtypes.int and len(leaves)!=1: return None
    value=leaves[0]; body=value.src[0]; ranges=list(value.src[1:]); external=tuple(axis for axis in body.toposort() if axis.op in (Ops.RANGE,Ops.SPECIAL) and axis not in ranges) if out.dtype.scalar() is not dtypes.int else (); rows,out_index=(int(external[0].src[0].arg),external[0]) if len(external)==1 and external[0].src and external[0].src[0].op is Ops.CONST else (rows,out_index) if out.dtype.scalar() is dtypes.int else (0,out_index)  # noqa: E501
  if rows<1 or not ranges or any(axis.op not in (Ops.RANGE,Ops.SPECIAL) or not axis.src or axis.src[0].op is not Ops.CONST for axis in ranges): return None  # noqa: E501
  loaded_indices={node.src[0] for node in body.toposort() if node.op is Ops.LOAD}; body=body.substitute({node:node.load() for node in body.toposort() if node.op is Ops.INDEX and node not in loaded_indices},walk=True); extents=tuple(int(axis.src[0].arg) for axis in ranges); total=math.prod(extents); graph=body.toposort(); loads=_semantic_loads(body); unit_sum=total<=_FP16_EXACT_INTEGER and (candidate:=_strip_cast(body)).op is Ops.WHERE and _is_static_expr(candidate.src[0]) and all(src.op is Ops.CONST for src in candidate.src[1:]) and {float(src.arg) for src in candidate.src[1:]}<={0.0,1.0}  # noqa: E501
  if not 2<=total<=_MAX_GENERIC_UNROLL or rows>16 and out.dtype.scalar() is dtypes.half and value.arg[0] is Ops.ADD and total>416 and (total*round_up(rows,8)>_MAX_GENERIC_UNROLL or not (any(node.op is Ops.WHERE and _is_static_expr(node.src[0]) for node in graph) or any(len(load.src)>2 and _is_static_expr(load.src[2]) for load in loads))) or not loads and not unit_sum: return None  # noqa: E501
  try: converted=body if out.dtype.scalar() in (dtypes.int,dtypes.bool) else _fp32_expr_to_half(body); product=body if out.dtype.scalar() in (dtypes.int,dtypes.bool) else _strip_cast(converted)  # noqa: E501
  except _RKGenericReject: product=_strip_cast(body)
  boolean=product.dtype.scalar() is dtypes.bool; short_math=value.arg[0] is Ops.ADD and total==16 and rows<=4096 and body.op is Ops.EXP2 and body.dtype.scalar() is dtypes.float and product.op is Ops.EXP2 and product.dtype.scalar() is dtypes.half; square=product.op is Ops.MUL and product.src[0] is product.src[1]; integer=boolean or dtypes.is_int(product.dtype.scalar()); bounds=(0,1) if boolean else _int_info(product)[0] if integer else None; bounded_sum=value.arg[0] is Ops.ADD and rows>1 and out.dtype.scalar() is dtypes.int and bounds is not None and -32768<=total*bounds[0]<=total*bounds[1]<=32767  # noqa: E501
  if boolean and out.dtype.scalar() is not dtypes.bool or product.dtype.scalar() is not dtypes.half and (bounds is None or not -32768<=bounds[0]<=bounds[1]<=32767) or len(reductions)==1 and (total<32 and not bounded_sum and not short_math and not (integer and value.arg[0] is Ops.MAX) or not (rows>1 and out.dtype.scalar() is dtypes.int or total>416 or len(loads)>2 or any(node.op in (Ops.SQRT,Ops.EXP2,Ops.LOG2,Ops.SIN,Ops.CMPLT,Ops.CMPNE,Ops.WHERE) for node in graph))): return None  # noqa: E501
  mapped_dtype=dtypes.int16 if integer else dtypes.half; gated=_gate_zero_term(product) if product.op is Ops.WHERE and _strip_cast(product.src[1]).op is Ops.LOAD else product; mapped_terms:tuple[UOp,...]=(gated if gated is not product or unit_sum else body,); product=gated  # noqa: E501
  if product.op is Ops.MUL and product.dtype.scalar() is dtypes.half and any(_strip_cast(source).op is Ops.LOAD for source in product.src) and any(_strip_cast(source).op is not Ops.LOAD for source in product.src):  # noqa: E501
    factor,multiplier=next(((a,b) for a,b in (product.src,product.src[::-1]) if a.op is Ops.ADD and _strip_cast(b).op is Ops.LOAD),product.src); products=tuple(term.alu(Ops.MUL,multiplier) for term in _iter_binary(factor,Ops.ADD,dtypes.half,plain=True)) if factor.op is Ops.ADD else (product,); pairs=tuple(_two_product(term,UOp.const(-1.0,dtypes.half),UOp.const(65.0,dtypes.half)) for term in products); mapped_terms=tuple(pair[0] for pair in pairs)+tuple(_tag_precise_adds(pair[1]) for pair in pairs)  # noqa: E501
  out_affine=typing_cast(tuple[int,dict[UOp,int]]|None,_linear_index(out_index)); output_axes=_affine_output_axes(out_affine,rows) if out_affine is not None and out_affine[0]==0 else None; block=rows if boolean else round_up(rows,8) if rows>1 else 1; groups=total*len(mapped_terms); lanes=groups*block; source_loads=_semantic_loads(body) if boolean else (); source_param=_root_param(source_loads[0].src[0]) if len(source_loads)==1 else None; source_affine=_linear_index(source_loads[0].src[0].src[1]) if len(source_loads)==1 else None  # noqa: E501
  if rows>1 and (value.arg[0] not in ((Ops.MUL,) if boolean else (Ops.ADD,Ops.MAX,Ops.MUL)) or output_axes is None or lanes>(_MAX_GENERIC_EXPANDED_NODES if boolean else min(_MAX_STATIC_RANGE_ENVS,16*_MAX_GENERIC_UNROLL) if short_math or value.arg[0] is Ops.MUL or value.arg[0] is Ops.ADD and product.op in (Ops.EXP2,Ops.LOG2,Ops.SQRT,Ops.SIN) else _MAX_STATIC_RANGE_ENVS if integer or value.arg[0] is Ops.MAX else _MAX_GENERIC_UNROLL)) or boolean and (source_param is None or source_param.src[0].op is not Ops.CONST or int(source_param.src[0].arg)!=rows*total or source_affine!=(0,{axis:stride*total for axis,stride,_ in output_axes or ()}|{axis:math.prod(extents[i+1:]) for i,axis in enumerate(ranges)})): return None  # noqa: E501
  lane=UOp.range(lanes,1+max((u.arg[0] for u in uops if u.op is Ops.RANGE and isinstance(u.arg,tuple)),default=-1),dtype=dtypes.int); row_lane=lane.alu(Ops.CDIV,lane.const_like(groups)) if boolean else lane.alu(Ops.CMOD,lane.const_like(block)); logical=lane.alu(Ops.CMOD,lane.const_like(total)) if boolean else lane.alu(Ops.CDIV,lane.const_like(block)); logical=logical.alu(Ops.CMOD,logical.const_like(total)) if len(mapped_terms)>1 else logical; output_row=row_lane.alu(Ops.CMPLT,row_lane.const_like(rows)).where(row_lane,row_lane.const_like(0)) if block>rows else row_lane  # noqa: E501
  replacements={axis:(logical if boolean and stride==1 else logical.alu(Ops.CDIV,logical.const_like(stride)).alu(Ops.CMOD,logical.const_like(extent)) if stride>1 else logical.alu(Ops.CMOD,logical.const_like(extent))) for axis,extent,stride in  # noqa: E501
    zip(ranges,extents,(math.prod(extents[i+1:]) for i in range(len(extents))))}
  replacements.update({axis:output_row if boolean and stride==1 else output_row.alu(Ops.CDIV,output_row.const_like(stride)).alu(Ops.CMOD,output_row.const_like(extent)) if stride>1 else output_row.alu(Ops.CMOD,output_row.const_like(extent)) for axis,stride,extent in output_axes or ()}); terms=tuple(graph_rewrite(mapped,sym) if boolean else mapped for term in mapped_terms for mapped in (term.substitute(replacements,walk=True),))  # noqa: E501
  mapped_body=functools.reduce(lambda selected,item:lane.alu(Ops.CMPLT,lane.const_like((item[0]+1)*total*block)).where(item[1],selected),reversed(tuple(enumerate(terms[:-1]))),terms[-1]); mapped_body=row_lane.alu(Ops.CMPLT,row_lane.const_like(rows)).where(mapped_body,mapped_body.const_like(0 if value.arg[0] is Ops.ADD else dtypes.int16.min if integer else -math.inf)) if block>rows else mapped_body; mapped_body=mapped_body.substitute({load.src[0]:load.src[0].replace(src=(load.src[0].src[0],lane)) for load in _semantic_loads(mapped_body)},walk=True) if boolean else mapped_body  # noqa: E501
  padded=mapped_body.op is Ops.WHERE and mapped_body.src[2].op is Ops.CONST and mapped_body.src[2].arg==0 and _is_static_expr(mapped_body.src[0]); weighted_body,pad=(mapped_body.src[1],mapped_body.src[0]) if padded else (mapped_body,None); weighted=next((condition.cast(dtypes.int16)*weight.cast(dtypes.int16) for cast,weight in (weighted_body.src,weighted_body.src[::-1]) if (condition:=_typed_cast_source(cast,dtypes.int,dtypes.bool)) is not None and _is_static_expr(weight)),None) if integer and value.arg[0] is Ops.ADD and bounds is not None and -32768<=total*bounds[0]<=total*bounds[1]<=32767 and weighted_body.op is Ops.MUL else None; weighted=weighted if weighted is None or pad is None else weighted*pad.cast(dtypes.int16); slot=1+max((u.arg.slot for u in uops if u.op is Ops.PARAM),default=out.arg.slot); fake=UOp.param(slot,mapped_dtype,(lanes,)); mapped_sink=fake.index(lane).store(mapped_body.cast(mapped_dtype) if boolean else weighted if weighted is not None else mapped_body).end(lane).sink(); mapped=_lower_uop_program(list(graph_rewrite(mapped_sink,pm_lower_index_dtype,ctx={}).toposort() if integer else mapped_sink.toposort()),vectorize_reductions=False)  # noqa: E501
  if mapped is None or any(isinstance(op,RKCMAC) or isinstance(op,RKGather) and op.index is not None and op.dst.kind is RKBufferKind.ARG for op in mapped.program): return None  # noqa: E501
  direct=mapped_dtype is dtypes.half and product.op is Ops.LOAD and len(mapped.scratch)==1 and len(mapped.program)==2 and isinstance(mapped.program[0],RKGather) and isinstance(mapped.program[1],RKEWOp) and mapped.program[1].dst==RKArg(RKBufferKind.ARG,slot) and mapped.program[1].lhs==mapped.program[1].rhs; value_slot=len(mapped.scratch); source=typing_cast(RKEWOp,mapped.program[1]).lhs if direct else RKArg(RKBufferKind.SCRATCH,value_slot); scratch=list(mapped.scratch)+([] if direct else [_scratch_bytes(lanes)])  # noqa: E501
  mapped=_alias_image_args(mapped,{slot:source}); prefix_program=mapped.program[:1] if direct else mapped.program; gathers:list[RKGather]=[]; ops:list[RKEWOp]=[]  # noqa: E501
  reduced=_reduce_boolean_rows(ops,scratch,gathers,source,lanes,_EW_CFG[value.arg[0]],rows) if boolean else _reduce_mapped_rows(ops,scratch,gathers,source,lanes,_EW_CFG[value.arg[0]],rows,int16=integer,kahan=value.arg[0] is Ops.ADD and product.op is not Ops.LOAD and not square and not unit_sum and groups<=4096,pairwise=value.arg[0] is Ops.ADD and square and groups<=4096,barrier=not direct)  # noqa: E501
  commit:tuple[RKGather|RKEWOp,...]=()
  if boolean: packed,unit=(RKArg(RKBufferKind.SCRATCH,len(scratch)+i) for i in range(2)); scratch.extend((_scratch_bytes(rows),)*2); commit=(RKGather(reduced,packed,rows,offsets=tuple(range(rows))),RKGather(None,unit,rows,values=(_fp16_bits(1),)),RKEWOp(packed,packed,unit,rows,_EW_CFG[Ops.MUL],mode=RKEWMode.INT16)); reduced=packed  # noqa: E501
  prefix=RKImage(tuple(scratch),program=prefix_program+tuple(gathers)+tuple(ops)+commit); scalar=UOp.param(slot+1,dtypes.half if boolean else mapped_dtype,(rows,)); replacement=scalar.index(out_index).load().cast(value.dtype); replacement=replacement.alu(Ops.MAX,replacement.const_like(bounds[0])) if integer and not boolean and bounds is not None and value.arg[0] is Ops.MAX and total<32 else replacement; replacement=replacement.const_like(0).alu(Ops.SUB,replacement.const_like(0).alu(Ops.SUB,replacement).alu(Ops.MAX,replacement.const_like(-bounds[1]))) if integer and not boolean and bounds is not None and value.arg[0] is Ops.MAX and total<32 else replacement; suffix_root=root.substitute({value:replacement})  # noqa: E501
  return None if (suffix:=_lower_uop_program(list(store.replace(src=(store.src[0],suffix_root)).sink().toposort()),vectorize_reductions=any(node.op is Ops.REDUCE for node in suffix_root.toposort()))) is None else _append_inplace_image(prefix,suffix,link=(slot+1,reduced,rows),chain=direct or integer and value.arg[0] is Ops.MAX and total<32)  # noqa: E501

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
    delta=b.alu(Ops.SUB,a); one=delta.const_like(1); component_less=_i16_bit(delta); component_equal=one.alu(Ops.SUB,_i16_min(_i16_abs(delta),one))
    less=less.alu(Ops.MAX,equal.alu(Ops.MUL,component_less)); equal=equal.alu(Ops.MUL,component_equal)
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
  return RKImage(program=(_raw_gather(RKArg(RKBufferKind.ARG,low.param.arg.slot),out.arg.slot,n,itemsize=4)._replace(axes=(),offsets=tuple(offset//2 for offset in low.gather.offsets)),))  # noqa: E501

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
def _int_info(u:UOp) -> tuple[tuple[int, int]|None, UOp|None]:
  """Share exact range admission and the optional HALF arithmetic recipe for one integer graph."""
  dtype=u.dtype.scalar(); bounds_src=u.src[:1] if u.op is Ops.CAST else u.src[1:] if u.op is Ops.WHERE else u.src
  valid=dtype in (dtypes.int,dtypes.weakint) and (_is_static_expr(u) or
    u.op in (Ops.CAST,Ops.WHERE) and len(u.src)==(1 if u.op is Ops.CAST else 3) and all(bound is not None or u.op is Ops.CAST and source.dtype.scalar() in (dtypes.bool,dtypes.int16) for source,bound in zip(bounds_src,(_int_info(node)[0] for node in bounds_src))) or  # noqa: E501
    u.op is Ops.XOR and len(u.src)==2 and any(marker.op is Ops.CONST and marker.arg==-1 and _int_info(source)[0] is not None for marker,source in (u.src,u.src[::-1])) or  # noqa: E501
    u.op is Ops.CMOD and len(u.src)==2 and (right:=_int_info(u.src[1])[0]) is not None and right[0]==right[1]!=0 or u.op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX) and len(u.src)==2 and all(_int_info(node)[0] is not None for node in bounds_src))  # noqa: E501
  bounds=((0,max(0,high)) if u.op is Ops.RANGE else (low,high)) if valid and dtype.min <= (low:=int(u.vmin)) <= (high:=int(u.vmax)) <= dtype.max else None  # noqa: E501
  if u.op is Ops.CONST: recipe=UOp.const(float(u.arg),dtypes.half)
  elif (source:=_typed_cast_source(u,dtypes.int,dtypes.half)) is not None: recipe=_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source,)))
  elif (source:=_typed_cast_source(u,dtypes.int,dtypes.bool)) is not None: recipe=source.cast(dtypes.half)
  elif u.op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.CMOD) and len(u.src)==2 and (mapped:=tuple(_int_info(src)[1] for src in u.src)) and all(x is not None for x in mapped):  # noqa: E501
    lhs,rhs=typing_cast(tuple[UOp,UOp],mapped); recipe=lhs.alu(Ops.SUB,_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(lhs.alu(Ops.FDIV,rhs),))).alu(Ops.MUL,rhs)) if u.op is Ops.CMOD else u.replace(dtype=dtypes.half,src=(lhs,rhs))  # noqa: E501
  elif u.op is Ops.WHERE and len(u.src)==3:
    condition=u.src[0]; compared=tuple(_int_info(src)[1] for src in condition.src) if condition.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ) and all(src.dtype.scalar() is dtypes.int for src in condition.src) else (); condition=condition.replace(src=typing_cast(tuple[UOp,...],compared)) if compared and all(x is not None for x in compared) else condition  # noqa: E501
    arms=tuple(_int_info(src)[1] for src in u.src[1:]); recipe=UOp(Ops.WHERE,dtypes.half,src=(condition,*typing_cast(tuple[UOp,...],arms))) if all(x is not None for x in arms) and (not compared or all(x is not None for x in compared)) else None  # noqa: E501
  else: recipe=None
  return bounds,recipe

def _exact_int_range(u:UOp) -> tuple[int, int]|None: return _int_info(u)[0]

class _RKGenericReject(Exception): pass

def _has_runtime_address(root:UOp) -> bool:
  """True when a value LOAD obtains its address or gate from another runtime LOAD."""
  return any(_semantic_loads(load.src[0].src[1]) or len(load.src)>2 and _semantic_loads(load.src[2]) for load in _semantic_loads(root) if load.src and load.src[0].op is Ops.INDEX)  # noqa: E501

def _fp32_expr_to_half(u:UOp) -> UOp:
  """Represent a float ADD/MUL expression with a three-half expansion at its FP16 storage boundary."""
  if u.dtype.scalar() is dtypes.half or u.op is Ops.CONST and u.dtype.scalar() is dtypes.weakfloat: return u if u.dtype.scalar() is dtypes.half else UOp.const(float(u.arg),dtypes.half)  # noqa: E501
  if u.dtype.scalar() is not dtypes.float: raise _RKGenericReject
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.half: return u.src[0]
  if u.op is Ops.CAST and len(u.src) == 1 and u.src[0].dtype.scalar() in (dtypes.int, dtypes.int16, dtypes.bool): return u.src[0].cast(dtypes.half)
  if u.op is Ops.LOAD: return u.cast(dtypes.half)
  if u.op is Ops.CONST: return UOp.const(float(u.arg), dtypes.half)
  if _is_static_expr(u) and u.op is not Ops.WHERE: return u.cast(dtypes.half)
  if ((u.op in (Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.SIN, Ops.NEG) and len(u.src) == 1) or (u.op in (Ops.MUL, Ops.SUB, Ops.MAX) and len(u.src) == 2) or
      (u.op is Ops.WHERE and len(u.src) == 3 and _is_static_expr(u.src[0]))):
    return UOp(u.op, dtypes.half, src=(u.src[0],*tuple(_fp32_expr_to_half(src) for src in u.src[1:])) if u.op is Ops.WHERE else tuple(_fp32_expr_to_half(src) for src in u.src), arg=u.arg if u.op not in (Ops.MUL, Ops.NEG) else None)  # noqa: E501
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
    self.values:dict[UOp,UOp]={}; self.scratch:list[int]=[]; self.materialized_slots:dict[tuple,int]={}
    self.raw_components:dict[RKArg,tuple[UOp,...]]={}
    self.recipe_owners:dict[UOp,UOp]={}
    self.program:list[RKGather|RKEWOp|RKCMAC]=[]
    nodes=self.root.toposort(); value_nodes=self.root.toposort(gate=lambda node:node.op is not Ops.LOAD); self.semantic_nodes=set(nodes); self.bounded_chain=any(node.op is Ops.MAX and node.arg == _NATIVE_POSITIVE_MASK for node in nodes)  # noqa: E501
    int_range=_int_info(self.root)[0] if self.root.dtype.scalar() is dtypes.int else None
    dynamic_int_load=any(node.op is Ops.LOAD and node.dtype.scalar() in (dtypes.int,dtypes.uint) and node.src and _root_param(node.src[0]) is not None for node in nodes)  # noqa: E501
    cmod_support=tuple(_int_info(node)[1] is not None for node in value_nodes if node.op is Ops.CMOD and not _is_static_expr(node))
    wide_int=dynamic_int_load or any(node.op is Ops.CDIV and not _is_static_expr(node) for node in value_nodes) or False in cmod_support
    narrow_int=not wide_int and (self.root.dtype.scalar() is dtypes.int and int_range is not None and -32768 <= int_range[0] <= int_range[1] <= 32767 or self.root.dtype.scalar() is not dtypes.int and bool(cmod_support))  # noqa: E501
    self.int_layout = dtypes.int16 if narrow_int else dtypes.int

  def _layout(self, dtype:DType) -> DType:
    if (layout:={dtypes.half:dtypes.half, dtypes.float:dtypes.half, dtypes.int16:dtypes.int16,
      dtypes.uchar:dtypes.int16, dtypes.bool:dtypes.int16, dtypes.uint:dtypes.int}.get(
        dtype,self.int_layout if dtype is dtypes.int else None)) is None: raise _RKGenericReject(f"layout {dtype}")
    return layout

  def _carrier(self, arg:RKArg, dtype:DType) -> UOp: return UOp(Ops.NOOP,dtype,src=(UOp.const(0,dtype),),arg=arg)

  def _scratch(self, layout:DType, size:int|None=None, u:UOp|None=None) -> UOp:
    if u is not None and (u is self.root or self.recipe_owners.get(u) is self.root) and self.out_param.dtype.scalar() is u.dtype.scalar() and layout is {dtypes.half:dtypes.half,dtypes.int16:dtypes.int16,dtypes.int:dtypes.int}.get(u.dtype.scalar()): return self._carrier(self.out,layout)  # noqa: E501
    self.scratch.append(size if size is not None else self.count*layout.itemsize if layout is dtypes.int else _scratch_bytes(self.count))  # noqa: E501
    return self._carrier(RKArg(RKBufferKind.SCRATCH,len(self.scratch)-1),layout)

  def _slot(self, cache:dict, source:RKGather|tuple, layout:DType, size:int|None=None, key:tuple|None=None) -> UOp:
    if isinstance(source, RKGather):
      plan, cache_key = source, typing_cast(bytes|tuple,("gather",layout,_gather_cache_key((source,))) if key is None else ("gather",key))
    else:
      plan=RKGather(None,RKArg(RKBufferKind.SCRATCH,0),self.count,values=source,itemsize=layout.itemsize); cache_key=("static",layout,source)  # noqa: E501
    if cache_key not in cache:
      value = self._scratch(layout,size)
      self.program.append(plan._replace(dst=value.arg))
      cache[cache_key] = value.arg.index
    return self._carrier(RKArg(RKBufferKind.SCRATCH,cache[cache_key]),layout)

  def _constant(self, u:UOp, dtype_hint:DType|None=None) -> UOp:
    layout = self._layout(dtype_hint or u.dtype.scalar())
    bits=int(u.arg)&0xffffffff if layout is dtypes.int else _fp16_bits(u.arg) if layout is dtypes.half else _int16_bits(u.arg)
    return self._slot(self.materialized_slots,(bits,),layout)

  def _operand(self, u:UOp, dtype:DType, finite_min:bool=False) -> UOp:
    if finite_min and u.op is Ops.LOAD and len(u.src)>2 and u.src[1].op is Ops.CONST and math.isinf(float(u.src[1].arg)) and float(u.src[1].arg)<0 and _is_static_expr(u.src[2]): return self._load(u,_fp16_bits(-65504.0))  # noqa: E501
    return self._constant(u, dtype) if u.op is Ops.CONST and (u.dtype.scalar() in dtypes.weaks or dtype is dtypes.half and u.dtype.scalar() is dtypes.float) else self.lower(u)  # noqa: E501

  def _static(self, u:UOp) -> UOp:
    dtype, layout = u.dtype.scalar(), self._layout(u.dtype.scalar())
    if not _index_ranges(u): return self._constant(UOp.const(typing_cast(int|float|bool,_eval_static(u,{}).item()),dtype))
    values = _static_values(self.out_index,u,self.count,_fp16_bits if layout is dtypes.half else int)
    if dtype is dtypes.int and layout is dtypes.int16 and any(not -32768 <= value <= 32767 for value in values): raise _RKGenericReject
    encoded = values if layout is dtypes.half else tuple(value&0xffffffff if layout is dtypes.int else value&0xffff for value in values)
    return self._slot(self.materialized_slots,encoded,layout)

  def _masked_load_default(self, u:UOp, dtype:DType, layout:DType, gate:UOp|None, default:UOp, runtime_address:bool) -> UOp:  # noqa: E501
    """Overlay a static masked load on a separately materialized default."""
    if dtype not in (dtypes.half,dtypes.int16,dtypes.int,dtypes.uint) or gate is None or runtime_address: raise _RKGenericReject
    schedule, fallback = len(self.program),self.lower(default)
    if any(not isinstance(op,RKGather) or op.index is not None for op in self.program[schedule:]): raise _RKGenericReject
    if (plan:=_typed_load_plan(u,dtype,self.out_index,self.count,fill_bits=0)) is None: raise _RKGenericReject
    value = self._scratch(layout,self.count*dtype.itemsize)
    self.program.extend((RKGather(fallback.arg,value.arg,self.count,axes=((1,self.count,1),),itemsize=dtype.itemsize),
      plan.gather._replace(dst=value.arg,partial=True,itemsize=dtype.itemsize)))
    return value

  def _host_address_load(self, param:UOp, index:UOp, gate:UOp|None, address_loads:tuple[UOp, ...], dtype:DType, layout:DType, fill_bits:int) -> UOp:  # noqa: E501
    """Compute a dynamic address and predicate on the NPU, then move only the selected raw lane on the host.

    The host never interprets tensor arithmetic or a boolean gate: an invalid lane is encoded as index -1 by the
    ordinary typed UOp path. This replaces candidate-domain value selection with one exact physical address carrier.
    """
    if os.getenv("ROCKCHIP_HOST_GATHER","1") != "1" or not address_loads: raise _RKGenericReject
    if (any((load:=_strip_cast(node)).op is Ops.LOAD and len(load.src)==1 and load.src[0].op is Ops.INDEX and (owner:=_root_param(load.src[0])) is not None and owner.src[0].op is Ops.CONST and  # noqa: E501
            owner.dtype.scalar() in (dtypes.int,dtypes.int16) and int(owner.src[0].arg)*owner.dtype.scalar().itemsize-1>dtypes.int.max for node in address_loads) or  # noqa: E501
        int(param.src[0].arg)*dtype.itemsize-1>dtypes.int.max): raise _RKGenericReject
    physical=self.lower(index if gate is None else gate.where(index,index.const_like(-1)))
    if physical.dtype not in (dtypes.int16,dtypes.int): raise _RKGenericReject
    value=self._scratch(layout,self.count*dtype.itemsize)
    self.program.append(RKGather(RKArg(RKBufferKind.ARG,param.arg.slot),value.arg,self.count,fill_bits=fill_bits,
      itemsize=dtype.itemsize,index=physical.arg,index_itemsize=physical.dtype.itemsize))
    return value

  def _load(self, u:UOp, fill_override:int|None=None) -> UOp:
    dtype,layout = u.dtype.scalar(),self._layout(u.dtype.scalar())
    if not u.src or u.src[0].op is not Ops.INDEX or (param:=_root_param(u.src[0])) is None or param.arg.slot == self.out_param.arg.slot or param.src[0].op is not Ops.CONST: raise _RKGenericReject  # noqa: E501
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
      return self._host_address_load(param,index,gate,address_loads,dtype,layout,fill_bits)
    if (plan:=_typed_load_plan(u,dtype,self.out_index,self.count,fill_bits=fill_bits,
                              require_offsets=dtype is dtypes.bool)) is None: raise _RKGenericReject
    if dtype is dtypes.float:
      raw=self._slot(self.materialized_slots,plan.gather._replace(itemsize=4),dtypes.float,ceildiv(self.count,_EW_ELEMS_32BIT)*16,("fp32_raw",_gather_cache_key((plan.gather._replace(itemsize=4),))))  # noqa: E501
      return self._convert(u,raw,dtypes.half)
    if dtype is dtypes.bool:
      return self._slot(self.materialized_slots,plan.gather._replace(
        fill_bits=int(bool(default.arg)) if default is not None else 0,dst_stride=2,itemsize=1),
        dtypes.int16,self.count*2)
    if gate is None and index.key == self.out_index.key and int(plan.param.src[0].arg) == self.count:
      return self._carrier(RKArg(RKBufferKind.ARG,plan.param.arg.slot),layout)
    return self._slot(self.materialized_slots,plan.gather._replace(itemsize=dtype.itemsize),
      layout,self.count*dtype.itemsize)

  def _emit(self, dst:UOp, lhs:UOp, rhs:UOp, cfg:int, *, compare:bool=False) -> UOp:
    integer16, integer32 = dst.dtype is dtypes.int16, dst.dtype is dtypes.int
    if lhs.dtype is not dst.dtype or rhs.dtype is not dst.dtype: raise _RKGenericReject
    barrier = not integer16 and not integer32 and cfg in (_EW_CFG_FLOOR, _EW_CFG[Ops.FDIV])
    mode=RKEWMode.INT16 if integer16 else RKEWMode.INT32 if integer32 else RKEWMode.COMPARE if compare else RKEWMode.BOUNDED if self.bounded_chain or barrier else RKEWMode.HALF  # noqa: E501
    self.program.append(RKEWOp(dst.arg,lhs.arg,rhs.arg,self.count,cfg,submit_barrier=barrier,mode=mode)); self.bounded_chain |= compare; return dst

  def _byte_gather(self, source:RKArg, dest:RKArg, count:int, *, base:int=0, source_stride:int=1, source_limit:int|None=None, dst_stride:int=1, dst_addend:int=0, itemsize:int=2, partial:bool=False) -> RKArg:  # noqa: E501
    self.program.append(RKGather(source._replace(addend=0),dest._replace(addend=0),count,base=base,
      axes=((1, count if source_limit is None else source_limit, source_stride),), dst_stride=dst_stride, dst_addend=dst_addend,
      itemsize=itemsize,partial=partial)); return dest

  def _raw(self, source:UOp|Iterable[UOp], layout:DType|None=None, *, u:UOp|None=None, dst:UOp|None=None, cache:bool=True, copy_wide:bool=True) -> Any:  # noqa: E501
    if isinstance(source,UOp):
      value = source
      if cache and value.arg in self.raw_components: return self.raw_components[value.arg]
      itemsize = value.dtype.itemsize
      if itemsize == 4 and copy_wide:
        source = self._scratch(dtypes.int)
        self._emit(source, value, value, _EW_CFG[Ops.MAX])
      parts = tuple(self._scratch(dtypes.int16) for _ in range(itemsize))
      for byte,part in enumerate(parts): self._byte_gather(source.arg,part.arg,self.count,base=source.arg.addend+byte,
        source_stride=itemsize,dst_stride=2,itemsize=1)
      if cache: self.raw_components[value.arg] = parts
      return parts
    if layout is None or u is None: raise _RKGenericReject
    parts = tuple(source)
    itemsize = layout.itemsize
    if len(parts) != itemsize: raise _RKGenericReject
    result = self._scratch(layout,u=u) if dst is None else dst
    for byte,part in enumerate(parts): self._byte_gather(part.arg,result.arg,self.count,base=part.arg.addend,source_stride=2,
      dst_stride=itemsize,dst_addend=byte,itemsize=1,partial=bool(byte))
    if cache: self.raw_components[result.arg] = parts
    return result

  def _lower_recipe(self, owner:UOp, recipe:UOp) -> UOp:
    self.recipe_owners[recipe]=owner; return self.lower(recipe)

  def _bitplanes(self, value:UOp) -> tuple[UOp, ...]:
    return tuple(itertools.chain.from_iterable(map(_byte_bits,self._raw(value,copy_wide=False))))

  def _pack_bits(self, bits:Iterable[UOp], layout:DType, u:UOp) -> UOp:
    planes=tuple(bits)
    if len(planes)!=layout.itemsize*8: raise _RKGenericReject
    raw=tuple(functools.reduce(lambda x,y:x.alu(Ops.ADD,y),
      (planes[byte*8+bit].alu(Ops.MUL,planes[byte*8+bit].const_like(1<<bit)) for bit in range(1,8)),planes[byte*8])
      for byte in range(layout.itemsize))
    return self._raw(tuple(self.lower(part) for part in raw),layout,u=u)

  def _alu(self, u:UOp) -> UOp:
    if u.op in (Ops.RECIPROCAL, Ops.NEG):
      src = self.lower(u.src[0])
      if u.op is Ops.RECIPROCAL:
        one = self.lower(UOp.const(1.0, dtypes.half))
        return self._emit(self._scratch(dtypes.half,u=u),one,src,_EW_CFG[Ops.FDIV])
      return self._emit(self._scratch(src.dtype,u=u),src,src,_EW_CFG_NEG)
    if len(u.src) != 2: raise _RKGenericReject
    if u.op is Ops.ADD and (recipe:=_fold_relu_cap(u)) is not None: return self.lower(recipe)
    if u.op is Ops.FDIV and (recipe:=_preserve_infinite_division_sign(u)) is not None:
      return self.lower(recipe)
    dtype, int_range = u.dtype.scalar(), _int_info(u)[0] if u.dtype.scalar() is dtypes.int else None
    bounded = self.int_layout is dtypes.int or self.int_layout is dtypes.int16 and int_range is not None and -32768 <= int_range[0] <= int_range[1] <= 32767  # noqa: E501
    if dtype is dtypes.int and not bounded: raise _RKGenericReject(f"alu {u.op.name} {dtype} bounds={int_range}")
    expected = self._layout(dtype); finite_min=u.op is Ops.MAX and dtype is dtypes.half
    sources=tuple(UOp.const(-65504.0,dtypes.half) if finite_min and src.op is Ops.CONST and math.isinf(float(src.arg)) and float(src.arg)<0 else src for src in u.src)  # noqa: E501
    lhs,rhs=(self._operand(src,dtype,finite_min) for src in sources)
    left,right=lhs,rhs
    if u.op is Ops.SUB and u.arg == _NATIVE_SIGN:
      if expected is not dtypes.half: raise _RKGenericReject
      zero=left.const_like(0.0)
      return self._lower_recipe(u,_positive_mask(left).alu(Ops.SUB,_positive_mask(zero.alu(Ops.SUB,left))))
    if u.op is Ops.MAX and u.arg == _NATIVE_MIN:
      if expected is not dtypes.half: return self._emit(self._scratch(dtypes.int16,u=u),lhs,rhs,_EW_CFG_MIN)
      zero=left.const_like(0.0)
      return self._lower_recipe(u,zero.alu(Ops.SUB,zero.alu(Ops.SUB,left).alu(Ops.MAX,zero.alu(Ops.SUB,right))))
    cfg = _EW_CFG_ABS if u.op is Ops.MAX and u.arg == _NATIVE_ABS else _EW_CFG_FLOOR if u.op is Ops.MAX and u.arg == _NATIVE_FLOOR else _EW_CFG_CEIL if u.op is Ops.MAX and u.arg == _NATIVE_CEIL else _EW_CFG_RELU6 if u.op is Ops.MAX and u.arg == _NATIVE_RELU6 else _EW_CFG[u.op]  # noqa: E501
    compare = u.op is Ops.MAX and u.arg == _NATIVE_POSITIVE_MASK
    return self._emit(self._scratch(expected,u=u),lhs,rhs,cfg,compare=compare)

  def _convert(self, u:UOp|None, source:UOp, target:DType, barrier:bool=False, dst:RKArg|None=None) -> UOp:
    """Cross one physical carrier boundary using the native DPU conversion stage."""
    if source.dtype is target: return source
    pair, cfg = (source.dtype,target), _EW_CFG[Ops.MAX]
    if dtypes.float in pair:
      groups=tuple(range(0,self.count,_EW_ELEMS_32BIT)); aligned=self._scratch(dtypes.half,len(groups)*16)
      if pair==(dtypes.float,dtypes.half):
        zero=self._scratch(dtypes.float,16); self.program.append(RKGather(None,zero.arg,_EW_ELEMS_32BIT,values=(0,)*_EW_ELEMS_32BIT,itemsize=4))  # noqa: E501
        self.program.extend(RKEWOp(aligned.arg._replace(addend=group*16),source.arg._replace(addend=group*16),zero.arg,min(_EW_ELEMS_32BIT,self.count-start),_EW_CFG[Ops.ADD],mode=RKEWMode.FLOAT_TO_HALF) for group,start in enumerate(groups))  # noqa: E501
        result=self._scratch(dtypes.half,self.count*2); self.program.append(RKGather(aligned.arg,result.arg,self.count,offsets=tuple((lane//_EW_ELEMS_32BIT)*8+lane%_EW_ELEMS_32BIT for lane in range(self.count)),itemsize=2))  # noqa: E501
      elif pair==(dtypes.half,dtypes.float) and dst is not None:
        result=self._carrier(dst,dtypes.float); chunks=tuple((group,start,min(_EW_ELEMS_32BIT,self.count-start)) for group,start in enumerate(groups))  # noqa: E501
        self.program.extend(tuple(RKGather(source.arg._replace(addend=0),aligned.arg,lanes,offsets=tuple(source.arg.addend//2+lane for lane in range(start,start+lanes)),dst_addend=group*8,partial=bool(group)) for group,start,lanes in chunks)+tuple(RKEWOp(result.arg._replace(addend=start*4),(chunk:=aligned.arg._replace(addend=group*16)),chunk,lanes,cfg,mode=RKEWMode.HALF_TO_FLOAT) for group,start,lanes in chunks))  # noqa: E501
      else: raise _RKGenericReject(f"convert {source.dtype}->{target}")
      return result
    result=self._scratch(target,u=None if pair==(dtypes.int,dtypes.half) else u)
    if pair in ((dtypes.half,dtypes.int),(dtypes.int,dtypes.half)):
      atoms=tuple((start,min(4,self.count-start)) for start in range(0,self.count,4)); tile=self._scratch(dtypes.int,len(atoms)<<6)
      src_size,dst_size=source.dtype.itemsize,target.itemsize
      self.program.extend(RKGather(source.arg._replace(addend=source.arg.addend+start*src_size),tile.arg._replace(addend=start//4*64),count,axes=((1,count,1),),itemsize=src_size) for start,count in atoms)  # noqa: E501
      mode=RKEWMode.HALF_TO_INT32 if target is dtypes.int else RKEWMode.INT32_TO_HALF
      self.program.extend(RKEWOp((arg:=tile.arg._replace(addend=start//4*64)),arg,arg,count,cfg,mode=mode) for start,count in atoms)
      self.program.extend(RKGather(tile.arg._replace(addend=start//4*64),result.arg._replace(addend=result.arg.addend+start*dst_size),count,axes=((1,count,1),),itemsize=dst_size) for start,count in atoms)  # noqa: E501
      return result
    if pair == (dtypes.half,dtypes.int16): rhs,mode=source.arg,RKEWMode.HALF_TO_INT16
    elif pair == (dtypes.int16,dtypes.int): rhs,mode,cfg=self._constant(UOp.const(0,dtypes.int16)).arg,RKEWMode.INT16_TO_INT32,_EW_CFG[Ops.ADD]  # noqa: E501
    else: raise _RKGenericReject(f"convert {source.dtype}->{target}")
    self.program.append(RKEWOp(result.arg,source.arg,rhs,self.count,cfg,barrier and pair==(dtypes.half,dtypes.int16),mode)); return result

  def _integer_bitwise(self, u:UOp) -> UOp:
    if len(u.src) != 2: raise _RKGenericReject
    dtype,layout=u.dtype.scalar(),self._layout(u.dtype.scalar())
    if dtype not in (dtypes.int16,dtypes.int) or u.op not in (Ops.AND,Ops.OR,Ops.XOR): raise _RKGenericReject
    if u.op is Ops.XOR and (pair:=next(((source,marker) for source,marker in (u.src,u.src[::-1])
      if marker.op is Ops.CONST and int(marker.arg)==-1),None)) is not None:
      value=self.lower(pair[0])
      if value.dtype is dtypes.int16: return self._lower_recipe(u,UOp.const(-1,dtypes.int16).alu(Ops.SUB,value))
      inverted=tuple(self.lower(component.const_like(255).alu(Ops.SUB,component)) for component in self._raw(value))
      return self._raw(inverted,dtypes.int,u=u)
    masked=tuple(_const_operand(term,Ops.AND) for term in u.src) if dtype is dtypes.int16 and u.op is Ops.OR else ()
    if len(masked)==2 and all(pair is not None for pair in masked) and {int(typing_cast(tuple[UOp,UOp],pair)[1].arg)&0xffff for pair in masked}=={0x7fff,0x8000}:  # noqa: E501
      sources={int(typing_cast(tuple[UOp,UOp],pair)[1].arg)&0xffff:typing_cast(tuple[UOp,UOp],pair)[0] for pair in masked}
      (low,high),(_,sign_high)=(self._raw(self.lower(sources[mask])) for mask in (0x7fff,0x8000))
      hi,shi=high,sign_high
      magnitude_sign=_i16_bit(hi.alu(Ops.SUB,hi.const_like(127))).alu(Ops.MUL,hi.const_like(128))
      sign=_i16_bit(shi.alu(Ops.SUB,shi.const_like(127))).alu(Ops.MUL,shi.const_like(128))
      return self._raw(tuple(self.lower(part) for part in (low,hi.alu(Ops.SUB,magnitude_sign).alu(Ops.ADD,sign))),layout,u=u)
    values = tuple(self.lower(source) for source in u.src)
    if not 1 <= self.count*layout.itemsize <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    lhs_bits,rhs_bits=(self._bitplanes(value) for value in values)
    combined=tuple(left.alu(Ops.MUL,right) if u.op is Ops.AND else left.alu(Ops.MAX,right) if u.op is Ops.OR else
                   _i16_abs(left.alu(Ops.SUB,right))
                   for left,right in zip(lhs_bits,rhs_bits))
    return self._pack_bits(combined,layout,u)

  def _int32_shift(self, u:UOp) -> UOp:
    if len(u.src) != 2 or u.dtype.scalar() not in (dtypes.int, dtypes.uint) or u.src[1].dtype.scalar() not in (dtypes.int, dtypes.uint) or self.int_layout is not dtypes.int:  # noqa: E501
      raise _RKGenericReject
    value=self.lower(u.src[0])
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
    return self._pack_bits(current,dtypes.int,u)

  def _compare(self, u:UOp) -> UOp:
    if len(u.src) != 2: raise _RKGenericReject
    if all(src.dtype.scalar() is dtypes.bool for src in u.src):
      expression=next((v for v,m in (u.src,u.src[::-1]) if u.op is Ops.CMPNE and m.op is Ops.CONST and bool(m.arg) and m.dtype.scalar() is dtypes.bool),None); sources=tuple(_half_backed_value(src) for src in expression.src) if expression is not None and expression.op is Ops.CMPLT else ()  # noqa: E501,E702
      if sources and all(src is not None for src in sources):
        less=self.lower(typing_cast(UOp,expression)); unordered=tuple(self._fp16_component_values(self._operand(src,dtypes.half))[2] for src in typing_cast(tuple[UOp,UOp],sources)); one=less.const_like(1)  # noqa: E501
        return self._lower_recipe(u,one.alu(Ops.SUB,less).alu(Ops.MUL,one.alu(Ops.SUB,unordered[0].alu(Ops.MAX,unordered[1]))))
      lhs,rhs=(self.lower(src) for src in u.src); op=Ops.AND if u.op is Ops.MUL else Ops.OR if u.op is Ops.MAX else u.op
      if op not in (Ops.AND,Ops.OR,Ops.XOR,Ops.CMPNE,Ops.CMPEQ): raise _RKGenericReject
      complement=next((other for source,other in zip(u.src,(rhs,lhs)) if op in (Ops.XOR,Ops.CMPNE) and source.op is Ops.CONST and bool(source.arg)),None)  # noqa: E501
      result=lhs.const_like(1).alu(Ops.SUB,complement) if complement is not None else lhs.alu(Ops.MUL if op is Ops.AND else Ops.MAX,rhs) if op in (Ops.AND,Ops.OR) else _i16_compare(op,lhs,rhs)  # noqa: E501
      return self._lower_recipe(u,result)
    if u.op is Ops.CMPNE and (u is self.root or any(src.op is Ops.INDEX for src in u.src)) and (nonzero:=_fp16_nonzero_mask(u)) is not None:  # noqa: E501
      value = self.lower(nonzero)
      return self._convert(u,value,dtypes.int16,True)
    nan:tuple[UOp,...]=()
    if all(src.dtype.scalar() is dtypes.int or src.op is Ops.CONST and src.dtype.scalar() is dtypes.weakint for src in u.src):
      int_sources = typing_cast(tuple[UOp, UOp], tuple(UOp.const(int(src.arg), dtypes.int) if src.dtype.scalar() is dtypes.weakint else src for src in u.src))  # noqa: E501
      half_sources=tuple(_int_info(src)[1] for src in int_sources)
      if self.int_layout is dtypes.int16 and all(src is not None for src in half_sources): return self.lower(u.replace(src=typing_cast(tuple[UOp,...],half_sources)))  # noqa: E501
      if self.int_layout is dtypes.int16: return self._lower_recipe(u,_i16_compare(u.op,*(self._operand(src,dtypes.int) for src in int_sources)))  # noqa: E501
      values=tuple(self._operand(src,dtypes.int) for src in int_sources)
      components=tuple(self._raw(value) for value in values)
      if u.op is Ops.CMPLT: components=tuple((_sign_bias(parts[3]),*parts[2::-1]) for parts in components)
    elif all(src.dtype.scalar() is dtypes.int16 for src in u.src): return self._lower_recipe(u,_i16_compare(u.op,*(self._operand(src,dtypes.int16) for src in u.src)))  # noqa: E501
    else:
      half_sources = u.src if all(src.dtype.scalar() is dtypes.half for src in u.src) else tuple(_half_backed_value(src) for src in u.src)
      if u.op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) or any(src is None for src in half_sources): raise _RKGenericReject
      classified_sources:tuple[tuple[UOp,bool],...]=tuple((pair[0],True) if pair is not None else (src,False) for src in typing_cast(tuple[UOp,UOp],half_sources) for pair in ((_const_operand(src,Ops.MUL,-1.0) if u.op is not Ops.CMPLT else None),))  # noqa: E501
      def classify(source:UOp, negated:bool) -> tuple[UOp,...]:
        if source.op is not Ops.MAX or source.arg is not None or len(source.src)!=2 or negated: return self._fp16_component_values(self._operand(source,dtypes.half),negated=negated)  # noqa: E501
        values=tuple(self._operand(src,dtypes.half) for src in source.src); ordered=tuple(self._fp16_component_values(value,True) for value in values); choose=ordered[0][2].const_like(1).alu(Ops.SUB,ordered[0][2]).alu(Ops.MUL,ordered[1][2].alu(Ops.MAX,_ordered_bits(ordered[0][:2],ordered[1][:2])))  # noqa: E501
        parts=tuple(self._fp16_component_values(value) for value in values); return tuple(_i16_select(choose,right,left) for left,right in zip(parts[0][:2],parts[1][:2]))+(ordered[0][2].alu(Ops.MAX,ordered[1][2]),)  # noqa: E501
      classified=tuple(self._fp16_component_values(value,True) for value in tuple(self._operand(src,dtypes.half) for src,_ in classified_sources)) if u.op is Ops.CMPLT else tuple(classify(*source) for source in classified_sources)  # noqa: E501
      nan=tuple(parts[2] for parts in classified); components=tuple(parts[:2] for parts in classified)
    result=_ordered_bits(*components) if u.op is Ops.CMPLT else functools.reduce(
      lambda equal,pair:equal.alu(Ops.MUL,_i16_equal(*pair)),zip(*components),components[0][0].const_like(1))
    if nan: result=result.alu(Ops.MUL,result.const_like(1).alu(Ops.SUB,nan[0].alu(Ops.MAX,nan[1])))
    if u.op is Ops.CMPNE: result=result.const_like(1).alu(Ops.SUB,result)
    return self.lower(result)

  def _int32_divmod(self, u:UOp) -> UOp:
    if len(u.src) != 2 or not 1 <= self.count <= _MAX_EW_ELEMS_FP16: raise _RKGenericReject
    values = tuple(self._operand(src, dtypes.int) for src in u.src)
    raw=tuple(self._raw(value) for value in values)
    signs=tuple(_i16_bit(value[3].alu(Ops.SUB,value[3].const_like(127))) for value in raw)
    numerator,denominator=(_twos_complement(value,sign) for value,sign in zip(raw,signs))
    denominator_nonzero=functools.reduce(lambda x,y:x.alu(Ops.MAX,y),map(_i16_bit,denominator)); one=denominator_nonzero.const_like(1)
    numerator_bits=tuple(itertools.chain.from_iterable(map(_byte_bits,numerator)))
    zero=numerator[0].const_like(0); remainder,quotient=[zero]*4,[zero]*4
    for bit_index in range(31, -1, -1):
      shifted,incoming=[],numerator_bits[bit_index]
      for byte in remainder:
        carry=_i16_bit(byte.alu(Ops.SUB,byte.const_like(127)))
        wrapped=byte.alu(Ops.ADD,byte).alu(Ops.SUB,carry.alu(Ops.MUL,byte.const_like(256)))
        shifted.append(wrapped.alu(Ops.ADD,incoming)); incoming=carry
      remainder=shifted
      borrow,reduced=zero,[]
      for left,right in zip(remainder, denominator):
        delta=left.alu(Ops.SUB,right).alu(Ops.SUB,borrow)
        borrow=_i16_bit(zero.alu(Ops.SUB,delta)); reduced.append(delta.alu(Ops.ADD,borrow.alu(Ops.MUL,zero.const_like(256))))
      ge=denominator_nonzero.alu(Ops.MUL,one.alu(Ops.SUB,borrow))
      remainder=[left.alu(Ops.ADD,ge.alu(Ops.MUL,right.alu(Ops.SUB,left))) for left,right in zip(remainder,reduced)]; byte_index,weight=bit_index>>3,1<<(bit_index&7)  # noqa: E501
      quotient[byte_index]=quotient[byte_index].alu(Ops.ADD,ge.alu(Ops.MUL,zero.const_like(weight)))
    quotient_raw, remainder_raw, remainder_sign, quotient_sign = tuple(quotient),tuple(remainder),signs[0],_i16_abs(signs[0].alu(Ops.SUB,signs[1]))
    packed_raw, sign = (quotient_raw, quotient_sign) if u.op is Ops.CDIV else (remainder_raw, remainder_sign)
    return self._raw(tuple(self.lower(value) for value in _twos_complement(packed_raw,sign)),dtypes.int,u=u)

  def _fp16_component_values(self, value:UOp, ordered:bool=False, negated:bool=False) -> tuple[UOp, ...]:
    """Split and classify one physical FP16 value once so composed comparison UOps can reuse it."""
    lo,hi=self._raw(value); hi=_sign_bias(hi) if negated else hi; sign_scale=_i16_bit(hi.alu(Ops.SUB,hi.const_like(127))).alu(Ops.MUL,hi.const_like(128)); magnitude=hi.alu(Ops.SUB,sign_scale)  # noqa: E501
    one=magnitude.const_like(1); clean=hi.alu(Ops.SUB,sign_scale.alu(Ops.MUL,one.alu(Ops.SUB,_i16_min(magnitude,one)).alu(Ops.MUL,one.alu(Ops.SUB,_i16_min(lo,one))))); nan=_i16_bit(magnitude.alu(Ops.SUB,magnitude.const_like(124)).alu(Ops.ADD,_i16_min(lo,one)))  # noqa: E501
    if not ordered: return lo,clean,nan
    sign=_i16_bit(clean.alu(Ops.SUB,clean.const_like(127))); positive=clean.alu(Ops.ADD,clean.const_like(128)); high_delta=clean.const_like(255).alu(Ops.SUB,clean).alu(Ops.SUB,positive); low_delta=lo.const_like(255).alu(Ops.SUB,lo).alu(Ops.SUB,lo)  # noqa: E501
    return positive.alu(Ops.ADD,sign.alu(Ops.MUL,high_delta)),lo.alu(Ops.ADD,sign.alu(Ops.MUL,low_delta)),nan

  def _raw_where(self, u:UOp, selector:UOp|None=None) -> UOp:
    """Select typed values through one canonical INT16 mask, preserving nonfinite arms as raw bytes."""
    gate,arms=_unwrap_condition(u.src[0]),tuple(_unwrap_condition(src) for src in u.src[1:])
    lhs=gate.src[0] if gate.op is Ops.CMPLT and gate.src[1].op is Ops.CONST and math.isfinite(float(gate.src[1].arg)) and all(src.dtype.scalar() in (dtypes.half,dtypes.float) for src in gate.src) else None  # noqa: E501
    if selector is None and lhs is not None and any(dynamic.key==lhs.key and constant.op is Ops.CONST and math.isfinite(float(constant.arg)) and float(constant.arg)!=float(gate.src[1].arg) for dynamic,constant in (arms,arms[::-1])):  # noqa: E501
      value=self.lower(lhs.cast(dtypes.half)); nan=self._fp16_component_values(value)[2]
      selector=self._convert(None,self.lower(_positive_mask(gate.src[1].cast(dtypes.half).alu(Ops.SUB,lhs.cast(dtypes.half)))),dtypes.int16,True)
      selector=self.lower(selector.alu(Ops.MUL,nan.const_like(1).alu(Ops.SUB,nan)))
    yes, no = (self.lower(src) for src in u.src[1:])
    if selector is None: selector = self.lower(u.src[0])
    if yes.dtype is dtypes.int16: return self._lower_recipe(u,_i16_select(selector,yes,no))
    yes_bytes,no_bytes=(self._raw(x,cache=False,copy_wide=False) for x in (yes,no))
    selected_bytes=[self.lower((n:=no_byte).alu(Ops.ADD,selector.alu(Ops.MUL,yes_byte.alu(Ops.SUB,n))))
                    for yes_byte,no_byte in zip(yes_bytes,no_bytes)]
    return self._raw(selected_bytes,yes.dtype,u=u,dst=self._scratch(yes.dtype),cache=False)

  def _where(self, u:UOp) -> UOp:
    if len(u.src) != 3: raise _RKGenericReject
    if u is self.root and u.dtype.scalar() is dtypes.uchar and (source:=_typed_cast_source(u.src[1],dtypes.uchar,dtypes.half)) is not None and (condition:=u.src[0]).op is Ops.CMPLT and condition.src[0].op is Ops.CONST and float(condition.src[0].arg)==0.0 and condition.src[1].key==source.key and u.src[2].op is Ops.CONST and int(u.src[2].arg)==0:  # noqa: E501
      return self.lower(source.alu(Ops.MAX,UOp.const(0.0,dtypes.half)).cast(dtypes.uchar))
    if u is self.root and u.dtype.scalar() is dtypes.int and all(arm.op is Ops.CONST and -32768<=int(arm.arg)<=32767 for arm in u.src[1:]):  # noqa: E501
      selector=self.lower(u.src[0]); yes,no=(self._constant(UOp.const(int(arm.arg),dtypes.int16)) for arm in u.src[1:]); selected=self._lower_recipe(u,_i16_select(selector,yes,no))  # noqa: E501
      return selected if self.out_param.dtype.scalar() is dtypes.int16 else self._convert(u,selected,dtypes.int)
    if u is self.root and u.dtype.scalar() in (dtypes.half, dtypes.int16) and _is_static_expr(u.src[0]):
      dtype = u.dtype.scalar()
      routes:dict[UOp, np.ndarray] = {}
      def route(node:UOp, active:np.ndarray) -> None:
        if node.op is Ops.WHERE and _is_static_expr(node.src[0]):
          selector = np.asarray(_static_values(self.out_index, node.src[0], self.count, int),dtype=bool)
          for child,take in zip(node.src[1:], (selector, ~selector)):
            route(child, active&take)
        else: routes[node] = active if node not in routes else routes[node]|active
      route(u, np.ones(self.count,dtype=bool))
      expected, itemsize, commits, lanes = dtype, dtype.itemsize, [], np.arange(self.count,dtype=np.int64)
      for partial,(leaf,mask) in enumerate(routes.items()):
        value=self._operand(leaf,dtype,dtype is dtypes.half and leaf.op is Ops.LOAD and (param:=_root_param(leaf.src[0])) is not None and param.src[0].op is Ops.CONST and int(param.src[0].arg)<self.count)  # noqa: E501
        offsets = tuple(np.where(mask,lanes+value.arg.addend//itemsize,-1).tolist())
        commits.append(RKGather(value.arg._replace(addend=0),self.out,self.count,offsets=offsets,partial=bool(partial),itemsize=itemsize))
      self.program.extend(commits)
      return self._carrier(self.out,expected)
    for fold in (_fold_where_abs, _fold_ordered_where):
      if (recipe:=fold(u)) is not None: return self.lower(recipe)
    return self._raw_where(u)

  def _cast(self, u:UOp) -> UOp:
    dtype,source_dtype=u.dtype.scalar(),u.src[0].dtype.scalar()
    if dtype in (dtypes.bool,dtypes.uchar) and source_dtype is not dtypes.half or \
       dtype is dtypes.float and source_dtype not in (dtypes.half,dtypes.int,dtypes.bool) or \
       dtype is dtypes.int and source_dtype is dtypes.float: raise _RKGenericReject(f"cast {source_dtype}->{dtype}")
    source_u=u.src[0]
    if dtype is dtypes.uchar:
      if (relu:=_relu_operand(source_u)) is not None: source_u=relu.alu(Ops.MAX,UOp.const(0.0,dtypes.half))
      truncated=_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source_u,)))
      source_u=truncated.alu(Ops.SUB,_native_same(truncated.alu(Ops.MUL,UOp.const(1.0/256.0,dtypes.half)),_NATIVE_FLOOR).alu(
        Ops.MUL,UOp.const(256.0,dtypes.half)))
    elif dtype is dtypes.bool: source_u=_positive_mask(UOp(Ops.MAX,dtypes.half,src=(source_u,source_u),arg=_NATIVE_ABS))
    elif dtype is dtypes.int and source_dtype is dtypes.half: source_u=_fold_trunc(UOp(Ops.TRUNC,dtypes.half,src=(source_u,)))
    elif source_dtype is dtypes.bool and dtype in (dtypes.half,dtypes.float): source_u=source_u.where(UOp.const(1.0,dtypes.half),UOp.const(0.0,dtypes.half))  # noqa: E501
    if dtype is dtypes.half and source_dtype is dtypes.float:
      if _is_static_expr(u.src[0]): return self._static(u)
      source=self._load(u.src[0]) if u.src[0].op is Ops.LOAD else self.lower(_fp32_expr_to_half(u.src[0]))
    elif dtype is dtypes.half and source_dtype is dtypes.int and (recipe:=_int_info(u.src[0])[1]) is not None: source=self.lower(recipe)
    else: source=self.lower(source_u)
    if source_dtype is dtypes.int and source.dtype is dtypes.int16 and dtype in (dtypes.half,dtypes.float): source=self._convert(u.src[0],source,dtypes.int)  # noqa: E501
    if dtype in (dtypes.int16,dtypes.uint) and source.dtype is not self._layout(dtype): raise _RKGenericReject
    return self._convert(u,source,self._layout(dtype),dtype in (dtypes.bool,dtypes.uchar))

  def _finish_value(self, result:UOp, dtype:DType) -> None:
    expected=dtypes.int if dtype is dtypes.int else self._layout(dtype)
    if dtype is dtypes.int and result.dtype is dtypes.int16: result=self._convert(self.root,result,expected)
    if result.dtype is not expected: raise _RKGenericReject
    if dtype is dtypes.float: self._convert(self.root,result,dtypes.float,dst=self.out); return
    if result.arg == self.out: return
    if dtype in (dtypes.half,dtypes.int16): self._emit(self._carrier(self.out,expected),result,result,_EW_CFG[Ops.MAX]); return
    self.program.append(_raw_gather(result.arg,self.out_param.arg.slot,self.count,stride=1 if dtype is dtypes.int else 2,itemsize=dtype.itemsize))  # noqa: E501

  def lower(self, u:UOp) -> UOp:
    if u in self.values: return self.values[u]
    if u.op is Ops.NOOP and isinstance(u.arg,RKArg): return u
    dtype = u.dtype.scalar()
    if u.op is Ops.CONST: value = self._constant(u)
    elif (dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.uint, dtypes.bool, dtypes.uchar) and u in self.semantic_nodes and _is_static_expr(u) and  # noqa: E501
          not any(isinstance(node.arg, str) and node.arg.startswith("rockchip_") for node in u.toposort())):
      value = self._static(u)
    elif u.op in (Ops.INDEX, Ops.LOAD): value = self.lower(u.load()) if u.op is Ops.INDEX else self._load(u)
    elif u.op is Ops.BITCAST and len(u.src) == 1:
      source = self.lower(u.src[0])
      if dtype is dtypes.int16 and u.src[0].dtype.scalar() is dtypes.half and source.dtype is dtypes.half:
        value = self._carrier(source.arg,dtypes.int16)
      elif dtype is dtypes.half and u.src[0].dtype.scalar() is dtypes.int16 and source.dtype is dtypes.int16:
        value = self._carrier(source.arg,dtypes.half)
      else: raise _RKGenericReject(f"bitcast {u.src[0].dtype.scalar()}->{dtype}")
      if u is self.root and value.arg != self.out:
        self.program.append(_raw_gather(value.arg,self.out_param.arg.slot,self.count,stride=1,itemsize=2))
        value = self._carrier(self.out,value.dtype)
    elif u.op is Ops.CAST and len(u.src) == 1: value=self._cast(u)
    elif dtype is dtypes.bool and u.op in (Ops.MUL,Ops.MAX,Ops.AND,Ops.OR,Ops.XOR): value=self._compare(u)
    elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.NEG, Ops.RECIPROCAL): value = self._alu(u)
    elif u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): value = self._compare(u)
    elif u.op in (Ops.AND,Ops.OR,Ops.XOR) and dtype in (dtypes.int16,dtypes.int): value=self._integer_bitwise(u)
    elif u.op in (Ops.SHL, Ops.SHR) and dtype in (dtypes.int, dtypes.uint): value = self._int32_shift(u)
    elif u.op is Ops.CMOD and dtype is dtypes.int and self.int_layout is dtypes.int16 and (recipe:=_int_info(u)[1]) is not None: value=self.lower(recipe.cast(dtypes.int))  # noqa: E501
    elif u.op in (Ops.CDIV,Ops.CMOD) and dtype is dtypes.int and (u.op is not Ops.CMOD or self.int_layout is not dtypes.int16): value=self._int32_divmod(u)  # noqa: E501
    elif u.op is Ops.WHERE: value = self._where(u)
    elif u.op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN) and len(u.src) == 1 and dtype is dtypes.half and \
         (recipe:=_DPU_MATH[u.op](u.src[0])) is not None:
      value = self.lower(_physical_recipe(recipe,(u.src[0],)))
    else: raise _RKGenericReject(f"uop {u.op.name} {dtype}")
    return self.values.setdefault(u, value)

  def finish(self) -> RKImage:
    nodes=self.root.toposort(); raw_predicate_inputs={src for node in nodes if node.op in (Ops.CMPNE,Ops.CMPEQ) for src in node.src if src.dtype.scalar() is dtypes.half and (src.op is Ops.MAX and src.arg is None and len(src.src)==2 or src.op is Ops.MUL and any(term.op is Ops.CONST and float(term.arg)==-1.0 for term in src.src))}; predicated=any(node.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ,Ops.WHERE) and not _is_static_expr(node) for node in nodes); blocked:set[UOp]=set(); typed_loads={load for load in set(itertools.chain.from_iterable(map(_semantic_loads,nodes))) if load.dtype.scalar() is dtypes.half and _typed_load_plan(load,dtypes.half,self.out_index,self.count) is not None} if len(nodes)>800 else set()  # noqa: E501
    # A dynamic predicate taints only its consumers; independent FP16 arithmetic can form one physical prelude.
    if len(nodes)>800:
      for node in nodes:
        if (node.op in (Ops.CMPLT,Ops.CMPNE,Ops.CMPEQ,Ops.WHERE) and not _is_static_expr(node)) or any(src in blocked for src in node.src): blocked.add(node)  # noqa: E501
        # A maximal compensated ADD owns its prefixes; eagerly lowering each prefix only creates unused physical copies.
        elif ((not predicated and node.dtype.scalar() in (dtypes.half,dtypes.int16,dtypes.bool,dtypes.uchar) and node.op in (Ops.CONST,Ops.LOAD,Ops.CAST,*GroupOp.ALU)) or (node.dtype.scalar() is dtypes.half and node.op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.FDIV,Ops.NEG,Ops.RECIPROCAL) and node not in raw_predicate_inputs and all(load in typed_loads for load in _semantic_loads(node)))): self.lower(node)  # noqa: E501
      for node in nodes:
        if node.dtype.scalar() is dtypes.bool and node.op in (Ops.MUL,Ops.MAX,Ops.AND,Ops.OR): self.lower(node)
    result, dtype = self.lower(self.root), self.out_param.dtype.scalar(); self._finish_value(result,dtype)
    return _reuse_linear_scratch(RKImage(tuple(self.scratch),program=tuple(self.program)))

def _expand_math_uops(root:UOp, *, accurate_adds:bool=True) -> UOp:
  """Expand semantic math UOps before physical allocation so the complete recipe has one liveness graph."""
  if (ratio:=_fp32_ratio_to_half(root)) is not None: return ratio
  bounded_recipes = len(root.toposort()) <= _MAX_OPTIONAL_RECIPE_NODES
  if bounded_recipes: root=root.substitute({u:recipe for u in root.toposort() if (recipe:=_fold_quadratic(u)) is not None})
  @functools.cache
  def rewrite(u:UOp) -> UOp:
    if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and len(u.src) == 1 and u.src[0].dtype.scalar() is dtypes.float and not _has_runtime_address(u.src[0]):  # noqa: E501
      if u.src[0].op is Ops.SIN: return rewrite(_physical_recipe(_dpu_sin(u.src[0].src[0]),(u.src[0].src[0],)))
      try: return _canonical_half_storage(u.src[0])
      except _RKGenericReject: pass
    if accurate_adds and bounded_recipes and u.op is Ops.ADD and u.dtype.scalar() is dtypes.half and u.arg is None and (recipe:=_accurate_add_recipe(u)) is not None: return recipe  # noqa: E501
    mapped = u.replace(src=tuple(rewrite(src) for src in u.src))
    if mapped.dtype.scalar() is dtypes.float and mapped.op in (Ops.WHERE,Ops.ADD,Ops.MUL) and not _is_static_expr(mapped): mapped=UOp(Ops.WHERE,dtypes.half,src=(mapped.src[0],mapped.src[1].cast(dtypes.half),mapped.src[2].cast(dtypes.half)),arg=mapped.arg) if mapped.op is Ops.WHERE else mapped.src[0].cast(dtypes.half).alu(mapped.op,mapped.src[1].cast(dtypes.half))  # noqa: E501
    if mapped.op is Ops.CAST and mapped.dtype.scalar() is dtypes.half and len(mapped.src)==1 and mapped.src[0].dtype.scalar() is dtypes.half: mapped=mapped.src[0]  # noqa: E501
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
  if root.op is Ops.MAX: root=root.substitute({u:u.replace(src=(u.src[0],u.src[1].const_like(-65504.0),u.src[2])) for u in root.toposort() if u.op is Ops.WHERE and u.src[1].op is Ops.CONST and u.src[1].dtype.scalar() in (dtypes.half,dtypes.float) and math.isinf(float(u.src[1].arg)) and float(u.src[1].arg)<0.0})  # noqa: E501
  nodes=root.toposort()
  neutrals={u:u.const_like(-2048) for u in nodes if u.op is Ops.CONST and u.dtype.scalar() is dtypes.int and int(u.arg)==dtypes.int.min}
  return root.substitute({maximum:maximum.substitute(neutrals) for maximum in reversed(nodes) if maximum.op is Ops.MAX and maximum.dtype.scalar() is dtypes.int})  # noqa: E501

def _fold_static_terms(op:Ops, dtype:DType, terms:list[UOp], balanced:bool) -> UOp:
  while balanced and len(terms)>1: terms=[UOp(op,dtype,src=(terms[i],terms[i+1])) for i in range(0,len(terms)-1,2)]+(terms[-1:] if len(terms)&1 else [])  # noqa: E501
  return terms[0] if balanced else functools.reduce(lambda value,term:UOp(op,dtype,src=(value,term)),terms[1:],terms[0])

def _unroll_static_reduces(root:UOp, precise:bool=True) -> UOp:
  """Interpret canonical static REDUCE structure; horizontal reductions retain their specified order."""
  cache:dict[UOp, UOp] = {}; half_storage=root.dtype.scalar() is dtypes.half
  for u in root.toposort():
    if (mapped:=u.replace(src=tuple(cache[src] for src in u.src))).op is Ops.REDUCE:
      reduce_op,ranges=mapped.arg[0],list(mapped.src[1:])
      if reduce_op not in (Ops.ADD,Ops.MAX,Ops.MUL) or not ranges or any(r.op not in (Ops.RANGE,Ops.SPECIAL) for r in ranges): raise _RKGenericReject  # noqa: E501
      lanes=_static_lanes(tuple(ranges),*ranges,limit=_MAX_GENERIC_UNROLL,dependencies=False)
      if len(lanes[0])*len(mapped.src[0].toposort())>_MAX_GENERIC_EXPANDED_NODES: raise _RKGenericReject
      terms=[UOp.const(identity_element(reduce_op,u.dtype),u.dtype)]
      terms.extend(mapped.src[0].substitute({r:r.const_like(int(value)) for r,value in zip(ranges,values)},walk=True) for values in zip(*lanes))
      fold_dtype=dtypes.half if half_storage and reduce_op is Ops.ADD and u.dtype.scalar() is dtypes.float else u.dtype
      if fold_dtype is dtypes.half and u.dtype.scalar() is dtypes.float: terms=[_fp32_expr_to_half(term) for term in terms]
      nonzero=[term for term in terms if not (term.op is Ops.CONST and float(term.arg)==0.0)] if reduce_op is Ops.ADD and fold_dtype.scalar() is dtypes.half else []  # noqa: E501
      if precise and nonzero and all(term.op is Ops.MUL and term.dtype.scalar() is dtypes.half for term in nonzero): reduced=_precise_mul_sum(nonzero)  # noqa: E501
      elif precise and nonzero and u.dtype.scalar() is dtypes.float and any(axis not in ranges for axis in mapped.src[0].toposort() if axis.op in (Ops.RANGE,Ops.SPECIAL)): reduced=_kahan_sum(nonzero)  # noqa: E501
      else: reduced=_fold_static_terms(reduce_op,fold_dtype,terms,reduce_op is Ops.ADD and u.dtype.scalar() is dtypes.float or reduce_op is Ops.MAX and u.dtype.scalar() is dtypes.int)  # noqa: E501
      mapped=reduced.cast(u.dtype) if fold_dtype is not u.dtype else reduced
    cache[u] = mapped
  result=cache[root].substitute({u:u.const_like(typing_cast(int|float|bool,_eval_static(u,{}).item())) for u in cache[root].toposort() if _is_static_expr(u) and not _index_ranges(u)},walk=True)  # noqa: E501
  return result.substitute({u:u.replace(src=(u.src[0],u.src[1].simplify(),*u.src[2:])) for u in result.toposort() if u.op is Ops.INDEX and len(u.src)>1},walk=True)  # noqa: E501

def _lower_uop_program(uops:list[UOp], *, vectorize_reductions:bool=True) -> RKImage|None:
  """Lower a composable typed UOp program; return None for the legacy correctness oracle."""
  if any(u.op is Ops.PARAM and not 0 <= u.arg.slot <= _RKIMAGE_U16_MAX for u in uops): return None
  accepted = (dtypes.half, dtypes.float, dtypes.int16, dtypes.int, dtypes.bool, dtypes.uchar)
  strict_output, local_output, output_stores = _outs(uops)
  if len(output_stores) > 1:
    lower_store = functools.partial(_lower_uop_program, vectorize_reductions=vectorize_reductions)
    if (combined:=lower_store(list(UOp(Ops.SINK, src=(output_stores[0],)).toposort()))) is None: return None
    for store in output_stores[1:]:
      if (child:=lower_store(list(UOp(Ops.SINK,src=(store,)).toposort()))) is None or \
         (combined:=_append_inplace_image(combined,child)) is None: return None
    return combined
  strict_output, local_output = (_admit(output, accepted) for output in (strict_output, local_output))
  if (selected:=_try(strict_output,dtypes.half,_lower_output_selector,uops,v=vectorize_reductions)) is not None: return selected
  if (cmac:=_try(local_output, (dtypes.half,dtypes.float,dtypes.int,dtypes.bool), _lower_reduction, uops, v=vectorize_reductions)) is not None: return cmac  # noqa: E501
  if (mixed:=_try(strict_output,dtypes.half,_lower_cmac_storage_epilogue,uops,v=vectorize_reductions)) is not None: return mixed
  if (image:=_try(strict_output,dtypes.int,_lower_raw_fp16_bitcast)) is not None: return image
  storage_precision,storage_product_adds=any(u.dtype.scalar() is dtypes.float for u in uops),False
  if storage_precision and (storage_output:=_admit(local_output,dtypes.half)) is not None:
    try:
      storage_root=storage_output[4]; storage_product_adds=any((boundary is not storage_root or len(boundary.src[0].toposort())>64) and _accurate_add_recipe(boundary) is not None for boundary in storage_root.toposort() if _typed_cast_source(boundary,dtypes.half,dtypes.float) is not None)  # noqa: E501
      storage_root=_expand_math_uops(storage_root,accurate_adds=False); local_output=(storage_output[0].replace(src=(storage_output[0].src[0],storage_root)),*storage_output[1:4],storage_root)  # noqa: E501
    except _RKGenericReject: pass
  if (output:=local_output) is None or len(output[0].src) != 2:
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject("output store")
    return None
  if output[2] <= 0: return RKImage()
  try:
    if not ((affine:=typing_cast(tuple[int, dict[UOp, int]]|None, _linear_index(output[3]))) is not None and affine[0] == 0 and
            set(affine[1]) == set(_index_ranges(output[3])) and _affine_output_axes(affine, output[2]) is not None) and \
       _static_values(output[3], output[3], output[2], int) != tuple(range(output[2])): return None
    root=_finite_int_max_neutrals(_unroll_static_reduces(output[4]) if Ops.REDUCE in (u.op for u in uops) else output[4])
    root = _expand_math_uops(root,accurate_adds=not storage_precision or storage_product_adds) if len(root.toposort()) <= 256 else recipe if (base:=_strip_cast(root)).dtype.scalar() is dtypes.half and (recipe:=_accurate_add_recipe(base,pure=True)) is not None else root  # noqa: E501
    if len(n:=root.toposort()) > _MAX_GENERIC_EXPANDED_NODES and os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise _RKGenericReject(f"expanded nodes {len(n)}")  # noqa: E501
    if len(n) > _MAX_GENERIC_EXPANDED_NODES: return None
    if root is not output[4]: output = (output[0].replace(src=(output[0].src[0], root)), *output[1:4], root)
    image = RKContext(output).finish()
    if len(image.scratch)>_RKIMAGE_U16_MAX and os.getenv("ROCKCHIP_UOPS_DEBUG","0")=="1": raise _RKGenericReject(f"image scratch count {len(image.scratch)}")  # noqa: E501
    if len(image.scratch)>_RKIMAGE_U16_MAX: return None
    return image
  except (_RKGenericReject, RuntimeError, ValueError, KeyError):
    if os.getenv("ROCKCHIP_UOPS_DEBUG", "0") == "1": raise
    return None


class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

def _const_operand(u:UOp, op:Ops, value:float|None=None) -> tuple[UOp, UOp]|None: return None if u.op is not op else next(
  ((a, b) for a,b in (u.src, u.src[::-1]) if b.op is Ops.CONST and (value is None or float(b.arg) == value)), None)

def _positive_mask(u:UOp) -> UOp: return UOp(Ops.MAX, dtypes.half, src=(u, u), arg=_NATIVE_POSITIVE_MASK)

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
  magnitude = u.alu(Ops.MAX, UOp.const(0.0, dtypes.half)).alu(Ops.MUL, UOp.const(256.0, dtypes.half)).alu(Ops.MUL, UOp.const(256.0, dtypes.half)).alu(Ops.MUL, UOp.const(256.0, dtypes.half))  # noqa: E501
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

def _preserve_infinite_division_sign(x:UOp) -> UOp|None:
  """RK3588 FDIV ignores the denominator sign for an infinite numerator; rebuild it with finite DPU intermediates."""
  numerator, denominator = x.src
  if numerator.op is not Ops.CONST or not math.isinf(value:=float(numerator.arg)): return None
  return UOp.const(-1.0 if value < 0 else 1.0, dtypes.half).alu(Ops.FDIV, denominator).alu(Ops.FDIV, UOp.const(0.0, dtypes.half))

def _fold_quadratic(root:UOp) -> UOp|None:
  """Scale sqrt(x*x +/- 1), and stabilize its canonical natural-log envelope."""
  logarithm=next((logarithm for logarithm,scale in ((root.src,root.src[::-1]) if root.op is Ops.MUL and len(root.src)==2 else ()) if scale.op is Ops.CONST and abs(float(scale.arg)-math.log(2))<1e-12 and logarithm.op is Ops.LOG2 and len(logarithm.src)==1 and logarithm.src[0].op is Ops.ADD),None)  # noqa: E501
  radical=root if root.op is Ops.SQRT and len(root.src)==1 else next((term for term in logarithm.src[0].src if term.op is Ops.SQRT and len(term.src)==1),None) if logarithm is not None else None  # noqa: E501
  if (matched:=next(((square.src[0],float(offset.arg)) for square,offset in ((radical.src[0].src,radical.src[0].src[::-1]) if radical is not None and radical.src[0].op is Ops.ADD else ()) if square.op is Ops.MUL and len(square.src)==2 and square.src[0].key==square.src[1].key and offset.op is Ops.CONST and float(offset.arg) in (-1.0,1.0) and (logarithm is None or any(term is not radical and term.key==square.src[0].key for term in logarithm.src[0].src))),None)) is None: return None  # noqa: E501
  source,offset=matched; source=source.cast(dtypes.half); magnitude=UOp(Ops.MAX,dtypes.half,src=(source,source),arg=_NATIVE_ABS)
  scale=_native_min(magnitude,_half(65504.0)).alu(Ops.MAX,_half(1.0)); ratio=source.alu(Ops.FDIV,scale)
  scaled=scale.alu(Ops.MUL,ratio.alu(Ops.MUL,ratio).alu(Ops.ADD,_half(offset).alu(Ops.FDIV,scale.alu(Ops.MUL,scale))).sqrt())
  if logarithm is None: return scaled
  result=(magnitude if offset==1 else source).alu(Ops.ADD,scaled).log2().alu(Ops.MUL,_half(math.log(2)))
  return result.alu(Ops.MUL,source.alu(Ops.FDIV,magnitude.alu(Ops.MAX,_half(2**-24)))) if offset==1 else result.alu(Ops.ADD,(valid:=_half(1).alu(Ops.SUB,_finite_positive_mask(_half(1).alu(Ops.SUB,source)))).alu(Ops.FDIV,valid).alu(Ops.SUB,_half(1)))  # noqa: E501
def _dpu_math_base(source:UOp) -> tuple[UOp, UOp, UOp, Callable[[UOp], UOp]]:
  source, zero, one = source.cast(dtypes.half), _half(0.0), _half(1.0)
  return source, zero, one, _positive_mask if source.op in (Ops.INDEX, Ops.LOAD) else _finite_positive_mask

def _dpu_sqrt(source:UOp) -> UOp|None:
  """Approximate FP16 sqrt with range-independent Babylonian iterations on DPU EW."""
  source, zero, one, _ = _dpu_math_base(source); finite = UOp(Ops.MAX, source.dtype, src=(source.alu(Ops.MAX, zero), UOp.const(65504.0, dtypes.half)), arg=_NATIVE_MIN)  # noqa: E501
  safe = finite.alu(Ops.MAX, UOp.const(2**-24, dtypes.half))
  estimate = safe.alu(Ops.MAX, one)
  for _ in range(14): estimate = estimate.alu(Ops.ADD, safe.alu(Ops.FDIV, estimate)).alu(Ops.MUL, UOp.const(0.5, dtypes.half))
  valid = one.alu(Ops.SUB, _positive_mask(zero.alu(Ops.SUB, source))); return source.alu(Ops.FDIV, estimate).alu(Ops.ADD, valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one))  # noqa: E501

def _dpu_periodic_reduce(source:UOp, reciprocal_period:float, split:tuple[float, ...]) -> tuple[UOp,UOp]:
  """Reduce a finite FP16 angle with split constants so large products retain their residual."""
  half,one=dtypes.half,UOp.const(1.0,dtypes.half); reduced=UOp(Ops.MAX,half,src=(source.cast(half).alu(Ops.MAX,UOp.const(-10000.0,half)),UOp.const(10000.0,half)),arg=_NATIVE_MIN); correction=UOp.const(0.0,half)  # noqa: E501
  # A second quotient removes the small residual left by the rounded FP16 bulk quotient.
  for _ in range(2):
    quotient=reduced.alu(Ops.MUL,UOp.const(reciprocal_period,half)); magnitude=UOp(Ops.MAX,half,src=(quotient,quotient),arg=_NATIVE_ABS)
    multiple=_native_same(magnitude.alu(Ops.ADD,UOp.const(0.5,half)),_NATIVE_FLOOR).alu(Ops.MUL,_finite_positive_mask(quotient).alu(Ops.MUL,UOp.const(2.0,half)).alu(Ops.SUB,one))  # noqa: E501
    reduced,correction=_precise_add_parts([reduced,correction,*(multiple.alu(Ops.MUL,UOp.const(-coefficient,half)) for coefficient in split)])
  return reduced,correction

def _dpu_sin(source:UOp) -> UOp:
  """Lower SIN and Tinygrad's COS phase spelling to one bounded FP16 DPU polynomial."""
  cosine=False
  if source.dtype.scalar() is dtypes.float:
    phase=_const_operand(source,Ops.ADD); negative=_const_operand(phase[0],Ops.MUL,-1.0) if phase is not None and abs(float(phase[1].arg)-math.pi/2)<1e-12 else None  # noqa: E501
    base=_typed_cast_source(negative[0],dtypes.float,dtypes.half) if negative is not None else None
    source,cosine=(base,True) if base is not None else (_canonical_half_storage(source),False)
  source=source.cast(dtypes.half); one=UOp.const(1.0,dtypes.half); split=(4.0,2.0,0.25,0.03125,2*math.pi-6.28125)
  reduced,reduction_error=_dpu_periodic_reduce(source,1/(2*math.pi),split); invalid=source.alu(Ops.MUL,UOp.const(0.0,dtypes.half))
  magnitude=UOp(Ops.MAX,dtypes.half,src=(reduced,reduced),arg=_NATIVE_ABS); reflected=_finite_positive_mask(magnitude.alu(Ops.SUB,UOp.const(math.pi/2,dtypes.half)))  # noqa: E501
  pi_minus=UOp.const(3.0,dtypes.half).alu(Ops.SUB,magnitude).alu(Ops.ADD,UOp.const(0.140625,dtypes.half)).alu(Ops.ADD,UOp.const(math.pi-3.140625,dtypes.half)); angle=magnitude.alu(Ops.MUL,one.alu(Ops.SUB,reflected)).alu(Ops.ADD,pi_minus.alu(Ops.MUL,reflected))  # noqa: E501
  if cosine:
    reduced_sign=one.alu(Ops.SUB,_finite_positive_mask(UOp.const(0.0,dtypes.half).alu(Ops.SUB,reduced)).alu(Ops.MUL,UOp.const(2.0,dtypes.half))); direction=reflected.alu(Ops.MUL,UOp.const(2.0,dtypes.half)).alu(Ops.SUB,one)  # noqa: E501
    angle,correction=_precise_add_parts([magnitude,UOp.const(-math.pi/2,dtypes.half),reduction_error.alu(Ops.MUL,reduced_sign),UOp.const(float_to_fp16(math.pi/2)-math.pi/2,dtypes.half)])  # noqa: E501
    angle,correction=angle.alu(Ops.MUL,direction),correction.alu(Ops.MUL,direction)
  square=angle.alu(Ops.MUL,angle); coefficients=(1/362880,-1/5040,1/120,-1/6,1)
  sign=one.alu(Ops.SUB,(reflected if cosine else _finite_positive_mask(UOp.const(0.0,dtypes.half).alu(Ops.SUB,reduced))).alu(Ops.MUL,UOp.const(2.0,dtypes.half)))  # noqa: E501
  result=angle.alu(Ops.MUL,polyN(square,list(coefficients))); return (result.alu(Ops.ADD,correction) if cosine else result).alu(Ops.MUL,sign).alu(Ops.ADD,invalid)  # noqa: E501

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
  result = polyN(bounded.alu(Ops.SUB,integer),[0.0013333558,0.0096181291,0.0555041087,0.2402265069,0.6931471806,1]).alu(Ops.MUL,scale)
  below, above = mask_fn(UOp.const(-24.0, dtypes.half).alu(Ops.SUB, source)), mask_fn(source.alu(Ops.SUB, UOp.const(15.9921875, dtypes.half)))
  finite = UOp(Ops.MUL, dtypes.half, src=(result, one.alu(Ops.SUB, below)), arg=_NATIVE_MASK_MUL)
  return finite.alu(Ops.ADD, one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one))

def _dpu_log2(source:UOp) -> UOp:
  """Approximate FP16 LOG2 without LUTs using threshold exponent extraction and an atanh polynomial."""
  source, zero, one, _ = _dpu_math_base(source); mask_fn=_finite_positive_mask
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
  result = exponent.alu(Ops.ADD, z.alu(Ops.MUL, polyN(z.alu(Ops.MUL, z), [1/9,1/7,1/5,1/3,1])).alu(Ops.MUL, _half(2/math.log(2))))
  nonzero = mask_fn(source).alu(Ops.MAX, mask_fn(zero.alu(Ops.SUB, source)))
  zero_correction, valid = UOp.const(-1.0, dtypes.half).alu(Ops.FDIV, nonzero).alu(Ops.ADD, one), one.alu(Ops.SUB, mask_fn(zero.alu(Ops.SUB, source)))
  negative_correction, above = valid.alu(Ops.FDIV, valid).alu(Ops.SUB, one), mask_fn(source.alu(Ops.SUB, UOp.const(65504.0, dtypes.half)))
  inf_correction = one.alu(Ops.FDIV, one.alu(Ops.SUB, above)).alu(Ops.SUB, one)
  return result.alu(Ops.ADD, zero_correction).alu(Ops.ADD, negative_correction).alu(Ops.ADD, inf_correction)

_DPU_MATH = {Ops.SQRT:_dpu_sqrt, Ops.EXP2:_dpu_exp2, Ops.LOG2:_dpu_log2, Ops.SIN:_dpu_sin}
class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4, direct_reduces = False, False, False, True
  code_for_op = {Ops.ADD: lambda: None, Ops.SUB: lambda: None, Ops.MUL: lambda: None, Ops.MAX: lambda: None,
                 Ops.FDIV: lambda: None, Ops.SQRT: lambda: None, Ops.EXP2: lambda: None, Ops.LOG2: lambda: None, Ops.SIN: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def supported_dtypes(self): return {dtypes.half, dtypes.int16}
  def render(self, uops:list[UOp]) -> str:
    image = _lower_uop_program(uops)
    if image is None: raise RuntimeError("RKPLAN_REJECT:generic_uops " + repr([(i, u.op.name, str(u.dtype)) for i,u in enumerate(uops)]))
    return base64.b64encode(encode_image(image,validate=False)).decode()

class RockchipBoolRenderer(RockchipRenderer):
  """Expose one 16-lane local bool tile that the renderer consumes as grouped DPU reduction work."""
  has_local, shared_max = True, 16
