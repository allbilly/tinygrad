import ctypes, functools, hashlib, itertools, math, struct, threading
import numpy as np
import pytest
from collections.abc import Callable
from types import SimpleNamespace
from tinygrad import Tensor
from tinygrad.codegen import expand_horizontal_reduce, to_program, to_program_cache
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKCMAC, RKImage, RKEWMode, RKEWOp,
  RKGather,
  _EW_CFG, _EW_CFG_ABS, _EW_CFG_FLOOR, _EW_CFG_MIN, _NATIVE_SIGN, _MAX_EW_ELEMS_FP16, _RKIMAGE_U16_MAX,
  _canonical_half_storage, _finite_int_max_neutrals, _fp32_expr_to_half, _gather_plan, _static_lanes,
  _lower_uop_program, _reuse_linear_scratch, _unroll_static_reduces, RockchipRenderer, decode_image, emit_cmac_stage, encode_image)
from tinygrad.runtime import ops_rockchip as rockchip_runtime
import tinygrad.renderer.rockchip as rockchip_renderer
from tinygrad.uop.ops import AxisType, Ops, UOp

def _typed_ops(image:RKImage, cls): return tuple(op for op in image.program if isinstance(op,cls))
def _ew_ops(image:RKImage) -> tuple[RKEWOp, ...]: return _typed_ops(image,RKEWOp)
_TYPED_EW_MODES=(RKEWMode.INT16,RKEWMode.INT32,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT32,
                 RKEWMode.HALF_TO_INT16,RKEWMode.INT32_TO_HALF)
_INT32_OUTPUT_MODES=(RKEWMode.INT32,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT32)
def _cmac(image:RKImage) -> RKCMAC|None:
  cmacs=_typed_ops(image,RKCMAC)
  assert len(cmacs)<=1
  return cmacs[0] if cmacs else None
def _static_gathers(image:RKImage) -> tuple[RKGather, ...]:
  return tuple(op for op in image.program if isinstance(op,RKGather) and op.index is None)
def _constant_gathers(image:RKImage) -> tuple[RKGather, ...]:
  return tuple(op for op in _static_gathers(image) if op.src is None and len(op.values)==1)
def _constant_bytes(image:RKImage) -> bytes:
  return b"".join(int(op.values[0]).to_bytes(op.itemsize,"little") for op in _constant_gathers(image))
def _runtime_gathers(image:RKImage, scatter:bool|None=None) -> tuple[RKGather, ...]:
  return tuple(op for op in image.program if isinstance(op,RKGather) and op.index is not None and
               (scatter is None or (op.dst.kind is RKBufferKind.ARG) is scatter))
def _address_gather(image:RKImage) -> RKGather:
  gathers=_runtime_gathers(image,False)
  assert len(gathers)==1 and gathers[0].index is not None
  return gathers[0]
def _initial_gathers(image:RKImage) -> tuple[RKGather, ...]:
  boundary=next((i for i,op in enumerate(image.program) if isinstance(op,(RKEWOp,RKCMAC))),len(image.program))
  return tuple(op for op in image.program[:boundary] if isinstance(op,RKGather) and op.index is None and op not in _constant_gathers(image))
def _output_gathers(image:RKImage) -> tuple[RKGather, ...]:
  boundary=max((i for i,op in enumerate(image.program) if isinstance(op,(RKEWOp,RKCMAC))),default=-1)
  return tuple(op for op in image.program[boundary+1:] if isinstance(op,RKGather) and op.index is None and op.dst.kind is RKBufferKind.ARG)
def _intermediate_gathers(image:RKImage) -> tuple[RKGather, ...]:
  excluded={id(op) for op in _constant_gathers(image)+_initial_gathers(image)+_output_gathers(image)}
  return tuple(op for op in _static_gathers(image) if id(op) not in excluded)
def _gather_point(image:RKImage, gather:RKGather) -> int:
  return sum(isinstance(op,RKEWOp) for op in image.program[:next(i for i,op in enumerate(image.program) if op is gather)])
def _gather_after(image:RKImage) -> int: return min((_gather_point(image,g) for g in _intermediate_gathers(image)),default=0)


def _program(dtype, value, count:int=4):
  out, axis = UOp.param(0, dtype, (count,)), UOp.range(count, 0)
  return list(out.index(axis).store(value(axis)).end(axis).sink().toposort())

def _cmac_weights(image:RKImage) -> tuple[int, ...]:
  assert _cmac(image) is not None and _cmac(image).n == 1 and _initial_gathers(image)[1].values
  return tuple(_initial_gathers(image)[1].values[(lane//32)*16*32+lane%32] for lane in range(_cmac(image).k))

def _slot_program(slot:int) -> list[UOp]:
  out, axis = UOp.param(slot, dtypes.half, (1,)), UOp.range(1, 0)
  return list(out.index(axis).store(UOp.const(0.0, dtypes.half)).end(axis).sink().toposort())

def test_argument_slots_reject_wire_overflow_before_encoding():
  for slot in (1 << 16, (1 << 16) + 1):
    assert _lower_uop_program(_slot_program(slot)) is None

def _int32_binary_program(value:Callable[[UOp, UOp], UOp], count:int=4) -> list[UOp]:
  out, lhs, rhs = (UOp.param(slot, dtypes.int, (count,)) for slot in range(3))
  axis = UOp.range(count, 0)
  return list(out.index(axis).store(value(lhs.index(axis).load(), rhs.index(axis).load())).end(axis).sink().toposort())

def _dynamic_load_program(count:int=4, extents:tuple[int, ...]=(9,), dtype=dtypes.half, *, normalized:bool=False,
                          external_gate:bool=False, repeat:int=1) -> list[UOp]:
  out, source, lane = UOp.param(0, dtype, (count,)), UOp.param(1, dtype, (math.prod(extents)*repeat,)), UOp.range(count, 0)
  coordinates:list[UOp] = []
  gate:UOp|None = None
  for axis,extent in enumerate(extents):
    raw = UOp.param(axis+2, dtypes.int, (count//repeat,)).index(lane//repeat).load()
    coordinate = (raw < 0).where(raw+extent, raw) if normalized else raw
    valid = ((coordinate < 0) != UOp.const(True, dtypes.bool)) & (coordinate < extent)
    coordinates.append(coordinate)
    gate = valid if gate is None else gate & valid
  index = coordinates[0]
  for coordinate,extent in zip(coordinates[1:], extents[1:]): index = index*extent+coordinate
  index = index*repeat+lane%repeat
  if external_gate: gate = gate & UOp.param(len(extents)+2, dtypes.bool, (count//repeat,)).index(lane//repeat).load()  # type: ignore[operator]
  assert gate is not None
  zero = UOp.const(0.0, dtype) if dtype is dtypes.half else UOp.const(0, dtype)
  return list(out.index(lane).store(source.index(index).load(zero, gate)).end(lane).sink().toposort())

def _dynamic_offset_program(data_offset:int=0, index_offset:int=0) -> list[UOp]:
  out, source, indices = UOp.param(0, dtypes.int, (1,)), UOp.param(1, dtypes.int, (data_offset+1,)), \
    UOp.param(2, dtypes.int, (index_offset+1,))
  lane = UOp.range(1, 0)
  dynamic = indices.index(lane+index_offset).load()
  gate = ((dynamic < 0) != UOp.const(True, dtypes.bool)) & (dynamic < 1)
  return list(out.index(lane).store(source.index(dynamic+data_offset).load(UOp.const(0, dtypes.int), gate)).end(lane).sink().toposort())

def _dynamic_total_load_program(dtype=dtypes.int, count:int=4, source_count:int=5, fill:int=-7) -> list[UOp]:
  out, source = UOp.param(0, dtype, (count,)), UOp.param(1, dtype, (source_count,))
  indices, mask, lane = UOp.param(2, dtypes.int, (count,)), UOp.param(3, dtypes.bool, (source_count,)), UOp.range(count, 0)
  total = mask.index(0).load().cast(dtypes.int)
  for index in range(1, source_count): total = total+mask.index(index).load().cast(dtypes.int)
  dynamic = indices.index(lane).load()
  gate = ((dynamic < 0) != UOp.const(True, dtypes.bool)) & (dynamic < source_count)
  selected = source.index(dynamic).load(UOp.const(0, dtype), gate)
  return list(out.index(lane).store((lane < total).where(selected, UOp.const(fill, dtype))).end(lane).sink().toposort())

def _apply_test_gather(gather:RKGather, buffer, linear:dict[int,np.ndarray]) -> None:
  dtype={1:np.uint8,2:np.uint16,4:np.uint32}[gather.itemsize]
  def view(arg:RKArg, lane_dtype=dtype, itemsize:int=gather.itemsize):
    assert arg.addend%itemsize==0
    return np.frombuffer(buffer(arg.kind,arg.index),dtype=lane_dtype)[arg.addend//itemsize:]
  lanes=linear.setdefault(gather.count,np.arange(gather.count,dtype=np.intp))
  dst=view(gather.dst)
  if gather.index is not None:
    assert gather.src is not None
    src=view(gather.src)
    indices=view(gather.index,{2:np.int16,4:np.int32}[gather.index_itemsize],gather.index_itemsize)[:gather.count].astype(np.intp)
    valid=(indices>=0)&(indices<len(dst) if gather.dst.kind is RKBufferKind.ARG else indices<len(src))
    if gather.dst.kind is RKBufferKind.ARG:
      for lane in range(gather.count):
        if valid[lane]: dst[indices[lane]]=src[lane]
    else:
      dst[:gather.count]=gather.fill_bits
      dst[np.nonzero(valid)[0]]=src[indices[valid]]
    return
  if gather.dst.kind is RKBufferKind.SCRATCH and not gather.partial and not gather.dst.addend and not gather.dst_addend: dst[:]=0
  dst_index=gather.dst_addend+lanes*gather.dst_stride
  if gather.values: dst[dst_index]=gather.values[0] if len(gather.values)==1 else gather.values
  elif gather.offsets:
    assert gather.src is not None
    src=view(gather.src)
    indices=np.asarray(gather.offsets,dtype=np.intp)
    valid=indices>=0
    if not gather.partial: dst[dst_index]=gather.fill_bits
    dst[dst_index[valid]]=src[indices[valid]]
  else:
    assert gather.src is not None
    src=view(gather.src)
    indices=np.full(gather.count,gather.base,dtype=np.intp)
    for divisor,limit,stride in gather.axes: indices+=(lanes//divisor%limit)*stride
    dst[dst_index]=src[indices]

def _execute_raw_dynamic_image(image:RKImage, output_bytes:int, *inputs:bytes) -> bytes:
  """Execute the selector's raw gathers plus native INT16 mask/reduction subset."""
  args, scratch = [bytearray(output_bytes), *(bytearray(value) for value in inputs)], [bytearray(size) for size in image.scratch]
  def buffer(kind:RKBufferKind, index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  def execute(op:RKEWOp) -> None:
    if op.mode in (RKEWMode.HALF,RKEWMode.STATEFUL,RKEWMode.COMPARE,RKEWMode.BOUNDED):
      def fp16(arg:RKArg) -> np.ndarray: return np.frombuffer(buffer(arg.kind,arg.index),dtype="<f2",count=op.count,offset=arg.addend)
      lhs,rhs=fp16(op.lhs).copy(),fp16(op.rhs).copy()
      if op.mode==RKEWMode.COMPARE: value=(lhs>0).astype("<f2")
      else:
        value=(lhs+rhs if op.ew_cfg==_EW_CFG[Ops.ADD] else lhs-rhs if op.ew_cfg==_EW_CFG[Ops.SUB] else
               lhs*rhs if op.ew_cfg==_EW_CFG[Ops.MUL] else np.maximum(lhs,rhs) if op.ew_cfg==_EW_CFG[Ops.MAX] else
               lhs/rhs if op.ew_cfg==_EW_CFG[Ops.FDIV] else np.floor(lhs) if op.ew_cfg==_EW_CFG_FLOOR else
               np.minimum(lhs,rhs) if op.ew_cfg==_EW_CFG_MIN else np.abs(lhs) if op.ew_cfg==_EW_CFG_ABS else None)
      assert value is not None, hex(op.ew_cfg)
      fp16(op.dst)[:]=value.astype("<f2")
      return
    if op.mode==RKEWMode.INT16: source_dtype=destination_dtype=np.dtype("<i2")
    elif op.mode==RKEWMode.INT32: source_dtype=destination_dtype=np.dtype("<i4")
    elif op.mode==RKEWMode.INT16_TO_INT32:
      source_dtype,destination_dtype=np.dtype("<i2"),np.dtype("<i4")
    elif op.mode==RKEWMode.HALF_TO_INT16:
      def view_fp(arg:RKArg,dtype) -> np.ndarray: return np.frombuffer(buffer(arg.kind,arg.index),dtype=dtype,count=op.count,offset=arg.addend)
      assert op.ew_cfg == _EW_CFG[Ops.MAX]
      view_fp(op.dst,"<i2")[:] = np.maximum(view_fp(op.lhs,"<f2"),view_fp(op.rhs,"<f2")).astype("<i2")
      return
    else: raise AssertionError(f"unsupported dynamic selector EW precision {op}")
    def view(arg:RKArg, dtype) -> np.ndarray: return np.frombuffer(buffer(arg.kind, arg.index), dtype=dtype, count=op.count, offset=arg.addend)
    lhs, rhs = view(op.lhs,source_dtype).astype(np.int64), view(op.rhs,source_dtype).astype(np.int64)
    if op.ew_cfg == _EW_CFG[Ops.ADD]: value = lhs+rhs
    elif op.ew_cfg == _EW_CFG[Ops.SUB]: value = lhs-rhs
    elif op.ew_cfg == _EW_CFG[Ops.MUL]: value = lhs*rhs
    elif op.ew_cfg == _EW_CFG[Ops.MAX]: value = np.maximum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_MIN: value = np.minimum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_ABS: value = np.abs(lhs)
    else: raise AssertionError(f"unsupported dynamic selector EW config {op.ew_cfg:#x}")
    value=np.clip(value,-32768,32767) if destination_dtype.itemsize==2 else (value+(1<<31))%(1<<32)-(1<<31)
    view(op.dst,destination_dtype)[:] = value.astype(destination_dtype)
  def execute_cmac(op:RKCMAC) -> None:
    ai,ao,_=rockchip_renderer._cmac_layout(op.n,op.k)
    lhs=np.frombuffer(buffer(op.lhs.kind,op.lhs.index),dtype="<f2",count=op.m*ai,offset=op.lhs.addend).reshape(op.m,ai)
    packed=np.frombuffer(buffer(op.rhs.kind,op.rhs.index),dtype="<f2",count=ao*ai,offset=op.rhs.addend)
    rhs=packed.reshape(ao//16,ai//32,16,32).transpose(1,3,0,2).reshape(ai,ao)
    value=lhs[:,:op.k].astype("<f4")@rhs[:op.k,:op.n].astype("<f4")
    if op.relu: value=np.maximum(value,0)
    storage=buffer(op.dst.kind,op.dst.index)
    if op.out_fp16:
      dst=np.frombuffer(storage,dtype="<f2",count=op.m*ao*2,offset=op.dst.addend)
      for row,col in itertools.product(range(op.m),range(op.n)): dst[row*ao*2+col//16*32+col%16]=value[row,col]
    else: np.frombuffer(storage,dtype="<f4",count=op.m*ao,offset=op.dst.addend).reshape(op.m,ao)[:,:op.n]=value
  linear:dict[int,np.ndarray]={}
  for op in image.program:
    if isinstance(op,RKGather): _apply_test_gather(op,buffer,linear)
    elif isinstance(op,RKEWOp): execute(op)
    elif isinstance(op,RKCMAC): execute_cmac(op)
    else: raise AssertionError(type(op))
  return bytes(args[0])

def _execute_integer_image(image:RKImage, *inputs:np.ndarray) -> np.ndarray:
  """Test-only physical executor for raw gathers and the signed integer EW subset used by INT32 division."""
  count = len(inputs[0])
  args = [bytearray(count*4), *(bytearray(value.astype("<i4").tobytes()) for value in inputs)]
  scratch = [bytearray(size) for size in image.scratch]
  def buffer(kind:RKBufferKind, index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  def view(arg:RKArg, dtype, lanes:int) -> np.ndarray:
    return np.frombuffer(buffer(arg.kind, arg.index), dtype=dtype, count=lanes, offset=arg.addend)
  def execute(op:RKEWOp) -> None:
    if op.mode==RKEWMode.INT16:
      source_dtype = destination_dtype = np.dtype("<i2")
    elif op.mode==RKEWMode.INT32:
      source_dtype = destination_dtype = np.dtype("<i4")
    elif op.mode==RKEWMode.INT16_TO_INT32:
      source_dtype, destination_dtype = np.dtype("<i2"), np.dtype("<i4")
    else: raise AssertionError(f"unsupported integer EW precision {op}")
    lhs = view(op.lhs, source_dtype, op.count).astype(np.int64)
    rhs = view(op.rhs, source_dtype, op.count).astype(np.int64)
    if op.ew_cfg == _EW_CFG[Ops.ADD]: result = lhs+rhs
    elif op.ew_cfg == _EW_CFG[Ops.SUB]: result = lhs-rhs
    elif op.ew_cfg == _EW_CFG[Ops.MUL]: result = lhs*rhs
    elif op.ew_cfg == _EW_CFG[Ops.MAX]: result = np.maximum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_MIN: result = np.minimum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_ABS: result = np.abs(lhs)
    else: raise AssertionError(f"unsupported integer EW config {op.ew_cfg:#x}")
    result = np.clip(result, -32768, 32767) if destination_dtype.itemsize == 2 else (result+(1<<31)) % (1<<32) - (1<<31)
    view(op.dst, destination_dtype, op.count)[:] = result.astype(destination_dtype)
  linear:dict[int,np.ndarray]={}
  for op in image.program:
    if isinstance(op,RKGather): _apply_test_gather(op,buffer,linear)
    elif isinstance(op,RKEWOp): execute(op)
    else: raise AssertionError("CMAC is outside the integer-image test executor")
  return np.frombuffer(args[0], dtype="<i4").copy()

def _assert_scratch_extent(image:RKImage, arg:RKArg, need:int) -> None:
  if arg.kind is RKBufferKind.SCRATCH:
    assert 0 <= arg.index < len(image.scratch)
    assert arg.addend >= 0 and need <= image.scratch[arg.index]

def _assert_decoded_image_bounds(image:RKImage) -> RKImage:
  decoded = decode_image(encode_image(image))
  def gather_indices(gather:RKGather) -> tuple[int, ...]:
    if gather.offsets: return tuple(index for index in gather.offsets if index >= 0)
    return tuple(gather.base+sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes) for lane in range(gather.count))
  for gather in _static_gathers(decoded):
    if gather.count <= 0: continue
    dst_indices=tuple(gather.dst_addend+lane*gather.dst_stride for lane in range(gather.count))
    _assert_scratch_extent(decoded, gather.dst,(max(dst_indices)+1)*gather.itemsize)
    if gather.src is not None:
      indices = gather_indices(gather)
      if indices: _assert_scratch_extent(decoded, gather.src, (max(indices)+1)*gather.itemsize)
  for op in _ew_ops(decoded):
    source_width = 4 if op.mode in (RKEWMode.INT32,RKEWMode.INT32_TO_HALF,RKEWMode.FLOAT_TO_HALF) else 2
    destination_width = 4 if op.mode in (RKEWMode.INT32,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT32,
                                         RKEWMode.HALF_TO_FLOAT) else 2
    _assert_scratch_extent(decoded, op.lhs, op.lhs.addend+op.count*source_width)
    _assert_scratch_extent(decoded, op.rhs, op.rhs.addend+op.count*source_width)
    _assert_scratch_extent(decoded, op.dst, op.dst.addend+op.count*destination_width)
  return decoded

def _execute_fp16_reduction_tail(image:RKImage, values:np.ndarray) -> np.ndarray:
  args = [bytearray(max(64,_ew_ops(image)[-1].count)*2)]
  scratch = [bytearray(size) for size in image.scratch]
  def raw_buffer(kind:RKBufferKind,index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  def view(arg:RKArg,count:int|None=None) -> np.ndarray:
    return np.frombuffer(raw_buffer(arg.kind,arg.index),dtype="<f2",count=-1 if count is None else count,offset=arg.addend)
  def execute(op:RKEWOp) -> None:
    lhs,rhs=view(op.lhs,op.count).copy(),view(op.rhs,op.count).copy()
    value = (lhs+rhs if op.ew_cfg == _EW_CFG[Ops.ADD] else lhs-rhs if op.ew_cfg == _EW_CFG[Ops.SUB] else
             lhs*rhs if op.ew_cfg == _EW_CFG[Ops.MUL] else np.maximum(lhs, rhs) if op.ew_cfg == _EW_CFG[Ops.MAX] else
             lhs/rhs if op.ew_cfg == _EW_CFG[Ops.FDIV] else np.floor(lhs) if op.ew_cfg == _EW_CFG_FLOOR else
             np.minimum(lhs, rhs) if op.ew_cfg == _EW_CFG_MIN else np.abs(lhs) if op.ew_cfg == _EW_CFG_ABS else None)
    assert value is not None, hex(op.ew_cfg)
    view(op.dst,op.count)[:]=value.astype(np.float16)
  spread=next((g for g in reversed(_static_gathers(image)) if g.dst_stride in (8,32)),
              next(g for g in reversed(_static_gathers(image)) if g.src is not None and g.src.kind is RKBufferKind.SCRATCH))
  split=next(i for i,op in enumerate(image.program) if op is spread)
  linear:dict[int,np.ndarray]={}
  for op in image.program[:split]:
    if isinstance(op,RKGather) and op.values: _apply_test_gather(op,raw_buffer,linear)
  assert spread.src is not None
  view(spread.src)[:len(values)]=values
  for op in image.program[split:]:
    if isinstance(op,RKGather): _apply_test_gather(op,raw_buffer,linear)
    elif isinstance(op,RKEWOp): execute(op)
    else: raise AssertionError("CMAC is outside the reduction-tail test executor")
  return np.frombuffer(args[0],dtype="<f2")

def _execute_scalar_reduction_image(image:RKImage, values:np.ndarray) -> float:
  """Execute the native scalar FP16 reduction subset without opening a device."""
  args = [bytearray(4), bytearray(np.asarray(values, dtype="<f2").tobytes())]
  scratch = [bytearray(size) for size in image.scratch]
  def raw_buffer(kind:RKBufferKind,index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  def storage(arg:RKArg) -> bytearray: return raw_buffer(arg.kind,arg.index)
  def read(arg:RKArg, dtype:np.dtype, count:int) -> np.ndarray:
    return np.frombuffer(storage(arg), dtype=dtype, count=count, offset=arg.addend).copy()
  def write(arg:RKArg, dtype:np.dtype, value:np.ndarray) -> None:
    np.frombuffer(storage(arg), dtype=dtype, count=len(value), offset=arg.addend)[:] = value.astype(dtype)
  linear:dict[int,np.ndarray]={}
  for op in image.program:
    if isinstance(op,RKGather):
      _apply_test_gather(op,raw_buffer,linear)
      continue
    assert isinstance(op,RKEWOp)
    input_dtype = np.dtype("<f4") if op.mode==RKEWMode.FLOAT_TO_HALF else np.dtype("<f2")
    output_dtype = np.dtype("<f4") if op.mode==RKEWMode.HALF_TO_FLOAT else np.dtype("<f2")
    lhs, rhs = read(op.lhs, input_dtype, op.count), read(op.rhs, input_dtype, op.count)
    if output_dtype == np.dtype("<f4"): lhs, rhs = lhs.astype(np.float32), rhs.astype(np.float32)
    cfg=op.ew_cfg
    assert cfg in (_EW_CFG[Ops.ADD],_EW_CFG[Ops.MAX])
    write(op.dst, output_dtype, lhs+rhs if cfg == _EW_CFG[Ops.ADD] else np.maximum(lhs,rhs))
  return float(np.frombuffer(args[0], dtype="<f4")[0])

def test_cmac_codec_and_body_match_the_proven_gemm_contract():
  cmac = RKCMAC(RKArg(RKBufferKind.SCRATCH, 2), RKArg(RKBufferKind.SCRATCH, 0),
    RKArg(RKBufferKind.SCRATCH, 1), 3, 4, 5)
  image = RKImage((192,2048,384),(cmac,))
  assert decode_image(encode_image(image)) == image
  for invalid in (image._replace(program=(cmac._replace(lhs=RKArg(RKBufferKind.ARG,0)),)),
                  image._replace(scratch=(191, *image.scratch[1:]))):
    try: encode_image(invalid)
    except ValueError: pass
    else: raise AssertionError("invalid CMAC image was encoded")
  try: encode_image(image._replace(program=(cmac._replace(m=2,k=385),)))
  except ValueError: pass
  else: raise AssertionError("multi-row CMAC exceeded the donor CBUF contract")
  try: encode_image(image._replace(program=(cmac._replace(m=1,k=417),)))
  except ValueError: pass
  else: raise AssertionError("CMAC exceeded the thirteen encoded K blocks")
  stage = emit_cmac_stage(cmac,lambda _arg:0)
  assert len(stage) == 45
  body = emit_cmac_stage(cmac,lambda arg:(0x100000,0x200000,0x300000)[arg.index]+arg.addend)
  assert hashlib.sha256(struct.pack("<45Q", *body)).hexdigest() == "d754ae668b210999c7d568131c0387e46be9c934ad3812b51fb956c789e3db22"
  relu_image = image._replace(program=(cmac._replace(relu=True),))
  assert decode_image(encode_image(relu_image)) == relu_image
  relu_stage = emit_cmac_stage(_cmac(relu_image),lambda _arg:0)
  changed = [(old,new) for old,new in zip(stage,relu_stage) if old != new]
  assert len(relu_stage) == 45 and len(changed) == 1
  assert changed[0][0]&0xffff == changed[0][1]&0xffff == rockchip_renderer.rk.REG_DPU_BS_CFG
  assert ((changed[0][0]>>16)&0xffffffff,(changed[0][1]>>16)&0xffffffff) == (0x53,0x12)
  mixed = image._replace(program=(cmac,RKGather(cmac.dst,RKArg(RKBufferKind.SCRATCH,0),1),
    RKEWOp(RKArg(RKBufferKind.ARG,0),cmac.dst,cmac.dst,1,_EW_CFG[Ops.ADD])))
  assert decode_image(encode_image(mixed)) == mixed

def test_image_codec_rejects_malformed_and_trailing_payloads():
  blob = encode_image(RKImage((64,)))
  scratch = RKArg(RKBufferKind.SCRATCH,0)
  with pytest.raises(ValueError,match="invalid RKEWOp flags"):
    encode_image(RKImage((64,),(RKEWOp(scratch,scratch,scratch,5,_EW_CFG[Ops.MAX],mode=RKEWMode.HALF_TO_INT32),)))
  payload = rockchip_renderer.zlib.decompress(blob[6:])
  malformed = (b"", blob[:4]+b"\0\0"+blob[6:], blob[:-1], blob+b"\0",
    blob[:6]+rockchip_renderer.zlib.compress(payload+b"\0", 1),
    blob[:6]+rockchip_renderer.zlib.compress(rockchip_renderer.pickle.dumps((),5),1))
  for candidate in malformed:
    with pytest.raises(ValueError, match="invalid RKImage"): decode_image(candidate)

def test_large_image_integer_validation_matches_scalar_contract():
  offsets=tuple(range(2048))
  gather=RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.ARG,0),len(offsets),offsets=offsets)
  image=RKImage(program=(gather,))
  assert decode_image(encode_image(image))==image
  for invalid in (offsets[:-1]+(1<<31,),offsets[:-1]+(1.0,),offsets[:-1]+("1",)):
    with pytest.raises(ValueError,match="invalid RKGather"): encode_image(image._replace(program=(gather._replace(offsets=invalid),)))

def _int32_division_samples() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(0x3588)
  lhs = rng.integers(-(1<<31), 1<<31, 100, dtype=np.int64).astype(np.int32)
  rhs = rng.integers(-(1<<31), 1<<31, 100, dtype=np.int64).astype(np.int32)
  lhs[:12] = (-(1<<31), -(1<<31), (1<<31)-1, -7, -7, 7, 0, 7, -1, 1, -(1<<31), (1<<31)-1)
  rhs[:12] = (-1, 1, -1, 3, -3, -3, 3, 0, 0, 0, -(1<<31), -(1<<31))
  return lhs, rhs

def _wrap_int32(value:int) -> int: return (value+(1<<31)) % (1<<32) - (1<<31)

def _trunc_divmod_int32(lhs:int, rhs:int) -> tuple[int, int]:
  quotient = 0 if rhs == 0 else abs(lhs)//abs(rhs) * (-1 if (lhs < 0) != (rhs < 0) else 1)
  return _wrap_int32(quotient), _wrap_int32(lhs-quotient*rhs)


def test_binary_tree_iteration_is_ordered_and_bounded_on_shared_dags():
  leaves = [UOp.const(x, dtypes.int) for x in range(8)]
  root = leaves[0]
  for leaf in leaves[1:]: root = root+leaf
  assert list(rockchip_renderer._iter_binary(root, Ops.ADD)) == leaves
  shared = root
  for _ in range(30): shared = shared+shared
  assert len(list(itertools.islice(rockchip_renderer._iter_binary(shared, Ops.ADD), 256))) == 256


def test_static_vector_values_match_scalar_typed_evaluation():
  outer, inner = UOp.range(5, 100), UOp.range(4, 101)
  out_index = outer.cast(dtypes.int)*4+inner.cast(dtypes.int)
  truncated_negative = UOp(Ops.TRUNC, dtypes.half, src=(UOp.const(-0.5, dtypes.half),))
  max_nan_rhs = UOp(Ops.MAX, dtypes.half, src=(UOp.const(1.0, dtypes.half), UOp.const(math.nan, dtypes.half)))
  expressions = (outer*7-inner*3+5, (outer < 3).where(inner+11, outer-7),
                 ((outer*7-inner*3+5).cast(dtypes.half)*0.5+1.25).cast(dtypes.half), (outer < 3) & (inner != 2),
                 truncated_negative, max_nan_rhs)
  for expr,encode in zip(expressions, (int, int, rockchip_renderer._fp16_bits, lambda x:int(bool(x)),
                                       rockchip_renderer._fp16_bits, rockchip_renderer._fp16_bits)):
    expected = [None]*20
    for values in itertools.product(range(5),range(4)):
      env=dict(zip((outer,inner),values))
      cache = {}
      expected[int(rockchip_renderer._eval_static(out_index,env,cache))] = encode(rockchip_renderer._eval_static(expr,env,cache).item())
    assert rockchip_renderer._static_values(out_index, expr, 20, encode) == tuple(expected)


def test_index_range_analysis_keeps_first_seen_order_and_hides_range_dependencies():
  first,second = UOp.range(3,20),UOp.range(4,21)
  root = UOp(Ops.ADD,dtypes.int,src=(UOp(Ops.ADD,dtypes.int,src=(second,first)),UOp(Ops.MUL,dtypes.int,src=(second,first))))
  assert rockchip_renderer._static_ranges(root) == (second,first)
  dependent = UOp(Ops.RANGE,dtypes.int,src=(UOp.const(2,dtypes.int),first),arg=(22,AxisType.REDUCE))
  assert rockchip_renderer._static_ranges(dependent) == (dependent,)


def test_exact_integer_range_analysis_covers_supported_carriers_only():
  lane = UOp.range(4,23)
  expressions = (lane,(lane-2)*3,UOp(Ops.WHERE,dtypes.int,src=(lane<2,lane,UOp.const(-5,dtypes.int))),
                 UOp(Ops.XOR,dtypes.int,src=(lane,UOp.const(-1,dtypes.int))),UOp.const(True,dtypes.bool).cast(dtypes.int),lane<<1)
  assert tuple(map(rockchip_renderer._exact_int_range,expressions)) == ((0,3),(-6,3),(-5,3),(-4,-1),(1,1),None)


def test_uop_is_the_typed_physical_abi():
  program = _program(dtypes.half, lambda _:UOp.const(0.0,dtypes.half), 1)
  output=rockchip_renderer._outs(program)[0]
  assert output is not None
  value = rockchip_renderer.RKContext(output)._carrier(RKArg(RKBufferKind.ARG,0),dtypes.half)
  assert value.op is Ops.NOOP and value.dtype is dtypes.half and value.arg == RKArg(RKBufferKind.ARG,0)

def _runtime_memory(size:int, dma:int):
  storage=ctypes.create_string_buffer(size)
  base=SimpleNamespace(va_addr=ctypes.addressof(storage))
  memory=SimpleNamespace(va_addr=base.va_addr,size=size,meta=SimpleNamespace(dma_addr=dma,obj_addr=dma),base=base,storage=storage)
  memory.offset=lambda offset,size:SimpleNamespace(va_addr=memory.va_addr+offset,size=size,meta=memory.meta,base=memory.base)
  return memory


def test_submit_timeout_poison_prevents_driver_retry(monkeypatch):
  class FakeDevice:
    fd_ctl, submit_count, task_count, timeout_retries, _poisoned = object(), 0, 0, 0, False
    def _check_healthy(self):
      if self._poisoned: raise RuntimeError("poisoned")
    def _sync_buffer(self, _buffer, _flags): self._check_healthy()
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = FakeDevice()
  buffer = SimpleNamespace(meta=SimpleNamespace(obj_addr=1))
  calls = 0
  def submit(_fd, **_kwargs):
    nonlocal calls
    calls += 1
    raise TimeoutError
  monkeypatch.setattr(rockchip_runtime.rk, "DRM_IOCTL_RKNPU_SUBMIT", submit)
  with pytest.raises(RuntimeError,match="platform NPU reset or power cycle required"): program._submit(buffer, buffer, 1)
  with pytest.raises(RuntimeError,match="poisoned"): program._submit(buffer, buffer, 1)
  assert calls == program.dev.timeout_retries == 1 and program.dev._poisoned
  assert program.dev.submit_count == program.dev.task_count == 0

def test_device_workspace_reuses_scratch_replaces_submit_pair_and_finalizes():
  device=object.__new__(rockchip_runtime.RockchipDevice)
  device._buffers,allocated,freed,timeline={},[],[],[]
  def alloc(size,flags=0):
    allocated.append(SimpleNamespace(size=size,flags=flags))
    timeline.append(("alloc",allocated[-1]))
    return allocated[-1]
  def free(buf):
    freed.append(buf)
    timeline.append(("free",buf))
  device._gpu_alloc,device._gpu_free=alloc,free
  first=device._ensure_buffer("cmd",4,8)
  assert first.size == 8 and device._ensure_buffer("cmd",7,8) is first and freed == []
  second=device._ensure_buffer("cmd",9,8,3)
  scratch=device._ensure_buffer("scratch",2,2)
  assert second is not first and (second.size,second.flags) == (9,3) and freed == [first]
  before=len(timeline)
  cmd,task=device._replace_submit_buffers(4,4)
  assert timeline[before:] == [("alloc",cmd),("alloc",task),("free",second)]
  assert (cmd.size,cmd.flags,task.size,task.flags) == \
    (rockchip_runtime._CMD_BUF_MIN,0,rockchip_runtime._TASK_BUF_MIN,rockchip_runtime.rk.RKNPU_MEM_KERNEL_MAPPING)
  device.finalize()
  assert device._buffers == {} and freed == [first,second,cmd,scratch,task]
  device.finalize()
  assert freed == [first,second,cmd,scratch,task]


def test_runtime_bounds_risky_and_oversize_pc_chains(monkeypatch):
  arg = RKArg(RKBufferKind.ARG,0)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = SimpleNamespace(reset_npu=lambda:None)
  monkeypatch.setattr(rockchip_runtime,"emit_ew_stage",lambda *_args,**_kwargs:(1,))
  half = RKEWOp(arg,arg,arg,1,_EW_CFG[Ops.ADD])
  submitted = []
  program._submit_bodies=lambda bodies,*_args,**_kwargs:submitted.append(len(tuple(bodies)))
  program._run_ew_ops(lambda _arg:0,(half,)*(rockchip_runtime._MAX_EW_GROUP_OPS+1))
  assert submitted == [rockchip_runtime._MAX_EW_GROUP_OPS+1]
  submitted.clear()
  program._run_ew_ops(lambda _arg:0,(half,)*(rockchip_runtime._MAX_PC_TASKS+1))
  assert submitted == [rockchip_runtime._MAX_PC_TASKS,1]
  submitted.clear()
  int16 = half._replace(mode=RKEWMode.INT16)
  program._run_ew_ops(lambda _arg:0,(half,)*rockchip_runtime._MAX_EW_GROUP_OPS+(int16,))
  assert submitted == [rockchip_runtime._MAX_EW_GROUP_OPS,1]
  submitted.clear()
  conversion = half._replace(count=8*(rockchip_runtime._MAX_EW_GROUP_OPS+1),mode=RKEWMode.INT16_TO_INT32)
  program._run_ew_ops(lambda _arg:0,(half,int16,conversion))
  assert submitted == [1,rockchip_runtime._MAX_EW_GROUP_OPS,2]
  submitted.clear()
  program._run_ew_ops(lambda _arg:0,(int16,)*77+(conversion._replace(count=25),))
  assert submitted == [81]
  submitted.clear()
  program._ew_modes = {half.mode,int16.mode}
  program._run_ew_ops(lambda _arg:0,(half,)*(rockchip_runtime._MAX_EW_GROUP_OPS+1))
  assert submitted == [rockchip_runtime._MAX_EW_GROUP_OPS,1]
  with pytest.raises(ValueError,match="invalid NPU task count"): program._submit(None,None,rockchip_runtime._MAX_PC_TASKS+1)  # type: ignore[arg-type]


def test_numeric_output_program_resets_before_its_first_dpu_stage():
  class FakeDevice:
    _native_int16, resets = False, 0
    def __init__(self): self._lock=threading.Lock()
    def _ensure_buffer(self, *_args): raise AssertionError("scratchless program allocated a workspace")
    def _sync_buffers(self, _buffers, _flags): pass
    def reset_npu(self): self.resets += 1
  op = RKEWOp(RKArg(RKBufferKind.ARG,0),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,0),1,
              _EW_CFG[Ops.ADD],mode=RKEWMode.HALF_TO_FLOAT)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev,program.image,program._scratch_offsets=FakeDevice(),RKImage(program=(op,)),(0,)
  program._run_ew_ops=lambda *_args,**_kwargs:None
  program()
  assert program.dev.resets == 1


def test_program_records_only_the_terminal_native_int16_mode():
  class FakeDevice:
    _native_int16 = False
    def __init__(self): self._lock=threading.Lock()
    def _ensure_buffer(self, *_args): raise AssertionError("scratchless program allocated a workspace")
    def _sync_buffers(self, _buffers, _flags): pass
    def reset_npu(self): self._native_int16=False
  arg=RKArg(RKBufferKind.ARG,0)
  for modes,expected in (((RKEWMode.INT16,RKEWMode.HALF),False),((RKEWMode.HALF,RKEWMode.INT16),True)):
    program=object.__new__(rockchip_runtime.RockchipProgram)
    program.dev=FakeDevice()
    program.image=RKImage(program=tuple(RKEWOp(arg,arg,arg,1,_EW_CFG[Ops.ADD],mode=mode) for mode in modes))
    program._scratch_offsets=(0,)
    program._run_ew_ops=lambda *_args,**_kwargs:None
    program(arg)
    assert program.dev._native_int16 is expected


def test_int32_to_half_gather_chain_rearms_only_into_bounded_fp16():
  class FakeDevice:
    _native_int16 = False
    def __init__(self): self._lock,self.arena=threading.Lock(),_runtime_memory(16384,0x100000)
    def _ensure_buffer(self, *_args): return self.arena
    def _sync_buffers(self, _buffers, _flags): pass
    def reset_npu(self): raise AssertionError("rearmed conversion reset the NPU")
  scratch=tuple(RKArg(RKBufferKind.SCRATCH,index) for index in range(4))
  out=RKArg(RKBufferKind.ARG,0)
  def convert(arg): return RKEWOp(arg,arg,arg,4,_EW_CFG[Ops.MAX],mode=RKEWMode.INT32_TO_HALF)
  def gather(src,dst): return RKGather(src,dst,4,axes=((1,4,1),))
  for terminal,expected in (((RKEWMode.BOUNDED,),(True,True,False)),
                            ((RKEWMode.BOUNDED,RKEWMode.HALF),(True,True,False)),
                            ((RKEWMode.HALF,),(True,False,False))):
    program=object.__new__(rockchip_runtime.RockchipProgram)
    program.dev=FakeDevice()
    calls=[]
    program.image=RKImage((64,)*4,(convert(scratch[0]),gather(scratch[0],scratch[1]),convert(scratch[2]),
      gather(scratch[2],scratch[3]),*(RKEWOp(out,scratch[1],scratch[3],4,_EW_CFG[Ops.FDIV],mode=mode) for mode in terminal)))
    program._scratch_offsets=(0,4096,8192,12288,16384)
    program._run_ew_ops=lambda _address,ops,rearm=False:calls.append(rearm)
    program(_runtime_memory(8,0x200000))
    assert tuple(calls) == expected


def test_cmac_runtime_keeps_the_45_qword_body_and_four_qword_tail_separate():
  class FakeDevice:
    resets, _poisoned = 0, False
    def reset_npu(self): self.resets += 1
  def memory(size, dma):
    storage = ctypes.create_string_buffer(size)
    base = SimpleNamespace(va_addr=ctypes.addressof(storage))
    return SimpleNamespace(va_addr=base.va_addr,size=size,meta=SimpleNamespace(dma_addr=dma,obj_addr=dma),base=base,storage=storage)
  cmac = RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),3,4,5)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev,program.image=FakeDevice(),RKImage(program=(cmac,))
  cmd, task, submits = memory(8192,0x400000), memory(4096,0x500000), []
  program.dev._replace_submit_buffers = lambda *_args,**_kwargs: (cmd,task)
  program._submit = lambda *args,**kwargs: submits.append((args,kwargs))
  addresses = (0x100000,0x200000,0x300000)
  program._submit_bodies((emit_cmac_stage(cmac,lambda arg:addresses[arg.index]+arg.addend),),True,True)
  commands = tuple((ctypes.c_uint64*49).from_address(cmd.va_addr))
  assert commands[:45] == emit_cmac_stage(cmac,lambda arg:addresses[arg.index]+arg.addend)
  assert commands[45:] == (rockchip_runtime._pc(0x0001,0),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_PC_REG,rockchip_runtime.rk.REG_PC_REGISTER_AMOUNTS),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_VERSION,0),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_PC,rockchip_runtime.rk.REG_PC_OPERATION_ENABLE,0xd))
  desc = rockchip_runtime.rk.struct_rknpu_task.from_address(task.va_addr)
  assert (desc.op_idx,desc.enable_mask,desc.int_mask,desc.int_clear,desc.regcfg_amount) == (0,0xd,0x300,0x1ffff,45)
  assert program.dev.resets == 2 and len(submits) == 1 and submits[0][1] == {"standalone":True}
  body = (1,2,3)
  program._submit_bodies((body,),True)
  assert tuple((ctypes.c_uint64*4).from_address(cmd.va_addr)) == body+(rockchip_runtime._pc(
    rockchip_runtime.rk.TARGET_PC,rockchip_runtime.rk.REG_PC_OPERATION_ENABLE,0x18),)
  desc = rockchip_runtime.rk.struct_rknpu_task.from_address(task.va_addr)
  assert (desc.op_idx,desc.enable_mask,desc.int_mask,desc.int_clear,desc.regcfg_amount) == (4,0x18,0x300,0x1ffff,4)
  assert program.dev.resets == 4 and len(submits) == 2 and submits[1][1] == {"standalone":True}


def test_mixed_cmac_runtime_runs_fixed_stage_before_ew_epilogue():
  events = []
  class FakeDevice:
    _native_int16 = False
    def __init__(self): self._lock,self.arena=threading.Lock(),_runtime_memory(16384,0x100000)
    def _ensure_buffer(self, *_args): return self.arena
    def _sync_buffers(self, _buffers, _flags): pass
    def reset_npu(self): pass
  cmac=RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),1,1,4,True)
  ew=RKEWOp(RKArg(RKBufferKind.ARG,0),cmac.dst,cmac.dst,1,_EW_CFG[Ops.ADD])
  program=object.__new__(rockchip_runtime.RockchipProgram)
  program.dev,program.image=FakeDevice(),RKImage((2,)*3,program=(cmac,ew))
  program._scratch_offsets=(0,4096,8192,12288)
  program._submit_bodies=lambda *_args,**_kwargs:events.append("cmac")
  program._run_ew_ops=lambda *_args,**_kwargs:events.append("ew")
  program()
  assert events == ["cmac","ew"]


def test_runtime_tiling_modes_keep_exact_stage_bodies():
  class FakeDevice:
    def __init__(self): self.resets = 0
    def reset_npu(self): self.resets += 1
    def _forget_program(self, _program): pass
    def _gpu_free(self, _buffer): pass
  scratch = tuple(RKArg(RKBufferKind.SCRATCH, index, index*32) for index in range(3))
  external = tuple(RKArg(RKBufferKind.ARG, index, index*32) for index in range(3))
  add, large = _EW_CFG[Ops.ADD], _MAX_EW_ELEMS_FP16+3
  cases = (
    ("6fef992e01c99f01880ddd0363b4f4472c05097528700dc4f4c1748c695e6056", ((31,31),),
      RKEWOp(scratch[0],scratch[1],scratch[2],large,add,mode=RKEWMode.INT16)),
    ("d8269792e064feb474483980a269a3df704b71fde4dd366b58afed686da58db2", ((32,32),),
      RKEWOp(scratch[0],scratch[1],scratch[2],11,add,mode=RKEWMode.INT16_TO_INT32)),
    ("f7d8fb289a5ee91ebb11e059ebab0ee16d46ad93094d6937e56a4bb83060c270", ((31,31),),
      RKEWOp(scratch[0],external[1],scratch[2],large,add,mode=RKEWMode.INT16)),
    ("a8dffe2406a897e0d7f225ac7b4fba7879ca8d476dc410ba15b552d9e733145f", ((31,31),),
      RKEWOp(scratch[0],scratch[1],scratch[2],_MAX_EW_ELEMS_FP16//2+3,add,mode=RKEWMode.INT32)),
    ("977a482cf35150b50d77165dff1011411d14b1fe6b70a3bd6846252176c07d40", ((31,31),),
      RKEWOp(external[0],external[1],external[2],large,add)),
    ("bb30c192a115976317e2ed0341666192a4d3bcf758e0910f5f1ec09a1d44b215", ((31,),),
      RKEWOp(external[0],external[1],external[2],17,add,mode=RKEWMode.STATEFUL)),
    ("2f075321eab33020cf9ae176f91235110c26e2e01e3d4790070c5f6e85f00d88", ((31,31),),
      RKEWOp(external[0],external[1],external[2],large,add,mode=RKEWMode.HALF_TO_INT16)),)
  def address(arg:RKArg) -> int: return 0x10000000+int(arg.kind)*0x01000000+arg.index*0x00100000+arg.addend
  for expected_hash,expected_shape,op in cases:
    program = object.__new__(rockchip_runtime.RockchipProgram)
    program.dev,program.image=FakeDevice(),SimpleNamespace(ew_ops=(op,))
    submissions = []
    program._submit_bodies = lambda bodies,*_args,**_kwargs:submissions.append(tuple(bodies))
    program._run_ew_ops(address,(op,))
    assert tuple(tuple(map(len, submission)) for submission in submissions) == expected_shape and program.dev.resets == 0
    packed = b"".join(struct.pack(f"<{len(body)}Q", *body) for submission in submissions for body in submission)
    assert hashlib.sha256(packed).hexdigest() == expected_hash


def test_runtime_chain_flush_preserves_mixed_boundaries():
  scratch = tuple(RKArg(RKBufferKind.SCRATCH,index,index*64) for index in range(3))
  external = tuple(RKArg(RKBufferKind.ARG,index,index*64) for index in range(3))
  add, large = _EW_CFG[Ops.ADD], _MAX_EW_ELEMS_FP16+3
  def op(*,count=17,barrier=False,mode=RKEWMode.HALF,scratch_args=False):
    args = scratch if scratch_args else external
    return RKEWOp(args[0],args[1],args[2],count,add,submit_barrier=barrier,mode=mode)
  cases = (
    ((op(),op(barrier=True),op()), (("submit",(31,)),("submit",(31,18)))),
    ((op(mode=RKEWMode.INT16,scratch_args=True),op(mode=RKEWMode.INT32,scratch_args=True),
      op(mode=RKEWMode.INT16_TO_INT32,scratch_args=True),op()),
     (("submit",(31,)),("submit",(31,)),("submit",(32,32,32,18)))),
    ((op(mode=RKEWMode.INT16,scratch_args=True),op()),
     (("submit",(31,)),("submit",(31,)))),
    ((op(),op(count=4,mode=RKEWMode.INT32_TO_HALF,scratch_args=True),op()),
     (("submit",(31,)),("submit",(31,)),("reset",),("submit",(31,)))),
    ((op(),op(mode=RKEWMode.COMPARE),op()),
     (("submit",(31,)),("standalone",37),("submit",(31,)))),
    ((op(),*(op(count=1,mode=RKEWMode.HALF_TO_FLOAT) for _ in range(17))),
     (("submit",(31,)),("submit",(32,)*16),("reset",),("submit",(32,)),("reset",))),
    ((op(count=large),op(count=large)), (("submit",(31,18)),("submit",(31,18)))),
    ((op(),op(count=large,barrier=True,mode=RKEWMode.STATEFUL),op(count=large),op(barrier=True)),
     (("submit",(31,)),("submit",(31,31)),("submit",(31,31)),("submit",(31,)))),
    ((op(mode=RKEWMode.STATEFUL),*(op() for _ in range(rockchip_runtime._MAX_EW_GROUP_OPS))),
     (("submit",(31,)+(18,)*(rockchip_runtime._MAX_EW_GROUP_OPS-1)),("submit",(31,)))),
  )
  def address(arg:RKArg) -> int: return 0x10000000+int(arg.kind)*0x100000+arg.index*0x10000+arg.addend
  for ops,expected in cases:
    events = []
    class FakeDevice:
      def reset_npu(self): events.append(("reset",))
    program = object.__new__(rockchip_runtime.RockchipProgram)
    program.dev,program.image=FakeDevice(),SimpleNamespace(ew_ops=ops)
    program._submit_bodies = lambda bodies,standalone=False,*_args,**_kwargs:events.append(
      ("standalone",len(tuple(bodies)[0])) if standalone else ("submit",tuple(map(len,bodies))))
    program._run_ew_ops(address,ops)
    assert tuple(events) == expected
  events=[]
  conversion=op(count=4,mode=RKEWMode.INT32_TO_HALF,scratch_args=True)
  program._run_ew_ops(address,(conversion,),True)
  assert tuple(events) == (("submit",(31,)),)
  with pytest.raises(ValueError,match="invalid RKEWOp sequence"):
    encode_image(RKImage(program=(op(mode=RKEWMode.HALF_TO_FLOAT),op(barrier=True))))
  with pytest.raises(ValueError,match="invalid RKEWOp sequence"):
    encode_image(RKImage(program=(op(mode=RKEWMode.INT16_TO_INT32),op(barrier=True))))


def test_native_ew_configs_keep_their_exact_register_values():
  assert tuple(rockchip_renderer._EW_CFG[op] for op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.FDIV)) == (
    0x108202c0,0x108402c0,0x108003c4,0x108002c0,0x108303c0)
  assert tuple(getattr(rockchip_renderer,name) for name in
    ("_EW_CFG_RELU6","_EW_CFG_MIN","_EW_CFG_ABS","_EW_CFG_NEG","_EW_CFG_FLOOR","_EW_CFG_CEIL")) == (
    0x108004c0,0x108102c0,0x108502c0,0x108602c0,0x108702c0,0x108802c0)
  arg=RKArg(RKBufferKind.ARG,0)
  relu6=RKEWOp(arg,arg,arg,4,rockchip_renderer._EW_CFG_RELU6,mode=RKEWMode.STATEFUL)
  assert rockchip_renderer._cmd(rockchip_renderer._DPU,rockchip_renderer.rk.REG_DPU_EW_RELUX_CMP_VALUE,
    rockchip_renderer._EW_RELUX_CMP_RELU6) in rockchip_renderer.emit_ew_stage(relu6,lambda _arg:0x10000000)


def test_cmac_packs_fp16_exact_per_term_weights_and_rejects_invalid_weights():
  def lower(weights:tuple[float, ...], nested:tuple[float, ...]=()) -> RKImage|None:
    out, source = UOp.param(0,dtypes.half,(1,)), UOp.param(1,dtypes.half,(len(weights),))
    terms = [source.index(i).load().cast(dtypes.float)*UOp.const(weight,dtypes.float) for i,weight in enumerate(weights)]
    for factor in nested: terms[0] = terms[0]*UOp.const(factor,dtypes.float)
    value = terms[0]
    for term in terms[1:]: value = value+term
    return _lower_uop_program(list(out.index(0).store(value.cast(dtypes.half)).sink().toposort()))
  weights = (0.5,-0.25,2.0,0.125)
  image = lower(weights)
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,4)
  assert not _constant_bytes(image) and len(image.scratch) == 3
  assert _cmac_weights(image) == tuple(rockchip_renderer._fp16_bits(weight) for weight in weights)
  for invalid in (0.1,float("inf")):
    fallback = lower((invalid,*weights[1:]))
    assert fallback is not None and _cmac(fallback) is None and _ew_ops(fallback)
  nested = lower(weights, (2.0,))
  assert nested is not None and _cmac(nested) is not None and _cmac_weights(nested) == (
    rockchip_renderer._fp16_bits(1.0), *tuple(rockchip_renderer._fp16_bits(weight) for weight in weights[1:]))
  for unsafe in (lower((10.0,*weights[1:]), (0.1,)), lower(weights, (2.0,1.0))):
    assert unsafe is not None and _cmac(unsafe) is None and _ew_ops(unsafe)
  half_out, half_source = UOp.param(0,dtypes.half,(1,)), UOp.param(1,dtypes.half,(4,))
  half_terms = [half_source.index(i).load() for i in range(4)]
  half_terms[0] = half_terms[0]*UOp.const(-53248.0,dtypes.half)*UOp.const(-0.03955078125,dtypes.half)
  half_value = functools.reduce(lambda total,term:total+term,half_terms[1:],half_terms[0])
  half_nested = _lower_uop_program(list(half_out.index(0).store(half_value).sink().toposort()))
  assert half_nested is not None and _cmac(half_nested) is None and _ew_ops(half_nested)


def test_generic_fp16_uops_lower_in_dependency_order():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:lhs.index(i).load() + rhs.index(i).load() * 2.0))
  assert image is not None
  assert len(_ew_ops(image)) == 2
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG and _ew_ops(image)[-1].dst.index == 0


def test_physical_half_recipe_materializes_strong_float_constant_at_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    UOp(Ops.ADD, dtypes.half, src=(source.index(i).load(), UOp.const(0.25, dtypes.float)))))
  assert image is not None and struct.pack("<e", 0.25) in _constant_bytes(image)


def test_fp32_load_materializes_through_canonical_fp16_physical_abi():
  source = UOp.param(1, dtypes.float, (9,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:source.index(i).load().cast(dtypes.half), count=9))
  assert image is not None and len(_initial_gathers(image)) == 2
  assert any(gather.itemsize == 4 and gather.count == 9 and not gather.values for gather in _initial_gathers(image))
  converters = [op for op in _ew_ops(image) if op.mode==RKEWMode.FLOAT_TO_HALF]
  assert [op.count for op in converters] == [4, 4, 1]
  assert any(gather.count == 9 and gather.itemsize == 2 for gather in _intermediate_gathers(image))
  assert decode_image(encode_image(image)) == image


def test_fp32_constant_uses_canonical_half_value_before_output_conversion():
  image = _lower_uop_program(_program(dtypes.float, lambda _i:UOp.const(4.0, dtypes.float), count=6))
  assert image is not None and _constant_bytes(image) == struct.pack("<e", 4.0)
  assert [op.count for op in _ew_ops(image)] == [4, 2]
  assert all(op.mode==RKEWMode.HALF_TO_FLOAT for op in _ew_ops(image))


def test_infinite_numerator_fdiv_preserves_dynamic_denominator_sign():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    UOp(Ops.FDIV, dtypes.half, src=(UOp.const(math.inf, dtypes.half), source.index(i).load()))))
  assert image is not None and len(_ew_ops(image)) == 3
  assert all(op.ew_cfg == _EW_CFG[Ops.FDIV] for op in _ew_ops(image)[:2])
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_production_scatter_add_keeps_nonfinite_terms_out_of_kahan():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",120,dtypes.half,num=13020)).reshape(4,5,6)
    index=Tensor(UOp.new_buffer("ROCKCHIP",60,dtypes.int,num=13021)).reshape(3,4,5)
    linear=source.scatter(dim=1,index=index,src=math.inf,reduce="add").schedule_linear()
    to_program_cache.clear()
    image=decode_image(to_program(linear.src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src[-1].arg)
  half_ops=[op for op in _ew_ops(image) if op.mode in (RKEWMode.HALF,RKEWMode.STATEFUL)]
  assert half_ops and all(op.ew_cfg==_EW_CFG[Ops.ADD] for op in half_ops)


def test_generic_where_owns_ternary_arity():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left, right)
  image = _lower_uop_program(_program(dtypes.half, select))
  assert image is not None
  assert any(op.mode==RKEWMode.COMPARE or op.ew_cfg == _EW_CFG[Ops.MAX] for op in _ew_ops(image))
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG and _ew_ops(image)[-1].dst.index == 0


def test_production_abs_and_minimum_keep_generic_typed_images():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    x = Tensor(UOp.new_buffer("ROCKCHIP",24,dtypes.half,num=12001)).reshape(2,3,4)
    y = Tensor(UOp.new_buffer("ROCKCHIP",24,dtypes.half,num=12002)).reshape(2,3,4)
    records = []
    for output in (x.abs(), x.minimum(y)):
      to_program_cache.clear()
      program = to_program(output.schedule_linear().src[0].src[0], RockchipRenderer(Target(device="ROCKCHIP")))
      blob = next(u.arg for u in program.src if u.op is Ops.BINARY)
      image = decode_image(blob)
      records.append((hashlib.sha256(blob).hexdigest(), len(blob), len(_ew_ops(image)), len(_intermediate_gathers(image))))
  assert records == [("765c50748dc3fdc7eab909360bcd22002fc790a70525c4bb18bb26fcde8c2e1b",1195,98,6),
                     ("6b4f9c0fc3228c0d4672c2d1a39e3740806fc4c32fe83268e628729698845bda",258,4,0)]


def test_static_nested_load_default_materializes_as_ordered_partial_gathers():
  fallback, selected = UOp.param(1, dtypes.half, (6,)), UOp.param(2, dtypes.half, (6,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    selected.index(i).load(fallback.index(i).load(), i < UOp.const(3, dtypes.int)), count=6))
  assert image is not None and len(_initial_gathers(image)) == 2
  assert not _initial_gathers(image)[0].partial and _initial_gathers(image)[1].partial
  assert _initial_gathers(image)[0].src.index == 1 and _initial_gathers(image)[1].src.index == 2
  assert _initial_gathers(image)[1].offsets == (0, 1, 2, -1, -1, -1)
  assert decode_image(encode_image(image)) == image


def test_bitcast_and_int16_masks_preserve_raw_fp16_sign_and_payload():
  magnitude, sign = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    ((magnitude.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.max, dtypes.int16)) |
     (sign.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.min, dtypes.int16))).bitcast(dtypes.half)))
  assert image is not None and len(_ew_ops(image))==10
  assert sum(g.src is not None and g.src.kind is RKBufferKind.ARG for g in _static_gathers(image))==4
  assert all(isinstance(op,RKGather) for op in image.program[-2:]) and _output_gathers(image)[0] is image.program[-1]
  assert len(_output_gathers(image)) == 1 and _output_gathers(image)[0].itemsize == 2
  assert decode_image(encode_image(image)) == image


def test_production_fp16_pair_bitcast_fused_transfer_uses_raw_gather():
  values = np.arange(24,dtype=np.float16).reshape(2,3,4)
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    images = []
    for source in (Tensor(values),Tensor(values).permute(1,0,2)):
      schedule = source.bitcast(dtypes.int).schedule_linear()
      ast = next(u.src[0] for u in schedule.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK)
      to_program_cache.clear()
      program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
      images.append(decode_image(next(u.arg for u in program.src if u.op is Ops.BINARY)))
  assert all(len(_initial_gathers(image))==1 and _initial_gathers(image)[0].itemsize==4 and not _ew_ops(image) for image in images)
  assert [_initial_gathers(image)[0].offsets for image in images] == [tuple(range(12)),(0,1,6,7,2,3,8,9,4,5,10,11)]


def test_zero_count_raw_fp16_bitcast_uses_empty_generic_image():
  lhs, rhs = UOp.param(1, dtypes.half, (0,)), UOp.param(2, dtypes.half, (0,))
  def packed(i):
    low = lhs.index(i).load().bitcast(dtypes.ushort).cast(dtypes.uint)
    high = rhs.index(i).load().bitcast(dtypes.ushort).cast(dtypes.uint).alu(Ops.SHL, UOp.const(16, dtypes.int))
    return (low + high).bitcast(dtypes.int)
  image = _lower_uop_program(_program(dtypes.int, packed, count=0))
  assert image is not None and not image.program and len(encode_image(image))==69


def test_generic_bool_where_uses_canonical_int16_ternary():
  lhs, rhs = UOp.param(1, dtypes.int, (4,)), UOp.param(2, dtypes.int, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left != UOp.const(0, dtypes.int), UOp.const(False, dtypes.bool))
  image = _lower_uop_program(_program(dtypes.bool, select))
  assert image is not None and _ew_ops(image)[-1].mode==RKEWMode.INT16
  assert len(_output_gathers(image)) == 1 and _output_gathers(image)[0].itemsize == 1


def test_inverted_fp16_comparison_keeps_ieee_unordered_semantics():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def greater_equal(i):
    less = UOp(Ops.CMPLT, dtypes.bool, src=(lhs.index(i).load(), rhs.index(i).load()))
    return UOp(Ops.CMPNE, dtypes.bool, src=(less, UOp.const(True, dtypes.bool)))
  image = _lower_uop_program(_program(dtypes.bool, greater_equal))
  assert image is not None and len(_ew_ops(image)) > 10
  assert _output_gathers(image) and _output_gathers(image)[-1].itemsize == 1
  assert not any(op.mode==RKEWMode.COMPARE for op in _ew_ops(image))
  assert _ew_ops(image)[-1].ew_cfg == _EW_CFG[Ops.MUL] and _ew_ops(image)[-1].mode==RKEWMode.INT16


def test_half_backed_fp32_inverted_comparison_reuses_exact_raw_path():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def image(as_float:bool) -> RKImage:
    def greater_equal(i:UOp) -> UOp:
      left, right = lhs.index(i).load(), rhs.index(i).load()
      if as_float: left, right = left.cast(dtypes.float), right.cast(dtypes.float)
      less = UOp(Ops.CMPLT, dtypes.bool, src=(left, right))
      return UOp(Ops.CMPNE, dtypes.bool, src=(less, UOp.const(True, dtypes.bool)))
    result = _lower_uop_program(_program(dtypes.bool, greater_equal))
    assert result is not None
    return result
  assert encode_image(image(True)) == encode_image(image(False))


def test_fp16_equality_uses_exact_raw_bytes_without_compare_resets():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() != rhs.index(i).load()))
  assert image is not None and len(_output_gathers(image))==1 and image.program[-1] is _output_gathers(image)[0]
  assert all(isinstance(op,RKGather) for op in image.program[:4])
  assert not any(op.mode==RKEWMode.COMPARE for op in _ew_ops(image)) and all(op.mode==RKEWMode.INT16 for op in _ew_ops(image))


def test_fp16_negated_max_equality_covers_every_bit_pattern():
  count=32768
  lhs,rhs,other=(UOp.param(slot,dtypes.half,(count,)) for slot in range(1,4))
  image=_lower_uop_program(_program(dtypes.bool,
    lambda i:lhs.index(i).load()*UOp.const(-1.0,dtypes.half) != rhs.index(i).load().maximum(other.index(i).load()),count))
  assert image is not None and all(op.mode==RKEWMode.INT16 for op in _ew_ops(image))
  patterns=np.arange(1<<16,dtype="<u2")
  right=np.roll(patterns,12345)
  third=patterns[::-1]
  for start in (0,count):
    left=patterns[start:start+count].view("<f2")
    maximum=np.maximum(right[start:start+count].view("<f2"),third[start:start+count].view("<f2"))
    actual=np.frombuffer(_execute_raw_dynamic_image(image,count,left.tobytes(),right[start:start+count].tobytes(),third[start:start+count].tobytes()),dtype=np.bool_)
    with np.errstate(invalid="ignore"): np.testing.assert_array_equal(actual,left*np.float16(-1.0)!=maximum)


def test_generic_where_selects_infinity_without_mask_multiplication():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(-math.inf, dtypes.half))))
  assert image is not None and not _ew_ops(image) and len(_output_gathers(image)) == 2


def test_max_uses_finite_neutral_for_selected_negative_infinity():
  source = UOp.param(1, dtypes.half, (4,))
  def maximum(i):
    selected = (i < UOp.const(3, dtypes.int)).where(UOp.const(-math.inf, dtypes.half), source.index(i).load())
    return selected.maximum(UOp.const(-2.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, maximum))
  assert image is not None and struct.pack("<e", -65504.0) in _constant_bytes(image)
  overlays=tuple(g for g in _static_gathers(image) if g.src is not None and g.src.kind is RKBufferKind.SCRATCH)
  assert len(overlays)==4 and tuple(_gather_point(image,g) for g in overlays)==(0,0,6,6)


def test_generic_where_predicates_nonfinite_exp2_input():
  source = UOp.param(1, dtypes.half, (4,))
  def power(i):
    exponent = UOp.const(-math.inf, dtypes.half) * source.index(i).load()
    return (source.index(i).load() != UOp.const(0.0, dtypes.half)).where(exponent.exp2(), UOp.const(1.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, power))
  assert image is not None and len(_ew_ops(image)) > 10


def test_generic_where_materializes_nan_only_on_selected_lanes():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    (source.index(i).load() < UOp.const(0.0, dtypes.half)).where(UOp.const(math.nan, dtypes.half), source.index(i).load())))
  assert image is not None and _intermediate_gathers(image) and not any(op.mode==RKEWMode.COMPARE or op.submit_barrier for op in _ew_ops(image))


def test_nested_where_around_math_preserves_raw_uop_selection():
  base, exponent = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def power(i):
    x, y = base.index(i).load(), exponent.index(i).load()
    absolute = (x < UOp.const(0.0, dtypes.half)).where(-x, x)
    magnitude = (absolute.log2() * y).exp2()
    invalid = (y != y.cast(dtypes.int).cast(dtypes.half)).where(UOp.const(math.nan, dtypes.half), magnitude)
    return (x < UOp.const(0.0, dtypes.half)).where(invalid, magnitude)
  image = _lower_uop_program(_program(dtypes.half, power))
  assert image is not None and len(_intermediate_gathers(image)) >= 6
  assert len({_gather_point(image,gather) for gather in _intermediate_gathers(image)}) >= 2
  assert decode_image(encode_image(image)) == image


def test_generic_where_abs_recipe_avoids_infinite_arm_blend():
  source = UOp.param(1, dtypes.half, (4,))
  def absolute(i):
    value = source.index(i).load()
    return (value < UOp.const(0.0, dtypes.half)).where(value * UOp.const(-1.0, dtypes.half), value)
  image = _lower_uop_program(_program(dtypes.half, absolute))
  assert image is not None and len(_ew_ops(image)) == 1 and _ew_ops(image)[-1].ew_cfg == _EW_CFG_ABS
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_threshold_where_uses_bounded_selection_for_dynamic_infinity():
  source = UOp.param(1, dtypes.half, (4,))
  def selected(i):
    value = source.index(i).load()
    return (value < UOp.const(0.0, dtypes.half)).where(value, UOp.const(1.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, selected))
  assert image is not None and any(op.ew_cfg == _EW_CFG[Ops.MAX] for op in _ew_ops(image))


def test_shifted_relu_difference_becomes_bounded_cap():
  source = UOp.param(1, dtypes.half, (4,))
  def bounded(i):
    scaled = source.index(i).load() * UOp.const(1/6, dtypes.half)
    lower, upper, zero = scaled + UOp.const(0.5, dtypes.half), scaled + UOp.const(-0.5, dtypes.half), UOp.const(0.0, dtypes.half)
    return (zero < lower).where(lower, zero) + (zero < upper).where(upper, zero) * UOp.const(-1.0, dtypes.half)
  image = _lower_uop_program(_program(dtypes.half, bounded))
  assert image is not None and len(_ew_ops(image)) < 10
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_where_abs_remains_native_inside_math_recipe():
  source = UOp.param(1, dtypes.half, (4,))
  def logarithm(i):
    value = source.index(i).load()
    absolute = (value < UOp.const(0.0, dtypes.half)).where(value * UOp.const(-1.0, dtypes.half), value)
    return absolute.log2()
  image = _lower_uop_program(_program(dtypes.half, logarithm))
  assert image is not None and any(op.ew_cfg == _EW_CFG_ABS for op in _ew_ops(image))


def test_where_abs_recognizes_negated_reciprocal_arms():
  source = UOp.param(1, dtypes.half, (4,))
  def power_magnitude(i):
    value = source.index(i).load()
    reciprocal = UOp(Ops.FDIV, dtypes.half, src=(UOp.const(1.0, dtypes.half), value))
    negative = UOp(Ops.FDIV, dtypes.half, src=(UOp.const(-1.0, dtypes.half), value))
    absolute = (reciprocal < UOp.const(0.0, dtypes.half)).where(negative, reciprocal)
    return (absolute.log2() * UOp.const(0.3, dtypes.half)).exp2()
  image = _lower_uop_program(_program(dtypes.half, power_magnitude))
  assert image is not None and any(op.ew_cfg == _EW_CFG_ABS for op in _ew_ops(image))


def test_generic_static_index_becomes_gather():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:source.index(3-i).load()))
  assert image is not None and len(_initial_gathers(image)) == 1
  assert _initial_gathers(image)[0].base == 3 and _initial_gathers(image)[0].axes == ((1, 4, -1),)


def test_max_materializes_negative_infinity_fill_as_finite_neutral():
  source = UOp.param(1, dtypes.half, (4,))
  def padded(i):
    value = source.index(i).load(UOp.const(-math.inf, dtypes.half), i < UOp.const(3, dtypes.int))
    return value.maximum(UOp.const(0.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, padded))
  assert image is not None and _initial_gathers(image)[0].fill_bits == 0xfbff


def test_guarded_load_with_infinite_fill_falls_through_dynamic_address_probes():
  source = UOp.param(1, dtypes.half, (2,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    source.index(i).load(UOp.const(math.inf, dtypes.half), i < UOp.const(2, dtypes.int))))
  assert image is not None and any(gather.fill_bits == 0x7c00 for gather in _initial_gathers(image))


def test_static_root_where_uses_exact_gathers_and_finite_padding_neutral():
  source = UOp.param(1, dtypes.half, (3,))
  def selected(i):
    padded = source.index(i).load(UOp.const(-math.inf, dtypes.half), i < UOp.const(3, dtypes.int))
    return (i < UOp.const(4, dtypes.int)).where(padded, UOp.const(0.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, selected))
  assert image is not None and not _ew_ops(image) and len(_output_gathers(image)) == 2
  assert any(gather.fill_bits == 0xfbff for gather in _initial_gathers(image))


def test_static_root_where_preserves_nonzero_constant_route():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(3.5, dtypes.half))))
  assert image is not None and not _ew_ops(image) and len(_output_gathers(image)) == 2
  assert struct.pack("<e", 3.5) in _constant_bytes(image)


def test_generic_bool_store_has_explicit_boundary_conversion():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() < rhs.index(i).load()))
  assert image is not None
  assert _output_gathers(image) and _output_gathers(image)[-1].itemsize == 1
  assert not any(op.mode==RKEWMode.COMPARE for op in _ew_ops(image))


def test_generic_int16_uses_canonical_native_layout():
  source = UOp.param(1, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int16, lambda i:source.index(i).load() + UOp.const(3, dtypes.int16)))
  assert image is not None and len(_ew_ops(image)) == 1
  assert _ew_ops(image)[0].mode==RKEWMode.INT16


def test_generic_int_constant_where_keeps_int16_output_carrier():
  out,source=UOp.param(0,dtypes.int16,(4,)),UOp.param(1,dtypes.int,(4,))
  lane=UOp.range(4,0)
  selected=(source.index(lane).load()!=lane).where(UOp.const(1,dtypes.int),UOp.const(0,dtypes.int))
  image=_lower_uop_program(list(out.index(lane).store(selected).end(lane).sink().toposort()))
  assert image is not None and not any(op.mode==RKEWMode.INT16_TO_INT32 for op in _ew_ops(image))
  actual=np.frombuffer(_execute_raw_dynamic_image(image,8,np.asarray((0,9,2,8),dtype="<i4").tobytes()),dtype="<i2")
  np.testing.assert_array_equal(actual,np.asarray((0,1,0,1),dtype=np.int16))


def test_generic_int16_complement_recipe_composes_with_max():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  def minimum(i):
    left, right, complement = lhs.index(i).load(), rhs.index(i).load(), UOp.const(-1, dtypes.int16)
    inverted_left = UOp(Ops.XOR, dtypes.int16, src=(left, complement))
    inverted_right = UOp(Ops.XOR, dtypes.int16, src=(right, complement))
    return UOp(Ops.XOR, dtypes.int16, src=(inverted_left.maximum(inverted_right), complement))
  image = _lower_uop_program(_program(dtypes.int16, minimum))
  assert image is not None and len(_ew_ops(image)) == 4
  assert all(op.mode==RKEWMode.INT16 for op in _ew_ops(image))


def test_generic_int16_where_avoids_saturating_difference():
  source = UOp.param(1, dtypes.int16, (4,))
  def clipped(i):
    value = source.index(i).load()
    return (value < UOp.const(100, dtypes.int16)).where(value, UOp.const(-100, dtypes.int16))
  image = _lower_uop_program(_program(dtypes.int16, clipped))
  assert image is not None
  assert len(_ew_ops(image)) == 7
  assert all(op.mode==RKEWMode.INT16 for op in _ew_ops(image))


def test_static_bool_materializes_in_int16_consumer_layout():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int16,
    lambda i:(i < UOp.const(2, dtypes.int)).where(lhs.index(i).load(), rhs.index(i).load())))
  assert image is not None and not _ew_ops(image) and len(_output_gathers(image)) == 2
  assert all(gather.itemsize == 2 for gather in _output_gathers(image))


def test_int16_to_int32_is_an_explicit_output_boundary():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int,
    lambda i:(lhs.index(i).load() + rhs.index(i).load()).cast(dtypes.int)))
  assert image is not None and len(_ew_ops(image)) == 2
  assert _ew_ops(image)[-1].mode==RKEWMode.INT16_TO_INT32


def test_bounded_int_root_keeps_dynamic_int32_load_in_canonical_layout():
  out, source = UOp.param(0, dtypes.int, (18,)), UOp.param(1, dtypes.int, (3,))
  row = UOp.range(3, 1)
  cls = UOp.range(6, 0, src=(row,))
  different = source.index(row).load() != cls
  value = different.where(UOp.const(0, dtypes.int), UOp.const(1, dtypes.int))
  image = _lower_uop_program(list(out.index(row*6+cls).store(value).end(cls, row).sink().toposort()))
  assert image is not None and _intermediate_gathers(image) and _ew_ops(image)[-1].mode in (RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT32)


def test_bounded_semantic_int_does_not_alias_int32_output_before_widening():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    (UOp.const(100.0, dtypes.half) < source.index(i).load()).cast(dtypes.int) * UOp.const(2500, dtypes.int)))
  assert image is not None and _ew_ops(image)[-1].mode==RKEWMode.INT16_TO_INT32
  assert all(op.dst.kind is RKBufferKind.SCRATCH for op in _ew_ops(image)[:-1])


def test_int32_load_store_is_raw_four_byte_materialization():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load()))
  assert image is not None and not _ew_ops(image) and len(_output_gathers(image)) == 1
  assert _output_gathers(image)[0].itemsize == 4 and _output_gathers(image)[0].dst.kind is RKBufferKind.ARG


def test_native_int32_mul_uses_the_canonical_wide_layout():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load() * source.index(i).load()))
  assert image is not None and len(_ew_ops(image)) == 1
  assert _ew_ops(image)[0].mode==RKEWMode.INT32


def test_int32_where_constants_convert_at_the_output_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    (UOp.const(0.5, dtypes.half) < source.index(i).load()).where(UOp.const(4, dtypes.int), UOp.const(2, dtypes.int))))
  assert image is not None and len(_ew_ops(image)) > 3
  assert _ew_ops(image)[-1].mode==RKEWMode.INT16_TO_INT32


def test_math_uops_own_multi_stage_recipes():
  source = UOp.param(1, dtypes.half, (4,))
  for op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN):
    image = _lower_uop_program(_program(dtypes.half, lambda i, op=op:UOp(op, dtypes.half, src=(source.index(i).load(),))))
    assert image is not None and len(_ew_ops(image)) > 1
    assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG
    if op is Ops.SIN: assert not any(stage.mode==RKEWMode.COMPARE for stage in _ew_ops(image))


def test_sqrt_recipe_covers_the_positive_binary16_domain():
  values=np.concatenate((np.arange(0x7c00,dtype="<u2").view("<f2"),np.asarray((-4.0,math.inf,math.nan),dtype="<f2")))
  source=UOp.param(1,dtypes.half,(len(values),))
  for reciprocal,divisions in ((False,16),(True,17)):
    def operation(i:UOp) -> UOp:
      value=source.index(i).load().sqrt()
      return value.reciprocal() if reciprocal else value
    image=_lower_uop_program(_program(dtypes.half,operation,len(values)))
    assert image is not None and sum(op.ew_cfg==_EW_CFG[Ops.FDIV] for op in _ew_ops(image))==divisions
    with np.errstate(all="ignore"):
      expected=np.sqrt(values)
      expected=(np.float16(1)/expected).astype("<f2") if reciprocal else expected
      actual=np.frombuffer(_execute_raw_dynamic_image(image,values.nbytes,values.tobytes()),dtype="<f2")
    np.testing.assert_allclose(actual,expected,rtol=5e-3,atol=5e-3,equal_nan=True)


def test_quadratic_math_rewrite_requires_one_shared_log_source():
  source, unrelated = UOp.param(1,dtypes.half,(1,)), UOp.param(2,dtypes.half,(1,))
  radical=(source*source+UOp.const(1.0,dtypes.half)).sqrt()
  scale=UOp.const(math.log(2),dtypes.half)
  assert rockchip_renderer._fold_quadratic(radical) is not None
  assert rockchip_renderer._fold_quadratic((source+radical).log2()*scale) is not None
  assert rockchip_renderer._fold_quadratic((unrelated+radical).log2()*scale) is None


def test_generic_sign_recipe_owns_tagged_semantics():
  source = UOp.param(1, dtypes.half, (4,))
  def sign(i):
    value = source.index(i).load()
    return UOp(Ops.SUB, dtypes.half, src=(value, value), arg=_NATIVE_SIGN)
  image = _lower_uop_program(_program(dtypes.half, sign))
  assert image is not None and len(_ew_ops(image)) == 4
  assert sum(op.mode==RKEWMode.COMPARE for op in _ew_ops(image)) == 2


def test_unrolled_math_reduction_vectorizes_periodic_indices():
  out = UOp.param(0, dtypes.half, (1,))
  lhs, rhs, weights = (UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.half, (8,)),
                       UOp.param(3, dtypes.half, (2,)))
  terms = [lhs.index(i).load().exp2() * rhs.index(i).load() * weights.index(i%2).load() for i in range(8)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(0).store(value).sink().toposort()))
  assert image is not None and _cmac(image) is None and _ew_ops(image)
  assert not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)
  assert decode_image(encode_image(image)) == image


def test_batched_unrolled_math_reduction_materializes_each_uop_result():
  rows, groups = 8, 4
  out, source = UOp.param(0, dtypes.half, (rows,)), UOp.param(1, dtypes.half, (rows*groups,))
  normalizer, lane = UOp.param(2, dtypes.half, (rows,)), UOp.range(rows, 0)
  terms = [(source.index(lane*groups+k).load() - normalizer.index(lane).load()).exp2() for k in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and _cmac(image) is None and _ew_ops(image)
  assert not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)
  assert decode_image(encode_image(image)) == image


def test_attention_softmax_sum_maps_production_exp_rows():
  rows,groups=4096,16
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    scores=Tensor(UOp.new_buffer("ROCKCHIP",rows*groups,dtypes.half,num=1060).reshape((rows,groups)))
    maxima=Tensor(UOp.new_buffer("ROCKCHIP",rows,dtypes.half,num=1061).reshape((rows,1)))
    to_program_cache.clear()
    program=to_program((scores-maxima).exp().sum(-1).schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
  image=decode_image(next(u.arg for u in program.src if u.op is Ops.BINARY))
  assert len(_ew_ops(image))<300 and sum(op.mode is RKEWMode.BOUNDED for op in _ew_ops(image))==9
  assert not _runtime_gathers(image) and decode_image(encode_image(image))==image
  values=np.random.default_rng(0x3588).uniform(-4,0,size=(rows,groups)).astype("<f2")
  maxima_values=values.max(1).astype("<f2")
  actual=np.frombuffer(_execute_raw_dynamic_image(image,rows*2,values.tobytes(),maxima_values.tobytes()),dtype="<f2")
  expected=np.exp(values.astype(np.float32)-maxima_values[:,None].astype(np.float32)).sum(1).astype("<f2")
  np.testing.assert_allclose(actual,expected,rtol=5e-3,atol=5e-3)


def test_attention_softmax_sum_keeps_gqa_surface_fail_closed():
  rows,groups=16384,16
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    scores=Tensor(UOp.new_buffer("ROCKCHIP",rows*groups,dtypes.half,num=1062).reshape((rows,groups)))
    maxima=Tensor(UOp.new_buffer("ROCKCHIP",rows,dtypes.half,num=1063).reshape((rows,1)))
    to_program_cache.clear()
    program=to_program((scores-maxima).exp().sum(-1).schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
  image=decode_image(next(u.arg for u in program.src if u.op is Ops.BINARY))
  assert len(_ew_ops(image))>1000 and not _runtime_gathers(image) and decode_image(encode_image(image))==image


def test_static_reduce_uops_are_structurally_executed():
  values=np.asarray(((1.5,-2,0.5),(-1,3,2)),dtype=np.float16)
  for op in (Ops.ADD, Ops.MAX, Ops.MUL):
    out, source = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,))
    row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
    term = source.index(row*3+axis).load()
    reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(op,0))
    image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
    assert image is not None
    if op is Ops.ADD: assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,1,3)
    else: assert _cmac(image) is None and len(_ew_ops(image))==3
    expected={Ops.ADD:values.sum(1,dtype=np.float16),Ops.MAX:values.max(1),Ops.MUL:values.prod(1,dtype=np.float16)}[op]
    assert _execute_raw_dynamic_image(image,4,values.tobytes()) == expected.tobytes()
    assert not _runtime_gathers(image) and decode_image(encode_image(image)) == image

def test_production_scatter_product_padding_uses_fp16_one_neutral():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    base,source,index=(Tensor(UOp.new_buffer("ROCKCHIP",5,dtype,num=12060+i)) for i,dtype in enumerate((dtypes.half,dtypes.half,dtypes.int)))
    output=base.scatter_reduce(0,index,source,"prod",include_self=False); call=next(u for u in output.schedule_linear().toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK)  # noqa: E501,E702
    image=decode_image(next(u.arg for u in to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY))
  indices=np.asarray((0,0,2,2,2),dtype="<i4"); values=np.asarray((2,3,4,5,6),dtype="<f2"); initial=np.asarray((7,8,9,10,11),dtype="<f2")  # noqa: E702
  actual=np.frombuffer(_execute_raw_dynamic_image(image,10,indices.tobytes(),values.tobytes(),initial.tobytes()),dtype="<f2")
  np.testing.assert_array_equal(actual,np.asarray((6,8,120,10,11),dtype="<f2"))
  assert any(gather.fill_bits==0x3c00 for gather in _static_gathers(image) if gather.offsets and -1 in gather.offsets)

def test_multi_axis_reduce_routes_cmac_unrolling():
  out, source = UOp.param(0,dtypes.half,(2,)), UOp.param(1,dtypes.half,(12,))
  row,outer,inner = UOp.range(2,0),UOp.range(2,1,AxisType.REDUCE),UOp.range(3,2,AxisType.REDUCE)
  term = source.index(row*6+outer*3+inner).load()
  reduced = UOp(Ops.REDUCE,dtypes.float,src=(term.cast(dtypes.float),outer,inner),arg=(Ops.ADD,0))
  image = _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).end(row,outer,inner).sink().toposort()))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,1,6)
  assert not _ew_ops(image) and decode_image(encode_image(image)) == image
  assert _initial_gathers(image)[0].offsets[:6] == tuple(range(6)) and _initial_gathers(image)[0].offsets[32:38] == tuple(range(6,12))

def test_noopt_multi_axis_tensor_sum_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=1):
    source = Tensor(UOp.new_buffer("ROCKCHIP",24,dtypes.half,num=1005).reshape((2,3,4)))
    ast = source.sum((0,2)).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (3,1,8)
  assert not _ew_ops(image) and not _runtime_gathers(image)

def test_scalar_half_to_float_sum_preserves_two_stage_dpu_contract():
  values = np.random.RandomState(0).uniform(-2, 2, size=(45, 3)).astype(np.float16).reshape(-1)
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP", values.size, dtypes.half, num=1001).reshape((45, 3)))
    output = source.sum(dtype=dtypes.float32)
    ast = output.schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is None and _constant_bytes(image) == struct.pack("<e",0.0) and len(_ew_ops(image)) == values.size+1
  sources=tuple(g for g in _static_gathers(image) if g.src is not None and g.src.kind is RKBufferKind.ARG)
  spread=next(g for g in reversed(_static_gathers(image)) if g.src is not None and g.src.kind is RKBufferKind.SCRATCH)
  assert len(sources)==values.size and _gather_point(image,spread)==values.size
  assert sum(op.mode==RKEWMode.HALF_TO_FLOAT for op in _ew_ops(image)) == 1
  assert _ew_ops(image)[-1].dst == RKArg(RKBufferKind.ARG,0) and _ew_ops(image)[-1].mode==RKEWMode.HALF_TO_FLOAT
  assert not _output_gathers(image) and decode_image(encode_image(image)) == image
  assert not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)


def test_scalar_sum_beyond_cmac_k_blocks_uses_production_dpu_reduction():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=1):
    source=Tensor(UOp.new_buffer("ROCKCHIP",417,dtypes.half,num=1037))
    to_program_cache.clear()
    program=to_program(source.sum().schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
  image=decode_image(next(u.arg for u in program.src if u.op is Ops.BINARY))
  tree=_ew_ops(image)
  spread=next(gather for gather in _static_gathers(image) if gather.dst_stride==8)
  assert _cmac(image) is None and tuple(op.count for op in tree)==(2048,1024,512,256,128,64,32,16,8,1)
  assert spread.count==512 and sorted(offset for offset in spread.offsets if offset>=0)==list(range(417))
  assert all(arg.addend%16==0 for op in tree for arg in (op.lhs,op.rhs)) and tree[-1].dst==RKArg(RKBufferKind.ARG,0)
  values=(np.arange(417)%7-3).astype(np.float16)
  assert _execute_fp16_reduction_tail(image,values)[0] == values.sum(dtype=np.float16)
  assert not _runtime_gathers(image) and decode_image(encode_image(image)) == image


def test_real_matmul_routes_production_cmac_and_packs_the_output_surface():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP", 15, dtypes.half, num=1002).reshape((3,5)))
    rhs = Tensor(UOp.new_buffer("ROCKCHIP", 20, dtypes.half, num=1003).reshape((5,4)))
    ast = (lhs@rhs).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (3,4,5)
  assert len(_initial_gathers(image)) == 2 and len(_output_gathers(image)) == 1 and not _ew_ops(image)
  assert _output_gathers(image)[0].offsets == tuple(row*64+col for row in range(3) for col in range(4))
  assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)
  lhs_values, rhs_values = np.arange(15, dtype=np.float16), np.arange(20, dtype=np.float16).reshape(5,4)
  sources = {_initial_gathers(image)[0].src.index:lhs_values.view(np.uint16),
             _initial_gathers(image)[1].src.index:rhs_values.reshape(-1).view(np.uint16)}
  scratch = [np.zeros(size//2, dtype=np.uint16) for size in image.scratch]
  for gather in _initial_gathers(image):
    destination, offsets = scratch[gather.dst.index], np.asarray(gather.offsets)
    valid = offsets >= 0
    destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src.index][offsets[valid]]
  packed_lhs = scratch[_cmac(image).lhs.index].view(np.float16).reshape(3,32)
  packed_rhs = scratch[_cmac(image).rhs.index].view(np.float16).reshape(2,1,16,32).transpose(0,2,1,3).reshape(32,32)
  np.testing.assert_array_equal(packed_lhs[:,:5], lhs_values.reshape(3,5))
  np.testing.assert_array_equal(packed_rhs[:4,:5], rhs_values.T)
  np.testing.assert_array_equal(packed_lhs[:,:5].astype(np.float32)@packed_rhs[:4,:5].T.astype(np.float32),
                                lhs_values.reshape(3,5).astype(np.float32)@rhs_values.astype(np.float32))


@pytest.mark.parametrize(("formula","lhs_shape","rhs_shape"), (
  ("pqrs,tuqvr->pstuv",(3,5,8,10),(11,7,5,13,8)),
  ("prsq,tquvr->pstuv",(3,8,10,5),(11,5,7,13,8)),
  ("zqrs,tuqvr->zstuv",(3,5,8,10),(11,7,5,13,8))))
def test_large_einsum_uses_transposed_contiguous_cmac_view(formula:str, lhs_shape:tuple[int,...], rhs_shape:tuple[int,...]):
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    lhs=Tensor(UOp.new_buffer("ROCKCHIP",math.prod(lhs_shape),dtypes.half,num=1064).reshape(lhs_shape))
    rhs=Tensor(UOp.new_buffer("ROCKCHIP",math.prod(rhs_shape),dtypes.half,num=1065).reshape(rhs_shape))
    to_program_cache.clear()
    program=to_program(Tensor.einsum(formula,lhs,rhs).schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
  image=decode_image(next(u.arg for u in program.src if u.op is Ops.BINARY))
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k)==(1001,30,40)
  assert not _ew_ops(image) and not _runtime_gathers(image) and _assert_decoded_image_bounds(image)==image
  lhs_values=(np.arange(math.prod(lhs_shape))%5-2).astype("<f2").reshape(lhs_shape)
  rhs_values=(np.arange(math.prod(rhs_shape))%5-2).astype("<f2").reshape(rhs_shape)
  actual=np.frombuffer(_execute_raw_dynamic_image(image,30030*2,lhs_values.tobytes(),rhs_values.tobytes()),dtype="<f2").reshape(3,10,11,7,13)
  expected=np.einsum(formula,lhs_values.astype(np.float32),rhs_values.astype(np.float32)).astype("<f2")
  np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize(("m","k","n"), ((256,256,256),(192,256,160),(64,384,384),(512,128,128)))
def test_large_affine_matmul_keeps_cmac_planning_compact(m:int, k:int, n:int):
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP",m*k,dtypes.half,num=1040)).reshape(m,k)
    rhs = Tensor(UOp.new_buffer("ROCKCHIP",k*n,dtypes.half,num=1041)).reshape(k,n)
    ast = (lhs@rhs).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (m,n,k)
  assert len(_initial_gathers(image)) == 2 and not _ew_ops(image) and decode_image(encode_image(image)) == image
  def offsets(gather:RKGather) -> np.ndarray:
    lanes = np.arange(gather.count,dtype=np.int64)
    indices = np.full(gather.count,gather.base,dtype=np.int64)
    for divisor,limit,stride in gather.axes: indices += lanes//divisor%limit*stride
    return indices
  assert len(encode_image(image)) < 512
  assert all(gather.axes and not gather.offsets for gather in (*_initial_gathers(image),*_output_gathers(image)))
  lhs_offsets,rhs_offsets = map(offsets,_initial_gathers(image))
  np.testing.assert_array_equal(lhs_offsets,np.arange(m*k))
  expected_rhs = np.asarray([(ib*32+ki)*n+ob*16+ni for ob in range(n//16) for ib in range(k//32) for ni in range(16) for ki in range(32)])
  np.testing.assert_array_equal(rhs_offsets,expected_rhs)
  lhs_values = (np.arange(m*k).reshape(m,k)%7-3).astype(np.float16)
  rhs_values = (np.arange(k*n).reshape(k,n)%5-2).astype(np.float16)
  packed_rhs = rhs_values.reshape(-1)[rhs_offsets].reshape(n//16,k//32,16,32).transpose(0,2,1,3).reshape(n,k)
  np.testing.assert_array_equal(lhs_values.astype(np.float32)@packed_rhs.T.astype(np.float32),
                                lhs_values.astype(np.float32)@rhs_values.astype(np.float32))
  expected_output = np.asarray([row*n*2+col//16*32+col%16 for row in range(m) for col in range(n)])
  np.testing.assert_array_equal(offsets(_output_gathers(image)[0]),expected_output)


def test_real_matmul_relu_routes_one_production_cmac_stage():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP", 15, dtypes.half, num=1030).reshape((3,5)))
    rhs = Tensor(UOp.new_buffer("ROCKCHIP", 20, dtypes.half, num=1031).reshape((5,4)))
    ast = (lhs@rhs).relu().schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and _cmac(image).relu and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (3,4,5)
  assert not _ew_ops(image) and len(emit_cmac_stage(_cmac(image),lambda _arg:0)) == 45
  lhs_values = (np.arange(15)-7).astype(np.float16).reshape(3,5)
  rhs_values = (np.arange(20)%5-2).astype(np.float16).reshape(5,4)
  sources = {_initial_gathers(image)[0].src.index:lhs_values.view(np.uint16).reshape(-1),
             _initial_gathers(image)[1].src.index:rhs_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(size//2,dtype=np.uint16) for size in image.scratch]
  for gather in _initial_gathers(image):
    offsets = np.asarray(gather.offsets)
    valid = offsets >= 0
    scratch[gather.dst.index][:gather.count] = gather.fill_bits
    scratch[gather.dst.index][:gather.count][valid] = sources[gather.src.index][offsets[valid]]
  packed_lhs = scratch[_cmac(image).lhs.index].view(np.float16).reshape(3,32)
  packed_rhs = scratch[_cmac(image).rhs.index].view(np.float16).reshape(2,1,16,32).transpose(0,2,1,3).reshape(32,32)
  physical = np.zeros(3*32*2,dtype=np.float16)
  physical.reshape(3,64)[:,:4] = np.maximum(packed_lhs.astype(np.float32)@packed_rhs.astype(np.float32).T,0)[:,:4].astype(np.float16)
  np.testing.assert_array_equal(physical[np.asarray(_output_gathers(image)[0].offsets)],
                                np.maximum(lhs_values.astype(np.float32)@rhs_values.astype(np.float32),0).astype(np.float16).reshape(-1))


def test_zero_gated_padded_convolution_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",324,dtypes.half,num=1012)).reshape(1,4,9,9)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",144,dtypes.half,num=1013)).reshape(4,4,3,3)
    ast = source.conv2d(weight,padding=1).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (4,81,36)
  assert not _ew_ops(image) and len(_initial_gathers(image))==2
  assert any(offset<0 for gather in _initial_gathers(image) for offset in gather.offsets)
  source_values = (np.arange(324)%5-2).astype(np.float16).reshape(1,4,9,9)
  weight_values = (np.arange(144)%5-2).astype(np.float16).reshape(4,4,3,3)
  sources = {1:source_values.view(np.uint16).reshape(-1),2:weight_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(size//2,dtype=np.uint16) for size in image.scratch]
  for gather in _initial_gathers(image):
    destination,offsets = scratch[gather.dst.index],np.asarray(gather.offsets)
    valid = offsets >= 0
    if not gather.partial: destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src.index][offsets[valid]]
  packed_weight = scratch[_cmac(image).lhs.index].view(np.float16).reshape(4,96)
  packed_source = scratch[_cmac(image).rhs.index].view(np.float16).reshape(6,3,16,32).transpose(0,2,1,3).reshape(96,96)
  expected = np.asarray([[sum(float(weight_values[oc,ic,ky,kx])*float(source_values[0,ic,y+ky-1,x+kx-1])
    for ic in range(4) for ky in range(3) for kx in range(3) if 0 <= y+ky-1 < 9 and 0 <= x+kx-1 < 9)
    for y in range(9) for x in range(9)] for oc in range(4)],dtype=np.float32)
  np.testing.assert_array_equal(packed_weight[:,:36].astype(np.float32)@packed_source[:81,:36].T.astype(np.float32),expected)
  assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)
  out,lhs,rhs = UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(4,)),UOp.param(2,dtypes.half,(4,))
  for fill in (-0.0,1.0):
    terms = [lhs.index(UOp.const(-1,dtypes.int)).load(UOp.const(fill,dtypes.half),UOp.const(False,dtypes.bool))*rhs.index(i).load()
             for i in range(4)]
    fallback = _lower_uop_program(list(out.index(0).store(functools.reduce(lambda total,term:total+term,terms[1:],terms[0])).sink().toposort()))
    assert fallback is not None and _cmac(fallback) is None and _ew_ops(fallback)


def test_batched_zero_gated_convolution_reorders_one_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",648,dtypes.half,num=1014)).reshape(2,4,9,9)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",144,dtypes.half,num=1015)).reshape(4,4,3,3)
    ast = source.conv2d(weight,padding=1).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (4,162,36)
  assert not _ew_ops(image) and len(_initial_gathers(image)) == 2
  source_values = (np.arange(648)%5-2).astype(np.float16).reshape(2,4,9,9)
  weight_values = (np.arange(144)%5-2).astype(np.float16).reshape(4,4,3,3)
  sources = {1:source_values.view(np.uint16).reshape(-1),2:weight_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(size//2,dtype=np.uint16) for size in image.scratch]
  for gather in _initial_gathers(image):
    destination,offsets = scratch[gather.dst.index],np.asarray(gather.offsets)
    valid = offsets >= 0
    if not gather.partial: destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src.index][offsets[valid]]
  packed_weight = scratch[_cmac(image).lhs.index].view(np.float16).reshape(4,192)
  packed_source = scratch[_cmac(image).rhs.index].view(np.float16).reshape(12,6,16,32).transpose(0,2,1,3).reshape(192,192)
  expected = np.asarray([[[[sum(float(weight_values[oc,ic,ky,kx])*float(source_values[batch,ic,y+ky-1,x+kx-1])
    for ic in range(4) for ky in range(3) for kx in range(3) if 0 <= y+ky-1 < 9 and 0 <= x+kx-1 < 9)
    for x in range(9)] for y in range(9)] for oc in range(4)] for batch in range(2)],dtype=np.float32)
  np.testing.assert_array_equal(packed_weight[:,:36].astype(np.float32)@packed_source[:162,:36].T.astype(np.float32),
                                expected.transpose(1,0,2,3).reshape(4,162))
  assert _output_gathers(image)[0].offsets == tuple(oc*384+(batch*81+lane)//16*32+(batch*81+lane)%16
    for batch in range(2) for oc in range(4) for lane in range(81))
  assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)


def test_output_padded_transpose_convolution_avoids_stateful_vector_reduction():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",240,dtypes.half,num=14001)).reshape(2,4,6,5)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",144,dtypes.half,num=14002)).reshape(4,4,3,3)
    bias = Tensor(UOp.new_buffer("ROCKCHIP",4,dtypes.half,num=14003))
    ast = source.conv_transpose2d(weight,bias,output_padding=(1,1),stride=(2,3)).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    image = decode_image(next(u for u in to_program(ast,RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is None and len(_ew_ops(image)) == 2214 and len(_initial_gathers(image)) == 125 and not _intermediate_gathers(image)
  assert not any(op.submit_barrier or op.mode != RKEWMode.HALF for op in _ew_ops(image)) and not _runtime_gathers(image)
  source_values,weight_values,bias_values=np.ones((2,4,6,5),dtype=np.float16),np.ones((4,4,3,3),dtype=np.float16),np.ones(4,dtype=np.float16)
  expected=np.ones((2,4,14,16),dtype=np.float16)
  for y,x in itertools.product(range(6),range(5)): expected[:,:,y*2:y*2+3,x*3:x*3+3] += np.float16(4)
  assert _execute_raw_dynamic_image(image,expected.nbytes,source_values.tobytes(),weight_values.tobytes(),bias_values.tobytes()) == expected.tobytes()
  assert decode_image(encode_image(image)) == image


def test_biased_eight_channel_convolutions_use_shared_kahan_recipe():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",200,dtypes.half,num=13001)).reshape(1,8,5,5)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",64,dtypes.half,num=13002)).reshape(8,8,1,1)
    bias = Tensor(UOp.new_buffer("ROCKCHIP",8,dtypes.half,num=13003))
    schedule = source.conv2d(weight,bias).relu().conv2d(weight,bias).schedule_linear()
    calls = [u for u in schedule.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    records,images = [],[]
    for call in calls:
      to_program_cache.clear()
      program = to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP")))
      blob = next(u.arg for u in program.src if u.op is Ops.BINARY)
      image = decode_image(blob)
      assert encode_image(image) == blob and _cmac(image) is None
      inputs=sum(g.src is not None and g.src.kind is RKBufferKind.ARG for g in _static_gathers(image))
      records.append((len(_ew_ops(image)),inputs))
      images.append(image)
  assert records == [(253,17),(251,17)]
  rng=np.random.RandomState(0)
  source_values=rng.uniform(-1,1,(1,8,5,5)).astype(np.float16)
  weight_values=rng.uniform(-1,1,(8,8,1,1)).astype(np.float16)
  bias_values=rng.uniform(-1,1,8).astype(np.float16)
  expected_first=np.maximum(np.einsum("bcyx,oc->boyx",source_values.astype(np.float32),weight_values[:,:,0,0].astype(np.float32))+bias_values[None,:,None,None],0).astype(np.float16)  # noqa: E501
  got_first=np.frombuffer(_execute_raw_dynamic_image(images[0],400,source_values.tobytes(),weight_values.tobytes(),bias_values.tobytes()),dtype="<f2").reshape(1,8,5,5)  # noqa: E501
  expected_second=(np.einsum("bcyx,oc->boyx",expected_first.astype(np.float32),weight_values[:,:,0,0].astype(np.float32))+bias_values[None,:,None,None]).astype(np.float16)  # noqa: E501
  got_second=np.frombuffer(_execute_raw_dynamic_image(images[1],400,got_first.tobytes(),weight_values.tobytes(),bias_values.tobytes()),dtype="<f2").reshape(1,8,5,5)  # noqa: E501
  np.testing.assert_allclose(got_first,expected_first,atol=5e-3,rtol=5e-3)
  np.testing.assert_allclose(got_second,expected_second,atol=5e-3,rtol=5e-3)


def test_fp32_contraction_biases_route_one_production_cmac_surface():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP",6,dtypes.half,num=1007)).reshape(2,3)
    rhs = Tensor(UOp.new_buffer("ROCKCHIP",6,dtypes.half,num=1008)).reshape(3,2)
    biases = ((Tensor(UOp.new_buffer("ROCKCHIP",2,dtypes.half,num=1009)),np.asarray((7,8),dtype=np.float16)),
      (Tensor(UOp.new_buffer("ROCKCHIP",2,dtypes.half,num=1010)).reshape(2,1),np.asarray(((7,),(8,)),dtype=np.float16)),
      (Tensor(UOp.new_buffer("ROCKCHIP",1,dtypes.half,num=1011)),np.asarray((7,),dtype=np.float16)))
  lhs_values = np.arange(1,7,dtype=np.float16).reshape(2,3)
  rhs_values = np.arange(1,7,dtype=np.float16).reshape(3,2)
  for bias,bias_values in biases:
    ast = (lhs.matmul(rhs,dtype=dtypes.float)+bias.cast(dtypes.float)).cast(dtypes.half).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
    image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
    assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,2,4)
    assert not _ew_ops(image) and len(_initial_gathers(image)) == 4 and sum(bool(gather.values) for gather in _initial_gathers(image)) == 1
    got=np.frombuffer(_execute_raw_dynamic_image(image,8,lhs_values.tobytes(),rhs_values.tobytes(),bias_values.tobytes()),dtype="<f2").reshape(2,2)  # noqa: E501
    expected=(lhs_values.astype(np.float32)@rhs_values.astype(np.float32)+bias_values.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(got,expected)
    assert not _runtime_gathers(image) and decode_image(encode_image(image)) == image


def test_literal_fp32_bias_and_large_broadcast_share_cmac_candidate_planner():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",64,dtypes.half,num=1026)).reshape(64,1).expand(64,64)
    ast = (source.cast(dtypes.float)+3.0).cast(dtypes.half).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (64,64,2)
  assert not _ew_ops(image) and len(_initial_gathers(image)) == 3 and sum(bool(gather.values) for gather in _initial_gathers(image)) == 2
  source_values = np.arange(64,dtype=np.float16)
  scratch = [np.zeros(size//2,dtype=np.uint16) for size in image.scratch]
  for gather in _initial_gathers(image):
    destination = scratch[gather.dst.index]
    if gather.values: destination[:gather.count] = gather.values
    else:
      offsets = np.asarray(gather.offsets)
      valid = offsets >= 0
      if not gather.partial: destination[:gather.count] = gather.fill_bits
      destination[:gather.count][valid] = source_values.view(np.uint16)[offsets[valid]]
  ai,ao,_ = rockchip_renderer._cmac_layout(_cmac(image).n,_cmac(image).k)
  lhs = scratch[_cmac(image).lhs.index].view(np.float16).reshape(_cmac(image).m,ai)
  rhs = scratch[_cmac(image).rhs.index].view(np.float16).reshape(ao//16,ai//32,16,32).transpose(0,2,1,3).reshape(ao,ai)
  result = (lhs.astype(np.float32)@rhs.astype(np.float32).T).astype(np.float16)
  physical = np.zeros(_cmac(image).m*ao*2,dtype=np.float16)
  for row in range(_cmac(image).m):
    for col in range(_cmac(image).n): physical[row*ao*2+col//16*32+col%16] = result[row,col]
  got = physical[np.asarray(_output_gathers(image)[0].offsets)]
  expected = (source_values[:,None]+np.float16(3)).repeat(64,axis=1).reshape(-1)
  np.testing.assert_array_equal(got,expected)
  assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)


def test_tensor_mean_routes_scaled_production_cmac_weights():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",8,dtypes.half,num=1004))
    ast = source.mean().schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  images = [decode_image(u.arg) for u in program.src if u.op is Ops.BINARY]
  image = next(image for image in images if _cmac(image) is not None)
  assert (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,8)
  assert not _constant_bytes(image) and len(image.scratch) == 3 and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*8
  assert _cmac(image).out_fp16 and not _ew_ops(image)
  assert not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)


def test_avg_pool3d_static_denominator_stays_compile_time_data():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",4096,dtypes.half,num=1005)).reshape(1,1,16,16,16)
    output = source.avg_pool2d(kernel_size=(8,8,8),stride=5,padding=1,count_include_pad=False)
    to_program_cache.clear()
    program = to_program(output.schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert len(_ew_ops(image)) == 20 and len(_static_gathers(image)) == 7 and len(image.scratch) == 9
  assert not any(op.mode in _TYPED_EW_MODES for op in _ew_ops(image))
  values=np.ones((1,1,16,16,16),dtype=np.float16)
  assert _execute_raw_dynamic_image(image,54,values.tobytes()) == np.ones((1,1,3,3,3),dtype=np.float16).tobytes()
  assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)


def test_arange_weighted_tensor_sum_routes_binary_outer_scale_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",5,dtypes.half,num=1006))
    weighted = source*Tensor.arange(1,6).cast(dtypes.half)
    def scaled(*factors:float) -> Tensor:
      value = weighted.sum(dtype=dtypes.float)
      for factor in factors: value = value*factor
      return value.cast(dtypes.half)
    outputs = (weighted.sum(),scaled(0.5),scaled(0.75),scaled(-0.5),(source.sum(dtype=dtypes.float)*0.75).cast(dtypes.half))
    images = []
    for output in outputs:
      to_program_cache.clear()
      program = to_program(output.schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
      images.append(decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg))
  for image,scale in zip(images[:2],(1.0,0.5)):
    assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,5)
    assert _initial_gathers(image)[0].offsets[:5] == tuple(range(5))
    assert _cmac_weights(image) == tuple(rockchip_renderer._fp16_bits(scale*value) for value in range(1,6))
    assert not _constant_bytes(image) and not _ew_ops(image) and not _runtime_gathers(image)
  assert all(_cmac(image) is None and _ew_ops(image) for image in images[2:])
  def lower_factors(factors:tuple[float, ...]) -> RKImage|None:
    out,source = UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(4,))
    terms = [source.index(i).load().cast(dtypes.float)*UOp.const(weight,dtypes.float)
             for i,weight in enumerate((0.5,-0.25,2.0,0.125))]
    value = functools.reduce(lambda total,term:total+term,terms[1:],terms[0])
    for factor in factors: value = value*UOp.const(factor,dtypes.float)
    return _lower_uop_program(list(out.index(0).store(value.cast(dtypes.half)).sink().toposort()))
  exact = lower_factors((0.25,2.0))
  rejected = tuple(lower_factors(factors) for factors in ((0.1,5.0),(0.1,10.0),(2.0**127,2.0**-127)))
  assert exact is not None and _cmac(exact) is not None and _cmac_weights(exact) == tuple(
    rockchip_renderer._fp16_bits(value) for value in (0.25,-0.125,1.0,0.0625))
  assert all(image is not None and _cmac(image) is None and _ew_ops(image) for image in rejected)


def test_static_dot_reduce_owns_accurate_physical_recipe():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,)), UOp.param(2, dtypes.half, (6,))
  row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
  term = lhs.index(row*3+axis).load() * rhs.index(row*3+axis).load()
  reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(Ops.ADD,0))
  image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,2,3)
  assert _output_gathers(image)[0].offsets == (0, 65) and not _ew_ops(image)


def test_vectorized_mul_add_reduction_retains_product_residuals_and_relu():
  groups = 64
  rows = _MAX_EW_ELEMS_FP16+1
  out = UOp.param(0, dtypes.half, (rows,))
  lhs, rhs = UOp.param(1, dtypes.half, (rows*groups,)), UOp.param(2, dtypes.half, (rows*groups,))
  lane = UOp.range(rows, 0)
  terms = [lhs.index(lane*groups+k).load() * rhs.index(lane*groups+k).load() for k in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  zero = UOp.const(0.0, dtypes.half)
  value = (zero < value).where(value, zero)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and _cmac(image) is None and _ew_ops(image)
  assert _ew_ops(image)[-1].dst == RKArg(RKBufferKind.ARG, 0)
  assert not _runtime_gathers(image) and decode_image(encode_image(image)) == image


def test_production_batched_dot_retains_product_residuals_after_cmac_rejects():
  lhs_shape, rhs_shape = (8,45,65), (8,65,100)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP",math.prod(lhs_shape),dtypes.half,num=1033)).reshape(*lhs_shape)
    rhs = Tensor(UOp.new_buffer("ROCKCHIP",math.prod(rhs_shape),dtypes.half,num=1034)).reshape(*rhs_shape)
    ast = lhs.dot(rhs).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  inputs=sum(g.src is not None and g.src.kind is RKBufferKind.ARG for g in _static_gathers(image))
  assert _cmac(image) is None and len(_ew_ops(image))==4012 and len(image.scratch)==269 and inputs==130
  assert not _runtime_gathers(image) and _assert_decoded_image_bounds(image) == image


def test_production_composite_product_sum_uses_mapped_reduction_after_cmac_rejects():
  shape = (32,10)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    logits = Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=1035)).reshape(*shape)
    target = Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=1036)).reshape(*shape)
    ast = logits.cross_entropy(target,reduction="sum").schedule_linear().src[-1].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  spread=next(gather for gather in reversed(_intermediate_gathers(image)) if gather.dst_stride == 8)
  assert _cmac(image) is None and spread.count == math.prod(shape)*6
  assert len(_ew_ops(image)) == _gather_point(image,spread)+4*(spread.count-1)+2
  assert all(arg.addend%16==0 for op in _ew_ops(image) for arg in (op.lhs,op.rhs))
  values=np.ones(spread.count,dtype=np.float16)
  assert _execute_fp16_reduction_tail(image,values)[0] == np.float16(-len(values))
  assert not _runtime_gathers(image) and decode_image(encode_image(image)) == image
  assert _assert_decoded_image_bounds(image) == image


@pytest.mark.parametrize(("smoothing","expected_hash"), ((0.3,"42c639d9e2df33714b56e9f9dcd8dd12c5cbdafc1c971bc14fdef3a54194ee5e"),
  (0.9,"e47ed027d3224b83b7a69e005a6c8b73b0867f680809408ed62d676bcf7a42d1")))
def test_production_sparse_label_smoothing_stages_nested_reductions(smoothing:float, expected_hash:str):
  shape=(12,10)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    logits=Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=1037)).reshape(*shape)
    classes=Tensor(UOp.new_buffer("ROCKCHIP",shape[0],dtypes.int,num=1038))
    calls=[u for u in logits.sparse_categorical_crossentropy(classes,label_smoothing=smoothing).schedule_linear().toposort()
           if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    images=[]
    for call in calls:
      to_program_cache.clear()
      images.append(decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)))
  assert len(images)==3 and len(_ew_ops(images[2]))==174 and len(images[2].program)<250
  assert not _runtime_gathers(images[2]) and all(_assert_decoded_image_bounds(image)==image for image in images)
  digest=hashlib.sha256()
  for seed in range(16):
    values=np.random.default_rng(seed).uniform(-4,4,size=shape).astype("<f2")
    labels=((np.arange(shape[0],dtype=np.int32)*7+seed)%shape[1]).astype("<i4")
    with np.errstate(over="ignore",invalid="ignore"):
      maximum=_execute_raw_dynamic_image(images[0],shape[0]*2,values.tobytes())
      summed=_execute_raw_dynamic_image(images[1],shape[0]*2,values.tobytes(),maximum)
      loss=_execute_raw_dynamic_image(images[2],2,values.tobytes(),maximum,summed,labels.tobytes())
    digest.update(loss)
    shifted=values.astype(np.float64)-values.max(axis=1,keepdims=True).astype(np.float64)
    log_probs=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    expected=-np.mean((1-smoothing)*log_probs[np.arange(shape[0]),labels]+smoothing*log_probs.mean(axis=1))
    np.testing.assert_allclose(np.frombuffer(loss,dtype="<f2")[0],expected,rtol=2e-3,atol=2e-3)
  assert digest.hexdigest()==expected_hash


def test_production_causal_attention_applies_infinite_mask_after_precise_dot():
  shape = (32,8,16,64)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    q,k,v = (Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=1030+i)).reshape(*shape) for i in range(3))
    ast = q.scaled_dot_product_attention(k,v,is_causal=True).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  terminal = _ew_ops(image)[-1]
  masks = [gather for gather in _static_gathers(image) if gather.dst.kind is terminal.rhs.kind and gather.dst.index == terminal.rhs.index and
           gather.values and set(gather.values) == {0,rockchip_renderer._fp16_bits(-math.inf)}]
  assert _cmac(image) is None and len(_ew_ops(image)) > 1000 and terminal.dst == RKArg(RKBufferKind.ARG,0)
  assert terminal.ew_cfg == _EW_CFG[Ops.ADD] and len(masks) == 1
  assert _gather_point(image,masks[0]) == 0 and image.program[-1] is terminal
  assert not _runtime_gathers(image) and _assert_decoded_image_bounds(image) == image


def test_generic_image_allows_many_small_ew_stages():
  count = 1237
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.int, (count,)), UOp.param(2, dtypes.int, (count,))
  value = UOp.const(0.0, dtypes.half)
  for index in range(count):
    value = value + (lhs.index(index).load() < rhs.index(index).load()).where(
      UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half))
  uops = list(out.index(0).store(value).sink().toposort())
  image = _lower_uop_program(uops)
  assert image is not None and len(_ew_ops(image)) > _RKIMAGE_U16_MAX > _MAX_EW_ELEMS_FP16
  assert decode_image(encode_image(image)) == image


def test_short_fp32_add_mul_tree_routes_cmac_at_output_boundary():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,2,2) and not _ew_ops(image)


def test_short_tensor_dot_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    for k in (2,3):
      for fp16 in (True,False):
        lhs = Tensor(UOp.new_buffer("ROCKCHIP",k,dtypes.half,num=1020+2*k))
        rhs = Tensor(UOp.new_buffer("ROCKCHIP",k,dtypes.half,num=1021+2*k))
        result = (lhs*rhs).sum(dtype=dtypes.float)
        ast = (result.cast(dtypes.half) if fp16 else result).schedule_linear().src[0].src[0]
        to_program_cache.clear()
        program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
        image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
        assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k,_cmac(image).out_fp16) == (1,1,k,fp16)
        assert not _ew_ops(image) and tuple(gather.src.index for gather in _initial_gathers(image) if not gather.values) == (1,2)
        assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)


def test_static_fp32_subgraph_rounds_only_after_coordinate_cancellation():
  lane = UOp.range(31, 0)
  coordinate = (lane.cast(dtypes.float)+UOp.const(0.5, dtypes.float))*UOp.const(20/31, dtypes.float) - \
    UOp.const(0.5, dtypes.float)
  fraction = coordinate - UOp(Ops.TRUNC, dtypes.float, src=(coordinate,))
  lowered = _fp32_expr_to_half(fraction)
  assert lowered.op is Ops.CAST and lowered.dtype.scalar() is dtypes.half and lowered.src == (fraction,)


def test_terminal_half_to_float_cast_uses_chunked_dpu_output_conversion():
  source = UOp.param(1, dtypes.half, (9,))
  image = _lower_uop_program(_program(dtypes.float, lambda i:source.index(i).load().cast(dtypes.float), count=9))
  assert image is not None and len(_ew_ops(image))==3 and tuple(type(op) for op in image.program)==(RKGather,)*3+(RKEWOp,)*3
  assert tuple(gather.count for gather in _static_gathers(image))==(4,4,1)
  assert all(op.mode==RKEWMode.HALF_TO_FLOAT and op.dst.kind is RKBufferKind.ARG for op in _ew_ops(image))
  assert decode_image(encode_image(image)) == image


def test_terminal_int_to_float_cast_composes_integer_and_fp32_converters():
  source = UOp.param(1, dtypes.int, (9,))
  image = _lower_uop_program(_program(dtypes.float, lambda i:source.index(i).load().cast(dtypes.float), count=9))
  assert image is not None and len(_ew_ops(image)) == 6
  assert sum(op.mode==RKEWMode.HALF_TO_FLOAT for op in _ew_ops(image)) == 3
  assert decode_image(encode_image(image)) == image


def test_remapped_integer_and_bool_to_float_casts_use_generic_typed_values():
  for dtype,stages in ((dtypes.int, 6), (dtypes.bool, 9)):
    source = UOp.param(1, dtype, (9,))
    image = _lower_uop_program(_program(dtypes.float, lambda i:source.index(8-i).load().cast(dtypes.float), count=9))
    assert image is not None and len(_ew_ops(image)) == stages and decode_image(encode_image(image)) == image


def test_terminal_half_casts_use_typed_integer_and_canonical_bool_abis():
  source = UOp.param(1, dtypes.half, (9,))
  integer = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load().cast(dtypes.int), count=9))
  boolean = _lower_uop_program(_program(dtypes.bool, lambda i:source.index(i).load().cast(dtypes.bool), count=9))
  assert integer is not None and _ew_ops(integer)[-1].mode in _INT32_OUTPUT_MODES
  assert boolean is not None and _ew_ops(boolean)[-1].mode==RKEWMode.HALF_TO_INT16 and _ew_ops(boolean)[-1].dst.kind is RKBufferKind.SCRATCH
  assert len(_output_gathers(boolean)) == 1 and _output_gathers(boolean)[0].itemsize == 1
  assert decode_image(encode_image(integer)) == integer and decode_image(encode_image(boolean)) == boolean


def test_fp32_pure_add_tree_routes_cmac_at_the_half_output_boundary():
  source = UOp.param(1, dtypes.half, (64,))
  terms = [source.index(i).load().cast(dtypes.float) for i in range(64)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(_program(dtypes.half, lambda _i:value.cast(dtypes.half), count=1))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,64)
  assert _cmac(image).out_fp16 and not _ew_ops(image) and decode_image(encode_image(image)) == image


def test_multiple_production_fp32_reductions_share_cmac_surfaces():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    sources = [Tensor(UOp.new_buffer("ROCKCHIP",8,dtypes.half,num=1010+i)) for i in range(4)]
    outputs = ((sources[0].sum(dtype=dtypes.float)+sources[1].sum(dtype=dtypes.float)).cast(dtypes.half),
      ((sources[0]*sources[1]).sum(dtype=dtypes.float)+(sources[2]*sources[3]).sum(dtype=dtypes.float)).cast(dtypes.half))
    images = []
    for output in outputs:
      to_program_cache.clear()
      program = to_program(output.schedule_linear().src[0].src[0],RockchipRenderer(Target(device="ROCKCHIP")))
      images.append(decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg))
  for image,source_slots in zip(images,((1,2),(1,3,2,4))):
    assert _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,16) and not _ew_ops(image)
    dynamic = tuple(gather for gather in _initial_gathers(image) if not gather.values)
    assert tuple(gather.src.index for gather in dynamic) == source_slots
    assert tuple(gather.partial for gather in dynamic) == tuple(i%2 == 1 for i in range(len(dynamic)))
    assert tuple(tuple(i for i,offset in enumerate(gather.offsets) if offset >= 0) for gather in dynamic) == \
      ((tuple(range(8)),tuple(range(8,16)))*2)[:len(dynamic)]
    assert all(tuple(offset for offset in gather.offsets if offset >= 0) == tuple(range(8)) for gather in dynamic)
    assert decode_image(encode_image(image)) == image and not _runtime_gathers(image)


def test_std_mean_outer_selector_commits_two_aligned_output_surfaces():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",13125,dtypes.half,num=14004)).reshape(15,25,35)
    linear = Tensor.stack(*source.std_mean()).schedule_linear()
    to_program_cache.clear()
    image = decode_image(next(u for u in to_program(linear.src[-1].src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY).arg)
  commits=tuple(gather for gather in _static_gathers(image) if gather.dst.kind is RKBufferKind.ARG)
  atoms=tuple(gather for gather in _static_gathers(image) if gather.dst_stride==8)
  assert _cmac(image) is None and len(_ew_ops(image))==90 and [(g.count,g.dst_stride) for g in atoms]==[(16384,8)]*2
  assert all(arg.addend%16==0 for op in _ew_ops(image) for arg in (op.lhs,op.rhs))
  assert [(g.count,g.dst_addend) for g in commits] == [(1,0),(1,1)]
  assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and not _runtime_gathers(image)
  assert decode_image(encode_image(image)) == image


def test_multisource_cmac_resource_cap_charges_each_partial_surface(monkeypatch):
  sources = (UOp.param(1,dtypes.half,(8,)),UOp.param(2,dtypes.half,(8,)))
  terms = [source.index(i).load().cast(dtypes.float) for source in sources for i in range(8)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  monkeypatch.setattr(rockchip_renderer,"_MAX_DYNAMIC_SELECTOR_CELLS",1070)
  uops = _program(dtypes.half,lambda _i:value.cast(dtypes.half),count=1)
  assert (output:=rockchip_renderer._outs(uops)[1]) is not None and rockchip_renderer._lower_reduction(output,uops) is None


def test_scaled_pure_sum_routes_cmac_weights_but_scaled_dot_stays_generic():
  lhs, rhs = UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.half, (8,))
  sums = [lhs.index(i).load().cast(dtypes.float) for i in range(8)]
  dots = [lhs.index(i).load().cast(dtypes.float)*rhs.index(i).load().cast(dtypes.float) for i in range(8)]
  for terms in (sums,dots):
    value = terms[0]
    for term in terms[1:]: value = value+term
    value = value*0.125
    image = _lower_uop_program(_program(dtypes.half,lambda _i:value.cast(dtypes.half),count=1))
    assert image is not None
    if terms is sums:
      assert _cmac(image) is not None and not _constant_bytes(image) and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*8
      assert not _ew_ops(image)
    else: assert _cmac(image) is None and _ew_ops(image)


def test_nested_scaled_direct_reduce_routes_cmac_but_scaled_dot_stays_generic():
  out, lhs, rhs = UOp.param(0,dtypes.half,(2,)),UOp.param(1,dtypes.half,(8,)),UOp.param(2,dtypes.half,(8,))
  row,axis = UOp.range(2,0),UOp.range(4,1,AxisType.REDUCE)
  for dot in (False,True):
    value = lhs.index(row*4+axis).load().cast(dtypes.float)
    if dot: value = value*rhs.index(row*4+axis).load().cast(dtypes.float)
    reduced = UOp(Ops.REDUCE,dtypes.float,src=(value,axis),arg=(Ops.ADD,0))*0.5*0.25
    image = _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).end(row,axis).sink().toposort()))
    assert image is not None
    if not dot:
      assert _cmac(image) is not None and not _constant_bytes(image) and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*4
      assert not _ew_ops(image)
    else: assert _cmac(image) is None and _ew_ops(image)


def test_nested_fp32_product_sum_is_committed_before_outer_half_add():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  bias = UOp.param(3, dtypes.half, (1,)).index(0).load()
  products = [(lhs.index(i).load() * rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  product_sum = products[0]
  for product in products[1:]: product_sum = product_sum + product
  image = _lower_uop_program(_program(dtypes.half, lambda _i:product_sum.cast(dtypes.half) + bias, count=1))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,4)
  assert _cmac(image).out_fp16 and len(_ew_ops(image)) == 1 and len(_intermediate_gathers(image)) == 3 and not _output_gathers(image)
  assert _intermediate_gathers(image)[0].src.kind is RKBufferKind.SCRATCH and _intermediate_gathers(image)[0].dst.kind is RKBufferKind.ARG
  assert _intermediate_gathers(image)[1].src.kind is RKBufferKind.ARG and _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG
  assert decode_image(encode_image(image)) == image


def test_cmac_storage_epilogue_does_not_clobber_an_inplace_input():
  out, lhs, rhs = UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(4,)),UOp.param(2,dtypes.half,(4,))
  products = [(lhs.index(i).load()*rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  reduced = functools.reduce(lambda total,term:total+term,products).cast(dtypes.half)
  image = _lower_uop_program(_program(dtypes.half,lambda _i:reduced+out.index(0).load(),count=1))
  assert image is None


def test_independent_fp32_reductions_are_committed_before_outer_half_division():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  products = [(lhs.index(i).load() * rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  weights = [rhs.index(i).load().cast(dtypes.float) for i in range(4)]
  numerator, denominator = products[0], weights[0]
  for product,weight in zip(products[1:], weights[1:]): numerator, denominator = numerator+product, denominator+weight
  ratio = UOp(Ops.FDIV, dtypes.half, src=(numerator.cast(dtypes.half), denominator.cast(dtypes.half)))
  image = _lower_uop_program(_program(dtypes.half, lambda _i:ratio, count=1))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,4)
  assert len(_ew_ops(image)) == 55 and _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_fp32_math_uop_converts_at_half_storage_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:source.index(i).load().cast(dtypes.float).exp2().cast(dtypes.half)))
  assert image is not None and len(_ew_ops(image)) > 10


def test_trunc_uop_expands_to_native_floor_and_ceil_stages():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:UOp(Ops.TRUNC, dtypes.half, src=(source.index(i).load(),))))
  assert image is not None and any(op.ew_cfg == _EW_CFG_FLOOR for op in _ew_ops(image))


def test_fp32_sin_additive_phase_reduces_terms_before_half_storage_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  def shifted_sin(i):
    value = source.index(i).load().cast(dtypes.float)
    phase = UOp.const(math.pi/2, dtypes.float) + value * UOp.const(-1.0, dtypes.float)
    return UOp(Ops.SIN, dtypes.float, src=(phase,)).cast(dtypes.half)
  image = _lower_uop_program(_program(dtypes.half, shifted_sin))
  assert image is not None and sum(op.ew_cfg == _EW_CFG_FLOOR for op in _ew_ops(image)) >= 2
  assert _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG and decode_image(encode_image(image)) == image


def test_fp32_storage_reuses_generic_algebra_after_nested_half_casts():
  source = UOp.param(1, dtypes.half, (1,)).index(UOp.const(0, dtypes.int)).load()
  exponent = source.cast(dtypes.float) * UOp.const(1/math.log(2), dtypes.float)
  exponential = UOp(Ops.EXP2, dtypes.float, src=(exponent,))
  denominator = UOp.const(1.0, dtypes.half) + exponential.cast(dtypes.half)
  inverse = UOp(Ops.FDIV, dtypes.half, src=(UOp.const(1.0, dtypes.half), denominator))
  correction = inverse + (UOp.const(1.0, dtypes.half) - inverse) / denominator * UOp.const(-1.0, dtypes.half)
  canonical = _canonical_half_storage(exponential * correction.cast(dtypes.float))
  assert canonical.op is Ops.ADD and sum(node.op is Ops.EXP2 for node in canonical.toposort()) == 1
  assert not any(node.dtype.scalar() is dtypes.float for node in canonical.toposort())
  image = _lower_uop_program(_program(dtypes.half, lambda _i:canonical, count=1))
  assert image is not None and _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_fp32_boundary_activation_routes_terminal_cmac_relu():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).maximum(UOp.const(0.0, dtypes.float)).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and _cmac(image) is not None and _cmac(image).relu and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,2,2)
  assert not _ew_ops(image) and decode_image(encode_image(image)) == image


def test_terminal_minimum_is_not_misclassified_as_cmac_relu():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane, zero = UOp.range(2, 0), UOp.const(0.0, dtypes.half)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  reduced = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  value = (reduced < zero).where(reduced, zero)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and _cmac(image) is not None and not _cmac(image).relu
  assert (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (2,2,2) and _ew_ops(image)[-1].dst.kind is RKBufferKind.ARG


def test_horizontal_reduces_are_structurally_executed():
  for op in (Ops.ADD,Ops.MAX,Ops.MUL):
    out,source=UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(3,))
    packed=UOp(Ops.STACK,dtypes.half,src=tuple(source.index(i).load() for i in range(3)))
    reduced=UOp(Ops.REDUCE,dtypes.half,src=(packed,),arg=(op,1))
    image=_lower_uop_program(list(out.index(0).store(expand_horizontal_reduce(reduced)).sink().toposort()))
    assert image is not None


def test_static_reduce_preserves_range_order_dependencies():
  """Range AFTER edges remain semantic inputs to direct reduction planning."""
  out = UOp.param(0, dtypes.half, (2,))
  dependency = UOp.range(4, 0, AxisType.WEAK)
  lane = UOp.range(2, 1, AxisType.WEAK, src=(dependency,))
  reduce_axis = UOp.range(3, 2, AxisType.REDUCE, src=(lane,))
  reduced=UOp(Ops.REDUCE,dtypes.half,src=(reduce_axis.cast(dtypes.half),reduce_axis),arg=(Ops.ADD,0))
  root=reduced+lane.cast(dtypes.half)*UOp.const(0.0,dtypes.half)
  uops=list(out.index(lane).store(root).end(lane,reduce_axis).sink().toposort())
  expanded = _unroll_static_reduces(root)
  assert dependency in expanded.toposort()
  assert rockchip_renderer._static_ranges(lane) == (lane,)
  assert any(node.key == lane.key and len(node.src) > 1 and node.src[1].key == dependency.key
             for node in expanded.toposort() if node.op is Ops.RANGE)
  assert _lower_uop_program(uops) is not None


def _boolean_reduce_program(source_dtype,op:Ops,groups:int=2,width:int=4):
  out,source=UOp.param(0,dtypes.bool,(groups,)),UOp.param(1,source_dtype,(groups*width,))
  group,axis=UOp.range(groups,1),UOp.range(width,0,AxisType.REDUCE)
  loaded=source.index(group*width+axis).load()
  present=loaded!=UOp.const(0.0,dtypes.half) if source_dtype is dtypes.half else loaded
  reduced=UOp(Ops.REDUCE,dtypes.bool,src=(present,axis),arg=(op,0))
  return list(out.index(group).store(reduced).end(group,axis).sink().toposort())


def test_boolean_reductions_are_physically_executed():
  for groups,source_dtype,op in itertools.product((1,2),(dtypes.half,dtypes.bool),(Ops.MUL,Ops.MAX)):
    for width in (1,2,4):
      image=_lower_uop_program(_boolean_reduce_program(source_dtype,op,groups=groups,width=width))
      assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False)
      values=([1]*width+[1]*(width-1)+[0]) if op is Ops.MUL else ([0]*width+[0]*(width-1)+[1])
      source=np.asarray(values[:groups*width],dtype=np.float16 if source_dtype is dtypes.half else np.uint8)
      assert _execute_raw_dynamic_image(image,groups,source.tobytes())==(bytes((1,0)) if op is Ops.MUL else bytes((0,1)))[:groups]
      assert decode_image(encode_image(image))==image


def test_boolean_arg_extrema_keeps_integer_suffix_together():
  renderer=RockchipRenderer(Target(device="ROCKCHIP"))
  for data,fxn,expected in (([False,True],Tensor.argmax,1),([True,False],Tensor.argmin,1)):
    with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
      source=Tensor(UOp.new_buffer("ROCKCHIP",2,dtypes.bool,num=12043))
      call=next(u for u in fxn(source).schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK)
      uops=list(call.src[0].toposort())
      output=rockchip_renderer._outs(uops)[1] or rockchip_renderer._outs(uops)[0]
      assert output is not None and rockchip_renderer._lower_mapped_reduce(output,uops) is None
      to_program_cache.clear()
      image=decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY))
    actual=np.frombuffer(_execute_raw_dynamic_image(image,4,np.asarray(data,dtype=np.uint8).tobytes()),dtype="<i4")
    np.testing.assert_array_equal(actual,np.asarray([expected],dtype="<i4"))


@pytest.mark.parametrize("kind",("argmin","argmax"))
@pytest.mark.parametrize("axis",(0,1))
def test_production_short_axis_arg_extrema_uses_bounded_mapped_max(kind:str,axis:int):
  shape=(10,20)
  rows=shape[1-axis]
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=12044)).reshape(shape)
    calls=[u for u in getattr(source,kind)(axis).schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    to_program_cache.clear()
    image=decode_image(next(u.arg for u in to_program(calls[1].src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY))
  assert _cmac(image) is None and len(_ew_ops(image))<100 and max(op.count for op in _ew_ops(image))>rows
  assert not _runtime_gathers(image) and decode_image(encode_image(image))==image
  rng=np.random.default_rng(12044+axis)
  values=rng.uniform(-4,4,size=shape).astype("<f2")
  tie=-5 if kind=="argmin" else 5
  if axis==0: values[2,:]=values[7,:]=tie
  else: values[:,3]=values[:,9]=tie
  best=((-values).max(axis=axis) if kind=="argmin" else values.max(axis=axis)).astype("<f2")
  actual=np.frombuffer(_execute_raw_dynamic_image(image,rows*4,values.tobytes(),best.tobytes()),dtype="<i4")
  np.testing.assert_array_equal(actual,getattr(np,kind)(values,axis=axis).astype("<i4"))


def test_production_large_all_uses_bounded_row_reduction():
  count=1 << 20
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    output=Tensor.ones(count,device="ROCKCHIP").bool().all()
    calls=[u for u in output.schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    to_program_cache.clear()
    images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
  assert len(images)==3 and len(_ew_ops(images[1]))<400
  assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for image in images)
  values=np.ones(count,dtype="<f2")
  values[:4]=np.asarray((np.nan,np.inf,-np.inf,np.nextafter(np.float16(0),np.float16(1))),dtype="<f2")
  values[[4095,4096,500000,count-1]]=np.float16(0)
  expected=np.all(values.reshape(256,4096)!=0,axis=1).astype(np.uint8)
  rows=_execute_raw_dynamic_image(images[1],256,values.tobytes())
  np.testing.assert_array_equal(np.frombuffer(rows,dtype=np.uint8),expected)
  assert _execute_raw_dynamic_image(images[2],1,rows)==bytes((int(expected.all()),))


def test_dependent_scalar_extrema_uses_direct_native_lowering():
  for extents in ((4,), (45,65)):
    count=math.prod(extents)
    out,source=UOp.param(0,dtypes.int,(1,)),UOp.param(1,dtypes.half,(count,))
    value_axes=tuple(UOp.range(extent,axis,AxisType.REDUCE) for axis,extent in enumerate(extents))
    index_axes=tuple(UOp.range(extent,axis+len(extents),AxisType.REDUCE) for axis,extent in enumerate(extents))
    def flatten(axes:tuple[UOp,...])->UOp: return functools.reduce(lambda value,item:value*item[1]+item[0],zip(axes,extents),UOp.const(0,dtypes.int))  # noqa: E501
    value_candidate=source.index(flatten(value_axes)).load()
    best=UOp(Ops.REDUCE,dtypes.half,src=(value_candidate,*value_axes),arg=(Ops.MAX,0))
    index_candidate=source.index(flatten(index_axes)).load()
    equal=(index_candidate!=best)!=UOp.const(True,dtypes.bool)
    coordinate=UOp.const(count,dtypes.int)-flatten(index_axes)
    selected=UOp(Ops.REDUCE,dtypes.int,src=(equal.cast(dtypes.int)*coordinate,*index_axes),arg=(Ops.MAX,0))
    output=out.index(0).store(UOp.const(count,dtypes.int)-selected)

    image=_lower_uop_program(list(output.sink().toposort()))
    assert image is not None and len(_ew_ops(image))<2*count+256 and not _runtime_gathers(image)
    assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and decode_image(encode_image(image))==image
    assert _assert_decoded_image_bounds(image)==image and _ew_ops(image)[-1].dst==RKArg(RKBufferKind.ARG,0)


def test_nested_static_reductions_materialize_load_addresses():
  count = 4
  out, source = UOp.param(0, dtypes.half, (count,)), UOp.param(1, dtypes.half, (count,))
  lane = UOp.range(count, 2)
  outer = UOp.range(count, 1, AxisType.REDUCE, src=(lane,))
  inner = UOp.range(count, 0, AxisType.REDUCE, src=(outer,))
  inner_term = ((inner+outer < UOp.const(count-1, dtypes.int)) != UOp.const(True, dtypes.bool)).cast(dtypes.int)
  inner_result=UOp(Ops.REDUCE,dtypes.int,src=(inner_term,inner),arg=(Ops.ADD,0))
  outer_term = (inner_result != outer+lane+UOp.const(1-count, dtypes.int)).where(0, 1)
  dynamic_index=UOp(Ops.REDUCE,dtypes.int,src=(outer_term,outer),arg=(Ops.ADD,0))
  normalized = (dynamic_index < 0).where(dynamic_index+count, dynamic_index)
  gate = ((normalized < 0) != UOp.const(True, dtypes.bool)) & (normalized < count)
  output = out.index(lane).store(source.index(normalized).load(UOp.const(0.0, dtypes.half), gate)).end(lane)

  image = _lower_uop_program(list(output.sink().toposort()))
  assert image is not None and _cmac(image) is not None and _initial_gathers(image)[0].offsets[::32] == (0, 0, 0, 0)
  assert not _runtime_gathers(image,False) and decode_image(encode_image(image)) == image


def test_static_reduce_preserves_sequential_fp16_updates():
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (2,))
  axis = UOp.range(3, 0, AxisType.REDUCE)
  term = (axis < 1).where(UOp.const(2048.0, dtypes.half),
                         (axis < 2).where(UOp.const(1.0, dtypes.half), UOp.const(-2048.0, dtypes.half)))
  index=UOp(Ops.REDUCE,dtypes.half,src=(term,axis),arg=(Ops.ADD,0)).cast(dtypes.int)
  store = out.index(0).store(source.index(index).load())
  uops = list(store.sink().toposort())
  assert (image:=_lower_uop_program(uops)) is not None and _initial_gathers(image)[0].offsets[0] == 0
  assert decode_image(encode_image(image)) == image


def test_multiple_fp16_reduces_preserve_sequential_updates():
  out, lane = UOp.param(0, dtypes.half, (2,)), UOp.range(2, 3)
  def reduction(axis_id:int) -> UOp:
    axis = UOp.range(3, axis_id, AxisType.REDUCE)
    term = (axis < 1).where(UOp.const(2048.0, dtypes.half), (axis < 2).where(1.0, -2048.0))
    return UOp(Ops.REDUCE,dtypes.half,src=(term,axis),arg=(Ops.ADD,0))
  first,second=reduction(0),reduction(1)
  image = _lower_uop_program(list(out.index(lane).store(first+second).sink().toposort()))
  assert image is not None and not _runtime_gathers(image)
  assert _constant_bytes(image)==struct.pack("<e",0.0) and len(_ew_ops(image))==1
  assert not _initial_gathers(image) and not _intermediate_gathers(image)
  assert decode_image(encode_image(image)) == image


def test_static_reduce_preflights_reducer_product():
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (2,))
  outer = UOp.range(16384, 0, AxisType.REDUCE)
  inner = UOp.range(4096, 1, AxisType.REDUCE, src=(outer,))
  index=UOp(Ops.REDUCE,dtypes.int,src=(UOp.const(1,dtypes.int),outer,inner),arg=(Ops.ADD,0))
  store = out.index(0).store(source.index(index).load())
  assert _lower_uop_program(list(store.sink().toposort())) is None


def test_packed_bool_load_uses_canonical_int16_lanes():
  out, mask = UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.bool, (4,))
  lane = UOp.range(4, 0)
  image = _lower_uop_program(list(out.index(lane).store(mask.index(lane).load().cast(dtypes.int)+1).end(lane).sink().toposort()))
  assert image is not None and len(_initial_gathers(image)) == 1
  assert _initial_gathers(image)[0].itemsize == 1 and _initial_gathers(image)[0].dst_stride == 2
  assert _ew_ops(image)[-1].mode==RKEWMode.INT16_TO_INT32


def test_fp16_predicate_prefix_executes_generic_uops():
  source = UOp.param(1, dtypes.half, (4,))
  def prefix(lane:UOp) -> UOp:
    terms = []
    for source_lane in range(4):
      predicate = UOp(Ops.CMPLT, dtypes.bool, src=(UOp.const(0.0, dtypes.half), source.index(source_lane).load()))
      active = UOp(Ops.CMPLT, dtypes.bool, src=(UOp.const(source_lane, dtypes.int), lane+1))
      terms.append(active.where(predicate.cast(dtypes.int), UOp.const(0, dtypes.int)))
    value = terms[0]
    for term in terms[1:]: value = value+term
    return value
  image = _lower_uop_program(_program(dtypes.int, prefix))
  assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)
  assert any(op.mode in _INT32_OUTPUT_MODES for op in _ew_ops(image)) and decode_image(encode_image(image)) == image


def test_fixed_nonzero_rank_two_static_images_preserve_coordinate_matrix_bounds():
  source = Tensor([[1, 0], [0, 2]], device="ROCKCHIP")
  result = source.nonzero(size=2)
  linear, var_vals = result.linear_with_vars()
  assert not var_vals
  renderer = RockchipRenderer(Target.parse("ROCKCHIP"))
  images = [decode_image(to_program(call.src[0], renderer).src[-1].arg) for call in linear.src if call.src[0].op is Ops.SINK]
  assert len(images) == 4
  for image in images:
    assert not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)
    assert decode_image(encode_image(image)) == image
    for gather in _static_gathers(image):
      if gather.dst.kind is RKBufferKind.SCRATCH:
        assert gather.dst_addend+(gather.count-1)*gather.dst_stride < image.scratch[gather.dst.index]//gather.itemsize

  coordinate = images[-1]
  assert (len(coordinate.scratch),len(_static_gathers(coordinate)),len(_ew_ops(coordinate)),len(_output_gathers(coordinate))) == (135,148,7861,1)
  lanes=np.arange(4,dtype="<i4").tobytes()
  assert _execute_raw_dynamic_image(coordinate,16,lanes,lanes) == bytes.fromhex("00000000000000000000000001000000")
  assert hashlib.sha256(encode_image(coordinate)).hexdigest()=="35e50a800397d5fccee67c44a1b284f7ed977f1c1499c86d43072f88cc1e7f63"
  np.testing.assert_array_equal(_execute_integer_image(coordinate, np.asarray([1, 0, 0, 2], dtype=np.int32),
                                                       np.asarray([0, 1, 6, 7], dtype=np.int32)),
                                np.asarray([0, 0, 1, 1], dtype=np.int32))

  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",320,dtypes.half,num=12042).reshape((32,10)))
    large_linear=(source>0.5).nonzero(size=120).schedule_linear()
    large_calls=[u for u in large_linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    to_program_cache.clear()
    large_images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in large_calls]
  assert len(large_images)==5
  prefix,mapped_coordinate,large_coordinate=large_images[0],large_images[2],large_images[-1]
  assert (len(prefix.scratch),len(_static_gathers(prefix)),len(_ew_ops(prefix)),len(_output_gathers(prefix))) == (27,16,1099,0)
  assert (len(mapped_coordinate.scratch),len(_static_gathers(mapped_coordinate)),len(_ew_ops(mapped_coordinate)),
          len(_output_gathers(mapped_coordinate))) == (51,52,141,0)
  assert (len(large_coordinate.scratch),len(_static_gathers(large_coordinate)),len(_ew_ops(large_coordinate)),
          len(_output_gathers(large_coordinate))) == (153,147,10451,1)
  values=np.random.default_rng(1001).uniform(-1,1,size=320).astype("<f2")
  assert hashlib.sha256(_execute_raw_dynamic_image(prefix,768*4,values.tobytes())).hexdigest() == \
    "60fbd91c95bad020944066ff92feac468dc5f4713dcf489e24df1988dff7f79d"
  indices=((np.arange(240,dtype=np.int64)*7+3)%17).astype("<i4")
  assert hashlib.sha256(_execute_raw_dynamic_image(mapped_coordinate,240*4,
    ((np.arange(768,dtype=np.int64)*7+3)%17).astype("<i4").tobytes(),((np.arange(3,dtype=np.int64)*7+3)%17).astype("<i4").tobytes())).hexdigest() == \
    "49c0396708d050e2d1f4f99a07a824d7c8eb8e287373fff0770354c6a415f5e7"
  assert hashlib.sha256(_execute_raw_dynamic_image(large_coordinate,240*4,
    np.random.default_rng(1005).uniform(-1,1,size=320).astype("<f2").tobytes(),indices.tobytes())).hexdigest() == \
    "5eb5c4574e3ce639b7e9920d1d6698b5bf2d9110ceab18833a1a76296f5fcd22"


def test_int32_selector_images_preserve_full_width_values_and_bounds():
  count,candidates=5,8
  out,source=UOp.param(0,dtypes.int,(count,)),UOp.param(1,dtypes.int,(candidates,))
  lane,axis=UOp.range(count,0),UOp.range(candidates,1,AxisType.REDUCE)
  different=UOp(Ops.CMPNE,dtypes.bool,src=(source.index(axis).load(),lane.cast(dtypes.int)))
  term=different.where(UOp.const(0,dtypes.int),UOp.const(1,dtypes.int))
  reduced=UOp(Ops.REDUCE,dtypes.int,src=(term,axis),arg=(Ops.ADD,0))
  histogram=_lower_uop_program(list(out.index(lane).store(reduced).end(lane,axis).sink().toposort()))
  assert histogram is not None and _ew_ops(histogram)[-1].mode is RKEWMode.INT16_TO_INT32
  values=np.asarray((-2**31,-1,0,1,1,4,2**31-1,5),dtype="<i4")
  np.testing.assert_array_equal(np.frombuffer(_execute_raw_dynamic_image(histogram,count*4,values.tobytes()),dtype="<i4"),
                                np.asarray((1,2,0,0,1),dtype="<i4"))

  count,limit=8,4
  out,source=UOp.param(0,dtypes.int,(count,)),UOp.param(1,dtypes.int,(count,))
  lane=UOp.range(count,0)
  loaded=source.index(lane).load()
  negative=UOp(Ops.CMPLT,dtypes.bool,src=(loaded,loaded.const_like(0)))
  nonnegative=UOp(Ops.CMPNE,dtypes.bool,src=(negative,UOp.const(True,dtypes.bool)))
  upper=UOp(Ops.CMPLT,dtypes.bool,src=(loaded,loaded.const_like(limit)))
  gate=UOp(Ops.AND,dtypes.bool,src=(upper,nonnegative))
  selected=gate.where(loaded*10+lane.cast(dtypes.int),loaded.const_like(0))
  lookup=_lower_uop_program(list(out.index(lane).store(selected).end(lane).sink().toposort()))
  assert lookup is not None and _ew_ops(lookup)[-1].mode is RKEWMode.INT16_TO_INT32
  values=np.asarray((-2**31,-1,0,1,3,4,2**31-1,2),dtype="<i4")
  np.testing.assert_array_equal(np.frombuffer(_execute_raw_dynamic_image(lookup,count*4,values.tobytes()),dtype="<i4"),
                                np.asarray((0,0,2,13,34,0,0,27),dtype="<i4"))


def test_normalized_int_prefix_executes_generic_int32_uops():
  source = UOp.param(1, dtypes.int, (4,))
  def prefix(lane:UOp) -> UOp:
    terms = []
    for source_lane in range(4):
      active = UOp(Ops.CMPLT, dtypes.bool, src=(UOp.const(source_lane, dtypes.int), lane+1))
      terms.append(source.index(source_lane).load(UOp.const(0, dtypes.int), active))
    value = terms[0]
    for term in terms[1:]: value = value+term
    return (value < 0).where(value+4, value)
  image = _lower_uop_program(_program(dtypes.int, prefix))
  assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False) and not _runtime_gathers(image,True)
  assert any(op.mode in _INT32_OUTPUT_MODES for op in _ew_ops(image)) and decode_image(encode_image(image)) == image


def test_direct_dynamic_int32_load_selects_all_raw_bytes():
  out, source, indices = (UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.int, (9,)), UOp.param(2, dtypes.int, (4,)))
  lane = UOp.range(4, 0)
  index = indices.index(lane).load()
  gate = ((index < 0) != UOp.const(True, dtypes.bool)) & (index < 9)
  load = source.index(index).load(UOp.const(0, dtypes.int), gate)
  image = _lower_uop_program(list(out.index(lane).store(load).end(lane).sink().toposort()))
  assert image is not None and sum(gather.itemsize for gather in _output_gathers(image)) == 4
  assert decode_image(encode_image(image)) == image


def test_dynamic_address_gather_preserves_plain_raw_payloads():
  indices = np.asarray((0, 1, 2, 8), dtype="<i4")
  sources = {
    dtypes.half:np.asarray((0x0000, 0x8000, 0x7e01, 0x7fff, 0x7c01, 0xfc01, 0x3555, 0xbc00, 0x0400), dtype="<u2"),
    dtypes.int16:np.asarray((0x0000, 0x8000, 0xffff, 0x7fff, 0x00ff, 0xff00, 0x5555, 0xaaaa, 0x1234), dtype="<u2"),
    dtypes.int:np.asarray((0, 0x80000000, 0x7fc01234, 0xffffffff, 0x7fffffff, 0x00800000,
                           0xff800001, 0x55aa55aa, 0x12345678), dtype="<u4"),
  }
  for dtype,source in sources.items():
    image = _lower_uop_program(_dynamic_load_program(dtype=dtype))
    assert image is not None and _address_gather(image).index is not None and _address_gather(image).index.kind is RKBufferKind.SCRATCH
    assert _execute_raw_dynamic_image(image, 4*dtype.itemsize, source.tobytes(), indices.tobytes()) == source[indices].tobytes()
    assert decode_image(encode_image(image)) == image


def test_dynamic_npu_address_normalizes_negative_indices_exactly():
  source = np.asarray((0x0000, 0x8000, 0x7e01, 0x7fff, 0x7c01, 0xfc01, 0x3555, 0xbc00, 0x0400), dtype="<u2")
  indices = np.asarray((-1, -9, -5, -10), dtype="<i4")
  image = _lower_uop_program(_dynamic_load_program(normalized=True))
  assert image is not None
  gather=_address_gather(image)
  assert gather.index is not None and gather.index.kind is RKBufferKind.SCRATCH and _gather_point(image,gather)>0
  expected = np.asarray((source[8], source[0], source[4], 0), dtype="<u2")
  assert _execute_raw_dynamic_image(image, 8, source.tobytes(), indices.tobytes()) == expected.tobytes()


def test_production_multi_fancy_one_hot_uses_one_runtime_gather():
  def symbolic(shape, dtype, num): return Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtype,num=num)).reshape(shape)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=symbolic((2,5,6,5,3,4),dtypes.half,12500)
    i=symbolic((2,1,1,1,1,1),dtypes.int,12501)
    j=symbolic((1,3,1,1,1,1),dtypes.int,12502)
    o=symbolic((2,1,1,5,1,1),dtypes.int,12503)
    p=symbolic((1,1,1,1,6,1),dtypes.int,12504)
    call=next(u for u in source[i,j,None,o,p].schedule_linear().toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK)
    image=decode_image(next(u.arg for u in to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY))
  assert len(_runtime_gathers(image,False))==1 and len(_ew_ops(image))<600 and decode_image(encode_image(image))==image
  raw=np.arange(3600,dtype=np.uint16).reshape(2,5,6,5,3,4)
  ii=np.asarray((-1,0),dtype="<i4").reshape(2,1,1,1,1,1)
  jj=np.asarray((-5,-2,4),dtype="<i4").reshape(1,3,1,1,1,1)
  oo=np.resize(np.asarray((-5,-1,0,2,4),dtype="<i4"),10).reshape(2,1,1,5,1,1)
  pp=np.asarray((-3,-1,0,1,2,-2),dtype="<i4").reshape(1,1,1,1,6,1)
  expected=raw[ii,jj,None,oo,pp]
  got=np.frombuffer(_execute_raw_dynamic_image(image,expected.nbytes,ii.tobytes(),jj.tobytes(),oo.tobytes(),pp.tobytes(),raw.tobytes()),dtype=np.uint16).reshape(expected.shape)  # noqa: E501
  np.testing.assert_array_equal(got,expected)


def test_affine_gather_bounds_reject_negative_low_but_keep_offset_sentinel():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    for count in (1, 2):
      invalid = RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),count,base=-1,axes=((1,count,1),),itemsize=dtype.itemsize)
      try: rockchip_renderer._validate_gather_bounds(invalid, count)
      except RuntimeError: pass
      else: raise AssertionError(f"negative affine low admitted for {dtype} lane{count}")
      rockchip_renderer._validate_gather_bounds(RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),count,axes=((1,count,1),),itemsize=dtype.itemsize),count)
      rockchip_renderer._validate_gather_bounds(RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),count,offsets=(-1,)+(0,)*(count-1)),1)
      try: rockchip_renderer._validate_gather_bounds(RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),
        count,offsets=(-2,0)[:count]),count)
      except RuntimeError: pass
      else: raise AssertionError(f"offset below sentinel admitted for {dtype} lane{count}")
      if dtype is not dtypes.bool:
        assert _lower_uop_program(_dynamic_load_program(count=count, dtype=dtype, normalized=True)) is not None


def test_scalar_gather_bounds_reject_negative_low_for_all_typed_lanes():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    try: rockchip_renderer._validate_gather_bounds(RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),1,
      base=-1,itemsize=dtype.itemsize),1)
    except RuntimeError: pass
    else: raise AssertionError(f"negative scalar low admitted for {dtype}")
    rockchip_renderer._validate_gather_bounds(RKGather(RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.SCRATCH,0),1,itemsize=dtype.itemsize),1)


def test_gather_offsets_reject_true_gate_negative_and_allow_false_sentinel():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    out, source, lane = UOp.param(0, dtype, (4,)), UOp.param(1, dtype, (4,)), UOp.range(4, 0)
    default = UOp.const(0.0, dtype) if dtype is dtypes.half else UOp.const(0, dtype)
    counterexample = list(out.index(lane).store(source.index(lane-1).load(default, lane < 4)).end(lane).sink().toposort())
    assert _lower_uop_program(counterexample) is None
    for gate in (lane < 0, lane > 0):
      valid = list(out.index(lane).store(source.index(lane-1).load(default, gate)).end(lane).sink().toposort())
      image = _lower_uop_program(valid)
      assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False)


def test_gather_offsets_normalize_inactive_raw_negative_to_fill():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    out, source, lane = UOp.param(0, dtype, (4,)), UOp.param(1, dtype, (4,)), UOp.range(4, 0)
    default = UOp.const(0.0, dtype) if dtype is dtypes.half else UOp.const(0, dtype)
    padded = list(out.index(lane).store(source.index(lane-31).load(default, lane < 0)).end(lane).sink().toposort())
    image = _lower_uop_program(padded)
    assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False)
    assert any(gather.offsets==(-1,-1,-1,-1) for gather in _static_gathers(image))


def test_dynamic_npu_address_composes_multiple_axes_and_external_gate():
  source = (np.arange(81, dtype=np.uint32)*257+0x8000).astype("<u2")
  first, second = np.asarray((-1, 0, 2, -10), dtype="<i4"), np.asarray((0, -1, 3, 4), dtype="<i4")
  gate = np.asarray((1, 1, 0, 1), dtype=np.uint8)
  image = _lower_uop_program(_dynamic_load_program(extents=(9, 9), normalized=True, external_gate=True))
  assert image is not None
  gather=_address_gather(image)
  assert gather.index is not None and gather.index.kind is RKBufferKind.SCRATCH and _gather_point(image,gather)>0
  expected = np.asarray((source[72], source[8], 0, 0), dtype="<u2")
  assert _execute_raw_dynamic_image(image, 8, source.tobytes(), first.tobytes(), second.tobytes(), gate.tobytes()) == expected.tobytes()
  assert decode_image(encode_image(image)) == image


def test_dynamic_address_keeps_non_address_predicates_on_npu():
  out,source,indices=UOp.param(0,dtypes.half,(4,)),UOp.param(1,dtypes.half,(9,)),UOp.param(2,dtypes.int,(4,))
  lane=UOp.range(4,0)
  dynamic=indices.index(lane).load()
  bounds=((dynamic < 0) != UOp.const(True,dtypes.bool)) & (dynamic < 3)
  values=np.asarray((0x100,0x101,0x102,0x103,0x104,0x105,0x106,0x107,0x108),dtype="<u2")
  cases=((dynamic,bounds & (lane < 2),np.asarray((0,1,2,2),dtype="<i4"),np.asarray((values[0],values[1],0,0),dtype="<u2")),
         (dynamic*dynamic,bounds,np.asarray((-1,0,2,3),dtype="<i4"),np.asarray((0,values[0],values[4],0),dtype="<u2")))
  for index,gate,raw_indices,expected in cases:
    load=source.index(index).load(UOp.const(0.0,dtypes.half),gate)
    image=_lower_uop_program(list(out.index(lane).store(load).end(lane).sink().toposort()))
    assert image is not None
    gather=_address_gather(image)
    assert gather.index is not None and gather.index.kind is RKBufferKind.SCRATCH and _gather_point(image,gather)>0
    assert _execute_raw_dynamic_image(image,8,values.tobytes(),raw_indices.tobytes())==expected.tobytes()


def test_dynamic_npu_address_repeats_raw_channels():
  source = (np.arange(21, dtype=np.uint32)*131+0x8000).astype("<u2")
  indices, gate = np.asarray((6, 0, 3, 7), dtype="<i4"), np.asarray((1, 0, 1, 1), dtype=np.uint8)
  image = _lower_uop_program(_dynamic_load_program(count=12, extents=(7,), external_gate=True, repeat=3))
  assert image is not None
  gather=_address_gather(image)
  assert gather.index is not None and gather.index.kind is RKBufferKind.SCRATCH and _gather_point(image,gather)>0
  expected = np.zeros(12, dtype="<u2")
  expected[:3], expected[6:9] = source[18:21], source[9:12]
  assert _execute_raw_dynamic_image(image, 24, source.tobytes(), indices.tobytes(), gate.tobytes()) == expected.tobytes()


def test_dynamic_address_avoids_1001_candidate_expansion():
  source = (np.arange(1001, dtype=np.uint32)+0x8000).astype("<u2")
  image = _lower_uop_program(_dynamic_load_program(count=64, extents=(1001,)))
  assert image is not None and _address_gather(image).index is not None and _address_gather(image).index.kind is RKBufferKind.SCRATCH and len(_ew_ops(image)) < 200  # noqa: E501
  for start in range(0, 1024, 64):
    indices = np.arange(start, start+64, dtype="<i4")
    expected = np.zeros(64, dtype="<u2")
    valid = indices < len(source)
    expected[valid] = source[indices[valid]]
    assert _execute_raw_dynamic_image(image, 128, source.tobytes(), indices.tobytes()) == expected.tobytes()
  assert decode_image(encode_image(image)) == image


def test_dynamic_address_has_no_candidate_domain_allocation():
  cases = ((1, rockchip_renderer._MAX_STATIC_RANGE_ENVS+1),
           (4096, rockchip_renderer._MAX_DYNAMIC_SELECTOR_CELLS//4096+1))
  for count,extent in cases:
    image=_lower_uop_program(_dynamic_load_program(count=count, extents=(extent,)))
    assert image is not None and _address_gather(image).index is not None and _address_gather(image).index.kind is RKBufferKind.SCRATCH and len(encode_image(image)) < 1 << 20  # noqa: E501


def test_dynamic_address_rejects_unencodable_slots_and_offsets():
  out, source, indices = UOp.param(0, dtypes.half, (1,)), UOp.param(rockchip_renderer._RKIMAGE_U16_MAX+1, dtypes.half, (1,)), \
    UOp.param(2, dtypes.int, (1,))
  lane = UOp.range(1, 0)
  dynamic = indices.index(lane).load()
  gate = ((dynamic < 0) != UOp.const(True, dtypes.bool)) & (dynamic < 1)
  program = list(out.index(lane).store(source.index(dynamic).load(UOp.const(0.0, dtypes.half), gate)).end(lane).sink().toposort())
  assert _lower_uop_program(program) is None
  safe, unsafe = (1 << 29)-1, 1 << 29
  for program in (_dynamic_offset_program(data_offset=safe), _dynamic_offset_program(index_offset=safe)):
    assert (image:=_lower_uop_program(program)) is not None
    assert decode_image(encode_image(image)) == image
  for program in (_dynamic_offset_program(data_offset=unsafe), _dynamic_offset_program(index_offset=unsafe)):
    assert _lower_uop_program(program) is None


def test_dynamic_npu_address_composes_exact_bool_total_fill_gate():
  indices, mask = np.asarray((4, 1, -1, 8), dtype="<i4"), np.asarray((1, 0, 1, 0, 0), dtype=np.uint8)
  for dtype,source in ((dtypes.int16, np.asarray((0x8000, 0xffff, 0x7fff, 0x1234, 0xabcd), dtype="<u2")),
                       (dtypes.int, np.asarray((0x80000000, 0xffffffff, 0x7fffffff, 0x12345678, 0xabcdef01), dtype="<u4"))):
    image = _lower_uop_program(_dynamic_total_load_program(dtype))
    assert image is not None
    gather=_address_gather(image)
    assert gather.index is not None and any(op.mode in _TYPED_EW_MODES for op in _ew_ops(image))
    expected = np.asarray((source[4], source[1], (1 << (dtype.itemsize*8))-7, (1 << (dtype.itemsize*8))-7), dtype=source.dtype)
    assert _execute_raw_dynamic_image(image, 4*dtype.itemsize, source.tobytes(), indices.tobytes(), mask.tobytes()) == expected.tobytes()
    assert decode_image(encode_image(image)) == image


def test_bounded_int32_lookup_executes_as_exact_selector_uops():
  out, indices = UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.int, (4,))
  lane = UOp.range(4, 0)
  index = indices.index(lane).load()
  valid = ((index < 0) != UOp.const(True, dtypes.bool)) & (index < 5)
  value = valid.where(index+lane*4, UOp.const(0, dtypes.int))
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and not _runtime_gathers(image) and not _runtime_gathers(image,False)
  assert _ew_ops(image)[-1].mode==RKEWMode.INT16_TO_INT32 and not any(op.mode==RKEWMode.INT32 for op in _ew_ops(image))
  assert decode_image(encode_image(image)) == image
  np.testing.assert_array_equal(_execute_integer_image(image, np.asarray((-1, 0, 2, 6), dtype=np.int32)),
                                np.asarray((0, 4, 10, 0), dtype=np.int32))


def test_int32_bitwise_uop_executes_over_raw_byte_planes():
  rng = np.random.default_rng(0x2608)
  samples = [rng.integers(-(1<<31), 1<<31, 64, dtype=np.int64).astype(np.int32) for _ in range(3)]
  edges = np.asarray((-(1<<31), (1<<31)-1, -1, 0, 1, -1431655766, 1431655765, 0x00ff00ff), dtype=np.int32)
  for index in range(3): samples[index][:len(edges)] = np.roll(edges, index)
  functions = {Ops.AND:np.bitwise_and, Ops.OR:np.bitwise_or, Ops.XOR:np.bitwise_xor}
  for op,fn in functions.items():
    out, lhs, rhs, third = (UOp.param(slot, dtypes.int, (len(samples[0]),)) for slot in range(4))
    lane = UOp.range(len(samples[0]), 0)
    direct = UOp(op, dtypes.int, src=(lhs.index(lane).load(), rhs.index(lane).load()))
    for value,inputs,expected in ((direct, samples[:2], fn(samples[0], samples[1])),
      (UOp(op, dtypes.int, src=(direct, third.index(lane).load())), samples, fn(fn(samples[0], samples[1]), samples[2]))):
      image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
      assert image is not None and not _runtime_gathers(image)
      assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and all(x.mode==RKEWMode.INT16 for x in _ew_ops(image))
      assert decode_image(encode_image(image)) == image
      np.testing.assert_array_equal(_execute_integer_image(image, *inputs), expected)
    for constant in (0, 1, -1, 0x00ff00ff, -1431655766):
      image = _lower_uop_program(_int32_binary_program(
        lambda left,_right,op=op,constant=constant:UOp(op, dtypes.int, src=(left, UOp.const(constant, dtypes.int))), len(samples[0])))
      assert image is not None and decode_image(encode_image(image)) == image
      np.testing.assert_array_equal(_execute_integer_image(image, samples[0]), fn(samples[0], np.int32(constant)))
  maximum = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:lhs & rhs, _MAX_EW_ELEMS_FP16//4))
  assert maximum is not None and decode_image(encode_image(maximum)) == maximum
  assert _lower_uop_program(_int32_binary_program(lambda lhs,rhs:lhs & rhs, _MAX_EW_ELEMS_FP16//4+1)) is None


def test_wide_int32_cdiv_cmod_physical_semantics_and_composition():
  lhs, rhs = _int32_division_samples()
  expressions = (
    lambda left,right:UOp(Ops.CDIV, dtypes.int, src=(left, right)),
    lambda left,right:UOp(Ops.CMOD, dtypes.int, src=(left, right)),
    lambda left,right:UOp(Ops.CDIV, dtypes.int, src=(left+1, right*3)),
  )
  for select,expression in enumerate(expressions):
    image = _lower_uop_program(_int32_binary_program(expression, len(lhs)))
    assert image is not None and not _runtime_gathers(image)
    assert len(_ew_ops(image)) == (2545,2499,2547)[select] and decode_image(encode_image(image)) == image
    expected = []
    for left,right in zip(lhs.tolist(), rhs.tolist()):
      if select == 2: left, right = _wrap_int32(left+1), _wrap_int32(right*3)
      expected.append(_trunc_divmod_int32(left, right)[select == 1])
    np.testing.assert_array_equal(_execute_integer_image(image, lhs, rhs), np.asarray(expected, dtype=np.int32))


def test_sibling_int32_cdiv_cmod_share_one_restoring_core():
  direct = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs))))
  combined = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:
    UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs)) + UOp(Ops.CMOD, dtypes.int, src=(lhs, rhs))))
  assert direct is not None and combined is not None
  assert len(_ew_ops(direct)) < len(_ew_ops(combined)) < len(_ew_ops(direct))+100
  assert decode_image(encode_image(combined)) == combined


def test_int32_floor_division_and_modulo_execute_ordinary_uops():
  lhs_values, rhs_values = _int32_division_samples()
  zero = UOp.const(0, dtypes.int)
  def expressions(lhs:UOp, rhs:UOp) -> tuple[UOp, UOp]:
    quotient, remainder = UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs)), UOp(Ops.CMOD, dtypes.int, src=(lhs, rhs))
    correction = (remainder != zero) & ((lhs < zero) != (rhs < zero))
    return quotient + correction.cast(dtypes.int)*-1, remainder + correction.where(rhs, zero)
  for select in (0, 1):
    image = _lower_uop_program(_int32_binary_program(lambda lhs,rhs,select=select:expressions(lhs, rhs)[select], len(lhs_values)))
    assert image is not None and len(_ew_ops(image)) < 4000 and decode_image(encode_image(image)) == image
    expected = []
    for lhs,rhs in zip(lhs_values.tolist(), rhs_values.tolist()):
      quotient, remainder = _trunc_divmod_int32(lhs, rhs)
      correction = remainder != 0 and (lhs < 0) != (rhs < 0)
      expected.append(_wrap_int32(quotient-int(correction) if select == 0 else remainder+(rhs if correction else 0)))
    np.testing.assert_array_equal(_execute_integer_image(image, lhs_values, rhs_values), np.asarray(expected, dtype=np.int32))


def test_embedded_int32_not_preserves_all_raw_bytes_before_wide_arithmetic():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    UOp(Ops.XOR, dtypes.int, src=(source.index(i).load(), UOp.const(-1, dtypes.int)))+1))
  assert image is not None and len(_ew_ops(image)) == 6 and len(_intermediate_gathers(image)) == 8
  assert sum(op.mode==RKEWMode.INT32 for op in _ew_ops(image)) == 2


def test_int32_shift_uops_compose_over_signed_and_unsigned_raw_bytes():
  values = np.asarray((-(1<<31), -7, -1, 0, 1, 7, (1<<31)-1, 0x55aa55aa), dtype=np.int32)
  shifts = np.asarray((0, 7, 8, 15, 16, 31, 32, 2), dtype=np.int32)
  marker = np.int32(0x13579bdf)
  def expected(op:Ops, dtype, amount, base=values):
    amount = np.asarray(amount, dtype=np.uint32)&31
    if op is Ops.SHL: return (base.view(np.uint32).astype(np.uint64) << amount).astype(np.uint32).view(np.int32)
    return (base.view(np.uint32) >> amount).view(np.int32) if dtype is dtypes.uint else base >> amount
  for dtype in (dtypes.int, dtypes.uint):
    physical = values if dtype is dtypes.int else values.view(np.uint32)
    for op in (Ops.SHL, Ops.SHR):
      for amount in (0, 7, 8, 15, 16, 31, 32):
        out, source = UOp.param(0, dtypes.int, (len(values),)), UOp.param(1, dtype, (len(values),))
        lane = UOp.range(len(values), 0)
        shifted = UOp(op, dtype, src=(source.index(lane).load(), UOp.const(amount, dtype)))
        result = shifted.cast(dtypes.int) if dtype is dtypes.uint else shifted
        image = _lower_uop_program(list(out.index(lane).store(result).end(lane).sink().toposort()))
        assert image is not None and not _runtime_gathers(image)
        assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and decode_image(encode_image(image)) == image
        np.testing.assert_array_equal(_execute_integer_image(image, physical), expected(op, dtype, amount))
      out, source, amount = (UOp.param(slot, dtype if slot else dtypes.int, (len(values),)) for slot in range(3))
      lane = UOp.range(len(values), 0)
      shifted = UOp(op, dtype, src=(source.index(lane).load(), amount.index(lane).load()))
      shifted = shifted.cast(dtypes.int) if dtype is dtypes.uint else shifted
      nested = UOp(Ops.XOR, dtypes.int, src=(shifted, UOp.const(int(marker), dtypes.int)))
      image = _lower_uop_program(list(out.index(lane).store(nested).end(lane).sink().toposort()))
      assert image is not None and not _runtime_gathers(image)
      assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and decode_image(encode_image(image)) == image
      np.testing.assert_array_equal(_execute_integer_image(image, physical, shifts if dtype is dtypes.int else shifts.view(np.uint32)),
                                    np.bitwise_xor(expected(op, dtype, shifts), marker))
      inner = UOp(Ops.SHL, dtype, src=(source.index(lane).load(), UOp.const(1, dtype)))
      shifted = UOp(op, dtype, src=(inner, amount.index(lane).load()))
      result = shifted.cast(dtypes.int) if dtype is dtypes.uint else shifted
      image = _lower_uop_program(list(out.index(lane).store(result).end(lane).sink().toposort()))
      assert image is not None and not _runtime_gathers(image)
      assert not _runtime_gathers(image,False) and not _runtime_gathers(image,True) and decode_image(encode_image(image)) == image
      inner_expected = expected(Ops.SHL, dtype, 1)
      np.testing.assert_array_equal(_execute_integer_image(image, physical, shifts if dtype is dtypes.int else shifts.view(np.uint32)),
                                    expected(op, dtype, shifts, inner_expected))


def test_cmod_range_keeps_expanded_parity_arithmetic_in_exact_fp16_lanes():
  source = UOp.param(1, dtypes.half, (4,))
  def parity(i):
    value = source.index(i).load().cast(dtypes.int)
    remainder = UOp(Ops.CMOD, dtypes.int, src=(value, UOp.const(2, dtypes.int)))
    negative = UOp(Ops.CMPLT, dtypes.bool, src=(remainder, UOp.const(0, dtypes.int)))
    return remainder + negative.where(UOp.const(2, dtypes.int), UOp.const(0, dtypes.int))
  image = _lower_uop_program(_program(dtypes.int, parity))
  assert image is not None and any(op.mode in _INT32_OUTPUT_MODES for op in _ew_ops(image))


def test_dependent_reduction_range_preserves_vector_output_axis():
  def lower(rows:int, depth:int=65):
    out = UOp.param(0, dtypes.half, (rows,))
    lhs, rhs = UOp.param(1, dtypes.half, (rows*depth,)), UOp.param(2, dtypes.half, (depth,))
    row = UOp.range(rows, 1)
    axis = UOp.range(depth, 0, AxisType.REDUCE, src=(row,))
    product = lhs.index(row*depth+axis).load() * rhs.index(axis).load()
    reduced=UOp(Ops.REDUCE,dtypes.float,src=(product.cast(dtypes.float),axis),arg=(Ops.ADD,0))
    return _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).sink().toposort()))

  scalar, vector, large = lower(1), lower(45), lower(128, 128)
  assert scalar is not None and vector is not None and large is not None
  assert len(_ew_ops(vector)) == len(_ew_ops(scalar)) < 200 and len(_ew_ops(large)) < 300

  def lower_mapped(rows:int, depth:int, centered:bool):
    out=UOp.param(0,dtypes.half,(rows,))
    data,center=UOp.param(1,dtypes.half,(rows*depth,)),UOp.param(2,dtypes.half,(rows,))
    row=UOp.range(rows,1)
    axis=UOp.range(depth,0,AxisType.REDUCE,src=(row,))
    value=data.index(row*depth+axis).load()
    if centered:
      value=value-center.index(row).load()
      value=value*value
    reduced=UOp(Ops.REDUCE,dtypes.float,src=(value.cast(dtypes.float),axis),arg=(Ops.ADD,0))
    return _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).sink().toposort()))
  for depth,centered in ((65,True),(513,False)):
    scalar_mapped,vector_mapped=lower_mapped(1,depth,centered),lower_mapped(15,depth,centered)
    assert scalar_mapped is not None and vector_mapped is not None and len(_ew_ops(vector_mapped))==len(_ew_ops(scalar_mapped))
    values=(np.arange(15*depth,dtype=np.float16)%17-8)/8
    centers=(np.arange(15,dtype=np.float16)%5-2)/4
    actual=np.frombuffer(_execute_raw_dynamic_image(vector_mapped,30,values.tobytes(),centers.tobytes()),dtype="<f2")
    expected=np.asarray([np.frombuffer(_execute_raw_dynamic_image(scalar_mapped,2,
      values[row*depth:(row+1)*depth].tobytes(),centers[row:row+1].tobytes()),dtype="<f2")[0] for row in range(15)])
    np.testing.assert_array_equal(actual,expected)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",15*25*35,dtypes.half,num=1041).reshape((15,25,35)))
    linear=source.var((1,2)).schedule_linear()
    calls=[u for u in linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    to_program_cache.clear()
    images=[decode_image(next(u for u in to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src
      if u.op is Ops.BINARY).arg) for call in calls]
  assert len(images)==2 and tuple(len(_ew_ops(image)) for image in images)==(11,115)
  assert all(_assert_decoded_image_bounds(image)==image for image in images)
  values=np.random.default_rng(0).uniform(-2,2,size=(15,25,35)).astype(np.float16)
  mean=_execute_raw_dynamic_image(images[0],30,values.tobytes())
  actual=np.frombuffer(_execute_raw_dynamic_image(images[1],30,values.tobytes(),mean),dtype="<f2")
  np.testing.assert_allclose(actual,values.astype(np.float32).var(axis=(1,2),ddof=1).astype(np.float16),atol=1e-6,rtol=1e-3)


def test_production_cumprod_maps_balanced_half_product_rows():
  rendered=[]
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    for case,count in enumerate((512,1022)):
      source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=1050+case).reshape((count,)))
      calls=[u for u in source.cumprod(0).schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
      to_program_cache.clear()
      rendered.append([decode_image(next(u.arg for u in to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src
        if u.op is Ops.BINARY)) for call in calls])
  assert tuple(tuple(len(_ew_ops(image)) for image in images) for images in rendered)==((10,),(9,52,1))
  assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for images in rendered for image in images)
  for count,images in zip((512,1022),rendered):
    values=np.random.default_rng(7).uniform(-2,2,size=count).astype("<f2")
    if len(images)==1: output=_execute_raw_dynamic_image(images[0],count*2,values.tobytes())
    else:
      padded=(count+255)//256*256
      chunks=_execute_raw_dynamic_image(images[0],padded*2,values.tobytes())
      base=_execute_raw_dynamic_image(images[1],padded//256*2,chunks)
      output=_execute_raw_dynamic_image(images[2],padded*2,chunks,base)[-count*2:]
    np.testing.assert_allclose(np.frombuffer(output,dtype="<f2"),np.cumprod(values,dtype="<f2"),atol=5e-3,rtol=5e-3)


def test_production_masked_sibling_reductions_map_sequentially():
  shape=(1,1,16,16,16)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",math.prod(shape),dtypes.half,num=13015).reshape(shape))
    output=source.avg_pool2d(kernel_size=(8,8,8),stride=5,padding=1,count_include_pad=False)
    call=next(u for u in output.schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK)
    to_program_cache.clear()
    image=decode_image(next(u.arg for u in to_program(call.src[0],RockchipRenderer(Target(device="ROCKCHIP"))).src if u.op is Ops.BINARY))
  assert len(_ew_ops(image))<50 and _assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image
  data=np.random.default_rng(3).uniform(-4,4,size=shape).astype("<f2")
  actual=np.frombuffer(_execute_raw_dynamic_image(image,27*2,data.tobytes()),dtype="<f2").reshape(3,3,3)
  expected=np.empty((3,3,3),dtype="<f2")
  for z,y,x in itertools.product(range(3),repeat=3):
    starts=(z*5-1,y*5-1,x*5-1)
    slices=tuple(slice(max(0,start),min(16,start+8)) for start in starts)
    expected[z,y,x]=data[(0,0,*slices)].astype(np.float32).mean()
  np.testing.assert_allclose(actual,expected,atol=5e-3,rtol=5e-3)


def test_production_normalize_maps_predicated_rows():
  rng=np.random.default_rng(5)
  for case,kwargs in enumerate(({}, {"dim":0}, {"p":1}, {"p":3,"dim":0}, {"p":-1})):
    shape=(45,65)
    count=math.prod(shape)
    data=rng.uniform(-2,2,size=shape).astype("<f2")
    with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
      source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=13020+case).reshape(shape))
      output=source.normalize(**kwargs)
      calls=[u for u in output.schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
      renderer=RockchipRenderer(Target(device="ROCKCHIP"))
      to_program_cache.clear()
      images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
    assert len(images)==2 and len(_ew_ops(images[0]))<700
    assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for image in images)
    norm_count=rockchip_renderer._outs(list(calls[0].src[0].toposort()))[1][2]
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
      denominator=_execute_raw_dynamic_image(images[0],norm_count*2,data.tobytes())
      actual=np.frombuffer(_execute_raw_dynamic_image(images[1],count*2,data.tobytes(),denominator),dtype="<f2").reshape(shape)
    wide=data.astype(np.float32)
    p=float(kwargs.get("p",2))
    dim=int(kwargs.get("dim",1))
    norm=np.sum(np.abs(wide)**p,axis=dim,keepdims=True)**(1/p)
    expected=(wide/np.maximum(norm,1e-12)).astype("<f2")
    np.testing.assert_allclose(actual,expected,atol=5e-3,rtol=5e-3)


def test_cmac_candidate_filter_keeps_later_valid_layout():
  def lower(depth:int) -> RKImage:
    with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
      source = Tensor(UOp.new_buffer("ROCKCHIP",2*depth,dtypes.half,num=1040).reshape((2,depth)))
      ast = source.sum(axis=1,dtype=dtypes.float).cast(dtypes.half).schedule_linear().src[0].src[0]
      to_program_cache.clear()
      program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
    return decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  boundary, expanded = lower(384), lower(385)
  assert _cmac(boundary) is not None and (_cmac(boundary).m,_cmac(boundary).n,_cmac(boundary).k) == (2,1,384)
  assert _cmac(expanded) is not None and (_cmac(expanded).m,_cmac(expanded).n,_cmac(expanded).k) == (1,2,385)
  assert _output_gathers(boundary)[0].offsets == (0,64) and _output_gathers(expanded)[0].offsets == (0,1)
  assert all(not _ew_ops(image) and len(_initial_gathers(image)) == 2 and decode_image(encode_image(image)) == image for image in (boundary,expanded))
  source_values = (np.arange(770)%17-8).astype(np.float16).reshape(2,385)
  scratch = [np.zeros(size//2,dtype=np.uint16) for size in expanded.scratch]
  for gather in _initial_gathers(expanded):
    destination = scratch[gather.dst.index]
    if gather.values: destination[:gather.count] = gather.values
    else:
      offsets, source_bits = np.asarray(gather.offsets), source_values.view(np.uint16).reshape(-1)
      valid = offsets >= 0
      if not gather.partial: destination[:gather.count] = gather.fill_bits
      destination[:gather.count][valid] = source_bits[offsets[valid]]
  cmac = _cmac(expanded)
  lhs = scratch[cmac.lhs.index].view(np.float16).reshape(cmac.m,-1)
  ai,ao = lhs.shape[1],len(scratch[cmac.rhs.index])//lhs.shape[1]
  rhs = scratch[cmac.rhs.index].view(np.float16).reshape(ao//16,ai//32,16,32).transpose(0,2,1,3).reshape(ao,ai)
  product, physical = lhs.astype(np.float32)@rhs.astype(np.float32).T, np.zeros(cmac.m*ao*2,dtype=np.float16)
  for row in range(cmac.m):
    for col in range(cmac.n): physical[row*ao*2+col//16*32+col%16] = product[row,col]
  np.testing.assert_array_equal(physical[np.asarray(_output_gathers(expanded)[0].offsets)],source_values.sum(axis=1,dtype=np.float32))


def test_cmac_extent_bound_rejects_before_static_unroll(monkeypatch):
  def reduction(depth:int):
    out,source=UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(depth,))
    axis=UOp.range(depth,0,AxisType.REDUCE)
    reduced=UOp(Ops.REDUCE,dtypes.float,src=(source.index(axis).load().cast(dtypes.float),axis),arg=(Ops.ADD,0))
    uops=list(out.index(0).store(reduced.cast(dtypes.half)).sink().toposort())
    output=rockchip_renderer._outs(uops)[1]
    assert output is not None
    return output,uops
  assert rockchip_renderer._lower_cmac_reduce(*reduction(416)) is not None
  def forbidden(*_args,**_kwargs): raise AssertionError("oversized CMAC candidate was statically unrolled")
  monkeypatch.setattr(rockchip_renderer,"_unroll_static_reduces",forbidden)
  assert rockchip_renderer._lower_cmac_reduce(*reduction(417)) is None


def test_large_static_reduce_balances_before_generic_post_uops():
  size = 1025
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (size,))
  axis = UOp.range(size, 0, AxisType.REDUCE)
  value = source.index(axis).load()
  reduced=UOp(Ops.REDUCE,dtypes.float,src=((value*value).cast(dtypes.float),axis),arg=(Ops.ADD,0))
  post = (reduced.cast(dtypes.half)*UOp.const(1/size, dtypes.half)).sqrt()
  output = out.index(0).store(post)
  uops = list(output.sink().toposort())
  expanded = _unroll_static_reduces(post,precise=False)
  depth:dict[UOp,int] = {}
  for node in expanded.toposort(): depth[node] = 1+max((depth[source] for source in node.src),default=0)
  assert depth[expanded] < 64
  image = _lower_uop_program(uops)
  assert image is not None and _cmac(image) is None and _gather_after(image) < len(_ew_ops(image))
  assert _ew_ops(image)[-1].dst == RKArg(RKBufferKind.ARG, 0) and sum(op.submit_barrier for op in _ew_ops(image))==28


def test_large_predicate_graph_keeps_fp16_prelude_before_typed_comparisons():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",1022,dtypes.half,num=13010))
    linear,_ = source.cummin(0)[1].linear_with_vars()
    to_program_cache.clear()
    renderer = RockchipRenderer(Target(device="ROCKCHIP"))
    images = [decode_image(to_program(call.src[0],renderer).src[-1].arg) for call in linear.src if call.src[0].op is Ops.SINK]
  def typed(op): return op.mode in _TYPED_EW_MODES
  assert images and all(typed(op) for op in _ew_ops(images[-1]))


def test_production_sort_maps_short_bounded_integer_sums():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",384,dtypes.half,num=13013).reshape((8,8,6)))
    values,indices=source.sort(-1,descending=True)
    calls=[u for u in values.schedule_linear(indices).toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    ranked=[]
    weighted=[]
    for call in calls:
      uops=list(call.src[0].toposort())
      output=rockchip_renderer._outs(uops)[1] or rockchip_renderer._outs(uops)[0]
      reductions=tuple(node for node in uops if node.op is Ops.REDUCE)
      if output is None or len(reductions)!=1 or reductions[0].dtype.scalar() is not dtypes.int or rockchip_renderer._lower_mapped_reduce(output,uops) is None: continue  # noqa: E501
      to_program_cache.clear()
      image=decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY))
      params=tuple(param for param in uops if param.op is Ops.PARAM)
      input_count=next(int(param.src[0].arg) for param in params if param.arg.slot==1)
      (ranked if len(params)==2 else weighted).append((input_count,image))
  assert [count for count,_ in ranked]==[384,512] and all(len(_ew_ops(image))==71 for _,image in ranked)
  special=np.asarray((0.0,-0.0,np.inf,-np.inf,np.nan,np.nextafter(np.float16(0),np.float16(1)),-1.0),dtype="<f2")
  for input_count,image in ranked:
    width=input_count//64
    data=np.random.default_rng(input_count).uniform(-4,4,size=input_count).astype("<f2")
    data[(np.arange(len(special))*37+3)%input_count]=special
    actual=np.frombuffer(_execute_raw_dynamic_image(image,384*4,data.tobytes()),dtype="<i4")
    expected=[]
    for outer0,outer1,position in itertools.product(range(8),range(8),range(6)):
      row=data[(outer0*8+outer1)*width:(outer0*8+outer1)*width+6]
      expected.append(sum(index<position+1 and row[index]==row[position] for index in range(6)))
    np.testing.assert_array_equal(actual,np.asarray(expected,dtype=np.int32))
    assert _assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image
  assert len(weighted)==1 and weighted[0][0]==384 and len(_ew_ops(weighted[0][1]))<200
  weighted_image=weighted[0][1]
  assert not any(op.mode==RKEWMode.HALF_TO_INT32 for op in _ew_ops(weighted_image))
  rank_values=np.asarray((dtypes.int.min,dtypes.int.max,-65537,-32769,-1,0,1,32768,65537),dtype="<i4")
  for seed in range(24):
    rng=np.random.default_rng(seed)
    source_values=rng.integers(-3,4,size=(8,8,6)).astype("<f2")
    sorted_values=rng.integers(-3,4,size=(8,8,8)).astype("<f2")
    source_ranks=rng.choice(rank_values,size=(8,8,6))
    sorted_ranks=rng.choice(rank_values,size=(8,8,6))
    if seed==0:
      source_values.reshape(-1)[:7]=special
      sorted_values.reshape(-1)[:7]=special
    actual=np.frombuffer(_execute_raw_dynamic_image(weighted_image,384*4,source_values.tobytes(),sorted_values.tobytes(),
      source_ranks.tobytes(),sorted_ranks.tobytes()),dtype="<i4")
    expected=[sum(position for position in range(6) if source_values[outer0,outer1,position]==sorted_values[outer0,outer1,output] and
      source_ranks[outer0,outer1,position]==sorted_ranks[outer0,outer1,output])
      for outer0,outer1,output in itertools.product(range(8),range(8),range(6))]
    np.testing.assert_array_equal(actual,np.asarray(expected,dtype="<i4"))
  assert _assert_decoded_image_bounds(weighted_image)==weighted_image and decode_image(encode_image(weighted_image))==weighted_image
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",18,dtypes.half,num=13014))
    values,indices=source.sort(-1)
    calls=[u for u in values.schedule_linear(indices).toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    weighted_call=next(call for call in calls if len(tuple(u for u in call.src[0].toposort() if u.op is Ops.PARAM))==5)
    to_program_cache.clear()
    padded_image=decode_image(next(u.arg for u in to_program(weighted_call.src[0],renderer).src if u.op is Ops.BINARY))
  assert len(_ew_ops(padded_image))<200 and not any(op.mode==RKEWMode.HALF_TO_INT32 for op in _ew_ops(padded_image))
  for seed in range(8):
    rng=np.random.default_rng(seed+100)
    source_values=rng.integers(-3,4,size=18).astype("<f2")
    sorted_values=rng.integers(-3,4,size=32).astype("<f2")
    source_ranks=rng.choice(rank_values,size=18)
    sorted_ranks=rng.choice(rank_values,size=18)
    actual=np.frombuffer(_execute_raw_dynamic_image(padded_image,18*4,source_values.tobytes(),sorted_values.tobytes(),
      source_ranks.tobytes(),sorted_ranks.tobytes()),dtype="<i4")
    expected=[sum(position for position in range(18) if source_values[position]==sorted_values[output] and
      source_ranks[position]==sorted_ranks[output]) for output in range(18)]
    np.testing.assert_array_equal(actual,np.asarray(expected,dtype="<i4"))
  assert _assert_decoded_image_bounds(padded_image)==padded_image and decode_image(encode_image(padded_image))==padded_image


def test_mapped_weight_padding_does_not_coerce_half_infinity():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",45*65,dtypes.half,num=13015).reshape((45,65)))
    calls=[u for u in source.logcumsumexp(0).schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    to_program_cache.clear()
    images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
  assert len(images)==2 and all(_assert_decoded_image_bounds(image)==image for image in images)


def test_production_cumulative_index_uses_bounded_mapped_max():
  count=512
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=13011))
    values,indices=source.cummax(0)
    calls=[u for u in values.schedule_linear(indices).toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    to_program_cache.clear()
    images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
  assert len(images)==2 and len(_ew_ops(images[0]))<20 and len(_ew_ops(images[1]))<100
  assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for image in images)
  data=np.random.default_rng(0).uniform(-2,2,size=count).astype("<f2")
  data[:12]=np.asarray((0,-0.0,1,1,-1,-1,np.inf,np.inf,-np.inf,-np.inf,np.nan,2),dtype="<f2")
  running=np.frombuffer(_execute_raw_dynamic_image(images[0],count*2,data.tobytes()),dtype="<f2")
  actual=np.frombuffer(_execute_raw_dynamic_image(images[1],count*4,data.tobytes(),running.tobytes()),dtype="<i4")
  match=(data[:,None]==running[None,:])&np.triu(np.ones((count,count),dtype=bool))
  expected=count-(match*np.arange(count,0,-1,dtype=np.int32)[:,None]).max(axis=0)
  np.testing.assert_array_equal(actual,expected)


def test_production_large_cumulative_index_uses_bounded_mapped_max():
  count=1022
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=13013))
    values,indices=source.cummax(0)
    calls=[u for u in values.schedule_linear(indices).toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    images=[]
    for call in calls:
      to_program_cache.clear()
      images.append(decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)))
  coordinate=images[-1]
  assert len(images)==4 and (len(coordinate.scratch),len(_static_gathers(coordinate)),len(_ew_ops(coordinate)))==(26,19,66)
  assert not _runtime_gathers(coordinate) and _assert_decoded_image_bounds(coordinate)==coordinate
  data=np.random.default_rng(1022).uniform(-8,8,size=count).astype("<f2")
  data[:8]=np.asarray((0,-0.0,1,1,-1,-1,2,2),dtype="<f2")
  data[900:902]=np.inf
  data[950]=-np.inf
  partial=_execute_raw_dynamic_image(images[0],1024*2,data.tobytes())
  blocks=_execute_raw_dynamic_image(images[1],4*2,partial)
  running_storage=_execute_raw_dynamic_image(images[2],1024*2,partial,blocks)
  actual=np.frombuffer(_execute_raw_dynamic_image(coordinate,count*4,data.tobytes(),running_storage),dtype="<i4")
  running=np.maximum.accumulate(data)
  match=(data[:,None]==running[None,:])&np.triu(np.ones((count,count),dtype=bool))
  expected=count-(match*np.arange(count,0,-1,dtype=np.int32)[:,None]).max(axis=0)
  np.testing.assert_array_equal(actual,expected)


def test_production_cumulative_min_maps_negated_max():
  count=512
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=13012))
    values,indices=source.cummin(0)
    calls=[u for u in values.schedule_linear(indices).toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    to_program_cache.clear()
    images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
  assert len(images)==3 and len(_ew_ops(images[0]))<20 and len(_ew_ops(images[2]))<100
  assert all(op.mode in _TYPED_EW_MODES for op in _ew_ops(images[2]))
  assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for image in images)
  data=np.random.default_rng(1).uniform(-2,2,size=count).astype("<f2")
  data[:12]=np.asarray((0,-0.0,1,1,-1,-1,np.inf,np.inf,-np.inf,-np.inf,np.nan,2),dtype="<f2")
  running=np.frombuffer(_execute_raw_dynamic_image(images[0],count*2,data.tobytes()),dtype="<f2")
  actual_values=np.frombuffer(_execute_raw_dynamic_image(images[1],count*2,running.tobytes()),dtype="<f2")
  actual_indices=np.frombuffer(_execute_raw_dynamic_image(images[2],count*4,data.tobytes(),running.tobytes()),dtype="<i4")
  negative=(data*np.float16(-1)).astype("<f2")
  expected_running=np.maximum.accumulate(negative)
  np.testing.assert_array_equal(running.view("<u2"),expected_running.view("<u2"))
  np.testing.assert_array_equal(actual_values.view("<u2"),(expected_running*np.float16(-1)).astype("<f2").view("<u2"))
  match=(negative[:,None]==running[None,:])&np.triu(np.ones((count,count),dtype=bool))
  expected_indices=count-(match*np.arange(count,0,-1,dtype=np.int32)[:,None]).max(axis=0)
  np.testing.assert_array_equal(actual_indices,expected_indices)


def test_production_logcumsumexp_maps_masked_half_rows():
  shape=(45,65)
  count=math.prod(shape)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF"):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count,dtypes.half,num=13014).reshape(shape))
    output=source.logcumsumexp(1)
    calls=[u for u in output.schedule_linear().toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.SINK]
    renderer=RockchipRenderer(Target(device="ROCKCHIP"))
    to_program_cache.clear()
    images=[decode_image(next(u.arg for u in to_program(call.src[0],renderer).src if u.op is Ops.BINARY)) for call in calls]
  assert len(images)==2 and len(_ew_ops(images[1]))<600
  assert all(_assert_decoded_image_bounds(image)==image and decode_image(encode_image(image))==image for image in images)
  data=np.random.default_rng(2).uniform(-4,4,size=shape).astype("<f2")
  maximum=_execute_raw_dynamic_image(images[0],count*2,data.tobytes())
  actual=np.frombuffer(_execute_raw_dynamic_image(images[1],count*2,data.tobytes(),maximum),dtype="<f2").reshape(shape)
  expected=np.logaddexp.accumulate(data.astype(np.float32),axis=1).astype("<f2")
  np.testing.assert_allclose(actual,expected,atol=5e-3,rtol=5e-3)


def test_multiple_output_stores_execute_sequentially():
  first, second = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (4,))
  source, lane = UOp.param(2, dtypes.half, (4,)), UOp.range(4, 0)
  program = list(UOp.sink(first.index(lane).store(source.index(lane).load()+1.0),
                          second.index(lane).store(first.index(lane).load()*2.0)).toposort())
  image = _lower_uop_program(program)
  assert image is not None and len(_ew_ops(image)) == 2
  assert _ew_ops(image)[1].lhs == RKArg(RKBufferKind.ARG, 0)
  assert _ew_ops(image)[1].submit_barrier and _ew_ops(image)[1].mode==RKEWMode.HALF


def test_static_structural_expansion_is_bounded():
  limit = (1 << 14) + 1
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (limit,))
  lane, axis = UOp.range(1, 0), UOp.range(limit, 1, AxisType.REDUCE)
  reduced = UOp(Ops.REDUCE, dtypes.half, src=(source.index(axis).load(), axis), arg=(Ops.ADD,0))
  uops = list(out.index(lane).store(reduced).end(lane, axis).sink().toposort())
  assert _lower_uop_program(uops) is None


def test_deep_generic_graph_canonicalization_is_iterative():
  value = UOp.const(0, dtypes.int)
  for _ in range(4096): value = value + UOp.const(1, dtypes.int)
  rewritten = _finite_int_max_neutrals(value)
  assert rewritten.key == value.key


def test_static_range_environment_allocation_is_bounded():
  axes = [UOp.range(1024, 0), UOp.range(1024, 1)]
  try: _static_lanes(tuple(axes),*axes,limit=1024)
  except RuntimeError as error: assert "static_index_budget" in str(error)
  else: raise AssertionError("oversized static RANGE product was materialized")


def test_large_range_independent_static_value_is_materialized_once():
  image = _lower_uop_program(_program(dtypes.half,
    lambda _i:UOp.const(0.0, dtypes.half) * UOp.const(-1.0, dtypes.half), count=1 << 20))
  assert image is not None and not _initial_gathers(image) and _constant_bytes(image) == struct.pack("<e", -0.0)


def test_generic_ew_chain_reuses_dead_scratch_values():
  source = UOp.param(1, dtypes.half, (1024,))
  def chain(i):
    value = source.index(i).load()
    for _ in range(128): value = value * UOp.const(1.001, dtypes.half)
    return value
  image = _lower_uop_program(_program(dtypes.half, chain, count=1024))
  assert image is not None and len(_ew_ops(image)) == 128 and len(image.scratch) <= 4


def test_scratch_reuse_follows_the_ordered_program_lifetime():
  scratch = (64,)*5
  arg, slots = RKArg(RKBufferKind.ARG, 0), tuple(RKArg(RKBufferKind.SCRATCH, i) for i in range(5))
  gather=RKGather(arg,slots[3],1,offsets=(0,),partial=True)
  image = RKImage(scratch,program=(RKGather(None,slots[0],1,values=(0,)),
    RKEWOp(slots[1], slots[0], slots[0], 1, _EW_CFG[Ops.ADD]),
    RKEWOp(slots[2], slots[1], slots[0], 1, _EW_CFG[Ops.ADD]),
    gather,RKEWOp(slots[4],slots[3],slots[3],1,_EW_CFG[Ops.ADD])))
  colored = _reuse_linear_scratch(image)
  assert len(colored.scratch) <= 4
  physical_gather=next(op for op in colored.program if isinstance(op,RKGather) and op.src is not None)
  assert colored.program.index(physical_gather)==3 and physical_gather.dst==_ew_ops(colored)[-1].lhs
  assert decode_image(encode_image(colored)) == colored

  runtime=object.__new__(rockchip_runtime.RockchipProgram)
  rockchip_runtime.RockchipProgram.__init__(runtime,SimpleNamespace(),SimpleNamespace(name="alignment",lib=encode_image(RKImage((1,4097,64)))))
  assert runtime._scratch_offsets == (0,4096,12288,16384)


def test_large_divided_range_address_uses_compact_gather_axes():
  outer = UOp.range(16384, 1)
  inner = UOp.range(64, 4, src=(outer,))
  out_index = outer*64+inner
  grouped = UOp(Ops.CDIV, dtypes.int, src=(outer, UOp.const(64, dtypes.int)))*1024+inner
  plan = _gather_plan(1, 0, out_index, grouped, None, 1 << 20)
  assert not plan.offsets and plan.base == 0
  assert set(plan.axes) == {(1, 64, 1), (4096, 256, 1024)}


def test_dynamic_host_gather_is_explicit_and_direct_scatter_fails_closed(monkeypatch):
  monkeypatch.delenv("ROCKCHIP_HOST_GATHER", raising=False)
  indices = UOp.param(2, dtypes.int, (4,))
  axis = UOp.range(4, 0)

  gather_out, gather_source = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (8,))
  gather = gather_out.index(axis).store(gather_source.index(indices.index(axis).load()).load())
  gather_uops = list(gather.end(axis).sink().toposort())
  gather_image = _lower_uop_program(gather_uops)
  assert gather_image is not None and bool(_runtime_gathers(gather_image))
  assert len(_runtime_gathers(gather_image,False)) == 1 and not _runtime_gathers(gather_image,True)
  assert decode_image(encode_image(gather_image)) == gather_image

  scatter_out, scatter_source = UOp.param(0, dtypes.half, (8,)), UOp.param(1, dtypes.half, (4,))
  scatter = scatter_out.index(indices.index(axis).load()).store(scatter_source.index(axis).load())
  scatter_uops = list(scatter.end(axis).sink().toposort())
  assert _lower_uop_program(scatter_uops) is None

  monkeypatch.setenv("ROCKCHIP_HOST_GATHER", "0")
  assert _lower_uop_program(gather_uops) is None and _lower_uop_program(scatter_uops) is None


def test_dynamic_host_gather_materializes_affine_lane_address_on_npu(monkeypatch):
  monkeypatch.setenv("ROCKCHIP_HOST_GATHER", "1")
  count = 4
  out, source = UOp.param(0, dtypes.half, (count,)), UOp.param(1, dtypes.half, (count*10,))
  indices, lane = UOp.param(2, dtypes.int, (count,)), UOp.range(count, 0)
  value = source.index(lane*10+indices.index(lane).load()).load()
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(_runtime_gathers(image,False)) == 1
  address = _runtime_gathers(image,False)[0]
  assert address.index is not None and address.index.kind is RKBufferKind.SCRATCH and _gather_point(image,address)>0
  source_values=np.arange(count*10,dtype="<u2")
  index_values=np.arange(1,count+1,dtype="<i4")
  assert _execute_raw_dynamic_image(image,count*2,source_values.tobytes(),index_values.tobytes()) == source_values[[1,12,23,34]].tobytes()
  assert decode_image(encode_image(image)) == image


def test_dynamic_host_gather_materializes_nonaffine_static_lane_bases(monkeypatch):
  monkeypatch.delenv("ROCKCHIP_HOST_GATHER", raising=False)
  out, source = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (8,))
  indices, lane = UOp.param(2, dtypes.int, (4,)), UOp.range(4, 0)
  runtime = indices.index(lane).load()
  batch, spatial = lane.alu(Ops.CDIV, lane.const_like(2)), lane.alu(Ops.CMOD, lane.const_like(2))
  address = batch*4 + spatial + runtime*2
  gate = ((runtime < UOp.const(0, dtypes.int)) != UOp.const(True, dtypes.bool)) & (runtime < UOp.const(2, dtypes.int))
  value = source.index(address).load(UOp.const(0.0, dtypes.half), gate) * UOp.const(2.0, dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(_runtime_gathers(image,False)) == 1
  host = _runtime_gathers(image,False)[0]
  assert host.src.kind is RKBufferKind.ARG and host.index is not None and host.index.kind is RKBufferKind.SCRATCH
  assert _gather_point(image,host)>0
  assert decode_image(encode_image(image)) == image


def test_nonaffine_scalar_dot_uses_cmac_static_packing():
  groups = 64
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (groups,)), UOp.param(2, dtypes.half, (groups,))
  permutation = tuple((lane*17)%groups for lane in range(groups))
  terms = [lhs.index(permutation[lane]).load()*rhs.index(lane).load() for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  image = _lower_uop_program(list(out.index(UOp.const(0, dtypes.int)).store(value).sink().toposort()))
  assert image is not None and _cmac(image) is not None and (_cmac(image).m,_cmac(image).n,_cmac(image).k) == (1,1,groups)
  assert any(gather.offsets[:groups] == permutation for gather in _initial_gathers(image))
  assert not _ew_ops(image) and _output_gathers(image)[0].offsets == (0,)
  assert decode_image(encode_image(image)) == image
