import ctypes, functools, hashlib, itertools, math, struct
import numpy as np
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from tinygrad import Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKCMAC, RKExecutionClass, RKImage, RKLayout, RKTarget, RKValue, RKEWOp,
  RKGather, RKScratch,
  _EW_CFG, _EW_CFG_ABS, _EW_CFG_FLOOR, _EW_CFG_MIN, _EW_STAGE_FP32_IN, _EW_STAGE_FP32_OUT, _NATIVE_SIGN, _MAX_EW_ELEMS_FP16, _RKIMAGE_U16_MAX,
  _canonical_half_storage, _finite_int_max_neutrals, _fp32_expr_to_half, _gather_plan, _iter_range_env,
  _lower_uop_program, _reuse_linear_scratch, _unroll_static_local, RockchipRenderer, decode_image, emit_cmac_stage, encode_image, patch_stage)
from tinygrad.runtime import ops_rockchip as rockchip_runtime
import tinygrad.renderer.rockchip as rockchip_renderer
from tinygrad.uop.ops import AxisType, Ops, UOp


def _program(dtype, value, count:int=4):
  out, axis = UOp.param(0, dtype, (count,)), UOp.range(count, 0)
  return list(out.index(axis).store(value(axis)).end(axis).sink().toposort())

def _cmac_weights(image:RKImage) -> tuple[int, ...]:
  assert image.cmac is not None and image.cmac.n == 1 and image.gathers[1].values
  return tuple(image.gathers[1].values[(lane//32)*16*32+lane%32] for lane in range(image.cmac.k))

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

def _execute_raw_dynamic_image(image:RKImage, output_bytes:int, *inputs:bytes) -> bytes:
  """Execute the selector's raw gathers plus native INT16 mask/reduction subset."""
  args, scratch = [bytearray(output_bytes), *(bytearray(value) for value in inputs)], [bytearray(spec.size) for spec in image.scratch]
  for slot in range(len(image.constants)//2):
    lane = image.constants[slot*2:slot*2+2]
    scratch[slot][:] = lane*(len(scratch[slot])//2)
  def buffer(kind:RKBufferKind, index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  def apply_gathers(items) -> None:
    for gather in items:
      lane_dtype = {1:np.uint8, 2:np.uint16, 4:np.uint32}[gather.itemsize]
      dst, lanes = np.frombuffer(buffer(gather.dst_kind, gather.dst_index), dtype=lane_dtype), np.arange(gather.count, dtype=np.intp)
      dst_index = gather.dst_addend+lanes*gather.dst_stride
      if gather.values: dst[dst_index] = gather.values
      elif gather.offsets:
        src, offsets = np.frombuffer(buffer(gather.src_kind, gather.src_index), dtype=lane_dtype), np.asarray(gather.offsets)
        valid = offsets >= 0
        if not gather.partial: dst[dst_index] = gather.fill_bits
        dst[dst_index[valid]] = src[offsets[valid]]
      else:
        src, offsets = np.frombuffer(buffer(gather.src_kind, gather.src_index), dtype=lane_dtype), np.full(gather.count, gather.base)
        for divisor,limit,stride in gather.axes: offsets += lanes//divisor%limit*stride
        dst[dst_index] = src[offsets]
  def execute(op:RKEWOp) -> None:
    assert op.int16_input and op.int16_output and not op.int32_input and not op.int32_output
    def view(arg:RKArg) -> np.ndarray: return np.frombuffer(buffer(arg.kind, arg.index), dtype="<i2", count=op.count, offset=arg.addend)
    lhs, rhs = view(op.lhs).astype(np.int32), view(op.rhs).astype(np.int32)
    if op.ew_cfg == _EW_CFG[Ops.ADD]: value = lhs+rhs
    elif op.ew_cfg == _EW_CFG[Ops.SUB]: value = lhs-rhs
    elif op.ew_cfg == _EW_CFG[Ops.MUL]: value = lhs*rhs
    elif op.ew_cfg == _EW_CFG[Ops.MAX]: value = np.maximum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_MIN: value = np.minimum(lhs, rhs)
    elif op.ew_cfg == _EW_CFG_ABS: value = np.abs(lhs)
    else: raise AssertionError(f"unsupported dynamic selector EW config {op.ew_cfg:#x}")
    view(op.dst)[:] = np.clip(value, -32768, 32767).astype("<i2")
  apply_gathers(image.gathers)
  mid:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers: mid.setdefault(gather.after if gather.after >= 0 else image.gather_after, []).append(gather)
  for index in range(len(image.ew_ops)+1):
    apply_gathers(mid.get(index, ()))
    if index < len(image.ew_ops): execute(image.ew_ops[index])
  apply_gathers(image.post_gathers)
  return bytes(args[0])

def _execute_integer_image(image:RKImage, *inputs:np.ndarray) -> np.ndarray:
  """Test-only physical executor for raw gathers and the signed integer EW subset used by INT32 division."""
  count = len(inputs[0])
  args = [bytearray(count*4), *(bytearray(value.astype("<i4").tobytes()) for value in inputs)]
  scratch = [bytearray(spec.size) for spec in image.scratch]
  for slot in range(len(image.constants)//2):
    lane = image.constants[slot*2:slot*2+2]
    scratch[slot][:] = lane*(len(scratch[slot])//2)
  def buffer(kind:RKBufferKind, index:int) -> bytearray: return args[index] if kind is RKBufferKind.ARG else scratch[index]
  linear:dict[int, np.ndarray] = {}
  def apply_gathers(gathers:tuple[RKGather, ...]) -> None:
    for gather in gathers:
      dtype = {1:np.uint8, 2:np.uint16, 4:np.uint32}[gather.itemsize]
      dst = np.frombuffer(buffer(gather.dst_kind, gather.dst_index), dtype=dtype)
      lanes = linear.setdefault(gather.count, np.arange(gather.count, dtype=np.intp))
      dst_index = gather.dst_addend + lanes*gather.dst_stride
      if gather.values: dst[dst_index] = gather.values
      elif gather.offsets:
        src = np.frombuffer(buffer(gather.src_kind, gather.src_index), dtype=dtype)
        index = np.asarray(gather.offsets, dtype=np.intp)
        valid = index >= 0
        if not gather.partial: dst[dst_index] = gather.fill_bits
        dst[dst_index[valid]] = src[index[valid]]
      else:
        src = np.frombuffer(buffer(gather.src_kind, gather.src_index), dtype=dtype)
        index = np.full(gather.count, gather.base, dtype=np.intp)
        for divisor,limit,stride in gather.axes: index += (lanes//divisor%limit)*stride
        dst[dst_index] = src[index]
  def view(arg:RKArg, dtype, lanes:int) -> np.ndarray:
    return np.frombuffer(buffer(arg.kind, arg.index), dtype=dtype, count=lanes, offset=arg.addend)
  def execute(op:RKEWOp) -> None:
    if op.int16_input and op.int16_output and not (op.int32_input or op.int32_output):
      source_dtype = destination_dtype = np.dtype("<i2")
    elif op.int32_input and op.int32_output and not (op.int16_input or op.int16_output):
      source_dtype = destination_dtype = np.dtype("<i4")
    elif op.int16_input and op.int32_output and not (op.int16_output or op.int32_input):
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
  apply_gathers(image.gathers)
  mid:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers: mid.setdefault(gather.after if gather.after >= 0 else image.gather_after, []).append(gather)
  for index in range(len(image.ew_ops)+1):
    apply_gathers(tuple(mid.get(index, ())))
    if index < len(image.ew_ops): execute(image.ew_ops[index])
  apply_gathers(image.post_gathers)
  return np.frombuffer(args[0], dtype="<i4").copy()

def _assert_scratch_extent(image:RKImage, arg:RKArg, need:int) -> None:
  if arg.kind is RKBufferKind.SCRATCH:
    assert 0 <= arg.index < len(image.scratch)
    assert arg.addend >= 0 and need <= image.scratch[arg.index].size

def _assert_decoded_image_bounds(image:RKImage) -> RKImage:
  decoded = decode_image(encode_image(image))
  def gather_indices(gather:RKGather) -> tuple[int, ...]:
    if gather.offsets: return tuple(index for index in gather.offsets if index >= 0)
    return tuple(gather.base+sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes) for lane in range(gather.count))
  for gather in (*decoded.gathers, *decoded.mid_gathers, *decoded.post_gathers):
    if gather.count <= 0: continue
    _assert_scratch_extent(decoded, RKArg(gather.dst_kind, gather.dst_index),
      (gather.dst_addend+(gather.count-1)*gather.dst_stride+1)*gather.itemsize)
    if not gather.values:
      indices = gather_indices(gather)
      if indices: _assert_scratch_extent(decoded, RKArg(gather.src_kind, gather.src_index), (max(indices)+1)*gather.itemsize)
  for op in decoded.ew_ops:
    source_width = 4 if op.int32_input or op.ew_cfg & _EW_STAGE_FP32_IN else 2
    destination_width = (1 if op.bool_output else 4) if op.int32_output else 4 if op.ew_cfg & _EW_STAGE_FP32_OUT else 2
    _assert_scratch_extent(decoded, op.lhs, op.lhs.addend+op.count*source_width)
    _assert_scratch_extent(decoded, op.rhs, op.rhs.addend+op.count*source_width)
    _assert_scratch_extent(decoded, op.dst, op.dst.addend+op.count*destination_width)
  return decoded

def _execute_fp16_reduction_tail(image:RKImage, values:np.ndarray) -> np.ndarray:
  args = [np.zeros(max(64, image.ew_ops[-1].count), dtype=np.float16)]
  scratch = [bytearray(spec.size) for spec in image.scratch]
  for slot in range(len(image.constants)//2): scratch[slot][:] = image.constants[slot*2:slot*2+2]*(len(scratch[slot])//2)
  source = np.frombuffer(scratch[image.mid_gathers[0].src_index], dtype="<f2")
  source[:len(values)] = values
  def buffer(arg:RKArg) -> np.ndarray: return args[arg.index] if arg.kind is RKBufferKind.ARG else np.frombuffer(scratch[arg.index], dtype="<f2")
  def read(arg:RKArg, count:int) -> np.ndarray: return buffer(arg)[arg.addend//2:arg.addend//2+count].copy()
  def write(arg:RKArg, value:np.ndarray) -> None: buffer(arg)[arg.addend//2:arg.addend//2+len(value)] = value.astype(np.float16)
  def apply_gather(gather:RKGather) -> None:
    src, dst, lanes = buffer(RKArg(gather.src_kind, gather.src_index)), buffer(RKArg(gather.dst_kind, gather.dst_index)), np.arange(gather.count)
    indices = np.asarray(gather.offsets) if gather.offsets else np.asarray(
      [gather.base+sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes) for lane in lanes])
    destination = gather.dst_addend+lanes*gather.dst_stride
    if gather.offsets and gather.partial:
      valid = indices >= 0
      dst[destination[valid]] = src[indices[valid]]
    else: dst[destination] = gather.values or src[indices]
  def execute(op:RKEWOp) -> None:
    lhs, rhs = read(op.lhs, op.count), read(op.rhs, op.count)
    value = (lhs+rhs if op.ew_cfg == _EW_CFG[Ops.ADD] else lhs-rhs if op.ew_cfg == _EW_CFG[Ops.SUB] else
             lhs*rhs if op.ew_cfg == _EW_CFG[Ops.MUL] else np.maximum(lhs, rhs) if op.ew_cfg == _EW_CFG[Ops.MAX] else
             lhs/rhs if op.ew_cfg == _EW_CFG[Ops.FDIV] else np.floor(lhs) if op.ew_cfg == _EW_CFG_FLOOR else
             np.minimum(lhs, rhs) if op.ew_cfg == _EW_CFG_MIN else np.abs(lhs) if op.ew_cfg == _EW_CFG_ABS else None)
    assert value is not None, hex(op.ew_cfg)
    write(op.dst, value)
  for gather in image.mid_gathers:
    if (gather.after if gather.after >= 0 else image.gather_after) == image.gather_after: apply_gather(gather)
  for op in image.ew_ops[image.gather_after:]: execute(op)
  return args[0]

def _execute_scalar_reduction_image(image:RKImage, values:np.ndarray) -> float:
  """Execute the native scalar FP16 reduction subset without opening a device."""
  args = [bytearray(4), bytearray(np.asarray(values, dtype="<f2").tobytes())]
  scratch = [bytearray(spec.size) for spec in image.scratch]
  for slot in range(len(image.constants)//2):
    np.frombuffer(scratch[slot], dtype="<f2")[:] = np.frombuffer(image.constants[slot*2:slot*2+2], dtype="<f2")[0]
  def storage(arg:RKArg) -> bytearray: return args[arg.index] if arg.kind is RKBufferKind.ARG else scratch[arg.index]
  def read(arg:RKArg, dtype:np.dtype, count:int) -> np.ndarray:
    return np.frombuffer(storage(arg), dtype=dtype, count=count, offset=arg.addend).copy()
  def write(arg:RKArg, dtype:np.dtype, value:np.ndarray) -> None:
    np.frombuffer(storage(arg), dtype=dtype, count=len(value), offset=arg.addend)[:] = value.astype(dtype)
  for gather in image.gathers:
    source, destination = np.frombuffer(storage(RKArg(gather.src_kind, gather.src_index)), dtype="<f2"), \
      np.frombuffer(storage(RKArg(gather.dst_kind, gather.dst_index)), dtype="<f2")
    for lane in range(gather.count):
      index = gather.offsets[lane] if gather.offsets else gather.base + sum(lane//divisor%limit*stride for divisor,limit,stride in gather.axes)
      destination[gather.dst_addend+lane*gather.dst_stride] = source[index]
  for op in image.ew_ops:
    input_dtype = np.dtype("<f4") if op.ew_cfg & _EW_STAGE_FP32_IN else np.dtype("<f2")
    output_dtype = np.dtype("<f4") if op.ew_cfg & _EW_STAGE_FP32_OUT else np.dtype("<f2")
    lhs, rhs = read(op.lhs, input_dtype, op.count), read(op.rhs, input_dtype, op.count)
    if output_dtype == np.dtype("<f4"): lhs, rhs = lhs.astype(np.float32), rhs.astype(np.float32)
    assert op.ew_cfg & ~(_EW_STAGE_FP32_IN | _EW_STAGE_FP32_OUT) == _EW_CFG[Ops.ADD]
    write(op.dst, output_dtype, lhs+rhs)
  return float(np.frombuffer(args[0], dtype="<f4")[0])

def test_cmac_codec_and_body_match_the_proven_gemm_contract():
  cmac = RKCMAC(RKArg(RKBufferKind.SCRATCH, 2), RKArg(RKBufferKind.SCRATCH, 0),
    RKArg(RKBufferKind.SCRATCH, 1), 3, 4, 5)
  image = RKImage(RKTarget.RK3588, (RKScratch(192), RKScratch(2048), RKScratch(384)), cmac=cmac)
  assert decode_image(encode_image(image)) == image
  for invalid in (replace(image, cmac=replace(cmac, lhs=RKArg(RKBufferKind.ARG, 0))),
                  replace(image, scratch=(RKScratch(191), *image.scratch[1:]))):
    try: encode_image(invalid)
    except ValueError: pass
    else: raise AssertionError("invalid CMAC image was encoded")
  try: emit_cmac_stage(replace(cmac, m=2, k=385))
  except ValueError: pass
  else: raise AssertionError("multi-row CMAC exceeded the donor CBUF contract")
  stage = emit_cmac_stage(cmac)
  assert len(stage.commands) == 45 and tuple(index for index,_ in stage.relocs) == (18, 24, 31)
  body = patch_stage(stage, lambda _kind,index:(0x100000,0x200000,0x300000)[index])
  assert hashlib.sha256(struct.pack("<45Q", *body)).hexdigest() == "d754ae668b210999c7d568131c0387e46be9c934ad3812b51fb956c789e3db22"
  relu_image = replace(image, cmac=replace(cmac, relu=True))
  assert decode_image(encode_image(relu_image)) == relu_image
  relu_stage = emit_cmac_stage(relu_image.cmac)
  changed = [(old,new) for old,new in zip(stage.commands,relu_stage.commands) if old != new]
  assert len(relu_stage.commands) == 45 and len(changed) == 1
  assert changed[0][0]&0xffff == changed[0][1]&0xffff == rockchip_renderer.rk.REG_DPU_BS_CFG
  assert ((changed[0][0]>>16)&0xffffffff,(changed[0][1]>>16)&0xffffffff) == (0x53,0x12)

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
    for env in rockchip_renderer._iter_range_env([outer, inner]):
      cache = {}
      expected[int(rockchip_renderer._eval_expr(out_index, env, cache))] = encode(rockchip_renderer._eval_expr(expr, env, cache))
    assert rockchip_renderer._static_values(out_index, expr, 20, encode) == tuple(expected)


def test_index_range_analysis_keeps_first_seen_order_and_hides_range_dependencies():
  first,second = UOp.range(3,20),UOp.range(4,21)
  root = UOp(Ops.ADD,dtypes.int,src=(UOp(Ops.ADD,dtypes.int,src=(second,first)),UOp(Ops.MUL,dtypes.int,src=(second,first))))
  assert rockchip_renderer._index_ranges(root) == [second,first]
  dependent = UOp(Ops.RANGE,dtypes.int,src=(UOp.const(2,dtypes.int),first),arg=(22,AxisType.REDUCE))
  assert rockchip_renderer._index_ranges(dependent) == [dependent]


def test_exact_integer_range_analysis_covers_supported_carriers_only():
  lane = UOp.range(4,23)
  expressions = (lane,(lane-2)*3,UOp(Ops.WHERE,dtypes.int,src=(lane<2,lane,UOp.const(-5,dtypes.int))),
                 UOp(Ops.XOR,dtypes.int,src=(lane,UOp.const(-1,dtypes.int))),UOp.const(True,dtypes.bool).cast(dtypes.int),lane<<1)
  assert tuple(map(rockchip_renderer._exact_int_range,expressions)) == ((0,3),(-6,3),(-5,3),(-4,-1),(1,1),None)


def test_rkvalue_is_the_typed_physical_abi():
  value = RKValue(RKArg(RKBufferKind.ARG, 0), dtypes.half, 1, RKLayout.FP16)
  assert value.dtype is dtypes.half and value.count == 1 and value.layout is RKLayout.FP16


def test_submit_retries_once_after_driver_timeout(monkeypatch):
  class FakeDevice:
    fd_ctl, submit_count, task_count, timeout_retries, resets = object(), 0, 0, 0, 0
    def _sync_buffer(self, _buffer, _flags): pass
    def reset_npu(self): self.resets += 1
    def _forget_program(self, _program): pass
    def _gpu_free(self, _buffer): pass
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev, program.submit_count = FakeDevice(), 0
  buffer = SimpleNamespace(meta=SimpleNamespace(obj_addr=1))
  calls = 0
  def submit(_fd, **_kwargs):
    nonlocal calls
    calls += 1
    if calls == 1: raise TimeoutError
  monkeypatch.setattr(rockchip_runtime.rk, "DRM_IOCTL_RKNPU_SUBMIT", submit)
  program._submit(buffer, buffer, 1)
  assert calls == 2 and program.dev.resets == program.dev.timeout_retries == 1
  assert program.submit_count == program.dev.submit_count == 1 and program.dev.task_count == 1


def test_cmac_submit_never_replays_a_timeout(monkeypatch):
  class FakeDevice:
    fd_ctl, submit_count, task_count, timeout_retries, resets = object(), 0, 0, 0, 0
    def _sync_buffer(self, _buffer, _flags): pass
    def reset_npu(self): self.resets += 1
    def _forget_program(self, _program): pass
    def _gpu_free(self, _buffer): pass
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev, program.submit_count = FakeDevice(), 0
  buffer = SimpleNamespace(meta=SimpleNamespace(obj_addr=1))
  calls = 0
  def submit(_fd, **_kwargs):
    nonlocal calls
    calls += 1
    raise TimeoutError
  monkeypatch.setattr(rockchip_runtime.rk, "DRM_IOCTL_RKNPU_SUBMIT", submit)
  try:
    program._submit(buffer, buffer, 1, retry=False)
  except TimeoutError: pass
  else: raise AssertionError("CMAC timeout was swallowed")
  assert calls == 1 and program.dev.resets == 0 and program.dev.timeout_retries == 1
  assert program.submit_count == program.dev.submit_count == program.dev.task_count == 0


def test_cmac_runtime_keeps_the_45_qword_body_and_four_qword_tail_separate():
  class FakeDevice:
    resets = 0
    def reset_npu(self): self.resets += 1
    def _forget_program(self, _program): pass
    def _gpu_free(self, _buffer): pass
  def memory(size, dma):
    storage = ctypes.create_string_buffer(size)
    base = SimpleNamespace(va_addr=ctypes.addressof(storage))
    return SimpleNamespace(va_addr=base.va_addr,size=size,meta=SimpleNamespace(dma_addr=dma,obj_addr=dma),base=base,storage=storage)
  cmac = RKCMAC(RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,1),3,4,5)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev, program.image = FakeDevice(), RKImage(RKTarget.RK3588,cmac=cmac)
  cmd, task, submits = memory(8192,0x400000), memory(4096,0x500000), []
  program._ensure_buffer = lambda attr,*_args,**_kwargs: cmd if "cmd" in attr else task
  program._submit = lambda *args,**kwargs: submits.append((args,kwargs))
  addresses = (0x100000,0x200000,0x300000)
  program._submit_standalone(patch_stage(emit_cmac_stage(cmac),lambda _kind,index:addresses[index]),True)
  commands = tuple((ctypes.c_uint64*49).from_address(cmd.va_addr))
  assert commands[:45] == patch_stage(emit_cmac_stage(cmac),lambda _kind,index:addresses[index])
  assert commands[45:] == (rockchip_runtime._pc(0x0001,0),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_PC_REG,rockchip_runtime.rk.REG_PC_REGISTER_AMOUNTS),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_VERSION,0),
    rockchip_runtime._pc(rockchip_runtime.rk.TARGET_PC,rockchip_runtime.rk.REG_PC_OPERATION_ENABLE,0xd))
  desc = rockchip_runtime.rk.struct_rknpu_task.from_address(task.va_addr)
  assert (desc.op_idx,desc.enable_mask,desc.int_mask,desc.int_clear,desc.regcfg_amount) == (0,0xd,0x300,0x1ffff,45)
  assert program.dev.resets == 1 and len(submits) == 1 and submits[0][1] == {"standalone":True,"retry":False}
  body = (1,2,3)
  program._submit_standalone(body)
  assert tuple((ctypes.c_uint64*4).from_address(cmd.va_addr)) == body+(rockchip_runtime._pc(
    rockchip_runtime.rk.TARGET_PC,rockchip_runtime.rk.REG_PC_OPERATION_ENABLE,0x18),)
  desc = rockchip_runtime.rk.struct_rknpu_task.from_address(task.va_addr)
  assert (desc.op_idx,desc.enable_mask,desc.int_mask,desc.int_clear,desc.regcfg_amount) == (4,0x18,0x300,0x1ffff,4)
  assert program.dev.resets == 1 and len(submits) == 2 and submits[1][1] == {"standalone":True}


def test_native_ew_configs_keep_their_exact_register_values():
  assert tuple(rockchip_renderer._EW_CFG[op] for op in (Ops.ADD,Ops.SUB,Ops.MUL,Ops.MAX,Ops.FDIV)) == (
    0x108202c0,0x108402c0,0x108003c4,0x108002c0,0x108303c0)
  assert tuple(getattr(rockchip_renderer,name) for name in
    ("_EW_CFG_RELU6","_EW_CFG_MIN","_EW_CFG_ABS","_EW_CFG_NEG","_EW_CFG_FLOOR","_EW_CFG_CEIL")) == (
    0x108004c0,0x108102c0,0x108502c0,0x108602c0,0x108702c0,0x108802c0)


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
  assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,4)
  assert not image.constants and len(image.scratch) == 3
  assert _cmac_weights(image) == tuple(rockchip_renderer._fp16_bits(weight) for weight in weights)
  for invalid in (0.1,float("inf")):
    fallback = lower((invalid,*weights[1:]))
    assert fallback is not None and fallback.cmac is None and fallback.ew_ops
  nested = lower(weights, (2.0,))
  assert nested is not None and nested.cmac is not None and _cmac_weights(nested) == (
    rockchip_renderer._fp16_bits(1.0), *tuple(rockchip_renderer._fp16_bits(weight) for weight in weights[1:]))
  for unsafe in (lower((10.0,*weights[1:]), (0.1,)), lower(weights, (2.0,1.0))):
    assert unsafe is not None and unsafe.cmac is None and unsafe.ew_ops
  half_out, half_source = UOp.param(0,dtypes.half,(1,)), UOp.param(1,dtypes.half,(4,))
  half_terms = [half_source.index(i).load() for i in range(4)]
  half_terms[0] = half_terms[0]*UOp.const(-53248.0,dtypes.half)*UOp.const(-0.03955078125,dtypes.half)
  half_value = functools.reduce(lambda total,term:total+term,half_terms[1:],half_terms[0])
  half_nested = _lower_uop_program(list(half_out.index(0).store(half_value).sink().toposort()))
  assert half_nested is not None and half_nested.cmac is None and half_nested.ew_ops


def test_generic_fp16_uops_lower_in_dependency_order():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:lhs.index(i).load() + rhs.index(i).load() * 2.0))
  assert image is not None
  assert len(image.ew_ops) == 2
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG and image.ew_ops[-1].dst.index == 0


def test_physical_half_recipe_materializes_strong_float_constant_at_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    UOp(Ops.ADD, dtypes.half, src=(source.index(i).load(), UOp.const(0.25, dtypes.float)))))
  assert image is not None and struct.pack("<e", 0.25) in image.constants


def test_fp32_load_materializes_through_canonical_fp16_physical_abi():
  source = UOp.param(1, dtypes.float, (9,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:source.index(i).load().cast(dtypes.half), count=9))
  assert image is not None and len(image.gathers) == 2
  assert any(gather.itemsize == 4 and gather.count == 9 and not gather.values for gather in image.gathers)
  converters = [op for op in image.ew_ops if op.ew_cfg & _EW_STAGE_FP32_IN]
  assert [op.count for op in converters] == [4, 4, 1]
  assert any(gather.count == 9 and gather.itemsize == 2 for gather in image.mid_gathers)
  assert decode_image(encode_image(image)) == image


def test_fp32_constant_uses_canonical_half_value_before_output_conversion():
  image = _lower_uop_program(_program(dtypes.float, lambda _i:UOp.const(4.0, dtypes.float), count=6))
  assert image is not None and image.constants == struct.pack("<e", 4.0)
  assert [op.count for op in image.ew_ops] == [4, 2]
  assert all(op.ew_cfg & _EW_STAGE_FP32_OUT for op in image.ew_ops)


def test_infinite_numerator_fdiv_preserves_dynamic_denominator_sign():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    UOp(Ops.FDIV, dtypes.half, src=(UOp.const(math.inf, dtypes.half), source.index(i).load()))))
  assert image is not None and len(image.ew_ops) == 3
  assert all(op.ew_cfg == _EW_CFG[Ops.FDIV] for op in image.ew_ops[:2])
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_generic_where_owns_ternary_arity():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left, right)
  image = _lower_uop_program(_program(dtypes.half, select))
  assert image is not None
  assert any(op.compare or op.ew_cfg == _EW_CFG[Ops.MAX] for op in image.ew_ops)
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG and image.ew_ops[-1].dst.index == 0


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
      records.append((hashlib.sha256(blob).hexdigest(), len(blob), len(image.ew_ops), len(image.mid_gathers)))
  assert records == [("3f3c1465a2e4970839a8a15675d53f56e2f58afa793b36d08ac556b3f8c85533",5954,123,16),
                     ("7230b73ee7b39133536d38615b32f316fd80ad49aaee468de3678845baab446e",234,4,0)]


def test_static_nested_load_default_materializes_as_ordered_partial_gathers():
  fallback, selected = UOp.param(1, dtypes.half, (6,)), UOp.param(2, dtypes.half, (6,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    selected.index(i).load(fallback.index(i).load(), i < UOp.const(3, dtypes.int)), count=6))
  assert image is not None and len(image.gathers) == 2
  assert not image.gathers[0].partial and image.gathers[1].partial
  assert image.gathers[0].src_index == 1 and image.gathers[1].src_index == 2
  assert image.gathers[1].offsets == (0, 1, 2, -1, -1, -1)
  assert decode_image(encode_image(image)) == image


def test_bitcast_and_int16_masks_preserve_raw_fp16_sign_and_payload():
  magnitude, sign = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    ((magnitude.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.max, dtypes.int16)) |
     (sign.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.min, dtypes.int16))).bitcast(dtypes.half)))
  assert image is not None and len(image.ew_ops) == 11 and len(image.mid_gathers) == 10
  assert len(image.post_gathers) == 1 and image.post_gathers[0].itemsize == 2
  assert decode_image(encode_image(image)) == image


def test_production_fp16_pair_bitcast_remains_zero_kernel_movement():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP", 24, dtypes.half, num=1001)).reshape(2,3,4)
    assert not source.bitcast(dtypes.int).schedule_linear().src


def test_zero_count_raw_fp16_bitcast_uses_empty_generic_image():
  lhs, rhs = UOp.param(1, dtypes.half, (0,)), UOp.param(2, dtypes.half, (0,))
  def packed(i):
    low = lhs.index(i).load().bitcast(dtypes.ushort).cast(dtypes.uint)
    high = rhs.index(i).load().bitcast(dtypes.ushort).cast(dtypes.uint).alu(Ops.SHL, UOp.const(16, dtypes.int))
    return (low + high).bitcast(dtypes.int)
  image = _lower_uop_program(_program(dtypes.int, packed, count=0))
  assert image is not None and not image.gathers and not image.ew_ops and len(encode_image(image)) == 40


def test_generic_bool_where_uses_canonical_int16_ternary():
  lhs, rhs = UOp.param(1, dtypes.int, (4,)), UOp.param(2, dtypes.int, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left != UOp.const(0, dtypes.int), UOp.const(False, dtypes.bool))
  image = _lower_uop_program(_program(dtypes.bool, select))
  assert image is not None and image.ew_ops[-1].int16_output
  assert len(image.post_gathers) == 1 and image.post_gathers[0].itemsize == 1


def test_inverted_fp16_comparison_keeps_ieee_unordered_semantics():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def greater_equal(i):
    less = UOp(Ops.CMPLT, dtypes.bool, src=(lhs.index(i).load(), rhs.index(i).load()))
    return UOp(Ops.CMPNE, dtypes.bool, src=(less, UOp.const(True, dtypes.bool)))
  image = _lower_uop_program(_program(dtypes.bool, greater_equal))
  assert image is not None and len(image.ew_ops) > 10
  assert image.post_gathers and image.post_gathers[-1].itemsize == 1
  assert not any(op.compare for op in image.ew_ops)
  assert image.ew_ops[-1].ew_cfg == _EW_CFG[Ops.MUL] and image.ew_ops[-1].int16_output


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
  assert image is not None and image.mid_gathers and image.post_gathers
  assert not any(op.compare for op in image.ew_ops) and all(op.int16_input and op.int16_output for op in image.ew_ops)


def test_generic_where_selects_infinity_without_mask_multiplication():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(-math.inf, dtypes.half))))
  assert image is not None and not image.ew_ops and len(image.post_gathers) == 2


def test_max_uses_finite_neutral_for_selected_negative_infinity():
  source = UOp.param(1, dtypes.half, (4,))
  def maximum(i):
    selected = (i < UOp.const(3, dtypes.int)).where(UOp.const(-math.inf, dtypes.half), source.index(i).load())
    return selected.maximum(UOp.const(-2.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, maximum))
  assert image is not None and struct.pack("<e", -65504.0) in image.constants
  assert len(image.mid_gathers) == 6 and len({gather.after for gather in image.mid_gathers}) == 2


def test_generic_where_predicates_nonfinite_exp2_input():
  source = UOp.param(1, dtypes.half, (4,))
  def power(i):
    exponent = UOp.const(-math.inf, dtypes.half) * source.index(i).load()
    return (source.index(i).load() != UOp.const(0.0, dtypes.half)).where(exponent.exp2(), UOp.const(1.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, power))
  assert image is not None and len(image.ew_ops) > 10


def test_generic_where_materializes_nan_only_on_selected_lanes():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    (source.index(i).load() < UOp.const(0.0, dtypes.half)).where(UOp.const(math.nan, dtypes.half), source.index(i).load())))
  assert image is not None and image.mid_gathers and not any(op.compare or op.submit_barrier for op in image.ew_ops)


def test_nested_where_around_math_preserves_raw_uop_selection():
  base, exponent = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def power(i):
    x, y = base.index(i).load(), exponent.index(i).load()
    absolute = (x < UOp.const(0.0, dtypes.half)).where(-x, x)
    magnitude = (absolute.log2() * y).exp2()
    invalid = (y != y.cast(dtypes.int).cast(dtypes.half)).where(UOp.const(math.nan, dtypes.half), magnitude)
    return (x < UOp.const(0.0, dtypes.half)).where(invalid, magnitude)
  image = _lower_uop_program(_program(dtypes.half, power))
  assert image is not None and len(image.mid_gathers) >= 6
  assert len({gather.after for gather in image.mid_gathers}) >= 2
  assert decode_image(encode_image(image)) == image


def test_generic_where_abs_recipe_avoids_infinite_arm_blend():
  source = UOp.param(1, dtypes.half, (4,))
  def absolute(i):
    value = source.index(i).load()
    return (value < UOp.const(0.0, dtypes.half)).where(value * UOp.const(-1.0, dtypes.half), value)
  image = _lower_uop_program(_program(dtypes.half, absolute))
  assert image is not None and len(image.ew_ops) == 1 and image.ew_ops[-1].ew_cfg == _EW_CFG_ABS
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_threshold_where_uses_bounded_selection_for_dynamic_infinity():
  source = UOp.param(1, dtypes.half, (4,))
  def selected(i):
    value = source.index(i).load()
    return (value < UOp.const(0.0, dtypes.half)).where(value, UOp.const(1.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, selected))
  assert image is not None and any(op.ew_cfg == _EW_CFG[Ops.MAX] for op in image.ew_ops)


def test_shifted_relu_difference_becomes_bounded_cap():
  source = UOp.param(1, dtypes.half, (4,))
  def bounded(i):
    scaled = source.index(i).load() * UOp.const(1/6, dtypes.half)
    lower, upper, zero = scaled + UOp.const(0.5, dtypes.half), scaled + UOp.const(-0.5, dtypes.half), UOp.const(0.0, dtypes.half)
    return (zero < lower).where(lower, zero) + (zero < upper).where(upper, zero) * UOp.const(-1.0, dtypes.half)
  image = _lower_uop_program(_program(dtypes.half, bounded))
  assert image is not None and len(image.ew_ops) < 10
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_where_abs_remains_native_inside_math_recipe():
  source = UOp.param(1, dtypes.half, (4,))
  def logarithm(i):
    value = source.index(i).load()
    absolute = (value < UOp.const(0.0, dtypes.half)).where(value * UOp.const(-1.0, dtypes.half), value)
    return absolute.log2()
  image = _lower_uop_program(_program(dtypes.half, logarithm))
  assert image is not None and any(op.ew_cfg == _EW_CFG_ABS for op in image.ew_ops)


def test_where_abs_recognizes_negated_reciprocal_arms():
  source = UOp.param(1, dtypes.half, (4,))
  def power_magnitude(i):
    value = source.index(i).load()
    reciprocal = UOp(Ops.FDIV, dtypes.half, src=(UOp.const(1.0, dtypes.half), value))
    negative = UOp(Ops.FDIV, dtypes.half, src=(UOp.const(-1.0, dtypes.half), value))
    absolute = (reciprocal < UOp.const(0.0, dtypes.half)).where(negative, reciprocal)
    return (absolute.log2() * UOp.const(0.3, dtypes.half)).exp2()
  image = _lower_uop_program(_program(dtypes.half, power_magnitude))
  assert image is not None and any(op.ew_cfg == _EW_CFG_ABS for op in image.ew_ops)


def test_generic_static_index_becomes_gather():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:source.index(3-i).load()))
  assert image is not None and len(image.gathers) == 1
  assert image.gathers[0].base == 3 and image.gathers[0].axes == ((1, 4, -1),)


def test_max_materializes_negative_infinity_fill_as_finite_neutral():
  source = UOp.param(1, dtypes.half, (4,))
  def padded(i):
    value = source.index(i).load(UOp.const(-math.inf, dtypes.half), i < UOp.const(3, dtypes.int))
    return value.maximum(UOp.const(0.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, padded))
  assert image is not None and image.gathers[0].fill_bits == 0xfbff


def test_guarded_load_with_infinite_fill_falls_through_dynamic_address_probes():
  source = UOp.param(1, dtypes.half, (2,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    source.index(i).load(UOp.const(math.inf, dtypes.half), i < UOp.const(2, dtypes.int))))
  assert image is not None and any(gather.fill_bits == 0x7c00 for gather in image.gathers)


def test_static_root_where_uses_exact_gathers_and_finite_padding_neutral():
  source = UOp.param(1, dtypes.half, (3,))
  def selected(i):
    padded = source.index(i).load(UOp.const(-math.inf, dtypes.half), i < UOp.const(3, dtypes.int))
    return (i < UOp.const(4, dtypes.int)).where(padded, UOp.const(0.0, dtypes.half))
  image = _lower_uop_program(_program(dtypes.half, selected))
  assert image is not None and not image.ew_ops and len(image.post_gathers) == 2
  assert any(gather.fill_bits == 0xfbff for gather in image.gathers)


def test_static_root_where_preserves_nonzero_constant_route():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(3.5, dtypes.half))))
  assert image is not None and not image.ew_ops and len(image.post_gathers) == 2
  assert struct.pack("<e", 3.5) in image.constants


def test_generic_bool_store_has_explicit_boundary_conversion():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() < rhs.index(i).load()))
  assert image is not None
  assert image.post_gathers and image.post_gathers[-1].itemsize == 1
  assert not any(op.compare for op in image.ew_ops)


def test_generic_int16_uses_canonical_native_layout():
  source = UOp.param(1, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int16, lambda i:source.index(i).load() + UOp.const(3, dtypes.int16)))
  assert image is not None and len(image.ew_ops) == 1
  assert image.ew_ops[0].int16_input and image.ew_ops[0].int16_output


def test_generic_int16_complement_recipe_composes_with_max():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  def minimum(i):
    left, right, complement = lhs.index(i).load(), rhs.index(i).load(), UOp.const(-1, dtypes.int16)
    inverted_left = UOp(Ops.XOR, dtypes.int16, src=(left, complement))
    inverted_right = UOp(Ops.XOR, dtypes.int16, src=(right, complement))
    return UOp(Ops.XOR, dtypes.int16, src=(inverted_left.maximum(inverted_right), complement))
  image = _lower_uop_program(_program(dtypes.int16, minimum))
  assert image is not None and len(image.ew_ops) == 4
  assert all(op.int16_input and op.int16_output for op in image.ew_ops)


def test_generic_int16_where_avoids_saturating_difference():
  source = UOp.param(1, dtypes.int16, (4,))
  def clipped(i):
    value = source.index(i).load()
    return (value < UOp.const(100, dtypes.int16)).where(value, UOp.const(-100, dtypes.int16))
  image = _lower_uop_program(_program(dtypes.int16, clipped))
  assert image is not None
  assert len(image.ew_ops) == 7
  assert all(op.int16_input and op.int16_output for op in image.ew_ops)


def test_static_bool_materializes_in_int16_consumer_layout():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int16,
    lambda i:(i < UOp.const(2, dtypes.int)).where(lhs.index(i).load(), rhs.index(i).load())))
  assert image is not None and not image.ew_ops and len(image.post_gathers) == 2
  assert all(gather.itemsize == 2 for gather in image.post_gathers)


def test_int16_to_int32_is_an_explicit_output_boundary():
  lhs, rhs = UOp.param(1, dtypes.int16, (4,)), UOp.param(2, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int,
    lambda i:(lhs.index(i).load() + rhs.index(i).load()).cast(dtypes.int)))
  assert image is not None and len(image.ew_ops) == 2
  assert image.ew_ops[-1].int16_input and image.ew_ops[-1].int32_output


def test_bounded_int_root_keeps_dynamic_int32_load_in_canonical_layout():
  out, source = UOp.param(0, dtypes.int, (18,)), UOp.param(1, dtypes.int, (3,))
  row = UOp.range(3, 1)
  cls = UOp.range(6, 0, src=(row,))
  different = source.index(row).load() != cls
  value = different.where(UOp.const(0, dtypes.int), UOp.const(1, dtypes.int))
  image = _lower_uop_program(list(out.index(row*6+cls).store(value).end(cls, row).sink().toposort()))
  assert image is not None and image.mid_gathers and image.ew_ops[-1].int32_output


def test_bounded_semantic_int_does_not_alias_int32_output_before_widening():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    (UOp.const(100.0, dtypes.half) < source.index(i).load()).cast(dtypes.int) * UOp.const(2500, dtypes.int)))
  assert image is not None and image.ew_ops[-1].int16_input and image.ew_ops[-1].int32_output
  assert all(op.dst.kind is RKBufferKind.SCRATCH for op in image.ew_ops[:-1])


def test_int32_load_store_is_raw_four_byte_materialization():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load()))
  assert image is not None and not image.ew_ops and len(image.post_gathers) == 1
  assert image.post_gathers[0].itemsize == 4 and image.post_gathers[0].dst_kind is RKBufferKind.ARG


def test_native_int32_mul_uses_the_canonical_wide_layout():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load() * source.index(i).load()))
  assert image is not None and len(image.ew_ops) == 1
  assert image.ew_ops[0].int32_input and image.ew_ops[0].int32_output


def test_int32_where_constants_convert_at_the_output_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    (UOp.const(0.5, dtypes.half) < source.index(i).load()).where(UOp.const(4, dtypes.int), UOp.const(2, dtypes.int))))
  assert image is not None and len(image.ew_ops) > 3
  assert image.ew_ops[-1].int32_output and image.ew_ops[-1].int16_input


def test_math_uops_own_multi_stage_recipes():
  source = UOp.param(1, dtypes.half, (4,))
  for op in (Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN):
    image = _lower_uop_program(_program(dtypes.half, lambda i, op=op:UOp(op, dtypes.half, src=(source.index(i).load(),))))
    assert image is not None and len(image.ew_ops) > 1
    assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_generic_sign_recipe_owns_tagged_semantics():
  source = UOp.param(1, dtypes.half, (4,))
  def sign(i):
    value = source.index(i).load()
    return UOp(Ops.SUB, dtypes.half, src=(value, value), arg=_NATIVE_SIGN)
  image = _lower_uop_program(_program(dtypes.half, sign))
  assert image is not None and len(image.ew_ops) == 4
  assert sum(op.compare for op in image.ew_ops) == 2


def test_unrolled_math_reduction_vectorizes_periodic_indices():
  out = UOp.param(0, dtypes.half, (1,))
  lhs, rhs, weights = (UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.half, (8,)),
                       UOp.param(3, dtypes.half, (2,)))
  terms = [lhs.index(i).load().exp2() * rhs.index(i).load() * weights.index(i%2).load() for i in range(8)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(0).store(value).sink().toposort()))
  assert image is not None and image.cmac is None and image.ew_ops
  assert image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters
  assert decode_image(encode_image(image)) == image


def test_batched_unrolled_math_reduction_materializes_each_uop_result():
  rows, groups = 8, 4
  out, source = UOp.param(0, dtypes.half, (rows,)), UOp.param(1, dtypes.half, (rows*groups,))
  normalizer, lane = UOp.param(2, dtypes.half, (rows,)), UOp.range(rows, 0)
  terms = [(source.index(lane*groups+k).load() - normalizer.index(lane).load()).exp2() for k in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.cmac is None and image.ew_ops
  assert image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters
  assert decode_image(encode_image(image)) == image


def test_static_reduce_uops_are_structurally_executed():
  for op in (Ops.ADD, Ops.MAX, Ops.MUL):
    out, source = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,))
    row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
    term = source.index(row*3+axis).load()
    reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(op,))
    image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
    assert image is not None
    if op is Ops.ADD: assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,1,3)
    else: assert image.cmac is None and len(image.gathers) == 3 and len(image.ew_ops) == 2

def test_multi_axis_reduce_and_fp32_local_add_share_cmac_unrolling():
  out, source = UOp.param(0,dtypes.half,(2,)), UOp.param(1,dtypes.half,(12,))
  row,outer,inner = UOp.range(2,0),UOp.range(2,1,AxisType.REDUCE),UOp.range(3,2,AxisType.REDUCE)
  term = source.index(row*6+outer*3+inner).load()
  reduced = UOp(Ops.REDUCE,dtypes.float,src=(term.cast(dtypes.float),outer,inner),arg=(Ops.ADD,0))
  direct = _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).end(row,outer,inner).sink().toposort()))
  local = UOp.placeholder((1,),dtypes.float,0,addrspace=AddrSpace.REG)
  initialize = local.index(0).store(0.0)
  pointer = local.after(initialize,outer,inner).index(0)
  update = pointer.store(pointer.load()+term.cast(dtypes.float))
  result = local.after(update.end(outer,inner)).index(0).load().cast(dtypes.half)
  loop = _lower_uop_program(list(UOp.sink(initialize,update,out.index(row).store(result)).toposort()))
  for image in (direct,loop):
    assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,1,6)
    assert not image.ew_ops and decode_image(encode_image(image)) == image
    assert image.gathers[0].offsets[:6] == tuple(range(6)) and image.gathers[0].offsets[32:38] == tuple(range(6,12))
  assert encode_image(direct) == encode_image(loop)

def test_noopt_multi_axis_tensor_sum_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=1):
    source = Tensor(UOp.new_buffer("ROCKCHIP",24,dtypes.half,num=1005).reshape((2,3,4)))
    ast = source.sum((0,2)).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (3,1,8)
  assert not image.ew_ops and image.execution_class is RKExecutionClass.NATIVE

def test_scalar_half_to_float_sum_routes_production_cmac():
  values = np.random.RandomState(0).uniform(-2, 2, size=(45, 3)).astype(np.float16).reshape(-1)
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP", values.size, dtypes.half, num=1001).reshape((45, 3)))
    output = source.sum(dtype=dtypes.float32)
    ast = output.schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k,image.cmac.out_fp16) == (1,1,values.size,False)
  assert not image.constants and len(image.scratch) == 3 and len(image.gathers) == 2 and len(image.post_gathers) == 1
  assert _cmac_weights(image) == (rockchip_renderer._fp16_bits(1.0),)*values.size
  assert image.gathers[0].offsets[:values.size] == tuple(range(values.size))
  assert all(offset == -1 for offset in image.gathers[0].offsets[values.size:])
  assert image.post_gathers[0].itemsize == 4 and image.post_gathers[0].offsets == (0,)
  assert len(emit_cmac_stage(image.cmac).commands) == 45 and decode_image(encode_image(image)) == image
  assert image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters


def test_real_matmul_routes_production_cmac_and_packs_the_output_surface():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP", 15, dtypes.half, num=1002).reshape((3,5)))
    rhs = Tensor(UOp.new_buffer("ROCKCHIP", 20, dtypes.half, num=1003).reshape((5,4)))
    ast = (lhs@rhs).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (3,4,5)
  assert len(image.gathers) == 2 and len(image.post_gathers) == 1 and not image.ew_ops
  assert image.post_gathers[0].offsets == tuple(row*64+col for row in range(3) for col in range(4))
  assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE
  lhs_values, rhs_values = np.arange(15, dtype=np.float16), np.arange(20, dtype=np.float16).reshape(5,4)
  sources = {image.gathers[0].src_index:lhs_values.view(np.uint16), image.gathers[1].src_index:rhs_values.reshape(-1).view(np.uint16)}
  scratch = [np.zeros(spec.size//2, dtype=np.uint16) for spec in image.scratch]
  for gather in image.gathers:
    destination, offsets = scratch[gather.dst_index], np.asarray(gather.offsets)
    valid = offsets >= 0
    destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src_index][offsets[valid]]
  packed_lhs = scratch[image.cmac.lhs.index].view(np.float16).reshape(3,32)
  packed_rhs = scratch[image.cmac.rhs.index].view(np.float16).reshape(2,1,16,32).transpose(0,2,1,3).reshape(32,32)
  np.testing.assert_array_equal(packed_lhs[:,:5], lhs_values.reshape(3,5))
  np.testing.assert_array_equal(packed_rhs[:4,:5], rhs_values.T)
  np.testing.assert_array_equal(packed_lhs[:,:5].astype(np.float32)@packed_rhs[:4,:5].T.astype(np.float32),
                                lhs_values.reshape(3,5).astype(np.float32)@rhs_values.astype(np.float32))


def test_real_matmul_relu_routes_one_production_cmac_stage():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    lhs = Tensor(UOp.new_buffer("ROCKCHIP", 15, dtypes.half, num=1030).reshape((3,5)))
    rhs = Tensor(UOp.new_buffer("ROCKCHIP", 20, dtypes.half, num=1031).reshape((5,4)))
    ast = (lhs@rhs).relu().schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast, RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and image.cmac.relu and (image.cmac.m,image.cmac.n,image.cmac.k) == (3,4,5)
  assert not image.ew_ops and len(emit_cmac_stage(image.cmac).commands) == 45
  lhs_values = (np.arange(15)-7).astype(np.float16).reshape(3,5)
  rhs_values = (np.arange(20)%5-2).astype(np.float16).reshape(5,4)
  sources = {image.gathers[0].src_index:lhs_values.view(np.uint16).reshape(-1),
             image.gathers[1].src_index:rhs_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(spec.size//2,dtype=np.uint16) for spec in image.scratch]
  for gather in image.gathers:
    offsets = np.asarray(gather.offsets)
    valid = offsets >= 0
    scratch[gather.dst_index][:gather.count] = gather.fill_bits
    scratch[gather.dst_index][:gather.count][valid] = sources[gather.src_index][offsets[valid]]
  packed_lhs = scratch[image.cmac.lhs.index].view(np.float16).reshape(3,32)
  packed_rhs = scratch[image.cmac.rhs.index].view(np.float16).reshape(2,1,16,32).transpose(0,2,1,3).reshape(32,32)
  physical = np.zeros(3*32*2,dtype=np.float16)
  physical.reshape(3,64)[:,:4] = np.maximum(packed_lhs.astype(np.float32)@packed_rhs.astype(np.float32).T,0)[:,:4].astype(np.float16)
  np.testing.assert_array_equal(physical[np.asarray(image.post_gathers[0].offsets)],
                                np.maximum(lhs_values.astype(np.float32)@rhs_values.astype(np.float32),0).astype(np.float16).reshape(-1))


def test_zero_gated_padded_convolution_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",324,dtypes.half,num=1012)).reshape(1,4,9,9)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",144,dtypes.half,num=1013)).reshape(4,4,3,3)
    ast = source.conv2d(weight,padding=1).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (4,81,36)
  assert not image.ew_ops and len(image.gathers) == 2 and any(offset < 0 for gather in image.gathers for offset in gather.offsets)
  source_values = (np.arange(324)%5-2).astype(np.float16).reshape(1,4,9,9)
  weight_values = (np.arange(144)%5-2).astype(np.float16).reshape(4,4,3,3)
  sources = {1:source_values.view(np.uint16).reshape(-1),2:weight_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(spec.size//2,dtype=np.uint16) for spec in image.scratch]
  for gather in image.gathers:
    destination,offsets = scratch[gather.dst_index],np.asarray(gather.offsets)
    valid = offsets >= 0
    if not gather.partial: destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src_index][offsets[valid]]
  packed_weight = scratch[image.cmac.lhs.index].view(np.float16).reshape(4,96)
  packed_source = scratch[image.cmac.rhs.index].view(np.float16).reshape(6,3,16,32).transpose(0,2,1,3).reshape(96,96)
  expected = np.asarray([[sum(float(weight_values[oc,ic,ky,kx])*float(source_values[0,ic,y+ky-1,x+kx-1])
    for ic in range(4) for ky in range(3) for kx in range(3) if 0 <= y+ky-1 < 9 and 0 <= x+kx-1 < 9)
    for y in range(9) for x in range(9)] for oc in range(4)],dtype=np.float32)
  np.testing.assert_array_equal(packed_weight[:,:36].astype(np.float32)@packed_source[:81,:36].T.astype(np.float32),expected)
  assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE
  out,lhs,rhs = UOp.param(0,dtypes.half,(1,)),UOp.param(1,dtypes.half,(4,)),UOp.param(2,dtypes.half,(4,))
  for fill in (-0.0,1.0):
    terms = [lhs.index(UOp.const(-1,dtypes.int)).load(UOp.const(fill,dtypes.half),UOp.const(False,dtypes.bool))*rhs.index(i).load()
             for i in range(4)]
    fallback = _lower_uop_program(list(out.index(0).store(functools.reduce(lambda total,term:total+term,terms[1:],terms[0])).sink().toposort()))
    assert fallback is not None and fallback.cmac is None and fallback.ew_ops


def test_batched_zero_gated_convolution_reorders_one_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",648,dtypes.half,num=1014)).reshape(2,4,9,9)
    weight = Tensor(UOp.new_buffer("ROCKCHIP",144,dtypes.half,num=1015)).reshape(4,4,3,3)
    ast = source.conv2d(weight,padding=1).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (4,162,36)
  assert not image.ew_ops and len(image.gathers) == 2
  source_values = (np.arange(648)%5-2).astype(np.float16).reshape(2,4,9,9)
  weight_values = (np.arange(144)%5-2).astype(np.float16).reshape(4,4,3,3)
  sources = {1:source_values.view(np.uint16).reshape(-1),2:weight_values.view(np.uint16).reshape(-1)}
  scratch = [np.zeros(spec.size//2,dtype=np.uint16) for spec in image.scratch]
  for gather in image.gathers:
    destination,offsets = scratch[gather.dst_index],np.asarray(gather.offsets)
    valid = offsets >= 0
    if not gather.partial: destination[:gather.count] = gather.fill_bits
    destination[:gather.count][valid] = sources[gather.src_index][offsets[valid]]
  packed_weight = scratch[image.cmac.lhs.index].view(np.float16).reshape(4,192)
  packed_source = scratch[image.cmac.rhs.index].view(np.float16).reshape(12,6,16,32).transpose(0,2,1,3).reshape(192,192)
  expected = np.asarray([[[[sum(float(weight_values[oc,ic,ky,kx])*float(source_values[batch,ic,y+ky-1,x+kx-1])
    for ic in range(4) for ky in range(3) for kx in range(3) if 0 <= y+ky-1 < 9 and 0 <= x+kx-1 < 9)
    for x in range(9)] for y in range(9)] for oc in range(4)] for batch in range(2)],dtype=np.float32)
  np.testing.assert_array_equal(packed_weight[:,:36].astype(np.float32)@packed_source[:162,:36].T.astype(np.float32),
                                expected.transpose(1,0,2,3).reshape(4,162))
  assert image.post_gathers[0].offsets == tuple(oc*384+(batch*81+lane)//16*32+(batch*81+lane)%16
    for batch in range(2) for oc in range(4) for lane in range(81))
  assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE


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
    assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,2,4)
    assert not image.ew_ops and len(image.gathers) == 4 and sum(bool(gather.values) for gather in image.gathers) == 1
    sources = {1:lhs_values.view(np.uint16).reshape(-1),2:rhs_values.view(np.uint16).reshape(-1),
               3:bias_values.view(np.uint16).reshape(-1)}
    scratch = [np.zeros(spec.size//2,dtype=np.uint16) for spec in image.scratch]
    for gather in image.gathers:
      destination,offsets = scratch[gather.dst_index],np.asarray(gather.offsets)
      if gather.values: destination[:gather.count] = gather.values
      else:
        valid = offsets >= 0
        if not gather.partial: destination[:gather.count] = gather.fill_bits
        destination[:gather.count][valid] = sources[gather.src_index][offsets[valid]]
    packed_lhs = scratch[image.cmac.lhs.index].view(np.float16).reshape(2,32)
    packed_rhs = scratch[image.cmac.rhs.index].view(np.float16).reshape(2,1,16,32).transpose(0,2,1,3).reshape(32,32)
    np.testing.assert_array_equal(packed_lhs[:,:4].astype(np.float32)@packed_rhs[:2,:4].T.astype(np.float32),
                                  lhs_values.astype(np.float32)@rhs_values.astype(np.float32)+bias_values.astype(np.float32))
    assert image.execution_class is RKExecutionClass.NATIVE and decode_image(encode_image(image)) == image


def test_literal_fp32_bias_and_large_broadcast_share_cmac_candidate_planner():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",64,dtypes.half,num=1026)).reshape(64,1).expand(64,64)
    ast = (source.cast(dtypes.float)+3.0).cast(dtypes.half).schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (128,32,2)
  assert not image.ew_ops and len(image.gathers) == 3 and sum(bool(gather.values) for gather in image.gathers) == 2
  source_values = np.arange(64,dtype=np.float16)
  scratch = [np.zeros(spec.size//2,dtype=np.uint16) for spec in image.scratch]
  for gather in image.gathers:
    destination = scratch[gather.dst_index]
    if gather.values: destination[:gather.count] = gather.values
    else:
      offsets = np.asarray(gather.offsets)
      valid = offsets >= 0
      if not gather.partial: destination[:gather.count] = gather.fill_bits
      destination[:gather.count][valid] = source_values.view(np.uint16)[offsets[valid]]
  ai,ao,_ = rockchip_renderer._cmac_layout(image.cmac.n,image.cmac.k)
  lhs = scratch[image.cmac.lhs.index].view(np.float16).reshape(image.cmac.m,ai)
  rhs = scratch[image.cmac.rhs.index].view(np.float16).reshape(ao//16,ai//32,16,32).transpose(0,2,1,3).reshape(ao,ai)
  result = (lhs.astype(np.float32)@rhs.astype(np.float32).T).astype(np.float16)
  physical = np.zeros(image.cmac.m*ao*2,dtype=np.float16)
  for row in range(image.cmac.m):
    for col in range(image.cmac.n): physical[row*ao*2+col//16*32+col%16] = result[row,col]
  got = physical[np.asarray(image.post_gathers[0].offsets)]
  expected = (source_values[:,None]+np.float16(3)).repeat(64,axis=1).reshape(-1)
  np.testing.assert_array_equal(got,expected)
  assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE


def test_tensor_mean_routes_scaled_production_cmac_weights():
  with Context(DEV="ROCKCHIP", DEFAULT_FLOAT="HALF", NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",8,dtypes.half,num=1004))
    ast = source.mean().schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  images = [decode_image(u.arg) for u in program.src if u.op is Ops.BINARY]
  image = next(image for image in images if image.cmac is not None)
  assert (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,8)
  assert not image.constants and len(image.scratch) == 3 and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*8
  assert image.cmac.out_fp16 and not image.ew_ops
  assert image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters


def test_arange_weighted_tensor_sum_routes_production_cmac():
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source = Tensor(UOp.new_buffer("ROCKCHIP",5,dtypes.half,num=1006))
    ast = (source*Tensor.arange(1,6).cast(dtypes.half)).sum().schedule_linear().src[0].src[0]
    to_program_cache.clear()
    program = to_program(ast,RockchipRenderer(Target(device="ROCKCHIP")))
  image = decode_image(next(u for u in program.src if u.op is Ops.BINARY).arg)
  assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,5)
  assert image.gathers[0].offsets[:5] == tuple(range(5))
  assert _cmac_weights(image) == tuple(rockchip_renderer._fp16_bits(value) for value in range(1,6))
  assert not image.constants and not image.ew_ops and image.execution_class is RKExecutionClass.NATIVE


def test_static_dot_reduce_owns_accurate_physical_recipe():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,)), UOp.param(2, dtypes.half, (6,))
  row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
  term = lhs.index(row*3+axis).load() * rhs.index(row*3+axis).load()
  reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(Ops.ADD,))
  image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
  assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,2,3)
  assert image.post_gathers[0].offsets == (0, 65) and not image.ew_ops


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
  assert image is not None and image.cmac is None and image.ew_ops
  assert image.ew_ops[-1].dst == RKArg(RKBufferKind.ARG, 0)
  assert image.execution_class is RKExecutionClass.NATIVE and decode_image(encode_image(image)) == image


def test_generic_image_allows_many_small_ew_stages():
  count = 1080
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.int, (count,)), UOp.param(2, dtypes.int, (count,))
  value = UOp.const(0.0, dtypes.half)
  for index in range(count):
    value = value + (lhs.index(index).load() < rhs.index(index).load()).where(
      UOp.const(1.0, dtypes.half), UOp.const(0.0, dtypes.half))
  uops = list(out.index(0).store(value).sink().toposort())
  image = _lower_uop_program(uops)
  assert image is not None and len(image.ew_ops) > _RKIMAGE_U16_MAX > _MAX_EW_ELEMS_FP16
  assert decode_image(encode_image(image)) == image


def test_short_fp32_add_mul_tree_routes_cmac_at_output_boundary():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,2,2) and not image.ew_ops


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
        assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k,image.cmac.out_fp16) == (1,1,k,fp16)
        assert not image.ew_ops and tuple(gather.src_index for gather in image.gathers if not gather.values) == (1,2)
        assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE


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
  assert image is not None and len(image.ew_ops) == len(image.mid_gathers) == 3
  assert tuple(gather.dst_addend for gather in image.mid_gathers) == (0, 8, 16)
  assert all(op.ew_cfg & _EW_STAGE_FP32_OUT and op.dst.kind is RKBufferKind.ARG for op in image.ew_ops)
  assert decode_image(encode_image(image)) == image


def test_terminal_int_to_float_cast_composes_integer_and_fp32_converters():
  source = UOp.param(1, dtypes.int, (9,))
  image = _lower_uop_program(_program(dtypes.float, lambda i:source.index(i).load().cast(dtypes.float), count=9))
  assert image is not None and len(image.ew_ops) == 4
  assert sum(bool(op.ew_cfg & _EW_STAGE_FP32_OUT) for op in image.ew_ops) == 3
  assert decode_image(encode_image(image)) == image


def test_remapped_integer_and_bool_to_float_casts_use_generic_typed_values():
  for dtype,stages in ((dtypes.int, 4), (dtypes.bool, 9)):
    source = UOp.param(1, dtype, (9,))
    image = _lower_uop_program(_program(dtypes.float, lambda i:source.index(8-i).load().cast(dtypes.float), count=9))
    assert image is not None and len(image.ew_ops) == stages and decode_image(encode_image(image)) == image


def test_terminal_half_casts_use_typed_integer_and_bool_output_abis():
  source = UOp.param(1, dtypes.half, (9,))
  integer = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load().cast(dtypes.int), count=9))
  boolean = _lower_uop_program(_program(dtypes.bool, lambda i:source.index(i).load().cast(dtypes.bool), count=9))
  assert integer is not None and integer.ew_ops[-1].int32_output
  assert boolean is not None and boolean.ew_ops[-1].bool_output
  assert decode_image(encode_image(integer)) == integer and decode_image(encode_image(boolean)) == boolean


def test_fp32_pure_add_tree_routes_cmac_at_the_half_output_boundary():
  source = UOp.param(1, dtypes.half, (64,))
  terms = [source.index(i).load().cast(dtypes.float) for i in range(64)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(_program(dtypes.half, lambda _i:value.cast(dtypes.half), count=1))
  assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,64)
  assert image.cmac.out_fp16 and not image.ew_ops and decode_image(encode_image(image)) == image


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
    assert image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,16) and not image.ew_ops
    dynamic = tuple(gather for gather in image.gathers if not gather.values)
    assert tuple(gather.src_index for gather in dynamic) == source_slots
    assert tuple(gather.partial for gather in dynamic) == tuple(i%2 == 1 for i in range(len(dynamic)))
    assert tuple(tuple(i for i,offset in enumerate(gather.offsets) if offset >= 0) for gather in dynamic) == \
      ((tuple(range(8)),tuple(range(8,16)))*2)[:len(dynamic)]
    assert all(tuple(offset for offset in gather.offsets if offset >= 0) == tuple(range(8)) for gather in dynamic)
    assert decode_image(encode_image(image)) == image and image.execution_class is RKExecutionClass.NATIVE


def test_multisource_cmac_resource_cap_charges_each_partial_surface(monkeypatch):
  sources = (UOp.param(1,dtypes.half,(8,)),UOp.param(2,dtypes.half,(8,)))
  terms = [source.index(i).load().cast(dtypes.float) for source in sources for i in range(8)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  monkeypatch.setattr(rockchip_renderer,"_MAX_DYNAMIC_SELECTOR_CELLS",1070)
  uops = _program(dtypes.half,lambda _i:value.cast(dtypes.half),count=1)
  assert (output:=rockchip_renderer._outs(uops)[1]) is not None and rockchip_renderer._lower_cmac_reduction(output,uops) is None


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
      assert image.cmac is not None and not image.constants and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*8
      assert not image.ew_ops
    else: assert image.cmac is None and image.ew_ops


def test_nested_scaled_direct_reduce_routes_cmac_but_scaled_dot_stays_generic():
  out, lhs, rhs = UOp.param(0,dtypes.half,(2,)),UOp.param(1,dtypes.half,(8,)),UOp.param(2,dtypes.half,(8,))
  row,axis = UOp.range(2,0),UOp.range(4,1,AxisType.REDUCE)
  for dot in (False,True):
    value = lhs.index(row*4+axis).load().cast(dtypes.float)
    if dot: value = value*rhs.index(row*4+axis).load().cast(dtypes.float)
    reduced = UOp(Ops.REDUCE,dtypes.float,src=(value,axis),arg=(Ops.ADD,))*0.5*0.25
    image = _lower_uop_program(list(out.index(row).store(reduced.cast(dtypes.half)).end(row,axis).sink().toposort()))
    assert image is not None
    if not dot:
      assert image.cmac is not None and not image.constants and _cmac_weights(image) == (rockchip_renderer._fp16_bits(0.125),)*4
      assert not image.ew_ops
    else: assert image.cmac is None and image.ew_ops


def test_nested_fp32_product_sum_is_committed_before_outer_half_add():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  bias = UOp.param(3, dtypes.half, (1,)).index(0).load()
  products = [(lhs.index(i).load() * rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  product_sum = products[0]
  for product in products[1:]: product_sum = product_sum + product
  image = _lower_uop_program(_program(dtypes.half, lambda _i:product_sum.cast(dtypes.half) + bias, count=1))
  assert image is not None and image.cmac is None and len(image.ew_ops) > 20 and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_independent_fp32_reductions_are_committed_before_outer_half_division():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  products = [(lhs.index(i).load() * rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  weights = [rhs.index(i).load().cast(dtypes.float) for i in range(4)]
  numerator, denominator = products[0], weights[0]
  for product,weight in zip(products[1:], weights[1:]): numerator, denominator = numerator+product, denominator+weight
  ratio = UOp(Ops.FDIV, dtypes.half, src=(numerator.cast(dtypes.half), denominator.cast(dtypes.half)))
  image = _lower_uop_program(_program(dtypes.half, lambda _i:ratio, count=1))
  assert image is not None and len(image.ew_ops) > 40 and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_fp32_math_uop_converts_at_half_storage_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:source.index(i).load().cast(dtypes.float).exp2().cast(dtypes.half)))
  assert image is not None and len(image.ew_ops) > 10


def test_trunc_uop_expands_to_native_floor_and_ceil_stages():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:UOp(Ops.TRUNC, dtypes.half, src=(source.index(i).load(),))))
  assert image is not None and any(op.ew_cfg == _EW_CFG_FLOOR for op in image.ew_ops)


def test_fp32_sin_additive_phase_reduces_terms_before_half_storage_boundary():
  source = UOp.param(1, dtypes.half, (4,))
  def shifted_sin(i):
    value = source.index(i).load().cast(dtypes.float)
    phase = UOp.const(math.pi/2, dtypes.float) + value * UOp.const(-1.0, dtypes.float)
    return UOp(Ops.SIN, dtypes.float, src=(phase,)).cast(dtypes.half)
  image = _lower_uop_program(_program(dtypes.half, shifted_sin))
  assert image is not None and sum(op.ew_cfg == _EW_CFG_FLOOR for op in image.ew_ops) >= 2
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG and decode_image(encode_image(image)) == image


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
  assert image is not None and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_fp32_boundary_activation_routes_terminal_cmac_relu():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).maximum(UOp.const(0.0, dtypes.float)).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.cmac is not None and image.cmac.relu and (image.cmac.m,image.cmac.n,image.cmac.k) == (2,2,2)
  assert not image.ew_ops and decode_image(encode_image(image)) == image


def test_terminal_minimum_is_not_misclassified_as_cmac_relu():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane, zero = UOp.range(2, 0), UOp.const(0.0, dtypes.half)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  reduced = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  value = (reduced < zero).where(reduced, zero)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.cmac is None and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


def test_static_local_accumulator_is_structurally_executed():
  for op,initial in ((Ops.ADD, 0.0), (Ops.MAX, -100.0), (Ops.MUL, 1.0)):
    out, source = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,))
    row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
    local = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG).index(0)
    initialize = local.store(initial)
    update = local.store(UOp(op, dtypes.half, src=(local.load(), source.index(row*3+axis).load())))
    output = out.index(row).store(local.load())
    image = _lower_uop_program(list(UOp.sink(initialize, update, output).toposort()))
    assert image is not None and len(image.gathers) == 3 and len(image.ew_ops) == 3


def test_static_local_unroll_preserves_range_order_dependencies():
  """Range AFTER edges remain semantic inputs to later static planning."""
  out = UOp.param(0, dtypes.half, (2,))
  dependency = UOp.range(4, 0, AxisType.WEAK)
  lane = UOp.range(2, 1, AxisType.WEAK, src=(dependency,))
  reduce_axis = UOp.range(3, 2, AxisType.REDUCE, src=(lane,))
  local = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG)
  initialize = local.index(0).store(0.0)
  update = local.after(initialize, reduce_axis).index(0).store(local.index(0).load()+reduce_axis.cast(dtypes.half))
  result = local.after(update.end(reduce_axis)).index(0).load()
  root = result + lane.cast(dtypes.half)*UOp.const(0.0, dtypes.half)
  uops = list(UOp.sink(initialize, update, out.index(lane).store(root)).toposort())
  expanded = _unroll_static_local(uops, root)
  assert dependency in expanded.toposort()
  assert any(node.key == lane.key and len(node.src) > 1 and node.src[1].key == dependency.key
             for node in expanded.toposort() if node.op is Ops.RANGE)


def _indexed_local_bridge_program(source_dtype, op:Ops, *, groups:int=2, workers:int=1, local_size:int=2, reduce:int=2, carrier=dtypes.bool):
  out, source = UOp.param(0, carrier, (groups,)), UOp.param(1, source_dtype, (groups*local_size*reduce,))
  group, worker = UOp.special(groups, "gidx0", dtypes.int), UOp.special(workers, "lidx0", dtypes.int)
  initial = (op is Ops.AND) if carrier is dtypes.bool else (-1 if op is Ops.AND else 0)
  first = UOp.placeholder((1,), carrier, 0, addrspace=AddrSpace.REG)
  first_init = first.index(0).store(initial)
  first_axis = UOp.range(reduce, 0, AxisType.REDUCE)
  first_ptr = first.after(first_init, first_axis).index(0)
  loaded = source.index(group*(local_size*reduce)+worker*reduce+first_axis).load()
  present = loaded != UOp.const(0.0, dtypes.half) if source_dtype is dtypes.half else loaded
  if carrier is not dtypes.bool: present = present.cast(carrier)
  def combine(lhs:UOp, rhs:UOp) -> UOp: return lhs & rhs if op is Ops.AND else lhs | rhs
  first_update = first_ptr.store(combine(first_ptr.load(), present))
  first_end = first_update.end(first_axis)
  bridge = UOp.placeholder((local_size,), carrier, 0, addrspace=AddrSpace.LOCAL)
  bridge_store = bridge.index(worker).store(first.after(first_end).index(0).load())
  second = UOp.placeholder((1,), carrier, 1, addrspace=AddrSpace.REG)
  second_init = second.index(0).store(initial)
  second_axis = UOp.range(local_size, 1, AxisType.REDUCE, src=(first_end,))
  second_ptr = second.after(second_init, second_axis).index(0)
  second_update = second_ptr.store(combine(second_ptr.load(), bridge.after(bridge_store.barrier()).index(second_axis).load()))
  result = second.after(second_update.end(second_axis)).index(0).load()
  output = out.index(group).store(result)
  return list(UOp.sink(first_init, first_update, bridge_store, second_init, second_update, output).toposort())

def test_indexed_local_bridge_and_boolean_accumulators_are_physically_executed():
  for source_dtype,op in ((dtypes.half, Ops.AND), (dtypes.half, Ops.OR), (dtypes.bool, Ops.AND), (dtypes.bool, Ops.OR)):
    for local_size,workers in ((1, 1), (2, 1), (4, 3), (4, 4), (4, 0)):
      image = _lower_uop_program(_indexed_local_bridge_program(source_dtype, op, workers=workers, local_size=local_size))
      if workers == 0:
        assert image is None
        continue
      assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
      values = []
      for group in range(2):
        for lane in range(local_size):
          for _ in range(2):
            values.append((lane >= workers) if op is Ops.OR and group == 0 else
                          (lane < workers) if op is Ops.AND and group == 0 else
                          (lane == 0) if op is Ops.OR else (lane != 0))
      source = np.asarray(values, dtype=np.float16 if source_dtype is dtypes.half else np.uint8)
      expected = bytes((1, 0)) if op is Ops.AND else bytes((0, 1))
      assert _execute_raw_dynamic_image(image, 2, source.tobytes()) == expected
      assert decode_image(encode_image(image)) == image

    counterexample = _lower_uop_program(_indexed_local_bridge_program(source_dtype, Ops.OR, local_size=2, workers=1))
    assert counterexample is not None
    values = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.float16 if source_dtype is dtypes.half else np.uint8)
    assert _execute_raw_dynamic_image(counterexample, 2, values.tobytes()) == bytes((0, 0))

  integer = _lower_uop_program(_indexed_local_bridge_program(dtypes.int, Ops.AND, carrier=dtypes.int))
  assert integer is None


def test_dependent_scalar_local_extrema_is_vectorized_from_uop_structure():
  count = 4
  out, source = UOp.param(0, dtypes.int, (1,)), UOp.param(1, dtypes.half, (count,))
  value_buffer = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG)
  value_init = value_buffer.index(0).store(UOp.const(-math.inf, dtypes.half))
  value_axis = UOp.range(count, 0, AxisType.REDUCE)
  value_ptr = value_buffer.after(value_init, value_axis).index(0)
  value_candidate = source.index(value_axis).load()
  value_update = value_ptr.store(value_ptr.load().maximum(value_candidate))
  value_end = value_update.end(value_axis)
  best = value_buffer.after(value_end).index(0).load()

  index_buffer = UOp.placeholder((1,), dtypes.int, 1, addrspace=AddrSpace.REG)
  index_init = index_buffer.after(value_end).index(0).store(UOp.const(dtypes.int.min, dtypes.int))
  index_axis = UOp.range(count, 1, AxisType.REDUCE, src=(value_end,))
  index_ptr = index_buffer.after(index_init, index_axis).index(0)
  index_candidate = source.index(index_axis).load()
  equal = (index_candidate != best) != UOp.const(True, dtypes.bool)
  coordinate = UOp.const(count, dtypes.int)-index_axis
  index_update = index_ptr.store(index_ptr.load().maximum(equal.cast(dtypes.int)*coordinate))
  index_end = index_update.end(index_axis)
  selected = index_buffer.after(index_end).index(0).load()
  output = out.index(0).store(UOp.const(count, dtypes.int)-selected)

  image = _lower_uop_program(list(UOp.sink(value_init, value_update, index_init, index_update, output).toposort()))
  assert image is not None and len(image.ew_ops) < 100 and len(image.mid_gathers) == 7
  assert image.constants == struct.pack("<6h", 0, 1, 123, 124, 127, 128) and decode_image(encode_image(image)) == image
  assert _assert_decoded_image_bounds(image) == image
  assert any(gather.dst_stride == 32 for gather in image.mid_gathers)


def test_nested_static_local_accumulators_materialize_load_addresses():
  count = 4
  out, source = UOp.param(0, dtypes.half, (count,)), UOp.param(1, dtypes.half, (count,))
  lane = UOp.range(count, 2)
  outer = UOp.range(count, 1, AxisType.REDUCE, src=(lane,))
  inner = UOp.range(count, 0, AxisType.REDUCE, src=(outer,))

  inner_buffer = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  inner_init = inner_buffer.after(outer).index(0).store(0)
  inner_ptr = inner_buffer.after(inner_init, inner).index(0)
  inner_term = ((inner+outer < UOp.const(count-1, dtypes.int)) != UOp.const(True, dtypes.bool)).cast(dtypes.int)
  inner_update = inner_ptr.store(inner_ptr.load()+inner_term)
  inner_result = inner_buffer.after(inner_update.end(inner)).index(0).load()

  outer_buffer = UOp.placeholder((1,), dtypes.int, 1, addrspace=AddrSpace.REG)
  outer_init = outer_buffer.after(lane).index(0).store(0)
  outer_ptr = outer_buffer.after(outer_init, outer).index(0)
  outer_term = (inner_result != outer+lane+UOp.const(1-count, dtypes.int)).where(0, 1)
  outer_update = outer_ptr.store(outer_ptr.load()+outer_term)
  dynamic_index = outer_buffer.after(outer_update.end(outer)).index(0).load()
  normalized = (dynamic_index < 0).where(dynamic_index+count, dynamic_index)
  gate = ((normalized < 0) != UOp.const(True, dtypes.bool)) & (normalized < count)
  output = out.index(lane).store(source.index(normalized).load(UOp.const(0.0, dtypes.half), gate)).end(lane)

  image = _lower_uop_program(list(UOp.sink(inner_init, inner_update, outer_init, outer_update, output).toposort()))
  assert image is not None and len(image.gathers) == 1 and image.gathers[0].offsets == (0, 0, 0, 0)
  assert not image.host_gathers and decode_image(encode_image(image)) == image


def test_static_local_address_preserves_sequential_fp16_updates():
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (2,))
  buffer = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG)
  initialize, axis = buffer.index(0).store(0.0), UOp.range(3, 0, AxisType.REDUCE)
  pointer = buffer.after(initialize, axis).index(0)
  term = (axis < 1).where(UOp.const(2048.0, dtypes.half),
                         (axis < 2).where(UOp.const(1.0, dtypes.half), UOp.const(-2048.0, dtypes.half)))
  update = pointer.store(pointer.load()+term)
  index = buffer.after(update.end(axis)).index(0).load().cast(dtypes.int)
  store = out.index(0).store(source.index(index).load())
  uops = list(UOp.sink(initialize, update, store).toposort())
  assert (image:=_lower_uop_program(uops)) is not None and image.gathers[0].offsets == (0,)
  assert decode_image(encode_image(image)) == image


def test_multiple_fp16_locals_preserve_sequential_store_updates():
  out, lane = UOp.param(0, dtypes.half, (2,)), UOp.range(2, 3)
  def local(slot:int, axis_id:int) -> tuple[UOp, UOp, UOp]:
    buffer = UOp.placeholder((1,), dtypes.half, slot, addrspace=AddrSpace.REG)
    initialize, axis = buffer.index(0).store(0.0), UOp.range(3, axis_id, AxisType.REDUCE)
    pointer = buffer.after(initialize, axis).index(0)
    term = (axis < 1).where(UOp.const(2048.0, dtypes.half), (axis < 2).where(1.0, -2048.0))
    update = pointer.store(pointer.load()+term)
    return initialize, update, buffer.after(update.end(axis)).index(0).load()
  first_init, first_update, first = local(0, 0)
  second_init, second_update, second = local(1, 1)
  image = _lower_uop_program(list(UOp.sink(first_init, first_update, second_init, second_update,
                                          out.index(lane).store(first+second)).toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE
  assert image.constants == struct.pack("<e", 0.0) and len(image.ew_ops) == 1 and not image.gathers and not image.mid_gathers
  assert decode_image(encode_image(image)) == image


def test_static_local_address_preflights_reducer_product(monkeypatch):
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (2,))
  buffer = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  initialize = buffer.index(0).store(0)
  outer = UOp.range(16384, 0, AxisType.REDUCE)
  inner = UOp.range(4096, 1, AxisType.REDUCE, src=(outer,))
  pointer = buffer.after(initialize, outer, inner).index(0)
  update = pointer.store(pointer.load()+1)
  index = buffer.after(update.end(outer, inner)).index(0).load()
  store = out.index(0).store(source.index(index).load())
  original = rockchip_renderer._iter_range_env
  def guarded_iterator(ranges, max_envs=rockchip_renderer._MAX_STATIC_RANGE_ENVS, dependencies=True):
    if not dependencies: raise AssertionError("iterator reached")
    return original(ranges, max_envs, dependencies)
  monkeypatch.setattr(rockchip_renderer, "_iter_range_env", guarded_iterator)
  assert _lower_uop_program(list(UOp.sink(initialize, update, store).toposort())) is None


def test_packed_bool_load_uses_canonical_int16_lanes():
  out, mask = UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.bool, (4,))
  lane = UOp.range(4, 0)
  image = _lower_uop_program(list(out.index(lane).store(mask.index(lane).load().cast(dtypes.int)+1).end(lane).sink().toposort()))
  assert image is not None and len(image.gathers) == 1
  assert image.gathers[0].itemsize == 1 and image.gathers[0].dst_stride == 2
  assert image.ew_ops[-1].int16_input and image.ew_ops[-1].int32_output


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
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters
  assert any(op.int32_output for op in image.ew_ops) and decode_image(encode_image(image)) == image


def test_fixed_nonzero_rank_two_static_images_preserve_coordinate_matrix_bounds():
  source = Tensor([[1, 0], [0, 2]], device="ROCKCHIP")
  result = source.nonzero(size=2)
  linear, var_vals = result.linear_with_vars()
  assert not var_vals
  renderer = RockchipRenderer(Target.parse("ROCKCHIP"))
  images = [decode_image(to_program(call.src[0], renderer).src[-1].arg) for call in linear.src if call.src[0].op is Ops.SINK]
  assert len(images) == 4
  for image in images:
    assert image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters
    assert decode_image(encode_image(image)) == image
    for gather in (*image.gathers, *image.mid_gathers, *image.post_gathers):
      if gather.dst_kind is RKBufferKind.SCRATCH:
        assert gather.dst_addend+(gather.count-1)*gather.dst_stride < image.scratch[gather.dst_index].size//gather.itemsize

  coordinate = images[-1]
  coordinate_rows = [g for g in coordinate.gathers if g.dst_kind is RKBufferKind.SCRATCH and g.itemsize == 2 and
                    g.count == result.numel() and len(g.values) == result.numel()]
  row_slots = {g.dst_index for g in coordinate_rows}
  assert len(coordinate_rows) == 10 and len(row_slots) == 3
  matrix_slot = next(slot for slot in row_slots if sum(g.dst_index == slot for g in coordinate_rows) == 8)
  matrix_rows = [g for g in coordinate_rows if g.dst_index == matrix_slot]
  assert coordinate.scratch[matrix_slot].size == 512
  assert tuple(g.dst_addend for g in matrix_rows) == tuple(range(0, 256, 32))
  assert tuple(g.values for g in matrix_rows) == ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1),
                                                  (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1))
  np.testing.assert_array_equal(_execute_integer_image(coordinate, np.asarray([1, 0, 0, 2], dtype=np.int32),
                                                       np.asarray([0, 1, 6, 7], dtype=np.int32)),
                                np.asarray([0, 0, 1, 1], dtype=np.int32))


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
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers and not image.host_scatters
  assert any(op.int32_output for op in image.ew_ops) and decode_image(encode_image(image)) == image


def test_direct_dynamic_int32_load_selects_all_raw_bytes():
  out, source, indices = (UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.int, (9,)), UOp.param(2, dtypes.int, (4,)))
  lane = UOp.range(4, 0)
  index = indices.index(lane).load()
  gate = ((index < 0) != UOp.const(True, dtypes.bool)) & (index < 9)
  load = source.index(index).load(UOp.const(0, dtypes.int), gate)
  image = _lower_uop_program(list(out.index(lane).store(load).end(lane).sink().toposort()))
  assert image is not None and len(image.post_gathers) == 4
  assert {gather.dst_addend for gather in image.post_gathers} == {0, 1, 2, 3}
  assert decode_image(encode_image(image)) == image


def test_dynamic_candidate_selector_preserves_plain_raw_payloads():
  indices = np.asarray((0, 1, 2, 8), dtype="<i4")
  sources = {
    dtypes.half:np.asarray((0x0000, 0x8000, 0x7e01, 0x7fff, 0x7c01, 0xfc01, 0x3555, 0xbc00, 0x0400), dtype="<u2"),
    dtypes.int16:np.asarray((0x0000, 0x8000, 0xffff, 0x7fff, 0x00ff, 0xff00, 0x5555, 0xaaaa, 0x1234), dtype="<u2"),
    dtypes.int:np.asarray((0, 0x80000000, 0x7fc01234, 0xffffffff, 0x7fffffff, 0x00800000,
                           0xff800001, 0x55aa55aa, 0x12345678), dtype="<u4"),
  }
  for dtype,source in sources.items():
    image = _lower_uop_program(_dynamic_load_program(dtype=dtype))
    assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
    assert _execute_raw_dynamic_image(image, 4*dtype.itemsize, source.tobytes(), indices.tobytes()) == source[indices].tobytes()
    assert decode_image(encode_image(image)) == image


def test_dynamic_candidate_selector_normalizes_negative_indices_exactly():
  source = np.asarray((0x0000, 0x8000, 0x7e01, 0x7fff, 0x7c01, 0xfc01, 0x3555, 0xbc00, 0x0400), dtype="<u2")
  indices = np.asarray((-1, -9, -5, -10), dtype="<i4")
  image = _lower_uop_program(_dynamic_load_program(normalized=True))
  assert image is not None and not image.host_gathers
  expected = np.asarray((source[8], source[0], source[4], 0), dtype="<u2")
  assert _execute_raw_dynamic_image(image, 8, source.tobytes(), indices.tobytes()) == expected.tobytes()


def test_affine_gather_bounds_reject_negative_low_but_keep_offset_sentinel():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    for count in (1, 2):
      invalid = RKGather(1, 0, count, base=-1, axes=((1, count, 1),), itemsize=dtype.itemsize)
      try: rockchip_renderer._validate_gather_bounds(invalid, count)
      except RuntimeError: pass
      else: raise AssertionError(f"negative affine low admitted for {dtype} lane{count}")
      rockchip_renderer._validate_gather_bounds(RKGather(1, 0, count, base=0, axes=((1, count, 1),), itemsize=dtype.itemsize), count)
      rockchip_renderer._validate_gather_bounds(RKGather(1, 0, count, offsets=(-1,)+(0,)*(count-1)), 1)
      try: rockchip_renderer._validate_gather_bounds(RKGather(1, 0, count, offsets=(-2, 0)[:count]), count)
      except RuntimeError: pass
      else: raise AssertionError(f"offset below sentinel admitted for {dtype} lane{count}")
      if dtype is not dtypes.bool:
        assert _lower_uop_program(_dynamic_load_program(count=count, dtype=dtype, normalized=True)) is not None


def test_scalar_gather_bounds_reject_negative_low_for_all_typed_lanes():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    try: rockchip_renderer._validate_gather_bounds(RKGather(1, 0, 1, base=-1, itemsize=dtype.itemsize), 1)
    except RuntimeError: pass
    else: raise AssertionError(f"negative scalar low admitted for {dtype}")
    rockchip_renderer._validate_gather_bounds(RKGather(1, 0, 1, itemsize=dtype.itemsize), 1)


def test_gather_offsets_reject_true_gate_negative_and_allow_false_sentinel():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    out, source, lane = UOp.param(0, dtype, (4,)), UOp.param(1, dtype, (4,)), UOp.range(4, 0)
    default = UOp.const(0.0, dtype) if dtype is dtypes.half else UOp.const(0, dtype)
    counterexample = list(out.index(lane).store(source.index(lane-1).load(default, lane < 4)).end(lane).sink().toposort())
    assert _lower_uop_program(counterexample) is None
    for gate in (lane < 0, lane > 0):
      valid = list(out.index(lane).store(source.index(lane-1).load(default, gate)).end(lane).sink().toposort())
      image = _lower_uop_program(valid)
      assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers


def test_gather_offsets_normalize_inactive_raw_negative_to_fill():
  for dtype in (dtypes.half, dtypes.int16, dtypes.int, dtypes.bool):
    out, source, lane = UOp.param(0, dtype, (4,)), UOp.param(1, dtype, (4,)), UOp.range(4, 0)
    default = UOp.const(0.0, dtype) if dtype is dtypes.half else UOp.const(0, dtype)
    padded = list(out.index(lane).store(source.index(lane-31).load(default, lane < 0)).end(lane).sink().toposort())
    image = _lower_uop_program(padded)
    assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
    assert image.gathers and image.gathers[-1].offsets == (-1, -1, -1, -1)


def test_dynamic_candidate_selector_composes_multiple_axes_and_external_gate():
  source = (np.arange(81, dtype=np.uint32)*257+0x8000).astype("<u2")
  first, second = np.asarray((-1, 0, 2, -10), dtype="<i4"), np.asarray((0, -1, 3, 4), dtype="<i4")
  gate = np.asarray((1, 1, 0, 1), dtype=np.uint8)
  image = _lower_uop_program(_dynamic_load_program(extents=(9, 9), normalized=True, external_gate=True))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  expected = np.asarray((source[72], source[8], 0, 0), dtype="<u2")
  assert _execute_raw_dynamic_image(image, 8, source.tobytes(), first.tobytes(), second.tobytes(), gate.tobytes()) == expected.tobytes()
  assert decode_image(encode_image(image)) == image


def test_dynamic_candidate_selector_repeats_raw_channels():
  source = (np.arange(21, dtype=np.uint32)*131+0x8000).astype("<u2")
  indices, gate = np.asarray((6, 0, 3, 7), dtype="<i4"), np.asarray((1, 0, 1, 1), dtype=np.uint8)
  image = _lower_uop_program(_dynamic_load_program(count=12, extents=(7,), external_gate=True, repeat=3))
  assert image is not None and not image.host_gathers
  expected = np.zeros(12, dtype="<u2")
  expected[:3], expected[6:9] = source[18:21], source[9:12]
  assert _execute_raw_dynamic_image(image, 24, source.tobytes(), indices.tobytes(), gate.tobytes()) == expected.tobytes()


def test_dynamic_candidate_selector_blocks_1001_candidates():
  source = (np.arange(1001, dtype=np.uint32)+0x8000).astype("<u2")
  image = _lower_uop_program(_dynamic_load_program(count=64, extents=(1001,)))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  for start in range(0, 1024, 64):
    indices = np.arange(start, start+64, dtype="<i4")
    expected = np.zeros(64, dtype="<u2")
    valid = indices < len(source)
    expected[valid] = source[indices[valid]]
    assert _execute_raw_dynamic_image(image, 128, source.tobytes(), indices.tobytes()) == expected.tobytes()
  assert decode_image(encode_image(image)) == image


def test_dynamic_candidate_selector_rejects_before_table_allocation(monkeypatch):
  def unexpected(*_args, **_kwargs): raise AssertionError("candidate table escaped pre-allocation admission")
  monkeypatch.setattr(rockchip_renderer, "_gather_offsets", unexpected)
  cases = ((1, rockchip_renderer._MAX_STATIC_RANGE_ENVS+1),
           (4096, rockchip_renderer._MAX_DYNAMIC_SELECTOR_CELLS//4096+1))
  for count,extent in cases:
    output = rockchip_renderer._output_store(_dynamic_load_program(count=count, extents=(extent,)), dtypes.half)
    assert output is not None and rockchip_renderer._lower_dynamic_typed_load(output, dtypes.half) is None


def test_dynamic_candidate_selector_rejects_unencodable_slots_and_offsets(monkeypatch):
  out, source, indices = UOp.param(0, dtypes.half, (1,)), UOp.param(rockchip_renderer._RKIMAGE_U16_MAX+1, dtypes.half, (1,)), \
    UOp.param(2, dtypes.int, (1,))
  lane = UOp.range(1, 0)
  dynamic = indices.index(lane).load()
  gate = ((dynamic < 0) != UOp.const(True, dtypes.bool)) & (dynamic < 1)
  program = list(out.index(lane).store(source.index(dynamic).load(UOp.const(0.0, dtypes.half), gate)).end(lane).sink().toposort())
  output = rockchip_renderer._output_store(program, dtypes.half)
  assert output is not None and rockchip_renderer._lower_dynamic_typed_load(output, dtypes.half) is None
  safe, unsafe = (1 << 29)-1, 1 << 29
  for program in (_dynamic_offset_program(data_offset=safe), _dynamic_offset_program(index_offset=safe)):
    output = rockchip_renderer._output_store(program, dtypes.int)
    assert output is not None and (image:=rockchip_renderer._lower_dynamic_typed_load(output, dtypes.int)) is not None
    assert decode_image(encode_image(image)) == image
  monkeypatch.setattr(rockchip_renderer, "_candidate_gather",
                      lambda *_args, **_kwargs:(_ for _ in ()).throw(AssertionError("unsafe offset reached physical allocation")))
  for program in (_dynamic_offset_program(data_offset=unsafe), _dynamic_offset_program(index_offset=unsafe)):
    output = rockchip_renderer._output_store(program, dtypes.int)
    assert output is not None and rockchip_renderer._lower_dynamic_typed_load(output, dtypes.int) is None


def test_dynamic_candidate_selector_composes_exact_bool_total_fill_gate():
  indices, mask = np.asarray((4, 1, -1, 8), dtype="<i4"), np.asarray((1, 0, 1, 0, 0), dtype=np.uint8)
  for dtype,source in ((dtypes.int16, np.asarray((0x8000, 0xffff, 0x7fff, 0x1234, 0xabcd), dtype="<u2")),
                       (dtypes.int, np.asarray((0x80000000, 0xffffffff, 0x7fffffff, 0x12345678, 0xabcdef01), dtype="<u4"))):
    image = _lower_uop_program(_dynamic_total_load_program(dtype))
    assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
    expected = np.asarray((source[4], source[1], (1 << (dtype.itemsize*8))-7, (1 << (dtype.itemsize*8))-7), dtype=source.dtype)
    assert _execute_raw_dynamic_image(image, 4*dtype.itemsize, source.tobytes(), indices.tobytes(), mask.tobytes()) == expected.tobytes()
    assert decode_image(encode_image(image)) == image


def test_bounded_int32_lookup_executes_as_ordinary_uops():
  out, indices = UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.int, (4,))
  lane = UOp.range(4, 0)
  index = indices.index(lane).load()
  valid = ((index < 0) != UOp.const(True, dtypes.bool)) & (index < 5)
  value = valid.where(index+lane*4, UOp.const(0, dtypes.int))
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  assert any(op.int32_input and op.int32_output for op in image.ew_ops)
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
      assert image is not None and image.execution_class is RKExecutionClass.NATIVE
      assert not image.host_gathers and not image.host_scatters and all(x.int16_input and x.int16_output for x in image.ew_ops)
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
    assert image is not None and image.execution_class is RKExecutionClass.NATIVE
    assert len(image.ew_ops) > 3000 and decode_image(encode_image(image)) == image
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
  assert len(direct.ew_ops) < len(combined.ew_ops) < len(direct.ew_ops)+100
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
    assert image is not None and len(image.ew_ops) < 4000 and decode_image(encode_image(image)) == image
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
  assert image is not None and len(image.ew_ops) == 6 and len(image.mid_gathers) == 8
  assert sum(op.int32_input and op.int32_output for op in image.ew_ops) == 2


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
        assert image is not None and image.execution_class is RKExecutionClass.NATIVE
        assert not image.host_gathers and not image.host_scatters and decode_image(encode_image(image)) == image
        np.testing.assert_array_equal(_execute_integer_image(image, physical), expected(op, dtype, amount))
      out, source, amount = (UOp.param(slot, dtype if slot else dtypes.int, (len(values),)) for slot in range(3))
      lane = UOp.range(len(values), 0)
      shifted = UOp(op, dtype, src=(source.index(lane).load(), amount.index(lane).load()))
      shifted = shifted.cast(dtypes.int) if dtype is dtypes.uint else shifted
      nested = UOp(Ops.XOR, dtypes.int, src=(shifted, UOp.const(int(marker), dtypes.int)))
      image = _lower_uop_program(list(out.index(lane).store(nested).end(lane).sink().toposort()))
      assert image is not None and image.execution_class is RKExecutionClass.NATIVE
      assert not image.host_gathers and not image.host_scatters and decode_image(encode_image(image)) == image
      np.testing.assert_array_equal(_execute_integer_image(image, physical, shifts if dtype is dtypes.int else shifts.view(np.uint32)),
                                    np.bitwise_xor(expected(op, dtype, shifts), marker))
      inner = UOp(Ops.SHL, dtype, src=(source.index(lane).load(), UOp.const(1, dtype)))
      shifted = UOp(op, dtype, src=(inner, amount.index(lane).load()))
      result = shifted.cast(dtypes.int) if dtype is dtypes.uint else shifted
      image = _lower_uop_program(list(out.index(lane).store(result).end(lane).sink().toposort()))
      assert image is not None and image.execution_class is RKExecutionClass.NATIVE
      assert not image.host_gathers and not image.host_scatters and decode_image(encode_image(image)) == image
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
  assert image is not None and any(op.int32_output for op in image.ew_ops)


def test_dependent_reduction_range_preserves_vector_output_axis():
  def lower(rows:int, depth:int=65):
    out = UOp.param(0, dtypes.half, (rows,))
    lhs, rhs = UOp.param(1, dtypes.half, (rows*depth,)), UOp.param(2, dtypes.half, (depth,))
    row = UOp.range(rows, 1)
    axis = UOp.range(depth, 0, AxisType.REDUCE, src=(row,))
    local = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG).index(0)
    initialize = local.store(UOp.const(0.0, dtypes.float))
    product = lhs.index(row*depth+axis).load() * rhs.index(axis).load()
    update = local.store(local.load() + product.cast(dtypes.float))
    output = out.index(row).store(local.load().cast(dtypes.half))
    return _lower_uop_program(list(UOp.sink(initialize, update, output).toposort()))

  scalar, vector, large = lower(1), lower(45), lower(128, 128)
  assert scalar is not None and vector is not None and large is not None
  assert len(vector.ew_ops) == len(scalar.ew_ops) < 200 and len(large.ew_ops) < 300


def test_mapped_loop_reduction_composes_generic_post_uops():
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (65,))
  axis = UOp.range(65, 0, AxisType.REDUCE)
  local = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG).index(0)
  initialize = local.store(UOp.const(0.0, dtypes.float))
  value = source.index(axis).load()
  update = local.store(local.load() + (value*value).cast(dtypes.float))
  post = (local.load().cast(dtypes.half)*UOp.const(1/65, dtypes.half)).sqrt()
  output = out.index(0).store(post)
  image = _lower_uop_program(list(UOp.sink(initialize, update, output).toposort()))
  assert image is not None and image.gather_after < len(image.ew_ops)
  assert image.ew_ops[-1].dst == RKArg(RKBufferKind.ARG, 0)


def test_multiple_output_stores_execute_sequentially():
  first, second = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (4,))
  source, lane = UOp.param(2, dtypes.half, (4,)), UOp.range(4, 0)
  program = list(UOp.sink(first.index(lane).store(source.index(lane).load()+1.0),
                          second.index(lane).store(first.index(lane).load()*2.0)).toposort())
  image = _lower_uop_program(program)
  assert image is not None and len(image.ew_ops) == 2
  assert image.ew_ops[1].lhs == RKArg(RKBufferKind.ARG, 0)
  assert image.ew_ops[1].submit_barrier and image.ew_ops[1].stateful


def test_static_structural_expansion_is_bounded():
  limit = (1 << 14) + 1
  out, source = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (limit,))
  lane, axis = UOp.range(1, 0), UOp.range(limit, 1, AxisType.REDUCE)
  reduced = UOp(Ops.REDUCE, dtypes.half, src=(source.index(axis).load(), axis), arg=(Ops.ADD,))
  uops = list(out.index(lane).store(reduced).end(lane, axis).sink().toposort())
  assert _lower_uop_program(uops) is None


def test_deep_generic_graph_canonicalization_is_iterative():
  value = UOp.const(0, dtypes.int)
  for _ in range(4096): value = value + UOp.const(1, dtypes.int)
  rewritten = _finite_int_max_neutrals(value)
  assert rewritten.key == value.key


def test_static_range_environment_allocation_is_bounded():
  axes = [UOp.range(1024, 0), UOp.range(1024, 1)]
  try: _iter_range_env(axes, max_envs=1024)
  except RuntimeError as error: assert "static_index_budget" in str(error)
  else: raise AssertionError("oversized static RANGE product was materialized")


def test_large_range_independent_static_value_is_materialized_once():
  image = _lower_uop_program(_program(dtypes.half,
    lambda _i:UOp.const(0.0, dtypes.half) * UOp.const(-1.0, dtypes.half), count=1 << 20))
  assert image is not None and not image.gathers and image.constants == struct.pack("<e", -0.0)


def test_generic_ew_chain_reuses_dead_scratch_values():
  source = UOp.param(1, dtypes.half, (1024,))
  def chain(i):
    value = source.index(i).load()
    for _ in range(128): value = value * UOp.const(1.001, dtypes.half)
    return value
  image = _lower_uop_program(_program(dtypes.half, chain, count=1024))
  assert image is not None and len(image.ew_ops) == 128 and len(image.scratch) <= 4


def test_scratch_reuse_spans_mid_gathers_without_aliasing_their_state():
  scratch = tuple(RKScratch(64) for _ in range(5))
  arg, slots = RKArg(RKBufferKind.ARG, 0), tuple(RKArg(RKBufferKind.SCRATCH, i) for i in range(5))
  image = RKImage(RKTarget.RK3588, scratch, gathers=(RKGather(0, 0, 1, values=(0,)),), ew_ops=(
    RKEWOp(slots[1], slots[0], slots[0], 1, _EW_CFG[Ops.ADD]),
    RKEWOp(slots[2], slots[1], slots[0], 1, _EW_CFG[Ops.ADD]),
    RKEWOp(slots[4], slots[3], slots[3], 1, _EW_CFG[Ops.ADD]),),
    mid_gathers=(RKGather(arg.index, slots[3].index, 1, offsets=(0,), partial=True, after=2),))
  colored = _reuse_linear_scratch(image, {})
  assert len(colored.scratch) <= 4
  assert colored.mid_gathers[0].dst_index not in {colored.ew_ops[0].dst.index, colored.ew_ops[1].dst.index, colored.ew_ops[2].dst.index}
  assert decode_image(encode_image(colored)) == colored


def test_large_divided_range_address_uses_compact_gather_axes():
  outer = UOp.range(16384, 1)
  inner = UOp.range(64, 4, src=(outer,))
  out_index = outer*64+inner
  grouped = UOp(Ops.CDIV, dtypes.int, src=(outer, UOp.const(64, dtypes.int)))*1024+inner
  plan = _gather_plan(1, 0, out_index, grouped, None, 1 << 20)
  assert not plan.offsets and plan.base == 0
  assert set(plan.axes) == {(1, 64, 1), (4096, 256, 1024)}


def test_dynamic_host_gather_and_scatter_are_explicit_and_opt_out(monkeypatch):
  monkeypatch.delenv("ROCKCHIP_HOST_GATHER", raising=False)
  indices = UOp.param(2, dtypes.int, (4,))
  axis = UOp.range(4, 0)

  gather_out, gather_source = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (8,))
  gather = gather_out.index(axis).store(gather_source.index(indices.index(axis).load()).load())
  gather_uops = list(gather.end(axis).sink().toposort())
  gather_image = _lower_uop_program(gather_uops)
  assert gather_image is not None and gather_image.execution_class is RKExecutionClass.HOST_ADDRESS
  assert len(gather_image.host_gathers) == 1 and not gather_image.host_scatters
  assert decode_image(encode_image(gather_image)) == gather_image

  scatter_out, scatter_source = UOp.param(0, dtypes.half, (8,)), UOp.param(1, dtypes.half, (4,))
  scatter = scatter_out.index(indices.index(axis).load()).store(scatter_source.index(axis).load())
  scatter_uops = list(scatter.end(axis).sink().toposort())
  scatter_image = _lower_uop_program(scatter_uops)
  assert scatter_image is not None and scatter_image.execution_class is RKExecutionClass.HOST_ADDRESS
  assert len(scatter_image.host_scatters) == 1 and not scatter_image.host_gathers
  assert decode_image(encode_image(scatter_image)) == scatter_image

  monkeypatch.setenv("ROCKCHIP_HOST_GATHER", "0")
  assert _lower_uop_program(gather_uops) is None and _lower_uop_program(scatter_uops) is None


def test_dynamic_host_gather_carries_affine_lane_address_abi(monkeypatch):
  monkeypatch.setenv("ROCKCHIP_HOST_GATHER", "1")
  count = 4
  out, source = UOp.param(0, dtypes.half, (count,)), UOp.param(1, dtypes.half, (count*10,))
  indices, lane = UOp.param(2, dtypes.int, (count,)), UOp.range(count, 0)
  value = source.index(lane*10+indices.index(lane).load()).load()
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(image.host_gathers) == 1
  address = image.host_gathers[0]
  assert (address.base, address.index_scale, address.lane_stride, address.index_limit) == (0, 1, 10, count*10)
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
  assert image is not None and len(image.host_gathers) == 1
  host = image.host_gathers[0]
  assert host.src.kind is RKBufferKind.SCRATCH and (host.src_count, host.index_scale, host.lane_stride, host.index_limit) == (8, 1, 2, 2)
  assert any(gather.offsets == (0, 2, 1, 3, 4, 6, 5, 7) for gather in image.gathers)
  assert decode_image(encode_image(image)) == image


def test_nonaffine_scalar_dot_uses_cmac_static_packing():
  groups = 64
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (groups,)), UOp.param(2, dtypes.half, (groups,))
  permutation = tuple((lane*17)%groups for lane in range(groups))
  terms = [lhs.index(permutation[lane]).load()*rhs.index(lane).load() for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  image = _lower_uop_program(list(out.index(UOp.const(0, dtypes.int)).store(value).sink().toposort()))
  assert image is not None and image.cmac is not None and (image.cmac.m,image.cmac.n,image.cmac.k) == (1,1,groups)
  assert any(gather.offsets[:groups] == permutation for gather in image.gathers)
  assert not image.ew_ops and image.post_gathers[0].offsets == (0,)
  assert decode_image(encode_image(image)) == image
