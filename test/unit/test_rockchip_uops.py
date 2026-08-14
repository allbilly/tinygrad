import itertools, math, struct
import numpy as np
from collections.abc import Callable
from types import SimpleNamespace
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKExecutionClass, RKImage, RKLayout, RKTarget, RKValue, RKEWOp, RKGather, RKScratch,
  _EW_CFG, _EW_CFG_ABS, _EW_CFG_FLOOR, _EW_CFG_MIN, _EW_STAGE_FP32_IN, _EW_STAGE_FP32_OUT, _NATIVE_SIGN, _MAX_EW_ELEMS_FP16,
  _canonical_half_storage, _finite_int_max_neutrals, _fp32_expr_to_half, _gather_plan, _iter_range_env,
  _lower_uop_program, _reuse_linear_scratch, decode_image, encode_image)
from tinygrad.runtime import ops_rockchip as rockchip_runtime
import tinygrad.renderer.rockchip as rockchip_renderer
from tinygrad.uop.ops import AxisType, Ops, UOp


def _program(dtype, value, count:int=4):
  out, axis = UOp.param(0, dtype, (count,)), UOp.range(count, 0)
  return list(out.index(axis).store(value(axis)).end(axis).sink().toposort())

def _int32_binary_program(value:Callable[[UOp, UOp], UOp], count:int=4) -> list[UOp]:
  out, lhs, rhs = (UOp.param(slot, dtypes.int, (count,)) for slot in range(3))
  axis = UOp.range(count, 0)
  return list(out.index(axis).store(value(lhs.index(axis).load(), rhs.index(axis).load())).end(axis).sink().toposort())

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
  for gather in image.mid_gathers: mid.setdefault(gather.after, []).append(gather)
  for index in range(len(image.ew_ops)+1):
    apply_gathers(tuple(mid.get(index, ())))
    if index < len(image.ew_ops): execute(image.ew_ops[index])
  apply_gathers(image.post_gathers)
  return np.frombuffer(args[0], dtype="<i4").copy()

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
      expected[rockchip_renderer._eval_int(out_index, env, cache)] = encode(rockchip_renderer._eval_expr(expr, env, cache))
    assert rockchip_renderer._static_values(out_index, expr, 20, encode) == tuple(expected)


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
  assert image is not None and image.mid_gathers
  assert any(gather.src_index == 3 and gather.offsets == (0, 1, 0, 1, 0, 1, 0, 1) for gather in image.gathers)
  assert len(image.ew_ops) < 300


def test_batched_unrolled_math_reduction_materializes_each_uop_result():
  rows, groups = 8, 4
  out, source = UOp.param(0, dtypes.half, (rows,)), UOp.param(1, dtypes.half, (rows*groups,))
  normalizer, lane = UOp.param(2, dtypes.half, (rows,)), UOp.range(rows, 0)
  terms = [(source.index(lane*groups+k).load() - normalizer.index(lane).load()).exp2() for k in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(image.mid_gathers) == groups
  assert image.gather_after > 1 and image.ew_ops[image.gather_after].dst.kind is RKBufferKind.SCRATCH


def test_static_reduce_uops_are_structurally_executed():
  for op in (Ops.ADD, Ops.MAX, Ops.MUL):
    out, source = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,))
    row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
    term = source.index(row*3+axis).load()
    reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(op,))
    image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
    assert image is not None and len(image.gathers) == 3 and len(image.ew_ops) == 2


def test_static_dot_reduce_owns_accurate_physical_recipe():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (6,)), UOp.param(2, dtypes.half, (6,))
  row, axis = UOp.range(2, 0), UOp.range(3, 1, AxisType.REDUCE)
  term = lhs.index(row*3+axis).load() * rhs.index(row*3+axis).load()
  reduced = UOp(Ops.REDUCE, dtypes.half, src=(term, axis), arg=(Ops.ADD,))
  image = _lower_uop_program(list(out.index(row).store(reduced).end(row, axis).sink().toposort()))
  assert image is not None and len(image.gathers) >= 6 and len(image.ew_ops) > 20


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
  assert image is not None and image.constants == struct.pack("<eee", 1.0, 65.0, 0.0)
  assert len(image.gathers) == groups*2 and image.gather_after >= 17 and image.gather_after%17 == 0
  assert len(image.mid_gathers) == groups*2 and len(image.ew_ops) > image.gather_after
  assert image.ew_ops[-1].ew_cfg == _EW_CFG[Ops.MAX]


def test_fp32_add_mul_tree_uses_half_expansion_at_output_boundary():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(image.ew_ops) > 10


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
  assert image is not None and len(image.ew_ops) == 6
  assert sum(bool(op.ew_cfg & _EW_STAGE_FP32_OUT) for op in image.ew_ops) == 3
  assert decode_image(encode_image(image)) == image


def test_terminal_half_casts_use_typed_integer_and_bool_output_abis():
  source = UOp.param(1, dtypes.half, (9,))
  integer = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load().cast(dtypes.int), count=9))
  boolean = _lower_uop_program(_program(dtypes.bool, lambda i:source.index(i).load().cast(dtypes.bool), count=9))
  assert integer is not None and integer.ew_ops[-1].int32_output
  assert boolean is not None and boolean.ew_ops[-1].bool_output
  assert decode_image(encode_image(integer)) == integer and decode_image(encode_image(boolean)) == boolean


def test_fp32_pure_add_tree_uses_compensated_half_expansion_at_output_boundary():
  source = UOp.param(1, dtypes.half, (64,))
  terms = [source.index(i).load().cast(dtypes.float) for i in range(64)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(_program(dtypes.half, lambda _i:value.cast(dtypes.half), count=1))
  assert image is not None and 64 < len(image.ew_ops) < 2000


def test_nested_fp32_product_sum_is_committed_before_outer_half_add():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  bias = UOp.param(3, dtypes.half, (1,)).index(0).load()
  products = [(lhs.index(i).load() * rhs.index(i).load()).cast(dtypes.float) for i in range(4)]
  product_sum = products[0]
  for product in products[1:]: product_sum = product_sum + product
  image = _lower_uop_program(_program(dtypes.half, lambda _i:product_sum.cast(dtypes.half) + bias, count=1))
  assert image is not None and len(image.ew_ops) > 20 and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


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


def test_fp32_boundary_activation_preserves_accurate_reduction():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).maximum(UOp.const(0.0, dtypes.float)).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(image.ew_ops) > 10 and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


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
  output = rockchip_renderer._output_store(uops, dtypes.half, allow_local=True)
  assert output is not None
  assert next(iter(rockchip_renderer._static_local_load_offsets(uops, output, output[4]).values())) == (0,)
  assert (image:=_lower_uop_program(uops)) is not None and decode_image(encode_image(image)) == image


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


def test_unrolled_fp16_predicate_prefix_uses_blocked_uop_recipe():
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
  assert image is not None and len(image.ew_ops) == 16 and sum(op.compare for op in image.ew_ops) == 3


def test_normalized_int_prefix_avoids_compare_submission():
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
  assert image is not None and len(image.ew_ops) == 10 and not any(op.compare for op in image.ew_ops)


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
  assert len(vector.ew_ops) == len(scalar.ew_ops)


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


def test_nonaffine_scalar_mul_sum_uses_static_gather_product_residual_and_kahan():
  groups = 64
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (groups,)), UOp.param(2, dtypes.half, (groups,))
  permutation = tuple((lane*17)%groups for lane in range(groups))
  terms = [lhs.index(permutation[lane]).load()*rhs.index(lane).load() for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  image = _lower_uop_program(list(out.index(UOp.const(0, dtypes.int)).store(value).sink().toposort()))
  assert image is not None and any(gather.offsets == permutation for gather in image.gathers)
  assert len(image.mid_gathers) == groups*2 and sum(op.submit_barrier for op in image.ew_ops) >= groups*2
  assert any(gather.values and 0x5410 in gather.values for gather in image.gathers)  # FP16 splitter 65
  assert decode_image(encode_image(image)) == image
