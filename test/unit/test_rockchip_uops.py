import ctypes, math, struct
import numpy as np
from collections.abc import Callable
from types import SimpleNamespace
from tinygrad.codegen import line_rewrite, pm_linearize_cleanups
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKExecutionClass, RKImage, RKLayout, RKStaticIndexEvaluator, RKValue,
  RKEWOp, RKGather, RKScratch,
  _EW_CFG, _EW_CFG_ABS, _EW_CFG_FLOOR, _EW_CFG_MIN, _EW_STAGE_FP32_IN, _EW_STAGE_FP32_OUT, _NATIVE_SIGN, _MAX_EW_ELEMS_FP16,
  _bool_tautology, _bounded_int32_fp16_root, _canonical_half_storage, _exact_int_range, _finite_max_neutrals, _fp16_bits,
  _fp32_expr_to_half, _gather_plan, _iter_range_env,
  _hoist_leading_vector_materialization, _lower_uop_program, _reuse_linear_scratch, _static_int_vector, decode_image, encode_image)
from tinygrad.runtime import ops_rockchip as rockchip_runtime
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
    if destination_dtype.itemsize == 2: result = np.clip(result, -32768, 32767)
    else: result = (result+(1<<31)) % (1<<32) - (1<<31)
    view(op.dst, destination_dtype, op.count)[:] = result.astype(destination_dtype)
  apply_gathers(image.gathers)
  mid:dict[int, list[RKGather]] = {}
  for gather in image.mid_gathers: mid.setdefault(gather.after, []).append(gather)
  for index in range(len(image.ew_ops)+1):
    apply_gathers(tuple(mid.get(index, ())))
    if index < len(image.ew_ops): execute(image.ew_ops[index])
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

def _terminal_gathers(image:RKImage) -> tuple[RKGather, ...]:
  return tuple(gather for gather in image.mid_gathers if gather.after == len(image.ew_ops))


def _bounded_int32_narrowing_program(tautological:bool=True) -> tuple[list[UOp], UOp]:
  """The generic bounded CDIV/CMOD and gated STORE shape emitted by expanded coordinate selection."""
  out, source = UOp.param(0, dtypes.int, (236,)), UOp.param(1, dtypes.int, (236,))
  lane = UOp.special(4, "lidx0", dtypes.int) + UOp.special(59, "gidx0", dtypes.int)*4
  value, zero, true = source.index(lane).load(), UOp.const(0, dtypes.int), UOp.const(True, dtypes.bool)
  negative = value < zero
  mod2, mod20 = (UOp(op, dtypes.int, src=(value, UOp.const(divisor, dtypes.int)))
                 for op,divisor in ((Ops.CMOD, 2), (Ops.CMOD, 20)))
  floor2 = UOp(Ops.CDIV, dtypes.int, src=(value, UOp.const(2, dtypes.int))) + \
    ((mod2 != zero) & negative).cast(dtypes.int)*-1
  floor20 = UOp(Ops.CDIV, dtypes.int, src=(value, UOp.const(20, dtypes.int))) + \
    ((mod20 != zero) & negative).cast(dtypes.int)*-1
  floor2_mod10 = UOp(Ops.CMOD, dtypes.int, src=(floor2, UOp.const(10, dtypes.int)))
  coordinate = (mod2 + ((mod2 != zero) & negative).where(UOp.const(2, dtypes.int), zero) != zero).where(
    floor2_mod10 + ((floor2_mod10 != zero) & (floor2 < zero)).where(UOp.const(10, dtypes.int), zero), floor20)
  valid = (negative != true) & (value < UOp.const(640, dtypes.int))
  gate = (valid != true) | (valid & (((coordinate != zero) != true) | valid) & ((coordinate != zero) | valid)) \
    if tautological else valid
  store = out.index(lane).store(valid.where(coordinate, zero), gate)
  return line_rewrite(list(store.sink().toposort()), pm_linearize_cleanups), gate


def test_rkvalue_is_the_typed_physical_abi():
  value = RKValue(RKArg(RKBufferKind.ARG, 0), dtypes.half, 1, RKLayout.FP16)
  assert value.dtype is dtypes.half and value.count == 1 and value.layout is RKLayout.FP16


def test_static_range_expressions_vectorize_in_output_order():
  row, col = UOp.range(3, 0), UOp.range(4, 1)
  out_index = col * 3 + row
  value = (row < 2).where(col + row * 4, UOp.const(-1, dtypes.int))
  assert _static_int_vector(out_index, value, 12) == (0, 4, -1, 1, 5, -1, 2, 6, -1, 3, 7, -1)


def test_static_gather_rows_share_output_range_materialization(monkeypatch):
  row, col = UOp.range(3, 0), UOp.range(4, 1)
  calls, original = 0, _iter_range_env
  def counted(ranges):
    nonlocal calls
    calls += 1
    return original(ranges)
  monkeypatch.setattr("tinygrad.renderer.rockchip._iter_range_env", counted)
  evaluator = RKStaticIndexEvaluator(col * 3 + row, 12)
  assert evaluator.offsets(row * 4 + col, None) == (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
  assert evaluator.offsets(11 - (row * 4 + col), None) == (11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0)
  assert evaluator.values(row * 4 + col, int) == (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
  assert calls == 1


def test_special_has_exact_static_integer_bounds():
  lane = UOp.special(7, "gidx0", dtypes.int)
  assert _exact_int_range(lane*3+2) == (2, 20)


def test_submit_timeout_poison_prevents_driver_retry(monkeypatch):
  class FakeDevice:
    fd_ctl, submit_count, task_count, _poisoned = object(), 0, 0, False
    def _sync_buffer(self, _buffer, _flags):
      if self._poisoned: raise RuntimeError("poisoned")
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
  try: program._submit(buffer, buffer, 1)
  except RuntimeError as exc: assert "platform NPU reset or power cycle required" in str(exc)
  else: raise AssertionError("timeout must poison the device")
  try: program._submit(buffer, buffer, 1)
  except RuntimeError as exc: assert "poisoned" in str(exc)
  else: raise AssertionError("poisoned device must reject submission")
  assert calls == 1 and program.dev._poisoned
  assert program.submit_count == program.dev.submit_count == program.dev.task_count == 0


def test_native_int16_fast_path_bounds_every_pc_chain(monkeypatch):
  scratch = RKArg(RKBufferKind.SCRATCH, 0)
  op = RKEWOp(scratch, scratch, scratch, 1, _EW_CFG[Ops.ADD], int16_input=True, int16_output=True)
  ops = (op,)*129
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = SimpleNamespace(_forget_program=lambda _program:None)
  program.image, program._scratch_ew_bodies = RKImage(ew_ops=ops), {ops:tuple((1,) for _ in ops)}
  submitted = []
  monkeypatch.setattr(program, "_submit_pcchain", lambda bodies:submitted.append(len(bodies)))
  program._run_ew_ops(lambda *_args:0, lambda *_args:None)
  assert submitted == [48, 48, 33]


def test_mixed_precision_path_bounds_native_pc_chains(monkeypatch):
  scratch = RKArg(RKBufferKind.SCRATCH, 0)
  native = RKEWOp(scratch, scratch, scratch, 1, _EW_CFG[Ops.ADD], int16_input=True, int16_output=True)
  terminal = RKEWOp(scratch, scratch, scratch, 1, _EW_CFG[Ops.MAX], int32_input=True, int32_output=True)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = SimpleNamespace(_forget_program=lambda _program:None)
  submitted = []
  monkeypatch.setattr(rockchip_runtime, "emit_ew_stage", lambda *_args, **_kwargs:(1,))
  monkeypatch.setattr(rockchip_runtime, "patch_stage", lambda stage,_address:stage)
  monkeypatch.setattr(program, "_submit_pcchain", lambda bodies:submitted.append(len(bodies)))
  monkeypatch.setattr(program, "_run_int32_conversion", lambda *_args:None)
  program._run_ew_ops(lambda *_args:0, lambda *_args:None, (native,)*54+(terminal,))
  assert submitted == [48, 6, 1]


def test_spatially_tiled_path_bounds_native_pc_chains(monkeypatch):
  scratch = RKArg(RKBufferKind.SCRATCH, 0)
  ops = (RKEWOp(scratch, scratch, scratch, _MAX_EW_ELEMS_FP16+1, _EW_CFG[Ops.ADD], stateful=True),) + \
        (RKEWOp(scratch, scratch, scratch, _MAX_EW_ELEMS_FP16+1, _EW_CFG[Ops.ADD]),)*48
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = SimpleNamespace(_forget_program=lambda _program:None)
  submitted = []
  monkeypatch.setattr(rockchip_runtime, "emit_ew_stage", lambda *_args, **_kwargs:(1,))
  monkeypatch.setattr(rockchip_runtime, "patch_stage", lambda stage,_address:stage)
  monkeypatch.setattr(program, "_submit_pcchain", lambda bodies:submitted.append(len(bodies)))
  program._run_ew_ops(lambda *_args:0, lambda *_args:None, ops)
  assert submitted == [48, 1, 48, 1]


def test_changed_pcchains_use_fresh_buffers_and_rearm_first_body(monkeypatch):
  events, freed = [], []
  class FakeBuffer:
    def __init__(self, size:int, object_id:int):
      self.storage = (ctypes.c_ubyte * size)()
      self.va_addr, self.size = ctypes.addressof(self.storage), size
      self.meta, self.base = SimpleNamespace(dma_addr=object_id << 20, obj_addr=object_id), self
  class FakeDevice:
    def __init__(self): self.next_object, self._ew_precision = 1, 0
    def _gpu_alloc(self, size:int, _flags:int=0):
      result = FakeBuffer(size, self.next_object)
      self.next_object += 1
      return result
    def _gpu_free(self, buffer): freed.append(buffer.meta.obj_addr)
    def _forget_program(self, _program): pass
    def reset_npu(self):
      events.append(("reset",))
      self._ew_precision = 0
  def format_body(process:int, output:int) -> tuple[int, ...]:
    return (rockchip_runtime._pc(rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DATA_FORMAT,
                                (output << 29) | (process << 26) | process),)
  def body(precision:int) -> tuple[int, ...]: return format_body(precision, precision)

  assert rockchip_runtime._rearm_body(body(4))[2] == rockchip_runtime._pc(
    rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 1 << 4)
  assert rockchip_runtime._rearm_body(format_body(1, 4))[2] == rockchip_runtime._pc(
    rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 1 << 4)
  assert rockchip_runtime._rearm_body(format_body(2, 5))[2] == rockchip_runtime._pc(
    rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 1 << 4)
  assert rockchip_runtime._rearm_body(format_body(4, 2))[2] == rockchip_runtime._pc(
    rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 0)
  assert rockchip_runtime._rearm_body(body(2))[2] == rockchip_runtime._pc(
    rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 0)

  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev, program._cmd_buf, program._task_buf, program._pcchain_bodies = FakeDevice(), None, None, None
  monkeypatch.setattr(program, "_submit", lambda cmd,task,n:events.append(("submit", n, cmd.meta.obj_addr, task.meta.obj_addr)))
  program._submit_pcchain([body(4)])
  program._submit_pcchain([body(1)])
  program._submit_pcchain([body(1)])
  program._submit_pcchain([body(4)])
  final_body = body(1)
  program._submit_pcchain([final_body, final_body])

  assert [event[0] for event in events] == ["submit", "reset", "submit", "submit", "submit", "reset", "submit"]
  submits = [event for event in events if event[0] == "submit"]
  assert submits[1][2:] == submits[2][2:]
  assert len({event[2:] for event in (submits[0], submits[1], submits[3], submits[4])}) == 4
  assert program._pcchain_bodies[0][:3] == (
    rockchip_runtime._pc(rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_S_POINTER, 0x30),
    rockchip_runtime._pc(rockchip_runtime._TARGET_DPU_RDMA, rockchip_runtime.rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30),
    rockchip_runtime._pc(rockchip_runtime._TARGET_DPU, rockchip_runtime.rk.REG_DPU_DST_SURF_STRIDE, 0))
  assert program._pcchain_bodies[1] == final_body
  assert len(freed) == 6

  event_start = len(events)
  program._submit_pcchain([body(2)])
  program._submit_pcchain([body(4)])
  program._submit_pcchain([body(2)])
  assert [event[0] for event in events[event_start:]] == ["submit", "submit", "reset", "submit"]
  assert program.dev._ew_precision == 2


def test_every_independent_fp16_chain_starts_with_full_state(monkeypatch):
  states, submitted = [], []
  def emit(*_args, stateful=False, **_kwargs):
    states.append(stateful)
    return (len(states),)
  monkeypatch.setattr(rockchip_runtime, "emit_ew_stage", emit)
  monkeypatch.setattr(rockchip_runtime, "patch_stage", lambda stage,_address:stage)
  program = object.__new__(rockchip_runtime.RockchipProgram)
  program.dev = SimpleNamespace(_forget_program=lambda _program:None)
  arg = RKArg(RKBufferKind.ARG, 0)
  ops = (RKEWOp(arg, arg, arg, 1, _EW_CFG[Ops.ADD]), RKEWOp(arg, arg, arg, 1, _EW_CFG[Ops.ADD]),
         RKEWOp(arg, arg, arg, 1, _EW_CFG[Ops.ADD], submit_barrier=True))
  monkeypatch.setattr(program, "_submit_pcchain", lambda bodies:submitted.append(tuple(bodies)))
  program._run_ew_ops(lambda *_args:0, lambda *_args:None, ops, tile_groups=False)
  assert states == [True, False, True] and tuple(map(len, submitted)) == (2, 1)


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


def test_masked_scaled_load_materializes_before_nonfinite_neutral_rewrite():
  source = UOp.param(1, dtypes.half, (4,))
  def selected(i:UOp) -> UOp:
    gate = i < UOp.const(2, dtypes.int)
    load = source.index(i-2).load(UOp.const(0.0, dtypes.half), gate != UOp.const(True, dtypes.bool))
    return gate.where(UOp.const(-math.inf, dtypes.half), gate.where(UOp.const(0.0, dtypes.half), load * -1.0))
  image = _lower_uop_program(_program(dtypes.half, selected))
  assert image is not None and not image.mid_gathers and len(image.ew_ops) == 1
  assert decode_image(encode_image(image)) == image


def test_bitcast_and_int16_masks_preserve_raw_fp16_sign_and_payload():
  magnitude, sign = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:
    ((magnitude.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.max, dtypes.int16)) |
     (sign.index(i).load().bitcast(dtypes.int16) & UOp.const(dtypes.int16.min, dtypes.int16))).bitcast(dtypes.half)))
  assert image is not None and len(image.ew_ops) == 11 and len(image.mid_gathers) == 11
  output = tuple(gather for gather in _terminal_gathers(image) if gather.dst_kind is RKBufferKind.ARG)
  assert len(output) == 1 and output[0].itemsize == 2
  assert decode_image(encode_image(image)) == image


def test_generic_bool_where_uses_canonical_int16_ternary():
  lhs, rhs = UOp.param(1, dtypes.int, (4,)), UOp.param(2, dtypes.int, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left != UOp.const(0, dtypes.int), UOp.const(False, dtypes.bool))
  image = _lower_uop_program(_program(dtypes.bool, select))
  assert image is not None and image.ew_ops[-1].int16_output
  assert len(_terminal_gathers(image)) == 1 and _terminal_gathers(image)[0].itemsize == 1


def test_inverted_fp16_comparison_keeps_ieee_unordered_semantics():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def greater_equal(i):
    less = UOp(Ops.CMPLT, dtypes.bool, src=(lhs.index(i).load(), rhs.index(i).load()))
    return UOp(Ops.CMPNE, dtypes.bool, src=(less, UOp.const(True, dtypes.bool)))
  image = _lower_uop_program(_program(dtypes.bool, greater_equal))
  assert image is not None and len(image.ew_ops) > 10
  assert _terminal_gathers(image) and _terminal_gathers(image)[-1].itemsize == 1
  assert not any(op.compare for op in image.ew_ops)
  assert image.ew_ops[-1].ew_cfg == _EW_CFG[Ops.MUL] and image.ew_ops[-1].int16_output


def test_fp16_equality_uses_exact_raw_bytes_without_compare_resets():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() != rhs.index(i).load()))
  assert image is not None and image.mid_gathers and _terminal_gathers(image)
  assert not any(op.compare for op in image.ew_ops) and all(op.int16_input and op.int16_output for op in image.ew_ops)


def test_generic_where_selects_infinity_without_mask_multiplication():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(-math.inf, dtypes.half))))
  assert image is not None and not image.ew_ops and len(_terminal_gathers(image)) == 2


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
  assert image is not None and not image.ew_ops and len(_terminal_gathers(image)) == 2
  assert any(gather.fill_bits == 0xfbff for gather in image.gathers)


def test_static_root_where_preserves_nonzero_constant_route():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half,
    lambda i:(i < UOp.const(2, dtypes.int)).where(source.index(i).load(), UOp.const(3.5, dtypes.half))))
  assert image is not None and not image.ew_ops and len(_terminal_gathers(image)) == 2
  assert struct.pack("<e", 3.5) in image.constants


def test_generic_bool_store_has_explicit_boundary_conversion():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() < rhs.index(i).load()))
  assert image is not None
  assert _terminal_gathers(image) and _terminal_gathers(image)[-1].itemsize == 1
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
  assert image is not None and not image.ew_ops and len(_terminal_gathers(image)) == 2
  assert all(gather.itemsize == 2 for gather in _terminal_gathers(image))


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


def test_bounded_int32_narrowing_keeps_runtime_bounds_exact():
  program, gate = _bounded_int32_narrowing_program()
  assert _bool_tautology(gate)
  image = _lower_uop_program(program)
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE
  assert not image.host_gathers and not image.host_scatters
  assert any(op.int32_input and op.int32_output for op in image.ew_ops)
  assert image.ew_ops[-1].int32_output and not image.ew_ops[-1].int32_input
  assert decode_image(encode_image(image)) == image


def test_bounded_int32_narrowing_rejects_inexact_intermediate():
  source, lane = UOp.param(1, dtypes.int, (4,)), UOp.range(1, 0)
  value, zero, true = source.index(lane).load(), UOp.const(0, dtypes.int), UOp.const(True, dtypes.bool)
  valid = ((value < zero) != true) & (value < UOp.const(4, dtypes.int))
  product = UOp(Ops.MUL, dtypes.int, src=(value, UOp.const(1023, dtypes.int)))
  remainder = UOp(Ops.CMOD, dtypes.int, src=(product, UOp.const(2, dtypes.int)))
  root = UOp(Ops.WHERE, dtypes.int, src=(valid, remainder, zero))
  bounded = UOp.special(4, "rockchip_bound", dtypes.int)
  assert _exact_int_range(remainder.substitute({value:bounded})) == (0, 1)
  assert _exact_int_range(product.substitute({value:bounded})) == (0, 3069)
  assert _bounded_int32_fp16_root(root) is None


def test_non_tautological_if_store_is_not_executed_unconditionally():
  program, gate = _bounded_int32_narrowing_program(False)
  assert not _bool_tautology(gate)
  assert _lower_uop_program(program) is None


def test_bounded_semantic_int_does_not_alias_int32_output_before_widening():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    (UOp.const(100.0, dtypes.half) < source.index(i).load()).cast(dtypes.int) * UOp.const(2500, dtypes.int)))
  assert image is not None and image.ew_ops[-1].int16_input and image.ew_ops[-1].int32_output
  assert all(op.dst.kind is RKBufferKind.SCRATCH for op in image.ew_ops[:-1])


def test_int32_load_store_is_raw_four_byte_materialization():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load()))
  assert image is not None and not image.ew_ops and len(_terminal_gathers(image)) == 1
  assert _terminal_gathers(image)[0].itemsize == 4 and _terminal_gathers(image)[0].dst_kind is RKBufferKind.ARG


def test_native_int32_mul_uses_the_canonical_wide_layout():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load() * source.index(i).load()))
  assert image is not None and len(image.ew_ops) == 1
  assert image.ew_ops[0].int32_input and image.ew_ops[0].int32_output


def test_wide_int32_cdiv_cmod_are_composable_uops():
  for op in (Ops.CDIV, Ops.CMOD):
    image = _lower_uop_program(_int32_binary_program(lambda lhs,rhs,op=op:UOp(op, dtypes.int, src=(lhs, rhs))))
    assert image is not None and image.execution_class is RKExecutionClass.NATIVE
    assert len(image.ew_ops) > 3000 and len(_terminal_gathers(image)) == 4
    assert decode_image(encode_image(image)) == image


def test_wide_int32_cdiv_cmod_physical_semantics():
  lhs, rhs = _int32_division_samples()
  for op in (Ops.CDIV, Ops.CMOD):
    image = _lower_uop_program(_int32_binary_program(lambda left,right,op=op:UOp(op, dtypes.int, src=(left, right)), len(lhs)))
    assert image is not None
    expected = [_trunc_divmod_int32(left, right)[op is Ops.CMOD] for left,right in zip(lhs.tolist(), rhs.tolist())]
    np.testing.assert_array_equal(_execute_integer_image(image, lhs, rhs), np.asarray(expected, dtype=np.int32))


def test_sibling_int32_cdiv_cmod_share_one_restoring_core():
  direct = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs))))
  combined = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:
    UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs)) + UOp(Ops.CMOD, dtypes.int, src=(lhs, rhs))))
  assert direct is not None and combined is not None
  assert len(direct.ew_ops) < len(combined.ew_ops) < len(direct.ew_ops)+100
  assert decode_image(encode_image(combined)) == combined


def test_int32_division_accepts_composed_operands():
  lhs, rhs = _int32_division_samples()
  image = _lower_uop_program(_int32_binary_program(lambda lhs,rhs:
    UOp(Ops.CDIV, dtypes.int, src=(lhs+1, rhs*3)), len(lhs)))
  assert image is not None and any(op.int32_input and op.int32_output for op in image.ew_ops)
  assert decode_image(encode_image(image)) == image
  expected = [_trunc_divmod_int32(_wrap_int32(left+1), _wrap_int32(right*3))[0]
              for left,right in zip(lhs.tolist(), rhs.tolist())]
  np.testing.assert_array_equal(_execute_integer_image(image, lhs, rhs), np.asarray(expected, dtype=np.int32))


def test_int32_floor_division_and_modulo_execute_ordinary_uops():
  lhs_values, rhs_values = _int32_division_samples()
  zero = UOp.const(0, dtypes.int)
  def expressions(lhs:UOp, rhs:UOp) -> tuple[UOp, UOp]:
    quotient, remainder = UOp(Ops.CDIV, dtypes.int, src=(lhs, rhs)), UOp(Ops.CMOD, dtypes.int, src=(lhs, rhs))
    correction = (remainder != zero) & ((lhs < zero) != (rhs < zero))
    return quotient + correction.cast(dtypes.int)*-1, remainder + correction.where(rhs, zero)
  for select in (0, 1):
    image = _lower_uop_program(_int32_binary_program(
      lambda lhs,rhs,select=select:expressions(lhs, rhs)[select], len(lhs_values)))
    assert image is not None and len(image.ew_ops) < 4000
    assert decode_image(encode_image(image)) == image
    expected = []
    for lhs,rhs in zip(lhs_values.tolist(), rhs_values.tolist()):
      quotient, remainder = _trunc_divmod_int32(lhs, rhs)
      correction = remainder != 0 and (lhs < 0) != (rhs < 0)
      expected.append(_wrap_int32(quotient-int(correction) if select == 0 else remainder+(rhs if correction else 0)))
    np.testing.assert_array_equal(_execute_integer_image(image, lhs_values, rhs_values), np.asarray(expected, dtype=np.int32))


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


def test_unrolled_math_reduction_executes_periodic_indices():
  out = UOp.param(0, dtypes.half, (1,))
  lhs, rhs, weights = (UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.half, (8,)),
                       UOp.param(3, dtypes.half, (2,)))
  terms = [lhs.index(i).load().exp2() * rhs.index(i).load() * weights.index(i%2).load() for i in range(8)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(0).store(value).sink().toposort()))
  assert image is not None and image.ew_ops and decode_image(encode_image(image)) == image


def test_batched_unrolled_math_reduction_executes_each_uop():
  rows, groups = 8, 4
  out, source = UOp.param(0, dtypes.half, (rows,)), UOp.param(1, dtypes.half, (rows*groups,))
  normalizer, lane = UOp.param(2, dtypes.half, (rows,)), UOp.range(rows, 0)
  terms = [(source.index(lane*groups+k).load() - normalizer.index(lane).load()).exp2() for k in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and image.ew_ops and decode_image(encode_image(image)) == image


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
  split = min(gather.after for gather in image.mid_gathers)
  assert len(image.gathers) == groups*2 and split >= 17 and split%17 == 0
  assert len(image.mid_gathers) == groups*2 and len(image.ew_ops) > split
  assert image.ew_ops[-1].ew_cfg == _EW_CFG[Ops.MAX]


def test_fp32_add_mul_tree_uses_half_expansion_at_output_boundary():
  out, lhs, rhs = UOp.param(0, dtypes.half, (2,)), UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  lane = UOp.range(2, 0)
  products = [lhs.index(lane*2+k).load().cast(dtypes.float) * rhs.index(lane*2+k).load().cast(dtypes.float) for k in range(2)]
  value = products[0].alu(Ops.ADD, products[1]).cast(dtypes.half)
  image = _lower_uop_program(list(out.index(lane).store(value).end(lane).sink().toposort()))
  assert image is not None and len(image.ew_ops) > 10


def test_repeated_composed_products_vectorize_then_retain_residuals():
  count = 128
  out = UOp.param(0, dtypes.half, (1,))
  lhs, bias, rhs = (UOp.param(slot, dtypes.half, (count,)) for slot in range(1, 4))
  terms = [(lhs.index(i).load()+bias.index(i//2).load())*rhs.index(count-1-i).load() for i in range(count)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  image = _lower_uop_program(list(out.index(0).store(value*-1.0).sink().toposort()))
  assert image is not None and len(image.mid_gathers) == count*2 and len(image.ew_ops) < 2000
  assert any(gather.values and set(gather.values) == {_fp16_bits(65.0)} for gather in image.gathers)
  assert any(op.submit_barrier for op in image.ew_ops)


def test_static_fp32_subgraph_rounds_only_after_coordinate_cancellation():
  lane = UOp.range(31, 0)
  coordinate = (lane.cast(dtypes.float)+UOp.const(0.5, dtypes.float))*UOp.const(20/31, dtypes.float) - \
    UOp.const(0.5, dtypes.float)
  fraction = coordinate - UOp(Ops.TRUNC, dtypes.float, src=(coordinate,))
  lowered = _fp32_expr_to_half(fraction)
  assert lowered.op is Ops.CAST and lowered.dtype.scalar() is dtypes.half and lowered.src == (fraction,)


def test_static_fp32_where_and_trunc_survive_precise_dynamic_storage():
  source, lane = UOp.param(1, dtypes.half, (8,)), UOp.range(4, 0)
  coordinate = (lane.cast(dtypes.float)+UOp.const(0.5, dtypes.float))*UOp.const(3/7, dtypes.float)
  fraction = coordinate - UOp(Ops.TRUNC, dtypes.float, src=(coordinate,))
  weight = (lane < UOp.const(2, dtypes.int)).where(fraction, UOp.const(1.0, dtypes.float)-fraction)
  value = source.index(lane*2).load().cast(dtypes.float)*weight + \
          source.index(lane*2+1).load().cast(dtypes.float)*(UOp.const(1.0, dtypes.float)-weight)
  image = _lower_uop_program(_program(dtypes.half, lambda i:value.substitute({lane:i}).cast(dtypes.half)))
  assert image is not None and image.ew_ops[-1].dst.kind is RKBufferKind.ARG


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
  composed = _lower_uop_program(_program(dtypes.bool, lambda i:(source.index(8-i).load()+1.0).cast(dtypes.bool), count=9))
  assert integer is not None and integer.ew_ops[-1].int32_output
  assert boolean is not None and boolean.ew_ops[-1].bool_output and composed is not None and composed.ew_ops[-1].bool_output
  assert decode_image(encode_image(integer)) == integer and decode_image(encode_image(boolean)) == boolean and \
    decode_image(encode_image(composed)) == composed


def test_terminal_uint8_cast_and_where_use_int16_physical_values():
  source = UOp.param(1, dtypes.half, (9,))
  for value in (lambda i:source.index(i).load().cast(dtypes.uchar),
                lambda i:(UOp.const(0.0, dtypes.half) < source.index(i).load()).where(
                  source.index(i).load().cast(dtypes.uchar), UOp.const(0, dtypes.uchar))):
    image = _lower_uop_program(_program(dtypes.uchar, value, count=9))
    assert image is not None and image.ew_ops[-1].int16_output and _terminal_gathers(image)[-1].itemsize == 1
    assert decode_image(encode_image(image)) == image


def test_bounded_sums_of_native_half_comparisons_stay_int16_until_output():
  out, source = UOp.param(0, dtypes.int, (1,)), UOp.param(1, dtypes.half, (4,))
  terms = [(source.index(i).load() < UOp.const(0.5, dtypes.half)).cast(dtypes.int) for i in range(4)]
  image = _lower_uop_program(list(out.index(0).store(terms[0]+terms[1]+terms[2]+terms[3]).sink().toposort()))
  assert image is not None and len(image.mid_gathers) == 10 and len({g.after for g in image.mid_gathers}) == 5
  assert image.ew_ops[-1].int16_input and image.ew_ops[-1].int32_output


def test_fp32_pure_add_tree_uses_compensated_half_expansion_at_output_boundary():
  source = UOp.param(1, dtypes.half, (64,))
  terms = [source.index(i).load().cast(dtypes.float) for i in range(64)]
  value = terms[0]
  for term in terms[1:]: value = value + term
  image = _lower_uop_program(_program(dtypes.half, lambda _i:value.cast(dtypes.half), count=1))
  assert image is not None and 64 < len(image.ew_ops) < 2000


def test_fp32_nonfinite_add_uses_plain_half_storage_without_residuals():
  source = UOp.param(1, dtypes.half, (1,)).index(0).load()
  gate = source.alu(Ops.CMPLT, UOp.const(0.0, dtypes.half))
  selected = UOp(Ops.WHERE, dtypes.float, src=(gate, UOp.const(math.inf, dtypes.float), UOp.const(0.0, dtypes.float)))
  lowered = _fp32_expr_to_half(UOp(Ops.ADD, dtypes.float, src=(source.cast(dtypes.float), selected)))
  assert lowered.op is Ops.ADD and lowered.arg is None
  assert sum(node.op is Ops.ADD for node in lowered.toposort()) == 1


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


def test_indexed_local_bridge_and_boolean_accumulators_are_structurally_executed():
  out, source = UOp.param(0, dtypes.bool, (2,)), UOp.param(1, dtypes.half, (8,))
  group, worker = UOp.special(2, "gidx0", dtypes.int), UOp.special(2, "lidx0", dtypes.int)
  first = UOp.placeholder((1,), dtypes.bool, 0, addrspace=AddrSpace.REG)
  first_init = first.index(0).store(True)
  first_axis = UOp.range(2, 0, AxisType.REDUCE)
  first_ptr = first.after(first_init, first_axis).index(0)
  present = source.index(group*4+worker*2+first_axis).load() != UOp.const(0.0, dtypes.half)
  first_update = first_ptr.store(first_ptr.load() & present)
  first_end = first_update.end(first_axis)

  bridge = UOp.placeholder((2,), dtypes.bool, 0, addrspace=AddrSpace.LOCAL)
  bridge_store = bridge.index(worker).store(first.after(first_end).index(0).load())
  second = UOp.placeholder((1,), dtypes.bool, 1, addrspace=AddrSpace.REG)
  second_init = second.index(0).store(True)
  second_axis = UOp.range(2, 1, AxisType.REDUCE, src=(first_end,))
  second_ptr = second.after(second_init, second_axis).index(0)
  second_update = second_ptr.store(second_ptr.load() & bridge.after(bridge_store.barrier()).index(second_axis).load())
  result = second.after(second_update.end(second_axis)).index(0).load()
  output = out.index(group).store(result)

  image = _lower_uop_program(list(UOp.sink(first_init, first_update, bridge_store, second_init, second_update, output).toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  assert decode_image(encode_image(image)) == image


def test_dependent_scalar_local_extrema_executes_as_ordinary_uops():
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
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  assert any(op.int32_input or op.int32_output for op in image.ew_ops)
  assert decode_image(encode_image(image)) == image


def test_large_mapped_scalar_max_uses_compact_affine_materialization():
  count = 1022
  out, source = UOp.param(0, dtypes.int, (count,)), UOp.param(1, dtypes.half, (count,))
  prefix, blocks = UOp.param(2, dtypes.half, (1024,)), UOp.param(3, dtypes.half, (4,))
  group, worker = UOp.special(511, "gidx0", dtypes.int), UOp.special(2, "lidx0", dtypes.int)
  lane, axis = worker+group*2, UOp.range(count, 0, AxisType.REDUCE)
  local = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)
  initialize = local.index(0).store(UOp.const(dtypes.int.min, dtypes.int))
  pointer = local.after(initialize, axis).index(0)
  block = UOp(Ops.CDIV, dtypes.int, src=(group+1, UOp.const(128, dtypes.int)))
  reference = prefix.index(lane+2).load().maximum(blocks.index(block).load())
  equal = (source.index(axis).load() != reference) != UOp.const(True, dtypes.bool)
  candidate = (equal & ((lane < axis) != UOp.const(True, dtypes.bool))).cast(dtypes.int)*(count-axis)
  update = pointer.store(pointer.load().maximum(candidate))
  result = local.after(update.end(axis)).index(0).load()
  image = _lower_uop_program(list(UOp.sink(initialize, update, out.index(lane).store(count-result)).toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE
  raw = encode_image(image)
  assert len(raw) < 105_000 and len(image.ew_ops) < 1_120 and len(image.mid_gathers) == count
  assert max(max(len(gather.offsets), len(gather.values)) for gather in image.gathers) <= count
  assert decode_image(raw) == image


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
  assert image is not None and any(op.int32_output for op in image.ew_ops)
  assert decode_image(encode_image(image)) == image


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
  assert image is not None and any(op.int32_output for op in image.ew_ops)
  assert decode_image(encode_image(image)) == image


def test_direct_dynamic_int32_load_selects_all_raw_bytes():
  out, source, indices = (UOp.param(0, dtypes.int, (4,)), UOp.param(1, dtypes.int, (9,)), UOp.param(2, dtypes.int, (4,)))
  lane = UOp.range(4, 0)
  index = indices.index(lane).load()
  gate = ((index < 0) != UOp.const(True, dtypes.bool)) & (index < 9)
  load = source.index(index).load(UOp.const(0, dtypes.int), gate)
  image = _lower_uop_program(list(out.index(lane).store(load).end(lane).sink().toposort()))
  assert image is not None and len(_terminal_gathers(image)) == 4
  assert {gather.dst_addend for gather in _terminal_gathers(image)} == {0, 1, 2, 3}
  assert decode_image(encode_image(image)) == image


def test_dynamic_index_materializer_composes_external_bool_gate():
  out, source = UOp.param(0, dtypes.half, (4,)), UOp.param(1, dtypes.half, (4,))
  indices, mask, lane = UOp.param(2, dtypes.int, (4,)), UOp.param(3, dtypes.bool, (4,)), UOp.range(4, 0)
  index = indices.index(lane).load()
  gate = (((index < 0) != UOp.const(True, dtypes.bool)) & (index < 4)) & mask.index(lane).load()
  image = _lower_uop_program(list(out.index(lane).store(source.index(index).load(UOp.const(0.0, dtypes.half), gate)).end(lane).sink().toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  assert any(gather.src_index == mask.arg.slot and gather.itemsize == 1 for gather in image.gathers)
  assert decode_image(encode_image(image)) == image


def test_int32_bounds_predicates_execute_as_ordinary_uops():
  out, lhs, rhs = UOp.param(0, dtypes.bool, (4,)), UOp.param(1, dtypes.int, (4,)), UOp.param(2, dtypes.int, (4,))
  lane = UOp.range(4, 0)
  values = [source.index(lane).load() for source in (lhs, rhs)]
  bounded = [(value < 0).where(value+5, value) for value in values]
  valid = [((value < 0) != UOp.const(True, dtypes.bool)) & (value < 5) for value in bounded]
  image = _lower_uop_program(list(out.index(lane).store(valid[0] & valid[1]).end(lane).sink().toposort()))
  assert image is not None and image.execution_class is RKExecutionClass.NATIVE and not image.host_gathers
  assert any(op.int32_input and op.int32_output for op in image.ew_ops)
  assert any(gather.dst_kind is RKBufferKind.ARG and gather.itemsize == 1 for gather in _terminal_gathers(image))
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


def test_int32_bitwise_uop_executes_over_raw_byte_planes():
  lhs, rhs = UOp.param(1, dtypes.int, (4,)), UOp.param(2, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:lhs.index(i).load() & rhs.index(i).load()))
  assert image is not None and len(_terminal_gathers(image)) == 1
  assert _terminal_gathers(image)[0].count == 16 and all(op.int16_input and op.int16_output for op in image.ew_ops)


def test_embedded_int32_not_preserves_all_raw_bytes_before_wide_arithmetic():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:
    UOp(Ops.XOR, dtypes.int, src=(source.index(i).load(), UOp.const(-1, dtypes.int)))+1))
  assert image is not None and len(image.ew_ops) == 6 and len(image.mid_gathers) == 8
  assert sum(op.int32_input and op.int32_output for op in image.ew_ops) == 2


def test_int32_shift_uop_executes_with_byte_plane_barrel_recipe():
  source = UOp.param(1, dtypes.int, (4,))
  image = _lower_uop_program(_program(dtypes.int, lambda i:source.index(i).load() << UOp.const(2, dtypes.int)))
  assert image is not None and len(_terminal_gathers(image)) == 4
  assert {gather.dst_addend for gather in _terminal_gathers(image)} == {0, 1, 2, 3}


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
  assert image is not None and image.ew_ops
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
  rewritten, _ = _finite_max_neutrals(value, value.toposort())
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


def test_leading_vector_materialization_is_composed_into_initial_lane_gathers():
  arg, vector, copied, arena = RKArg(RKBufferKind.ARG, 1), *(RKArg(RKBufferKind.SCRATCH, i) for i in range(3))
  image = RKImage(tuple(RKScratch(128) for _ in range(3)),
    gathers=(RKGather(arg.index, vector.index, 4, axes=((1, 4, 1),)),),
    ew_ops=(RKEWOp(copied, vector, vector, 4, _EW_CFG[Ops.MAX]),
            RKEWOp(copied, arena, RKArg(arena.kind, arena.index, 32), 1, _EW_CFG[Ops.ADD]),
            RKEWOp(RKArg(RKBufferKind.ARG, 0), copied, RKArg(arena.kind, arena.index, 64), 1, _EW_CFG[Ops.ADD])),
    mid_gathers=tuple(RKGather(copied.index, arena.index, 1, base=lane, dst_addend=lane*32,
                               src_kind=RKBufferKind.SCRATCH, after=1) for lane in range(4)))
  folded = _hoist_leading_vector_materialization(image)
  assert len(folded.ew_ops) == 2 and not folded.mid_gathers and len(folded.gathers) == 5
  assert tuple(gather.offsets for gather in folded.gathers[1:]) == ((0,), (1,), (2,), (3,))


def test_scratch_reuse_spans_mid_gathers_without_aliasing_their_state():
  scratch = tuple(RKScratch(64) for _ in range(5))
  arg, slots = RKArg(RKBufferKind.ARG, 0), tuple(RKArg(RKBufferKind.SCRATCH, i) for i in range(5))
  image = RKImage(scratch, gathers=(RKGather(0, 0, 1, values=(0,)),), ew_ops=(
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


def test_nonaffine_scalar_mul_sum_executes_as_uops():
  groups = 64
  out, lhs, rhs = UOp.param(0, dtypes.half, (1,)), UOp.param(1, dtypes.half, (groups,)), UOp.param(2, dtypes.half, (groups,))
  permutation = tuple((lane*17)%groups for lane in range(groups))
  terms = [lhs.index(permutation[lane]).load()*rhs.index(lane).load() for lane in range(groups)]
  value = terms[0]
  for term in terms[1:]: value = value+term
  image = _lower_uop_program(list(out.index(UOp.const(0, dtypes.int)).store(value).sink().toposort()))
  assert image is not None and image.ew_ops and decode_image(encode_image(image)) == image
