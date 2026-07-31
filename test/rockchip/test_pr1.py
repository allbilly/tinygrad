# PR 1 hardware-free tests: classifier, emitter, codec, determinism.
# These tests do not require an NPU and run under DEV=NULL.
import unittest, struct
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.codegen import early_simplify
from tinygrad.uop.ops import Ops, ProgramInfo, graph_rewrite
from tinygrad.codegen import pm_to_program
from tinygrad.runtime.support.rockchip import (plan_rk, emit_rk, encode_rk, decode_rk, encode_rk_multi, decode_rk_multi,
                                               build_native_program, RKPlan, RKSubTask, _HOST_FP32_HALF_LAYOUT,
                                               _HOST_FP32_RESIDUAL_LAYOUT, _HOST_FP32_COMBINE_LAYOUT, _HOST_HALF_FP32_LAYOUT,
                                               _HOST_ELEMENTWISE_LAYOUT, _HOST_VARIANCE_LAYOUT, _HOST_SOFTMAX_ARGMAX_LAYOUT,
                                               _HOST_STATIC_HALF_LAYOUT, _HOST_SCATTER_LAYOUT, _HOST_ARGMAX_LAYOUT,
                                               _HOST_AVG_POOL_LAYOUT, _HOST_ELEMENTWISE_REDUCE_LAYOUT, _HOST_BCE_LAYOUT,
                                               _HOST_CROSS_ENTROPY_LAYOUT, _HOST_NLL_LAYOUT, _HOST_EINSUM_LAYOUT,
                                               _HOST_BILINEAR_LAYOUT)
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipRenderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.helpers import Target

class TestSubmitBufferLifecycle(unittest.TestCase):
  def test_refresh_replaces_then_frees_submit_buffers(self):
    class FakeDevice:
      cmd_buf_size = 16
      cmd_buf, task_buf = object(), object()
      def __init__(self): self.allocated, self.freed = [], []
      def _gpu_alloc(self, size, flags, name):
        ret = object()
        self.allocated.append((size, flags, name, ret))
        return ret
      def _gpu_free(self, buf): self.freed.append(buf)
    dev = FakeDevice()
    old_cmd, old_task = dev.cmd_buf, dev.task_buf
    RockchipDevice.refresh_submit_buffers(dev)
    self.assertEqual([(x[0], x[1], x[2]) for x in dev.allocated],
                     [(dev.cmd_buf_size * 8, 0, "cmd_buf"), (1024, rk.RKNPU_MEM_KERNEL_MAPPING, "task_buf")])
    self.assertEqual(dev.freed, [old_cmd, old_task])
    self.assertIs(dev.cmd_buf, dev.allocated[0][3])
    self.assertIs(dev.task_buf, dev.allocated[1][3])

def _get_sink(expr):
  lin = expr.schedule_linear()
  ks = [c.src[0] for c in lin.src if c.src[0].op is Ops.SINK]
  return early_simplify(ks[0])

def _classify(sink):
  result = plan_rk(sink)
  return result.kind if isinstance(result, RKPlan) else result

def _emit(sink):
  plan = plan_rk(sink)
  assert isinstance(plan, RKPlan), f"plan_rk returned reject: {plan}"
  return emit_rk(plan)

class TestClassifier(unittest.TestCase):
  def test_dpu_add(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+b)), "dpu")

  def test_dpu_mul(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a*b)), "dpu")

  def test_dpu_sub(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a-b)), "dpu")

  def test_dpu_max(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.maximum(b))), "dpu")

  def test_dpu_relu_via_where_max(self):
    # relu(x) = WHERE(CMPLT(0, x), x, 0) = MAX(x, 0) → DPU
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.relu())), "dpu")

  def test_dpu_2d_contiguous_add(self):
    # 2D row-major contiguous output — _is_flat_contiguous accepts 2D
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+b)), "dpu")

  def test_dpu_3d_contiguous_add(self):
    # 3D row-major contiguous — exercises nested 2D row-major index
    a = Tensor.rand(8,4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(8,4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+b)), "dpu")

  def test_dpu_neg(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(-a)), "dpu")

  def test_dpu_copy(self):
    # Copy (STORE(INDEX)) is a real NPU DMA pass-through, not host-side memmove
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+0)), "dpu")

  def test_dpu_scalar_add(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+1)), "dpu")

  def test_dpu_scalar_mul(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a*2)), "dpu")

  def test_dpu_scalar_max(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.maximum(1))), "dpu")

  def test_dpu_single_element(self):
    a = Tensor.rand(1,1,dtype=dtypes.half).realize()
    b = Tensor.rand(1,1,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a+b)), "dpu")

  def test_cmac_matmul(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@b)), "cmac")

  def test_cmac_1x1_conv(self):
    # 1x1 conv = pointwise GEMM with transposed A/B pattern
    x = Tensor.rand(1,4,3,3,dtype=dtypes.half).realize()
    w = Tensor.rand(4,4,1,1,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(x.conv2d(w))), "cmac")

  def test_cmac_fused_bias_conv(self):
    # Channel bias is applied to the raw fp32 CMAC accumulator before fp16 rounding.
    x = Tensor.rand(1,4,3,3,dtype=dtypes.half).realize()
    w = Tensor.rand(4,4,1,1,dtype=dtypes.half).realize()
    b = Tensor.rand(4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(x.conv2d(w, b))), "cmac")

  def test_cmac_avg_pool_scale_and_padding(self):
    x = Tensor.rand(1,2,5,6,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(x.avg_pool2d(kernel_size=(3,2), padding=(1,0)))), "cmac")

  def test_ppu_max(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.max(axis=0))), "ppu")

  def test_ppu_max_flexible_channels(self):
    for shape in [(8,4),(4,4),(8,8),(8,2),(4,2),(8,6),(4,6),(8,3),(8,1),(4,1)]:
      a = Tensor.rand(*shape,dtype=dtypes.half).realize()
      self.assertEqual(_classify(_get_sink(a.max(axis=0))), "ppu", f"shape {shape}")

  def test_ppu_max_k_split(self):
    # K=64 splits as (8,8) — both factors in [2,16]
    a = Tensor.rand(64,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.max(axis=0))), "ppu")

  def test_ppu_max_k_prime_fallback(self):
    # K=5 (prime ≤ 16) uses in_h=1 fallback
    a = Tensor.rand(5,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.max(axis=0))), "ppu")

  def test_ppu_max_k_prime_rejected(self):
    # K=17 (prime > 16) cannot be split — rejected
    a = Tensor.rand(17,8,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a.max(axis=0))))

  def test_ppu_max_chan1(self):
    # channels=1: MUL eliminated, INDEX uses RANGE directly
    a = Tensor.rand(8,1,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.max(axis=0))), "ppu")

  def test_reject_int(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize().cast(dtypes.int)
    b = Tensor.rand(4,4,dtype=dtypes.half).realize().cast(dtypes.int)
    self.assertTrue(_classify(_get_sink(a+b)).startswith("RKPLAN_REJECT"))

  def test_float32_dpu(self):
    a = Tensor.rand(4,4,dtype=dtypes.float).realize()
    b = Tensor.rand(4,4,dtype=dtypes.float).realize()
    self.assertEqual(_classify(_get_sink(a+b)), "dpu")

  def test_sum_full(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.sum())), "cmac")  # full sum → M=1,N=1 via ones-vector

  def test_mean_full(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.mean())), "cmac")

  def test_sum_axis1(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.sum(axis=1))), "cmac")

  def test_sum_axis0(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.sum(axis=0))), "cmac")

  def test_mean_axis1(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.mean(axis=1))), "cmac")

  def test_mean_axis0(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.mean(axis=0))), "cmac")

  def test_broadcast_row_rejected(self):
    # Broadcast is rejected in PR1 — no host-side materialization
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,1,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a+b)))

  def test_broadcast_col_rejected(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a+b)))

  def test_const_fill_zeros(self):
    # Constant fill is a DPU ADD(zero, const) — zero buffer is prep, DPU does the fill
    self.assertEqual(_classify(_get_sink(Tensor.zeros(4,4,dtype=dtypes.half))), "dpu")

  def test_const_fill_ones(self):
    self.assertEqual(_classify(_get_sink(Tensor.ones(4,4,dtype=dtypes.half))), "dpu")

  def test_const_fill_full(self):
    self.assertEqual(_classify(_get_sink(Tensor.full((4,4), 3.0, dtype=dtypes.half))), "dpu")

  def test_reject_transpose(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a.T+b)))

  def test_reject_ppu_wrong_channels(self):
    a = Tensor.rand(8,4,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a.max(axis=1))))

  def test_cmac_transposed_b(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@b.T)), "cmac")

  def test_cmac_transposed_a(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.T@b)), "cmac")

  def test_cmac_gemv_vector_a(self):
    # GEMV: (K,) @ (K,N) → (N,) — vector is A, M=1
    a = Tensor.rand(4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@b)), "cmac")

  def test_cmac_scalar_dot(self):
    a = Tensor.rand(5,dtype=dtypes.half).realize()
    b = Tensor.rand(5,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a.dot(b))), "cmac")

  def test_cmac_batched_n_tiled_matmul(self):
    a = Tensor.rand(3,4,5,dtype=dtypes.half).realize()
    b = Tensor.rand(3,5,40,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@b)), "cmac")

  def test_cmac_gemv_vector_b(self):
    # GEMV: (M,K) @ (K,) → (M,) — vector is B, N=1
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    v = Tensor.rand(4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@v)), "cmac")

  def test_reject_ppu_k_too_large(self):
    a = Tensor.rand(17,8,dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a.max(axis=0))))

  def test_reject_cmac_strided_a(self):
    a = Tensor.rand(8,4,dtype=dtypes.half).realize()
    b = Tensor.eye(4, dtype=dtypes.half).realize()
    self.assertIn("REJECT", _classify(_get_sink(a[::2]@b)))

  def test_cmac_same_buffer(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink(a@a)), "cmac")

  def test_cmac_scaled_sum_full(self):
    # REDUCE(ADD, MUL(INDEX, CONST(c))) — scaled full sum
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink((a*2).sum())), "cmac")

  def test_cmac_scaled_sum_axis0(self):
    # Scaled sum over axis=0: ones(1,K) @ (a*c)
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink((a*3).sum(axis=0))), "cmac")

  def test_dpu_cast_wrapping_ew(self):
    # CAST(half→half, ADD(INDEX, INDEX)) should be classified as DPU, not rejected as Ops.CAST
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    self.assertEqual(_classify(_get_sink((a+b).cast(dtypes.half))), "dpu")

  def test_reject_raises_runtime_error(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    sink = _get_sink(a.T+b)  # transpose is rejected
    with self.assertRaises(RuntimeError) as cm:
      build_native_program(sink)
    self.assertIn("RKPLAN_REJECT", str(cm.exception))

  def test_long_cumprod_uses_logarithmic_scan(self):
    sink = _get_sink(Tensor.empty(512, device="ROCKCHIP").cumprod(0))
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertTrue(all(isinstance(task, RKSubTask) for task in subtasks))
    self.assertLess(len(subtasks), 600)

  def test_cumsum_uses_bounded_typed_stages(self):
    for dtype in (dtypes.half, dtypes.float):
      for size, expected_tags in ((512, (_HOST_ELEMENTWISE_REDUCE_LAYOUT,)),
                                  (1022, (_HOST_ELEMENTWISE_REDUCE_LAYOUT, _HOST_ELEMENTWISE_REDUCE_LAYOUT,
                                          _HOST_ELEMENTWISE_LAYOUT))):
        source = Tensor.empty(size, dtype=dtype, device="ROCKCHIP")
        sinks = [early_simplify(call.src[0]) for call in source.cumsum(0).schedule_linear().src if call.src[0].op is Ops.SINK]
        self.assertEqual(len(sinks), len(expected_tags))
        for sink, expected_tag in zip(sinks, expected_tags):
          program = build_native_program(sink)
          self.assertIsNotNone(program)
          subtasks = program.src[1].src[0].arg
          self.assertEqual(len(subtasks), 1)
          self.assertEqual(subtasks[0].task.layout[1], expected_tag)

  def test_fp16_cumulative_extrema_use_typed_reductions_and_indices(self):
    for kind in ("max", "min"):
      source = Tensor.empty(1022, dtype=dtypes.half, device="ROCKCHIP")
      values = (source.cummax(0) if kind == "max" else source.cummin(0))[0]
      value_sinks = [early_simplify(call.src[0]) for call in values.schedule_linear().src if call.src[0].op is Ops.SINK]
      self.assertEqual(len(value_sinks), 3)
      for program in (build_native_program(sink) for sink in value_sinks[:2]):
        self.assertIsNotNone(program)
        subtasks = program.src[1].src[0].arg
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      if kind == "min":
        final_program = build_native_program(value_sinks[2])
        self.assertIsNotNone(final_program)
        final_subtasks = final_program.src[1].src[0].arg
        self.assertEqual(len(final_subtasks), 1)
        self.assertEqual(final_subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

      index_source = Tensor.empty(5, dtype=dtypes.half, device="ROCKCHIP")
      indices = (index_source.cummax(0) if kind == "max" else index_source.cummin(0))[1]
      index_sinks = [early_simplify(call.src[0]) for call in indices.schedule_linear().src if call.src[0].op is Ops.SINK]
      index_program = build_native_program(index_sinks[-1])
      self.assertIsNotNone(index_program)
      index_subtasks = index_program.src[1].src[0].arg
      self.assertEqual(len(index_subtasks), 1)
      self.assertEqual(index_subtasks[0].task.layout[1], _HOST_ARGMAX_LAYOUT)
      self.assertEqual(index_subtasks[0].task.layout[4] == 10, kind == "min")
      mapping_offset = 5 if kind == "min" else 4
      self.assertEqual(index_subtasks[0].task.layout[mapping_offset+20:mapping_offset+25], (4, 3, 2, 1, 0))

      long_source = Tensor.empty(1022, dtype=dtypes.half, device="ROCKCHIP")
      long_indices = (long_source.cummax(0) if kind == "max" else long_source.cummin(0))[1]
      long_sinks = [early_simplify(call.src[0]) for call in long_indices.schedule_linear().src if call.src[0].op is Ops.SINK]
      long_program = build_native_program(long_sinks[-1])
      self.assertIsNotNone(long_program)
      long_subtasks = long_program.src[1].src[0].arg
      self.assertEqual(len(long_subtasks), 1)
      self.assertEqual(long_subtasks[0].task.layout, (1022, _HOST_ARGMAX_LAYOUT, 0, 1022, 10 if kind == "min" else 2))

      if kind == "min":
        short_source = Tensor.empty(512, dtype=dtypes.half, device="ROCKCHIP")
        short_indices = short_source.cummin(0)[1]
        short_sinks = [early_simplify(call.src[0]) for call in short_indices.schedule_linear().src if call.src[0].op is Ops.SINK]
        short_program = build_native_program(short_sinks[0])
        self.assertIsNotNone(short_program)
        short_subtasks = short_program.src[1].src[0].arg
        self.assertEqual(len(short_subtasks), 1)
        self.assertEqual(short_subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_fp32_sum_uses_typed_cmac_boundary(self):
    sink = _get_sink(Tensor.empty(3,3, dtype=dtypes.float, device="ROCKCHIP").sum())
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertGreaterEqual(len(subtasks), 4)
    copy_layouts = [task.task.layout[1] for task in subtasks if task.task.is_copy]
    self.assertIn(_HOST_FP32_HALF_LAYOUT, copy_layouts)
    self.assertIn(_HOST_FP32_RESIDUAL_LAYOUT, copy_layouts)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_large_fp32_sum_uses_two_limb_cmac_boundary(self):
    sink = _get_sink(Tensor.empty(4,6,8, dtype=dtypes.float, device="ROCKCHIP").sum())
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    copy_layouts = [task.task.layout[1] for task in subtasks if task.task.is_copy]
    self.assertIn(_HOST_FP32_HALF_LAYOUT, copy_layouts)
    self.assertIn(_HOST_FP32_RESIDUAL_LAYOUT, copy_layouts)
    self.assertEqual(sum(task.task.kind == "cmac" for task in subtasks), 2)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_full_fp32_mean_uses_scalar_factorized_epilogue(self):
    program = build_native_program(_get_sink(Tensor.empty(3,4,5,6, dtype=dtypes.float, device="ROCKCHIP").mean()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertTrue(any(task.task.kind == "cmac" for task in subtasks))
    scalar_views = [task.task.layout for task in subtasks if task.task.is_copy and len(task.task.layout) == 4]
    self.assertIn((1, _HOST_FP32_HALF_LAYOUT, 0, 0), scalar_views)
    self.assertIn((1, _HOST_FP32_RESIDUAL_LAYOUT, 0, 0), scalar_views)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_variance_uses_one_strict_serialized_task(self):
    expression = Tensor.empty(15,25,35, dtype=dtypes.float, device="ROCKCHIP").var((1,2))
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertTrue(subtasks[0].task.fp32_output)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_VARIANCE_LAYOUT)
    self.assertEqual(subtasks[0].task.layout[4], 0)

  def test_fp32_std_sets_strict_variance_sqrt_epilogue(self):
    expression = Tensor.empty(15,25,35, dtype=dtypes.float, device="ROCKCHIP").std((1,2))
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_VARIANCE_LAYOUT)
    self.assertEqual(subtasks[0].task.layout[4], 1)

  def test_fp32_std_mean_sets_strict_stacked_epilogue(self):
    expression = Tensor.stack(*Tensor.empty(3,4,5,6, dtype=dtypes.float, device="ROCKCHIP").std_mean(axis=(1,2)))
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_VARIANCE_LAYOUT)
    self.assertEqual(subtasks[0].task.layout[4], 2)

  def test_fp32_topology_uses_exact_host_boundary(self):
    x = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink((x+x)*x))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_broadcast_uses_exact_host_boundary(self):
    for op in (Tensor.div, Tensor.pow):
      a = Tensor.empty(1,3,1,7,1, dtype=dtypes.float, device="ROCKCHIP")
      b = Tensor.empty(2,1,5,1,8, dtype=dtypes.float, device="ROCKCHIP")
      program = build_native_program(_get_sink(op(a,b)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertTrue(subtasks[0].task.is_copy)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp16_broadcast_pow_uses_exact_host_boundary(self):
    for lhs_shape, rhs_shape in (((5,3,14,16), (5,1,14,1)), ((1,3,1,7,1), (2,1,5,1,8))):
      lhs = Tensor.empty(*lhs_shape, dtype=dtypes.half, device="ROCKCHIP")
      rhs = Tensor.empty(*rhs_shape, dtype=dtypes.half, device="ROCKCHIP")
      program = build_native_program(_get_sink(lhs.pow(rhs)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertTrue(subtasks[0].task.is_copy)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp16_constant_base_integer_pow_uses_exact_host_boundary(self):
    default_float = dtypes.default_float
    try:
      dtypes.default_float = dtypes.half
      sink = _get_sink(0.7**Tensor.empty(6, dtype=dtypes.int, device="ROCKCHIP"))
    finally:
      dtypes.default_float = default_float
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_padded_add_uses_exact_host_boundary(self):
    a = Tensor.empty(64,64, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(60,60, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a+b.pad((2,2,2,2))))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_conditional_movement_uses_exact_host_boundary(self):
    for mode in ("reflect", "replicate"):
      expression = Tensor.empty(1,1,5,5, dtype=dtypes.float, device="ROCKCHIP").pad((0,2,3,2), mode=mode)
      program = build_native_program(_get_sink(expression))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertTrue(subtasks[0].task.is_copy)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_nearest_interpolation_keeps_float_coordinate_scale(self):
    expression = Tensor.empty(2,3,13, dtype=dtypes.float, device="ROCKCHIP").interpolate((9,), mode="nearest")
    program = build_native_program(_get_sink(expression))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fancy_index_preprocessing_uses_typed_host_boundary(self):
    x = Tensor.empty(2,5,6,5,3,4, dtype=dtypes.float, device="ROCKCHIP")
    indices = (Tensor.empty(2,1,1,1,1,1, dtype=dtypes.int, device="ROCKCHIP"),
               Tensor.empty(1,3,1,1,1,1, dtype=dtypes.int, device="ROCKCHIP"),
               Tensor.empty(1,1,4,1,1,1, dtype=dtypes.int, device="ROCKCHIP"),
               Tensor.empty(2,1,1,5,1,1, dtype=dtypes.int, device="ROCKCHIP"),
               Tensor.empty(1,1,1,1,6,1, dtype=dtypes.int, device="ROCKCHIP"))
    sinks = [early_simplify(call.src[0]) for call in x[indices].schedule_linear().src if call.src[0].op is Ops.SINK]
    self.assertEqual(len(sinks), 3)
    for sink in sinks[:2]:
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    reduced = x[indices[0], ..., indices[-1]]
    reduced_sinks = [early_simplify(call.src[0]) for call in reduced.schedule_linear().src if call.src[0].op is Ops.SINK]
    self.assertEqual(len(reduced_sinks), 1)
    reduced_program = build_native_program(reduced_sinks[0])
    self.assertIsNotNone(reduced_program)
    self.assertEqual(reduced_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
    injected = x[indices[0], indices[1], None, indices[3], indices[4]]
    injected_sink = next(early_simplify(call.src[0]) for call in injected.schedule_linear().src if call.src[0].op is Ops.SINK)
    injected_program = build_native_program(injected_sink)
    self.assertIsNotNone(injected_program)
    self.assertEqual(injected_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
    x_half = Tensor.empty(2,5,6,5,3,4, dtype=dtypes.half, device="ROCKCHIP")
    for expression in (x_half[indices[0], ..., indices[-1]],
                       x_half[indices[0], indices[1], None, indices[3], indices[4]]):
      half_sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      half_program = build_native_program(half_sink)
      self.assertIsNotNone(half_program)
      self.assertEqual(half_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_scatter_uses_typed_update_selection(self):
    for shape in ((4,5,6), (3,4,5)):
      x = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP")
      src = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP")
      indices = Tensor.empty(3,4,5, dtype=dtypes.int, device="ROCKCHIP")
      program = build_native_program(_get_sink(x.scatter(dim=1, index=indices, src=src)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    x = Tensor.empty(4,5,6, dtype=dtypes.float, device="ROCKCHIP")
    indices = Tensor.empty(3,4,5, dtype=dtypes.int, device="ROCKCHIP")
    scalar_program = build_native_program(_get_sink(x.scatter(dim=1, index=indices, src=3)))
    self.assertIsNotNone(scalar_program)
    self.assertEqual(scalar_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    x_half = Tensor.empty(4,5,6, dtype=dtypes.half, device="ROCKCHIP")
    src_half = Tensor.empty(4,5,6, dtype=dtypes.half, device="ROCKCHIP")
    for expression in (x_half.scatter(dim=1, index=indices, src=src_half),
                       x_half.scatter(dim=1, index=indices, src=3)):
      half_program = build_native_program(_get_sink(expression))
      self.assertIsNotNone(half_program)
      self.assertEqual(half_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_scatter_scalar_reductions_use_typed_stages(self):
    x = Tensor.empty(4,5,6, dtype=dtypes.float, device="ROCKCHIP")
    indices = Tensor.empty(3,4,5, dtype=dtypes.int, device="ROCKCHIP")
    for mode, value in (("add", float("inf")), ("multiply", float("nan"))):
      program = build_native_program(_get_sink(x.scatter(dim=1, index=indices, src=value, reduce=mode)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 2)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      self.assertEqual(subtasks[1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    x_half = Tensor.empty(4,5,6, dtype=dtypes.half, device="ROCKCHIP")
    for mode, value in (("add", float("inf")), ("multiply", float("nan"))):
      half_program = build_native_program(_get_sink(x_half.scatter(dim=1, index=indices, src=value, reduce=mode)))
      self.assertIsNotNone(half_program)
      half_subtasks = half_program.src[1].src[0].arg
      self.assertEqual(len(half_subtasks), 2)
      self.assertEqual(half_subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      self.assertEqual(half_subtasks[1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_scatter_reduce_tensor_uses_bounded_typed_boundary(self):
    x = Tensor.empty(3,4,5, dtype=dtypes.float, device="ROCKCHIP")
    indices = Tensor.empty(3,4,5, dtype=dtypes.int, device="ROCKCHIP")
    src = Tensor.empty(3,4,5, dtype=dtypes.float, device="ROCKCHIP")
    for mode in ("sum", "prod", "mean", "amin", "amax"):
      for include_self in (True, False):
        program = build_native_program(_get_sink(x.scatter_reduce(-1, indices, src, mode, include_self=include_self)))
        self.assertIsNotNone(program)
        subtasks = program.src[1].src[0].arg
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    padded = Tensor.zeros(4,5,6, dtype=dtypes.float, device="ROCKCHIP").scatter_reduce(
      1, indices, Tensor.empty(4,5,6, dtype=dtypes.float, device="ROCKCHIP"), "prod")
    padded_sinks = [early_simplify(call.src[0]) for call in padded.schedule_linear().src if call.src[0].op is Ops.SINK]
    padded_program = build_native_program(padded_sinks[-1])
    self.assertIsNotNone(padded_program)
    self.assertEqual(padded_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    x_half = Tensor.empty(3,4,5, dtype=dtypes.half, device="ROCKCHIP")
    src_half = Tensor.empty(3,4,5, dtype=dtypes.half, device="ROCKCHIP")
    for mode in ("sum", "prod", "mean", "amin", "amax"):
      for include_self in (True, False):
        half_program = build_native_program(_get_sink(x_half.scatter_reduce(
          -1, indices, src_half, mode, include_self=include_self)))
        self.assertIsNotNone(half_program)
        self.assertEqual(half_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_biased_convolution_keeps_cmac_and_serializes_epilogue(self):
    x = Tensor.empty(1,8,5,5, dtype=dtypes.float, device="ROCKCHIP")
    w = Tensor.empty(8,8,1,1, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(8, dtype=dtypes.float, device="ROCKCHIP")
    expression = x.conv2d(w,b).relu().conv2d(w,b)
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    self.assertEqual(len(sinks), 2)
    for sink in sinks:
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertTrue(any(task.task.kind == "cmac" for task in subtasks))
      self.assertEqual(subtasks[-1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_strided_transposed_convolution_keeps_guarded_cmac(self):
    for stride in ((2,1), (1,2)):
      x = Tensor.empty(2,4,4,5, dtype=dtypes.float, device="ROCKCHIP")
      w = Tensor.empty(4,4,3,3, dtype=dtypes.float, device="ROCKCHIP")
      program = build_native_program(_get_sink(x.conv_transpose2d(w, stride=stride)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertTrue(any(task.task.kind == "cmac" for task in subtasks))

  def test_fp32_softmax_stages_use_strict_serialized_tasks(self):
    for shape, axis in (((45,65), 1), ((45,), 0)):
      expression = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP").softmax(axis)
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      programs = [build_native_program(sink) for sink in sinks]
      self.assertTrue(all(program is not None for program in programs))
      host_stages = [program.src[1].src[0].arg for sink, program in zip(sinks, programs)
                     if any(u.op in (Ops.EXP2, Ops.RECIPROCAL) for u in sink.toposort())]
      self.assertTrue(host_stages)
      self.assertTrue(all(len(subtasks) == 1 and subtasks[0].task.layout[1] == _HOST_ELEMENTWISE_LAYOUT
                          for subtasks in host_stages))

  def test_fp32_softmax_argmax_uses_linear_strict_task(self):
    for axis, compact_mapping in ((0, 1), (1, 0)):
      expression = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").softmax(axis).argmax()
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      program = build_native_program(sinks[-1])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_SOFTMAX_ARGMAX_LAYOUT)
      self.assertEqual(subtasks[0].task.layout[3], compact_mapping)

  def test_fp32_log_softmax_stages_use_strict_serialized_tasks(self):
    for shape, axis in (((45,65), 1), ((45,), 0)):
      expression = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP").log_softmax(axis)
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      programs = [build_native_program(sink) for sink in sinks]
      self.assertTrue(all(program is not None for program in programs))
      for sink, program in zip(sinks, programs):
        if any(u.op is Ops.LOG2 for u in sink.toposort()):
          subtasks = program.src[1].src[0].arg
          self.assertEqual(len(subtasks), 1)
          self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_normalize_denominators_use_strict_serialized_tasks(self):
    for p in (2, 1, 3, 0, -1):
      expression = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").normalize(p=p)
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src
                  if call.src[0].op is Ops.SINK and any(u.op is Ops.REDUCE for u in call.src[0].toposort()))
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_logsumexp_uses_strict_serialized_reduction(self):
    for shape, axis in (((45,65), 0), ((45,), 0)):
      expression = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP").logsumexp(axis)
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      program = build_native_program(sinks[-1])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_logcumsumexp_stages_use_strict_serialized_reductions(self):
    expression = Tensor.empty(6,6,6, dtype=dtypes.float, device="ROCKCHIP").logcumsumexp(2)
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    programs = [build_native_program(sink) for sink in sinks]
    self.assertEqual(len(programs), 2)
    for program in programs:
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_log_and_logaddexp_use_strict_serialized_tasks(self):
    lhs = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    rhs = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    scalar = Tensor.empty(1, dtype=dtypes.float, device="ROCKCHIP")
    vector = Tensor.empty(3, dtype=dtypes.float, device="ROCKCHIP")
    for expression in (lhs.log(), lhs.logaddexp(rhs), scalar.logaddexp(vector)):
      program = build_native_program(_get_sink(expression))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_sinh_cosh_avoid_generic_dpu_splitter(self):
    for expression in (Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").sinh(),
                       Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").cosh()):
      program = build_native_program(_get_sink(expression))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_tanh_avoids_half_two_lut_path(self):
    expression = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").tanh()
    program = build_native_program(_get_sink(expression))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp32_atanh_avoids_half_two_lut_path(self):
    expression = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").atanh()
    program = build_native_program(_get_sink(expression))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_small_fp32_gemm_uses_typed_cmac_boundary(self):
    a = Tensor.empty(9,9, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(9,9, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a@b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "cmac" for task in subtasks), 3)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_batched_fp32_gemm_serializes_shared_axis(self):
    a = Tensor.empty(8,45,65, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(8,65,100, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a@b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertEqual(len(cmac_tasks), 3*8)
    self.assertTrue(all(task.layout[:3] == (45,100,65) for task in cmac_tasks))
    self.assertEqual(sum(task.fp32_output for task in cmac_tasks), 8)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_large_shared_fp32_gemm_serializes_before_k_limit(self):
    query = Tensor.empty(256,16,64, dtype=dtypes.float, device="ROCKCHIP")
    key = Tensor.empty(256,16,64, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink((query @ key.transpose(-2,-1)) * 0.25))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertEqual(len(cmac_tasks), 3*256)
    self.assertTrue(all(task.layout[:3] == (16,16,64) for task in cmac_tasks))
    self.assertEqual(sum(task.fp32_output for task in cmac_tasks), 256)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_attention_score_gemm_uses_bounded_typed_reduction(self):
    query = Tensor.empty(256,16,64, dtype=dtypes.float, device="ROCKCHIP")
    key = Tensor.empty(256,16,64, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink((query @ key.transpose(-2,-1)) * 0.125))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 2)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
    self.assertEqual(subtasks[1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    causal_query = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    causal_key = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    causal_value = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    causal_sink = next(early_simplify(call.src[0]) for call in
                       causal_query.scaled_dot_product_attention(causal_key, causal_value, is_causal=True).schedule_linear().src
                       if call.src[0].op is Ops.SINK)
    causal_program = build_native_program(causal_sink)
    self.assertIsNotNone(causal_program)
    self.assertEqual([task.task.layout[1] for task in causal_program.src[1].src[0].arg],
                     [_HOST_ELEMENTWISE_REDUCE_LAYOUT, _HOST_ELEMENTWISE_LAYOUT])

  def test_attention_value_gemm_uses_bounded_typed_reduction(self):
    query = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    key = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    value = Tensor.empty(32,8,16,64, dtype=dtypes.float, device="ROCKCHIP")
    sinks = [early_simplify(call.src[0]) for call in query.scaled_dot_product_attention(key, value).schedule_linear().src
             if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 2)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
    self.assertEqual(subtasks[1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    gqa_query = Tensor.empty(32,32,16,64, dtype=dtypes.float, device="ROCKCHIP")
    gqa_sinks = [early_simplify(call.src[0]) for call in
                 gqa_query.scaled_dot_product_attention(key, value, enable_gqa=True).schedule_linear().src
                 if call.src[0].op is Ops.SINK]
    for sink in (gqa_sinks[0], gqa_sinks[-1]):
      gqa_program = build_native_program(sink)
      self.assertIsNotNone(gqa_program)
      self.assertEqual([task.task.layout[1] for task in gqa_program.src[1].src[0].arg],
                       [_HOST_ELEMENTWISE_REDUCE_LAYOUT, _HOST_ELEMENTWISE_LAYOUT])

  def test_half_attention_uses_typed_score_and_softmax_reductions(self):
    query = Tensor.empty(32,8,16,64, dtype=dtypes.half, device="ROCKCHIP")
    key = Tensor.empty(32,8,16,64, dtype=dtypes.half, device="ROCKCHIP")
    value = Tensor.empty(32,8,16,64, dtype=dtypes.half, device="ROCKCHIP")
    sinks = [early_simplify(call.src[0]) for call in query.scaled_dot_product_attention(key, value).schedule_linear().src
             if call.src[0].op is Ops.SINK]
    for sink_index, expected_count in ((0, 2), (1, 1), (3, 1)):
      program = build_native_program(sinks[sink_index])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), expected_count)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      if expected_count == 2: self.assertEqual(subtasks[1].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    denominator_program = build_native_program(sinks[2])
    self.assertIsNotNone(denominator_program)
    self.assertEqual(denominator_program.src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_bce_uses_one_bounded_typed_task(self):
    source = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    target = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP").clip(0, 1)
    expressions = (source.sigmoid().binary_crossentropy(target),
                   source.binary_crossentropy_logits(target),
                   source.sigmoid().binary_crossentropy(target, reduction="none"),
                   source.binary_crossentropy_logits(target, reduction="none"),
                   source.binary_crossentropy_logits(target, pos_weight=Tensor(
                     [.25,.5,.75,1,2,3,4,5,6,7], dtype=dtypes.half, device="ROCKCHIP")))
    for expression in expressions:
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_BCE_LAYOUT)

  def test_probability_cross_entropy_uses_exact_bounded_task(self):
    def half_bits(value:float) -> int: return struct.unpack("<H", struct.pack("<e", value))[0]
    source_2d = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    target_2d = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    source_4d = Tensor.empty(32,4,4,4, dtype=dtypes.half, device="ROCKCHIP")
    target_4d = Tensor.empty(32,4,4,4, dtype=dtypes.half, device="ROCKCHIP")
    expressions = (
      (source_2d.cross_entropy(target_2d), (1, _HOST_CROSS_ENTROPY_LAYOUT, 2, 32, 10, 1, half_bits(1), half_bits(0))),
      (source_2d.cross_entropy(target_2d, reduction="none"),
       (32, _HOST_CROSS_ENTROPY_LAYOUT, 0, 32, 10, 1, half_bits(1), half_bits(0))),
      (source_2d.cross_entropy(target_2d, label_smoothing=.3),
       (1, _HOST_CROSS_ENTROPY_LAYOUT, 2, 32, 10, 1, half_bits(.7), half_bits(.03))),
      (source_2d.cross_entropy(target_2d, label_smoothing=1),
       (1, _HOST_CROSS_ENTROPY_LAYOUT, 2, 32, 10, 1, half_bits(0), half_bits(.1))),
      (source_4d.cross_entropy(target_4d),
       (1, _HOST_CROSS_ENTROPY_LAYOUT, 2, 512, 4, 16, half_bits(1), half_bits(0))))
    for expression, expected_layout in expressions:
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      program = build_native_program(sinks[-1])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout, expected_layout)

  def test_sparse_cross_entropy_combined_args_use_bounded_task(self):
    source = Tensor.empty(12,10, dtype=dtypes.half, device="ROCKCHIP")
    target = Tensor.empty(12, dtype=dtypes.int, device="ROCKCHIP")
    expression = source.sparse_categorical_crossentropy(target, reduction="mean", ignore_index=3, label_smoothing=.3)
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    self.assertEqual(len(subtasks[0].relocs), 5)

  def test_nll_uses_exact_bounded_task(self):
    source_2d = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    target_2d = Tensor.empty(32, dtype=dtypes.int, device="ROCKCHIP")
    source_3d = Tensor.empty(2,10,3,3,3, dtype=dtypes.half, device="ROCKCHIP")
    target_3d = Tensor.empty(2,3,3,3, dtype=dtypes.int, device="ROCKCHIP")
    weight = Tensor.empty(10, dtype=dtypes.half, device="ROCKCHIP")
    expressions = (
      (source_2d.log_softmax(axis=1).nll_loss(target_2d),
       (1, _HOST_NLL_LAYOUT, 2, 32, 10, 1, 0, 0, 0)),
      (source_2d.log_softmax(axis=1).nll_loss(target_2d, reduction="none"),
       (32, _HOST_NLL_LAYOUT, 0, 32, 10, 1, 0, 0, 0)),
      (source_2d.log_softmax(axis=1).nll_loss(target_2d, weight, reduction="sum"),
       (1, _HOST_NLL_LAYOUT, 1, 32, 10, 1, 0, 0, 1)),
      (source_2d.log_softmax(axis=1).nll_loss(target_2d, ignore_index=3),
       (1, _HOST_NLL_LAYOUT, 2, 32, 10, 1, 1, 3, 0)),
      (source_3d.log_softmax(axis=1).nll_loss(target_3d),
       (1, _HOST_NLL_LAYOUT, 2, 54, 10, 27, 0, 0, 0)))
    for expression, expected_layout in expressions:
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      program = build_native_program(sinks[-1])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout, expected_layout)

  def test_masked_select_prefix_count_uses_bounded_typed_reduction(self):
    source = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    prefix = (source > 0.5).flatten().cumsum()
    bool_prefix = Tensor.empty(9, dtype=dtypes.bool, device="ROCKCHIP").cumsum()
    indices = Tensor.empty(320, dtype=dtypes.int, device="ROCKCHIP")
    histogram = Tensor.zeros(118, dtype=dtypes.int, device="ROCKCHIP", buffer=False).scatter(0, indices, 1, reduce="add")
    for expression in (prefix, bool_prefix, histogram, histogram.cumsum()):
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      sink = sinks[-1]
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_nonzero_expanded_prefix_uses_bounded_typed_reduction(self):
    source = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    mask = source > 0.5
    expression = mask.unsqueeze(-1).expand(*mask.shape, mask.ndim).flatten().cumsum()[-1]
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    programs = [build_native_program(sink) for sink in sinks]
    self.assertTrue(all(program is not None for program in programs))
    layouts = [program.src[1].src[0].arg[0].task.layout[1] for program in programs]
    self.assertEqual(layouts[:2], [_HOST_ELEMENTWISE_REDUCE_LAYOUT, _HOST_ELEMENTWISE_REDUCE_LAYOUT])

  def test_constant_true_masked_select_uses_typed_copy(self):
    source = Tensor.empty(32,10, dtype=dtypes.half, device="ROCKCHIP")
    expression = source.masked_select(Tensor(True, device="ROCKCHIP"))
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fixed_masked_select_uses_bounded_typed_reduction(self):
    source = Tensor.empty(9, dtype=dtypes.int, device="ROCKCHIP")
    mask = Tensor.empty(9, dtype=dtypes.bool, device="ROCKCHIP")
    expression = source.masked_select(mask, size=4)
    sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
    program = build_native_program(sinks[-1])
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_fixed_nonzero_uses_bounded_typed_reduction(self):
    expressions = (Tensor.empty(5, dtype=dtypes.int, device="ROCKCHIP").nonzero(size=3),
                   Tensor.empty(2,2, dtype=dtypes.int, device="ROCKCHIP").nonzero(size=2))
    for expression in expressions:
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      programs = [build_native_program(sink) for sink in sinks]
      self.assertTrue(all(program is not None for program in programs))
      self.assertEqual(programs[-1].src[1].src[0].arg[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_fp32_bitcast_uses_typed_reinterpret(self):
    expression = Tensor.empty(3,3, dtype=dtypes.float, device="CPU").to("ROCKCHIP").bitcast(dtypes.int)
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_uint8_min_uses_typed_reduction(self):
    expression = Tensor.empty(2,3, dtype=dtypes.int, device="ROCKCHIP").cast(dtypes.uint8).min()
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_fp32_cos_uses_typed_host_math(self):
    expression = Tensor.empty(4, dtype=dtypes.float, device="ROCKCHIP").cos()
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp16_floor_ceil_use_typed_host_where(self):
    for expression in (Tensor.empty(4, dtype=dtypes.half, device="ROCKCHIP").floor(),
                       Tensor.empty(4, dtype=dtypes.half, device="ROCKCHIP").ceil()):
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_round_quantization_uses_one_typed_host_task(self):
    source = Tensor.empty(6, dtype=dtypes.half, device="ROCKCHIP")
    expression = source + 0.125*(source.round()-source)
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_mod_uses_one_typed_host_task(self):
    expressions = []
    for lhs_dtype in (dtypes.half, dtypes.int):
      for rhs_dtype in (dtypes.half, dtypes.int):
        lhs = Tensor.empty(7, dtype=lhs_dtype, device="ROCKCHIP")
        rhs = Tensor.empty(7, dtype=rhs_dtype, device="ROCKCHIP")
        expressions.append(lhs % rhs)
    for dtype in (dtypes.half, dtypes.int):
      source = Tensor.empty(7, dtype=dtype, device="ROCKCHIP")
      expressions.extend((source % 2, source % 3, 100 % source))
      if dtype is dtypes.half: expressions.extend((source % 3.5, 100.5 % source))
    for expression in expressions:
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fmod_uses_one_typed_host_task(self):
    expressions = []
    for lhs_dtype in (dtypes.half, dtypes.int):
      for rhs_dtype in (dtypes.half, dtypes.int):
        lhs = Tensor.empty(7, dtype=lhs_dtype, device="ROCKCHIP")
        rhs = Tensor.empty(7, dtype=rhs_dtype, device="ROCKCHIP")
        expressions.append(lhs.fmod(rhs))
    for dtype in (dtypes.half, dtypes.int):
      source = Tensor.empty(7, dtype=dtype, device="ROCKCHIP")
      expressions.append(source.fmod(2))
      if dtype is dtypes.half: expressions.append(source.fmod(3.5))
    for expression in expressions:
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_int_true_div_uses_one_typed_fp16_task(self):
    default_float = dtypes.default_float
    try:
      dtypes.default_float = dtypes.half
      cases = [(dtypes.int, dtypes.int, 7, None), (dtypes.int, dtypes.int, 7, "floor"),
               (dtypes.int, dtypes.int, 7, "trunc"), (dtypes.int, dtypes.int, 1, None),
               (dtypes.int, dtypes.int, 1, "floor"), (dtypes.int, dtypes.int, 1, "trunc")]
      cases += [(lhs_dtype, rhs_dtype, 1, mode)
                for lhs_dtype, rhs_dtype in ((dtypes.half, dtypes.int), (dtypes.int, dtypes.half), (dtypes.half, dtypes.half))
                for mode in (None, "trunc", "floor")]
      expressions = [Tensor.empty(7, dtype=lhs_dtype, device="ROCKCHIP").div(
        Tensor.empty(rhs_size, dtype=rhs_dtype, device="ROCKCHIP"), rounding_mode=mode)
                     for lhs_dtype, rhs_dtype, rhs_size, mode in cases]
      expressions.append(Tensor.empty(7, dtype=dtypes.int, device="ROCKCHIP")//2)
      sinks = [next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
               for expression in expressions]
    finally:
      dtypes.default_float = default_float
    for sink in sinks:
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_fp16_cumprod_uses_one_typed_float32_reduction(self):
    for shape, axis in (((20,), 0), ((20,30), 0), ((20,30), 1), ((20,30,40), 2)):
      expression = Tensor.empty(*shape, dtype=dtypes.half, device="ROCKCHIP").cumprod(axis)
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      self.assertEqual(subtasks[0].task.layout[5], 1)
    long_sinks = [early_simplify(call.src[0]) for call in
                  Tensor.empty(1022, dtype=dtypes.half, device="ROCKCHIP").cumprod(0).schedule_linear().src
                  if call.src[0].op is Ops.SINK]
    long_programs = [build_native_program(sink) for sink in long_sinks]
    self.assertTrue(all(program is not None for program in long_programs))
    for program in long_programs[:2]:
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)
      self.assertEqual(subtasks[0].task.layout[5], 1)

  def test_large_ellipsis_einsum_uses_one_typed_float32_reduction(self):
    lhs = Tensor.empty(32, 7, 24, 24, 24, dtype=dtypes.half, device="ROCKCHIP")
    rhs = Tensor.empty(32, 7, 24, 24, 24, dtype=dtypes.half, device="ROCKCHIP")
    expression = Tensor.einsum("ij...,ij...->ij", lhs, rhs)
    sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout, (224, _HOST_EINSUM_LAYOUT, 13824))

  def test_int_power_uses_one_typed_host_task(self):
    for exponent in (2, 7, 29):
      expression = Tensor.empty(3, dtype=dtypes.int, device="ROCKCHIP")**exponent
      sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
      program = build_native_program(sink)
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_bilinear_interpolate_uses_two_typed_host_stages(self):
    for input_size, output_size in (((12,20),(9,31)), ((12,9),(31,20)), ((9,31),(20,12))):
      for align_corners in (False, True):
        expression = Tensor.empty(2,3,*input_size, dtype=dtypes.half, device="ROCKCHIP").interpolate(
          size=output_size, mode="linear", align_corners=align_corners)
        sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
        programs = [build_native_program(sink) for sink in sinks]
        self.assertEqual(len(programs), 2)
        for program in programs:
          self.assertIsNotNone(program)
          subtasks = program.src[1].src[0].arg
          self.assertEqual(len(subtasks), 1)
          self.assertEqual(subtasks[0].task.layout[1], _HOST_BILINEAR_LAYOUT)

  def test_linear_interpolate_uses_one_typed_host_stage(self):
    for input_size, output_size in ((52,29), (29,52)):
      for align_corners in (False, True):
        expression = Tensor.empty(2,3,input_size, dtype=dtypes.half, device="ROCKCHIP").interpolate(
          size=(output_size,), mode="linear", align_corners=align_corners)
        sink = next(early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK)
        program = build_native_program(sink)
        self.assertIsNotNone(program)
        subtasks = program.src[1].src[0].arg
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].task.layout[1], _HOST_BILINEAR_LAYOUT)

  def test_fp16_axis_arg_extrema_use_typed_coordinate_reduction(self):
    source = Tensor.empty(10,20, dtype=dtypes.half, device="ROCKCHIP")
    for expression in (source.argmax(0, False), source.argmin(0, False)):
      sinks = [early_simplify(call.src[0]) for call in expression.schedule_linear().src if call.src[0].op is Ops.SINK]
      program = build_native_program(sinks[-1])
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_REDUCE_LAYOUT)

  def test_fp32_factorized_zero_stride_sum_stays_native(self):
    a = Tensor.empty(2,4,1, dtype=dtypes.float, device="ROCKCHIP").expand(2,4,3)
    b = Tensor.empty(1,4,1, dtype=dtypes.float, device="ROCKCHIP").expand(2,4,3)
    program = build_native_program(_get_sink((a*b).sum((0,2))))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertTrue(any(task.task.kind == "cmac" for task in subtasks))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_padded_fp32_gemm_materializes_where_sources(self):
    a = Tensor.empty(9,9, dtype=dtypes.float, device="ROCKCHIP").pad(((0,7),(0,7)))
    b = Tensor.empty(9,9, dtype=dtypes.float, device="ROCKCHIP").pad(((0,7),(0,7)))
    program = build_native_program(_get_sink(a@b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertEqual(len(cmac_tasks), 3)
    self.assertTrue(all(task.layout[:3] == (16,16,16) for task in cmac_tasks))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_explicit_half_gemm_tags_fused_fp32_inputs(self):
    a = Tensor.empty(64,64, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(64,64, dtype=dtypes.float, device="ROCKCHIP")
    plan = plan_rk(_get_sink(a.half() @ b.half()))
    self.assertIsInstance(plan, RKPlan)
    self.assertEqual(plan.kind, "cmac")
    self.assertEqual(plan.fp32_inputs, (1,2))
    self.assertFalse(plan.fp32_output)

  def test_large_fp32_contraction_uses_tiled_raw_cmac_boundary(self):
    a = Tensor.empty(3,5,8,10, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(11,7,5,13,8, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(Tensor.einsum("pqrs,tuqvr->pstuv", a, b)))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertEqual(len(cmac_tasks), 3)
    self.assertTrue(cmac_tasks[0].fp32_output)
    self.assertEqual(cmac_tasks[0].layout[0] * cmac_tasks[0].layout[1], 30030)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_long_fp32_batched_dot_uses_proven_cmac_k_chunks(self):
    a = Tensor.empty(3,13824, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(3,13824, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(Tensor.einsum("ij,ij->i", a, b)))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertGreater(len(subtasks), 128)
    self.assertTrue(cmac_tasks)
    self.assertLessEqual(max(task.layout[2] for task in cmac_tasks), 416)
    self.assertTrue(any(task.task.is_copy and len(task.task.layout) > 2 and
                        task.task.layout[1] == _HOST_FP32_HALF_LAYOUT for task in subtasks))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_long_fp32_sum_uses_two_level_raw_cmac(self):
    a = Tensor.empty(16384, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a.sum()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    cmac_tasks = [task.task for task in subtasks if task.task.kind == "cmac"]
    self.assertTrue(cmac_tasks)
    self.assertLessEqual(max(task.layout[2] for task in cmac_tasks), 4096)
    self.assertTrue(any(task.fp32_output for task in cmac_tasks))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_small_axis_fp32_sum_uses_both_input_limbs(self):
    for shape, axis in (((4,2,2), (0,2)), ((3,4,5,6), 3)):
      a = Tensor.empty(*shape, dtype=dtypes.float, device="ROCKCHIP")
      program = build_native_program(_get_sink(a.sum(axis)))
      self.assertIsNotNone(program)
      subtasks = program.src[1].src[0].arg
      self.assertEqual(sum(task.task.kind == "cmac" for task in subtasks), 2)
      copy_layouts = [task.task.layout[1] for task in subtasks if task.task.is_copy]
      self.assertIn(_HOST_FP32_HALF_LAYOUT, copy_layouts)
      self.assertIn(_HOST_FP32_RESIDUAL_LAYOUT, copy_layouts)

  def test_fp32_relu_sum_preserves_both_input_limbs(self):
    a = Tensor.empty(3,4,5, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a.relu().sum().relu()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    copy_layouts = [task.task.layout[1] for task in subtasks if task.task.is_copy]
    self.assertIn(_HOST_FP32_HALF_LAYOUT, copy_layouts)
    self.assertIn(_HOST_FP32_RESIDUAL_LAYOUT, copy_layouts)
    self.assertEqual(sum(task.task.kind == "cmac" for task in subtasks), 2)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_fp32_add_uses_compensated_boundary(self):
    a = Tensor.empty(45,68, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(45,68, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a+b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 9)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 5)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_add3_keeps_compensated_boundary(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    c = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a+b+c))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 18)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 7)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_broadcast_add_uses_affine_limb_views(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(45,1, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a+b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 9)
    self.assertTrue(all(len(task.task.layout) > 2 for task in subtasks[:4]))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_scalar_add_uses_compensated_constant_limbs(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a+0.1251))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 9)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 3)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_sub_uses_signed_compensated_limbs(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a-b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 11)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 5)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_scalar_rsub_negates_limbs_on_npu(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(2-a))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 11)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_mul_uses_compensated_boundary(self):
    a = Tensor.empty(45,68, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(45,68, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a*b))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 25)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 5)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_scalar_mul_uses_compensated_constant_limbs(self):
    a = Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a*0.1251))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(task.task.kind == "dpu" and not task.task.is_copy for task in subtasks), 25)
    self.assertEqual(sum(task.task.is_copy for task in subtasks), 3)
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

  def test_fp32_all_uses_both_nonzero_limbs(self):
    a = Tensor.empty(3,4,5,6, dtype=dtypes.float, device="ROCKCHIP")
    program = build_native_program(_get_sink(a.all()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    layouts = [task.task.layout[1] for task in subtasks if task.task.is_copy and len(task.task.layout) == 2]
    self.assertIn(_HOST_FP32_HALF_LAYOUT, layouts)
    self.assertIn(_HOST_FP32_RESIDUAL_LAYOUT, layouts)
    self.assertEqual(sum(task.task.kind == "cmac" for task in subtasks), 1)

  def test_large_fp32_fill_uses_tiled_npu_boundary(self):
    program = build_native_program(_get_sink(Tensor.ones(2**19, dtype=dtypes.float, device="ROCKCHIP")))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertEqual(sum(not task.task.is_copy for task in subtasks), 2)
    self.assertEqual(sum(task.task.is_copy and task.task.layout[1] == _HOST_HALF_FP32_LAYOUT for task in subtasks), 2)

  def test_fp32_acos_uses_specialized_lut_boundary(self):
    program = build_native_program(_get_sink(Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").acos()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertLess(len(subtasks), 80)
    self.assertTrue(any(task.task.kind == "dpu_lut" for task in subtasks))
    self.assertEqual(sum(task.task.fp32_output for task in subtasks), 1)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_fp32_acosh_uses_specialized_lut_boundary(self):
    program = build_native_program(_get_sink(Tensor.empty(45,65, dtype=dtypes.float, device="ROCKCHIP").acosh()))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertLess(len(subtasks), 80)
    self.assertEqual(sum(task.task.kind == "dpu_lut" for task in subtasks), 2)
    self.assertEqual(sum(task.task.fp32_output for task in subtasks), 1)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_reject_cmac_exceeds_cbuf(self):
    # M=6000 with K=4: align_in=32, input_row_bytes=64, data_banks=ceil(6000*64/32768)=12 > 11
    a = Tensor.rand(6000,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    sink = _get_sink(a@b)
    with self.assertRaises(RuntimeError) as cm:
      build_native_program(sink)
    self.assertIn("RKPLAN_REJECT", str(cm.exception))

class TestEmitter(unittest.TestCase):
  def test_dpu_emits_commands(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a+b))
    self.assertGreater(len(cmds), 0)
    self.assertEqual(len(relocs), 3)
    self.assertEqual(task.enable_mask, 0x18)

  def test_dpu_inplace_add_emits(self):
    # a.assign(a+b) — in-place ADD where output slot == input slot A
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    c = a.assign(a+b)
    cmds, task, relocs = _emit(_get_sink(c))
    self.assertGreater(len(cmds), 0)
    self.assertEqual(len(relocs), 3)
    # relocs[0]=out, relocs[1]=a, relocs[2]=b; out==a for in-place
    self.assertEqual(relocs[0].globals_slot, relocs[1].globals_slot)

  def test_cmac_emits_commands(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a@b))
    self.assertGreater(len(cmds), 0)
    self.assertEqual(len(relocs), 3)
    self.assertEqual(task.enable_mask, 0xd)

  def test_cmac_same_buffer_emits(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a@a))
    self.assertGreater(len(cmds), 0)
    self.assertEqual(len(relocs), 3)
    self.assertEqual(relocs[0].globals_slot, relocs[1].globals_slot)
    self.assertEqual(task.enable_mask, 0xd)

  def test_ppu_emits_commands(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a.max(axis=0)))
    self.assertGreater(len(cmds), 0)
    self.assertEqual(len(relocs), 2)
    self.assertEqual(task.enable_mask, 0x60)

  def test_relocs_reference_valid_cmds(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    sink = _get_sink(a+b)
    pi = ProgramInfo.from_sink(sink)
    cmds, task, relocs = _emit(sink)
    for r in relocs:
      self.assertLess(r.word_index, len(cmds))
      self.assertIn(r.globals_slot, pi.globals)

class TestCodec(unittest.TestCase):
  def test_roundtrip_dpu(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a+b))
    packed = encode_rk(cmds, task, relocs)
    dec_cmds, dec_task, dec_relocs = decode_rk(packed)
    self.assertEqual(len(dec_cmds), len(cmds))
    self.assertEqual(len(dec_relocs), len(relocs))
    self.assertEqual(dec_task.enable_mask, task.enable_mask)
    self.assertEqual(dec_task.int_mask, task.int_mask)
    for c1, c2 in zip(cmds, dec_cmds):
      self.assertEqual(c1, c2)

  def test_roundtrip_dpu_fill(self):
    # Fill roundtrip: verify is_fill survives codec
    cmds, task, relocs = _emit(_get_sink(Tensor.zeros(4,4,dtype=dtypes.half)))
    self.assertTrue(task.is_fill)
    packed = encode_rk(cmds, task, relocs)
    dec_cmds, dec_task, dec_relocs = decode_rk(packed)
    self.assertTrue(dec_task.is_fill)
    self.assertEqual(len(dec_cmds), len(cmds))

  def test_roundtrip_multi_fp32_residual_input(self):
    cmds, task, relocs = _emit(_get_sink(Tensor.rand(4,4,dtype=dtypes.half).realize()+1))
    encoded = encode_rk_multi((RKSubTask(cmds, replace(task, fp32_residual_input=True), relocs),))
    decoded = decode_rk_multi(encoded)
    self.assertEqual(len(decoded), 1)
    self.assertTrue(decoded[0].task.fp32_residual_input)

  def test_roundtrip_cmac_scaled_sum(self):
    # Scaled sum roundtrip: verify const_val survives codec
    cmds, task, relocs = _emit(_get_sink((Tensor.rand(4,4,dtype=dtypes.half).realize()*2).sum()))
    self.assertAlmostEqual(task.const_val, 2.0)
    packed = encode_rk(cmds, task, relocs)
    dec_cmds, dec_task, dec_relocs = decode_rk(packed)
    self.assertAlmostEqual(dec_task.const_val, 2.0)

  def test_determinism(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    cmds, task, relocs = _emit(_get_sink(a+b))
    self.assertEqual(encode_rk(cmds, task, relocs), encode_rk(cmds, task, relocs))

  def test_determinism_across_compiles(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    def compile_once():
      sink = _get_sink(a+b)
      prg = build_native_program(sink)
      r = RockchipRenderer(Target())
      final = graph_rewrite(prg, pm_to_program, ctx=r, name='linearize/render')
      for s in final.src:
        if s.op == Ops.BINARY: return s.arg
      return None
    self.assertEqual(compile_once(), compile_once())

  def test_no_pickle_in_binary(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    sink = _get_sink(a+b)
    prg = build_native_program(sink)
    r = RockchipRenderer(Target())
    final = graph_rewrite(prg, pm_to_program, ctx=r, name='linearize/render')
    for s in final.src:
      if s.op == Ops.BINARY:
        binary = s.arg
        self.assertNotEqual(binary[0], 0x80)
        magic = struct.unpack_from("<I", binary, 0)[0]
        self.assertEqual(magic, 0x524b494d)
        break

  def test_decode_bad_magic(self):
    header = struct.pack("<IIIIIIIIi", 0xDEAD, 2, 0, 0, 0, 0, 0, 0, 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header)
    self.assertIn("bad magic", str(cm.exception))

  def test_decode_truncated_header(self):
    with self.assertRaises(RuntimeError) as cm: decode_rk(b'\x00' * 10)
    self.assertIn("truncated header", str(cm.exception))

  def test_decode_truncated_commands(self):
    header = struct.pack("<IIIIIIIIi", 0x524b494d, 2, 100, 0, 0, 0, 0, 0, 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header)
    self.assertIn("truncated commands", str(cm.exception))

  def test_decode_out_of_range_reloc(self):
    header = struct.pack("<IIIIIIIIi", 0x524b494d, 2, 1, 1, 0, 0, 0, 0, 0)
    cmd = struct.pack("<Q", 0)
    reloc = struct.pack("<IIIIII", 99, 0, 0, 0, 0, 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header + cmd + reloc)
    self.assertIn("out of range", str(cm.exception))

  def test_decode_bad_version(self):
    header = struct.pack("<IIIIIIIIi", 0x524b494d, 99, 0, 0, 0, 0, 0, 0, 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header)
    self.assertIn("version", str(cm.exception))

  def test_decode_invalid_kind(self):
    header = struct.pack("<IIIIIIIIi", 0x524b494d, 2, 0, 0, 0, 0, 0, (99 << 24), 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header)
    self.assertIn("kind", str(cm.exception))

  def test_decode_truncated_layout(self):
    header = struct.pack("<IIIIIIIIi", 0x524b494d, 2, 0, 0, 0, 0, 0, (0 << 24) | 5, 0)
    with self.assertRaises(RuntimeError) as cm: decode_rk(header)
    self.assertIn("truncated layout", str(cm.exception))

class TestPipeline(unittest.TestCase):
  def _compile(self, expr):
    sink = _get_sink(expr)
    try: prg = build_native_program(sink)
    except RuntimeError as e:
      if "RKPLAN_REJECT" in str(e): return None
      raise
    r = RockchipRenderer(Target())
    final = graph_rewrite(prg, pm_to_program, ctx=r, name='linearize/render')
    for s in final.src:
      if s.op == Ops.BINARY: return s.arg
    return None

  def test_add_produces_binary(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    binary = self._compile(a+b)
    self.assertIsNotNone(binary)
    self.assertGreater(len(binary), 24)

  def test_matmul_produces_binary(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    binary = self._compile(a@b)
    self.assertIsNotNone(binary)
    self.assertGreater(len(binary), 24)

  def test_multifactor_einsum_produces_two_cmac_stages(self):
    a = Tensor.rand(2,3,dtype=dtypes.half).realize()
    b = Tensor.rand(5,3,7,dtype=dtypes.half).realize()
    c = Tensor.rand(2,7,dtype=dtypes.half).realize()
    prg = build_native_program(_get_sink(Tensor.einsum("ik,jkl,il->ij", a, b, c)))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 2)
    self.assertTrue(all(st.task.kind == "cmac" for st in subtasks))

  def test_fp32_multifactor_einsum_produces_compensated_cmac_stages(self):
    a = Tensor.empty(2,3,dtype=dtypes.float,device="ROCKCHIP")
    b = Tensor.empty(5,3,7,dtype=dtypes.float,device="ROCKCHIP")
    c = Tensor.empty(2,7,dtype=dtypes.float,device="ROCKCHIP")
    prg = build_native_program(_get_sink(Tensor.einsum("ik,jkl,il->ij", a, b, c)))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(sum(st.task.kind == "cmac" for st in subtasks), 6)
    self.assertEqual(sum(st.task.layout[1] == _HOST_FP32_COMBINE_LAYOUT for st in subtasks if st.task.is_copy), 2)
    self.assertTrue(subtasks[-1].task.fp32_output)

  def test_avg_pool_variable_divisor_serializes_counts(self):
    x = Tensor.rand(1,1,6,6,dtype=dtypes.half).realize()
    prg = build_native_program(_get_sink(x.avg_pool2d(kernel_size=(3,3), padding=1, count_include_pad=False)))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.kind, "cmac")
    layout = subtasks[0].task.layout
    n_counts = layout[11]
    self.assertEqual(n_counts, 4)
    self.assertEqual(set(layout[12:12+n_counts]), {4, 6, 9})

  def test_fp32_avg_pool_uses_typed_reduction_boundary(self):
    x = Tensor.empty(1,1,8,8, dtype=dtypes.float, device="ROCKCHIP")
    for output in (x.avg_pool2d(kernel_size=(3,2)),
                   x.avg_pool2d(kernel_size=(1,2), padding=(0,1), stride=(5,1)),
                   x.avg_pool2d(kernel_size=(3,3), padding=1, count_include_pad=False),
                   Tensor.empty(1,1,8,8,8, dtype=dtypes.float, device="ROCKCHIP").avg_pool2d(
                     kernel_size=(4,4,4), stride=3, padding=1, count_include_pad=False)):
      prg = build_native_program(_get_sink(output))
      self.assertIsNotNone(prg)
      subtasks = prg.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_AVG_POOL_LAYOUT)

  def test_local_max_pool_gathers_then_reduces_on_dpu(self):
    x = Tensor.rand(1,1,4,5,dtype=dtypes.half).realize()
    prg = build_native_program(_get_sink(x.max_pool2d(kernel_size=(2,2))))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 7)
    self.assertEqual(sum(st.task.is_copy for st in subtasks), 4)
    self.assertTrue(all(st.task.kind == "dpu" for st in subtasks))
    self.assertEqual(subtasks[-1].task.out_slot, 0)

  def test_int32_padded_max_pool_uses_bounded_static_reduction(self):
    x = Tensor.empty(4,2,11,28, dtype=dtypes.float, device="ROCKCHIP")
    prg = build_native_program(_get_sink(x.int().max_pool2d(kernel_size=(2,2), padding=1)))
    self.assertIsNotNone(prg)
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_max_pool_indices_encode_spatial_not_window_offsets(self):
    x = Tensor.empty(2,3,6,6, dtype=dtypes.float, device="ROCKCHIP")
    _, indices = x.max_pool2d(kernel_size=(2,2), return_indices=True)
    sinks = [early_simplify(call.src[0]) for call in indices.schedule_linear().src if call.src[0].op is Ops.SINK]
    prg = build_native_program(sinks[-1])
    self.assertIsNotNone(prg)
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ARGMAX_LAYOUT)
    self.assertEqual(subtasks[0].task.layout[4], 4)
    self.assertIn(7, subtasks[0].task.layout[5:])

    large = Tensor.empty(8,3,50,50, dtype=dtypes.half, device="ROCKCHIP")
    _, large_indices = large.max_pool2d(kernel_size=(5,5), stride=(6,5), return_indices=True)
    large_sinks = [early_simplify(call.src[0]) for call in large_indices.schedule_linear().src if call.src[0].op is Ops.SINK]
    large_prg = build_native_program(large_sinks[-1])
    self.assertIsNotNone(large_prg)
    large_subtasks = large_prg.src[1].src[0].arg
    self.assertEqual(len(large_subtasks), 1)
    self.assertEqual(large_subtasks[0].task.layout[1], _HOST_ARGMAX_LAYOUT)
    self.assertEqual(large_subtasks[0].task.layout[2:4], (25, 2500))

    overlap = Tensor.empty(1,1,6,6, dtype=dtypes.int, device="ROCKCHIP")
    _, overlap_indices = overlap.max_pool2d(kernel_size=(2,2), stride=1, return_indices=True)
    overlap_sinks = [early_simplify(call.src[0]) for call in overlap_indices.schedule_linear().src if call.src[0].op is Ops.SINK]
    overlap_prg = build_native_program(overlap_sinks[-1])
    self.assertIsNotNone(overlap_prg)
    overlap_subtasks = overlap_prg.src[1].src[0].arg
    self.assertEqual(len(overlap_subtasks), 1)
    self.assertEqual(overlap_subtasks[0].task.layout[1], _HOST_ARGMAX_LAYOUT)
    self.assertEqual(overlap_subtasks[0].task.layout[4], 8)
    self.assertEqual(overlap_subtasks[0].task.layout[5:9], (0, 1, 6, 7))

  def test_max_unpool_uses_typed_scatter_boundary(self):
    for dtype, itemsize in ((dtypes.half, 2), (dtypes.float, 4)):
      x = Tensor.empty(1,1,4,4, dtype=dtype, device="ROCKCHIP")
      values, indices = x.max_pool2d(kernel_size=(2,2), return_indices=True)
      output = values.max_unpool2d(indices, kernel_size=(2,2))
      sinks = [early_simplify(call.src[0]) for call in output.schedule_linear().src if call.src[0].op is Ops.SINK]
      prg = build_native_program(sinks[-1])
      self.assertIsNotNone(prg)
      subtasks = prg.src[1].src[0].arg
      self.assertEqual(len(subtasks), 1)
      self.assertEqual(subtasks[0].task.layout[1], _HOST_SCATTER_LAYOUT)
      self.assertEqual(subtasks[0].task.layout[-1], itemsize)

  def test_single_output_fp32_max_pool_keeps_spatial_index_boundary(self):
    x = Tensor.empty(1,1,2,3, dtype=dtypes.float, device="ROCKCHIP")
    _, indices = x.max_pool2d(kernel_size=(2,2), return_indices=True)
    sinks = [early_simplify(call.src[0]) for call in indices.schedule_linear().src if call.src[0].op is Ops.SINK]
    prg = build_native_program(sinks[-1])
    self.assertIsNotNone(prg)
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertEqual(subtasks[0].task.layout[:5], (1, _HOST_ARGMAX_LAYOUT, 4, 6, 4))

  def test_fp32_axis_argmax_reuses_low_typed_gather_slot(self):
    x = Tensor.rand(4,5, dtype=dtypes.float).realize()
    sinks = [c.src[0] for c in x.argmax(0).schedule_linear().src if c.src[0].op is Ops.SINK]
    self.assertEqual(len(sinks), 2)
    max_program = build_native_program(sinks[0])
    self.assertIsNotNone(max_program)
    subtasks = max_program.src[1].src[0].arg
    typed_slots = [slot for st in subtasks for slot in st.task.fp32_inputs]
    self.assertEqual(len(typed_slots), 4)
    self.assertEqual(len(set(typed_slots)), 1)
    self.assertIsNotNone(build_native_program(sinks[1]))

  def test_fp32_argsort_keeps_typed_gathers_in_encodable_slots(self):
    x = Tensor.rand(1,8,2, dtype=dtypes.float).realize()
    sinks = [c.src[0] for c in x.argsort(1, True).schedule_linear().src if c.src[0].op is Ops.SINK]
    programs = [build_native_program(sink) for sink in sinks]
    self.assertTrue(all(program is not None for program in programs))
    typed_slots = [slot for program in programs if program is not None
                   for st in program.src[1].src[0].arg for slot in st.task.fp32_inputs]
    self.assertTrue(typed_slots)
    self.assertLess(max(typed_slots), 7)
    subtasks = [st for program in programs if program is not None for st in program.src[1].src[0].arg]
    self.assertFalse(any(st.task.native_int32_input or st.task.native_int32_output for st in subtasks))
    self.assertFalse(any((cmd & 0xffff) == rk.REG_DPU_BN_RELUX_CMP_VALUE and
                         ((cmd >> 16) & 0xffffffff) == 0x3f800000 for st in subtasks for cmd in st.cmds))

  def test_isclose_uses_one_strict_serialized_task(self):
    a = Tensor.empty(8, dtype=dtypes.float, device="ROCKCHIP")
    b = Tensor.empty(8, dtype=dtypes.float, device="ROCKCHIP")
    prg = build_native_program(_get_sink(a.isclose(b)))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 1)
    self.assertTrue(subtasks[0].task.is_copy)
    self.assertEqual(subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)
    a_half = Tensor.empty(8, dtype=dtypes.half, device="ROCKCHIP")
    b_half = Tensor.empty(8, dtype=dtypes.half, device="ROCKCHIP")
    for expression in (a_half.isclose(b_half), a_half.isclose(1.0, equal_nan=True)):
      half_program = build_native_program(_get_sink(expression))
      half_subtasks = half_program.src[1].src[0].arg
      self.assertEqual(len(half_subtasks), 1)
      self.assertTrue(half_subtasks[0].task.is_copy)
      self.assertEqual(half_subtasks[0].task.layout[1], _HOST_ELEMENTWISE_LAYOUT)

  def test_k_tiled_dot_produces_binary(self):
    a = Tensor.rand(5000,dtype=dtypes.half).realize()
    b = Tensor.rand(5000,dtype=dtypes.half).realize()
    self.assertIsNotNone(self._compile(a.dot(b)))

  def test_max_produces_binary(self):
    a = Tensor.rand(4,8,dtype=dtypes.half).realize()
    binary = self._compile(a.max(axis=0))
    self.assertIsNotNone(binary)
    self.assertGreater(len(binary), 24)

  def test_command_words_match_known_patterns(self):
    a = Tensor.rand(4,4,dtype=dtypes.half).realize()
    b = Tensor.rand(4,4,dtype=dtypes.half).realize()
    binary = self._compile(a+b)
    dec_cmds, _, _ = decode_rk(binary)
    pc_cmd = dec_cmds[-1]
    target = (pc_cmd >> 48) & 0xFFFF
    reg = pc_cmd & 0xFFFF
    self.assertEqual(target, 0x81)
    self.assertEqual(reg, 0x8)

if __name__ == '__main__':
  unittest.main()
