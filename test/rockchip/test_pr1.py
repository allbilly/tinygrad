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
                                               _HOST_FP32_RESIDUAL_LAYOUT, _HOST_FP32_COMBINE_LAYOUT, _HOST_HALF_FP32_LAYOUT)
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

  def test_fp32_sum_uses_typed_cmac_boundary(self):
    sink = _get_sink(Tensor.empty(3,3, dtype=dtypes.float, device="ROCKCHIP").sum())
    program = build_native_program(sink)
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertGreaterEqual(len(subtasks), 2)
    self.assertTrue(subtasks[0].task.fp32_inputs)
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

  def test_fp32_factorized_zero_stride_sum_stays_native(self):
    a = Tensor.empty(2,4,1, dtype=dtypes.float, device="ROCKCHIP").expand(2,4,3)
    b = Tensor.empty(1,4,1, dtype=dtypes.float, device="ROCKCHIP").expand(2,4,3)
    program = build_native_program(_get_sink((a*b).sum((0,2))))
    self.assertIsNotNone(program)
    subtasks = program.src[1].src[0].arg
    self.assertTrue(any(task.task.kind == "cmac" for task in subtasks))
    self.assertEqual(subtasks[-1].task.layout[1], _HOST_FP32_COMBINE_LAYOUT)

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

  def test_local_max_pool_gathers_then_reduces_on_dpu(self):
    x = Tensor.rand(1,1,4,5,dtype=dtypes.half).realize()
    prg = build_native_program(_get_sink(x.max_pool2d(kernel_size=(2,2))))
    subtasks = prg.src[1].src[0].arg
    self.assertEqual(len(subtasks), 7)
    self.assertEqual(sum(st.task.is_copy for st in subtasks), 4)
    self.assertTrue(all(st.task.kind == "dpu" for st in subtasks))
    self.assertEqual(subtasks[-1].task.out_slot, 0)

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
