"""Rockchip NPU census: ops known to pass, with DRM_IOCTL_RKNPU_SUBMIT counts.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip.py -q -n0

The backend accepts FP16 inputs and emits contiguous FP16 DPU EW output. It does not
perform host arithmetic or FP32→FP16 conversion. GEMM gather layout is prepared on
the host, then every MUL/ADD is submitted to DPU EW.

Reduction via ROCKCHIP_EW_REDUCE=sequential|kahan|twoproduct (default sequential).
Each program uses one PC chain sized from its actual register-command and task-descriptor bytes.
"""
from __future__ import annotations
import math, unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context
from test.backend import test_ops as _test_ops
from test.backend.test_ops import helper_test_op, slow_test

# fp16 tol matches test_ops.test_gemm_fp16
_FP16 = dict(atol=5e-3, rtol=5e-3)
_FP16_WITH_GRAD = dict(atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
_TEST_OPS_HELPER = _test_ops.helper_test_op

def _fp16_test_op(*args, **kwargs):
  """Run a test_ops case at the same FP16 tolerance as test_gemm_fp16."""
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(*args, **kwargs)

def _fp16_fp32_golden_test_op(shps, torch_fxn, tinygrad_fxn, **kwargs):
  """Use an FP32-accumulated golden when CPU FP16 reduction is less accurate than DPU EW."""
  def fp32_golden(*tensors):
    out = torch_fxn(*(x.float() if x.is_floating_point() else x for x in tensors))
    return out.to(tensors[0].dtype) if tensors and tensors[0].is_floating_point() else out
  kwargs.update(_FP16_WITH_GRAD)
  return _TEST_OPS_HELPER(shps, fp32_golden, tinygrad_fxn, **kwargs)

def _ew_submits(n:int) -> int:
  """All tiles for one realized Rockchip program share one dynamically sized PC-chain."""
  return int(n > 0)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["ROCKCHIP"]

  def _half(self, shape, seed:int=0) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-2, 2, size=shape).astype(np.float16))

  def _check(self, expected_submits:int|None, out:Tensor, ref:np.ndarray, atol=5e-3, rtol=5e-3):
    """Realize `out`, compare to `ref`, assert ioctl submit delta.

    Default tol matches test_ops half gemm (DEFAULT_FLOAT=HALF / test_gemm_fp16).
    """
    before, before_tasks = self.dev.submit_count, self.dev.task_count
    got = out.realize().numpy()
    submits, tasks = self.dev.submit_count-before, self.dev.task_count-before_tasks
    if expected_submits is None: expected_submits = int(tasks > 0)
    print(f"  {self._testMethodName}: tasks={tasks} submits={submits} (expected {expected_submits})")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    self.assertEqual(submits, expected_submits, f"{self._testMethodName}: submits={submits} expected={expected_submits}")

  def _check_conv2d(self, x_shape:tuple[int, ...], w_shape:tuple[int, ...], seed:int, **kwargs):
    rng = np.random.default_rng(seed)
    xn = rng.uniform(-2, 2, size=x_shape).astype(np.float16)
    wn = rng.uniform(-2, 2, size=w_shape).astype(np.float16)
    ref = torch.nn.functional.conv2d(torch.from_numpy(xn), torch.from_numpy(wn), **kwargs).numpy()
    self._check(None, Tensor(xn).conv2d(Tensor(wn), **kwargs), ref, **_FP16)

  # ---- ADD ----
  def test_tiny_add(self):
    a, b = self._half((3,), 1), self._half((3,), 2)
    self._check(_ew_submits(3), a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add(self):
    a, b = self._half((45, 68), 3), self._half((45, 68), 4)
    self._check(_ew_submits(45*68), a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add_scalar_constfold(self):
    # Tensor(1)+0.5 folds on device=None — no NPU submit
    self._check(0, Tensor(1) + 0.5, np.array(1.5, dtype=np.float16))

  def test_add_empty(self):
    # rank-0 buffers materialize without NPU submit
    a, b = self._half((), 5), self._half((), 6)
    self._check(0, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add3(self):
    # two logical EW ops share one PC-chain ioctl
    a, b, c = self._half((45, 65), 7), self._half((45, 65), 8), self._half((45, 65), 9)
    ref = (a.numpy().astype(np.float32) + b.numpy() + c.numpy()).astype(np.float16)
    self._check(1, a + b + c, ref)

  # ---- MUL ----
  def test_tiny_mul(self):
    a, b = self._half((64,), 10), self._half((64,), 11)
    self._check(_ew_submits(64), a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_mul(self):
    a, b = self._half((64, 64), 12), self._half((64, 64), 13)
    self._check(_ew_submits(64*64), a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_scalar_mul(self):
    a = self._half((45, 65), 14)
    n = _ew_submits(45*65)
    self._check(n, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(n, a * -1, (a.numpy().astype(np.float32) * -1).astype(np.float16))
    self._check(n, 255 * a, (a.numpy().astype(np.float32) * 255).astype(np.float16))
    self._check(n, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_scalar_mul_empty(self):
    # rank-0 scalar mul — no NPU submit
    a = self._half((), 15)
    self._check(0, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(0, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_mul_naninf(self):
    a = self._half((45, 65), 16)
    n = _ew_submits(45*65)
    self._check(n, a * math.inf, (a.numpy().astype(np.float32) * np.float32(np.inf)).astype(np.float16))
    self._check(n, a * -math.inf, (a.numpy().astype(np.float32) * np.float32(-np.inf)).astype(np.float16))
    self._check(n, a * math.nan, (a.numpy().astype(np.float32) * np.float32(np.nan)).astype(np.float16))

  # ---- DIV ----
  def test_tiny_div(self):
    lhs = np.array([-2.0, 3.0, 4.0], dtype=np.float16)
    rhs = np.array([0.5, -2.0, 8.0], dtype=np.float16)
    self._check(1, Tensor(lhs) / Tensor(rhs), (lhs.astype(np.float32) / rhs).astype(np.float16))

  def test_infinite_division_sign(self):
    values = np.array([-3.0, -0.5, 0.5, 2.0], dtype=np.float16)
    self._check(1, math.inf / Tensor(values), (np.float16(np.inf) / values).astype(np.float16))
    self._check(1, -math.inf / Tensor(values), (np.float16(-np.inf) / values).astype(np.float16))

  # ---- MAX ----
  def test_maximum_fp16(self):
    a, b = self._half((45, 65), 17), self._half((45, 65), 18)
    self._check(_ew_submits(45*65), a.maximum(b), np.maximum(a.numpy(), b.numpy()))

  def test_max_pool2d_simple_submit(self):
    xn = np.array([[[[-1, 2, 0], [3, -4, 1]]]], dtype=np.float16)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(xn), kernel_size=(2, 2)).numpy()
    self._check(1, Tensor(xn).max_pool2d(kernel_size=(2, 2)), ref)

  def test_avg_pool2d_valid_count_submit(self):
    xn = np.arange(16, dtype=np.float16).reshape(1, 1, 4, 4)
    args = dict(kernel_size=(3, 3), padding=1, count_include_pad=False)
    ref = torch.nn.functional.avg_pool2d(torch.from_numpy(xn), **args).numpy()
    self._check(1, Tensor(xn).avg_pool2d(**args), ref)

  def test_interpolate_nearest_submit(self):
    x = self._half((2, 3, 13), 502)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(9,), mode="nearest").numpy()
    self._check(1, x.interpolate((9,), mode="nearest"), ref, **_FP16)

  def test_interpolate_linear_submit(self):
    x = self._half((2, 3, 52), 503)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(29,), mode="linear").numpy()
    self._check(1, x.interpolate((29,), mode="linear"), ref, **_FP16)

  def test_interpolate_bilinear_submit(self):
    x = self._half((2, 3, 12, 20), 504)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(9, 31), mode="bilinear").numpy()
    self._check(2, x.interpolate((9, 31), mode="linear"), ref, **_FP16)

  def test_interpolate_trilinear_submit(self):
    x = self._half((1, 1, 3, 2, 4), 505)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(2, 4, 3), mode="trilinear").numpy()
    self._check(3, x.interpolate((2, 4, 3), mode="linear"), ref, **_FP16)

  # ---- GEMM / MATMUL (from test_ops, fp16 tol) ----
  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  @slow_test
  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_matmul_batched(self):
    helper_test_op([(3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_matmul_batched_vector(self):
    helper_test_op([(4,3), (1,3,3,5)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
  def test_small_gemm(self):
    helper_test_op([(8,8), (8,8)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_9_gemm(self):
    helper_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_small_gemm_padded(self):
    helper_test_op([(9,9), (9,9)],
                   lambda x,y: torch.nn.functional.pad(x, (0,7,0,7)).matmul(torch.nn.functional.pad(y, (0,7,0,7))),
                   lambda x,y: x.pad(((0,7),(0,7)))@y.pad(((0,7),(0,7))), **_FP16)
  def test_small_gemm_range(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.arange(0,64,dtype=np.float16).reshape(8,8),
                         np.arange(64,128,dtype=np.float16).reshape(8,8)], **_FP16)
  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.eye(8).astype(np.float16), np.eye(8).astype(np.float16)], **_FP16)
  @slow_test
  def test_gemm_fp16(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()), **_FP16)
  @slow_test
  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.matmul(y), **_FP16)
  @slow_test
  def test_big_gemm(self):
    helper_test_op([(256,256), (256,256)], lambda x,y: x.matmul(y), **_FP16)
  def test_gemm_with_zeros_shape(self):
    helper_test_op([(8,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,8), (8,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,8), (8,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(8,0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0,0), (0,0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0), (0,8)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)
    helper_test_op([(0), (0)], lambda x,y: x.matmul(y), Tensor.dot, **_FP16)

  # ---- CONV2D (from test_ops, fp16 tol) ----
  def test_simple_conv2d_1x1(self):
    self._check_conv2d((1,4,9,9), (4,4,1,1), 100)

  def test_simple_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 101)

  def test_simple_conv2d_batched(self):
    self._check_conv2d((2,4,9,9), (4,4,3,3), 102)

  def test_padded_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 103, padding=1)

  def test_strided_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 104, stride=2)

  def test_depthwise_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,1,3,3), 105, groups=4)

  def test_simple_conv2d_reduce63(self):
    self._check_conv2d((1,7,9,9), (4,7,3,3), 207)

  def test_simple_conv2d_reduce72(self):
    self._check_conv2d((1,8,9,9), (4,8,3,3), 208)

  def test_simple_conv2d_m4(self):
    self._check_conv2d((1,16,9,9), (16,16,3,3), 300)

  def test_simple_conv2d_1x1_m4(self):
    self._check_conv2d((1,16,32,32), (16,16,1,1), 301)

  def test_grouped_conv2d(self):
    self._check_conv2d((1,8,9,9), (8,4,3,3), 302, groups=2)

  def test_dilated_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 303, dilation=2)

  def test_asymmetric_padding_conv2d(self):
    rng = np.random.default_rng(400)
    xn = rng.uniform(-2, 2, size=(1,1,4,4)).astype(np.float16)
    wn = rng.uniform(-2, 2, size=(1,1,2,2)).astype(np.float16)
    ref = torch.nn.functional.conv2d(torch.nn.functional.pad(torch.from_numpy(xn), (2,1,2,1)), torch.from_numpy(wn)).numpy()
    self._check(1, Tensor(xn).conv2d(Tensor(wn), padding=(2,1,2,1)), ref, **_FP16)

  def test_output_padded_conv_transpose2d(self):
    rng = np.random.default_rng(401)
    xn = rng.uniform(-2, 2, size=(2,4,6,5)).astype(np.float16)
    wn = rng.uniform(-2, 2, size=(4,4,3,3)).astype(np.float16)
    bn = rng.uniform(-2, 2, size=(4,)).astype(np.float16)
    args = dict(output_padding=(1,1), stride=(2,3))
    ref = torch.nn.functional.conv_transpose2d(torch.from_numpy(xn), torch.from_numpy(wn), torch.from_numpy(bn), **args).numpy()
    self._check(None, Tensor(xn).conv_transpose2d(Tensor(wn), Tensor(bn), **args), ref, **_FP16)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipConvOps(unittest.TestCase):
  """Every test*conv* case from test_ops, rerun with the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception
  _test_conv2d = _test_ops.TestOps._test_conv2d

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_bias_conv_transpose2d(self):
    # PyTorch CPU accumulates this FP16 ConvTranspose in FP16 and differs from the exact product sum after bias cancellation.
    _fp16_fp32_golden_test_op([(2,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b), lambda x,w,b: Tensor.conv_transpose2d(x,w,b))

  @slow_test
  def test_simple_conv_transpose3d(self):
    # The DPU expansion is also closer to FP32 accumulation than PyTorch CPU's FP16 ConvTranspose3D reduction.
    _fp16_fp32_golden_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w), lambda x,w: Tensor.conv_transpose2d(x,w))

# Keep the Rockchip convolution census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test") and "conv" in _name and _name not in ("test_bias_conv_transpose2d", "test_simple_conv_transpose3d"):
    setattr(TestRockchipConvOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxPoolOps(unittest.TestCase):
  """Every numeric FP16 MaxPool2D case from test_ops at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  @unittest.skip("Rockchip accepts FP16 inputs only")
  def test_max_pool2d_padding_int(self): pass

  @unittest.skip("Rockchip DPU EW has no integer index output")
  def test_max_pool2d_return_indices(self): pass

# Keep the numeric MaxPool2D census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test_max_pool2d") and _name not in ("test_max_pool2d_padding_int", "test_max_pool2d_return_indices"):
    setattr(TestRockchipMaxPoolOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAvgPoolOps(unittest.TestCase):
  """Every FP16 AvgPool case from test_ops at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_avg_pool3d(self):
    # PyTorch CPU does not implement FP16 AvgPool3D; use its FP32 result cast to the FP16 device contract.
    _fp16_fp32_golden_test_op([(1,1,16,16,16)],
      lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False), forward_only=True)

# Keep the FP16 AvgPool census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if (_name.startswith("test_avg_pool") and _name != "test_avg_pool3d") or _name == "test_global_avg_pool2d":
    setattr(TestRockchipAvgPoolOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAdaptivePoolOps(unittest.TestCase):
  """Adaptive-pooling-equivalent divisible windows; tinygrad has no adaptive pooling API."""

  def test_adaptive_avg_pool2d_equivalent(self):
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (2,2)),
                  lambda x: x.avg_pool2d(kernel_size=2, stride=2))
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1,1)),
                  lambda x: x.avg_pool2d(kernel_size=4, stride=4))

  def test_adaptive_max_pool2d_equivalent(self):
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_max_pool2d(x, (2,2)),
                  lambda x: x.max_pool2d(kernel_size=2, stride=2))
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_max_pool2d(x, (1,1)),
                  lambda x: x.max_pool2d(kernel_size=4, stride=4))

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPolynomialLossOps(unittest.TestCase):
  """Polynomial and simple loss expressions composed only from FP16 DPU EW primitives."""

  def test_square_and_cubic(self):
    _fp16_test_op([(4,5)], lambda x: x.square(), lambda x: x.square())
    _fp16_test_op([(4,5)], lambda x: x*x*x, lambda x: x*x*x)

  def test_horner_polynomial(self):
    _fp16_test_op([(4,5)], lambda x: (x*0.5-1.25)*x+0.75, lambda x: (x*0.5-1.25)*x+0.75)

  def test_mse_loss(self):
    _fp16_fp32_golden_test_op([(3,4), (3,4)], lambda x,y: (x-y).square().mean(), lambda x,y: (x-y).square().mean())

  def test_l1_loss(self):
    _fp16_fp32_golden_test_op([(3,4), (3,4)], lambda x,y: (x-y).abs().mean(), lambda x,y: (x-y).abs().mean())

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipScatterOps(unittest.TestCase):
  """FP16 scatter with compile-time-static indices; external integer NPU inputs are unsupported."""

  def test_scatter_static_tensor_source(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(2,4), (2,2)], lambda x,src: x.scatter(1, index, src),
                  lambda x,src: x.scatter(1, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src), forward_only=True)

  def test_scatter_static_dim0(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(4,2), (2,2)], lambda x,src: x.scatter(0, index, src),
                  lambda x,src: x.scatter(0, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src), forward_only=True)

  def test_scatter_static_scalar_source(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(2,4)], lambda x: x.scatter(1, index, value=0.5),
                  lambda x: x.scatter(1, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src=0.5), forward_only=True)

  def test_scatter_reduce_static_sum(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="sum"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="sum"), forward_only=True)

  def test_scatter_reduce_static_max(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="amax"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="amax"), forward_only=True)

  def test_scatter_reduce_static_product(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="prod"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="prod"), forward_only=True)

  def test_scatter_reduce_static_min(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="amin"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="amin"), forward_only=True)

  def test_scatter_reduce_static_mean(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_fp32_golden_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="mean"),
      lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="mean"), forward_only=True)

  @unittest.skip("Rockchip accepts FP16 inputs only; external integer index buffers are excluded")
  def test_scatter_dynamic_integer_index(self): pass

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBroadcastOps(unittest.TestCase):
  """FP16 broadcasting through static gather and DPU EW arithmetic."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_broadcast_simple = _test_ops.TestOps.test_broadcast_simple

  def test_broadcast_full_arithmetic(self):
    for torch_op, tinygrad_op in ((torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul), (torch.div, Tensor.div)):
      for shapes in (((5,3,14,16), (5,1,14,1)), ((1,3,1,7,1), (2,1,5,1,8))):
        with self.subTest(op=torch_op.__name__, shapes=shapes): _fp16_test_op(shapes, torch_op, tinygrad_op)

  def test_broadcast_partial_arithmetic(self):
    shapes = (((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)), ((4,1), (4,5)), ((1,4), (5,4)))
    for torch_op, tinygrad_op in ((torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul), (torch.div, Tensor.div)):
      for pair in shapes:
        with self.subTest(op=torch_op.__name__, shapes=pair): _fp16_test_op(pair, torch_op, tinygrad_op)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipDotOps(unittest.TestCase):
  """FP16 dot, batched dot, and matvec compositions lowered through DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_dot_1d = _test_ops.TestOps.test_dot_1d
  test_dot = _test_ops.TestOps.test_dot
  test_broadcastdot = _test_ops.TestOps.test_broadcastdot
  test_multidot = _test_ops.TestOps.test_multidot
  test_matvec = _test_ops.TestOps.test_matvec
  test_matvecmat = _test_ops.TestOps.test_matvecmat

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipInterpolateOps(unittest.TestCase):
  """FP16 interpolation cases advanced one test_ops method at a time."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_interpolate_nearest = _test_ops.TestOps.test_interpolate_nearest
  test_interpolate_nearest_exact = _test_ops.TestOps.test_interpolate_nearest_exact
  test_interpolate_linear = _test_ops.TestOps.test_interpolate_linear
  test_interpolate_linear_corners_aligned = _test_ops.TestOps.test_interpolate_linear_corners_aligned
  test_interpolate_bilinear = _test_ops.TestOps.test_interpolate_bilinear
  test_interpolate_bilinear_corners_aligned = _test_ops.TestOps.test_interpolate_bilinear_corners_aligned
  test_interpolate_trilinear = _test_ops.TestOps.test_interpolate_trilinear
  test_interpolate_trilinear_corners_aligned = _test_ops.TestOps.test_interpolate_trilinear_corners_aligned

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMovementOps(unittest.TestCase):
  """Static FP16 view and layout methods proven by the Rockchip gather path."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_transpose = _test_ops.TestOps.test_transpose
  test_permute = _test_ops.TestOps.test_permute
  test_reshape = _test_ops.TestOps.test_reshape
  test_view = _test_ops.TestOps.test_view
  test_flip = _test_ops.TestOps.test_flip
  test_squeeze = _test_ops.TestOps.test_squeeze
  test_unsqueeze = _test_ops.TestOps.test_unsqueeze
  test_flatten = _test_ops.TestOps.test_flatten
  test_unflatten = _test_ops.TestOps.test_unflatten
  test_detach = _test_ops.TestOps.test_detach
  test_expand = _test_ops.TestOps.test_expand

  def test_slice_negative_strides(self):
    rng = np.random.default_rng(600)
    a = rng.standard_normal((10,10,10)).astype(np.float16)
    t = Tensor(a)
    for idx in (np.s_[::-1], np.s_[::-2], np.s_[:,2:0:-1], np.s_[:,2:0:-1,3:1:-2], np.s_[4:0:-3,2:0:-1,-1:-5:-2],
                np.s_[2:5:-1,:,:], np.s_[:,2:5:-1,:], np.s_[:,:,2:5:-1]):
      np.testing.assert_allclose(a[idx], t[idx].numpy(), **_FP16)

  def test_slice_with_const_tensor(self):
    self.skipTest("Rockchip DPU accepts FP16 inputs only; this case requires an integer index tensor")

  test_slice_in_bounds_1dim = _test_ops.TestOps.test_slice_in_bounds_1dim
  test_slice_on_0dim_tensor = _test_ops.TestOps.test_slice_on_0dim_tensor
  test_slice_int_indexing = _test_ops.TestOps.test_slice_int_indexing
  test_slice_in_bounds_multidim = _test_ops.TestOps.test_slice_in_bounds_multidim
  test_slice_with_none = _test_ops.TestOps.test_slice_with_none
  test_slice_one_endpoint_out_of_bounds = _test_ops.TestOps.test_slice_one_endpoint_out_of_bounds
  test_slice_stride_gt_one = _test_ops.TestOps.test_slice_stride_gt_one
  test_slice_both_endpoints_out_of_bounds = _test_ops.TestOps.test_slice_both_endpoints_out_of_bounds
  test_slice_start_gt_end = _test_ops.TestOps.test_slice_start_gt_end
  test_slice_zero_in_shape = _test_ops.TestOps.test_slice_zero_in_shape
  test_slice_errors = _test_ops.TestOps.test_slice_errors
  test_slice_ellipsis = _test_ops.TestOps.test_slice_ellipsis
  test_double_slice = _test_ops.TestOps.test_double_slice
  test_pad_reshape = _test_ops.TestOps.test_pad_reshape
  test_pad_slice = _test_ops.TestOps.test_pad_slice
  test_diag = _test_ops.TestOps.test_diag
  test_diagonal = _test_ops.TestOps.test_diagonal
  test_roll = _test_ops.TestOps.test_roll

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipConcatOps(unittest.TestCase):
  """FP16 concatenation, stacking, and repetition through partial raw gathers."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_stack(self):
    for dim in range(-1, 3):
      _fp16_test_op([(5,6,3)]*3, lambda x,y,z: torch.stack((x,y,z), dim), lambda x,y,z: Tensor.stack(x,y,z,dim=dim))
      _fp16_test_op([(5,6,3)]*3, lambda x,y,z: torch.stack((x,y,z), dim), lambda x,y,z: Tensor.stack((x,y,z),dim=dim))
    with self.assertRaises(IndexError): Tensor.stack(Tensor.randn(45,65,3), dim=77)
    with self.assertRaises(ValueError): Tensor.stack((Tensor([1,2]), Tensor([3,4])), Tensor([5,6]))
    np.testing.assert_allclose(Tensor.stack(Tensor(3.14), Tensor(3.14)).numpy(), np.array([3.14,3.14]), **_FP16)

  test_cat = _test_ops.TestOps.test_cat
  test_multicat = _test_ops.TestOps.test_multicat
  test_stack_slice = _test_ops.TestOps.test_stack_slice
  test_stack_max = _test_ops.TestOps.test_stack_max
  test_repeat = _test_ops.TestOps.test_repeat
  test_repeat_interleave = _test_ops.TestOps.test_repeat_interleave
  test_simple_repeat = _test_ops.TestOps.test_simple_repeat

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPaddingOps(unittest.TestCase):
  """Constant, reflect, replicate, and circular FP16 padding through raw gathers."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_pad = _test_ops.TestOps.test_pad
  test_pad_reflect_mode = _test_ops.TestOps.test_pad_reflect_mode
  test_pad_replicate_mode = _test_ops.TestOps.test_pad_replicate_mode
  test_pad_circular_mode = _test_ops.TestOps.test_pad_circular_mode

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipReductionOps(unittest.TestCase):
  """FP16 scalar sum, mean, minimum/maximum, and product reductions on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_min(self):
    for shape in ((3,3), (45,3)): _fp16_test_op([shape], lambda x: x.min())
    _fp16_test_op([(45,3)], lambda x: x.min().mul(0.5))
    _fp16_test_op([()], lambda x: x.min())

  def test_max(self):
    _fp16_test_op([(45,3)], lambda x: x.max())
    _fp16_test_op([(45,3)], lambda x: x.max().mul(0.5))
    _fp16_test_op(None, lambda x: x.max().mul(0.5), vals=[[[1.0,1.0,0.0,1.0]]])
    _fp16_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: x.max(axis=1))
    _fp16_test_op([()], lambda x: x.max())

  def test_sum_full(self):
    before = Device["ROCKCHIP"].submit_count
    with Context(NOOPT=1): _fp16_test_op([(16384)], lambda x: x.sum())
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 1)

  def test_sum_dtype_arg(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(45,3)], lambda x: x.sum(dtype=torch.float32), lambda x: x.sum(dtype=dtypes.float32), forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 2)
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).sum(dtype="")

  def test_non_fp16_reductions(self):
    self.skipTest("Rockchip DPU accepts FP16 tensors only; FP32 dtype, boolean, and integer reductions are excluded")

  test_sum_fake = _test_ops.TestOps.test_sum_fake
  test_sum_collapse = _test_ops.TestOps.test_sum_collapse
  test_sum_collapse_neg = _test_ops.TestOps.test_sum_collapse_neg
  test_sum_pad_collapse = _test_ops.TestOps.test_sum_pad_collapse
  test_sum_twice = _test_ops.TestOps.test_sum_twice
  test_sum_cat_collapse = _test_ops.TestOps.test_sum_cat_collapse
  test_max_dont_collapse = _test_ops.TestOps.test_max_dont_collapse
  test_sum_simple = _test_ops.TestOps.test_sum_simple
  test_sum_relu = _test_ops.TestOps.test_sum_relu
  test_sum_tiny = _test_ops.TestOps.test_sum_tiny
  test_sum = _test_ops.TestOps.test_sum
  test_sum_with_zeros_shape = _test_ops.TestOps.test_sum_with_zeros_shape
  test_prod = _test_ops.TestOps.test_prod
  test_prod_dtype_arg = _test_ops.TestOps.test_prod_dtype_arg
  test_const_reduce = _test_ops.TestOps.test_const_reduce
  test_mean = _test_ops.TestOps.test_mean
  test_mean_axis = _test_ops.TestOps.test_mean_axis
  test_mean_zero_axis = _test_ops.TestOps.test_mean_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIncrementalOps(unittest.TestCase):
  """Remaining test_ops methods, admitted and debugged in source order."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_full_like = _test_ops.TestOps.test_full_like
  test_full = _test_ops.TestOps.test_full
  test_negative_dims = _test_ops.TestOps.test_negative_dims
  test_negative_dims_full = _test_ops.TestOps.test_negative_dims_full
  test_negative_dims_eye = _test_ops.TestOps.test_negative_dims_eye
  test_negative_dims_kaiming = _test_ops.TestOps.test_negative_dims_kaiming
  test_zeros = _test_ops.TestOps.test_zeros
  test_zeros_like = _test_ops.TestOps.test_zeros_like
  test_empty_0 = _test_ops.TestOps.test_empty_0
  test_ones = _test_ops.TestOps.test_ones
  test_ones_like = _test_ops.TestOps.test_ones_like
  test_eye = _test_ops.TestOps.test_eye
  test_split = _test_ops.TestOps.test_split
  test_chunk = _test_ops.TestOps.test_chunk
  test_unfold = _test_ops.TestOps.test_unfold
  test_meshgrid = _test_ops.TestOps.test_meshgrid
  test_arange = _test_ops.TestOps.test_arange
  test_arange_big = _test_ops.TestOps.test_arange_big
  test_arange_4096 = _test_ops.TestOps.test_arange_4096
  test_linspace = _test_ops.TestOps.test_linspace
  test_lerp = _test_ops.TestOps.test_lerp
  test_broadcasted_add = _test_ops.TestOps.test_broadcasted_add
  test_broadcasted_add_2 = _test_ops.TestOps.test_broadcasted_add_2
  test_sub = _test_ops.TestOps.test_sub
  test_scalar_sub = _test_ops.TestOps.test_scalar_sub
  test_scalar_rsub = _test_ops.TestOps.test_scalar_rsub
  test_neg = _test_ops.TestOps.test_neg
  test_tiny_add = _test_ops.TestOps.test_tiny_add
  test_tiny_mul = _test_ops.TestOps.test_tiny_mul
  test_add = _test_ops.TestOps.test_add
  test_add3 = _test_ops.TestOps.test_add3
  test_mul = _test_ops.TestOps.test_mul
  test_scalar_mul = _test_ops.TestOps.test_scalar_mul
  test_div = _test_ops.TestOps.test_div
  test_scalar_div = _test_ops.TestOps.test_scalar_div
  test_mul_naninf = _test_ops.TestOps.test_mul_naninf
  test_div_naninf = _test_ops.TestOps.test_div_naninf
  test_relu = _test_ops.TestOps.test_relu
  test_relu_exact = _test_ops.TestOps.test_relu_exact
  test_relu_maximum_exact = _test_ops.TestOps.test_relu_maximum_exact
  test_leaky_relu = _test_ops.TestOps.test_leaky_relu
  test_abs = _test_ops.TestOps.test_abs
  test_abs_exact = _test_ops.TestOps.test_abs_exact
  test_relu6 = _test_ops.TestOps.test_relu6
  test_clip = _test_ops.TestOps.test_clip
  test_hardtanh = _test_ops.TestOps.test_hardtanh
  test_hardsigmoid = _test_ops.TestOps.test_hardsigmoid
  test_hardsigmoid_extreme = _test_ops.TestOps.test_hardsigmoid_extreme
  test_hardswish = _test_ops.TestOps.test_hardswish

if __name__ == "__main__":
  unittest.main()
