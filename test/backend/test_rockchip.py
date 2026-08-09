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

def _fp16_pool_test_op(*args, **kwargs):
  """Keep explicit MaxPool fixtures on the backend's FP16 input contract."""
  if (vals:=kwargs.get("vals")) is not None: kwargs["vals"] = [np.asarray(value, dtype=np.float16) for value in vals]
  return _fp16_test_op(*args, **kwargs)

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
  """Every FP16 MaxPool2D case from test_ops at the test_gemm_fp16 tolerance."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_pool_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  @unittest.skip("Rockchip accepts FP16 inputs only")
  def test_max_pool2d_padding_int(self): pass

  def test_max_pool2d_return_indices_wide(self):
    data = np.zeros((1,1,50,50), dtype=np.float16)
    data[0,0,46,49] = 1
    expected = torch.nn.functional.max_pool2d(torch.from_numpy(data), kernel_size=(5,5), stride=(6,5), return_indices=True)[1].int().numpy()
    got = Tensor(data, device="ROCKCHIP").max_pool2d((5,5), stride=(6,5), return_indices=True)[1].realize().numpy()
    self.assertGreater(int(expected.max()), 2048)
    np.testing.assert_array_equal(got, expected)

# Keep the MaxPool2D census synchronized as test_ops grows.
for _name, _test in vars(_test_ops.TestOps).items():
  if _name.startswith("test_max_pool2d") and _name != "test_max_pool2d_padding_int":
    setattr(TestRockchipMaxPoolOps, _name, _test)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxUnpoolOps(unittest.TestCase):
  """FP16 MaxUnpool scatter from the upstream operator census."""

  def test_max_unpool2d_nonfinite_bits(self):
    values = np.array([math.inf, -math.inf, math.nan, 3.5], dtype=np.float16).reshape(4,1,1,1)
    indices = np.arange(4, dtype=np.int32).reshape(4,1,1,1)
    expected = np.zeros((4,1,2,2), dtype=np.float16)
    expected.reshape(4,4)[np.arange(4),np.arange(4)] = values.reshape(4)
    got = Tensor(values, device="ROCKCHIP").max_unpool2d(Tensor(indices, device="ROCKCHIP"), 2).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

  def test_max_unpool2d_inf(self):
    data = np.array([[[[math.inf, -math.inf, math.nan], [1.0, 2.0, 3.0]]]], dtype=np.float16)
    expected = torch.nn.functional.max_unpool2d(
      *torch.nn.functional.max_pool2d(torch.from_numpy(data), kernel_size=(2,2), return_indices=True), kernel_size=(2,2)).numpy()
    got = Tensor.max_unpool2d(
      *Tensor.max_pool2d(Tensor(data, device="ROCKCHIP"), kernel_size=(2,2), return_indices=True), kernel_size=(2,2)).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

  def test_max_unpool2d_wide_indices(self):
    values = np.arange(1, 15, dtype=np.float16).reshape(7,1,1,2)
    indices = np.array([[2049,2499]]*7, dtype=np.int32).reshape(7,1,1,2)
    expected = np.zeros((7,1,50,50), dtype=np.float16)
    expected.reshape(7,2500)[:,2049] = values.reshape(7,2)[:,0]
    expected.reshape(7,2500)[:,2499] = values.reshape(7,2)[:,1]
    got = Tensor(values, device="ROCKCHIP").max_unpool2d(Tensor(indices, device="ROCKCHIP"), (1,2), output_size=(50,50)).realize().numpy()
    np.testing.assert_array_equal(got, expected)

  @slow_test
  def test_max_unpool2d_wide(self):
    args = {"kernel_size":(5,5), "stride":(6,5)}
    _fp16_test_op([(8,3,50,50)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args), forward_only=True)

  def test_max_unpool2d_bounded(self):
    _fp16_test_op([(1,3,7,6)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                                 kernel_size=(2,2), output_size=(99,99,7,6)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                    kernel_size=(2,2), output_size=(99,99,7,6)), forward_only=True)

  def test_max_unpool2d_padded(self):
    args = {"kernel_size":(3,3), "stride":(6,7), "padding":1}
    _fp16_test_op([(8,3,30,30)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args),
                                                 **args, output_size=(30,30)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args),
                                    **args, output_size=(30,30)), forward_only=True)

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

  test_topo_sort = _test_ops.TestOps.test_topo_sort

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
class TestRockchipIntegerPowerOps(unittest.TestCase):
  """Constant integer powers composed from FP16 DPU EW multiplication and division."""

  def test_pow(self):
    before = Device["ROCKCHIP"].submit_count
    for exponent in (0, 1, 2, 3, -2):
      _fp16_test_op([(45,65)], lambda x,exponent=exponent:x**exponent, forward_only=True)
    for exponent in (2, -2):
      _fp16_test_op([()], lambda x,exponent=exponent:x**exponent, forward_only=True)
    _fp16_test_op([(45,65)], lambda x:x**3, forward_only=True, low=-30, high=-27)
    _fp16_test_op([()], lambda x:x**3, forward_only=True, low=-30, high=-27)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 5)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipOneHotOps(unittest.TestCase):
  """Exact INT32 one-hot equality through four raw bytes and DPU EW masks."""

  test_one_hot = _test_ops.TestOps.test_one_hot

  def test_one_hot_full_int32_bytes(self):
    values = np.array([0, 5, 256, 65536, 1 << 24, -1], dtype=np.int32)
    expected = np.zeros((len(values), 6), dtype=np.int32)
    expected[0,0] = expected[1,5] = 1
    got = Tensor(values, device="ROCKCHIP").one_hot(6).realize().numpy()
    np.testing.assert_array_equal(got, expected)

  def test_one_hot_beyond_fp16_integer_range(self):
    expected = np.zeros((1, 2050), dtype=np.int32)
    expected[0,2049] = 1
    got = Tensor(np.array([2049], dtype=np.int32), device="ROCKCHIP").one_hot(2050).realize().numpy()
    np.testing.assert_array_equal(got, expected)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipGatherOps(unittest.TestCase):
  """Dynamic INT32 gather with exact index and raw FP16 representation selection on DPU."""

  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_gather = _test_ops.TestOps.test_gather

  def test_gather_nonfinite_full_index_bytes(self):
    source = np.array([math.inf, -math.inf, math.nan], dtype=np.float16)
    indices = np.array([0, 1, 2, 256, 65536, 1 << 24, -1], dtype=np.int32)
    expected = np.zeros(len(indices), dtype=np.float16)
    expected[:3] = source
    got = Tensor(source, device="ROCKCHIP").gather(0, Tensor(indices, device="ROCKCHIP")).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipFancyIndexOps(unittest.TestCase):
  """Dynamic INT32 fancy indices with exact negative normalization and FP16 bits."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_fancy_indexing_inf = _test_ops.TestOps.test_fancy_indexing_inf
  test_slice_fancy_indexing_with_tensors = _test_ops.TestOps.test_slice_fancy_indexing_with_tensors

  def test_fancy_indexing_negative_nonfinite_bits(self):
    source = np.array([math.inf, -math.inf, math.nan], dtype=np.float16)
    indices = np.array([-1, -2, -3], dtype=np.int32)
    got = Tensor(source, device="ROCKCHIP")[Tensor(indices, device="ROCKCHIP")].realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), source[::-1].view(np.uint16))

  def test_multi_fancy_indexing_negative_nonfinite_bits(self):
    source = np.array([[math.inf, -math.inf, math.nan], [1.0, -0.0, 0.0]], dtype=np.float16)
    rows = Tensor(np.array([[0, 0, 0], [-1, -1, -1]], dtype=np.int32), device="ROCKCHIP")
    columns = Tensor(np.array([-3, -2, -1], dtype=np.int32), device="ROCKCHIP")
    got = Tensor(source, device="ROCKCHIP")[rows, columns].realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), source.view(np.uint16))

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipScatterOps(unittest.TestCase):
  """FP16 Scatter and ScatterReduce through static or exact dynamic INT32 selection."""

  test_scatter_add = _test_ops.TestOps.test_scatter_add
  test_scatter_mul = _test_ops.TestOps.test_scatter_mul

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

  def test_scatter_dynamic_integer_index(self):
    base = np.array([0x7c00, 0xfc00, 0x7e01, 0x8000], dtype=np.uint16).view(np.float16)
    for index,source in (([1,3], [0x0000,0xfe01]), ([0,0], [0x7c00,0xfe01])):
      with self.subTest(index=index):
        indices = np.array(index, dtype=np.int32)
        values = np.array(source, dtype=np.uint16).view(np.float16)
        expected = base.copy()
        for lane,dst in enumerate(indices): expected[dst] = values[lane]
        got = Tensor(base).scatter(0, Tensor(indices), Tensor(values)).realize().numpy()
        np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

  def test_scatter_dynamic_scalar_reductions(self):
    base = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float16)
    indices = Tensor(np.array([0, 0, 2, 1], dtype=np.int32)).realize()
    for mode,value,expected in (("add", 2.0, [5.0, 0.0, 5.0, 4.0]),
                                ("multiply", -1.0, [1.0, 2.0, -3.0, 4.0])):
      with self.subTest(mode=mode):
        got = Tensor(base).scatter(0, indices, value, reduce=mode).realize().numpy()
        np.testing.assert_array_equal(got, np.array(expected, dtype=np.float16))

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
class TestRockchipEinsumOps(unittest.TestCase):
  """FP16 einsum contractions composed from static gathers and DPU EW reductions."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_einsum = _test_ops.TestOps.test_einsum

  def test_einsum_ellipsis(self):
    _fp16_test_op([(3,8,9), (3,8,9)], lambda a,b:torch.einsum("...id,...jd->...ij", a, b),
                  lambda a,b:Tensor.einsum("...id,...jd->...ij", a, b))
    _fp16_test_op([(3,8,9), (3,8,9)], lambda a,b:torch.einsum("...id,...jd", a, b),
                  lambda a,b:Tensor.einsum("...id,...jd", a, b))
    _fp16_test_op([(2,3,4,5), (5,2,4)], lambda a,b:torch.einsum("i...j,ji...->...", a, b),
                  lambda a,b:Tensor.einsum("i...j,ji...->...", a, b))
    self.helper_test_exception([(2,3,4), (2,3,4)], lambda a,b:torch.einsum("...ik...,...jk->", a, b),
                               lambda a,b:Tensor.einsum("...ik...,...jk->", a, b), expected=(RuntimeError, IndexError))
    self.helper_test_exception([(2,3,4), (2,3,4)], lambda a,b:torch.einsum("i...j,ji...->...", a, b),
                               lambda a,b:Tensor.einsum("i...j,ji...->...", a, b), expected=RuntimeError)

  test_einsum_trace = _test_ops.TestOps.test_einsum_trace
  test_einsum_shape_check = _test_ops.TestOps.test_einsum_shape_check
  test_einsum_arity_check1 = _test_ops.TestOps.test_einsum_arity_check1
  test_einsum_arity_check2 = _test_ops.TestOps.test_einsum_arity_check2

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeOps(unittest.TestCase):
  """FP16 cumulative reductions on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_small_cumsum = _test_ops.TestOps.test_small_cumsum
  test_simple_cumsum = _test_ops.TestOps.test_simple_cumsum
  test_cumsum = _test_ops.TestOps.test_cumsum
  test_cumsum_zero_axis = _test_ops.TestOps.test_cumsum_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeProductOps(unittest.TestCase):
  """FP16 cumulative products on DPU EW."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_small_cumprod = _test_ops.TestOps.test_small_cumprod
  test_simple_cumprod = _test_ops.TestOps.test_simple_cumprod
  test_cumprod = _test_ops.TestOps.test_cumprod
  test_cumprod_zero_axis = _test_ops.TestOps.test_cumprod_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCumulativeExtremaOps(unittest.TestCase):
  """Unchanged FP16 cumulative-extrema methods with exact DPU-selected INT32 indices."""
  helper_test_exception = _test_ops.TestOps.helper_test_exception

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def _test_simple(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.cummax, Tensor.cummax) if kind == "max" else (torch.cummin, Tensor.cummin)
    for count in (512, 1022):
      np.random.seed(0)
      data = np.random.uniform(-2, 2, size=(count,)).astype(np.float16)
      expected = torch_fxn(torch.from_numpy(data), dim=0)
      if count == 512:
        values, indices = tinygrad_fxn(Tensor(data), axis=0)
        Tensor.realize(values, indices)
      else:
        values = tinygrad_fxn(Tensor(data), axis=0)[0].realize()
        indices = tinygrad_fxn(Tensor(data), axis=0)[1].realize()
      np.testing.assert_allclose(values.numpy(), expected.values.numpy(), **_FP16)
      np.testing.assert_equal(indices.numpy(), expected.indices.int().numpy())

  test_small_cummax = _test_ops.TestOps.test_small_cummax
  @slow_test
  def test_simple_cummax(self): self._test_simple("max")
  test_cummax = _test_ops.TestOps.test_cummax
  test_cummax_zero_axis = _test_ops.TestOps.test_cummax_zero_axis
  test_small_cummin = _test_ops.TestOps.test_small_cummin
  @slow_test
  def test_simple_cummin(self): self._test_simple("min")
  test_cummin = _test_ops.TestOps.test_cummin
  test_cummin_zero_axis = _test_ops.TestOps.test_cummin_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipArgExtremaOps(unittest.TestCase):
  """FP16 ArgMax/ArgMin selected entirely by DPU EW, including first-index ties."""

  def _test(self, kind:str, values:tuple[tuple[float, ...], ...]):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    for case in values:
      with self.subTest(kind=kind, values=case):
        _TEST_OPS_HELPER(None, lambda x: torch_fxn(x).int(), tinygrad_fxn, vals=[list(case)], forward_only=True)

  def test_argmax_first_tie(self): self._test("max", ((2.0, 2.0), (1.0, 2.0, 2.0)))

  def test_argmin_first_tie(self): self._test("min", ((2.0, 2.0), (3.0, 2.0, 2.0)))

  def _test_axes(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    for axis,keepdim in ((0, False), (1, False), (1, True)):
      with self.subTest(kind=kind, axis=axis, keepdim=keepdim):
        _TEST_OPS_HELPER([(10,20)], lambda x, axis=axis, keepdim=keepdim: torch_fxn(x, dim=axis, keepdim=keepdim).int(),
                         lambda x, axis=axis, keepdim=keepdim: tinygrad_fxn(x, axis=axis, keepdim=keepdim), forward_only=True)

  def test_argmax_axes(self): self._test_axes("max")

  def test_argmin_axes(self): self._test_axes("min")

  def _test_global(self, kind:str):
    torch_fxn, tinygrad_fxn = (torch.argmax, Tensor.argmax) if kind == "max" else (torch.argmin, Tensor.argmin)
    def run(shps, vals=None):
      _TEST_OPS_HELPER(shps, lambda x: torch_fxn(x).int(), tinygrad_fxn, vals=vals, forward_only=True)
    run([(10,20)])
    tied = np.zeros((10,20), dtype=np.float16)
    tied.flat[[37,99]] = 1.0 if kind == "max" else -1.0
    run(None, [tied.tolist()])

  def test_argmax_global(self): self._test_global("max")

  def test_argmin_global(self): self._test_global("min")

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortValueOps(unittest.TestCase):
  """FP16 stable sort values using static bitonic lane maps and DPU EW MIN/MAX."""

  def test_sort_values_trivial(self):
    for shape in ((0,), (0,5), (1,), (1,5)):
      with self.subTest(shape=shape):
        _fp16_test_op([shape], lambda x: x.sort(0).values, lambda x: x.sort(0)[0], forward_only=True)

  def test_sort_values_axes(self):
    for axis in (-1, 0, 1):
      for descending in (True, False):
        with self.subTest(axis=axis, descending=descending):
          _fp16_test_op([(8,8,6)], lambda x, axis=axis, descending=descending: x.sort(axis, descending).values,
                        lambda x, axis=axis, descending=descending: x.sort(axis, descending)[0], forward_only=True)

  def test_sort_values_repeated(self):
    values = np.array([0, 1] * 9, dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _fp16_test_op(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).values,
                      lambda x, descending=descending: x.sort(descending=descending)[0], vals=[values], forward_only=True)

  def test_sort_values_infinity(self):
    values = np.array([-np.inf, 2.0], dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _fp16_test_op(None, lambda x, descending=descending: x.sort(descending=descending).values,
                      lambda x, descending=descending: x.sort(descending=descending)[0], vals=[values], forward_only=True)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortIndexOps(unittest.TestCase):
  """Exact stable INT32 sort indices selected by FP16 value/count equality on DPU EW."""

  def _axis(self, axis:int, descending:bool):
    _TEST_OPS_HELPER([(8,8,6)], lambda x: x.sort(axis, descending).indices.int(),
                     lambda x: x.sort(axis, descending)[1], forward_only=True)

  def test_sort_indices_trivial(self):
    for shape in ((0,), (0,5), (1,), (1,5)):
      with self.subTest(shape=shape):
        _TEST_OPS_HELPER([shape], lambda x: x.sort(0).indices.int(), lambda x: x.sort(0)[1], forward_only=True)

  def test_sort_indices_last_descending(self): self._axis(-1, True)
  def test_sort_indices_last_ascending(self): self._axis(-1, False)
  def test_sort_indices_axis0_descending(self): self._axis(0, True)
  def test_sort_indices_axis0_ascending(self): self._axis(0, False)
  def test_sort_indices_axis1_ascending(self): self._axis(1, False)
  test_argsort = _test_ops.TestOps.test_argsort

  def test_sort_indices_repeated(self):
    values = np.array([0, 1] * 9, dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _TEST_OPS_HELPER(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).indices.int(),
                         lambda x, descending=descending: x.sort(descending=descending)[1], vals=[values], forward_only=True)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTopKOps(unittest.TestCase):
  """FP16 TopK values and stable INT32 indices composed from native sort and static slicing."""

  def _test(self, shape:tuple[int, ...], k:int, axis:int, largest:bool):
    _fp16_test_op([shape], lambda x: x.topk(k, axis, largest, True).values,
                  lambda x: x.topk(k, axis, largest, True)[0], forward_only=True)
    _TEST_OPS_HELPER([shape], lambda x: x.topk(k, axis, largest, True).indices.int(),
                     lambda x: x.topk(k, axis, largest, True)[1], forward_only=True)

  def test_topk_1d(self): self._test((8,), 3, -1, True)
  def test_topk_axis0_largest(self): self._test((5,5,4), 4, 0, True)
  def test_topk_axis1_smallest(self): self._test((5,5,4), 4, 1, False)

  def test_topk_repeated(self):
    values = np.array([1,1,0,1,0,1,0,0,1,0,0,0,1,0], dtype=np.float16)
    for largest,expected in ((True, [0,1,3]), (False, [2,4,6])):
      with self.subTest(largest=largest):
        result_values, result_indices = Tensor(values).topk(3, largest=largest)
        np.testing.assert_array_equal(result_values.numpy(), values[expected])
        np.testing.assert_array_equal(result_indices.numpy(), expected)
    with self.assertRaises((RuntimeError, ValueError)): Tensor(np.zeros(4, dtype=np.float16)).topk(5)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipElementwiseExtremaOps(unittest.TestCase):
  """FP16 portions of test_ops maximum/minimum; integer and boolean inputs are outside the NPU contract."""

  def _test(self, torch_fxn, tinygrad_fxn):
    _fp16_test_op([(45,65), (45,65)], torch_fxn, tinygrad_fxn)
    _fp16_test_op([(), ()], torch_fxn, tinygrad_fxn)
    _fp16_test_op(None, torch_fxn, tinygrad_fxn, vals=[[1., 0., 3., -4.], 3.])
    _fp16_test_op(None, torch_fxn, tinygrad_fxn, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    _fp16_test_op(None, torch_fxn, tinygrad_fxn,
                  vals=[[math.inf, -math.inf, math.nan, -0.0], [math.inf, math.inf, 1.0, 0.0]], forward_only=True)

  def test_maximum(self): self._test(torch.maximum, Tensor.maximum)
  def test_minimum(self): self._test(torch.minimum, Tensor.minimum)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSignOps(unittest.TestCase):
  """FP16 sign and softsign computed entirely by DPU EW stages."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_sign = _test_ops.TestOps.test_sign
  test_sign_exact = _test_ops.TestOps.test_sign_exact
  test_copysign = _test_ops.TestOps.test_copysign
  test_softsign = _test_ops.TestOps.test_softsign
  test_softsign_exact = _test_ops.TestOps.test_softsign_exact

  def test_copysign_exact(self):
    values = np.array([-1., -0., 0., 1., math.inf, -math.inf, math.nan], dtype=np.float16)
    magnitude, sign = np.repeat(values, len(values)), np.tile(values, len(values))
    expected = np.copysign(magnitude, sign)
    got = Tensor(magnitude).copysign(Tensor(sign)).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

  def test_sign_nonfinite(self):
    values = np.array([-math.inf, -1., -0., 0., 1., math.inf, math.nan], dtype=np.float16)
    before = Device["ROCKCHIP"].submit_count
    got = Tensor(values).sign().realize().numpy()
    expected = np.array([-1., -1., 0., 0., 1., 1., 0.], dtype=np.float16)
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 4)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCastOps(unittest.TestCase):
  """Direct FP16-to-bool conversion through native ABS and DPU positive-mask stages."""

  def test_cast_bool(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(3,3)], lambda x:x.bool(), forward_only=True)
    _fp16_test_op(None, lambda x:x.bool(), vals=[[-2., -0., 0., 1., math.inf, -math.inf, math.nan]], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 6)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBitcastOps(unittest.TestCase):
  """Exact FP16-pair to INT32 representation movement; the upstream odd-width FP16 shape is invalid."""

  def test_bitcast(self):
    values = np.array([0x0000, 0x8000, 0x3c00, 0xc000, 0x7c00, 0xfc00, 0x7e01, 0xfe01,
                       0x3555, 0xb555, 0x7bff, 0xfbff, 0x0001, 0x8001, 0x03ff, 0x83ff,
                       0x0400, 0x8400, 0x3c01, 0xbc01, 0x4000, 0xc000, 0x4200, 0xc200], dtype=np.uint16).view(np.float16).reshape(2,3,4)
    before = Device["ROCKCHIP"].submit_count
    np.testing.assert_array_equal(Tensor(values).bitcast(dtypes.int32).numpy(), values.view(np.int32))
    np.testing.assert_array_equal(Tensor(values).permute(1,0,2).bitcast(dtypes.int32).numpy(),
                                  values.transpose(1,0,2).view(np.int32))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 0)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipClassificationOps(unittest.TestCase):
  """FP16 IEEE predicates computed as DPU masks and packed through the typed bool-output ABI."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_isinf = _test_ops.TestOps.test_isinf
  test_isnan = _test_ops.TestOps.test_isnan
  test_isfinite = _test_ops.TestOps.test_isfinite

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipComparisonOps(unittest.TestCase):
  """FP16 portions of test_ops comparisons; integer and boolean inputs are outside the NPU contract."""

  def _test_cmp(self, fxn, reverse:bool=True):
    _fp16_test_op(None, fxn, fxn, forward_only=True, vals=[[0., 1., 2.], [2., 1., 0.]])
    for shapes in [[(3,4,5), (3,4,5)], [(3,4,5), (5,)], [(5,), (3,4,5)]]:
      _fp16_test_op(shapes, fxn, fxn, forward_only=True)
    _fp16_test_op(None, lambda x,y:fxn(x, 2), lambda x,y:fxn(x, 2), forward_only=True,
                  vals=[[0., 1., 2.], [2., 1., 0.]])
    if reverse:
      _fp16_test_op(None, lambda x,y:fxn(2, y), lambda x,y:fxn(2, y), forward_only=True,
                    vals=[[0., 1., 2.], [2., 1., 0.]])
    specials = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan]
    pairs = [(lhs, rhs) for lhs in specials for rhs in specials]
    _fp16_test_op(None, fxn, fxn, forward_only=True, vals=[[x[0] for x in pairs], [x[1] for x in pairs]])

  def test_cmp_eq(self): self._test_cmp(lambda x,y:x == y, reverse=False)
  def test_cmp_ne(self): self._test_cmp(lambda x,y:x != y, reverse=False)
  def test_cmp_gt(self): self._test_cmp(lambda x,y:x > y)
  def test_cmp_ge(self): self._test_cmp(lambda x,y:x >= y)
  def test_cmp_lt(self): self._test_cmp(lambda x,y:x < y)
  def test_cmp_le(self): self._test_cmp(lambda x,y:x <= y)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipLogicalPredicateOps(unittest.TestCase):
  """FP16 logical-not and scalar isclose compositions over native DPU comparison masks."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  def test_logical_not(self):
    _fp16_test_op(None, torch.logical_not, Tensor.logical_not,
                  vals=[[1., 2., 0., 0.5, -0.0, math.inf, -math.inf, math.nan]], forward_only=True)

  def test_and(self):
    _fp16_test_op(None, lambda x:(1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])

  def test_or(self):
    _fp16_test_op(None, lambda x:(x < -1) | (x > 1), forward_only=True,
                  vals=[[-math.inf, -2., -1., -0., 0., 1., 2., math.inf, math.nan]])

  def test_xor(self):
    _fp16_test_op(None, lambda x:(x < 0) ^ (x > 1), forward_only=True,
                  vals=[[-math.inf, -2., -1., -0., 0., 1., 2., math.inf, math.nan]])

  test_isclose_scalar = _test_ops.TestOps.test_isclose_scalar

  def test_isclose_edge_cases(self):
    values = (math.inf, -math.inf, math.nan, 0.0)
    lhs, rhs = [a for a in values for _ in values], [b for _ in values for b in values]
    before = Device["ROCKCHIP"].submit_count
    for equal_nan in (False, True):
      _fp16_test_op(None, lambda x,y,equal_nan=equal_nan:x.isclose(y, equal_nan=equal_nan),
                    vals=[lhs, rhs], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 104)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegralRoundingOps(unittest.TestCase):
  """Native FP16 floor/ceil plus DPU-composed truncation; round-to-even still requires a LUT."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_floor = _test_ops.TestOps.test_floor
  test_ceil = _test_ops.TestOps.test_ceil
  test_trunc = _test_ops.TestOps.test_trunc

  def test_all_fp16_encodings(self):
    values = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    before = Device["ROCKCHIP"].submit_count
    with np.errstate(invalid="ignore"):
      for name in ("floor", "ceil", "trunc"):
        np.testing.assert_equal(getattr(Tensor(values), name)().numpy(), getattr(np, name)(values))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 3)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipModuloOps(unittest.TestCase):
  """FP16 remainder compositions over DPU FDIV, FLOOR/TRUNC, MUL, and SUB."""

  def test_mod(self):
    a = [-4., 7., 5., 4., -7., 8., -9.]
    b = [2., -3., 8., -2., 3., 5., -5.]
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op(None, lambda x,y:x%y, Tensor.mod, forward_only=True, vals=[a, b])
    _fp16_test_op(None, lambda x,y:x%y, forward_only=True, vals=[a, b])
    _fp16_test_op(None, lambda x:x%2, forward_only=True, vals=[a])
    _fp16_test_op(None, lambda x:x%3, forward_only=True, vals=[a])
    _fp16_test_op(None, lambda x:x%3.5, lambda x:(x%3.5).clone(), forward_only=True, vals=[a])
    _fp16_test_op(None, lambda x:100%x, forward_only=True, vals=[a])
    _fp16_test_op(None, lambda x:100.5%x, lambda x:(100.5%x).clone(), forward_only=True, vals=[a])
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 7)

  def test_fmod(self):
    a = [-4., 7., 5., 4., -7., 8., -9.]
    b = [2., -3., 8., -2., 3., 5., -5.]
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op(None, lambda x,y:x.fmod(y), forward_only=True, vals=[a, b])
    _fp16_test_op(None, lambda x:x.fmod(2), forward_only=True, vals=[a])
    _fp16_test_op(None, lambda x:x.fmod(3.5), lambda x:x.fmod(3.5).clone(), forward_only=True, vals=[a])
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 3)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipDivisionRoundingOps(unittest.TestCase):
  """FP16 scalar-broadcast division with native DPU FLOOR/TRUNC epilogues."""

  def test_div_rounding_mode(self):
    numerator = [5., 6., 7., 0., -5., -6., -7.]
    before = Device["ROCKCHIP"].submit_count
    for denominator in (-10., -5., -3., -2., -1., 1., 2., 3., 5., 10.):
      for mode in (None, "trunc", "floor"):
        _fp16_test_op(None, lambda x,y,mode=mode:x.div(y, rounding_mode=mode), forward_only=True,
                      vals=[numerator, [denominator]])
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 30)

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBooleanReductionOps(unittest.TestCase):
  """ANY/ALL over FP16 inputs; external boolean tensors remain outside the DPU contract."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  @staticmethod
  def _check(name:str, values:np.ndarray, axis=None):
    _fp16_test_op(None, lambda x:getattr(x, name)(axis=axis), lambda x:getattr(x, name)(axis=axis),
                  vals=[values], forward_only=True)

  def test_any(self):
    self._check("any", np.array([0., -0., 0.], dtype=np.float16))
    self._check("any", np.array([0., math.nan, math.inf, -math.inf], dtype=np.float16))
    _fp16_test_op([()], lambda x:x.any(), forward_only=True)

  def test_any_axis(self):
    values = np.zeros((3,4,5,6), dtype=np.float16)
    values[0,0,0,0], values[1,1,2,3], values[2,3,4,5] = 1., math.inf, math.nan
    self._check("any", values, axis=(1,2))

  def test_all(self):
    self._check("all", np.array([1., -2., math.nan, math.inf, -math.inf], dtype=np.float16))
    self._check("all", np.array([1., 0., math.nan], dtype=np.float16))
    _fp16_test_op([()], lambda x:x.all(), forward_only=True)

  def test_all_axis(self):
    values = np.ones((3,4,5,6), dtype=np.float16)
    values[0,0,0,0], values[1,1,2,3] = 0., -0.
    self._check("all", values, axis=(1,2))

  test_any_zero_axis = _test_ops.TestOps.test_any_zero_axis
  test_all_zero_axis = _test_ops.TestOps.test_all_zero_axis

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipWhereOps(unittest.TestCase):
  """FP16 WHERE and masked arithmetic lowered to DPU comparison masks and selection."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_inf_where = _test_ops.TestOps.test_inf_where
  test_masked_fill = _test_ops.TestOps.test_masked_fill

  def test_where_permute(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(5,5)], lambda x: torch.where(x > .5, 4, 2).type(torch.int32).permute((1,0)),
                  lambda x: (x > .5).where(4, 2).clone().permute((1,0)), forward_only=True)
    values = [[math.nan, math.inf, -math.inf], [-0., 0., 1.]]
    _fp16_test_op(None, lambda x: torch.where(x > .5, 4, 2).type(torch.int32).permute((1,0)),
                  lambda x: (x > .5).where(4, 2).clone().permute((1,0)), vals=[values], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 24)

  def test_where_tensor(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x > .1, a, b), lambda x,a,b: (x > .1).where(a, b),
                  forward_only=True)

  def test_where_scalar(self):
    _fp16_test_op([(2,3,4,5)], lambda x: torch.where(x < -.2, 3.0, -2.0), lambda x: (x < -.2).where(3.0, -2.0),
                  forward_only=True)

  def test_where_broadcast(self):
    _fp16_test_op([(2,3,4,5), (5,), (1,3,1,1)], lambda x,a,b: torch.where(x > 0, a, b),
                  lambda x,a,b: (x > 0).where(a, b), forward_only=True)

  def test_where_cmpne_exact(self):
    values = [[-2., -1., 0., 1., 2.], [-2., 0., 0., 0., 2.], [4., 3., 2., 1., 0.], [-4., -3., -2., -1., 0.]]
    _fp16_test_op(None, lambda x,y,a,b: torch.where(x != y, a, b), lambda x,y,a,b: (x != y).where(a, b), vals=values,
                  forward_only=True)

  def test_where_boolean_composition(self):
    _fp16_test_op([(100,)], lambda x: torch.where((x > -.5) & (x < .5), x*2, x-1),
                  lambda x: ((x > -.5) & (x < .5)).where(x*2, x-1), forward_only=True)
    _fp16_test_op([(100,)], lambda x: torch.where((x < -.5) | (x > .5), x+2, x-2),
                  lambda x: ((x < -.5) | (x > .5)).where(x+2, x-2), forward_only=True)

  def test_where_nested(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x < -.5, a, torch.where(x > .5, b, x)),
                  lambda x,a,b: (x < -.5).where(a, (x > .5).where(b, x)), forward_only=True)

  def test_masked_fill_finite(self):
    _fp16_test_op([(32,10)], lambda x: x.masked_fill(x > .1, -3.25), lambda x: x.masked_fill(x > .1, -3.25), forward_only=True)

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
class TestRockchipTriangularOps(unittest.TestCase):
  """Numeric FP16 portions of test_ops tril/triu; external boolean inputs are unsupported."""

  def _test(self, name:str):
    cases = (((3,3), 0), ((3,3), 1), ((3,3), 2), ((3,3), -1), ((3,3), -2),
             ((4,5), 4), ((4,5), 5), ((4,5), 6), ((4,5), -4), ((4,5), -5), ((4,5), -6),
             ((5,3,3), 0), ((5,0,3), 0), ((5,3,3), 1))
    for shape, diagonal in cases:
      with self.subTest(shape=shape, diagonal=diagonal):
        _fp16_test_op([shape], lambda x: getattr(x, name)(diagonal), lambda x: getattr(x, name)(diagonal))

  def test_tril(self): self._test("tril")
  def test_triu(self): self._test("triu")

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
  test_padding_add = _test_ops.TestOps.test_padding_add

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

  def test_max_nan(self):
    before = Device["ROCKCHIP"].submit_count
    for values in ([1.0, math.nan], [math.nan, 1.0]):
      self.assertTrue(math.isnan(Tensor(np.array(values, dtype=np.float16)).max().item()))
    values = np.array([[1.0, math.nan], [2.0, 3.0]], dtype=np.float16)
    for reduction,finite in ((Tensor.max, 3.0), (Tensor.min, 2.0)):
      result = reduction(Tensor(values), axis=1).numpy()
      self.assertTrue(math.isnan(result[0]))
      self.assertEqual(result[1], finite)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 4)

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
  test_var = _test_ops.TestOps.test_var
  test_var_axis = _test_ops.TestOps.test_var_axis
  test_var_zero_in_axis = _test_ops.TestOps.test_var_zero_in_axis
  test_var_one_in_axis = _test_ops.TestOps.test_var_one_in_axis
  test_var_keepdim = _test_ops.TestOps.test_var_keepdim

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
  test_softsign = _test_ops.TestOps.test_softsign
  test_softsign_exact = _test_ops.TestOps.test_softsign_exact

if __name__ == "__main__":
  unittest.main()
