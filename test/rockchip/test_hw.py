# PR 1 hardware numerical tests: one per compute family (DPU, CNA+CORE, PPU).
# These tests require an RK3588 NPU and /dev/dri/card1.
import os, unittest, numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import to_mv

_NPU_AVAILABLE = os.path.exists("/dev/dri/card1")

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestDPU(unittest.TestCase):
  def test_dpu_add(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float16)
    b_np = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_repeated_invocation(self):
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c1 = (a + b).realize().numpy()
    c2 = (a + b).realize().numpy()
    np.testing.assert_allclose(c1, c2, rtol=1e-3, atol=1e-3)

  def test_dpu_inplace_add(self):
    # a.assign(a+b).realize() — in-place ADD where output buffer == input buffer A
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    a.assign(a + b).realize()
    np.testing.assert_allclose(a.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_sub(self):
    a_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    b_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a - b).realize()
    np.testing.assert_allclose(c.numpy(), a_np - b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_mul(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a * b).realize()
    np.testing.assert_allclose(c.numpy(), a_np * b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_max(self):
    a_np = np.array([[1,5,3,4],[8,6,2,7]], dtype=np.float16)
    b_np = np.array([[4,2,6,8],[3,9,1,5]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = a.maximum(b).realize()
    np.testing.assert_allclose(c.numpy(), np.maximum(a_np, b_np), rtol=1e-3, atol=1e-3)

  def test_dpu_copy(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a + 0).realize()  # a+0 lowers to copy (STORE(INDEX)) — NPU DMA pass-through
    np.testing.assert_allclose(c.numpy(), a_np, rtol=1e-3, atol=1e-3)

  def test_dpu_fill_zeros(self):
    # Fill via DPU ADD(zero, const) — zero buffer is prep, DPU does the fill
    c = Tensor.zeros(2,4, dtype=dtypes.half, device="ROCKCHIP").realize()
    np.testing.assert_allclose(c.numpy(), np.zeros((2,4), dtype=np.float16), rtol=1e-3, atol=1e-3)

  def test_dpu_fill_ones(self):
    c = Tensor.ones(2,4, dtype=dtypes.half, device="ROCKCHIP").realize()
    np.testing.assert_allclose(c.numpy(), np.ones((2,4), dtype=np.float16), rtol=1e-3, atol=1e-3)

  def test_dpu_fill_full(self):
    c = Tensor.full((2,4), 3.5, dtype=dtypes.half, device="ROCKCHIP").realize()
    np.testing.assert_allclose(c.numpy(), np.full((2,4), 3.5, dtype=np.float16), rtol=1e-3, atol=1e-3)

  def test_dpu_typed_fills(self):
    for dtype, np_dtype, value in ((dtypes.float, np.float32, 3.5), (dtypes.int, np.int32, 4),
                                   (dtypes.bool, np.bool_, True), (dtypes.uint8, np.uint8, 7)):
      c = Tensor.full((2,4), value, dtype=dtype, device="ROCKCHIP").realize()
      np.testing.assert_array_equal(c.numpy(), np.full((2,4), value, dtype=np_dtype))

  def test_dpu_scalar_add(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a + 1).realize()
    np.testing.assert_allclose(c.numpy(), a_np + 1, rtol=1e-3, atol=1e-3)

  def test_dpu_scalar_mul(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a * 2).realize()
    np.testing.assert_allclose(c.numpy(), a_np * 2, rtol=1e-3, atol=1e-3)

  def test_dpu_neg(self):
    a_np = np.array([[1,-2,3,-4],[5,-6,7,-8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (-a).realize()
    np.testing.assert_allclose(c.numpy(), -a_np, rtol=1e-3, atol=1e-3)

  def test_dpu_scalar_max(self):
    a_np = np.array([[1,-2,3,-4],[5,-6,7,-8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.maximum(0).realize()  # ReLU
    np.testing.assert_allclose(c.numpy(), np.maximum(a_np, 0), rtol=1e-3, atol=1e-3)

  def test_dpu_relu(self):
    # relu(x) = MAX(x, 0) via WHERE→MAX rewrite
    a_np = np.array([[-1,-2,3,4],[5,-6,-7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.relu().realize()
    np.testing.assert_allclose(c.numpy(), np.maximum(a_np, 0), rtol=1e-3, atol=1e-3)

  def test_dpu_abs(self):
    # Preserve the sign-WHERE pattern for the stable staged max(x, -x) path.
    a_np = np.array([[-8,-2,-0.0,4],[5,-6,-7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.abs().realize()
    np.testing.assert_array_equal(c.numpy(), np.abs(a_np))

  def test_dpu_sign(self):
    a_np = np.array([-np.inf,-8,-0.0,0,5,np.inf], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    np.testing.assert_array_equal(a.sign().realize().numpy(), np.sign(a_np))

  def test_dpu_rounding(self):
    a_np = np.array([-8.75,-2,-0.5,-0.0,0,0.5,3.75,8], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    for op, expected in ((a.trunc, np.trunc(a_np)), (a.floor, np.floor(a_np)), (a.ceil, np.ceil(a_np)),
                         (a.round, np.round(a_np))):
      np.testing.assert_array_equal(op().realize().numpy(), expected)

  def test_dpu_silu_staged(self):
    a_np = np.concatenate((np.linspace(-2, 2, 257, dtype=np.float16),
                           np.array([-1.9599609375, -1.9541015625], dtype=np.float16)))
    a = Tensor(a_np, device="ROCKCHIP").realize()
    expected = (a_np.astype(np.float32) / (1.0 + np.exp(-a_np.astype(np.float32)))).astype(np.float16)
    np.testing.assert_allclose(a.silu().realize().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_dpu_exp2_special_values(self):
    a_np = np.array([np.inf, -np.inf, np.nan, -2, 0, 2], dtype=np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").exp2().realize().numpy()
    np.testing.assert_allclose(actual, np.exp2(a_np), rtol=1e-3, atol=1e-6)

  def test_dpu_exp_two_lut(self):
    a_np = np.concatenate((np.linspace(-2, 2, 513, dtype=np.float16),
                           np.array([np.inf, -np.inf, np.nan], dtype=np.float16)))
    actual = Tensor(a_np, device="ROCKCHIP").exp().realize().numpy()
    with np.errstate(over="ignore", invalid="ignore"):
      expected = np.exp(a_np.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_dpu_log2_special_values(self):
    a_np = np.array([np.inf, -np.inf, np.nan, -2, -0.0, 0, 0.25, 1, 4], dtype=np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").log2().realize().numpy()
    np.testing.assert_allclose(actual, np.log2(a_np), rtol=1e-3, atol=2e-4)

  def test_dpu_sigmoid_extreme_values(self):
    a_np = np.array([np.inf, -np.inf, np.nan, -400, -8, -2, 0, 2, 8, 400], dtype=np.float16)
    with np.errstate(over="ignore"):
      expected = (1 / (1 + np.exp(-a_np.astype(np.float32)))).astype(np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").sigmoid().realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_dpu_sqrt_special_values(self):
    a_np = np.array([np.inf, -np.inf, np.nan, -2, -0.0, 0, 0.25, 4], dtype=np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").sqrt().realize().numpy()
    np.testing.assert_allclose(actual, np.sqrt(a_np), rtol=1e-3, atol=1e-6)

  def test_dpu_rsqrt_special_values(self):
    # DPU FDIV does not preserve the sign of a zero denominator, so -0 is
    # tracked separately from the TestOps contract covered here.
    a_np = np.array([np.inf, -np.inf, np.nan, -2, 0, 0.25, 4], dtype=np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").rsqrt().realize().numpy()
    np.testing.assert_allclose(actual, 1 / np.sqrt(a_np), rtol=1e-3, atol=1e-6)

  def test_dpu_comparison_outputs(self):
    a_np = np.array([-2,-0.0,0,1,3], dtype=np.float16)
    b_np = np.array([-1,0,0,2,2], dtype=np.float16)
    a, b = Tensor(a_np, device="ROCKCHIP").realize(), Tensor(b_np, device="ROCKCHIP").realize()
    for op, expected in ((lambda: a < b, a_np < b_np), (lambda: a == b, a_np == b_np),
                         (lambda: a != b, a_np != b_np), (lambda: a >= b, a_np >= b_np)):
      np.testing.assert_array_equal(op().realize().numpy(), expected)

  def test_dpu_where_infinities(self):
    a_np = np.array([-np.inf, -1, 0, 0.5, 2, np.inf], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    np.testing.assert_array_equal((a < 0).where(a, 1).realize().numpy(), np.where(a_np < 0, a_np, 1))
    b_np = np.array([-2, -1, 0, 0.5, 1, 2], dtype=np.float16)
    b = Tensor(b_np, device="ROCKCHIP").realize()
    np.testing.assert_array_equal((b > 0).where(-np.inf, b).realize().numpy(), np.where(b_np > 0, -np.inf, b_np))
    np.testing.assert_array_equal((b > 0).where(b, np.inf).realize().numpy(), np.where(b_np > 0, b_np, np.inf))

  def test_dpu_infinity_division_sign(self):
    # RK3588 FDIV keeps the numerator's infinity sign but drops the sign of a
    # nonzero denominator. Signed-zero division remains a separate limitation.
    a_np = np.array([-2, -1, -0.5, 0.5, 1, 2], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    for numerator in (np.inf, -np.inf, np.nan):
      with np.errstate(invalid="ignore"):
        expected = numerator / a_np
      np.testing.assert_array_equal((numerator / a).realize().numpy(), expected)

  def test_dpu_hardsigmoid_saturation(self):
    a_np = np.array([-400, -3, -2, 0, 2, 3, 381.25, 382, 383.5, 400], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    expected = np.clip(a_np.astype(np.float32) / 6 + 0.5, 0, 1).astype(np.float16)
    actual = a.hardsigmoid().realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(actual[[0, 1, 5, 6, 7, 8, 9]], expected[[0, 1, 5, 6, 7, 8, 9]])

  def test_dpu_hardswish_two_lut(self):
    a_np = np.concatenate((np.linspace(-2, 2, 1025, dtype=np.float16),
                           np.array([-400, -4, -3, -0.125, -0.0, 0.003313, 0.117, 0.125, 3, 4, 400], dtype=np.float16)))
    a = Tensor(a_np, device="ROCKCHIP").realize()
    expected = (a_np.astype(np.float32) * np.clip(a_np.astype(np.float32)+3, 0, 6) / 6).astype(np.float16)
    actual = a.hardswish().realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(actual[a_np == 0], expected[a_np == 0])

  def test_dpu_tanh_extreme_saturation(self):
    a_np = np.array([-np.inf, -300, -5, 5, 300, np.inf, np.nan], dtype=np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").tanh().realize().numpy()
    np.testing.assert_allclose(actual, np.tanh(a_np), rtol=1e-3, atol=1e-6)

  def test_dpu_tanh_direct_lut(self):
    a_np = np.concatenate((np.linspace(-4, 4, 2049, dtype=np.float16),
                           np.array([-np.inf, -300, -4.01, -0.0, 0.0, 4.01, 300, np.inf, np.nan], dtype=np.float16)))
    actual = Tensor(a_np, device="ROCKCHIP").tanh().realize().numpy()
    np.testing.assert_allclose(actual, np.tanh(a_np), rtol=1e-3, atol=1e-6)

  def test_dpu_quick_gelu_two_lut(self):
    a_np = np.array([-400, -300, -10.5, -2, -1.9, -1.6, -1.5, -1.4, -1,
                     -0.918457, -0.534668, -0.403809, -0.331787, -0.161, -0.16, -0.159,
                     -0.0, 0.0, 0.159, 0.16, 0.161, 0.19165, 2, 5.5, 300, 400], dtype=np.float16)
    with np.errstate(over="ignore"):
      scaled = (a_np.astype(np.float32)*np.float32(1.702)).astype(np.float16)
      sigmoid = (1/(1+np.exp(-scaled.astype(np.float32)))).astype(np.float16)
      expected = (a_np*sigmoid).astype(np.float16)
    actual = Tensor(a_np, device="ROCKCHIP").quick_gelu().realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_dpu_celu_two_lut(self):
    a_np = np.array([-2, -1.999, -1.5, -1, -0.5, -0.127, -0.1255, -0.125, -0.1249,
                     -0.1, -0.01, -0.002153, -0.0, 0.0, 0.125, 1, 1.999], dtype=np.float16)
    for alpha in range(1, 5):
      expected = np.where(a_np > 0, a_np, alpha*np.expm1(a_np.astype(np.float32)/alpha)).astype(np.float16)
      actual = Tensor(a_np, device="ROCKCHIP").celu(alpha).realize().numpy()
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6, err_msg=f"alpha={alpha}")

  def test_dpu_direct_casts(self):
    half_np = np.array([-2.5,0,1.75,255], dtype=np.float16)
    half = Tensor(half_np, device="ROCKCHIP").realize()
    np.testing.assert_array_equal(half.cast(dtypes.float).realize().numpy(), half_np.astype(np.float32))
    np.testing.assert_array_equal(half.cast(dtypes.int).realize().numpy(), half_np.astype(np.int32))
    np.testing.assert_array_equal(half.cast(dtypes.bool).realize().numpy(), half_np.astype(np.bool_))
    np.testing.assert_array_equal(half.cast(dtypes.uint8).realize().numpy(), half_np.astype(np.uint8))
    int_np = np.array([-2,0,1,255], dtype=np.int32)
    bool_np = np.array([True,False,True,False], dtype=np.bool_)
    np.testing.assert_array_equal(Tensor(int_np, device="ROCKCHIP").cast(dtypes.float).realize().numpy(), int_np.astype(np.float32))
    np.testing.assert_array_equal(Tensor(bool_np, device="ROCKCHIP").cast(dtypes.float).realize().numpy(), bool_np.astype(np.float32))

  def test_dpu_typed_minimum_boundaries(self):
    int_np = np.array([-1234,0,1234,np.iinfo(np.int32).max,np.iinfo(np.int32).min], dtype=np.int32)
    ints = Tensor(int_np, device="ROCKCHIP").realize()
    int_min = Tensor(np.iinfo(np.int32).min, dtype=dtypes.int, device="ROCKCHIP").realize()
    np.testing.assert_array_equal(ints.minimum(int_min).realize().numpy(), np.minimum(int_np, np.iinfo(np.int32).min))
    bool_a, bool_b = np.array([True,False,False]), np.array([True,True,False])
    a = Tensor(bool_a, device="ROCKCHIP").realize()
    b = Tensor(bool_b, device="ROCKCHIP").realize()
    np.testing.assert_array_equal(a.minimum(b).realize().numpy(), np.minimum(bool_a, bool_b))
    np.testing.assert_array_equal(a.maximum(b).realize().numpy(), np.maximum(bool_a, bool_b))
    expected = np.minimum(int_np.astype(np.float16), np.float16(1.2))
    np.testing.assert_array_equal(ints.minimum(Tensor(1.2, dtype=dtypes.half, device="ROCKCHIP")).realize().numpy(), expected)

  def test_dpu_2d_add(self):
    # 2D row-major contiguous element-wise add
    a_np = np.random.randn(8,16).astype(np.float16)
    b_np = np.random.randn(8,16).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_3d_add(self):
    # 3D contiguous element-wise add
    a_np = np.random.randn(4,8,8).astype(np.float16)
    b_np = np.random.randn(4,8,8).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_cast_wrapping_ew(self):
    # CAST(half→half) wrapping EW op — no-op cast should be stripped
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[9,10,11,12],[13,14,15,16]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).cast(dtypes.half).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_relu_cast_uint8(self):
    # Keep the output larger than one page: an int32-sized write into this uint8
    # allocation caused the old conversion path to overrun and segfault.
    a_np = np.linspace(-16, 255, 4097, dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.relu().cast(dtypes.uint8).realize()
    np.testing.assert_array_equal(c.numpy(), np.maximum(a_np, 0).astype(np.uint8))

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestCMAC(unittest.TestCase):
  def test_cmac_matmul(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float16)
    b_np = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-3, atol=1e-3)

  def test_cmac_sum(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.sum(axis=1).realize()
    np.testing.assert_allclose(c.numpy(), a_np.sum(axis=1), rtol=1e-2, atol=1e-2)

  def test_cmac_sum_axis0(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.sum(axis=0).realize()
    np.testing.assert_allclose(c.numpy(), a_np.sum(axis=0), rtol=1e-2, atol=1e-2)

  def test_cmac_sum_full(self):
    # Full reduction: sum all elements to scalar via ones-vector GEMM (M=1, N=1)
    a_np = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.sum().realize()
    np.testing.assert_allclose(c.numpy(), a_np.sum(), rtol=1e-2, atol=1e-2)

  def test_cmac_scaled_sum_full(self):
    # Scaled full sum: (a * 2).sum() → ones @ (a*2)
    a_np = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a * 2).sum().realize()
    np.testing.assert_allclose(c.numpy(), (a_np * 2).sum(), rtol=1e-2, atol=1e-2)

  def test_cmac_scaled_sum_axis0(self):
    # Scaled sum over axis=0: (a * 3).sum(axis=0)
    a_np = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1],[2,2,2,2]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a * 3).sum(axis=0).realize()
    np.testing.assert_allclose(c.numpy(), (a_np * 3).sum(axis=0), rtol=1e-2, atol=1e-2)

  def test_cmac_matmul_non_identity(self):
    a_np = np.array([[1,2],[3,4]], dtype=np.float16)
    b_np = np.array([[5,6],[7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_gemv_vector_a(self):
    # GEMV: (K,) @ (K,N) → (N,) — vector is A, M=1
    a_np = np.array([1,2,3,4], dtype=np.float16)
    b_np = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_gemv_vector_b(self):
    # GEMV: (M,K) @ (K,) → (M,) — vector is B, N=1
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([1,2,3,4], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_same_buffer(self):
    a_np = np.array([[1,2],[3,4]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a @ a).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ a_np, rtol=1e-2, atol=1e-1)

  def test_cmac_subnormal_output(self):
    # 4x4 matmul where each element = 4 * (2^-12)^2 = 4 * 2^-24 = 2^-22
    # 2^-22 is a subnormal in FP16 (min normal is 2^-14, min subnormal is 2^-24)
    # The old FP32→FP16 conversion returned 0 for subnormals; this verifies the fix
    a_np = np.ones((4,4), dtype=np.float16) * np.float16(2**-12)
    b_np = np.ones((4,4), dtype=np.float16) * np.float16(2**-12)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    expected = a_np @ b_np
    np.testing.assert_allclose(c.numpy(), expected, rtol=0, atol=0)
    # verify the result is actually subnormal (nonzero but below min normal)
    self.assertGreater(c.numpy().max(), 0, "subnormal result should not be zero")

  def test_cmac_accuracy(self):
    # Broader CMAC accuracy: random 8x8 matmul with values spanning normal range
    np.random.seed(42)
    a_np = np.random.randn(8,8).astype(np.float16)
    b_np = np.random.randn(8,8).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_1x1_conv(self):
    # 1x1 conv = pointwise GEMM (transposed A/B pattern)
    np.random.seed(42)
    C, H, W = 4, 3, 3
    x_np = np.random.randn(1, C, H, W).astype(np.float16)
    w_np = np.random.randn(C, C, 1, 1).astype(np.float16)
    x = Tensor(x_np, device="ROCKCHIP").realize()
    w = Tensor(w_np, device="ROCKCHIP").realize()
    c = x.conv2d(w).realize()
    expected = np.zeros((1, C, H, W), dtype=np.float16)
    for co in range(C):
      for oy in range(H):
        for ox in range(W):
          expected[0, co, oy, ox] = np.sum(x_np[0, :, oy, ox].astype(np.float32) * w_np[co, :, 0, 0].astype(np.float32))
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-2, atol=1e-1)

  def test_cmac_fp32_to_fp16_rounding(self):
    # Judge's test case: 0.5180664 * 0.5258789 should give FP16 bits 0x345c, not 0x345b
    # The old conversion truncated (mt >> 13) without round-to-nearest-even
    # Use a sparse 4x4 matmul so only one product contributes to result[0,0]
    a_np = np.zeros((4,4), dtype=np.float16)
    b_np = np.zeros((4,4), dtype=np.float16)
    a_np[0,0] = np.float16(0.5180664)
    b_np[0,0] = np.float16(0.5258789)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    result_bits = int(c.numpy().view(np.uint16)[0, 0])
    expected_bits = int(np.float16(np.float32(0.5180664) * np.float32(0.5258789)).view(np.uint16))
    self.assertEqual(result_bits, expected_bits,
                     f"rounding: got 0x{result_bits:04x}, expected 0x{expected_bits:04x}")

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestPPU(unittest.TestCase):
  def test_ppu_globalmax(self):
    a_np = np.arange(1, 33, dtype=np.float16).reshape(2, 2, 8)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.max(axis=(0,1)).realize()
    np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=(0,1)), rtol=1e-3, atol=1e-3)

  def test_ppu_globalmax_axis0(self):
    a_np = np.array([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1],[9,10,11,12,13,14,15,16],[16,15,14,13,12,11,10,9]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.max(axis=0).realize()
    np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=0), rtol=1e-3, atol=1e-3)

  def test_ppu_globalmax_flexible_channels(self):
    np.random.seed(123)
    for shape in [(8,4),(4,4),(8,8),(8,2),(4,2),(8,6),(4,6),(8,3),(8,1),(4,1),(12,4),(16,4),(6,8),(64,8)]:
      a_np = np.random.randn(*shape).astype(np.float16)
      a = Tensor(a_np, device="ROCKCHIP").realize()
      c = a.max(axis=0).realize()
      np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=0), rtol=1e-3, atol=1e-3,
                                 err_msg=f"shape {shape}")

  def test_ppu_globalmax_prime_k(self):
    # K=5 (prime ≤ 16) uses in_h=1 fallback — verified on hardware
    np.random.seed(42)
    for K in [5, 7, 11, 13]:
      a_np = np.random.uniform(-2, 2, (1, K, 8)).astype(np.float16)
      a = Tensor(a_np, device="ROCKCHIP").realize()
      c = a.max(axis=(0,1)).realize()
      np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=(0,1)), rtol=1e-2, atol=1e-2,
                                 err_msg=f"K={K}")

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestCounters(unittest.TestCase):
  def test_all_units_submit(self):
    from tinygrad.device import Device
    dev = Device["ROCKCHIP"]
    dev.submitted_masks.clear()
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    dpu_result = (a + b).realize().numpy()
    np.testing.assert_allclose(dpu_result, a_np + b_np, rtol=1e-3, atol=1e-3)
    cmac_result = (a @ b).realize().numpy()
    np.testing.assert_allclose(cmac_result, a_np @ b_np, rtol=1e-2, atol=1e-1)
    a8_np = np.random.randn(4,8).astype(np.float16)
    a8 = Tensor(a8_np, device="ROCKCHIP").realize()
    ppu_result = a8.max(axis=0).realize().numpy()
    np.testing.assert_allclose(ppu_result, np.max(a8_np, axis=0), rtol=1e-3, atol=1e-3)
    self.assertIn(0x18, dev.submitted_masks)
    self.assertIn(0xd, dev.submitted_masks)
    self.assertIn(0x60, dev.submitted_masks)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestGuardBuffer(unittest.TestCase):
  def test_small_buffer_canary(self):
    # 12 elements = 24 bytes; DPU rounds up to 2 lanes of 8 = 32 bytes written
    a_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float16)
    b_np = np.array([12,11,10,9,8,7,6,5,4,3,2,1], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    expected = a_np + b_np
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-3, atol=1e-3)
    out_buf = c._buffer().get_buf("ROCKCHIP")
    page_mv = to_mv(out_buf.va_addr, 4096)
    canary = b'\xDE\xAD' * ((4096 - 32) // 2)
    page_mv[32:4096] = canary
    page_mv[0:32] = b'\x00' * 32
    c.assign(a + b).realize()
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-3, atol=1e-3)
    self.assertEqual(bytes(page_mv[32:4096]), canary)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestJit(unittest.TestCase):
  def test_two_kernel_jit(self):
    from tinygrad import TinyJit
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    @TinyJit
    def f(x, y):
      return (x + y).realize()
    c1 = f(a, b).numpy()
    c2 = f(a, b).numpy()
    np.testing.assert_allclose(c1, c2, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
  unittest.main()
