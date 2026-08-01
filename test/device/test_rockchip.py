import os, unittest
import numpy as np
from tinygrad import Tensor, dtypes

@unittest.skipUnless(os.path.exists("/dev/dri/card1"), "no RK3588 NPU")
class TestRockchip(unittest.TestCase):
  def test_dpu_binary_and_multistage(self):
    rng = np.random.default_rng(1)
    values = [rng.uniform(-2, 2, 16).astype(np.float16) for _ in range(4)]
    a, b, c, d = (Tensor(x, device="ROCKCHIP").realize() for x in values)
    for out, expected in ((a+b, values[0]+values[1]), (a*b, values[0]*values[1]),
                          (((a+b)*c)+d, ((values[0]+values[1])*values[2])+values[3])):
      np.testing.assert_allclose(out.realize().numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_dpu_scalar_and_fill(self):
    data = np.linspace(-2, 2, 16, dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_allclose((x*2).realize().numpy(), data*2, rtol=1e-3, atol=1e-3)
    np.testing.assert_equal(Tensor.full((16,), 3.5, dtype=dtypes.half, device="ROCKCHIP").realize().numpy(),
                            np.full(16, 3.5, np.float16))
    np.testing.assert_equal(Tensor.ones((), dtype=dtypes.half, device="ROCKCHIP").realize().numpy(), np.ones((), np.float16))

  def test_ordered_where_extrema(self):
    data = np.linspace(-2, 2, 16, dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_equal(x.relu().realize().numpy(), np.maximum(data, 0))
    np.testing.assert_equal(x.clip(-1, 1).realize().numpy(), np.clip(data, -1, 1))

  def test_generic_where_mask(self):
    lhs, rhs = np.linspace(-2, 2, 16, dtype=np.float16), np.linspace(3, 4, 16, dtype=np.float16)
    x, y = Tensor(lhs, device="ROCKCHIP").realize(), Tensor(rhs, device="ROCKCHIP").realize()
    np.testing.assert_equal((x<0).where(x, y).realize().numpy(), np.where(lhs<0, lhs, rhs))

  def test_reciprocal_and_division(self):
    lhs, rhs = np.linspace(1, 2, 16, dtype=np.float16), np.linspace(2, 4, 16, dtype=np.float16)
    x, y = Tensor(lhs, device="ROCKCHIP").realize(), Tensor(rhs, device="ROCKCHIP").realize()
    np.testing.assert_allclose(x.reciprocal().realize().numpy(), 1/lhs, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose((x/y).realize().numpy(), lhs/rhs, rtol=2e-3, atol=2e-3)

  def test_abs_specials(self):
    data = np.array([-2, -0., 0., 2., np.inf, -np.inf, np.nan, -np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(data, device="ROCKCHIP").abs().realize().numpy(), np.abs(data))

  def test_direct_fp16_contract(self):
    rng = np.random.default_rng(2)
    a_np, packed_b_np = rng.uniform(-1,1,(1,32)).astype(np.float16), rng.uniform(-1,1,(8,32)).astype(np.float16)
    a, packed_b = Tensor(a_np, device="ROCKCHIP").realize(), Tensor(packed_b_np, device="ROCKCHIP").realize()
    np.testing.assert_allclose((a@packed_b.T).realize().numpy(), a_np@packed_b_np.T, rtol=5e-3, atol=5e-3)

  def test_fp16_row_sum_contract(self):
    data = np.arange(8*32, dtype=np.float16).reshape(8,32) / 128
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").sum(axis=1).realize().numpy(), data.sum(axis=1), rtol=5e-3, atol=5e-3)

  def test_explicit_hwc_global_max(self):
    data = np.arange(64, dtype=np.float16).reshape(8,8)
    np.testing.assert_equal(Tensor(data, device="ROCKCHIP").max(axis=0).realize().numpy(), data.max(axis=0))

  def test_generated_exp2_lut(self):
    for count in (1, 128, 129, 2925):
      with self.subTest(count=count):
        data = np.linspace(-2, 2, count, dtype=np.float16)
        actual = Tensor(data, device="ROCKCHIP").realize().exp2().realize().numpy()
        np.testing.assert_allclose(actual, np.exp2(data), rtol=5e-3, atol=5e-3)
    special = np.array([np.inf, -np.inf, np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").exp2().numpy(), np.array([np.inf, 0, np.nan], dtype=np.float16))

  def test_two_level_hardswish_lut(self):
    data = np.linspace(-4, 4, 1025, dtype=np.float16)
    expected = (data.astype(np.float32)*np.clip(data.astype(np.float32)+3, 0, 6)/6).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").hardswish().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_two_level_tanh_lut(self):
    data = np.linspace(-8, 8, 2049, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").tanh().numpy(), np.tanh(data.astype(np.float32)).astype(np.float16),
                               rtol=1e-3, atol=1e-6)

  def test_two_level_sigmoid_lut(self):
    data = np.linspace(-2, 2, 2049, dtype=np.float16)
    expected = (1/(1+np.exp(-data.astype(np.float32)))).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").sigmoid().numpy(), expected, rtol=1e-3, atol=1e-6)

if __name__ == "__main__": unittest.main()
