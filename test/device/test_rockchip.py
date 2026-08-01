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
                          (a.maximum(b), np.maximum(values[0], values[1])), (a/b, values[0]/values[1]),
                          (((a+b)*c)+d, ((values[0]+values[1])*values[2])+values[3])):
      np.testing.assert_allclose(out.realize().numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_dpu_scalar_and_fill(self):
    data = np.linspace(-2, 2, 16, dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_allclose((x*2).realize().numpy(), data*2, rtol=1e-3, atol=1e-3)
    np.testing.assert_equal(Tensor.full((16,), 3.5, dtype=dtypes.half, device="ROCKCHIP").realize().numpy(), np.full(16, 3.5, np.float16))
    np.testing.assert_equal(Tensor.ones((), dtype=dtypes.half, device="ROCKCHIP").realize().numpy(), np.ones((), np.float16))
    np.testing.assert_equal(Tensor.full((2925,), 4, dtype=dtypes.int, device="ROCKCHIP").realize().numpy(), np.full(2925, 4, np.int32))
    np.testing.assert_equal(Tensor.full((6,), 4, dtype=dtypes.float, device="ROCKCHIP").realize().numpy(), np.full(6, 4, np.float32))

  def test_where_uses_native_mask(self):
    a = np.linspace(-2, 2, 16, dtype=np.float16)
    b = np.linspace(3, 6, 16, dtype=np.float16)
    ta, tb = Tensor(a, device="ROCKCHIP").realize(), Tensor(b, device="ROCKCHIP").realize()
    np.testing.assert_equal((ta<0).where(ta, tb).realize().numpy(), np.where(a<0, a, b))
    special = np.array([-np.inf, -2, 0, 2, np.inf], dtype=np.float16)
    ts = Tensor(special, device="ROCKCHIP").realize()
    np.testing.assert_equal((ts<0).where(ts, 1).realize().numpy(), np.where(special<0, special, 1))

  def test_fp16_abs_specials_and_finite_extrema(self):
    data = np.array([-2, -0., 0., 2., np.inf, -np.inf, np.nan, -np.nan], dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_equal(x.abs().realize().numpy(), np.abs(data))
    finite = data[:4]
    np.testing.assert_equal(Tensor(finite, device="ROCKCHIP").maximum(0).realize().numpy(), np.maximum(finite, np.float16(0)))
    np.testing.assert_equal(Tensor(finite, device="ROCKCHIP").sign().realize().numpy(), np.sign(finite))

  def test_generated_exp2_lut(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    data = encodings.view(np.float16)
    data = data[np.isfinite(data) & (data >= -2) & (data <= 2)]
    actual = Tensor(data, device="ROCKCHIP").realize().exp2().realize().numpy()
    reference = np.exp2(data.astype(np.float32))
    absolute = np.abs(actual.astype(np.float32)-reference)
    relative = absolute/reference
    ulp = np.abs(actual.view(np.uint16).astype(np.int32)-reference.astype(np.float16).view(np.uint16).astype(np.int32))
    order = np.argsort(data.astype(np.float32), kind="stable")
    self.assertEqual(data.size, 32770)
    self.assertLessEqual(float(absolute.max()), 0.0011)
    self.assertLessEqual(float(relative.max()), 0.0009)
    self.assertLessEqual(int(ulp.max()), 1)
    self.assertTrue(np.all(np.diff(actual[order].astype(np.float32)) >= 0))

  def test_generated_roundoff_lut(self):
    data = np.linspace(-16, 16, 4097, dtype=np.float16)
    np.testing.assert_equal(Tensor(data, device="ROCKCHIP").round().realize().numpy(), np.round(data))
    special = np.array([-np.inf,-2.5,-1.5,-.5,-0.,0.,.5,1.5,2.5,np.inf,np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").round().realize().numpy(), np.round(special))
    for function, reference in ((lambda x:x.trunc(), np.trunc), (lambda x:x.floor(), np.floor), (lambda x:x.ceil(), np.ceil)):
      np.testing.assert_equal(function(Tensor(data, device="ROCKCHIP")).realize().numpy(), reference(data))

  def test_linear_sigmoid_workload(self):
    rng = np.random.default_rng(2)
    a_np = rng.uniform(-0.25, 0.25, (1,32)).astype(np.float16)
    w_np = rng.uniform(-0.25, 0.25, (8,32)).astype(np.float16)
    a, w = Tensor(a_np, device="ROCKCHIP").realize(), Tensor(w_np, device="ROCKCHIP").realize()
    actual = (a@w.T).realize().sigmoid().realize().numpy()
    logits = a_np@w_np.T
    np.testing.assert_allclose(actual, 1/(1+np.exp(-logits)), rtol=6e-3, atol=6e-3)

if __name__ == "__main__": unittest.main()
