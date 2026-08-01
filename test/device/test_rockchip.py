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

  def test_where_uses_native_mask(self):
    a = np.linspace(-2, 2, 16, dtype=np.float16)
    b = np.linspace(3, 6, 16, dtype=np.float16)
    ta, tb = Tensor(a, device="ROCKCHIP").realize(), Tensor(b, device="ROCKCHIP").realize()
    np.testing.assert_equal((ta<0).where(ta, tb).realize().numpy(), np.where(a<0, a, b))

if __name__ == "__main__": unittest.main()
