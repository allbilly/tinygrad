# PR1 Rockchip NPU tests: fp16 ADD (DPU), matmul (CMAC), globalmax (PPU).
# Replaces the legacy FP32/torch-dependent test file. See test/rockchip/ for the full suite.
import os, unittest
import numpy as np

if os.getenv("ROCKCHIP") == "1" and "DEV" not in os.environ: os.environ["DEV"] = "ROCKCHIP"
os.environ.setdefault("DEFAULT_FLOAT", "half")

from tinygrad import Tensor

_NPU_AVAILABLE = os.getenv("DEV") == "ROCKCHIP"

@unittest.skipUnless(_NPU_AVAILABLE, "set DEV=ROCKCHIP to run NPU tests")
class TestRockchipPR1(unittest.TestCase):
  def test_dpu_add(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_inplace_add(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    a.assign(a + b).realize()
    np.testing.assert_allclose(a.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_cmac_matmul(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_ppu_globalmax(self):
    a_np = np.random.randn(2,2,8).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.max(axis=(0,1)).realize()
    np.testing.assert_allclose(c.numpy(), a_np.max(axis=(0,1)), rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
  unittest.main(verbosity=2)
