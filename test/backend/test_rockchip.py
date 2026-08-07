"""Rockchip NPU census: ops known to pass, with DRM_IOCTL_RKNPU_SUBMIT counts.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip.py -q -n0
"""
from __future__ import annotations
import math, unittest
import numpy as np
from tinygrad import Tensor, Device

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["ROCKCHIP"]

  def _half(self, shape, seed:int=0) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-2, 2, size=shape).astype(np.float16))

  def _check(self, expected_submits:int, out:Tensor, ref:np.ndarray, atol=1e-2, rtol=1e-2):
    """Realize `out`, compare to `ref`, assert ioctl submit delta."""
    before = self.dev.submit_count
    got = out.realize().numpy()
    submits = self.dev.submit_count - before
    print(f"  {self._testMethodName}: submits={submits} (expected {expected_submits})")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    self.assertEqual(submits, expected_submits, f"{self._testMethodName}: submits={submits} expected={expected_submits}")

  # ---- ADD ----
  def test_tiny_add(self):
    a, b = self._half((3,), 1), self._half((3,), 2)
    self._check(1, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add(self):
    a, b = self._half((45, 68), 3), self._half((45, 68), 4)
    self._check(1, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add_scalar_constfold(self):
    # Tensor(1)+0.5 folds on device=None — no NPU submit
    self._check(0, Tensor(1) + 0.5, np.array(1.5, dtype=np.float16))

  def test_add_empty(self):
    # rank-0 buffers materialize without NPU submit
    a, b = self._half((), 5), self._half((), 6)
    self._check(0, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))

  def test_add3(self):
    # two DPU stages, one PC-chained submit
    a, b, c = self._half((45, 65), 7), self._half((45, 65), 8), self._half((45, 65), 9)
    ref = (a.numpy().astype(np.float32) + b.numpy() + c.numpy()).astype(np.float16)
    self._check(1, a + b + c, ref)

  # ---- MUL ----
  def test_tiny_mul(self):
    a, b = self._half((64,), 10), self._half((64,), 11)
    self._check(1, a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_mul(self):
    a, b = self._half((64, 64), 12), self._half((64, 64), 13)
    self._check(1, a * b, (a.numpy().astype(np.float32) * b.numpy()).astype(np.float16))

  def test_scalar_mul(self):
    a = self._half((45, 65), 14)
    self._check(1, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(1, a * -1, (a.numpy().astype(np.float32) * -1).astype(np.float16))
    self._check(1, 255 * a, (a.numpy().astype(np.float32) * 255).astype(np.float16))
    self._check(1, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_scalar_mul_empty(self):
    # rank-0 scalar mul — no NPU submit
    a = self._half((), 15)
    self._check(0, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(0, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))

  def test_mul_naninf(self):
    a = self._half((45, 65), 16)
    self._check(1, a * math.inf, (a.numpy().astype(np.float32) * np.float32(np.inf)).astype(np.float16))
    self._check(1, a * -math.inf, (a.numpy().astype(np.float32) * np.float32(-np.inf)).astype(np.float16))
    self._check(1, a * math.nan, (a.numpy().astype(np.float32) * np.float32(np.nan)).astype(np.float16))

  # ---- GEMM ----
  def test_small_gemm(self):
    a, b = self._half((8, 8), 20), self._half((8, 8), 21)
    An, Bn = a.numpy(), b.numpy()
    ref = np.zeros((8, 8), dtype=np.float16)
    for k in range(8):
      ref = (ref + (An[:, k, None] * Bn[None, k, :])).astype(np.float16)
    self._check(1, a @ b, ref)

if __name__ == "__main__":
  unittest.main()
