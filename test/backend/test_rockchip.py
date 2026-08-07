"""Rockchip NPU census: ops known to pass, with DRM_IOCTL_RKNPU_SUBMIT counts.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip.py -q -n0

OUT precision via ROCKCHIP_EW_OUT=fp32|fp16 (default fp32):
  fp32: mtx512 ≤8/chunk, PC-chain 64, host f32→half between stages
  fp16: contiguous half, chain ops without host cvt (fewer ioctls)
"""
from __future__ import annotations
import math, os, unittest
import numpy as np
import torch
from tinygrad import Tensor, Device
from test.backend.test_ops import helper_test_op, slow_test

# fp16 tol matches test_ops.test_gemm_fp16
_FP16 = dict(atol=5e-3, rtol=5e-3)
_EW_CHUNK, _EW_CHAIN = 8, 64
_OUT_FP16 = os.getenv("ROCKCHIP_EW_OUT", "fp32").strip().lower() in ("fp16", "half", "2")
_EW_KAHAN = os.getenv("ROCKCHIP_EW_REDUCE", "sequential").strip().lower() == "kahan"

def _ew_submits(n:int) -> int:
  """EW ioctl count for one logical op over n half elements."""
  if _OUT_FP16:
    # one contiguous task per op (tiled only above 64k); one op → one ioctl unless tiled
    from tinygrad.renderer.rockchip import _MAX_EW_ELEMS_FP16
    tiles = (n + _MAX_EW_ELEMS_FP16 - 1) // _MAX_EW_ELEMS_FP16
    return (tiles + _EW_CHAIN - 1) // _EW_CHAIN
  chunks = (n + _EW_CHUNK - 1) // _EW_CHUNK
  return (chunks + _EW_CHAIN - 1) // _EW_CHAIN

@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["ROCKCHIP"]

  def _half(self, shape, seed:int=0) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.uniform(-2, 2, size=shape).astype(np.float16))

  def _check(self, expected_submits:int, out:Tensor, ref:np.ndarray, atol=5e-3, rtol=5e-3):
    """Realize `out`, compare to `ref`, assert ioctl submit delta.

    Default tol matches test_ops half gemm (DEFAULT_FLOAT=HALF / test_gemm_fp16).
    """
    before = self.dev.submit_count
    got = out.realize().numpy()
    submits = self.dev.submit_count - before
    print(f"  {self._testMethodName}: submits={submits} (expected {expected_submits})")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    self.assertEqual(submits, expected_submits, f"{self._testMethodName}: submits={submits} expected={expected_submits}")

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
    # two logical EW ops; fp16 out chains them in one ioctl, fp32 submits per op
    a, b, c = self._half((45, 65), 7), self._half((45, 65), 8), self._half((45, 65), 9)
    ref = (a.numpy().astype(np.float32) + b.numpy() + c.numpy()).astype(np.float16)
    expected = 1 if _OUT_FP16 else 2 * _ew_submits(45*65)
    self._check(expected, a + b + c, ref)

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
  def test_medium_gemm(self):
    # Sequential EW first misses tolerance at N=21; compensated EW reaches the current full-unroll ceiling at N=32.
    n = 32 if _EW_KAHAN else 20
    helper_test_op([(n,n), (n,n)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_9_gemm(self):
    helper_test_op([(9,9), (9,9)], lambda x,y: x.matmul(y), lambda x,y: x@y, **_FP16)
  def test_small_gemm_padded(self):
    helper_test_op([(9,9), (9,9)],
                   lambda x,y: torch.nn.functional.pad(x, (0,7,0,7)).matmul(torch.nn.functional.pad(y, (0,7,0,7))),
                   lambda x,y: x.pad(((0,7),(0,7)))@y.pad(((0,7),(0,7))), **_FP16)
  def test_small_gemm_range(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.arange(0,64,dtype=np.float32).reshape(8,8),
                         np.arange(64,128,dtype=np.float32).reshape(8,8)], **_FP16)
  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.matmul(y), lambda x,y: x@y,
                   vals=[np.eye(8).astype(np.float32), np.eye(8).astype(np.float32)], **_FP16)
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

if __name__ == "__main__":
  unittest.main()
