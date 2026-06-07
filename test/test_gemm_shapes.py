"""Test GEMM on ROCKCHIP device with various shapes.

Run with:
  TC=0 ROCKCHIP_FUSED_MATMUL=0 DEFAULT_FLOAT=HALF DEV=ROCKCHIP FORWARD_ONLY=1 python test/test_gemm_shapes.py [shape1 shape2 ...]

Without arguments, tests a default set of shapes.
"""
import os, sys
os.environ.setdefault("TC", "0")
os.environ.setdefault("ROCKCHIP_FUSED_MATMUL", "0")
os.environ.setdefault("DEFAULT_FLOAT", "HALF")
os.environ.setdefault("DEV", "ROCKCHIP")
os.environ.setdefault("FORWARD_ONLY", "1")

import numpy as np
from tinygrad import Tensor

DEFAULT_SHAPES = [
  (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8),
  (16, 16, 16), (32, 32, 32),
  (32, 2, 2), (32, 4, 4), (32, 8, 8), (32, 16, 16),
  (2, 32, 2), (2, 32, 4), (2, 32, 8), (2, 32, 16),
  (3, 32, 3), (5, 32, 5), (6, 32, 6), (7, 32, 7),
  (32, 32, 2), (32, 32, 16),
  (2, 2, 32), (4, 4, 32), (8, 8, 32), (16, 16, 32),
  (32, 16, 32), (16, 32, 32),
  (3, 5, 7), (7, 5, 3), (5, 8, 3),
  (2, 3, 5), (3, 5, 2), (8, 3, 4),
  (2, 16, 4), (4, 2, 16), (16, 4, 2),
]

def parse_shapes(args):
  out = []
  for a in args:
    parts = a.lower().split('x')
    if len(parts) != 3: continue
    try:
      m, n, k = int(parts[0]), int(parts[1]), int(parts[2])
      out.append((m, n, k))
    except ValueError: pass
  return out

def test_shape(m, n, k, seed=0):
  rng = np.random.default_rng(seed)
  a_np = rng.standard_normal((m, k)).astype(np.float16)
  b_np = rng.standard_normal((k, n)).astype(np.float16)
  a = Tensor(a_np)
  b = Tensor(b_np)
  out = a.matmul(b).realize().numpy()
  expected = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)
  diff = float(np.max(np.abs(out.astype(np.float32) - expected.astype(np.float32))))
  ok = diff < 5e-2
  return ok, diff

shapes = parse_shapes(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_SHAPES
print(f"Testing {len(shapes)} shapes...\n")
passed, failed = 0, []
for (m, n, k) in shapes:
  try:
    ok, diff = test_shape(m, n, k)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {m}x{k} @ {k}x{n}  max_diff={diff:.4f}")
    if ok: passed += 1
    else: failed.append((m, n, k, diff))
  except Exception as e:
    print(f"  [ERR ] {m}x{k} @ {k}x{n}  {type(e).__name__}: {e}")
    failed.append((m, n, k, str(e)))

print(f"\n{passed}/{len(shapes)} passed")
if failed:
  print("failed:")
  for m, n, k, d in failed:
    print(f"  {m}x{k} @ {k}x{n}: {d}")
  sys.exit(1)
