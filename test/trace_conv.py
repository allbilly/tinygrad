"""Trace the simple conv2d shape from test_ops.py on ROCKCHIP.

Shape from test/backend/test_ops.py:test_simple_conv2d:
  input  = (1, 4, 9, 9)
  weight = (4, 4, 3, 3)
  bias   = (4,)
  output = (1, 4, 7, 7)

Logical GEMM view of this conv is:
  M = batch * out_h * out_w = 49
  N = cout = 4
  K = cin * kh * kw = 36

Run with:
  DEBUG=7 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 python test/trace_conv.py
"""
import os
import numpy as np

os.environ.setdefault("DEFAULT_FLOAT", "HALF")
os.environ.setdefault("DEV", "ROCKCHIP")
os.environ.setdefault("FORWARD_ONLY", "1")

from tinygrad import Tensor

BS, CIN, IH, IW = 1, 4, 9, 9
COUT, KH, KW = 4, 3, 3
OH, OW = IH - KH + 1, IW - KW + 1
M, N, K = BS * OH * OW, COUT, CIN * KH * KW

np.random.seed(0)
x_np = (np.arange(BS*CIN*IH*IW, dtype=np.float32).reshape(BS, CIN, IH, IW) * 0.01).astype(np.float16)
w_np = (np.arange(COUT*CIN*KH*KW, dtype=np.float32).reshape(COUT, CIN, KH, KW) * 0.01).astype(np.float16)
b_np = (np.arange(COUT, dtype=np.float32) * 0.1).astype(np.float16)

print(f"conv input={x_np.shape} weight={w_np.shape} bias={b_np.shape} output=({BS}, {COUT}, {OH}, {OW})")
print(f"logical gemm M={M} N={N} K={K}")
print(f"flat buffers out={BS*COUT*OH*OW} input={BS*CIN*IH*IW} weight={COUT*CIN*KH*KW} bias={COUT}")

x = Tensor(x_np)
w = Tensor(w_np)
b = Tensor(b_np)
out = x.conv2d(w, b).realize().numpy().astype(np.float32)

expected = np.zeros((BS, COUT, OH, OW), dtype=np.float32)
for bs in range(BS):
  for co in range(COUT):
    for oy in range(OH):
      for ox in range(OW):
        expected[bs, co, oy, ox] = np.sum(x_np[bs, :, oy:oy+KH, ox:ox+KW].astype(np.float32) * w_np[co].astype(np.float32)) + b_np[co].astype(np.float32)

max_diff = float(np.max(np.abs(out - expected)))
status = "PASS" if np.allclose(out, expected, atol=0.5, rtol=0.02) else "FAIL"
print(f"{status} max_diff={max_diff:.4f}")
print("result sample", out.reshape(-1)[:8])
print("expected sample", expected.reshape(-1)[:8])
