"""Trace a (M, K, N) = (2, 33, 17) gemm +1 with TC=0 and ROCKCHIP_FUSED_MATMUL=0.

The Rockchip CMAC atom is (N=16, M=1, K=32) — see tinygrad/codegen/opt/tc.py:143.
This test uses the atom + 1 in each dim, i.e. (M+1, K+1, N+1) = (2, 33, 17), to
exercise non-aligned packing and verify that the +1 ADD is emitted at the end
of the unrolled matmul ops (or, with TC=1, after the WMMA coalesced into CNA).

  A = (2, 33), B = (33, 17)
  A @ B = (2, 17), then +1

Run with:
  DEBUG=7 DEV=ROCKCHIP FORWARD_ONLY=1 python test/trace_gemm.py

Env vars (set with shell prefix, not in this file, to match the usual tinygrad pattern):
  TC=0                   disable tensor cores (env key for USE_TC is "TC", not "USE_TC")
  ROCKCHIP_FUSED_MATMUL=0 disable rockchip fused matmul template
  DEFAULT_FLOAT=HALF     make every Tensor default to fp16
  DEV=ROCKCHIP           run on the rockchip device
  FORWARD_ONLY=1         skip backward
"""
import os, struct
import numpy as np
os.environ.setdefault("TC", "0")
os.environ.setdefault("ROCKCHIP_FUSED_MATMUL", "0")
os.environ.setdefault("DEFAULT_FLOAT", "HALF")
os.environ.setdefault("DEV", "ROCKCHIP")
os.environ.setdefault("FORWARD_ONLY", "1")

from tinygrad import Tensor

# Tencore atom (N=16, M=1, K=32) + 1 in each dim.
# align_in = next multiple of 32 >= K = 64 (see RK_MIN_CHANNEL_TILE in ops_rockchip.py).
M, K, N = 2, 33, 17

# Sequential values scaled into fp16-safe range. K=33 accumulations of up to
# 0.66 * 5.61 ≈ 3.7 per term yield at most ~60 for any output, well under the
# fp16 ceiling of 65504, so the +1 stays representable.
A = (np.arange(1, M*K+1, dtype=np.float32) * 0.01).reshape(M, K)
B = (np.arange(1, K*N+1, dtype=np.float32) * 0.01).reshape(K, N)
EXPECTED = (A @ B + 1).astype(np.float32)

a = Tensor(A.astype(np.float16))
b = Tensor(B.astype(np.float16))
out = a.matmul(b) + 1
out = out.realize().numpy().astype(np.float32)

ALIGN = 64

def fp16_bytes(f: float) -> bytes:
  return np.array([f], dtype=np.float16).tobytes()

def fp16_hex(f: float) -> str:
  h = struct.unpack('<H', fp16_bytes(f))[0]
  return f"0x{h:04x}={f:.2f}"

def print_packed_memory(name: str, rows: int, cols: int, row_data, align: int, max_rows: int | None = None):
  """Print the packed fp16 memory layout of a 2D matrix padded to `align` fp16 per row.

  row_data: callable row_data(r, c) -> float (the value at logical position (r, c))
  max_rows: if set, only print the first/last few rows with a "..." separator.
  """
  row_bytes = align * 2
  print(f"\npacked {name} memory (rows={rows}, cols={cols}, align={align} fp16/row, "
        f"row_stride={row_bytes} bytes, total={rows*row_bytes} bytes):")
  rows_to_print = list(range(rows))
  if max_rows is not None and rows > max_rows * 2:
    rows_to_print = list(range(max_rows)) + [-1] + list(range(rows - max_rows, rows))
  for r in rows_to_print:
    if r == -1:
      print(f"  ...")
      continue
    addr = r * row_bytes
    hex_pairs, decoded = [], []
    for c in range(cols):
      v = float(row_data(r, c))
      h = struct.unpack('<H', fp16_bytes(v))[0]
      hex_pairs.append(f"{h:04x}")
      decoded.append(f"{v:>4.1f}")
    pad = align - cols
    for _ in range(pad):
      hex_pairs.append("0000")
      decoded.append("  --")
    bytes_line = ' '.join(hex_pairs)
    values_line = ' '.join(decoded)
    print(f"  0x{addr:04x}: {bytes_line}   {values_line}")

# Pack A as (M, K) row-major, each row padded to `align` fp16
print_packed_memory("A", M, K, lambda r, c: A[r, c], ALIGN)

# Pack B as (N, K) row-major = B^T flat, each row padded to `align` fp16
# This is the layout the rockchip NPU uses for the rhs of a matmul
print_packed_memory("B (as N,K row-major = B^T flat)", N, K, lambda r, c: B[c, r], ALIGN, max_rows=3)

# M*N*K = 1122 entries is too long; show the first few plus the final ones.
print(f"\n{M}x{N}x{K} (fp16) — first 4 and last 2 of {M*N*K} entries:")
print(f" {'idx':>4} | {'input':<14} | {'weight':<14}")
print(f"-----+----------------+----------------")
a_flat = [(i*ALIGN + j, float(A[i, j])) for i in range(M) for j in range(K)]
b_flat = [(i*ALIGN + j, float(B[j, i])) for i in range(N) for j in range(K)]
pairs = list(zip(a_flat, b_flat))
for (idx_a, av), (idx_b, bv) in pairs[:4]:
  assert idx_a == idx_b
  print(f"{idx_a:>5} | {fp16_hex(av):<14} | {fp16_hex(bv):<14}")
print(f"  ...")
for (idx_a, av), (idx_b, bv) in pairs[-2:]:
  assert idx_a == idx_b
  print(f"{idx_a:>5} | {fp16_hex(av):<14} | {fp16_hex(bv):<14}")

print(f"\nraw output surface (m={M}, n={N}, align_out={ALIGN}, dtype={type(out.dtype).__name__}):")
for i in range(M):
  for j in range(N):
    idx = i*ALIGN + j
    print(f"  [{idx:>4}] r{i} c{j} = {out[i, j]:.1f}")

max_diff = float(np.max(np.abs(out - EXPECTED)))
max_rel = float(np.max(np.abs(out - EXPECTED) / np.maximum(np.abs(EXPECTED), 1e-6)))
# fp16 accumulation over K=33 terms loses some precision, but must still catch a dropped +1 post-op.
status = "PASS" if (max_diff < 0.5 and max_rel < 0.02) else "FAIL"
print(f"  {status} (max_diff={max_diff:.4f}, max_rel={max_rel:.4f})")
print(f"result:   {out}")
print(f"expected: {EXPECTED}")

assert np.allclose(out, EXPECTED, atol=0.5, rtol=0.02), "matmul result wrong"
print("OK")
