"""Trace a 2x2 gemm with TC=0 and ROCKCHIP_FUSED_MATMUL=0 to see raw mul/add ops.

  A = [[1, 2],     B = [[5, 6],
       [3, 4]]          [7, 8]]
  A @ B = [[19, 22], [43, 50]]

Run with:
  DEBUG=6 DEV=ROCKCHIP FORWARD_ONLY=1 python test/trace_gemm.py

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

A = np.array([[1, 2], [3, 4]], dtype=np.int32)
B = np.array([[5, 6], [7, 8]], dtype=np.int32)
EXPECTED = np.array([[19, 22], [43, 50]], dtype=np.float32)

a = Tensor(A).half()
b = Tensor(B).half()
out = a.matmul(b).realize().numpy().astype(np.float32)

M, K = A.shape
K2, N = B.shape
ALIGN = 32

def fp16_bytes(f: float) -> bytes:
  return np.array([f], dtype=np.float16).tobytes()

def fp16_hex(f: float) -> str:
  h = struct.unpack('<H', fp16_bytes(f))[0]
  return f"0x{h:04x}={f:.2f}"

def print_packed_memory(name: str, rows: int, cols: int, row_data, align: int):
  """Print the packed fp16 memory layout of a 2D matrix padded to `align` fp16 per row.

  row_data: callable row_data(r, c) -> float (the value at logical position (r, c))
  """
  row_bytes = align * 2
  print(f"\npacked {name} memory (rows={rows}, cols={cols}, align={align} fp16/row, "
        f"row_stride={row_bytes} bytes, total={rows*row_bytes} bytes):")
  for r in range(rows):
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
print_packed_memory("B (as N,K row-major = B^T flat)", N, K, lambda r, c: B[c, r], ALIGN)

print(f"\n{M}x{N}x{K} (fp16):")
print(f" {'idx':>4} | {'input':<14} | {'weight':<14}")
print(f"-----+----------------+----------------")
a_flat = [(i*ALIGN + j, float(A[i, j])) for i in range(M) for j in range(K)]
b_flat = [(i*ALIGN + j, float(B[j, i])) for i in range(N) for j in range(K)]
for (idx_a, av), (idx_b, bv) in zip(a_flat, b_flat):
  assert idx_a == idx_b
  print(f"{idx_a:>5} | {fp16_hex(av):<14} | {fp16_hex(bv):<14}")

print(f"\nraw output surface (m={M}, align_out={ALIGN}, dtype={type(out.dtype).__name__}):")
for i in range(M):
  for j in range(N):
    idx = i*ALIGN + j
    print(f"  [{idx:>4}] r{i} c{j} = {out[i, j]:.1f}")

max_diff = float(np.max(np.abs(out - EXPECTED)))
status = "PASS" if max_diff < 5e-3 else "FAIL"
print(f"  {status} (max_diff={max_diff:.4f})")
print(f"result:   {out}")
print(f"expected: {EXPECTED}")

assert (out == EXPECTED).all(), "matmul result wrong"
print("OK")
