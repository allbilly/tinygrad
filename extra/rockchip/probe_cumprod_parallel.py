#!/usr/bin/env python3
"""Compare a logarithmic FP16 prefix product with the unchanged TestOps contract."""
from __future__ import annotations
import numpy as np
import torch

def parallel_prefix(values:np.ndarray, axis:int) -> np.ndarray:
  output = values.copy()
  distance = 1
  while distance < values.shape[axis]:
    previous = output.copy()
    dst, src = [slice(None)]*values.ndim, [slice(None)]*values.ndim
    dst[axis], src[axis] = slice(distance,None), slice(None,-distance)
    output[tuple(dst)] = (previous[tuple(dst)]*previous[tuple(src)]).astype(np.float16)
    distance *= 2
  return output

def main() -> None:
  for shape,axes in (((20,),(0,)),((20,30),(0,1)),((20,30,40),(2,-1))):
    for axis in axes:
      np.random.seed(0)
      values = np.random.uniform(-2,2,shape).astype(np.float16)
      expected = torch.cumprod(torch.from_numpy(values),dim=axis).numpy()
      sequential = np.cumprod(values,axis=axis,dtype=np.float16)
      actual = parallel_prefix(values,axis)
      allowed = np.isclose(actual,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
      sequential_allowed = np.isclose(sequential,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
      finite = np.isfinite(actual)&np.isfinite(expected)
      absolute = np.abs(actual[finite].astype(np.float32)-expected[finite].astype(np.float32))
      relative = absolute/np.maximum(np.abs(expected[finite].astype(np.float32)),1e-30)
      print(f"shape={shape} axis={axis} sequential_mismatches={np.count_nonzero(~sequential_allowed)}/{expected.size} "
            f"mismatches={np.count_nonzero(~allowed)}/{expected.size} "
            f"max_abs={float(absolute.max(initial=0))} max_rel={float(relative.max(initial=0))}")

if __name__ == "__main__": main()
