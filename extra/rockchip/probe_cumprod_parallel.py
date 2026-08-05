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

def parallel_prefix_base4(values:np.ndarray, axis:int) -> np.ndarray:
  """Model one DPU task multiplying the current value and three prefix-shifted values in FP32, then rounding once."""
  output = values.copy()
  distance = 1
  while distance < values.shape[axis]:
    previous = output.copy()
    factors = []
    for multiple in range(4):
      factor = np.ones_like(previous)
      if multiple == 0: factor = previous
      elif multiple*distance < values.shape[axis]:
        dst, src = [slice(None)]*values.ndim, [slice(None)]*values.ndim
        dst[axis], src[axis] = slice(multiple*distance,None), slice(None,-multiple*distance)
        factor[tuple(dst)] = previous[tuple(src)]
      factors.append(factor.astype(np.float32))
    output = np.prod(np.stack(factors),axis=0,dtype=np.float32).astype(np.float16)
    distance *= 4
  return output

def main() -> None:
  for shape,axes in (((20,),(0,)),((20,30),(0,1)),((20,30,40),(2,-1))):
    for axis in axes:
      np.random.seed(0)
      values = np.random.uniform(-2,2,shape).astype(np.float16)
      expected = torch.cumprod(torch.from_numpy(values),dim=axis).numpy()
      sequential = np.cumprod(values,axis=axis,dtype=np.float16)
      actual, base4 = parallel_prefix(values,axis), parallel_prefix_base4(values,axis)
      allowed = np.isclose(actual,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
      base4_allowed = np.isclose(base4,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
      sequential_allowed = np.isclose(sequential,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
      finite = np.isfinite(actual)&np.isfinite(expected)
      absolute = np.abs(actual[finite].astype(np.float32)-expected[finite].astype(np.float32))
      relative = absolute/np.maximum(np.abs(expected[finite].astype(np.float32)),1e-30)
      print(f"shape={shape} axis={axis} sequential_mismatches={np.count_nonzero(~sequential_allowed)}/{expected.size} "
            f"base2_mismatches={np.count_nonzero(~allowed)}/{expected.size} base4_mismatches={np.count_nonzero(~base4_allowed)}/{expected.size} "
            f"max_abs={float(absolute.max(initial=0))} max_rel={float(relative.max(initial=0))}")

if __name__ == "__main__": main()
