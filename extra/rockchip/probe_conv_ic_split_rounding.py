#!/usr/bin/env python3
"""Characterize the FP16-rounding cost of channel-split 1x1 convolution."""
from __future__ import annotations
import numpy as np
import torch

_SHAPE = (1, 16, 32, 32)
_WEIGHT_SHAPE = (16, 16, 1, 1)
_RTOL, _ATOL = 1e-3, 1e-6

def _partials(source:np.ndarray, weight:np.ndarray, channels:int) -> list[np.ndarray]:
  return [np.sum(source[:,None,start:start+channels].astype(np.float32)*
    weight[None,:,start:start+channels,0,0,None,None].astype(np.float32),axis=2).astype(np.float16)
    for start in range(0,source.shape[1],channels)]

def _sequential(values:list[np.ndarray]) -> np.ndarray:
  result = values[0]
  for value in values[1:]: result = (result+value).astype(np.float16)
  return result

def _balanced(values:list[np.ndarray]) -> np.ndarray:
  while len(values) > 1:
    values = [(values[i]+values[i+1]).astype(np.float16) for i in range(0,len(values),2)]
  return values[0]

def main() -> None:
  np.random.seed(0)
  source = np.random.uniform(-2,2,_SHAPE).astype(np.float16)
  weight = np.random.uniform(-2,2,_WEIGHT_SHAPE).astype(np.float16)
  reference = torch.nn.functional.conv2d(torch.tensor(source),torch.tensor(weight)).numpy()
  for channels in (1,2,4,8):
    values = _partials(source,weight,channels)
    for order,actual in (("sequential",_sequential(values)),("balanced",_balanced(values))):
      mismatch = ~np.isclose(actual,reference,rtol=_RTOL,atol=_ATOL,equal_nan=True)
      max_abs = float(np.max(np.abs(actual.astype(np.float32)-reference.astype(np.float32))))
      print(f"channels={channels:2d} order={order:10s} mismatch={int(mismatch.sum()):5d}/{reference.size} max_abs={max_abs}")
      assert mismatch.any(), "an FP16-intermediate split unexpectedly met the official convolution tolerance"

if __name__ == "__main__": main()
