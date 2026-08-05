#!/usr/bin/env python3
"""Measure the rejected unreduced FP16 BCE recipes at the official tolerance."""
from __future__ import annotations
import numpy as np
import torch

from tinygrad import Tensor
import tinygrad.renderer.rockchip.elementwise as rk_elementwise

_REJECT = "unreduced BCE LUT composition exceeds the FP16 relative-error contract"

def _stats(actual:np.ndarray, expected:np.ndarray) -> tuple[int,float,float]:
  difference = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
  failed = difference > 1e-6+1e-3*np.abs(expected.astype(np.float32))
  relative = difference/np.maximum(np.abs(expected.astype(np.float32)),1e-30)
  return int(np.count_nonzero(failed)),float(difference[failed].max(initial=0)),float(relative[failed].max(initial=0))

def main() -> None:
  original = rk_elementwise._numerical_contract
  rk_elementwise._numerical_contract = lambda u: None if original(u) == _REJECT else original(u)
  np.random.seed(0)
  x,y = (np.random.uniform(-2,2,(32,10)).astype(np.float16) for _ in range(2))
  tx,ty = torch.from_numpy(x),torch.from_numpy(y)
  nx,ny = Tensor(x,device="ROCKCHIP"),Tensor(y,device="ROCKCHIP")
  cases = (
    ("probability", nx.sigmoid().binary_crossentropy(ny.clip(0,1),reduction="none"),
      torch.nn.functional.binary_cross_entropy(tx.sigmoid(),ty.clip(0,1),reduction="none")),
    ("logits", nx.binary_crossentropy_logits(ny.clip(0,1),reduction="none"),
      torch.nn.functional.binary_cross_entropy_with_logits(tx,ty.clip(0,1),reduction="none")),
  )
  for name,result,reference in cases:
    actual,expected = result.realize().numpy(),reference.numpy()
    failed,max_abs,max_rel = _stats(actual,expected)
    print(f"{name}: mismatches={failed}/{actual.size} max_abs={max_abs} max_rel={max_rel}")
    assert failed

if __name__ == "__main__": main()
