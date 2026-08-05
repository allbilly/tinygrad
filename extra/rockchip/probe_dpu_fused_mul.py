#!/usr/bin/env python3
"""Characterize compact four-vector multiplication through DPU BS, BN, and EW."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKDPUProgram, RKFusedMulStage, encode_image, emit_dpu
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def main() -> None:
  dev, capacity, canary = RockchipDevice("ROCKCHIP"), 288, np.float16(123.0)
  rng = np.random.default_rng(0)
  for count in (1,7,8,16,64,256):
    values = tuple(rng.uniform(0.5,1.5,capacity).astype(np.float16) for _ in range(4))
    expected = np.prod(np.stack([value[:count].astype(np.float32) for value in values]),axis=0,dtype=np.float32).astype(np.float16)
    output = np.full(capacity,canary,dtype=np.float16)
    buffers = [dev._gpu_alloc(capacity*2) for _ in range(5)]
    try:
      ctypes.memmove(int(buffers[0].va_addr),output.ctypes.data,output.nbytes)
      for buffer,value in zip(buffers[1:],values): ctypes.memmove(int(buffer.va_addr),value.ctypes.data,value.nbytes)
      args = tuple(RKArg(RKBufferKind.ARG,index) for index in range(5))
      image = emit_dpu(RKDPUProgram((RKFusedMulStage(*args,count),)))
      RockchipProgram(dev,TinyELF(encode_image(image),f"dpu_fused_mul_{count}",Target("ROCKCHIP"),()))(*buffers,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),capacity*2),dtype=np.float16)
      physical_count = (count+7)&-8
      exact, atom_tail_clean = np.array_equal(actual[:count],expected), np.all(actual[physical_count:] == canary)
      print(f"count={count:3d} physical={physical_count:3d} exact_fp32_once={exact} atom_tail_clean={atom_tail_clean}")
      assert exact and atom_tail_clean
    finally:
      for buffer in buffers: dev._gpu_free(buffer)

if __name__ == "__main__": main()
