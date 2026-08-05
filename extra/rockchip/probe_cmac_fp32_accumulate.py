#!/usr/bin/env python3
"""Probe FP32 CMAC writeback with a FP32 BRDMA accumulator epilogue."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKCMACTask, RKEpilogue, RKLayout, RKLayoutKind,
  RKTensorRef, encode_image, emit_cmac_task)
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def image():
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((1,32),(1,64),(256,4),dtypes.float,padding=((0,0),(0,32))))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((1,32),(1,32),(64,2),dtypes.half))
  rhs = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((32,32),(32,32),(64,2),dtypes.half,kind=RKLayoutKind.CMAC_WEIGHT))
  bias = RKTensorRef(RKArg(RKBufferKind.ARG,3),RKLayout((32,),(32,),(4,),dtypes.float))
  return emit_cmac_task(RKCMACTask(out,lhs,rhs,0,epilogue=RKEpilogue(bias)))

def main() -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608)
  lhs = rng.uniform(-2,2,32).astype(np.float16)
  weights = rng.uniform(-1,1,(32,32)).astype(np.float16)
  packed = weights.reshape(2,16,1,32).transpose(0,2,1,3).ravel()
  bias = rng.uniform(-4,4,32).astype(np.float32)
  buffers = [dev._gpu_alloc(size) for size in (256,64,packed.nbytes,128)]
  try:
    for buf,value in zip(buffers[1:],(lhs,packed,bias)): ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
    ctypes.memset(int(buffers[0].va_addr),0,256)
    RockchipProgram(dev,TinyELF(encode_image(image()),"cmac_fp32_accumulate",Target("ROCKCHIP"),()))(*buffers,wait=True)
    physical = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),256),dtype=np.float32).copy()
    actual = physical[:32]
    expected = lhs.astype(np.float32) @ weights.T.astype(np.float32) + bias
    print(f"finite={np.isfinite(actual).all()} max_abs={np.max(np.abs(actual-expected))} actual={actual[:8].tolist()}")
    print(f"expected={expected[:8].tolist()}")
  finally:
    for buf in buffers: dev._gpu_free(buf)

if __name__ == "__main__": main()
