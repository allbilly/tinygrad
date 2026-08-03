#!/usr/bin/env python3
"""Probe direct RK3588 CMAC output widths above the original 16-channel contract."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKContract, RKLayout, RKLayoutKind, RKTensorRef,
  encode_image, emit_contract)
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def image(channels:int):
  align = max(32,(channels+31)&-32)
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((1,channels),(1,align*2),(align*4,2),dtypes.half,
    padding=((0,0),(0,align*2-channels))))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((1,align),(1,align),(align*2,2),dtypes.half))
  rhs = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((channels,align),(align,align),(align*2,2),dtypes.half,
    padding=((0,align-channels),(0,0)),kind=RKLayoutKind.CMAC_WEIGHT))
  return emit_contract(RKContract(out,lhs,rhs,0))

def main() -> None:
  dev = RockchipDevice("ROCKCHIP")
  for channels in (16,20,24,28,32,64,96,128):
    align = max(32,(channels+31)&-32)
    lhs = np.linspace(-3,3,align,dtype=np.float16)
    weight = np.eye(align,dtype=np.float16)
    packed = weight.reshape(align//16,16,align//32,32).transpose(0,2,1,3).ravel()
    out, lhs_buf, rhs = dev._gpu_alloc(align*4), dev._gpu_alloc(lhs.nbytes), dev._gpu_alloc(packed.nbytes)
    try:
      ctypes.memmove(int(lhs_buf.va_addr),lhs.ctypes.data,lhs.nbytes)
      ctypes.memmove(int(rhs.va_addr),packed.ctypes.data,packed.nbytes)
      ctypes.memset(int(out.va_addr),0,align*4)
      program = RockchipProgram(dev,TinyELF(encode_image(image(channels)),f"cmac_n{channels}",Target("ROCKCHIP"),()))
      program(out,lhs_buf,rhs,wait=True)
      physical = np.frombuffer(ctypes.string_at(int(out.va_addr),align*4),dtype=np.float16).copy()
      actual = physical[[channel//16*32+channel%16 for channel in range(channels)]]
      print(f"N={channels} exact={np.array_equal(actual,lhs[:channels])} actual={actual.tolist()}")
    finally:
      dev._gpu_free(out)
      dev._gpu_free(lhs_buf)
      dev._gpu_free(rhs)

if __name__ == "__main__": main()
