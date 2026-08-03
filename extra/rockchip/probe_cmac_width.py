#!/usr/bin/env python3
"""Probe direct RK3588 CMAC output widths above the original 16-channel contract."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKContract, RKLayout, RKLayoutKind, RKTensorRef,
  RKImage, RKTarget, encode_image, emit_contract)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU = 0x1001

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image(channels:int, compact:bool=False, rows:int=1):
  align = max(32,(channels+31)&-32)
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((rows,channels),(rows,align*2),(align*4,2),dtypes.half,
    padding=((0,0),(0,align*2-channels))))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((rows,align),(rows,align),(align*2,2),dtypes.half))
  rhs = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((channels,align),(align,align),(align*2,2),dtypes.half,
    padding=((0,align-channels),(0,0)),kind=RKLayoutKind.CMAC_WEIGHT))
  base = emit_contract(RKContract(out,lhs,rhs,0))
  if not compact: return base
  # The normal FP16 WDMA layout places each 16-channel group in a 32-lane atom.
  # Probe whether a one-atom SURFACE_ADD packs those groups without another task.
  return RKImage(RKTarget.RK3588,(_replace(base.stages[0],_DPU,rk.REG_DPU_SURFACE_ADD,0x20),),base.scratch,base.constants)

def selector_image(span:int, indexes:list[int]) -> RKImage:
  align = (span+31)&-32
  weights = np.zeros((32,align),dtype=np.float16)
  for output,index in enumerate(indexes): weights[output,index] = 1
  packed = weights.reshape(2,16,align//32,32).transpose(0,2,1,3).ravel()
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((1,len(indexes)),(1,32),(64,2),dtypes.half,padding=((0,0),(0,32-len(indexes)))))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((1,span),(1,align),(align*2,2),dtypes.half,padding=((0,0),(0,align-span))))
  rhs = RKTensorRef(RKArg(RKBufferKind.CONSTANT,0),RKLayout((len(indexes),span),(32,align),(align*2,2),dtypes.half,
    padding=((0,32-len(indexes)),(0,align-span)),kind=RKLayoutKind.CMAC_WEIGHT))
  return emit_contract(RKContract(out,lhs,rhs,0,packed.tobytes()))

def main() -> None:
  dev = RockchipDevice("ROCKCHIP")
  for channels in (16,20,24,28,32,40,64,96,99,128):
    align = max(32,(channels+31)&-32)
    rows = 4 if channels in (40,99) else 1
    lhs = np.linspace(-3,3,rows*align,dtype=np.float16).reshape(rows,align)
    weight = np.eye(align,dtype=np.float16)
    packed = weight.reshape(align//16,16,align//32,32).transpose(0,2,1,3).ravel()
    out, lhs_buf, rhs = dev._gpu_alloc(rows*align*4), dev._gpu_alloc(lhs.nbytes), dev._gpu_alloc(packed.nbytes)
    try:
      ctypes.memmove(int(lhs_buf.va_addr),lhs.ctypes.data,lhs.nbytes)
      ctypes.memmove(int(rhs.va_addr),packed.ctypes.data,packed.nbytes)
      ctypes.memset(int(out.va_addr),0,align*4)
      for compact in (False,True):
        ctypes.memset(int(out.va_addr),0,rows*align*4)
        program = RockchipProgram(dev,TinyELF(encode_image(image(channels,compact,rows)),
          f"cmac_n{channels}_{'compact' if compact else 'normal'}",Target("ROCKCHIP"),()))
        program(out,lhs_buf,rhs,wait=True)
        physical = np.frombuffer(ctypes.string_at(int(out.va_addr),rows*align*4),dtype=np.float16).copy()
        actual = physical[:rows*channels].reshape(rows,channels) if compact else physical.reshape(rows,-1)[:,
          [channel//16*32+channel%16 for channel in range(channels)]]
        print(f"M={rows} N={channels} compact={compact} exact={np.array_equal(actual,lhs[:,:channels])} actual={actual.tolist()}")
    finally:
      dev._gpu_free(out)
      dev._gpu_free(lhs_buf)
      dev._gpu_free(rhs)

  indexes = [99*index for index in range(16)]
  span = indexes[-1]+1
  values = np.arange(span,dtype=np.float16)
  out, src = dev._gpu_alloc(64), dev._gpu_alloc(values.nbytes)
  try:
    ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
    ctypes.memset(int(out.va_addr),0,64)
    program = RockchipProgram(dev,TinyELF(encode_image(selector_image(span,indexes)),"cmac_selector_k99",Target("ROCKCHIP"),()))
    program(out,src,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(out.va_addr),64),dtype=np.float16).copy()[:16]
    print(f"selector align=1504 exact={np.array_equal(actual,values[indexes])} actual={actual.tolist()}")
  finally:
    dev._gpu_free(out)
    dev._gpu_free(src)

if __name__ == "__main__": main()
