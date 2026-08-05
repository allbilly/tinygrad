#!/usr/bin/env python3
"""Reject FP32-accumulator scaling through the RK3588 DPU FP32-to-FP16 output converter."""
from __future__ import annotations
import ctypes, struct
from dataclasses import replace
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKCMACTask, RKEpilogue, RKLayout, RKLayoutKind,
  RKTensorRef, encode_image, emit_cmac_task)
from tinygrad.renderer.rockchip.image import RKImage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def _replace_register(image:RKImage, reg:int, value:int) -> RKImage:
  commands = tuple(((command & ~0xffffffff0000) | ((value & 0xffffffff) << 16))
                   if command >> 48 == 0x1001 and command & 0xffff == reg else command for command in image.stages[0].commands)
  return replace(image, stages=(replace(image.stages[0], commands=commands),))

def _set_register(image:RKImage, reg:int, value:int) -> RKImage:
  if any(command >> 48 == 0x1001 and command & 0xffff == reg for command in image.stages[0].commands):
    return _replace_register(image,reg,value)
  commands = list(image.stages[0].commands)
  trigger = next(index for index,command in enumerate(commands) if command >> 48 == 0x81)
  commands.insert(trigger,(0x1001 << 48) | ((value & 0xffffffff) << 16) | reg)
  return replace(image,stages=(replace(image.stages[0],commands=tuple(commands)),))

def image(scale:int, shift:int) -> RKImage:
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((1,32),(1,64),(128,2),dtypes.half,padding=((0,0),(0,32))))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((1,32),(1,32),(64,2),dtypes.half))
  rhs = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((32,32),(32,32),(64,2),dtypes.half,kind=RKLayoutKind.CMAC_WEIGHT))
  bias = RKTensorRef(RKArg(RKBufferKind.ARG,3),RKLayout((32,),(32,),(4,),dtypes.float))
  result = emit_cmac_task(RKCMACTask(out,lhs,rhs,0,epilogue=RKEpilogue(bias)))
  result = _replace_register(result, rk.REG_DPU_OUT_CVT_SCALE, 0x10000 | scale)
  return _replace_register(result, rk.REG_DPU_OUT_CVT_SHIFT, shift)

def factored_image(bs:float, bn:float) -> RKImage:
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((1,32),(1,32),(64,2),dtypes.half))
  lhs = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((1,32),(1,32),(64,2),dtypes.half))
  rhs = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((32,32),(32,32),(64,2),dtypes.half,kind=RKLayoutKind.CMAC_WEIGHT))
  result = emit_cmac_task(RKCMACTask(out,lhs,rhs,0,compact_output=True))
  for prefix,factor in (("BS",bs),("BN",bn)):
    result = _set_register(result,getattr(rk,f"REG_DPU_{prefix}_CFG"),0x20040)
    result = _set_register(result,getattr(rk,f"REG_DPU_{prefix}_ALU_CFG"),0x80000000)
    result = _set_register(result,getattr(rk,f"REG_DPU_{prefix}_MUL_CFG"),struct.unpack("<H",struct.pack("<e",factor))[0]<<16)
  return result

def main() -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608)
  lhs = rng.uniform(-2,2,32).astype(np.float16)
  weights = rng.uniform(-1,1,(32,32)).astype(np.float16)
  packed = weights.reshape(2,16,1,32).transpose(0,2,1,3).ravel()
  bias = rng.uniform(-4,4,32).astype(np.float32)
  buffers = [dev._gpu_alloc(size) for size in (128,64,packed.nbytes,128)]
  variants = (("unit",1,0),("integer_shift3",1,3),("integer_ninth",29127,18),
              ("minus_exp3",1,3 << 12),("typed_shift3",1,(1 << 31) | 3),
              ("typed_ninth",29127,(1 << 31) | 18),("typed_minus_exp3",1,(1 << 31) | (3 << 12)))
  try:
    for buf,value in zip(buffers[1:],(lhs,packed,bias)): ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
    outputs = []
    for name,scale,shift in variants:
      ctypes.memset(int(buffers[0].va_addr),0,128)
      RockchipProgram(dev,TinyELF(encode_image(image(scale,shift)),f"cmac_fp16_cvt_{name}",Target("ROCKCHIP"),()))(*buffers,wait=True)
      outputs.append(np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),64),dtype=np.float16).copy())
      print(f"{name}: {outputs[-1][:8].tolist()}")
    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])
    print("all converter scale/shift variants produced the unit-scale output")

    factors = (0.1265869140625,0.1339111328125,6.5546875)
    scaled_weights = np.full((32,32),factors[0],dtype=np.float16).reshape(2,16,1,32).transpose(0,2,1,3).ravel()
    ctypes.memmove(int(buffers[2].va_addr),scaled_weights.ctypes.data,scaled_weights.nbytes)
    ctypes.memset(int(buffers[0].va_addr),0,128)
    RockchipProgram(dev,TinyELF(encode_image(factored_image(*factors[1:])),"cmac_three_factor_ninth",Target("ROCKCHIP"),()))(
      *buffers,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),64),dtype=np.float16).copy()
    expected = np.full(32,np.float16(np.sum(lhs.astype(np.float32))*np.float32(1/9)),dtype=np.float16)
    print(f"three_factor_ninth exact={np.array_equal(actual,expected)} actual={actual[:8].tolist()} expected={expected[:8].tolist()}")
  finally:
    for buf in buffers: dev._gpu_free(buf)

if __name__ == "__main__": main()
