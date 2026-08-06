#!/usr/bin/env python3
"""Compare pointwise, spatial, and hybrid register families for blocked IC16 1x1 CONV.

The raw C1HWC2-looking allocation intentionally records a rejected assumption:
it does not reproduce the compiler selector's legal CNA surface. The decisive
compiler-packed tests patch only the three candidate register families while
retaining normal lowering. All three route IC16/OC16 identity and every
IC16/OC1 one-hot channel exactly, so this probe does not justify a production
register change.
"""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad import Tensor
from tinygrad.codegen import to_program_cache
from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip.image import RKImage, RKStage, encode_image
import tinygrad.renderer.rockchip.emit as rk_emit
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from extra.rockchip.probe_conv_channel_bias import image as base_image

_CNA, _CORE = 0x0201, 0x0801

def patch_registers(image:RKImage, values:dict[tuple[int,int],int]) -> RKImage:
  stages:list[RKStage] = []
  for stage in image.stages:
    commands = tuple(((command & ~0xffffffff0000) | (values[(command>>48,command&0xffff)]<<16))
                     if (command>>48,command&0xffff) in values else command for command in stage.commands)
    stages.append(RKStage(stage.engine,commands,stage.relocs,stage.flags))
  return RKImage(image.target,tuple(stages),image.scratch,image.constants,image.version)

def main() -> None:
  modes = {
    "pointwise": {(_CNA,rk.REG_CNA_CVT_CON0):1, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200},
    "pointwise_mask7": {(_CNA,rk.REG_CNA_CVT_CON0):1, (_CNA,0x1180):7, (_CORE,rk.REG_CORE_MISC_CFG):0x200},
    "spatial": {(_CNA,rk.REG_CNA_CVT_CON0):0xb, (_CNA,0x1180):7, (_CORE,rk.REG_CORE_MISC_CFG):0x201},
    "hybrid": {(_CNA,rk.REG_CNA_CVT_CON0):0xb, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200},
    "hybrid_mask7": {(_CNA,rk.REG_CNA_CVT_CON0):0xb, (_CNA,0x1180):7, (_CORE,rk.REG_CORE_MISC_CFG):0x200},
    "pointwise_cbuf1": {(_CNA,rk.REG_CNA_CVT_CON0):1, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200,
                         (_CNA,rk.REG_CNA_CBUF_CON0):0xb1, (_CNA,rk.REG_CNA_CBUF_CON1):6},
    "hybrid_cbuf1": {(_CNA,rk.REG_CNA_CVT_CON0):0xb, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200,
                      (_CNA,rk.REG_CNA_CBUF_CON0):0xb1, (_CNA,rk.REG_CNA_CBUF_CON1):6},
    "pointwise_grain6": {(_CNA,rk.REG_CNA_CVT_CON0):1, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200,
                          (_CNA,rk.REG_CNA_CONV_CON2):6<<4},
    "hybrid_grain6": {(_CNA,rk.REG_CNA_CVT_CON0):0xb, (_CNA,0x1180):0, (_CORE,rk.REG_CORE_MISC_CFG):0x200,
                       (_CNA,rk.REG_CNA_CONV_CON2):6<<4},
  }
  logical = np.fromfunction(lambda y,x,c: 1000*y+100*x+c,(5,5,16),dtype=np.float32).astype(np.float16)
  packed = np.zeros((2,5,6,8),dtype=np.float16)
  for channel in range(16): packed[channel//8,:,:5,channel%8] = logical[:,:,channel]
  bias = np.zeros(32,dtype=np.float32)
  dev = RockchipDevice("ROCKCHIP")
  try:
    for name,registers in modes.items():
      failures:list[tuple[int,int,float,float]] = []
      for selected in range(16):
        weight = np.zeros((1,1,1,16),dtype=np.float16)
        weight[0,0,0,selected] = 1
        buffers = [dev._gpu_alloc(size) for size in (2*28*8*2,packed.nbytes,weight.nbytes,bias.nbytes)]
        try:
          for buffer,value in zip(buffers[1:],(packed,weight,bias)):
            ctypes.memmove(int(buffer.va_addr),value.ctypes.data,value.nbytes)
          probe = patch_registers(base_image(16,1,False),registers)
          RockchipProgram(dev,TinyELF(encode_image(probe),f"wide_pointwise_{name}_{selected}",Target("ROCKCHIP"),()))(*buffers,wait=True)
          surface = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),2*28*8*2),dtype=np.float16).reshape(2,28,8)
          actual = surface[0,:25,0].reshape(5,5)
          expected = logical[:,:,selected]
          mismatch = np.flatnonzero(actual.view(np.uint16) != expected.view(np.uint16))
          if mismatch.size:
            first = int(mismatch[0])
            failures.append((selected,int(mismatch.size),float(actual.flat[first]),float(expected.flat[first])))
        finally:
          for buffer in buffers: dev._gpu_free(buffer)
      print(f"{name}: exact_channels={16-len(failures)}/16 failures={failures}")
    print("identity OC16:")
    identity_weight = np.zeros((1,1,16,16),dtype=np.float16)
    identity_weight[0,0] = np.eye(16,dtype=np.float16)
    for name,registers in modes.items():
      buffers = [dev._gpu_alloc(size) for size in (2*28*8*2,packed.nbytes,identity_weight.nbytes,bias.nbytes)]
      try:
        for buffer,value in zip(buffers[1:],(packed,identity_weight,bias)):
          ctypes.memmove(int(buffer.va_addr),value.ctypes.data,value.nbytes)
        probe = patch_registers(base_image(16,16,False),registers)
        RockchipProgram(dev,TinyELF(encode_image(probe),f"wide_pointwise_identity_{name}",Target("ROCKCHIP"),()))(*buffers,wait=True)
        surface = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),2*28*8*2),dtype=np.float16).reshape(2,28,8)
        actual = np.stack(tuple(surface[channel//8,:25,channel%8] for channel in range(16)),axis=-1).reshape(5,5,16)
        mismatch = np.flatnonzero(actual.view(np.uint16) != logical.view(np.uint16))
        print(f"  {name}: exact={not mismatch.size} mismatches={mismatch.size}")
      finally:
        for buffer in buffers: dev._gpu_free(buffer)
    print("compiler-packed identity OC16:")
    original_emit = rk_emit.emit_spatial_conv
    nchw = logical.transpose(2,0,1)[None]
    oihw = np.eye(16,dtype=np.float16).reshape(16,16,1,1)
    try:
      for name,registers in modes.items():
        rk_emit.emit_spatial_conv = lambda plan,target,registers=registers: patch_registers(original_emit(plan,target),registers)
        to_program_cache.clear()
        actual = Tensor(nchw,device="ROCKCHIP").conv2d(Tensor(oihw,device="ROCKCHIP")).realize().numpy()
        mismatch = np.flatnonzero(actual.view(np.uint16) != nchw.view(np.uint16))
        print(f"  {name}: exact={not mismatch.size} mismatches={mismatch.size}")
      print("compiler-packed one-hot OC1:")
      for name in ("pointwise","spatial","hybrid"):
        registers = modes[name]
        rk_emit.emit_spatial_conv = lambda plan,target,registers=registers: patch_registers(original_emit(plan,target),registers)
        to_program_cache.clear()
        failures:list[tuple[int,int]] = []
        for selected in range(16):
          one_hot = np.zeros((1,16,1,1),dtype=np.float16)
          one_hot[0,selected,0,0] = 1
          actual = Tensor(nchw,device="ROCKCHIP").conv2d(Tensor(one_hot,device="ROCKCHIP")).realize().numpy()[0,0]
          mismatch = np.flatnonzero(actual.view(np.uint16) != logical[:,:,selected].view(np.uint16))
          if mismatch.size: failures.append((selected,int(mismatch.size)))
        print(f"  {name}: exact_channels={16-len(failures)}/16 failures={failures}")
    finally:
      rk_emit.emit_spatial_conv = original_emit
  finally:
    del dev

if __name__ == "__main__": main()
