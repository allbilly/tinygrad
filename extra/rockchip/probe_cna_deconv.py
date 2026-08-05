#!/usr/bin/env python3
"""Probe the RK3588 CNA deconvolution stride fields from a known-good 1x1 CONV task."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKConvTask, RKImage, RKLayout, RKLayoutKind, RKTarget,
  RKTensorRef, emit_spatial_conv, encode_image)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_CNA, _CORE, _DPU = 0x201, 0x801, 0x1001

def _command(target:int, reg:int, value:int) -> int:
  return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)

def _mutate(stage:RKStage, values:dict[tuple[int,int],int]) -> RKStage:
  commands = tuple(_command(target,reg,values.get((target,reg),value))
                   for command in stage.commands
                   for target,value,reg in ((command>>48,(command>>16)&0xffffffff,command&0xffff),))
  missing = set(values)-{(command>>48,command&0xffff) for command in stage.commands}
  if missing: raise ValueError(f"registers absent from base CONV task: {sorted(missing)}")
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image() -> RKImage:
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((2,2,1),(2,2,1),(4,2,2),dtypes.half,
    kind=RKLayoutKind.CNA_ACTIVATION))
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((1,1,1,1),(1,1,1,8),(16,16,16,2),dtypes.half,
    padding=((0,0),(0,0),(0,0),(0,7)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((2,12,8),(2,12,8),(192,16,2),dtypes.half,
    kind=RKLayoutKind.CONV_OUTPUT))
  base = emit_spatial_conv(RKConvTask(out,src,weight,1,1,2,2,1,1,2,2,1,1,2,12))
  registers = {(command>>48,command&0xffff):(command>>16)&0xffffffff for command in base.stages[0].commands}
  oh = ow = 3
  values = {
    (_CNA,rk.REG_CNA_CONV_CON1):registers[(_CNA,rk.REG_CNA_CONV_CON1)]|(1<<16),
    (_CNA,rk.REG_CNA_CONV_CON3):registers[(_CNA,rk.REG_CNA_CONV_CON3)]|(1<<11)|(1<<8),
    (_CNA,rk.REG_CNA_DATA_SIZE2):ow,
    (_CNA,rk.REG_CNA_DATA_SIZE3):oh*ow,
    (_CORE,rk.REG_CORE_DATAOUT_SIZE_0):((oh-1)<<16)|(ow-1),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH):ow-1,
    (_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT):oh-1,
    (_DPU,rk.REG_DPU_WDMA_SIZE_1):((oh-1)<<16)|(ow-1),
  }
  return RKImage(RKTarget.RK3588,(_mutate(base.stages[0],values),),base.scratch,base.constants)

def main() -> None:
  source = np.array([1,2,3,4],dtype=np.float16)
  weight = np.zeros(8,dtype=np.float16)
  weight[0] = 1
  output = np.zeros(2*12*8,dtype=np.float16)
  expected = np.zeros((3,3),dtype=np.float16)
  expected[::2,::2] = source.reshape(2,2)
  dev = RockchipDevice("ROCKCHIP")
  out_buf,src_buf,weight_buf = dev._gpu_alloc(output.nbytes),dev._gpu_alloc(source.nbytes),dev._gpu_alloc(weight.nbytes)
  try:
    ctypes.memset(int(out_buf.va_addr),0,output.nbytes)
    ctypes.memmove(int(src_buf.va_addr),source.ctypes.data,source.nbytes)
    ctypes.memmove(int(weight_buf.va_addr),weight.ctypes.data,weight.nbytes)
    RockchipProgram(dev,TinyELF(encode_image(image()),"cna_deconv_1x1_s2",Target("ROCKCHIP"),()))(
      out_buf,src_buf,weight_buf,wait=True)
    physical = np.frombuffer(ctypes.string_at(int(out_buf.va_addr),output.nbytes),dtype=np.float16).reshape(2,12,8)
    actual = physical[0,:9,0].reshape(3,3)
    print(f"CNA deconv 1x1 stride2 exact={np.array_equal(actual,expected)} actual={actual.tolist()}")
    np.testing.assert_equal(actual,expected)
  finally:
    dev._gpu_free(out_buf)
    dev._gpu_free(src_buf)
    dev._gpu_free(weight_buf)

if __name__ == "__main__": main()
