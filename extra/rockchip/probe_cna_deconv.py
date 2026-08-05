#!/usr/bin/env python3
"""Probe RK3588 CNA deconvolution stride and kernel orientation from known-good CONV tasks."""
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

def image(kernel_size:int, output_height:int, output_width:int, output_stride:int,
          stride:tuple[int,int], dilation:tuple[int,int]) -> RKImage:
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((2,2,1),(2,2,1),(4,2,2),dtypes.half,
    kind=RKLayoutKind.CNA_ACTIVATION))
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((kernel_size,kernel_size,1,1),(kernel_size,kernel_size,1,8),
    (kernel_size*16,16,16,2),dtypes.half,padding=((0,0),(0,0),(0,0),(0,7)),
    kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((2,output_stride,8),(2,output_stride,8),(output_stride*16,16,2),dtypes.half,
    kind=RKLayoutKind.CONV_OUTPUT))
  # A same-size ordinary task gives the emitter legal base geometry. Full deconvolution needs effective-kernel-1 padding.
  pad_y, pad_x = (kernel_size-1)*dilation[0]//2, (kernel_size-1)*dilation[1]//2
  base = emit_spatial_conv(RKConvTask(out,src,weight,1,1,2,2,kernel_size,kernel_size,2,2,1,1,2,output_stride,
                                     pad_top=pad_y,pad_bottom=pad_y,pad_left=pad_x,pad_right=pad_x,
                                     dilation_y=dilation[0],dilation_x=dilation[1]))
  registers = {(command>>48,command&0xffff):(command>>16)&0xffffffff for command in base.stages[0].commands}
  values = {
    (_CNA,rk.REG_CNA_CONV_CON1):registers[(_CNA,rk.REG_CNA_CONV_CON1)]|(1<<16),
    (_CNA,rk.REG_CNA_CONV_CON3):registers[(_CNA,rk.REG_CNA_CONV_CON3)]|((stride[0]-1)<<11)|((stride[1]-1)<<8),
    (_CNA,rk.REG_CNA_PAD_CON0):((kernel_size-1)*dilation[1]<<4)|((kernel_size-1)*dilation[0]),
    (_CNA,rk.REG_CNA_DATA_SIZE2):output_width,
    (_CNA,rk.REG_CNA_DATA_SIZE3):output_height*output_width,
    (_CORE,rk.REG_CORE_DATAOUT_SIZE_0):((output_height-1)<<16)|(output_width-1),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH):output_width-1,
    (_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT):output_height-1,
    (_DPU,rk.REG_DPU_WDMA_SIZE_1):((output_height-1)<<16)|(output_width-1),
  }
  return RKImage(RKTarget.RK3588,(_mutate(base.stages[0],values),),base.scratch,base.constants)

def _reference(source:np.ndarray, kernel:np.ndarray, stride:tuple[int,int], dilation:tuple[int,int]) -> np.ndarray:
  output = np.zeros(((source.shape[0]-1)*stride[0]+(kernel.shape[0]-1)*dilation[0]+1,
                     (source.shape[1]-1)*stride[1]+(kernel.shape[1]-1)*dilation[1]+1),dtype=np.float16)
  for y in range(source.shape[0]):
    for x in range(source.shape[1]):
      output[y*stride[0]:y*stride[0]+kernel.shape[0]*dilation[0]:dilation[0],
             x*stride[1]:x*stride[1]+kernel.shape[1]*dilation[1]:dilation[1]] += source[y,x]*kernel
  return output

def _run(dev:RockchipDevice, source:np.ndarray, kernel:np.ndarray, stride:tuple[int,int]=(2,2),
         dilation:tuple[int,int]=(1,1)) -> None:
  output_height = (source.shape[0]-1)*stride[0]+(kernel.shape[0]-1)*dilation[0]+1
  output_width = (source.shape[1]-1)*stride[1]+(kernel.shape[1]-1)*dilation[1]+1
  output_stride = (output_height*output_width+3)&-4
  weight = np.zeros((kernel.shape[0],kernel.shape[1],1,8),dtype=np.float16)
  # CNA correlates the zero-inserted feature surface. Spatially reverse weights to implement tinygrad's transpose-convolution contract.
  weight[:,:,0,0] = kernel[::-1,::-1]
  output_nbytes = 2*output_stride*8*2
  expected = _reference(source,kernel,stride,dilation)
  out_buf,src_buf,weight_buf = dev._gpu_alloc(output_nbytes),dev._gpu_alloc(source.nbytes),dev._gpu_alloc(weight.nbytes)
  try:
    ctypes.memset(int(out_buf.va_addr),0,output_nbytes)
    ctypes.memmove(int(src_buf.va_addr),source.ctypes.data,source.nbytes)
    ctypes.memmove(int(weight_buf.va_addr),weight.ctypes.data,weight.nbytes)
    name = f"cna_deconv_{kernel.shape[0]}x{kernel.shape[1]}_s{stride[0]}x{stride[1]}_d{dilation[0]}x{dilation[1]}"
    task = image(kernel.shape[0],output_height,output_width,output_stride,stride,dilation)
    RockchipProgram(dev,TinyELF(encode_image(task),name,Target("ROCKCHIP"),()))(
      out_buf,src_buf,weight_buf,wait=True)
    physical = np.frombuffer(ctypes.string_at(int(out_buf.va_addr),output_nbytes),dtype=np.float16).reshape(2,output_stride,8)
    actual = physical[0,:output_height*output_width,0].reshape(output_height,output_width)
    flipped = _reference(source,kernel[::-1,::-1],stride,dilation)
    print(f"CNA deconv {kernel.shape[0]}x{kernel.shape[1]} stride={stride} dilation={dilation} "
          f"direct={np.array_equal(actual,expected)} "
          f"flipped={np.array_equal(actual,flipped)} actual={actual.tolist()}")
    np.testing.assert_equal(actual,expected)
  finally:
    dev._gpu_free(out_buf)
    dev._gpu_free(src_buf)
    dev._gpu_free(weight_buf)

def main() -> None:
  dev = RockchipDevice("ROCKCHIP")
  source = np.array([[1,2],[3,4]],dtype=np.float16)
  _run(dev,source,np.array([[1]],dtype=np.float16))
  kernel = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.float16)
  for stride,dilation in (((1,1),(1,1)),((2,1),(1,1)),((1,2),(1,1)),((2,2),(1,1)),((1,1),(2,2))):
    _run(dev,source,kernel,stride,dilation)

if __name__ == "__main__": main()
