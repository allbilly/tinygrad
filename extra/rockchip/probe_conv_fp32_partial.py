#!/usr/bin/env python3
"""Probe tall one-channel CNA convolution with FP32 DPU writeback."""
from __future__ import annotations
import ctypes, os
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKConvTask, RKImage, RKLayout, RKLayoutKind, RKReloc, RKTarget,
  RKTensorRef, encode_image, emit_spatial_conv)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x81
_BRDMA_CFG, _BS_BASE = 0x501c, 0x5020
def _command(target:int, reg:int, value:int) -> int:
  return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _replace(stage:RKStage, reg:int, value:int) -> RKStage:
  commands = tuple(_command(_DPU,reg,value) if command>>48 == _DPU and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image(k:int, n:int, accumulate:bool=False) -> RKImage:
  width_stride, output_width_stride = n, (n+3)&-4
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((k,n,1),(k,n,1),(n*2,2,2),dtypes.half,
    kind=RKLayoutKind.CNA_ACTIVATION))
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((k,1,1,1),(k,1,1,8),(16,16,16,2),dtypes.half,
    padding=((0,0),(0,0),(0,0),(0,7)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((2,output_width_stride,8),(2,output_width_stride,8),
    (output_width_stride*16,16,2),dtypes.half,kind=RKLayoutKind.CONV_OUTPUT))
  base = emit_spatial_conv(RKConvTask(out,src,weight,1,1,k,n,k,1,1,n,1,1,width_stride,output_width_stride))
  stage = _replace(base.stages[0],rk.REG_DPU_DATA_FORMAT,(5<<29)|(2<<26)|2)
  stage = _replace(stage,rk.REG_DPU_BS_OW_CFG,0x36e)
  stage = _replace(stage,rk.REG_DPU_OUT_CVT_SCALE,0)
  if accumulate:
    commands = list(stage.commands)
    if commands[-1]>>48 != _PC: raise RuntimeError("CONV operation enable is not last")
    commands.pop()
    stage = RKStage(stage.engine,tuple(commands),stage.relocs,stage.flags)
    stage = _replace(stage,rk.REG_DPU_BS_CFG,0x20150)
    commands = list(stage.commands)
    commands.extend(_command(_DPU,reg,value) for reg,value in ((rk.REG_DPU_BS_ALU_CFG,0),(rk.REG_DPU_BS_MUL_CFG,0)))
    commands.extend(_command(_RDMA,reg,value) for reg,value in (
      (rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,n-1),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
       int(os.getenv("ROCKCHIP_CONV_ACCUM_CHANNELS","16"))-1),(_BRDMA_CFG,2)))
    bias_word = len(commands); commands.append(_command(_RDMA,_BS_BASE,0))
    commands.extend(_command(_RDMA,reg,value) for reg,value in (
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,1),(rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,0x2f850),(rk.REG_DPU_RDMA_RDMA_WEIGHT,0x01010101)))
    commands.append(_command(_PC,rk.REG_PC_OPERATION_ENABLE,0x1d))
    stage = RKStage(stage.engine,tuple(commands),(*stage.relocs,RKReloc(0,bias_word,RKBufferKind.ARG,3)),stage.flags)
  return RKImage(RKTarget.RK3588,(stage,))

def main() -> None:
  dev, rng, k, n = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608), 16, 8
  matrix = rng.uniform(-2,2,(k,n)).astype(np.float16)
  vectors = [rng.uniform(-2,2,k).astype(np.float16) for _ in range(2)]
  packed = [np.zeros((k,1,1,8),dtype=np.float16) for _ in vectors]
  for storage,vector in zip(packed,vectors): storage[:,0,0,0] = vector
  buffers = [dev._gpu_alloc(size) for size in (4096,matrix.nbytes,packed[0].nbytes,4096)]
  try:
    ctypes.memmove(int(buffers[1].va_addr),matrix.ctypes.data,matrix.nbytes)
    ctypes.memmove(int(buffers[2].va_addr),packed[0].ctypes.data,packed[0].nbytes)
    ctypes.memset(int(buffers[0].va_addr),0,4096)
    RockchipProgram(dev,TinyELF(encode_image(image(k,n)),"conv_fp32_partial",Target("ROCKCHIP"),()))(*buffers,wait=True)
    raw = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),1024),dtype=np.float32).copy()
    expected = vectors[0].astype(np.float32)@matrix.astype(np.float32)
    print(f"expected={expected.tolist()}")
    print(f"raw_f32={raw[:64].tolist()}")
    for stride in (8,16,32): print(f"stride{stride}={raw[::stride][:n].tolist()}")
    # Opt-in rejected experiment. With 16 channels BRDMA broadcasts the first FP32 partial to every spatial output;
    # 4/8-channel geometries time out. This is not enabled by any compiler path.
    if os.getenv("ROCKCHIP_UNSAFE_CONV_ACCUM") == "1":
      ctypes.memmove(int(buffers[2].va_addr),packed[1].ctypes.data,packed[1].nbytes)
      ctypes.memset(int(buffers[3].va_addr),0,4096)
      RockchipProgram(dev,TinyELF(encode_image(image(k,n,True)),"conv_fp32_accumulate",Target("ROCKCHIP"),()))(
        buffers[3],buffers[1],buffers[2],buffers[0],wait=True)
      accumulated = np.frombuffer(ctypes.string_at(int(buffers[3].va_addr),1024),dtype=np.float32).copy()[::4][:n]
      expected_accum = expected + vectors[1].astype(np.float32)@matrix.astype(np.float32)
      print(f"accumulate max_abs={np.max(np.abs(accumulated-expected_accum))} actual={accumulated.tolist()}")
      print(f"accumulate expected={expected_accum.tolist()}")
  finally:
    for buf in buffers: dev._gpu_free(buf)

if __name__ == "__main__": main()
