#!/usr/bin/env python3
"""Probe a full FP16 ERDMA surface as a flying-CONV elementwise ADD operand."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKConvTask, RKImage, RKLayout, RKLayoutKind, RKReloc, RKTarget,
  RKTensorRef, encode_image, emit_spatial_conv)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x81
_EW_SURF_STRIDE = 0x5040
def _command(target:int, reg:int, value:int) -> int:
  return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)

def _replace(commands:tuple[int,...], target:int, reg:int, value:int) -> tuple[int,...]:
  return tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in commands)

def image(k:int, n:int) -> RKImage:
  output_width_stride = (n+3)&-4
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((k,n,1),(k,n,1),(n*2,2,2),dtypes.half,
    kind=RKLayoutKind.CNA_ACTIVATION))
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((k,1,1,1),(k,1,1,8),(16,16,16,2),dtypes.half,
    padding=((0,0),(0,0),(0,0),(0,7)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((2,output_width_stride,8),(2,output_width_stride,8),
    (output_width_stride*16,16,2),dtypes.half,kind=RKLayoutKind.CONV_OUTPUT))
  base = emit_spatial_conv(RKConvTask(out,src,weight,1,1,k,n,k,1,1,n,1,1,n,output_width_stride))
  commands = list(base.stages[0].commands)
  if commands[-1]>>48 != _PC: raise RuntimeError("CONV operation enable is not last")
  commands.pop()
  commands = list(_replace(tuple(commands),_DPU,rk.REG_DPU_EW_CFG,0x108202c0))
  commands.extend(_command(_RDMA,reg,value) for reg,value in (
    (rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,2*n-1),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,15),
    (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x40000008),
    (_EW_SURF_STRIDE,output_width_stride*16),
    (rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,0x17850)))
  operand_word = len(commands)
  commands.append(_command(_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,0))
  commands.append(_command(_PC,rk.REG_PC_OPERATION_ENABLE,0x1d))
  stage = RKStage(base.stages[0].engine,tuple(commands),
    (*base.stages[0].relocs,RKReloc(0,operand_word,RKBufferKind.ARG,3)),base.stages[0].flags)
  return RKImage(RKTarget.RK3588,(stage,))

def main() -> None:
  dev, rng, k, n = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608), 16, 8
  matrix = rng.uniform(-2,2,(k,n)).astype(np.float16)
  vector = rng.uniform(-2,2,k).astype(np.float16)
  packed_weight = np.zeros((k,1,1,8),dtype=np.float16)
  packed_weight[:,0,0,0] = vector
  prior = np.zeros(2*((n+3)&-4)*8,dtype=np.float16)
  prior[::16][:n] = rng.uniform(-2,2,n).astype(np.float16)
  buffers = [dev._gpu_alloc(size) for size in (prior.nbytes,matrix.nbytes,packed_weight.nbytes,prior.nbytes)]
  try:
    for buf,value in zip(buffers[1:],(matrix,packed_weight,prior)): ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
    ctypes.memset(int(buffers[0].va_addr),0,prior.nbytes)
    RockchipProgram(dev,TinyELF(encode_image(image(k,n)),"conv_erdma_accumulate",Target("ROCKCHIP"),()))(*buffers,wait=True)
    raw = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),prior.nbytes),dtype=np.float16).copy()
    actual = raw[::8][:n]
    expected = (vector.astype(np.float32)@matrix.astype(np.float32)+prior[::16][:n]).astype(np.float16)
    print(f"max_abs={np.max(np.abs(actual-expected))} actual={actual.tolist()}")
    print(f"expected={expected.tolist()}")
    assert np.array_equal(actual,expected)
  finally:
    for buf in buffers: dev._gpu_free(buf)

if __name__ == "__main__": main()
