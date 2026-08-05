#!/usr/bin/env python3
"""Reject a channel-packed FP16-mask to public-int8 DPU conversion."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip import (RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKImage, RKMaskStage,
  encode_image, emit_dpu)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

_DPU, _RDMA = 0x1001, 0x2001

def _command(target:int, reg:int, value:int) -> int:
  return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)

def _replace(commands:tuple[int, ...], target:int, reg:int, value:int) -> tuple[int, ...]:
  return tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in commands)

def mask_image() -> RKImage:
  """Pack sixteen FP16 mask lanes into one sixteen-byte public-int8 atom."""
  base = emit_dpu(RKDPUProgram((RKMaskStage(RKArg(RKBufferKind.ARG,0),RKArg(RKBufferKind.ARG,1),16),)))
  stage, commands = base.stages[0], base.stages[0].commands
  for target,reg,value in (
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<26)|2),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0xf000f),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,15),
    (_DPU,rk.REG_DPU_WDMA_SIZE_1,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SCALE,1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,15),
  ): commands = _replace(commands,target,reg,value)
  return RKImage(base.target,(RKStage(stage.engine,commands,stage.relocs,stage.flags),),base.scratch,base.constants)

def main() -> None:
  dev = RockchipDevice("ROCKCHIP")
  output, source = dev._gpu_alloc(64), dev._gpu_alloc(32)
  values = np.array([0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,0],dtype=np.float16)
  try:
    ctypes.memset(int(output.va_addr),0x5a,64)
    ctypes.memmove(int(source.va_addr),values.ctypes.data,values.nbytes)
    try:
      RockchipProgram(dev,TinyELF(encode_image(mask_image()),"fp16_mask_to_int8_channel",Target("ROCKCHIP"),()))(
        output,source,wait=True)
    except TimeoutError as error:
      print(f"channel-packed FP16-to-int8 rejected with {error!r}")
    else: raise AssertionError("channel-packed FP16-to-int8 unexpectedly completed")

    recovery = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD,RKArg(RKBufferKind.ARG,0),0.0,3.0,32),)))
    RockchipProgram(dev,TinyELF(encode_image(recovery),"fp16_fill_recovery",Target("ROCKCHIP"),()))(output,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(output.va_addr),64),dtype=np.float16).copy()
    print(f"recovery_exact={np.array_equal(actual,np.full(32,3,dtype=np.float16))}")
    assert np.array_equal(actual,np.full(32,3,dtype=np.float16))
  finally:
    dev._gpu_free(output)
    dev._gpu_free(source)

if __name__ == "__main__": main()
