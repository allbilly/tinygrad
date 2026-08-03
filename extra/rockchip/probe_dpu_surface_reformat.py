#!/usr/bin/env python3
"""Probe strided scalar MRDMA gathers for native dynamic CMAC-weight packing."""
from __future__ import annotations
import ctypes, os
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip import (RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKImage, RKTarget,
  encode_image, emit_dpu)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

_DPU, _RDMA = 0x1001, 0x2001
_RDMA_SRC_DMA_CFG, _RDMA_SURF_NOTCH = 0x5048, 0x504c

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image(rows:int, stride:int, column:int, line_notch:int, nonalign:bool=False) -> RKImage:
  base = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD,RKArg(RKBufferKind.ARG,0),
    RKArg(RKBufferKind.ARG,1,column*2),0.0,rows),)))
  stage = base.stages[0]
  regs = ((_DPU,rk.REG_DPU_FEATURE_MODE_CFG,0x1e5 | ((1<<25) if nonalign else 0)),
    (_DPU,rk.REG_DPU_DST_SURF_STRIDE,0x10), (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,rows-1), (_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0),
    (_DPU,rk.REG_DPU_WDMA_SIZE_0,0), (_DPU,rk.REG_DPU_WDMA_SIZE_1,(rows-1)<<16),
    (_DPU,rk.REG_DPU_SURFACE_ADD,0x10), (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,rows-1), (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,0),
    (_RDMA,_RDMA_SRC_DMA_CFG,line_notch<<19), (_RDMA,_RDMA_SURF_NOTCH,stride*2))
  for target,reg,value in regs: stage = _replace(stage,target,reg,value)
  return RKImage(RKTarget.RK3588,(stage,),base.scratch,base.constants)

def main() -> None:
  rows, stride, column = 8, 5, 2
  values = np.arange(rows*stride,dtype=np.float16).reshape(rows,stride)
  dev = RockchipDevice("ROCKCHIP")
  src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(256)
  try:
    ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
    modes = (False,True) if os.getenv("ROCKCHIP_UNSAFE_NONALIGN") == "1" else (False,)
    for nonalign in modes:
      for notch in (0,stride-1,stride):
        ctypes.memset(int(out.va_addr),0,256)
        program = RockchipProgram(dev,TinyELF(encode_image(image(rows,stride,column,notch,nonalign)),
          f"dpu_gather_n{notch}_{'nonalign' if nonalign else 'aligned'}",Target("ROCKCHIP"),()))
        program(out,src,wait=True)
        actual = np.frombuffer(ctypes.string_at(int(out.va_addr),256),dtype=np.float16).copy()
        expected = values[:,column]
        positions = [int(index) for index in np.flatnonzero(np.isin(actual,expected))]
        print(f"line_notch={notch} nonalign={nonalign} expected={expected.tolist()} positions={positions} first64={actual[:64].tolist()}")
  finally:
    dev._gpu_free(src)
    dev._gpu_free(out)

if __name__ == "__main__": main()
