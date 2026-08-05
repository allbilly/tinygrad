#!/usr/bin/env python3
"""Characterize DPU BS/BN/EW channel operands across a padded spatial surface.

This is a hardware probe, not a compiler path.  The main and output tensors use
four physical pixels with one complete eight-channel FP16 atom per pixel.  Each
external operand has a live first atom followed by deliberately different
poison atoms, so the result distinguishes per-channel reuse from per-pixel
consumption without reading outside an allocation.
"""
from __future__ import annotations
import ctypes
from dataclasses import replace
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKDPUProgram, RKFusedMulStage, RKImage, RKTarget, encode_image, emit_dpu
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU, _RDMA, _EW_SURF_STRIDE = 0x1001, 0x2001, 0x5040

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _set(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = list(stage.commands)
  for index, command in enumerate(commands):
    if command>>48 == target and command&0xffff == reg:
      commands[index] = _command(target,reg,value)
      return RKStage(stage.engine,tuple(commands),stage.relocs,stage.flags)
  trigger = next(index for index,command in enumerate(commands)
                 if command>>48 == 0x81 and command&0xffff == rk.REG_PC_OPERATION_ENABLE)
  commands.insert(trigger,_command(target,reg,value))
  relocs = tuple(replace(reloc,word=reloc.word+(reloc.word >= trigger)) for reloc in stage.relocs)
  return RKStage(stage.engine,tuple(commands),relocs,stage.flags)

def image(spatial:int, channels:int, channel_operand:str) -> RKImage:
  args = tuple(RKArg(RKBufferKind.ARG,index) for index in range(5))
  base = emit_dpu(RKDPUProgram((RKFusedMulStage(*args,channels),)))
  stage = base.stages[0]
  for target,reg,value in (
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,spatial-1), (_DPU,rk.REG_DPU_WDMA_SIZE_1,spatial-1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,spatial-1)):
    stage = _set(stage,target,reg,value)
  if channel_operand == "ew":
    # TRM: ERDMA data mode 0 is per-channel; EW core must use the matching mode,
    # and EW_SURF_STRIDE must be one in per-channel mode.
    stage = _set(stage,_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x00000008)
    stage = _set(stage,_RDMA,_EW_SURF_STRIDE,1)
    ew_cfg = next(command>>16 & 0xffffffff for command in stage.commands
                  if command>>48 == _DPU and command&0xffff == rk.REG_DPU_EW_CFG)
    stage = _set(stage,_DPU,rk.REG_DPU_EW_CFG,ew_cfg & ~0x30000000)
  return RKImage(RKTarget.RK3588,(stage,),base.scratch,base.constants)

def main() -> None:
  dev = RockchipDevice("ROCKCHIP")
  results:dict[tuple[int,int,str],bool] = {}
  try:
    for spatial,channels in ((1,8),(4,8),(16,8),(4,16),(4,64),(4,256)):
      main_value = (np.arange(spatial*channels,dtype=np.float32).reshape(spatial,channels)/256+1).astype(np.float16)
      live = np.linspace(np.float16(0.5),np.float16(1.25),channels,dtype=np.float16)
      poison = np.stack(tuple(live+2*row for row in range(spatial))).astype(np.float16)
      ones = np.ones_like(main_value)
      buffers = [dev._gpu_alloc(main_value.nbytes) for _ in range(5)]
      try:
        ctypes.memmove(int(buffers[1].va_addr),main_value.ctypes.data,main_value.nbytes)
        for candidate,slot in (("bs",2),("bn",3),("ew",4)):
          operands = [ones.copy(),ones.copy(),ones.copy()]
          operands[slot-2] = poison
          for buf,value in zip(buffers[2:],operands): ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
          ctypes.memset(int(buffers[0].va_addr),0xa5,main_value.nbytes)
          program = RockchipProgram(dev,TinyELF(encode_image(image(spatial,channels,candidate)),
            f"dpu_{candidate}_channel_broadcast_{spatial}x{channels}",Target("ROCKCHIP"),()))
          program(*buffers,wait=True)
          actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),main_value.nbytes),dtype=np.float16).reshape(spatial,channels).copy()
          repeated = (main_value.astype(np.float32)*live.astype(np.float32)).astype(np.float16)
          per_pixel = (main_value.astype(np.float32)*poison.astype(np.float32)).astype(np.float16)
          exact, pixel = np.array_equal(actual.view(np.uint16),repeated.view(np.uint16)), \
                         np.array_equal(actual.view(np.uint16),per_pixel.view(np.uint16))
          results[(spatial,channels,candidate)] = exact
          mismatch = np.flatnonzero(actual.view(np.uint16) != repeated.view(np.uint16))
          print(f"{spatial}x{channels} {candidate}: repeated={exact} per_pixel={pixel} mismatches={mismatch.size} "
                f"first={mismatch[:8].tolist()}")
      finally:
        for buf in buffers: dev._gpu_free(buf)
  finally:
    del dev
  assert all(results[(spatial,8,candidate)] for spatial in (1,4,16) for candidate in ("bs","bn","ew"))
  assert not any(results[(4,channels,candidate)] for channels in (16,64,256) for candidate in ("bs","bn","ew"))

if __name__ == "__main__": main()
