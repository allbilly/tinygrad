#!/usr/bin/env python3
"""Probe RK3588 PPU global-MAX legality below the currently proven HWC8 format."""
from __future__ import annotations
import ctypes, os
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKImage, RKLayout, RKReduce,
  RKScratch, RKTarget, RKTensorRef, encode_image, emit_dpu, emit_reduce)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

_PPU, _RDMA = 0x4001, 0x8001

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine, commands, stage.relocs, stage.flags)

def image(channels:int, nonalign:bool=False) -> RKImage:
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1), RKLayout((16,16,8),(16,16,8),(256,16,2),dtypes.half))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0), RKLayout((1,1,8),(1,1,8),(16,16,2),dtypes.half))
  base = emit_reduce(RKReduce(out,src,Ops.MAX,0))
  stage = base.stages[0]
  for target,reg,value in ((_PPU,rk.REG_PPU_DATA_CUBE_IN_CHANNEL,channels-1),
                           (_PPU,rk.REG_PPU_DATA_CUBE_OUT_CHANNEL,channels-1),
                           (_PPU,rk.REG_PPU_MISC_CTRL,0x83 if nonalign else 3),
                           (_RDMA,rk.REG_PPU_RDMA_CUBE_IN_CHANNEL,channels-1),
                           (_RDMA,rk.REG_PPU_RDMA_SRC_LINE_STRIDE,16*channels*2),
                           (_RDMA,rk.REG_PPU_RDMA_SRC_SURF_STRIDE,16*16*channels*2)):
    stage = _replace(stage,target,reg,value)
  return RKImage(RKTarget.RK3588,(stage,))

def main() -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(23)
  # NONALIGN timed out for HWC1. Keep it opt-in so the default probe is a safe reproducible negative result.
  modes = (False,True) if os.getenv("ROCKCHIP_UNSAFE_NONALIGN") == "1" else (False,)
  for nonalign in modes:
    for channels in (1,2,4,8):
      values = rng.uniform(-4,4,(16,16,channels)).astype(np.float16)
      src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(16)
      try:
        ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
        ctypes.memset(int(out.va_addr),0,16)
        name = f"ppu_hwc{channels}_{'nonalign' if nonalign else 'aligned'}"
        program = RockchipProgram(dev,TinyELF(encode_image(image(channels,nonalign)),name,Target("ROCKCHIP"),()))
        program(out,src,wait=True)
        actual = np.frombuffer(ctypes.string_at(int(out.va_addr),16),dtype=np.float16).copy()[:channels]
        expected = values.max(axis=(0,1))
        print(f"HWC{channels} nonalign={nonalign} exact={np.array_equal(actual,expected)} actual={actual.tolist()} expected={expected.tolist()}")
      finally:
        dev._gpu_free(src)
        dev._gpu_free(out)

  # Test whether DPU source addends can reduce each eight-lane atom without first transposing it.
  atoms = rng.uniform(-4,4,(16,)).astype(np.float16)
  src, out = dev._gpu_alloc(atoms.nbytes), dev._gpu_alloc(atoms.nbytes)
  try:
    ctypes.memmove(int(src.va_addr),atoms.ctypes.data,atoms.nbytes)
    scratch = (RKScratch(atoms.nbytes),RKScratch(atoms.nbytes))
    stages = (RKALUStage(Ops.MAX,RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.ARG,1),RKArg(RKBufferKind.ARG,1,2),16),
      RKALUStage(Ops.MAX,RKArg(RKBufferKind.SCRATCH,1),RKArg(RKBufferKind.SCRATCH,0),RKArg(RKBufferKind.SCRATCH,0,4),16),
      RKALUStage(Ops.MAX,RKArg(RKBufferKind.ARG,0),RKArg(RKBufferKind.SCRATCH,1),RKArg(RKBufferKind.SCRATCH,1,8),16))
    program = RockchipProgram(dev,TinyELF(encode_image(emit_dpu(RKDPUProgram(stages,scratch))),"dpu_atom_max",Target("ROCKCHIP"),()))
    program(out,src,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(out.va_addr),atoms.nbytes),dtype=np.float16).copy()[::8]
    expected = atoms.reshape(-1,8).max(axis=1)
    print(f"unaligned DPU atom MAX exact={np.array_equal(actual,expected)} actual={actual.tolist()} expected={expected.tolist()}")
  finally:
    dev._gpu_free(src)
    dev._gpu_free(out)

if __name__ == "__main__": main()
