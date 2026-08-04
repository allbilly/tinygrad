#!/usr/bin/env python3
"""Characterize partial-channel and one-dimensional RK3588 PPU sliding MAX."""
from __future__ import annotations
import ctypes
import argparse
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKLayout, RKLayoutKind, RKPool, RKTarget, RKTensorRef,
  encode_image, emit_pool)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

_PPU, _PPU_RDMA = 0x4001, 0x8001

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image(ih:int, iw:int, channels:int, kh:int, kw:int, sy:int, sx:int) -> RKImage:
  """Start from a proven HWC8 task, then alter only characterized PPU geometry fields."""
  base_kh, base_kw = min(2,ih), min(8,iw)
  base_oh, base_ow = ih-base_kh+1, iw-base_kw+1
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),
    RKLayout((ih,iw,8),(ih,iw,8),(iw*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),
    RKLayout((base_oh,base_ow,8),(base_oh,base_ow,8),(base_ow*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC))
  stage = emit_pool(RKPool(out,src,Ops.MAX,0,base_kh,base_kw,1,1)).stages[0]
  oh, ow, c = (ih-kh)//sy+1, (iw-kw)//sx+1, channels-1
  line_stride, output_index_add = iw*channels*2, iw*oh
  fields = ((_PPU,rk.REG_PPU_DATA_CUBE_IN_WIDTH,iw-1),(_PPU,rk.REG_PPU_DATA_CUBE_IN_HEIGHT,ih-1),
    (_PPU,rk.REG_PPU_DATA_CUBE_IN_CHANNEL,c),(_PPU,rk.REG_PPU_DATA_CUBE_OUT_WIDTH,ow-1),
    (_PPU,rk.REG_PPU_DATA_CUBE_OUT_HEIGHT,oh-1),(_PPU,rk.REG_PPU_DATA_CUBE_OUT_CHANNEL,c),
    (_PPU,rk.REG_PPU_POOLING_KERNEL_CFG,((sy-1)<<20)|((sx-1)<<16)|((kh-1)<<8)|(kw-1)),
    (_PPU,rk.REG_PPU_DST_SURF_STRIDE,output_index_add),(_PPU,rk.REG_PPU_DATA_FORMAT,(output_index_add<<16)|2),
    (_PPU_RDMA,rk.REG_PPU_RDMA_CUBE_IN_WIDTH,iw-1),(_PPU_RDMA,rk.REG_PPU_RDMA_CUBE_IN_HEIGHT,ih-1),
    (_PPU_RDMA,rk.REG_PPU_RDMA_CUBE_IN_CHANNEL,c),(_PPU_RDMA,rk.REG_PPU_RDMA_SRC_LINE_STRIDE,line_stride),
    (_PPU_RDMA,rk.REG_PPU_RDMA_SRC_SURF_STRIDE,ih*line_stride))
  for target,reg,value in fields: stage = _replace(stage,target,reg,value)
  return RKImage(RKTarget.RK3588,(stage,))

def reference(values:np.ndarray, kh:int, kw:int, sy:int, sx:int) -> np.ndarray:
  oh, ow = (values.shape[0]-kh)//sy+1, (values.shape[1]-kw)//sx+1
  return np.stack(tuple(values[y*sy:y*sy+kh,x*sx:x*sx+kw].max(axis=(0,1))
                        for y in range(oh) for x in range(ow))).reshape(oh,ow,values.shape[2])

_CASES = ((9,13,2,2,2,1,1),(17,2,2,1,2,1,2),(9,17,4,1,16,1,1),
          (9,17,8,1,8,1,1),(256,32,8,1,8,1,8),(256,4,8,1,4,1,4),
          (9,17,8,1,16,1,1),(17,32,2,1,16,1,16))

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--case",type=int,choices=range(len(_CASES)),required=True,
                      help="run one geometry in a fresh process; cases 1, 2, 6, and 7 are known-bad or timeout probes")
  args = parser.parse_args()
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(42)
  for ih,iw,channels,kh,kw,sy,sx in (_CASES[args.case],):
    values = rng.uniform(-8,8,(ih,iw,channels)).astype(np.float16)
    expected = reference(values,kh,kw,sy,sx)
    src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(expected.nbytes)
    try:
      ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
      ctypes.memset(int(out.va_addr),0,expected.nbytes)
      name = f"ppu_max_hwc{channels}_{ih}x{iw}_k{kh}x{kw}_s{sy}x{sx}"
      RockchipProgram(dev,TinyELF(encode_image(image(ih,iw,channels,kh,kw,sy,sx)),name,Target("ROCKCHIP"),()))(out,src,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(out.va_addr),expected.nbytes),dtype=np.float16).copy().reshape(expected.shape)
      print(f"{name} exact={np.array_equal(actual,expected)} mismatches={np.count_nonzero(actual != expected)}/{actual.size} "
            f"actual_head={actual.reshape(-1)[:8].tolist()} expected_head={expected.reshape(-1)[:8].tolist()}")
    finally:
      dev._gpu_free(src)
      dev._gpu_free(out)

if __name__ == "__main__": main()
