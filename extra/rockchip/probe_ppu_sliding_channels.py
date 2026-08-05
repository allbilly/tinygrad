#!/usr/bin/env python3
"""Characterize partial-channel RK3588 PPU sliding MAX and average modes."""
from __future__ import annotations
import ctypes, struct
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
_REG_PPU_RECIP_KERNEL_WIDTH, _REG_PPU_RECIP_KERNEL_HEIGHT, _REG_PPU_POOLING_PADDING_CFG = 0x6038, 0x603c, 0x6040

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target,reg,value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine,commands,stage.relocs,stage.flags)

def image(ih:int, iw:int, channels:int, kh:int, kw:int, sy:int, sx:int,
          padding:tuple[int,int,int,int]=(0,0,0,0), average:bool=False, reciprocal:int|None=None, proc_precision:int=2) -> RKImage:
  """Start from a proven HWC8 task, then alter only characterized PPU geometry fields."""
  pt,pb,pl,pr = padding
  base_kh, base_kw = min(2,ih), min(8,iw)
  base_oh, base_ow = ih-base_kh+1, iw-base_kw+1
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),
    RKLayout((ih,iw,8),(ih,iw,8),(iw*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),
    RKLayout((base_oh,base_ow,8),(base_oh,base_ow,8),(base_ow*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC))
  stage = emit_pool(RKPool(out,src,Ops.MAX,0,base_kh,base_kw,1,1)).stages[0]
  oh, ow, c = (ih+pt+pb-kh)//sy+1, (iw+pl+pr-kw)//sx+1, channels-1
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
  if average:
    stage = _replace(stage,_PPU,rk.REG_PPU_OPERATION_MODE_CFG,0x10)
    stage = _replace(stage,_PPU,rk.REG_PPU_DATA_FORMAT,(output_index_add<<16)|proc_precision)
    reciprocal_width = (struct.unpack("<H",struct.pack("<e",1/kw))[0]|0x4000) if reciprocal is None else reciprocal
    reciprocal_height = (struct.unpack("<H",struct.pack("<e",1/kh))[0]|0x4000) if reciprocal is None else reciprocal
    reciprocals = (_command(_PPU,_REG_PPU_RECIP_KERNEL_WIDTH,reciprocal_width),
                   _command(_PPU,_REG_PPU_RECIP_KERNEL_HEIGHT,reciprocal_height))
    stage = RKStage(stage.engine,(*stage.commands[:-1],*reciprocals,stage.commands[-1]),stage.relocs,stage.flags)
  padding_cfg = (pb<<12)|(pr<<8)|(pt<<4)|pl
  stage = RKStage(stage.engine,(*stage.commands[:-1],_command(_PPU,_REG_PPU_POOLING_PADDING_CFG,padding_cfg),stage.commands[-1]),
                  stage.relocs,stage.flags)
  return RKImage(RKTarget.RK3588,(stage,))

def reference(values:np.ndarray, kh:int, kw:int, sy:int, sx:int,
              padding:tuple[int,int,int,int]=(0,0,0,0), average:bool=False) -> np.ndarray:
  pt,pb,pl,pr = padding
  values = np.pad(values,((pt,pb),(pl,pr),(0,0)),constant_values=0 if average else -np.inf)
  oh, ow = (values.shape[0]-kh)//sy+1, (values.shape[1]-kw)//sx+1
  return np.stack(tuple((values[y*sy:y*sy+kh,x*sx:x*sx+kw].astype(np.float32).mean(axis=(0,1)).astype(np.float16)
                         if average else values[y*sy:y*sy+kh,x*sx:x*sx+kw].max(axis=(0,1)))
                        for y in range(oh) for x in range(ow))).reshape(oh,ow,values.shape[2])

_CASES = ((9,13,2,2,2,1,1),(17,2,2,1,2,1,2),(9,17,4,1,16,1,1),
          (9,17,8,1,8,1,1),(256,32,8,1,8,1,8),(256,4,8,1,4,1,4),
          (9,17,8,1,16,1,1),(17,32,2,1,16,1,16),(9,13,8,2,2,1,1),(9,13,8,3,3,3,3))
_PAD_CASES = ((11,28,8,5,5,5,5,1,0,1,0),(11,28,8,5,5,5,5,2,1,2,1),(11,28,8,3,2,3,2,1,1,0,1))

def main() -> None:
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--case",type=int,choices=range(len(_CASES)),
                     help="run one geometry in a fresh process; cases 1, 2, 6, and 7 are known-bad or timeout probes")
  group.add_argument("--padding-case",type=int,choices=range(len(_PAD_CASES)),help="run one asymmetric-padding geometry")
  parser.add_argument("--average",action="store_true",help="use PPU average mode with Q16 reciprocal registers")
  parser.add_argument("--reciprocal",type=lambda value:int(value,0),help="override both average reciprocal registers")
  parser.add_argument("--proc-precision",type=int,default=2,choices=(2,5),help="PPU process precision")
  args = parser.parse_args()
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(42)
  case = _CASES[args.case]+(0,0,0,0) if args.case is not None else _PAD_CASES[args.padding_case]
  for ih,iw,channels,kh,kw,sy,sx,pt,pb,pl,pr in (case,):
    values = rng.uniform(-8,8,(ih,iw,channels)).astype(np.float16)
    expected = reference(values,kh,kw,sy,sx,(pt,pb,pl,pr),args.average)
    src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(expected.nbytes)
    try:
      ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
      ctypes.memset(int(out.va_addr),0,expected.nbytes)
      name = f"ppu_{'avg' if args.average else 'max'}_hwc{channels}_{ih}x{iw}_k{kh}x{kw}_s{sy}x{sx}_p{pt}_{pb}_{pl}_{pr}"
      RockchipProgram(dev,TinyELF(encode_image(image(ih,iw,channels,kh,kw,sy,sx,(pt,pb,pl,pr),args.average,args.reciprocal,
        args.proc_precision)),
        name,Target("ROCKCHIP"),()))(out,src,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(out.va_addr),expected.nbytes),dtype=np.float16).copy().reshape(expected.shape)
      allowed = 1e-6+1e-5*np.abs(expected.astype(np.float32))
      official_misses = np.abs(actual.astype(np.float32)-expected.astype(np.float32)) > allowed
      print(f"{name} exact={np.array_equal(actual,expected)} mismatches={np.count_nonzero(actual != expected)}/{actual.size} "
            f"official_misses={np.count_nonzero(official_misses)}/{actual.size} "
            f"actual_head={actual.reshape(-1)[:8].tolist()} expected_head={expected.reshape(-1)[:8].tolist()}")
    finally:
      dev._gpu_free(src)
      dev._gpu_free(out)

if __name__ == "__main__": main()
