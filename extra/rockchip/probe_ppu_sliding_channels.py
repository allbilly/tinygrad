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
          padding:tuple[int,int,int,int]=(0,0,0,0), average:bool=False, reciprocal:int|None=None, proc_precision:int=2,
          index:bool=False, index_add:int|None=None, surf_len:int=0, mc_surf_out:bool=False, minimum:bool=False) -> RKImage:
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
  if minimum:
    mode = next(command>>16&0xffffffff for command in stage.commands
                if command>>48 == _PPU and command&0xffff == rk.REG_PPU_OPERATION_MODE_CFG)
    stage = _replace(stage,_PPU,rk.REG_PPU_OPERATION_MODE_CFG,(mode&~3)|2)
  if index:
    mode = next(command>>16&0xffffffff for command in stage.commands
                if command>>48 == _PPU and command&0xffff == rk.REG_PPU_OPERATION_MODE_CFG)
    stage = _replace(stage,_PPU,rk.REG_PPU_OPERATION_MODE_CFG,mode|(1<<30))
    # INDEX_ADD occupies bits 31:4 and is expressed in 16-byte output atoms.  The
    # value-only path historically placed an unused stride in bits 31:16; that
    # becomes an invalid DMA offset as soon as INDEX_EN makes the field live.
    stage = _replace(stage,_PPU,rk.REG_PPU_DATA_FORMAT,((oh*ow if index_add is None else index_add)<<4)|proc_precision)
    stage = _replace(stage,_PPU,rk.REG_PPU_MISC_CTRL,(surf_len<<16)|(int(mc_surf_out)<<8)|3)
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
              padding:tuple[int,int,int,int]=(0,0,0,0), average:bool=False, minimum:bool=False) -> np.ndarray:
  pt,pb,pl,pr = padding
  values = np.pad(values,((pt,pb),(pl,pr),(0,0)),constant_values=0 if average else -np.inf)
  oh, ow = (values.shape[0]-kh)//sy+1, (values.shape[1]-kw)//sx+1
  return np.stack(tuple((values[y*sy:y*sy+kh,x*sx:x*sx+kw].astype(np.float32).mean(axis=(0,1)).astype(np.float16)
                         if average else (values[y*sy:y*sy+kh,x*sx:x*sx+kw].min(axis=(0,1)) if minimum else
                                          values[y*sy:y*sy+kh,x*sx:x*sx+kw].max(axis=(0,1))))
                        for y in range(oh) for x in range(ow))).reshape(oh,ow,values.shape[2])

_CASES = ((9,13,2,2,2,1,1),(17,2,2,1,2,1,2),(9,17,4,1,16,1,1),
          (9,17,8,1,8,1,1),(256,32,8,1,8,1,8),(256,4,8,1,4,1,4),
          (9,17,8,1,16,1,1),(17,32,2,1,16,1,16),(9,13,8,2,2,1,1),(9,13,8,3,3,3,3),
          (4,4,8,4,4,4,4))
_PAD_CASES = ((11,28,8,5,5,5,5,1,0,1,0),(11,28,8,5,5,5,5,2,1,2,1),(11,28,8,3,2,3,2,1,1,0,1))

def main() -> None:
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--case",type=int,choices=range(len(_CASES)),
                     help="run one geometry in a fresh process; cases 1, 2, 6, and 7 are known-bad or timeout probes")
  group.add_argument("--padding-case",type=int,choices=range(len(_PAD_CASES)),help="run one asymmetric-padding geometry")
  parser.add_argument("--average",action="store_true",help="use PPU average mode with Q16 reciprocal registers")
  parser.add_argument("--minimum",action="store_true",help="use the documented PPU minimum mode")
  parser.add_argument("--reciprocal",type=lambda value:int(value,0),help="override both average reciprocal registers")
  parser.add_argument("--proc-precision",type=int,default=2,choices=(2,5),help="PPU process precision")
  parser.add_argument("--index",action="store_true",help="enable the documented PPU kernel-position output plane")
  parser.add_argument("--index-add",type=lambda value:int(value,0),help="override INDEX_ADD in 16-byte atoms")
  parser.add_argument("--surf-len",type=lambda value:int(value,0),default=0,help="override PPU multi-surface length")
  parser.add_argument("--mc-surf-out",action="store_true",help="enable PPU multiple-surface output")
  parser.add_argument("--ties",action="store_true",help="fill the input with equal values to characterize index tie-breaking")
  parser.add_argument("--signed-zero",action="store_true",help="characterize positive/negative-zero values and indices")
  args = parser.parse_args()
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(42)
  case = _CASES[args.case]+(0,0,0,0) if args.case is not None else _PAD_CASES[args.padding_case]
  for ih,iw,channels,kh,kw,sy,sx,pt,pb,pl,pr in (case,):
    values = (np.ones((ih,iw,channels),dtype=np.float16) if args.ties else rng.uniform(-8,8,(ih,iw,channels)).astype(np.float16))
    if args.signed_zero:
      values.fill(0)
      values.reshape(-1,channels)[0] = np.array([0 if channel%2 == 0 else -0.0 for channel in range(channels)],dtype=np.float16)
      values.reshape(-1,channels)[1] = np.array([-0.0 if channel%2 == 0 else 0 for channel in range(channels)],dtype=np.float16)
    expected = reference(values,kh,kw,sy,sx,(pt,pb,pl,pr),args.average,args.minimum)
    output_bytes = max(expected.nbytes*(4 if args.index else 1),4096 if args.index else expected.nbytes)
    src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(output_bytes)
    try:
      ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
      ctypes.memset(int(out.va_addr),0xa5,output_bytes)
      name = f"ppu_{'avg' if args.average else ('min' if args.minimum else 'max')}_hwc{channels}_{ih}x{iw}_k{kh}x{kw}_s{sy}x{sx}_p{pt}_{pb}_{pl}_{pr}"
      RockchipProgram(dev,TinyELF(encode_image(image(ih,iw,channels,kh,kw,sy,sx,(pt,pb,pl,pr),args.average,args.reciprocal,
        args.proc_precision,args.index,args.index_add,args.surf_len,args.mc_surf_out,args.minimum)),
        name,Target("ROCKCHIP"),()))(out,src,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(out.va_addr),expected.nbytes),dtype=np.float16).copy().reshape(expected.shape)
      allowed = 1e-6+1e-5*np.abs(expected.astype(np.float32))
      official_misses = np.abs(actual.astype(np.float32)-expected.astype(np.float32)) > allowed
      print(f"{name} exact={np.array_equal(actual,expected)} mismatches={np.count_nonzero(actual != expected)}/{actual.size} "
            f"official_misses={np.count_nonzero(official_misses)}/{actual.size} "
            f"actual_head={actual.reshape(-1)[:8].tolist()} expected_head={expected.reshape(-1)[:8].tolist()} "
            f"actual_bits={actual.reshape(-1)[:8].view(np.uint16).tolist()} expected_bits={expected.reshape(-1)[:8].view(np.uint16).tolist()}")
      if args.index:
        raw = np.frombuffer(ctypes.string_at(int(out.va_addr),output_bytes),dtype=np.uint8).copy()
        changed = np.flatnonzero(raw != 0xa5)
        index_offset = (expected.shape[0]*expected.shape[1] if args.index_add is None else args.index_add)*16
        index_plane = raw[index_offset:index_offset+expected.nbytes]
        windows = np.stack(tuple(values[y*sy:y*sy+kh,x*sx:x*sx+kw] for y in range(expected.shape[0])
                                 for x in range(expected.shape[1])))
        reduced_windows = windows.reshape(expected.shape[0]*expected.shape[1],kh*kw,channels)
        expected_local = (reduced_windows.argmin(axis=1) if args.minimum else reduced_windows.argmax(axis=1)).reshape(-1)
        expected_hw = (expected_local//kw)*8+expected_local%kw
        actual_index = index_plane.view(np.uint16)[:expected_local.size]
        index_mismatches = np.flatnonzero(actual_index != expected_hw)
        actual_rows, expected_rows = actual_index.reshape(expected.shape), expected_hw.reshape(expected.shape)
        row_matches = [next((yy for yy in range(expected.shape[0]) if np.array_equal(actual_rows[y],expected_rows[yy])),-1)
                       for y in range(expected.shape[0])]
        print(f"changed=[{changed[0] if changed.size else -1},{changed[-1]+1 if changed.size else -1}) "
              f"index_offset={index_offset} index_exact={np.array_equal(actual_index,expected_hw)} "
              f"index_mismatches={index_mismatches.size}/{expected_local.size}")
        print(f"index_head={actual_index[:16].tolist()} expected_hw_head={expected_hw[:16].tolist()} "
              f"expected_local_head={expected_local[:16].tolist()} mismatch_positions={index_mismatches[:16].tolist()} "
              f"mismatch_actual={actual_index[index_mismatches[:16]].tolist()} "
              f"mismatch_expected={expected_hw[index_mismatches[:16]].tolist()} row_matches={row_matches}")
    finally:
      dev._gpu_free(src)
      dev._gpu_free(out)

if __name__ == "__main__": main()
