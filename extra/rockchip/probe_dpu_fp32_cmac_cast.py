#!/usr/bin/env python3
"""Probe compact FP32-to-FP16 writeback through the DPU external BS operand."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip.image import RK_STAGE_RESET, RKImage, RKReloc, RKStage, encode_image
from tinygrad.renderer.rockchip.ir import RKBufferKind, RKEngine, RKTarget
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x81
_BRDMA_CFG, _BS_BASE, _NRDMA_CFG, _BN_BASE = 0x501c, 0x5020, 0x5028, 0x502c

def _command(target:int, reg:int, value:int) -> int:
  return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)

def image(rows:int, channels:int, dst_stride:int=0, surface_add:int=0x10, bs_stride:int=0, width_geometry:bool=False,
          full_pipeline:bool=False) -> RKImage:
  width, c = ((channels+7)//8-1,7) if width_geometry else (0,channels-1)
  dpu = ((rk.REG_DPU_S_POINTER,0xe),(rk.REG_DPU_FEATURE_MODE_CFG,0x1e5),(rk.REG_DPU_DATA_FORMAT,0x48000002),
    (rk.REG_DPU_DATA_CUBE_WIDTH,width),(rk.REG_DPU_DATA_CUBE_HEIGHT,rows-1),(rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL,(c<<16)|c),(rk.REG_DPU_DST_SURF_STRIDE,dst_stride),
    (rk.REG_DPU_BS_CFG,(2<<16)|0x150),(rk.REG_DPU_BN_CFG,0x42 if full_pipeline else 0x53),
    (rk.REG_DPU_BS_ALU_CFG,0),(rk.REG_DPU_BS_MUL_CFG,0),(rk.REG_DPU_BS_OW_CFG,2),(rk.REG_DPU_WDMA_SIZE_0,c),
    (rk.REG_DPU_WDMA_SIZE_1,((rows-1)<<16)|width),(rk.REG_DPU_BN_MUL_CFG,1 if full_pipeline else 0),
    (rk.REG_DPU_EW_CFG,0x108202c0 if full_pipeline else 0x383),
    (rk.REG_DPU_OUT_CVT_SCALE,0x10001),(rk.REG_DPU_SURFACE_ADD,surface_add))
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,rows-1),(rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,c),
    (_BRDMA_CFG,2),(0x5024,bs_stride),(_NRDMA_CFG,8 if full_pipeline else 0),
    (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x40000008 if full_pipeline else 1),(rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,0x17849),
    (rk.REG_DPU_RDMA_RDMA_WEIGHT,0x01010101))
  commands = [_command(_DPU,*item) for item in dpu]+[_command(_RDMA,*item) for item in rdma]
  relocs = []
  for target,reg,slot in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,1),(_RDMA,_BS_BASE,2)):
    commands.append(_command(target,reg,0))
    relocs.append(RKReloc(0,len(commands)-1,RKBufferKind.ARG,slot))
  if full_pipeline:
    for reg,slot in ((_BN_BASE,3),(rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,4)):
      commands.append(_command(_RDMA,reg,0))
      relocs.append(RKReloc(0,len(commands)-1,RKBufferKind.ARG,slot))
  commands.append(_command(_PC,rk.REG_PC_OPERATION_ENABLE,0x18))
  return RKImage(RKTarget.RK3588,(RKStage(RKEngine.DPU,tuple(commands),tuple(relocs),RK_STAGE_RESET),))

def main() -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(256)
  for rows,channels in ((1,16),(1,256),(1,384),(1,416),(1,512),(1,1024),(1,2048),(1,4096),(1,8192),(1,16384)):
    count = rows*channels
    fp32 = rng.uniform(-4,4,count).astype(np.float32)
    fp32[0:4] = (-0.0,0.0,np.inf,np.nan)
    zeros = np.full(count,np.float16(-0.0),dtype=np.float16)
    expected = fp32.astype(np.float16)
    buffers = [dev._gpu_alloc(size) for size in (count*2,count*2,count*4)]
    try:
      ctypes.memset(int(buffers[0].va_addr),0,count*2)
      ctypes.memmove(int(buffers[1].va_addr),zeros.ctypes.data,zeros.nbytes)
      ctypes.memmove(int(buffers[2].va_addr),fp32.ctypes.data,fp32.nbytes)
      RockchipProgram(dev,TinyELF(encode_image(image(rows,channels)),f"dpu_fp32_cast_{rows}x{channels}",Target("ROCKCHIP"),()))(
        *buffers,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),count*2),dtype=np.float16).copy()
      same = (actual.view(np.uint16)==expected.view(np.uint16))|np.isnan(actual)&np.isnan(expected)
      print(f"shape={rows}x{channels} mismatches={np.count_nonzero(~same)} first={np.flatnonzero(~same)[:8].tolist()}")
    finally:
      for buffer in buffers: dev._gpu_free(buffer)
  for channels in (16,256,384,416):
    fp32 = rng.uniform(-4,4,channels).astype(np.float32)
    zeros, negative_ones, expected = np.zeros(channels,dtype=np.float16), np.full(channels,-1,dtype=np.float16), fp32.astype(np.float16)
    buffers = [dev._gpu_alloc(size) for size in (channels*2,channels*2,channels*4,channels*2,channels*2)]
    try:
      for buffer,value in zip(buffers[1:],(zeros,fp32,negative_ones,zeros)):
        ctypes.memmove(int(buffer.va_addr),value.ctypes.data,value.nbytes)
      RockchipProgram(dev,TinyELF(encode_image(image(1,channels,full_pipeline=True)),
        f"dpu_fp32_full_pipeline_cast_{channels}",Target("ROCKCHIP"),()))(*buffers,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),channels*2),dtype=np.float16).copy()
      print(f"full_pipeline={channels} exact={np.array_equal(actual,expected)} mismatches={np.count_nonzero(actual != expected)}")
    finally:
      for buffer in buffers: dev._gpu_free(buffer)
  for rows,channels in ((2,16),(4,256)):
    count = rows*channels
    fp32 = np.arange(1,count+1,dtype=np.float32)
    zeros, expected = np.full(count,np.float16(-0.0),dtype=np.float16), fp32.astype(np.float16)
    buffers = [dev._gpu_alloc(size) for size in (count*2,count*2,count*4)]
    try:
      ctypes.memset(int(buffers[0].va_addr),0,count*2)
      ctypes.memmove(int(buffers[1].va_addr),zeros.ctypes.data,zeros.nbytes)
      ctypes.memmove(int(buffers[2].va_addr),fp32.ctypes.data,fp32.nbytes)
      cast = image(rows,channels,channels*2,0x10,channels*4,True)
      RockchipProgram(dev,TinyELF(encode_image(cast),f"dpu_fp32_width_cast_{rows}x{channels}",Target("ROCKCHIP"),()))(
        *buffers,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),count*2),dtype=np.float16).copy()
      print(f"width_shape={rows}x{channels} exact={np.array_equal(actual,expected)} first={actual[:32].tolist()}")
    finally:
      for buffer in buffers: dev._gpu_free(buffer)
  rows, channels = 2, 16
  for dst_stride,surface_add,bs_stride in ((0,0x10,0),(32,0x10,64),(32,0x20,64),(32,32,64),(64,0x20,64),(64,64,64)):
    count = rows*channels
    fp32 = np.arange(1,count+1,dtype=np.float32)
    zeros, expected = np.full(count,np.float16(-0.0),dtype=np.float16), fp32.astype(np.float16)
    buffers = [dev._gpu_alloc(size) for size in (count*2,count*2,count*4)]
    try:
      ctypes.memset(int(buffers[0].va_addr),0,count*2)
      ctypes.memmove(int(buffers[1].va_addr),zeros.ctypes.data,zeros.nbytes)
      ctypes.memmove(int(buffers[2].va_addr),fp32.ctypes.data,fp32.nbytes)
      cast = image(rows,channels,dst_stride,surface_add,bs_stride)
      RockchipProgram(dev,TinyELF(encode_image(cast),f"dpu_fp32_cast_stride_{dst_stride}_{surface_add}_{bs_stride}",
        Target("ROCKCHIP"),()))(*buffers,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),count*2),dtype=np.float16).copy()
      print(f"stride=({dst_stride:#x},{surface_add:#x},{bs_stride:#x}) exact={np.array_equal(actual,expected)} "
            f"rows={actual.reshape(rows,channels).tolist()}")
    finally:
      for buffer in buffers: dev._gpu_free(buffer)

if __name__ == "__main__": main()
