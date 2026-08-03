#!/usr/bin/env python3
"""Characterize RK3588 FP16 sliding-average PPU on the current DRM runtime."""
from __future__ import annotations
import ctypes
import numpy as np
import torch

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKLayout, RKPool, RKTarget, RKTensorRef,
  encode_image, emit_pool)
from tinygrad.renderer.rockchip.image import RKStage
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

_PPU = 0x4001
_RECIP_WIDTH, _RECIP_HEIGHT = 0x6038, 0x603c
# NVDLA's FP17 reciprocal encodings for 1/K, K=1..8. RKNN's 2x2 AVG command uses the same 0x7800 value.
_RECIP = (0, 0x7c00, 0x7800, 0x7555, 0x7400, 0x7266, 0x7155, 0x7092, 0x7000)

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def image(ih:int, iw:int, kh:int, kw:int, sy:int=1, sx:int=1) -> RKImage:
  oh, ow = (ih-kh)//sy+1, (iw-kw)//sx+1
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1), RKLayout((ih,iw,8),(ih,iw,8),(iw*16,16,2),dtypes.half))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0), RKLayout((oh,ow,8),(oh,ow,8),(ow*16,16,2),dtypes.half))
  base = emit_pool(RKPool(out,src,Ops.MAX,0,kh,kw,sy,sx))
  stage = base.stages[0]
  commands = tuple(_command(_PPU,rk.REG_PPU_OPERATION_MODE_CFG,0x10)
                   if command>>48 == _PPU and command&0xffff == rk.REG_PPU_OPERATION_MODE_CFG else command
                   for command in stage.commands)
  commands = commands[:-1] + (_command(_PPU,_RECIP_WIDTH,_RECIP[kw]),
                              _command(_PPU,_RECIP_HEIGHT,_RECIP[kh])) + commands[-1:]
  return RKImage(RKTarget.RK3588,(RKStage(stage.engine,commands,stage.relocs,stage.flags),))

def reference(values:np.ndarray, kh:int, kw:int, sy:int, sx:int) -> np.ndarray:
  oh, ow = (values.shape[0]-kh)//sy+1, (values.shape[1]-kw)//sx+1
  out = np.empty((oh,ow,8),dtype=np.float16)
  for y in range(oh):
    for x in range(ow): out[y,x] = values[y*sy:y*sy+kh,x*sx:x*sx+kw].astype(np.float32).mean(axis=(0,1)).astype(np.float16)
  return out

def main() -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(34)
  for ih,iw,kh,kw,sy,sx in ((7,9,2,2,1,1),(8,10,3,3,1,1),(9,11,3,2,2,1),(9,9,5,5,1,1)):
    values = rng.uniform(-4,4,(ih,iw,8)).astype(np.float16)
    expected = reference(values,kh,kw,sy,sx)
    src, out = dev._gpu_alloc(values.nbytes), dev._gpu_alloc(expected.nbytes)
    try:
      ctypes.memmove(int(src.va_addr),values.ctypes.data,values.nbytes)
      ctypes.memset(int(out.va_addr),0,expected.nbytes)
      name = f"ppu_avg_{kh}x{kw}_s{sy}x{sx}"
      RockchipProgram(dev,TinyELF(encode_image(image(ih,iw,kh,kw,sy,sx)),name,Target("ROCKCHIP"),()))(out,src,wait=True)
      actual = np.frombuffer(ctypes.string_at(int(out.va_addr),expected.nbytes),dtype=np.float16).copy().reshape(expected.shape)
      torch_expected = torch.nn.functional.avg_pool2d(torch.from_numpy(values.transpose(2,0,1)[None]),
        kernel_size=(kh,kw),stride=(sy,sx))[0].numpy().transpose(1,2,0)
      diff = np.abs(actual.astype(np.float32)-torch_expected.astype(np.float32))
      print(f"{name} exact={np.array_equal(actual,torch_expected)} official_tol="
            f"{np.allclose(actual,torch_expected,rtol=1e-5,atol=1e-6)} max_abs={float(diff.max())} "
            f"mismatches={int(np.count_nonzero(diff))}/{diff.size} fp32_ref_equal={np.array_equal(expected,torch_expected)}")
    finally:
      dev._gpu_free(src)
      dev._gpu_free(out)

if __name__ == "__main__": main()
