#!/usr/bin/env python3
"""Direct RK3588 DPU probe for a fused x + (y-x)*z pipeline.

This is a hardware-research probe, not part of the renderer. It deliberately uses
MRDMA -> BS(subtract BRDMA) -> BN(multiply NRDMA) -> EW(add ERDMA) in one task.
"""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.renderer.rockchip.image import RK_STAGE_RESET, RKImage, RKReloc, RKStage, encode_image
from tinygrad.renderer.rockchip.ir import RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKEngine, RKTarget
from tinygrad.renderer.rockchip.emit import emit_dpu
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x81
_BRDMA_CFG, _BS_BASE, _NRDMA_CFG, _BN_BASE = 0x501c, 0x5020, 0x5028, 0x502c

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _replace(stage:RKStage, target:int, reg:int, value:int) -> RKStage:
  commands = tuple(_command(target, reg, value) if command>>48 == target and command&0xffff == reg else command for command in stage.commands)
  return RKStage(stage.engine, commands, stage.relocs, stage.flags)

def image(bs_algo:int, full:bool=True, count:int=8) -> RKImage:
  width = (count+7)//8-1
  dpu = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
    (rk.REG_DPU_DATA_FORMAT, (2<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
    # BS: external ALU, no multiply/ReLU. BN: bypass ALU, external multiply, no ReLU.
    (rk.REG_DPU_BS_CFG, (bs_algo<<16)|0x150), (rk.REG_DPU_BN_CFG, 0x42 if full else 0x53),
    (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width),
    (rk.REG_DPU_BN_MUL_CFG, 1), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0),
    (rk.REG_DPU_EW_CFG, (0x108002c0|(2<<16)) if full else 0x383), (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, count-1),
    (_BRDMA_CFG, 2), (_NRDMA_CFG, 8 if full else 0), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008 if full else 1),
    (rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), (rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101))
  commands = [_command(_DPU, *item) for item in dpu] + [_command(_RDMA, *item) for item in rdma]
  relocs:list[RKReloc] = []
  # ABI: output, x_fp16, y_fp16, z_fp16, x_operand, z_operand. Main=y, BS=x, BN=z, EW=x.
  addresses = [(_DPU, rk.REG_DPU_DST_BASE_ADDR, 0), (_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 2), (_RDMA, _BS_BASE, 4)]
  if full: addresses += [(_RDMA, _BN_BASE, 5), (_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 1)]
  for target, reg, slot in addresses:
    commands.append(_command(target, reg, 0))
    relocs.append(RKReloc(0, len(commands)-1, RKBufferKind.ARG, slot))
  commands.append(_command(_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
  return RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, tuple(commands), tuple(relocs), RK_STAGE_RESET),))

def main() -> None:
  x = np.array([-3, -1, -.25, 0, .5, 1, 2, 4], dtype=np.float16)
  y = np.array([4, 2, .75, -.5, 1.5, -2, 3, -1], dtype=np.float16)
  z = np.array([0, .125, .25, .5, .75, 1, -1, 2], dtype=np.float16)
  dev = RockchipDevice("ROCKCHIP")
  buffers = [dev._gpu_alloc(size) for size in (16,16,16,16,32,32)]
  try:
    x_operand, z_operand = x.astype(np.float32), z
    for buf, value in zip(buffers[1:], (x,y,z,x_operand,z_operand)):
      ctypes.memmove(int(buf.va_addr), value.ctypes.data, value.nbytes)
    for addend in (0,8):
      conversion = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD, RKArg(RKBufferKind.ARG, 0),
        RKArg(RKBufferKind.ARG, 1, addend), 0.0, 4, dtypes.float),)))
      ctypes.memset(int(buffers[0].va_addr), 0, 16)
      program = RockchipProgram(dev, TinyELF(encode_image(conversion), f"convert_{addend}", Target("ROCKCHIP"), ()))
      program(*buffers, wait=True)
      converted = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr), 16), dtype=np.float32).copy()
      print(f"convert source addend {addend}: {converted.tolist()}")
    wide_conversion = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD, RKArg(RKBufferKind.ARG, 4),
      RKArg(RKBufferKind.ARG, 1), 0.0, 8, dtypes.float),)))
    ctypes.memset(int(buffers[4].va_addr), 0, 32)
    RockchipProgram(dev, TinyELF(encode_image(wide_conversion), "convert_8", Target("ROCKCHIP"), ()))(*buffers, wait=True)
    print(f"convert eight: {np.frombuffer(ctypes.string_at(int(buffers[4].va_addr), 32), dtype=np.float32).tolist()}")
    padded = np.zeros(16,dtype=np.float16)
    padded[:4], padded[8:12] = x[:4], x[4:]
    padded_buf, converted_buf = dev._gpu_alloc(32), dev._gpu_alloc(32)
    try:
      ctypes.memmove(int(padded_buf.va_addr), padded.ctypes.data, 32)
      grouped = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD, RKArg(RKBufferKind.ARG, 0), RKArg(RKBufferKind.ARG, 1),
        0.0, 4, dtypes.float),)))
      grouped_stage = _replace(grouped.stages[0], _DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 1)
      grouped_stage = _replace(grouped_stage, _DPU, rk.REG_DPU_WDMA_SIZE_1, 1)
      grouped_stage = _replace(grouped_stage, _RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, 1)
      grouped = RKImage(RKTarget.RK3588, (grouped_stage,), constants=grouped.constants)
      RockchipProgram(dev, TinyELF(encode_image(grouped), "convert_grouped", Target("ROCKCHIP"), ()))(converted_buf,padded_buf,wait=True)
      converted = np.frombuffer(ctypes.string_at(int(converted_buf.va_addr),32),dtype=np.float32).copy()
      print(f"convert grouped exact={np.array_equal(converted,x.astype(np.float32))}: {converted.tolist()}")
    finally:
      dev._gpu_free(padded_buf)
      dev._gpu_free(converted_buf)
    source32 = np.linspace(-4,4,32,dtype=np.float16)
    source32_buf, output32_buf = dev._gpu_alloc(64), dev._gpu_alloc(128)
    try:
      ctypes.memmove(int(source32_buf.va_addr), source32.ctypes.data, 64)
      base = emit_dpu(RKDPUProgram((RKALUStage(Ops.ADD, RKArg(RKBufferKind.ARG, 0), RKArg(RKBufferKind.ARG, 1),
        0.0, 4, dtypes.float),)))
      stage = _replace(base.stages[0], _DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x1f001f)
      # Rejected WIP: 0x36e is Mesa's FP32 BS_OW_CFG for convolution output, but using it here
      # in flying mode times out. Keep the proven four-lane bypass value for this diagnostic.
      stage = _replace(stage, _DPU, rk.REG_DPU_BS_OW_CFG, 2)
      stage = _replace(stage, _DPU, rk.REG_DPU_WDMA_SIZE_0, 31)
      stage = _replace(stage, _RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 31)
      wide32 = RKImage(RKTarget.RK3588, (stage,), constants=base.constants)
      program32 = RockchipProgram(dev, TinyELF(encode_image(wide32), "convert_32_channels", Target("ROCKCHIP"), ()))
      program32(output32_buf, source32_buf, wait=True)
      actual32 = np.frombuffer(ctypes.string_at(int(output32_buf.va_addr), 128), dtype=np.float32).copy()
      print(f"convert 32 channels exact={np.array_equal(actual32, source32.astype(np.float32))}: {actual32.tolist()}")
    finally:
      dev._gpu_free(source32_buf)
      dev._gpu_free(output32_buf)
    for name,operand in (("dense_fp16", x), ("dense_fp32", x.astype(np.float32)),
                         ("low16_spaced", x.view(np.uint16).astype(np.uint32)),
                         ("high16_spaced", x.view(np.uint16).astype(np.uint32)<<16)):
      ctypes.memset(int(buffers[4].va_addr), 0, 32)
      ctypes.memmove(int(buffers[4].va_addr), operand.ctypes.data, operand.nbytes)
      for algo in (2,4):
        ctypes.memset(int(buffers[0].va_addr), 0, 16)
        program = RockchipProgram(dev, TinyELF(encode_image(image(algo, False)), f"bs_probe_{algo}", Target("ROCKCHIP"), ()))
        program(*buffers, wait=True)
        actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr), 16), dtype=np.float16).copy()
        print(f"BS-only {name} algo {algo}: {actual.tolist()}")
    ctypes.memmove(int(buffers[4].va_addr), x.astype(np.float32).ctypes.data, 32)
    for algo in (2,4):
      ctypes.memset(int(buffers[0].va_addr), 0, 16)
      program = RockchipProgram(dev, TinyELF(encode_image(image(algo)), f"lerp_probe_{algo}", Target("ROCKCHIP"), ()))
      program(*buffers, wait=True)
      actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr), 16), dtype=np.float16).copy()
      print(f"BS algo {algo}: {actual.tolist()}")
    print(f"lerp expected: {(x.astype(np.float32)+(y.astype(np.float32)-x.astype(np.float32))*z.astype(np.float32)).astype(np.float16).tolist()}")
    print(f"reverse expected: {(x.astype(np.float32)+(x.astype(np.float32)-y.astype(np.float32))*z.astype(np.float32)).astype(np.float16).tolist()}")
    x16 = np.linspace(-3,4,16,dtype=np.float16)
    y16 = np.linspace(4,-2,16,dtype=np.float16)
    z16 = np.linspace(-1,1,16,dtype=np.float16)
    buffers16 = [dev._gpu_alloc(size) for size in (32,32,32,32,64,32)]
    try:
      for buf,value in zip(buffers16[1:],(x16,y16,z16,x16.astype(np.float32),z16)):
        ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
      RockchipProgram(dev,TinyELF(encode_image(image(4,True,16)),"lerp16",Target("ROCKCHIP"),()))(*buffers16,wait=True)
      actual16=np.frombuffer(ctypes.string_at(int(buffers16[0].va_addr),32),dtype=np.float16).copy()
      expected16=(x16.astype(np.float32)+(y16.astype(np.float32)-x16.astype(np.float32))*z16.astype(np.float32)).astype(np.float16)
      print(f"lerp 16 BRDMA broadcast exact={np.array_equal(actual16,expected16)}: {actual16.tolist()}")
    finally:
      for buf in buffers16: dev._gpu_free(buf)
  finally:
    for buf in buffers: dev._gpu_free(buf)

if __name__ == "__main__": main()
