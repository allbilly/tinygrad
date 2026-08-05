#!/usr/bin/env python3
"""Probe ordered overlapping DPU copies at four-byte word offsets."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKCopyStage, RKDPUProgram, RKTarget, encode_image, emit_dpu
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def main() -> None:
  dev, side, padded = RockchipDevice("ROCKCHIP"), 5, 32
  source = np.arange(padded,dtype=np.uint32)^np.uint32(0x5a5a0000)
  mapping = np.arange(side*side,dtype=np.int32).reshape(side,side).T.reshape(-1)
  stages = tuple(RKCopyStage(RKArg(RKBufferKind.ARG,0,dst*4),RKArg(RKBufferKind.ARG,1,int(src)*4),1,dtypes.int)
                 for dst,src in enumerate(mapping))
  image = emit_dpu(RKDPUProgram(stages),RKTarget.RK3588)
  buffers = [dev._gpu_alloc(padded*4) for _ in range(2)]
  try:
    ctypes.memset(int(buffers[0].va_addr),0,padded*4)
    ctypes.memmove(int(buffers[1].va_addr),source.ctypes.data,source.nbytes)
    RockchipProgram(dev,TinyELF(encode_image(image),"unaligned_word_copy",Target("ROCKCHIP"),()))(*buffers,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),padded*4),dtype=np.uint32).copy()
    desired = source[mapping]
    aligned = np.zeros(padded,dtype=np.uint32)
    for dst,src in enumerate(mapping): aligned[dst&-4:(dst&-4)+4] = source[(int(src)&-4):(int(src)&-4)+4]
    print(f"stages={len(stages)} desired={np.array_equal(actual[:len(mapping)],desired)} "
          f"aligned_down={np.array_equal(actual,aligned)}")
    print(f"actual={actual[:len(mapping)].tolist()}")
    print(f"desired={desired.tolist()}")
    assert not np.array_equal(actual[:len(mapping)],desired)
    assert np.array_equal(actual,aligned)
  finally:
    for buffer in buffers: dev._gpu_free(buffer)

if __name__ == "__main__": main()
