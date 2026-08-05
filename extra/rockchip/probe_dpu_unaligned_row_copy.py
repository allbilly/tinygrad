#!/usr/bin/env python3
"""Characterize DPU FP16 source-base alignment while copying dense rows into an atom-aligned surface."""
import argparse, ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.uop.ops import Ops
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKDPUProgram, RKALUStage, emit_dpu, encode_image
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rows",type=int,default=4)
  parser.add_argument("--columns",type=int,default=99)
  parser.add_argument("--physical-columns",type=int,default=104)
  args = parser.parse_args()
  if min(args.rows,args.columns,args.physical_columns) <= 0 or args.physical_columns%8 or args.physical_columns < args.columns:
    raise ValueError("the physical row must be an aligned superset of the logical row")
  source_values = np.arange(args.rows*args.columns,dtype=np.float16)
  dev = RockchipDevice("ROCKCHIP")
  source,output = dev._gpu_alloc(source_values.nbytes),dev._gpu_alloc(args.rows*args.physical_columns*2)
  try:
    ctypes.memmove(int(source.va_addr),source_values.ctypes.data,source_values.nbytes)
    ctypes.memset(int(output.va_addr),0,args.rows*args.physical_columns*2)
    stages = tuple(RKALUStage(Ops.ADD,RKArg(RKBufferKind.ARG,0,row*args.physical_columns*2),
      RKArg(RKBufferKind.ARG,1,row*args.columns*2),0.0,args.columns) for row in range(args.rows))
    image = encode_image(emit_dpu(RKDPUProgram(stages)))
    RockchipProgram(dev,TinyELF(image,"dpu_unaligned_row_copy",Target("ROCKCHIP"),()))(output,source,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(output.va_addr),args.rows*args.physical_columns*2),
                           dtype=np.float16).reshape(args.rows,args.physical_columns)
    for row in range(args.rows):
      matches = np.flatnonzero(source_values == actual[row,0])
      print(f"row={row} requested_source={row*args.columns} observed_source={matches[0] if matches.size else None} "
            f"first={actual[row,:8].tolist()} tail={actual[row,-8:].tolist()}")
  finally:
    dev._gpu_free(source)
    dev._gpu_free(output)

if __name__ == "__main__": main()
