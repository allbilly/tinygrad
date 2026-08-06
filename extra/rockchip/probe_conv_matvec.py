#!/usr/bin/env python3
"""Archive the rejected wide-spatial-CONV matrix-vector experiment.

The corresponding emitter candidate remains commented in ``emit.py``. It was
exact at K=8, timed out at K=32, and at K=65 behaved like a wrapped one-tap
operation. With the production emitter this probe reports the typed rejection
instead of silently re-enabling the unsafe register family.
"""
from __future__ import annotations
import argparse, ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKConvTask, RKImage, RKLayout, RKLayoutKind, RKTarget,
  RKTensorRef, encode_image, emit_spatial_conv)
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def image(rows:int, k:int) -> RKImage:
  output_stride = (rows+3)&-4
  src = RKTensorRef(RKArg(RKBufferKind.ARG,1),
    RKLayout((rows,k,1),(rows,k,1),(k*2,2,2),dtypes.half,kind=RKLayoutKind.CNA_ACTIVATION))
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),
    RKLayout((1,k,1,1),(1,k,1,8),(k*16,16,16,2),dtypes.half,
             padding=((0,0),(0,0),(0,0),(0,7)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),
    RKLayout((2,output_stride,8),(2,output_stride,8),(output_stride*16,16,2),dtypes.half,kind=RKLayoutKind.CONV_OUTPUT))
  return emit_spatial_conv(RKConvTask(out,src,weight,1,1,rows,k,1,k,rows,1,1,1,k,output_stride))

def probe(rows:int, k:int) -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608)
  matrix = rng.uniform(-1,1,(rows,k)).astype(np.float16)
  vector = rng.uniform(-1,1,k).astype(np.float16)
  packed_weight = np.zeros((1,k,1,8),dtype=np.float16)
  packed_weight[0,:,0,0] = vector
  output_stride = (rows+3)&-4
  out,src,weight = dev._gpu_alloc(2*output_stride*8*2),dev._gpu_alloc(matrix.nbytes),dev._gpu_alloc(packed_weight.nbytes)
  try:
    ctypes.memmove(int(src.va_addr),matrix.ctypes.data,matrix.nbytes)
    ctypes.memmove(int(weight.va_addr),packed_weight.ctypes.data,packed_weight.nbytes)
    ctypes.memset(int(out.va_addr),0xa5,2*output_stride*8*2)
    try: probe_image = image(rows,k)
    except ValueError as exc:
      print(f"M={rows} K={k} production_reject={exc}")
      return
    RockchipProgram(dev,TinyELF(encode_image(probe_image),f"conv_matvec_m{rows}_k{k}",Target("ROCKCHIP"),()))(out,src,weight,wait=True)
    physical = np.frombuffer(ctypes.string_at(int(out.va_addr),2*output_stride*8*2),dtype=np.float16).copy()
    actual = physical.reshape(2,output_stride,8)[0,:rows,0]
    expected = (matrix.astype(np.float32)@vector.astype(np.float32)).astype(np.float16)
    mismatch = np.flatnonzero(actual != expected)
    print(f"M={rows} K={k} task_count=1 exact={mismatch.size == 0} mismatches={mismatch.size} "
          f"max_abs={np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))):.6g} "
          f"actual_head={actual[:8].tolist()} expected_head={expected[:8].tolist()}")
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-5)
  finally:
    dev._gpu_free(out)
    dev._gpu_free(src)
    dev._gpu_free(weight)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("rows",type=int,nargs="?")
  parser.add_argument("k",type=int,nargs="?")
  args = parser.parse_args()
  if (args.rows is None) != (args.k is None): parser.error("provide both ROWS and K")
  for shape in (((args.rows,args.k),) if args.rows is not None else ((4,8),(4,32),(8,65),(45,65),(360,65))): probe(*shape)
