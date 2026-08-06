#!/usr/bin/env python3
"""Probe bounded row-major matrix-vector CMAC tiles with FP32 continuation."""
from __future__ import annotations
import argparse, ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKCastStage, RKDPUProgram, RKEpilogue, RKLayout, RKLayoutKind,
  RKScratch, RKTensorRef, RKCMACTask, RKProgram, encode_image, emit_program)
from tinygrad.renderer.rockchip.ir import RKALUStage
from tinygrad.renderer.rockchip.selector import _cmac_weight_ref, _windowed_cmac_pipeline
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.uop.ops import Ops

def program(rows:int, k:int) -> RKProgram:
  chunks = (k+31)//32
  vector = RKArg(RKBufferKind.SCRATCH,0)
  packed = RKArg(RKBufferKind.SCRATCH,1)
  accumulators = (RKArg(RKBufferKind.SCRATCH,2),RKArg(RKBufferKind.SCRATCH,3))
  half_out = RKArg(RKBufferKind.SCRATCH,4)
  scratch = (RKScratch(chunks*64),RKScratch(2048),RKScratch(256),RKScratch(256),RKScratch(64))
  vector_stages = []
  for chunk in range(chunks):
    valid = min(32,k-chunk*32)
    dst = RKArg(vector.kind,vector.index,chunk*64)
    vector_stages.extend((RKALUStage(Ops.ADD,dst,0.0,0.0,32),
                          RKALUStage(Ops.ADD,dst,RKArg(RKBufferKind.ARG,2,chunk*64),0.0,valid)))
  steps:list = [RKDPUProgram(tuple(vector_stages),scratch)]
  source_capacity = ((rows*k*2+4095)&-4096)//2
  vector_layout = RKLayout((1,32),(1,32),(64,2),dtypes.half)
  weight_layout = RKLayout((32,32),(32,32),(64,2),dtypes.half,kind=RKLayoutKind.CMAC_WEIGHT)
  fp32_out_layout = RKLayout((1,32),(1,64),(256,4),dtypes.float,padding=((0,0),(0,32)))
  fp32_bias_layout = RKLayout((32,),(32,),(4,),dtypes.float)
  for row_start in range(0,rows,32):
    valid_rows = min(32,rows-row_start)
    current = 0
    for chunk in range(chunks):
      chunk_start, valid_k = chunk*32, min(32,k-chunk*32)
      mapping = [[(row_start+out_lane)*k+chunk_start+k_lane]
                 if out_lane < valid_rows and k_lane < valid_k else []
                 for out_lane in range(32) for k_lane in range(32)]
      if any(not row for row in mapping):
        steps.append(RKDPUProgram((RKALUStage(Ops.ADD,packed,0.0,0.0,1024),),scratch))
      packed_plan = _windowed_cmac_pipeline(packed,RKArg(RKBufferKind.ARG,1),mapping,scratch=scratch,
        direct_count=source_capacity,max_window=512,max_outputs=128,clear_empty=False)
      if packed_plan is None: raise ValueError("matrix tile cannot be packed inside the selector contract")
      scratch = packed_plan.scratch
      steps.extend(packed_plan.steps)
      out = accumulators[current]
      epilogue = None if chunk == 0 else RKEpilogue(RKTensorRef(accumulators[1-current],fp32_bias_layout))
      steps.append(RKCMACTask(RKTensorRef(out,fp32_out_layout),
        RKTensorRef(RKArg(vector.kind,vector.index,chunk*64),vector_layout),RKTensorRef(packed,weight_layout),0,b"",epilogue))
      current = 1-current
    final = accumulators[1-current]
    cast_dst = RKArg(RKBufferKind.ARG,0,row_start*2) if valid_rows == 32 else half_out
    tail_copy = () if valid_rows == 32 else (RKALUStage(Ops.ADD,RKArg(RKBufferKind.ARG,0,row_start*2),half_out,0.0,valid_rows),)
    steps.append(RKDPUProgram((RKCastStage(cast_dst,final,32,dtypes.float,dtypes.half),*tail_copy),scratch))
  return RKProgram(tuple(steps),scratch)

def probe(rows:int, k:int, special_values:bool=False) -> None:
  dev, rng = RockchipDevice("ROCKCHIP"), np.random.default_rng(2608)
  matrix = rng.uniform(-1,1,(rows,k)).astype(np.float16)
  vector = rng.uniform(-1,1,k).astype(np.float16)
  if special_values:
    matrix[:4,:4] = np.eye(4,dtype=np.float16)
    matrix[0,0],matrix[1,1],matrix[2,2],matrix[3,3] = np.inf,-np.inf,np.nan,-0.0
    vector[:4] = 1
  out,matrix_buf,vector_buf = dev._gpu_alloc(rows*2),dev._gpu_alloc(matrix.nbytes),dev._gpu_alloc(vector.nbytes)
  try:
    ctypes.memmove(int(matrix_buf.va_addr),matrix.ctypes.data,matrix.nbytes)
    ctypes.memmove(int(vector_buf.va_addr),vector.ctypes.data,vector.nbytes)
    image = emit_program(program(rows,k))
    RockchipProgram(dev,TinyELF(encode_image(image),f"cmac_matvec_m{rows}_k{k}",Target("ROCKCHIP"),()))(
      out,matrix_buf,vector_buf,wait=True)
    actual = np.frombuffer(ctypes.string_at(int(out.va_addr),rows*2),dtype=np.float16).copy()
    expected = (matrix.astype(np.float32)@vector.astype(np.float32)).astype(np.float16)
    mismatch = np.flatnonzero(~((actual == expected)|(np.isnan(actual)&np.isnan(expected))))
    print(f"M={rows} K={k} tasks={len(image.stages)} constants={len(image.constants)} exact={mismatch.size == 0} "
          f"mismatches={mismatch.size} max_abs={np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))):.6g}")
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6,equal_nan=True)
  finally:
    dev._gpu_free(out)
    dev._gpu_free(matrix_buf)
    dev._gpu_free(vector_buf)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("rows",type=int,nargs="?",default=360)
  parser.add_argument("k",type=int,nargs="?",default=65)
  parser.add_argument("--special-values",action="store_true")
  args = parser.parse_args()
  probe(args.rows,args.k,args.special_values)
