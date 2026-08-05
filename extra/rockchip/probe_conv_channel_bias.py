#!/usr/bin/env python3
"""Probe one flying CNA/CORE/DPU task with FP32 channel bias and optional ReLU."""
from __future__ import annotations
import ctypes
import numpy as np

from tinygrad.device import Target, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKConvTask, RKEpilogue, RKLayout, RKLayoutKind,
  RKTensorRef, encode_image, emit_spatial_conv)
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram

def image(in_channels:int, out_channels:int, relu:bool):
  ih = iw = oh = ow = 5
  input_width_stride, output_width_stride = 6, 28
  if in_channels <= 4:
    src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((ih,iw,in_channels),(ih,input_width_stride,in_channels),
      (input_width_stride*in_channels*2,in_channels*2,2),dtypes.half,
      padding=((0,0),(0,1),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0))
    align_in = 8
  else:
    src = RKTensorRef(RKArg(RKBufferKind.ARG,1),RKLayout((in_channels//8,ih,iw,8),(in_channels//8,ih,input_width_stride,8),
      (ih*input_width_stride*16,input_width_stride*16,16,2),dtypes.half,
      padding=((0,0),(0,0),(0,1),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0))
    align_in = 16
  weight = RKTensorRef(RKArg(RKBufferKind.ARG,2),RKLayout((1,1,out_channels,in_channels),(1,1,out_channels,align_in),
    (out_channels*align_in*2,out_channels*align_in*2,align_in*2,2),dtypes.half,
    padding=((0,0),(0,0),(0,0),(0,align_in-in_channels)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0))
  out = RKTensorRef(RKArg(RKBufferKind.ARG,0),RKLayout((2,output_width_stride,8),(2,output_width_stride,8),
    (output_width_stride*16,16,2),dtypes.half,kind=RKLayoutKind.CONV_OUTPUT))
  bias = RKTensorRef(RKArg(RKBufferKind.ARG,3),
    RKLayout((out_channels,),(32,),(4,),dtypes.float,padding=((0,32-out_channels),)))
  return emit_spatial_conv(RKConvTask(out,src,weight,in_channels,out_channels,ih,iw,1,1,oh,ow,1,1,input_width_stride,output_width_stride,
                                      epilogue=RKEpilogue(bias,relu)))

def main() -> None:
  rng = np.random.default_rng(260806)
  dev = RockchipDevice("ROCKCHIP")
  try:
    for in_channels in (4,8,16):
      logical_src = rng.uniform(-2,2,(5,5,in_channels)).astype(np.float16)
      if in_channels <= 4:
        packed_src = np.zeros((5,6,in_channels),dtype=np.float16)
        packed_src[:,:5] = logical_src
      else:
        packed_src = np.zeros((in_channels//8,5,6,8),dtype=np.float16)
        for channel in range(in_channels): packed_src[channel//8,:,:5,channel%8] = logical_src[:,:,channel]
      for out_channels in (1,2,3,4,5,7,8,9,12,16):
        logical_weight = rng.uniform(-1,1,(out_channels,in_channels)).astype(np.float16)
        bias = rng.uniform(-1,1,out_channels).astype(np.float16)
        packed_weight = np.zeros((1,1,out_channels,8 if in_channels <= 4 else 16),dtype=np.float16)
        packed_weight[0,0,:,:in_channels] = logical_weight
        packed_bias = np.zeros(32,dtype=np.float32)
        packed_bias[:out_channels] = bias
        buffers = [dev._gpu_alloc(size) for size in (2*28*8*2,packed_src.nbytes,packed_weight.nbytes,packed_bias.nbytes)]
        try:
          for buf,value in zip(buffers[1:],(packed_src,packed_weight,packed_bias)):
            ctypes.memmove(int(buf.va_addr),value.ctypes.data,value.nbytes)
          expected = logical_src.astype(np.float32) @ logical_weight.astype(np.float32).T + bias.astype(np.float32)
          for relu in (False,True):
            ctypes.memset(int(buffers[0].va_addr),0xa5,2*28*8*2)
            RockchipProgram(dev,TinyELF(encode_image(image(in_channels,out_channels,relu)),
              f"conv_channel_bias_{in_channels}_{out_channels}_{relu}",Target("ROCKCHIP"),()))(*buffers,wait=True)
            surface = np.frombuffer(ctypes.string_at(int(buffers[0].va_addr),2*28*8*2),dtype=np.float16).reshape(2,28,8)
            actual = np.stack(tuple(surface[channel//8,:25,channel%8] for channel in range(out_channels)),axis=-1).reshape(5,5,out_channels)
            reference = np.maximum(expected,0).astype(np.float16) if relu else expected.astype(np.float16)
            mismatch = np.flatnonzero(actual.view(np.uint16) != reference.view(np.uint16))
            max_abs = float(np.max(np.abs(actual.astype(np.float32)-reference.astype(np.float32))))
            print(f"in_channels={in_channels} out_channels={out_channels} relu={relu} bit_exact={not mismatch.size} "
                  f"mismatches={mismatch.size} max_abs={max_abs} first={mismatch[:8].tolist()}")
            assert (not mismatch.size) if in_channels == 4 else bool(mismatch.size)
        finally:
          for buf in buffers: dev._gpu_free(buf)
  finally:
    del dev

if __name__ == "__main__": main()
