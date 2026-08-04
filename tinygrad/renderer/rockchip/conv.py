from __future__ import annotations

from tinygrad.renderer.rockchip.ir import RKConvSplit, RKConvTile, RKConvTiling

_CBUF_BANKS, _BANK_BYTES, _ENTRY_BYTES, _ENTRIES_PER_BANK = 12, 32768, 128, 256

def _ceildiv(value:int, divisor:int) -> int: return (value+divisor-1)//divisor

def _entries_per_slice(width:int, channels:int) -> int:
  atomics_per_entry, channel_atomics = _ENTRY_BYTES//16, _ceildiv(channels*2,16)
  whole, partial = divmod(channel_atomics,atomics_per_entry)
  return whole*width + (width if partial == 3 else _ceildiv(partial*width,atomics_per_entry))

def _windows(total:int, step:int, min_tail:int=6) -> tuple[tuple[int,int], ...]:
  if step >= total: return ((0,total),)
  windows, start = [], 0
  while start < total:
    remaining = total-start
    if remaining <= step:
      windows.append((start,remaining))
      break
    tail = remaining%step
    width = step if step < min_tail or tail == 0 or tail >= min_tail or remaining <= step+min_tail else max(1,step-(min_tail-tail))
    windows.append((start,width))
    start += width
  return tuple(windows)

def _pointwise_k_step(in_channels:int, out_channels:int, feature_banks:int) -> int:
  available_weight_banks = 2 if feature_banks < _CBUF_BANKS-2 else 1
  maximum = available_weight_banks*_BANK_BYTES//(in_channels*2)
  for granule in (32,16,8,4,1):
    if (step:=min(out_channels,maximum)//granule*granule): return min(32,step)
  return 1

def plan_conv_cbuf(in_h:int, in_w:int, in_channels:int, out_channels:int, kernel_h:int, kernel_w:int, stride:int=1,
                   width_stride:int|None=None, output_width_stride:int|None=None, aligned_in:int|None=None,
                   use_nhwc:bool=False, max_k_step:int=32) -> RKConvTiling|None:
  """Plan dense valid FP16 convolution tiles from simultaneous RK3588 feature/weight CBUF pressure."""
  if min(in_h,in_w,in_channels,out_channels,kernel_h,kernel_w,stride,max_k_step) <= 0 or stride > 7 or \
     kernel_h > in_h or kernel_w > in_w:
    return None
  width_stride = in_w if width_stride is None else width_stride
  spatial, out_h, out_w = kernel_h != 1 or kernel_w != 1, (in_h-kernel_h)//stride+1, (in_w-kernel_w)//stride+1
  output_width_stride = out_h*out_w if output_width_stride is None else output_width_stride
  aligned_in = (8 if in_channels <= 4 else _ceildiv(in_channels,32)*32 if not spatial and in_channels >= 32 else _ceildiv(in_channels,16)*16) \
    if aligned_in is None else aligned_in
  if width_stride < in_w or aligned_in < in_channels: return None
  row_bytes, weight_bytes_per_k = width_stride*aligned_in*2, kernel_h*kernel_w*aligned_in*2
  full_weight_banks = max(1,_ceildiv(weight_bytes_per_k*out_channels,_BANK_BYTES))
  even_rows = (_ceildiv(2*_BANK_BYTES,row_bytes)+1)&-2
  feature_grains = in_h+kernel_h if use_nhwc and spatial else min(in_h+kernel_h,even_rows)
  k_step = out_channels
  if spatial and (full_weight_banks > 3 or feature_grains < in_h): k_step = min(32,out_channels)
  elif not spatial and in_channels >= 32:
    pointwise = _pointwise_k_step(aligned_in,out_channels,_ceildiv(row_bytes*in_h,_BANK_BYTES))
    if full_weight_banks > 3 or out_channels > pointwise: k_step = pointwise
  k_step = min(k_step,max_k_step)
  tile_weight_banks = max(1,_ceildiv(weight_bytes_per_k*k_step,_BANK_BYTES))
  if tile_weight_banks >= _CBUF_BANKS: return None
  data_banks = _CBUF_BANKS-tile_weight_banks

  entries = max(1,_entries_per_slice(width_stride,aligned_in))
  max_input_rows = max(1,_ENTRIES_PER_BANK*data_banks//entries)
  if use_nhwc and spatial:
    max_input_rows = min(max_input_rows,max(1,((_CBUF_BANKS-1)//2)*_BANK_BYTES//row_bytes))
    headroom = 2*kernel_h
  else:
    if spatial: max_input_rows = min(max_input_rows,even_rows)
    headroom = 2*kernel_h if spatial and max(kernel_h,kernel_w) >= 5 else kernel_h
  y_step = min(out_h,max(1,(max_input_rows-headroom)//stride+1))
  if not spatial and in_channels > 64: y_step = min(y_step,even_rows+1)
  needed = max(1,_ceildiv(row_bytes*feature_grains,_BANK_BYTES))
  if needed > data_banks: y_step = min(y_step,max(1,out_h*data_banks//needed))
  if not spatial and in_channels <= 4 and output_width_stride > 992: y_step = min(y_step,max(1,992//out_w))
  if spatial and (out_h > 32 or output_width_stride > 992): y_step = min(y_step,32 if output_width_stride > 992 else 50)

  split = RKConvSplit.BY_YK if y_step < out_h and k_step < out_channels else RKConvSplit.BY_Y if y_step < out_h else \
    RKConvSplit.BY_K if k_step < out_channels else RKConvSplit.NONE
  y_windows, k_windows = _windows(out_h,y_step), _windows(out_channels,k_step)
  tiles = []
  for y_start,output_height in y_windows:
    input_height = min((output_height-1)*stride+kernel_h,in_h-y_start*stride)
    actual_data_banks = max(1,_ceildiv(entries*input_height,_ENTRIES_PER_BANK))
    for k_start,channels in k_windows:
      actual_weight_banks = max(1,_ceildiv(weight_bytes_per_k*channels,_BANK_BYTES))
      if actual_data_banks+actual_weight_banks > _CBUF_BANKS: return None
      tiles.append(RKConvTile(y_start,y_start*stride,input_height,output_height,k_start,channels,
                              actual_data_banks,actual_weight_banks))
  return RKConvTiling(split,y_step,k_step,tuple(tiles))
