from __future__ import annotations
import struct
from itertools import islice

from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops
from tinygrad.renderer.rockchip.access import compact_access_map, RKMultiSourceMap
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.ir import (RKBufferKind, RKLayoutKind, RKReformatKind, RKArg, RKALUStage, RKDPUProgram, RKScratch,
  RKLayout, RKTensorRef, RKCMACTask, RKConvTask, RKReduce, RKReformatPlan, RKLegalizedReformat, RKProgram)
from tinygrad.renderer.rockchip.limits import RK_MAX_CONSTANT_BYTES, RK_MAX_PROGRAM_STAGES

def _cmac_tiled_output_bytes(count:int) -> int:
  # Each logical 16-lane tile is a physical 32-lane write, including the final tile's tail.
  return (((count+15)&-16)+16)*2

def _dense_ref(slot:int, shape:tuple[int, ...], dtype:DType, kind:RKBufferKind=RKBufferKind.ARG) -> RKTensorRef:
  stride, strides = dtype.itemsize, []
  for extent in reversed(shape):
    strides.append(stride)
    stride *= extent
  return RKTensorRef(RKArg(kind, slot), RKLayout(shape, shape, tuple(reversed(strides)), dtype))

def _dense_half_ref(slot:int, shape:tuple[int, ...], kind:RKBufferKind=RKBufferKind.ARG) -> RKTensorRef:
  return _dense_ref(slot,shape,dtypes.half,kind)

def _cmac_weight_ref(slot:int, logical_n:int, k:int, kind:RKBufferKind=RKBufferKind.ARG, physical_n:int|None=None) -> RKTensorRef:
  physical_n = max(32, (logical_n+31)&-32) if physical_n is None else physical_n
  return RKTensorRef(RKArg(kind, slot), RKLayout((logical_n,k), (physical_n,k), (k*2,2), dtypes.half, kind=RKLayoutKind.CMAC_WEIGHT))

def _cmac_mask_payload(count:int, align_in:int, outputs:int=4, scale:float=1.0) -> bytes:
  values = [0] * (32*align_in)
  active = struct.unpack("<H", struct.pack("<e", scale))[0]
  for out in range(outputs):
    for k in range(count): values[(((out//16)*(align_in//32)+(k//32))*16+(out%16))*32+(k%32)] = active
  return struct.pack(f"<{len(values)}H", *values)

def _cmac_weighted_payload(rows:list[list[tuple[int,float]]], align_in:int, align_out:int) -> bytes:
  values = [0.0] * (align_out*align_in)
  for out,terms in enumerate(rows):
    for k,weight in terms:
      packed = (((out//16)*(align_in//32)+(k//32))*16+(out%16))*32+(k%32)
      values[packed] += weight
  return b"".join(struct.pack("<e", value) for value in values)

def _cmac_selection_payload(rows:list[list[int]], align_in:int, align_out:int, scale:float) -> bytes:
  return _cmac_weighted_payload([[(index,scale) for index in row] for row in rows], align_in, align_out)

def _sparse_cmac_pipeline(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]], scale:float=1.0,
                          scratch:tuple[RKScratch, ...]=(), out_dtype:DType=dtypes.half) -> RKProgram:
  """Materialize one static selector matrix as sequential, proven-width CMAC tasks."""
  if out_dtype not in (dtypes.half,dtypes.float) or out_dtype is dtypes.float and len(rows) != 1:
    raise ValueError("sparse CMAC FP32 output requires one scalar row")
  align_in = max(32, (input_count+31)&-32)
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(align_in*2),)
  dpu = RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                      RKALUStage(Ops.ADD, packed, source, 0.0, input_count)), scratch)
  lhs_layout = RKLayout((1,input_count), (1,align_in), (align_in*2,2), dtypes.half,
                        padding=((0,0),(0,align_in-input_count)))
  contracts:list[RKCMACTask] = []
  for start in range(0, len(rows), 16):
    count = min(16, len(rows)-start)
    physical, strides = ((1,64),(256,4)) if out_dtype is dtypes.float else ((1,32),(64,2))
    out_layout = RKLayout((1,count), physical, strides, out_dtype, padding=((0,0),(0,physical[1]-count)))
    contracts.append(RKCMACTask(RKTensorRef(RKArg(output.kind, output.index, output.addend+start*out_dtype.itemsize), out_layout),
      RKTensorRef(packed, lhs_layout), _cmac_weight_ref(0, count, align_in, RKBufferKind.CONSTANT, 32), 0,
      _cmac_selection_payload(rows[start:start+count], align_in, 32, scale)))
  return RKProgram((dpu, *contracts), scratch)

def _windowed_weighted_cmac_pipeline(output:RKArg, source:RKArg, rows:list[list[tuple[int,float]]],
                                     scratch:tuple[RKScratch, ...]=(), direct_count:int=0, max_window:int=512,
                                     max_outputs:int=64, clear_empty:bool=True) -> RKProgram|None:
  """Apply consecutive static weighted rows from bounded, atom-aligned source windows."""
  chunks:list[tuple[int, int, int, int, list[list[tuple[int,float]]], bytes]] = []
  start = 0
  while start < len(rows):
    tile:list[list[tuple[int,float]]] = []
    for candidate in rows[start:start+max_outputs]:
      tile.append(candidate)
      selected = [index for row in tile for index,_ in row]
      if not selected: continue
      base, end = min(selected)&-8, max(selected)+1
      align_in = max(32, (end-base+31)&-32)
      if align_in > max_window:
        tile.pop()
        while len(tile)%8: tile.pop()
        break
    if not tile: return None
    selected = [index for row in tile for index,_ in row]
    if not selected:
      start += len(tile)
      continue
    base, end = min(selected)&-8, max(selected)+1
    align_in = max(32, (end-base+31)&-32)
    align_out = max(32, (len(tile)+31)&-32)
    payload = _cmac_weighted_payload([[(index-base,weight) for index,weight in row] for row in tile], align_in, align_out)
    chunks.append((start, base, end, align_in, tile, payload))
    start += len(tile)
  direct = tuple(base+align_in <= direct_count for _,base,_,align_in,_,_ in chunks)
  if sum(map(len, set(chunk[-1] for chunk in chunks))) > RK_MAX_CONSTANT_BYTES or \
     (1 if any(not row for row in rows) else 0)+sum(1 if safe else 3 for safe in direct) > RK_MAX_PROGRAM_STAGES: return None
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  if not all(direct): scratch += (RKScratch(max((chunk[3] for chunk,safe in zip(chunks,direct) if not safe), default=32)*2),)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = ([RKDPUProgram((RKALUStage(Ops.ADD, output, 0.0, 0.0, len(rows)),))]
                                                  if clear_empty and any(not row for row in rows) else [])
  for (start,base,end,align_in,tile,payload),safe in zip(chunks,direct):
    span, align_out = end-base, max(32, (len(tile)+31)&-32)
    if safe:
      lhs = RKTensorRef(RKArg(source.kind, source.index, source.addend+base*2),
        RKLayout((1,span), (1,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-span))))
    else:
      steps.append(RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
        RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, source.addend+base*2), 0.0, span)), scratch))
      lhs = _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH)
    valid = len(tile)
    out_layout = RKLayout((1,valid), (1,align_out), (align_out*2,2), dtypes.half, padding=((0,0),(0,align_out-valid)))
    steps.append(RKCMACTask(RKTensorRef(RKArg(output.kind, output.index, output.addend+start*2), out_layout),
      lhs, _cmac_weight_ref(0, valid, align_in, RKBufferKind.CONSTANT, align_out), 0, payload, compact_output=True))
  return RKProgram(tuple(steps), scratch)

def _windowed_cmac_pipeline(output:RKArg, source:RKArg, rows:list[list[int]], scale:float=1.0,
                            scratch:tuple[RKScratch, ...]=(), direct_count:int=0, max_window:int=512,
                            max_outputs:int=64, clear_empty:bool=True) -> RKProgram|None:
  """Reduce consecutive output tiles from bounded, atom-aligned source windows."""
  if struct.unpack("<e", struct.pack("<e", scale))[0] != scale: return None
  return _windowed_weighted_cmac_pipeline(output,source,[[(index,scale) for index in row] for row in rows],
                                          scratch,direct_count,max_window,max_outputs,clear_empty)

def _two_level_selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                                scratch:tuple[RKScratch, ...], scale:float=1.0, max_outputs:int=64) -> RKProgram|None:
  groups:list[list[list[int]]] = []
  for row in rows:
    selected = list(row)
    if selected and max(32, (max(selected)+1-(min(selected)&-8)+31)&-32) > 512: return None
    trial = [index for candidate in (*groups[-1],row) for index in candidate] if groups else selected
    if groups and (not trial or max(32, (max(trial)+1-(min(trial)&-8)+31)&-32) <= 512): groups[-1].append(row)
    else: groups.append([row])
  intermediate_rows:list[list[int]] = []
  compact_rows:list[list[int]] = []
  for group in groups:
    base = len(intermediate_rows)
    intermediate_rows.extend(group)
    compact_rows.extend([[base+index] for index in range(len(group))])
    intermediate_rows.extend([[] for _ in range((-len(group))%8)])
  intermediate = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(_cmac_tiled_output_bytes(len(intermediate_rows))),)
  first = _windowed_cmac_pipeline(intermediate, source, intermediate_rows, scratch=scratch, direct_count=input_count,
                                  max_outputs=max_outputs)
  if first is None: return None
  second = _windowed_cmac_pipeline(output, intermediate, compact_rows, scratch=first.scratch,
                                   direct_count=_cmac_tiled_output_bytes(len(intermediate_rows))//2, scale=scale,
                                   max_outputs=max_outputs)
  return None if second is None else _finish_program([*first.steps,*second.steps], second.scratch)

def _selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                      scratch:tuple[RKScratch, ...], direct_capacity:int|None=None, max_window:int=512,
                      max_outputs:int=64) -> RKProgram|None:
  sparse_bytes = ((len(rows)+15)//16)*32*max(32,(input_count+31)&-32)*2
  sparse = _sparse_cmac_pipeline(output,source,input_count,rows,scratch=scratch) \
    if input_count <= 512 and sparse_bytes <= RK_MAX_CONSTANT_BYTES else None
  candidates = (sparse,_windowed_cmac_pipeline(output, source, rows, scratch=scratch,
                                               direct_count=input_count if direct_capacity is None else direct_capacity,
                                               max_window=max_window, max_outputs=max_outputs))
  legal = tuple((cost,plan) for plan in candidates if plan is not None and (cost:=plan_cost(plan)).stage_count <= RK_MAX_PROGRAM_STAGES and
                cost.constant_bytes <= RK_MAX_CONSTANT_BYTES)
  if legal:
    return min(legal, key=lambda item:(item[0].reset_count, item[0].estimated_macs,
      item[0].estimated_read_bytes+item[0].estimated_write_bytes, item[0].command_words,
      item[0].constant_bytes, item[0].scratch_bytes))[1]
  two_level = _two_level_selector_program(output, source, input_count, rows, scratch, max_outputs=max_outputs)
  return two_level if two_level is not None and plan_cost(two_level).stage_count <= RK_MAX_PROGRAM_STAGES and \
    plan_cost(two_level).constant_bytes <= RK_MAX_CONSTANT_BYTES else None

def _multi_source_windowed_program(output:RKArg, sources:tuple[RKArg, ...], source_counts:tuple[int, ...],
                                   access:RKMultiSourceMap, max_outputs:int=128, max_window:int=512) -> RKProgram|None:
  """Select aligned output tiles directly from their source surfaces, combining only tiles which cross a source boundary."""
  scratch:tuple[RKScratch, ...] = (RKScratch(_cmac_tiled_output_bytes(max_outputs)),)
  partial = RKArg(RKBufferKind.SCRATCH, 0)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  values, start = access.values(), 0
  while tile := tuple(islice(values,max_outputs)):
    target = RKArg(output.kind,output.index,output.addend+start*2)
    source_ids = tuple(dict.fromkeys(sid for sid,_ in tile))
    for position,sid in enumerate(source_ids):
      rows = [[src] if row_sid == sid else [] for row_sid,src in tile]
      selected = _windowed_cmac_pipeline(target if position == 0 else partial,sources[sid],rows,scratch=scratch,
                                         direct_count=source_counts[sid],max_outputs=max_outputs,max_window=max_window)
      if selected is None: return None
      steps.extend(selected.steps)
      scratch = selected.scratch
      if position:
        steps.append(RKDPUProgram((RKALUStage(Ops.ADD,target,target,partial,len(tile)),),scratch))
    start += len(tile)
  return _finish_program(steps,scratch)

def _partitioned_selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                                  scratch:tuple[RKScratch, ...]=(), max_window:int=512,
                                  max_outputs:int=128) -> RKProgram|None:
  """Split aligned output tiles by bounded source windows, then combine their disjoint selector results on the DPU."""
  partial = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(_cmac_tiled_output_bytes(max_outputs)),)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  for start in range(0,len(rows),max_outputs):
    tile = rows[start:start+max_outputs]
    selected = sorted({index for row in tile for index in row})
    target = RKArg(output.kind,output.index,output.addend+start*2)
    if not selected:
      steps.append(RKDPUProgram((RKALUStage(Ops.ADD,target,0.0,0.0,len(tile)),),scratch))
      continue
    windows:list[tuple[int,int]] = []
    for index in selected:
      if not windows or index-(windows[-1][0]&-8) >= max_window: windows.append((index,index+1))
      else: windows[-1] = (windows[-1][0],index+1)
    for position,(lo,hi) in enumerate(windows):
      window_rows = [[index for index in row if lo <= index < hi] for row in tile]
      plan = _windowed_cmac_pipeline(target if position == 0 else partial,source,window_rows,scratch=scratch,
                                     direct_count=input_count,max_window=max_window,max_outputs=max_outputs,clear_empty=False)
      if plan is None: return None
      steps.extend(plan.steps)
      scratch = plan.scratch
      if position: steps.append(RKDPUProgram((RKALUStage(Ops.ADD,target,target,partial,len(tile)),),scratch))
  return _finish_program(steps,scratch)

def _best_partitioned_selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                                       scratch:tuple[RKScratch, ...]=()) -> RKProgram|None:
  candidates = tuple(_partitioned_selector_program(output,source,input_count,rows,scratch,max_outputs=width)
                     for width in range(64,129,8))
  legal = tuple((cost,plan) for plan in candidates if plan is not None and (cost:=plan_cost(plan)).stage_count <= RK_MAX_PROGRAM_STAGES and
                cost.constant_bytes <= RK_MAX_CONSTANT_BYTES)
  return None if not legal else min(legal,key=lambda item:(item[0].reset_count,item[0].estimated_macs,
    item[0].estimated_read_bytes+item[0].estimated_write_bytes,item[0].command_words,
    item[0].constant_bytes,item[0].scratch_bytes))[1]

def _finish_program(steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce], scratch:tuple[RKScratch, ...]) -> RKProgram:
  """Give every ordered DPU step the program's final resource table."""
  return RKProgram(tuple(RKDPUProgram(step.stages, scratch) if isinstance(step, RKDPUProgram) else step for step in steps), scratch)

def _legalized_reformat(out:RKTensorRef, src:RKTensorRef, mapping:tuple[int, ...], fill:float,
                        kind:RKReformatKind, program:RKProgram) -> RKLegalizedReformat:
  return RKLegalizedReformat(RKReformatPlan(out, src, compact_access_map(mapping), fill), kind, program)

def _periodic_selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                               scratch:tuple[RKScratch, ...]=()) -> RKProgram|None:
  """Materialize one aligned repeated-map period, then duplicate it through direct DPU copies."""
  count = len(rows)
  period = next((candidate for candidate in range(8,min(count,32769),8) if count%candidate == 0 and
                 all(rows[index] == rows[index%candidate] for index in range(candidate,count))), None)
  if period is None: return None
  prefix = _selector_program(output, source, input_count, rows[:period], scratch)
  if prefix is None: return None
  copy_stages:list[RKALUStage] = []
  filled = period
  while filled < count:
    copied = min(filled, count-filled, 32768)
    copy_stages.append(RKALUStage(Ops.ADD, RKArg(output.kind,output.index,filled*2), output, 0.0, copied))
    filled += copied
  copies = RKDPUProgram(tuple(copy_stages))
  completed = _finish_program([*prefix.steps,copies], prefix.scratch)
  cost = plan_cost(completed)
  return completed if cost.stage_count <= RK_MAX_PROGRAM_STAGES and cost.constant_bytes <= RK_MAX_CONSTANT_BYTES else None

def _constant_run_selector_program(output:RKArg, source:RKArg, input_count:int, mapping:list[int],
                                   scratch:tuple[RKScratch, ...]=()) -> RKProgram|None:
  """Materialize aligned constant-run heads, then expand each run through geometric DPU copies."""
  runs:list[tuple[int,int,int]] = []
  start = 0
  while start < len(mapping):
    end = start+1
    while end < len(mapping) and mapping[end] == mapping[start]: end += 1
    if start%8 or end-start < 32 or (end-start)%8 or not 0 <= mapping[start] < input_count: return None
    runs.append((start,end,mapping[start]))
    start = end
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  for start,end,source_index in runs:
    head = _selector_program(RKArg(output.kind,output.index,output.addend+start*2), source, input_count, [[source_index]]*8, scratch)
    if head is None: return None
    steps.extend(head.steps)
    scratch = head.scratch
  copies:list[RKALUStage] = []
  for start,end,_ in runs:
    filled = 8
    while start+filled < end:
      copied = min(filled, end-start-filled, 32768)
      copies.append(RKALUStage(Ops.ADD, RKArg(output.kind,output.index,output.addend+(start+filled)*2),
                               RKArg(output.kind,output.index,output.addend+start*2), 0.0, copied))
      filled += copied
  completed = _finish_program([*steps,RKDPUProgram(tuple(copies))], scratch)
  cost = plan_cost(completed)
  return completed if cost.stage_count <= RK_MAX_PROGRAM_STAGES and cost.constant_bytes <= RK_MAX_CONSTANT_BYTES else None
