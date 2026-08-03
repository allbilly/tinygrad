from __future__ import annotations
import math, os, struct
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable, cast
from tinygrad.dtype import dtypes, Invalid
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId as RKLUTId
from tinygrad.runtime.support.rockchip_telemetry import record as record_telemetry
from tinygrad.uop.ops import AddrSpace, Ops, ProgramInfo, UOp

from tinygrad.renderer.rockchip.ir import (RKTarget as RKTarget, RKEngine as RKEngine, RKBufferKind, RKLayoutKind, RKReformatKind, RKArg,
  RKALUStage, RKFusedALUStage as RKFusedALUStage,
  RKMaskStage as RKMaskStage, RKLUTStage as RKLUTStage, RKDPUStage,
  RKScratch, RKDPUProgram, RKLayout, RKTensorRef, RKEpilogue, RKContract, RKSpatialConv, RKReduce, RKReformat, RKProgram, RKPlanCost,
  RKRejectKind, RKReject, RKLowerKind, RKLowerResult)
from tinygrad.renderer.rockchip.image import (RK_STAGE_RESET as RK_STAGE_RESET, RKReloc as RKReloc, RKStage as RKStage, RKImage as RKImage,
  encode_image, decode_image as decode_image,
  patch_image as patch_image, validate_image as validate_image)
from tinygrad.renderer.rockchip.affine import affine as _affine, rk_fingerprint as rk_fingerprint
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr

RK_MAX_CONSTANT_BYTES = 2*1024*1024
RK_MAX_AFFINE_VISITS = 65536
RK_MAX_PROGRAM_STAGES = 400
RK_MAX_AFFINE_WINDOW = 192
RK_MAX_CMAC_SELECTOR_WINDOW = 1504

def _fp16_exact(value:float) -> bool:
  try: rounded = struct.unpack("<e", struct.pack("<e", value))[0]
  except OverflowError: return False
  return math.isnan(value) and math.isnan(rounded) or rounded == value

def _cmac_tiled_output_bytes(count:int) -> int:
  # Each logical 16-lane tile is a physical 32-lane write, including the final tile's tail.
  return (((count+15)&-16)+16)*2

def _dense_half_ref(slot:int, shape:tuple[int, ...], kind:RKBufferKind=RKBufferKind.ARG) -> RKTensorRef:
  stride, strides = 2, []
  for extent in reversed(shape):
    strides.append(stride)
    stride *= extent
  return RKTensorRef(RKArg(kind, slot), RKLayout(shape, shape, tuple(reversed(strides)), dtypes.half))

def _cmac_weight_ref(slot:int, logical_n:int, k:int, kind:RKBufferKind=RKBufferKind.ARG, physical_n:int|None=None) -> RKTensorRef:
  physical_n = max(32, (logical_n+31)&-32) if physical_n is None else physical_n
  return RKTensorRef(RKArg(kind, slot), RKLayout((logical_n,k), (physical_n,k), (k*2,2), dtypes.half, kind=RKLayoutKind.CMAC_WEIGHT))

def _cmac_mask_payload(count:int, align_in:int, outputs:int=4, scale:float=1.0) -> bytes:
  values = [0] * (32*align_in)
  active = struct.unpack("<H", struct.pack("<e", scale))[0]
  for out in range(outputs):
    for k in range(count): values[(((out//16)*(align_in//32)+(k//32))*16+(out%16))*32+(k%32)] = active
  return struct.pack(f"<{len(values)}H", *values)

def _cmac_selection_payload(rows:list[list[int]], align_in:int, align_out:int, scale:float) -> bytes:
  values = [0.0] * (align_out*align_in)
  for out,indexes in enumerate(rows):
    for k in indexes:
      packed = (((out//16)*(align_in//32)+(k//32))*16+(out%16))*32+(k%32)
      values[packed] += scale
  return b"".join(struct.pack("<e", value) for value in values)

def _sparse_cmac_pipeline(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]], scale:float=1.0,
                          scratch:tuple[RKScratch, ...]=()) -> RKProgram:
  """Materialize one static selector matrix as sequential, proven-width CMAC tasks."""
  align_in = max(32, (input_count+31)&-32)
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(align_in*2),)
  dpu = RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                      RKALUStage(Ops.ADD, packed, source, 0.0, input_count)), scratch)
  lhs_layout = RKLayout((1,input_count), (1,align_in), (align_in*2,2), dtypes.half,
                        padding=((0,0),(0,align_in-input_count)))
  contracts:list[RKContract] = []
  for start in range(0, len(rows), 16):
    count = min(16, len(rows)-start)
    out_layout = RKLayout((1,count), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-count)))
    contracts.append(RKContract(RKTensorRef(RKArg(output.kind, output.index, output.addend+start*2), out_layout),
      RKTensorRef(packed, lhs_layout), _cmac_weight_ref(0, count, align_in, RKBufferKind.CONSTANT, 32), 0,
      _cmac_selection_payload(rows[start:start+count], align_in, 32, scale)))
  return RKProgram((dpu, *contracts), scratch)

def _windowed_cmac_pipeline(output:RKArg, source:RKArg, rows:list[list[int]], scale:float=1.0,
                            scratch:tuple[RKScratch, ...]=(), direct_count:int=0, max_window:int=512) -> RKProgram|None:
  """Reduce consecutive output tiles from bounded, atom-aligned source windows."""
  if struct.unpack("<e", struct.pack("<e", scale))[0] != scale: return None
  chunks:list[tuple[int, int, int, list[list[int]], bytes]] = []
  start = 0
  while start < len(rows):
    tile:list[list[int]] = []
    for candidate in rows[start:start+32]:
      tile.append(candidate)
      selected = [index for row in tile for index in row]
      if not selected: continue
      base, end = min(selected)&-8, max(selected)+1
      span, align_in = end-base, max(32, (end-base+31)&-32)
      if align_in > max_window:
        tile.pop()
        while len(tile)%8: tile.pop()
        break
    if not tile: return None
    selected = [index for row in tile for index in row]
    if not selected:
      start += len(tile)
      continue
    base, end = min(selected)&-8, max(selected)+1
    span, align_in = end-base, max(32, (end-base+31)&-32)
    payload = _cmac_selection_payload([[index-base for index in row] for row in tile], align_in, 32, scale)
    chunks.append((start, base, end, tile, payload))
    start += len(tile)
  direct = tuple(base+len(payload)//64 <= direct_count for _,base,_,_,payload in chunks)
  if sum(map(len, set(chunk[-1] for chunk in chunks))) > RK_MAX_CONSTANT_BYTES or \
     (1 if any(not row for row in rows) else 0)+sum(1 if safe else 3 for safe in direct) > RK_MAX_PROGRAM_STAGES: return None
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  if not all(direct): scratch += (RKScratch(max((len(chunk[-1])//64 for chunk,safe in zip(chunks,direct) if not safe), default=32)*2),)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = ([RKDPUProgram((RKALUStage(Ops.ADD, output, 0.0, 0.0, len(rows)),))]
                                                  if any(not row for row in rows) else [])
  for (start,base,end,tile,payload),safe in zip(chunks,direct):
    span, align_in = end-base, len(payload)//64
    if safe:
      lhs = RKTensorRef(RKArg(source.kind, source.index, source.addend+base*2),
        RKLayout((1,span), (1,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-span))))
    else:
      steps.append(RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
        RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, source.addend+base*2), 0.0, span)), scratch))
      lhs = _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH)
    valid = len(tile)
    out_layout = RKLayout((1,valid), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-valid)))
    steps.append(RKContract(RKTensorRef(RKArg(output.kind, output.index, output.addend+start*2), out_layout),
      lhs, _cmac_weight_ref(0, valid, align_in, RKBufferKind.CONSTANT, 32), 0, payload, compact_output=True))
  return RKProgram(tuple(steps), scratch)

def plan_cost(plan:RKProgram) -> RKPlanCost:
  image = emit_program(plan)
  reads = writes = macs = 0
  for step in plan.steps:
    if isinstance(step, RKDPUProgram):
      for stage in step.stages:
        if isinstance(stage, RKALUStage):
          reads += sum(stage.count*2 for operand in (stage.lhs,stage.rhs) if isinstance(operand,RKArg))
          writes += stage.count*stage.out_dtype.itemsize
          macs += stage.count
        elif isinstance(stage, RKFusedALUStage):
          reads += stage.count*(2+4+2) + (stage.count*2 if isinstance(stage.bn,RKArg) else 0)
          writes += stage.count*2
          macs += stage.count*3
        elif isinstance(stage, (RKMaskStage,RKLUTStage)):
          reads += stage.count*2
          writes += stage.count*2
          macs += stage.count
    elif isinstance(step, RKContract):
      m, n, k = math.prod(step.lhs.layout.logical_shape[:-1]), step.rhs.layout.logical_shape[0], step.lhs.layout.logical_shape[-1]
      reads += math.prod(step.lhs.layout.logical_shape)*step.lhs.layout.dtype.itemsize
      reads += math.prod(step.rhs.layout.logical_shape)*step.rhs.layout.dtype.itemsize
      if step.epilogue is not None and step.epilogue.bias is not None:
        reads += math.prod(step.epilogue.bias.layout.logical_shape)*step.epilogue.bias.layout.dtype.itemsize
      writes += math.prod(step.out.layout.logical_shape)*step.out.layout.dtype.itemsize
      macs += m*n*k
    elif isinstance(step, RKSpatialConv):
      reads += math.prod(step.src.layout.logical_shape)*step.src.layout.dtype.itemsize
      reads += math.prod(step.weight.layout.logical_shape)*step.weight.layout.dtype.itemsize
      writes += step.out_channels*step.output_height*step.output_width*2
      macs += step.out_channels*step.output_height*step.output_width*step.in_channels*step.kernel_height*step.kernel_width
    elif isinstance(step, RKReformat):
      nested = plan_cost(RKProgram(step.steps, step.scratch))
      reads += nested.estimated_read_bytes
      writes += nested.estimated_write_bytes
      macs += nested.estimated_macs
    elif isinstance(step, RKReduce):
      reads += math.prod(step.src.layout.logical_shape)*step.src.layout.dtype.itemsize
      writes += math.prod(step.out.layout.logical_shape)*step.out.layout.dtype.itemsize
      macs += math.prod(step.src.layout.logical_shape)
    else: raise TypeError(f"unsupported Rockchip cost step {type(step).__name__}")
  return RKPlanCost(len(image.stages), sum(len(stage.commands) for stage in image.stages),
                    sum(bool(stage.flags & RK_STAGE_RESET) for stage in image.stages), len(image.constants),
                    sum(resource.size for resource in plan.scratch), reads, writes, macs)

def _two_level_selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                                scratch:tuple[RKScratch, ...], scale:float=1.0) -> RKProgram|None:
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
  first = _windowed_cmac_pipeline(intermediate, source, intermediate_rows, scratch=scratch, direct_count=input_count)
  if first is None: return None
  second = _windowed_cmac_pipeline(output, intermediate, compact_rows, scratch=first.scratch,
                                   direct_count=_cmac_tiled_output_bytes(len(intermediate_rows))//2, scale=scale)
  return None if second is None else _finish_program([*first.steps,*second.steps], second.scratch)

def _selector_program(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]],
                      scratch:tuple[RKScratch, ...], direct_capacity:int|None=None, max_window:int=512) -> RKProgram|None:
  sparse_bytes = ((len(rows)+15)//16)*32*max(32,(input_count+31)&-32)*2
  sparse = _sparse_cmac_pipeline(output,source,input_count,rows,scratch=scratch) if sparse_bytes <= RK_MAX_CONSTANT_BYTES else None
  candidates = (sparse,_windowed_cmac_pipeline(output, source, rows, scratch=scratch,
                                               direct_count=input_count if direct_capacity is None else direct_capacity,
                                               max_window=max_window))
  legal = tuple((cost,plan) for plan in candidates if plan is not None and (cost:=plan_cost(plan)).stage_count <= RK_MAX_PROGRAM_STAGES and
                cost.constant_bytes <= RK_MAX_CONSTANT_BYTES)
  if legal:
    return min(legal, key=lambda item:(item[0].reset_count, item[0].estimated_macs,
      item[0].estimated_read_bytes+item[0].estimated_write_bytes, item[0].command_words,
      item[0].constant_bytes, item[0].scratch_bytes))[1]
  two_level = _two_level_selector_program(output, source, input_count, rows, scratch)
  return two_level if two_level is not None and plan_cost(two_level).stage_count <= RK_MAX_PROGRAM_STAGES and \
    plan_cost(two_level).constant_bytes <= RK_MAX_CONSTANT_BYTES else None

def _finish_program(steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce], scratch:tuple[RKScratch, ...]) -> RKProgram:
  """Give every ordered DPU step the program's final resource table."""
  return RKProgram(tuple(RKDPUProgram(step.stages, scratch) if isinstance(step, RKDPUProgram) else step for step in steps), scratch)

def _native(plan:RKDPUProgram|RKContract|RKSpatialConv|RKReduce|RKReformat|RKProgram) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.NATIVE, plan=plan)
def _not_applicable() -> RKLowerResult: return RKLowerResult(RKLowerKind.NOT_APPLICABLE)
def _unsupported(kind:RKRejectKind, detail:str, node_op:Ops|None=None) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(kind, detail, node_op))

from tinygrad.renderer.rockchip.expr import (_ALUExpr, _MaskExpr, _LUTExpr, _Expr, _Value, _parse_alu, _unwrap_same_cast,
  _canonical_lerp, _numerical_contract)

def _lower_fused_lerp(output:RKArg, operands:tuple[UOp,UOp,UOp], count:int) -> RKLowerResult:
  x,y,z = operands
  if x.op is not Ops.INDEX or y.op is not Ops.INDEX or z.op not in (Ops.INDEX,Ops.CONST): return _not_applicable()
  source = RKArg(RKBufferKind.ARG, x.src[0].arg.slot)
  x_float = RKArg(RKBufferKind.SCRATCH, 0)
  scratch = (RKScratch((((count+31)//32)*32+32)*4),)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  for start in range(0,count,32):
    valid = min(32,count-start)
    lhs_layout = RKLayout((1,valid), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-valid)))
    out_layout = RKLayout((1,valid), (1,64), (256,4), dtypes.float, padding=((0,0),(0,64-valid)))
    steps.append(RKContract(RKTensorRef(RKArg(x_float.kind,x_float.index,start*4),out_layout),
      RKTensorRef(RKArg(source.kind,source.index,source.addend+start*2),lhs_layout),
      _cmac_weight_ref(0,valid,32,RKBufferKind.CONSTANT,32), 0,
      _cmac_selection_payload([[lane] for lane in range(32)],32,32,1.0)))
  bn:RKArg|float = RKArg(RKBufferKind.ARG, z.src[0].arg.slot) if z.op is Ops.INDEX else float(z.arg)
  stages:list[RKDPUStage] = []
  for start in range(0,count,8):
    tile = min(8,count-start)
    tile_bn = RKArg(bn.kind,bn.index,bn.addend+start*2) if isinstance(bn,RKArg) else bn
    stages.append(RKFusedALUStage(RKArg(output.kind,output.index,output.addend+start*2),
      RKArg(RKBufferKind.ARG,y.src[0].arg.slot,start*2), Ops.SUB, RKArg(x_float.kind,x_float.index,start*4),
      Ops.MUL, tile_bn, Ops.ADD, RKArg(RKBufferKind.ARG,x.src[0].arg.slot,start*2), tile))
  program = _finish_program([*steps,RKDPUProgram(tuple(stages))], scratch)
  cost = plan_cost(program)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"fused ALU needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", Ops.ADD)
  return _native(program)

def lower_dpu_result(sink:UOp) -> RKLowerResult:
  """Lower one contiguous expression or native wide constant fill to a typed DPU result."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, f"expected one store, found {len(stores)}", Ops.STORE)
  store = stores[0]
  if store.src[0].op is not Ops.INDEX: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "output is not an indexed surface", store.src[0].op)
  if store.src[0].dtype not in (dtypes.half, dtypes.int, dtypes.float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"output dtype {store.src[0].dtype.name}", store.src[0].op)
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op not in (Ops.RANGE, Ops.CONST) or out_param.src[0].op is not Ops.CONST:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "output surface is not contiguous", out_index.op)
  count = int(out_param.src[0].arg)
  if not 0 < count <= 65536 or (out_index.op is Ops.RANGE and int(out_index.src[0].arg) != count) or \
     (out_index.op is Ops.CONST and (count != 1 or int(out_index.arg) != 0)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"unsupported contiguous output extent {count}", out_index.op)
  input_indexes = [u for u in store.src[1].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  # Rejected WIP: DATA_FORMAT in_precision=precision_float32 exists in the register enum, but a direct FP32->FP16 ADD timed out on RK3588.
  # The exact typed-stage/emitter probe is preserved as wip-native-fp32-dpu-input-timeout.patch; do not restore 2607's CPU narrowing instead.
  if (bad_dtype:=next((u.dtype for u in input_indexes if u.dtype is not dtypes.half), None)) is not None:
    return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE, f"input dtype {bad_dtype.name}", Ops.INDEX)
  if any(u.src[1].key != out_index.key for u in input_indexes):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "input index map differs from output surface", Ops.INDEX)
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if (lerp:=_canonical_lerp(store.src[1])) is not None:
    result = _lower_fused_lerp(output, lerp, count)
    if result.kind is not RKLowerKind.NOT_APPLICABLE: return result
  if (reason:=_numerical_contract(store.src[1])) is not None:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, reason, _unwrap_same_cast(store.src[1]).op)
  root = _parse_alu(store.src[1], out_index, {})
  if root is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "expression is not legal DPU arithmetic", _unwrap_same_cast(store.src[1]).op)
  if store.src[0].dtype is not dtypes.half and not isinstance(root, float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"non-constant {store.src[0].dtype.name} arithmetic", store.src[1].op)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      if not isinstance(root, float):
        return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"non-constant {store.src[0].dtype.name} fill", store.src[1].op)
      if not _fp16_exact(root):
        return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,
          f"{store.src[0].dtype.name} fill value {root!r} is not exactly representable by the FP16 DPU input", store.src[1].op)
      tile = 64 if store.src[0].dtype is dtypes.int else 4
      fill_stages = tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*4), 0.0, root, min(tile, count-start),
                                     store.src[0].dtype) for start in range(0, count, tile))
      return _native(RKDPUProgram(fill_stages))
    return _native(RKDPUProgram(tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*2), 0.0, root,
      min(32768, count-start)) for start in range(0, count, 32768))))
  if (program:=_schedule_expr(root, output, count)) is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "stage source is not materializable")
  return _native(program)

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Compatibility helper for compiler probes; production lowering consumes `lower_dpu_result`."""
  return cast(RKDPUProgram|None, lower_dpu_result(sink).plan)

def _strip_casts(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _static_scalar(u:UOp, ranges:dict[int, int]) -> int|float|bool|None:
  """Evaluate one compile-time coordinate predicate; tensor loads are never accepted."""
  if u.op is Ops.CAST: return _static_scalar(u.src[0], ranges)
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.RANGE: return ranges.get(u.arg[0])
  values = tuple(_static_scalar(x, ranges) for x in u.src)
  if any(x is None for x in values): return None
  if u.op is Ops.ADD: return values[0]+values[1]  # type: ignore[operator]
  if u.op is Ops.MUL: return values[0]*values[1]  # type: ignore[operator]
  if u.op is Ops.MAX: return max(values)  # type: ignore[type-var]
  if u.op is Ops.FLOORDIV: return int(cast(int|float|bool, values[0]))//int(cast(int|float|bool, values[1]))
  if u.op is Ops.FLOORMOD: return int(cast(int|float|bool, values[0]))%int(cast(int|float|bool, values[1]))
  if u.op is Ops.CMPLT: return values[0] < values[1]  # type: ignore[operator]
  if u.op is Ops.CMPNE: return values[0] != values[1]
  if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
  if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
  if u.op is Ops.WHERE: return values[1] if values[0] else values[2]
  return None

def _conditional_index(u:UOp) -> tuple[UOp, UOp|None, bool]|None:
  """Return the indexed tensor and an optional static zero-mask around it."""
  value = _strip_casts(u)
  if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM: return value, None, True
  if value.op is not Ops.WHERE: return None
  condition, positive, negative = value.src
  positive, negative = _strip_casts(positive), _strip_casts(negative)
  if positive.op is Ops.INDEX and positive.src[0].op is Ops.PARAM and negative.op is Ops.CONST and float(negative.arg) == 0:
    return positive, condition, True
  if negative.op is Ops.INDEX and negative.src[0].op is Ops.PARAM and positive.op is Ops.CONST and float(positive.arg) == 0:
    return negative, condition, False
  return None

def _static_index_selected(u:UOp, index:UOp, ranges:dict[int, int]) -> bool|None:
  """Follow static WHERE branches and report whether one coordinate selects `index`."""
  value = _strip_casts(u)
  if value.key == index.key: return True
  if value.op in (Ops.CONST, Ops.INDEX): return False
  if value.op is not Ops.WHERE: return None
  predicate = _static_scalar(value.src[0], ranges)
  if predicate is None: return None
  return _static_index_selected(value.src[1] if predicate else value.src[2], index, ranges)

def _relu_source(u:UOp) -> UOp|None:
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  cond, positive, zero = u.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2 or cond.src[0].op is not Ops.CONST or float(cond.src[0].arg) != 0 or \
     zero.op is not Ops.CONST or float(zero.arg) != 0: return None
  source = _strip_casts(cond.src[1])
  return source if _strip_casts(positive).key == source.key else None

def _contract_bias_epilogue(stored:UOp, reduce:UOp) -> tuple[UOp, bool]|None:
  """Recognize a channel-bias ADD, optionally followed by ReLU, directly around one contraction."""
  relu_source, relu = _relu_source(stored), False
  if relu_source is not None: stored, relu = relu_source, True
  stored = _strip_casts(stored)
  if stored.op is not Ops.ADD: return None
  for reduced,bias in (stored.src, stored.src[::-1]):
    bias = _strip_casts(bias)
    if _strip_casts(reduced).key == reduce.key and bias.op is Ops.INDEX and bias.src[0].op is Ops.PARAM and bias.dtype is dtypes.half:
      return bias, relu
  return None

def lower_reformat_result(sink:UOp) -> RKLowerResult:
  """Lower a static affine movement through atom copies or a sparse CMAC selector."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM:
    return _not_applicable()
  value, condition, select_true = _strip_casts(store.src[1]), None, True
  if value.op is Ops.WHERE:
    cond, positive, negative = value.src
    positive, negative = _strip_casts(positive), _strip_casts(negative)
    if positive.op is Ops.INDEX and negative.op is Ops.CONST and float(negative.arg) == 0:
      value, condition = positive, cond
    elif negative.op is Ops.INDEX and positive.op is Ops.CONST and float(positive.arg) == 0:
      value, condition, select_true = negative, cond, False
    else: return _not_applicable()
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or len(value.src) != 2: return _not_applicable()
  if store.src[0].dtype is not dtypes.half or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "atom reformat requires FP16 input and output", store.src[0].op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  axes = tuple(sorted(out_aff[0].keys() | src_aff[0].keys()))
  if any(axis not in ranges for axis in axes): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "dynamic reformat range", Ops.RANGE)
  count, src_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  mapping = [-2] * count
  for coordinates in product(*(range(ranges[axis]) for axis in axes)):
    point = dict(zip(axes, coordinates))
    dst = out_aff[1] + sum(out_aff[0].get(axis, 0)*point[axis] for axis in axes)
    src = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in axes)
    selected = True
    if condition is not None:
      if (predicate:=_static_scalar(condition, point)) is None:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat predicate is not static", condition.op)
      selected = bool(predicate) is select_true
    if not 0 <= dst < count or mapping[dst] != -2 or selected and not 0 <= src < src_count:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat does not cover one dense output", Ops.INDEX)
    mapping[dst] = src if selected else -1
  if any(source == -2 for source in mapping): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat output has holes", Ops.INDEX)
  output, source = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  stages:list[RKDPUStage] = []
  atom_reject:RKLowerResult|None = None
  dst = 0
  while dst < count:
    valid, src = min(8, count-dst), mapping[dst]
    if src < 0:
      atom_reject = _unsupported(RKRejectKind.REQUIRES_REFORMAT, f"movement output atom {dst} includes a static zero", Ops.WHERE)
      break
    if src % 8:
      atom_reject = _unsupported(RKRejectKind.UNALIGNED_ROW, f"movement source atom begins at element {src}", Ops.INDEX)
      break
    if mapping[dst:dst+valid] != list(range(src, src+valid)):
      atom_reject = _unsupported(RKRejectKind.REQUIRES_REFORMAT, f"movement breaks FP16 destination atom at element {dst}", Ops.INDEX)
      break
    length = valid
    while dst+length < count:
      following = min(8, count-dst-length)
      if mapping[dst+length] != src+length or mapping[dst+length:dst+length+following] != list(range(src+length, src+length+following)): break
      length += following
    stages.append(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, dst*2), RKArg(source.kind, source.index, src*2), 0.0, length))
    dst += length
  out_ref, src_ref = _dense_half_ref(output.index, (count,)), _dense_half_ref(source.index, (src_count,))
  if atom_reject is None:
    return _native(RKReformat(out_ref, src_ref, tuple(mapping), RKReformatKind.COALESCED_DPU, (RKDPUProgram(tuple(stages)),)))
  align_in = max(32, (src_count+31)&-32)
  constant_bytes = ((count+15)//16)*32*align_in*2
  if 0 < src_count <= 512 and 0 < count <= 4096 and constant_bytes <= RK_MAX_CONSTANT_BYTES:
    implementation = _sparse_cmac_pipeline(output, source, src_count, [[src] if src >= 0 else [] for src in mapping])
    return _native(RKReformat(out_ref, src_ref, tuple(mapping), RKReformatKind.SELECTOR_CMAC,
      cast(tuple[RKDPUProgram|RKContract, ...], implementation.steps), implementation.scratch))
  return atom_reject

def lower_broadcast_alu_result(sink:UOp) -> RKLowerResult:
  """Materialize one static affine or zero-masked FP16 surface, then schedule generic DPU arithmetic."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  output_index, stored = store.src[0].src[1], _strip_casts(store.src[1])
  out_aff = _affine(output_index)
  if out_aff is None: return _not_applicable()
  indexes = [u for u in stored.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  broadcast = [u for u in indexes if u.src[1].key != output_index.key]
  if len(broadcast) != 1 or any(u.dtype is not dtypes.half for u in indexes): return _not_applicable()
  source_index = broadcast[0]
  surfaces = [(u, parsed) for u in stored.toposort() if (parsed:=_conditional_index(u)) is not None and parsed[0].key == source_index.key]
  if not surfaces: return _not_applicable()
  surface, (_, condition, select_true) = surfaces[-1]
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  axes = tuple(sorted(out_aff[0]))
  surface_axes = {u.arg[0] for u in surface.toposort() if u.op is Ops.RANGE}
  if not axes or any(axis not in ranges for axis in axes) or surface_axes-set(axes):
    return _unsupported(RKRejectKind.UNSUPPORTED_BROADCAST, "broadcast axes are not one static affine output", Ops.RANGE)
  count, src_count = int(store.src[0].src[0].src[0].arg), int(source_index.src[0].src[0].arg)
  mapping = [-2]*count
  for coordinates in product(*(range(ranges[axis]) for axis in axes)):
    point = dict(zip(axes, coordinates))
    dest_offset = out_aff[1] + sum(out_aff[0].get(axis, 0)*point[axis] for axis in axes)
    predicate = True if condition is None else _static_scalar(condition, point)
    if predicate is None:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "broadcast predicate is not static", condition.op if condition is not None else Ops.WHERE)
    selected = bool(predicate) is select_true
    source_offset = _static_scalar(source_index.src[1], point) if selected else -1
    if not 0 <= dest_offset < count or mapping[dest_offset] != -2 or selected and \
       (not isinstance(source_offset, int) or isinstance(source_offset, bool) or not 0 <= source_offset < src_count):
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "broadcast does not cover one dense output", Ops.INDEX)
    mapping[dest_offset] = cast(int, source_offset)
  if any(index == -2 for index in mapping): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "broadcast output has holes", Ops.INDEX)
  if not 0 < count <= 4096 or not 0 < src_count:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"broadcast surface is {count} from {src_count}", Ops.INDEX)

  canonical = source_index.replace(src=(source_index.src[0], output_index))
  root = _parse_alu(stored.substitute({surface:canonical}), output_index, {})
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "broadcast expression is not legal DPU arithmetic", stored.op)
  old_arg, expanded = RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot), RKArg(RKBufferKind.SCRATCH, 0)
  def remap(value:_Value) -> _Value:
    if value == old_arg: return expanded
    if isinstance(value, _ALUExpr): return _ALUExpr(value.op, (remap(value.src[0]), remap(value.src[1])))
    if isinstance(value, _MaskExpr): return _MaskExpr((cast(_Expr|RKArg, remap(value.src[0])),))
    if isinstance(value, _LUTExpr): return _LUTExpr(value.lut, (cast(_Expr|RKArg, remap(value.src[0])),))
    return value
  root = cast(_Expr, remap(root))

  # Group 16-output CMAC tiles while their source window remains small. This keeps a padded
  # row at 64 inputs instead of constructing one output_count x source_count selector.
  blocks = [(start, mapping[start:start+16]) for start in range(0, count, 16) if any(x >= 0 for x in mapping[start:start+16])]
  chunks:list[tuple[int, int, list[tuple[int, list[int]]]]] = []
  for start, rows in blocks:
    selected_indexes = [x for x in rows if x >= 0]
    base, end = min(selected_indexes)&-8, max(selected_indexes)+1
    if chunks:
      old_base, old_end, old_blocks = chunks[-1]
      merged_base, merged_end = min(old_base, base), max(old_end, end)
      if merged_end-merged_base <= 64:
        chunks[-1] = merged_base, merged_end, [*old_blocks, (start, rows)]
        continue
    if end-base > 512: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"broadcast tile source span is {end-base}", Ops.INDEX)
    chunks.append((base, end, [(start, rows)]))
  constant_bytes = sum(len(chunk)*32*max(32, (end-base+31)&-32)*2 for base,end,chunk in chunks)
  if constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"broadcast selectors require {constant_bytes} bytes", Ops.INDEX)

  packed = RKArg(RKBufferKind.SCRATCH, 1)
  max_align = max((max(32, (end-base+31)&-32) for base,end,_ in chunks), default=32)
  scratch:list[RKScratch] = [RKScratch((((count+31)&-32)+16)*2), RKScratch(max_align*2)]
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  if any(all(x < 0 for x in mapping[start:start+16]) for start in range(0, count, 16)):
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD, expanded, 0.0, 0.0, count),)))
  source = RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot)
  for base,end,blocks in chunks:
    span, align_in = end-base, max(32, (end-base+31)&-32)
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                               RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, base*2), 0.0, span))))
    for start,rows in blocks:
      valid = len(rows)
      out_layout = RKLayout((1,valid), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-valid)))
      steps.append(RKContract(RKTensorRef(RKArg(expanded.kind, expanded.index, start*2), out_layout),
        _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH),
        _cmac_weight_ref(0, valid, align_in, RKBufferKind.CONSTANT, 32), 0,
        _cmac_selection_payload([[index-base] if index >= 0 else [] for index in rows], align_in, 32, 1.0)))

  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  if (scheduled:=_schedule_expr(root, output, count, tuple(scratch))) is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "broadcast stage source is not materializable", stored.op)
  scratch_tuple = scheduled.scratch
  return _native(_finish_program([*steps, scheduled], scratch_tuple))

def lower_multi_broadcast_alu_result(sink:UOp) -> RKLowerResult:
  """Materialize multiple static affine FP16 broadcasts, then schedule one generic DPU expression."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  output_index, stored = store.src[0].src[1], _strip_casts(store.src[1])
  out_aff = _affine(output_index)
  if out_aff is None: return _not_applicable()
  broadcasts = list(dict.fromkeys(u for u in stored.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and
                                  u.src[1].key != output_index.key))
  if len(broadcasts) < 2 or any(u.dtype is not dtypes.half for u in broadcasts): return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  axes, count = tuple(sorted(out_aff[0])), int(store.src[0].src[0].src[0].arg)
  if not axes or any(axis not in ranges for axis in axes) or not 0 < count <= 4096:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"multi-broadcast output has {count} elements", Ops.RANGE)
  substitutions:dict[UOp,UOp] = {}
  remaps:dict[RKArg,RKArg] = {}
  scratch:tuple[RKScratch, ...] = ()
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  for source_index in broadcasts:
    src_count = int(source_index.src[0].src[0].arg)
    mapping = [-1]*count
    for coordinates in product(*(range(ranges[axis]) for axis in axes)):
      point = dict(zip(axes, coordinates))
      dst, src = (_static_scalar(index, point) for index in (output_index,source_index.src[1]))
      if not isinstance(dst, int) or isinstance(dst, bool) or not 0 <= dst < count or mapping[dst] != -1 or \
         not isinstance(src, int) or isinstance(src, bool) or not 0 <= src < src_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-broadcast does not cover one dense output", Ops.INDEX)
      mapping[dst] = src
    expanded = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(count)),)
    packed = _selector_program(expanded, RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot), src_count,
                               [[source] for source in mapping], scratch)
    if packed is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "multi-broadcast selector exceeds plan limits", Ops.INDEX)
    steps.extend(packed.steps)
    scratch = packed.scratch
    substitutions[source_index] = source_index.replace(src=(source_index.src[0], output_index))
    remaps[RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot)] = expanded
  root = _parse_alu(stored.substitute(substitutions), output_index, {})
  def remap(value:_Value) -> _Value:
    if isinstance(value, RKArg): return remaps.get(value, value)
    if isinstance(value, _ALUExpr): return _ALUExpr(value.op, (remap(value.src[0]), remap(value.src[1])))
    if isinstance(value, _MaskExpr): return _MaskExpr((cast(_Expr|RKArg, remap(value.src[0])),))
    if isinstance(value, _LUTExpr): return _LUTExpr(value.lut, (cast(_Expr|RKArg, remap(value.src[0])),))
    return value
  if not isinstance(root, (_ALUExpr,_MaskExpr,_LUTExpr)):
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "multi-broadcast expression is not legal DPU arithmetic", stored.op)
  scheduled = _schedule_expr(cast(_Expr, remap(root)), RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), count, scratch)
  return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "multi-broadcast expression is not materializable", stored.op) if scheduled is None else \
    _native(_finish_program([*steps,scheduled], scheduled.scratch))

def lower_add_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a dense FP16 global sum through aligned DPU block trees and one CMAC tail."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "DPU sum requires an FP16 output surface", store.src[0].op)
  if store.src[0].src[1].op is not Ops.CONST or int(store.src[0].src[1].arg) != 0 or int(store.src[0].src[0].src[0].arg) != 1:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "DPU sum requires one scalar output", store.src[0].op)
  stored, scale, final_relu = _strip_casts(store.src[1]), 1.0, False
  if (relu_source:=_relu_source(stored)) is not None and relu_source.key == reduce.key:
    stored, final_relu = relu_source, True
  if stored.key != reduce.key:
    if stored.op is not Ops.MUL:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "DPU sum only accepts a constant scale epilogue", stored.op)
    const, reduced = (stored.src[0], stored.src[1]) if stored.src[0].op is Ops.CONST else (stored.src[1], stored.src[0])
    if const.op is not Ops.CONST or _strip_casts(reduced).key != reduce.key:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "DPU sum scale is not a direct constant", stored.op)
    scale = float(const.arg)
  value, red = _strip_casts(reduce.src[0]), reduce.src[1]
  pre_relu = (relu_source:=_relu_source(value)) is not None
  if relu_source is not None: value = relu_source
  if final_relu and not pre_relu:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "DPU sum cannot drop an unproven final ReLU", stored.op)
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half or red.op is not Ops.RANGE:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "DPU sum input is not one direct FP16 surface", reduce.op)
  count, src_aff = int(red.src[0].arg), _affine(value.src[1])
  if not 2 <= count <= 65536:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, f"DPU sum extent {count} is outside 2..65536", red.op)
  if src_aff != ({red.arg[0]:1}, 0) or int(value.src[0].src[0].arg) != count:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "DPU sum requires one dense reduction axis", value.op)
  output, input_arg = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  stages:list[RKDPUStage] = []
  scratch:list[RKScratch] = []
  if pre_relu:
    relu_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    stages.append(RKALUStage(Ops.MAX, relu_arg, input_arg, 0.0, count))
    scratch.append(RKScratch(((count+31)&-32)*2))
    input_arg = relu_arg
  if 32 < count <= 512:
    align_in = (count+31)&-32
    packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
    stages.append(RKALUStage(Ops.ADD, packed, input_arg, 0.0, count))
    scratch.append(RKScratch(align_in*2))
    dpu = RKDPUProgram(tuple(stages), tuple(scratch))
    out_layout = RKLayout((1,1), (1,32), (64,2), dtypes.half, padding=((0,0),(0,31)))
    lhs_layout = RKLayout((1,count), (1,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-count)))
    contract = RKContract(RKTensorRef(output, out_layout), RKTensorRef(packed, lhs_layout),
      _cmac_weight_ref(0, 4, align_in, RKBufferKind.CONSTANT, 32), red.arg[0], _cmac_mask_payload(count, align_in, scale=scale))
    return _native(RKProgram((dpu, contract), tuple(scratch)))
  runs:list[tuple[RKArg, int]] = []
  prefix:list[RKContract] = []
  rows, tail = divmod(count, 32)
  if 4 <= rows <= 16 and tail <= 24:
    prefix_out = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch.append(RKScratch(64))
    out_layout = RKLayout((1,rows), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-rows)))
    prefix.append(RKContract(RKTensorRef(prefix_out, out_layout), _dense_half_ref(0, (1,32), RKBufferKind.CONSTANT),
      _cmac_weight_ref(input_arg.index, rows, 32), red.arg[0], struct.pack("<e", 1.0)*32))
    runs.append((prefix_out, rows))
    if tail: runs.append((RKArg(input_arg.kind, input_arg.index, rows*64), tail))
  else:
    blocks:list[tuple[int, int]] = []
    remaining, start = count, 0
    while remaining:
      block = 1 << (remaining.bit_length()-1)
      blocks.append((start, block))
      start, remaining = start+block, remaining-block
    for start,block in blocks:
      source, block_count = RKArg(input_arg.kind, input_arg.index, start*2), block
      while block_count > 8:
        half = block_count//2
        dst = RKArg(RKBufferKind.SCRATCH, len(scratch))
        stages.append(RKALUStage(Ops.ADD, dst, source, RKArg(source.kind, source.index, source.addend+half*2), half))
        scratch.append(RKScratch(((half+7)//8)*16))
        source, block_count = dst, half
      runs.append((source, block_count))
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch.append(RKScratch(64))
  term_offset, mask_values = 0, [0.0]*32
  for source,run_count in runs:
    term_offset = (term_offset+7)&-8
    if term_offset+run_count > 32:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "DPU sum needs more than 32 aligned CMAC terms", red.op)
    stages.append(RKALUStage(Ops.ADD, RKArg(packed.kind, packed.index, term_offset*2), source, 0.0, run_count))
    mask_values[term_offset:term_offset+run_count] = [scale]*run_count
    term_offset += run_count
  mask = b"".join(struct.pack("<e", x) for x in mask_values)
  constants = mask*4
  out_ref = RKTensorRef(output, RKLayout((1,1), (1,32), (64,2), dtypes.half, padding=((0,0),(0,31))))
  contract = RKContract(out_ref, _dense_half_ref(packed.index, (1,32), RKBufferKind.SCRATCH),
                        _cmac_weight_ref(0, 4, 32, RKBufferKind.CONSTANT), red.arg[0], constants)
  return _native(RKProgram((*prefix, RKDPUProgram(tuple(stages)), contract), tuple(scratch)))

def lower_nested_add_reduce_result(sink:UOp) -> RKLowerResult:
  """Compose two affine ADD reductions while preserving their intermediate FP16 rounding boundary."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 2 or any(u.arg[0] is not Ops.ADD or len(u.src) < 2 for u in reductions):
    return _not_applicable()
  store, outer = stores[0], _strip_casts(stores[0].src[1])
  if outer.op is not Ops.REDUCE or store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or \
     store.src[0].dtype is not dtypes.half or store.src[0].src[1].op is not Ops.CONST or \
     int(store.src[0].src[1].arg) != 0 or int(store.src[0].src[0].src[0].arg) != 1:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "nested ADD reduction requires one FP16 scalar output", store.src[0].op)
  chain:list[UOp] = []
  range_groups:list[tuple[UOp, ...]] = []
  value = outer
  while value.op is Ops.REDUCE:
    if value.arg[0] is not Ops.ADD or not value.src[1:] or any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in value.src[1:]):
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "nested ADD reduction is not one reduction chain", value.op)
    chain.append(value)
    range_groups.append(value.src[1:])
    value = _strip_casts(value.src[0])
  if set(chain) != set(reductions) or value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "nested ADD source is not one direct FP16 surface", value.op)
  input_count, index_map = int(value.src[0].src[0].arg), _affine(value.src[1])
  ranges = tuple(u for group in range_groups for u in group)
  axes, extents = tuple(u.arg[0] for u in ranges), tuple(int(u.src[0].arg) for u in ranges)
  if input_count != math.prod(extents) or input_count > RK_MAX_AFFINE_VISITS or index_map is None or set(index_map[0]) != set(axes):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "nested ADD axes do not describe the complete input surface", value.op)
  visited = {index_map[1]+sum(index_map[0][axis]*coord for axis,coord in zip(axes, point))
             for point in product(*(range(extent) for extent in extents))}
  if visited != set(range(input_count)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "nested ADD affine map is not a dense bijection", value.op)
  outer_ranges, inner_ranges = range_groups
  outer_axes, inner_axes = tuple(u.arg[0] for u in outer_ranges), tuple(u.arg[0] for u in inner_ranges)
  range_extents = {u.arg[0]:int(u.src[0].arg) for u in ranges}
  selectors:list[list[int]] = []
  for outer_point in product(*(range(range_extents[axis]) for axis in outer_axes)):
    point = dict(zip(outer_axes, outer_point))
    row = []
    for inner_point in product(*(range(range_extents[axis]) for axis in inner_axes)):
      point.update(zip(inner_axes, inner_point))
      row.append(index_map[1]+sum(index_map[0][axis]*point[axis] for axis in axes))
    selectors.append(row)
  intermediate_count = len(selectors)
  intermediate = RKArg(RKBufferKind.SCRATCH, 0)
  scratch = (RKScratch(_cmac_tiled_output_bytes(intermediate_count)),)
  first = _selector_program(intermediate, RKArg(RKBufferKind.ARG, value.src[0].arg.slot), input_count, selectors, scratch)
  if first is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "inner nested ADD selector exceeds plan limits", outer.op)
  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  second = _selector_program(output, intermediate, intermediate_count, [list(range(intermediate_count))], first.scratch)
  if second is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "outer nested ADD selector exceeds plan limits", outer.op)
  completed = _finish_program([*first.steps,*second.steps], second.scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"nested ADD needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", outer.op)
  return _native(completed)

def lower_scalar_mul_reduce_result(sink:UOp) -> RKLowerResult:
  """Reduce one short dense FP16 surface by gathering each lane into an addressable NPU atom, then multiplying in source order."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] is not Ops.MUL: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     store.src[0].src[1].op is not Ops.CONST or int(store.src[0].src[1].arg) != 0 or int(store.src[0].src[0].src[0].arg) != 1:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "scalar MUL reduction requires one FP16 output", store.src[0].op)
  value = _strip_casts(reduce.src[0])
  if len(reduce.src) != 2 or reduce.src[1].op is not Ops.RANGE or reduce.src[1].src[0].op is not Ops.CONST or \
     value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "scalar MUL reduction requires one direct FP16 surface", reduce.op)
  count, source_map = int(reduce.src[1].src[0].arg), _affine(value.src[1])
  if not 2 <= count <= 32:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"scalar MUL extent {count} is outside 2..32", reduce.op)
  if int(value.src[0].src[0].arg) != count or source_map != ({reduce.src[1].arg[0]:1}, 0):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "scalar MUL source is not one dense reduction axis", value.op)
  packed = RKArg(RKBufferKind.SCRATCH, 0)
  scratch:tuple[RKScratch, ...] = (RKScratch(64),)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = [RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, 32),
    RKALUStage(Ops.ADD, packed, RKArg(RKBufferKind.ARG, value.src[0].arg.slot), 0.0, count)), scratch)]
  operands:list[RKArg] = []
  for index in range(count):
    operand = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(64),)
    gathered = _windowed_cmac_pipeline(operand, packed, [[index]], scratch=scratch, direct_count=32)
    if gathered is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "scalar MUL lane gather exceeds plan limits", reduce.op)
    steps.extend(gathered.steps)
    scratch = gathered.scratch
    operands.append(operand)
  output, accumulator = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), operands[0]
  multiplies:list[RKDPUStage] = []
  for index,operand in enumerate(operands[1:], 1):
    final = index == len(operands)-1
    destination = output if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
    if not final: scratch += (RKScratch(16),)
    multiplies.append(RKALUStage(Ops.MUL, destination, accumulator, operand, 1))
    accumulator = destination
  completed = _finish_program([*steps,RKDPUProgram(tuple(multiplies))], scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"scalar MUL needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(completed)

def lower_affine_mul_reduce_result(sink:UOp) -> RKLowerResult:
  """Materialize each coordinate of a short affine FP16 MUL reduction, then fold full output surfaces on DPU."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] is not Ops.MUL: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if _strip_casts(store.src[1]).key != reduce.key:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "affine MUL does not yet accept an epilogue", store.src[1].op)
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine MUL requires an FP16 output surface", store.src[0].op)
  value = _strip_casts(reduce.src[0])
  if not reduce.src[1:] or any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in reduce.src[1:]) or \
     value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine MUL requires one direct FP16 input surface", reduce.op)
  out_map, source_map = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_map is None or source_map is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine MUL indexes are not static affine maps", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes, out_axes = tuple(u.arg[0] for u in reduce.src[1:]), tuple(sorted(out_map[0]))
  source_axes = set(source_map[0])
  if set(out_axes) & set(red_axes) or source_axes - set(out_axes) - set(red_axes) or \
     any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine MUL axes do not form one output/reduction partition", Ops.RANGE)
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
  if not 1 <= output_count <= 8192 or not 2 <= reduction_count <= 32 or output_count*reduction_count > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine MUL surface is {output_count} outputs by {reduction_count} terms", reduce.op)
  selectors:list[list[list[int]]] = [[[] for _ in range(output_count)] for _ in range(reduction_count)]
  seen:set[int] = set()
  for out_point in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_point))
    output_index = out_map[1]+sum(out_map[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= output_index < output_count or output_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine MUL output is not dense", store.src[0].op)
    seen.add(output_index)
    for term,red_point in enumerate(product(*(range(ranges[axis]) for axis in red_axes))):
      point.update(zip(red_axes, red_point))
      source_index = source_map[1]+sum(source_map[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes))
      if not 0 <= source_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine MUL input index is out of bounds", value.op)
      selectors[term][output_index] = [source_index]
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine MUL output has holes", store.src[0].op)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  scratch:tuple[RKScratch, ...] = ()
  operands:list[RKArg] = []
  source = RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  for rows in selectors:
    operand = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    materialized = _selector_program(operand, source, input_count, rows, scratch)
    if materialized is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "affine MUL operand selector exceeds plan limits", reduce.op)
    steps.extend(materialized.steps)
    scratch = materialized.scratch
    operands.append(operand)
  output, accumulator = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), operands[0]
  multiplies:list[RKDPUStage] = []
  for index,operand in enumerate(operands[1:], 1):
    final = index == len(operands)-1
    destination = output if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
    if not final: scratch += (RKScratch(((output_count+7)//8)*16),)
    multiplies.append(RKALUStage(Ops.MUL, destination, accumulator, operand, output_count))
    accumulator = destination
  completed = _finish_program([*steps,RKDPUProgram(tuple(multiplies))], scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine MUL needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(completed)

def lower_masked_affine_mul_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a small affine MUL scan by materializing selected values and multiplicative identities on NPU engines."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] is not Ops.MUL: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if _strip_casts(store.src[1]).key != reduce.key or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if len(indexes) != 1 or indexes[0].dtype is not dtypes.half or not reduce.src[1:] or \
     any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in reduce.src[1:]):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL requires one FP16 indexed source", reduce.op)
  source_index, out_map = indexes[0], _affine(store.src[0].src[1])
  if out_map is None: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL output is not affine", store.src[0].op)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes, out_axes = tuple(u.arg[0] for u in reduce.src[1:]), tuple(sorted(out_map[0]))
  source_axes = {u.arg[0] for u in source_index.src[1].toposort() if u.op is Ops.RANGE}
  if set(out_axes) & set(red_axes) or source_axes - set(out_axes) - set(red_axes) or \
     any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "masked affine MUL axes do not form one static partition", Ops.RANGE)
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(source_index.src[0].src[0].arg)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
  if output_count > 16:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,
      f"masked affine MUL output {output_count} exceeds the stable one-tile contract", reduce.op)
  if not 1 <= output_count <= 128 or not 2 <= reduction_count <= 32 or output_count*reduction_count > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"masked affine MUL surface is {output_count} outputs by {reduction_count} terms", reduce.op)
  selected:list[list[list[int]]] = [[[] for _ in range(output_count)] for _ in range(reduction_count)]
  identities:list[list[list[int]]] = [[[] for _ in range(output_count)] for _ in range(reduction_count)]
  zero_source = value.substitute({source_index:UOp.const(0, source_index.dtype)})
  seen:set[int] = set()
  for out_point in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_point))
    output_index = out_map[1]+sum(out_map[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= output_index < output_count or output_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL output is not dense", store.src[0].op)
    seen.add(output_index)
    for term,red_point in enumerate(product(*(range(ranges[axis]) for axis in red_axes))):
      point.update(zip(red_axes, red_point))
      active = _static_index_selected(value, source_index, point)
      if active is None:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL predicate is not static", value.op)
      if active:
        source_offset = _static_scalar(source_index.src[1], point)
        if not isinstance(source_offset, int) or isinstance(source_offset, bool) or not 0 <= source_offset < input_count:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL input index is out of bounds", source_index.op)
        selected[term][output_index] = [source_offset]
      else:
        inactive = _static_scalar(zero_source, point)
        if not isinstance(inactive, (int,float)) or float(inactive) != 1.0:
          return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "masked affine MUL inactive value is not one", value.op)
        identities[term][output_index] = [0]
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "masked affine MUL output has holes", store.src[0].op)
  one = RKArg(RKBufferKind.SCRATCH, 0)
  scratch:tuple[RKScratch, ...] = (RKScratch(64),)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = [RKDPUProgram((RKALUStage(Ops.ADD, one, 0.0, 1.0, 32),), scratch)]
  operands:list[RKArg] = []
  source = RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot)
  for value_rows,identity_rows in zip(selected, identities):
    selected_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    value_plan = _selector_program(selected_arg, source, input_count, value_rows, scratch)
    if value_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "masked MUL value selector exceeds limits", reduce.op)
    steps.extend(value_plan.steps)
    scratch = value_plan.scratch
    identity_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    identity_plan = _selector_program(identity_arg, one, 32, identity_rows, scratch)
    if identity_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "masked MUL identity selector exceeds limits", reduce.op)
    steps.extend(identity_plan.steps)
    scratch = identity_plan.scratch
    operand = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(((output_count+7)//8)*16),)
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD, operand, selected_arg, identity_arg, output_count),), scratch))
    operands.append(operand)
  output, accumulator = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), operands[0]
  multiplies:list[RKDPUStage] = []
  for index,operand in enumerate(operands[1:], 1):
    final = index == len(operands)-1
    destination = output if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
    if not final: scratch += (RKScratch(((output_count+7)//8)*16),)
    multiplies.append(RKALUStage(Ops.MUL, destination, accumulator, operand, output_count))
    accumulator = destination
  completed = _finish_program([*steps,RKDPUProgram(tuple(multiplies))], scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"masked affine MUL needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(completed)

def lower_multi_source_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Reduce a static affine selection among FP16 inputs, then combine source-local partials entirely on NPU engines."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] is not Ops.ADD: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if _strip_casts(store.src[1]).key != reduce.key or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  indexes = list(dict.fromkeys(u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM))
  if not 2 <= len(indexes) <= 8 or any(index.dtype is not dtypes.half for index in indexes) or not reduce.src[1:] or \
     any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in reduce.src[1:]):
    return _not_applicable()
  out_map = _affine(store.src[0].src[1])
  if out_map is None: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-source SUM output is not affine", store.src[0].op)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes, out_axes = tuple(u.arg[0] for u in reduce.src[1:]), tuple(sorted(out_map[0]))
  value_axes = {u.arg[0] for u in value.toposort() if u.op is Ops.RANGE}
  if set(out_axes) & set(red_axes) or value_axes - set(out_axes) - set(red_axes) or \
     any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "multi-source SUM axes do not form one static partition", Ops.RANGE)
  output_count, reduction_count = int(store.src[0].src[0].src[0].arg), math.prod(ranges[axis] for axis in red_axes)
  if not 1 <= output_count <= 8192 or output_count*reduction_count > 2*RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"multi-source SUM surface is {output_count}x{reduction_count}", reduce.op)
  selectors:dict[UOp, list[list[int]]] = {index:[[] for _ in range(output_count)] for index in indexes}
  zero_value = value.substitute({index:UOp.const(0, index.dtype) for index in indexes})
  seen:set[int] = set()
  for out_point in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_point))
    output_index = out_map[1]+sum(out_map[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= output_index < output_count or output_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-source SUM output is not dense", store.src[0].op)
    seen.add(output_index)
    for red_point in product(*(range(ranges[axis]) for axis in red_axes)):
      point.update(zip(red_axes, red_point))
      selected = [_static_index_selected(value, index, point) for index in indexes]
      if any(active is None for active in selected) or sum(active is True for active in selected) > 1:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-source SUM selection is not one static branch", value.op)
      if any(selected):
        index = indexes[selected.index(True)]
        source_offset = _static_scalar(index.src[1], point)
        input_count = int(index.src[0].src[0].arg)
        if not isinstance(source_offset, int) or isinstance(source_offset, bool) or not 0 <= source_offset < input_count:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-source SUM input index is out of bounds", index.op)
        selectors[index][output_index].append(source_offset)
      else:
        inactive = _static_scalar(zero_value, point)
        if not isinstance(inactive, (int,float)) or float(inactive) != 0.0:
          return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "multi-source SUM inactive value is not zero", value.op)
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-source SUM output has holes", store.src[0].op)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  scratch:tuple[RKScratch, ...] = ()
  partials:list[RKArg] = []
  for index,rows in selectors.items():
    if not any(rows): continue
    partial = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    plan = _selector_program(partial, RKArg(RKBufferKind.ARG, index.src[0].arg.slot), int(index.src[0].src[0].arg), rows, scratch)
    if plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "multi-source SUM selector exceeds plan limits", reduce.op)
    steps.extend(plan.steps)
    scratch = plan.scratch
    partials.append(partial)
  if not partials: return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "multi-source SUM has no selected input", reduce.op)
  output, accumulator = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), partials[0]
  combines:list[RKDPUStage] = []
  for combine_idx,partial in enumerate(partials[1:], 1):
    final = combine_idx == len(partials)-1
    destination = output if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
    if not final: scratch += (RKScratch(((output_count+7)//8)*16),)
    combines.append(RKALUStage(Ops.ADD, destination, accumulator, partial, output_count))
    accumulator = destination
  if len(partials) == 1: combines.append(RKALUStage(Ops.ADD, output, accumulator, 0.0, output_count))
  completed = _finish_program([*steps,RKDPUProgram(tuple(combines))], scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"multi-source SUM needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(completed)

def _finish_reduction_epilogue(program:RKProgram, stored:UOp, reduce:UOp, output_index:UOp, output:RKArg, reduced:RKArg,
                               output_count:int, out_axes:tuple[int, ...], ranges:dict[int, int]) -> RKLowerResult:
  """Materialize static pointwise operands and execute a reduction epilogue entirely on NPU engines."""
  reduction_nodes = set(reduce.toposort())
  indexes = list(dict.fromkeys(u for u in stored.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and
                               u not in reduction_nodes and u.src[1].key != output_index.key))
  memo:dict[UOp, _Expr|RKArg|float] = {reduce:reduced}
  steps = list(program.steps)
  scratch = program.scratch
  for index in indexes:
    if index.dtype is not dtypes.half:
      return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE, "reduction epilogue input must be FP16", index.op)
    src_count, mapping = int(index.src[0].src[0].arg), [-1]*output_count
    for coordinates in product(*(range(ranges[axis]) for axis in out_axes)):
      point = dict(zip(out_axes, coordinates))
      dst, src = (_static_scalar(node, point) for node in (output_index,index.src[1]))
      if not isinstance(dst, int) or isinstance(dst, bool) or not 0 <= dst < output_count or mapping[dst] != -1 or \
         not isinstance(src, int) or isinstance(src, bool) or not 0 <= src < src_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reduction epilogue is not one static output surface", index.op)
      mapping[dst] = src
    expanded = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    packed = _selector_program(expanded, RKArg(RKBufferKind.ARG, index.src[0].arg.slot), src_count,
                               [[src] for src in mapping], scratch)
    if packed is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "reduction epilogue selector exceeds plan limits", index.op)
    steps.extend(packed.steps)
    scratch = packed.scratch
    memo[index] = expanded
  root = _parse_alu(stored, output_index, memo)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "reduction epilogue is not legal DPU arithmetic", stored.op)
  scheduled = _schedule_expr(root, output, output_count, scratch)
  if scheduled is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "reduction epilogue is not materializable", stored.op)
  completed = _finish_program([*steps, scheduled], scheduled.scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"reduction epilogue needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", stored.op)
  return _native(completed)

def lower_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a small affine FP16 ADD reduction as generated sparse CMAC tiles."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or not reduce.src[1:]: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine CMAC requires an FP16 output", store.src[0].op)
  stored, scale, epilogue = _strip_casts(store.src[1]), 1.0, False
  if stored.key != reduce.key:
    if stored.op is Ops.MUL:
      const, reduced_term = (stored.src[0], stored.src[1]) if stored.src[0].op is Ops.CONST else (stored.src[1], stored.src[0])
      if const.op is Ops.CONST and _strip_casts(reduced_term).key == reduce.key: scale = float(const.arg)
      else: epilogue = True
    else: epilogue = True
  value, prepare = _strip_casts(reduce.src[0]), None
  condition, select_true = None, True
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM and value.dtype is dtypes.half:
    value_index, source = value, RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  elif (conditional:=_conditional_index(value)) is not None and conditional[0].dtype is dtypes.half:
    value_index, condition, select_true = conditional
    source = RKArg(RKBufferKind.ARG, value_index.src[0].arg.slot)
  elif indexes and all(u.dtype is dtypes.half for u in indexes) and len({u.src[1].key for u in indexes}) == 1 and \
       len({int(u.src[0].src[0].arg) for u in indexes}) == 1:
    value_index = indexes[0]
    root = _parse_alu(value, value_index.src[1], {})
    if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
      return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "affine reduction expression is not legal DPU arithmetic", value.op)
    input_count = int(value_index.src[0].src[0].arg)
    source = RKArg(RKBufferKind.SCRATCH, 0)
    prepare = _schedule_expr(root, source, input_count, (RKScratch(((input_count+7)//8)*16),))
    if prepare is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "affine reduction expression is not materializable", value.op)
  else: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC input is not one FP16 pointwise surface", reduce.op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value_index.src[1])
  if out_aff is None or (src_aff is None and condition is None):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value_index.src[0].src[0].arg)
  src_axes = set(src_aff[0]) if src_aff is not None else {u.arg[0] for u in value_index.src[1].toposort() if u.op is Ops.RANGE}
  if len(red_axes) != len(reduce.src)-1 or set(out_axes) & set(red_axes) or \
     src_axes - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine CMAC axes do not form one static output/reduction partition", Ops.RANGE)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
  if not 1 <= output_count <= 8192 or not 2 <= input_count <= 65536 or output_count*reduction_count > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"affine CMAC surface is {output_count}x{input_count}", reduce.op)
  selectors:list[list[int]] = [[] for _ in range(output_count)]
  seen:set[int] = set()
  for out_values in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_values))
    out_index = out_aff[1] + sum(out_aff[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= out_index < output_count or out_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC output is not dense", Ops.INDEX)
    seen.add(out_index)
    for red_values in product(*(range(ranges[axis]) for axis in red_axes)):
      point.update(zip(red_axes, red_values))
      predicate = True if condition is None else _static_scalar(condition, point)
      if predicate is None:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC predicate is not static", cast(UOp, condition).op)
      if bool(predicate) is not select_true: continue
      src_index = _static_scalar(value_index.src[1], point) if src_aff is None else \
        src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes))
      if not isinstance(src_index, int) or isinstance(src_index, bool) or not 0 <= src_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC input index is out of bounds", Ops.INDEX)
      selectors[out_index].append(src_index)
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC output has holes", Ops.INDEX)
  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  initial_scratch = () if prepare is None else prepare.scratch
  reduced = output
  if epilogue:
    reduced = RKArg(RKBufferKind.SCRATCH, len(initial_scratch))
    initial_scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
  program = _sparse_cmac_pipeline(reduced, source, input_count, selectors, scale, initial_scratch) if \
    input_count <= 512 and output_count <= 128 else _windowed_cmac_pipeline(
      reduced, source, selectors, scale, initial_scratch, direct_count=input_count)
  if program is None and struct.unpack("<e", struct.pack("<e", scale))[0] != scale:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, f"two-level affine scale {scale} is not exactly FP16", stored.op)
  if program is None: program = _two_level_selector_program(reduced, source, input_count, selectors, initial_scratch, scale)
  if program is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "affine CMAC output tiles exceed the source-window or constant budget", reduce.op)
  if prepare is not None: program = RKProgram((RKDPUProgram(prepare.stages, program.scratch), *program.steps), program.scratch)
  if epilogue: return _finish_reduction_epilogue(program, stored, reduce, store.src[0].src[1], output, reduced,
                                                  output_count, out_axes, ranges)
  return _native(program)

def lower_contract_result(sink:UOp) -> RKLowerResult:
  """Recognize directly legal M=1, K=32 FP16 CMAC contractions and dense row sums."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "contraction output is not indexed", store.src[0].op)
  if store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"contraction output dtype {store.src[0].dtype.name}", store.src[0].op)
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return _not_applicable()
  if _strip_casts(store.src[1]).key != reduce.key:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, "store epilogue is not a direct reduction", store.src[1].op)
  body, red = _strip_casts(reduce.src[0]), reduce.src[1]
  out_param, out_aff = store.src[0].src[0], _affine(store.src[0].src[1])
  if out_param.op is not Ops.PARAM or out_aff is None or out_aff[1] or len(out_aff[0]) != 1:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "contraction output map is not direct affine", store.src[0].op)
  if body.op is Ops.INDEX:
    inp_aff = _affine(body.src[1])
    out_axis, red_axis, n = next(iter(out_aff[0])), red.arg[0], int(out_param.src[0].arg)
    if body.src[0].op is not Ops.PARAM or body.dtype is not dtypes.half:
      return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE, "CMAC row-sum input must be half", body.op)
    if red.op is not Ops.RANGE or int(red.src[0].arg) != 32:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "direct CMAC row sum requires K=32", red.op)
    if not 4 <= n <= 16:
      return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, f"direct CMAC row sum requires 4<=N<=16, got {n}", red.op)
    if out_aff[0] != {out_axis:1} or inp_aff != ({out_axis:32, red_axis:1}, 0):
      return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "CMAC row sum requires dense N-by-32 input", body.op)
    if int(body.src[0].src[0].arg) != n*32:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "CMAC row-sum input extent does not match N-by-32 surface", body.op)
    ones = struct.pack("<e", 1.0)*32
    return _native(RKContract(_dense_half_ref(out_param.arg.slot, (1,n)), _dense_half_ref(0, (1,32), RKBufferKind.CONSTANT),
                              _cmac_weight_ref(body.src[0].arg.slot, n, 32), red_axis, ones))
  if body.op is not Ops.MUL: return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, "reduction body is not multiply", body.op)
  if red.op is not Ops.RANGE: return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "reduction axis is not a range", red.op)
  if int(red.src[0].arg) != 32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, f"direct CMAC requires K=32, got {int(red.src[0].arg)}", red.op)
  lhs, rhs = (_strip_casts(x) for x in body.src)
  if any(x.op is not Ops.INDEX or x.src[0].op is not Ops.PARAM for x in (lhs, rhs)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "contraction operands are not indexed parameter surfaces", body.op)
  if any(x.dtype is not dtypes.half for x in (lhs, rhs)):
    return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE, "CMAC operands must be half", Ops.INDEX)
  lhs_aff, rhs_aff = _affine(lhs.src[1]), _affine(rhs.src[1])
  if lhs_aff is None or rhs_aff is None or lhs_aff[1] or rhs_aff[1]:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "CMAC operands need affine zero-based surfaces", Ops.INDEX)
  red_axis, out_axes = red.arg[0], tuple(out_aff[0])
  if len(out_axes) != 1 or out_aff[0][out_axes[0]] != 1 or lhs_aff[0] != {red_axis:1}:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "CMAC output or lhs strides need reformatting", Ops.INDEX)
  n_axis = out_axes[0]
  n = next(int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.arg[0] == n_axis)
  if not 4 <= n <= 16: return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, f"direct CMAC requires 4<=N<=16, got {n}", red.op)
  if rhs_aff[0] != {n_axis:32, red_axis:1}:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "rhs is not a packed N-by-K surface", Ops.INDEX)
  if int(out_param.src[0].arg) != n or int(lhs.src[0].src[0].arg) != 32 or int(rhs.src[0].src[0].arg) != n*32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, "CMAC buffer extents do not match M=1,N,K=32", Ops.INDEX)
  return _native(RKContract(_dense_half_ref(out_param.arg.slot, (1,n)), _dense_half_ref(lhs.src[0].arg.slot, (1,32)),
                            _cmac_weight_ref(rhs.src[0].arg.slot, n, 32), red_axis))

def lower_spatial_contract_result(sink:UOp) -> RKLowerResult:
  """Recognize a proven dense NCHW/OIHW stride-one convolution and pack every surface on the NPU."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 4 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     _strip_casts(store.src[1]).key != reduce.key:
    return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[1] is not None or parsed[0].dtype is not dtypes.half or
         parsed[0].src[0].op is not Ops.PARAM for parsed in operands): return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]], operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] != 0 or len(out_aff[0]) not in (3,4): return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[UOp,UOp,int,int,int,int,int,int,int,int,int,int]|None = None
  for feature_parsed,weight_parsed in (parsed_operands, tuple(reversed(parsed_operands))):
    feature, weight = feature_parsed[0], weight_parsed[0]
    feature_aff, weight_aff = _affine(feature.src[1]), _affine(weight.src[1])
    if feature_aff is None or weight_aff is None or feature_aff[1] or weight_aff[1]: continue
    output_roles = ((None,*axes) for axes in permutations(out_axes)) if len(out_axes) == 3 else permutations(out_axes)
    for batch_axis,channel_axis,out_y_axis,out_x_axis in output_roles:
      batch = 1 if batch_axis is None else ranges[batch_axis]
      out_c, out_h, out_w = (ranges[x] for x in (channel_axis,out_y_axis,out_x_axis))
      for in_channel_axis,kernel_y_axis,kernel_x_axis in permutations(red_axes):
        in_c, kernel_h, kernel_w = (ranges[x] for x in (in_channel_axis,kernel_y_axis,kernel_x_axis))
        in_h, in_w = out_h+kernel_h-1, out_w+kernel_w-1
        expected_out = {channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}
        expected_feature = {in_channel_axis:in_h*in_w,kernel_y_axis:in_w,kernel_x_axis:1,out_y_axis:in_w,out_x_axis:1}
        if batch_axis is not None:
          expected_out[batch_axis], expected_feature[batch_axis] = out_c*out_h*out_w, in_c*in_h*in_w
        if out_aff[0] != expected_out or feature_aff[0] != expected_feature: continue
        if weight_aff[0] != {channel_axis:in_c*kernel_h*kernel_w, in_channel_axis:kernel_h*kernel_w,
                            kernel_y_axis:kernel_w, kernel_x_axis:1}: continue
        feature_count, weight_count, output_count = (int(x.src[0].src[0].arg) for x in (feature,weight,store.src[0]))
        if (feature_count,weight_count,output_count) != (batch*in_c*in_h*in_w,out_c*in_c*kernel_h*kernel_w,
                                                        batch*out_c*out_h*out_w): continue
        match = (feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,output_count)
        break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()
  feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,output_count = match
  if in_c not in (4,16) or not 1 <= out_c <= 16 or not (kernel_h > 1 or kernel_w > 1) or max(kernel_h,kernel_w) > 3 or \
     max(in_h,in_w) > 16 or batch > 4:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct spatial convolution is B={batch},IC={in_c},OC={out_c},H={in_h},W={in_w},K={kernel_h}x{kernel_w}", reduce.op)

  align_in, input_c2 = (8,4) if in_c == 4 else (16,8)
  width_alignment = max(1,(16+align_in-1)//align_in)
  input_width_stride, output_width_stride = ((in_w+width_alignment-1)//width_alignment)*width_alignment, (out_h*out_w+3)&-4
  input_batch_count, output_batch_count = in_h*input_width_stride*in_c, 2*output_width_stride*8
  if in_c == 4:
    input_rows = [[b*in_c*in_h*in_w+c*in_h*in_w+y*in_w+x] if x < in_w else []
                  for b in range(batch) for y in range(in_h) for x in range(input_width_stride) for c in range(input_c2)]
  else:
    input_rows = [[b*in_c*in_h*in_w+(c1*input_c2+c2)*in_h*in_w+y*in_w+x] if x < in_w else []
                  for b in range(batch) for c1 in range(in_c//input_c2) for y in range(in_h)
                  for x in range(input_width_stride) for c2 in range(input_c2)]
  weight_rows = [[oc*in_c*kernel_h*kernel_w+c*kernel_h*kernel_w+ky*kernel_w+kx] if c < in_c else []
                 for ky in range(kernel_h) for kx in range(kernel_w) for oc in range(out_c) for c in range(align_in)]
  output_rows = [[b*output_batch_count+(oc//8)*output_width_stride*8+(y*out_w+x)*8+oc%8]
                 for b in range(batch) for oc in range(out_c) for y in range(out_h) for x in range(out_w)]
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),
                                 int(feature.src[0].src[0].arg),input_rows,scratch,
                                 direct_capacity=((int(feature.src[0].src[0].arg)*2+4095)&-4096)//2,
                                 max_window=RK_MAX_CMAC_SELECTOR_WINDOW)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"direct convolution input pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_weight = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(weight_rows)*2),)
  weight_plan = _selector_program(packed_weight,RKArg(RKBufferKind.ARG,weight.src[0].arg.slot),
                                  int(weight.src[0].src[0].arg),weight_rows,scratch,
                                  direct_capacity=((int(weight.src[0].src[0].arg)*2+4095)&-4096)//2)
  if weight_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"direct convolution weight pack exceeds plan limits",Ops.INDEX)
  scratch, steps = weight_plan.scratch, [*steps,*weight_plan.steps]
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(batch*output_batch_count*2),)
  if in_c == 4:
    input_layout = RKLayout((in_h,in_w,in_c),(in_h,input_width_stride,in_c),(input_width_stride*in_c*2,in_c*2,2),dtypes.half,
                            padding=((0,0),(0,input_width_stride-in_w),(0,0)))
  else:
    input_layout = RKLayout((in_c//input_c2,in_h,in_w,input_c2),(in_c//input_c2,in_h,input_width_stride,input_c2),
      (in_h*input_width_stride*input_c2*2,input_width_stride*input_c2*2,input_c2*2,2),dtypes.half,
      padding=((0,0),(0,0),(0,input_width_stride-in_w),(0,0)))
  weight_layout = RKLayout((kernel_h,kernel_w,out_c,in_c),(kernel_h,kernel_w,out_c,align_in),
                           (kernel_w*out_c*align_in*2,out_c*align_in*2,align_in*2,2),dtypes.half,
                           padding=((0,0),(0,0),(0,0),(0,align_in-in_c)))
  output_layout = RKLayout((2,output_width_stride,8),(2,output_width_stride,8),(output_width_stride*16,16,2),dtypes.half)
  for b in range(batch):
    steps.append(RKSpatialConv(RKTensorRef(RKArg(packed_output.kind,packed_output.index,b*output_batch_count*2),output_layout),
      RKTensorRef(RKArg(packed_input.kind,packed_input.index,b*input_batch_count*2),input_layout),
      RKTensorRef(packed_weight,weight_layout),in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,
      input_width_stride,output_width_stride))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
                             batch*output_batch_count,output_rows,scratch,direct_capacity=batch*output_batch_count)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"direct convolution output unpack exceeds plan limits",Ops.INDEX)
  program = _finish_program([*steps,*unpack.steps],unpack.scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"direct convolution needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def lower_tiled_contract_result(sink:UOp) -> RKLowerResult:
  """Pack, execute, and unpack one bounded affine FP16 MxNxK contraction entirely on the NPU."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) < 2 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  stored = _strip_casts(store.src[1])
  epilogue = stored.key != reduce.key
  fused_epilogue = _contract_bias_epilogue(stored, reduce) if epilogue else None
  remaining_epilogue = epilogue and fused_epilogue is None
  body, red_ranges = _strip_casts(reduce.src[0]), reduce.src[1:]
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in red_ranges): return _not_applicable()
  lhs_value, rhs_value = (_strip_casts(x) for x in body.src)
  lhs_parsed, rhs_parsed = _conditional_index(lhs_value), _conditional_index(rhs_value)
  if lhs_parsed is None or rhs_parsed is None: return _not_applicable()
  lhs, rhs = lhs_parsed[0], rhs_parsed[0]
  if lhs.dtype is not dtypes.half or rhs.dtype is not dtypes.half: return _not_applicable()
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None: return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(sorted(out_aff[0])), tuple(red.arg[0] for red in red_ranges)
  operand_axes = {u.arg[0] for value in (lhs_value,rhs_value) for u in value.toposort() if u.op is Ops.RANGE}
  if any(axis not in ranges for axis in red_axes) or operand_axes - set(out_axes) - set(red_axes) or \
     any(axis not in ranges for axis in out_axes): return _not_applicable()
  k = math.prod(ranges[axis] for axis in red_axes)
  lhs_count, rhs_count, output_count = (int(x.src[0].src[0].arg) for x in (lhs,rhs,store.src[0]))
  if not 1 <= k <= 64 or lhs_count > 8192 or rhs_count > 8192 or output_count*k > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"tiled CMAC surfaces are out={output_count},lhs={lhs_count},rhs={rhs_count},K={k}", reduce.op)
  records:list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
  bias_sources:list[int] = []
  if fused_epilogue is not None and {u.arg[0] for u in fused_epilogue[0].src[1].toposort() if u.op is Ops.RANGE} - set(out_axes):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "CMAC bias depends on a reduction or unrelated axis", fused_epilogue[0].op)
  for coordinates in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, coordinates))
    out_offset = out_aff[1] + sum(out_aff[0].get(axis, 0)*point[axis] for axis in out_axes)
    lhs_row, rhs_column = [], []
    for red_coordinates in product(*(range(ranges[axis]) for axis in red_axes)):
      point.update(zip(red_axes, red_coordinates))
      indexes:list[int] = []
      for index,condition,select_true in (lhs_parsed,rhs_parsed):
        predicate = True if condition is None else _static_scalar(condition, point)
        if predicate is None:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "contraction predicate is not static",
                              condition.op if condition is not None else Ops.WHERE)
        if bool(predicate) is not select_true:
          indexes.append(-1)
          continue
        source = _static_scalar(index.src[1], point)
        if not isinstance(source, int) or isinstance(source, bool):
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "contraction index is not static", index.op)
        indexes.append(source)
      if indexes[0] >= lhs_count or indexes[1] >= rhs_count or min(indexes) < -1:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "contraction index is out of bounds", Ops.INDEX)
      lhs_row.append(indexes[0])
      rhs_column.append(indexes[1])
    records.append((out_offset, tuple(lhs_row), tuple(rhs_column)))
    if fused_epilogue is not None:
      bias_source = _static_scalar(fused_epilogue[0].src[1], point)
      if not isinstance(bias_source, int) or isinstance(bias_source, bool):
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "CMAC bias index is not static", fused_epilogue[0].op)
      bias_sources.append(bias_source)
  lhs_rows = list(dict.fromkeys(row for _,row,_ in records))
  rhs_columns = list(dict.fromkeys(column for _,_,column in records))
  m, n = len(lhs_rows), len(rhs_columns)
  align_out, align_in = max(32,(n+31)&-32), max(32,(n+31)&-32,(k+31)&-32)
  if len(records) != output_count or {output for output,_,_ in records} != set(range(output_count)): return _not_applicable()
  if not 1 <= m <= 512 or not 1 <= n <= 128:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"tiled CMAC contraction is M={m},N={n},K={k}", reduce.op)
  lhs_values = sum(source >= 0 for row in lhs_rows for source in row)
  rhs_values = sum(source >= 0 for column in rhs_columns for source in column)
  lhs_base = lhs_rows[0][0] if m == 1 and lhs_rows[0] else -1
  lhs_capacity = ((lhs_count*2+4095)&-4096)//2
  direct_lhs = lhs_base >= 0 and lhs_rows[0] == tuple(range(lhs_base,lhs_base+k)) and lhs_base+align_in <= lhs_capacity
  channel_ids = {column:index for index,column in enumerate(rhs_columns)}
  compact_output = m == 1 and all(out_index == channel_ids[rhs_key] for out_index,_,rhs_key in records)
  selector_floor = (0 if direct_lhs else (lhs_values+15)//16) + (rhs_values+15)//16 + \
    (1 if compact_output else (output_count+15)//16) + \
    (m+(4096//align_in)-1)//(4096//align_in)
  if selector_floor > RK_MAX_PROGRAM_STAGES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"tiled CMAC selector lower bound is {selector_floor} tasks", reduce.op)
  if fused_epilogue is not None and align_out != 32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION, "wide CMAC bias epilogue is not yet legalized", fused_epilogue[0].op)
  channel_bias:list[int]|None = None
  if fused_epilogue is not None:
    bias_count = int(fused_epilogue[0].src[0].src[0].arg)
    channel_bias = [-1]*n
    for (_,_,column),source in zip(records,bias_sources):
      channel = rhs_columns.index(column)
      if not 0 <= source < bias_count or channel_bias[channel] not in (-1, source):
        return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "CMAC bias is not one value per output channel", fused_epilogue[0].op)
      channel_bias[channel] = source
    if -1 in channel_bias:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "CMAC bias does not cover every output channel", fused_epilogue[0].op)
  a_selector = [entry for row in lhs_rows for entry in (([[source] if source >= 0 else [] for source in row]) +
                [[] for _ in range(align_in-k)])]
  b_selector:list[list[int]] = []
  for out_block in range(align_out//16):
    for in_block in range(align_in//32):
      for out_lane in range(16):
        for in_lane in range(32):
          out_channel, reduction_index = out_block*16+out_lane, in_block*32+in_lane
          source = rhs_columns[out_channel][reduction_index] if out_channel < n and reduction_index < k else -1
          b_selector.append([source] if source >= 0 else [])
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  scratch:tuple[RKScratch, ...] = ()
  if direct_lhs:
    # CMAC may read the allocator's page-rounded tail, but padded K lanes have zero weights and cannot affect the result.
    a_arg = RKArg(RKBufferKind.ARG,lhs.src[0].arg.slot,lhs_base*2)
  else:
    a_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(len(a_selector))),)
    packed_a = _selector_program(a_arg, RKArg(RKBufferKind.ARG, lhs.src[0].arg.slot), lhs_count, a_selector, scratch)
    if packed_a is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "tiled CMAC lhs selector exceeds plan limits", reduce.op)
    steps.extend(packed_a.steps)
    scratch = packed_a.scratch
  b_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(_cmac_tiled_output_bytes(len(b_selector))),)
  # Rockchip GEM allocations are page-rounded. Zero-weight selector lanes may read that physical tail without changing semantics.
  rhs_capacity = ((rhs_count*2+4095)&-4096)//2
  packed_b = _selector_program(b_arg, RKArg(RKBufferKind.ARG, rhs.src[0].arg.slot), rhs_count, b_selector, scratch,
                               rhs_capacity, RK_MAX_CMAC_SELECTOR_WINDOW)
  if packed_b is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "tiled CMAC rhs selector exceeds plan limits", reduce.op)
  steps.extend(packed_b.steps)
  scratch = packed_b.scratch
  contract_epilogue:RKEpilogue|None = None
  if fused_epilogue is not None:
    assert channel_bias is not None
    bias_half = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(32)),)
    bias_rows:list[list[int]] = [[] for _ in range(32)]
    for channel,source in enumerate(channel_bias): bias_rows[(channel//4)*8+channel%4] = [source]
    bias_plan = _selector_program(bias_half, RKArg(RKBufferKind.ARG, fused_epilogue[0].src[0].arg.slot),
                                  int(fused_epilogue[0].src[0].src[0].arg), bias_rows, scratch)
    if bias_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "CMAC bias selector exceeds plan limits", fused_epilogue[0].op)
    steps.extend(bias_plan.steps)
    scratch = bias_plan.scratch
    bias_float = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(32*4),)
    steps.append(RKDPUProgram(tuple(RKALUStage(Ops.ADD, RKArg(bias_float.kind, bias_float.index, start*4),
      RKArg(bias_half.kind, bias_half.index, start*4), 0.0, 4, dtypes.float) for start in range(0,32,4)), scratch))
    contract_epilogue = RKEpilogue(RKTensorRef(bias_float, RKLayout((n,), (32,), (4,), dtypes.float, padding=((0,32-n),))),
                                   fused_epilogue[1])
  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  reduced = output
  if remaining_epilogue:
    reduced = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
  cmac_out = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(m*align_out*(2 if compact_output else 4)),)
  rhs_layout = RKLayout((n,k), (align_out,align_in), (align_in*2,2), dtypes.half,
                        padding=((0,align_out-n),(0,align_in-k)), kind=RKLayoutKind.CMAC_WEIGHT)
  for row_start in range(0, m, 4096//align_in):
    tile_m = min(4096//align_in, m-row_start)
    lhs_layout = RKLayout((tile_m,k), (tile_m,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-k)))
    out_physical = align_out if compact_output else align_out*2
    out_layout = RKLayout((tile_m,n), (tile_m,out_physical), (out_physical*2,2), dtypes.half,
                          padding=((0,0),(0,out_physical-n)))
    steps.append(RKContract(RKTensorRef(RKArg(cmac_out.kind, cmac_out.index, row_start*out_physical*2), out_layout),
      RKTensorRef(RKArg(a_arg.kind, a_arg.index, row_start*align_in*2), lhs_layout), RKTensorRef(b_arg, rhs_layout), red_axes[0],
      epilogue=contract_epilogue, compact_output=compact_output))
  if compact_output:
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD,reduced,cmac_out,0.0,output_count),),scratch))
  else:
    unpack:list[list[int]] = [[] for _ in range(output_count)]
    for out_index,row,rhs_key in records:
      channel = channel_ids[rhs_key]
      unpack[out_index] = [lhs_rows.index(row)*align_out*2+(channel//16)*32+channel%16]
    dense = _selector_program(reduced, cmac_out, m*align_out*2, unpack, scratch)
    if dense is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "tiled CMAC output selector exceeds plan limits", reduce.op)
    steps.extend(dense.steps)
    scratch = dense.scratch
  stage_count = sum(len(step.stages) if isinstance(step, RKDPUProgram) else 1 for step in steps)
  constant_bytes = sum(map(len, {step.constants for step in steps if isinstance(step, RKContract)}))
  if stage_count > RK_MAX_PROGRAM_STAGES or constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"tiled CMAC plan needs {stage_count} stages and {constant_bytes} constant bytes", reduce.op)
  program = _finish_program(steps, scratch)
  if remaining_epilogue: return _finish_reduction_epilogue(program, stored, reduce, store.src[0].src[1], output, reduced,
                                                            output_count, out_axes, ranges)
  return _native(program)

def lower_contract(sink:UOp) -> RKContract|None:
  """Compatibility helper for compiler probes; production lowering consumes `lower_contract_result`."""
  return cast(RKContract|None, lower_contract_result(sink).plan)

def _pool_hw_shape(extent:int) -> tuple[int, int]|None:
  """Return a proven PPU global-pool surface: both dimensions fit its four-bit fields."""
  return min(((height, extent//height) for height in range(2, min(16, extent)+1)
              if extent % height == 0 and 2 <= extent//height <= 16), key=lambda shape:abs(shape[0]-shape[1]), default=None)

def lower_reduce_result(sink:UOp) -> RKLowerResult:
  """Recognize global FP16 MAX over the spatial dimensions of a dense HWC8 surface."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) != 2: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "PPU output is not an indexed parameter surface", store.src[0].op)
  if store.src[0].dtype is not dtypes.half or reduce.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "PPU reduction requires FP16 input and output", reduce.op)
  value, red = _strip_casts(reduce.src[0]), reduce.src[1]
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half or red.op is not Ops.RANGE:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "PPU reduction input is not a direct FP16 surface", reduce.op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None or out_aff[1] or src_aff[1] or len(out_aff[0]) != 1:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "PPU reduction needs zero-based affine HWC8 surfaces", Ops.INDEX)
  out_axis, red_axis = next(iter(out_aff[0])), red.arg[0]
  channels, extent = int(store.src[0].src[0].src[0].arg), int(red.src[0].arg)
  hw_shape = _pool_hw_shape(extent)
  if channels != 8 or out_aff[0] != {out_axis:1} or src_aff[0] != {red_axis:8, out_axis:1}:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "PPU global MAX requires dense HWC8 indexing", Ops.INDEX)
  if hw_shape is None:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, f"PPU global MAX spatial extent {extent} needs tiling or reformat", reduce.op)
  if int(value.src[0].src[0].arg) != extent*channels:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "PPU input buffer extent does not match HWC8 surface", Ops.INDEX)
  height, width = hw_shape
  out = RKTensorRef(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot),
                    RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half))
  src = RKTensorRef(RKArg(RKBufferKind.ARG, value.src[0].arg.slot),
                    RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half))
  return _native(RKReduce(out, src, Ops.MAX, red_axis))

def lower_global_max_result(sink:UOp) -> RKLowerResult:
  """Lower one dense FP16 global MAX through a padded pairwise DPU tree."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) != 2: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "DPU global MAX requires an FP16 output", store.src[0].op)
  if store.src[0].src[1].op is not Ops.CONST or int(store.src[0].src[1].arg) != 0 or int(store.src[0].src[0].src[0].arg) != 1:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "DPU global MAX requires one scalar output", store.src[0].op)

  stored, output_scale = _strip_casts(store.src[1]), 1.0
  if stored.key != reduce.key:
    if stored.op is not Ops.MUL: return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "DPU global MAX epilogue is not a scale", stored.op)
    const, reduced = (stored.src[0], stored.src[1]) if stored.src[0].op is Ops.CONST else (stored.src[1], stored.src[0])
    if const.op is not Ops.CONST or _strip_casts(reduced).key != reduce.key:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "DPU global MAX scale is not direct", stored.op)
    output_scale = float(const.arg)

  value, red, input_scale = _strip_casts(reduce.src[0]), reduce.src[1], 1.0
  if value.op is Ops.MUL:
    const, candidate = (value.src[0], value.src[1]) if value.src[0].op is Ops.CONST else (value.src[1], value.src[0])
    if const.op is Ops.CONST:
      value, input_scale = _strip_casts(candidate), float(const.arg)
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half or red.op is not Ops.RANGE:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "DPU global MAX input is not one FP16 surface", reduce.op)
  count, src_aff = int(red.src[0].arg), _affine(value.src[1])
  if not 2 <= count <= 65536:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, f"DPU global MAX extent {count} is outside 2..65536", red.op)
  if src_aff != ({red.arg[0]:1}, 0) or int(value.src[0].src[0].arg) != count:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "DPU global MAX requires one dense reduction axis", value.op)

  output, source = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  extent, stages, scratch = max(8, 1 << (count-1).bit_length()), [], []
  padded = RKArg(RKBufferKind.SCRATCH, 0)
  scratch.append(RKScratch(extent*2))
  stages.append(RKALUStage(Ops.ADD, padded, 0.0, -math.inf, extent))
  stages.append(RKALUStage(Ops.ADD if input_scale == 1.0 else Ops.MUL, padded, source, 0.0 if input_scale == 1.0 else input_scale, count))
  source, active = padded, extent
  while active > 8:
    half = active//2
    dst = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch.append(RKScratch(((half+7)//8)*16))
    # Stop before either source address would fall inside one 16-byte atom.
    stages.append(RKALUStage(Ops.MAX, dst, RKArg(source.kind, source.index, source.addend+half*2), source, half))
    source, active = dst, half

  # DPU elementwise addresses are atom-aligned, so reduce the final eight lanes by placing them in PPU channel 0 over an 8-pixel surface.
  packed = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch.append(RKScratch(64))
  stages.extend((RKALUStage(Ops.ADD, packed, 0.0, 0.0, 32), RKALUStage(Ops.ADD, packed, source, 0.0, 8)))
  hwc = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch.append(RKScratch(160))  # final 16-output CMAC tile writes one padded 32-lane surface
  rows = [[index//8] if index%8 == 0 else [] for index in range(64)]
  contracts:list[RKContract] = []
  for start in range(0, 64, 16):
    out_layout = RKLayout((1,16), (1,32), (64,2), dtypes.half, padding=((0,0),(0,16)))
    contracts.append(RKContract(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
      _dense_half_ref(packed.index, (1,32), RKBufferKind.SCRATCH), _cmac_weight_ref(0, 16, 32, RKBufferKind.CONSTANT, 32),
      red.arg[0], _cmac_selection_payload(rows[start:start+16], 32, 32, 1.0)))
  pooled = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch.append(RKScratch(16))
  scratch_tuple = tuple(scratch)
  reduce_plan = RKReduce(RKTensorRef(pooled, RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half)),
    RKTensorRef(hwc, RKLayout((2,4,8), (2,4,8), (64,16,2), dtypes.half)), Ops.MAX, red.arg[0])
  final = RKDPUProgram((RKALUStage(Ops.ADD if output_scale == 1.0 else Ops.MUL, output, pooled,
                                   0.0 if output_scale == 1.0 else output_scale, 1),), scratch_tuple)
  return _native(RKProgram((RKDPUProgram(tuple(stages), scratch_tuple), *contracts, reduce_plan, final), scratch_tuple))

def _scalar_affine_max_program(output:RKArg, source:RKArg, selectors:list[list[int]], pool_extent:int,
                               pool_shape:tuple[int, int], reduce_axis:int) -> RKProgram|None:
  """Reduce source-local scalar windows, then gather their aligned PPU atoms into dense output."""
  specs:list[tuple[int, int, int, int|None, list[int]]] = []
  for row in selectors:
    selected = [index for index in row if index >= 0]
    if not selected: return None
    base, end = min(selected)&-8, max(selected)+1
    span, align_in = end-base, max(32, (end-base+31)&-32)
    sentinel = align_in if any(index < 0 for index in row) else None
    if sentinel is not None: align_in += 32
    if align_in > 512: return None
    specs.append((base, span, align_in, sentinel, [index-base if index >= 0 else -1 for index in row]))
  surface_count, hwc_elements = pool_extent*8, ((pool_extent*8-1)//16)*16+32
  packed, hwc, atoms = (RKArg(RKBufferKind.SCRATCH, index) for index in range(3))
  scratch:tuple[RKScratch, ...] = (RKScratch(max(spec[2] for spec in specs)*2), RKScratch(hwc_elements*2), RKScratch(len(selectors)*16))
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  payloads:set[bytes] = set()
  height, width = pool_shape
  for output_index,(base,span,align_in,sentinel,row) in enumerate(specs):
    prepare = [RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
               RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, source.addend+base*2), 0.0, span)]
    if sentinel is not None:
      prepare.append(RKALUStage(Ops.ADD, RKArg(packed.kind, packed.index, sentinel*2), 0.0, -math.inf, 1))
    steps.append(RKDPUProgram(tuple(prepare), scratch))
    flat = [sentinel if row[min(spatial,len(row)-1)] < 0 else row[min(spatial,len(row)-1)]
            for spatial in range(pool_extent) for _ in range(8)]
    for start in range(0, surface_count, 16):
      count = min(16, surface_count-start)
      payload = _cmac_selection_payload([[cast(int,index)] for index in flat[start:start+count]], align_in, 32, 1.0)
      payloads.add(payload)
      out_layout = RKLayout((1,count), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-count)))
      steps.append(RKContract(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
        _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH),
        _cmac_weight_ref(0, count, align_in, RKBufferKind.CONSTANT, 32), reduce_axis, payload))
    out = RKTensorRef(RKArg(atoms.kind, atoms.index, output_index*16), RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half))
    src = RKTensorRef(hwc, RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half))
    steps.append(RKReduce(out, src, Ops.MAX, reduce_axis))
  gather = _sparse_cmac_pipeline(output, atoms, len(selectors)*8, [[index*8] for index in range(len(selectors))], scratch=scratch)
  steps.extend(gather.steps)
  scratch = gather.scratch
  payloads.update(step.constants for step in gather.steps if isinstance(step, RKContract))
  stage_count = sum(len(step.stages) if isinstance(step, RKDPUProgram) else 1 for step in steps)
  if stage_count > RK_MAX_PROGRAM_STAGES or sum(map(len,payloads)) > RK_MAX_CONSTANT_BYTES: return None
  return _finish_program(steps, scratch)

def lower_affine_max_result(sink:UOp) -> RKLowerResult:
  """Lower a static affine FP16 MAX by reformatting eight outputs into PPU channels per batch."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) < 2 or _strip_casts(store.src[1]).key != reduce.key: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine PPU MAX requires an FP16 output", store.src[0].op)
  value = _strip_casts(reduce.src[0])
  indexes = tuple(u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.dtype is dtypes.half)
  if len(indexes) != 1:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX input is not one FP16 surface", reduce.op)
  value_index = indexes[0]
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value_index.src[1])
  if src_aff is None and value_index.src[1].op is Ops.WHERE:
    affine_branches = tuple(x for branch in value_index.src[1].src[1:] if branch.arg is not Invalid and (x:=_affine(branch)) is not None)
    if len(affine_branches) == 1: src_aff = affine_branches[0]
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value_index.src[0].src[0].arg)
  if len(red_axes) != len(reduce.src)-1 or not out_axes and len(red_axes) == 1 or set(out_axes) & set(red_axes) or \
     set(src_aff[0]) - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine PPU MAX axes do not form a static output/reduction partition", Ops.RANGE)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
  if not 1 <= output_count <= 4096 or input_count < 2 or output_count*reduction_count > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine PPU MAX surface is {output_count}x{input_count} with {output_count*reduction_count} visits", reduce.op)
  pool_extent = next((extent for extent in range(max(4, reduction_count), 257) if _pool_hw_shape(extent) is not None), None)
  if pool_extent is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"affine PPU MAX reduction extent is {reduction_count}", reduce.op)
  pool_shape = _pool_hw_shape(pool_extent)
  assert pool_shape is not None

  selectors:list[list[int]] = [[] for _ in range(output_count)]
  seen:set[int] = set()
  for out_values in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_values))
    out_index = out_aff[1] + sum(out_aff[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= out_index < output_count or out_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX output is not dense", Ops.INDEX)
    seen.add(out_index)
    for red_values in product(*(range(ranges[axis]) for axis in red_axes)):
      point.update(zip(red_axes, red_values))
      selected = True if value.key == value_index.key else _static_index_selected(value, value_index, point)
      if selected is None:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX predicate is not static", Ops.WHERE)
      src_index = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes)) if selected else -1
      if selected and not 0 <= src_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX input index is out of bounds", Ops.INDEX)
      selectors[out_index].append(src_index)
  if seen != set(range(output_count)) or any(len(row) != reduction_count for row in selectors):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX output has holes", Ops.INDEX)

  surface_count, whole_surface = pool_extent*8, input_count <= 512 and output_count <= 128
  raw_groups = [(start, selectors[start:start+8]) for start in range(0, output_count, 8)]
  if any(max(32, (max(index for row in rows for index in row if index >= 0)+1-
                      (min(index for row in rows for index in row if index >= 0)&-8)+31)&-32) +
         (32 if any(index < 0 for row in rows for index in row) else 0) > 512 for _,rows in raw_groups):
    scalar = _scalar_affine_max_program(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot),
      RKArg(RKBufferKind.ARG, value_index.src[0].arg.slot), selectors, pool_extent, pool_shape, red_axes[0])
    if scalar is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "affine PPU MAX scalar windows exceed plan limits", reduce.op)
    return _native(scalar)
  windows:list[tuple[int, int, int, int|None, list[tuple[int, list[list[int]]]]]] = []
  next_group = 0
  while next_group < len(raw_groups):
    window_groups:list[tuple[int, list[list[int]]]] = []
    for candidate in raw_groups[next_group:]:
      trial_rows = [row for _,rows in (*window_groups,candidate) for row in rows]
      selected_indices = [index for row in trial_rows for index in row if index >= 0]
      base, end = (0,input_count) if whole_surface else ((min(selected_indices)&-8,max(selected_indices)+1) if selected_indices else (0,0))
      masked = any(index < 0 for row in trial_rows for index in row)
      align_in = max(32, (end-base+31)&-32) + (32 if masked else 0)
      if align_in > 512 or not whole_surface and masked and align_in > RK_MAX_AFFINE_WINDOW and window_groups: break
      window_groups.append(candidate)
      if whole_surface: continue
    if not window_groups:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"affine PPU MAX window exceeds the {RK_MAX_AFFINE_WINDOW}-lane cost target", reduce.op)
    trial_rows = [row for _,rows in window_groups for row in rows]
    selected_indices = [index for row in trial_rows for index in row if index >= 0]
    base, end = (0,input_count) if whole_surface else ((min(selected_indices)&-8,max(selected_indices)+1) if selected_indices else (0,0))
    span, align_in = end-base, max(32, (end-base+31)&-32)
    sentinel = align_in if any(index < 0 for row in trial_rows for index in row) else None
    if sentinel is not None: align_in += 32
    windows.append((base, span, align_in, sentinel, window_groups))
    next_group += len(window_groups)

  packed, hwc = RKArg(RKBufferKind.SCRATCH, 0), RKArg(RKBufferKind.SCRATCH, 1)
  hwc_elements = ((surface_count-1)//16)*16+32
  scratch = (RKScratch(max(window[2] for window in windows)*2), RKScratch(hwc_elements*2))
  source = RKArg(RKBufferKind.ARG, value_index.src[0].arg.slot)
  steps:list[RKDPUProgram|RKContract|RKSpatialConv|RKReduce] = []
  payloads:set[bytes] = set()
  height, width = pool_shape
  for base,span,align_in,sentinel,window_groups in windows:
    prepare = [RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in)]
    if span: prepare.append(RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, source.addend+base*2), 0.0, span))
    if sentinel is not None:
      prepare.append(RKALUStage(Ops.ADD, RKArg(packed.kind, packed.index, sentinel*2), 0.0, -math.inf, 1))
    steps.append(RKDPUProgram(tuple(prepare), scratch))
    for group_start,group_rows in window_groups:
      group_indices = [index for row in group_rows for index in row if index >= 0]
      group_base = base if sentinel is not None else min(group_indices)&-8
      contract_align = align_in if sentinel is not None else max(32, (max(group_indices)+1-group_base+31)&-32)
      local_rows = [[index-group_base if index >= 0 else -1 for index in row] for row in group_rows]
      channels = min(8, output_count-group_start)
      rows = [[local_rows[min(channel, channels-1)][min(spatial, reduction_count-1)] for channel in range(8)]
              for spatial in range(pool_extent)]
      flat_rows = [cast(int, sentinel) if rows[spatial][channel] < 0 else rows[spatial][channel]
                   for spatial in range(pool_extent) for channel in range(8)]
      for start in range(0, surface_count, 16):
        out_layout = RKLayout((1,min(16, surface_count-start)), (1,32), (64,2), dtypes.half,
                              padding=((0,0),(0,32-min(16, surface_count-start))))
        payload = _cmac_selection_payload([[index] for index in flat_rows[start:start+16]], contract_align, 32, 1.0)
        payloads.add(payload)
        lhs = _dense_half_ref(packed.index, (1,contract_align), RKBufferKind.SCRATCH)
        lhs = RKTensorRef(RKArg(packed.kind, packed.index, (group_base-base)*2), lhs.layout)
        steps.append(RKContract(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
          lhs, _cmac_weight_ref(0, min(16, surface_count-start), contract_align, RKBufferKind.CONSTANT, 32), reduce.arg[0], payload))
      out = RKTensorRef(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot, group_start*2),
                        RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half))
      src = RKTensorRef(hwc, RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half))
      steps.append(RKReduce(out, src, Ops.MAX, red_axes[0]))
  stage_count = sum(len(step.stages) if isinstance(step, RKDPUProgram) else 1 for step in steps)
  if stage_count > RK_MAX_PROGRAM_STAGES or sum(map(len, payloads)) > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine PPU MAX needs {stage_count} stages and {sum(map(len, payloads))} constant bytes", reduce.op)
  return _native(RKProgram(tuple(steps), scratch))

@dataclass(frozen=True)
class RKLowerer:
  name: str
  applies: Callable[[tuple[UOp, ...]], bool]
  lower: Callable[[UOp], RKLowerResult]

def _has_reduction(nodes:tuple[UOp, ...], op:Ops|None=None) -> bool:
  reductions = tuple(u for u in nodes if u.op is Ops.REDUCE)
  return bool(reductions) and (op is None or all(u.arg[0] is op for u in reductions))

_LOWERERS = (
  RKLowerer("dpu", lambda nodes:not _has_reduction(nodes), lower_dpu_result),
  RKLowerer("multi_broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_multi_broadcast_alu_result),
  RKLowerer("broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_broadcast_alu_result),
  RKLowerer("reformat", lambda nodes:not _has_reduction(nodes), lower_reformat_result),
  RKLowerer("nested_sum", lambda nodes:_has_reduction(nodes, Ops.ADD) and sum(u.op is Ops.REDUCE for u in nodes) > 1,
            lower_nested_add_reduce_result),
  RKLowerer("scalar_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_scalar_mul_reduce_result),
  RKLowerer("masked_affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_masked_affine_mul_reduce_result),
  RKLowerer("affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_affine_mul_reduce_result),
  RKLowerer("sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_add_reduce_result),
  RKLowerer("multi_source_sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_multi_source_affine_reduce_result),
  RKLowerer("affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_affine_reduce_result),
  RKLowerer("ppu_reduce", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_reduce_result),
  RKLowerer("affine_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_affine_max_result),
  RKLowerer("global_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_global_max_result),
  RKLowerer("spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_spatial_contract_result),
  RKLowerer("tiled_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_tiled_contract_result),
  RKLowerer("contract", lambda nodes:_has_reduction(nodes) and not _has_reduction(nodes, Ops.MAX), lower_contract_result),
)

_REJECT_PRIORITY = {
  RKRejectKind.NUMERICAL_CONTRACT:90, RKRejectKind.LUT_DOMAIN_UNPROVEN:85, RKRejectKind.PLAN_STAGE_LIMIT:80,
  RKRejectKind.UNSUPPORTED_INPUT_DTYPE:70, RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE:70,
  RKRejectKind.UNALIGNED_ROW:60, RKRejectKind.REQUIRES_REFORMAT:60, RKRejectKind.UNSUPPORTED_DYNAMIC_PACK:60,
  RKRejectKind.UNSUPPORTED_LAYOUT:50, RKRejectKind.UNSUPPORTED_BROADCAST:50,
  RKRejectKind.UNSUPPORTED_REDUCTION:40, RKRejectKind.UNSUPPORTED_CONTRACTION:40, RKRejectKind.UNSUPPORTED_ALU:30,
}

def lower_native(sink:UOp) -> RKLowerResult:
  nodes, rejects = tuple(sink.toposort()), []
  for lowerer in _LOWERERS:
    result = lowerer.lower(sink) if lowerer.applies(nodes) else _not_applicable()
    if result.kind is RKLowerKind.NATIVE: return result
    if result.kind is RKLowerKind.UNSUPPORTED:
      assert result.reject is not None
      rejects.append(result.reject)
  if not rejects: rejects.append(RKReject(RKRejectKind.UNSUPPORTED_ALU, "no Rockchip lowerer applies", sink.op))
  reject = max(enumerate(rejects), key=lambda item:(_REJECT_PRIORITY[item[1].kind], item[0]))[1]
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(reject.kind, reject.detail, reject.node_op, rk_fingerprint(sink)))

from tinygrad.renderer.rockchip.emit import (emit_dpu as emit_dpu, emit_contract as emit_contract, emit_spatial_conv as emit_spatial_conv,
  emit_program as emit_program, emit_reduce as emit_reduce, emit_reformat as emit_reformat)

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int, dtypes.float}
  def native_program(self, ast:UOp) -> UOp|None:
    info = ProgramInfo.from_sink(ast, self.target)
    params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM and u.arg.slot >= 0), key=lambda u:u.arg.slot))
    result = lower_native(ast)
    if result.reject is not None:
      reject = result.reject
      record_telemetry("reject", lane="REJECT", program=info.name, reject_kind=reject.kind.value, detail=reject.detail,
        node_op=reject.node_op.name if reject.node_op is not None else None, fingerprint=reject.fingerprint,
        fingerprint_digest=dict(reject.fingerprint)["graph"],
        signature=[{"slot": u.arg.slot, "dtype": u.dtype.name,
                    "shape": [x if isinstance(x, int) else str(x) for x in u.shape]} for u in params])
      fallback = os.getenv("ROCKCHIP_FALLBACK", "0").upper()
      if fallback == "PYTHON":
        from tinygrad.runtime.rockchip_fallback import build_rkpy_program
        return build_rkpy_program(ast, self.target)
      if fallback not in ("", "0"): raise RuntimeError(f"invalid ROCKCHIP_FALLBACK={fallback!r}")
      raise RuntimeError(f"RKPLAN_REJECT:{reject.kind.value}:{reject.detail}")
    if isinstance(result.plan, RKDPUProgram): image = emit_dpu(result.plan)
    elif isinstance(result.plan, RKContract): image = emit_contract(result.plan)
    elif isinstance(result.plan, RKSpatialConv): image = emit_spatial_conv(result.plan)
    elif isinstance(result.plan, RKReduce): image = emit_reduce(result.plan)
    elif isinstance(result.plan, RKReformat): image = emit_reformat(result.plan)
    elif isinstance(result.plan, RKProgram): image = emit_program(result.plan)
    else: raise RuntimeError("invalid Rockchip lowering result")
    linear = UOp(Ops.LINEAR, src=tuple(u for u in params if u.addrspace is not AddrSpace.ALU))
    return UOp(Ops.PROGRAM, src=(ast, linear, UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=encode_image(image))),
               arg=info)
