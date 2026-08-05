from __future__ import annotations
import math, struct
from itertools import product
from typing import cast

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.analysis import (strip_casts as _strip_casts, static_scalar as _static_scalar,
  conditional_index as _conditional_index, static_index_selected as _static_index_selected,
  relu_source as _relu_source, fp16_exact as _fp16_exact)
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.expr import _ALUExpr, _MaskExpr, _LUTExpr, _Expr, _parse_alu
from tinygrad.renderer.rockchip.ir import (RKBufferKind, RKArg, RKALUStage, RKDPUStage, RKScratch, RKDPUProgram, RKLayout, RKTensorRef,
  RKCMACTask, RKConvTask, RKReduce, RKProgram, RKRejectKind, RKLowerResult)
from tinygrad.renderer.rockchip.limits import (RK_MAX_CONSTANT_BYTES, RK_MAX_AFFINE_VISITS, RK_MAX_STATIC_MASK_VISITS,
  RK_MAX_PROGRAM_STAGES, RK_MAX_PREFIX_WINDOW, RK_MAX_PREFIX_VISITS)
from tinygrad.renderer.rockchip.lower import native as _native, not_applicable as _not_applicable, unsupported as _unsupported
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr
from tinygrad.renderer.rockchip.selector import (_cmac_tiled_output_bytes, _dense_half_ref, _cmac_weight_ref,
  _cmac_mask_payload, _cmac_selection_payload, _sparse_cmac_pipeline, _windowed_cmac_pipeline, _two_level_selector_program,
  _selector_program, _finish_program)

def lower_add_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a dense FP16 global sum through aligned DPU block trees and one CMAC tail."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype not in (dtypes.half,dtypes.float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "DPU sum requires an FP16 or FP32 output surface", store.src[0].op)
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
  fp32_out = store.src[0].dtype is dtypes.float
  def output_ref() -> RKTensorRef:
    return RKTensorRef(output, RKLayout((1,1), (1,64 if fp32_out else 32), (256,4) if fp32_out else (64,2),
      store.src[0].dtype, padding=((0,0),(0,63 if fp32_out else 31))))
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
    lhs_layout = RKLayout((1,count), (1,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-count)))
    contract = RKCMACTask(output_ref(), RKTensorRef(packed, lhs_layout),
      _cmac_weight_ref(0, 4, align_in, RKBufferKind.CONSTANT, 32), red.arg[0], _cmac_mask_payload(count, align_in, scale=scale))
    return _native(RKProgram((dpu, contract), tuple(scratch)))
  runs:list[tuple[RKArg, int]] = []
  prefix:list[RKCMACTask] = []
  rows, tail = divmod(count, 32)
  if 4 <= rows <= 16 and tail <= 24:
    prefix_out = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch.append(RKScratch(64))
    out_layout = RKLayout((1,rows), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-rows)))
    prefix.append(RKCMACTask(RKTensorRef(prefix_out, out_layout), _dense_half_ref(0, (1,32), RKBufferKind.CONSTANT),
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
  contract = RKCMACTask(output_ref(), _dense_half_ref(packed.index, (1,32), RKBufferKind.SCRATCH),
                        _cmac_weight_ref(0, 4, 32, RKBufferKind.CONSTANT), red.arg[0], constants)
  return _native(RKProgram((*prefix, RKDPUProgram(tuple(stages)), contract), tuple(scratch)))

def lower_affine_mean_result(sink:UOp) -> RKLowerResult:
  """Reject sibling ADD-reduction ratios until hardware can reproduce their FP16 accumulation contract."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 2 or any(u.arg[0] is not Ops.ADD or not u.src[1:] for u in reductions):
    return _not_applicable()
  stored = _strip_casts(stores[0].src[1])
  if stored.op is not Ops.MUL: return _not_applicable()
  pair = next((( _strip_casts(numerator), _strip_casts(reciprocal.src[0])) for numerator,reciprocal in
    (stored.src, stored.src[::-1]) if reciprocal.op is Ops.RECIPROCAL and
    _strip_casts(numerator).op is Ops.REDUCE and _strip_casts(reciprocal.src[0]).op is Ops.REDUCE), None)
  if pair is None: return _not_applicable()
  numerator, denominator = pair
  if {numerator,denominator} != set(reductions): return _not_applicable()
  numerator_value = _strip_casts(numerator.src[0])
  if _conditional_index(numerator_value) is None: return _not_applicable()
  denominator_value = _strip_casts(denominator.src[0])
  if denominator_value.op is not Ops.WHERE: return _not_applicable()
  arms = tuple(_strip_casts(x) for x in denominator_value.src[1:])
  if not all(x.op is Ops.CONST for x in arms) or {float(x.arg) for x in arms} != {0.0,1.0}: return _not_applicable()
  # Preserved hardware probes tried both materialized numerator/count division and row-scaled CMAC weights. Both differed from
  # the official avg-pool result by one FP16 ULP because CMAC does not reproduce the source reduction's accumulation contract.
  return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,
    "affine mean selector CMAC does not preserve the required FP16 accumulation rounding", stored.op)

def lower_nested_add_reduce_result(sink:UOp) -> RKLowerResult:
  """Compose two affine ADD reductions while preserving their intermediate FP16 rounding boundary."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 2 or any(u.arg[0] is not Ops.ADD or len(u.src) < 2 for u in reductions):
    return _not_applicable()
  store, outer = stores[0], _strip_casts(stores[0].src[1])
  if outer.op is not Ops.REDUCE: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     store.src[0].src[1].op is not Ops.CONST or int(store.src[0].src[1].arg) != 0 or int(store.src[0].src[0].src[0].arg) != 1:
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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = [RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, 32),
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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
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
  if not any(u.op is Ops.WHERE for u in value.toposort()): return _not_applicable()
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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = [RKDPUProgram((RKALUStage(Ops.ADD, one, 0.0, 1.0, 32),), scratch)]
  source = RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot)
  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  # Keep every prefix-product schedule inside the proven single-tile contract. Public tile starts are 16 FP16 values
  # (one 32-byte atom) apart, so independent tiles neither share selector state nor require an output compaction pass.
  for tile_start in range(0, output_count, 16):
    tile_count = min(16, output_count-tile_start)
    operands:list[RKArg] = []
    for value_rows,identity_rows in zip(selected, identities):
      selected_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
      scratch += (RKScratch(_cmac_tiled_output_bytes(tile_count)),)
      value_plan = _selector_program(selected_arg, source, input_count, value_rows[tile_start:tile_start+tile_count], scratch)
      if value_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "masked MUL value selector exceeds limits", reduce.op)
      steps.extend(value_plan.steps)
      scratch = value_plan.scratch
      identity_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
      scratch += (RKScratch(_cmac_tiled_output_bytes(tile_count)),)
      identity_plan = _selector_program(identity_arg, one, 32, identity_rows[tile_start:tile_start+tile_count], scratch)
      if identity_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "masked MUL identity selector exceeds limits", reduce.op)
      steps.extend(identity_plan.steps)
      scratch = identity_plan.scratch
      operand = RKArg(RKBufferKind.SCRATCH, len(scratch))
      scratch += (RKScratch(((tile_count+7)//8)*16),)
      steps.append(RKDPUProgram((RKALUStage(Ops.ADD, operand, selected_arg, identity_arg, tile_count),), scratch))
      operands.append(operand)
    accumulator = operands[0]
    multiplies:list[RKDPUStage] = []
    for index,operand in enumerate(operands[1:], 1):
      final = index == len(operands)-1
      destination = RKArg(output.kind, output.index, output.addend+tile_start*2) if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
      if not final: scratch += (RKScratch(((tile_count+7)//8)*16),)
      multiplies.append(RKALUStage(Ops.MUL, destination, accumulator, operand, tile_count))
      accumulator = destination
    steps.append(RKDPUProgram(tuple(multiplies)))
  completed = _finish_program(steps, scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"masked affine MUL needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(completed)

def lower_multi_source_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Reduce a static affine selection among FP16 inputs, then combine source-local partials entirely on NPU engines."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] not in (Ops.ADD,Ops.MAX): return _not_applicable()
  store, reduce = stores[0], reductions[0]
  op, label = reduce.arg[0], reduce.arg[0].name
  if _strip_casts(store.src[1]).key != reduce.key or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  indexes = list(dict.fromkeys(u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM))
  if not 2 <= len(indexes) <= 8 or any(index.dtype is not dtypes.half for index in indexes) or not reduce.src[1:] or \
     any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in reduce.src[1:]):
    return _not_applicable()
  out_map = _affine(store.src[0].src[1])
  if out_map is None: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"multi-source {label} output is not affine", store.src[0].op)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes, out_axes = tuple(u.arg[0] for u in reduce.src[1:]), tuple(sorted(out_map[0]))
  value_axes = {u.arg[0] for u in value.toposort() if u.op is Ops.RANGE}
  if set(out_axes) & set(red_axes) or value_axes - set(out_axes) - set(red_axes) or \
     any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, f"multi-source {label} axes do not form one static partition", Ops.RANGE)
  output_count, reduction_count = int(store.src[0].src[0].src[0].arg), math.prod(ranges[axis] for axis in red_axes)
  if not 1 <= output_count <= 8192 or output_count*reduction_count > 2*RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"multi-source {label} surface is {output_count}x{reduction_count}", reduce.op)
  selectors:dict[UOp, list[list[int]]] = {index:[[] for _ in range(output_count)] for index in indexes}
  identity = 0.0 if op is Ops.ADD else -math.inf
  identity_value = value.substitute({index:UOp.const(identity,index.dtype) for index in indexes})
  seen:set[int] = set()
  for out_point in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, out_point))
    output_index = out_map[1]+sum(out_map[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= output_index < output_count or output_index in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"multi-source {label} output is not dense", store.src[0].op)
    seen.add(output_index)
    for red_point in product(*(range(ranges[axis]) for axis in red_axes)):
      point.update(zip(red_axes, red_point))
      selected = [_static_index_selected(value, index, point) for index in indexes]
      if any(active is None for active in selected) or sum(active is True for active in selected) > 1:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"multi-source {label} selection is not one static branch", value.op)
      if any(selected):
        index = indexes[selected.index(True)]
        source_offset = _static_scalar(index.src[1], point)
        input_count = int(index.src[0].src[0].arg)
        if not isinstance(source_offset, int) or isinstance(source_offset, bool) or not 0 <= source_offset < input_count:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"multi-source {label} input index is out of bounds", index.op)
        selectors[index][output_index].append(source_offset)
      else:
        inactive = _static_scalar(identity_value, point)
        if not isinstance(inactive, (int,float)) or float(inactive) != identity:
          return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, f"multi-source {label} inactive value is not its identity", value.op)
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"multi-source {label} output has holes", store.src[0].op)
  if op is Ops.MAX and any(len(row) != 1 for rows in selectors.values() for row in rows):
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION,"multi-source MAX needs exactly one value from every source per output",reduce.op)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  scratch:tuple[RKScratch, ...] = ()
  partials:list[RKArg] = []
  for index,rows in selectors.items():
    if not any(rows): continue
    if int(index.src[0].src[0].arg) == output_count and rows == [[output] for output in range(output_count)]:
      partials.append(RKArg(RKBufferKind.ARG,index.src[0].arg.slot))
      continue
    partial = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    plan = _selector_program(partial, RKArg(RKBufferKind.ARG, index.src[0].arg.slot), int(index.src[0].src[0].arg), rows, scratch)
    if plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"multi-source {label} selector exceeds plan limits", reduce.op)
    steps.extend(plan.steps)
    scratch = plan.scratch
    partials.append(partial)
  if not partials: return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, f"multi-source {label} has no selected input", reduce.op)
  output, accumulator = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), partials[0]
  combines:list[RKDPUStage] = []
  for combine_idx,partial in enumerate(partials[1:], 1):
    final = combine_idx == len(partials)-1
    destination = output if final else RKArg(RKBufferKind.SCRATCH, len(scratch))
    if not final: scratch += (RKScratch(((output_count+7)//8)*16),)
    combines.append(RKALUStage(op, destination, accumulator, partial, output_count))
    accumulator = destination
  if len(partials) == 1: combines.append(RKALUStage(Ops.ADD, output, accumulator, 0.0, output_count))
  completed = _finish_program([*steps,RKDPUProgram(tuple(combines))], scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"multi-source {label} needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
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
  scale_terms = tuple(_strip_casts(x) for x in stored.src) if stored.op is Ops.MUL else ()
  positive_inf_scale = len(scale_terms) == 2 and any(x.op is Ops.CONST and float(x.arg) == math.inf for x in scale_terms) and \
    any(x.key == reduce.key for x in scale_terms)
  reduced_value = _strip_casts(reduce.src[0])
  squared_sum = reduced_value.op is Ops.MUL and _strip_casts(reduced_value.src[0]).key == _strip_casts(reduced_value.src[1]).key
  if positive_inf_scale and squared_sum:
    # SUM(square) is nonnegative, so x*(+inf) has exactly the same IEEE result as x/(+0): +inf for x>0 and NaN for x=0.
    # RK3588's native FDIV preserves that contract after CMAC, while its multiply-by-infinity epilogue does not.
    root:_Expr|RKArg|float|None = _ALUExpr(Ops.FDIV,(reduced,0.0))
  else: root = _parse_alu(stored, output_index, memo)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "reduction epilogue is not legal DPU arithmetic", stored.op)
  # FDIV after CMAC is stable only when the final DPU atom is complete. Keep the epilogue in aligned scratch, then copy the
  # full page-backed atom to the public surface; do not issue an unaligned tail-address write (DPU base addresses are atom-granular).
  schedule_count, scheduled_output = (output_count+7)&-8, output
  if schedule_count != output_count:
    scheduled_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(schedule_count*2),)
  scheduled = _schedule_expr(root, scheduled_output, schedule_count, scratch)
  if scheduled is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "reduction epilogue is not materializable", stored.op)
  tail = () if scheduled_output is output else (RKDPUProgram((RKALUStage(Ops.ADD,output,scheduled_output,0.0,schedule_count),)),)
  completed = _finish_program([*steps, scheduled, *tail], scheduled.scratch)
  cost = plan_cost(completed)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"reduction epilogue needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", stored.op)
  return _native(completed)

def lower_pointwise_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Materialize a multi-input pointwise expression, then reduce its static affine output rows."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1 or reductions[0].arg[0] is not Ops.ADD: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  indexes = list(dict.fromkeys(u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM))
  if not 2 <= len(indexes) <= 4 or any(index.dtype is not dtypes.half for index in indexes): return _not_applicable()
  if value.op is Ops.MUL and all(_strip_casts(operand).op is Ops.INDEX for operand in value.src):
    return _not_applicable()  # direct contractions must retain CMAC/CNA accumulation instead of FP16 pointwise products
  output_index, out_aff = store.src[0].src[1], _affine(store.src[0].src[1])
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0])) if out_aff is not None else ()
  value_axes = {u.arg[0] for u in value.toposort() if u.op is Ops.RANGE}
  if out_aff is None or len(red_axes) != len(reduce.src)-1 or set(out_axes) & set(red_axes) or \
     value_axes-set(out_axes)-set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT,
      "pointwise affine SUM axes do not form one static output/reduction partition", Ops.RANGE)
  output_count, reduction_count = int(store.src[0].src[0].src[0].arg), math.prod(ranges[axis] for axis in red_axes)
  visit_count = output_count*reduction_count
  if not 1 <= output_count <= 128 or not 2 <= visit_count <= RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"pointwise affine SUM surface is {output_count}x{reduction_count}", reduce.op)

  mappings:dict[UOp,list[int]] = {index:[-1]*visit_count for index in indexes}
  rows:list[list[int]] = [[] for _ in range(output_count)]
  seen:set[int] = set()
  for out_point in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes,out_point))
    out_offset = out_aff[1]+sum(out_aff[0][axis]*point[axis] for axis in out_axes)
    if not 0 <= out_offset < output_count or out_offset in seen:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"pointwise affine SUM output is not dense",store.src[0].op)
    seen.add(out_offset)
    for red_offset,red_point in enumerate(product(*(range(ranges[axis]) for axis in red_axes))):
      point.update(zip(red_axes,red_point))
      visit = out_offset*reduction_count+red_offset
      rows[out_offset].append(visit)
      for index in indexes:
        source_offset = _static_scalar(index.src[1],point)
        input_count = int(index.src[0].src[0].arg)
        if not isinstance(source_offset,int) or isinstance(source_offset,bool) or not 0 <= source_offset < input_count:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"pointwise affine SUM input is not static",index.op)
        mappings[index][visit] = source_offset
  if seen != set(range(output_count)) or any(-1 in mapping for mapping in mappings.values()):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"pointwise affine SUM surface has holes",Ops.INDEX)

  scratch:tuple[RKScratch,...] = ()
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  memo:dict[UOp,_Expr|RKArg|float] = {}
  for index,mapping in mappings.items():
    # Rejected WIP: a logically identity map is not enough to alias the ARG here. The selector's physical output layout
    # is part of the pointwise scheduler contract; bypassing it corrupted a previously native variance subcase.
    expanded = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(visit_count)),)
    packed = _selector_program(expanded,RKArg(RKBufferKind.ARG,index.src[0].arg.slot),int(index.src[0].src[0].arg),
                               [[source] for source in mapping],scratch,max_outputs=64)
    if packed is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"pointwise affine SUM input materialization exceeds limits",index.op)
    steps.extend(packed.steps)
    scratch = packed.scratch
    memo[index] = expanded
  root = _parse_alu(value,indexes[0].src[1],memo)
  if not isinstance(root,(_ALUExpr,_MaskExpr,_LUTExpr)):
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU,"pointwise affine SUM expression is not legal DPU arithmetic",value.op)
  pointwise = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(((visit_count+7)//8)*16),)
  scheduled = _schedule_expr(root,pointwise,visit_count,scratch)
  if scheduled is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU,"pointwise affine SUM expression is not materializable",value.op)
  steps.append(scheduled)
  scratch = scheduled.scratch

  stored, output = _strip_casts(store.src[1]), RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot)
  epilogue = stored.key != reduce.key
  reduced = output
  reduction = _selector_program(reduced,pointwise,visit_count,rows,scratch,direct_capacity=visit_count,max_outputs=64)
  if reduction is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"pointwise affine SUM reduction exceeds limits",reduce.op)
  program = _finish_program([*steps,*reduction.steps],reduction.scratch)
  if epilogue:
    return _finish_reduction_epilogue(program,stored,reduce,output_index,output,reduced,output_count,out_axes,ranges)
  cost = plan_cost(program)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"pointwise affine SUM needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def lower_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a small affine FP16 ADD reduction as generated sparse CMAC tiles."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or not reduce.src[1:]: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype not in (dtypes.half,dtypes.float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine CMAC requires an FP16 or scalar FP32 output", store.src[0].op)
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
  fp32_out = store.src[0].dtype is dtypes.float
  if fp32_out and (output_count != 1 or epilogue):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine CMAC FP32 output requires one direct scalar reduction", store.src[0].op)
  src_axes = set(src_aff[0]) if src_aff is not None else {u.arg[0] for u in value_index.src[1].toposort() if u.op is Ops.RANGE}
  if len(red_axes) != len(reduce.src)-1 or set(out_axes) & set(red_axes) or \
     src_axes - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine CMAC axes do not form one static output/reduction partition", Ops.RANGE)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
  # A statically masked reduction may describe identity padding that never reaches CMAC. Inspect a bounded amount of that
  # logical padding, then apply the unchanged affine-visit budget to the selected source terms below.
  logical_visits = output_count*reduction_count
  prefix_candidate = condition is not None and not epilogue and len(red_axes) == 1 and reduction_count <= RK_MAX_PREFIX_WINDOW and \
                     (output_count == input_count or max(output_count,input_count) <= RK_MAX_PREFIX_WINDOW)
  visit_limit = RK_MAX_PREFIX_VISITS if prefix_candidate else RK_MAX_STATIC_MASK_VISITS if condition is not None else RK_MAX_AFFINE_VISITS
  output_limit = 64*RK_MAX_PROGRAM_STAGES if prefix_candidate else 8192
  input_limit = output_limit if prefix_candidate else 65536
  if not 1 <= output_count <= output_limit or not 2 <= input_count <= input_limit or logical_visits > visit_limit:
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
  selected_visits = sum(map(len, selectors))
  ordinary_prefix = output_count%reduction_count == 0 and all(len(row) == index%reduction_count+1 and
    (index%reduction_count == 0 or row[:-1] == selectors[index-1]) for index,row in enumerate(selectors))
  # Tinygrad's long scan pads 1,022 inputs to four 256-output groups. The first group begins with identity rows and the
  # final group repeats its saturated tail. Retain this only when every group is one monotone contiguous source prefix.
  def monotone_prefix_group(rows:list[list[int]]) -> bool:
    nonempty, lengths = [row for row in rows if row], [len(row) for row in rows]
    return all(row == list(range(row[0],row[0]+len(row))) for row in nonempty) and len({row[0] for row in nonempty}) <= 1 and \
      all(lhs <= rhs for lhs,rhs in zip(lengths,lengths[1:]))
  padded_prefix = output_count <= RK_MAX_PREFIX_WINDOW and input_count <= RK_MAX_PREFIX_WINDOW and \
    all(monotone_prefix_group(selectors[start:start+reduction_count]) for start in range(0,output_count,reduction_count))
  prefix_scan = prefix_candidate and (ordinary_prefix or padded_prefix)
  if selected_visits > (RK_MAX_PREFIX_VISITS if prefix_scan else RK_MAX_AFFINE_VISITS):
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine CMAC selects {selected_visits} source terms from {logical_visits} logical visits", reduce.op)
  if prefix_scan and output_count <= 32 and prepare is None and _fp16_exact(scale):
    align_in, align_out = max(32,(input_count+31)&-32), 32
    packed = RKArg(RKBufferKind.SCRATCH,0)
    scratch = (RKScratch(align_in*2),)
    prep = RKDPUProgram((RKALUStage(Ops.ADD,packed,0.0,0.0,align_in),RKALUStage(Ops.ADD,packed,source,0.0,input_count)),scratch)
    out_layout = RKLayout((1,output_count),(1,align_out),(align_out*2,2),dtypes.half,padding=((0,0),(0,align_out-output_count)))
    lhs_layout = RKLayout((1,input_count),(1,align_in),(align_in*2,2),dtypes.half,padding=((0,0),(0,align_in-input_count)))
    contract = RKCMACTask(RKTensorRef(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),out_layout),
      RKTensorRef(packed,lhs_layout),_cmac_weight_ref(0,output_count,align_in,RKBufferKind.CONSTANT,align_out),reduce.op,
      _cmac_selection_payload(selectors,align_in,align_out,scale),compact_output=True)
    return _native(_finish_program([prep,contract],scratch))
  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  initial_scratch = () if prepare is None else prepare.scratch
  reduced = output
  if epilogue:
    reduced = RKArg(RKBufferKind.SCRATCH, len(initial_scratch))
    initial_scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
  program = _sparse_cmac_pipeline(reduced, source, input_count, selectors, scale, initial_scratch, store.src[0].dtype) if \
    input_count <= 512 and output_count <= 128 else (None if fp32_out else _windowed_cmac_pipeline(
      reduced, source, selectors, scale, initial_scratch, direct_count=input_count,
      max_window=RK_MAX_PREFIX_WINDOW if prefix_scan else 512))
  # Rejected WIP: reducing with unit weights and applying a non-FP16 coefficient afterward submitted hundreds of tasks,
  # then left the average-pool hardware probe blocked in device wait. Per-term scaling also misses the official rounding.
  if program is None and struct.unpack("<e", struct.pack("<e", scale))[0] != scale:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, f"two-level affine scale {scale} is not exactly FP16", stored.op)
  if program is None: program = _two_level_selector_program(reduced, source, input_count, selectors, initial_scratch, scale)
  if program is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "affine CMAC output tiles exceed the source-window or constant budget", reduce.op)
  if prepare is not None: program = RKProgram((RKDPUProgram(prepare.stages, program.scratch), *program.steps), program.scratch)
  if epilogue: return _finish_reduction_epilogue(program, stored, reduce, store.src[0].src[1], output, reduced,
                                                  output_count, out_axes, ranges)
  return _native(program)
