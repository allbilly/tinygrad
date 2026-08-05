from __future__ import annotations
from itertools import product
from typing import cast
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops, UOp

from tinygrad.renderer.rockchip.ir import (RKBufferKind, RKArg, RKALUStage, RKFusedALUStage, RKCopyStage, RKCastStage,
  RKDPUStage, RKScratch, RKDPUProgram, RKLayout, RKTensorRef, RKCMACTask, RKConvTask, RKReduce, RKRejectKind, RKLowerKind, RKLowerResult)
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.analysis import (strip_casts as _strip_casts, static_scalar as _static_scalar,
  conditional_index as _conditional_index, fp16_exact as _fp16_exact)
from tinygrad.renderer.rockchip.lower import native as _native, not_applicable as _not_applicable, unsupported as _unsupported
from tinygrad.renderer.rockchip.limits import RK_MAX_CONSTANT_BYTES, RK_MAX_AFFINE_VISITS, RK_MAX_PROGRAM_STAGES
from tinygrad.renderer.rockchip.selector import (_cmac_tiled_output_bytes, _dense_half_ref, _cmac_weight_ref,
  _cmac_selection_payload, _selector_program, _finish_program,
  _periodic_selector_program, _constant_run_selector_program)
from tinygrad.renderer.rockchip.expr import (_ALUExpr, _MaskExpr, _LUTExpr, _Expr, _Value, _parse_alu, _unwrap_same_cast,
  _canonical_lerp, _canonical_tensor_pow, _numerical_contract)

def _lower_fused_lerp(output:RKArg, operands:tuple[UOp,UOp,UOp], count:int) -> RKLowerResult:
  x,y,z = operands
  if x.op is not Ops.INDEX or y.op is not Ops.INDEX or z.op not in (Ops.INDEX,Ops.CONST): return _not_applicable()
  source = RKArg(RKBufferKind.ARG, x.src[0].arg.slot)
  x_float = RKArg(RKBufferKind.SCRATCH, 0)
  scratch = (RKScratch((((count+31)//32)*32+32)*4),)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  for start in range(0,count,32):
    valid = min(32,count-start)
    lhs_layout = RKLayout((1,valid), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-valid)))
    out_layout = RKLayout((1,valid), (1,64), (256,4), dtypes.float, padding=((0,0),(0,64-valid)))
    steps.append(RKCMACTask(RKTensorRef(RKArg(x_float.kind,x_float.index,start*4),out_layout),
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

def _int_fill_program(output:RKArg, count:int, value:int) -> RKDPUProgram:
  tile = 64
  return RKDPUProgram(tuple(RKALUStage(Ops.ADD,RKArg(output.kind,output.index,start*4),0.0,0.0,min(tile,count-start),
    dtypes.int,value&0xffffffff) for start in range(0,count,tile)))

def lower_dpu_result(sink:UOp) -> RKLowerResult:
  """Lower one contiguous expression or native wide constant fill to a typed DPU result."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, f"expected one store, found {len(stores)}", Ops.STORE)
  store = stores[0]
  if store.src[0].op is not Ops.INDEX: return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "output is not an indexed surface", store.src[0].op)
  if store.src[0].dtype not in (dtypes.bool, dtypes.half, dtypes.int, dtypes.float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"output dtype {store.src[0].dtype.name}", store.src[0].op)
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op not in (Ops.RANGE, Ops.CONST) or out_param.src[0].op is not Ops.CONST:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "output surface is not contiguous", out_index.op)
  count = int(out_param.src[0].arg)
  if not 0 < count <= 65536 or (out_index.op is Ops.RANGE and int(out_index.src[0].arg) != count) or \
     (out_index.op is Ops.CONST and (count != 1 or int(out_index.arg) != 0)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, f"unsupported contiguous output extent {count}", out_index.op)
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  identity = _strip_casts(store.src[1])
  if store.src[0].dtype in (dtypes.bool,dtypes.int,dtypes.float) and identity.op is Ops.INDEX and identity.dtype is store.src[0].dtype and \
     identity.src[0].op is Ops.PARAM and identity.src[1].key == out_index.key:
    return _native(RKDPUProgram((RKCopyStage(output,RKArg(RKBufferKind.ARG,identity.src[0].arg.slot),count,store.src[0].dtype),)))
  if store.src[0].dtype is dtypes.int and identity.op is Ops.BITCAST and identity.src[0].op is Ops.INDEX and \
     identity.src[0].dtype is dtypes.float and identity.src[0].src[0].op is Ops.PARAM and identity.src[0].src[1].key == out_index.key:
    # All-bypass int32 transport preserves each source word; no FP32 arithmetic or conversion is involved.
    return _native(RKDPUProgram((RKCopyStage(output,RKArg(RKBufferKind.ARG,identity.src[0].src[0].arg.slot),count,dtypes.int),)))
  if store.src[0].dtype is dtypes.int and identity.op is Ops.OR and any(
      operand.op is Ops.CONST and int(operand.arg)&0xffffffff == 0xffffffff for operand in identity.src):
    return _native(_int_fill_program(output,count,-1))
  if store.src[0].dtype is dtypes.bool:
    if identity.op is Ops.CONST:
      return _native(RKDPUProgram((RKCopyStage(output,bool(identity.arg),count,dtypes.bool),)))
    bool_sources,physical_op = (identity.src,Ops.MAX) if identity.op is Ops.OR else (None,None)
    if identity.op is Ops.CMPNE and identity.src[1].op is Ops.CONST and identity.src[1].arg is True and \
       identity.src[0].op is Ops.OR and all(u.op is Ops.CMPNE and u.src[1].op is Ops.CONST and u.src[1].arg is True
                                          for u in identity.src[0].src):
      bool_sources,physical_op = (tuple(u.src[0] for u in identity.src[0].src),Ops.MUL)
    if bool_sources is not None and all(u.op is Ops.INDEX and u.dtype is dtypes.bool and u.src[0].op is Ops.PARAM and
                                        u.src[1].key == out_index.key for u in bool_sources):
      lhs,rhs = (RKArg(RKBufferKind.ARG,u.src[0].arg.slot) for u in bool_sources)
      assert physical_op is not None
      return _native(RKDPUProgram((RKALUStage(physical_op,output,lhs,rhs,count,dtypes.bool),)))
    if identity.op is Ops.CMPNE and identity.src[1].op is Ops.CONST and identity.src[1].arg is True and \
       identity.src[0].op is Ops.INDEX and identity.src[0].dtype is dtypes.bool and identity.src[0].src[0].op is Ops.PARAM and \
       identity.src[0].src[1].key == out_index.key:
      source = RKArg(RKBufferKind.ARG,identity.src[0].src[0].arg.slot)
      return _native(RKDPUProgram((RKALUStage(Ops.SUB,output,1.0,source,count,dtypes.bool),)))
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE,"non-identity bool output",store.src[1].op)
  input_indexes = [u for u in store.src[1].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  bool_indexes = tuple(u for u in input_indexes if u.dtype is dtypes.bool)
  if any(u.src[1].key != out_index.key for u in bool_indexes):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "bool input index map differs from output surface", Ops.INDEX)
  if bool_indexes and count > 8:
    return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE,"bool input conversion requires one eight-value atom",Ops.INDEX)
  bool_params = tuple(dict.fromkeys(u.src[0] for u in bool_indexes))
  scratch = tuple(RKScratch(((count+7)//8)*16) for _ in bool_params)
  bool_refs = {param.key:RKArg(RKBufferKind.SCRATCH,slot) for slot,param in enumerate(bool_params)}
  cast_stages = tuple(RKCastStage(ref,RKArg(RKBufferKind.ARG,param.arg.slot),count,dtypes.bool,dtypes.half)
                      for param,ref in zip(bool_params,bool_refs.values()))
  memo:dict[UOp,_Expr|RKArg|float] = {u:bool_refs[u.src[0].key] for u in bool_indexes}
  # Rejected WIP: DATA_FORMAT in_precision=precision_float32 exists in the register enum, but a direct FP32->FP16 ADD timed out on RK3588.
  # The exact typed-stage/emitter probe is preserved as wip-native-fp32-dpu-input-timeout.patch; do not restore 2607's CPU narrowing instead.
  if (bad_dtype:=next((u.dtype for u in input_indexes if u.dtype not in (dtypes.bool,dtypes.half)), None)) is not None:
    return _unsupported(RKRejectKind.UNSUPPORTED_INPUT_DTYPE, f"input dtype {bad_dtype.name}", Ops.INDEX)
  if any(u.src[1].key != out_index.key for u in input_indexes):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "input index map differs from output surface", Ops.INDEX)
  if (lerp:=_canonical_lerp(store.src[1])) is not None:
    result = _lower_fused_lerp(output, lerp, count)
    if result.kind is not RKLowerKind.NOT_APPLICABLE: return result
  if (reason:=_numerical_contract(store.src[1])) is not None:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, reason, _unwrap_same_cast(store.src[1]).op)
  root = _parse_alu(store.src[1], out_index, memo)
  if root is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "expression is not legal DPU arithmetic", _unwrap_same_cast(store.src[1]).op)
  int_where = store.src[0].dtype is dtypes.int and identity.op is Ops.WHERE and all(
    arm.op is Ops.CONST and isinstance(arm.arg,(int,float)) and float(arm.arg).is_integer() and _fp16_exact(float(arm.arg))
    for arm in identity.src[1:])
  if store.src[0].dtype is not dtypes.half and not isinstance(root, float) and not int_where:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"non-constant {store.src[0].dtype.name} arithmetic", store.src[1].op)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      if not isinstance(root, float):
        return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"non-constant {store.src[0].dtype.name} fill", store.src[1].op)
      if store.src[0].dtype is dtypes.int:
        if not root.is_integer() or not dtypes.int.min <= root <= dtypes.int.max:
          return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, f"int fill value {root!r} is outside signed int32", store.src[1].op)
        return _native(_int_fill_program(output,count,int(root)))
      if not _fp16_exact(root):
        return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,
          f"{store.src[0].dtype.name} fill value {root!r} is not exactly representable by the FP16 DPU input", store.src[1].op)
      tile = 4
      fill_stages = tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*4), 0.0, root, min(tile, count-start),
                                     store.src[0].dtype) for start in range(0, count, tile))
      return _native(RKDPUProgram(fill_stages))
    return _native(RKDPUProgram(tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*2), 0.0, root,
      min(32768, count-start)) for start in range(0, count, 32768))))
  # Rejected WIP: materializing every partial-atom DPU input is correct but unnecessarily duplicates ordinary elementwise work.
  # The allocator clears upload padding instead; selector planners remain responsible for initializing scratch padding.
  scheduled_output = output
  if int_where:
    scheduled_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(((count+7)//8)*16),)
  if (program:=_schedule_expr(root, scheduled_output, count, scratch)) is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "stage source is not materializable")
  stages:list[RKDPUStage] = [*cast_stages,*program.stages]
  if int_where:
    if count > 4:
      padded_count = ((count+3)//4)*8
      padded = RKArg(RKBufferKind.SCRATCH,len(program.scratch))
      padded_scratch = program.scratch+(RKScratch(padded_count*2),)
      rows = [[[start+lane] if start+lane < count and lane < 4 else [] for lane in range(8)]
              for start in range(0,count,4)]
      packed = _selector_program(padded,scheduled_output,count,[row for group in rows for row in group],padded_scratch,
                                 direct_capacity=count)
      if packed is None:
        return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"int WHERE packing exceeds the native cost contract",Ops.WHERE)
      conversions = tuple(RKALUStage(Ops.ADD,RKArg(output.kind,output.index,start*4),
        RKArg(padded.kind,padded.index,start//4*16),0.0,min(4,count-start),dtypes.int) for start in range(0,count,4))
      return _native(_finish_program([RKDPUProgram(tuple(stages),packed.scratch),*packed.steps,
        RKDPUProgram(conversions,packed.scratch)],packed.scratch))
    stages.append(RKALUStage(Ops.ADD,output,scheduled_output,0.0,count,dtypes.int))
  return _native(RKDPUProgram(tuple(stages),program.scratch))

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Compatibility helper for compiler probes; production lowering consumes `lower_dpu_result`."""
  return cast(RKDPUProgram|None, lower_dpu_result(sink).plan)

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
  # Direct tensor POW is calibrated against exact paired FP16 inputs. Reusing one operand through a materialized
  # broadcast changes the deterministic rounding groups: the two official small layouts miss one lane by 0.015625.
  # Keep this native-only path rejected until broadcast-output calibration is proven over the complete FP16 domain.
  if _canonical_tensor_pow(stored) is not None:
    return _unsupported(RKRejectKind.NUMERICAL_CONTRACT, "broadcast tensor POW lacks an output-domain error contract", Ops.EXP2)
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
  if not 0 < count <= RK_MAX_AFFINE_VISITS or not 0 < src_count:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"broadcast surface is {count} from {src_count}", Ops.INDEX)

  canonical = source_index.replace(src=(source_index.src[0], output_index))
  root = _parse_alu(stored.substitute({source_index if condition is None else surface:canonical}), output_index, {})
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

  source = RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot)
  if count > 4096:
    initial_scratch = (RKScratch(_cmac_tiled_output_bytes(count)),)
    expanded_plan = _periodic_selector_program(expanded, source, src_count,
      [[index] if index >= 0 else [] for index in mapping], initial_scratch) or \
      _constant_run_selector_program(expanded, source, src_count, mapping, initial_scratch)
    if expanded_plan is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "large broadcast has no legal aligned periodic plan", Ops.INDEX)
    output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
    if (scheduled:=_schedule_expr(root, output, count, expanded_plan.scratch)) is None:
      return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "broadcast stage source is not materializable", stored.op)
    completed = _finish_program([*expanded_plan.steps,scheduled], scheduled.scratch)
    cost = plan_cost(completed)
    if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
        f"large periodic broadcast needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", stored.op)
    return _native(completed)

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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  if any(all(x < 0 for x in mapping[start:start+16]) for start in range(0, count, 16)):
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD, expanded, 0.0, 0.0, count),)))
  for base,end,blocks in chunks:
    span, align_in = end-base, max(32, (end-base+31)&-32)
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                               RKALUStage(Ops.ADD, packed, RKArg(source.kind, source.index, base*2), 0.0, span))))
    for start,rows in blocks:
      valid = len(rows)
      out_layout = RKLayout((1,valid), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-valid)))
      steps.append(RKCMACTask(RKTensorRef(RKArg(expanded.kind, expanded.index, start*2), out_layout),
        _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH),
        _cmac_weight_ref(0, valid, align_in, RKBufferKind.CONSTANT, 32), 0,
        _cmac_selection_payload([[index-base] if index >= 0 else [] for index in rows], align_in, 32, 1.0)))

  output = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot)
  if (scheduled:=_schedule_expr(root, output, count, tuple(scratch))) is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "broadcast stage source is not materializable", stored.op)
  completed = _finish_program([*steps, scheduled], scheduled.scratch)
  cost = plan_cost(completed)
  return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
    f"broadcast expression needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", stored.op) \
    if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES else _native(completed)

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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  for source_index in broadcasts:
    surfaces = [(u, parsed) for u in stored.toposort() if (parsed:=_conditional_index(u)) is not None and parsed[0].key == source_index.key]
    if not surfaces: return _not_applicable()
    surface,(_,condition,select_true) = surfaces[-1]
    if {u.arg[0] for u in surface.toposort() if u.op is Ops.RANGE}-set(axes):
      return _unsupported(RKRejectKind.UNSUPPORTED_BROADCAST,"multi-broadcast surface has axes outside its output",Ops.RANGE)
    src_count = int(source_index.src[0].src[0].arg)
    mapping = [-1]*count
    for coordinates in product(*(range(ranges[axis]) for axis in axes)):
      point = dict(zip(axes, coordinates))
      dst, predicate = _static_scalar(output_index,point), True if condition is None else _static_scalar(condition,point)
      selected = predicate is not None and bool(predicate) is select_true
      src = _static_scalar(source_index.src[1],point) if selected else -1
      if not isinstance(dst, int) or isinstance(dst, bool) or not 0 <= dst < count or mapping[dst] != -1 or \
         predicate is None or not isinstance(src, int) or isinstance(src, bool) or selected and not 0 <= src < src_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "multi-broadcast does not cover one dense output", Ops.INDEX)
      mapping[dst] = src
    expanded = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(count)),)
    packed = _selector_program(expanded, RKArg(RKBufferKind.ARG, source_index.src[0].arg.slot), src_count,
                               [[source] for source in mapping], scratch)
    if packed is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "multi-broadcast selector exceeds plan limits", Ops.INDEX)
    steps.extend(packed.steps)
    scratch = packed.scratch
    substitutions[source_index if condition is None else surface] = source_index.replace(src=(source_index.src[0], output_index))
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
  remapped_root = cast(_Expr,remap(root))
  def contains_unbounded_exp(value:_Value) -> bool:
    return isinstance(value,_LUTExpr) and value.lut in (RKLUTId.EXP,RKLUTId.EXP_LOCAL) or \
      isinstance(value,(_ALUExpr,_MaskExpr,_LUTExpr)) and any(contains_unbounded_exp(source) for source in value.src)
  # The generated EXP tables are characterized only on [-2,2]. Multi-broadcast expressions currently have no range proof;
  # rejecting here prevents centered softmax/softmin graphs from silently clamping legal inputs below -2.
  if contains_unbounded_exp(remapped_root):
    return _unsupported(RKRejectKind.LUT_DOMAIN_UNPROVEN,"multi-broadcast EXP input is not proven inside [-2,2]",Ops.EXP2)
  scheduled = _schedule_expr(remapped_root, RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), count, scratch)
  if scheduled is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "multi-broadcast expression is not materializable", stored.op)
  completed = _finish_program([*steps,scheduled], scheduled.scratch)
  cost = plan_cost(completed)
  return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
    f"multi-broadcast expression needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", stored.op) \
    if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES else _native(completed)
