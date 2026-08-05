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
  RKALUStage, RKFusedALUStage as RKFusedALUStage, RKCopyStage, RKCastStage,
  RKMaskStage as RKMaskStage, RKLUTStage as RKLUTStage, RKDPUStage,
  RKScratch, RKDPUProgram, RKLayout, RKTensorRef, RKEpilogue, RKContractionPlan, RKCMACTask, RKConvTask, RKConvPlan as RKConvPlan,
  RKConvSplit as RKConvSplit, RKConvTile as RKConvTile, RKConvTiling as RKConvTiling, RKReduce, RKPool, RKReformatPlan as RKReformatPlan,
  RKMultiSourceReformatPlan, RKLegalizedReformat, RKProgram, RKPlanCost as RKPlanCost,
  RKRejectKind, RKReject, RKLowerKind, RKLowerResult)
from tinygrad.renderer.rockchip.image import (RK_STAGE_RESET as RK_STAGE_RESET, RKReloc as RKReloc, RKStage as RKStage, RKImage as RKImage,
  encode_image, decode_image as decode_image,
  patch_image as patch_image, validate_image as validate_image)
from tinygrad.renderer.rockchip.affine import affine as _affine, rk_fingerprint as rk_fingerprint
from tinygrad.renderer.rockchip.access import (RKAccessMap as RKAccessMap, RKIdentityMap as RKIdentityMap, RKAffineMap as RKAffineMap,
  RKPadMap as RKPadMap, RKPeriodicMap as RKPeriodicMap, RKAffineSegment as RKAffineSegment,
  RKPiecewiseAffineMap as RKPiecewiseAffineMap, RKStaticSelectorMap as RKStaticSelectorMap)
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr
from tinygrad.renderer.rockchip.conv import plan_conv_cbuf as plan_conv_cbuf, legalize_conv_plan as legalize_conv_plan
from tinygrad.renderer.rockchip.contract import legalize_contraction_plan as legalize_contraction_plan
from tinygrad.renderer.rockchip.cost import plan_cost as plan_cost
from tinygrad.renderer.rockchip.analysis import (strip_casts as _strip_casts, static_scalar as _static_scalar,
  static_linear_form as _static_linear_form, conditional_index as _conditional_index,
  conditional_index_affine as _conditional_index_affine, conv_zero_padding as _conv_zero_padding,
  static_index_selected as _static_index_selected, static_selected_index as _static_selected_index,
  relu_source as _relu_source, contract_bias_epilogue as _contract_bias_epilogue)

from tinygrad.renderer.rockchip.limits import (RK_MAX_CONSTANT_BYTES, RK_MAX_AFFINE_VISITS, RK_MAX_STATIC_MASK_VISITS,
  RK_MAX_PROGRAM_STAGES, RK_MAX_AFFINE_WINDOW, RK_MAX_CMAC_SELECTOR_WINDOW, RK_MAX_TILED_CMAC_SELECTOR_WINDOW,
  RK_MAX_TILED_CONTRACT_VISITS, RK_MAX_PREFIX_WINDOW, RK_MAX_PREFIX_VISITS)
from tinygrad.renderer.rockchip.selector import (_cmac_tiled_output_bytes, _dense_ref, _dense_half_ref, _cmac_weight_ref,
  _cmac_mask_payload, _cmac_selection_payload, _sparse_cmac_pipeline, _windowed_weighted_cmac_pipeline,
  _windowed_cmac_pipeline, _two_level_selector_program, _selector_program, _multi_source_windowed_program,
  _best_partitioned_selector_program, _finish_program, _legalized_reformat,
  _periodic_selector_program, _constant_run_selector_program)

def _fp16_exact(value:float) -> bool:
  try: rounded = struct.unpack("<e", struct.pack("<e", value))[0]
  except OverflowError: return False
  return math.isnan(value) and math.isnan(rounded) or rounded == value

def _native(plan:RKDPUProgram|RKCMACTask|RKConvTask|RKReduce|RKLegalizedReformat|RKProgram) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.NATIVE, plan=plan)
def _not_applicable() -> RKLowerResult: return RKLowerResult(RKLowerKind.NOT_APPLICABLE)
def _unsupported(kind:RKRejectKind, detail:str, node_op:Ops|None=None) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(kind, detail, node_op))

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

def lower_static_two_tap_result(sink:UOp) -> RKLowerResult:
  """Lower a bounded dense FP32 affine blend of statically indexed FP16 values through CMAC."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or \
     store.src[0].dtype not in (dtypes.half,dtypes.float):
    return _not_applicable()
  output_index,stored = store.src[0].src[1],_strip_casts(store.src[1])
  if stored.dtype is not dtypes.float: return _not_applicable()
  indexes = tuple(u for u in stored.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM)
  sources = tuple(dict.fromkeys(u.src[0] for u in indexes))
  if not indexes or len(sources) != 1 or any(u.dtype is not dtypes.half for u in indexes): return _not_applicable()
  source = sources[0]
  ranges = tuple(u for u in sink.toposort() if u.op is Ops.RANGE)
  visits = math.prod(int(u.src[0].arg) for u in ranges)
  count,source_count = int(store.src[0].src[0].src[0].arg),int(source.src[0].arg)
  if not ranges or visits != count or visits > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,f"static two-tap transform visits {visits} outputs",Ops.RANGE)
  rows:list[list[tuple[int,float]]|None] = [None]*count
  for coordinates in product(*(range(int(u.src[0].arg)) for u in ranges)):
    point = dict(zip(ranges,coordinates))
    dst,form = _static_scalar(output_index,point),_static_linear_form(stored,point,source)
    if not isinstance(dst,int) or isinstance(dst,bool) or not 0 <= dst < count or rows[dst] is not None or form is None:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static two-tap transform does not cover one dense output",Ops.INDEX)
    constant,terms = form
    term_limit = 8 if store.src[0].dtype is dtypes.float else 2
    if constant != 0.0 or not 1 <= len(terms) <= term_limit or any(not 0 <= index < source_count for index in terms) or \
       not math.isclose(sum(terms.values()),1.0,rel_tol=0.0,abs_tol=1e-12):
      return _unsupported(RKRejectKind.UNSUPPORTED_ALU,"static transform is not a bounded zero-bias affine blend",stored.op)
    ordered = sorted(terms.items())
    if store.src[0].dtype is dtypes.half and len(ordered) == 2 and ordered[1][0] != ordered[0][0]+1:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static two-tap sources are not adjacent",Ops.INDEX)
    high = struct.unpack("<e",struct.pack("<e",ordered[-1][1]))[0]
    rounded = [(ordered[0][0],struct.unpack("<e",struct.pack("<e",1.0-high))[0]),(ordered[1][0],high)] if len(ordered) == 2 else \
      [(index,struct.unpack("<e",struct.pack("<e",weight))[0]) for index,weight in ordered]
    if any(not math.isfinite(weight) or not 0.0 <= weight <= 1.0 for _,weight in rounded):
      return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,"static two-tap coefficients do not fit FP16",stored.op)
    rows[dst] = rounded
  if any(row is None for row in rows):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static two-tap transform has output holes",Ops.INDEX)
  output = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot)
  source_arg = RKArg(RKBufferKind.ARG,source.arg.slot)
  program = _windowed_weighted_cmac_pipeline(output,source_arg,cast(list[list[tuple[int,float]]],rows),direct_count=source_count,
                                             out_dtype=store.src[0].dtype)
  if program is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"static two-tap transform exceeds native plan limits",stored.op)
  cost = plan_cost(program)
  return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
    f"static two-tap transform needs {cost.task_count} tasks and {cost.constant_bytes} constant bytes",stored.op) \
    if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES else _native(program)

def lower_reformat_result(sink:UOp) -> RKLowerResult:
  """Lower a static affine movement through atom copies or a sparse CMAC selector."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM:
    return _not_applicable()
  value, condition, select_true, fill = _strip_casts(store.src[1]), None, True, 0.0
  if value.op is Ops.WHERE:
    cond, positive, negative = value.src
    positive, negative = _strip_casts(positive), _strip_casts(negative)
    # Padding with a nonzero value is simplified as WHERE(p, WHERE(p, load, 0), fill). Inside the outer true/false arms,
    # an identical nested predicate has a statically known branch and can be removed without changing tensor semantics.
    if positive.op is Ops.WHERE and positive.src[0].key == cond.key: positive = _strip_casts(positive.src[1])
    if negative.op is Ops.WHERE and negative.src[0].key == cond.key: negative = _strip_casts(negative.src[2])
    if positive.op is Ops.INDEX and negative.op is Ops.CONST:
      value, condition, fill = positive, cond, float(negative.arg)
    elif negative.op is Ops.INDEX and positive.op is Ops.CONST:
      value, condition, select_true, fill = negative, cond, False, float(positive.arg)
    else: return _not_applicable()
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or len(value.src) != 2: return _not_applicable()
  if store.src[0].dtype is not value.dtype or value.dtype not in (dtypes.bool,dtypes.half,dtypes.float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "atom reformat requires matching bool, FP16, or FP32 input/output", store.src[0].op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  count, src_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  mapping = [-2] * count
  range_uops = tuple(dict.fromkeys(u for root in (store.src[0].src[1], value.src[1], condition) if root is not None
                                  for u in root.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST))
  split_axes = len({u.arg[0] for u in range_uops}) != len(range_uops)
  if out_aff is None or src_aff is None or split_axes:
    visits = math.prod(int(u.src[0].arg) for u in range_uops)
    if visits > RK_MAX_AFFINE_VISITS:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"static reformat needs {visits} coordinate visits", Ops.RANGE)
    for coordinates in product(*(range(int(u.src[0].arg)) for u in range_uops)):
      static_point = dict(zip(range_uops, coordinates))
      dst = _static_scalar(store.src[0].src[1], static_point)
      selected = True
      if condition is not None:
        if (predicate:=_static_scalar(condition, static_point)) is None:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat predicate is not static", condition.op)
        selected = bool(predicate) is select_true
      src = _static_scalar(value.src[1], static_point) if selected else -1
      if not isinstance(dst, int) or isinstance(dst, bool) or not isinstance(src, int) or isinstance(src, bool) or \
         not 0 <= dst < count or selected and not 0 <= src < src_count or mapping[dst] not in (-2, src):
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat static indexes do not cover one output", Ops.INDEX)
      mapping[dst] = src
  else:
    ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
    axes = tuple(sorted(out_aff[0].keys() | src_aff[0].keys()))
    if any(axis not in ranges for axis in axes): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "dynamic reformat range", Ops.RANGE)
    for coordinates in product(*(range(ranges[axis]) for axis in axes)):
      affine_point = dict(zip(axes, coordinates))
      dst = out_aff[1] + sum(out_aff[0].get(axis, 0)*affine_point[axis] for axis in axes)
      src = src_aff[1] + sum(src_aff[0].get(axis, 0)*affine_point[axis] for axis in axes)
      selected = True
      if condition is not None:
        if (predicate:=_static_scalar(condition, affine_point)) is None:
          return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat predicate is not static", condition.op)
        selected = bool(predicate) is select_true
      if not 0 <= dst < count or mapping[dst] != -2 or selected and not 0 <= src < src_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat does not cover one dense output", Ops.INDEX)
      mapping[dst] = src if selected else -1
  if any(source == -2 for source in mapping): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat output has holes", Ops.INDEX)
  output, source = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  if value.dtype is dtypes.bool:
    if fill == 0.0 and all(src in (-1,dst) for dst,src in enumerate(mapping)):
      mask = bytes(src >= 0 for src in mapping)
      out_ref = RKTensorRef(output,RKLayout((count,),(count,),(1,),dtypes.bool))
      src_ref = RKTensorRef(source,RKLayout((src_count,),(src_count,),(1,),dtypes.bool))
      program = RKProgram((RKDPUProgram((RKALUStage(Ops.MUL,output,source,mask,count,dtypes.bool),)),))
      return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.COALESCED_DPU,program))
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT,"bool movement needs an identity-or-zero static mask",Ops.INDEX)
  if value.dtype is dtypes.float:
    if any(src < 0 for src in mapping):
      return _unsupported(RKRejectKind.REQUIRES_REFORMAT,"FP32 movement cannot materialize static fill lanes",Ops.WHERE)
    runs:list[tuple[int,int,int]] = []
    for dst,src in enumerate(mapping):
      if runs and runs[-1][1]+runs[-1][2] == src:
        start,run_src,length = runs[-1]
        runs[-1] = (start,run_src,length+1)
      else: runs.append((dst,src,1))
    if any(dst*4%16 or src*4%16 for dst,src,_ in runs):
      return _unsupported(RKRejectKind.UNALIGNED_ROW,"FP32 reformat copy begins outside a 16-byte atom",Ops.INDEX)
    if len(runs) > RK_MAX_PROGRAM_STAGES:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,f"FP32 reformat needs {len(runs)} copy tasks",Ops.INDEX)
    copy_stages = tuple(RKCopyStage(RKArg(output.kind,output.index,dst*4),RKArg(source.kind,source.index,src*4),length,dtypes.float)
                        for dst,src,length in runs)
    out_ref,src_ref = _dense_ref(output.index,(count,),dtypes.float),_dense_ref(source.index,(src_count,),dtypes.float)
    return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.COALESCED_DPU,
      RKProgram((RKDPUProgram(copy_stages),))))
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
    return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.COALESCED_DPU,
      RKProgram((RKDPUProgram(tuple(stages)),))))
  rows = [[src] if src >= 0 else [] for src in mapping]
  # A finite fill can be appended to an aligned source scratch and selected like any other lane. Non-finite fills cannot enter CMAC:
  # zero selector weights multiplied by infinity would contaminate ordinary source rows with NaN. Select a finite padding mask instead,
  # then construct signed infinity as +/-mask/(1-mask) in DPU arithmetic.
  positive_zero = fill == 0.0 and math.copysign(1.0,fill) > 0
  if any(src < 0 for src in mapping) and not positive_zero:
    if math.isnan(fill) or fill == 0.0:
      return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,
        f"reformat fill {fill!r} has an unproven NaN or signed-zero contract",Ops.CONST)
    try: fill = struct.unpack("<e",struct.pack("<e",fill))[0]
    except OverflowError:
      return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,f"reformat fill {fill!r} overflows FP16",Ops.CONST)
    if math.isfinite(fill):
      fill_index, augmented = (src_count+7)&-8, RKArg(RKBufferKind.SCRATCH, 0)
      augmented_count, aligned_count = fill_index+1, max(32, (fill_index+32)&-32)
      finite_scratch = (RKScratch(aligned_count*2),)
      seed = RKDPUProgram((RKALUStage(Ops.ADD, augmented, 0.0, 0.0, aligned_count),
        RKALUStage(Ops.ADD, augmented, source, 0.0, src_count),
        RKALUStage(Ops.ADD, RKArg(augmented.kind,augmented.index,fill_index*2), 0.0, fill, 1)), finite_scratch)
      filled_rows = [[src if src >= 0 else fill_index] for src in mapping]
      implementation = _selector_program(output,augmented,augmented_count,filled_rows,finite_scratch,direct_capacity=aligned_count)
      if implementation is not None:
        completed = _finish_program([seed,*implementation.steps],implementation.scratch)
        cost = plan_cost(completed)
        if cost.stage_count <= RK_MAX_PROGRAM_STAGES and cost.constant_bytes <= RK_MAX_CONSTANT_BYTES:
          return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.SELECTOR_CMAC,completed))
    else:
      selected_surface, padding_mask, one = (RKArg(RKBufferKind.SCRATCH,index) for index in range(3))
      mask_scratch = (RKScratch(_cmac_tiled_output_bytes(count)),RKScratch(_cmac_tiled_output_bytes(count)),RKScratch(64))
      seed = RKDPUProgram((RKALUStage(Ops.ADD,one,0.0,0.0,32),RKALUStage(Ops.ADD,one,0.0,1.0,1)),mask_scratch)
      selected_plan = _selector_program(selected_surface,source,src_count,rows,mask_scratch)
      if selected_plan is not None:
        mask_rows = [[0] if src < 0 else [] for src in mapping]
        mask_plan = _selector_program(padding_mask,one,1,mask_rows,selected_plan.scratch,direct_capacity=32)
        if mask_plan is not None:
          numerator = _ALUExpr(Ops.MUL,(padding_mask,math.copysign(1.0,fill)))
          denominator = _ALUExpr(Ops.ADD,(_ALUExpr(Ops.MUL,(padding_mask,-1.0)),1.0))
          root = _ALUExpr(Ops.ADD,(selected_surface,_ALUExpr(Ops.FDIV,(numerator,denominator))))
          if (final:=_schedule_expr(root,output,count,mask_plan.scratch)) is not None:
            completed = _finish_program([seed,*selected_plan.steps,*mask_plan.steps,final],final.scratch)
            cost = plan_cost(completed)
            if cost.stage_count <= RK_MAX_PROGRAM_STAGES and cost.constant_bytes <= RK_MAX_CONSTANT_BYTES:
              return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.SELECTOR_CMAC,completed))
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"constant-filled reformat exceeds the native cost contract",Ops.WHERE)
  implementation = (_periodic_selector_program(output,source,src_count,rows) or
                    _selector_program(output,source,src_count,rows,()) or
                    _best_partitioned_selector_program(output,source,src_count,rows)) if 0 < src_count and 0 < count else None
  if implementation is not None:
    return _native(_legalized_reformat(out_ref,src_ref,tuple(mapping),fill,RKReformatKind.SELECTOR_CMAC,implementation))
  if 0 < src_count and 0 < count:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "static reformat selector exceeds the native cost contract", Ops.INDEX)
  return atom_reject

def lower_multi_source_reformat_result(sink:UOp) -> RKLowerResult:
  """Lower a static WHERE-selected FP16/FP32 layout from several parameter surfaces."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype not in (dtypes.half,dtypes.float):
    return _not_applicable()
  indexes = tuple(dict.fromkeys(u for u in store.src[1].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM))
  params = tuple(sorted(dict.fromkeys(index.src[0] for index in indexes),key=lambda param:param.arg.slot))
  dtype = store.src[0].dtype
  if len(params) < 2 or any(index.dtype is not dtype for index in indexes): return _not_applicable()
  output_count = int(store.src[0].src[0].src[0].arg)
  ranges = tuple(dict.fromkeys(u for root in (store.src[0].src[1],store.src[1]) for u in root.toposort()
                               if u.op is Ops.RANGE and u.src[0].op is Ops.CONST))
  visits = math.prod(int(u.src[0].arg) for u in ranges)
  if visits > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,f"multi-source reformat needs {visits} coordinate visits",Ops.RANGE)
  source_id = {param.key:index for index,param in enumerate(params)}
  source_counts = tuple(int(param.src[0].arg) for param in params)
  mapping:list[tuple[int,int]|None] = [None]*output_count
  for coordinates in product(*(range(int(u.src[0].arg)) for u in ranges)):
    point = dict(zip(ranges,coordinates))
    dst = _static_scalar(store.src[0].src[1],point)
    selected = _static_selected_index(store.src[1],point)
    src = None if selected is None else _static_scalar(selected.src[1],point)
    if not isinstance(dst,int) or isinstance(dst,bool) or selected is None or not isinstance(src,int) or isinstance(src,bool) or \
       not 0 <= dst < output_count or selected.src[0].key not in source_id:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"multi-source reformat has a dynamic selection",Ops.WHERE)
    sid = source_id[selected.src[0].key]
    if not 0 <= src < source_counts[sid] or mapping[dst] not in (None,(sid,src)):
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"multi-source reformat does not cover one dense output",Ops.INDEX)
    mapping[dst] = (sid,src)
  if any(item is None for item in mapping):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"multi-source reformat output has holes",Ops.INDEX)
  typed_mapping = cast(tuple[tuple[int,int], ...],tuple(mapping))
  out_ref = _dense_ref(store.src[0].src[0].arg.slot,(output_count,),dtype)
  source_refs = tuple(_dense_ref(param.arg.slot,(count,),dtype) for param,count in zip(params,source_counts))
  semantic = RKMultiSourceReformatPlan(out_ref,source_refs,typed_mapping)
  if dtype is dtypes.float:
    runs:list[tuple[int,int,int,int]] = []
    for dst,(sid,src) in enumerate(typed_mapping):
      if runs and runs[-1][1] == sid and runs[-1][2]+runs[-1][3] == src:
        start,run_sid,run_src,length = runs[-1]
        runs[-1] = (start,run_sid,run_src,length+1)
      else: runs.append((dst,sid,src,1))
    if any(dst*4%16 or src*4%16 for dst,_,src,_ in runs):
      return _unsupported(RKRejectKind.UNALIGNED_ROW,"FP32 multi-source copy begins outside a 16-byte atom",Ops.INDEX)
    if len(runs) > RK_MAX_PROGRAM_STAGES:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,f"FP32 multi-source reformat needs {len(runs)} copy tasks",Ops.WHERE)
    stages = tuple(RKCopyStage(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot,dst*4),
      RKArg(RKBufferKind.ARG,params[sid].arg.slot,src*4),count,dtypes.float) for dst,sid,src,count in runs)
    return _native(RKLegalizedReformat(semantic,RKReformatKind.COALESCED_DPU,RKProgram((RKDPUProgram(stages),))))
  sources = tuple(RKArg(RKBufferKind.ARG,param.arg.slot) for param in params)
  direct = _multi_source_windowed_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),sources,source_counts,typed_mapping)
  bases, packed_count = [], 0
  for source_count in source_counts:
    bases.append(packed_count)
    packed_count += (source_count+7)&-8
  scratch = (RKScratch(max(32,packed_count)*2),)
  packed = RKArg(RKBufferKind.SCRATCH,0)
  seed_stages = [RKALUStage(Ops.ADD,packed,0.0,0.0,max(32,packed_count))]
  seed_stages.extend(RKALUStage(Ops.ADD,RKArg(packed.kind,packed.index,bases[sid]*2),RKArg(RKBufferKind.ARG,param.arg.slot),0.0,count)
                     for sid,(param,count) in enumerate(zip(params,source_counts)))
  rows = [[bases[sid]+src] for sid,src in typed_mapping]
  packed_implementation = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed,packed_count,rows,scratch,
                                            direct_capacity=max(32,packed_count))
  if packed_implementation is not None:
    packed_implementation = _finish_program([RKDPUProgram(tuple(seed_stages),scratch),*packed_implementation.steps],
                                            packed_implementation.scratch)
  legal = tuple((cost,candidate) for candidate in (direct,packed_implementation) if candidate is not None and
                (cost:=plan_cost(candidate)).stage_count <= RK_MAX_PROGRAM_STAGES and cost.constant_bytes <= RK_MAX_CONSTANT_BYTES)
  if not legal:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"multi-source reformat selector exceeds plan limits",Ops.WHERE)
  _,program = min(legal,key=lambda item:(item[0].reset_count,item[0].estimated_macs,
    item[0].estimated_read_bytes+item[0].estimated_write_bytes,item[0].command_words,item[0].constant_bytes,item[0].scratch_bytes))
  return _native(RKLegalizedReformat(semantic,RKReformatKind.SELECTOR_CMAC,program))

def lower_static_selector_reformat_result(sink:UOp) -> RKLowerResult:
  """Resolve a disjoint static ADD/WHERE expression of indexes from one FP16 surface."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  stored, output_index = _strip_casts(store.src[1]), store.src[0].src[1]
  indexes = tuple(dict.fromkeys(u for u in stored.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM))
  params = tuple(dict.fromkeys(index.src[0] for index in indexes))
  if len(indexes) < 2 or len(params) != 1 or any(index.dtype is not dtypes.half for index in indexes): return _not_applicable()
  count, src_count = int(store.src[0].src[0].src[0].arg), int(params[0].src[0].arg)
  range_uops = tuple(dict.fromkeys(u for root in (output_index,stored) for u in root.toposort()
                                  if u.op is Ops.RANGE and u.src[0].op is Ops.CONST))
  visits = math.prod(int(u.src[0].arg) for u in range_uops)
  if not 0 < count <= RK_MAX_AFFINE_VISITS or visits > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"static selector reformat is {count} outputs and {visits} visits",Ops.WHERE)
  def selected_indexes(value:UOp, point:dict[UOp,int]) -> list[UOp]|None:
    value = _strip_casts(value)
    if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM: return [value]
    if value.op is Ops.CONST and float(value.arg) == 0.0: return []
    if value.op is Ops.ADD:
      parts = tuple(selected_indexes(source,point) for source in value.src)
      return None if any(part is None for part in parts) else \
        [index for part in cast(tuple[list[UOp],...],parts) for index in part]
    if value.op is Ops.WHERE:
      predicate = _static_scalar(value.src[0],point)
      return None if predicate is None else selected_indexes(value.src[1] if predicate else value.src[2],point)
    return None
  mapping = [-2]*count
  for coordinates in product(*(range(int(u.src[0].arg)) for u in range_uops)):
    point = dict(zip(range_uops,coordinates))
    dst, selected = _static_scalar(output_index,point), selected_indexes(stored,point)
    if not isinstance(dst,int) or isinstance(dst,bool) or not 0 <= dst < count or mapping[dst] != -2 or selected is None or len(selected) != 1:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static selector does not cover one dense output",Ops.WHERE)
    source_offset = _static_scalar(selected[0].src[1],point)
    if not isinstance(source_offset,int) or isinstance(source_offset,bool) or not 0 <= source_offset < src_count:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static selector source is out of bounds",Ops.INDEX)
    mapping[dst] = source_offset
  if any(source_offset < 0 for source_offset in mapping):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static selector reformat has output holes",Ops.WHERE)
  output, source_arg = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG,params[0].arg.slot)
  rows = [[index] for index in mapping]
  implementation = _periodic_selector_program(output,source_arg,src_count,rows) or _selector_program(output,source_arg,src_count,rows,())
  if implementation is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"static selector reformat exceeds the native cost contract",Ops.INDEX)
  return _native(_legalized_reformat(_dense_half_ref(output.index,(count,)),_dense_half_ref(source_arg.index,(src_count,)),tuple(mapping),
    0.0,RKReformatKind.SELECTOR_CMAC,implementation))

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
    plan = RKContractionPlan(_dense_half_ref(out_param.arg.slot, (1,n)),
      _dense_half_ref(0, (1,32), RKBufferKind.CONSTANT), _cmac_weight_ref(body.src[0].arg.slot, n, 32),
      1,n,32,(red_axis,),ones)
    return _native(legalize_contraction_plan(plan)[0])
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
  plan = RKContractionPlan(_dense_half_ref(out_param.arg.slot, (1,n)), _dense_half_ref(lhs.src[0].arg.slot, (1,32)),
                           _cmac_weight_ref(rhs.src[0].arg.slot, n, 32), 1,n,32,(red_axis,))
  return _native(legalize_contraction_plan(plan)[0])

def lower_depthwise_spatial_contract_result(sink:UOp) -> RKLowerResult:
  """Run dense NCHW depthwise convolution as independent channel-native CNA tasks."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 3 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     _strip_casts(store.src[1]).key != reduce.key: return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[1] is not None or parsed[0].dtype is not dtypes.half or
         parsed[0].src[0].op is not Ops.PARAM for parsed in operands): return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]], operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] or len(out_aff[0]) not in (3,4): return _not_applicable()
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
      if channel_axis is None or out_y_axis is None or out_x_axis is None: continue
      batch = 1 if batch_axis is None else ranges[batch_axis]
      channels, out_h, out_w = ranges[channel_axis], ranges[out_y_axis], ranges[out_x_axis]
      for kernel_y_axis,kernel_x_axis in permutations(red_axes):
        kernel_h, kernel_w = ranges[kernel_y_axis], ranges[kernel_x_axis]
        if channel_axis not in feature_aff[0]: continue
        in_w = feature_aff[0].get(kernel_y_axis,0)
        if in_w <= 0 or feature_aff[0].get(channel_axis,0)%in_w: continue
        in_h = feature_aff[0][channel_axis]//in_w
        stride_y_coeff, stride_x = feature_aff[0].get(out_y_axis,0), feature_aff[0].get(out_x_axis,0)
        if stride_y_coeff <= 0 or stride_y_coeff%in_w: continue
        stride_y = stride_y_coeff//in_w
        if not 1 <= stride_y <= 7 or not 1 <= stride_x <= 7: continue
        expected_out = {channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}
        expected_feature = {channel_axis:in_h*in_w,kernel_y_axis:in_w,kernel_x_axis:1,
                            out_y_axis:stride_y*in_w,out_x_axis:stride_x}
        if batch_axis is not None:
          expected_out[batch_axis], expected_feature[batch_axis] = channels*out_h*out_w, channels*in_h*in_w
        if out_aff[0] != expected_out or feature_aff[0] != expected_feature or \
           weight_aff[0] != {channel_axis:kernel_h*kernel_w,kernel_y_axis:kernel_w,kernel_x_axis:1}: continue
        counts = (int(feature.src[0].src[0].arg),int(weight.src[0].src[0].arg),int(store.src[0].src[0].src[0].arg))
        if counts != (batch*channels*in_h*in_w,channels*kernel_h*kernel_w,batch*channels*out_h*out_w): continue
        if out_h != (in_h-kernel_h)//stride_y+1 or out_w != (in_w-kernel_w)//stride_x+1: continue
        match = feature,weight,batch,channels,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x
        break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()
  feature,weight,batch,channels,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x = match
  if channels > 32 or max(in_h,in_w) > 32 or max(kernel_h,kernel_w) > 3 or in_w%2 or batch*channels > 32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct depthwise convolution is B={batch},C={channels},H={in_h},W={in_w},K={kernel_h}x{kernel_w}", reduce.op)

  scratch:tuple[RKScratch, ...] = ()
  input_count, input_plane = batch*channels*in_h*in_w, (in_h*in_w+7)&-8
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  input_rows = [[tile*in_h*in_w+index] if index < in_h*in_w else []
                for tile in range(batch*channels) for index in range(input_plane)]
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),input_count,input_rows,scratch,
    direct_capacity=((input_count*2+4095)&-4096)//2)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"depthwise input alignment pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_weight = RKArg(RKBufferKind.SCRATCH,len(scratch))
  weight_rows = [[channel*kernel_h*kernel_w+ky*kernel_w+kx] if out_channel == 0 and lane == 0 else []
                 for channel in range(channels) for ky in range(kernel_h) for kx in range(kernel_w)
                 for out_channel in range(2) for lane in range(8)]
  scratch += (RKScratch(len(weight_rows)*2),)
  weight_plan = _selector_program(packed_weight,RKArg(RKBufferKind.ARG,weight.src[0].arg.slot),
    channels*kernel_h*kernel_w,weight_rows,scratch,direct_capacity=channels*kernel_h*kernel_w)
  if weight_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"depthwise weight pack exceeds plan limits",Ops.INDEX)
  scratch, steps = weight_plan.scratch, [*steps,*weight_plan.steps]
  out_width_stride, out_plane = (out_h*out_w+3)&-4, 2*((out_h*out_w+3)&-4)*8
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(batch*channels*out_plane*2),)
  input_layout = RKLayout((in_h,in_w,1),(in_h,in_w,1),(in_w*2,2,2),dtypes.half,kind=RKLayoutKind.CNA_ACTIVATION)
  weight_layout = RKLayout((kernel_h,kernel_w,2,1),(kernel_h,kernel_w,2,8),
    (kernel_w*32,32,16,2),dtypes.half,padding=((0,0),(0,0),(0,0),(0,7)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0)
  output_layout = RKLayout((2,out_width_stride,8),(2,out_width_stride,8),(out_width_stride*16,16,2),dtypes.half,
                           kind=RKLayoutKind.CONV_OUTPUT)
  for b in range(batch):
    for channel in range(channels):
      tile = b*channels+channel
      steps.append(RKConvTask(RKTensorRef(RKArg(packed_output.kind,packed_output.index,tile*out_plane*2),output_layout),
        RKTensorRef(RKArg(packed_input.kind,packed_input.index,tile*input_plane*2),input_layout),
        RKTensorRef(RKArg(packed_weight.kind,packed_weight.index,channel*kernel_h*kernel_w*32),weight_layout),
        1,2,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,in_w,out_width_stride))
  output_rows = [[(b*channels+channel)*out_plane+(y*out_w+x)*8]
                 for b in range(batch) for channel in range(channels) for y in range(out_h) for x in range(out_w)]
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
    batch*channels*out_plane,output_rows,scratch,direct_capacity=batch*channels*out_plane)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"depthwise output unpack exceeds plan limits",Ops.INDEX)
  program = _finish_program([*steps,*unpack.steps],unpack.scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"direct depthwise convolution needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def lower_grouped_spatial_contract_result(sink:UOp) -> RKLowerResult:
  """Run dense grouped NCHW convolution as independent batch/group CNA tiles."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 4 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     _strip_casts(store.src[1]).key != reduce.key: return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[1] is not None or parsed[0].dtype is not dtypes.half or
         parsed[0].src[0].op is not Ops.PARAM for parsed in operands): return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]], operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] or len(out_aff[0]) not in (4,5): return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[UOp,UOp,int,int,int,int,int,int,int,int,int,int,int,int]|None = None
  for feature_parsed,weight_parsed in (parsed_operands, tuple(reversed(parsed_operands))):
    feature, weight = feature_parsed[0], weight_parsed[0]
    feature_aff, weight_aff = _affine(feature.src[1]), _affine(weight.src[1])
    if feature_aff is None or weight_aff is None or feature_aff[1] or weight_aff[1]: continue
    output_roles = ((None,*axes) for axes in permutations(out_axes)) if len(out_axes) == 4 else permutations(out_axes)
    for batch_axis,group_axis,out_channel_axis,out_y_axis,out_x_axis in output_roles:
      if group_axis is None or out_channel_axis is None or out_y_axis is None or out_x_axis is None: continue
      batch = 1 if batch_axis is None else ranges[batch_axis]
      groups, out_c, out_h, out_w = (ranges[x] for x in (group_axis,out_channel_axis,out_y_axis,out_x_axis))
      if groups <= 1: continue
      for in_channel_axis,kernel_y_axis,kernel_x_axis in permutations(red_axes):
        in_c, kernel_h, kernel_w = (ranges[x] for x in (in_channel_axis,kernel_y_axis,kernel_x_axis))
        in_w = feature_aff[0].get(kernel_y_axis,0)
        if in_w <= 0 or feature_aff[0].get(in_channel_axis,0)%in_w: continue
        in_h = feature_aff[0][in_channel_axis]//in_w
        stride_y_coeff, stride_x = feature_aff[0].get(out_y_axis,0), feature_aff[0].get(out_x_axis,0)
        if stride_y_coeff <= 0 or stride_y_coeff%in_w: continue
        stride_y = stride_y_coeff//in_w
        expected_out = {group_axis:out_c*out_h*out_w,out_channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}
        expected_feature = {group_axis:in_c*in_h*in_w,in_channel_axis:in_h*in_w,
          kernel_y_axis:in_w,kernel_x_axis:1,out_y_axis:stride_y*in_w,out_x_axis:stride_x}
        expected_weight = {group_axis:out_c*in_c*kernel_h*kernel_w,out_channel_axis:in_c*kernel_h*kernel_w,
          in_channel_axis:kernel_h*kernel_w,kernel_y_axis:kernel_w,kernel_x_axis:1}
        if batch_axis is not None:
          expected_out[batch_axis], expected_feature[batch_axis] = groups*out_c*out_h*out_w, groups*in_c*in_h*in_w
        counts = (int(feature.src[0].src[0].arg),int(weight.src[0].src[0].arg),int(store.src[0].src[0].src[0].arg))
        if out_aff[0] != expected_out or feature_aff[0] != expected_feature or weight_aff[0] != expected_weight or \
           counts != (batch*groups*in_c*in_h*in_w,groups*out_c*in_c*kernel_h*kernel_w,batch*groups*out_c*out_h*out_w): continue
        if not 1 <= stride_y <= 7 or not 1 <= stride_x <= 7 or \
           out_h != (in_h-kernel_h)//stride_y+1 or out_w != (in_w-kernel_w)//stride_x+1: continue
        match = feature,weight,batch,groups,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x
        break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()
  feature,weight,batch,groups,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x = match
  if in_c not in (1,2,3,4) or not 1 <= out_c <= 16 or max(kernel_h,kernel_w) > 3 or \
     max(in_h,in_w) > 32 or batch*groups > 32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct grouped convolution is B={batch},G={groups},IC/G={in_c},OC/G={out_c},H={in_h},W={in_w},K={kernel_h}x{kernel_w}",reduce.op)

  align_in, input_c2 = 8, in_c
  width_alignment = max(1,(16+align_in-1)//align_in)
  input_width_stride, output_width_stride = ((in_w+width_alignment-1)//width_alignment)*width_alignment, (out_h*out_w+3)&-4
  input_surface_count, output_tile_count = in_h*input_width_stride*in_c, 2*output_width_stride*8
  input_tile_count = (input_surface_count+7)&-8
  weight_surface_count = kernel_h*kernel_w*out_c*align_in
  weight_tile_count = (weight_surface_count+15)&-16
  weight_banks = max(1,(weight_surface_count*2+32767)//32768)
  if weight_banks != 1:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"grouped CNA tile needs {weight_banks} weight CBUF banks; split-K is not legalized",reduce.op)
  input_rows:list[list[int]] = []
  for b in range(batch):
    for group in range(groups):
      input_rows.extend([[((b*groups+group)*in_c+c)*in_h*in_w+y*in_w+x] if x < in_w else []
                         for y in range(in_h) for x in range(input_width_stride) for c in range(input_c2)])
      input_rows.extend([[] for _ in range(input_tile_count-input_surface_count)])
  weight_rows:list[list[int]] = []
  for group in range(groups):
    weight_rows.extend([[((group*out_c+oc)*in_c+c)*kernel_h*kernel_w+ky*kernel_w+kx] if c < in_c else []
                        for ky in range(kernel_h) for kx in range(kernel_w) for oc in range(out_c) for c in range(align_in)])
    weight_rows.extend([[] for _ in range(weight_tile_count-weight_surface_count)])
  output_rows = [[(b*groups+group)*output_tile_count+(oc//8)*output_width_stride*8+(y*out_w+x)*8+oc%8]
                 for b in range(batch) for group in range(groups) for oc in range(out_c)
                 for y in range(out_h) for x in range(out_w)]
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),
    int(feature.src[0].src[0].arg),input_rows,scratch,direct_capacity=((int(feature.src[0].src[0].arg)*2+4095)&-4096)//2,
    max_window=RK_MAX_CMAC_SELECTOR_WINDOW,max_outputs=64)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"grouped convolution input pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_weight = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(weight_rows)*2),)
  weight_plan = _selector_program(packed_weight,RKArg(RKBufferKind.ARG,weight.src[0].arg.slot),
    int(weight.src[0].src[0].arg),weight_rows,scratch,direct_capacity=((int(weight.src[0].src[0].arg)*2+4095)&-4096)//2,
    max_outputs=64)
  if weight_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"grouped convolution weight pack exceeds plan limits",Ops.INDEX)
  scratch, steps = weight_plan.scratch, [*steps,*weight_plan.steps]
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(batch*groups*output_tile_count*2),)
  input_layout = RKLayout((in_h,in_w,in_c),(in_h,input_width_stride,in_c),(input_width_stride*in_c*2,in_c*2,2),dtypes.half,
                          padding=((0,0),(0,input_width_stride-in_w),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0)
  weight_layout = RKLayout((kernel_h,kernel_w,out_c,in_c),(kernel_h,kernel_w,out_c,align_in),
    (kernel_w*out_c*align_in*2,out_c*align_in*2,align_in*2,2),dtypes.half,padding=((0,0),(0,0),(0,0),(0,align_in-in_c)),
    kind=RKLayoutKind.CNA_WEIGHT,padding_value=0)
  output_layout = RKLayout((2,output_width_stride,8),(2,output_width_stride,8),(output_width_stride*16,16,2),dtypes.half,
                           kind=RKLayoutKind.CONV_OUTPUT)
  for b in range(batch):
    for group in range(groups):
      tile = b*groups+group
      steps.append(RKConvTask(RKTensorRef(RKArg(packed_output.kind,packed_output.index,tile*output_tile_count*2),output_layout),
        RKTensorRef(RKArg(packed_input.kind,packed_input.index,tile*input_tile_count*2),input_layout),
        RKTensorRef(RKArg(packed_weight.kind,packed_weight.index,group*weight_tile_count*2),weight_layout),
        in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,input_width_stride,output_width_stride))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
    batch*groups*output_tile_count,output_rows,scratch,direct_capacity=batch*groups*output_tile_count,max_outputs=64)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"grouped convolution output unpack exceeds plan limits",Ops.INDEX)
  program = _finish_program([*steps,*unpack.steps],unpack.scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"direct grouped convolution needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def lower_nhwc_spatial_contract_result(sink:UOp) -> RKLowerResult:
  """Pack one dense NHWC/HWIO convolution, splitting output channels from CBUF pressure."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 3 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     _strip_casts(store.src[1]).key != reduce.key: return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[1] is not None or parsed[0].dtype is not dtypes.half or
         parsed[0].src[0].op is not Ops.PARAM for parsed in operands): return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]], operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] or len(out_aff[0]) != 4: return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if len(red_axes) != 2 or any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[UOp,UOp,int,int,int,int,int,int,int,int,int,int,int]|None = None
  for feature_parsed,weight_parsed in (parsed_operands, tuple(reversed(parsed_operands))):
    feature, weight = feature_parsed[0], weight_parsed[0]
    feature_aff, weight_aff = _affine(feature.src[1]), _affine(weight.src[1])
    if feature_aff is None or weight_aff is None or feature_aff[1] or weight_aff[1]: continue
    for batch_axis,out_channel_axis,out_y_axis,out_x_axis in permutations(out_axes):
      batch, out_c, out_h, out_w = (ranges[x] for x in (batch_axis,out_channel_axis,out_y_axis,out_x_axis))
      if out_aff[0] != {batch_axis:out_c*out_h*out_w,out_channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}: continue
      for kernel_y_axis,packed_xc_axis in permutations(red_axes):
        kernel_h, packed_extent = ranges[kernel_y_axis], ranges[packed_xc_axis]
        for in_c in range(1,17):
          feature_x = feature_aff[0].get(out_x_axis,0)
          if packed_extent%in_c or feature_x <= 0 or feature_x%in_c: continue
          kernel_w, stride_x = packed_extent//in_c, feature_x//in_c
          if not 5 <= in_c <= 16 or not 1 <= kernel_w <= 3 or not 1 <= stride_x <= 7 or \
             feature_aff[0].get(kernel_y_axis,0) <= 0 or feature_aff[0].get(kernel_y_axis,0)%in_c: continue
          in_w = feature_aff[0][kernel_y_axis]//in_c
          feature_batch, feature_y = feature_aff[0].get(batch_axis,0), feature_aff[0].get(out_y_axis,0)
          if in_w < 1 or feature_batch <= 0 or feature_y <= 0 or feature_batch%(in_w*in_c) or feature_y%(in_w*in_c): continue
          in_h, stride_y = feature_batch//(in_w*in_c), feature_y//(in_w*in_c)
          expected_feature = {batch_axis:in_h*in_w*in_c,kernel_y_axis:in_w*in_c,packed_xc_axis:1,
                              out_y_axis:stride_y*in_w*in_c,out_x_axis:stride_x*in_c}
          expected_weight = {kernel_y_axis:kernel_w*in_c*out_c,packed_xc_axis:out_c,out_channel_axis:1}
          counts = (int(feature.src[0].src[0].arg),int(weight.src[0].src[0].arg),int(store.src[0].src[0].src[0].arg))
          if feature_aff[0] != expected_feature or weight_aff[0] != expected_weight or \
             counts != (batch*in_h*in_w*in_c,kernel_h*kernel_w*in_c*out_c,batch*out_c*out_h*out_w): continue
          if out_h != (in_h-kernel_h)//stride_y+1 or out_w != (in_w-kernel_w)//stride_x+1: continue
          match = feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x
          break
        if match is not None: break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()
  feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x = match
  if not 5 <= in_c <= 16 or not 4 <= out_c <= 64 or max(kernel_h,kernel_w) > 3 or max(in_h,in_w) > 32 or batch > 4 or \
     max(stride_y,stride_x) > 7:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct NHWC convolution is B={batch},IC={in_c},OC={out_c},H={in_h},W={in_w},K={kernel_h}x{kernel_w},S={stride_y}x{stride_x}",reduce.op)

  align_in, input_c2 = 16, 8
  input_width_stride, output_width_stride = in_w, (out_h*out_w+3)&-4
  tiling = plan_conv_cbuf(in_h,in_w,in_c,out_c,kernel_h,kernel_w,(stride_y,stride_x),input_width_stride,output_width_stride,
                          align_in,use_nhwc=False,max_k_step=16)
  if tiling is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"NHWC convolution has no legal CBUF tile",reduce.op)
  if tiling.split in (RKConvSplit.BY_Y,RKConvSplit.BY_YK):
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,"NHWC convolution needs unlegalized CBUF Y tiling",reduce.op)
  channel_tiles = tuple(dict.fromkeys((tile.k_start,tile.out_channels) for tile in tiling.tiles))
  input_surface_count = ((in_c+7)//8)*in_h*input_width_stride*8
  input_batch_count, output_tile_count = (input_surface_count+7)&-8, 2*output_width_stride*8
  input_rows:list[list[int]] = []
  for b in range(batch):
    input_rows.extend([[b*in_h*in_w*in_c+y*in_w*in_c+x*in_c+c1*input_c2+c2] if c1*input_c2+c2 < in_c else []
                       for c1 in range((in_c+7)//8) for y in range(in_h) for x in range(input_width_stride) for c2 in range(input_c2)])
    input_rows.extend([[] for _ in range(input_batch_count-input_surface_count)])
  weight_rows:list[list[int]] = []
  weight_offsets:list[int] = []
  for start,tile_c in channel_tiles:
    weight_offsets.append(len(weight_rows))
    weight_rows.extend([[((ky*kernel_w+kx)*in_c+c)*out_c+start+oc] if c < in_c else []
                        for ky in range(kernel_h) for kx in range(kernel_w) for oc in range(tile_c) for c in range(align_in)])
  output_rows:list[list[int]] = []
  for b in range(batch):
    for tile,(start,tile_c) in enumerate(channel_tiles):
      output_rows.extend([[(b*len(channel_tiles)+tile)*output_tile_count+(oc//8)*output_width_stride*8+(y*out_w+x)*8+oc%8]
                          for oc in range(tile_c) for y in range(out_h) for x in range(out_w)])
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),
    int(feature.src[0].src[0].arg),input_rows,scratch,direct_capacity=((int(feature.src[0].src[0].arg)*2+4095)&-4096)//2,
    max_window=RK_MAX_AFFINE_WINDOW,max_outputs=64)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"NHWC convolution input pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_weight = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(weight_rows)*2),)
  weight_plan = _selector_program(packed_weight,RKArg(RKBufferKind.ARG,weight.src[0].arg.slot),
    int(weight.src[0].src[0].arg),weight_rows,scratch,direct_capacity=((int(weight.src[0].src[0].arg)*2+4095)&-4096)//2,
    max_window=RK_MAX_AFFINE_WINDOW,max_outputs=64)
  if weight_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"NHWC convolution weight pack exceeds plan limits",Ops.INDEX)
  scratch, steps = weight_plan.scratch, [*steps,*weight_plan.steps]
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(batch*len(channel_tiles)*output_tile_count*2),)
  input_layout = RKLayout(((in_c+7)//8,in_h,in_w,8),((in_c+7)//8,in_h,input_width_stride,8),
    (in_h*input_width_stride*16,input_width_stride*16,16,2),dtypes.half,
    padding=((0,0),(0,0),(0,input_width_stride-in_w),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0)
  output_layout = RKLayout((2,output_width_stride,8),(2,output_width_stride,8),(output_width_stride*16,16,2),dtypes.half,
                           kind=RKLayoutKind.CONV_OUTPUT)
  for b in range(batch):
    for tile,(start,tile_c) in enumerate(channel_tiles):
      weight_layout = RKLayout((kernel_h,kernel_w,tile_c,in_c),(kernel_h,kernel_w,tile_c,align_in),
        (kernel_w*tile_c*align_in*2,tile_c*align_in*2,align_in*2,2),dtypes.half,
        padding=((0,0),(0,0),(0,0),(0,align_in-in_c)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0)
      steps.append(RKConvTask(
        RKTensorRef(RKArg(packed_output.kind,packed_output.index,(b*len(channel_tiles)+tile)*output_tile_count*2),output_layout),
        RKTensorRef(RKArg(packed_input.kind,packed_input.index,b*input_batch_count*2),input_layout),
        RKTensorRef(RKArg(packed_weight.kind,packed_weight.index,weight_offsets[tile]*2),weight_layout),
        in_c,tile_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,input_width_stride,output_width_stride))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
    batch*len(channel_tiles)*output_tile_count,output_rows,scratch,direct_capacity=batch*len(channel_tiles)*output_tile_count,
    max_window=RK_MAX_AFFINE_WINDOW,max_outputs=64)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"NHWC convolution output unpack exceeds plan limits",Ops.INDEX)
  program = _finish_program([*steps,*unpack.steps],unpack.scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"direct NHWC convolution needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def lower_spatial_contract_result(sink:UOp) -> RKLowerResult:
  """Recognize a proven dense NCHW/OIHW convolution and pack every surface on the NPU."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) not in (2,4) or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or \
     _strip_casts(store.src[1]).key != reduce.key:
    return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[0].dtype is not dtypes.half or
         parsed[0].src[0].op is not Ops.PARAM for parsed in operands): return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]], operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] != 0 or len(out_aff[0]) not in (2,3,4): return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[UOp,UOp,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int]|None = None
  if len(out_axes) == 4 and len(red_axes) == 1:
    point_in_channel_axis = red_axes[0]
    for feature_parsed,weight_parsed in (parsed_operands, tuple(reversed(parsed_operands))):
      feature, weight = feature_parsed[0], weight_parsed[0]
      feature_aff, weight_aff = _conditional_index_affine(feature), _affine(weight.src[1])
      if weight_parsed[1] is not None or feature_aff is None or weight_aff is None or weight_aff[1] or feature_aff[1] > 0: continue
      for point_batch_axis,point_channel_axis,point_y_axis,point_x_axis in permutations(out_axes):
        batch,in_c,out_c,out_h,out_w = (ranges[x] for x in
          (point_batch_axis,point_in_channel_axis,point_channel_axis,point_y_axis,point_x_axis))
        if out_aff[0] != {point_batch_axis:out_c*out_h*out_w,point_channel_axis:out_h*out_w,
                          point_y_axis:out_w,point_x_axis:1} or \
           weight_aff[0] != {point_channel_axis:in_c,point_in_channel_axis:1}: continue
        input_plane, row_step, stride_x = (feature_aff[0].get(axis,0) for axis in
          (point_in_channel_axis,point_y_axis,point_x_axis))
        if min(input_plane,row_step,stride_x) <= 0 or stride_x > 7: continue
        for in_w in range(1,33):
          if input_plane%in_w or row_step%in_w: continue
          in_h,stride_y = input_plane//in_w,row_step//in_w
          if not 1 <= in_h <= 32 or not 1 <= stride_y <= 7: continue
          expected_feature = {point_batch_axis:in_c*in_h*in_w,point_in_channel_axis:in_h*in_w,
                              point_y_axis:stride_y*in_w,point_x_axis:stride_x}
          padding = _conv_zero_padding(feature_aff,feature_parsed[1],feature_parsed[2],ranges,
            (-1,-1,point_y_axis,point_x_axis),in_h,in_w,1,1,out_h,out_w,stride_y,stride_x)
          if feature_aff[0] != expected_feature or padding is None: continue
          feature_count,weight_count,output_count = (int(x.src[0].src[0].arg) for x in (feature,weight,store.src[0]))
          if (feature_count,weight_count,output_count) != (batch*in_c*in_h*in_w,out_c*in_c,batch*out_c*out_h*out_w): continue
          match = (feature,weight,batch,in_c,out_c,in_h,in_w,1,1,out_h,out_w,stride_y,stride_x,1,1,output_count,*padding)
          break
        if match is not None: break
      if match is not None: break
  if len(out_axes) == 2 and len(red_axes) == 1:
    point_reduction_axis = red_axes[0]
    for feature_parsed,weight_parsed in (parsed_operands, tuple(reversed(parsed_operands))):
      feature, weight = feature_parsed[0], weight_parsed[0]
      feature_aff, weight_aff = _conditional_index_affine(feature), _affine(weight.src[1])
      if feature_parsed[1] is not None or weight_parsed[1] is not None or feature_aff is None or weight_aff is None or \
         feature_aff[1] or weight_aff[1]: continue
      for point_channel_axis,point_spatial_axis in permutations(out_axes):
        in_c, out_c, spatial = ranges[point_reduction_axis], ranges[point_channel_axis], ranges[point_spatial_axis]
        if out_aff[0] != {point_channel_axis:spatial,point_spatial_axis:1} or \
           feature_aff[0] != {point_reduction_axis:spatial,point_spatial_axis:1} or \
           weight_aff[0] != {point_channel_axis:in_c,point_reduction_axis:1}: continue
        shapes = [(h,spatial//h) for h in range(1,min(32,spatial)+1) if spatial%h == 0 and spatial//h <= 32]
        if not shapes: continue
        in_h,in_w = min(shapes,key=lambda shape:abs(shape[0]-shape[1]))
        feature_count, weight_count, output_count = (int(x.src[0].src[0].arg) for x in (feature,weight,store.src[0]))
        if (feature_count,weight_count,output_count) != (in_c*spatial,out_c*in_c,out_c*spatial): continue
        match = (feature,weight,1,in_c,out_c,in_h,in_w,1,1,in_h,in_w,1,1,1,1,output_count,0,0,0,0)
        break
      if match is not None: break
  for feature_parsed,weight_parsed in (() if match is not None or len(red_axes) != 3 or len(out_axes) not in (3,4) else
                                       (parsed_operands, tuple(reversed(parsed_operands)))):
    feature, weight = feature_parsed[0], weight_parsed[0]
    feature_aff, weight_aff = _conditional_index_affine(feature), _affine(weight.src[1])
    if weight_parsed[1] is not None or feature_aff is None or weight_aff is None or weight_aff[1] or feature_aff[1] > 0: continue
    output_roles = ((None,*axes) for axes in permutations(out_axes)) if len(out_axes) == 3 else permutations(out_axes)
    for batch_axis,channel_axis,out_y_axis,out_x_axis in output_roles:
      if channel_axis is None or out_y_axis is None or out_x_axis is None: continue
      batch = 1 if batch_axis is None else ranges[batch_axis]
      out_c, out_h, out_w = (ranges[x] for x in (channel_axis,out_y_axis,out_x_axis))
      for in_channel_axis,kernel_y_axis,kernel_x_axis in permutations(red_axes):
        in_c, kernel_h, kernel_w = (ranges[x] for x in (in_channel_axis,kernel_y_axis,kernel_x_axis))
        input_plane, kernel_y_coeff, dilation_x, stride_y_coeff, stride_x = (feature_aff[0].get(axis,0) for axis in
          (in_channel_axis,kernel_y_axis,kernel_x_axis,out_y_axis,out_x_axis))
        if min(input_plane,kernel_y_coeff,dilation_x,stride_y_coeff,stride_x) <= 0: continue
        for in_w in range(1,33):
          if input_plane%in_w or kernel_y_coeff%in_w or stride_y_coeff%in_w: continue
          in_h, dilation_y, stride_y = input_plane//in_w, kernel_y_coeff//in_w, stride_y_coeff//in_w
          if not 1 <= in_h <= 32 or not 1 <= stride_y <= 7 or not 1 <= stride_x <= 7 or \
             not 1 <= dilation_y <= 32 or not 1 <= dilation_x <= 32: continue
          expected_out = {channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}
          expected_feature = {in_channel_axis:in_h*in_w,kernel_y_axis:dilation_y*in_w,kernel_x_axis:dilation_x,
                              out_y_axis:stride_y*in_w,out_x_axis:stride_x}
          if batch_axis is not None:
            expected_out[batch_axis], expected_feature[batch_axis] = out_c*out_h*out_w, in_c*in_h*in_w
          if out_aff[0] != expected_out or feature_aff[0] != expected_feature: continue
          if weight_aff[0] != {channel_axis:in_c*kernel_h*kernel_w, in_channel_axis:kernel_h*kernel_w,
                              kernel_y_axis:kernel_w, kernel_x_axis:1}: continue
          padding = _conv_zero_padding(feature_aff,feature_parsed[1],feature_parsed[2],ranges,
            (kernel_y_axis,kernel_x_axis,out_y_axis,out_x_axis),in_h,in_w,kernel_h,kernel_w,out_h,out_w,
            stride_y,stride_x,dilation_y,dilation_x)
          if padding is None: continue
          feature_count, weight_count, output_count = (int(x.src[0].src[0].arg) for x in (feature,weight,store.src[0]))
          if (feature_count,weight_count,output_count) != (batch*in_c*in_h*in_w,out_c*in_c*kernel_h*kernel_w,
                                                          batch*out_c*out_h*out_w): continue
          match = (feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,
                   dilation_y,dilation_x,output_count,*padding)
          break
        if match is not None: break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()
  feature,weight,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,dilation_y,dilation_x,\
    output_count,pt,pb,pl,pr = match
  if in_c not in (1,2,3,4,16) or not 1 <= out_c <= 16 or not 1 <= kernel_h <= 3 or not 1 <= kernel_w <= 3 or \
     max(in_h,in_w) > 32 or batch > 4:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct spatial convolution is B={batch},IC={in_c},OC={out_c},H={in_h},W={in_w},K={kernel_h}x{kernel_w},"
      f"S={stride_y}x{stride_x},D={dilation_y}x{dilation_x}",
      reduce.op)

  align_in, input_c2 = (8,in_c) if in_c <= 4 else (16,8)
  width_alignment = max(1,(16+align_in-1)//align_in)
  input_width_stride, output_width_stride = ((in_w+width_alignment-1)//width_alignment)*width_alignment, (out_h*out_w+3)&-4
  input_surface_count = in_h*input_width_stride*in_c
  input_batch_count, output_batch_count = (input_surface_count+7)&-8, 2*output_width_stride*8
  input_rows:list[list[int]] = []
  if in_c <= 4:
    for b in range(batch):
      input_rows.extend([[b*in_c*in_h*in_w+c*in_h*in_w+y*in_w+x] if x < in_w else []
                         for y in range(in_h) for x in range(input_width_stride) for c in range(input_c2)])
      input_rows.extend([[] for _ in range(input_batch_count-input_surface_count)])
  else:
    for b in range(batch):
      input_rows.extend([[b*in_c*in_h*in_w+(c1*input_c2+c2)*in_h*in_w+y*in_w+x] if x < in_w else []
                         for c1 in range(in_c//input_c2) for y in range(in_h)
                         for x in range(input_width_stride) for c2 in range(input_c2)])
      input_rows.extend([[] for _ in range(input_batch_count-input_surface_count)])
  weight_rows = [[oc*in_c*kernel_h*kernel_w+c*kernel_h*kernel_w+ky*kernel_w+kx] if c < in_c else []
                 for ky in range(kernel_h) for kx in range(kernel_w) for oc in range(out_c) for c in range(align_in)]
  output_rows = [[b*output_batch_count+(oc//8)*output_width_stride*8+(y*out_w+x)*8+oc%8]
                 for b in range(batch) for oc in range(out_c) for y in range(out_h) for x in range(out_w)]
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  selector_outputs = 128 if kernel_h == kernel_w == 1 and not any((pt,pb,pl,pr)) else 64
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),
                                 int(feature.src[0].src[0].arg),input_rows,scratch,
                                 direct_capacity=((int(feature.src[0].src[0].arg)*2+4095)&-4096)//2,
                                 max_window=RK_MAX_CMAC_SELECTOR_WINDOW,max_outputs=selector_outputs)
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
  if in_c <= 4:
    input_layout = RKLayout((in_h,in_w,in_c),(in_h,input_width_stride,in_c),(input_width_stride*in_c*2,in_c*2,2),dtypes.half,
                            padding=((0,0),(0,input_width_stride-in_w),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0)
  else:
    input_layout = RKLayout((in_c//input_c2,in_h,in_w,input_c2),(in_c//input_c2,in_h,input_width_stride,input_c2),
      (in_h*input_width_stride*input_c2*2,input_width_stride*input_c2*2,input_c2*2,2),dtypes.half,
      padding=((0,0),(0,0),(0,input_width_stride-in_w),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0)
  weight_layout = RKLayout((kernel_h,kernel_w,out_c,in_c),(kernel_h,kernel_w,out_c,align_in),
                           (kernel_w*out_c*align_in*2,out_c*align_in*2,align_in*2,2),dtypes.half,
                           padding=((0,0),(0,0),(0,0),(0,align_in-in_c)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0)
  output_layout = RKLayout((2,output_width_stride,8),(2,output_width_stride,8),(output_width_stride*16,16,2),dtypes.half,
                           kind=RKLayoutKind.CONV_OUTPUT)
  tiling = plan_conv_cbuf(in_h,in_w,in_c,out_c,kernel_h,kernel_w,(stride_y,stride_x),input_width_stride,output_width_stride,
                          align_in,use_nhwc=in_c <= 4,max_k_step=out_c,padding=(pt,pb,pl,pr),
                          dilation=(dilation_y,dilation_x))
  if tiling is None or tiling.split is not RKConvSplit.NONE:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,"direct convolution needs an unlegalized CBUF split",reduce.op)
  for b in range(batch):
    conv_plan = RKConvPlan(RKTensorRef(RKArg(packed_output.kind,packed_output.index,b*output_batch_count*2),output_layout),
      RKTensorRef(RKArg(packed_input.kind,packed_input.index,b*input_batch_count*2),input_layout),RKTensorRef(packed_weight,weight_layout),
      in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x,input_width_stride,output_width_stride,tiling,
      pt,pb,pl,pr,dilation_y,dilation_x)
    steps.extend(legalize_conv_plan(conv_plan))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
                             batch*output_batch_count,output_rows,scratch,direct_capacity=batch*output_batch_count,
                             max_outputs=selector_outputs)
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
  if not 1 <= k <= 96 or lhs_count > 8192 or rhs_count > 8192 or output_count*k > RK_MAX_TILED_CONTRACT_VISITS:
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
  lhs_base = lhs_rows[0][0] if lhs_rows and lhs_rows[0] else -1
  lhs_capacity = ((lhs_count*2+4095)&-4096)//2
  direct_lhs = lhs_base >= 0 and all(row == tuple(range(lhs_base+index*align_in,lhs_base+index*align_in+k))
                                     for index,row in enumerate(lhs_rows)) and lhs_base+m*align_in <= lhs_capacity
  channel_ids = {column:index for index,column in enumerate(rhs_columns)}
  compact_output = m == 1 and all(out_index == channel_ids[rhs_key] for out_index,_,rhs_key in records)
  selector_floor = (0 if direct_lhs else (lhs_values+31)//32) + (rhs_values+31)//32 + \
    (1 if compact_output else (output_count+31)//32) + \
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
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  scratch:tuple[RKScratch, ...] = ()
  if direct_lhs:
    # CMAC may read the allocator's page-rounded tail, but padded K lanes have zero weights and cannot affect the result.
    a_arg = RKArg(RKBufferKind.ARG,lhs.src[0].arg.slot,lhs_base*2)
  else:
    a_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(len(a_selector))),)
    packed_a = _selector_program(a_arg, RKArg(RKBufferKind.ARG, lhs.src[0].arg.slot), lhs_count, a_selector, scratch, max_outputs=32)
    if packed_a is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "tiled CMAC lhs selector exceeds plan limits", reduce.op)
    steps.extend(packed_a.steps)
    scratch = packed_a.scratch
  b_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(_cmac_tiled_output_bytes(len(b_selector))),)
  # Rockchip GEM allocations are page-rounded. Zero-weight selector lanes may read that physical tail without changing semantics.
  rhs_capacity = ((rhs_count*2+4095)&-4096)//2
  packed_b = _selector_program(b_arg, RKArg(RKBufferKind.ARG, rhs.src[0].arg.slot), rhs_count, b_selector, scratch,
                               rhs_capacity, RK_MAX_TILED_CMAC_SELECTOR_WINDOW, 32)
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
                                  int(fused_epilogue[0].src[0].src[0].arg), bias_rows, scratch, max_outputs=32)
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
    steps.append(RKCMACTask(RKTensorRef(RKArg(cmac_out.kind, cmac_out.index, row_start*out_physical*2), out_layout),
      RKTensorRef(RKArg(a_arg.kind, a_arg.index, row_start*align_in*2), lhs_layout), RKTensorRef(b_arg, rhs_layout), red_axes[0],
      epilogue=contract_epilogue, compact_output=compact_output))
  if compact_output:
    steps.append(RKDPUProgram((RKALUStage(Ops.ADD,reduced,cmac_out,0.0,output_count),),scratch))
  else:
    unpack:list[list[int]] = [[] for _ in range(output_count)]
    for out_index,row,rhs_key in records:
      channel = channel_ids[rhs_key]
      unpack[out_index] = [lhs_rows.index(row)*align_out*2+(channel//16)*32+channel%16]
    dense = _selector_program(reduced, cmac_out, m*align_out*2, unpack, scratch, max_outputs=32)
    if dense is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "tiled CMAC output selector exceeds plan limits", reduce.op)
    steps.extend(dense.steps)
    scratch = dense.scratch
  stage_count = sum(len(step.stages) if isinstance(step, RKDPUProgram) else 1 for step in steps)
  constant_bytes = sum(map(len, {step.constants for step in steps if isinstance(step, RKCMACTask)}))
  if stage_count > RK_MAX_PROGRAM_STAGES or constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"tiled CMAC plan needs {stage_count} stages and {constant_bytes} constant bytes", reduce.op)
  program = _finish_program(steps, scratch)
  if remaining_epilogue: return _finish_reduction_epilogue(program, stored, reduce, store.src[0].src[1], output, reduced,
                                                            output_count, out_axes, ranges)
  return _native(program)

def lower_contract(sink:UOp) -> RKCMACTask|None:
  """Compatibility helper for compiler probes; production lowering consumes `lower_contract_result`."""
  return cast(RKCMACTask|None, lower_contract_result(sink).plan)

_PPU_BAD_SPLITS = frozenset({(3,6),(6,3),(12,12)})
def _pool_hw_shape(extent:int) -> tuple[int, int]|None:
  """Return a characterized PPU global-pool surface; reject known-bad RK3588 geometries."""
  return min(((height, extent//height) for height in range(2, min(16, extent)+1) if extent%height == 0 and
              2 <= extent//height <= 16 and height != 9 and extent//height != 9 and
              (height,extent//height) not in _PPU_BAD_SPLITS), key=lambda shape:abs(shape[0]-shape[1]), default=None)

def lower_reduce_result(sink:UOp) -> RKLowerResult:
  """Recognize global FP16 MAX over the spatial dimensions of a dense HWC surface."""
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
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "PPU reduction needs zero-based affine HWC surfaces", Ops.INDEX)
  out_axis, red_axis = next(iter(out_aff[0])), red.arg[0]
  channels, extent = int(store.src[0].src[0].src[0].arg), int(red.src[0].arg)
  hw_shape = _pool_hw_shape(extent)
  if not 2 <= channels <= 8 or out_aff[0] != {out_axis:1} or src_aff[0] != {red_axis:channels, out_axis:1}:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "PPU global MAX requires dense HWC indexing", Ops.INDEX)
  if hw_shape is None:
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, f"PPU global MAX spatial extent {extent} needs tiling or reformat", reduce.op)
  if int(value.src[0].src[0].arg) != extent*channels:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "PPU input buffer extent does not match HWC surface", Ops.INDEX)
  height, width = hw_shape
  out = RKTensorRef(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot),
                    RKLayout((1,1,channels), (1,1,channels), (channels*2,channels*2,2), dtypes.half,kind=RKLayoutKind.PPU_HWC))
  src = RKTensorRef(RKArg(RKBufferKind.ARG, value.src[0].arg.slot),
                    RKLayout((height,width,channels), (height,width,channels), (width*channels*2,channels*2,2), dtypes.half,
                             kind=RKLayoutKind.PPU_HWC))
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
  contracts:list[RKCMACTask] = []
  for start in range(0, 64, 16):
    out_layout = RKLayout((1,16), (1,32), (64,2), dtypes.half, padding=((0,0),(0,16)))
    contracts.append(RKCMACTask(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
      _dense_half_ref(packed.index, (1,32), RKBufferKind.SCRATCH), _cmac_weight_ref(0, 16, 32, RKBufferKind.CONSTANT, 32),
      red.arg[0], _cmac_selection_payload(rows[start:start+16], 32, 32, 1.0)))
  pooled = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch.append(RKScratch(16))
  scratch_tuple = tuple(scratch)
  reduce_plan = RKReduce(RKTensorRef(pooled, RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half,
    kind=RKLayoutKind.PPU_HWC)), RKTensorRef(hwc, RKLayout((2,4,8), (2,4,8), (64,16,2), dtypes.half,
    kind=RKLayoutKind.PPU_HWC)), Ops.MAX, red.arg[0])
  final = RKDPUProgram((RKALUStage(Ops.ADD if output_scale == 1.0 else Ops.MUL, output, pooled,
                                   0.0 if output_scale == 1.0 else output_scale, 1),), scratch_tuple)
  return _native(RKProgram((RKDPUProgram(tuple(stages), scratch_tuple), *contracts, reduce_plan, final), scratch_tuple))

def _scalar_affine_max_program(output:RKArg, source:RKArg, selectors:list[list[int]], pool_extent:int,
                               pool_shape:tuple[int, int], reduce_axis:int, scratch:tuple[RKScratch, ...]=()) -> RKProgram|None:
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
  packed, hwc, atoms = (RKArg(RKBufferKind.SCRATCH, len(scratch)+index) for index in range(3))
  scratch += (RKScratch(max(spec[2] for spec in specs)*2), RKScratch(hwc_elements*2), RKScratch(len(selectors)*16))
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
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
      steps.append(RKCMACTask(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
        _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH),
        _cmac_weight_ref(0, count, align_in, RKBufferKind.CONSTANT, 32), reduce_axis, payload))
    out = RKTensorRef(RKArg(atoms.kind, atoms.index, output_index*16), RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half,
      kind=RKLayoutKind.PPU_HWC))
    src = RKTensorRef(hwc, RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half,
      kind=RKLayoutKind.PPU_HWC))
    steps.append(RKReduce(out, src, Ops.MAX, reduce_axis))
  gather = _sparse_cmac_pipeline(output, atoms, len(selectors)*8, [[index*8] for index in range(len(selectors))], scratch=scratch)
  steps.extend(gather.steps)
  scratch = gather.scratch
  payloads.update(step.constants for step in gather.steps if isinstance(step, RKCMACTask))
  stage_count = sum(len(step.stages) if isinstance(step, RKDPUProgram) else 1 for step in steps)
  if stage_count > RK_MAX_PROGRAM_STAGES or sum(map(len,payloads)) > RK_MAX_CONSTANT_BYTES: return None
  return _finish_program(steps, scratch)

def lower_sliding_max_result(sink:UOp) -> RKLowerResult:
  """Pack dense planar valid-pool surfaces once, then run one sliding PPU task per HWC8 group."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) != 3 or _strip_casts(store.src[1]).key != reduce.key or \
     store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half: return _not_applicable()
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None or out_aff[1] or src_aff[1] or len(out_aff[0]) != 3:
    return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if len(red_axes) != 2 or any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[int,int,int,int,int,int,int,int,int]|None = None
  for plane_axis,out_y_axis,out_x_axis in permutations(out_axes):
    planes, out_h, out_w = (ranges[x] for x in (plane_axis,out_y_axis,out_x_axis))
    for kernel_y_axis,kernel_x_axis in permutations(red_axes):
      kernel_h, kernel_w = ranges[kernel_y_axis], ranges[kernel_x_axis]
      in_w = src_aff[0].get(kernel_y_axis,0)
      if in_w <= 0 or src_aff[0].get(plane_axis,0)%in_w: continue
      in_h = src_aff[0][plane_axis]//in_w
      stride_y_coeff, stride_x = src_aff[0].get(out_y_axis,0), src_aff[0].get(out_x_axis,0)
      if stride_y_coeff <= 0 or stride_y_coeff%in_w: continue
      stride_y = stride_y_coeff//in_w
      if out_aff[0] != {plane_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1} or \
         src_aff[0] != {plane_axis:in_h*in_w,out_y_axis:stride_y*in_w,out_x_axis:stride_x,
                        kernel_y_axis:in_w,kernel_x_axis:1}: continue
      input_count, output_count = int(value.src[0].src[0].arg), int(store.src[0].src[0].src[0].arg)
      if (input_count,output_count) != (planes*in_h*in_w,planes*out_h*out_w) or \
         out_h != (in_h-kernel_h)//stride_y+1 or out_w != (in_w-kernel_w)//stride_x+1: continue
      match = planes,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x
      break
    if match is not None: break
  if match is None: return _not_applicable()
  planes,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x = match
  if not 1 <= planes <= 32 or not 2 <= max(kernel_h,kernel_w) <= 8 or min(kernel_h,kernel_w) < 2 or \
     max(in_h,in_w) > 256 or max(stride_y,stride_x) > 8:
    return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION,
      f"sliding PPU MAX is C={planes},H={in_h},W={in_w},K={kernel_h}x{kernel_w},S={stride_y}x{stride_x}",reduce.op)

  groups, input_tile_count, output_tile_count = (planes+7)//8, in_h*in_w*8, out_h*out_w*8
  input_rows = [[(group*8+channel)*in_h*in_w+y*in_w+x] if group*8+channel < planes else []
                for group in range(groups) for y in range(in_h) for x in range(in_w) for channel in range(8)]
  output_rows = [[(plane//8)*output_tile_count+(y*out_w+x)*8+plane%8]
                 for plane in range(planes) for y in range(out_h) for x in range(out_w)]
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,value.src[0].arg.slot),planes*in_h*in_w,input_rows,scratch,
    direct_capacity=((planes*in_h*in_w*2+4095)&-4096)//2,max_window=RK_MAX_CMAC_SELECTOR_WINDOW,max_outputs=128)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"sliding PPU input pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(groups*output_tile_count*2),)
  input_layout = RKLayout((in_h,in_w,8),(in_h,in_w,8),(in_w*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC)
  output_layout = RKLayout((out_h,out_w,8),(out_h,out_w,8),(out_w*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC)
  for group in range(groups):
    steps.append(RKPool(RKTensorRef(RKArg(packed_output.kind,packed_output.index,group*output_tile_count*2),output_layout),
      RKTensorRef(RKArg(packed_input.kind,packed_input.index,group*input_tile_count*2),input_layout),Ops.MAX,red_axes[0],
      kernel_h,kernel_w,stride_y,stride_x))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
    groups*output_tile_count,output_rows,scratch,direct_capacity=groups*output_tile_count,max_outputs=128)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"sliding PPU output unpack exceeds plan limits",Ops.INDEX)
  program = _finish_program([*steps,*unpack.steps],unpack.scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"sliding PPU MAX needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

def _ppu_width_factors(width:int) -> tuple[int, ...]|None:
  """Factor one dense HWC8 width into characterized 1xK, stride-K PPU tasks."""
  factors:list[int] = []
  while width > 1:
    factor = next((candidate for candidate in range(min(8,width),1,-1) if width%candidate == 0),None)
    if factor is None: return None
    factors.append(factor)
    width //= factor
  return tuple(factors)

def _dense_axis_coefficients(axes:tuple[int, ...], ranges:dict[int, int]) -> dict[int, int]:
  coefficients, stride = {}, 1
  for axis in reversed(axes): coefficients[axis], stride = stride, stride*ranges[axis]
  return coefficients

def lower_dense_row_max_result(sink:UOp) -> RKLowerResult:
  """Reduce contiguous FP16 rows through direct HWC8 width pooling and an eight-channel DPU fold."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) < 2 or _strip_casts(store.src[1]).key != reduce.key:
    return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _not_applicable()
  value = _strip_casts(reduce.src[0])
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _not_applicable()
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None or out_aff[1] or src_aff[1]: return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(sorted(out_aff[0])), tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  if len(red_axes) != len(reduce.src)-1 or not out_axes or set(out_axes)&set(red_axes) or \
     set(src_aff[0]) != set(out_axes)|set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _not_applicable()
  output_count, reduction_count = math.prod(ranges[axis] for axis in out_axes), math.prod(ranges[axis] for axis in red_axes)
  input_count = int(value.src[0].src[0].arg)
  if not 2 <= output_count <= 256 or input_count != output_count*reduction_count or reduction_count%8:
    return _not_applicable()
  factors = _ppu_width_factors(reduction_count//8)
  if factors is None or not factors or out_aff[0] != _dense_axis_coefficients(out_axes,ranges) or \
     src_aff[0] != _dense_axis_coefficients((*out_axes,*red_axes),ranges): return _not_applicable()

  scratch:tuple[RKScratch, ...] = ()
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  current = RKArg(RKBufferKind.ARG,value.src[0].arg.slot)
  width = reduction_count//8
  for factor in factors:
    next_width = width//factor
    target = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(output_count*next_width*16),)
    src_layout = RKLayout((output_count,width,8),(output_count,width,8),(width*16,16,2),dtypes.half,kind=RKLayoutKind.PPU_HWC)
    out_layout = RKLayout((output_count,next_width,8),(output_count,next_width,8),(next_width*16,16,2),dtypes.half,
                          kind=RKLayoutKind.PPU_HWC)
    steps.append(RKPool(RKTensorRef(target,out_layout),RKTensorRef(current,src_layout),Ops.MAX,red_axes[0],1,factor,1,factor))
    current, width = target, next_width

  planar = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(output_count*16),)
  transpose = _selector_program(planar,current,output_count*8,
    [[row*8+channel] for channel in range(8) for row in range(output_count)],scratch,
    direct_capacity=output_count*8,max_outputs=64)
  if transpose is None:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"dense row MAX channel transpose exceeds plan limits",reduce.op)
  scratch, steps = transpose.scratch, [*steps,*transpose.steps]
  level = [RKArg(planar.kind,planar.index,(channel*output_count)*2) for channel in range(8)]
  dpu_stages:list[RKALUStage] = []
  while len(level) > 1:
    last = len(level) == 2
    target = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot) if last else RKArg(RKBufferKind.SCRATCH,len(scratch))
    if not last: scratch += (RKScratch((len(level)//2)*output_count*2),)
    next_level:list[RKArg] = []
    for index in range(0,len(level),2):
      dst = RKArg(target.kind,target.index,target.addend+(index//2)*output_count*2)
      dpu_stages.append(RKALUStage(Ops.MAX,dst,level[index],level[index+1],output_count))
      next_level.append(dst)
    level = next_level
  steps.append(RKDPUProgram(tuple(dpu_stages),scratch))
  program = _finish_program(steps,scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"dense row MAX needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
  return _native(program)

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
  input_count, source = int(value_index.src[0].src[0].arg), RKArg(RKBufferKind.ARG,value_index.src[0].arg.slot)
  source_prepare:RKDPUProgram|None = None
  # A pointwise sign flip is exact in FP16 and is needed before MAX for MIN/softmin-style reductions. Keep this deliberately
  # narrower than arbitrary expression materialization so padding WHEREs continue through the proven selector path below.
  if value.key != value_index.key and value.op is Ops.MUL:
    constant, candidate = (value.src[0],value.src[1]) if value.src[0].op is Ops.CONST else (value.src[1],value.src[0])
    if constant.op is Ops.CONST and float(constant.arg) == -1.0 and _strip_casts(candidate).key == value_index.key:
      source = RKArg(RKBufferKind.SCRATCH,0)
      source_prepare = RKDPUProgram((RKALUStage(Ops.MUL,source,RKArg(RKBufferKind.ARG,value_index.src[0].arg.slot),-1.0,input_count),),
                                    (RKScratch(((input_count+7)//8)*16),))
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value_index.src[1])
  if src_aff is None and value_index.src[1].op is Ops.WHERE:
    affine_branches = tuple(x for branch in value_index.src[1].src[1:] if branch.arg is not Invalid and (x:=_affine(branch)) is not None)
    if len(affine_branches) == 1: src_aff = affine_branches[0]
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count = int(store.src[0].src[0].src[0].arg)
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
      selected = True if value.key == value_index.key or source_prepare is not None else _static_index_selected(value, value_index, point)
      if selected is None:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX predicate is not static", Ops.WHERE)
      src_index = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes)) if selected else -1
      if selected and not 0 <= src_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX input index is out of bounds", Ops.INDEX)
      selectors[out_index].append(src_index)
  if seen != set(range(output_count)) or any(len(row) != reduction_count for row in selectors):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX output has holes", Ops.INDEX)

  surface_count, whole_surface = pool_extent*8, input_count <= 512 and output_count <= 128
  def group_input_width(rows:list[list[int]]) -> int:
    selected = [index for row in rows for index in row if index >= 0]
    return max(32,(max(selected)+1-(min(selected)&-8)+31)&-32) + (32 if any(index < 0 for row in rows for index in row) else 0)
  raw_groups:list[tuple[int,list[list[int]]]] = [(start,selectors[start:start+8]) for start in range(0,output_count,8)]
  if source_prepare is not None:
    raw_groups, next_output = [], 0
    while next_output < output_count:
      selected_rows = next((selectors[next_output:next_output+channels] for channels in range(min(8,output_count-next_output),0,-1)
                            if group_input_width(selectors[next_output:next_output+channels]) <= 512),None)
      if selected_rows is None:
        raw_groups.clear()
        break
      raw_groups.append((next_output,selected_rows))
      next_output += len(selected_rows)
  elif any(group_input_width(rows) > 512 for _,rows in raw_groups): raw_groups.clear()
  if not raw_groups:
    scalar = _scalar_affine_max_program(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot),
      source, selectors, pool_extent, pool_shape, red_axes[0], () if source_prepare is None else source_prepare.scratch)
    if scalar is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, "affine PPU MAX scalar windows exceed plan limits", reduce.op)
    return _native(scalar if source_prepare is None else
      RKProgram((RKDPUProgram(source_prepare.stages,scalar.scratch),*scalar.steps),scalar.scratch))
  windows:list[tuple[int, int, int, int|None, list[tuple[int, list[list[int]]]]]] = []
  next_group = 0
  while next_group < len(raw_groups):
    window_groups:list[tuple[int, list[list[int]]]] = []
    for group_candidate in raw_groups[next_group:]:
      trial_rows = [row for _,rows in (*window_groups,group_candidate) for row in rows]
      selected_indices = [index for row in trial_rows for index in row if index >= 0]
      base, end = (0,input_count) if whole_surface else ((min(selected_indices)&-8,max(selected_indices)+1) if selected_indices else (0,0))
      masked = any(index < 0 for row in trial_rows for index in row)
      align_in = max(32, (end-base+31)&-32) + (32 if masked else 0)
      if align_in > 512 or not whole_surface and masked and align_in > RK_MAX_AFFINE_WINDOW and window_groups: break
      window_groups.append(group_candidate)
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

  initial_scratch = () if source_prepare is None else source_prepare.scratch
  packed, hwc = RKArg(RKBufferKind.SCRATCH, len(initial_scratch)), RKArg(RKBufferKind.SCRATCH, len(initial_scratch)+1)
  hwc_elements = ((surface_count-1)//16)*16+32
  scratch = initial_scratch+(RKScratch(max(window[2] for window in windows)*2), RKScratch(hwc_elements*2))
  grouped_output:RKArg|None = None
  group_slots = {start:index for index,(start,_) in enumerate(raw_groups)}
  if source_prepare is not None:
    grouped_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(len(raw_groups)*16),)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  if source_prepare is not None: steps.append(RKDPUProgram(source_prepare.stages,scratch))
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
      channels = len(group_rows)
      rows = [[local_rows[min(channel, channels-1)][min(spatial, reduction_count-1)] for channel in range(8)]
              for spatial in range(pool_extent)]
      flat_rows = [cast(int, sentinel) if rows[spatial][channel] < 0 else rows[spatial][channel]
                   for spatial in range(pool_extent) for channel in range(8)]
      packed_group = RKArg(packed.kind,packed.index,(group_base-base)*2)
      selector = _windowed_cmac_pipeline(hwc,packed_group,[[index] for index in flat_rows],scratch=scratch,
                                         direct_count=contract_align,max_outputs=64)
      if selector is None:
        return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"affine PPU MAX HWC selector exceeds plan limits",reduce.op)
      steps.extend(selector.steps)
      scratch = selector.scratch
      out_arg = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot,group_start*2) if grouped_output is None else \
                RKArg(grouped_output.kind,grouped_output.index,group_slots[group_start]*16)
      out = RKTensorRef(out_arg,
                        RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half,kind=RKLayoutKind.PPU_HWC))
      src = RKTensorRef(hwc, RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half,
        kind=RKLayoutKind.PPU_HWC))
      steps.append(RKReduce(out, src, Ops.MAX, red_axes[0]))
  if grouped_output is not None:
    compact_rows = [[group_slots[start]*8+channel] for start,rows in raw_groups for channel in range(len(rows))]
    compact = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),grouped_output,
      len(raw_groups)*8,compact_rows,scratch,direct_capacity=len(raw_groups)*8,max_outputs=64)
    if compact is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"affine PPU MAX output compaction exceeds plan limits",reduce.op)
    steps.extend(compact.steps)
    scratch = compact.scratch
  program = _finish_program(steps,scratch)
  cost = plan_cost(program)
  if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"affine PPU MAX needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes", reduce.op)
  return _native(program)

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
  RKLowerer("multi_source_reformat", lambda nodes:not _has_reduction(nodes), lower_multi_source_reformat_result),
  RKLowerer("static_selector_reformat", lambda nodes:not _has_reduction(nodes), lower_static_selector_reformat_result),
  RKLowerer("static_two_tap", lambda nodes:not _has_reduction(nodes), lower_static_two_tap_result),
  RKLowerer("multi_broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_multi_broadcast_alu_result),
  RKLowerer("broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_broadcast_alu_result),
  RKLowerer("reformat", lambda nodes:not _has_reduction(nodes), lower_reformat_result),
  RKLowerer("affine_mean", lambda nodes:_has_reduction(nodes, Ops.ADD) and sum(u.op is Ops.REDUCE for u in nodes) == 2,
            lower_affine_mean_result),
  RKLowerer("nested_sum", lambda nodes:_has_reduction(nodes, Ops.ADD) and sum(u.op is Ops.REDUCE for u in nodes) > 1,
            lower_nested_add_reduce_result),
  RKLowerer("scalar_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_scalar_mul_reduce_result),
  RKLowerer("masked_affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_masked_affine_mul_reduce_result),
  RKLowerer("affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_affine_mul_reduce_result),
  RKLowerer("sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_add_reduce_result),
  RKLowerer("multi_source_sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_multi_source_affine_reduce_result),
  RKLowerer("pointwise_affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_pointwise_affine_reduce_result),
  RKLowerer("affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_affine_reduce_result),
  RKLowerer("multi_source_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_multi_source_affine_reduce_result),
  RKLowerer("ppu_reduce", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_reduce_result),
  RKLowerer("sliding_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_sliding_max_result),
  RKLowerer("dense_row_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_dense_row_max_result),
  RKLowerer("affine_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_affine_max_result),
  RKLowerer("global_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_global_max_result),
  RKLowerer("depthwise_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_depthwise_spatial_contract_result),
  RKLowerer("grouped_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_grouped_spatial_contract_result),
  RKLowerer("nhwc_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_nhwc_spatial_contract_result),
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

from tinygrad.renderer.rockchip.emit import (emit_dpu as emit_dpu, emit_cmac_task as emit_cmac_task, emit_spatial_conv as emit_spatial_conv,
  emit_program as emit_program, emit_reduce as emit_reduce, emit_pool as emit_pool, emit_reformat as emit_reformat)

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
    elif isinstance(result.plan, RKCMACTask): image = emit_cmac_task(result.plan)
    elif isinstance(result.plan, RKConvTask): image = emit_spatial_conv(result.plan)
    elif isinstance(result.plan, RKPool): image = emit_pool(result.plan)
    elif isinstance(result.plan, RKReduce): image = emit_reduce(result.plan)
    elif isinstance(result.plan, RKLegalizedReformat): image = emit_reformat(result.plan)
    elif isinstance(result.plan, RKProgram): image = emit_program(result.plan)
    else: raise RuntimeError("invalid Rockchip lowering result")
    linear = UOp(Ops.LINEAR, src=tuple(u for u in params if u.addrspace is not AddrSpace.ALU))
    return UOp(Ops.PROGRAM, src=(ast, linear, UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=encode_image(image))),
               arg=info)
