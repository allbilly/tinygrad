from __future__ import annotations
import math, struct
from itertools import product
from typing import cast

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.access import (RKMultiSourceAffineSegment, RKMultiSourceAccessMap, RKMultiSourceAffineGridMap,
  RKMultiSourceMap, compact_multi_source_map)
from tinygrad.renderer.rockchip.analysis import (strip_casts as _strip_casts, static_scalar as _static_scalar,
  static_linear_form as _static_linear_form, static_selected_index as _static_selected_index)
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.expr import _ALUExpr
from tinygrad.renderer.rockchip.ir import (RKBufferKind, RKReformatKind, RKArg, RKALUStage, RKCopyStage, RKDPUStage, RKScratch,
  RKDPUProgram, RKLayout, RKTensorRef, RKMultiSourceReformatPlan, RKLegalizedReformat, RKProgram, RKRejectKind, RKLowerResult)
from tinygrad.renderer.rockchip.limits import (RK_MAX_CONSTANT_BYTES, RK_MAX_AFFINE_VISITS, RK_MAX_PROGRAM_STAGES,
  RK_MAX_TILED_CMAC_SELECTOR_WINDOW)
from tinygrad.renderer.rockchip.lower import native as _native, not_applicable as _not_applicable, unsupported as _unsupported
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr
from tinygrad.renderer.rockchip.selector import (_cmac_tiled_output_bytes, _dense_ref, _dense_half_ref,
  _windowed_weighted_cmac_pipeline, _selector_program, _multi_source_windowed_program, _best_partitioned_selector_program,
  _finish_program, _legalized_reformat, _periodic_selector_program)

def _range_affine(u:UOp) -> tuple[dict[UOp,int],int]|None:
  """Preserve split RANGE identities instead of collapsing their shared logical axis id."""
  if u.op is Ops.RANGE: return ({u:1},0)
  if u.op is Ops.CONST: return ({},int(u.arg))
  if u.op is Ops.ADD:
    lhs,rhs = _range_affine(u.src[0]),_range_affine(u.src[1])
    if lhs is None or rhs is None: return None
    return ({axis:lhs[0].get(axis,0)+rhs[0].get(axis,0) for axis in lhs[0].keys()|rhs[0].keys()},lhs[1]+rhs[1])
  if u.op is Ops.MUL:
    constant,value = (u.src[0],u.src[1]) if u.src[0].op is Ops.CONST else (u.src[1],u.src[0])
    if constant.op is not Ops.CONST or (parsed:=_range_affine(value)) is None: return None
    return ({axis:coefficient*int(constant.arg) for axis,coefficient in parsed[0].items()},parsed[1]*int(constant.arg))
  return None

def lower_static_two_tap_result(sink:UOp) -> RKLowerResult:
  """Lower a dense FP32 convex blend of at most two statically indexed FP16 values through CMAC."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
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
    if constant != 0.0 or not 1 <= len(terms) <= 2 or any(not 0 <= index < source_count for index in terms) or \
       not math.isclose(sum(terms.values()),1.0,rel_tol=0.0,abs_tol=1e-12):
      return _unsupported(RKRejectKind.UNSUPPORTED_ALU,"static transform is not a zero-bias two-tap blend",stored.op)
    ordered = sorted(terms.items())
    if len(ordered) == 2 and ordered[1][0] != ordered[0][0]+1:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static two-tap sources are not adjacent",Ops.INDEX)
    high = struct.unpack("<e",struct.pack("<e",ordered[-1][1]))[0]
    rounded = [(ordered[0][0],struct.unpack("<e",struct.pack("<e",1.0-high))[0]),(ordered[1][0],high)] if len(ordered) == 2 else \
      [(ordered[0][0],struct.unpack("<e",struct.pack("<e",ordered[0][1]))[0])]
    if any(not math.isfinite(weight) or not 0.0 <= weight <= 1.0 for _,weight in rounded):
      return _unsupported(RKRejectKind.NUMERICAL_CONTRACT,"static two-tap coefficients do not fit FP16",stored.op)
    rows[dst] = rounded
  if any(row is None for row in rows):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT,"static two-tap transform has output holes",Ops.INDEX)
  output = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot)
  source_arg = RKArg(RKBufferKind.ARG,source.arg.slot)
  program = _windowed_weighted_cmac_pipeline(output,source_arg,cast(list[list[tuple[int,float]]],rows),direct_count=source_count)
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
  source_id = {param.key:index for index,param in enumerate(params)}
  source_counts = tuple(int(param.src[0].arg) for param in params)
  # A concatenation of complete dense surfaces simplifies to one outer selector and one full-surface inner range.
  # Prove that compact structure directly so large concatenations never require an element-per-output compiler map.
  concat_access:RKMultiSourceMap|None = None
  exact_out = _range_affine(store.src[0].src[1])
  source_indexes = tuple(u for u in store.src[1].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM)
  conditions = tuple(u.src[0] for u in store.src[1].toposort() if u.op is Ops.WHERE)
  condition_ranges = {u for condition in conditions for u in condition.toposort() if u.op is Ops.RANGE}
  exact_sources = tuple(_range_affine(index.src[1]) for index in source_indexes)
  if exact_out is not None and not exact_out[1] and set(exact_out[0]) == set(ranges) and len(condition_ranges) == 1 and \
     all(parsed is not None and not parsed[1] for parsed in exact_sources):
    selector = next(iter(condition_ranges))
    source_coefficients = cast(tuple[tuple[dict[UOp,int],int], ...],exact_sources)
    common_source = source_coefficients[0][0]
    if all(parsed[0] == common_source for parsed in source_coefficients) and selector not in common_source and \
       set(common_source) == set(ranges)-{selector}:
      points:list[int] = []
      zero_point = {axis:0 for axis in ranges}
      for coordinate in range(int(selector.src[0].arg)):
        point = {**zero_point,selector:coordinate}
        selected = _static_selected_index(store.src[1],point)
        if selected is None or selected.src[0].key not in source_id or _static_scalar(selected.src[1],point) != 0:
          points = []
          break
        points.append(source_id[selected.src[0].key])
      if points:
        concat_access = RKMultiSourceAffineGridMap(tuple(int(axis.src[0].arg) for axis in ranges),
          tuple(exact_out[0][axis] for axis in ranges),tuple(common_source.get(axis,0) for axis in ranges),ranges.index(selector),
          tuple(points),(0,)*len(points))
  out_aff = _affine(store.src[0].src[1])
  if concat_access is None and out_aff is not None and not out_aff[1] and len(ranges) == 2 and len(out_aff[0]) == 2:
    inner = next((axis for axis in ranges if out_aff[0].get(axis.arg[0]) == 1),None)
    outer = next((axis for axis in ranges if inner is not None and axis is not inner),None)
    if inner is not None and outer is not None:
      inner_count, outer_count = int(inner.src[0].arg), int(outer.src[0].arg)
      dense_sources = source_counts and all(count == inner_count for count in source_counts) and all(
        _affine(index.src[1]) == ({inner.arg[0]:1},0) for index in source_indexes)
      outer_only = all({u.arg[0] for u in condition.toposort() if u.op is Ops.RANGE} <= {outer.arg[0]} for condition in conditions)
      if dense_sources and outer_only and out_aff == ({outer.arg[0]:inner_count,inner.arg[0]:1},0) and \
         output_count == outer_count*inner_count:
        segments:list[RKMultiSourceAffineSegment] = []
        for coordinate in range(outer_count):
          selected = _static_selected_index(store.src[1],{outer:coordinate,inner:0})
          if selected is None or selected.src[0].key not in source_id or _static_scalar(selected.src[1],{outer:coordinate,inner:0}) != 0:
            segments = []
            break
          segments.append(RKMultiSourceAffineSegment(source_id[selected.src[0].key],0,1,inner_count))
        if segments: concat_access = RKMultiSourceAccessMap(tuple(segments))
  if concat_access is not None and visits > RK_MAX_AFFINE_VISITS:
    out_ref = _dense_ref(store.src[0].src[0].arg.slot,(output_count,),dtype)
    source_refs = tuple(_dense_ref(param.arg.slot,(count,),dtype) for param,count in zip(params,source_counts))
    semantic = RKMultiSourceReformatPlan(out_ref,source_refs,concat_access)
    direct = _multi_source_windowed_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),
      tuple(RKArg(RKBufferKind.ARG,param.arg.slot) for param in params),source_counts,concat_access,max_outputs=256,
      max_window=RK_MAX_TILED_CMAC_SELECTOR_WINDOW)
    if direct is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"dense concatenation exceeds native window limits",Ops.WHERE)
    cost = plan_cost(direct)
    if cost.stage_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
        f"dense concatenation needs {cost.stage_count} stages and {cost.constant_bytes} constant bytes",Ops.WHERE)
    return _native(RKLegalizedReformat(semantic,RKReformatKind.SELECTOR_CMAC,direct))
  if visits > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,f"multi-source reformat needs {visits} coordinate visits",Ops.RANGE)
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
  access = compact_multi_source_map(typed_mapping)
  semantic = RKMultiSourceReformatPlan(out_ref,source_refs,access)
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
  direct = _multi_source_windowed_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),sources,source_counts,access)
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
