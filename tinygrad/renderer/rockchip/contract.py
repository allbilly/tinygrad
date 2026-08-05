from __future__ import annotations
import math, struct
from itertools import permutations, product
from typing import cast

from tinygrad.dtype import dtypes, Invalid
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.analysis import (strip_casts as _strip_casts, static_scalar as _static_scalar,
  conditional_index as _conditional_index, conditional_index_affine as _conditional_index_affine,
  conv_zero_padding as _conv_zero_padding, contract_bias_epilogue as _contract_bias_epilogue)
from tinygrad.renderer.rockchip.conv import plan_conv_cbuf, legalize_conv_plan
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.ir import (RKEngine, RKBufferKind, RKLayoutKind, RKArg, RKALUStage, RKStridedAtomGatherStage, RKScratch,
  RKDPUProgram, RKLayout, RKTensorRef, RKEpilogue, RKContractionPlan, RKCMACTask, RKConvTask, RKDeconvTask, RKConvPlan, RKConvSplit, RKReduce,
  RKRejectKind, RKLowerResult)
from tinygrad.renderer.rockchip.limits import (RK_MAX_CONSTANT_BYTES, RK_MAX_PROGRAM_STAGES,
  RK_MAX_AFFINE_WINDOW, RK_MAX_CMAC_SELECTOR_WINDOW, RK_MAX_TILED_CMAC_SELECTOR_WINDOW, RK_MAX_TILED_CONTRACT_VISITS)
from tinygrad.renderer.rockchip.lower import native as _native, not_applicable as _not_applicable, unsupported as _unsupported
from tinygrad.renderer.rockchip.reduce import _finish_reduction_epilogue
from tinygrad.renderer.rockchip.selector import (_cmac_tiled_output_bytes, _dense_half_ref, _cmac_weight_ref, _cmac_selection_payload,
  _selector_program, _finish_program)

def _pack_row_major_rhs(rhs:RKArg, n:int, k:int, scratch:tuple[RKScratch, ...]) -> \
    tuple[list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce], tuple[RKScratch, ...], RKArg]:
  """Gather and transpose one aligned row-major KxN RHS into the proven blocked CMAC weight stream."""
  if n%32 or k%32 or not 32 <= n <= k <= 128: raise ValueError(f"unsupported row-major RHS N={n},K={k}")
  packed = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(n*k*2),)
  gathered = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(k*8*2),)
  transpose = _cmac_selection_payload([[k_lane*8+n_lane] for n_lane in range(8) for k_lane in range(32)],256,256,1.0)
  lhs_layout = RKLayout((1,256),(1,256),(512,2),dtypes.half)
  out_layout = RKLayout((1,256),(1,256),(512,2),dtypes.half)
  weight = _cmac_weight_ref(0,256,256,RKBufferKind.CONSTANT,256)
  steps:list[RKDPUProgram|RKCMACTask|RKConvTask|RKReduce] = []
  for n_start in range(0,n,8):
    steps.append(RKDPUProgram((RKStridedAtomGatherStage(gathered,RKArg(rhs.kind,rhs.index,rhs.addend+n_start*2),k,n),),scratch))
    for k_start in range(0,k,32):
      packed_offset = (((n_start//16)*(k//32)+k_start//32)*512+(n_start%16//8)*256)*2
      steps.append(RKCMACTask(RKTensorRef(RKArg(packed.kind,packed.index,packed_offset),out_layout),
        RKTensorRef(RKArg(gathered.kind,gathered.index,k_start*8*2),lhs_layout),weight,0,transpose,compact_output=True))
  return steps,scratch,packed

def legalize_contraction_plan(plan:RKContractionPlan) -> tuple[RKCMACTask, ...]:
  """Legalize one already-packed dense contraction into one physical CMAC task."""
  plan.lhs.layout.validate_for(RKEngine.CMAC)
  plan.rhs.layout.validate_for(RKEngine.CMAC)
  plan.out.layout.validate_for(RKEngine.CMAC)
  if plan.rhs.layout.kind is not RKLayoutKind.CMAC_WEIGHT:
    raise ValueError("RK contraction RHS is not in CMAC weight layout")
  if plan.lhs.layout.logical_shape != (plan.logical_m,plan.logical_k) or \
     plan.rhs.layout.logical_shape != (plan.logical_n,plan.logical_k) or \
     plan.out.layout.logical_shape != (plan.logical_m,plan.logical_n):
    raise ValueError("RK direct contraction layouts do not match logical geometry")
  align_out, align_in = max(32,(plan.logical_n+31)&-32), max(32,(plan.logical_n+31)&-32,(plan.logical_k+31)&-32)
  if plan.lhs.layout.physical_shape != (plan.logical_m,align_in) or plan.lhs.layout.strides_bytes != (align_in*2,2):
    raise ValueError("RK direct contraction LHS is not in the proven aligned row surface")
  if plan.rhs.layout.physical_shape != (align_out,align_in) or plan.rhs.layout.strides_bytes != (align_in*2,2):
    raise ValueError("RK direct contraction RHS is not in the proven 16x32 blocked weight surface")
  compact = plan.logical_m == 1 and plan.logical_n <= 16 and plan.out.layout.physical_shape == (1,plan.logical_n) and \
    plan.out.layout.strides_bytes == (plan.logical_n*2,2)
  padded = plan.out.layout.physical_shape == (plan.logical_m,align_out*2) and plan.out.layout.strides_bytes == (align_out*4,2)
  if not compact and not padded:
    raise ValueError("RK direct contraction output is not in the proven FP16 CMAC surface")
  return (RKCMACTask(plan.out,plan.lhs,plan.rhs,plan.reduction_axes[0],plan.constants,plan.epilogue),)

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

def _deconv_source(parsed:tuple[UOp,UOp|None,bool], point:dict[int,int]) -> int|None:
  index, condition, select_true = parsed
  predicate = True if condition is None else _static_scalar(condition,point)
  if predicate is None: return None
  if bool(predicate) is not select_true: return -1
  source = _static_scalar(index.src[1],point)
  return source if isinstance(source,int) and not isinstance(source,bool) else None

def _deconv_dimensions(input_size:int, output_size:int, kernel_size:int) -> tuple[tuple[int,int,int,int], ...]:
  """Return transpose-stride, dilation, symmetric-padding, and output-padding candidates."""
  return tuple((stride,dilation,padding,output_padding) for stride in range(1,9) for dilation in range(1,33)
               for padding in range(16) for output_padding in range(stride)
               if output_size == (input_size-1)*stride-2*padding+(kernel_size-1)*dilation+1+output_padding and
                  padding <= (kernel_size-1)*dilation)

def lower_deconv_result(sink:UOp) -> RKLowerResult:
  """Recognize decomposed FP16 transpose convolution and preserve its per-kernel-position FP16 accumulation."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) != 4 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half: return _not_applicable()
  stored = _strip_casts(store.src[1])
  bias_epilogue = None if stored.key == reduce.key else _contract_bias_epilogue(stored,reduce)
  if stored.key != reduce.key and bias_epilogue is None: return _not_applicable()
  body = _strip_casts(reduce.src[0])
  if body.op is Ops.WHERE:
    branches = tuple(_strip_casts(x) for x in body.src[1:])
    real = tuple(x for x in branches if not (x.op is Ops.CONST and x.arg is Invalid))
    if len(real) == 1: body = real[0]
  if body.op is not Ops.MUL or any(red.op is not Ops.RANGE for red in reduce.src[1:]): return _not_applicable()
  operands = tuple(_conditional_index(_strip_casts(value)) for value in body.src)
  if any(parsed is None or parsed[0].dtype is not dtypes.half or parsed[0].src[0].op is not Ops.PARAM for parsed in operands):
    return _not_applicable()
  parsed_operands = cast(tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool]],operands)
  out_aff = _affine(store.src[0].src[1])
  if out_aff is None or out_aff[1] or len(out_aff[0]) not in (4,5): return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if len(red_axes) != 3 or any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[tuple[UOp,UOp|None,bool],tuple[UOp,UOp|None,bool],int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int]|None = None
  matched_bias_rows:list[list[int]]|None = None
  for feature_parsed,weight_parsed in (parsed_operands,tuple(reversed(parsed_operands))):
    feature, weight = feature_parsed[0], weight_parsed[0]
    # A fully cropped transpose can simplify every feature coordinate in bounds, removing the WHERE predicate.
    # The exhaustive source/weight address validation below still distinguishes it from an ordinary contraction.
    if weight_parsed[1] is not None: continue
    feature_count, weight_count, output_count = (int(x.src[0].src[0].arg) for x in (feature,weight,store.src[0]))
    output_candidates = tuple((b,None,oc,y,x) for b,oc,y,x in permutations(out_axes)) if len(out_axes) == 4 else \
      tuple(permutations(out_axes))
    for batch_axis,group_axis,out_channel_axis,out_y_axis,out_x_axis in output_candidates:
      batch,out_c_group,out_h,out_w = (ranges[x] for x in (batch_axis,out_channel_axis,out_y_axis,out_x_axis))
      explicit_groups = 1 if group_axis is None else ranges[group_axis]
      out_c = explicit_groups*out_c_group
      expected_output = {batch_axis:out_c*out_h*out_w,out_channel_axis:out_h*out_w,out_y_axis:out_w,out_x_axis:1}
      if group_axis is not None: expected_output[group_axis] = out_c_group*out_h*out_w
      if out_aff[0] != expected_output or output_count != batch*out_c*out_h*out_w: continue
      for in_channel_axis,kernel_y_axis,kernel_x_axis in permutations(red_axes):
        in_c_group,kernel_h,kernel_w = (ranges[x] for x in (in_channel_axis,kernel_y_axis,kernel_x_axis))
        if not 1 <= kernel_h <= 3 or not 1 <= kernel_w <= 3 or weight_count != in_c_group*out_c*kernel_h*kernel_w: continue
        group_candidates = (explicit_groups,) if group_axis is not None else tuple(group for group in range(1,out_c+1) if out_c%group == 0)
        for groups in group_candidates:
          in_c, out_c_group = in_c_group*groups, out_c//groups
          if feature_count%(batch*in_c): continue
          spatial = feature_count//(batch*in_c)
          for in_h in range(1,min(32,spatial)+1):
            if spatial%in_h or spatial//in_h > 32: continue
            in_w = spatial//in_h
            for stride_y,dilation_y,pad_y,_output_pad_y in _deconv_dimensions(in_h,out_h,kernel_h):
              for stride_x,dilation_x,pad_x,_output_pad_x in _deconv_dimensions(in_w,out_w,kernel_w):
                valid = True
                for b,oc,oy,ox,icg,ky,kx in product(range(batch),range(out_c),range(out_h),range(out_w),
                                                      range(in_c_group),range(kernel_h),range(kernel_w)):
                  point = {batch_axis:b,out_channel_axis:oc,out_y_axis:oy,out_x_axis:ox,
                           in_channel_axis:icg,kernel_y_axis:ky,kernel_x_axis:kx}
                  if group_axis is not None:
                    point[group_axis], point[out_channel_axis] = oc//out_c_group, oc%out_c_group
                  iy_num = oy+pad_y-(kernel_h-1-ky)*dilation_y
                  ix_num = ox+pad_x-(kernel_w-1-kx)*dilation_x
                  group, source = oc//out_c_group, -1
                  if iy_num%stride_y == 0 and ix_num%stride_x == 0:
                    iy, ix = iy_num//stride_y, ix_num//stride_x
                    if 0 <= iy < in_h and 0 <= ix < in_w:
                      source = b*in_c*in_h*in_w+(group*in_c_group+icg)*in_h*in_w+iy*in_w+ix
                  weight_source = (group*in_c_group+icg)*out_c_group*kernel_h*kernel_w+\
                    (oc%out_c_group)*kernel_h*kernel_w+(kernel_h-1-ky)*kernel_w+(kernel_w-1-kx)
                  if _deconv_source(feature_parsed,point) != source or _deconv_source(weight_parsed,point) != weight_source:
                    valid = False
                    break
                candidate_bias_rows:list[list[int]]|None = None
                if valid and bias_epilogue is not None:
                  bias, _relu = bias_epilogue
                  bias_count = int(bias.src[0].src[0].arg)
                  candidate_bias_rows = []
                  for b,oc,oy,ox in product(range(batch),range(out_c),range(out_h),range(out_w)):
                    point = {batch_axis:b,out_channel_axis:oc,out_y_axis:oy,out_x_axis:ox}
                    if group_axis is not None:
                      point[group_axis], point[out_channel_axis] = oc//out_c_group, oc%out_c_group
                    bias_source = _static_scalar(bias.src[1],point)
                    if not isinstance(bias_source,int) or isinstance(bias_source,bool) or not 0 <= bias_source < bias_count:
                      valid = False
                      break
                    candidate_bias_rows.append([bias_source])
                if valid:
                  match = (feature_parsed,weight_parsed,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,
                           stride_y,stride_x,dilation_y,dilation_x,pad_y,pad_x,groups)
                  matched_bias_rows = candidate_bias_rows
                  break
              if match is not None: break
            if match is not None: break
          if match is not None: break
        if match is not None: break
      if match is not None: break
    if match is not None: break
  if match is None: return _not_applicable()

  feature_parsed,weight_parsed,batch,in_c,out_c,in_h,in_w,kernel_h,kernel_w,out_h,out_w,\
    stride_y,stride_x,dilation_y,dilation_x,pad_y,pad_x,groups = match
  feature, weight = feature_parsed[0], weight_parsed[0]
  if in_c not in (1,2,3,4,16) or not 1 <= out_c <= 16 or batch > 4:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"direct deconvolution is B={batch},IC={in_c},OC={out_c},H={in_h},W={in_w},K={kernel_h}x{kernel_w}",reduce.op)
  # The characterized CNA inserted-zero fields implement strides one and two. For a wider axis, materialize the
  # zero-inserted feature surface through the existing native selector and leave that CNA axis at stride one.
  expand_y, expand_x = stride_y > 2, stride_x > 2
  hardware_in_h = (in_h-1)*stride_y+1 if expand_y else in_h
  hardware_in_w = (in_w-1)*stride_x+1 if expand_x else in_w
  hardware_stride_y, hardware_stride_x = (1 if expand_y else stride_y), (1 if expand_x else stride_x)
  if hardware_in_h > 32 or hardware_in_w > 32:
    return _unsupported(RKRejectKind.UNSUPPORTED_CONTRACTION,
      f"deconvolution zero-inserted CNA surface is {hardware_in_h}x{hardware_in_w}",reduce.op)
  align_in, input_c2 = (8,in_c) if in_c <= 4 else (16,8)
  width_alignment = max(1,(16+align_in-1)//align_in)
  full_h, full_w = out_h+2*pad_y, out_w+2*pad_x
  input_width_stride = ((hardware_in_w+width_alignment-1)//width_alignment)*width_alignment
  output_width_stride = (full_h*full_w+3)&-4
  input_surface_count = hardware_in_h*input_width_stride*in_c
  input_batch_count, output_batch_count = (input_surface_count+7)&-8, 2*output_width_stride*8
  input_rows:list[list[int]] = []
  if in_c <= 4:
    for b in range(batch):
      input_rows.extend([[b*in_c*in_h*in_w+c*in_h*in_w+(y//stride_y if expand_y else y)*in_w+(x//stride_x if expand_x else x)]
                         if x < hardware_in_w and (not expand_y or y%stride_y == 0) and (not expand_x or x%stride_x == 0) else []
                         for y in range(hardware_in_h) for x in range(input_width_stride) for c in range(input_c2)])
      input_rows.extend([[] for _ in range(input_batch_count-input_surface_count)])
  else:
    for b in range(batch):
      input_rows.extend([[b*in_c*in_h*in_w+(c1*input_c2+c2)*in_h*in_w+
                          (y//stride_y if expand_y else y)*in_w+(x//stride_x if expand_x else x)]
                         if x < hardware_in_w and (not expand_y or y%stride_y == 0) and (not expand_x or x%stride_x == 0) else []
                         for c1 in range(in_c//input_c2) for y in range(hardware_in_h)
                         for x in range(input_width_stride) for c2 in range(input_c2)])
      input_rows.extend([[] for _ in range(input_batch_count-input_surface_count)])
  in_c_group = in_c//groups
  # Each output group consumes one contiguous input-channel group.
  out_c_group = out_c//groups
  weight_rows = [[c*out_c_group*kernel_h*kernel_w+(oc%out_c_group)*kernel_h*kernel_w+ky*kernel_w+kx]
                  if oc//out_c_group*in_c_group <= c < (oc//out_c_group+1)*in_c_group else []
                 for ky in range(kernel_h) for kx in range(kernel_w) for oc in range(out_c) for c in range(align_in)]
  output_rows = [[b*output_batch_count+(oc//8)*output_width_stride*8+((y+pad_y)*full_w+x+pad_x)*8+oc%8]
                 for b in range(batch) for oc in range(out_c) for y in range(out_h) for x in range(out_w)]
  scratch:tuple[RKScratch, ...] = ()
  packed_input = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(input_rows)*2),)
  input_plan = _selector_program(packed_input,RKArg(RKBufferKind.ARG,feature.src[0].arg.slot),feature_count,input_rows,scratch,
    direct_capacity=((feature_count*2+4095)&-4096)//2,max_window=RK_MAX_CMAC_SELECTOR_WINDOW,max_outputs=64)
  if input_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"deconvolution input pack exceeds plan limits",Ops.INDEX)
  scratch, steps = input_plan.scratch, list(input_plan.steps)
  packed_weight = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(len(weight_rows)*2),)
  weight_plan = _selector_program(packed_weight,RKArg(RKBufferKind.ARG,weight.src[0].arg.slot),weight_count,weight_rows,scratch,
    direct_capacity=((weight_count*2+4095)&-4096)//2,max_outputs=64)
  if weight_plan is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"deconvolution weight pack exceeds plan limits",Ops.INDEX)
  scratch, steps = weight_plan.scratch, [*steps,*weight_plan.steps]
  packed_output = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(batch*output_batch_count*2),)
  contribution = RKArg(RKBufferKind.SCRATCH,len(scratch))
  scratch += (RKScratch(output_batch_count*2),)
  input_layout = RKLayout((hardware_in_h,hardware_in_w,in_c),(hardware_in_h,input_width_stride,in_c),
                          (input_width_stride*in_c*2,in_c*2,2),dtypes.half,
                          padding=((0,0),(0,input_width_stride-hardware_in_w),(0,0)),
                          kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0) if in_c <= 4 else \
    RKLayout((in_c//input_c2,hardware_in_h,hardware_in_w,input_c2),(in_c//input_c2,hardware_in_h,input_width_stride,input_c2),
      (hardware_in_h*input_width_stride*input_c2*2,input_width_stride*input_c2*2,input_c2*2,2),dtypes.half,
      padding=((0,0),(0,0),(0,input_width_stride-hardware_in_w),(0,0)),kind=RKLayoutKind.CNA_ACTIVATION,padding_value=0)
  weight_layout = RKLayout((1,1,out_c,in_c),(1,1,out_c,align_in),(out_c*align_in*2,out_c*align_in*2,align_in*2,2),dtypes.half,
    padding=((0,0),(0,0),(0,0),(0,align_in-in_c)),kind=RKLayoutKind.CNA_WEIGHT,padding_value=0)
  output_layout = RKLayout((2,output_width_stride,8),(2,output_width_stride,8),(output_width_stride*16,16,2),dtypes.half,
                           kind=RKLayoutKind.CONV_OUTPUT)
  for b in range(batch):
    accumulator = RKArg(packed_output.kind,packed_output.index,b*output_batch_count*2)
    for ky in range(kernel_h):
      for kx in range(kernel_w):
        first = ky == kx == 0
        task_output = accumulator if first else contribution
        weight_offset = (ky*kernel_w+kx)*out_c*align_in*2
        steps.append(RKDeconvTask(RKTensorRef(task_output,output_layout),
          RKTensorRef(RKArg(packed_input.kind,packed_input.index,b*input_batch_count*2),input_layout),
          RKTensorRef(RKArg(packed_weight.kind,packed_weight.index,weight_offset),weight_layout),
          in_c,out_c,hardware_in_h,hardware_in_w,1,1,full_h,full_w,1,1,input_width_stride,output_width_stride,
          transpose_stride_y=hardware_stride_y,transpose_stride_x=hardware_stride_x,
          hardware_pad_top=ky*dilation_y,hardware_pad_left=kx*dilation_x))
        if not first: steps.append(RKDPUProgram((RKALUStage(Ops.ADD,accumulator,accumulator,contribution,output_batch_count),),scratch))
  unpack = _selector_program(RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot),packed_output,
    batch*output_batch_count,output_rows,scratch,direct_capacity=batch*output_batch_count,max_outputs=64)
  if unpack is None: return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"deconvolution output unpack exceeds plan limits",Ops.INDEX)
  steps, scratch = [*steps,*unpack.steps], unpack.scratch
  if bias_epilogue is not None:
    bias, relu = bias_epilogue
    assert matched_bias_rows is not None
    expanded_bias = RKArg(RKBufferKind.SCRATCH,len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(output_count)),)
    bias_plan = _selector_program(expanded_bias,RKArg(RKBufferKind.ARG,bias.src[0].arg.slot),int(bias.src[0].src[0].arg),
                                  matched_bias_rows,scratch,max_outputs=64)
    if bias_plan is None:
      return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,"deconvolution bias broadcast exceeds plan limits",bias.op)
    scratch, steps = bias_plan.scratch, [*steps,*bias_plan.steps]
    output = RKArg(RKBufferKind.ARG,store.src[0].src[0].arg.slot)
    epilogue_stages = [RKALUStage(Ops.ADD,output,output,expanded_bias,output_count)]
    if relu: epilogue_stages.append(RKALUStage(Ops.MAX,output,output,0.0,output_count))
    steps.append(RKDPUProgram(tuple(epilogue_stages),scratch))
  program = _finish_program(steps,scratch)
  cost = plan_cost(program)
  if cost.task_count > RK_MAX_PROGRAM_STAGES or cost.constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"direct deconvolution needs {cost.task_count} stages and {cost.constant_bytes} constant bytes",reduce.op)
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
  k128_candidate = (k,lhs_count,rhs_count,output_count) == (128,128,128*128,128)
  if (not 1 <= k <= 96 or lhs_count > 8192 or rhs_count > 8192 or output_count*k > RK_MAX_TILED_CONTRACT_VISITS) and not k128_candidate:
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
  direct_row_major_rhs = direct_lhs and n%32 == 0 and k%32 == 0 and 32 <= n <= k <= 128 and rhs_count == n*k and \
    lhs_parsed[1] is None and rhs_parsed[1] is None and \
    rhs_columns == [tuple(red*n+channel for red in range(k)) for channel in range(n)]
  # One-row compact CMAC writes are proven through 128 outputs. Keep conditional/padded contractions on the older 32-output
  # schedule because changing their selector grouping can change the final FP16 accumulation contract.
  rhs_selector_outputs = 64 if align_out <= 64 and lhs_parsed[1] is None and rhs_parsed[1] is None else 32
  selector_floor = (0 if direct_lhs else (lhs_values+31)//32) + (rhs_values+rhs_selector_outputs-1)//rhs_selector_outputs + \
    (1 if compact_output else (output_count+31)//32) + \
    (m+(4096//align_in)-1)//(4096//align_in)
  if selector_floor > RK_MAX_PROGRAM_STAGES and not direct_row_major_rhs:
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
  if not direct_row_major_rhs:
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
  if direct_row_major_rhs:
    packed_steps,scratch,b_arg = _pack_row_major_rhs(RKArg(RKBufferKind.ARG,rhs.src[0].arg.slot),n,k,scratch)
    steps.extend(packed_steps)
  else:
    b_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
    scratch += (RKScratch(_cmac_tiled_output_bytes(len(b_selector))),)
    # Rockchip GEM allocations are page-rounded. Zero-weight selector lanes may read that physical tail without changing semantics.
    rhs_capacity = ((rhs_count*2+4095)&-4096)//2
    packed_b = _selector_program(b_arg, RKArg(RKBufferKind.ARG, rhs.src[0].arg.slot), rhs_count, b_selector, scratch,
                                 rhs_capacity, RK_MAX_TILED_CMAC_SELECTOR_WINDOW, rhs_selector_outputs)
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
    dense = _selector_program(reduced, cmac_out, m*align_out*2, unpack, scratch,
                              max_outputs=64 if direct_row_major_rhs else 32)
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
