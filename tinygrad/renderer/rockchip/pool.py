from __future__ import annotations
import math
from itertools import permutations, product
from typing import cast

from tinygrad.dtype import dtypes, Invalid
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.analysis import strip_casts as _strip_casts, static_index_selected as _static_index_selected
from tinygrad.renderer.rockchip.cost import plan_cost
from tinygrad.renderer.rockchip.ir import (RKBufferKind, RKLayoutKind, RKArg, RKALUStage, RKScratch, RKDPUProgram, RKLayout, RKTensorRef,
  RKCMACTask, RKConvTask, RKReduce, RKPool, RKProgram, RKRejectKind, RKLowerResult)
from tinygrad.renderer.rockchip.limits import (RK_MAX_CONSTANT_BYTES, RK_MAX_AFFINE_VISITS, RK_MAX_PROGRAM_STAGES,
  RK_MAX_AFFINE_WINDOW, RK_MAX_CMAC_SELECTOR_WINDOW)
from tinygrad.renderer.rockchip.lower import native as _native, not_applicable as _not_applicable, unsupported as _unsupported
from tinygrad.renderer.rockchip.selector import (_dense_half_ref, _cmac_weight_ref, _cmac_selection_payload, _sparse_cmac_pipeline,
  _windowed_cmac_pipeline, _selector_program, _finish_program)

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
  if out_aff is None or src_aff is None or out_aff[1] or src_aff[1] or len(out_aff[0]) < 3:
    return _not_applicable()
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  out_axes, red_axes = tuple(out_aff[0]), tuple(red.arg[0] for red in reduce.src[1:])
  if len(red_axes) != 2 or any(axis not in ranges for axis in (*out_axes,*red_axes)): return _not_applicable()

  match:tuple[int,int,int,int,int,int,int,int,int]|None = None
  for ordered_out_axes in permutations(out_axes):
    *plane_axes,out_y_axis,out_x_axis = ordered_out_axes
    planes, out_h, out_w = math.prod(ranges[x] for x in plane_axes), ranges[out_y_axis], ranges[out_x_axis]
    for kernel_y_axis,kernel_x_axis in permutations(red_axes):
      kernel_h, kernel_w = ranges[kernel_y_axis], ranges[kernel_x_axis]
      in_w = src_aff[0].get(kernel_y_axis,0)
      if in_w <= 0 or not plane_axes or src_aff[0].get(plane_axes[-1],0)%in_w: continue
      in_h = src_aff[0][plane_axes[-1]]//in_w
      stride_y_coeff, stride_x = src_aff[0].get(out_y_axis,0), src_aff[0].get(out_x_axis,0)
      if stride_y_coeff <= 0 or stride_y_coeff%in_w: continue
      stride_y = stride_y_coeff//in_w
      expected_src = {axis:coefficient*in_h*in_w for axis,coefficient in _dense_axis_coefficients(tuple(plane_axes),ranges).items()}
      expected_src.update({out_y_axis:stride_y*in_w,out_x_axis:stride_x,kernel_y_axis:in_w,kernel_x_axis:1})
      if out_aff[0] != _dense_axis_coefficients(ordered_out_axes,ranges) or src_aff[0] != expected_src: continue
      input_count, output_count = int(value.src[0].src[0].arg), int(store.src[0].src[0].src[0].arg)
      if (input_count,output_count) != (planes*in_h*in_w,planes*out_h*out_w) or \
         out_h != (in_h-kernel_h)//stride_y+1 or out_w != (in_w-kernel_w)//stride_x+1: continue
      match = planes,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x
      break
    if match is not None: break
  if match is None: return _not_applicable()
  planes,in_h,in_w,kernel_h,kernel_w,out_h,out_w,stride_y,stride_x = match
  if not 1 <= planes <= 64 or not 2 <= max(kernel_h,kernel_w) <= 8 or min(kernel_h,kernel_w) < 2 or \
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
