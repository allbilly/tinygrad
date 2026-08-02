from __future__ import annotations
import math, os, struct
from dataclasses import dataclass
from itertools import product
from typing import Callable, cast
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId as RKLUTId
from tinygrad.runtime.support.rockchip_telemetry import record as record_telemetry
from tinygrad.uop.ops import AddrSpace, Ops, ProgramInfo, UOp

from tinygrad.renderer.rockchip.ir import (RKTarget as RKTarget, RKEngine as RKEngine, RKBufferKind, RKLayoutKind, RKArg,
  RKALUStage, RKMaskStage as RKMaskStage, RKLUTStage as RKLUTStage, RKDPUStage,
  RKScratch, RKDPUProgram, RKLayout, RKTensorRef, RKContract, RKReduce, RKProgram, RKRejectKind, RKReject, RKLowerKind, RKLowerResult)
from tinygrad.renderer.rockchip.image import (RK_STAGE_RESET as RK_STAGE_RESET, RKReloc as RKReloc, RKStage as RKStage, RKImage as RKImage,
  encode_image, decode_image as decode_image,
  patch_image as patch_image, validate_image as validate_image)
from tinygrad.renderer.rockchip.affine import affine as _affine, rk_fingerprint as rk_fingerprint
from tinygrad.renderer.rockchip.schedule import schedule_expr as _schedule_expr

RK_MAX_CONSTANT_BYTES = 2*1024*1024
RK_MAX_AFFINE_VISITS = 65536

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

def _finish_program(steps:list[RKDPUProgram|RKContract|RKReduce], scratch:tuple[RKScratch, ...]) -> RKProgram:
  """Give every ordered DPU step the program's final resource table."""
  return RKProgram(tuple(RKDPUProgram(step.stages, scratch) if isinstance(step, RKDPUProgram) else step for step in steps), scratch)

def _native(plan:RKDPUProgram|RKContract|RKReduce|RKProgram) -> RKLowerResult: return RKLowerResult(RKLowerKind.NATIVE, plan=plan)
def _not_applicable() -> RKLowerResult: return RKLowerResult(RKLowerKind.NOT_APPLICABLE)
def _unsupported(kind:RKRejectKind, detail:str, node_op:Ops|None=None) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(kind, detail, node_op))

from tinygrad.renderer.rockchip.expr import _ALUExpr, _MaskExpr, _LUTExpr, _Expr, _Value, _parse_alu, _unwrap_same_cast

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
  root = _parse_alu(store.src[1], out_index, {})
  if root is None: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "expression is not legal DPU arithmetic", _unwrap_same_cast(store.src[1]).op)
  if store.src[0].dtype is not dtypes.half and not isinstance(root, float):
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, f"non-constant {store.src[0].dtype.name} arithmetic", store.src[1].op)
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      tile = 64 if store.src[0].dtype is dtypes.int else 4
      fill_stages = tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*4), 0.0, root, min(tile, count-start),
                                     store.src[0].dtype) for start in range(0, count, tile))
      return _native(RKDPUProgram(fill_stages))
    return _native(RKDPUProgram((RKALUStage(Ops.ADD, output, 0.0, root, count),)))
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

def _relu_source(u:UOp) -> UOp|None:
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  cond, positive, zero = u.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2 or cond.src[0].op is not Ops.CONST or float(cond.src[0].arg) != 0 or \
     zero.op is not Ops.CONST or float(zero.arg) != 0: return None
  source = _strip_casts(cond.src[1])
  return source if _strip_casts(positive).key == source.key else None

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
  if atom_reject is None: return _native(RKDPUProgram(tuple(stages)))
  align_in = max(32, (src_count+31)&-32)
  constant_bytes = ((count+15)//16)*32*align_in*2
  if 0 < src_count <= 512 and 0 < count <= 4096 and constant_bytes <= RK_MAX_CONSTANT_BYTES:
    return _native(_sparse_cmac_pipeline(output, source, src_count, [[src] if src >= 0 else [] for src in mapping]))
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
  steps:list[RKDPUProgram|RKContract|RKReduce] = []
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

def lower_affine_reduce_result(sink:UOp) -> RKLowerResult:
  """Lower a small affine FP16 ADD reduction as generated sparse CMAC tiles."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or not reduce.src[1:]: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine CMAC requires an FP16 output", store.src[0].op)
  stored, scale = _strip_casts(store.src[1]), 1.0
  if stored.key != reduce.key:
    if stored.op is not Ops.MUL:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "affine CMAC only accepts a constant scale", stored.op)
    const, reduced = (stored.src[0], stored.src[1]) if stored.src[0].op is Ops.CONST else (stored.src[1], stored.src[0])
    if const.op is not Ops.CONST or _strip_casts(reduced).key != reduce.key:
      return _unsupported(RKRejectKind.UNSUPPORTED_REDUCTION, "affine CMAC scale is not direct", stored.op)
    scale = float(const.arg)
  value, prepare = _strip_casts(reduce.src[0]), None
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM and value.dtype is dtypes.half:
    value_index, source = value, RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
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
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value_index.src[0].src[0].arg)
  if len(red_axes) != len(reduce.src)-1 or set(out_axes) & set(red_axes) or \
     set(src_aff[0]) - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine CMAC axes do not form one static output/reduction partition", Ops.RANGE)
  if not 1 <= output_count <= 128 or not 2 <= input_count <= 512:
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
      src_index = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes))
      if not 0 <= src_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC input index is out of bounds", Ops.INDEX)
      selectors[out_index].append(src_index)
  if seen != set(range(output_count)):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC output has holes", Ops.INDEX)
  program = _sparse_cmac_pipeline(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), source, input_count, selectors, scale,
                                  () if prepare is None else prepare.scratch)
  if prepare is not None: program = RKProgram((RKDPUProgram(prepare.stages, program.scratch), *program.steps), program.scratch)
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

def lower_tiled_contract_result(sink:UOp) -> RKLowerResult:
  """Pack, execute, and unpack one bounded affine FP16 MxNxK contraction entirely on the NPU."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.ADD or len(reduce.src) < 2 or store.src[0].op is not Ops.INDEX or \
     store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half or _strip_casts(store.src[1]).key != reduce.key:
    return _not_applicable()
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
  if not 1 <= k <= 64 or lhs_count > 512 or rhs_count > 512 or output_count*k > RK_MAX_AFFINE_VISITS:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT,
      f"tiled CMAC surfaces are out={output_count},lhs={lhs_count},rhs={rhs_count},K={k}", reduce.op)
  records:list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
  for coordinates in product(*(range(ranges[axis]) for axis in out_axes)):
    point = dict(zip(out_axes, coordinates))
    output = out_aff[1] + sum(out_aff[0].get(axis, 0)*point[axis] for axis in out_axes)
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
    records.append((output, tuple(lhs_row), tuple(rhs_column)))
  lhs_rows = list(dict.fromkeys(row for _,row,_ in records))
  rhs_columns = list(dict.fromkeys(column for _,_,column in records))
  m, n, align_in = len(lhs_rows), len(rhs_columns), max(32, (k+31)&-32)
  pairs = {(lhs_rows.index(row), rhs_columns.index(column)) for _,row,column in records}
  if len(records) != output_count or {output for output,_,_ in records} != set(range(output_count)) or \
     pairs != set(product(range(m), range(n))): return _not_applicable()
  if not 1 <= m <= 64 or not 1 <= n <= 16 or m*align_in > 4096:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"tiled CMAC contraction is M={m},N={n},K={k}", reduce.op)
  a_selector = [entry for row in lhs_rows for entry in (([[source] if source >= 0 else [] for source in row]) +
                [[] for _ in range(align_in-k)])]
  b_selector:list[list[int]] = []
  for out_block in range(2):
    for in_block in range(align_in//32):
      for out_lane in range(16):
        for in_lane in range(32):
          out_channel, reduction_index = out_block*16+out_lane, in_block*32+in_lane
          source = rhs_columns[out_channel][reduction_index] if out_channel < n and reduction_index < k else -1
          b_selector.append([source] if source >= 0 else [])
  constant_bytes = sum(((count+15)//16)*32*max(32,(source_count+31)&-32)*2 for count,source_count in
                       ((len(a_selector),lhs_count),(len(b_selector),rhs_count),(output_count,m*64)))
  if constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"tiled CMAC selectors need {constant_bytes} bytes", reduce.op)

  steps:list[RKDPUProgram|RKContract|RKReduce] = []
  scratch:tuple[RKScratch, ...] = (RKScratch(len(a_selector)*2),)
  a_arg = RKArg(RKBufferKind.SCRATCH, 0)
  packed_a = _sparse_cmac_pipeline(a_arg, RKArg(RKBufferKind.ARG, lhs.src[0].arg.slot), lhs_count, a_selector, scratch=scratch)
  steps.extend(packed_a.steps)
  scratch = packed_a.scratch
  b_arg = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(len(b_selector)*2),)
  packed_b = _sparse_cmac_pipeline(b_arg, RKArg(RKBufferKind.ARG, rhs.src[0].arg.slot), rhs_count, b_selector, scratch=scratch)
  steps.extend(packed_b.steps)
  scratch = packed_b.scratch
  cmac_out = RKArg(RKBufferKind.SCRATCH, len(scratch))
  scratch += (RKScratch(m*128),)
  lhs_layout = RKLayout((m,k), (m,align_in), (align_in*2,2), dtypes.half, padding=((0,0),(0,align_in-k)))
  rhs_layout = RKLayout((n,k), (32,align_in), (align_in*2,2), dtypes.half,
                        padding=((0,32-n),(0,align_in-k)), kind=RKLayoutKind.CMAC_WEIGHT)
  out_layout = RKLayout((m,n), (m,64), (128,2), dtypes.half, padding=((0,0),(0,64-n)))
  steps.append(RKContract(RKTensorRef(cmac_out, out_layout), RKTensorRef(a_arg, lhs_layout), RKTensorRef(b_arg, rhs_layout), red_axes[0]))
  unpack:list[list[int]] = [[] for _ in range(output_count)]
  for output,row,rhs_key in records: unpack[output] = [lhs_rows.index(row)*64+rhs_columns.index(rhs_key)]
  dense = _sparse_cmac_pipeline(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), cmac_out, m*64, unpack, scratch=scratch)
  steps.extend(dense.steps)
  scratch = dense.scratch
  return _native(_finish_program(steps, scratch))

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

def lower_affine_max_result(sink:UOp) -> RKLowerResult:
  """Lower a static affine FP16 MAX by reformatting eight outputs into PPU channels per batch."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return _not_applicable()
  store, reduce = stores[0], reductions[0]
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) < 2 or _strip_casts(store.src[1]).key != reduce.key: return _not_applicable()
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM or store.src[0].dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "affine PPU MAX requires an FP16 output", store.src[0].op)
  value = _strip_casts(reduce.src[0])
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX input is not one FP16 surface", reduce.op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  if len(red_axes) != len(reduce.src)-1 or not out_axes or set(out_axes) & set(red_axes) or \
     set(src_aff[0]) - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine PPU MAX axes do not form a static output/reduction partition", Ops.RANGE)
  if not 2 <= output_count <= 128 or not 2 <= input_count <= 512:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"affine PPU MAX surface is {output_count}x{input_count}", reduce.op)
  reduction_count = math.prod(ranges[axis] for axis in red_axes)
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
      src_index = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in (*out_axes,*red_axes))
      if not 0 <= src_index < input_count:
        return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX input index is out of bounds", Ops.INDEX)
      selectors[out_index].append(src_index)
  if seen != set(range(output_count)) or any(len(row) != reduction_count for row in selectors):
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine PPU MAX output has holes", Ops.INDEX)

  align_in, surface_count = max(32, (input_count+31)&-32), pool_extent*8
  constant_bytes = ((output_count+7)//8)*((surface_count+15)//16)*32*align_in*2
  if constant_bytes > RK_MAX_CONSTANT_BYTES:
    return _unsupported(RKRejectKind.PLAN_STAGE_LIMIT, f"affine PPU MAX needs {constant_bytes} constant bytes", reduce.op)
  packed, hwc = RKArg(RKBufferKind.SCRATCH, 0), RKArg(RKBufferKind.SCRATCH, 1)
  hwc_elements = ((surface_count-1)//16)*16+32
  scratch = (RKScratch(align_in*2), RKScratch(hwc_elements*2))
  source = RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  steps:list = [RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                              RKALUStage(Ops.ADD, packed, source, 0.0, input_count)), scratch)]
  height, width = pool_shape
  for group_start in range(0, output_count, 8):
    channels = min(8, output_count-group_start)
    rows = [[selectors[group_start+min(channel, channels-1)][min(spatial, reduction_count-1)] for channel in range(8)]
            for spatial in range(pool_extent)]
    flat_rows = [rows[spatial][channel] for spatial in range(pool_extent) for channel in range(8)]
    for start in range(0, surface_count, 16):
      out_layout = RKLayout((1,min(16, surface_count-start)), (1,32), (64,2), dtypes.half,
                            padding=((0,0),(0,32-min(16, surface_count-start))))
      steps.append(RKContract(RKTensorRef(RKArg(hwc.kind, hwc.index, start*2), out_layout),
        _dense_half_ref(packed.index, (1,align_in), RKBufferKind.SCRATCH),
        _cmac_weight_ref(0, min(16, surface_count-start), align_in, RKBufferKind.CONSTANT, 32), reduce.arg[0],
        _cmac_selection_payload([[index] for index in flat_rows[start:start+16]], align_in, 32, 1.0)))
    out = RKTensorRef(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot, group_start*2),
                      RKLayout((1,1,8), (1,1,8), (16,16,2), dtypes.half))
    src = RKTensorRef(hwc, RKLayout((height,width,8), (height,width,8), (width*16,16,2), dtypes.half))
    steps.append(RKReduce(out, src, Ops.MAX, red_axes[0]))
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
  RKLowerer("broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_broadcast_alu_result),
  RKLowerer("reformat", lambda nodes:not _has_reduction(nodes), lower_reformat_result),
  RKLowerer("sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_add_reduce_result),
  RKLowerer("affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_affine_reduce_result),
  RKLowerer("ppu_reduce", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_reduce_result),
  RKLowerer("affine_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_affine_max_result),
  RKLowerer("global_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_global_max_result),
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

from tinygrad.renderer.rockchip.emit import (emit_dpu as emit_dpu, emit_contract as emit_contract, emit_program as emit_program,
  emit_reduce as emit_reduce)

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
    elif isinstance(result.plan, RKReduce): image = emit_reduce(result.plan)
    elif isinstance(result.plan, RKProgram): image = emit_program(result.plan)
    else: raise RuntimeError("invalid Rockchip lowering result")
    linear = UOp(Ops.LINEAR, src=tuple(u for u in params if u.addrspace is not AddrSpace.ALU))
    return UOp(Ops.PROGRAM, src=(ast, linear, UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=encode_image(image))),
               arg=info)
