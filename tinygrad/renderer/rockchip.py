from __future__ import annotations
import hashlib, math, os, struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import product
from typing import Callable, cast
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk, rockchip_lut as rklut
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.runtime.support.rockchip_telemetry import record as record_telemetry
from tinygrad.uop.ops import AddrSpace, Ops, ProgramInfo, UOp

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 3, 1
_HEADER, _STAGE = struct.Struct("<4sHHHHHHIII"), struct.Struct("<BBHIIII")
_RELOC, _SCRATCH = struct.Struct("<HHBBIqIH"), struct.Struct("<II")

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum):
  DPU = 1
  CMAC = 2
  PPU = 3
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
class RKLayoutKind(Enum):
  LINEAR = "linear"
  CMAC_WEIGHT = "cmac_weight"
@dataclass(frozen=True)
class RKArg:
  kind: RKBufferKind
  index: int
  addend: int = 0

_RK_ALU_OPS = frozenset((Ops.ADD, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.SUB))

@dataclass(frozen=True)
class RKALUStage:
  op: Ops
  dst: RKArg
  lhs: RKArg|float
  rhs: RKArg|float
  count: int
  out_dtype: DType = dtypes.half
  def __post_init__(self):
    if self.op not in _RK_ALU_OPS: raise ValueError(f"unsupported RK DPU ALU operation {self.op}")

@dataclass(frozen=True)
class RKMaskStage:
  dst: RKArg
  src: RKArg
  count: int

@dataclass(frozen=True)
class RKLUTStage:
  lut: RKLUTId
  dst: RKArg
  src: RKArg
  count: int

RKDPUStage = RKALUStage|RKMaskStage|RKLUTStage

@dataclass(frozen=True)
class RKScratch:
  size: int
  alignment: int = 4096

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...] = ()

@dataclass(frozen=True)
class RKLayout:
  logical_shape: tuple[int, ...]
  physical_shape: tuple[int, ...]
  strides_bytes: tuple[int, ...]
  dtype: DType
  base_offset: int = 0
  row_alignment: int = 16
  channel_alignment: int = 8
  padding: tuple[tuple[int, int], ...] = ()
  kind: RKLayoutKind = RKLayoutKind.LINEAR
  def __post_init__(self):
    rank = len(self.logical_shape)
    if len(self.physical_shape) != rank or len(self.strides_bytes) != rank or self.padding and len(self.padding) != rank:
      raise ValueError("RKLayout rank mismatch")
    if any(logical < 0 or physical < logical for logical,physical in zip(self.logical_shape, self.physical_shape)):
      raise ValueError("RKLayout physical shape does not contain its logical shape")
    if self.base_offset < 0 or self.row_alignment <= 0 or self.channel_alignment <= 0: raise ValueError("invalid RKLayout alignment")

@dataclass(frozen=True)
class RKTensorRef:
  buffer: RKArg
  layout: RKLayout

@dataclass(frozen=True)
class RKContract:
  out: RKTensorRef
  lhs: RKTensorRef
  rhs: RKTensorRef
  reduce_axis: int
  constants: bytes = b""

@dataclass(frozen=True)
class RKReduce:
  out: RKTensorRef
  src: RKTensorRef
  op: Ops
  reduce_axis: int

RKProgramStep = RKDPUProgram|RKContract|RKReduce

@dataclass(frozen=True)
class RKProgram:
  steps: tuple[RKProgramStep, ...]
  scratch: tuple[RKScratch, ...] = ()
  def __post_init__(self):
    if not self.steps: raise ValueError("Rockchip program has no steps")
    if any(isinstance(step, RKDPUProgram) and step.scratch and step.scratch != self.scratch for step in self.steps):
      raise ValueError("Rockchip step scratch does not match program resources")

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

def _sparse_cmac_pipeline(output:RKArg, source:RKArg, input_count:int, rows:list[list[int]], scale:float=1.0) -> RKProgram:
  """Materialize one static selector matrix as sequential, proven-width CMAC tasks."""
  align_in = max(32, (input_count+31)&-32)
  packed = RKArg(RKBufferKind.SCRATCH, 0)
  dpu = RKDPUProgram((RKALUStage(Ops.ADD, packed, 0.0, 0.0, align_in),
                      RKALUStage(Ops.ADD, packed, source, 0.0, input_count)), (RKScratch(align_in*2),))
  lhs_layout = RKLayout((1,input_count), (1,align_in), (align_in*2,2), dtypes.half,
                        padding=((0,0),(0,align_in-input_count)))
  contracts:list[RKContract] = []
  for start in range(0, len(rows), 16):
    count = min(16, len(rows)-start)
    out_layout = RKLayout((1,count), (1,32), (64,2), dtypes.half, padding=((0,0),(0,32-count)))
    contracts.append(RKContract(RKTensorRef(RKArg(output.kind, output.index, output.addend+start*2), out_layout),
      RKTensorRef(packed, lhs_layout), _cmac_weight_ref(0, count, align_in, RKBufferKind.CONSTANT, 32), 0,
      _cmac_selection_payload(rows[start:start+count], align_in, 32, scale)))
  return RKProgram((dpu, *contracts), dpu.scratch)

class RKRejectKind(Enum):
  UNSUPPORTED_INPUT_DTYPE = "unsupported_input_dtype"
  UNSUPPORTED_OUTPUT_DTYPE = "unsupported_output_dtype"
  UNSUPPORTED_ALU = "unsupported_alu"
  UNSUPPORTED_LAYOUT = "unsupported_layout"
  UNALIGNED_ROW = "unaligned_row"
  REQUIRES_REFORMAT = "requires_reformat"
  UNSUPPORTED_BROADCAST = "unsupported_broadcast"
  UNSUPPORTED_REDUCTION = "unsupported_reduction"
  UNSUPPORTED_CONTRACTION = "unsupported_contraction"
  UNSUPPORTED_DYNAMIC_PACK = "unsupported_dynamic_pack"
  PLAN_STAGE_LIMIT = "plan_stage_limit"
  LUT_DOMAIN_UNPROVEN = "lut_domain_unproven"
  NUMERICAL_CONTRACT = "numerical_contract"

@dataclass(frozen=True)
class RKReject:
  kind: RKRejectKind
  detail: str
  node_op: Ops|None = None
  fingerprint: tuple = ()

class RKLowerKind(Enum):
  NATIVE = "native"
  NOT_APPLICABLE = "not_applicable"
  UNSUPPORTED = "unsupported"

@dataclass(frozen=True)
class RKLowerResult:
  kind: RKLowerKind
  plan: RKDPUProgram|RKContract|RKReduce|RKProgram|None = None
  reject: RKReject|None = None
  def __post_init__(self):
    valid = {RKLowerKind.NATIVE:self.plan is not None and self.reject is None,
             RKLowerKind.NOT_APPLICABLE:self.plan is None and self.reject is None,
             RKLowerKind.UNSUPPORTED:self.plan is None and self.reject is not None}
    if not valid[self.kind]: raise ValueError(f"invalid {self.kind.value} Rockchip lowering result")

def _native(plan:RKDPUProgram|RKContract|RKReduce|RKProgram) -> RKLowerResult: return RKLowerResult(RKLowerKind.NATIVE, plan=plan)
def _not_applicable() -> RKLowerResult: return RKLowerResult(RKLowerKind.NOT_APPLICABLE)
def _unsupported(kind:RKRejectKind, detail:str, node_op:Ops|None=None) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(kind, detail, node_op))

@dataclass(frozen=True)
class RKReloc:
  stage: int
  word: int
  kind: RKBufferKind
  index: int
  addend: int = 0
  shift: int = 0
  mask: int = 0xffffffff
  field_shift: int = 0

@dataclass(frozen=True)
class RKStage:
  engine: RKEngine
  commands: tuple[int, ...]
  relocs: tuple[RKReloc, ...] = ()
  flags: int = 0

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  stages: tuple[RKStage, ...]
  scratch: tuple[RKScratch, ...] = ()
  constants: bytes = b""
  version: int = RKIMAGE_VERSION

def validate_image(image:RKImage) -> None:
  if image.version != RKIMAGE_VERSION: raise ValueError(f"unsupported RKImage version {image.version}")
  for stage_idx, stage in enumerate(image.stages):
    for reloc in stage.relocs:
      if reloc.stage != stage_idx or not 0 <= reloc.word < len(stage.commands): raise ValueError("invalid relocation location")
      if reloc.index < 0 or reloc.index >> 32 or not 0 <= reloc.shift < 64 or not 0 <= reloc.field_shift < 32 or reloc.mask >> 32:
        raise ValueError("invalid relocation field")
  for scratch in image.scratch:
    if scratch.size < 0 or scratch.alignment <= 0 or scratch.alignment & (scratch.alignment-1): raise ValueError("invalid scratch declaration")

def encode_image(image:RKImage) -> bytes:
  validate_image(image)
  commands:list[int] = []
  relocs:list[RKReloc] = []
  rows:list[tuple[int, ...]] = []
  for stage in image.stages:
    command_start, reloc_start = len(commands), len(relocs)
    commands.extend(stage.commands)
    relocs.extend(stage.relocs)
    rows.append((int(stage.engine), stage.flags, 0, command_start, len(stage.commands), reloc_start, len(stage.relocs)))
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(rows), len(relocs), len(image.scratch), 0,
                               len(commands), len(image.constants), 0))
  for row in rows: out += _STAGE.pack(*row)
  for r in relocs: out += _RELOC.pack(r.stage, r.word, int(r.kind), r.shift, r.index, r.addend, r.mask, r.field_shift)
  for scratch in image.scratch: out += _SCRATCH.pack(scratch.size, scratch.alignment)
  if commands: out += struct.pack(f"<{len(commands)}Q", *commands)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  if len(blob) < _HEADER.size: raise ValueError("truncated RKImage header")
  magic, version, target, stage_count, reloc_count, scratch_count, reserved, command_count, constant_size, reserved2 = _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or reserved or reserved2: raise ValueError("invalid RKImage header")
  expected = _HEADER.size + stage_count*_STAGE.size + reloc_count*_RELOC.size + scratch_count*_SCRATCH.size + command_count*8 + constant_size
  if expected != len(blob): raise ValueError("invalid RKImage size")
  off, rows = _HEADER.size, []
  for _ in range(stage_count):
    rows.append(_STAGE.unpack_from(blob, off))
    off += _STAGE.size
  relocs = []
  for _ in range(reloc_count):
    stage, word, kind, shift, index, addend, mask, field_shift = _RELOC.unpack_from(blob, off)
    off += _RELOC.size
    relocs.append(RKReloc(stage, word, RKBufferKind(kind), index, addend, shift, mask, field_shift))
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(scratch_count))
  off += scratch_count*_SCRATCH.size
  commands = struct.unpack_from(f"<{command_count}Q", blob, off) if command_count else ()
  off += command_count*8
  stages = []
  for idx, (engine, flags, row_reserved, command_start, command_len, reloc_start, reloc_len) in enumerate(rows):
    if row_reserved or command_start+command_len > command_count or reloc_start+reloc_len > reloc_count: raise ValueError("invalid RKImage stage")
    stages.append(RKStage(RKEngine(engine), tuple(commands[command_start:command_start+command_len]),
                          tuple(relocs[reloc_start:reloc_start+reloc_len]), flags))
    if any(r.stage != idx for r in stages[-1].relocs): raise ValueError("relocation belongs to wrong stage")
  image = RKImage(RKTarget(target), tuple(stages), scratch, blob[off:], version)
  validate_image(image)
  return image

def patch_image(image:RKImage, address:Callable[[RKBufferKind, int], int]) -> tuple[tuple[int, ...], ...]:
  validate_image(image)
  patched = [list(stage.commands) for stage in image.stages]
  for stage in image.stages:
    for reloc in stage.relocs:
      word, value = patched[reloc.stage][reloc.word], (patched[reloc.stage][reloc.word] >> 16) & 0xffffffff
      field = ((address(reloc.kind, reloc.index)+reloc.addend) >> reloc.shift) & reloc.mask
      field_mask = (reloc.mask << reloc.field_shift) & 0xffffffff
      patched[reloc.stage][reloc.word] = (word & ~0xffffffff0000) | (((value & ~field_mask) | ((field << reloc.field_shift) & field_mask)) << 16)
  return tuple(tuple(stage) for stage in patched)

@dataclass(frozen=True)
class _ALUExpr:
  op: Ops
  src: tuple[_ALUExpr|_MaskExpr|_LUTExpr|RKArg|float, _ALUExpr|_MaskExpr|_LUTExpr|RKArg|float]

@dataclass(frozen=True)
class _MaskExpr:
  src: tuple[_ALUExpr|_MaskExpr|_LUTExpr|RKArg]

@dataclass(frozen=True)
class _LUTExpr:
  lut: RKLUTId
  src: tuple[_ALUExpr|_MaskExpr|_LUTExpr|RKArg]

_Expr = _ALUExpr|_MaskExpr|_LUTExpr
_Value = _Expr|RKArg|float

def _sub(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _ALUExpr:
  return _ALUExpr(Ops.SUB, (lhs, rhs))

def _fp16(value:float) -> float: return struct.unpack("<e", struct.pack("<e", value))[0]
def _fp16_previous(value:float) -> float:
  bits = struct.unpack("<H", struct.pack("<e", value))[0]
  return struct.unpack("<e", struct.pack("<H", bits-1))[0]

def _nonzero_lut(lut:RKLUTId, source:_Expr|RKArg) -> _Expr:
  nonzero = _ALUExpr(Ops.MAX, (_MaskExpr((_sub(source, 0.0),)), _MaskExpr((_sub(0.0, source),))))
  return _ALUExpr(Ops.MUL, (_LUTExpr(lut, (source,)), nonzero))

def _round_expr(source:_Expr|RKArg|float) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  negative = _ALUExpr(Ops.MUL, (source, -1.0))
  magnitude = _ALUExpr(Ops.MAX, (source, negative))
  positive_mask, negative_mask = positive(source, 0.0), positive(0.0, source)
  sign = _sub(positive_mask, negative_mask)
  rounded = _ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.ROUNDOFF, (magnitude,)), sign))
  high = positive(magnitude, 65472.0)
  high_result = _ALUExpr(Ops.FDIV, (sign, _sub(1.0, high)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (rounded, _sub(1.0, high))), _ALUExpr(Ops.MUL, (high_result, high))))
  valid = _sub(1.0, _ALUExpr(Ops.MUL, (positive_mask, negative_mask)))
  return _ALUExpr(Ops.MUL, (selected, _ALUExpr(Ops.FDIV, (valid, valid))))

def _trunc_expr(source:_Expr|RKArg|float) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  rounded = _round_expr(source)
  decrement = _ALUExpr(Ops.MUL, (positive(rounded, source), positive(source, 0.0)))
  increment = _ALUExpr(Ops.MUL, (positive(source, rounded), positive(0.0, source)))
  return _ALUExpr(Ops.ADD, (_sub(rounded, decrement), increment))

def _exp_expr(source:_Expr|RKArg|float) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg|float, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  broad:_Expr = _LUTExpr(RKLUTId.EXP, (clamp(source, 2.0),))
  broad = _ALUExpr(Ops.MUL, (broad, _ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (positive(source, 0.0), 7.0))))))
  local = _LUTExpr(RKLUTId.EXP_LOCAL, (clamp(source, .25),))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.25), positive(.25, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))
  high, low = positive(source, 65472.0), positive(-65472.0, source)
  finite = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.FDIV, (selected, _sub(1.0, high))), _sub(1.0, low)))
  nan_denom = _sub(1.0, _ALUExpr(Ops.MUL, (high, low)))
  return _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.MUL, (finite, nan_denom)), nan_denom))

def _expm1_expr(source:_Expr|RKArg|float) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg|float, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  positive_input = positive(source, 0.0)
  broad_scale = _ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (positive_input, 7.0))))
  broad = _ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.EXPM1, (clamp(source, 2.0),)), broad_scale))
  local = _ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.EXPM1_LOCAL, (clamp(source, .25),)), _ALUExpr(Ops.ADD, (1.0, positive_input))))
  # WIP rejected on RK3588: x+x*x/2+x*x*x/6+x*x*x*x/24 simulates well but staged DPU rounding regresses CELU.
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.25), positive(.25, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))
  nonzero = _ALUExpr(Ops.MAX, (positive_input, positive(0.0, source)))
  return _ALUExpr(Ops.MUL, (selected, nonzero))

def _sigmoid_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  broad, local = _LUTExpr(RKLUTId.SIGMOID, (source,)), _LUTExpr(RKLUTId.SIGMOID_LOCAL, (source,))
  local_outside = _ALUExpr(Ops.MAX, (positive(-2.0, source), positive(source, 2.0)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, local_outside)),
    _ALUExpr(Ops.MUL, (local, _sub(1.0, local_outside)))))
  high, low = positive(source, 8.0), positive(-8.0, source)
  high_result = _ALUExpr(Ops.ADD, (selected, _ALUExpr(Ops.MUL, (_sub(1.0, selected), high))))
  bounded = _ALUExpr(Ops.MUL, (high_result, _sub(1.0, low)))
  nan_denom = _sub(1.0, _ALUExpr(Ops.MUL, (high, low)))
  return _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.MUL, (bounded, nan_denom)), nan_denom))

def _tanh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  broad, mid = _LUTExpr(RKLUTId.TANH, (clamp(source, 8.0),)), _LUTExpr(RKLUTId.TANH_MID, (clamp(source, .5),))
  local_source = clamp(source, .125)
  square = _ALUExpr(Ops.MUL, (local_source, local_source))
  local = _sub(local_source, _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (square, local_source)), 1/3)))
  mid_inside = _ALUExpr(Ops.MUL, (positive(source, -.5), positive(.5, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.125), positive(.125, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, mid_inside))), _ALUExpr(Ops.MUL, (mid, _sub(mid_inside, local_inside)))))
  selected = _ALUExpr(Ops.ADD, (selected, _ALUExpr(Ops.MUL, (local, local_inside))))
  high, low = positive(source, 8.0), positive(-8.0, source)
  high_result = _ALUExpr(Ops.ADD, (selected, _ALUExpr(Ops.MUL, (_sub(1.0, selected), high))))
  bounded = _ALUExpr(Ops.ADD, (high_result, _ALUExpr(Ops.MUL, (_sub(-1.0, high_result), low))))
  nan_denom = _sub(1.0, _ALUExpr(Ops.MUL, (high, low)))
  return _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.MUL, (bounded, nan_denom)), nan_denom))

def _asin_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  lower = _ALUExpr(Ops.MAX, (source, -1.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -1.0)), -1.0))
  local_source = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, -.125)), -1.0)), -.125)), -1.0))
  local_square = _ALUExpr(Ops.MUL, (local_source, local_source))
  local_cube = _ALUExpr(Ops.MUL, (local_source, local_square))
  local = _ALUExpr(Ops.ADD, (local_source, _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (local_cube, 1/6)),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (local_cube, local_square)), 3/40))))))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.125), positive(.125, source)))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  distance = _ALUExpr(Ops.MAX, (_sub(1.0, magnitude), 0.0))
  distance = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (distance, -1.0)), -.125)), -1.0))
  # WIP alternative kept for hardware study: sqrt(2*d)*(1+d/12+3*d*d/160+5*d*d*d/896).
  # It is accurate but needs enough generic stages to exceed RKImage's dependency mask on atan.
  edge = _nonzero_lut(RKLUTId.ASIN_EDGE, distance)
  sign = _sub(positive(source, 0.0), positive(0.0, source))
  edge_result = _ALUExpr(Ops.MUL, (_sub(math.pi/2, edge), sign))
  edge_inside = positive(magnitude, .875)
  broad_inside = _sub(1.0, _ALUExpr(Ops.MAX, (local_inside, edge_inside)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.ASIN, (bounded,)), broad_inside)),
    _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (local, local_inside)), _ALUExpr(Ops.MUL, (edge_result, edge_inside))))))
  invalid = _ALUExpr(Ops.MAX, (positive(source, 1.0), positive(-1.0, source)))
  valid = _sub(1.0, invalid)
  return _ALUExpr(Ops.MUL, (selected, _ALUExpr(Ops.FDIV, (valid, valid))))

def _acos_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  lower = _ALUExpr(Ops.MAX, (source, -1.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -1.0)), -1.0))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  distance = _ALUExpr(Ops.MAX, (_sub(1.0, magnitude), 0.0))
  distance = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (distance, -1.0)), -.125)), -1.0))
  edge = _nonzero_lut(RKLUTId.ASIN_EDGE, distance)
  positive_input, negative_input = positive(source, 0.0), positive(0.0, source)
  edge_result = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (edge, positive_input)),
    _ALUExpr(Ops.MUL, (_sub(math.pi, edge), negative_input))))
  edge_inside = positive(magnitude, .875)
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.ACOS, (bounded,)), _sub(1.0, edge_inside))),
    _ALUExpr(Ops.MUL, (edge_result, edge_inside))))
  invalid = _ALUExpr(Ops.MAX, (positive(source, 1.0), positive(-1.0, source)))
  valid = _sub(1.0, invalid)
  return _ALUExpr(Ops.MUL, (selected, _ALUExpr(Ops.FDIV, (valid, valid))))

def _atan_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  broad = _LUTExpr(RKLUTId.ATAN, (clamp(source, 8.0),))
  local_source = clamp(source, .3)
  local_square = _ALUExpr(Ops.MUL, (local_source, local_source))
  local_cube = _ALUExpr(Ops.MUL, (local_source, local_square))
  local = _ALUExpr(Ops.ADD, (_sub(local_source, _ALUExpr(Ops.MUL, (local_cube, 1/3))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (local_cube, local_square)), 1/5))))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.3), positive(.3, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))
  high, low = positive(source, 8.0), positive(-8.0, source)
  tail_inside = _ALUExpr(Ops.MAX, (high, low))
  safe_denominator = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (source, tail_inside)), _sub(1.0, tail_inside)))
  tail = _sub(_ALUExpr(Ops.MUL, (_sub(high, low), math.pi/2)), _ALUExpr(Ops.FDIV, (1.0, safe_denominator)))
  bounded = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (selected, _sub(1.0, tail_inside))), _ALUExpr(Ops.MUL, (tail, tail_inside))))
  valid = _sub(1.0, _ALUExpr(Ops.MUL, (high, low)))
  return _ALUExpr(Ops.MUL, (bounded, _ALUExpr(Ops.FDIV, (valid, valid))))

def _atanh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  broad = _LUTExpr(RKLUTId.ATANH, (clamp(source, .875),))
  local_source = clamp(source, .3)
  local_square = _ALUExpr(Ops.MUL, (local_source, local_source))
  local_cube = _ALUExpr(Ops.MUL, (local_source, local_square))
  local = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (local_source, _ALUExpr(Ops.MUL, (local_cube, 1/3)))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (local_cube, local_square)), 1/5))))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.3), positive(.3, source)))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  distance = _ALUExpr(Ops.MAX, (_sub(1.0, magnitude), 0.0))
  distance = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (distance, -1.0)), -.125)), -1.0))
  edge = _LUTExpr(RKLUTId.ATANH_EDGE, (distance,))
  sign = _sub(positive(source, 0.0), positive(0.0, source))
  edge_result = _ALUExpr(Ops.MUL, (edge, sign))
  edge_inside = positive(magnitude, .875)
  broad_inside = _sub(1.0, _ALUExpr(Ops.MAX, (local_inside, edge_inside)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, broad_inside)),
    _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (local, local_inside)), _ALUExpr(Ops.MUL, (edge_result, edge_inside))))))
  inside_domain, outside = positive(1.0, magnitude), positive(magnitude, 1.0)
  bounded = _ALUExpr(Ops.FDIV, (selected, inside_domain))
  valid = _sub(1.0, outside)
  return _ALUExpr(Ops.MUL, (bounded, _ALUExpr(Ops.FDIV, (valid, valid))))

def _asinh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  broad, mid = _LUTExpr(RKLUTId.ASINH, (clamp(source, 512.0),)), _LUTExpr(RKLUTId.ASINH_MID, (clamp(source, 8.0),))
  near = _LUTExpr(RKLUTId.ASINH_NEAR, (clamp(source, 2.0),))
  local_source = clamp(source, .3)
  square = _ALUExpr(Ops.MUL, (local_source, local_source))
  cube = _ALUExpr(Ops.MUL, (local_source, square))
  local = _ALUExpr(Ops.ADD, (_sub(local_source, _ALUExpr(Ops.MUL, (cube, 1/6))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (cube, square)), 3/40))))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.3), positive(.3, source)))
  near_inside = _ALUExpr(Ops.MUL, (positive(source, -2.0), positive(2.0, source)))
  mid_inside = _ALUExpr(Ops.MUL, (positive(source, -8.0), positive(8.0, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, mid_inside))),
    _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (mid, _sub(mid_inside, near_inside))), _ALUExpr(Ops.ADD,
      (_ALUExpr(Ops.MUL, (near, _sub(near_inside, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))))))
  infinite = _ALUExpr(Ops.MAX, (positive(source, 65472.0), positive(-65472.0, source)))
  return _ALUExpr(Ops.FDIV, (selected, _sub(1.0, infinite)))

def _acosh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  bounded_low = _ALUExpr(Ops.MAX, (source, 1.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (bounded_low, -1.0)), -512.0)), -1.0))
  distance = _sub(bounded, 1.0)
  mid_distance = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (distance, -1.0)), -8.0)), -1.0))
  edge_distance = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (distance, -1.0)), -.125)), -1.0))
  broad, mid = _LUTExpr(RKLUTId.ACOSH, (bounded,)), _LUTExpr(RKLUTId.ACOSH_MID, (mid_distance,))
  # Exact LUT input zero overflows on this steep edge payload; duplicate zero and address it one table step above zero.
  edge_value = _LUTExpr(RKLUTId.ACOSH_EDGE, (_ALUExpr(Ops.ADD, (edge_distance, 32/65504)),))
  edge = _ALUExpr(Ops.MUL, (edge_value, _MaskExpr((_sub(edge_distance, 0.0),))))
  edge_inside, mid_inside = positive(1.125, bounded), positive(9.0, bounded)
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, mid_inside))),
    _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (mid, _sub(mid_inside, edge_inside))), _ALUExpr(Ops.MUL, (edge, edge_inside))))))
  finite = _ALUExpr(Ops.FDIV, (selected, _sub(1.0, positive(source, 65472.0))))
  valid = positive(source, 1.0-2**-11)
  return _ALUExpr(Ops.MUL, (finite, _ALUExpr(Ops.FDIV, (valid, valid))))

def _sinh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  lower = _ALUExpr(Ops.MAX, (source, -2.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -2.0)), -1.0))
  local_lower = _ALUExpr(Ops.MAX, (source, -.3))
  local_source = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (local_lower, -1.0)), -.3)), -1.0))
  square, broad = _ALUExpr(Ops.MUL, (local_source, local_source)), _LUTExpr(RKLUTId.SINH, (bounded,))
  cube = _ALUExpr(Ops.MUL, (local_source, square))
  local = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (local_source, _ALUExpr(Ops.MUL, (cube, 1/6)))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (cube, square)), 1/120))))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.3), positive(.3, source)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))
  overflow = _ALUExpr(Ops.MAX, (positive(source, 11.78), positive(-11.78, source)))
  return _ALUExpr(Ops.FDIV, (selected, _sub(1.0, overflow)))

def _cosh_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  lower = _ALUExpr(Ops.MAX, (source, -2.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -2.0)), -1.0))
  overflow = _ALUExpr(Ops.MAX, (positive(source, 11.78), positive(-11.78, source)))
  return _ALUExpr(Ops.FDIV, (_LUTExpr(RKLUTId.COSH, (bounded,)), _sub(1.0, overflow)))

def _erf_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  local_source = clamp(source, .05)
  square = _ALUExpr(Ops.MUL, (local_source, local_source))
  cube = _ALUExpr(Ops.MUL, (local_source, square))
  series = _ALUExpr(Ops.ADD, (_sub(local_source, _ALUExpr(Ops.MUL, (cube, 1/3))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (cube, square)), 1/10))))
  local = _ALUExpr(Ops.MUL, (series, 2/math.sqrt(math.pi)))
  polynomial_inside = _ALUExpr(Ops.MUL, (positive(source, -.05), positive(.05, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.25), positive(.25, source)))
  # Q16 LUT conversion is undefined for its near-zero payload on DPU; that result is eagerly evaluated even when masked.
  # Move only the polynomial-selected input away from zero, where the LUT result is dead, without changing the live LUT domain.
  safe_local_source = _ALUExpr(Ops.ADD, (source, _ALUExpr(Ops.MUL, (polynomial_inside, .125))))
  broad, local_table = _LUTExpr(RKLUTId.ERF, (clamp(source, 2.0),)), _LUTExpr(RKLUTId.ERF_LOCAL, (clamp(safe_local_source, .25),))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))), _ALUExpr(Ops.ADD,
    (_ALUExpr(Ops.MUL, (local_table, _sub(local_inside, polynomial_inside))), _ALUExpr(Ops.MUL, (local, polynomial_inside))))))
  high, low = positive(source, 2.0), positive(-2.0, source)
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (selected, _sub(1.0, _ALUExpr(Ops.MAX, (high, low))))), _sub(high, low)))

def _softplus_expr(source:_Expr|RKArg, scale:float=1.0, input_scale:float=1.0) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  negative_abs = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0)))), -1.0))
  if math.isclose(input_scale, 3.0) and math.isclose(scale, 1/3):
    near_source = _ALUExpr(Ops.MAX, (negative_abs, -1.0))
    far_lower = _ALUExpr(Ops.MAX, (negative_abs, -2.0))
    far_source = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (far_lower, -1.0)), 2.5/3)), -1.0))
    near_inside, domain_inside = positive(negative_abs, -.834), positive(negative_abs, -2.01)
    negative_part = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.SOFTPLUS_DIV3_NEAR, (near_source,)), near_inside)),
      _ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.SOFTPLUS_DIV3_FAR, (far_source,)), _sub(domain_inside, near_inside)))))
    return _ALUExpr(Ops.ADD, (negative_part, _ALUExpr(Ops.MAX, (source, 0.0))))
  bounded = _ALUExpr(Ops.MAX, (negative_abs, -4.0))
  inside = positive(negative_abs, -4.01)
  residual:_Expr = _LUTExpr(RKLUTId.SOFTPLUS_NEG, (bounded,))
  positive_part:_Expr = _ALUExpr(Ops.MAX, (source, 0.0))
  if scale != 1.0:
    residual, positive_part = _ALUExpr(Ops.MUL, (residual, scale)), _ALUExpr(Ops.MUL, (positive_part, scale))
  negative_part = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.ADD, (residual, .5*scale)), inside))
  return _ALUExpr(Ops.ADD, (negative_part, positive_part))

def _mish_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  mid_inside = _ALUExpr(Ops.MUL, (positive(source, -.5), positive(.5, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.125), positive(.125, source)))
  broad_source = _ALUExpr(Ops.ADD, (source, mid_inside))
  lower = _ALUExpr(Ops.MAX, (broad_source, -2.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -2.0)), -1.0))
  mid_source = _ALUExpr(Ops.ADD, (source, _ALUExpr(Ops.MUL, (local_inside, .25))))
  local = _ALUExpr(Ops.MUL, (source, _ALUExpr(Ops.ADD, (.6, _ALUExpr(Ops.MUL, (source, _ALUExpr(Ops.ADD,
    (.32, _ALUExpr(Ops.MUL, (source, _ALUExpr(Ops.ADD, (-.016, _ALUExpr(Ops.MUL, (source, -86/1875))))))))))))))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.MISH, (bounded,)), _sub(1.0, mid_inside))),
    _ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.MISH_MID, (mid_source,)), _sub(mid_inside, local_inside))))),
    _ALUExpr(Ops.MUL, (local, local_inside))))
  nonzero = _ALUExpr(Ops.MAX, (positive(source, 0.0), positive(0.0, source)))
  return _ALUExpr(Ops.MUL, (selected, nonzero))

def _hardswish_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  broad_inside = _ALUExpr(Ops.MUL, (positive(source, -2.0), positive(2.0, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.125), positive(15/128, source)))
  broad_source = _ALUExpr(Ops.ADD, (source, local_inside))
  lower = _ALUExpr(Ops.MAX, (broad_source, -2.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -2.0)), -1.0))
  positive_plus = _ALUExpr(Ops.MAX, (_ALUExpr(Ops.ADD, (source, 3.0)), 0.0))
  relu6 = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (positive_plus, -1.0)), -6.0)), -1.0))
  fallback = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (source, relu6)), 1/6))
  local = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (source, source)), 1/6)), _ALUExpr(Ops.MUL, (source, .5))))
  positive_outer = positive(source, 2.0)
  positive_curve = _ALUExpr(Ops.MUL, (positive_outer, positive(3.0, source)))
  positive_tail = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (local, positive_curve)),
    _ALUExpr(Ops.MUL, (source, _sub(positive_outer, positive_curve)))))
  negative_fallback = _ALUExpr(Ops.MUL, (fallback, _ALUExpr(Ops.MUL, (_sub(1.0, broad_inside), _sub(1.0, positive_outer)))))
  wide = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_LUTExpr(RKLUTId.HARDSWISH, (bounded,)), broad_inside)),
    _ALUExpr(Ops.ADD, (negative_fallback, positive_tail))))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (wide, _sub(1.0, local_inside))), _ALUExpr(Ops.MUL, (local, local_inside))))

def _quick_gelu_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  inside = _ALUExpr(Ops.MUL, (positive(source, -2.0), positive(2.0, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -2.0), positive(-1.0, source)))
  poly_inside = _ALUExpr(Ops.MUL, (positive(source, -.16), positive(.16, source)))
  lower = _ALUExpr(Ops.MAX, (source, -2.0))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -2.0)), -1.0))
  safe_source = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (bounded, local_inside)), poly_inside))
  broad = _LUTExpr(RKLUTId.QUICK_GELU, (safe_source,))
  local_input = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.ADD, (source, 1.5)), 4.0))
  local = _LUTExpr(RKLUTId.QUICK_GELU_LOCAL, (local_input,))
  polynomial = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (bounded, .5)),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (bounded, bounded)), .4253))))
  broad_mask = _sub(_sub(inside, local_inside), poly_inside)
  fallback = _ALUExpr(Ops.MUL, (source, _sigmoid_expr(_ALUExpr(Ops.MUL, (source, 1.702)))))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, broad_mask)),
    _ALUExpr(Ops.MUL, (local, local_inside)))), _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (polynomial, poly_inside)),
    _ALUExpr(Ops.MUL, (fallback, _sub(1.0, inside)))))))

def _gelu_expr(source:_Expr|RKArg, approximate_tanh:bool) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  def clamp(value:_Expr|RKArg, limit:float) -> _Expr:
    lower = _ALUExpr(Ops.MAX, (value, -limit))
    return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lower, -1.0)), -limit)), -1.0))
  range_inside = _ALUExpr(Ops.MUL, (positive(source, -4.0), positive(4.0, source)))
  local_inside = _ALUExpr(Ops.MUL, (positive(source, -.5), positive(.5, source)))
  poly_inside = _ALUExpr(Ops.MUL, (positive(source, -.04), positive(.04, source)))
  broad_id, local_id = (RKLUTId.GELU_TANH, RKLUTId.GELU_TANH_LOCAL) if approximate_tanh else \
    (RKLUTId.GELU_EXACT, RKLUTId.GELU_EXACT_LOCAL)
  broad = _LUTExpr(broad_id, (_ALUExpr(Ops.ADD, (clamp(source, 4.0), local_inside)),))
  local_input = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (clamp(source, .5), 8.0)), poly_inside))
  local = _ALUExpr(Ops.MUL, (_LUTExpr(local_id, (local_input,)), .5))
  poly_source = clamp(source, .04)
  polynomial = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (poly_source, .5)),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (poly_source, poly_source)), 1/math.sqrt(2*math.pi)))))
  positive_scale = _ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (positive(source, 0.0), 3.0))))
  broad_selected = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (broad, positive_scale)), _sub(range_inside, local_inside)))
  local_selected = _ALUExpr(Ops.MUL, (local, _sub(local_inside, poly_inside)))
  interior = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (broad_selected, local_selected)),
    _ALUExpr(Ops.MUL, (polynomial, poly_inside))))
  fallback = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, 0.0)), _sub(1.0, range_inside)))
  return _ALUExpr(Ops.ADD, (interior, fallback))

def _elu_expr(source:_Expr|RKArg, negative_scale:float, positive_scale:float) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  if negative_scale < .2: broad_id, local_id, broad_gain, local_gain = RKLUTId.ELU01, RKLUTId.ELU01_LOCAL, 8.0, 16.0
  elif negative_scale > 1.5: broad_id, local_id, broad_gain, local_gain = RKLUTId.SELU, RKLUTId.SELU_LOCAL, .5, 1.0
  else: broad_id, local_id, broad_gain, local_gain = RKLUTId.ELU1, RKLUTId.ELU1_LOCAL, 1.0, 2.0
  broad_input = _ALUExpr(Ops.MAX, (source, -8.0))
  local_input = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, -.5)), 4.0))
  broad, local = _LUTExpr(broad_id, (broad_input,)), _LUTExpr(local_id, (local_input,))
  below, local_below, poly_below, negative = (positive(x, source) for x in (-8.0, -.5, -.03, 0.0))
  broad_mask, local_mask = _sub(local_below, below), _sub(poly_below, local_below)
  poly_mask, positive_mask = _sub(negative, poly_below), _sub(1.0, negative)
  poly_input = _ALUExpr(Ops.MAX, (source, -.03))
  polynomial = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (poly_input, negative_scale)),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (poly_input, poly_input)), negative_scale/2))))
  negative_sum = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL,
    (_ALUExpr(Ops.MUL, (broad, 1/broad_gain)), broad_mask)), _ALUExpr(Ops.MUL,
    (_ALUExpr(Ops.MUL, (local, 1/local_gain)), local_mask)))), _ALUExpr(Ops.MUL, (polynomial, poly_mask))))
  tails = _ALUExpr(Ops.ADD, (negative_sum, _ALUExpr(Ops.MUL, (-negative_scale, below))))
  positive_result = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, 0.0)), positive_scale)), positive_mask))
  return _ALUExpr(Ops.ADD, (tails, positive_result))

def _celu_expr(source:_Expr|RKArg, alpha:int) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  broad_id, local_id = {2:(RKLUTId.CELU2,RKLUTId.CELU2_LOCAL), 3:(RKLUTId.CELU3,RKLUTId.CELU3_LOCAL),
                        4:(RKLUTId.CELU4,RKLUTId.CELU4_LOCAL)}[alpha]
  broad, local = _LUTExpr(broad_id, (_ALUExpr(Ops.MAX, (source, -4.0)),)), \
                 _LUTExpr(local_id, (_ALUExpr(Ops.MAX, (source, -.5)),))
  below, local_below, poly_below, negative = (positive(x, source) for x in (-4.0, -.5, -.03, 0.0))
  broad_mask, local_mask = _sub(local_below, below), _sub(poly_below, local_below)
  poly_mask, positive_mask = _sub(negative, poly_below), _sub(1.0, negative)
  poly_input = _ALUExpr(Ops.MAX, (source, -.03))
  polynomial = _ALUExpr(Ops.ADD, (poly_input, _ALUExpr(Ops.MUL,
    (_ALUExpr(Ops.MUL, (poly_input, poly_input)), 1/(2*alpha)))))
  negative_sum = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, broad_mask)),
    _ALUExpr(Ops.MUL, (local, local_mask)))), _ALUExpr(Ops.MUL, (polynomial, poly_mask))))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (negative_sum, _ALUExpr(Ops.MUL, (-float(alpha), below)))),
                            _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (source, 0.0)), positive_mask))))

def _sqrt_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  refined:_Expr = _LUTExpr(RKLUTId.SQRT, (source,))
  for _ in range(3): refined = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.ADD, (refined, _ALUExpr(Ops.FDIV, (source, refined)))), .5))
  high, negative = positive(source, 65472.0), positive(0.0, source)
  nonzero = _ALUExpr(Ops.MAX, (positive(source, 0.0), negative))
  not_number = _ALUExpr(Ops.MUL, (positive(source, 0.0), negative))
  positive_result = _ALUExpr(Ops.FDIV, (refined, _sub(1.0, high)))
  zero_result = _ALUExpr(Ops.MUL, (positive_result, nonzero))
  valid = _sub(1.0, _ALUExpr(Ops.MAX, (negative, not_number)))
  return _ALUExpr(Ops.MUL, (zero_result, _ALUExpr(Ops.FDIV, (valid, valid))))

def _rsqrt_expr(source:_Expr|RKArg) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  greater_zero, below_1, below_2 = positive(source, 0.0), positive(.0625, source), positive(.00390625, source)
  low_1, low_2 = _ALUExpr(Ops.MUL, (greater_zero, below_1)), _ALUExpr(Ops.MUL, (greater_zero, below_2))
  factors = tuple(_ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (mask, 15.0)))) for mask in (low_1, low_2))
  scaled = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (source, factors[0])), factors[1]))
  seed = _LUTExpr(RKLUTId.RSQRT, (scaled,))
  safe = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (scaled, -1.0)), -4.0)), -1.0))
  correction = _sub(1.5, _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (safe, _ALUExpr(Ops.MUL, (seed, seed)))), .5)))
  refined = _ALUExpr(Ops.MUL, (seed, correction))
  out_factors = tuple(_ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (mask, 3.0)))) for mask in (low_1, low_2))
  scaled_out = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (refined, out_factors[0])), out_factors[1]))
  negative, high = positive(0.0, source), positive(source, 65472.0)
  nonzero = _ALUExpr(Ops.MAX, (greater_zero, negative))
  finite = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.FDIV, (scaled_out, nonzero)), _sub(1.0, high)))
  not_number = _ALUExpr(Ops.MUL, (greater_zero, negative))
  valid = _sub(1.0, _ALUExpr(Ops.MAX, (negative, not_number)))
  return _ALUExpr(Ops.MUL, (finite, _ALUExpr(Ops.FDIV, (valid, valid))))

def _log2_expr(source:_Expr|RKArg, scale:float=1.0) -> _Expr:
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  low_1, low_2 = positive(.25, source), positive(.015625, source)
  factor = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (1.0, _ALUExpr(Ops.MUL, (low_1, 15.0)))), _ALUExpr(Ops.MUL, (low_2, 240.0))))
  normalized = _ALUExpr(Ops.MUL, (source, factor))
  offset = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.ADD, (low_1, low_2)), -4.0*scale))
  bounded_low = _ALUExpr(Ops.MAX, (normalized, .25))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (bounded_low, -1.0)), -4.0)), -1.0))
  centered = _sub(bounded, 1.0)
  broad_id, local_id = (RKLUTId.LOG10, RKLUTId.LOG10_LOCAL) if math.isclose(scale, math.log10(2)) else \
    (RKLUTId.LOG2, RKLUTId.LOG2_LOCAL)
  broad = _LUTExpr(broad_id, (bounded,))
  local = _ALUExpr(Ops.MUL, (_LUTExpr(local_id, (_ALUExpr(Ops.MUL, (centered, 12.5)),)), .25))
  local_inside = _ALUExpr(Ops.MUL, (positive(bounded, .85), positive(1.15, bounded)))
  near_inside = _ALUExpr(Ops.MUL, (positive(centered, -.02), positive(.02, centered)))
  polynomial = _ALUExpr(Ops.MUL, (_sub(centered, _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (centered, centered)), .5))),
    scale*math.log2(math.e)))
  mantissa = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (broad, _sub(1.0, local_inside))),
    _ALUExpr(Ops.MUL, (local, _sub(local_inside, near_inside))))), _ALUExpr(Ops.MUL, (polynomial, near_inside))))
  corrected = _ALUExpr(Ops.ADD, (mantissa, offset))
  negative, greater_zero, high = positive(0.0, source), positive(source, 0.0), positive(source, 65472.0)
  nonzero = _ALUExpr(Ops.MAX, (greater_zero, negative))
  finite = _ALUExpr(Ops.FDIV, (corrected, _sub(nonzero, high)))
  valid = _sub(1.0, negative)
  return _ALUExpr(Ops.MUL, (finite, _ALUExpr(Ops.FDIV, (valid, valid))))

def _pow8_expr(source:_Expr|RKArg) -> _Expr:
  """Match float32-power accuracy with two normalized LUT ranges and a full-domain DPU fallback."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  square = _ALUExpr(Ops.MUL, (source, source))
  fourth = _ALUExpr(Ops.MUL, (square, square))
  repeated = _ALUExpr(Ops.MUL, (fourth, fourth))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  above_half, above_one, above_two = positive(magnitude, .5), positive(magnitude, 1.0), positive(magnitude, 2.0)
  bands:tuple[_Value,_Value,_Value,_Value] = (_sub(1.0, above_half), _sub(above_half, above_one), _sub(above_one, above_two), above_two)
  def weighted(weights:tuple[float,float,float,float]) -> _Expr:
    terms = tuple(_ALUExpr(Ops.MUL, (band, weight)) for band,weight in zip(bands, weights))
    return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (terms[0], terms[1])), _ALUExpr(Ops.ADD, (terms[2], terms[3]))))
  normalized = _ALUExpr(Ops.MUL, (magnitude, weighted((4.0, 2.0, 1.0, .5))))
  factor = weighted((2.0**-16, 2.0**-8, 1.0, 256.0))
  low, high = _LUTExpr(RKLUTId.POW8, (normalized,)), _LUTExpr(RKLUTId.POW8_HIGH, (normalized,))
  high_mask = positive(normalized, _fp16(math.sqrt(2.0)))
  normalized_power = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (low, _sub(1.0, high_mask))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (high, 256.0)), high_mask))))
  scaled = _ALUExpr(Ops.MUL, (normalized_power, factor))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (scaled, -1.0)), -65504.0)), -1.0))
  valid = _ALUExpr(Ops.MUL, (positive(magnitude, .25), positive(4.0, magnitude)))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (bounded, valid)), _ALUExpr(Ops.MUL, (repeated, _sub(1.0, valid)))))

def _pow55_expr(source:_Expr|RKArg) -> _Expr:
  """Evaluate x**5.5 with normalized Q11/Q15 LUT ranges and preserve the generic full-domain result."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  square = _ALUExpr(Ops.MUL, (source, source))
  fourth = _ALUExpr(Ops.MUL, (square, square))
  fallback = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (fourth, source)), _sqrt_expr(source)))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  above_half, above_one, above_two = positive(magnitude, .5), positive(magnitude, 1.0), positive(magnitude, 2.0)
  bands:tuple[_Value,_Value,_Value,_Value] = (_sub(1.0, above_half), _sub(above_half, above_one), _sub(above_one, above_two), above_two)
  def weighted(weights:tuple[float,float,float,float]) -> _Expr:
    terms = tuple(_ALUExpr(Ops.MUL, (band, weight)) for band,weight in zip(bands, weights))
    return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (terms[0], terms[1])), _ALUExpr(Ops.ADD, (terms[2], terms[3]))))
  normalized = _ALUExpr(Ops.MUL, (magnitude, weighted((4.0, 2.0, 1.0, .5))))
  factor = weighted((2.0**-11, 2.0**-5.5, 1.0, 2.0**5.5))
  low, local, high = (_LUTExpr(lut, (normalized,)) for lut in (RKLUTId.POW55, RKLUTId.POW55_LOCAL, RKLUTId.POW55_HIGH))
  local_mask = positive(1.125, normalized)
  low_range = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (local, 2.0)), local_mask)),
    _ALUExpr(Ops.MUL, (low, _sub(1.0, local_mask)))))
  high_mask = positive(normalized, _fp16_previous(_fp16_previous(16.0**(1.0/5.5))))
  normalized_power = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (low_range, _sub(1.0, high_mask))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (high, 2.0**5.5)), high_mask))))
  scaled = _ALUExpr(Ops.MUL, (normalized_power, factor))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (scaled, -1.0)), -65504.0)), -1.0))
  valid = _ALUExpr(Ops.MUL, (positive(magnitude, .25), positive(4.0, magnitude)))
  selected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (bounded, valid)), _ALUExpr(Ops.MUL, (fallback, _sub(1.0, valid)))))
  negative, negative_inf = positive(0.0, source), positive(-65472.0, source)
  invalid_denom = _sub(1.0, _sub(negative, negative_inf))
  return _ALUExpr(Ops.MUL, (selected, _ALUExpr(Ops.FDIV, (invalid_denom, invalid_denom))))

def _pow_neg55_expr(source:_Expr|RKArg) -> _Expr:
  """Evaluate x**-5.5 from x directly, avoiding the rounded reciprocal used by the generic decomposition."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  reciprocal = _ALUExpr(Ops.FDIV, (1.0, source))
  square = _ALUExpr(Ops.MUL, (reciprocal, reciprocal))
  fourth = _ALUExpr(Ops.MUL, (square, square))
  fallback = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (fourth, reciprocal)), _sqrt_expr(reciprocal)))
  magnitude = _ALUExpr(Ops.MAX, (source, _ALUExpr(Ops.MUL, (source, -1.0))))
  above_half, above_one = positive(magnitude, .5), positive(magnitude, 1.0)
  above_two, above_four = positive(magnitude, 2.0), positive(magnitude, 4.0)
  bands:tuple[_Value,_Value,_Value,_Value,_Value] = (_sub(1.0, above_half), _sub(above_half, above_one),
    _sub(above_one, above_two), _sub(above_two, above_four), above_four)
  def weighted(weights:tuple[float,float,float,float,float]) -> _Expr:
    terms = tuple(_ALUExpr(Ops.MUL, (band, weight)) for band,weight in zip(bands, weights))
    return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (terms[0], terms[1])),
      _ALUExpr(Ops.ADD, (terms[2], terms[3])))), terms[4]))
  normalized = _ALUExpr(Ops.MUL, (magnitude, weighted((4.0, 2.0, 1.0, .5, .25))))
  factor = weighted((2.0**11, 2.0**5.5, 1.0, 2.0**-5.5, 2.0**-11))
  low = _LUTExpr(RKLUTId.POW_NEG55_LOW, (normalized,))
  shifted = _sub(normalized, 1.0)
  high, far = _LUTExpr(RKLUTId.POW_NEG55_HIGH, (shifted,)), _LUTExpr(RKLUTId.POW_NEG55_FAR, (shifted,))
  far_mask = positive(normalized, 1.375)
  high_range = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (high, _sub(1.0, far_mask))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (far, .25)), far_mask))))
  high_mask = positive(normalized, 1.0)
  normalized_power = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (low, _sub(1.0, high_mask))), _ALUExpr(Ops.MUL, (high_range, high_mask))))
  scaled = _ALUExpr(Ops.MUL, (normalized_power, factor))
  bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (scaled, -1.0)), -65504.0)), -1.0))
  above_finite, below_eight = positive(magnitude, _fp16(.133056640625)), positive(8.0, magnitude)
  valid = _ALUExpr(Ops.MUL, (above_finite, below_eight))
  fallback_bounded = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (fallback, -1.0)), -65504.0)), -1.0))
  combined = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (bounded, valid)), _ALUExpr(Ops.MUL, (fallback_bounded, _sub(1.0, valid)))))
  overflow = _sub(1.0, above_finite)
  overflow_result = _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.ADD, (combined, overflow)), above_finite))
  above_first_finite = positive(magnitude, _fp16(.1331787109375))
  first_finite = _sub(above_finite, above_first_finite)
  rounded = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (overflow_result, _sub(1.0, first_finite))),
    _ALUExpr(Ops.MUL, (first_finite, 65408.0))))
  negative, negative_inf = positive(0.0, source), positive(-65472.0, source)
  invalid_denom = _sub(1.0, _sub(negative, negative_inf))
  return _ALUExpr(Ops.MUL, (rounded, _ALUExpr(Ops.FDIV, (invalid_denom, invalid_denom))))

def _exp2_expr(source:_Expr|RKArg) -> _Expr:
  """Preserve IEEE infinity behavior around the finite-domain hardware EXP2 LUT."""
  base = _LUTExpr(RKLUTId.EXP2, (source,))
  positive_inf, negative_inf = _MaskExpr((_sub(source, 65504.0),)), _MaskExpr((_sub(-65504.0, source),))
  return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.FDIV, (base, _sub(1.0, positive_inf))), _sub(1.0, negative_inf)))

def _pow_base55_expr(source:_Expr|RKArg) -> _Expr:
  """Evaluate 5.5**x with two Q15 ranges and the generic NPU EXP2 path outside [-2,2]."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  low, high = _LUTExpr(RKLUTId.POW_BASE55_LOW, (source,)), _LUTExpr(RKLUTId.POW_BASE55_HIGH, (source,))
  high_mask = positive(source, 0.0)
  corrected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (low, _sub(1.0, high_mask))),
    _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (high, 32.0)), high_mask))))
  inside = _ALUExpr(Ops.MUL, (positive(source, -2.001953125), positive(2.001953125, source)))
  fallback = _exp2_expr(_ALUExpr(Ops.MUL, (source, math.log2(5.5))))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (corrected, inside)), _ALUExpr(Ops.MUL, (fallback, _sub(1.0, inside)))))

def _pow_negative_base55_expr(source:_Expr|RKArg) -> _Expr:
  """Evaluate (-5.5)**x with native roundoff-LUT truncation, integer validity, and parity."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  truncated = _trunc_expr(source)
  half_truncated = _trunc_expr(_ALUExpr(Ops.MUL, (truncated, .5)))
  remainder = _sub(truncated, _ALUExpr(Ops.MUL, (half_truncated, 2.0)))
  odd = _ALUExpr(Ops.MAX, (remainder, _ALUExpr(Ops.MUL, (remainder, -1.0))))
  sign = _sub(1.0, _ALUExpr(Ops.MUL, (odd, 2.0)))
  noninteger = _ALUExpr(Ops.MAX, (positive(source, truncated), positive(truncated, source)))
  valid = _sub(1.0, noninteger)
  return _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (_pow_base55_expr(source), sign)), _ALUExpr(Ops.FDIV, (valid, valid))))

def _pow_base8_expr(source:_Expr|RKArg) -> _Expr:
  """Evaluate 8**x with four Q15 output-scale bands and native EXP2 outside [-2,2]."""
  def positive(lhs:_Expr|RKArg|float, rhs:_Expr|RKArg|float) -> _MaskExpr: return _MaskExpr((_sub(lhs, rhs),))
  above_negative_one, above_zero, above_one = positive(source, -1.0), positive(source, 0.0), positive(source, 1.0)
  bands:tuple[_Value,_Value,_Value,_Value] = (_sub(1.0, above_negative_one), _sub(above_negative_one, above_zero),
    _sub(above_zero, above_one), above_one)
  tables = tuple(_LUTExpr(lut, (source,)) for lut in
    (RKLUTId.POW_BASE8_FAR_LOW, RKLUTId.POW_BASE8_LOW, RKLUTId.POW_BASE8_HIGH, RKLUTId.POW_BASE8_FAR_HIGH))
  terms = tuple(_ALUExpr(Ops.MUL, (_ALUExpr(Ops.MUL, (table, decode)), band)) for table,decode,band in
    zip(tables, (.125, 1.0, 8.0, 64.0), bands))
  corrected = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.ADD, (terms[0], terms[1])), _ALUExpr(Ops.ADD, (terms[2], terms[3]))))
  inside = _ALUExpr(Ops.MUL, (positive(source, -2.001953125), positive(2.001953125, source)))
  fallback = _exp2_expr(_ALUExpr(Ops.MUL, (source, 3.0)))
  return _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (corrected, inside)), _ALUExpr(Ops.MUL, (fallback, _sub(1.0, inside)))))

def _unwrap_same_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype is u.src[0].dtype: u = u.src[0]
  return u

def _unwrap_fp_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype in (dtypes.half, dtypes.float) and u.src[0].dtype in (dtypes.half, dtypes.float): u = u.src[0]
  return _unwrap_same_cast(u)

def _canonical_mul_power(u:UOp, power:float, reciprocal:bool=False) -> UOp|None:
  """Recognize a multiplication tree containing exactly `power` copies of one FP16 indexed value."""
  u = _unwrap_fp_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.half]
  if len(indexes) != 1: return None
  source = indexes[0]
  if reciprocal:
    reciprocals = [x for x in u.toposort() if x.op is Ops.RECIPROCAL and len(x.src) == 1 and _unwrap_fp_cast(x.src[0]).key == source.key]
    if len(reciprocals) != 1: return None
    base = reciprocals[0]
  else: base = source
  def exponent(node:UOp) -> float|None:
    node = _unwrap_fp_cast(node)
    if node.key == base.key: return 1.0
    if node.op is Ops.SQRT and len(node.src) == 1 and _unwrap_fp_cast(node.src[0]).key == base.key: return .5
    if node.op is not Ops.MUL or len(node.src) != 2: return None
    lhs, rhs = exponent(node.src[0]), exponent(node.src[1])
    return None if lhs is None or rhs is None else lhs+rhs
  return source if exponent(u) == power else None

def _canonical_sigmoid(u:UOp) -> tuple[UOp,float]|None:
  """Recognize 1/(1+exp2(-log2(e)*x))."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.RECIPROCAL or (denominator:=_unwrap_same_cast(u.src[0])).op is not Ops.ADD: return None
  one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  exponential = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.EXP2), None)
  if one is None or float(one.arg) != 1 or exponential is None or (scaled:=_unwrap_same_cast(exponential.src[0])).op is not Ops.MUL: return None
  source = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None)
  return (source, factor/-math.log2(math.e)) if source is not None and factor is not None and math.isfinite(factor) else None

def _canonical_tanh(u:UOp) -> UOp|None:
  """Recognize 2*sigmoid(2*x)-1."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.ADD: return None
  for scaled, offset in (u.src, u.src[::-1]):
    scaled, offset = _unwrap_same_cast(scaled), _unwrap_same_cast(offset)
    if offset.op is not Ops.CONST or float(offset.arg) != -1 or scaled.op is not Ops.MUL: continue
    two = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    sigmoid_u = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
    if two is None or float(two.arg) != 2 or sigmoid_u is None or (sigmoid:=_canonical_sigmoid(sigmoid_u)) is None: continue
    if math.isclose(sigmoid[1], 2.0): return sigmoid[0]
  return None

def _canonical_expm1(u:UOp) -> tuple[UOp,float,float]|None:
  """Recognize +/- (exp(+/- x)-1) after EXP has become EXP2."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.ADD: return None
  for exponential, constant in (u.src, u.src[::-1]):
    exponential, constant = _unwrap_fp_cast(exponential), _unwrap_same_cast(constant)
    polarity = 1.0
    if exponential.op is Ops.MUL:
      neg = next((_unwrap_same_cast(x) for x in exponential.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      if neg is None or float(neg.arg) != -1: continue
      exp_source = exponential.src[0] if _unwrap_same_cast(exponential.src[1]).key == neg.key else exponential.src[1]
      exponential, polarity = _unwrap_fp_cast(exp_source), -1.0
    if constant.op is not Ops.CONST or float(constant.arg) != -polarity or exponential.op is not Ops.EXP2: continue
    scaled = _unwrap_fp_cast(exponential.src[0])
    if scaled.op is not Ops.MUL: continue
    factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None)
    source = next((_unwrap_same_cast(x) for x in scaled.src if x.op is not Ops.CONST), None)
    if source is not None and factor is not None and math.isclose(abs(factor), math.log2(math.e)): return source, math.copysign(1.0, factor), polarity
  return None

def _canonical_sign(u:UOp) -> UOp|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.WHERE: return None
  cond, nonzero, zero = (_unwrap_same_cast(x) for x in u.src)
  if cond.op is not Ops.CMPNE or zero.op is not Ops.CONST or float(zero.arg) != 0 or nonzero.op is not Ops.WHERE: return None
  less, negative, positive = (_unwrap_same_cast(x) for x in nonzero.src)
  compared = tuple(_unwrap_same_cast(x) for x in cond.src)
  for data, zero_cmp in (compared, compared[::-1]):
    if zero_cmp.op is not Ops.CONST or float(zero_cmp.arg) != 0 or less.op is not Ops.CMPLT: continue
    less_lhs, less_rhs = (_unwrap_same_cast(x) for x in less.src)
    if (less_lhs.key == data.key and less_rhs.op is Ops.CONST and float(less_rhs.arg) == 0 and
        negative.op is Ops.CONST and float(negative.arg) == -1 and positive.op is Ops.CONST and float(positive.arg) == 1): return data
  return None

def _canonical_abs(u:UOp) -> UOp|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for data, sign in (u.src, u.src[::-1]):
    data, sign = _unwrap_same_cast(data), _unwrap_same_cast(sign)
    sign_input = _canonical_sign(sign)
    if sign_input is not None and sign_input.key == data.key: return data
  return None

def _canonical_asin(u:UOp) -> UOp|None:
  """Recognize tinygrad's sign(x)*(pi/2-sqrt(1-abs(x))*poly(abs(x)))."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  coefficients = (-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                  -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)
  for sign, body in (u.src, u.src[::-1]):
    source = _canonical_sign(sign)
    if source is None: continue
    nodes = _unwrap_same_cast(body).toposort()
    constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
    if not any(x.op is Ops.SQRT for x in nodes) or not all(any(math.isclose(x, value) for x in constants) for value in coefficients): continue
    if any((absolute:=_canonical_abs(x)) is not None and absolute.key == source.key for x in nodes): return source
  return None

def _canonical_acos(u:UOp) -> UOp|None:
  """Recognize pi/2-asin(x) after asin has decomposed."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.ADD: return None
  for constant, negative in (u.src, u.src[::-1]):
    constant, negative = _unwrap_same_cast(constant), _unwrap_same_cast(negative)
    if constant.op is not Ops.CONST or not math.isclose(float(constant.arg), math.pi/2) or negative.op is not Ops.MUL: continue
    minus_one = next((_unwrap_same_cast(x) for x in negative.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    asin_u = next((_unwrap_same_cast(x) for x in negative.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
    if minus_one is not None and float(minus_one.arg) == -1 and asin_u is not None and (source:=_canonical_asin(asin_u)) is not None: return source
  return None

def _canonical_atan(u:UOp) -> UOp|None:
  """Recognize asin(x/sqrt(1+x*x)) and recover x."""
  if (normalized:=_canonical_asin(u)) is None or (normalized:=_unwrap_same_cast(normalized)).op is not Ops.MUL: return None
  for source, reciprocal in (normalized.src, normalized.src[::-1]):
    source, reciprocal = _unwrap_same_cast(source), _unwrap_same_cast(reciprocal)
    if reciprocal.op is not Ops.RECIPROCAL or (root:=_unwrap_same_cast(reciprocal.src[0])).op is not Ops.SQRT: continue
    denominator = _unwrap_same_cast(root.src[0])
    if denominator.op is not Ops.ADD: continue
    one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    square = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.MUL), None)
    if (one is not None and float(one.arg) == 1 and square is not None and
        all(_unwrap_same_cast(x).key == source.key for x in square.src)): return source
  return None

def _canonical_atanh(u:UOp) -> UOp|None:
  """Recognize log((1+x)/(1-x))/2 and recover x."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  factor = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  logarithm = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.LOG2), None)
  if factor is None or not math.isclose(float(factor.arg), math.log(2)/2) or logarithm is None: return None
  ratio = _unwrap_same_cast(logarithm.src[0])
  if ratio.op is not Ops.MUL: return None
  numerator = next((_unwrap_same_cast(x) for x in ratio.src if _unwrap_same_cast(x).op is Ops.ADD), None)
  reciprocal = next((_unwrap_same_cast(x) for x in ratio.src if _unwrap_same_cast(x).op is Ops.RECIPROCAL), None)
  if numerator is None or reciprocal is None or (denominator:=_unwrap_same_cast(reciprocal.src[0])).op is not Ops.ADD: return None
  one = next((_unwrap_same_cast(x) for x in numerator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  source = next((_unwrap_same_cast(x) for x in numerator.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
  if one is None or float(one.arg) != 1 or source is None: return None
  den_one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  negative = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.MUL), None)
  if den_one is None or float(den_one.arg) != 1 or negative is None: return None
  minus_one = next((_unwrap_same_cast(x) for x in negative.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  den_source = next((_unwrap_same_cast(x) for x in negative.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
  return source if minus_one is not None and float(minus_one.arg) == -1 and den_source is not None and den_source.key == source.key else None

def _canonical_inverse_hyperbolic(u:UOp, offset:float) -> UOp|None:
  """Recognize log(x+sqrt(x*x+offset)) and recover x."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  factor = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  logarithm = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.LOG2), None)
  if factor is None or not math.isclose(float(factor.arg), math.log(2)) or logarithm is None: return None
  argument = _unwrap_same_cast(logarithm.src[0])
  if argument.op is not Ops.ADD: return None
  root = next((_unwrap_same_cast(x) for x in argument.src if _unwrap_same_cast(x).op is Ops.SQRT), None)
  source = next((_unwrap_same_cast(x) for x in argument.src if _unwrap_same_cast(x).op is not Ops.SQRT), None)
  if root is None or source is None or (radicand:=_unwrap_same_cast(root.src[0])).op is not Ops.ADD: return None
  constant = next((_unwrap_same_cast(x) for x in radicand.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  square = next((_unwrap_same_cast(x) for x in radicand.src if _unwrap_same_cast(x).op is Ops.MUL), None)
  return source if constant is not None and math.isclose(float(constant.arg), offset) and square is not None and \
    all(_unwrap_same_cast(x).key == source.key for x in square.src) else None

def _canonical_hyperbolic(u:UOp) -> tuple[UOp,bool]|None:
  """Recognize (exp(x) +/- exp(-x))/2; bool is true for sinh."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  half_const = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  body = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
  if half_const is None or float(half_const.arg) != .5 or body is None or body.op is not Ops.ADD: return None
  def exponential(term:UOp) -> tuple[UOp,int,int]|None:
    term, outer = _unwrap_fp_cast(term), 1
    if term.op is Ops.MUL:
      negative = next((_unwrap_same_cast(x) for x in term.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      candidate = next((_unwrap_fp_cast(x) for x in term.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
      if negative is None or float(negative.arg) != -1 or candidate is None: return None
      term, outer = candidate, -1
    if term.op is not Ops.EXP2 or (scaled:=_unwrap_fp_cast(term.src[0])).op is not Ops.MUL: return None
    factor = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    operand = next((_unwrap_fp_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
    if factor is None or not math.isclose(float(factor.arg), math.log2(math.e)) or operand is None: return None
    exponent = 1
    if operand.op is Ops.MUL:
      negative = next((_unwrap_same_cast(x) for x in operand.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      source = next((_unwrap_same_cast(x) for x in operand.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
      if negative is None or float(negative.arg) != -1 or source is None: return None
      operand, exponent = source, -1
    return operand, outer, exponent
  terms = tuple(exponential(x) for x in body.src)
  if any(x is None for x in terms): return None
  lhs, rhs = cast(tuple[tuple[UOp,int,int], tuple[UOp,int,int]], terms)
  if lhs[0].key != rhs[0].key: return None
  signatures = {(lhs[1],lhs[2]), (rhs[1],rhs[2])}
  return (lhs[0], True) if signatures == {(1,1),(-1,-1)} else (lhs[0], False) if signatures == {(1,1),(1,-1)} else None

def _canonical_erf(u:UOp) -> UOp|None:
  """Recognize tinygrad's Abramowitz-Stegun erf expansion."""
  nodes = _unwrap_same_cast(u).toposort()
  indexes = [x for x in nodes if x.op is Ops.INDEX and x.dtype is dtypes.half]
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  coefficients = (0.3275911, 1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)
  return indexes[0] if len(indexes) == 1 and any(x.op is Ops.EXP2 for x in nodes) and \
    all(any(math.isclose(x, value) for x in constants) for value in coefficients) else None

def _canonical_softplus(u:UOp) -> tuple[UOp,float]|None:
  """Recognize +/-logaddexp(source, 0) after EXP and LOG decomposition."""
  u = _unwrap_same_cast(u)
  if u.op is Ops.MUL:
    factor = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    body = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is not Ops.CONST), None)
    if factor is not None and body is not None and (base:=_canonical_softplus(body)) is not None: return base[0], base[1]*float(factor.arg)
  if u.op is not Ops.ADD: return None
  for maximum, logarithm in (u.src, u.src[::-1]):
    maximum, polarity = _unwrap_same_cast(maximum), 1.0
    if maximum.op is Ops.MUL:
      factor = next((_unwrap_same_cast(x) for x in maximum.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      if factor is None or float(factor.arg) != -1: continue
      maximum, polarity = _unwrap_same_cast(maximum.src[0] if _unwrap_same_cast(maximum.src[1]).key == factor.key else maximum.src[1]), -1.0
    if maximum.op is not Ops.MAX: continue
    operands = tuple(_unwrap_same_cast(x) for x in maximum.src)
    zero = next((x for x in operands if x.op is Ops.CONST and float(x.arg) == 0), None)
    if zero is None: continue
    source = operands[0] if operands[1].key == zero.key else operands[1]
    nodes = _unwrap_same_cast(logarithm).toposort()
    constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
    if sum(x.op is Ops.EXP2 for x in nodes) == 2 and sum(x.op is Ops.LOG2 for x in nodes) == 1 and \
       all(any(math.isclose(x, value) for x in constants) for value in (math.log2(math.e), polarity*math.log(2), -1.0)): return source, polarity
  return None

def _canonical_mish(u:UOp) -> UOp|None:
  """Recognize source*tanh(softplus(source))."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for source, hyperbolic in (u.src, u.src[::-1]):
    source, hyperbolic = _unwrap_same_cast(source), _unwrap_same_cast(hyperbolic)
    softplus_u = _canonical_tanh(hyperbolic)
    if softplus_u is None or (softplus:=_canonical_softplus(softplus_u)) is None: continue
    if math.isclose(softplus[1], 1.0) and _unwrap_same_cast(softplus[0]).key == source.key: return source
  return None

def _canonical_hardswish(u:UOp) -> UOp|None:
  """Recognize x*relu6(x+3)/6 and recover x."""
  u = _unwrap_same_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.half]
  if len(indexes) != 1: return None
  counts = {op:sum(x.op is op for x in u.toposort()) for op in (Ops.MUL, Ops.ADD, Ops.WHERE, Ops.CMPLT)}
  constants = [float(x.arg) for x in u.toposort() if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  required = {Ops.MUL:3, Ops.ADD:3, Ops.WHERE:2, Ops.CMPLT:2}
  return indexes[0] if counts == required and all(any(math.isclose(x, value) for x in constants) for value in (-3,-1,0,1/6,3)) else None

def _canonical_quick_gelu(u:UOp) -> UOp|None:
  """Recognize source*sigmoid(1.702*source)."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for source, sigmoid_u in (u.src, u.src[::-1]):
    source = _unwrap_same_cast(source)
    if (sigmoid:=_canonical_sigmoid(sigmoid_u)) is not None and math.isclose(sigmoid[1], 1.702) and \
       _unwrap_same_cast(sigmoid[0]).key == source.key: return source
  return None

def _canonical_gelu(u:UOp) -> tuple[UOp,bool]|None:
  """Recognize tanh-approximate and exact GELU decompositions."""
  u = _unwrap_same_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.half]
  if len(indexes) != 1 or u.op is not Ops.MUL or sum(x.op is Ops.EXP2 for x in u.toposort()) != 1: return None
  constants = [float(x.arg) for x in u.toposort() if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  if any(math.isclose(x, .044715) for x in constants): return indexes[0], True
  if any(math.isclose(x, 1/math.sqrt(2)) for x in constants) and any(math.isclose(x, .231641888, rel_tol=1e-6) for x in constants):
    return indexes[0], False
  return None

def _canonical_elu(u:UOp) -> tuple[UOp,float,float]|None:
  """Recognize ELU and SELU decompositions and recover their shared exponential-tail parameters."""
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = [x for x in nodes if x.op is Ops.INDEX and x.dtype is dtypes.half]
  if len(indexes) != 1 or sum(x.op is Ops.EXP2 for x in nodes) != 1: return None
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  wheres = sum(x.op is Ops.WHERE for x in nodes)
  if u.op is Ops.ADD and wheres == 2: return indexes[0], (.1 if any(math.isclose(abs(x), .1) for x in constants) else 1.0), 1.0
  if u.op is Ops.MUL and wheres == 1 and any(math.isclose(x, 1.0507) for x in constants):
    return indexes[0], 1.0507*1.67326, 1.0507
  return None

def _canonical_celu(u:UOp) -> tuple[UOp,int]|None:
  """Recognize CELU for the integer alpha values exercised by the native generated-table contract."""
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = [x for x in nodes if x.op is Ops.INDEX and x.dtype is dtypes.half]
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  alpha = next((x for x in (4,3,2,1) if any(math.isclose(c, -x) for c in constants)), None)
  return (indexes[0], alpha) if u.op is Ops.ADD and len(indexes) == 1 and alpha is not None and \
    sum(x.op is Ops.EXP2 for x in nodes) == 1 and sum(x.op is Ops.MAX for x in nodes) >= 2 and \
    any(math.isclose(x, math.log2(math.e)) for x in constants) else None

def _canonical_round(u:UOp) -> UOp|None:
  """Recognize tinygrad's exact round-to-nearest-even expansion."""
  u = _unwrap_same_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX]
  if len(indexes) != 1 or (source:=indexes[0]).dtype is not dtypes.half: return None
  counts = {op:sum(x.op is op for x in u.toposort()) for op in (Ops.TRUNC, Ops.ADD, Ops.MUL, Ops.CMPLT, Ops.CMPNE, Ops.WHERE)}
  constants = [float(x.arg) for x in u.toposort() if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  required = {Ops.TRUNC:4, Ops.ADD:4, Ops.MUL:1, Ops.CMPLT:3, Ops.CMPNE:3, Ops.WHERE:3}
  return source if counts == required and all(any(math.isclose(x, value) for x in constants) for value in (-1,-.5,0,.5,1)) else None

def _canonical_negative_base55(u:UOp) -> UOp|None:
  """Recognize tinygrad's integer-validity/parity expansion for (-5.5)**x."""
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = [x for x in nodes if x.op is Ops.INDEX and x.dtype is dtypes.half]
  exponentials = [x for x in nodes if x.op is Ops.EXP2]
  if u.op is not Ops.WHERE or len(indexes) != 1 or len(exponentials) != 1 or sum(x.op is Ops.WHERE for x in nodes) != 3: return None
  source, exponential = indexes[0], exponentials[0]
  product = _unwrap_same_cast(exponential.src[0])
  factors = [float(x.arg) for x in product.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))] if product.op is Ops.MUL else []
  condition = _unwrap_same_cast(u.src[0])
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  return source if len(factors) == 1 and math.isclose(factors[0], math.log2(5.5), rel_tol=1e-3) and source in product.toposort() and \
    condition.op is Ops.CMPNE and source in condition.toposort() and any(math.isnan(x) for x in constants) and -1.0 in constants else None

def _canonical_relu_difference(u:UOp) -> UOp|None:
  """Recognize relu(x+0.5)-relu(x-0.5), the stable clip(x+0.5, 0, 1) form."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.ADD: return None
  def shifted_relu(v:UOp, negated:bool) -> tuple[UOp,float]|None:
    v = _unwrap_same_cast(v)
    if negated:
      if v.op is not Ops.MUL: return None
      const = next((_unwrap_same_cast(x) for x in v.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      if const is None or float(const.arg) != -1: return None
      v = _unwrap_same_cast(v.src[0] if _unwrap_same_cast(v.src[1]).key == const.key else v.src[1])
    if v.op is Ops.WHERE:
      cond, shifted, zero = (_unwrap_same_cast(x) for x in v.src)
      if cond.op is not Ops.CMPLT or _unwrap_same_cast(cond.src[0]).op is not Ops.CONST or float(_unwrap_same_cast(cond.src[0]).arg) != 0 or \
         _unwrap_same_cast(cond.src[1]).key != shifted.key or zero.op is not Ops.CONST or float(zero.arg) != 0: return None
    elif v.op is Ops.MAX:
      zero_const = next((_unwrap_same_cast(x) for x in v.src if _unwrap_same_cast(x).op is Ops.CONST and float(_unwrap_same_cast(x).arg) == 0), None)
      if zero_const is None: return None
      shifted = _unwrap_same_cast(v.src[0] if _unwrap_same_cast(v.src[1]).key == zero_const.key else v.src[1])
    else: return None
    if shifted.op is not Ops.ADD: return None
    offset = next((_unwrap_same_cast(x) for x in shifted.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    if offset is None: return None
    base = _unwrap_same_cast(shifted.src[0] if _unwrap_same_cast(shifted.src[1]).key == offset.key else shifted.src[1])
    return base, float(offset.arg)
  for positive, negative in (u.src, u.src[::-1]):
    pos, neg = shifted_relu(positive, False), shifted_relu(negative, True)
    if pos is not None and neg is not None and pos[0].key == neg[0].key and math.isclose(pos[1], .5) and math.isclose(neg[1], -.5):
      return pos[0]
  return None

def _parse_mask_expr(u:UOp, output_index:UOp, memo:dict[UOp, _Expr|RKArg|float]) -> _Expr|None:
  """Build an FP16 0/1 predicate from comparisons and boolean composition."""
  u = _unwrap_same_cast(u)
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    operands = tuple(_parse_alu(x, output_index, memo) for x in u.src)
    if any(x is None for x in operands): return None
    lhs, rhs = cast(tuple[_Value, _Value], operands)
    positive = _MaskExpr((_sub(rhs, lhs),))
    return positive if u.op is Ops.CMPLT else _ALUExpr(Ops.MAX, (positive, _MaskExpr((_sub(lhs, rhs),))))
  if u.op in (Ops.OR, Ops.AND):
    operands = tuple(_parse_mask_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in operands): return None
    return _ALUExpr(Ops.MAX if u.op is Ops.OR else Ops.MUL, cast(tuple[_Value, _Value], operands))
  return None

def _parse_alu(u:UOp, output_index:UOp, memo:dict[UOp, _Expr|RKArg|float]) -> _Expr|RKArg|float|None:
  while u.op is Ops.CAST and u.dtype in (dtypes.half, dtypes.float) and u.src[0].dtype in (dtypes.half, dtypes.float): u = u.src[0]
  u = _unwrap_same_cast(u)
  if u in memo: return memo[u]
  if u.op is Ops.INDEX and u.dtype is dtypes.half and u.src[0].op is Ops.PARAM and u.src[1].key == output_index.key:
    ret:_Expr|RKArg|float = RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  elif u.op is Ops.CONST and isinstance(u.arg, (int, float)): ret = float(u.arg)
  elif (negative_base55_input:=_canonical_negative_base55(u)) is not None:
    operand = _parse_alu(negative_base55_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _pow_negative_base55_expr(operand)
  elif (pow8_input:=_canonical_mul_power(u, 8)) is not None:
    operand = _parse_alu(pow8_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _pow8_expr(operand)
  elif (pow55_input:=_canonical_mul_power(u, 5.5)) is not None:
    operand = _parse_alu(pow55_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _pow55_expr(operand)
  elif (pow_neg55_input:=_canonical_mul_power(u, 5.5, reciprocal=True)) is not None:
    operand = _parse_alu(pow_neg55_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _pow_neg55_expr(operand)
  elif (hyperbolic:=_canonical_hyperbolic(u)) is not None:
    operand = _parse_alu(hyperbolic[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _sinh_expr(operand) if hyperbolic[1] else _cosh_expr(operand)
  elif (gelu:=_canonical_gelu(u)) is not None:
    operand = _parse_alu(gelu[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _gelu_expr(operand, gelu[1])
  elif (erf_input:=_canonical_erf(u)) is not None:
    operand = _parse_alu(erf_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _erf_expr(operand)
  elif (celu:=_canonical_celu(u)) is not None:
    operand = _parse_alu(celu[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _elu_expr(operand, 1.0, 1.0) if celu[1] == 1 else _celu_expr(operand, celu[1])
  elif (elu:=_canonical_elu(u)) is not None:
    operand = _parse_alu(elu[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _elu_expr(operand, elu[1], elu[2])
  elif (mish_input:=_canonical_mish(u)) is not None:
    operand = _parse_alu(mish_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _mish_expr(operand)
  elif (hardswish_input:=_canonical_hardswish(u)) is not None:
    operand = _parse_alu(hardswish_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _hardswish_expr(operand)
  elif (quick_gelu_input:=_canonical_quick_gelu(u)) is not None:
    operand = _parse_alu(quick_gelu_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _quick_gelu_expr(operand)
  elif (softplus_input:=_canonical_softplus(u)) is not None:
    softplus_source, input_scale = softplus_input[0], 1.0
    if softplus_source.op is Ops.MUL:
      factor = next((_unwrap_same_cast(x) for x in softplus_source.src if _unwrap_same_cast(x).op is Ops.CONST), None)
      if factor is not None and math.isclose(float(factor.arg), 3.0):
        input_scale = 3.0
        softplus_source = softplus_source.src[0] if _unwrap_same_cast(softplus_source.src[1]).key == factor.key else softplus_source.src[1]
    operand = _parse_alu(softplus_source, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _softplus_expr(operand, softplus_input[1], input_scale)
  elif (tanh_input:=_canonical_tanh(u)) is not None:
    operand = _parse_alu(tanh_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _tanh_expr(operand)
  elif (expm1:=_canonical_expm1(u)) is not None:
    operand = _parse_alu(expm1[0], output_index, memo)
    if operand is None: return None
    if expm1[1] < 0: operand = _ALUExpr(Ops.MUL, (operand, -1.0))
    ret = _expm1_expr(operand)
    if expm1[2] < 0: ret = _ALUExpr(Ops.MUL, (ret, -1.0))
  elif (atan_input:=_canonical_atan(u)) is not None:
    operand = _parse_alu(atan_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _atan_expr(operand)
  elif (atanh_input:=_canonical_atanh(u)) is not None:
    operand = _parse_alu(atanh_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _atanh_expr(operand)
  elif (asinh_input:=_canonical_inverse_hyperbolic(u, 1.0)) is not None:
    operand = _parse_alu(asinh_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _asinh_expr(operand)
  elif (acosh_input:=_canonical_inverse_hyperbolic(u, -1.0)) is not None:
    operand = _parse_alu(acosh_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _acosh_expr(operand)
  elif (acos_input:=_canonical_acos(u)) is not None:
    operand = _parse_alu(acos_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _acos_expr(operand)
  elif (asin_input:=_canonical_asin(u)) is not None:
    operand = _parse_alu(asin_input, output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _asin_expr(operand)
  elif (sigmoid:=_canonical_sigmoid(u)) is not None:
    operand = _parse_alu(sigmoid[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    if not math.isclose(sigmoid[1], 1.0): operand = _ALUExpr(Ops.MUL, (operand, sigmoid[1]))
    ret = _sigmoid_expr(operand)
  elif (rounded:=_canonical_round(u)) is not None:
    rounded_source = _parse_alu(rounded, output_index, memo)
    if rounded_source is None: return None
    ret = _round_expr(rounded_source)
  elif (abs_input:=_canonical_abs(u)) is not None:
    operand = _parse_alu(abs_input, output_index, memo)
    if operand is None: return None
    ret = _ALUExpr(Ops.MAX, (operand, _ALUExpr(Ops.MUL, (operand, -1.0))))
  elif (clamp_base:=_canonical_relu_difference(u)) is not None:
    base = _parse_alu(clamp_base, output_index, memo)
    if base is None: return None
    positive = _ALUExpr(Ops.MAX, (_ALUExpr(Ops.ADD, (base, .5)), 0.0))
    ret = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (positive, -1.0)), -1.0)), -1.0))
  elif u.op is Ops.MUL and any(x.op is Ops.RECIPROCAL for x in u.src):
    reciprocal = next(i for i,x in enumerate(u.src) if x.op is Ops.RECIPROCAL)
    if _canonical_sigmoid(u.src[reciprocal]) is not None:
      mul_src = tuple(_parse_alu(x, output_index, memo) for x in u.src)
      if any(x is None for x in mul_src): return None
      ret = _ALUExpr(Ops.MUL, cast(tuple[_Value, _Value], mul_src))
    else:
      div_src = (_parse_alu(u.src[1-reciprocal], output_index, memo), _parse_alu(u.src[reciprocal].src[0], output_index, memo))
      if any(x is None for x in div_src): return None
      numerator, denominator = cast(tuple[_Value, _Value], div_src)
      if isinstance(numerator, float) and math.isinf(numerator):
        sign = _sub(_MaskExpr((_sub(denominator, 0.0),)), _MaskExpr((_sub(0.0, denominator),)))
        ret = _ALUExpr(Ops.MUL, (numerator, sign))
      else: ret = _ALUExpr(Ops.FDIV, (numerator, denominator))
  elif u.op is Ops.RECIPROCAL:
    reciprocal_source = _unwrap_same_cast(u.src[0])
    if reciprocal_source.op is Ops.SQRT:
      operand = _parse_alu(reciprocal_source.src[0], output_index, memo)
      if operand is None or isinstance(operand, float): return None
      ret = _rsqrt_expr(operand)
      memo[u] = ret
      return ret
    reciprocal_denominator = _parse_alu(reciprocal_source, output_index, memo)
    if reciprocal_denominator is None: return None
    ret = _ALUExpr(Ops.FDIV, (1.0, reciprocal_denominator))
  elif u.op is Ops.TRUNC:
    operand = _parse_alu(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _trunc_expr(operand)
  elif u.op is Ops.SQRT:
    operand = _parse_alu(u.src[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _sqrt_expr(operand)
  elif u.op is Ops.LOG2:
    operand = _parse_alu(u.src[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _log2_expr(operand)
  elif u.op is Ops.MUL and (logarithm:=next((x for x in u.src if _unwrap_same_cast(x).op is Ops.LOG2), None)) is not None:
    factor = next((x for x in u.src if x is not logarithm and x.op is Ops.CONST and isinstance(x.arg, (int, float))), None)
    if factor is not None and math.isclose(float(factor.arg), math.log10(2)):
      operand = _parse_alu(_unwrap_same_cast(logarithm).src[0], output_index, memo)
      if operand is None or isinstance(operand, float): return None
      ret = _log2_expr(operand, float(factor.arg))
    else:
      src = tuple(_parse_alu(x, output_index, memo) for x in u.src)
      if len(src) != 2 or any(x is None for x in src): return None
      ret = _ALUExpr(Ops.MUL, (src[0], src[1]))  # type: ignore[arg-type]
  elif u.op is Ops.EXP2:
    exp_operand = _unwrap_same_cast(u.src[0])
    exp_factor = next((x for x in exp_operand.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None) \
      if exp_operand.op is Ops.MUL else None
    exp_source = next((x for x in exp_operand.src if x is not exp_factor), None) if exp_factor is not None else None
    exp_scale = float(exp_factor.arg) if exp_factor is not None else None
    is_exp = exp_scale is not None and math.isclose(abs(exp_scale), math.log2(math.e))
    is_pow_base55 = exp_scale is not None and math.isclose(exp_scale, math.log2(5.5), rel_tol=1e-3)
    is_pow_base8 = exp_scale is not None and math.isclose(exp_scale, 3.0, rel_tol=1e-3)
    operand = _parse_alu(exp_source if (is_exp or is_pow_base55 or is_pow_base8) and exp_source is not None else u.src[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    if is_exp: ret = _exp_expr(_ALUExpr(Ops.MUL, (operand, -1.0))) if cast(float, exp_scale) < 0 else _exp_expr(operand)
    elif is_pow_base55: ret = _pow_base55_expr(operand)
    elif is_pow_base8: ret = _pow_base8_expr(operand)
    else: ret = _exp2_expr(operand)
  elif u.op is Ops.WHERE:
    cond = _unwrap_same_cast(u.src[0])
    true_u, false_u = (_unwrap_same_cast(x) for x in u.src[1:])
    if cond.op is Ops.CMPLT:
      lhs_u, rhs_u = (_unwrap_same_cast(x) for x in cond.src)
      ordered_max = true_u.key == rhs_u.key and false_u.key == lhs_u.key
      ordered_min = true_u.key == lhs_u.key and false_u.key == rhs_u.key
    else:
      lhs_u = rhs_u = true_u
      ordered_max = ordered_min = False
    compared = tuple(_parse_alu(x, output_index, memo) for x in (lhs_u, rhs_u))
    if any(x is None for x in compared): return None
    lhs, rhs = cast(tuple[_Value, _Value], compared)
    threshold_select = cond.op is Ops.CMPLT and isinstance(rhs, float) and \
      ((true_u.key == lhs_u.key and false_u.op is Ops.CONST and math.isfinite(float(false_u.arg)) and float(false_u.arg) != rhs) or
       (false_u.key == lhs_u.key and true_u.op is Ops.CONST and math.isfinite(float(true_u.arg)) and float(true_u.arg) != rhs))
    if threshold_select:
      mask = _parse_mask_expr(cond, output_index, memo)
      if mask is None: return None
      threshold = cast(float, rhs)
      if true_u.key == lhs_u.key:
        base = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (lhs, -1.0)), -threshold)), -1.0))
        ret = _ALUExpr(Ops.ADD, (base, _ALUExpr(Ops.MUL, (float(false_u.arg)-threshold, _sub(1.0, mask)))))
      else:
        base = _ALUExpr(Ops.MAX, (lhs, threshold))
        ret = _ALUExpr(Ops.ADD, (base, _ALUExpr(Ops.MUL, (float(true_u.arg)-threshold, mask))))
    elif ordered_max: ret = _ALUExpr(Ops.MAX, (lhs, rhs))
    elif ordered_min:
      negative = (_ALUExpr(Ops.MUL, (lhs, -1.0)), _ALUExpr(Ops.MUL, (rhs, -1.0)))
      ret = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, negative), -1.0))
    else:
      mask = _parse_mask_expr(cond, output_index, memo)
      arms = tuple(_parse_alu(x, output_index, memo) for x in (true_u, false_u))
      if mask is None or any(x is None for x in arms) or any(isinstance(x, float) and not math.isfinite(x) for x in arms): return None
      true, false = cast(tuple[_Value, _Value], arms)
      ret = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (true, mask)),
        _ALUExpr(Ops.MUL, (false, _sub(1.0, mask)))))
  elif u.op in _RK_ALU_OPS:
    src = tuple(_parse_alu(x, output_index, memo) for x in u.src)
    if len(src) != 2 or any(x is None for x in src): return None
    ret = _ALUExpr(u.op, (src[0], src[1]))  # type: ignore[arg-type]
  else: return None
  memo[u] = ret
  return ret

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
  order:list[_Expr] = []
  def visit(expr:_Expr) -> None:
    for src in expr.src:
      if isinstance(src, (_ALUExpr, _MaskExpr, _LUTExpr)) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  visit(root)
  uses = {expr:sum(src == expr for node in order for src in node.src) for expr in order}
  values:dict[_Expr, RKArg] = {}
  free:list[int] = []
  scratch_count, stages = 0, cast(list[RKDPUStage], [])
  for expr in order:
    src = tuple(values[x] if isinstance(x, (_ALUExpr, _MaskExpr, _LUTExpr)) else x for x in expr.src)
    if expr is root: dst = output
    elif isinstance(expr, _ALUExpr) and (reuse:=next((values[x] for x in expr.src if isinstance(x, (_ALUExpr, _MaskExpr, _LUTExpr)) and
                                                     uses[x] == 1 and values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    if isinstance(expr, _ALUExpr): stages.append(RKALUStage(expr.op, dst, src[0], src[1], count))
    elif isinstance(expr, _LUTExpr) and isinstance(src[0], RKArg): stages.append(RKLUTStage(expr.lut, dst, src[0], count))
    elif isinstance(src[0], RKArg): stages.append(RKMaskStage(dst, src[0], count))
    else: return _unsupported(RKRejectKind.UNSUPPORTED_ALU, "stage source is not materializable")
    values[expr] = dst
    for source in expr.src:
      if isinstance(source, (_ALUExpr, _MaskExpr, _LUTExpr)):
        uses[source] -= 1
        arg = values[source]
        if uses[source] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  size = ((count+7)//8)*16
  return _native(RKDPUProgram(tuple(stages), tuple(RKScratch(size) for _ in range(scratch_count))))

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Compatibility helper for compiler probes; production lowering consumes `lower_dpu_result`."""
  return cast(RKDPUProgram|None, lower_dpu_result(sink).plan)

def _strip_casts(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _relu_source(u:UOp) -> UOp|None:
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  cond, positive, zero = u.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2 or cond.src[0].op is not Ops.CONST or float(cond.src[0].arg) != 0 or \
     zero.op is not Ops.CONST or float(zero.arg) != 0: return None
  source = _strip_casts(cond.src[1])
  return source if _strip_casts(positive).key == source.key else None

def _affine(u:UOp) -> tuple[dict[int, int], int]|None:
  if u.op is Ops.RANGE: return ({u.arg[0]:1}, 0)
  if u.op is Ops.CONST: return ({}, int(u.arg))
  if u.op is Ops.ADD:
    a, b = _affine(u.src[0]), _affine(u.src[1])
    if a is None or b is None: return None
    return ({k:a[0].get(k, 0)+b[0].get(k, 0) for k in a[0].keys()|b[0].keys()}, a[1]+b[1])
  if u.op is Ops.MUL:
    const, value = (u.src[0], u.src[1]) if u.src[0].op is Ops.CONST else (u.src[1], u.src[0])
    if const.op is not Ops.CONST or (aff:=_affine(value)) is None: return None
    return ({k:v*int(const.arg) for k,v in aff[0].items()}, aff[1]*int(const.arg))
  return None

def _const_category(value) -> str:
  if isinstance(value, float) and math.isnan(value): return "NAN"
  if isinstance(value, float) and math.isinf(value): return "POS_INF" if value > 0 else "NEG_INF"
  if value == 0: return "ZERO"
  if value == 1: return "ONE"
  if value == -1: return "NEG_ONE"
  if isinstance(value, int) or isinstance(value, float) and value.is_integer(): return "POS_INT" if value > 0 else "NEG_INT"
  if isinstance(value, (int, float)): return "POS_FRAC" if value > 0 else "NEG_FRAC"
  return type(value).__name__.upper()

def rk_fingerprint(sink:UOp) -> tuple:
  """Stable graph-family identity that omits buffer slots and exact constant values."""
  nodes = sink.toposort()
  axis_ids = {axis:i for i,axis in enumerate(sorted({u.arg[0] for u in nodes if u.op is Ops.RANGE}))}
  digest:dict[UOp, str] = {}
  indexes:list[tuple] = []
  reductions:list[tuple] = []
  for u in nodes:
    shape = tuple(x if isinstance(x, int) else str(x) for x in u._shape) if u._shape is not None else ()
    arg:tuple|None = None
    if u.op is Ops.PARAM: arg = (u.addrspace.name,)
    elif u.op is Ops.CONST: arg = (_const_category(u.arg),)
    elif u.op is Ops.RANGE: arg = (axis_ids[u.arg[0]], u.arg[-1].name, int(u.src[0].arg) if u.src[0].op is Ops.CONST else "dynamic")
    elif u.op is Ops.REDUCE:
      arg = (u.arg[0].name, u.arg[1])
      reductions.append(arg)
    elif u.op is Ops.INDEX:
      affine = _affine(u.src[1])
      index:tuple = ("nonaffine",) if affine is None else (tuple(sorted((axis_ids.get(k, -1), v) for k,v in affine[0].items())), affine[1])
      indexes.append((u.dtype.name, index))
      arg = index
    payload = (u.op.name, u.dtype.name, shape, arg, tuple(digest[x] for x in u.src))
    digest[u] = hashlib.sha256(repr(payload).encode()).hexdigest()[:16]
  op_counts = tuple((op.name, sum(u.op is op for u in nodes)) for op in sorted({u.op for u in nodes}, key=lambda x:x.name))
  params = tuple(sorted(((u.dtype.name, tuple(x if isinstance(x, int) else str(x) for x in u.shape), u.addrspace.name)
                         for u in nodes if u.op is Ops.PARAM), key=repr))
  constants = tuple(sorted(_const_category(u.arg) for u in nodes if u.op is Ops.CONST))
  return (("graph", digest[sink]), ("ops", op_counts), ("params", params), ("constants", constants),
          ("indexes", tuple(sorted(indexes, key=repr))), ("reductions", tuple(sorted(reductions, key=repr))))

def lower_reformat_result(sink:UOp) -> RKLowerResult:
  """Lower a static affine movement through atom copies or a sparse CMAC selector."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1: return _not_applicable()
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].src[0].op is not Ops.PARAM:
    return _not_applicable()
  value = _strip_casts(store.src[1])
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or len(value.src) != 2:
    return _not_applicable()
  if store.src[0].dtype is not dtypes.half or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE, "atom reformat requires FP16 input and output", store.src[0].op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  axes = tuple(sorted(out_aff[0].keys() | src_aff[0].keys()))
  if any(axis not in ranges for axis in axes): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "dynamic reformat range", Ops.RANGE)
  count, src_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  mapping = [-1] * count
  for coordinates in product(*(range(ranges[axis]) for axis in axes)):
    point = dict(zip(axes, coordinates))
    dst = out_aff[1] + sum(out_aff[0].get(axis, 0)*point[axis] for axis in axes)
    src = src_aff[1] + sum(src_aff[0].get(axis, 0)*point[axis] for axis in axes)
    if not 0 <= dst < count or mapping[dst] != -1 or not 0 <= src < src_count:
      return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat does not cover one dense output", Ops.INDEX)
    mapping[dst] = src
  if any(source < 0 for source in mapping): return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "reformat output has holes", Ops.INDEX)
  output, source = RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot), RKArg(RKBufferKind.ARG, value.src[0].arg.slot)
  stages:list[RKDPUStage] = []
  atom_reject:RKLowerResult|None = None
  dst = 0
  while dst < count:
    valid, src = min(8, count-dst), mapping[dst]
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
  if 0 < src_count <= 512 and 0 < count <= 4096 and constant_bytes <= 8*1024*1024:
    return _native(_sparse_cmac_pipeline(output, source, src_count, [[src] for src in mapping]))
  return atom_reject

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
  value = _strip_casts(reduce.src[0])
  if value.op is not Ops.INDEX or value.src[0].op is not Ops.PARAM or value.dtype is not dtypes.half:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC input is not one FP16 surface", reduce.op)
  out_aff, src_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or src_aff is None:
    return _unsupported(RKRejectKind.UNSUPPORTED_LAYOUT, "affine CMAC indexes are not affine", Ops.INDEX)
  ranges = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST}
  red_axes = tuple(u.arg[0] for u in reduce.src[1:] if u.op is Ops.RANGE)
  out_axes = tuple(sorted(out_aff[0]))
  output_count, input_count = int(store.src[0].src[0].src[0].arg), int(value.src[0].src[0].arg)
  if len(red_axes) != len(reduce.src)-1 or not out_axes or set(out_axes) & set(red_axes) or \
     set(src_aff[0]) - set(out_axes) - set(red_axes) or any(axis not in ranges for axis in (*out_axes,*red_axes)):
    return _unsupported(RKRejectKind.REQUIRES_REFORMAT, "affine CMAC axes do not form one static output/reduction partition", Ops.RANGE)
  if not 2 <= output_count <= 128 or not 2 <= input_count <= 512:
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
  return _native(_sparse_cmac_pipeline(RKArg(RKBufferKind.ARG, store.src[0].src[0].arg.slot),
    RKArg(RKBufferKind.ARG, value.src[0].arg.slot), input_count, selectors, scale))

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
  RKLowerer("reformat", lambda nodes:not _has_reduction(nodes), lower_reformat_result),
  RKLowerer("sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_add_reduce_result),
  RKLowerer("affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_affine_reduce_result),
  RKLowerer("ppu_reduce", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_reduce_result),
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

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_TARGET_CNA, _TARGET_CORE = 0x201, 0x801
_TARGET_PPU, _TARGET_PPU_RDMA = 0x4001, 0x8001
_EW_BASE = 0x108002c0
_ERDMA_FP16 = 0x40000008
_EW_CFG = {Ops.ADD:_EW_BASE | (2 << 16), Ops.MUL:_EW_BASE | (1 << 2) | (1 << 8), Ops.MAX:_EW_BASE,
           Ops.SUB:_EW_BASE | (4 << 16), Ops.FDIV:_EW_BASE | (3 << 16) | (1 << 8)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _emit_mask(stage_idx:int, plan:RKMaskStage) -> RKStage:
  width = (plan.count+7)//8-1
  # Rejected WIP: setting out_precision=int8 for a final public bool mask timed out on RK3588; exact probe is archived separately.
  regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53),
    (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width), (rk.REG_DPU_BN_MUL_CFG, 0),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_BS_CFG, 0x40040), (rk.REG_DPU_BS_ALU_CFG, 0x33800000),
    (rk.REG_DPU_BS_MUL_CFG, 0x40000000), (rk.REG_DPU_BN_CFG, 0x40082), (rk.REG_DPU_BN_MUL_CFG, 0x7c000000),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000), (rk.REG_DPU_EW_CFG, _EW_BASE|1), (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 0x10001), (rk.REG_DPU_OUT_CVT_SHIFT, 0),
    (rk.REG_DPU_SURFACE_ADD, 0x40))
  cmds = [_command(_TARGET_DPU, reg, value) for reg,value in regs]
  cmds += [_command(_TARGET_DPU_RDMA, reg, value) for reg,value in ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, _ERDMA_FP16))]
  relocs = []
  for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, plan.src),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, plan.src)):
    cmds.append(_command(target_id, reg, 0))
    relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET)

def _emit_roundoff(stage_idx:int, plan:RKLUTStage) -> RKStage:
  width, surf_stride, cmds = (plan.count+7)//8-1, ((plan.count+7)//8)*16, []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    cmds.extend(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value) for value in
                rklut.RK_LUT_ROUNDOFF[table_id*rklut.RK_LUT_ROUNDOFF_ENTRIES:(table_id+1)*rklut.RK_LUT_ROUNDOFF_ENTRIES])
  dpu = ((rk.REG_DPU_S_POINTER, 0x30), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
    (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width),
    (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_EW_CFG, 0x302), (rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (0x40c4, 0),
    (rk.REG_DPU_LUT_CFG, 0x68), (rk.REG_DPU_LUT_INFO, 0xe0e00), (rk.REG_DPU_LUT_LE_START, 0),
    (rk.REG_DPU_LUT_LE_END, 0x44000000), (rk.REG_DPU_LUT_LO_START, 0x44000000), (rk.REG_DPU_LUT_LO_END, 0x44800000),
    (0x4120, 23107), (0x4124, 22))
  cmds.extend(_command(_TARGET_DPU, *x) for x in dpu[:3])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(_TARGET_DPU, *x) for x in dpu[3:])
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1))
  cmds.extend(_command(_TARGET_DPU_RDMA, *x) for x in rdma)
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
           _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, plan.src.kind, plan.src.index, plan.src.addend))
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, RK_STAGE_RESET)

def _emit_lut(stage_idx:int, plan:RKLUTStage) -> RKStage:
  if plan.lut is RKLUTId.ROUNDOFF: return _emit_roundoff(stage_idx, plan)
  if plan.lut not in (RKLUTId.EXP2, RKLUTId.EXP, RKLUTId.EXP_LOCAL, RKLUTId.EXPM1, RKLUTId.EXPM1_LOCAL,
                      RKLUTId.SIGMOID, RKLUTId.SIGMOID_LOCAL, RKLUTId.TANH, RKLUTId.TANH_MID, RKLUTId.TANH_LOCAL,
                      RKLUTId.SQRT, RKLUTId.RSQRT, RKLUTId.LOG2, RKLUTId.LOG2_LOCAL, RKLUTId.LOG10, RKLUTId.LOG10_LOCAL,
                      RKLUTId.ASIN, RKLUTId.ASIN_LOCAL, RKLUTId.ASIN_EDGE, RKLUTId.ACOS, RKLUTId.ATAN,
                      RKLUTId.ATANH, RKLUTId.ATANH_EDGE, RKLUTId.ASINH, RKLUTId.ASINH_MID,
                      RKLUTId.ACOSH, RKLUTId.ACOSH_MID, RKLUTId.ACOSH_EDGE, RKLUTId.ASINH_NEAR,
                      RKLUTId.SINH, RKLUTId.COSH, RKLUTId.ERF, RKLUTId.ERF_LOCAL, RKLUTId.SOFTPLUS_NEG,
                      RKLUTId.SOFTPLUS_DIV3_NEAR, RKLUTId.SOFTPLUS_DIV3_FAR, RKLUTId.MISH, RKLUTId.MISH_MID,
                      RKLUTId.HARDSWISH, RKLUTId.QUICK_GELU, RKLUTId.QUICK_GELU_LOCAL, RKLUTId.GELU_TANH,
                      RKLUTId.GELU_TANH_LOCAL, RKLUTId.GELU_EXACT, RKLUTId.GELU_EXACT_LOCAL, RKLUTId.ELU1,
                      RKLUTId.ELU1_LOCAL, RKLUTId.ELU01, RKLUTId.ELU01_LOCAL, RKLUTId.SELU, RKLUTId.SELU_LOCAL,
                      RKLUTId.CELU2, RKLUTId.CELU2_LOCAL, RKLUTId.CELU3, RKLUTId.CELU3_LOCAL, RKLUTId.CELU4, RKLUTId.CELU4_LOCAL,
                      RKLUTId.POW8, RKLUTId.POW8_HIGH, RKLUTId.POW55, RKLUTId.POW55_LOCAL, RKLUTId.POW55_HIGH,
                      RKLUTId.POW_NEG55_LOW, RKLUTId.POW_NEG55_HIGH, RKLUTId.POW_NEG55_FAR,
                      RKLUTId.POW_BASE55_LOW, RKLUTId.POW_BASE55_HIGH, RKLUTId.POW_BASE8_FAR_LOW,
                      RKLUTId.POW_BASE8_LOW, RKLUTId.POW_BASE8_HIGH, RKLUTId.POW_BASE8_FAR_HIGH):
    raise ValueError(f"unimplemented Rockchip LUT {plan.lut}")
  name = plan.lut.name
  table, entries = getattr(rklut, f"RK_LUT_{name}"), getattr(rklut, f"RK_LUT_{name}_ENTRIES")
  bn_mul, minus_exp = getattr(rklut, f"RK_LUT_{name}_BN_MUL"), getattr(rklut, f"RK_LUT_{name}_MINUS_EXP")
  width, surf_stride, cmds = (plan.count+7)//8-1, ((plan.count+7)//8)*16, []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    cmds.extend(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value) for value in
                table[table_id*entries:(table_id+1)*entries])
  fixed = (
    (_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x30), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30),
    (_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width),
    (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    (_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 2), (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, 7),
    (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, width), (_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x20040),
    (_TARGET_DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000), (_TARGET_DPU, rk.REG_DPU_BN_MUL_CFG, bn_mul << 16),
    (_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x302), (_TARGET_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001), (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SHIFT, minus_exp << 12),
    (_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (_TARGET_DPU, 0x40c4, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_CFG, 0x68), (_TARGET_DPU, rk.REG_DPU_LUT_INFO, 0x50500),
    (_TARGET_DPU, rk.REG_DPU_LUT_LE_START, 0xffffc000), (_TARGET_DPU, rk.REG_DPU_LUT_LE_END, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_START, 0), (_TARGET_DPU, rk.REG_DPU_LUT_LO_END, 0x4000),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 16434 << 16), (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 13 << 5),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), (_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
  cmds.extend(_command(*x) for x in fixed[:4])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[4:30])
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[30:])
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, plan.src.kind, plan.src.index, plan.src.addend))
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, RK_STAGE_RESET)

def emit_dpu(program:RKDPUProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  constants, constant_offsets, stages = bytearray(), {}, []
  def materialize(value:RKArg|float, count:int) -> RKArg:
    if isinstance(value, RKArg): return value
    bits, key = struct.pack("<e", value), (value, count)
    if key not in constant_offsets:
      constant_offsets[key] = len(constants)
      constants.extend(bits * (((count+7)//8)*8))
    return RKArg(RKBufferKind.CONSTANT, constant_offsets[key])
  for stage_idx, plan in enumerate(program.stages):
    if isinstance(plan, RKLUTStage):
      stages.append(_emit_lut(stage_idx, plan))
      continue
    if isinstance(plan, RKMaskStage):
      stages.append(_emit_mask(stage_idx, plan))
      continue
    if not isinstance(plan, RKALUStage): raise ValueError(f"unimplemented Rockchip stage {type(plan).__name__}")
    material_count = plan.count*2 if plan.out_dtype is dtypes.int else (32 if plan.out_dtype is dtypes.float else plan.count)
    lhs, rhs = materialize(plan.lhs, material_count), materialize(plan.rhs, material_count)
    width = ((plan.count*2 if plan.out_dtype is dtypes.int else plan.count)+7)//8-1
    wide_out = plan.out_dtype in (dtypes.int, dtypes.float)
    lanes = 8 if wide_out or plan.count >= 8 else plan.count
    out_precision = 4 if plan.out_dtype is dtypes.int else (5 if plan.out_dtype is dtypes.float else 2)
    dpu_regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
      (rk.REG_DPU_DATA_FORMAT, (out_precision<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
      (rk.REG_DPU_DATA_CUBE_CHANNEL, ((lanes-1)<<16)|(lanes-1)),
      (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0),
      (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 3 if wide_out else lanes-1), (rk.REG_DPU_WDMA_SIZE_1, width),
      (rk.REG_DPU_BN_MUL_CFG, 0), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_EW_CFG, _EW_CFG[plan.op]),
      (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1), (rk.REG_DPU_OUT_CVT_OFFSET, 0),
      (rk.REG_DPU_OUT_CVT_SCALE, 0 if plan.out_dtype is dtypes.float else (1 if plan.op is Ops.FDIV or
       plan.out_dtype is dtypes.int else 0x10001)), (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
    rdma_regs = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, lanes-1),
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, _ERDMA_FP16))
    cmds = [_command(_TARGET_DPU, *x) for x in dpu_regs] + [_command(_TARGET_DPU_RDMA, *x) for x in rdma_regs]
    relocs = []
    for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, lhs),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, rhs)):
      cmds.append(_command(target_id, reg, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if plan.op is Ops.FDIV else 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

def emit_contract(plan:RKContract, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit one direct FP16 CMAC task; all surfaces are already hardware-legal."""
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  if plan.rhs.layout.kind is not RKLayoutKind.CMAC_WEIGHT: raise ValueError("CMAC RHS is not in weight layout")
  e, align_out, align_in = _command, plan.rhs.layout.physical_shape[0], plan.lhs.layout.physical_shape[-1]
  if align_in < 32 or align_in % 32: raise ValueError("CMAC K must be aligned to 32")
  if align_out != 32: raise ValueError("proven CMAC output tile is 32 physical channels")
  input_row_bytes = align_in*2
  feature_grains = max(80, (((2*256*128+input_row_bytes-1)//input_row_bytes)+1)&-2)
  line_stride = 4*min(align_in//32, 13)
  notch = 8*min(align_out//32, 13)-1
  commands = (
    e(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x0e), e(_TARGET_CNA, rk.REG_CNA_CONV_CON1, 0x20000120),
    e(_TARGET_CNA, rk.REG_CNA_CONV_CON2, feature_grains<<4), e(_TARGET_CNA, rk.REG_CNA_CONV_CON3, 9),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE1, ((align_in-1)<<16)|align_in),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE2, 1), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE3, 1),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE0, input_row_bytes*align_out), e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE1, input_row_bytes),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE2, 0x1010000|align_out), e(_TARGET_CNA, rk.REG_CNA_CBUF_CON0, 0xb1),
    e(_TARGET_CNA, rk.REG_CNA_CBUF_CON1, align_in//32), e(_TARGET_CNA, rk.REG_CNA_CVT_CON0, 0xb),
    *(e(_TARGET_CNA, reg, 0x10000) for reg in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4)),
    e(_TARGET_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0), e(_TARGET_CNA, rk.REG_CNA_DMA_CON0, 0xf000f),
    e(_TARGET_CNA, rk.REG_CNA_DMA_CON1, line_stride), e(_TARGET_CNA, rk.REG_CNA_DMA_CON2, 0),
    e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE1, align_in),
    e(_TARGET_CNA, rk.REG_CNA_DCOMP_ADDR0, 0), e(_TARGET_CORE, rk.REG_CORE_MISC_CFG, 0x201),
    e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_0, 0), e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out-1),
    e(_TARGET_CORE, rk.REG_CORE_RESERVED_3030, 0), e(_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e4),
    e(_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002), e(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    e(_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, 0x10), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch<<16)|notch),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out-1)<<16)|(align_out-1)), e(_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 0x126), e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, align_out-1),
    e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, 0), e(_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x383), e(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    e(_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 0x40), e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
  relocs = tuple(RKReloc(0, word, ref.buffer.kind, ref.buffer.index, ref.buffer.addend+ref.layout.base_offset)
                 for word,ref in ((18,plan.lhs), (24,plan.rhs), (31,plan.out)))
  return RKImage(target, (RKStage(RKEngine.CMAC, commands, relocs, RK_STAGE_RESET),), constants=plan.constants)

def emit_program(plan:RKProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Compose arbitrary typed engine steps into one ordered sequential image."""
  images:list[RKImage] = []
  for step in plan.steps:
    if isinstance(step, RKDPUProgram): images.append(emit_dpu(step, target))
    elif isinstance(step, RKContract): images.append(emit_contract(step, target))
    elif isinstance(step, RKReduce): images.append(emit_reduce(step, target))
    else: raise TypeError(f"unsupported Rockchip program step {type(step).__name__}")
  stages:list[RKStage] = []
  constants = bytearray()
  for image in images:
    constant_base = len(constants)
    for stage in image.stages:
      relocs = tuple(RKReloc(len(stages), reloc.word, reloc.kind,
        reloc.index+(constant_base if reloc.kind is RKBufferKind.CONSTANT else 0), reloc.addend, reloc.shift, reloc.mask, reloc.field_shift)
        for reloc in stage.relocs)
      stages.append(RKStage(stage.engine, stage.commands, relocs, stage.flags))
    constants.extend(image.constants)
  return RKImage(target, tuple(stages), plan.scratch, bytes(constants))

def emit_reduce(plan:RKReduce, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit the proven direct PPU global-MAX program for a dense FP16 HWC8 surface."""
  if target is not RKTarget.RK3588 or plan.op is not Ops.MAX: raise ValueError("unsupported Rockchip PPU reduction")
  height, width, channels = plan.src.layout.logical_shape
  if channels != 8 or not 2 <= height <= 16 or not 2 <= width <= 16: raise ValueError("PPU global MAX requires 2..16 x 2..16 x 8")
  h, w, c = height-1, width-1, channels-1
  regs = (
    (_TARGET_PPU, rk.REG_PPU_S_POINTER, 0xe), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_S_POINTER, 0xe),
    (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, w), (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, h),
    (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, c), (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, c),
    (_TARGET_PPU, rk.REG_PPU_OPERATION_MODE_CFG, 0x11),
    (_TARGET_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (h<<20)|(w<<16)|(h<<8)|w),
    (_TARGET_PPU, rk.REG_PPU_DST_SURF_STRIDE, 1), (_TARGET_PPU, rk.REG_PPU_DATA_FORMAT, 0x10002),
    (_TARGET_PPU, rk.REG_PPU_MISC_CTRL, 3), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_WIDTH, w),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_HEIGHT, h), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_CHANNEL, c),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_LINE_STRIDE, plan.src.layout.strides_bytes[0]),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_SURF_STRIDE, height*plan.src.layout.strides_bytes[0]),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_DATA_FORMAT, 2), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_OPERATION_ENABLE, 1))
  commands = [_command(*x) for x in regs]
  dst_word = len(commands)
  commands.append(_command(_TARGET_PPU, rk.REG_PPU_DST_BASE_ADDR, 0))
  src_word = len(commands)
  commands.append(_command(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_BASE_ADDR, 0))
  commands.append(_command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x60))
  relocs = (RKReloc(0, dst_word, plan.out.buffer.kind, plan.out.buffer.index, plan.out.buffer.addend+plan.out.layout.base_offset),
            RKReloc(0, src_word, plan.src.buffer.kind, plan.src.buffer.index, plan.src.buffer.addend+plan.src.layout.base_offset))
  return RKImage(target, (RKStage(RKEngine.PPU, tuple(commands), relocs, RK_STAGE_RESET),))

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
