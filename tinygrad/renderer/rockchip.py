from __future__ import annotations
import math, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, cast
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk, rockchip_lut as rklut
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 2, 1
_HEADER, _STAGE = struct.Struct("<4sHHHHHHIII"), struct.Struct("<BBHQIIIIQQ")
_RELOC, _SCRATCH = struct.Struct("<HHBBIqIH"), struct.Struct("<II")

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum):
  DPU = 1
  CMAC = 2
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
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
class RKView:
  slot: int
  shape: tuple[int, ...]
  strides: tuple[int, ...]

@dataclass(frozen=True)
class RKContract:
  out: RKView
  lhs: RKView
  rhs: RKView
  reduce_axis: int

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
  reads: tuple[int, ...] = ()
  writes: tuple[int, ...] = ()
  dependencies: int = 0
  flags: int = 0

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  stages: tuple[RKStage, ...]
  scratch: tuple[RKScratch, ...] = ()
  constants: bytes = b""
  version: int = RKIMAGE_VERSION

def _slot_mask(slots:tuple[int, ...]) -> int:
  if any(x < 0 or x >= 64 for x in slots): raise ValueError("RKImage supports argument slots 0..63")
  return sum(1 << x for x in slots)

def validate_image(image:RKImage) -> None:
  if image.version != RKIMAGE_VERSION: raise ValueError(f"unsupported RKImage version {image.version}")
  if len(image.stages) > 64: raise ValueError("too many RKImage stages")
  for stage_idx, stage in enumerate(image.stages):
    if stage.dependencies >> stage_idx: raise ValueError("stage dependency must refer to an earlier stage")
    _slot_mask(stage.reads), _slot_mask(stage.writes)
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
    rows.append((int(stage.engine), stage.flags, 0, stage.dependencies, command_start, len(stage.commands), reloc_start, len(stage.relocs),
                 _slot_mask(stage.reads), _slot_mask(stage.writes)))
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
  def slots(mask:int) -> tuple[int, ...]: return tuple(x for x in range(64) if mask & (1 << x))
  for idx, (engine, flags, row_reserved, deps, command_start, command_len, reloc_start, reloc_len, reads, writes) in enumerate(rows):
    if row_reserved or command_start+command_len > command_count or reloc_start+reloc_len > reloc_count: raise ValueError("invalid RKImage stage")
    stages.append(RKStage(RKEngine(engine), tuple(commands[command_start:command_start+command_len]),
                          tuple(relocs[reloc_start:reloc_start+reloc_len]), slots(reads), slots(writes), deps, flags))
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
  finite = _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.FDIV, (corrected, nonzero)), _sub(1.0, high)))
  not_number = _ALUExpr(Ops.MUL, (greater_zero, negative))
  valid = _sub(1.0, _ALUExpr(Ops.MAX, (negative, not_number)))
  return _ALUExpr(Ops.MUL, (finite, _ALUExpr(Ops.FDIV, (valid, valid))))

def _unwrap_same_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype is u.src[0].dtype: u = u.src[0]
  return u

def _canonical_sigmoid(u:UOp) -> tuple[UOp,float]|None:
  """Recognize 1/(1+exp2(-log2(e)*x))."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.RECIPROCAL or (denominator:=_unwrap_same_cast(u.src[0])).op is not Ops.ADD: return None
  one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  exponential = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.EXP2), None)
  if one is None or float(one.arg) != 1 or exponential is None or (scaled:=_unwrap_same_cast(exponential.src[0])).op is not Ops.MUL: return None
  source = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is Ops.INDEX), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None)
  return (source, factor/-math.log2(math.e)) if source is not None and factor is not None and math.isfinite(factor) else None

def _canonical_abs(u:UOp) -> UOp|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for data, sign in (u.src, u.src[::-1]):
    data, sign = _unwrap_same_cast(data), _unwrap_same_cast(sign)
    if data.op is not Ops.INDEX or sign.op is not Ops.WHERE: continue
    cond, nonzero, zero = (_unwrap_same_cast(x) for x in sign.src)
    if cond.op is not Ops.CMPNE or zero.op is not Ops.CONST or float(zero.arg) != 0 or nonzero.op is not Ops.WHERE: continue
    less, negative, positive = (_unwrap_same_cast(x) for x in nonzero.src)
    compared = tuple(_unwrap_same_cast(x) for x in cond.src)
    if (less.op is Ops.CMPLT and compared[0].key == data.key and compared[1].op is Ops.CONST and float(compared[1].arg) == 0 and
        _unwrap_same_cast(less.src[0]).key == data.key and _unwrap_same_cast(less.src[1]).op is Ops.CONST and
        float(_unwrap_same_cast(less.src[1]).arg) == 0 and negative.op is Ops.CONST and float(negative.arg) == -1 and
        positive.op is Ops.CONST and float(positive.arg) == 1): return data
  return None

def _canonical_round(u:UOp) -> UOp|None:
  """Recognize tinygrad's exact round-to-nearest-even expansion."""
  u = _unwrap_same_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX]
  if len(indexes) != 1 or (source:=indexes[0]).dtype is not dtypes.half: return None
  counts = {op:sum(x.op is op for x in u.toposort()) for op in (Ops.TRUNC, Ops.ADD, Ops.MUL, Ops.CMPLT, Ops.CMPNE, Ops.WHERE)}
  constants = [float(x.arg) for x in u.toposort() if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  required = {Ops.TRUNC:4, Ops.ADD:4, Ops.MUL:1, Ops.CMPLT:3, Ops.CMPNE:3, Ops.WHERE:3}
  return source if counts == required and all(any(math.isclose(x, value) for x in constants) for value in (-1,-.5,0,.5,1)) else None

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
  elif (sigmoid:=_canonical_sigmoid(u)) is not None:
    operand = _parse_alu(sigmoid[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    if not math.isclose(sigmoid[1], 1.0): operand = _ALUExpr(Ops.MUL, (operand, sigmoid[1]))
    ret = _sigmoid_expr(operand)
  elif (rounded:=_canonical_round(u)) is not None:
    source = _parse_alu(rounded, output_index, memo)
    if source is None: return None
    ret = _round_expr(source)
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
    if factor is None or not math.isclose(float(factor.arg), math.log10(2)): return None
    operand = _parse_alu(_unwrap_same_cast(logarithm).src[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    ret = _log2_expr(operand, float(factor.arg))
  elif u.op is Ops.EXP2:
    exp_operand = _unwrap_same_cast(u.src[0])
    exp_factor = next((x for x in exp_operand.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None) \
      if exp_operand.op is Ops.MUL else None
    exp_source = next((x for x in exp_operand.src if x is not exp_factor), None) if exp_factor is not None else None
    exp_scale = float(exp_factor.arg) if exp_factor is not None else None
    is_exp = exp_scale is not None and math.isclose(abs(exp_scale), math.log2(math.e))
    operand = _parse_alu(exp_source if is_exp and exp_source is not None else u.src[0], output_index, memo)
    if operand is None or isinstance(operand, float): return None
    if is_exp: ret = _exp_expr(_ALUExpr(Ops.MUL, (operand, -1.0))) if cast(float, exp_scale) < 0 else _exp_expr(operand)
    else: ret = _LUTExpr(RKLUTId.EXP2, (operand,))
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

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Lower one contiguous FP16 expression or native wide constant fill to a UOp-free primitive DPU plan."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1 or (store:=stores[0]).src[0].op is not Ops.INDEX or \
     store.src[0].dtype not in (dtypes.half, dtypes.int, dtypes.float): return None
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op not in (Ops.RANGE, Ops.CONST) or out_param.src[0].op is not Ops.CONST: return None
  count = int(out_param.src[0].arg)
  if not 0 < count <= 65536 or (out_index.op is Ops.RANGE and int(out_index.src[0].arg) != count) or \
     (out_index.op is Ops.CONST and (count != 1 or int(out_index.arg) != 0)): return None
  root = _parse_alu(store.src[1], out_index, {})
  if root is None: return None
  if store.src[0].dtype is not dtypes.half and not isinstance(root, float): return None
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if not isinstance(root, (_ALUExpr, _MaskExpr, _LUTExpr)):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      tile = 64 if store.src[0].dtype is dtypes.int else 4
      fill_stages = tuple(RKALUStage(Ops.ADD, RKArg(output.kind, output.index, start*4), 0.0, root, min(tile, count-start),
                                     store.src[0].dtype) for start in range(0, count, tile))
      return RKDPUProgram(fill_stages) if len(fill_stages) <= 64 else None
    return RKDPUProgram((RKALUStage(Ops.ADD, output, 0.0, root, count),))
  if isinstance(root, _LUTExpr) and root.lut is RKLUTId.EXP2 and isinstance(root.src[0], RKArg):
    input_arg, base = root.src[0], root
    positive_inf = _MaskExpr((_sub(input_arg, 65504.0),))
    negative_inf = _MaskExpr((_sub(-65504.0, input_arg),))
    finite = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.FDIV, (base, _sub(1.0, positive_inf))), _sub(1.0, negative_inf)))
    not_number = _ALUExpr(Ops.MUL, (positive_inf, negative_inf))
    nan_denom = _sub(1.0, not_number)
    root = _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.MUL, (finite, nan_denom)), nan_denom))
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
    else: return None
    values[expr] = dst
    for source in expr.src:
      if isinstance(source, (_ALUExpr, _MaskExpr, _LUTExpr)):
        uses[source] -= 1
        arg = values[source]
        if uses[source] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  size = ((count+7)//8)*16
  return RKDPUProgram(tuple(stages), tuple(RKScratch(size) for _ in range(scratch_count)))

def _strip_casts(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

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

def lower_contract(sink:UOp) -> RKContract|None:
  """Recognize the directly legal M=1, K=32 FP16 CMAC surface."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return None
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].dtype is not dtypes.half or reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return None
  if _strip_casts(store.src[1]).key != reduce.key: return None
  body, red = _strip_casts(reduce.src[0]), reduce.src[1]
  out_param, out_aff = store.src[0].src[0], _affine(store.src[0].src[1])
  if out_param.op is not Ops.PARAM or out_aff is None or out_aff[1] or len(out_aff[0]) != 1 or body.op is not Ops.MUL: return None
  if red.op is not Ops.RANGE or int(red.src[0].arg) != 32: return None
  lhs, rhs = (_strip_casts(x) for x in body.src)
  if any(x.op is not Ops.INDEX or x.dtype is not dtypes.half or x.src[0].op is not Ops.PARAM for x in (lhs, rhs)): return None
  lhs_aff, rhs_aff = _affine(lhs.src[1]), _affine(rhs.src[1])
  if lhs_aff is None or rhs_aff is None or lhs_aff[1] or rhs_aff[1]: return None
  red_axis, out_axes = red.arg[0], tuple(out_aff[0])
  if len(out_axes) != 1 or out_aff[0][out_axes[0]] != 1 or lhs_aff[0] != {red_axis:1}: return None
  n_axis = out_axes[0]
  n = next(int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.arg[0] == n_axis)
  if not 4 <= n <= 16 or rhs_aff[0] != {n_axis:32, red_axis:1}: return None
  if int(out_param.src[0].arg) != n or int(lhs.src[0].src[0].arg) != 32 or int(rhs.src[0].src[0].arg) != n*32: return None
  return RKContract(RKView(out_param.arg.slot, (1,n), (n,1)), RKView(lhs.src[0].arg.slot, (1,32), (32,1)),
                    RKView(rhs.src[0].arg.slot, (n,32), (32,1)), red_axis)

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_TARGET_CNA, _TARGET_CORE = 0x201, 0x801
_EW_BASE = 0x108002c0
_EW_CFG = {Ops.ADD:_EW_BASE | (2 << 16), Ops.MUL:_EW_BASE | (1 << 2) | (1 << 8), Ops.MAX:_EW_BASE,
           Ops.SUB:_EW_BASE | (4 << 16), Ops.FDIV:_EW_BASE | (3 << 16) | (1 << 8)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _emit_mask(stage_idx:int, plan:RKMaskStage) -> RKStage:
  width = (plan.count+7)//8-1
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
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))]
  relocs = []
  for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, plan.src),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, plan.src)):
    cmds.append(_command(target_id, reg, 0))
    relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  reads, writes = ((plan.src.index,) if plan.src.kind is RKBufferKind.ARG else ()), \
                   ((plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ())
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

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
  reads, writes = ((plan.src.index,) if plan.src.kind is RKBufferKind.ARG else ()), \
                   ((plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ())
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

def _emit_lut(stage_idx:int, plan:RKLUTStage) -> RKStage:
  if plan.lut is RKLUTId.ROUNDOFF: return _emit_roundoff(stage_idx, plan)
  if plan.lut not in (RKLUTId.EXP2, RKLUTId.EXP, RKLUTId.EXP_LOCAL, RKLUTId.SIGMOID, RKLUTId.SIGMOID_LOCAL,
                      RKLUTId.SQRT, RKLUTId.RSQRT, RKLUTId.LOG2, RKLUTId.LOG2_LOCAL, RKLUTId.LOG10, RKLUTId.LOG10_LOCAL):
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
  reads, writes = ((plan.src.index,) if plan.src.kind is RKBufferKind.ARG else ()), \
                   ((plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ())
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

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
    out_precision = 4 if plan.out_dtype is dtypes.int else (5 if plan.out_dtype is dtypes.float else 2)
    dpu_regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
      (rk.REG_DPU_DATA_FORMAT, (out_precision<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0), (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
      (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0),
      (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 3 if wide_out else 7), (rk.REG_DPU_WDMA_SIZE_1, width),
      (rk.REG_DPU_BN_MUL_CFG, 0), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_EW_CFG, _EW_CFG[plan.op]),
      (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1), (rk.REG_DPU_OUT_CVT_OFFSET, 0),
      (rk.REG_DPU_OUT_CVT_SCALE, 0 if plan.out_dtype is dtypes.float else (1 if plan.op is Ops.FDIV or
       plan.out_dtype is dtypes.int else 0x10001)), (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
    rdma_regs = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))
    cmds = [_command(_TARGET_DPU, *x) for x in dpu_regs] + [_command(_TARGET_DPU_RDMA, *x) for x in rdma_regs]
    relocs = []
    for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, lhs),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, rhs)):
      cmds.append(_command(target_id, reg, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if plan.op is Ops.FDIV else 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    operands = (lhs, rhs)
    reads = tuple(sorted({x.index for x in operands if x.kind is RKBufferKind.ARG}))
    writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

def emit_contract(plan:RKContract, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit one direct FP16 CMAC task; all surfaces are already hardware-legal."""
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  e, align = _command, 32
  commands = (
    e(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x0e), e(_TARGET_CNA, rk.REG_CNA_CONV_CON1, 0x20000120),
    e(_TARGET_CNA, rk.REG_CNA_CONV_CON2, 0x4000), e(_TARGET_CNA, rk.REG_CNA_CONV_CON3, 9),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE1, 0x1f0020),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE2, 1), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE3, 1),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE0, 0x800), e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE1, 0x40),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE2, 0x1010020), e(_TARGET_CNA, rk.REG_CNA_CBUF_CON0, 0xb1),
    e(_TARGET_CNA, rk.REG_CNA_CBUF_CON1, 1), e(_TARGET_CNA, rk.REG_CNA_CVT_CON0, 0xb),
    *(e(_TARGET_CNA, reg, 0x10000) for reg in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4)),
    e(_TARGET_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0), e(_TARGET_CNA, rk.REG_CNA_DMA_CON0, 0xf000f),
    e(_TARGET_CNA, rk.REG_CNA_DMA_CON1, 4), e(_TARGET_CNA, rk.REG_CNA_DMA_CON2, 0),
    e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE1, align),
    e(_TARGET_CNA, rk.REG_CNA_DCOMP_ADDR0, 0), e(_TARGET_CORE, rk.REG_CORE_MISC_CFG, 0x201),
    e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_0, 0), e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align-1),
    e(_TARGET_CORE, rk.REG_CORE_RESERVED_3030, 0), e(_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e4),
    e(_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002), e(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    e(_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, 0x10), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0x70007),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x1f001f), e(_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 0x126), e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, align-1),
    e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, 0), e(_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x383), e(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    e(_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 0x40), e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
  relocs = (RKReloc(0, 18, RKBufferKind.ARG, plan.lhs.slot), RKReloc(0, 24, RKBufferKind.ARG, plan.rhs.slot),
            RKReloc(0, 31, RKBufferKind.ARG, plan.out.slot))
  return RKImage(target, (RKStage(RKEngine.CMAC, commands, relocs, (plan.lhs.slot,plan.rhs.slot), (plan.out.slot,), flags=RK_STAGE_RESET),))

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int, dtypes.float}
  def native_program(self, ast:UOp) -> UOp|None:
    if (dpu:=lower_dpu(ast)) is not None: image = emit_dpu(dpu)
    elif (contract:=lower_contract(ast)) is not None: image = emit_contract(contract)
    else: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
    return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=encode_image(image))),
               arg=ProgramInfo.from_sink(ast, self.target))
