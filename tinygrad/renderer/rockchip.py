from __future__ import annotations
import math, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, cast
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk, rockchip_lut as rklut
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 3, 1
_HEADER = struct.Struct("<4sHHHHHHIII")
_STAGE = struct.Struct("<BBHQIIIIQQ")
_RELOC = struct.Struct("<HHBBIqIH")
_SCRATCH = struct.Struct("<II")

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum):
  DPU = 1
  CMAC = 2
  CONV = 3
  PPU = 4
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
class RKDPUOp(IntEnum):
  COPY = 0
  ADD = 1
  MUL = 2
  MAX = 3
  SUB = 4
  EXP2 = 5
  MASK = 6
  DIV = 7
  HARDSWISH = 8
  HARDSWISH_LOCAL = 9
  TANH = 10
  TANH_LOCAL = 11
  SIGMOID = 12
  SIGMOID_LOCAL = 13
  QUICK_GELU = 14
  QUICK_GELU_LOCAL = 15
  GELU_TANH = 16
  GELU_TANH_LOCAL = 17
  GELU_EXACT = 18
  GELU_EXACT_LOCAL = 19
  ERF = 20
  ERF_LOCAL = 21

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
class RKScratch:
  size: int
  alignment: int = 4096

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

@dataclass(frozen=True)
class RKArg:
  kind: RKBufferKind
  index: int
  addend: int = 0

@dataclass(frozen=True)
class RKDPUStage:
  op: RKDPUOp
  dst: RKArg
  lhs: RKArg|float
  rhs: RKArg|float|None
  count: int
  out_dtype: DType = dtypes.half

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...]

@dataclass(frozen=True)
class RKView:
  slot: int
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  offset: int = 0
  kind: RKBufferKind = RKBufferKind.ARG

@dataclass(frozen=True)
class RKContract:
  out: RKView
  lhs: RKView
  rhs: RKView
  reduce_axes: tuple[int, ...]

@dataclass(frozen=True)
class RKPool:
  out: RKView
  inp: RKView
  kernel: tuple[int, int]

@dataclass(frozen=True)
class _DPUExpr:
  op: RKDPUOp
  src: tuple[_DPUExpr|RKArg|float, ...]

def _sigmoid_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Two-level sigmoid LUT with analytic tails and preserved NaN behavior."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  broad, local = _DPUExpr(RKDPUOp.SIGMOID, (source,)), _DPUExpr(RKDPUOp.SIGMOID_LOCAL, (source,))
  local_outside = _DPUExpr(RKDPUOp.MAX, (positive(-2.0, source), positive(source, 2.0)))
  selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, local_outside)),
    _DPUExpr(RKDPUOp.MUL, (local, _DPUExpr(RKDPUOp.SUB, (1.0, local_outside))))))
  high, low = positive(source, 8.0), positive(-8.0, source)
  high_result = _DPUExpr(RKDPUOp.ADD, (selected, _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (1.0, selected)), high))))
  bounded = _DPUExpr(RKDPUOp.MUL, (high_result, _DPUExpr(RKDPUOp.SUB, (1.0, low))))
  nan_denom = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MUL, (high, low))))
  return _DPUExpr(RKDPUOp.DIV, (_DPUExpr(RKDPUOp.MUL, (bounded, nan_denom)), nan_denom))

def _quick_gelu_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Dedicated broad/local LUTs plus a polynomial near zero and bounded sigmoid tails."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  base = _DPUExpr(RKDPUOp.MUL, (source, _sigmoid_expr(_DPUExpr(RKDPUOp.MUL, (source, 1.702)))))
  broad, local = _DPUExpr(RKDPUOp.QUICK_GELU, (source,)), _DPUExpr(RKDPUOp.QUICK_GELU_LOCAL,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.ADD, (source, 1.5)), 4.0)),))
  below, above = positive(-2.0, source), positive(source, 2.0)
  outside = _DPUExpr(RKDPUOp.MAX, (below, above))
  inside = _DPUExpr(RKDPUOp.SUB, (1.0, outside))
  local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MAX, (below, positive(source, -1.0)))))
  poly_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.16), positive(0.16, source)))
  broad_mask = _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.SUB, (inside, local_inside)), poly_inside))
  poly_x = _DPUExpr(RKDPUOp.MUL, (source, poly_inside))
  polynomial = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (poly_x, 0.5)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (poly_x, poly_x)), 0.4253))))
  inner = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, broad_mask)),
    _DPUExpr(RKDPUOp.MUL, (local, local_inside)))), polynomial))
  return _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (base, outside)), inner))

def _gelu_expr(source:_DPUExpr|RKArg|float, approximate_tanh:bool) -> _DPUExpr:
  """Broad/local GELU LUTs, near-zero series, and exact zero/x asymptotes."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  def clamp(value:_DPUExpr|RKArg|float, limit:float) -> _DPUExpr:
    lower = _DPUExpr(RKDPUOp.MAX, (value, -limit))
    return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -limit)), -1.0))
  broad_op, local_op = ((RKDPUOp.GELU_TANH, RKDPUOp.GELU_TANH_LOCAL) if approximate_tanh else
                        (RKDPUOp.GELU_EXACT, RKDPUOp.GELU_EXACT_LOCAL))
  broad = _DPUExpr(broad_op, (clamp(source, 4.0),))
  local = _DPUExpr(local_op, (_DPUExpr(RKDPUOp.MUL, (clamp(source, 0.5), 8.0)),))
  range_outside = _DPUExpr(RKDPUOp.MAX, (positive(-4.0, source), positive(source, 4.0)))
  range_inside = _DPUExpr(RKDPUOp.SUB, (1.0, range_outside))
  local_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.5), positive(0.5, source)))
  poly_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.04), positive(0.04, source)))
  local_mask = _DPUExpr(RKDPUOp.SUB, (local_inside, poly_inside))
  broad_mask = _DPUExpr(RKDPUOp.SUB, (range_inside, local_inside))
  poly_input = clamp(source, 0.04)
  polynomial = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (poly_input, 0.5)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (poly_input, poly_input)), 1/math.sqrt(2*math.pi)))))
  broad_scale = _DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.MUL, (positive(source, 0.0), 3.0))))
  interior = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (broad, broad_scale)), broad_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (local, 0.5)), local_mask)))), _DPUExpr(RKDPUOp.MUL, (polynomial, poly_inside))))
  fallback = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, 0.0)), range_outside))
  return _DPUExpr(RKDPUOp.ADD, (interior, fallback))

def _erf_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Broad/local erf LUTs, a near-zero line, and exact signed tails."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  def clamp(value:_DPUExpr|RKArg|float, limit:float) -> _DPUExpr:
    lower = _DPUExpr(RKDPUOp.MAX, (value, -limit))
    return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -limit)), -1.0))
  broad = _DPUExpr(RKDPUOp.ERF, (clamp(source, 4.0),))
  local = _DPUExpr(RKDPUOp.ERF_LOCAL, (_DPUExpr(RKDPUOp.MUL, (clamp(source, 0.25), 16.0)),))
  low, high = positive(-4.0, source), positive(source, 4.0)
  outside = _DPUExpr(RKDPUOp.MAX, (low, high))
  inside = _DPUExpr(RKDPUOp.SUB, (1.0, outside))
  local_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.25), positive(0.25, source)))
  near_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.04), positive(0.04, source)))
  broad_mask, local_mask = _DPUExpr(RKDPUOp.SUB, (inside, local_inside)), _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside))
  identity = _DPUExpr(RKDPUOp.MUL, (clamp(source, 0.04), 2/math.sqrt(math.pi)))
  interior = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, broad_mask)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (local, 1/3)), local_mask)))), _DPUExpr(RKDPUOp.MUL, (identity, near_inside))))
  tail = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (high, low)), outside))
  return _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (interior, inside)), tail))

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
  stage_rows:list[tuple[int, ...]] = []
  for stage in image.stages:
    command_start, reloc_start = len(commands), len(relocs)
    commands.extend(stage.commands)
    relocs.extend(stage.relocs)
    stage_rows.append((int(stage.engine), stage.flags, 0, stage.dependencies, command_start, len(stage.commands), reloc_start, len(stage.relocs),
                       _slot_mask(stage.reads), _slot_mask(stage.writes)))
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.stages), len(relocs), len(image.scratch), 0,
                               len(commands), len(image.constants), 0))
  for engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes in stage_rows:
    out += _STAGE.pack(engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes)
  for reloc in relocs:
    out += _RELOC.pack(reloc.stage, reloc.word, int(reloc.kind), reloc.shift, reloc.index, reloc.addend, reloc.mask, reloc.field_shift)
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
      word = patched[reloc.stage][reloc.word]
      value = (word >> 16) & 0xffffffff
      field = ((address(reloc.kind, reloc.index) + reloc.addend) >> reloc.shift) & reloc.mask
      field_mask = (reloc.mask << reloc.field_shift) & 0xffffffff
      patched[reloc.stage][reloc.word] = (word & ~0xffffffff0000) | (((value & ~field_mask) | ((field << reloc.field_shift) & field_mask)) << 16)
  return tuple(tuple(stage) for stage in patched)

def _unwrap_same_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype is u.src[0].dtype: u = u.src[0]
  return u

def _canonical_abs(u:UOp) -> UOp|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for data, sign in (u.src, u.src[::-1]):
    data, sign = _unwrap_same_cast(data), _unwrap_same_cast(sign)
    if data.op is not Ops.INDEX or sign.op is not Ops.WHERE: continue
    cond, nonzero, zero = _unwrap_same_cast(sign.src[0]), _unwrap_same_cast(sign.src[1]), _unwrap_same_cast(sign.src[2])
    if cond.op is not Ops.CMPNE or zero.op is not Ops.CONST or float(zero.arg) != 0 or nonzero.op is not Ops.WHERE: continue
    less, negative, positive = (_unwrap_same_cast(x) for x in nonzero.src)
    compared = tuple(_unwrap_same_cast(x) for x in cond.src)
    if (less.op is Ops.CMPLT and compared[0].key == data.key and compared[1].op is Ops.CONST and float(compared[1].arg) == 0 and
        _unwrap_same_cast(less.src[0]).key == data.key and _unwrap_same_cast(less.src[1]).op is Ops.CONST and
        float(_unwrap_same_cast(less.src[1]).arg) == 0 and negative.op is Ops.CONST and float(negative.arg) == -1 and
        positive.op is Ops.CONST and float(positive.arg) == 1): return data
  return None

def _canonical_hardswish(u:UOp) -> UOp|None:
  """Recognize x*relu6(x+3)/6 and return its single FP16 source INDEX."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  scale = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST and float(x.arg) == 1/6), None)
  product = next((_unwrap_same_cast(x) for x in u.src if scale is not None and _unwrap_same_cast(x).key != scale.key), None)
  if product is None or product.op is not Ops.MUL: return None
  source = next((_unwrap_same_cast(x) for x in product.src if _unwrap_same_cast(x).op is Ops.INDEX), None)
  clamp = next((_unwrap_same_cast(x) for x in product.src if source is not None and _unwrap_same_cast(x).key != source.key), None)
  if source is None or source.dtype is not dtypes.half or clamp is None or clamp.op is not Ops.ADD: return None
  def shifted_relu(v:UOp, negated:bool) -> tuple[UOp,float]|None:
    v = _unwrap_same_cast(v)
    if negated:
      if v.op is not Ops.MUL: return None
      negative = next((_unwrap_same_cast(x) for x in v.src if _unwrap_same_cast(x).op is Ops.CONST and float(x.arg) == -1), None)
      if negative is None: return None
      v = _unwrap_same_cast(v.src[0] if _unwrap_same_cast(v.src[1]).key == negative.key else v.src[1])
    if v.op is not Ops.WHERE: return None
    cond, shifted, zero = (_unwrap_same_cast(x) for x in v.src)
    if (cond.op is not Ops.CMPLT or _unwrap_same_cast(cond.src[0]).op is not Ops.CONST or float(cond.src[0].arg) != 0 or
        _unwrap_same_cast(cond.src[1]).key != shifted.key or zero.op is not Ops.CONST or float(zero.arg) != 0 or shifted.op is not Ops.ADD):
      return None
    offset = next((_unwrap_same_cast(x) for x in shifted.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    base = next((_unwrap_same_cast(x) for x in shifted.src if offset is not None and _unwrap_same_cast(x).key != offset.key), None)
    return (base, float(offset.arg)) if base is not None and offset is not None else None
  for positive, negative in (clamp.src, clamp.src[::-1]):
    pos, neg = shifted_relu(positive, False), shifted_relu(negative, True)
    if pos is not None and neg is not None and pos[0].key == source.key == neg[0].key and pos[1] == 3 and neg[1] == -3: return source
  return None

def _canonical_tanh(u:UOp) -> UOp|None:
  """Recognize tinygrad's 2/(1+exp2(-2*log2(e)*x))-1 decomposition."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.ADD: return None
  term, offset = u.src
  if _unwrap_same_cast(offset).op is not Ops.CONST: term, offset = offset, term
  term, offset = _unwrap_same_cast(term), _unwrap_same_cast(offset)
  if offset.op is not Ops.CONST or float(offset.arg) != -1 or term.op is not Ops.MUL: return None
  scale = next((_unwrap_same_cast(x) for x in term.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  reciprocal = next((_unwrap_same_cast(x) for x in term.src if _unwrap_same_cast(x).op is Ops.RECIPROCAL), None)
  if scale is None or float(scale.arg) != 2 or reciprocal is None or (_unwrap_same_cast(reciprocal.src[0])).op is not Ops.ADD: return None
  denominator = _unwrap_same_cast(reciprocal.src[0])
  exponential = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.EXP2), None)
  one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  if exponential is None or one is None or float(one.arg) != 1 or (_unwrap_same_cast(exponential.src[0])).op is not Ops.MUL: return None
  scaled = _unwrap_same_cast(exponential.src[0])
  source = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is Ops.INDEX), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST), None)
  return source if source is not None and factor is not None and abs(factor + 2.8853900817779268) < 1e-3 else None

def _canonical_scaled_sigmoid(u:UOp, scale:float=1.0) -> UOp|None:
  """Recognize 1/(1+exp2(-scale*log2(e)*x))."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.RECIPROCAL or (_unwrap_same_cast(u.src[0])).op is not Ops.ADD: return None
  denominator = _unwrap_same_cast(u.src[0])
  one = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.CONST), None)
  exponential = next((_unwrap_same_cast(x) for x in denominator.src if _unwrap_same_cast(x).op is Ops.EXP2), None)
  if one is None or float(one.arg) != 1 or exponential is None or (_unwrap_same_cast(exponential.src[0])).op is not Ops.MUL: return None
  scaled = _unwrap_same_cast(exponential.src[0])
  source = next((_unwrap_same_cast(x) for x in scaled.src if _unwrap_same_cast(x).op is Ops.INDEX), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST), None)
  return source if source is not None and factor is not None and abs(factor + scale*1.4426950408889634) < 1e-3 else None

def _canonical_sigmoid(u:UOp) -> UOp|None: return _canonical_scaled_sigmoid(u)

def _canonical_quick_gelu(u:UOp) -> UOp|None:
  """Recognize x*sigmoid(1.702*x)."""
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for source, sigmoid in (u.src, u.src[::-1]):
    source, sigmoid = _unwrap_same_cast(source), _unwrap_same_cast(sigmoid)
    if source.op is Ops.INDEX and (sigmoid_source:=_canonical_scaled_sigmoid(sigmoid, 1.702)) is not None and sigmoid_source.key == source.key:
      return source
  return None

def _canonical_gelu(u:UOp) -> tuple[UOp,bool]|None:
  """Recognize current tanh and exact GELU decompositions."""
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  if len(indexes) != 1 or indexes[0].dtype is not dtypes.half or sum(x.op is Ops.EXP2 for x in nodes) != 1 or \
     sum(x.op is Ops.RECIPROCAL for x in nodes) != 1: return None
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  if sum(x.op is Ops.WHERE for x in nodes) == 0 and any(math.isclose(x, 0.044715) for x in constants): return indexes[0], True
  if sum(x.op is Ops.WHERE for x in nodes) == 2 and any(math.isclose(x, 1/math.sqrt(2)) for x in constants): return indexes[0], False
  return None

def _canonical_erf(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  return indexes[0] if len(indexes) == 1 and indexes[0].dtype is dtypes.half and sum(x.op is Ops.EXP2 for x in nodes) == 1 and \
    sum(x.op is Ops.WHERE for x in nodes) == 2 and any(math.isclose(x, 0.3275911) for x in constants) else None

def _canonical_silu(u:UOp) -> tuple[UOp,UOp]|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for source, sigmoid in (u.src, u.src[::-1]):
    source, sigmoid = _unwrap_same_cast(source), _unwrap_same_cast(sigmoid)
    if source.op is Ops.INDEX and (sigmoid_source:=_canonical_sigmoid(sigmoid)) is not None and sigmoid_source.key == source.key:
      return source, sigmoid
  return None

def _canonical_relu_difference(u:UOp) -> UOp|None:
  """Recognize relu(x+0.5)-relu(x-0.5), tinygrad's clip(x+0.5, 0, 1)."""
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
      cond, shifted, where_zero = (_unwrap_same_cast(x) for x in v.src)
      if (cond.op is not Ops.CMPLT or _unwrap_same_cast(cond.src[0]).op is not Ops.CONST or
          float(_unwrap_same_cast(cond.src[0]).arg) != 0 or _unwrap_same_cast(cond.src[1]).key != shifted.key or
          where_zero.op is not Ops.CONST or float(where_zero.arg) != 0): return None
    elif v.op is Ops.MAX:
      max_zero = next((_unwrap_same_cast(x) for x in v.src if _unwrap_same_cast(x).op is Ops.CONST and
                       float(_unwrap_same_cast(x).arg) == 0), None)
      if max_zero is None: return None
      shifted = _unwrap_same_cast(v.src[0] if _unwrap_same_cast(v.src[1]).key == max_zero.key else v.src[1])
    else: return None
    if shifted.op is not Ops.ADD: return None
    offset = next((_unwrap_same_cast(x) for x in shifted.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    if offset is None: return None
    base = _unwrap_same_cast(shifted.src[0] if _unwrap_same_cast(shifted.src[1]).key == offset.key else shifted.src[1])
    return base, float(offset.arg)
  for positive, negative in (u.src, u.src[::-1]):
    pos, neg = shifted_relu(positive, False), shifted_relu(negative, True)
    if pos is not None and neg is not None and pos[0].key == neg[0].key and pos[1] == 0.5 and neg[1] == -0.5: return pos[0]
  return None

def _parse_mask_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float]) -> _DPUExpr|None:
  """Build an FP16 0/1 predicate from comparisons and boolean composition."""
  u = _unwrap_same_cast(u)
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    operands = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in operands): return None
    lhs, rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], operands)
    positive = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (rhs, lhs)),))
    return positive if u.op is Ops.CMPLT else _DPUExpr(RKDPUOp.MAX,
      (positive, _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))))
  if u.op in (Ops.OR, Ops.AND):
    operands = tuple(_parse_mask_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in operands): return None
    return _DPUExpr(RKDPUOp.MAX if u.op is Ops.OR else RKDPUOp.MUL,
                    cast(tuple[_DPUExpr|RKArg|float, ...], operands))
  return None

def _parse_dpu_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float]) -> _DPUExpr|RKArg|float|None:
  u = _unwrap_same_cast(u)
  if u in memo: return memo[u]
  if u.op is Ops.INDEX and u.dtype is dtypes.half and u.src[0].op is Ops.PARAM and u.src[1].key == output_index.key:
    ret:RKArg|float|_DPUExpr = RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  elif u.op is Ops.CONST and isinstance(u.arg, (int, float)): ret = float(u.arg)
  elif (quick_gelu:=_canonical_quick_gelu(u)) is not None:
    source = _parse_dpu_expr(quick_gelu, output_index, memo)
    if source is None: return None
    ret = _quick_gelu_expr(source)
  elif (gelu:=_canonical_gelu(u)) is not None:
    source = _parse_dpu_expr(gelu[0], output_index, memo)
    if source is None: return None
    ret = _gelu_expr(source, gelu[1])
  elif (erf:=_canonical_erf(u)) is not None:
    source = _parse_dpu_expr(erf, output_index, memo)
    if source is None: return None
    ret = _erf_expr(source)
  elif (silu:=_canonical_silu(u)) is not None:
    operands = tuple(_parse_dpu_expr(x, output_index, memo) for x in silu)
    if any(x is None for x in operands): return None
    ret = _DPUExpr(RKDPUOp.MUL, cast(tuple[_DPUExpr|RKArg|float, ...], operands))
  elif (hs_input:=_canonical_hardswish(u)) is not None:
    source = _parse_dpu_expr(hs_input, output_index, memo)
    if source is None: return None
    broad = _DPUExpr(RKDPUOp.HARDSWISH, (source,))
    positive = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.ADD, (source, 3.0)), 0.0))
    relu_negative = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.ADD, (source, -3.0)), 0.0)), -1.0))
    relu6 = _DPUExpr(RKDPUOp.ADD, (positive, relu_negative))
    fallback = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (source, relu6)), 1/6))
    wide_outside = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (-2.0, source)),)),
      _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, 2.0)),))))
    wide = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, wide_outside)))),
      _DPUExpr(RKDPUOp.MUL, (fallback, wide_outside))))
    local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.HARDSWISH_LOCAL,
      (_DPUExpr(RKDPUOp.MUL, (source, 16.0)),)), 1/16))
    inside = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, -0.125)),)),
      _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (15/128, source)),))))
    nonzero = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (0.0, source)),)),
      _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, 0.0)),))))
    corrected = _DPUExpr(RKDPUOp.MUL, (local, nonzero))
    ret = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (wide, _DPUExpr(RKDPUOp.SUB, (1.0, inside)))),
      _DPUExpr(RKDPUOp.MUL, (corrected, inside))))
  elif (tanh_input:=_canonical_tanh(u)) is not None:
    source = _parse_dpu_expr(tanh_input, output_index, memo)
    if source is None: return None
    tanh_source:_DPUExpr|RKArg|float = source
    def posmask(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
      return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
    def interval(low:float, high:float) -> _DPUExpr: return _DPUExpr(RKDPUOp.MUL, (posmask(tanh_source, low), posmask(high, tanh_source)))
    broad, local_inside, near_inside = _DPUExpr(RKDPUOp.TANH, (source,)), interval(-0.25, 0.25), interval(-0.04, 0.04)
    local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.TANH_LOCAL, (_DPUExpr(RKDPUOp.MUL, (source, 16.0)),)), 0.25))
    lower = _DPUExpr(RKDPUOp.MAX, (source, -0.04))
    identity = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -0.04)), -1.0))
    interior = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
      (broad, _DPUExpr(RKDPUOp.SUB, (1.0, local_inside)))), _DPUExpr(RKDPUOp.MUL,
      (local, _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)))))), _DPUExpr(RKDPUOp.MUL, (identity, near_inside))))
    low_mask, high_mask = posmask(-4.0, source), posmask(source, 4.0)
    outside = _DPUExpr(RKDPUOp.MAX, (high_mask, low_mask))
    sign = _DPUExpr(RKDPUOp.SUB, (high_mask, low_mask))
    ret = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (interior, _DPUExpr(RKDPUOp.SUB, (1.0, outside)))),
      _DPUExpr(RKDPUOp.MUL, (sign, outside))))
  elif (sigmoid_input:=_canonical_sigmoid(u)) is not None:
    source = _parse_dpu_expr(sigmoid_input, output_index, memo)
    if source is None: return None
    ret = _sigmoid_expr(source)
  elif (clamp_base:=_canonical_relu_difference(u)) is not None:
    base = _parse_dpu_expr(clamp_base, output_index, memo)
    if base is None: return None
    positive = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.ADD, (base, 0.5)), 0.0))
    ret = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (positive, -1.0)), -1.0)), -1.0))
  elif (abs_input:=_canonical_abs(u)) is not None:
    operand = _parse_dpu_expr(abs_input, output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.MAX, (operand, _DPUExpr(RKDPUOp.MUL, (operand, -1.0))))
  elif u.op is Ops.MUL and any(_unwrap_same_cast(x).op is Ops.RECIPROCAL for x in u.src):
    reciprocal = next(i for i,x in enumerate(u.src) if _unwrap_same_cast(x).op is Ops.RECIPROCAL)
    numerator, denominator = u.src[1-reciprocal], _unwrap_same_cast(u.src[reciprocal]).src[0]
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in (numerator, denominator))
    if any(x is None for x in src): return None
    ret = _DPUExpr(RKDPUOp.DIV, cast(tuple[_DPUExpr|RKArg|float, ...], src))
  elif u.op in (Ops.ADD, Ops.MUL, Ops.MAX):
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in src): return None
    ret = _DPUExpr({Ops.ADD:RKDPUOp.ADD, Ops.MUL:RKDPUOp.MUL, Ops.MAX:RKDPUOp.MAX}[u.op], src)  # type: ignore[arg-type]
  elif u.op is Ops.EXP2:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.EXP2, (operand,))
  elif u.op is Ops.RECIPROCAL:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.DIV, (1.0, operand))
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
    operands = tuple(_parse_dpu_expr(x, output_index, memo) for x in (lhs_u, rhs_u))
    if any(x is None for x in operands): return None
    parsed = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], operands)
    if ordered_max: ret = _DPUExpr(RKDPUOp.MAX, parsed)
    elif ordered_min:
      negative = tuple(_DPUExpr(RKDPUOp.MUL, (x, -1.0)) for x in parsed)
      ret = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, negative), -1.0))
    else:
      mask = _parse_mask_expr(cond, output_index, memo)
      arms = tuple(_parse_dpu_expr(x, output_index, memo) for x in (true_u, false_u))
      if mask is None or any(x is None for x in arms): return None
      true, false = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], arms)
      ret = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (true, mask)),
        _DPUExpr(RKDPUOp.MUL, (false, _DPUExpr(RKDPUOp.SUB, (1.0, mask))))))
  else: return None
  memo[u] = ret
  return ret

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Lower one contiguous fp16/int32 store to a UOp-free primitive DPU plan."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1 or (store:=stores[0]).src[0].op is not Ops.INDEX or \
     store.src[0].dtype not in (dtypes.half, dtypes.int, dtypes.float): return None
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op not in (Ops.RANGE, Ops.CONST) or out_param.src[0].op is not Ops.CONST: return None
  count = int(out_param.src[0].arg)
  if count <= 0 or (out_index.op is Ops.RANGE and int(out_index.src[0].arg) != count) or \
     (out_index.op is Ops.CONST and (count != 1 or int(out_index.arg) != 0)): return None
  root = _parse_dpu_expr(store.src[1], out_index, {})
  if root is None: return None
  # Native int32 WDMA emits four values per eight-lane fp16 atom. Constant
  # fills can safely double their source lanes; dynamic conversion needs an
  # explicit packed layout and is deliberately not inferred here.
  if store.src[0].dtype is not dtypes.half and not isinstance(root, float): return None
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if not isinstance(root, _DPUExpr):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      tile = 64 if store.src[0].dtype is dtypes.int else 4
      fill_stages = tuple(RKDPUStage(RKDPUOp.ADD, RKArg(output.kind, output.index, start*4), 0.0, root,
                                    min(tile, count-start), store.src[0].dtype) for start in range(0, count, tile))
      return RKDPUProgram(fill_stages, ()) if len(fill_stages) <= 64 else None
    stage = RKDPUStage(RKDPUOp.ADD, output, 0.0, root, count, store.src[0].dtype)
    return RKDPUProgram((stage,), ())
  if root.op is RKDPUOp.EXP2 and len(root.src) == 1 and isinstance(root.src[0], RKArg):
    source, base = root.src[0], root
    positive_inf = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, 65504.0)),))
    negative_inf = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (-65504.0, source)),))
    finite = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.DIV, (base, _DPUExpr(RKDPUOp.SUB, (1.0, positive_inf)))),
      _DPUExpr(RKDPUOp.SUB, (1.0, negative_inf))))
    not_number = _DPUExpr(RKDPUOp.MUL, (positive_inf, negative_inf))
    nan_denom = _DPUExpr(RKDPUOp.SUB, (1.0, not_number))
    root = _DPUExpr(RKDPUOp.DIV, (_DPUExpr(RKDPUOp.MUL, (finite, nan_denom)), nan_denom))
  # Rejected HardSwish WIP: OUT_CVT post-scaling overflowed; BS FP16 1/6 missed 9 cases, its upper neighbor 92, and reordered stages 4.
  order:list[_DPUExpr] = []
  def visit(expr:_DPUExpr) -> None:
    for src in expr.src:
      if isinstance(src, _DPUExpr) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  visit(root)
  uses = {expr:sum(src == expr for node in order for src in node.src) for expr in order}
  values:dict[_DPUExpr, RKArg] = {}
  free:list[int] = []
  scratch_count, stages = 0, []
  for expr in order:
    src = tuple(values[x] if isinstance(x, _DPUExpr) else x for x in expr.src)
    if expr is root: dst = output
    elif (reuse:=next((values[x] for x in expr.src if isinstance(x, _DPUExpr) and uses[x] == 1 and
                       values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    stages.append(RKDPUStage(expr.op, dst, src[0], src[1] if len(src) > 1 else None, count,
                             store.src[0].dtype if expr is root else dtypes.half))
    values[expr] = dst
    for dependency in expr.src:
      if isinstance(dependency, _DPUExpr):
        uses[dependency] -= 1
        arg = values[dependency]
        if uses[dependency] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
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
  """Recognize directly legal M=1, K=32 affine contractions and row sums."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return None
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].dtype is not dtypes.half or reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return None
  body, red = _strip_casts(reduce.src[0]), reduce.src[1]
  out_param, out_aff = store.src[0].src[0], _affine(store.src[0].src[1])
  if out_param.op is not Ops.PARAM or out_aff is None or out_aff[1] or len(out_aff[0]) != 1: return None
  if body.op is Ops.INDEX:
    inp_aff = _affine(body.src[1])
    out_axis, red_axis, n = next(iter(out_aff[0])), red.arg[0], int(out_param.src[0].arg)
    if (body.dtype is not dtypes.half or body.src[0].op is not Ops.PARAM or red.op is not Ops.RANGE or int(red.src[0].arg) != 32 or
        not 4 <= n <= 16 or out_aff[0] != {out_axis:1} or inp_aff != ({out_axis:32, red_axis:1}, 0) or
        int(body.src[0].src[0].arg) != n*32): return None
    ones = RKView(0, (1,32), (32,1), kind=RKBufferKind.CONSTANT)
    return RKContract(RKView(out_param.arg.slot, (1,n), (n,1)), ones,
                      RKView(body.src[0].arg.slot, (n,32), (32,1)), (red_axis,))
  if body.op is not Ops.MUL or red.op is not Ops.RANGE or int(red.src[0].arg) != 32: return None
  lhs, rhs = (_strip_casts(x) for x in body.src)
  if any(x.op is not Ops.INDEX or x.dtype is not dtypes.half or x.src[0].op is not Ops.PARAM for x in (lhs, rhs)): return None
  lhs_aff, rhs_aff = _affine(lhs.src[1]), _affine(rhs.src[1])
  if lhs_aff is None or rhs_aff is None or any(x[1] for x in (lhs_aff, rhs_aff)): return None
  red_axis = red.arg[0]
  out_axes = tuple(out_aff[0])
  if len(out_axes) != 1 or out_aff[0][out_axes[0]] != 1 or lhs_aff[0] != {red_axis:1}: return None
  n_axis, n = out_axes[0], next(int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.arg[0] == out_axes[0])
  if not 1 <= n <= 16 or rhs_aff[0] != {n_axis:32, red_axis:1}: return None
  if out_param.op is not Ops.PARAM or int(out_param.src[0].arg) != n or int(lhs.src[0].src[0].arg) != 32 or int(rhs.src[0].src[0].arg) != n*32:
    return None
  return RKContract(RKView(out_param.arg.slot, (1,n), (n,1)), RKView(lhs.src[0].arg.slot, (1,32), (32,1)),
                    RKView(rhs.src[0].arg.slot, (n,32), (32,1)), (red_axis,))

def lower_pool(sink:UOp) -> RKPool|None:
  """Recognize global MAX over an explicitly legal (K, 8) HWC surface."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return None
  store, reduce = stores[0], reductions[0]
  value = _strip_casts(reduce.src[0])
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) != 2 or value.op is not Ops.INDEX or value.dtype is not dtypes.half: return None
  red, out_param, inp_param = reduce.src[1], store.src[0].src[0], value.src[0]
  if red.op is not Ops.RANGE or store.src[0].op is not Ops.INDEX or out_param.op is not Ops.PARAM or inp_param.op is not Ops.PARAM: return None
  out_aff, inp_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or inp_aff is None or out_aff[1] or inp_aff[1] or len(out_aff[0]) != 1: return None
  channel_axis, red_axis = next(iter(out_aff[0])), red.arg[0]
  k, channels = int(red.src[0].arg), int(out_param.src[0].arg)
  if channels != 8 or out_aff[0] != {channel_axis:1} or inp_aff[0] != {red_axis:channels, channel_axis:1}: return None
  split = next(((h, k//h) for h in range(2, min(k,16)+1) if k%h == 0 and 2 <= k//h <= 16 and h != 9 and k//h != 9 and
                (h,k//h) not in ((3,6),(6,3),(12,12))), (1,k) if 4 <= k <= 16 else None)
  if split is None or int(inp_param.src[0].arg) != k*channels: return None
  return RKPool(RKView(out_param.arg.slot, (channels,), (1,)), RKView(inp_param.arg.slot, (k,channels), (channels,1)), split)

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_TARGET_CNA, _TARGET_CORE = 0x201, 0x801
_TARGET_PPU, _TARGET_PPU_RDMA = 0x4001, 0x8001
_EW_BASE = 0x108002c0
_EW_CFG = {RKDPUOp.ADD:_EW_BASE | (2 << 16), RKDPUOp.MUL:_EW_BASE | (1 << 2) | (1 << 8),
           RKDPUOp.MAX:_EW_BASE, RKDPUOp.SUB:_EW_BASE | (4 << 16), RKDPUOp.DIV:_EW_BASE | (3 << 16) | (1 << 8)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _emit_lut(stage_idx:int, plan:RKDPUStage, src:RKArg) -> RKStage:
  """Emit one variable-width generated LUT task; fitting stays in extra/rockchip/gen_lut.py."""
  width, surf_stride = (plan.count+7)//8-1, ((plan.count+7)//8)*16
  table, bn_mul, minus_exp = {
    RKDPUOp.EXP2:(rklut.RK_LUT_EXP2, rklut.RK_LUT_EXP2_BN_MUL, rklut.RK_LUT_EXP2_MINUS_EXP),
    RKDPUOp.HARDSWISH:(rklut.RK_LUT_HARDSWISH, rklut.RK_LUT_HARDSWISH_BN_MUL, rklut.RK_LUT_HARDSWISH_MINUS_EXP),
    RKDPUOp.HARDSWISH_LOCAL:(rklut.RK_LUT_HARDSWISH_LOCAL, rklut.RK_LUT_HARDSWISH_LOCAL_BN_MUL,
                            rklut.RK_LUT_HARDSWISH_LOCAL_MINUS_EXP),
    RKDPUOp.TANH:(rklut.RK_LUT_TANH, rklut.RK_LUT_TANH_BN_MUL, rklut.RK_LUT_TANH_MINUS_EXP),
    RKDPUOp.TANH_LOCAL:(rklut.RK_LUT_TANH_LOCAL, rklut.RK_LUT_TANH_LOCAL_BN_MUL, rklut.RK_LUT_TANH_LOCAL_MINUS_EXP),
    RKDPUOp.SIGMOID:(rklut.RK_LUT_SIGMOID, rklut.RK_LUT_SIGMOID_BN_MUL, rklut.RK_LUT_SIGMOID_MINUS_EXP),
    RKDPUOp.SIGMOID_LOCAL:(rklut.RK_LUT_SIGMOID_LOCAL, rklut.RK_LUT_SIGMOID_LOCAL_BN_MUL,
                          rklut.RK_LUT_SIGMOID_LOCAL_MINUS_EXP),
    RKDPUOp.QUICK_GELU:(rklut.RK_LUT_QUICK_GELU, rklut.RK_LUT_QUICK_GELU_BN_MUL, rklut.RK_LUT_QUICK_GELU_MINUS_EXP),
    RKDPUOp.QUICK_GELU_LOCAL:(rklut.RK_LUT_QUICK_GELU_LOCAL, rklut.RK_LUT_QUICK_GELU_LOCAL_BN_MUL,
                             rklut.RK_LUT_QUICK_GELU_LOCAL_MINUS_EXP),
    RKDPUOp.GELU_TANH:(rklut.RK_LUT_GELU_TANH, rklut.RK_LUT_GELU_TANH_BN_MUL, rklut.RK_LUT_GELU_TANH_MINUS_EXP),
    RKDPUOp.GELU_TANH_LOCAL:(rklut.RK_LUT_GELU_TANH_LOCAL, rklut.RK_LUT_GELU_TANH_LOCAL_BN_MUL,
                            rklut.RK_LUT_GELU_TANH_LOCAL_MINUS_EXP),
    RKDPUOp.GELU_EXACT:(rklut.RK_LUT_GELU_EXACT, rklut.RK_LUT_GELU_EXACT_BN_MUL, rklut.RK_LUT_GELU_EXACT_MINUS_EXP),
    RKDPUOp.GELU_EXACT_LOCAL:(rklut.RK_LUT_GELU_EXACT_LOCAL, rklut.RK_LUT_GELU_EXACT_LOCAL_BN_MUL,
                             rklut.RK_LUT_GELU_EXACT_LOCAL_MINUS_EXP),
    RKDPUOp.ERF:(rklut.RK_LUT_ERF, rklut.RK_LUT_ERF_BN_MUL, rklut.RK_LUT_ERF_MINUS_EXP),
    RKDPUOp.ERF_LOCAL:(rklut.RK_LUT_ERF_LOCAL, rklut.RK_LUT_ERF_LOCAL_BN_MUL, rklut.RK_LUT_ERF_LOCAL_MINUS_EXP)}[plan.op]
  cmds = []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    for value in table[table_id*513:(table_id+1)*513]:
      cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value & 0xffffffff if value < 0 else value))
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
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), (_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
  cmds.extend(_command(*x) for x in fixed[:4])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[4:30])
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[30:])
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, src.kind, src.index, src.addend))
  reads = (src.index,) if src.kind is RKBufferKind.ARG else ()
  writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

def _emit_mask(stage_idx:int, plan:RKDPUStage, src:RKArg) -> RKStage:
  width = (plan.count+7)//8-1
  regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53),
    (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width), (rk.REG_DPU_BN_MUL_CFG, 0),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_BS_CFG, 0x40040), (rk.REG_DPU_BS_ALU_CFG, 0x33800000),
    (rk.REG_DPU_BS_MUL_CFG, 0x40000000), (rk.REG_DPU_BN_CFG, 0x40082), (rk.REG_DPU_BN_MUL_CFG, 0x7c000000),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000), (rk.REG_DPU_EW_CFG, _EW_BASE|1),
    (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1), (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
  cmds = [_command(_TARGET_DPU, reg, value) for reg,value in regs]
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))
  cmds.extend(_command(_TARGET_DPU_RDMA, reg, value) for reg,value in rdma)
  relocs = []
  for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, src),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, src)):
    cmds.append(_command(target_id, reg, 0))
    relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
           _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  reads = (src.index,) if src.kind is RKBufferKind.ARG else ()
  writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

def emit_dpu(program:RKDPUProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  constants, constant_offsets, stages = bytearray(), {}, []
  def materialize(value:RKArg|float, count:int) -> RKArg:
    if isinstance(value, RKArg): return value
    bits = struct.pack("<e", value)
    key = bits, count
    if key not in constant_offsets:
      constant_offsets[key] = len(constants)
      constants.extend(bits * (((count+7)//8)*8))
    return RKArg(RKBufferKind.CONSTANT, constant_offsets[key])
  for stage_idx, plan in enumerate(program.stages):
    material_count = plan.count*2 if plan.out_dtype is dtypes.int else (32 if plan.out_dtype is dtypes.float else plan.count)
    lhs, rhs = materialize(plan.lhs, material_count), materialize(plan.rhs, material_count) if plan.rhs is not None else None
    if plan.op in (RKDPUOp.EXP2, RKDPUOp.HARDSWISH, RKDPUOp.HARDSWISH_LOCAL, RKDPUOp.TANH, RKDPUOp.TANH_LOCAL,
                   RKDPUOp.SIGMOID, RKDPUOp.SIGMOID_LOCAL, RKDPUOp.QUICK_GELU, RKDPUOp.QUICK_GELU_LOCAL,
                   RKDPUOp.GELU_TANH, RKDPUOp.GELU_TANH_LOCAL, RKDPUOp.GELU_EXACT, RKDPUOp.GELU_EXACT_LOCAL,
                   RKDPUOp.ERF, RKDPUOp.ERF_LOCAL):
      stages.append(_emit_lut(stage_idx, plan, lhs))
      continue
    if plan.op is RKDPUOp.MASK:
      stages.append(_emit_mask(stage_idx, plan, lhs))
      continue
    width = ((plan.count*2 if plan.out_dtype is dtypes.int else plan.count)+7)//8-1
    wide_out = plan.out_dtype in (dtypes.int, dtypes.float)
    out_precision = 4 if plan.out_dtype is dtypes.int else (5 if plan.out_dtype is dtypes.float else 2)
    dpu_regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
      (rk.REG_DPU_DATA_FORMAT, (out_precision<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0), (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
      (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0),
      (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 3 if wide_out else 7), (rk.REG_DPU_WDMA_SIZE_1, width),
      (rk.REG_DPU_BN_MUL_CFG, 0), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0),
      (rk.REG_DPU_EW_CFG, 0 if plan.op is RKDPUOp.COPY else _EW_CFG[plan.op]), (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
      (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE,
       0 if plan.out_dtype is dtypes.float else (1 if plan.op is RKDPUOp.DIV or plan.out_dtype is dtypes.int else 0x10001)),
      (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
    rdma_regs = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))
    cmds = [_command(_TARGET_DPU, *x) for x in dpu_regs] + [_command(_TARGET_DPU_RDMA, *x) for x in rdma_regs]
    relocs = []
    for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, lhs)):
      cmds.append(_command(target_id, reg, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
    if rhs is not None:
      cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, rhs.kind, rhs.index, rhs.addend))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if plan.op is RKDPUOp.DIV else 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    operands = (lhs,) if rhs is None else (lhs, rhs)
    reads = tuple(sorted({x.index for x in operands if x.kind is RKBufferKind.ARG}))
    writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

def emit_contract(plan:RKContract, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit a direct FP16 CMAC surface; inputs and output are already in hardware-legal layouts."""
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  m, align = 1, 32
  def e(target_id:int, reg:int, value:int) -> int: return _command(target_id, reg, value)
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
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, m-1), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0x70007),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x1f001f), e(_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 0x126), e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, align-1),
    e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, 0), e(_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x383), e(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    e(_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 0x40), e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
  relocs = (RKReloc(0, 18, plan.lhs.kind, plan.lhs.slot), RKReloc(0, 24, plan.rhs.kind, plan.rhs.slot),
            RKReloc(0, 31, plan.out.kind, plan.out.slot))
  reads = tuple(x.slot for x in (plan.lhs, plan.rhs) if x.kind is RKBufferKind.ARG)
  constants = struct.pack("<e", 1.0)*32 if any(x.kind is RKBufferKind.CONSTANT for x in (plan.lhs, plan.rhs)) else b""
  return RKImage(target, (RKStage(RKEngine.CMAC, commands, relocs, reads, (plan.out.slot,), flags=RK_STAGE_RESET),), constants=constants)

def emit_pool(plan:RKPool, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  height, width = plan.kernel
  h, w, c, stride = height-1, width-1, plan.out.shape[0]-1, width*plan.out.shape[0]*2
  e = _command
  commands = (e(_TARGET_PPU, rk.REG_PPU_S_POINTER, 0x0e), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_S_POINTER, 0x0e),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, w), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, h),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, c), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_WIDTH, 0),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_HEIGHT, 0), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, c),
    e(_TARGET_PPU, rk.REG_PPU_OPERATION_MODE_CFG, 0x11), e(_TARGET_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (h<<20)|(w<<16)|(h<<8)|w),
    e(_TARGET_PPU, rk.REG_PPU_DST_BASE_ADDR, 0), e(_TARGET_PPU, rk.REG_PPU_DST_SURF_STRIDE, 1),
    e(_TARGET_PPU, rk.REG_PPU_DATA_FORMAT, 0x10002), e(_TARGET_PPU, rk.REG_PPU_MISC_CTRL, 3),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH, w), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, h),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, c), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, stride),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, stride*height),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_DATA_FORMAT, 2), e(_TARGET_PPU_RDMA, 0x7038, 1),
    e(_TARGET_PPU, rk.REG_PPU_RECIP_KERNEL_WIDTH, 0), e(_TARGET_PPU, rk.REG_PPU_RECIP_KERNEL_HEIGHT, 0),
    e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x60))
  relocs = (RKReloc(0, 10, RKBufferKind.ARG, plan.out.slot, shift=4, mask=0x0fffffff, field_shift=4),
            RKReloc(0, 17, RKBufferKind.ARG, plan.inp.slot))
  return RKImage(target, (RKStage(RKEngine.PPU, commands, relocs, (plan.inp.slot,), (plan.out.slot,), flags=RK_STAGE_RESET),))

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int, dtypes.float}
  def native_program(self, ast:UOp) -> UOp|None:
    if (dpu:=lower_dpu(ast)) is not None: image = emit_dpu(dpu)
    elif (contract:=lower_contract(ast)) is not None: image = emit_contract(contract)
    elif (pool:=lower_pool(ast)) is not None: image = emit_pool(pool)
    else: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
    return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""),
      UOp(Ops.BINARY, arg=encode_image(image))), arg=ProgramInfo.from_sink(ast, self.target))
