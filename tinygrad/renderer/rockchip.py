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

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 11, 1
_HEADER = struct.Struct("<4sHHHHHHIIIQQQQQQQQQ")
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
  MASK = 5
  DIV = 6
  LUT = 7

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
  fp32_inputs: tuple[int, ...] = ()
  fp32_outputs: tuple[int, ...] = ()
  bool_outputs: tuple[int, ...] = ()
  bool_inputs: tuple[int, ...] = ()
  int_inputs: tuple[int, ...] = ()
  tiled_inputs: tuple[int, ...] = ()
  int_outputs: tuple[int, ...] = ()
  transposed_int_inputs: tuple[int, ...] = ()
  tiled_int_inputs: tuple[int, ...] = ()
  raw_int_inputs: tuple[int, ...] = ()
  numeric_int_inputs: tuple[int, ...] = ()

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
  lut: rklut.RKLUT|None = None

  def __post_init__(self):
    if (self.op is RKDPUOp.LUT) != (self.lut is not None): raise ValueError("Rockchip LUT stage requires exactly one LUT asset")

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...]
  fp32_inputs: tuple[int, ...] = ()
  fp32_outputs: tuple[int, ...] = ()
  bool_outputs: tuple[int, ...] = ()
  bool_inputs: tuple[int, ...] = ()
  int_inputs: tuple[int, ...] = ()
  tiled_inputs: tuple[int, ...] = ()
  int_outputs: tuple[int, ...] = ()
  transposed_int_inputs: tuple[int, ...] = ()
  tiled_int_inputs: tuple[int, ...] = ()
  raw_int_inputs: tuple[int, ...] = ()
  numeric_int_inputs: tuple[int, ...] = ()

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
  op: RKDPUOp|rklut.RKLUT
  src: tuple[_DPUExpr|RKArg|float, ...]
  lut: rklut.RKLUT|None = None

  def __post_init__(self):
    if isinstance(self.op, rklut.RKLUT):
      object.__setattr__(self, "lut", self.op)
      object.__setattr__(self, "op", RKDPUOp.LUT)

def _sigmoid_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Two-level sigmoid LUT with analytic tails and preserved NaN behavior."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  broad, local = _DPUExpr(rklut.RKLUT.SIGMOID, (source,)), _DPUExpr(rklut.RKLUT.SIGMOID_LOCAL, (source,))
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
  broad, local = _DPUExpr(rklut.RKLUT.QUICK_GELU, (source,)), _DPUExpr(rklut.RKLUT.QUICK_GELU_LOCAL,
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
  broad_op, local_op = ((rklut.RKLUT.GELU_TANH, rklut.RKLUT.GELU_TANH_LOCAL) if approximate_tanh else
                        (rklut.RKLUT.GELU_EXACT, rklut.RKLUT.GELU_EXACT_LOCAL))
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
  broad = _DPUExpr(rklut.RKLUT.ERF, (clamp(source, 4.0),))
  local = _DPUExpr(rklut.RKLUT.ERF_LOCAL, (_DPUExpr(RKDPUOp.MUL, (clamp(source, 0.25), 16.0)),))
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

def _asin_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Two-task asin: broad interior plus one detail LUT for center and singular endpoints."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative_source = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  magnitude = _DPUExpr(RKDPUOp.MAX, (source, negative_source))
  bounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (magnitude, -1.0)), -1.0)), -1.0))
  negative, invalid = positive(0.0, source), positive(magnitude, 1.0)
  positive_sign = _DPUExpr(RKDPUOp.SUB, (1.0, negative))
  endpoint, near_outside, local_outside = (positive(bounded, x) for x in (.875, .04, .125))
  near_inside, local_inside = (_DPUExpr(RKDPUOp.SUB, (1.0, x)) for x in (near_outside, local_outside))
  local_mask, middle_mask = (_DPUExpr(RKDPUOp.SUB, x) for x in ((local_inside, near_inside), (local_outside, endpoint)))
  detail_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (bounded, -1.0)), local_inside)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (1.0, bounded)), endpoint))))
  broad, detail = _DPUExpr(rklut.RKLUT.ASIN, (bounded,)), _DPUExpr(rklut.RKLUT.ASIN_DETAIL, (detail_input,))
  absolute = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (broad, 2.0)), middle_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (detail, .25)), local_mask)))), _DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (detail, 2.0)), endpoint)), _DPUExpr(RKDPUOp.MUL, (bounded, near_inside))))))
  signed = _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.MUL, (absolute, positive_sign)), _DPUExpr(RKDPUOp.MUL, (absolute, negative))))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, invalid))
  return _DPUExpr(RKDPUOp.MUL, (signed, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _acos_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Asymmetric broad ACOS plus coarse/fine endpoint-distance LUTs."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative_source = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  magnitude = _DPUExpr(RKDPUOp.MAX, (source, negative_source))
  bounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (magnitude, -1.0)), -1.0)), -1.0))
  negative, invalid = positive(0.0, source), positive(magnitude, 1.0)
  positive_sign, endpoint = _DPUExpr(RKDPUOp.SUB, (1.0, negative)), positive(bounded, .85)
  distance = _DPUExpr(RKDPUOp.SUB, (1.0, bounded))
  broad = _DPUExpr(rklut.RKLUT.ACOS, (_DPUExpr(RKDPUOp.SUB, (bounded,
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (bounded, negative)), 2.0)))),))
  coarse, fine = _DPUExpr(rklut.RKLUT.ACOS_ENDPOINT, (distance,)), _DPUExpr(rklut.RKLUT.ACOS_FINE_ENDPOINT,
    (_DPUExpr(RKDPUOp.MUL, (distance, 64.0)),))
  coarse_mask = positive(distance, .003)
  endpoint_value = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (coarse, coarse_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (fine, .125)), _DPUExpr(RKDPUOp.SUB, (1.0, coarse_mask))))))
  broad_decoded = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (broad, 2.0)), positive_sign)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (broad, 4.0)), negative))))
  exact_one = positive(bounded, .99975)
  endpoint_value = _DPUExpr(RKDPUOp.MUL, (endpoint_value, _DPUExpr(RKDPUOp.SUB, (1.0, exact_one))))
  endpoint_decoded = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (endpoint_value, positive_sign)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.SUB, (math.pi, endpoint_value)), negative))))
  selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad_decoded, _DPUExpr(RKDPUOp.SUB, (1.0, endpoint)))),
    _DPUExpr(RKDPUOp.MUL, (endpoint_decoded, endpoint))))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, invalid))
  return _DPUExpr(RKDPUOp.MUL, (selected, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _atan_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Two-task atan after reciprocal folding maps every finite magnitude into [0,1]."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative_source = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  absolute = _DPUExpr(RKDPUOp.MAX, (source, negative_source))
  negative, large = positive(0.0, source), positive(absolute, 1.0)
  nonnegative, small = _DPUExpr(RKDPUOp.SUB, (1.0, negative)), _DPUExpr(RKDPUOp.SUB, (1.0, large))
  inverse = _DPUExpr(RKDPUOp.DIV, (1.0, _DPUExpr(RKDPUOp.ADD, (absolute, small))))
  transformed = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (absolute, small)), _DPUExpr(RKDPUOp.MUL, (inverse, large))))
  near_outside, local_outside = positive(transformed, .04), positive(transformed, .125)
  near_inside, local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, near_outside)), _DPUExpr(RKDPUOp.SUB, (1.0, local_outside))
  detail_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (transformed, -4.0)), _DPUExpr(RKDPUOp.MUL, (local_inside, small)))),
    _DPUExpr(RKDPUOp.MUL, (transformed, large))))
  broad, detail = _DPUExpr(rklut.RKLUT.ATAN, (transformed,)), _DPUExpr(rklut.RKLUT.ATAN_DETAIL, (detail_input,))
  magnitude = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (broad, _DPUExpr(RKDPUOp.MUL, (local_outside, small)))), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (detail, .25)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)), small)))))), _DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.MUL, (transformed, _DPUExpr(RKDPUOp.MUL, (near_inside, small)))),
     _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (detail, 2.0)), large))))))
  return _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.MUL, (magnitude, nonnegative)), _DPUExpr(RKDPUOp.MUL, (magnitude, negative))))

def _sin_cos_expr(source:_DPUExpr|RKArg|float, is_cos:bool) -> _DPUExpr:
  """Cody-Waite FP16 range reduction followed by regional sine/cosine LUTs."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  bounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, -10000.0)), -1.0)), -10000.0)), -1.0))
  rounded = _round_expr(_DPUExpr(RKDPUOp.MUL, (bounded, 1/(2*math.pi))))
  reduced:_DPUExpr|RKArg|float = bounded
  for coefficient in (4.0, 2.0, .25, .03125, 2*math.pi-6.28125):
    reduced = _DPUExpr(RKDPUOp.SUB, (reduced, _DPUExpr(RKDPUOp.MUL, (rounded, coefficient))))
  if is_cos:
    broad, local = _DPUExpr(rklut.RKLUT.COS, (reduced,)), _DPUExpr(rklut.RKLUT.COS_LOCAL, (reduced,))
    abs_broad = _DPUExpr(RKDPUOp.MAX, (broad, _DPUExpr(RKDPUOp.MUL, (broad, -1.0))))
    local_inside, near_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(abs_broad, .5))), _DPUExpr(RKDPUOp.SUB, (1.0, positive(abs_broad, .01)))
    middle, broad_mask = _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)), _DPUExpr(RKDPUOp.SUB, (1.0, local_inside))
    abs_reduced = _DPUExpr(RKDPUOp.MAX, (reduced, _DPUExpr(RKDPUOp.MUL, (reduced, -1.0))))
    near = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.SUB, (1.5703125, abs_reduced)), math.pi/2-1.5703125))
    normal = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, broad_mask)), _DPUExpr(RKDPUOp.ADD,
      (_DPUExpr(RKDPUOp.MUL, (near, near_inside)),
       _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (local, .5)), middle))))))
  else:
    broad, local = _DPUExpr(rklut.RKLUT.SIN, (reduced,)), _DPUExpr(rklut.RKLUT.SIN_LOCAL, (_DPUExpr(RKDPUOp.MUL, (reduced, 16.0)),))
    abs_reduced = _DPUExpr(RKDPUOp.MAX, (reduced, _DPUExpr(RKDPUOp.MUL, (reduced, -1.0))))
    local_inside, near_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(abs_reduced, .125))), _DPUExpr(RKDPUOp.SUB, (1.0, positive(abs_reduced, .04)))
    normal = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, local_inside)))),
      _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (local, .125)),
      _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)))), _DPUExpr(RKDPUOp.MUL, (reduced, near_inside))))))
  return _DPUExpr(RKDPUOp.ADD, (normal, _DPUExpr(RKDPUOp.MUL, (source, 0.0))))

def _atanh_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Broad/detail atanh with explicit endpoint infinities and invalid-domain NaN."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative_source = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  absolute = _DPUExpr(RKDPUOp.MAX, (source, negative_source))
  bounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (absolute, -1.0)), -1.0)), -1.0))
  negative, invalid = positive(0.0, source), positive(absolute, 1.0)
  nonnegative, endpoint = _DPUExpr(RKDPUOp.SUB, (1.0, negative)), positive(bounded, .875)
  exact, near_outside, local_outside = positive(bounded, .99975), positive(bounded, .04), positive(bounded, .125)
  near_inside, local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, near_outside)), _DPUExpr(RKDPUOp.SUB, (1.0, local_outside))
  local_mask, broad_mask = _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)), _DPUExpr(RKDPUOp.SUB, (local_outside, endpoint))
  detail_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (bounded, -1.0)), local_inside)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (1.0, bounded)), endpoint))))
  broad, detail = _DPUExpr(rklut.RKLUT.ATANH, (bounded,)), _DPUExpr(rklut.RKLUT.ATANH_DETAIL, (detail_input,))
  magnitude = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (broad, 4.0)), broad_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (detail, .25)), local_mask)))), _DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (detail, 8.0)), endpoint)),
     _DPUExpr(RKDPUOp.MUL, (bounded, near_inside))))))
  signed = _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.MUL, (magnitude, nonnegative)), _DPUExpr(RKDPUOp.MUL, (magnitude, negative))))
  signed = _DPUExpr(RKDPUOp.DIV, (signed, _DPUExpr(RKDPUOp.SUB, (1.0, exact))))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, invalid))
  return _DPUExpr(RKDPUOp.MUL, (signed, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _asinh_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Two physical tables cover ASINH center, broad, middle, and large ranges."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative_source = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  magnitude = _DPUExpr(RKDPUOp.MAX, (source, negative_source))
  negative, small_outside, huge = positive(0.0, source), positive(magnitude, 2.0), positive(magnitude, 16.0)
  nonnegative, small = _DPUExpr(RKDPUOp.SUB, (1.0, negative)), _DPUExpr(RKDPUOp.SUB, (1.0, small_outside))
  middle = _DPUExpr(RKDPUOp.SUB, (small_outside, huge))
  near_inside, local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(magnitude, .04))), \
    _DPUExpr(RKDPUOp.SUB, (1.0, positive(magnitude, .25)))
  local_mask, broad_region = _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)), _DPUExpr(RKDPUOp.SUB, (small, local_inside))
  core_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (magnitude, -8.0)), local_inside)), _DPUExpr(RKDPUOp.MUL, (magnitude, broad_region))))
  range_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (magnitude, 2.0)), -1.0)), middle)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (magnitude, 1/19)), huge))))
  core, ranged = _DPUExpr(rklut.RKLUT.ASINH_CORE, (core_input,)), _DPUExpr(rklut.RKLUT.ASINH_RANGE, (range_input,))
  result = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (core, .25)), local_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (core, 2.0)), broad_region)))), _DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (ranged, 4.0)), middle)),
     _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (ranged, 8.0)), huge)))),
     _DPUExpr(RKDPUOp.MUL, (magnitude, near_inside))))))
  return _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.MUL, (result, nonnegative)), _DPUExpr(RKDPUOp.MUL, (result, negative))))

def _acosh_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  """Endpoint-aware FP16 ACOSH with separate core and range tables."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  invalid, magnitude = positive(1.0, source), _DPUExpr(RKDPUOp.MAX, (source, 1.0))
  small_outside, huge = positive(magnitude, 2.0), positive(magnitude, 16.0)
  small, middle = _DPUExpr(RKDPUOp.SUB, (1.0, small_outside)), _DPUExpr(RKDPUOp.SUB, (small_outside, huge))
  coordinate = _DPUExpr(RKDPUOp.SUB, (source, 1.0))
  local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(coordinate, .04)))
  broad_region, nonexact = _DPUExpr(RKDPUOp.SUB, (small, local_inside)), positive(coordinate, .0005)
  core_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (coordinate, -48.0)), local_inside)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (coordinate, 2.0)), broad_region))))
  range_input = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB, (magnitude, 2.0)), -1.0)), middle)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (magnitude, 1/19)), huge))))
  core, ranged = _DPUExpr(rklut.RKLUT.ACOSH_CORE, (core_input,)), _DPUExpr(rklut.RKLUT.ACOSH_RANGE, (range_input,))
  result = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (core, .5)), local_inside)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (core, 2.0)), broad_region)))), _DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (ranged, 4.0)), middle)),
     _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (ranged, 8.0)), huge))))))
  exact_zeroed = _DPUExpr(RKDPUOp.MUL, (result, nonexact))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, invalid))
  return _DPUExpr(RKDPUOp.MUL, (exact_zeroed, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _elu_expr(source:_DPUExpr|RKArg|float, negative_scale:float, positive_scale:float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  if negative_scale < 0.2: broad_op, local_op, broad_gain, local_gain = rklut.RKLUT.ELU01, rklut.RKLUT.ELU01_LOCAL, 8.0, 16.0
  elif negative_scale > 1.5: broad_op, local_op, broad_gain, local_gain = rklut.RKLUT.SELU, rklut.RKLUT.SELU_LOCAL, 0.5, 1.0
  else: broad_op, local_op, broad_gain, local_gain = rklut.RKLUT.ELU1, rklut.RKLUT.ELU1_LOCAL, 1.0, 2.0
  broad_input = _DPUExpr(RKDPUOp.MAX, (source, -8.0))
  local_input = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, -0.5)), 4.0))
  broad, local = _DPUExpr(broad_op, (broad_input,)), _DPUExpr(local_op, (local_input,))
  below, local_below, poly_below, negative = (positive(x, source) for x in (-8.0, -0.5, -0.03, 0.0))
  broad_mask, local_mask = _DPUExpr(RKDPUOp.SUB, (local_below, below)), _DPUExpr(RKDPUOp.SUB, (poly_below, local_below))
  poly_mask, positive_mask = _DPUExpr(RKDPUOp.SUB, (negative, poly_below)), _DPUExpr(RKDPUOp.SUB, (1.0, negative))
  poly_input = _DPUExpr(RKDPUOp.MAX, (source, -0.03))
  polynomial = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (poly_input, negative_scale)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (poly_input, poly_input)), negative_scale/2))))
  negative_sum = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (broad, 1/broad_gain)), broad_mask)), _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (local, 1/local_gain)), local_mask)))), _DPUExpr(RKDPUOp.MUL, (polynomial, poly_mask))))
  tails = _DPUExpr(RKDPUOp.ADD, (negative_sum, _DPUExpr(RKDPUOp.MUL, (-negative_scale, below))))
  positive_result = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, 0.0)), positive_scale)), positive_mask))
  return _DPUExpr(RKDPUOp.ADD, (tails, positive_result))

def _mish_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  broad, local = _DPUExpr(rklut.RKLUT.MISH, (source,)), _DPUExpr(rklut.RKLUT.MISH_LOCAL, (_DPUExpr(RKDPUOp.MUL, (source, 2.0)),))
  range_outside = _DPUExpr(RKDPUOp.MAX, (positive(-8.0, source), positive(source, 8.0)))
  range_inside = _DPUExpr(RKDPUOp.SUB, (1.0, range_outside))
  local_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -1.0), positive(1.0, source)))
  poly_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -0.08), positive(0.08, source)))
  broad_mask, local_mask = _DPUExpr(RKDPUOp.SUB, (range_inside, local_inside)), _DPUExpr(RKDPUOp.SUB, (local_inside, poly_inside))
  poly_source = _DPUExpr(RKDPUOp.MUL, (source, poly_inside))
  polynomial = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (poly_source, 0.6)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (poly_source, poly_source)), 0.32))))
  broad_scale = _DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.MUL, (positive(source, 0.0), 7.0))))
  inner = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (broad, broad_scale)), broad_mask)), _DPUExpr(RKDPUOp.MUL, (local, local_mask)))), polynomial))
  fallback = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, 0.0)), range_outside))
  return _DPUExpr(RKDPUOp.ADD, (inner, fallback))

def _logsigmoid_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  correction, tail = _DPUExpr(rklut.RKLUT.LOGSIGMOID, (source,)), _DPUExpr(rklut.RKLUT.LOGSIGMOID_TAIL, (source,))
  positive_source = _DPUExpr(RKDPUOp.MAX, (source, 0.0))
  minimum = _DPUExpr(RKDPUOp.SUB, (source, positive_source))
  tail_mask = positive(source, 3.5)
  broad_mask = _DPUExpr(RKDPUOp.SUB, (1.0, tail_mask))
  selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (correction, broad_mask)),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (tail, 1/32)), tail_mask))))
  raw = _DPUExpr(RKDPUOp.ADD, (minimum, selected))
  return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (raw, -1.0)), 0.0)), -1.0))

def _softplus_expr(source:_DPUExpr|RKArg|float, beta:float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  def clamp(value:_DPUExpr|RKArg|float, limit:float) -> _DPUExpr:
    lower = _DPUExpr(RKDPUOp.MAX, (value, -limit))
    return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -limit)), -1.0))
  positive_source = _DPUExpr(RKDPUOp.MAX, (source, 0.0))
  if beta < 1:
    raw = _DPUExpr(RKDPUOp.SUB, (positive_source, _DPUExpr(rklut.RKLUT.SOFTPLUS13, (source,))))
    finite = _DPUExpr(RKDPUOp.SUB, (1.0, positive(-100.0, source)))
    return _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (raw, finite)), 0.0))
  broad_op, tail_op = (rklut.RKLUT.SOFTPLUS3, rklut.RKLUT.SOFTPLUS3_TAIL) if beta == 3 else (rklut.RKLUT.SOFTPLUS1, rklut.RKLUT.SOFTPLUS1_TAIL)
  broad, tail = _DPUExpr(broad_op, (clamp(source, 8/beta),)), _DPUExpr(tail_op, (clamp(source, 16/beta),))
  tail_mask = positive(-3.05/beta, source)
  correction = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, tail_mask)))),
    _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (tail, 1/21)), tail_mask))))
  correction_outside = _DPUExpr(RKDPUOp.MAX, (positive(-16/beta, source), positive(source, 16/beta)))
  correction = _DPUExpr(RKDPUOp.MUL, (correction, _DPUExpr(RKDPUOp.SUB, (1.0, correction_outside))))
  return _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.SUB, (positive_source, correction)), 0.0))

def _sinh_cosh_expr(source:_DPUExpr|RKArg|float, is_cosh:bool) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  def clamp(value:_DPUExpr|RKArg|float, limit:float) -> _DPUExpr:
    lower = _DPUExpr(RKDPUOp.MAX, (value, -limit))
    return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -limit)), -1.0))
  selected:_DPUExpr = _DPUExpr(rklut.RKLUT.COSH if is_cosh else rklut.RKLUT.SINH, (clamp(source, 2.0),))
  negative = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  magnitude = _DPUExpr(RKDPUOp.MAX, (source, negative))
  if not is_cosh:
    local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(rklut.RKLUT.SINH_LOCAL, (clamp(source, .25),)), .25))
    near_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(magnitude, .04)))
    local_inside = _DPUExpr(RKDPUOp.SUB, (1.0, positive(magnitude, .125)))
    selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (selected,
      _DPUExpr(RKDPUOp.SUB, (1.0, local_inside)))), _DPUExpr(RKDPUOp.MUL, (local,
      _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)))))), _DPUExpr(RKDPUOp.MUL, (source, near_inside))))
  denominator = _DPUExpr(RKDPUOp.SUB, (1.0, positive(magnitude, 10.0)))
  return _DPUExpr(RKDPUOp.DIV, (selected, denominator))

def _sqrt_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  refined:_DPUExpr = _DPUExpr(rklut.RKLUT.SQRT, (source,))
  for _ in range(3): refined = _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.ADD, (refined, _DPUExpr(RKDPUOp.DIV, (source, refined)))), .5))
  high, negative = positive(source, 65472.0), positive(0.0, source)
  nonzero = _DPUExpr(RKDPUOp.MAX, (positive(source, 0.0), negative))
  not_number = _DPUExpr(RKDPUOp.MUL, (positive(source, 0.0), negative))
  positive_result = _DPUExpr(RKDPUOp.DIV, (refined, _DPUExpr(RKDPUOp.SUB, (1.0, high))))
  zero_result = _DPUExpr(RKDPUOp.MUL, (positive_result, nonzero))
  invalid = _DPUExpr(RKDPUOp.MAX, (negative, not_number))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, invalid))
  return _DPUExpr(RKDPUOp.MUL, (zero_result, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _rsqrt_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  greater_zero, below_1, below_2 = positive(source, 0.0), positive(0.0625, source), positive(0.00390625, source)
  low_1, low_2 = _DPUExpr(RKDPUOp.MUL, (greater_zero, below_1)), _DPUExpr(RKDPUOp.MUL, (greater_zero, below_2))
  factor_1, factor_2 = (_DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.MUL, (mask, 15.0)))) for mask in (low_1, low_2))
  scaled = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (source, factor_1)), factor_2))
  seed = _DPUExpr(rklut.RKLUT.RSQRT, (scaled,))
  safe = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (scaled, -1.0)), -4.0)), -1.0))
  correction = _DPUExpr(RKDPUOp.SUB, (1.5, _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (safe, _DPUExpr(RKDPUOp.MUL, (seed, seed)))), .5))))
  refined = _DPUExpr(RKDPUOp.MUL, (seed, correction))
  out_factor_1, out_factor_2 = (_DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.MUL, (mask, 3.0)))) for mask in (low_1, low_2))
  scaled_out = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (refined, out_factor_1)), out_factor_2))
  negative, high = positive(0.0, source), positive(source, 65472.0)
  nonzero = _DPUExpr(RKDPUOp.MAX, (greater_zero, negative))
  # Rejected signed-zero repair: selecting 1/source on the zero mask still produced +inf for -0 because DPU division normalizes its zero sign.
  finite = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.DIV, (scaled_out, nonzero)), _DPUExpr(RKDPUOp.SUB, (1.0, high))))
  not_number = _DPUExpr(RKDPUOp.MUL, (greater_zero, negative))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MAX, (negative, not_number))))
  return _DPUExpr(RKDPUOp.MUL, (finite, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _exp_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  def clamp(value:_DPUExpr|RKArg|float, limit:float) -> _DPUExpr:
    lower = _DPUExpr(RKDPUOp.MAX, (value, -limit))
    return _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lower, -1.0)), -limit)), -1.0))
  broad = _DPUExpr(rklut.RKLUT.EXP, (clamp(source, 2.0),))
  broad = _DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.MUL, (positive(source, 0.0), 7.0))))))
  local = _DPUExpr(rklut.RKLUT.EXP_LOCAL, (clamp(source, .25),))
  local_inside = _DPUExpr(RKDPUOp.MUL, (positive(source, -.25), positive(.25, source)))
  selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, local_inside)))),
                                      _DPUExpr(RKDPUOp.MUL, (local, local_inside))))
  high, low = positive(source, 65472.0), positive(-65472.0, source)
  finite = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.DIV, (selected, _DPUExpr(RKDPUOp.SUB, (1.0, high)))),
                                  _DPUExpr(RKDPUOp.SUB, (1.0, low))))
  nan_denom = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MUL, (high, low))))
  return _DPUExpr(RKDPUOp.DIV, (_DPUExpr(RKDPUOp.MUL, (finite, nan_denom)), nan_denom))

def _log2_expr(source:_DPUExpr|RKArg|float, scale:float=1.0) -> _DPUExpr:
  """Normalize by powers of four, refine near one, and repair IEEE special values on the NPU."""
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  # Two power-of-16 bands cover the official FP16 domain with six fewer stages than four power-of-four bands.
  low_masks = tuple(positive(threshold, source) for threshold in (.25, .015625))
  factor = _DPUExpr(RKDPUOp.ADD, (1.0, _DPUExpr(RKDPUOp.ADD, tuple(
    _DPUExpr(RKDPUOp.MUL, (mask, weight)) for mask,weight in zip(low_masks, (15.0, 240.0))))))
  normalized = _DPUExpr(RKDPUOp.MUL, (source, factor))
  count = _DPUExpr(RKDPUOp.ADD, low_masks)
  offset = _DPUExpr(RKDPUOp.MUL, (count, -4.0*scale))
  bounded_low = _DPUExpr(RKDPUOp.MAX, (normalized, .25))
  bounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX,
    (_DPUExpr(RKDPUOp.MUL, (bounded_low, -1.0)), -4.0)), -1.0))
  centered = _DPUExpr(RKDPUOp.SUB, (bounded, 1.0))
  broad_op, local_op = ((rklut.RKLUT.LOG, rklut.RKLUT.LOG_LOCAL) if math.isclose(scale, math.log(2)) else
                        (rklut.RKLUT.LOG10, rklut.RKLUT.LOG10_LOCAL) if math.isclose(scale, math.log10(2)) else
                        (rklut.RKLUT.LOG2, rklut.RKLUT.LOG2_LOCAL))
  broad = _DPUExpr(broad_op, (bounded,))
  local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(local_op,
    (_DPUExpr(RKDPUOp.MUL, (centered, 12.5)),)), .25))
  local_inside = _DPUExpr(RKDPUOp.MUL, (positive(bounded, .85), positive(1.15, bounded)))
  near_inside = _DPUExpr(RKDPUOp.MUL, (positive(centered, -.02), positive(.02, centered)))
  polynomial = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.SUB,
    (centered, _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (centered, centered)), .5)))), scale*math.log2(math.e)))
  mantissa = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD,
    (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, local_inside)))),
     _DPUExpr(RKDPUOp.MUL, (local, _DPUExpr(RKDPUOp.SUB, (local_inside, near_inside)))))),
    _DPUExpr(RKDPUOp.MUL, (polynomial, near_inside))))
  corrected = _DPUExpr(RKDPUOp.ADD, (mantissa, offset))
  negative, greater_zero, high = positive(0.0, source), positive(source, 0.0), positive(source, 65472.0)
  nonzero = _DPUExpr(RKDPUOp.MAX, (greater_zero, negative))
  zero_result = _DPUExpr(RKDPUOp.DIV, (corrected, nonzero))
  finite = _DPUExpr(RKDPUOp.DIV, (zero_result, _DPUExpr(RKDPUOp.SUB, (1.0, high))))
  not_number = _DPUExpr(RKDPUOp.MUL, (greater_zero, negative))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MAX, (negative, not_number))))
  return _DPUExpr(RKDPUOp.MUL, (finite, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _round_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  negative = _DPUExpr(RKDPUOp.MUL, (source, -1.0))
  magnitude = _DPUExpr(RKDPUOp.MAX, (source, negative))
  positive_mask, negative_mask = positive(source, 0.0), positive(0.0, source)
  sign = _DPUExpr(RKDPUOp.SUB, (positive_mask, negative_mask))
  rounded = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(rklut.RKLUT.ROUNDOFF, (magnitude,)), sign))
  high = positive(magnitude, 65472.0)
  high_result = _DPUExpr(RKDPUOp.DIV, (sign, _DPUExpr(RKDPUOp.SUB, (1.0, high))))
  selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (rounded, _DPUExpr(RKDPUOp.SUB, (1.0, high)))),
                                        _DPUExpr(RKDPUOp.MUL, (high_result, high))))
  valid = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MUL, (positive_mask, negative_mask))))
  return _DPUExpr(RKDPUOp.MUL, (selected, _DPUExpr(RKDPUOp.DIV, (valid, valid))))

def _trunc_expr(source:_DPUExpr|RKArg|float) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  rounded = _round_expr(source)
  decrement = _DPUExpr(RKDPUOp.MUL, (positive(rounded, source), positive(source, 0.0)))
  increment = _DPUExpr(RKDPUOp.MUL, (positive(source, rounded), positive(0.0, source)))
  return _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.SUB, (rounded, decrement)), increment))

def _celu_expr(source:_DPUExpr|RKArg|float, alpha:int) -> _DPUExpr:
  def positive(lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float) -> _DPUExpr:
    return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))
  broad_op, local_op = {2:(rklut.RKLUT.CELU2,rklut.RKLUT.CELU2_LOCAL), 3:(rklut.RKLUT.CELU3,rklut.RKLUT.CELU3_LOCAL),
                        4:(rklut.RKLUT.CELU4,rklut.RKLUT.CELU4_LOCAL)}[alpha]
  broad, local = _DPUExpr(broad_op, (_DPUExpr(RKDPUOp.MAX, (source, -4.0)),)), \
                 _DPUExpr(local_op, (_DPUExpr(RKDPUOp.MAX, (source, -.5)),))
  below, local_below, poly_below, negative = (positive(x, source) for x in (-4.0, -.5, -.03, 0.0))
  broad_mask, local_mask = _DPUExpr(RKDPUOp.SUB, (local_below, below)), _DPUExpr(RKDPUOp.SUB, (poly_below, local_below))
  poly_mask, positive_mask = _DPUExpr(RKDPUOp.SUB, (negative, poly_below)), _DPUExpr(RKDPUOp.SUB, (1.0, negative))
  poly_input = _DPUExpr(RKDPUOp.MAX, (source, -.03))
  polynomial = _DPUExpr(RKDPUOp.ADD, (poly_input, _DPUExpr(RKDPUOp.MUL,
    (_DPUExpr(RKDPUOp.MUL, (poly_input, poly_input)), 1/(2*alpha)))))
  negative_sum = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, broad_mask)),
    _DPUExpr(RKDPUOp.MUL, (local, local_mask)))), _DPUExpr(RKDPUOp.MUL, (polynomial, poly_mask))))
  return _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (negative_sum, _DPUExpr(RKDPUOp.MUL, (-float(alpha), below)))),
                                _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (source, 0.0)), positive_mask))))

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
  if any(x < 0 or x >= 32 for x in image.fp32_inputs): raise ValueError("RKImage FP32 input slots must be in 0..31")
  if any(x < 0 or x >= 16 for x in image.fp32_outputs): raise ValueError("RKImage FP32 output slots must be in 0..15")
  _slot_mask(image.bool_outputs)
  _slot_mask(image.bool_inputs)
  _slot_mask(image.int_inputs)
  _slot_mask(image.tiled_inputs)
  _slot_mask(image.int_outputs)
  _slot_mask(image.transposed_int_inputs)
  _slot_mask(image.tiled_int_inputs)
  _slot_mask(image.raw_int_inputs)
  _slot_mask(image.numeric_int_inputs)

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
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.stages), len(relocs), len(image.scratch),
                               sum(1 << x for x in image.fp32_outputs),
                               len(commands), len(image.constants), sum(1 << x for x in image.fp32_inputs), _slot_mask(image.bool_outputs),
                               _slot_mask(image.bool_inputs), _slot_mask(image.int_inputs), _slot_mask(image.tiled_inputs),
                               _slot_mask(image.int_outputs), _slot_mask(image.transposed_int_inputs), _slot_mask(image.tiled_int_inputs),
                               _slot_mask(image.raw_int_inputs), _slot_mask(image.numeric_int_inputs)))
  for engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes in stage_rows:
    out += _STAGE.pack(engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes)
  for reloc in relocs:
    out += _RELOC.pack(reloc.stage, reloc.word, int(reloc.kind), reloc.shift, reloc.index, reloc.addend, reloc.mask, reloc.field_shift)
  for scratch in image.scratch: out += _SCRATCH.pack(scratch.size, scratch.alignment)
  if commands: out += struct.pack(f"<{len(commands)}Q", *commands)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  if len(blob) < _HEADER.size: raise ValueError("truncated RKImage header")
  magic, version, target, stage_count, reloc_count, scratch_count, reserved, command_count, constant_size, reserved2, \
    bool_output_mask, bool_input_mask, int_input_mask, tiled_input_mask, int_output_mask, transposed_int_input_mask, \
    tiled_int_input_mask, raw_int_input_mask, numeric_int_input_mask = _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC: raise ValueError("invalid RKImage header")
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
  image = RKImage(RKTarget(target), tuple(stages), scratch, blob[off:], version,
                  tuple(x for x in range(32) if reserved2 & (1 << x)), tuple(x for x in range(16) if reserved & (1 << x)),
                  slots(bool_output_mask), slots(bool_input_mask), slots(int_input_mask), slots(tiled_input_mask), slots(int_output_mask),
                  slots(transposed_int_input_mask), slots(tiled_int_input_mask), slots(raw_int_input_mask), slots(numeric_int_input_mask))
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

def _canonical_asin(u:UOp) -> tuple[UOp,bool]|None:
  """Recognize tinygrad's asin polynomial, optionally wrapped as pi/2-asin(x)."""
  u = _unwrap_same_cast(u)
  indexes = list(dict.fromkeys(x for x in u.toposort() if x.op is Ops.INDEX))
  if len(indexes) != 1 or indexes[0].dtype is not dtypes.half: return None
  core, is_acos = u, False
  if u.op is Ops.ADD:
    pi_half = next((x for x in u.src if x.op is Ops.CONST and math.isclose(float(x.arg), math.pi/2)), None)
    negated = next((_unwrap_same_cast(x) for x in u.src if x is not pi_half), None)
    if pi_half is None or negated is None or negated.op is not Ops.MUL: return None
    minus_one = next((x for x in negated.src if x.op is Ops.CONST and float(x.arg) == -1.0), None)
    if minus_one is None: return None
    core = next((_unwrap_same_cast(x) for x in negated.src if x is not minus_one), negated)
    is_acos = True
  nodes = core.toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  coefficients = (-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                  -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)
  allowed = {Ops.ADD, Ops.MUL, Ops.SQRT, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return (indexes[0], is_acos) if core.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    all(x.op in allowed for x in nodes) and sum(x.op is Ops.SQRT for x in nodes) == 1 and sum(x.op is Ops.WHERE for x in nodes) == 2 and \
    all(any(x.op is Ops.CONST and math.isclose(float(x.arg), coefficient) for x in nodes) for coefficient in coefficients) else None

def _canonical_atan(u:UOp) -> UOp|None:
  """Recognize atan(x) lowered as asin(x/sqrt(1+x*x))."""
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  coefficients = (-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                  -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.SQRT, Ops.RECIPROCAL, Ops.WHERE, Ops.CMPLT, Ops.CMPNE,
             Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return indexes[0] if u.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    all(x.op in allowed for x in nodes) and sum(x.op is Ops.SQRT for x in nodes) == 2 and \
    sum(x.op in (Ops.RECIPROCAL, Ops.FDIV) for x in nodes) == 1 and sum(x.op is Ops.WHERE for x in nodes) == 2 and \
    all(any(x.op is Ops.CONST and math.isclose(float(x.arg), coefficient) for x in nodes) for coefficient in coefficients) else None

def _canonical_sin_cos(u:UOp) -> tuple[UOp,bool]|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.SIN: return None
  phase = u.src[0]
  while phase.op is Ops.CAST and phase.dtype in (dtypes.half,dtypes.float): phase = phase.src[0]
  if phase.op is Ops.INDEX and phase.dtype is dtypes.half: return phase, False
  indexes = list(dict.fromkeys(x for x in phase.toposort() if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in phase.toposort() if x.op is Ops.CONST]
  return (indexes[0], True) if phase.op is Ops.ADD and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    any(math.isclose(x, math.pi/2) for x in constants) and any(math.isclose(x, -1.0) for x in constants) else None

def _canonical_atanh(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.RECIPROCAL, Ops.LOG2, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST]
  return indexes[0] if u.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    all(x.op in allowed for x in nodes) and sum(x.op is Ops.LOG2 for x in nodes) == 1 and \
    sum(x.op in (Ops.FDIV, Ops.RECIPROCAL) for x in nodes) == 1 and \
    sum(x.op is Ops.ADD for x in nodes) == 2 and any(x == -1.0 for x in constants) else None

def _canonical_asinh(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST]
  allowed = {Ops.ADD, Ops.MUL, Ops.LOG2, Ops.SQRT, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return indexes[0] if u.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    all(x.op in allowed for x in nodes) and sum(x.op is Ops.LOG2 for x in nodes) == 1 and sum(x.op is Ops.SQRT for x in nodes) == 1 and \
    any(x == 1.0 for x in constants) and not any(x == -1.0 for x in constants) else None

def _canonical_acosh(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST]
  allowed = {Ops.ADD, Ops.MUL, Ops.LOG2, Ops.SQRT, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return indexes[0] if u.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    all(x.op in allowed for x in nodes) and sum(x.op is Ops.LOG2 for x in nodes) == 1 and sum(x.op is Ops.SQRT for x in nodes) == 1 and \
    any(x == -1.0 for x in constants) else None

def _canonical_elu(u:UOp) -> tuple[UOp,float,float]|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  if len(indexes) != 1 or indexes[0].dtype is not dtypes.half or sum(x.op is Ops.EXP2 for x in nodes) != 1: return None
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  wheres = sum(x.op is Ops.WHERE for x in nodes)
  if u.op is Ops.ADD and wheres == 2: return indexes[0], (0.1 if any(math.isclose(abs(x), .1) for x in constants) else 1.0), 1.0
  if u.op is Ops.MUL and wheres == 1 and any(math.isclose(x, 1.0507) for x in constants): return indexes[0], 1.0507*1.67326, 1.0507
  return None

def _canonical_celu(u:UOp) -> tuple[UOp,float]|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  alpha = next((x for x in (4.0, 3.0, 2.0, 1.0) if any(math.isclose(c, -x) for c in constants)), None)
  return (indexes[0], alpha) if u.op is Ops.ADD and len(indexes) == 1 and indexes[0].dtype is dtypes.half and alpha is not None and \
    sum(x.op is Ops.EXP2 for x in nodes) == 1 and sum(x.op is Ops.MAX for x in nodes) >= 2 and \
    any(math.isclose(x, math.log2(math.e)) for x in constants) else None

def _canonical_round(u:UOp) -> UOp|None:
  """Recognize tinygrad's exact round-to-nearest-even expansion."""
  u = _unwrap_same_cast(u)
  indexes = [x for x in u.toposort() if x.op is Ops.INDEX]
  if len(indexes) != 1 or (source:=indexes[0]).dtype is not dtypes.half: return None
  def c(value:float) -> UOp: return UOp.const(value, dtypes.half)
  truncated = UOp(Ops.TRUNC, dtypes.half, (source,))
  half_truncated = UOp(Ops.MUL, dtypes.half, (truncated, c(.5)))
  positive = UOp(Ops.CMPLT, dtypes.bool, (c(0), source))
  even = UOp(Ops.CMPNE, dtypes.bool, (UOp(Ops.CMPNE, dtypes.bool,
    (UOp(Ops.TRUNC, dtypes.half, (half_truncated,)), half_truncated)), UOp.const(True, dtypes.bool)))
  condition = UOp(Ops.CMPNE, dtypes.bool, (positive, even))
  plus_half = UOp(Ops.ADD, dtypes.half, (source, c(.5)))
  plus_trunc = UOp(Ops.TRUNC, dtypes.half, (plus_half,))
  floor_plus = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (plus_half, plus_trunc)),
    UOp(Ops.ADD, dtypes.half, (plus_trunc, c(-1))), plus_trunc))
  minus_half = UOp(Ops.ADD, dtypes.half, (source, c(-.5)))
  minus_trunc = UOp(Ops.TRUNC, dtypes.half, (minus_half,))
  ceil_minus = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (minus_trunc, minus_half)),
    UOp(Ops.ADD, dtypes.half, (minus_trunc, c(1))), minus_trunc))
  expected = UOp(Ops.WHERE, dtypes.half, (condition, floor_plus, ceil_minus))
  if u is expected: return source
  # Current master keeps two ±0.5/±1 constants weak while 2607 made them half, so exact UOp identity does not match.
  counts = {op:sum(x.op is op for x in u.toposort()) for op in (Ops.TRUNC,Ops.ADD,Ops.MUL,Ops.CMPLT,Ops.CMPNE,Ops.WHERE)}
  constants = [float(x.arg) for x in u.toposort() if x.op is Ops.CONST]
  required = {Ops.TRUNC:4,Ops.ADD:4,Ops.MUL:1,Ops.CMPLT:3,Ops.CMPNE:3,Ops.WHERE:3}
  return source if counts == required and all(any(math.isclose(x, value) for x in constants) for value in (-1,-.5,0,.5,1)) else None

def _canonical_mish(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  return indexes[0] if u.op is Ops.MUL and len(indexes) == 1 and indexes[0].dtype is dtypes.half and \
    sum(x.op is Ops.EXP2 for x in nodes) == 3 and sum(x.op is Ops.LOG2 for x in nodes) == 1 and \
    sum(x.op is Ops.MAX for x in nodes) == 1 and sum(x.op is Ops.RECIPROCAL for x in nodes) == 1 else None

def _canonical_logsigmoid(u:UOp) -> UOp|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  return indexes[0] if len(indexes) == 1 and indexes[0].dtype is dtypes.half and sum(x.op is Ops.EXP2 for x in nodes) == 2 and \
    sum(x.op is Ops.LOG2 for x in nodes) == 1 and sum(x.op is Ops.MAX for x in nodes) == 1 and \
    any(math.isclose(x, -math.log(2)) for x in constants) else None

def _canonical_finite_predicate(u:UOp) -> tuple[UOp,str]|None:
  def logical_not(x:UOp) -> UOp|None:
    x = _unwrap_same_cast(x)
    if x.op is not Ops.CMPNE: return None
    for value, marker in (x.src, x.src[::-1]):
      marker = _unwrap_same_cast(marker)
      if marker.op is Ops.CONST and marker.dtype is dtypes.bool and marker.arg is True: return value
    return None
  def atom(x:UOp) -> tuple[UOp,str]|None:
    x = _unwrap_same_cast(x)
    if x.op is Ops.CMPNE and x.src[0].key == x.src[1].key and x.src[0].op is Ops.INDEX and x.src[0].dtype is dtypes.half:
      return x.src[0], "nan"
    if (unequal:=logical_not(x)) is None or (unequal:=_unwrap_same_cast(unequal)).op is not Ops.CMPNE: return None
    for source, constant in (unequal.src, unequal.src[::-1]):
      source, constant = _unwrap_same_cast(source), _unwrap_same_cast(constant)
      if source.op is Ops.INDEX and source.dtype is dtypes.half and constant.op is Ops.CONST and constant.arg in (math.inf, -math.inf):
        return source, "positive_inf" if constant.arg == math.inf else "negative_inf"
    return None
  def flatten_or(x:UOp) -> list[UOp]:
    x = _unwrap_same_cast(x)
    return [child for source in x.src for child in flatten_or(source)] if x.op is Ops.OR else [x]

  if (single:=atom(u)) is not None: return single
  inverted, kind = logical_not(u), "finite"
  expression = _unwrap_same_cast(inverted) if inverted is not None else _unwrap_same_cast(u)
  if expression.op is not Ops.OR: return None
  matches = [atom(x) for x in flatten_or(expression)]
  if any(x is None for x in matches): return None
  predicates = cast(list[tuple[UOp,str]], matches)
  if len({x.key for x,_ in predicates}) != 1: return None
  tags = sorted(tag for _,tag in predicates)
  if inverted is None and tags == ["negative_inf", "positive_inf"]: kind = "inf"
  elif inverted is None or tags != ["nan", "negative_inf", "positive_inf"]: return None
  return predicates[0][0], kind

def _canonical_softplus(u:UOp) -> tuple[UOp,float]|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  constants = [float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int, float))]
  if len(indexes) != 1 or indexes[0].dtype is not dtypes.half or sum(x.op is Ops.EXP2 for x in nodes) != 2 or \
     sum(x.op is Ops.LOG2 for x in nodes) != 1 or sum(x.op is Ops.MAX for x in nodes) != 1 or \
     not any(math.isclose(x, math.log(2)) for x in constants): return None
  if u.op is Ops.ADD: return indexes[0], 1.0
  root_scale = next((float(x.arg) for x in u.src if x.op is Ops.CONST), None)
  return (indexes[0], 3.0 if root_scale is not None and math.isclose(root_scale, 1/3) else 1/3) if root_scale is not None else None

def _canonical_sinh_cosh(u:UOp) -> tuple[UOp,bool]|None:
  u, nodes = _unwrap_same_cast(u), _unwrap_same_cast(u).toposort()
  indexes = list(dict.fromkeys(x for x in nodes if x.op is Ops.INDEX))
  if u.op is not Ops.MUL or len(indexes) != 1 or indexes[0].dtype is not dtypes.half or \
     sum(x.op is Ops.EXP2 for x in nodes) != 2 or sum(x.op is Ops.ADD for x in nodes) != 1: return None
  return indexes[0], sum(x.op is Ops.MUL for x in nodes) == 4

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

def _int_operand_bytes(u:UOp, output_index:UOp) -> tuple[RKArg|float, ...]|None:
  u = _unwrap_same_cast(u)
  output_count = int(output_index.vmax)+1 if int(output_index.vmin) == 0 else 0
  input_count = int(u.src[0].src[0].arg) if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM else 0
  suffix_tiled = output_count > 0 and input_count > 0 and output_count % input_count == 0 and \
    ((u.src[1].op is Ops.RANGE and input_count == int(u.src[1].src[0].arg)) or
     (u.src[1].op is Ops.CONST and input_count == 1 and int(u.src[1].arg) == 0))
  if u.op is Ops.INDEX and u.dtype is dtypes.int and u.src[0].op is Ops.PARAM and (u.src[1].key == output_index.key or suffix_tiled):
    slot, plane_size = u.src[0].arg.slot, ((output_count+7)//8)*16
    return tuple(RKArg(RKBufferKind.ARG, slot, plane*plane_size) for plane in range(4))
  if u.op is Ops.CONST and dtypes.is_int(u.dtype):
    value = int(u.arg) & 0xffffffff
    return tuple(float(((value >> shift) & 0xff) ^ (0x80 if shift == 24 else 0)) for shift in (24, 16, 8, 0))
  return None

def _int_cmp_expr(op:Ops, lhs:tuple[RKArg|float, ...], rhs:tuple[RKArg|float, ...]) -> _DPUExpr:
  prefix: _DPUExpr|float = 1.0
  less: _DPUExpr|float = 0.0
  unequal: _DPUExpr|float = 0.0
  for left, right in zip(lhs, rhs):
    lt = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (right, left)),))
    ne = _DPUExpr(RKDPUOp.MAX, (lt, _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (left, right)),))))
    less = _DPUExpr(RKDPUOp.MAX, (less, _DPUExpr(RKDPUOp.MUL, (prefix, lt))))
    unequal = _DPUExpr(RKDPUOp.MAX, (unequal, ne))
    prefix = _DPUExpr(RKDPUOp.MUL, (prefix, _DPUExpr(RKDPUOp.SUB, (1.0, ne))))
  return cast(_DPUExpr, less if op is Ops.CMPLT else unequal)

def _ieee_mask_expr(op:Ops, lhs:_DPUExpr|RKArg|float, rhs:_DPUExpr|RKArg|float, invert:bool=False) -> _DPUExpr:
  def classifications(x:_DPUExpr|RKArg|float) -> tuple[_DPUExpr,_DPUExpr,_DPUExpr,_DPUExpr]:
    high = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (x, 65504.0)),))
    low = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (-65504.0, x)),))
    nan = _DPUExpr(RKDPUOp.MUL, (high, low))
    return nan, _DPUExpr(RKDPUOp.SUB, (high, nan)), _DPUExpr(RKDPUOp.SUB, (low, nan)), \
      _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MAX, (high, low))))
  lhs_nan, lhs_pos, lhs_neg, lhs_finite = classifications(lhs)
  rhs_nan, rhs_pos, rhs_neg, rhs_finite = classifications(rhs)
  positive = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (rhs, lhs)),))
  if op is Ops.CMPLT:
    valid = _DPUExpr(RKDPUOp.SUB, (1.0, _DPUExpr(RKDPUOp.MAX, (lhs_nan, rhs_nan))))
    forced = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MUL, (lhs_neg, _DPUExpr(RKDPUOp.SUB, (1.0, rhs_neg)))),
      _DPUExpr(RKDPUOp.MUL, (rhs_pos, _DPUExpr(RKDPUOp.SUB, (1.0, lhs_pos))))))
    finite = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (lhs_finite, rhs_finite)), positive))
    comparison = _DPUExpr(RKDPUOp.MAX, (forced, finite))
    return _DPUExpr(RKDPUOp.MUL, (valid, _DPUExpr(RKDPUOp.SUB, (1.0, comparison)) if invert else comparison))
  unequal = _DPUExpr(RKDPUOp.MAX, (positive, _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)),))))
  finite_equal = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (lhs_finite, rhs_finite)),
    _DPUExpr(RKDPUOp.SUB, (1.0, unequal))))
  equal = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MAX, (finite_equal, _DPUExpr(RKDPUOp.MUL, (lhs_pos, rhs_pos)))),
    _DPUExpr(RKDPUOp.MUL, (lhs_neg, rhs_neg))))
  return _DPUExpr(RKDPUOp.SUB, (1.0, equal))

def _parse_mask_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float], ieee:bool=False) -> _DPUExpr|RKArg|float|None:
  """Build an FP16 0/1 predicate from comparisons and boolean composition."""
  u = _unwrap_same_cast(u)
  if u.op is Ops.INDEX and u.dtype is dtypes.bool and u.src[0].op is Ops.PARAM and u.src[1].key == output_index.key:
    return RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  if u.op is Ops.CONST and u.dtype is dtypes.bool: return float(u.arg)
  if u.op is Ops.CMPNE and any(x.dtype is dtypes.bool for x in u.src):
    if ieee:
      for expression, marker in (u.src, u.src[::-1]):
        expression, marker = _unwrap_same_cast(expression), _unwrap_same_cast(marker)
        if marker.op is Ops.CONST and marker.dtype is dtypes.bool and marker.arg is True and expression.op is Ops.CMPLT:
          integer_operands = tuple(_int_operand_bytes(x, output_index) for x in expression.src)
          if all(x is not None for x in integer_operands):
            compared_ints = cast(tuple[tuple[RKArg|float, ...], tuple[RKArg|float, ...]], integer_operands)
            return _DPUExpr(RKDPUOp.SUB, (1.0, _int_cmp_expr(Ops.CMPLT, *compared_ints)))
          if all(x.dtype is dtypes.bool for x in expression.src):
            compared_bools = tuple(_parse_mask_expr(x, output_index, memo) for x in expression.src)
            if any(x is None for x in compared_bools): return None
            bool_lhs, bool_rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], compared_bools)
            less = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (bool_rhs, bool_lhs)),))
            return _DPUExpr(RKDPUOp.SUB, (1.0, less))
          compared = tuple(_parse_dpu_expr(x, output_index, memo) for x in expression.src)
          if any(x is None for x in compared): return None
          scalar_lhs, scalar_rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], compared)
          return _ieee_mask_expr(Ops.CMPLT, scalar_lhs, scalar_rhs, invert=True)
    mask_operands = tuple(_parse_mask_expr(x, output_index, memo, ieee) for x in u.src)
    if any(x is None for x in mask_operands): return None
    lhs, rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], mask_operands)
    if isinstance(lhs, float): return rhs if lhs == 0.0 else _DPUExpr(RKDPUOp.SUB, (1.0, rhs))
    if isinstance(rhs, float): return lhs if rhs == 0.0 else _DPUExpr(RKDPUOp.SUB, (1.0, lhs))
    return _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.SUB, (lhs, rhs)), _DPUExpr(RKDPUOp.SUB, (rhs, lhs))))
  if u.op in (Ops.CMPLT, Ops.CMPNE):
    integer_operands = tuple(_int_operand_bytes(x, output_index) for x in u.src)
    if all(x is not None for x in integer_operands):
      return _int_cmp_expr(u.op, *cast(tuple[tuple[RKArg|float, ...], tuple[RKArg|float, ...]], integer_operands))
    if u.op is Ops.CMPLT and all(x.dtype is dtypes.bool for x in u.src):
      bool_operands = tuple(_parse_mask_expr(x, output_index, memo) for x in u.src)
      if any(x is None for x in bool_operands): return None
      bool_lhs, bool_rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], bool_operands)
      return _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (bool_rhs, bool_lhs)),))
    scalar_operands = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in scalar_operands): return None
    scalar_lhs, scalar_rhs = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], scalar_operands)
    if ieee: return _ieee_mask_expr(u.op, scalar_lhs, scalar_rhs)
    positive = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (scalar_rhs, scalar_lhs)),))
    return positive if u.op is Ops.CMPLT else _DPUExpr(RKDPUOp.MAX,
      (positive, _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (scalar_lhs, scalar_rhs)),))))
  if u.op in (Ops.OR, Ops.AND):
    operands = tuple(_parse_mask_expr(x, output_index, memo, ieee) for x in u.src)
    if any(x is None for x in operands): return None
    return _DPUExpr(RKDPUOp.MAX if u.op is Ops.OR else RKDPUOp.MUL,
                    cast(tuple[_DPUExpr|RKArg|float, ...], operands))
  return None

def _parse_dpu_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float]) -> _DPUExpr|RKArg|float|None:
  while u.op is Ops.CAST and u.dtype in (dtypes.half, dtypes.float) and u.src[0].dtype in (dtypes.half, dtypes.float): u = u.src[0]
  u = _unwrap_same_cast(u)
  if u in memo: return memo[u]
  output_count = int(output_index.vmax)+1 if int(output_index.vmin) == 0 else 0
  input_count = int(u.src[0].src[0].arg) if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM else 0
  suffix_tiled = u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and output_count > 0 and output_count % input_count == 0 and \
    ((u.src[1].op is Ops.RANGE and input_count == int(u.src[1].src[0].arg)) or
     (u.src[1].op is Ops.CONST and input_count == 1 and int(u.src[1].arg) == 0))
  if u.op is Ops.INDEX and u.dtype is dtypes.half and u.src[0].op is Ops.PARAM and (u.src[1].key == output_index.key or suffix_tiled):
    ret:RKArg|float|_DPUExpr = RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  elif u.op is Ops.CAST and u.dtype is dtypes.half and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM and \
       u.src[0].src[1].key == output_index.key and u.src[0].dtype in (dtypes.int, dtypes.bool):
    ret = RKArg(RKBufferKind.ARG, u.src[0].src[0].arg.slot)
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
  elif (atan:=_canonical_atan(u)) is not None:
    source = _parse_dpu_expr(atan, output_index, memo)
    if source is None: return None
    ret = _atan_expr(source)
  elif (trig:=_canonical_sin_cos(u)) is not None:
    source = _parse_dpu_expr(trig[0], output_index, memo)
    if source is None: return None
    ret = _sin_cos_expr(source, trig[1])
  elif (atanh:=_canonical_atanh(u)) is not None:
    source = _parse_dpu_expr(atanh, output_index, memo)
    if source is None: return None
    ret = _atanh_expr(source)
  elif (asinh:=_canonical_asinh(u)) is not None:
    source = _parse_dpu_expr(asinh, output_index, memo)
    if source is None: return None
    ret = _asinh_expr(source)
  elif (acosh:=_canonical_acosh(u)) is not None:
    source = _parse_dpu_expr(acosh, output_index, memo)
    if source is None: return None
    ret = _acosh_expr(source)
  elif (asin:=_canonical_asin(u)) is not None:
    source = _parse_dpu_expr(asin[0], output_index, memo)
    if source is None: return None
    # pi/2-asin missed 156/2925 official outputs: ACOS needs its own endpoint coordinate.
    ret = _acos_expr(source) if asin[1] else _asin_expr(source)
  elif (celu:=_canonical_celu(u)) is not None:
    source = _parse_dpu_expr(celu[0], output_index, memo)
    if source is None: return None
    ret = _elu_expr(source, 1.0, 1.0) if celu[1] == 1 else _celu_expr(source, int(celu[1]))
  elif (rounded:=_canonical_round(u)) is not None:
    source = _parse_dpu_expr(rounded, output_index, memo)
    if source is None: return None
    ret = _round_expr(source)
  elif (elu:=_canonical_elu(u)) is not None:
    source = _parse_dpu_expr(elu[0], output_index, memo)
    if source is None: return None
    ret = _elu_expr(source, elu[1], elu[2])
  elif (mish:=_canonical_mish(u)) is not None:
    source = _parse_dpu_expr(mish, output_index, memo)
    if source is None: return None
    ret = _mish_expr(source)
  elif (logsigmoid:=_canonical_logsigmoid(u)) is not None:
    source = _parse_dpu_expr(logsigmoid, output_index, memo)
    if source is None: return None
    ret = _logsigmoid_expr(source)
  elif (softplus:=_canonical_softplus(u)) is not None:
    source = _parse_dpu_expr(softplus[0], output_index, memo)
    if source is None: return None
    ret = _softplus_expr(source, softplus[1])
  elif (hyperbolic:=_canonical_sinh_cosh(u)) is not None:
    source = _parse_dpu_expr(hyperbolic[0], output_index, memo)
    if source is None: return None
    ret = _sinh_cosh_expr(source, hyperbolic[1])
  elif u.op is Ops.SQRT or (u.op is Ops.RECIPROCAL and (sqrt:=_unwrap_same_cast(u.src[0])).op is Ops.SQRT):
    raw_source = _unwrap_same_cast((sqrt if u.op is Ops.RECIPROCAL else u).src[0])
    source = (RKArg(RKBufferKind.ARG, raw_source.src[0].arg.slot) if raw_source.op is Ops.INDEX and raw_source.dtype is dtypes.float and
              raw_source.src[0].op is Ops.PARAM and raw_source.src[1].key == output_index.key else
              _parse_dpu_expr(raw_source, output_index, memo))
    if source is None: return None
    ret = _rsqrt_expr(source) if u.op is Ops.RECIPROCAL else _sqrt_expr(source)
  elif (silu:=_canonical_silu(u)) is not None:
    operands = tuple(_parse_dpu_expr(x, output_index, memo) for x in silu)
    if any(x is None for x in operands): return None
    ret = _DPUExpr(RKDPUOp.MUL, cast(tuple[_DPUExpr|RKArg|float, ...], operands))
  elif (hs_input:=_canonical_hardswish(u)) is not None:
    source = _parse_dpu_expr(hs_input, output_index, memo)
    if source is None: return None
    broad = _DPUExpr(rklut.RKLUT.HARDSWISH, (source,))
    positive = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.ADD, (source, 3.0)), 0.0))
    relu_negative = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.ADD, (source, -3.0)), 0.0)), -1.0))
    relu6 = _DPUExpr(RKDPUOp.ADD, (positive, relu_negative))
    fallback = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MUL, (source, relu6)), 1/6))
    wide_outside = _DPUExpr(RKDPUOp.MAX, (_DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (-2.0, source)),)),
      _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, 2.0)),))))
    wide = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (broad, _DPUExpr(RKDPUOp.SUB, (1.0, wide_outside)))),
      _DPUExpr(RKDPUOp.MUL, (fallback, wide_outside))))
    local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(rklut.RKLUT.HARDSWISH_LOCAL,
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
    broad, local_inside, near_inside = _DPUExpr(rklut.RKLUT.TANH, (source,)), interval(-0.25, 0.25), interval(-0.04, 0.04)
    local = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(rklut.RKLUT.TANH_LOCAL, (_DPUExpr(RKDPUOp.MUL, (source, 16.0)),)), 0.25))
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
    parsed_div = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], src)
    ret = _DPUExpr(RKDPUOp.MUL, (parsed_div[1], parsed_div[0])) if isinstance(parsed_div[0], float) and math.isinf(parsed_div[0]) \
      else _DPUExpr(RKDPUOp.DIV, parsed_div)
  elif u.op is Ops.MUL and (logarithm:=next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.LOG2), None)) is not None:
    constant = next((_unwrap_same_cast(x) for x in u.src if _unwrap_same_cast(x).op is Ops.CONST), None)
    # Rejected scaled-log/atanh WIP: accepting arbitrary finite scales lowered atanh in 61 stages, but its ratio reaches 199.
    # Adding symmetric >4/>64 power-of-16 bands made log/atanh 69/73 stages, beyond RKImage's 64-stage dependency contract.
    if constant is None or not isinstance(constant.arg, (int, float)) or not any(
       math.isclose(float(constant.arg), x) for x in (math.log(2), math.log10(2))): return None
    raw_source = _unwrap_same_cast(logarithm.src[0])
    if raw_source.op is Ops.INDEX and raw_source.dtype is dtypes.float and raw_source.src[0].op is Ops.PARAM and \
       raw_source.src[1].key == output_index.key:
      count, high = int(raw_source.src[0].src[0].arg), RKArg(RKBufferKind.ARG, raw_source.src[0].arg.slot)
      residual = RKArg(RKBufferKind.ARG, high.index, ((count+7)//8)*16)
      correction = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.DIV,
        (residual, _DPUExpr(RKDPUOp.MAX, (high, 2**-14)))), float(constant.arg)*math.log2(math.e)))
      ret = _DPUExpr(RKDPUOp.ADD, (_log2_expr(high, float(constant.arg)), correction))
    else:
      operand = _parse_dpu_expr(raw_source, output_index, memo)
      if operand is None: return None
      ret = _log2_expr(operand, float(constant.arg))
  elif u.op in (Ops.ADD, Ops.MUL, Ops.MAX):
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in src): return None
    ret = _DPUExpr({Ops.ADD:RKDPUOp.ADD, Ops.MUL:RKDPUOp.MUL, Ops.MAX:RKDPUOp.MAX}[u.op], src)  # type: ignore[arg-type]
  elif u.op is Ops.EXP2:
    exp_operand = _unwrap_same_cast(u.src[0])
    exp_factor = next((x for x in exp_operand.src if x.op is Ops.CONST and isinstance(x.arg, (int, float))), None) \
      if exp_operand.op is Ops.MUL else None
    exp_source = next((x for x in exp_operand.src if x is not exp_factor), None) if exp_factor is not None else None
    operand = _parse_dpu_expr(exp_source if exp_source is not None else u.src[0], output_index, memo)
    if operand is None: return None
    if exp_factor is not None and math.isclose(float(exp_factor.arg), math.log2(math.e)): ret = _exp_expr(operand)
    else: ret = _DPUExpr(rklut.RKLUT.EXP2, (operand,))
  elif u.op is Ops.LOG2:
    raw_source = _unwrap_same_cast(u.src[0])
    if raw_source.op is Ops.INDEX and raw_source.dtype is dtypes.float and raw_source.src[0].op is Ops.PARAM and \
       raw_source.src[1].key == output_index.key:
      count, high = int(raw_source.src[0].src[0].arg), RKArg(RKBufferKind.ARG, raw_source.src[0].arg.slot)
      residual = RKArg(RKBufferKind.ARG, high.index, ((count+7)//8)*16)
      correction = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.DIV,
        (residual, _DPUExpr(RKDPUOp.MAX, (high, 2**-14)))), math.log2(math.e)))
      ret = _DPUExpr(RKDPUOp.ADD, (_log2_expr(high), correction))
    else:
      operand = _parse_dpu_expr(raw_source, output_index, memo)
      if operand is None: return None
      ret = _log2_expr(operand)
  elif u.op is Ops.RECIPROCAL:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.DIV, (1.0, operand))
  elif u.op is Ops.TRUNC:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _trunc_expr(operand)
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
    reverse_inf_select = cond.op is Ops.CMPLT and isinstance(parsed[0], float) and \
      ((false_u.key == rhs_u.key and true_u.op is Ops.CONST and math.isinf(float(true_u.arg))) or
       (true_u.key == rhs_u.key and false_u.op is Ops.CONST and math.isinf(float(false_u.arg))))
    threshold_select = cond.op is Ops.CMPLT and isinstance(parsed[1], float) and \
      ((true_u.key == lhs_u.key and false_u.op is Ops.CONST and math.isfinite(float(false_u.arg)) and float(false_u.arg) != parsed[1]) or
       (false_u.key == lhs_u.key and true_u.op is Ops.CONST and math.isfinite(float(true_u.arg)) and float(true_u.arg) != parsed[1]))
    select_denominator:_DPUExpr|RKArg|float
    if reverse_inf_select:
      mask = _parse_mask_expr(cond, output_index, memo)
      if mask is None: return None
      threshold_value, source_value = cast(float, parsed[0]), parsed[1]
      if false_u.key == rhs_u.key:
        base_value, inf_value, select_denominator = (_DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX,
          (_DPUExpr(RKDPUOp.MUL, (source_value, -1.0)), -threshold_value)), -1.0)),
          float(true_u.arg), _DPUExpr(RKDPUOp.SUB, (1.0, mask)))
      else:
        base_value, inf_value, select_denominator = _DPUExpr(RKDPUOp.MAX, (source_value, threshold_value)), float(false_u.arg), mask
      correction = _DPUExpr(RKDPUOp.MUL, (math.copysign(1.0, inf_value),
        _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.DIV, (1.0, select_denominator)), 1.0))))
      ret = _DPUExpr(RKDPUOp.ADD, (base_value, correction))
    elif threshold_select:
      mask = _parse_mask_expr(cond, output_index, memo)
      if mask is None: return None
      source_value, threshold_value = parsed[0], cast(float, parsed[1])
      if true_u.key == lhs_u.key:
        base_value = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX,
          (_DPUExpr(RKDPUOp.MUL, (source_value, -1.0)), -threshold_value)), -1.0))
        ret = _DPUExpr(RKDPUOp.ADD, (base_value, _DPUExpr(RKDPUOp.MUL,
          (float(false_u.arg)-threshold_value, _DPUExpr(RKDPUOp.SUB, (1.0, mask))))))
      else:
        base_value = _DPUExpr(RKDPUOp.MAX, (source_value, threshold_value))
        ret = _DPUExpr(RKDPUOp.ADD, (base_value, _DPUExpr(RKDPUOp.MUL, (float(true_u.arg)-threshold_value, mask))))
    elif ordered_max: ret = _DPUExpr(RKDPUOp.MAX, parsed)
    elif ordered_min:
      negative = tuple(_DPUExpr(RKDPUOp.MUL, (x, -1.0)) for x in parsed)
      ret = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, negative), -1.0))
    else:
      mask = _parse_mask_expr(cond, output_index, memo)
      arms = tuple(_parse_dpu_expr(x, output_index, memo) for x in (true_u, false_u))
      if mask is None or any(x is None for x in arms): return None
      true, false = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], arms)
      inf_arm = next(((idx, arm) for idx,arm in enumerate((true, false)) if isinstance(arm, float) and math.isinf(arm)), None)
      other_arm = (false, true)[inf_arm[0]] if inf_arm is not None else None
      if inf_arm is not None and not (isinstance(other_arm, float) and math.isinf(other_arm)):
        inf_index, inf_value = inf_arm
        finite_value, select_denominator = (false, _DPUExpr(RKDPUOp.SUB, (1.0, mask))) if inf_index == 0 else (true, mask)
        correction = _DPUExpr(RKDPUOp.MUL, (math.copysign(1.0, inf_value),
          _DPUExpr(RKDPUOp.SUB, (_DPUExpr(RKDPUOp.DIV, (1.0, select_denominator)), 1.0))))
        ret = _DPUExpr(RKDPUOp.ADD, (finite_value, correction))
      else:
        ret = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (true, mask)),
          _DPUExpr(RKDPUOp.MUL, (false, _DPUExpr(RKDPUOp.SUB, (1.0, mask))))))
  else: return None
  memo[u] = ret
  return ret

def _lower_int_roots(roots:tuple[_DPUExpr, ...], output:RKArg, count:int) -> tuple[tuple[RKDPUStage, ...], tuple[RKScratch, ...]]|None:
  """Schedule shared exact-byte expressions once and write four raw int32 planes."""
  plane_size, stages, scratch_count = ((count+7)//8)*16, [], 0
  order:list[_DPUExpr] = []
  def visit(expr:_DPUExpr) -> None:
    for src in expr.src:
      if isinstance(src, _DPUExpr) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  for root in roots: visit(root)
  uses = {expr:sum(src == expr for node in order for src in node.src) for expr in order}
  values:dict[_DPUExpr, RKArg] = {}
  free:list[int] = []
  destinations = {root:RKArg(output.kind, output.index, plane*plane_size) for plane,root in enumerate(roots)}
  for expr in order:
    src = tuple(values[x] if isinstance(x, _DPUExpr) else x for x in expr.src)
    if expr in destinations: dst = destinations[expr]
    elif (reuse:=next((values[x] for x in expr.src if isinstance(x, _DPUExpr) and uses[x] == 1 and
                       values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    stages.append(RKDPUStage(cast(RKDPUOp, expr.op), dst, src[0], src[1] if len(src) > 1 else None, count))
    values[expr] = dst
    for dependency in expr.src:
      if isinstance(dependency, _DPUExpr):
        uses[dependency] -= 1
        arg = values[dependency]
        if uses[dependency] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  return (tuple(stages), tuple(RKScratch(plane_size) for _ in range(scratch_count))) if len(stages) <= 64 else None

def _lower_int_where(u:UOp, output_index:UOp, output:RKArg, count:int) -> RKDPUProgram|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.WHERE: return None
  true_u, false_u = (_unwrap_same_cast(x) for x in u.src[1:])
  if any(x.op is not Ops.CONST or not dtypes.is_int(x.dtype) for x in (true_u, false_u)): return None
  mask = _parse_mask_expr(u.src[0], output_index, {})
  if mask is None: return None
  roots = tuple(_DPUExpr(RKDPUOp.ADD, (float((int(false_u.arg) >> shift) & 0xff), _DPUExpr(RKDPUOp.MUL,
    (mask, float(((int(true_u.arg) >> shift) & 0xff)-((int(false_u.arg) >> shift) & 0xff)))))) for shift in (24, 16, 8, 0))
  if (lowered:=_lower_int_roots(roots, output, count)) is None: return None
  bool_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.bool and
                                   x.src[0].op is Ops.PARAM))
  return RKDPUProgram(*lowered, bool_inputs=bool_inputs, int_outputs=(output.index,))

def _lower_int_fill(u:UOp, output:RKArg, count:int) -> RKDPUProgram|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.CONST or not dtypes.is_int(u.dtype): return None
  roots = tuple(_DPUExpr(RKDPUOp.ADD, (float((int(u.arg) >> shift) & 0xff),
    _DPUExpr(RKDPUOp.MUL, (0.0, float(plane+1))))) for plane,shift in enumerate((24, 16, 8, 0)))
  return RKDPUProgram(*lowered, int_outputs=(output.index,)) if (lowered:=_lower_int_roots(roots, output, count)) is not None else None

def _lower_int_bool_cast(u:UOp, output_index:UOp, output:RKArg, count:int) -> RKDPUProgram|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.CAST or u.dtype is not dtypes.int or u.src[0].dtype is not dtypes.bool: return None
  mask = _parse_mask_expr(u.src[0], output_index, {})
  if mask is None: return None
  roots = tuple(_DPUExpr(RKDPUOp.ADD, ((mask if plane == 3 else 0.0),
    _DPUExpr(RKDPUOp.MUL, (0.0, float(plane+1))))) for plane in range(4))
  if (lowered:=_lower_int_roots(roots, output, count)) is None: return None
  bool_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.bool and
                                   x.src[0].op is Ops.PARAM))
  return RKDPUProgram(*lowered, bool_inputs=bool_inputs, int_outputs=(output.index,))

def _lower_int_extrema(u:UOp, output_index:UOp, output:RKArg, count:int) -> RKDPUProgram|None:
  u, minimum = _unwrap_same_cast(u), False
  if u.op is Ops.MAX: lhs_u, rhs_u = u.src
  elif u.op is Ops.XOR and any(x.op is Ops.CONST and int(x.arg) == -1 for x in u.src):
    inner = next((_unwrap_same_cast(x) for x in u.src if x.op is not Ops.CONST), None)
    if inner is None or inner.op is not Ops.MAX: return None
    inverted = tuple(_unwrap_same_cast(x) for x in inner.src)
    if any(x.op is not Ops.XOR or not any(y.op is Ops.CONST and int(y.arg) == -1 for y in x.src) for x in inverted): return None
    lhs_u, rhs_u = (next(y for y in x.src if y.op is not Ops.CONST) for x in inverted)
    minimum = True
  else: return None
  operands = tuple(_int_operand_bytes(x, output_index) for x in (lhs_u, rhs_u))
  if any(x is None for x in operands): return None
  lhs, rhs = cast(tuple[tuple[RKArg|float, ...], tuple[RKArg|float, ...]], operands)
  less = _int_cmp_expr(Ops.CMPLT, lhs, rhs)
  roots:list[_DPUExpr] = []
  for plane,(left,right) in enumerate(zip(lhs, rhs)):
    low, high = (right, left) if minimum else (left, right)
    selected = _DPUExpr(RKDPUOp.ADD, (low, _DPUExpr(RKDPUOp.MUL, (less, _DPUExpr(RKDPUOp.SUB, (high, low))))))
    if plane == 0:
      upper = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (selected, 127.0)),))
      selected = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.ADD, (selected, 128.0)), _DPUExpr(RKDPUOp.MUL, (upper, -256.0))))
    roots.append(selected)
  if (lowered:=_lower_int_roots(tuple(roots), output, count)) is None: return None
  int_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.int and
                                  x.src[0].op is Ops.PARAM))
  tiled_int_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in u.toposort() if x.op is Ops.INDEX and x.dtype is dtypes.int and
    x.src[0].op is Ops.PARAM and count % int(x.src[0].src[0].arg) == 0 and x.src[1].key != output_index.key))
  return RKDPUProgram(*lowered, int_inputs=int_inputs, int_outputs=(output.index,), tiled_int_inputs=tiled_int_inputs)

def _lower_int_copy(u:UOp, output_index:UOp, output:RKArg, count:int) -> RKDPUProgram|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.INDEX or u.dtype is not dtypes.int or u.src[0].op is not Ops.PARAM: return None
  source, input_count = u.src[0].arg.slot, int(u.src[0].src[0].arg)
  transposed, tiled = False, u.src[1].key != output_index.key and count % input_count == 0 and \
    ((u.src[1].op is Ops.RANGE and input_count == int(u.src[1].src[0].arg)) or
     (u.src[1].op is Ops.CONST and input_count == 1 and int(u.src[1].arg) == 0))
  if u.src[1].key != output_index.key and not tiled:
    out_aff, inp_aff = _affine(output_index), _affine(u.src[1])
    if out_aff is None or inp_aff is None or out_aff[1] != 0 or inp_aff[1] != 0 or set(out_aff[0]) != set(inp_aff[0]): return None
    axes = tuple(out_aff[0])
    sizes = {node.arg[0]:int(node.src[0].arg) for index in (output_index, u.src[1]) for node in index.toposort() if node.op is Ops.RANGE}
    if len(axes) != 2 or sizes.get(axes[0]) != sizes.get(axes[1]) or count != sizes.get(axes[0], 0)**2: return None
    side = sizes[axes[0]]
    if not (out_aff[0][axes[0]] == side and out_aff[0][axes[1]] == 1 and
            inp_aff[0][axes[0]] == 1 and inp_aff[0][axes[1]] == side): return None
    transposed = True
  plane_size = ((count+7)//8)*16
  stages = tuple(RKDPUStage(RKDPUOp.ADD, RKArg(output.kind, output.index, plane*plane_size), 0.0,
                            RKArg(RKBufferKind.ARG, source, plane*plane_size), count) for plane in range(4))
  return RKDPUProgram(stages, (), int_inputs=(source,), int_outputs=(output.index,),
                      transposed_int_inputs=(source,) if transposed else (), tiled_int_inputs=(source,) if tiled else (), raw_int_inputs=(source,))

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Lower one contiguous fp16/int32/bool-ABI store to a UOp-free primitive DPU plan."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1 or (store:=stores[0]).src[0].op is not Ops.INDEX or \
     store.src[0].dtype not in (dtypes.half, dtypes.int, dtypes.float, dtypes.bool): return None
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_param.src[0].op is not Ops.CONST: return None
  count = int(out_param.src[0].arg)
  if count <= 0 or int(out_index.vmin) != 0 or int(out_index.vmax) != count-1: return None
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if store.src[0].dtype is dtypes.int:
    if (int_fill:=_lower_int_fill(store.src[1], output, count)) is not None: return int_fill
    if (int_bool:=_lower_int_bool_cast(store.src[1], out_index, output, count)) is not None: return int_bool
    if (int_where:=_lower_int_where(store.src[1], out_index, output, count)) is not None: return int_where
    if (int_extrema:=_lower_int_extrema(store.src[1], out_index, output, count)) is not None: return int_extrema
    if (int_copy:=_lower_int_copy(store.src[1], out_index, output, count)) is not None: return int_copy
  root: _DPUExpr|RKArg|float|None
  predicate = _canonical_finite_predicate(store.src[1]) if store.src[0].dtype is dtypes.bool else None
  if predicate is not None:
    source = _parse_dpu_expr(predicate[0], out_index, {})
    if source is None: return None
    positive_inf = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (source, 65504.0)),))
    negative_inf = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (-65504.0, source)),))
    either, both = _DPUExpr(RKDPUOp.MAX, (positive_inf, negative_inf)), _DPUExpr(RKDPUOp.MUL, (positive_inf, negative_inf))
    root = (both if predicate[1] == "nan" else _DPUExpr(RKDPUOp.SUB, (
      positive_inf if predicate[1] == "positive_inf" else negative_inf if predicate[1] == "negative_inf" else either, both))
      if predicate[1] != "finite" else _DPUExpr(RKDPUOp.SUB, (1.0, either)))
  else:
    root = (_parse_mask_expr(store.src[1], out_index, {}, ieee=True) if store.src[0].dtype is dtypes.bool else
            _parse_dpu_expr(store.src[1], out_index, {}))
  if root is None: return None
  # Native int32 WDMA emits four values per eight-lane fp16 atom. Constant
  # fills can safely double their source lanes; dynamic conversion needs an
  # explicit packed layout and is deliberately not inferred here.
  if store.src[0].dtype is dtypes.int and not isinstance(root, float): return None
  if not isinstance(root, _DPUExpr):
    if store.src[0].dtype in (dtypes.int, dtypes.float):
      tile = 64 if store.src[0].dtype is dtypes.int else 4
      fill_stages = tuple(RKDPUStage(RKDPUOp.ADD, RKArg(output.kind, output.index, start*4), 0.0, root,
                                    min(tile, count-start), store.src[0].dtype) for start in range(0, count, tile))
      return RKDPUProgram(fill_stages, ()) if len(fill_stages) <= 64 else None
    stage = RKDPUStage(RKDPUOp.ADD, output, 0.0, root, count, dtypes.half if store.src[0].dtype is dtypes.bool else store.src[0].dtype)
    return RKDPUProgram((stage,), (), bool_outputs=(output.index,) if store.src[0].dtype is dtypes.bool else ())
  if root.op is RKDPUOp.LUT and root.lut is rklut.RKLUT.EXP2 and len(root.src) == 1 and isinstance(root.src[0], RKArg):
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
    assert isinstance(expr.op, RKDPUOp)
    stages.append(RKDPUStage(expr.op, dst, src[0], src[1] if len(src) > 1 else None, count,
                             dtypes.half if expr is root and store.src[0].dtype in (dtypes.float, dtypes.bool) else
                             (store.src[0].dtype if expr is root else dtypes.half), expr.lut))
    values[expr] = dst
    for dependency in expr.src:
      if isinstance(dependency, _DPUExpr):
        uses[dependency] -= 1
        arg = values[dependency]
        if uses[dependency] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  size = ((count+7)//8)*16
  fp32_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in store.src[1].toposort() if x.op is Ops.INDEX and x.dtype is dtypes.float and
                                   x.src[0].op is Ops.PARAM))
  fp32_outputs = (output.index,) if store.src[0].dtype is dtypes.float else ()
  bool_outputs = (output.index,) if store.src[0].dtype is dtypes.bool else ()
  bool_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in store.src[1].toposort() if x.op is Ops.INDEX and x.dtype is dtypes.bool and
                                   x.src[0].op is Ops.PARAM))
  int_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in store.src[1].toposort() if x.op is Ops.INDEX and x.dtype is dtypes.int and
                                  x.src[0].op is Ops.PARAM))
  numeric_int_inputs = tuple(dict.fromkeys(x.src[0].src[0].arg.slot for x in store.src[1].toposort() if x.op is Ops.CAST and
    x.dtype is dtypes.half and x.src[0].op is Ops.INDEX and x.src[0].dtype is dtypes.int and x.src[0].src[0].op is Ops.PARAM))
  int_inputs = tuple(x for x in int_inputs if x not in numeric_int_inputs)
  tiled_inputs = tuple(dict.fromkeys(x.src[0].arg.slot for x in store.src[1].toposort() if x.op is Ops.INDEX and x.dtype is dtypes.half and
    x.src[0].op is Ops.PARAM and int(out_index.vmin) == 0 and count % int(x.src[0].src[0].arg) == 0 and
    ((x.src[1].op is Ops.RANGE and int(x.src[0].src[0].arg) == int(x.src[1].src[0].arg)) or
     (x.src[1].op is Ops.CONST and int(x.src[0].src[0].arg) == 1 and int(x.src[1].arg) == 0)) and x.src[1].key != out_index.key))
  if len(stages) > 64: return None
  return RKDPUProgram(tuple(stages), tuple(RKScratch(size) for _ in range(scratch_count)), fp32_inputs, fp32_outputs, bool_outputs, bool_inputs,
                      int_inputs, tiled_inputs, numeric_int_inputs=numeric_int_inputs)

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
  assert plan.op is RKDPUOp.LUT and plan.lut is not None
  width, surf_stride = (plan.count+7)//8-1, ((plan.count+7)//8)*16
  table, bn_mul, minus_exp = {
    rklut.RKLUT.EXP2:(rklut.RK_LUT_EXP2, rklut.RK_LUT_EXP2_BN_MUL, rklut.RK_LUT_EXP2_MINUS_EXP),
    rklut.RKLUT.HARDSWISH:(rklut.RK_LUT_HARDSWISH, rklut.RK_LUT_HARDSWISH_BN_MUL, rklut.RK_LUT_HARDSWISH_MINUS_EXP),
    rklut.RKLUT.HARDSWISH_LOCAL:(rklut.RK_LUT_HARDSWISH_LOCAL, rklut.RK_LUT_HARDSWISH_LOCAL_BN_MUL,
                            rklut.RK_LUT_HARDSWISH_LOCAL_MINUS_EXP),
    rklut.RKLUT.TANH:(rklut.RK_LUT_TANH, rklut.RK_LUT_TANH_BN_MUL, rklut.RK_LUT_TANH_MINUS_EXP),
    rklut.RKLUT.TANH_LOCAL:(rklut.RK_LUT_TANH_LOCAL, rklut.RK_LUT_TANH_LOCAL_BN_MUL, rklut.RK_LUT_TANH_LOCAL_MINUS_EXP),
    rklut.RKLUT.SIGMOID:(rklut.RK_LUT_SIGMOID, rklut.RK_LUT_SIGMOID_BN_MUL, rklut.RK_LUT_SIGMOID_MINUS_EXP),
    rklut.RKLUT.SIGMOID_LOCAL:(rklut.RK_LUT_SIGMOID_LOCAL, rklut.RK_LUT_SIGMOID_LOCAL_BN_MUL,
                          rklut.RK_LUT_SIGMOID_LOCAL_MINUS_EXP),
    rklut.RKLUT.QUICK_GELU:(rklut.RK_LUT_QUICK_GELU, rklut.RK_LUT_QUICK_GELU_BN_MUL, rklut.RK_LUT_QUICK_GELU_MINUS_EXP),
    rklut.RKLUT.QUICK_GELU_LOCAL:(rklut.RK_LUT_QUICK_GELU_LOCAL, rklut.RK_LUT_QUICK_GELU_LOCAL_BN_MUL,
                             rklut.RK_LUT_QUICK_GELU_LOCAL_MINUS_EXP),
    rklut.RKLUT.GELU_TANH:(rklut.RK_LUT_GELU_TANH, rklut.RK_LUT_GELU_TANH_BN_MUL, rklut.RK_LUT_GELU_TANH_MINUS_EXP),
    rklut.RKLUT.GELU_TANH_LOCAL:(rklut.RK_LUT_GELU_TANH_LOCAL, rklut.RK_LUT_GELU_TANH_LOCAL_BN_MUL,
                            rklut.RK_LUT_GELU_TANH_LOCAL_MINUS_EXP),
    rklut.RKLUT.GELU_EXACT:(rklut.RK_LUT_GELU_EXACT, rklut.RK_LUT_GELU_EXACT_BN_MUL, rklut.RK_LUT_GELU_EXACT_MINUS_EXP),
    rklut.RKLUT.GELU_EXACT_LOCAL:(rklut.RK_LUT_GELU_EXACT_LOCAL, rklut.RK_LUT_GELU_EXACT_LOCAL_BN_MUL,
                             rklut.RK_LUT_GELU_EXACT_LOCAL_MINUS_EXP),
    rklut.RKLUT.ERF:(rklut.RK_LUT_ERF, rklut.RK_LUT_ERF_BN_MUL, rklut.RK_LUT_ERF_MINUS_EXP),
    rklut.RKLUT.ERF_LOCAL:(rklut.RK_LUT_ERF_LOCAL, rklut.RK_LUT_ERF_LOCAL_BN_MUL, rklut.RK_LUT_ERF_LOCAL_MINUS_EXP),
    **{op:(getattr(rklut, f"RK_LUT_{name}"), getattr(rklut, f"RK_LUT_{name}_BN_MUL"), getattr(rklut, f"RK_LUT_{name}_MINUS_EXP"))
       for op,name in ((rklut.RKLUT.ELU1,"ELU1"),(rklut.RKLUT.ELU1_LOCAL,"ELU1_LOCAL"),(rklut.RKLUT.ELU01,"ELU01"),
                       (rklut.RKLUT.ELU01_LOCAL,"ELU01_LOCAL"),(rklut.RKLUT.SELU,"SELU"),(rklut.RKLUT.SELU_LOCAL,"SELU_LOCAL"),
                       (rklut.RKLUT.MISH,"MISH"),(rklut.RKLUT.MISH_LOCAL,"MISH_LOCAL"),(rklut.RKLUT.LOGSIGMOID,"LOGSIGMOID"),
                       (rklut.RKLUT.LOGSIGMOID_TAIL,"LOGSIGMOID_TAIL"),(rklut.RKLUT.SOFTPLUS1,"SOFTPLUS1"),
                       (rklut.RKLUT.SOFTPLUS1_TAIL,"SOFTPLUS1_TAIL"),(rklut.RKLUT.SOFTPLUS3,"SOFTPLUS3"),
                       (rklut.RKLUT.SOFTPLUS3_TAIL,"SOFTPLUS3_TAIL"),(rklut.RKLUT.SOFTPLUS13,"SOFTPLUS13"),
                       (rklut.RKLUT.SINH,"SINH"),(rklut.RKLUT.SINH_LOCAL,"SINH_LOCAL"),(rklut.RKLUT.COSH,"COSH"),
                       (rklut.RKLUT.SQRT,"SQRT"),(rklut.RKLUT.RSQRT,"RSQRT"),(rklut.RKLUT.EXP,"EXP"),(rklut.RKLUT.EXP_LOCAL,"EXP_LOCAL"),
                       (rklut.RKLUT.CELU2,"CELU2"),(rklut.RKLUT.CELU2_LOCAL,"CELU2_LOCAL"),(rklut.RKLUT.CELU3,"CELU3"),
                       (rklut.RKLUT.CELU3_LOCAL,"CELU3_LOCAL"),(rklut.RKLUT.CELU4,"CELU4"),(rklut.RKLUT.CELU4_LOCAL,"CELU4_LOCAL"),
                       (rklut.RKLUT.LOG2,"LOG2"),(rklut.RKLUT.LOG2_LOCAL,"LOG2_LOCAL"),(rklut.RKLUT.LOG,"LOG"),
                       (rklut.RKLUT.LOG_LOCAL,"LOG_LOCAL"),(rklut.RKLUT.LOG10,"LOG10"),(rklut.RKLUT.LOG10_LOCAL,"LOG10_LOCAL"),
                       (rklut.RKLUT.ASIN,"ASIN"),(rklut.RKLUT.ASIN_DETAIL,"ASIN_DETAIL"),(rklut.RKLUT.ACOS,"ACOS"),
                       (rklut.RKLUT.ACOS_ENDPOINT,"ACOS_ENDPOINT"),(rklut.RKLUT.ACOS_FINE_ENDPOINT,"ACOS_FINE_ENDPOINT"),
                       (rklut.RKLUT.ATAN,"ATAN"),(rklut.RKLUT.ATAN_DETAIL,"ATAN_DETAIL"),(rklut.RKLUT.SIN,"SIN"),
                       (rklut.RKLUT.SIN_LOCAL,"SIN_LOCAL"),(rklut.RKLUT.COS,"COS"),(rklut.RKLUT.COS_LOCAL,"COS_LOCAL"),
                       (rklut.RKLUT.ATANH,"ATANH"),(rklut.RKLUT.ATANH_DETAIL,"ATANH_DETAIL"),
                       (rklut.RKLUT.ASINH_CORE,"ASINH_CORE"),(rklut.RKLUT.ASINH_RANGE,"ASINH_RANGE"),
                       (rklut.RKLUT.ACOSH_CORE,"ACOSH_CORE"),(rklut.RKLUT.ACOSH_RANGE,"ACOSH_RANGE"))}}[plan.lut]
  post_scale = {rklut.RKLUT.SOFTPLUS3:rklut.RK_LUT_SOFTPLUS3_POST_SCALE,
                rklut.RKLUT.SOFTPLUS3_TAIL:rklut.RK_LUT_SOFTPLUS3_TAIL_POST_SCALE}.get(plan.lut, 1.0)
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
    (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001 if post_scale == 1 else 0x10000 | round(post_scale*32768)),
    (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SHIFT, (minus_exp << 12) | (0 if post_scale == 1 else 15)),
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

def _emit_roundoff_lut(stage_idx:int, plan:RKDPUStage, src:RKArg) -> RKStage:
  """Emit the RK3588 algorithm-23 round-to-nearest-even LUT contract."""
  width, surf_stride, cmds = (plan.count+7)//8-1, ((plan.count+7)//8)*16, []
  table = rklut.RK_LUT_ROUNDOFF
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    cmds.extend(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value) for value in table[table_id*513:(table_id+1)*513])
  dpu = ((rk.REG_DPU_S_POINTER, 0x30), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width), (rk.REG_DPU_BN_CFG, 0x53),
    (rk.REG_DPU_EW_CFG, 0x302), (rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (0x40c4, 0),
    (rk.REG_DPU_LUT_CFG, 0x68), (rk.REG_DPU_LUT_INFO, 0xe0e00), (rk.REG_DPU_LUT_LE_START, 0),
    (rk.REG_DPU_LUT_LE_END, 0x44000000), (rk.REG_DPU_LUT_LO_START, 0x44000000),
    (rk.REG_DPU_LUT_LO_END, 0x44800000), (0x4120, 23107), (0x4124, 22))
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
           _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101),
           _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, src.kind, src.index, src.addend))
  reads, writes = ((src.index,) if src.kind is RKBufferKind.ARG else ()), \
                   ((plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ())
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
    # Rejected partial Maximum WIP: OUT_CVT_OFFSET can fill INT_MAX exactly, but later cases still require int copy/compare and bool packing.
    # int_fill = plan.out_dtype is dtypes.int and plan.op is RKDPUOp.ADD and plan.lhs == 0.0 and isinstance(plan.rhs, float)
    lhs, rhs = materialize(plan.lhs, material_count), materialize(plan.rhs, material_count) if plan.rhs is not None else None
    if plan.op is RKDPUOp.LUT:
      stages.append(_emit_roundoff_lut(stage_idx, plan, lhs) if plan.lut is rklut.RKLUT.ROUNDOFF else _emit_lut(stage_idx, plan, lhs))
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
  return RKImage(target, tuple(stages), program.scratch, bytes(constants), fp32_inputs=program.fp32_inputs,
                 fp32_outputs=program.fp32_outputs, bool_outputs=program.bool_outputs, bool_inputs=program.bool_inputs,
                 int_inputs=program.int_inputs, tiled_inputs=program.tiled_inputs, int_outputs=program.int_outputs,
                 transposed_int_inputs=program.transposed_int_inputs, tiled_int_inputs=program.tiled_int_inputs,
                 raw_int_inputs=program.raw_int_inputs, numeric_int_inputs=program.numeric_int_inputs)

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
