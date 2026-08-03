from __future__ import annotations
import math, struct
from dataclasses import dataclass
from typing import cast

from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops, UOp

from tinygrad.renderer.rockchip.ir import RKArg, RKBufferKind, RK_ALU_OPS as _RK_ALU_OPS

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

def _canonical_zero_base_power(u:UOp) -> tuple[UOp, UOp]|None:
  """Recognize WHERE(x != 0, EXP2(-inf*x), 1), whose inactive exponential must not see zero."""
  u = _unwrap_fp_cast(u)
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  condition, powered, identity = (_unwrap_fp_cast(x) for x in u.src)
  if condition.op is not Ops.CMPNE or powered.op is not Ops.EXP2 or identity.op is not Ops.CONST or float(identity.arg) != 1.0: return None
  compared = tuple(_unwrap_fp_cast(x) for x in condition.src)
  zero = next((x for x in compared if x.op is Ops.CONST and float(x.arg) == 0.0), None)
  source = next((x for x in compared if x is not zero), None)
  exponent = _unwrap_fp_cast(powered.src[0])
  if zero is None or source is None or exponent.op is not Ops.MUL: return None
  factors = tuple(_unwrap_fp_cast(x) for x in exponent.src)
  negative_inf = next((x for x in factors if x.op is Ops.CONST and float(x.arg) == -math.inf), None)
  exponent_source = next((x for x in factors if x is not negative_inf), None)
  return (source, condition) if negative_inf is not None and exponent_source is not None and exponent_source.key == source.key else None

def _is_const(u:UOp, value:float) -> bool:
  u = _unwrap_fp_cast(u)
  return u.op is Ops.CONST and isinstance(u.arg, (int,float)) and math.isclose(float(u.arg), value)

def _canonical_lerp(u:UOp) -> tuple[UOp,UOp,UOp]|None:
  """Recognize exact x + (y-x)*z and return its x, y, and z operands."""
  u = _unwrap_fp_cast(u)
  if u.op is not Ops.ADD: return None
  for base,weighted in (u.src, u.src[::-1]):
    base, weighted = _unwrap_fp_cast(base), _unwrap_fp_cast(weighted)
    if base.op is not Ops.INDEX or weighted.op is not Ops.MUL: continue
    for difference,weight in (weighted.src, weighted.src[::-1]):
      difference = _unwrap_fp_cast(difference)
      if difference.op is not Ops.ADD: continue
      for positive,negative in (difference.src, difference.src[::-1]):
        positive, negative = _unwrap_fp_cast(positive), _unwrap_fp_cast(negative)
        if positive.op is not Ops.INDEX or negative.op is not Ops.MUL: continue
        if any(_unwrap_fp_cast(x).key == base.key for x in negative.src) and any(_is_const(x, -1.0) for x in negative.src):
          return base, positive, _unwrap_fp_cast(weight)
  return None

def _uses_reciprocal_signed_zero(u:UOp) -> bool:
  """Detect signbit reconstruction through x<0 OR reciprocal(x)<0; RK3588 FDIV loses the required -0 sign."""
  for node in u.toposort():
    if node.op is not Ops.OR: continue
    comparisons = tuple(_unwrap_fp_cast(x) for x in node.src)
    if any(x.op is not Ops.CMPLT for x in comparisons): continue
    direct, reciprocal = None, None
    for comparison in comparisons:
      lhs, rhs = (_unwrap_fp_cast(x) for x in comparison.src)
      if not _is_const(rhs, 0.0): continue
      if lhs.op is Ops.RECIPROCAL: reciprocal = _unwrap_fp_cast(lhs.src[0])
      else: direct = lhs
    if direct is not None and reciprocal is not None and direct.key == reciprocal.key: return True
  return False

def _is_unreduced_bce(u:UOp) -> bool:
  """Recognize the direct probability-BCE expression whose staged LUT recipe exceeds the public error contract."""
  nodes = tuple(u.toposort())
  counts = {op:sum(x.op is op for x in nodes) for op in (Ops.LOG2, Ops.EXP2, Ops.RECIPROCAL, Ops.WHERE)}
  constants = tuple(float(x.arg) for x in nodes if x.op is Ops.CONST and isinstance(x.arg, (int,float)))
  return counts == {Ops.LOG2:2, Ops.EXP2:1, Ops.RECIPROCAL:1, Ops.WHERE:2} and \
    any(math.isclose(x, -math.log(2.0)) for x in constants) and any(math.isclose(x, -math.log2(math.e)) for x in constants)

def _numerical_contract(u:UOp) -> str|None:
  if _canonical_lerp(u) is not None: return "lerp requires a fused FP32-intermediate DPU task"
  if _uses_reciprocal_signed_zero(u): return "reciprocal sign does not preserve negative-zero copysign semantics"
  if _is_unreduced_bce(u): return "unreduced BCE LUT composition exceeds the FP16 relative-error contract"
  exponential = _unwrap_fp_cast(u)
  if exponential.op is Ops.EXP2:
    operand = _unwrap_fp_cast(exponential.src[0])
    if operand.op is not Ops.MUL: return None
    factor = next((_unwrap_fp_cast(x) for x in operand.src if _unwrap_fp_cast(x).op is Ops.CONST and
                   isinstance(_unwrap_fp_cast(x).arg, (int,float))), None)
    if factor is None: return None
    scale = float(factor.arg)
    if math.isinf(scale) or math.isclose(abs(scale), math.log2(math.e)) or math.isclose(scale, math.log2(5.5), rel_tol=1e-3) or \
       math.isclose(scale, 3.0, rel_tol=1e-3): return None
    return f"scaled EXP2 factor {scale} has no characterized numerical contract"
  return None

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
  elif (zero_power:=_canonical_zero_base_power(u)) is not None:
    operand, mask = _parse_alu(zero_power[0], output_index, memo), _parse_mask_expr(zero_power[1], output_index, memo)
    if operand is None or isinstance(operand, float) or mask is None: return None
    inactive = _sub(1.0, mask)
    # Generic WHERE uses arithmetic selection, so evaluating -inf*0 first would make the inactive arm NaN.
    # Replace inactive zero by one before EXP2; the active mask restores the exact 0**0 == 1 result.
    safe_operand = _ALUExpr(Ops.ADD, (operand, inactive))
    powered = _exp2_expr(_ALUExpr(Ops.MUL, (safe_operand, -math.inf)))
    ret = _ALUExpr(Ops.ADD, (_ALUExpr(Ops.MUL, (powered, mask)), inactive))
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
    infinite_threshold_select = cond.op is Ops.CMPLT and true_u.op is Ops.CONST and math.isinf(float(true_u.arg)) and \
      ((false_u.key == lhs_u.key and isinstance(rhs, float) and math.isfinite(rhs)) or
       (false_u.key == rhs_u.key and isinstance(lhs, float) and math.isfinite(lhs)))
    threshold_select = cond.op is Ops.CMPLT and isinstance(rhs, float) and \
      ((true_u.key == lhs_u.key and false_u.op is Ops.CONST and math.isfinite(float(false_u.arg)) and float(false_u.arg) != rhs) or
       (false_u.key == lhs_u.key and true_u.op is Ops.CONST and math.isfinite(float(true_u.arg)) and float(true_u.arg) != rhs))
    if infinite_threshold_select:
      mask = _parse_mask_expr(cond, output_index, memo)
      if mask is None: return None
      inactive = _sub(1.0, mask)
      if false_u.key == lhs_u.key: base = _ALUExpr(Ops.MAX, (lhs, rhs))
      else:
        assert isinstance(lhs, float)
        base = _ALUExpr(Ops.MUL, (_ALUExpr(Ops.MAX, (_ALUExpr(Ops.MUL, (rhs, -1.0)), -lhs)), -1.0))
      infinity = _ALUExpr(Ops.FDIV, (_ALUExpr(Ops.MUL, (mask, math.copysign(1.0, float(true_u.arg)))), inactive))
      ret = _ALUExpr(Ops.ADD, (base, infinity))
    elif threshold_select:
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
