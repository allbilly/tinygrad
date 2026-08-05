from __future__ import annotations
import math
from itertools import product
from typing import cast

from tinygrad.dtype import dtypes, Invalid
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import affine as _affine
from tinygrad.renderer.rockchip.limits import RK_MAX_AFFINE_VISITS

def strip_casts(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def static_scalar(u:UOp, ranges:dict[int, int]|dict[UOp, int]) -> int|float|bool|None:
  """Evaluate one compile-time coordinate predicate; tensor loads are never accepted."""
  if u.op is Ops.CAST:
    value = static_scalar(u.src[0], ranges)
    return None if value is None else cast(int|float|bool, u.dtype.const(value))
  if u.op is Ops.CONST: return u.arg
  if u.op is Ops.RANGE:
    return cast(dict[UOp,int], ranges)[u] if u in ranges else cast(dict[int,int], ranges).get(u.arg[0])
  values = tuple(static_scalar(x, ranges) for x in u.src)
  if any(x is None for x in values): return None
  if u.op is Ops.ADD: return values[0]+values[1]  # type: ignore[operator]
  if u.op is Ops.MUL: return values[0]*values[1]  # type: ignore[operator]
  if u.op is Ops.MAX: return max(values)  # type: ignore[type-var]
  if u.op is Ops.FLOORDIV: return int(cast(int|float|bool, values[0]))//int(cast(int|float|bool, values[1]))
  if u.op is Ops.FLOORMOD: return int(cast(int|float|bool, values[0]))%int(cast(int|float|bool, values[1]))
  if u.op is Ops.TRUNC: return math.trunc(cast(int|float, values[0]))
  if u.op is Ops.CMPLT: return values[0] < values[1]  # type: ignore[operator]
  if u.op is Ops.CMPNE: return values[0] != values[1]
  if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
  if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
  if u.op is Ops.WHERE: return values[1] if values[0] else values[2]
  return None

def static_linear_form(u:UOp, point:dict[UOp,int], source:UOp) -> tuple[float,dict[int,float]]|None:
  """Evaluate one statically indexed linear expression without reading runtime tensor data."""
  if u.op is Ops.INDEX:
    if u.src[0].key != source.key or u.dtype is not dtypes.half: return None
    offset = static_scalar(u.src[1],point)
    return (0.0,{offset:1.0}) if isinstance(offset,int) and not isinstance(offset,bool) else None
  if u.op is Ops.WHERE:
    predicate = static_scalar(u.src[0],point)
    return None if predicate is None else static_linear_form(u.src[1] if predicate else u.src[2],point,source)
  if not any(x.op is Ops.INDEX for x in u.toposort()):
    value = static_scalar(u,point)
    return (float(value),{}) if isinstance(value,(int,float,bool)) else None
  if u.op is Ops.CAST:
    # FP16 input promoted into one FP32 expression is linear. A rounded intermediate cast is not.
    return static_linear_form(u.src[0],point,source) if u.dtype is dtypes.float else None
  if u.op not in (Ops.ADD,Ops.MUL) or (lhs:=static_linear_form(u.src[0],point,source)) is None or \
     (rhs:=static_linear_form(u.src[1],point,source)) is None: return None
  lhs_const,lhs_terms = lhs
  rhs_const,rhs_terms = rhs
  if u.op is Ops.ADD:
    terms = dict(lhs_terms)
    for index,weight in rhs_terms.items(): terms[index] = terms.get(index,0.0)+weight
    return lhs_const+rhs_const,{index:weight for index,weight in terms.items() if weight != 0.0}
  if lhs_terms and rhs_terms: return None
  terms = {index:weight*rhs_const for index,weight in lhs_terms.items()}
  for index,weight in rhs_terms.items(): terms[index] = terms.get(index,0.0)+weight*lhs_const
  return lhs_const*rhs_const,{index:weight for index,weight in terms.items() if weight != 0.0}

def conditional_index(u:UOp) -> tuple[UOp, UOp|None, bool]|None:
  """Return the indexed tensor and an optional static zero-mask around it."""
  value = strip_casts(u)
  if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM: return value, None, True
  if value.op is not Ops.WHERE: return None
  condition, positive, negative = value.src
  positive, negative = strip_casts(positive), strip_casts(negative)
  if positive.op is Ops.INDEX and positive.src[0].op is Ops.PARAM and negative.op is Ops.CONST and float(negative.arg) == 0:
    return positive, condition, True
  if negative.op is Ops.INDEX and negative.src[0].op is Ops.PARAM and positive.op is Ops.CONST and float(positive.arg) == 0:
    return negative, condition, False
  return None

def conditional_index_affine(index:UOp) -> tuple[dict[int,int],int]|None:
  """Recover the one real affine address branch from a bounds-checked INDEX."""
  result = _affine(index.src[1])
  if result is None and index.src[1].op is Ops.WHERE:
    branches = tuple(x for branch in index.src[1].src[1:] if branch.arg is not Invalid and (x:=_affine(branch)) is not None)
    if len(branches) == 1: result = branches[0]
  return result

def proves_conv_zero_padding(condition:UOp|None, select_true:bool, ranges:dict[int,int], axes:tuple[int,int,int,int],
                             in_h:int, in_w:int, stride_y:int, stride_x:int, pad_top:int, pad_left:int,
                             dilation_y:int=1, dilation_x:int=1) -> bool:
  """Exhaustively prove that a feature mask selects exactly the in-bounds convolution coordinates."""
  if condition is None: return pad_top == pad_left == 0
  ky_axis,kx_axis,out_y_axis,out_x_axis = axes
  relevant = tuple(dict.fromkeys(u for u in condition.toposort() if u.op is Ops.RANGE))
  if any(u.arg[0] not in ranges for u in relevant) or math.prod(ranges[u.arg[0]] for u in relevant) > RK_MAX_AFFINE_VISITS:
    return False
  for coordinates in product(*(range(ranges[u.arg[0]]) for u in relevant)):
    point = dict(zip(relevant,coordinates))
    by_axis = {u.arg[0]:value for u,value in point.items()}
    selected = static_scalar(condition,point)
    if selected is None: return False
    iy = by_axis.get(ky_axis,0)*dilation_y+by_axis.get(out_y_axis,0)*stride_y-pad_top
    ix = by_axis.get(kx_axis,0)*dilation_x+by_axis.get(out_x_axis,0)*stride_x-pad_left
    if (bool(selected) is select_true) != (0 <= iy < in_h and 0 <= ix < in_w): return False
  return True

def conv_zero_padding(feature_aff:tuple[dict[int,int],int], condition:UOp|None, select_true:bool, ranges:dict[int,int],
                      axes:tuple[int,int,int,int], in_h:int, in_w:int, kernel_h:int, kernel_w:int, out_h:int, out_w:int,
                      stride_y:int, stride_x:int, dilation_y:int=1, dilation_x:int=1) -> tuple[int,int,int,int]|None:
  if feature_aff[1] > 0: return None
  pad_top,pad_left = divmod(-feature_aff[1],in_w)
  if max(pad_top,pad_left) > 15: return None
  effective_h, effective_w = (kernel_h-1)*dilation_y+1, (kernel_w-1)*dilation_x+1
  pad_bottom = next((pad for pad in range(16) if (in_h+pad_top+pad-effective_h)//stride_y+1 == out_h),-1)
  pad_right = next((pad for pad in range(16) if (in_w+pad_left+pad-effective_w)//stride_x+1 == out_w),-1)
  if min(pad_bottom,pad_right) < 0 or condition is None and any((pad_top,pad_bottom,pad_left,pad_right)) or \
     not proves_conv_zero_padding(condition,select_true,ranges,axes,
    in_h,in_w,stride_y,stride_x,pad_top,pad_left,dilation_y,dilation_x): return None
  return pad_top,pad_bottom,pad_left,pad_right

def static_index_selected(u:UOp, index:UOp, ranges:dict[int, int]) -> bool|None:
  """Follow static WHERE branches and report whether one coordinate selects `index`."""
  value = strip_casts(u)
  if value.key == index.key: return True
  if value.op in (Ops.CONST, Ops.INDEX): return False
  if value.op is not Ops.WHERE: return None
  predicate = static_scalar(value.src[0], ranges)
  if predicate is None: return None
  return static_index_selected(value.src[1] if predicate else value.src[2], index, ranges)

def static_selected_index(u:UOp, ranges:dict[UOp, int]) -> UOp|None:
  """Follow one statically decidable WHERE tree to its selected parameter INDEX."""
  value = strip_casts(u)
  if value.op is Ops.INDEX and value.src[0].op is Ops.PARAM: return value
  if value.op is not Ops.WHERE: return None
  predicate = static_scalar(value.src[0],ranges)
  return None if predicate is None else static_selected_index(value.src[1] if predicate else value.src[2],ranges)

def relu_source(u:UOp) -> UOp|None:
  if u.op is not Ops.WHERE or len(u.src) != 3: return None
  cond, positive, zero = u.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2 or cond.src[0].op is not Ops.CONST or float(cond.src[0].arg) != 0 or \
     zero.op is not Ops.CONST or float(zero.arg) != 0: return None
  source = strip_casts(cond.src[1])
  return source if strip_casts(positive).key == source.key else None

def contract_bias_epilogue(stored:UOp, reduce:UOp) -> tuple[UOp, bool]|None:
  """Recognize a channel-bias ADD, optionally followed by ReLU, directly around one contraction."""
  relu_value, relu = relu_source(stored), False
  if relu_value is not None: stored, relu = relu_value, True
  stored = strip_casts(stored)
  if stored.op is not Ops.ADD: return None
  for reduced,bias in (stored.src, stored.src[::-1]):
    bias = strip_casts(bias)
    if strip_casts(reduced).key == reduce.key and bias.op is Ops.INDEX and bias.src[0].op is Ops.PARAM and bias.dtype is dtypes.half:
      return bias, relu
  return None
