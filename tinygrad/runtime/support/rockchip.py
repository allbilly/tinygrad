# pylint: skip-file
# RK3588 NPU compiled backend: pure match/classify + register emission + codec.
# PR 1 native contract: DPU binary EW/copy, CMAC matmul/sum, PPU global max (fp16 only).
# Fill, broadcast, mean, and all other ops are rejected — no host-side tensor arithmetic.
from __future__ import annotations
import struct, math, numpy as np
from dataclasses import dataclass, replace
from tinygrad.dtype import dtypes
from tinygrad.helpers import ceildiv, round_up, prod
from tinygrad.uop.ops import Ops, UOp, ProgramInfo, PatternMatcher, graph_rewrite
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.upat import UPat

# Pre-classification rewrite: MUL(a, RECIPROCAL(b)) → FDIV(a, b) and MUL(RECIPROCAL(b), a) → FDIV(a, b)
# This lets the classifier see FDIV directly instead of nested MUL+RECIPROCAL
_pm_fdiv = PatternMatcher([
  (UPat.var("a") * UPat(Ops.RECIPROCAL, src=(UPat.var("b"),), name="r"), lambda a, b, r: UOp(Ops.FDIV, r.dtype, (a, b))),
  (UPat(Ops.RECIPROCAL, src=(UPat.var("b"),), name="r") * UPat.var("a"), lambda a, b, r: UOp(Ops.FDIV, r.dtype, (a, b))),
])

def _mul_where(w:UOp, x:UOp) -> UOp:
  return UOp(Ops.WHERE, w.dtype, (w.src[0], UOp(Ops.MUL, w.dtype, (w.src[1], x)), UOp(Ops.MUL, w.dtype, (w.src[2], x))))

_pm_where_mul = PatternMatcher([
  (UPat(Ops.MUL, src=(UPat(Ops.WHERE, name="w"), UPat.var("x"))), _mul_where),
  (UPat(Ops.MUL, src=(UPat.var("x"), UPat(Ops.WHERE, name="w"))), lambda x, w: _mul_where(w, x)),
])

# target ids for emit_raw (rkt_get_target(reg) + 1, from rkt_registers.h)
# PC=0x80 is the value used by the working allbilly reference (autogen has 0x100, which is wrong for PC)
_T_DPU, _T_DPU_RDMA, _T_CNA, _T_CORE, _T_PPU, _T_PPU_RDMA, _T_PC = 0x1001, 0x2001, 0x201, 0x801, 0x4001, 0x8001, 0x81
# PC chain tail targets (from ref/rk3588/conv_expt/conv_pershapepatch.py:280)
_T_PC_REG, _T_VERSION = 0x0101, 0x0041
_PC_CHAIN_TAIL_QWORDS = 4

# ---- image format ----
@dataclass(frozen=True)
class RKCmd:
  """One register command word: (target<<48) | (value<<16) | reg."""
  target: int
  reg: int
  value: int
  def pack(self) -> int: return ((self.target & 0xFFFF) << 48) | ((self.value & 0xFFFFFFFF) << 16) | (self.reg & 0xFFFF)

@dataclass(frozen=True)
class RKReloc:
  """Patch a command word's value field with a buffer DMA address at runtime.
  field_shift: left shift within the 32-bit value field (for sub-field patching like PPU DST_BASE_ADDR)."""
  word_index: int
  globals_slot: int
  addend: int
  shift: int
  mask: int
  field_shift: int = 0

@dataclass(frozen=True)
class RKPlan:
  """Output of plan_rk: classified kernel with slot mapping. One of six allowed dataclasses (§2.3)."""
  kind: str  # "dpu", "cmac", "ppu"
  sink: UOp
  out_slot: int
  in_slots: tuple[int, ...]
  input_scale: float = 1.0  # LUT input scale (for EXP2(MUL(x, CONST)) etc.)
  output_scale: float = 1.0  # LUT output scale (for MUL(LOG2(x), CONST) etc.)
  lut_op: Ops = Ops.EXP2  # which LUT builder to use
  fp32_inputs: tuple[int, ...] = ()   # slot numbers that are fp32 (need fp32→fp16 conversion before NPU)
  fp32_output: bool = False           # output is fp32 (need fp16→fp32 conversion after NPU)
  is_abs: bool = False                # abs(x) via BN negate + EW max (both operands same buffer)
  epilogue: str = "none"              # CMAC BS/BN epilogue: "none", "relu", "scale"
  epilogue_scale: float = 1.0         # scale factor for "scale" epilogue (BS MUL operand)

@dataclass(frozen=True)
class RKTask:
  """Task metadata carried in the image and INS nodes. One of six allowed dataclasses (§2.3)."""
  enable_mask: int
  int_mask: int
  op_idx: int
  kind: str
  layout: tuple
  out_slot: int
  is_copy: bool = False
  is_fill: bool = False
  const_val: float = 1.0
  fp32_inputs: tuple[int, ...] = ()   # slot numbers that are fp32 (need fp32→fp16 conversion before NPU)
  fp32_output: bool = False           # output slot is fp32 (need fp16→fp32 conversion after NPU)
  bool_inputs: tuple[int, ...] = ()   # boolean mask slots converted to fp16 before multi-task execution
  int32_inputs: tuple[int, ...] = ()  # int32 slots converted to fp16 before multi-task execution
  broadcast_inputs: tuple[int, ...] = () # scalar fp16 input slots expanded to the logical element count
  comparison_inputs: tuple[int, ...] = () # fp16 slots whose infinities are normalized before comparison
  int32_output: bool = False          # multi-task output converted from fp16 to int32
  uint8_output: bool = False          # multi-task output converted from fp16 to uint8
  bool_output: bool = False           # multi-task output converted from fp16 mask to bool
  trunc_output: bool = False          # multi-task fp16 output truncated through an integer cast round-trip

@dataclass(frozen=True)
class RKSubTask:
  """One task in a PC chain: cmds + task metadata + relocs. Used when a single kernel
  needs multiple DPU passes chained together (e.g. WHERE → SUB + MUL + ADD)."""
  cmds: tuple[int, ...]
  task: RKTask
  relocs: tuple[RKReloc, ...]

# ---- classifier ----
def _is_fp16_only(sink: UOp) -> bool:
  """All tensor-carrying nodes must be fp16 or fp32 (REDUCE may be fp32 for the NPU accumulator).
  fp32 inputs/outputs are handled via buffer-level conversion in RockchipProgram.__call__."""
  return all(not ((u.op is Ops.PARAM and u.dtype not in (dtypes.half, dtypes.float, dtypes.int)) or
                  (u.op is Ops.REDUCE and u.dtype not in (dtypes.half, dtypes.float)) or
                  (u.op is Ops.STORE and u.src[1].dtype not in (dtypes.half, dtypes.void, dtypes.float, dtypes.int))) for u in sink.toposort())

def _find_op(sink: UOp, op: Ops) -> UOp|None: return next((u for u in sink.toposort() if u.op is op), None)
def _reduce_node(sink: UOp) -> UOp|None: return _find_op(sink, Ops.REDUCE)
def _store_node(sink: UOp) -> UOp|None: return _find_op(sink, Ops.STORE)
def _unwrap(u: UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u
def _reduce_body(reduce: UOp) -> UOp: return _unwrap(reduce.src[0])
def _all_indexes(sink: UOp) -> list[UOp]: return [u for u in sink.toposort() if u.op is Ops.INDEX]
def _is_flat_contiguous(idx: UOp) -> bool:
  """Single flat RANGE or CONST(0) — rejects broadcast, transpose, strided access."""
  return idx.op is Ops.RANGE or (idx.op is Ops.CONST and int(idx.arg) == 0)

def _is_2d_index(idx: UOp, outer_kind: str = "LOOP", inner_kind: str = "LOOP", stride: int = -1) -> tuple[int, int, int]|None:
  """ADD(MUL(RANGE(outer_kind), CONST(stride)), RANGE(inner_kind)) → (outer, inner, stride) or None."""
  if idx.op is not Ops.ADD: return None
  ms, rs = idx.src
  if ms.op is not Ops.MUL or rs.op is not Ops.RANGE: return None
  mr, mc = ms.src
  if mr.op is not Ops.RANGE or mc.op is not Ops.CONST: return None
  if mr.arg[1].name != outer_kind or rs.arg[1].name != inner_kind or \
     (stride >= 0 and int(mc.arg) != stride) or (stride < 0 and mr.arg[0] != 0): return None
  return (int(mr.src[0].arg) if mr.src and mr.src[0].op is Ops.CONST else 0, int(mc.arg), int(mc.arg))

def _index_axes(idx:UOp) -> tuple[int,int]|None:
  if idx.op is not Ops.ADD: return None
  mul, inner = idx.src
  if mul.op is not Ops.MUL or inner.op is not Ops.RANGE: return None
  outer = next((x for x in mul.src if x.op is Ops.RANGE), None)
  return (outer.arg[0], inner.arg[0]) if outer is not None else None

def _affine_index(idx:UOp) -> tuple[dict[int,int], int]|None:
  if idx.op is Ops.RANGE: return ({idx.arg[0]: 1}, 0)
  if idx.op is Ops.CONST: return ({}, int(idx.arg))
  if idx.op is Ops.ADD:
    a, b = _affine_index(idx.src[0]), _affine_index(idx.src[1])
    if a is None or b is None: return None
    return ({k:a[0].get(k, 0)+b[0].get(k, 0) for k in a[0].keys()|b[0].keys()}, a[1]+b[1])
  if idx.op is Ops.MUL:
    c, x = (idx.src[0], idx.src[1]) if idx.src[0].op is Ops.CONST else (idx.src[1], idx.src[0])
    if c.op is not Ops.CONST or (aff := _affine_index(x)) is None: return None
    scale = int(c.arg)
    return ({k:v*scale for k,v in aff[0].items()}, aff[1]*scale)
  return None

def _try_sub(val: UOp) -> tuple[int, int]|None:
  """ADD(INDEX, MUL(INDEX, CONST(-1))) → (src_slot, ew_slot) for DPU SUB, or None."""
  if val.op is not Ops.ADD: return None
  for s, e in (val.src[0], val.src[1]), (val.src[1], val.src[0]):
    eu = _unwrap(e)
    if eu.op is Ops.MUL and _unwrap(s).op is Ops.INDEX:
      ma, mb = eu.src
      if mb.op is Ops.CONST and float(mb.arg) == -1.0 and _unwrap(ma).op is Ops.INDEX:
        return _unwrap(s).src[0].buf_uop.arg.slot, _unwrap(ma).src[0].buf_uop.arg.slot
  return None
_CONST_SLOT = 0xFFFF  # sentinel globals_slot for scalar constant buffer
_ZERO_SLOT = 0xFFFD  # sentinel globals_slot for zero-filled input buffer (fill)

def _try_scalar(val: UOp) -> tuple[int, float, bool]|None:
  """ADD/MUL/MAX/FDIV(INDEX, CONST(c)) → (index_slot, const_val, swap) for DPU scalar op, or None.
  swap=True when CONST is the first operand of FDIV (CONST/INDEX needs swapped DMA: CONST=input, INDEX=weight)."""
  if val.op not in _DPU_EW_CFGS: return None
  for i, (s, c) in enumerate(((val.src[0], val.src[1]), (val.src[1], val.src[0]))):
    if _unwrap(s).op is Ops.INDEX and c.op is Ops.CONST:
      # i=0: INDEX first (normal order). i=1: CONST first (swapped — only matters for non-commutative FDIV)
      swap = (i == 1 and val.op is Ops.FDIV)
      return _unwrap(s).src[0].buf_uop.arg.slot, float(c.arg), swap
  return None

def _try_reciprocal(val: UOp) -> tuple[int, float]|None:
  """RECIPROCAL(INDEX) → (index_slot, 1.0) for DPU scalar FDIV with swapped operands (1/x), or None."""
  if val.op is not Ops.RECIPROCAL: return None
  inner = _unwrap(val.src[0])
  if inner.op is not Ops.INDEX: return None
  return inner.src[0].buf_uop.arg.slot, 1.0

def _try_abs(val: UOp) -> int|None:
  """MUL(INDEX, WHERE(sign(INDEX))) → slot if both INDEX point to same buffer (abs pattern), or None.
  abs(x) = x * sign(x) where sign(x) = WHERE(CMPNE(x,0), WHERE(CMPLT(x,0),-1,1), 0).
  DPU handles this as MAX(x, -x) via BN mul-by-(-1) + EW MAX, both operands same buffer."""
  if val.op is not Ops.MUL: return None
  for a, b in (val.src[0], val.src[1]), (val.src[1], val.src[0]):
    au = _unwrap(a)
    if au.op is not Ops.INDEX: continue
    bu = _unwrap(b)
    if bu.op is not Ops.WHERE: continue
    # WHERE(CMPNE(x, 0), WHERE(CMPLT(x, 0), -1, 1), 0)
    cond, true_val, false_val = bu.src
    if cond.op is not Ops.CMPNE: continue
    ca, cb = _unwrap(cond.src[0]), cond.src[1]
    if ca.op is not Ops.INDEX or ca.src[0].buf_uop.arg.slot != au.src[0].buf_uop.arg.slot: continue
    if not (cb.op is Ops.CONST and float(cb.arg) == 0.0): continue
    if false_val.op is not Ops.CONST or float(false_val.arg) != 0.0: continue
    if true_val.op is not Ops.WHERE: continue
    tc, tt, tf = true_val.src
    if tc.op is not Ops.CMPLT: continue
    tca, tcb = _unwrap(tc.src[0]), tc.src[1]
    if tca.op is not Ops.INDEX or tca.src[0].buf_uop.arg.slot != au.src[0].buf_uop.arg.slot: continue
    if not (tcb.op is Ops.CONST and float(tcb.arg) == 0.0): continue
    if not (tt.op is Ops.CONST and float(tt.arg) == -1.0): continue
    if not (tf.op is Ops.CONST and float(tf.arg) == 1.0): continue
    return au.src[0].buf_uop.arg.slot
  return None

def _try_sign(val:UOp) -> int|None:
  """Recognize the sign WHERE tree by reusing the sign component of abs(x)."""
  val = _unwrap(val)
  if val.op is not Ops.WHERE or val.src[0].op is not Ops.CMPNE: return None
  source = _unwrap(val.src[0].src[0])
  if source.op is not Ops.INDEX: return None
  return _try_abs(UOp(Ops.MUL, val.dtype, (source, val)))

def _try_hardsigmoid(val:UOp) -> tuple[int,float,float]|None:
  """Recognize relu(alpha*x+beta)-relu(alpha*x+beta-1)."""
  val = _unwrap(val)
  if val.op is not Ops.ADD: return None

  def relu_inner(u:UOp) -> UOp|None:
    u = _unwrap(u)
    if u.op is not Ops.WHERE or _unwrap(u.src[0]).op is not Ops.CMPLT: return None
    lhs, rhs = (_unwrap(x) for x in _unwrap(u.src[0]).src)
    true, false = (_unwrap(x) for x in u.src[1:])
    if lhs.op is Ops.CONST and float(lhs.arg) == 0.0 and rhs is true and \
       false.op is Ops.CONST and float(false.arg) == 0.0: return true
    return None

  def negative_relu(u:UOp) -> UOp|None:
    u = _unwrap(u)
    if u.op is not Ops.MUL: return None
    for candidate, constant in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
      if constant.op is Ops.CONST and float(constant.arg) == -1.0: return relu_inner(candidate)
    return None

  for positive_term, negative_term in ((val.src[0], val.src[1]), (val.src[1], val.src[0])):
    positive, negative = relu_inner(positive_term), negative_relu(negative_term)
    if positive is None or negative is None or positive.op is not Ops.ADD or negative.op is not Ops.ADD: continue
    positive_shared, positive_const = positive.src
    negative_shared, negative_const = negative.src
    if positive_const.op is not Ops.CONST: positive_shared, positive_const = positive_const, positive_shared
    if negative_const.op is not Ops.CONST: negative_shared, negative_const = negative_const, negative_shared
    if positive_const.op is not Ops.CONST or negative_const.op is not Ops.CONST or positive_shared is not negative_shared: continue
    if not math.isclose(float(negative_const.arg), float(positive_const.arg)-1.0): continue
    shared = _unwrap(positive_shared)
    if shared.op is not Ops.MUL: continue
    for source, alpha in ((shared.src[0], shared.src[1]), (shared.src[1], shared.src[0])):
      source = _unwrap(source)
      if source.op is Ops.INDEX and alpha.op is Ops.CONST:
        return source.src[0].buf_uop.arg.slot, float(alpha.arg), float(positive_const.arg)
  return None

def _try_hardswish(val:UOp) -> UOp|None:
  """Recognize x*relu6(x+3)/6 and return its source INDEX."""
  val = _unwrap(val)
  indexes = [u for u in val.toposort() if u.op is Ops.INDEX]
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  def c(value:float) -> UOp: return UOp.const(dtypes.half, value)
  zero, plus = c(0.0), UOp(Ops.ADD, dtypes.half, (source, c(3.0)))
  relu_plus = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (zero, plus)), plus, zero))
  minus = UOp(Ops.ADD, dtypes.half, (source, c(-3.0)))
  relu_minus = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (zero, minus)), minus, zero))
  relu6 = UOp(Ops.ADD, dtypes.half, (relu_plus, UOp(Ops.MUL, dtypes.half, (relu_minus, c(-1.0)))))
  expected = UOp(Ops.MUL, dtypes.half, (UOp(Ops.MUL, dtypes.half, (source, relu6)), c(1/6)))
  return source if val is expected else None

def _try_tanh(val:UOp) -> UOp|None:
  """Recognize 2*sigmoid(2*x)-1 before or after the reciprocal-to-FDIV rewrite."""
  val = _unwrap(val)
  if val.op is not Ops.ADD: return None
  term, offset = val.src
  if offset.op is not Ops.CONST: term, offset = offset, term
  if offset.op is not Ops.CONST or float(offset.arg) != -1.0: return None
  term = _unwrap(term)
  if term.op is Ops.MUL:
    scale, reciprocal = term.src
    if scale.op is not Ops.CONST: scale, reciprocal = reciprocal, scale
    reciprocal = _unwrap(reciprocal)
    if scale.op is not Ops.CONST or float(scale.arg) != 2.0 or reciprocal.op is not Ops.RECIPROCAL: return None
    denominator = _unwrap(reciprocal.src[0])
  elif term.op is Ops.FDIV and term.src[0].op is Ops.CONST and float(term.src[0].arg) == 2.0:
    denominator = _unwrap(term.src[1])
  else: return None
  if denominator.op is not Ops.ADD: return None
  one, exponential = denominator.src
  if one.op is not Ops.CONST: one, exponential = exponential, one
  exponential = _unwrap(exponential)
  if one.op is not Ops.CONST or float(one.arg) != 1.0 or exponential.op is not Ops.EXP2: return None
  scaled = _unwrap(exponential.src[0])
  if scaled.op is not Ops.MUL: return None
  source = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.INDEX), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST), None)
  return source if source is not None and factor is not None and abs(factor + 2*math.log2(math.e)) < 1e-3 else None

def _try_quick_gelu(val:UOp) -> UOp|None:
  """Recognize x*sigmoid(1.702*x) before or after reciprocal-to-FDIV rewriting."""
  val = _unwrap(val)
  if val.op is Ops.MUL:
    source, reciprocal = val.src
    if _unwrap(source).op is not Ops.INDEX: source, reciprocal = reciprocal, source
    source, reciprocal = _unwrap(source), _unwrap(reciprocal)
    if source.op is not Ops.INDEX or reciprocal.op is not Ops.RECIPROCAL: return None
    denominator = _unwrap(reciprocal.src[0])
  elif val.op is Ops.FDIV:
    source, denominator = _unwrap(val.src[0]), _unwrap(val.src[1])
    if source.op is not Ops.INDEX: return None
  else: return None
  if denominator.op is not Ops.ADD: return None
  one, exponential = denominator.src
  if one.op is not Ops.CONST: one, exponential = exponential, one
  exponential = _unwrap(exponential)
  if one.op is not Ops.CONST or float(one.arg) != 1.0 or exponential.op is not Ops.EXP2: return None
  scaled = _unwrap(exponential.src[0])
  if scaled.op is not Ops.MUL: return None
  scaled_source = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.INDEX), None)
  factor = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST), None)
  return source if scaled_source is source and factor is not None and abs(factor + 1.702*math.log2(math.e)) < 1e-3 else None

# LUT ops: EXP2 uses DPU LUT table (513 entries × 2 tables) with BN_MUL scaling
_LUT_OPS = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL}
_LUT_SIGMOID = Ops.NOOP  # internal plan marker; tinygrad lowers sigmoid to reciprocal/add/exp2
_LUT_ROUNDOFF = Ops.CUSTOM  # internal plan marker for the RK3588 round-to-nearest-even LUT
_LUT_EXP_CORRECTION = Ops.NEG  # internal marker for the second, residual exp LUT
_LUT_HARDSWISH = Ops.POW  # internal marker for a fused hardswish LUT
_LUT_HARDSWISH_CORRECTION = Ops.CMOD
_LUT_SIZE = 513

def _build_exp2_lut(input_scale: float = 1.0) -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for EXP2 over x∈[-2,2] (scaled by input_scale).
  Returns (lut, bn_mul_operand, output_scale, index_scale, minus_exp).
  - index_scale: BN_MUL operand that maps x∈[-2,2] to int16 range [-16384,16384]
  - When input_scale != 1.0, index_scale is reduced so x*index_scale*input_scale stays in range.
  - output_scale: converts exp2(x*input_scale) to int16 range
  - minus_exp: OUT_CVT_SHIFT MINUS_EXP field (output is divided by 2^minus_exp)."""
  lut = [0] * _LUT_SIZE * 2
  # Map the original input x∈[-2,2] across the full table; the LUT values apply input_scale.
  index_scale = math.copysign(8192.0, input_scale)
  step = 32.0 / index_scale  # step in x domain
  # Determine output_scale and minus_exp based on max output value
  max_val = math.exp2(2.0 * abs(input_scale))  # either input endpoint can be the maximum for signed scaling
  if max_val <= 4.0:
    output_scale, minus_exp = 8192.0, 13
  elif max_val <= 8.0:
    output_scale, minus_exp = 4096.0, 12
  elif max_val <= 16.0:
    output_scale, minus_exp = 2048.0, 11
  else:
    output_scale, minus_exp = 1024.0, 10
  # Table 0 (LE): covers negative x. Entry i: x = -(512-i)*step (from -2.0 to 0)
  for i in range(_LUT_SIZE):
    x = -(512 - i) * step
    y = math.exp2(x * input_scale)
    lut[i] = max(-32768, min(32767, int(round(y * output_scale))))
  # Table 1 (LO): covers positive x. Entry i: x = i*step (from 0 to 2.0)
  for i in range(_LUT_SIZE):
    x = i * step
    y = math.exp2(x * input_scale)
    lut[_LUT_SIZE + i] = max(-32768, min(32767, int(round(y * output_scale))))
  if abs(input_scale + math.log2(math.e)) < 1e-3:
    # Preserve the curve through fp16 ADD/FDIV rounding boundaries in the staged SiLU path.
    lut[_LUT_SIZE + 500] -= 14
    lut[_LUT_SIZE + 502] += 14
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, minus_exp

def _build_exp_correction_lut() -> tuple[list[int], int, float, float, int]:
  """Signed Q12 residual for the low end of the Q12 EXP2 LUT used by exp(x).

  The first LUT must cover exp([-2, 2]) up to e**2, so its shared signed table
  is limited to Q12. The second LUT receives z=(x+1.75)*8, giving it four
  times the input resolution over x in [-2, -1.5]. It stores only failures
  outside TestOps' tolerance and saturates to zero correction outside that
  interval.
  """
  lut = [0] * _LUT_SIZE * 2
  base, _, output_scale, _, _ = _build_exp2_lut(math.log2(math.e))
  index_scale, correction_scale, correction_bias = 8192.0, 4096.0, 0.125
  step = 32.0 / index_scale
  for table, base_offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i) * step
      x = float(np.float16(z / 8.0 - 1.75))
      target = float(np.float16(math.exp(x)))
      position = (x + 2.0) * 256.0
      first_index = max(0, min(511, int(math.floor(position))))
      fraction = position - first_index
      first_raw = math.floor(base[first_index] + fraction * (base[first_index+1] - base[first_index]))
      approximate = float(np.float16(first_raw / output_scale))
      residual = target - approximate
      if i in (0, _LUT_SIZE-1) or abs(residual) <= 1e-6 + 1e-3 * abs(target): residual = 0.0
      # The RK3588 LUT path produces a spurious value for an exact zero table
      # entry. Keep the residual around a nonzero bias and subtract it later.
      lut[base_offset+i] = max(-32768, min(32767, int(round((residual+correction_bias) * correction_scale))))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, correction_scale, index_scale, 12

def _build_hardswish_lut() -> tuple[list[int], int, float, float, int]:
  """Q14 hardswish over [-2,2], with nonzero entries for the LUT zero erratum."""
  index_scale = 8192.0
  output_scale, minus_exp = 16384.0, 14
  step, lut = 32.0 / index_scale, [0] * _LUT_SIZE * 2
  for table, base in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i) * step
      y = x * min(6.0, max(0.0, x+3.0)) / 6.0
      raw = int(round(y * output_scale))
      lut[base+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_hardswish_correction_lut() -> tuple[list[int], int, float, float, int]:
  """Direct Q15 hardswish*16 near zero for tight relative accuracy."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0 / index_scale, [0] * _LUT_SIZE * 2
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i) * step
      x = float(np.float16(z / 16.0))
      target = x * min(6.0, max(0.0, x+3.0)) / 6.0
      raw = int(round(target*16.0*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_log2_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for LOG2 over x∈[0.25,4.0] → result∈[-2,2].
  Uses LUT_LE_START=-16384 (same as EXP2), index_scale=4096.
  LE table: underflow (x<0 → clip to log2(0+)=-2). LO table: x from 0 to 4.0.
  LO index = (bn_mul - 0) >> 5 = (x * 4096) >> 5. Entry i: x = i/128."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 4090.0  # avoid x=4.0 hitting LUT_LO_END=16384 exactly
  step = 32.0 / index_scale  # ≈ 1/127.8
  output_scale = 8192.0
  # LE table (table 0): underflow (bn_mul < 0, impossible for positive x → all clip to -2)
  for i in range(_LUT_SIZE):
    lut[i] = max(-32768, min(32767, int(round(-2.0 * output_scale))))
  # LO table (table 1): covers bn_mul from 0 to 16384 (x from 0 to ~4.01)
  for i in range(_LUT_SIZE):
    x = i * step  # i=0: x=0, i=128: x≈1.0, i=512: x≈4.01
    if x <= 0: y = -2.0  # clip log2(0) = -inf
    else: y = math.log2(x)
    y = max(-2.0, min(2.0, y))  # clip to [-2, 2]
    v = int(round(y * output_scale))
    # Avoid exact 0 in LUT — hardware produces wrong output for LUT result of 0
    if v == 0: v = 1
    lut[_LUT_SIZE + i] = max(-32768, min(32767, v))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 13

def _build_sin_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for SIN over x∈[-π,π] → result∈[-1,1].
  Uses LUT_LE_START=-16384, index_scale=16384/π≈5215.2.
  LE table: x∈[-π,0], LO table: x∈[0,π]."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 16384.0 / math.pi  # ≈5215.2, maps x∈[-π,π] to [-16384,16384]
  step = 32.0 / index_scale  # step in x per LUT entry
  output_scale = 16384.0  # maps [-1,1] to [-16384,16384] — doubled for precision
  # LE table (table 0): covers bn_mul from -16384 to 0, i.e., x from -π to 0
  for i in range(_LUT_SIZE):
    x = -i * step  # i=0: x=0, i=512: x=-π
    y = math.sin(x)
    v = int(round(y * output_scale))
    if v == 0: v = 1  # avoid exact 0 — hardware produces garbage (8.0) for LUT output of 0
    lut[i] = max(-32768, min(32767, v))
  # LO table (table 1): covers bn_mul from 0 to 16384, i.e., x from 0 to π
  for i in range(_LUT_SIZE):
    x = i * step  # i=0: x=0, i=512: x=π
    y = math.sin(x)
    v = int(round(y * output_scale))
    if v == 0: v = 1  # avoid exact 0 — hardware produces garbage (8.0) for LUT output of 0
    lut[_LUT_SIZE + i] = max(-32768, min(32767, v))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 14

def _build_sqrt_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for SQRT over x∈[0,4] → result∈[0,2].
  Uses LUT_LE_START=-16384 (LE handles underflow), index_scale=4090.
  LE table: underflow (x<0 → clip to 0). LO table: x from 0 to ~4.01.
  output_scale=16384 (doubled for precision); max entry 2*16384=32768 clips to 32767."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 4090.0  # avoid x=4.0 hitting LUT_LO_END=16384 exactly
  step = 32.0 / index_scale  # ≈ 1/127.8
  output_scale = 16384.0  # maps [0,2] to [0,32768] — doubled for precision (max clips to 32767)
  # LE table (table 0): underflow (x<0 → clip to 0, but use 1 to avoid exact 0 hardware bug)
  for i in range(_LUT_SIZE):
    lut[i] = 1  # sqrt(negative) ≈ 0 (clip, avoid exact 0)
  # LO table (table 1): covers bn_mul from 0 to 16384 (x from 0 to ~4.01)
  for i in range(_LUT_SIZE):
    x = i * step  # i=0: x=0, i=512: x≈4.01
    y = math.sqrt(x) if x > 0 else 0.0
    y = min(2.0, y)  # clip to [0, 2]
    v = int(round(y * output_scale))
    if v == 0: v = 1  # avoid exact 0
    lut[_LUT_SIZE + i] = max(-32768, min(32767, v))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 14

def _build_rsqrt_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for RSQRT over x∈[0.0625,4] → result∈[0.5,4].
  Uses LUT_LE_START=-16384, index_scale=4090.
  LE table: underflow (x<0 → clip to 4). LO table: x from 0 to ~4.01."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 4090.0
  step = 32.0 / index_scale
  output_scale = 8192.0  # maps [0,4] to [0,32768] — but we clip to [0.5,4] → [4096,32768]
  # LE table (table 0): underflow (x<0 → clip to rsqrt(0+) = inf, but clip to 4)
  for i in range(_LUT_SIZE):
    lut[i] = max(-32768, min(32767, int(round(4.0 * output_scale))))
  # LO table (table 1): covers bn_mul from 0 to 16384 (x from 0 to ~4.01)
  for i in range(_LUT_SIZE):
    x = i * step
    if x <= 0: y = 4.0  # clip rsqrt(0) = inf to 4
    else: y = 1.0 / math.sqrt(x)
    y = max(0.5, min(4.0, y))  # clip to [0.5, 4]
    v = int(round(y * output_scale))
    if v == 0: v = 1
    lut[_LUT_SIZE + i] = max(-32768, min(32767, v))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 13

def _build_sigmoid_lut() -> tuple[list[int], int, float, float, int]:
  """Build sigmoid directly over [-8, 8], avoiding EXP2+ADD+FDIV rounding."""
  index_scale, output_scale, minus_exp = 2048.0, 32768.0, 15
  step = 32.0 / index_scale
  lut = [0] * _LUT_SIZE * 2
  for i in range(_LUT_SIZE):
    x = -(512-i) * step
    lut[i] = max(1, min(32767, int(round(output_scale / (1.0 + math.exp(-x))))))
  for i in range(_LUT_SIZE):
    x = i * step
    lut[_LUT_SIZE+i] = max(1, min(32767, int(round(output_scale / (1.0 + math.exp(-x))))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _try_sigmoid(val:UOp) -> int|None:
  """RECIPROCAL(1 + EXP2(INDEX * -log2(e))) → input slot."""
  if val.op is not Ops.RECIPROCAL or (add := _unwrap(val.src[0])).op is not Ops.ADD: return None
  one, exp = add.src
  if exp.op is Ops.CONST: one, exp = exp, one
  if one.op is not Ops.CONST or float(one.arg) != 1.0 or (exp := _unwrap(exp)).op is not Ops.EXP2: return None
  mul = _unwrap(exp.src[0])
  if mul.op is not Ops.MUL: return None
  idx = next((_unwrap(x) for x in mul.src if _unwrap(x).op is Ops.INDEX), None)
  scale = next((float(x.arg) for x in mul.src if x.op is Ops.CONST), None)
  return idx.src[0].buf_uop.arg.slot if idx is not None and scale is not None and abs(scale + math.log2(math.e)) < 1e-3 else None

def _try_round(val:UOp) -> UOp|None:
  """Recognize tinygrad's exact round-to-nearest-even expansion and return its source INDEX."""
  val = _unwrap(val)
  indexes = [u for u in val.toposort() if u.op is Ops.INDEX]
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  def c(value:float) -> UOp: return UOp.const(dtypes.half, value)
  truncated = UOp(Ops.TRUNC, dtypes.half, (source,))
  half_truncated = UOp(Ops.MUL, dtypes.half, (truncated, c(0.5)))
  positive = UOp(Ops.CMPLT, dtypes.bool, (c(0.0), source))
  even = UOp(Ops.CMPNE, dtypes.bool, (
    UOp(Ops.CMPNE, dtypes.bool, (UOp(Ops.TRUNC, dtypes.half, (half_truncated,)), half_truncated)),
    UOp.const(dtypes.bool, True)))
  condition = UOp(Ops.CMPNE, dtypes.bool, (positive, even))
  plus_half = UOp(Ops.ADD, dtypes.half, (source, c(0.5)))
  plus_trunc = UOp(Ops.TRUNC, dtypes.half, (plus_half,))
  floor_plus = UOp(Ops.WHERE, dtypes.half, (
    UOp(Ops.CMPLT, dtypes.bool, (plus_half, plus_trunc)),
    UOp(Ops.ADD, dtypes.half, (plus_trunc, c(-1.0))), plus_trunc))
  minus_half = UOp(Ops.ADD, dtypes.half, (source, c(-0.5)))
  minus_trunc = UOp(Ops.TRUNC, dtypes.half, (minus_half,))
  ceil_minus = UOp(Ops.WHERE, dtypes.half, (
    UOp(Ops.CMPLT, dtypes.bool, (minus_trunc, minus_half)),
    UOp(Ops.ADD, dtypes.half, (minus_trunc, c(1.0))), minus_trunc))
  expected = UOp(Ops.WHERE, dtypes.half, (condition, floor_plus, ceil_minus))
  return source if val is expected else None

def _try_lut(val: UOp) -> tuple[int, float, float, Ops]|None:
  """Recognize LUT patterns and return (index_slot, input_scale, output_scale, lut_op).
  Patterns:
  - EXP2/LOG2/SIN/SQRT(INDEX) → (slot, 1.0, 1.0, op)
  - EXP2(MUL(INDEX, CONST)) → (slot, CONST, 1.0, EXP2)  [exp(x) = exp2(x*log2(e))]
  - MUL(LOG2(INDEX), CONST) → (slot, 1.0, CONST, LOG2)  [log(x) = log2(x)*ln(2)]
  - RECIPROCAL(SQRT(INDEX)) → (slot, 1.0, 1.0, RECIPROCAL)  [rsqrt(x)]
  Returns None if not a LUT pattern."""
  input_scale, output_scale = 1.0, 1.0
  if val.op is Ops.CUSTOM and val.arg == "rk_roundoff" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ROUNDOFF)
  if val.op is Ops.CUSTOM and val.arg == "rk_hardswish" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_HARDSWISH)
  if val.op is Ops.CUSTOM and val.arg == "rk_hardswish_correction" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_HARDSWISH_CORRECTION)
  if (sigmoid_slot := _try_sigmoid(val)) is not None: return (sigmoid_slot, 1.0, 1.0, _LUT_SIGMOID)
  # MUL(LUT_OP(INDEX), CONST) → output-scaled LUT (e.g., log(x) = log2(x) * ln(2))
  if val.op is Ops.MUL:
    a, b = val.src
    lut_op = None
    if a.op in _LUT_OPS and b.op is Ops.CONST: lut_op, inner, output_scale = a.op, _unwrap(a.src[0]), float(b.arg)
    elif b.op in _LUT_OPS and a.op is Ops.CONST: lut_op, inner, output_scale = b.op, _unwrap(b.src[0]), float(a.arg)
    if lut_op is None: return None
    # RECIPROCAL(SQRT(INDEX)) → RSQRT LUT
    if lut_op is Ops.RECIPROCAL:
      if inner.op is not Ops.SQRT: return None
      inner = _unwrap(inner.src[0])
    if inner.op is not Ops.INDEX: return None
    return (inner.src[0].buf_uop.arg.slot, input_scale, output_scale, lut_op)
  if val.op not in _LUT_OPS: return None
  inner = _unwrap(val.src[0])
  # RECIPROCAL(SQRT(INDEX)) → RSQRT LUT
  if val.op is Ops.RECIPROCAL:
    if inner.op is not Ops.SQRT: return None
    inner = _unwrap(inner.src[0])
    # Check for MUL(INDEX, CONST) scaling
    if inner.op is Ops.MUL:
      a, b = inner.src
      if a.op is Ops.CONST and _unwrap(b).op is Ops.INDEX:
        input_scale, inner = float(a.arg), _unwrap(b)
      elif b.op is Ops.CONST and _unwrap(a).op is Ops.INDEX:
        input_scale, inner = float(b.arg), _unwrap(a)
    if inner.op is not Ops.INDEX: return None
    return (inner.src[0].buf_uop.arg.slot, input_scale, output_scale, val.op)
  # Check for MUL(INDEX, CONST) scaling (e.g., exp(x) = exp2(x * log2(e)))
  if inner.op is Ops.MUL:
    a, b = inner.src
    if a.op is Ops.CONST and _unwrap(b).op is Ops.INDEX:
      input_scale, inner = float(a.arg), _unwrap(b)
    elif b.op is Ops.CONST and _unwrap(a).op is Ops.INDEX:
      input_scale, inner = float(b.arg), _unwrap(a)
  if inner.op is not Ops.INDEX: return None
  return (inner.src[0].buf_uop.arg.slot, input_scale, output_scale, val.op)

def _try_where_max(val: UOp) -> UOp|None:
  """WHERE(CMPLT(a,b), b, a) or WHERE(CMPLT(b,a), a, b) → synthetic MAX(a,b), or None."""
  if val.op is not Ops.WHERE or val.src[0].op is not Ops.CMPLT: return None
  cond, x, y = val.src
  p, q = cond.src
  if (x is q and y is p) or (cond.src[0] is q and cond.src[1] is p and x is p and y is q):
    return UOp(Ops.MAX, dtypes.half, (p, q))
  return None

def _ppu_channel_count(idx: UOp) -> int|None:
  """Channel count from input index: ADD(MUL(RANGE(REDUCE),CONST(ch)),RANGE(LOOP)) or simplified forms."""
  if idx.op is Ops.RANGE and idx.arg[1].name == "REDUCE": return 1
  if idx.op is not Ops.ADD: return None
  a, b = idx.src
  if a.op is Ops.MUL:
    mr, mc = a.src
    if mr.op is Ops.RANGE and mc.op is Ops.CONST and b.op is Ops.RANGE and mr.arg[1].name == "REDUCE" and b.arg[1].name == "LOOP": return int(mc.arg)
  if a.op is Ops.RANGE and b.op is Ops.RANGE and a.arg[1].name == "REDUCE" and b.arg[1].name == "LOOP": return 1
  return None

_PPU_BAD_SPLITS = frozenset({(3, 6), (6, 3), (12, 12)})
def _ppu_split_k(K: int) -> tuple[int, int]|None:
  """Split K into (in_h, in_w) for PPU global pooling. Kernel fields are 4 bits
  (max 16), so both in_h and in_w must be 1-16. Avoids in_h=9/in_w=9 and known-bad
  combinations (hardware bug on RK3588 PPU). Falls back to in_h=1 for primes ≤ 16."""
  if K < 4: return None
  for in_h in range(2, min(K, 16) + 1):
    if K % in_h: continue
    in_w = K // in_h
    if 2 <= in_w <= 16 and in_h != 9 and in_w != 9 and (in_h, in_w) not in _PPU_BAD_SPLITS: return (in_h, in_w)
  if K <= 16: return (1, K)  # in_h=1 fallback for primes/odd K ≤ 16
  return None

def _is_1d_index(idx: UOp, kind: str) -> bool:
  """Check idx is RANGE(axis_kind) — 1D vector access."""
  return idx.op is Ops.RANGE and idx.arg[1].name == kind

def _is_cmac_matmul_layout(sink: UOp, reduce: UOp) -> bool:
  """Validate INDEX patterns match row-major matmul: out=(i,j), A=(i,k), B=(k,j).
  Also accepts transposed (1x1 conv): A=(k,j), B=(i,k) — A and B swapped.
  GEMV: 1D output, one vector input (1D RANGE over REDUCE), one matrix input (2D)."""
  body = _reduce_body(reduce)
  if body.op is not Ops.MUL: return False
  a_idx, b_idx = _unwrap(body.src[0]), _unwrap(body.src[1])
  if a_idx.op is not Ops.INDEX or b_idx.op is not Ops.INDEX: return False
  out_shape = _shape_of_store(sink)
  K = _reduce_extent(reduce)
  if K < 0: return False
  store = _store_node(sink)
  if store is None or store.src[0].op is not Ops.INDEX: return False
  if len(out_shape) == 2:  # GEMM
    N = int(out_shape[1])
    if not _is_2d_index(store.src[0].src[1], "LOOP", "LOOP", N): return False
    return (_is_2d_index(a_idx.src[1], "LOOP", "REDUCE", K) is not None and _is_2d_index(b_idx.src[1], "REDUCE", "LOOP", N) is not None) or \
           (_is_2d_index(a_idx.src[1], "REDUCE", "LOOP", N) is not None and _is_2d_index(b_idx.src[1], "LOOP", "REDUCE", K) is not None)
  if len(out_shape) == 1:  # GEMV: (K,)@(K,D)→(D,) or (D,K)@(K,)→(D,)
    D = int(out_shape[0])
    if not _is_1d_index(store.src[0].src[1], "LOOP"): return False
    return (_is_1d_index(a_idx.src[1], "REDUCE") and _is_2d_index(b_idx.src[1], "REDUCE", "LOOP", D) is not None) or \
           (_is_2d_index(a_idx.src[1], "LOOP", "REDUCE", K) is not None and _is_1d_index(b_idx.src[1], "REDUCE"))
  return False

def _try_sum(sink: UOp, reduce: UOp) -> tuple[int, int, int, int, float]|None:
  """REDUCE(ADD, INDEX) or REDUCE(ADD, MUL(INDEX, CONST(c))) → (input_slot, M, N, K, const_val) for CMAC, or None.
  A-pattern (axis=1): a@ones(K,1), M=out[0], N=1, ones=B.
  B-pattern (axis=0): ones(1,K)@a, M=1, N=out[0], ones=A.
  C-pattern (full):   RANGE(REDUCE) only, M=1, N=1, ones=A or B.
  Scaled sum (MUL(INDEX, CONST(c))): same patterns, const_val=c instead of 1.0.
  Mean (post-reduce scalar MUL) is rejected — no host-side scaling in PR1."""
  body = _reduce_body(reduce)
  cv = 1.0
  if body.op is Ops.MUL:
    ws = [_unwrap(s) for s in body.src]
    if not any(w.op is Ops.CONST for w in ws) or not any(w.op is Ops.INDEX for w in ws): return None
    cv = float(next(w.arg for w in ws if w.op is Ops.CONST))
    body = next(w for w in ws if w.op is Ops.INDEX)
  if body.op is not Ops.INDEX: return None
  out_shape = _shape_of_store(sink)
  if len(out_shape) != 1: return None  # only 1D output
  K = _reduce_extent(reduce)
  if K < 0: return None
  store = _store_node(sink)
  if store is None or store.src[0].op is not Ops.INDEX or not _is_flat_contiguous(store.src[0].src[1]): return None
  input_slot = _unwrap(body).src[0].buf_uop.arg.slot
  # Accept post-reduce epilogue (relu, scale) — the classifier validates it separately
  # and the BS pipeline applies it. _try_sum only needs to verify the sum pattern.
  M = int(out_shape[0])
  if _is_2d_index(body.src[1], "LOOP", "REDUCE", K): return (input_slot, M, 1, K, cv)  # A-pattern
  if _is_2d_index(body.src[1], "REDUCE", "LOOP", M): return (input_slot, 1, M, K, cv)  # B-pattern
  if body.src[1].op is Ops.RANGE and body.src[1].arg[1].name == "REDUCE": return (input_slot, 1, 1, K, cv)  # C-pattern
  return None

def _check_dpu_layout(sink: UOp, allow_2d: bool, require_uniform: bool) -> str|None:
  """Check all INDEX nodes in sink have compatible layouts. Returns reject reason or None."""
  idx_nodes = _all_indexes(sink)
  has_2d = allow_2d and any(_is_2d_index(n.src[1]) is not None for n in idx_nodes)
  for n in idx_nodes:
    if not (_is_flat_contiguous(n.src[1]) or (allow_2d and _is_2d_index(n.src[1]) is not None)) or \
       (require_uniform and has_2d and _is_2d_index(n.src[1]) is None):
      return f"RKPLAN_REJECT:unsupported_layout:{n.src[1].op}"
  return None

def _try_cmac_epilogue(sink: UOp, reduce: UOp) -> tuple[str, float]|None:
  """Detect BS-fusable epilogue after CMAC reduce. Returns (epilogue_type, scale) or None.
  Supported: "relu" (WHERE(CMPLT(0,x), x, 0)), "scale" (MUL(x, const) or MUL(const, x)).
  The epilogue sits between the reduce and the store: store.src[1] = epilogue(reduce)."""
  store = _store_node(sink)
  if store is None: return None
  sv = _unwrap(store.src[1])
  if sv is reduce: return ("none", 1.0)
  # Scale: MUL(CAST(reduce), CONST(c)) or MUL(CONST(c), CAST(reduce))
  if sv.op is Ops.MUL and len(sv.src) == 2:
    a, b = _unwrap(sv.src[0]), _unwrap(sv.src[1])
    if a is reduce and b.op is Ops.CONST: return ("scale", float(b.arg))
    if b is reduce and a.op is Ops.CONST: return ("scale", float(a.arg))
  # ReLU: WHERE(CMPLT(CONST(0), CAST(reduce)), CAST(reduce), CONST(0))
  if sv.op is Ops.WHERE and len(sv.src) == 3:
    cond, t, f = sv.src
    cond_u, t_u, f_u = _unwrap(cond), _unwrap(t), _unwrap(f)
    # WHERE(CMPLT(0, x), x, 0) = relu(x)
    if cond_u.op is Ops.CMPLT and t_u is reduce and f_u.op is Ops.CONST and float(f_u.arg) == 0.0:
      return ("relu", 1.0)
    # WHERE(CMPLT(x, 0), 0, x) = relu(x) (alternative form)
    if cond_u.op is Ops.CMPLT and f_u is reduce and t_u.op is Ops.CONST and float(t_u.arg) == 0.0:
      return ("relu", 1.0)
  return None

def plan_rk(sink: UOp) -> RKPlan|str:
  """Classify a post-early_simplify SINK. Returns RKPlan on success, 'RKPLAN_REJECT:...' str on reject."""
  if not _is_fp16_only(sink): return "RKPLAN_REJECT:unsupported_dtype"
  reduce = _reduce_node(sink)
  lut_result = None  # set in DPU path only
  abs_slot = None  # set in DPU path only
  epilogue, epilogue_scale = "none", 1.0  # set in CMAC path only
  # R3 DPU: no REDUCE, single STORE with binary EW op (ADD/SUB/MUL/MAX), scalar operand, or DMA copy
  if reduce is None:
    store = _store_node(sink)
    if store is None: return "RKPLAN_REJECT:no_add_mul_reduction"
    val = store.src[1]
    where_max = _try_where_max(val)
    if where_max is not None: val = where_max
    val = _unwrap(val)  # strip no-op CASTs (half→half) so EW ops are recognized
    sub_slots, scalar = _try_sub(val), _try_scalar(val)
    reciprocal = _try_reciprocal(val)
    lut_result = _try_lut(val)
    abs_slot = _try_abs(val) if not (lut_result is not None or sub_slots is not None or scalar is not None or reciprocal is not None) else None
    # PR1 DPU contract: binary EW with two INDEX operands, scalar operand, DMA copy, or constant fill.
    # Broadcast and mean are rejected — no host-side tensor arithmetic.
    if lut_result is not None: kind, a2d, ru = "dpu_lut", False, False
    elif sub_slots is not None: kind, a2d, ru = "dpu", False, False
    elif reciprocal is not None: kind, a2d, ru = "dpu", True, True
    elif scalar is not None or (val.op in _DPU_EW_CFGS and all(_unwrap(s).op is Ops.INDEX for s in val.src)): kind, a2d, ru = "dpu", True, True
    elif abs_slot is not None: kind, a2d, ru = "dpu", True, True
    elif _unwrap(val).op is Ops.INDEX: kind, a2d, ru = "dpu", True, False
    elif _unwrap(val).op is Ops.CONST: kind, a2d, ru = "dpu", True, False
    else: return f"RKPLAN_REJECT:unsupported_op:{val.op if val.op not in _DPU_EW_CFGS else 'non_index_operand'}"
    if val.op is not Ops.INDEX and (r := _check_dpu_layout(sink, a2d, ru)): return r
  # R1 CMAC: REDUCE(ADD, MUL(INDEX, INDEX)) or REDUCE(ADD, INDEX) [sum via ones]
  elif reduce.arg[0] is Ops.ADD:
    body = _reduce_body(reduce)
    # Check for BS-fusable epilogue (relu, scale) after the reduce.
    # If the epilogue is recognized, we fuse it into the DPU BS pipeline.
    # If the store value is not the reduce and not a recognized epilogue, reject.
    store = _store_node(sink)
    if store is not None:
      sv = _unwrap(store.src[1])
      if sv is not reduce and not (sv.op is Ops.CAST and _unwrap(sv.src[0]) is reduce):
        epi = _try_cmac_epilogue(sink, reduce)
        if epi is None:
          return "RKPLAN_REJECT:unsupported_op:fused_epilogue"
        epilogue, epilogue_scale = epi
    if body.op is Ops.MUL and all(s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX) for s in body.src):
      if not _is_cmac_matmul_layout(sink, reduce): return "RKPLAN_REJECT:unsupported_layout"
      kind = "cmac"
    elif body.op is Ops.INDEX:
      # For epilogue fusion, _try_sum may reject because store.src[1] != reduce.
      # In that case, verify the sum pattern manually and accept with epilogue.
      if _try_sum(sink, reduce) is not None: kind = "cmac"
      elif epilogue != "none": kind = "cmac"  # sum with fusable epilogue
      else: return "RKPLAN_REJECT:no_add_mul_reduction"
    elif body.op is Ops.MUL and _try_sum(sink, reduce) is not None:
      kind = "cmac"  # scaled sum: REDUCE(ADD, MUL(INDEX, CONST(c)))
    else:
      return f"RKPLAN_REJECT:unsupported_op:{body.op}"
  # R4 PPU: REDUCE(MAX, INDEX) — global max pool over (H,W,C) → (C,)
  elif reduce.arg[0] is Ops.MAX:
    body = _reduce_body(reduce)
    if body.op in (Ops.INDEX, Ops.CAST):
      out_shape = _shape_of_store(sink)
      K = _reduce_extent(reduce)
      store = _store_node(sink)
      channels = int(out_shape[0]) if out_shape else 0
      if len(out_shape) != 1 or K < 0: return f"RKPLAN_REJECT:unsupported_layout:{out_shape}:{K}"
      if store is None or store.src[0].op is not Ops.INDEX or not _is_flat_contiguous(store.src[0].src[1]):
        return "RKPLAN_REJECT:unsupported_layout"
      if not all(_ppu_channel_count(n.src[1]) == channels for n in _all_indexes(sink) if n is not store.src[0]):
        return "RKPLAN_REJECT:unsupported_layout"
      if channels > 8 or _ppu_split_k(K) is None: return f"RKPLAN_REJECT:unsupported_layout:{out_shape}:{K}"
      kind = "ppu"
    else:
      return f"RKPLAN_REJECT:unsupported_op:{body.op}"
  else: return f"RKPLAN_REJECT:unsupported_op:{reduce.arg[0]}"
  prog_info = ProgramInfo.from_sink(sink)
  out_slots = list(prog_info.outs)
  if len(out_slots) != 1: return f"RKPLAN_REJECT:unsupported_layout:{len(out_slots)}-outputs"
  in_slots = tuple(s for s in prog_info.globals if s != out_slots[0])
  input_scale = lut_result[1] if lut_result is not None else 1.0
  output_scale = lut_result[2] if lut_result is not None else 1.0
  lut_op = lut_result[3] if lut_result is not None else Ops.EXP2
  # Detect fp32 inputs/output for buffer-level conversion (NPU processes fp16 internally).
  # fp32 is only supported for DPU/DPU_LUT (buffer-level fp32↔fp16 conversion in __call__).
  # CMAC and PPU require host-side data transforms (pad/swizzle) that assume fp16 — reject fp32 for them.
  fp32_param_slots = {u.arg.slot for u in sink.toposort() if u.op is Ops.PARAM and u.dtype is dtypes.float}
  int_param_slots = {u.arg.slot for u in sink.toposort() if u.op is Ops.PARAM and u.dtype is dtypes.int}
  is_copy = reduce is None and store is not None and _unwrap(store.src[1]).op is Ops.INDEX
  if int_param_slots and not is_copy: return "RKPLAN_REJECT:unsupported_dtype"
  wide_param_slots = fp32_param_slots | int_param_slots
  fp32_inputs = tuple(s for s in in_slots if s in wide_param_slots)  # four-byte input slots
  fp32_output = out_slots[0] in wide_param_slots
  if (fp32_inputs or fp32_output) and kind not in ("dpu", "dpu_lut"):
    return f"RKPLAN_REJECT:unsupported_dtype:fp32_{kind}"
  return RKPlan(kind, sink, out_slots[0], in_slots, input_scale=input_scale, output_scale=output_scale, lut_op=lut_op,
                fp32_inputs=fp32_inputs, fp32_output=fp32_output, is_abs=abs_slot is not None,
                epilogue=epilogue, epilogue_scale=epilogue_scale)

# ---- geometry extraction ----
def _loop_extents(sink: UOp) -> list[int]:
  """Extents of all LOOP RANGE nodes in topological order (-1 if non-const)."""
  return [int(u.src[0].arg) if u.src[0].op is Ops.CONST else -1 for u in sink.toposort() if u.op is Ops.RANGE and u.arg[1].name == "LOOP"]

def _shape_of_store(sink: UOp) -> tuple[int, ...]:
  """Extract the output shape from the LOOP RANGE extents."""
  return tuple(_loop_extents(sink)) or (1,)

def _reduce_extent(reduce: UOp) -> int:
  """Get the reduction axis extent (K for matmul, N for max)."""
  rng = reduce.src[1]
  return int(rng.src[0].arg) if rng.op is Ops.RANGE and rng.src[0].op is Ops.CONST else -1

# ---- emitter ----
def emitter_emit(cmds, target, reg, value): cmds.append(RKCmd(target, reg, value))
def emitter_reloc(cmds, relocs, globals_slot, addend=0, shift=0, mask=0xFFFFFFFF, field_shift=0):
  relocs.append(RKReloc(len(cmds)-1, globals_slot, addend, shift, mask, field_shift))
def emitter_pc_op_en(cmds, reserved_0): emitter_emit(cmds, _T_PC, rk.REG_PC_OPERATION_ENABLE, (reserved_0 << 1))

# DPU EW_CFG values for each op (from ref/rk3588/examples/elementwise.py)
# Base: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
_EW_BASE = 0x108002c0
# FDIV: ew_alu_algo=3, EW_OP_CVT_BYPASS=1 (bit 8) — output is fp16, needs OUT_CVT_SCALE=1
_DPU_EW_CFGS = {Ops.ADD: _EW_BASE | (2 << 16), Ops.SUB: _EW_BASE | (4 << 16),
                 Ops.MUL: _EW_BASE | (1 << 2) | (1 << 8), Ops.MAX: _EW_BASE,
                 Ops.FDIV: _EW_BASE | (3 << 16) | (1 << 8)}

def _where_arg(u: UOp) -> tuple[int, int]|None:
  u = _unwrap(u)
  if u.op is Ops.INDEX: return u.src[0].buf_uop.arg.slot, 0
  if u.op is Ops.CONST: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(u.arg)))[0]
  return None

def _emit_where_stage(total:int, out_slot:int, a:tuple[int,int], b:tuple[int,int], op:Ops, compare=False,
                      bool_inputs:tuple[int,...]=(), int32_inputs:tuple[int,...]=(), broadcast_inputs:tuple[int,...]=(), int32_output=False,
                      uint8_output=False, bool_output=False, trunc_output=False, comparison_inputs:tuple[int,...]=(),
                      fp32_inputs:tuple[int,...]=(), fp32_output=False) -> RKSubTask:
  """Fully-specified DPU stage used by the hardware-proven eight-pass WHERE lowering."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  def e(t, r, v): emitter_emit(cmds, t, r, v)
  e(_T_DPU, rk.REG_DPU_S_POINTER, 0xe)
  e(_T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  e(_T_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, (total+7)//8-1)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007)
  e(_T_DPU, rk.REG_DPU_BS_CFG, 0x53)
  e(_T_DPU, rk.REG_DPU_BN_CFG, 0x53)
  e(_T_DPU, rk.REG_DPU_BS_ALU_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BS_MUL_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BS_OW_CFG, 2)
  e(_T_DPU, rk.REG_DPU_WDMA_SIZE_0, 7)
  e(_T_DPU, rk.REG_DPU_WDMA_SIZE_1, (total+7)//8-1)
  e(_T_DPU, rk.REG_DPU_BN_MUL_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0)
  if compare:
    e(_T_DPU, rk.REG_DPU_BS_CFG, 0x40040)
    e(_T_DPU, rk.REG_DPU_BS_ALU_CFG, 0x33800000)
    e(_T_DPU, rk.REG_DPU_BS_MUL_CFG, 0x40000000)
    e(_T_DPU, rk.REG_DPU_BN_CFG, 0x40082)
    e(_T_DPU, rk.REG_DPU_BN_MUL_CFG, 0x7c000000)
    e(_T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000)
  e(_T_DPU, rk.REG_DPU_EW_CFG, _EW_BASE | 1 if compare else _DPU_EW_CFGS[op])
  e(_T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 1 if op is Ops.FDIV else 0x10001)
  e(_T_DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0)
  e(_T_DPU, rk.REG_DPU_SURFACE_ADD, 0x40)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, (total+7)//8-1)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008)
  for target, reg, arg in ((_T_DPU, rk.REG_DPU_DST_BASE_ADDR, (out_slot, 0)),
                           (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, a),
                           (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, b)):
    e(target, reg, 0)
    emitter_reloc(cmds, relocs, arg[0], arg[1])
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if op is Ops.FDIV else 0x17849)
  emitter_pc_op_en(cmds, 12)
  task = RKTask(0x18, 0x300, 4, "dpu", (total,), out_slot, bool_inputs=bool_inputs, int32_inputs=int32_inputs,
                broadcast_inputs=broadcast_inputs, int32_output=int32_output, uint8_output=uint8_output,
                bool_output=bool_output, trunc_output=trunc_output, comparison_inputs=comparison_inputs,
                fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  return RKSubTask(tuple(c.pack() for c in cmds), task, tuple(relocs))

def _emit_trunc_stage(total:int, out_slot:int, source:tuple[int,int]) -> RKSubTask:
  """Run an identity DPU stage, then apply the fp16→int32→fp16 cast boundary."""
  return _emit_where_stage(total, out_slot, source, (_ZERO_SLOT, 0), Ops.ADD, trunc_output=True)

def _try_cast_subtasks(sink:UOp) -> tuple[RKSubTask]|None:
  """Run a DPU identity stage in fp16, with buffer-level conversion at its edges."""
  store = _store_node(sink)
  if store is None: return None
  if store.src[1].op is Ops.CAST: output_dtype, source = store.src[1].dtype, _unwrap(store.src[1].src[0])
  elif store.src[1].op is Ops.INDEX and store.src[1].dtype is dtypes.bool: output_dtype, source = dtypes.bool, store.src[1]
  else: return None
  if source.op is not Ops.INDEX or output_dtype not in (dtypes.half, dtypes.float, dtypes.int, dtypes.bool, dtypes.uint8): return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  input_slot, out_slot = source.src[0].buf_uop.arg.slot, info.outs[0]
  return (_emit_where_stage(total, out_slot, (input_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                            bool_inputs=((input_slot,) if source.dtype is dtypes.bool else ()),
                            int32_inputs=((input_slot,) if source.dtype is dtypes.int else ()),
                            fp32_inputs=((input_slot,) if source.dtype is dtypes.float else ()),
                            fp32_output=output_dtype is dtypes.float, int32_output=output_dtype is dtypes.int,
                            uint8_output=output_dtype is dtypes.uint8, bool_output=output_dtype is dtypes.bool),)

def _try_typed_fill_subtasks(sink:UOp) -> tuple[RKSubTask]|None:
  """Fill non-fp16 outputs through the DPU, then convert the fp16 result buffer."""
  store = _store_node(sink)
  if store is None or store.src[1].op is not Ops.CONST: return None
  output_dtype = store.src[0].dtype
  if output_dtype not in (dtypes.float, dtypes.int, dtypes.bool, dtypes.uint8): return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  value = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(store.src[1].arg)))[0])
  return (_emit_where_stage(total, info.outs[0], (_ZERO_SLOT, 0), value, Ops.ADD,
                            fp32_output=output_dtype is dtypes.float, int32_output=output_dtype is dtypes.int,
                            uint8_output=output_dtype is dtypes.uint8, bool_output=output_dtype is dtypes.bool),)

def _try_round_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Replace tinygrad's expanded round graph with abs, RK3588 roundoff LUT, and sign stages."""
  store = _store_node(sink)
  if store is None or (source := _try_round(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  source_slot = source.src[0].buf_uop.arg.slot
  negative, magnitude, rounded = alloc(), alloc(), alloc()
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  tasks = [_emit_where_stage(total, negative, (source_slot, 0), negative_one, Ops.MUL),
           _emit_where_stage(total, magnitude, (source_slot, 0), (negative, 0), Ops.MAX)]
  roundoff = UOp(Ops.CUSTOM, dtypes.half, (temp_index(magnitude),), arg="rk_roundoff")
  stage_store = store.replace(src=(temp_index(rounded), roundoff))
  stage_sink = sink.substitute({store:stage_store})
  plan = plan_rk(stage_sink)
  if isinstance(plan, str) or plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  zero = (_ZERO_SLOT, 0)
  negative_diff, negative_mask, positive_diff, positive_mask, sign = alloc(), alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, zero, (source_slot, 0), Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, (source_slot, 0), zero, Ops.SUB),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, alloc(), (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, sign, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, alloc(), (rounded, 0), (sign, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (rounded, 0), (sign, 0), Ops.MUL)))
  return tuple(tasks)

def _try_sign_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower sign(x) as the difference of positive and negative comparison masks."""
  store = _store_node(sink)
  if store is None or (input_slot := _try_sign(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  tasks: list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  zero = (_ZERO_SLOT, 0)
  negative_diff, negative_mask, positive_diff, positive_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, zero, (input_slot, 0), Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, (input_slot, 0), zero, Ops.SUB),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, alloc(), (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, info.outs[0], (positive_mask, 0), (negative_mask, 0), Ops.SUB)))
  return tuple(tasks)

def _try_inf_div_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Restore the denominator sign that RK3588 FDIV drops for CONST(±inf) / INDEX.

  The first task deliberately keeps the hardware infinity result. Multiplying it
  by a separately reconstructed sign fixes nonzero denominators without trying
  to synthesize infinity from a large finite constant.
  """
  store = _store_node(sink)
  if store is None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.FDIV or val.src[0].op is not Ops.CONST or not math.isinf(float(val.src[0].arg)): return None
  denominator = _unwrap(val.src[1])
  if denominator.op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  denominator_arg = (denominator.src[0].buf_uop.arg.slot, 0)
  numerator_arg = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(val.src[0].arg)))[0])
  base, negative_diff, negative_mask, positive_diff, positive_mask = alloc(), alloc(), alloc(), alloc(), alloc()
  sign_scratch, sign, product_scratch = alloc(), alloc(), alloc()
  zero = (_ZERO_SLOT, 0)
  tasks.extend((_emit_where_stage(total, base, numerator_arg, denominator_arg, Ops.FDIV),
                _emit_where_stage(total, negative_diff, zero, denominator_arg, Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, denominator_arg, zero, Ops.SUB),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                # Repeat the first dependent reads of freshly materialized comparison masks.
                _emit_where_stage(total, sign_scratch, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, sign, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, product_scratch, (base, 0), (sign, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (base, 0), (sign, 0), Ops.MUL)))
  return tuple(tasks)

def _try_hardsigmoid_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Clamp hardsigmoid through neg/MAX/neg so saturated lanes are exactly 0/1."""
  store = _store_node(sink)
  if store is None or (match := _try_hardsigmoid(store.src[1])) is None: return None
  input_slot, alpha, beta = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  scaled, shifted, positive, negative, clamped = (alloc() for _ in range(5))
  source, zero, negative_one = (input_slot, 0), (_ZERO_SLOT, 0), scalar(-1.0)
  return (_emit_where_stage(total, scaled, source, scalar(alpha), Ops.MUL),
          _emit_where_stage(total, shifted, (scaled, 0), scalar(beta), Ops.ADD),
          _emit_where_stage(total, positive, (shifted, 0), zero, Ops.MAX),
          _emit_where_stage(total, negative, (positive, 0), negative_one, Ops.MUL),
          _emit_where_stage(total, clamped, (negative, 0), negative_one, Ops.MAX),
          _emit_where_stage(total, info.outs[0], (clamped, 0), negative_one, Ops.MUL))

def _try_hardswish_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate hardswish with a Q14 base LUT and a local Q15 correction LUT."""
  store = _store_node(sink)
  if store is None or (source := _try_hardswish(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  base_slot = alloc()
  lut_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_hardswish")
  stage_store = store.replace(src=(temp_index(base_slot), lut_val))
  lut_plan = plan_rk(sink.substitute({store:stage_store}))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  plus, positive, negative, clamped_negative, relu6, product, fallback = (alloc() for _ in range(7))
  tasks.extend((_emit_where_stage(total, plus, source_arg, scalar(3.0), Ops.ADD),
                _emit_where_stage(total, positive, (plus, 0), zero, Ops.MAX),
                _emit_where_stage(total, negative, (positive, 0), scalar(-1.0), Ops.MUL),
                _emit_where_stage(total, clamped_negative, (negative, 0), scalar(-6.0), Ops.MAX),
                _emit_where_stage(total, relu6, (clamped_negative, 0), scalar(-1.0), Ops.MUL),
                _emit_where_stage(total, product, source_arg, (relu6, 0), Ops.MUL),
                _emit_where_stage(total, fallback, (product, 0), scalar(1/6), Ops.MUL)))
  wide_negative_diff, wide_negative_mask, wide_positive_diff, wide_positive_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, wide_negative_diff, scalar(-2.0), source_arg, Ops.SUB),
                _emit_where_stage(total, wide_negative_mask, (wide_negative_diff, 0), (wide_negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, wide_positive_diff, source_arg, scalar(2.0), Ops.SUB),
                _emit_where_stage(total, wide_positive_mask, (wide_positive_diff, 0), (wide_positive_diff, 0), Ops.MAX, compare=True)))
  wide_outside_scratch, wide_outside, wide_inside_scratch, wide_inside = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, wide_outside_scratch, (wide_negative_mask, 0), (wide_positive_mask, 0), Ops.MAX),
                _emit_where_stage(total, wide_outside, (wide_negative_mask, 0), (wide_positive_mask, 0), Ops.MAX),
                _emit_where_stage(total, wide_inside_scratch, one, (wide_outside, 0), Ops.SUB),
                _emit_where_stage(total, wide_inside, one, (wide_outside, 0), Ops.SUB)))
  base_inner_scratch, base_inner, fallback_outer_scratch, fallback_outer, wide = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, base_inner_scratch, (base_slot, 0), (wide_inside, 0), Ops.MUL),
                _emit_where_stage(total, base_inner, (base_slot, 0), (wide_inside, 0), Ops.MUL),
                _emit_where_stage(total, fallback_outer_scratch, (fallback, 0), (wide_outside, 0), Ops.MUL),
                _emit_where_stage(total, fallback_outer, (fallback, 0), (wide_outside, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (base_inner, 0), (fallback_outer, 0), Ops.ADD),
                _emit_where_stage(total, wide, (base_inner, 0), (fallback_outer, 0), Ops.ADD)))

  scaled = alloc()
  tasks.append(_emit_where_stage(total, scaled, source_arg, scalar(16.0), Ops.MUL))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(scaled),), arg="rk_hardswish_correction")
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(1/16), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(1/16), Ops.MUL)))
  negative_diff, negative_mask, positive_diff, positive_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, scalar(-0.125), source_arg, Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, source_arg, scalar(15/128), Ops.SUB),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True)))
  outside_scratch, outside, inside_scratch, inside = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, outside_scratch, (negative_mask, 0), (positive_mask, 0), Ops.MAX),
                _emit_where_stage(total, outside, (negative_mask, 0), (positive_mask, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB)))
  negative_zero_diff, negative_zero_mask, positive_zero_diff, positive_zero_mask = alloc(), alloc(), alloc(), alloc()
  nonzero_scratch, nonzero = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_zero_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative_zero_mask, (negative_zero_diff, 0), (negative_zero_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_zero_diff, source_arg, zero, Ops.SUB),
                _emit_where_stage(total, positive_zero_mask, (positive_zero_diff, 0), (positive_zero_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, nonzero_scratch, (negative_zero_mask, 0), (positive_zero_mask, 0), Ops.MAX),
                _emit_where_stage(total, nonzero, (negative_zero_mask, 0), (positive_zero_mask, 0), Ops.MAX)))
  local_nonzero_scratch, local_nonzero = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_nonzero_scratch, (local_scaled, 0), (nonzero, 0), Ops.MUL),
                _emit_where_stage(total, local_nonzero, (local_scaled, 0), (nonzero, 0), Ops.MUL)))
  base_selected_scratch, base_selected, local_selected_scratch, local_selected = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (wide, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (wide, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_nonzero, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_nonzero, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (base_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (base_selected, 0), (local_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_tanh_saturation_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Keep the staged tanh interior and replace bounded-LUT tails with sign(x)."""
  store = _store_node(sink)
  if store is None or (source := _try_tanh(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  base_slot = alloc()
  base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
  base_tasks = _try_elementwise_subtasks(sink.substitute({store:base_store}))
  if base_tasks is None: return None
  tasks.extend(base_tasks)
  used_slots = [st.task.out_slot for st in base_tasks] + \
    [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  source_arg, one = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0)
  low_diff, low_mask, high_diff, high_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, low_diff, scalar(-4.0), source_arg, Ops.SUB),
                _emit_where_stage(total, low_mask, (low_diff, 0), (low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, high_diff, source_arg, scalar(4.0), Ops.SUB),
                _emit_where_stage(total, high_mask, (high_diff, 0), (high_diff, 0), Ops.MAX, compare=True)))
  sign_scratch, sign, outside_scratch, outside, inside_scratch, inside = (alloc() for _ in range(6))
  tasks.extend((_emit_where_stage(total, sign_scratch, (high_mask, 0), (low_mask, 0), Ops.SUB),
                _emit_where_stage(total, sign, (high_mask, 0), (low_mask, 0), Ops.SUB),
                _emit_where_stage(total, outside_scratch, (high_mask, 0), (low_mask, 0), Ops.MAX),
                _emit_where_stage(total, outside, (high_mask, 0), (low_mask, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB)))
  base_selected_scratch, base_selected, sign_selected_scratch, sign_selected, finite = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (base_slot, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (base_slot, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, sign_selected_scratch, (sign, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, sign_selected, (sign, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (base_selected, 0), (sign_selected, 0), Ops.ADD),
                _emit_where_stage(total, finite, (base_selected, 0), (sign_selected, 0), Ops.ADD)))
  not_number = alloc()
  comparison = UOp(Ops.CMPNE, dtypes.bool, (source, source))
  comparison_store = store.replace(src=(temp_index(not_number, dtypes.bool), comparison))
  comparison_tasks = _try_comparison_subtasks(sink.substitute({store:comparison_store}))
  if comparison_tasks is None: return None
  last = comparison_tasks[-1]
  comparison_tasks = (*comparison_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
  tasks.extend(comparison_tasks)
  nan_denom_scratch, nan_denom, nan_numerator_scratch, nan_numerator = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, nan_denom_scratch, one, (not_number, 0), Ops.SUB),
                _emit_where_stage(total, nan_denom, one, (not_number, 0), Ops.SUB),
                _emit_where_stage(total, nan_numerator_scratch, (finite, 0), (nan_denom, 0), Ops.MUL),
                _emit_where_stage(total, nan_numerator, (finite, 0), (nan_denom, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (nan_numerator, 0), (nan_denom, 0), Ops.FDIV),
                _emit_where_stage(total, info.outs[0], (nan_numerator, 0), (nan_denom, 0), Ops.FDIV)))
  return tuple(tasks)

def _try_quick_gelu_saturation_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Keep the staged QuickGELU interior and use its exact zero/x asymptotes."""
  store = _store_node(sink)
  if store is None or (source := _try_quick_gelu(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  base_slot = alloc()
  base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
  base_tasks = _try_elementwise_subtasks(sink.substitute({store:base_store}))
  if base_tasks is None: return None
  tasks.extend(base_tasks)
  used_slots = [st.task.out_slot for st in base_tasks] + \
    [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  source_arg, one = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0)
  low_diff, low_mask, high_diff, high_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, low_diff, scalar(-10.0), source_arg, Ops.SUB),
                _emit_where_stage(total, low_mask, (low_diff, 0), (low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, high_diff, source_arg, scalar(5.0), Ops.SUB),
                _emit_where_stage(total, high_mask, (high_diff, 0), (high_diff, 0), Ops.MAX, compare=True)))
  outside_scratch, outside, inside_scratch, inside = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, outside_scratch, (high_mask, 0), (low_mask, 0), Ops.MAX),
                _emit_where_stage(total, outside, (high_mask, 0), (low_mask, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB)))
  base_selected_scratch, base_selected, high_selected_scratch, high_selected = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (base_slot, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (base_slot, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, high_selected_scratch, source_arg, (high_mask, 0), Ops.MUL),
                _emit_where_stage(total, high_selected, source_arg, (high_mask, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (base_selected, 0), (high_selected, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (base_selected, 0), (high_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_abs_subtasks(sink:UOp) -> tuple[RKSubTask, RKSubTask]|None:
  """Lower abs(x) as neg=x*-1 followed by max(x,neg).

  The older single-task BS-negate/EW-MAX emitter remains below for reference,
  but that register combination times out when a CMAC task ran earlier in the
  process. Two ordinary DPU stages are stable across compute-family changes.
  """
  store = _store_node(sink)
  if store is None or (input_slot := _try_abs(_unwrap(store.src[1]))) is None: return None
  total, info = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink)
  out_slot, scratch = info.outs[0], max(info.globals, default=-1) + 1
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  return (_emit_where_stage(total, scratch, (input_slot, 0), negative_one, Ops.MUL),
          _emit_where_stage(total, out_slot, (input_slot, 0), (scratch, 0), Ops.MAX))

def _try_comparison_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Materialize CMPLT/CMPNE boolean expressions as fp16 0/1 masks, then pack to bool."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.bool: return None
  val, info = _unwrap(store.src[1]), ProgramInfo.from_sink(sink)
  if val.op not in (Ops.CMPLT, Ops.CMPNE, Ops.OR, Ops.AND, Ops.MAX): return None
  total, out_slot, next_slot = prod(_shape_of_store(sink)), info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def dependent(out:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops, bool_output=False,
                bool_inputs:tuple[int,...]=(), int32_inputs:tuple[int,...]=(),
                broadcast_inputs:tuple[int,...]=()) -> None:
    # The first DPU task consuming a freshly materialized comparison mask can
    # observe stale lanes. Repeat the identical read, as in the proven WHERE path.
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op, bool_inputs=bool_inputs,
                                   int32_inputs=int32_inputs, broadcast_inputs=broadcast_inputs))
    tasks.append(_emit_where_stage(total, out, lhs, rhs, op, bool_inputs=bool_inputs,
                                   int32_inputs=int32_inputs, broadcast_inputs=broadcast_inputs,
                                   bool_output=bool_output))

  def data_arg(u:UOp) -> tuple[tuple[int,int], tuple[int,...], tuple[int,...], tuple[int,...]]|None:
    u = _unwrap(u)
    if (arg := _where_arg(u)) is None: return None
    if u.op is Ops.CONST:
      if isinstance(u.arg, (float, np.floating)) and math.isinf(float(u.arg)):
        normalized = math.copysign(65504.0, float(u.arg))
        arg = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', normalized))[0])
      return arg, (), (), ()
    slot, source_n = arg[0], int(u.src[0].src[0].arg)
    return arg, ((slot,) if u.dtype is dtypes.bool else ()), \
      ((slot,) if u.dtype is dtypes.int else ()), ((slot,) if source_n < total else ())

  # Direct boolean OR/AND (including maximum lowered to OR) operates on
  # byte-packed buffers rather than freshly materialized comparison masks.
  # Convert inputs to fp16 0/1, run native MAX/MUL, then pack back to bool.
  if val.op in (Ops.OR, Ops.AND, Ops.MAX):
    lhs_info, rhs_info = (data_arg(x) for x in val.src)
    if lhs_info is not None and rhs_info is not None:
      bool_inputs, int32_inputs, broadcasts = \
        (tuple(dict.fromkeys(lhs_info[i] + rhs_info[i])) for i in range(1, 4))
      op = Ops.MUL if val.op is Ops.AND else Ops.MAX
      return (_emit_where_stage(total, out_slot, lhs_info[0], rhs_info[0], op,
                                bool_inputs=bool_inputs, int32_inputs=int32_inputs,
                                broadcast_inputs=broadcasts, bool_output=True),)

  def lower(u:UOp) -> tuple[int,int]|None:
    u = _unwrap(u)
    if u.op is Ops.CMPLT:
      lhs_info, rhs_info = (data_arg(x) for x in u.src)
      if lhs_info is None or rhs_info is None: return None
      lhs, rhs = lhs_info[0], rhs_info[0]
      bool_inputs, int32_inputs, broadcasts = (tuple(dict.fromkeys(lhs_info[i] + rhs_info[i])) for i in range(1, 4))
      comparison_inputs = tuple(dict.fromkeys(x[0] for x in (lhs, rhs) if x[0] not in (_CONST_SLOT, _ZERO_SLOT)))
      diff, mask = alloc(), alloc()
      tasks.extend((_emit_where_stage(total, diff, rhs, lhs, Ops.SUB, bool_inputs=bool_inputs,
                                      int32_inputs=int32_inputs, broadcast_inputs=broadcasts,
                                      comparison_inputs=comparison_inputs),
                    _emit_where_stage(total, mask, (diff, 0), (diff, 0), Ops.MAX, compare=True)))
      return (mask, 0)
    if u.op is Ops.CMPNE:
      # tinygrad represents logical NOT(x) as CMPNE(x, True).
      for logical, const in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
        if const.op is Ops.CONST and const.dtype is dtypes.bool and bool(const.arg):
          logical_u = _unwrap(logical)
          if logical_u.op is Ops.INDEX and logical_u.dtype is dtypes.bool:
            logical_info = data_arg(logical_u)
            if logical_info is None: return None
            mask_arg, bool_inputs, int32_inputs, broadcasts = logical_info
          else:
            lowered_mask = lower(logical)
            if lowered_mask is None: return None
            mask_arg = lowered_mask
            bool_inputs, int32_inputs, broadcasts = (), (), ()
          result = alloc()
          dependent(result, one, mask_arg, Ops.SUB, bool_inputs=bool_inputs,
                    int32_inputs=int32_inputs, broadcast_inputs=broadcasts)
          return (result, 0)
      lhs_data, rhs_data = (data_arg(x) for x in u.src)
      if lhs_data is None or rhs_data is None: return None
      lt, gt = lower(UOp(Ops.CMPLT, dtypes.bool, (u.src[0], u.src[1]))), \
               lower(UOp(Ops.CMPLT, dtypes.bool, (u.src[1], u.src[0])))
      if lt is None or gt is None: return None
      result = alloc()
      dependent(result, lt, gt, Ops.MAX)
      return (result, 0)
    if u.op in (Ops.OR, Ops.AND):
      lhs_mask, rhs_mask = (lower(x) for x in u.src)
      if lhs_mask is None or rhs_mask is None: return None
      result = alloc()
      dependent(result, lhs_mask, rhs_mask, Ops.MAX if u.op is Ops.OR else Ops.MUL)
      return (result, 0)
    return None

  if (result := lower(val)) is None: return None
  dependent(out_slot, result, (_ZERO_SLOT, 0), Ops.ADD, bool_output=True)
  return tuple(tasks)

def _try_exp2_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve IEEE EXP2 results for infinities and NaN around the bounded LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.EXP2 or len(val.src) != 1 or (source := _unwrap(val.src[0])).op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    # Repeat the first read of each freshly materialized value. Comparison and
    # LUT programs run as reset-separated stages and otherwise expose stale lanes.
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    # Intermediate masks stay as fp16 0/1 scratch. Only a user-visible boolean
    # output should be packed to the byte-wide bool representation.
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  # First materialize the normal bounded-domain LUT result.
  lut_slot = alloc()
  lut_plan = plan_rk(stage_sink(val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  # Comparison inputs are normalized from infinities to the fp16 extrema by
  # the runtime. Values beyond these thresholds already overflow/underflow
  # EXP2 in fp16, so the masks retain the required result semantics.
  hi, lo = UOp.const(dtypes.half, 65472.0), UOp.const(dtypes.half, -65472.0)
  positive = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  negative = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, lo)))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if positive is None or negative is None or not_number is None: return None

  # +inf: base / (1-positive) -> +inf.
  positive_denom, positive_result = alloc(), alloc()
  dependent(positive_denom, one, positive, Ops.SUB)
  dependent(positive_result, (lut_slot, 0), (positive_denom, 0), Ops.FDIV)
  # -inf: result * (1-negative) -> 0.
  negative_denom, finite_result = alloc(), alloc()
  dependent(negative_denom, one, negative, Ops.SUB)
  dependent(finite_result, (positive_result, 0), (negative_denom, 0), Ops.MUL)
  # NaN: first force the numerator to zero, then 0/0 produces NaN.
  nan_denom, nan_numerator = alloc(), alloc()
  dependent(nan_denom, one, not_number, Ops.SUB)
  dependent(nan_numerator, (finite_result, 0), (nan_denom, 0), Ops.MUL)
  dependent(out, (nan_numerator, 0), (nan_denom, 0), Ops.FDIV)
  return tuple(tasks)

def _try_exp_correction_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Use two LUT tasks for exp(x): Q12 exp plus a signed Q12 residual."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  lut_result = _try_lut(val)
  if lut_result is None or lut_result[3] is not Ops.EXP2 or abs(lut_result[1] - math.log2(math.e)) >= 1e-3: return None
  inner = _unwrap(val.src[0])
  if inner.op is not Ops.MUL: return None
  source = next((_unwrap(x) for x in inner.src if _unwrap(x).op is Ops.INDEX), None)
  if source is None or source.dtype is not dtypes.half: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  exp_slot, shifted_slot, correction_input, correction_slot = alloc(), alloc(), alloc(), alloc()
  exp_plan = plan_rk(stage_sink(val, exp_slot))
  if isinstance(exp_plan, str) or exp_plan.kind != "dpu_lut": return None
  exp_cmds, exp_task, exp_relocs = emit_rk(exp_plan)

  # EXP2(correction_index) is only a carrier that supplies the transformed slot to
  # the generic LUT emitter; the plan marker selects the residual table.
  correction_index = temp_index(correction_input)
  correction_val = UOp(Ops.EXP2, dtypes.half, (correction_index,))
  correction_sink = stage_sink(correction_val, correction_slot)
  correction_plan = RKPlan("dpu_lut", correction_sink, correction_slot,
                           (correction_input,), lut_op=_LUT_EXP_CORRECTION)
  correction_cmds, correction_task, correction_relocs = emit_rk(correction_plan)

  # Repeat the first read after the reset-separated LUT tasks; this is the
  # same stale-lane workaround used by comparison and special-value stages.
  unbiased, scratch, corrected = alloc(), alloc(), alloc()
  source_arg = (source.src[0].buf_uop.arg.slot, 0)
  shift = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.75))[0])
  zoom = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 8.0))[0])
  correction_bias = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 0.125))[0])
  tasks.extend((RKSubTask(exp_cmds, exp_task, exp_relocs),
                _emit_where_stage(total, scratch, source_arg, shift, Ops.ADD),
                _emit_where_stage(total, shifted_slot, source_arg, shift, Ops.ADD),
                _emit_where_stage(total, scratch, (shifted_slot, 0), zoom, Ops.MUL),
                _emit_where_stage(total, correction_input, (shifted_slot, 0), zoom, Ops.MUL),
                RKSubTask(correction_cmds, correction_task, correction_relocs),
                _emit_where_stage(total, scratch, (correction_slot, 0), correction_bias, Ops.SUB),
                _emit_where_stage(total, unbiased, (correction_slot, 0), correction_bias, Ops.SUB),
                _emit_where_stage(total, scratch, (exp_slot, 0), (unbiased, 0), Ops.ADD),
                _emit_where_stage(total, corrected, (exp_slot, 0), (unbiased, 0), Ops.ADD)))

  hi, lo = UOp.const(dtypes.half, 65472.0), UOp.const(dtypes.half, -65472.0)
  positive = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  negative = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, lo)))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if positive is None or negative is None or not_number is None: return None

  positive_denom, positive_result = alloc(), alloc()
  dependent(positive_denom, one, positive, Ops.SUB)
  dependent(positive_result, (corrected, 0), (positive_denom, 0), Ops.FDIV)
  negative_denom, finite_result = alloc(), alloc()
  dependent(negative_denom, one, negative, Ops.SUB)
  dependent(finite_result, (positive_result, 0), (negative_denom, 0), Ops.MUL)
  nan_denom, nan_numerator = alloc(), alloc()
  dependent(nan_denom, one, not_number, Ops.SUB)
  dependent(nan_numerator, (finite_result, 0), (nan_denom, 0), Ops.MUL)
  dependent(out, (nan_numerator, 0), (nan_denom, 0), Ops.FDIV)
  return tuple(tasks)

def _try_sigmoid_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve sigmoid saturation and NaN semantics outside the bounded LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  if _try_sigmoid(val) is None: return None
  indexes = [u for u in val.toposort() if u.op is Ops.INDEX]
  if len(indexes) != 1 or (source := _unwrap(indexes[0])).dtype is not dtypes.half: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  lut_slot = alloc()
  lut_plan = plan_rk(stage_sink(val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  high = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.half, 8.0), source)))
  low = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, UOp.const(dtypes.half, -8.0))))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if high is None or low is None or not_number is None: return None

  high_delta, high_adjustment, high_result = alloc(), alloc(), alloc()
  dependent(high_delta, one, (lut_slot, 0), Ops.SUB)
  dependent(high_adjustment, (high_delta, 0), high, Ops.MUL)
  dependent(high_result, (lut_slot, 0), (high_adjustment, 0), Ops.ADD)
  low_denom, bounded = alloc(), alloc()
  dependent(low_denom, one, low, Ops.SUB)
  dependent(bounded, (high_result, 0), (low_denom, 0), Ops.MUL)
  nan_denom, nan_numerator = alloc(), alloc()
  dependent(nan_denom, one, not_number, Ops.SUB)
  dependent(nan_numerator, (bounded, 0), (nan_denom, 0), Ops.MUL)
  dependent(out, (nan_numerator, 0), (nan_denom, 0), Ops.FDIV)
  return tuple(tasks)

def _try_log2_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve LOG2 zero, infinity, and NaN semantics around the bounded LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.LOG2 or len(val.src) != 1 or (source := _unwrap(val.src[0])).op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  zero = UOp.const(dtypes.half, 0.0)

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  lut_slot = alloc()
  lut_plan = plan_rk(stage_sink(val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  hi = UOp.const(dtypes.half, 65472.0)
  positive = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  negative = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, zero)))
  nonzero = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, zero)))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if positive is None or negative is None or nonzero is None or not_number is None: return None

  zero_result = alloc()
  dependent(zero_result, (lut_slot, 0), nonzero, Ops.FDIV)
  positive_denom, finite = alloc(), alloc()
  dependent(positive_denom, one, positive, Ops.SUB)
  dependent(finite, (zero_result, 0), (positive_denom, 0), Ops.FDIV)
  invalid, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid, negative, not_number, Ops.MAX)
  dependent(invalid_denom, one, (invalid, 0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV)
  dependent(out, (finite, 0), (invalid_factor, 0), Ops.MUL)
  return tuple(tasks)

def _try_sqrt_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve SQRT zero, infinity, and NaN semantics around the bounded LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.SQRT or len(val.src) != 1 or (source := _unwrap(val.src[0])).op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  half = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 0.5))[0])
  zero = UOp.const(dtypes.half, 0.0)
  source_arg = (source.src[0].buf_uop.arg.slot, 0)

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  lut_slot = alloc()
  lut_plan = plan_rk(stage_sink(val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  # Three Newton steps remove the linear LUT's curvature error near zero:
  # yₙ₊₁ = (yₙ + x/yₙ) / 2. Special values are repaired by the masks below.
  quotient, newton_sum, refined = alloc(), alloc(), alloc()
  dependent(quotient, source_arg, (lut_slot, 0), Ops.FDIV)
  dependent(newton_sum, (lut_slot, 0), (quotient, 0), Ops.ADD)
  dependent(refined, (newton_sum, 0), half, Ops.MUL)
  quotient2, newton_sum2, refined2 = alloc(), alloc(), alloc()
  dependent(quotient2, source_arg, (refined, 0), Ops.FDIV)
  dependent(newton_sum2, (refined, 0), (quotient2, 0), Ops.ADD)
  dependent(refined2, (newton_sum2, 0), half, Ops.MUL)
  quotient3, newton_sum3, refined3 = alloc(), alloc(), alloc()
  dependent(quotient3, source_arg, (refined2, 0), Ops.FDIV)
  dependent(newton_sum3, (refined2, 0), (quotient3, 0), Ops.ADD)
  dependent(refined3, (newton_sum3, 0), half, Ops.MUL)

  hi = UOp.const(dtypes.half, 65472.0)
  positive = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  negative = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, zero)))
  nonzero = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, zero)))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if positive is None or negative is None or nonzero is None or not_number is None: return None

  positive_denom, positive_result = alloc(), alloc()
  dependent(positive_denom, one, positive, Ops.SUB)
  dependent(positive_result, (refined3, 0), (positive_denom, 0), Ops.FDIV)
  zero_result = alloc()
  dependent(zero_result, (positive_result, 0), nonzero, Ops.MUL)
  invalid, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid, negative, not_number, Ops.MAX)
  dependent(invalid_denom, one, (invalid, 0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV)
  dependent(out, (zero_result, 0), (invalid_factor, 0), Ops.MUL)
  return tuple(tasks)

def _try_rsqrt_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve RSQRT zero, infinity, and NaN semantics around its dedicated LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.RECIPROCAL or len(val.src) != 1: return None
  sqrt = _unwrap(val.src[0])
  if sqrt.op is not Ops.SQRT or len(sqrt.src) != 1 or (source := _unwrap(sqrt.src[0])).op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  zero = UOp.const(dtypes.half, 0.0)
  out_idx = store.src[0]

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def stage_sink(stage_val:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_val))})

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  hi = UOp.const(dtypes.half, 65472.0)
  positive = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  negative = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, zero)))
  nonzero = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, zero)))
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  greater_zero = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (zero, source)))
  below_1 = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, UOp.const(dtypes.half, 0.0625))))
  below_2 = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, UOp.const(dtypes.half, 0.00390625))))
  if positive is None or negative is None or nonzero is None or not_number is None or \
     greater_zero is None or below_1 is None or below_2 is None: return None

  source_arg = (source.src[0].buf_uop.arg.slot, 0)
  fifteen = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 15.0))[0])
  three = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 3.0))[0])
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  negative_four = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -4.0))[0])
  half = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 0.5))[0])
  one_half = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.5))[0])

  # Exact powers-of-16 move small positive inputs into the LUT's [1/16, 4]
  # domain. The output is scaled back by the corresponding powers of four.
  low_1, low_2 = alloc(), alloc()
  dependent(low_1, greater_zero, below_1, Ops.MUL)
  dependent(low_2, greater_zero, below_2, Ops.MUL)
  factor_1_delta, factor_1, scaled_1 = alloc(), alloc(), alloc()
  dependent(factor_1_delta, (low_1, 0), fifteen, Ops.MUL)
  dependent(factor_1, one, (factor_1_delta, 0), Ops.ADD)
  dependent(scaled_1, source_arg, (factor_1, 0), Ops.MUL)
  factor_2_delta, factor_2, scaled_2 = alloc(), alloc(), alloc()
  dependent(factor_2_delta, (low_2, 0), fifteen, Ops.MUL)
  dependent(factor_2, one, (factor_2_delta, 0), Ops.ADD)
  dependent(scaled_2, (scaled_1, 0), (factor_2, 0), Ops.MUL)

  scaled_source = temp_index(scaled_2)
  scaled_val = val.substitute({source:scaled_source})
  lut_slot = alloc()
  lut_plan = plan_rk(stage_sink(scaled_val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  # One inverse-square-root Newton step removes residual linear interpolation
  # error. Clamp the refinement input at four so +inf stays a finite zero case.
  neg_scaled, clamped_neg, safe_scaled = alloc(), alloc(), alloc()
  dependent(neg_scaled, (scaled_2, 0), negative_one, Ops.MUL)
  dependent(clamped_neg, (neg_scaled, 0), negative_four, Ops.MAX)
  dependent(safe_scaled, (clamped_neg, 0), negative_one, Ops.MUL)
  square, product, half_product, correction, refined = (alloc() for _ in range(5))
  dependent(square, (lut_slot, 0), (lut_slot, 0), Ops.MUL)
  dependent(product, (safe_scaled, 0), (square, 0), Ops.MUL)
  dependent(half_product, (product, 0), half, Ops.MUL)
  dependent(correction, one_half, (half_product, 0), Ops.SUB)
  dependent(refined, (lut_slot, 0), (correction, 0), Ops.MUL)

  out_factor_1_delta, out_factor_1, scaled_out_1 = alloc(), alloc(), alloc()
  dependent(out_factor_1_delta, (low_1, 0), three, Ops.MUL)
  dependent(out_factor_1, one, (out_factor_1_delta, 0), Ops.ADD)
  dependent(scaled_out_1, (refined, 0), (out_factor_1, 0), Ops.MUL)
  out_factor_2_delta, out_factor_2, scaled_out_2 = alloc(), alloc(), alloc()
  dependent(out_factor_2_delta, (low_2, 0), three, Ops.MUL)
  dependent(out_factor_2, one, (out_factor_2_delta, 0), Ops.ADD)
  dependent(scaled_out_2, (scaled_out_1, 0), (out_factor_2, 0), Ops.MUL)

  zero_result = alloc()
  dependent(zero_result, (scaled_out_2, 0), nonzero, Ops.FDIV)
  positive_denom, finite = alloc(), alloc()
  dependent(positive_denom, one, positive, Ops.SUB)
  dependent(finite, (zero_result, 0), (positive_denom, 0), Ops.MUL)
  invalid, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid, negative, not_number, Ops.MAX)
  dependent(invalid_denom, one, (invalid, 0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV)
  dependent(out, (finite, 0), (invalid_factor, 0), Ops.MUL)
  return tuple(tasks)

def _try_where_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  store = _store_node(sink)
  if store is None or (val := _unwrap(store.src[1])).op is not Ops.WHERE: return None
  total, info = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink)
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  materialized:dict[UOp, int] = {}
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc(count:int=1) -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + count
    return ret

  def lower_arg(u:UOp) -> tuple[int,int]|None:
    if (arg := _where_arg(u)) is not None: return arg
    u = _unwrap(u)
    if u not in materialized:
      materialized[u] = alloc()
      if u.op is Ops.WHERE:
        if not lower(u, materialized[u], False): return None
      elif u.op in _DPU_EW_CFGS and len(u.src) == 2:
        a, b = (lower_arg(x) for x in u.src)
        if a is None or b is None: return None
        tasks.append(_emit_where_stage(total, materialized[u], a, b, u.op))
      elif u.op is Ops.TRUNC and (source := _where_arg(u.src[0])) is not None:
        tasks.append(_emit_trunc_stage(total, materialized[u], source))
      else: return None
    return (materialized[u], 0)

  def lower(w:UOp, out_slot:int, final:bool) -> bool:
    w = _unwrap(w)
    if w.op is not Ops.WHERE: return False
    cond = _unwrap(w.src[0])
    true, false = (lower_arg(x) for x in w.src[1:])
    if true is None or false is None: return False
    broadcasts = tuple(a[0] for u, a in zip(w.src[1:], (true, false))
                       if _unwrap(u).op is Ops.INDEX and int(_unwrap(u).src[0].src[0].arg) < total)
    first = alloc(4)
    t0, mask, scratch, selected_false = range(first, first+4)
    int_out = final and w.dtype is dtypes.int
    uint8_out = final and w.dtype is dtypes.uint8
    if cond.op is Ops.CMPLT:
      lhs, rhs = (lower_arg(x) for x in cond.src)
      if lhs is None or rhs is None: return False
      tasks.extend((_emit_where_stage(total, t0, rhs, lhs, Ops.SUB),
                    _emit_where_stage(total, mask, (t0,0), (t0,0), Ops.MAX, compare=True)))
      lhs_u, rhs_u, true_u, false_u = (_unwrap(x) for x in (*cond.src, *w.src[1:]))

      # WHERE(x<c, x, f) without 0*inf:
      # min(x,c) + (f-c)*(1-mask). This preserves selected -inf and discards
      # unselected +inf through MAX/negation rather than multiplication.
      if true_u is lhs_u and lhs_u.op is Ops.INDEX and rhs_u.op is Ops.CONST and false_u.op is Ops.CONST and \
         math.isfinite(float(rhs_u.arg)) and math.isfinite(float(false_u.arg)):
        negative, maximum, minimum, inverse, adjustment = (alloc() for _ in range(5))
        negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
        negative_threshold = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -float(rhs_u.arg)))[0])
        delta = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(false_u.arg)-float(rhs_u.arg)))[0])
        tasks.extend((_emit_where_stage(total, scratch, lhs, negative_one, Ops.MUL),
                      _emit_where_stage(total, negative, lhs, negative_one, Ops.MUL),
                      _emit_where_stage(total, scratch, (negative,0), negative_threshold, Ops.MAX),
                      _emit_where_stage(total, maximum, (negative,0), negative_threshold, Ops.MAX),
                      _emit_where_stage(total, scratch, (maximum,0), negative_one, Ops.MUL),
                      _emit_where_stage(total, minimum, (maximum,0), negative_one, Ops.MUL),
                      _emit_where_stage(total, scratch, one, (mask,0), Ops.SUB),
                      _emit_where_stage(total, inverse, one, (mask,0), Ops.SUB),
                      _emit_where_stage(total, scratch, (inverse,0), delta, Ops.MUL),
                      _emit_where_stage(total, adjustment, (inverse,0), delta, Ops.MUL),
                      _emit_where_stage(total, scratch, (minimum,0), (adjustment,0), Ops.ADD),
                      _emit_where_stage(total, out_slot, (minimum,0), (adjustment,0), Ops.ADD,
                                        int32_output=int_out, uint8_output=uint8_out)))
        return True

      # Literal infinity cannot use the ordinary arm*mask selector because
      # an unselected 0*inf becomes NaN. Gate a finite extremum first, then
      # divide by the opposite gate so only selected lanes become infinity.
      true_inf = true_u.op is Ops.CONST and math.isinf(float(true_u.arg))
      false_inf = false_u.op is Ops.CONST and math.isinf(float(false_u.arg))
      if true_inf or false_inf:
        selected_true, inverse, selected_false = alloc(), alloc(), alloc()
        tasks.extend((_emit_where_stage(total, scratch, one, (mask,0), Ops.SUB),
                      _emit_where_stage(total, inverse, one, (mask,0), Ops.SUB)))
        if true_inf:
          finite_true = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', math.copysign(65504.0, float(true_u.arg))))[0])
          gated_true = alloc()
          tasks.extend((_emit_where_stage(total, scratch, finite_true, (mask,0), Ops.MUL),
                        _emit_where_stage(total, gated_true, finite_true, (mask,0), Ops.MUL),
                        _emit_where_stage(total, scratch, (gated_true,0), (inverse,0), Ops.FDIV),
                        _emit_where_stage(total, selected_true, (gated_true,0), (inverse,0), Ops.FDIV)))
        else:
          tasks.extend((_emit_where_stage(total, scratch, true, (mask,0), Ops.MUL, broadcast_inputs=broadcasts),
                        _emit_where_stage(total, selected_true, true, (mask,0), Ops.MUL, broadcast_inputs=broadcasts)))
        if false_inf:
          finite_false = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', math.copysign(65504.0, float(false_u.arg))))[0])
          gated_false = alloc()
          tasks.extend((_emit_where_stage(total, scratch, finite_false, (inverse,0), Ops.MUL),
                        _emit_where_stage(total, gated_false, finite_false, (inverse,0), Ops.MUL),
                        _emit_where_stage(total, scratch, (gated_false,0), (mask,0), Ops.FDIV),
                        _emit_where_stage(total, selected_false, (gated_false,0), (mask,0), Ops.FDIV)))
        else:
          tasks.extend((_emit_where_stage(total, scratch, false, (inverse,0), Ops.MUL, broadcast_inputs=broadcasts),
                        _emit_where_stage(total, selected_false, false, (inverse,0), Ops.MUL, broadcast_inputs=broadcasts)))
        tasks.extend((_emit_where_stage(total, scratch, (selected_true,0), (selected_false,0), Ops.ADD),
                      _emit_where_stage(total, out_slot, (selected_true,0), (selected_false,0), Ops.ADD,
                                        int32_output=int_out, uint8_output=uint8_out)))
        return True

      tasks.extend((
                    _emit_where_stage(total, scratch, true, (mask,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, t0, true, (mask,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, scratch, one, (mask,0), Ops.SUB),
                    _emit_where_stage(total, mask, false, (scratch,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, selected_false, false, (scratch,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, out_slot, (t0,0), (selected_false,0), Ops.ADD,
                                      broadcast_inputs=broadcasts, int32_output=int_out, uint8_output=uint8_out)))
      return True
    if cond.op is Ops.OR and all(_unwrap(x).op is Ops.CMPLT for x in cond.src):
      masks = []
      for cmp in cond.src:
        lhs, rhs = (lower_arg(x) for x in _unwrap(cmp).src)
        if lhs is None or rhs is None: return False
        diff = alloc(2)
        cmp_mask = diff + 1
        tasks.extend((_emit_where_stage(total, diff, rhs, lhs, Ops.SUB),
                      _emit_where_stage(total, cmp_mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
        masks.append((cmp_mask, 0))
      combined = alloc()
      tasks.extend((_emit_where_stage(total, combined, masks[0], masks[1], Ops.MAX),
                    _emit_where_stage(total, scratch, true, (combined,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, t0, true, (combined,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, scratch, one, (combined,0), Ops.SUB),
                    _emit_where_stage(total, mask, false, (scratch,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, selected_false, false, (scratch,0), Ops.MUL, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, out_slot, (t0,0), (selected_false,0), Ops.ADD,
                                      broadcast_inputs=broadcasts, int32_output=int_out, uint8_output=uint8_out)))
      return True
    if cond.op is Ops.INDEX and (mask_arg := _where_arg(cond)) is not None:
      bool_slots = (mask_arg[0],)
      tasks.extend((_emit_where_stage(total, t0, true, mask_arg, Ops.MUL, bool_inputs=bool_slots, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, scratch, one, mask_arg, Ops.SUB, bool_inputs=bool_slots, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, selected_false, false, (scratch,0), Ops.MUL,
                                      bool_inputs=bool_slots, broadcast_inputs=broadcasts),
                    _emit_where_stage(total, out_slot, (t0,0), (selected_false,0), Ops.ADD, bool_inputs=bool_slots,
                                      broadcast_inputs=broadcasts, int32_output=int_out, uint8_output=uint8_out)))
      return True
    return False

  return tuple(tasks) if lower(val, out, True) else None

def _try_elementwise_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Split a nested fp16 elementwise expression into independently classifiable DPU stages."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val, info = _unwrap(store.src[1]), ProgramInfo.from_sink(sink)
  if val.op not in (_DPU_EW_CFGS.keys() | _LUT_OPS | {Ops.RECIPROCAL, Ops.TRUNC}): return None
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  materialized:dict[UOp, UOp] = {}

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot), *out_idx.src[1:]))

  def make_stage_sink(stage_val:UOp, out_slot:int) -> tuple[UOp, UOp]:
    out_idx = temp_index(out_slot, stage_val.dtype)
    return sink.substitute({store:store.replace(src=(out_idx, stage_val))}), out_idx

  def emit_stage(stage_val:UOp, out_slot:int) -> UOp|None:
    nonlocal next_slot
    stage_sink, out_idx = make_stage_sink(stage_val, out_slot)
    for special in (_try_exp_correction_subtasks, _try_sigmoid_special_subtasks, _try_exp2_special_subtasks, _try_log2_special_subtasks,
                    _try_rsqrt_special_subtasks, _try_sqrt_special_subtasks):
      if (special_tasks := special(stage_sink)) is None: continue
      tasks.extend(special_tasks)
      used_slots = [st.task.out_slot for st in special_tasks] + \
        [r.globals_slot for st in special_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
      next_slot = max(next_slot, max(used_slots, default=-1) + 1)
      return out_idx
    plan = plan_rk(stage_sink)
    if isinstance(plan, str) or plan.kind not in ("dpu", "dpu_lut"): return None
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return out_idx

  def lower_arg(u:UOp) -> UOp|None:
    nonlocal next_slot
    if u.op is Ops.CAST and u.dtype is dtypes.half and _unwrap(u.src[0]).op is Ops.INDEX:
      if u not in materialized:
        slot = alloc()
        stage_sink, idx = make_stage_sink(u, slot)
        if (cast_tasks := _try_cast_subtasks(stage_sink)) is None: return None
        tasks.extend(cast_tasks)
        materialized[u] = idx
      return materialized[u]
    u = _unwrap(u)
    if u.op in (Ops.INDEX, Ops.CONST): return u
    if u not in materialized:
      slot = alloc()
      if u.op is Ops.WHERE:
        stage_sink, idx = make_stage_sink(u, slot)
        if (where_tasks := _try_where_subtasks(stage_sink)) is None: return None
        tasks.extend(where_tasks)
        used_slots = [r.globals_slot for st in where_tasks for r in st.relocs
                      if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
        next_slot = max(next_slot, max(used_slots, default=-1) + 1)
      else:
        if (lowered := lower(u, slot)) is None: return None
        idx = lowered
      materialized[u] = idx
    return materialized[u]

  def lower(u:UOp, out_slot:int) -> UOp|None:
    u = _unwrap(u)
    if (idx := emit_stage(u, out_slot)) is not None: return idx
    if u.op is Ops.TRUNC and (source := _where_arg(u.src[0])) is not None:
      tasks.append(_emit_trunc_stage(prod(_shape_of_store(sink)), out_slot, source))
      return temp_index(out_slot, u.dtype)
    if u.op not in (_DPU_EW_CFGS.keys() | _LUT_OPS | {Ops.RECIPROCAL}): return None
    src = tuple(lower_arg(x) for x in u.src)
    if any(x is None for x in src): return None
    return emit_stage(u.replace(src=src), out_slot)  # type: ignore[arg-type]

  return tuple(tasks) if lower(val, out) is not None and (len(tasks) > 1 or val.op is Ops.TRUNC) else None

def _emit_dpu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """DPU elementwise op (ADD/SUB/MUL/MAX), scalar operand, or DMA copy. Register sequence from elementwise.py."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  sink = plan.sink
  store = _store_node(sink)
  assert store is not None, "dpu: no STORE node"
  val = store.src[1]
  where_max = _try_where_max(val)
  if where_max is not None: val = where_max
  val = _unwrap(val)  # strip no-op CASTs (half→half) so EW ops are recognized
  total = prod(_shape_of_store(sink))
  dw = (total + 7) // 8 - 1
  vu = _unwrap(val)
  is_copy = vu.op is Ops.INDEX
  is_fill = vu.op is Ops.CONST
  lut_result = _try_lut(val)
  sub_slots, scalar = (None, None) if (is_copy or is_fill or lut_result is not None) else (_try_sub(val), _try_scalar(val))
  reciprocal = None if (is_copy or is_fill or sub_slots is not None or scalar is not None or lut_result is not None) else _try_reciprocal(val)
  abs_slot = _try_abs(val) if not (is_copy or is_fill or lut_result is not None or sub_slots is not None or
                                   scalar is not None or reciprocal is not None) else None
  layout:tuple[int,...] = (total,)
  if is_copy and (aff := _affine_index(vu.src[1])) is not None:
    extents = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort()
               if u.op is Ops.RANGE and u.arg[1].name == "LOOP" and u.src[0].op is Ops.CONST}
    store_aff = _affine_index(store.src[0].src[1])
    axes = sorted(store_aff[0], key=lambda x: store_aff[0][x], reverse=True) if store_aff is not None else sorted(extents)
    shape = tuple(extents[i] for i in axes)
    strides = tuple(aff[0].get(i, 0) for i in axes)
    contiguous = tuple(prod(shape[i+1:]) for i in range(len(shape)))
    if strides != contiguous or aff[1]: layout = (total, len(shape), *shape, *strides, aff[1])
  # track the EW op for FDIV-specific register emissions (OUT_CVT_SCALE, FP16TOFP32_EN=0)
  ew_op = Ops.SUB if sub_slots else (Ops.FDIV if reciprocal else (val.op if scalar else
          (Ops.ADD if is_fill else (Ops.MAX if abs_slot else (None if is_copy else val.op)))))
  # S_POINTER: re-arm ping-pong pointers (pcchain.md §ADD Reference Captures)
  # Without these, PC chain later tasks don't execute correctly.
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_S_POINTER, 0x0e)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, dw)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0)
  # emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)  # old single-task stream; absent from the working ADD PC chain
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x0e)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, dw)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0)
  # emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0x70007)  # old single-task value
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0x7)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.out_slot)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  if sub_slots:
    emitter_reloc(cmds, relocs, sub_slots[0])
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, sub_slots[1])
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[Ops.SUB])
  elif reciprocal:
    # RECIPROCAL(x) = 1/x: swap operands — CONST(1) as input (RDMA_SRC), INDEX as weight (RDMA_EW)
    emitter_reloc(cmds, relocs, _CONST_SLOT, struct.unpack('<I', struct.pack('<f', reciprocal[1]))[0])
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, reciprocal[0])
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[Ops.FDIV])
  elif scalar:
    slot, const_val, swap = scalar
    if swap:
      # FDIV(CONST, INDEX) = CONST/INDEX: CONST as input (RDMA_SRC), INDEX as weight (RDMA_EW)
      emitter_reloc(cmds, relocs, _CONST_SLOT, struct.unpack('<I', struct.pack('<f', const_val))[0])
      emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
      emitter_reloc(cmds, relocs, slot)
    else:
      # Normal: INDEX as input (RDMA_SRC), CONST as weight (RDMA_EW)
      emitter_reloc(cmds, relocs, slot)
      emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
      emitter_reloc(cmds, relocs, _CONST_SLOT, struct.unpack('<I', struct.pack('<f', const_val))[0])
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[val.op])
  elif is_copy:
    emitter_reloc(cmds, relocs, vu.src[0].buf_uop.arg.slot)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, 0)  # EW disabled — DMA pass-through
  elif is_fill:
    emitter_reloc(cmds, relocs, _ZERO_SLOT)
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, _CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(vu.arg)))[0])
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[Ops.ADD])  # zero + const = fill
  elif abs_slot is not None:
    # abs(x) = max(x, -x): both RDMA_SRC and RDMA_EW point to x, BS negates input, EW=MAX
    emitter_reloc(cmds, relocs, abs_slot)
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, abs_slot)  # same buffer for EW (weight)
    # BS mul by -1 (output = -x), BN bypassed (output = x), EW MAX (output = max(-x, x) = abs(x))
    # Initialize the complete BS/BN/WDMA state. PPU submissions leave values that
    # make the shorter abs stream time out even after a subsequent ordinary DPU task.
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, 0x42)  # BS_RELU_BYPASS|BS_ALU_BYPASS (mul enabled, BS not bypassed)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_ALU_CFG, 0)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_MUL_CFG, 0xBC00 << 16)  # fp16 -1.0 at bits 16-31
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_OW_CFG, 2)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, 0x53)  # BN_RELU_BYPASS|BN_MUL_BYPASS|BN_ALU_BYPASS|BN_BYPASS
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_MUL_CFG, 0)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_0, 7)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_1, dw)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_SURFACE_ADD, 0x40)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[Ops.MAX])
  else:
    emitter_reloc(cmds, relocs, _unwrap(val.src[0]).src[0].buf_uop.arg.slot)
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, _unwrap(val.src[1]).src[0].buf_uop.arg.slot)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[val.op])
  # OUT_CVT_SCALE: always emit (reference elementwise.py emits for all ops)
  # FDIV: scale=1 (no FP32TOFP16), others: (1<<16)|1 (FP32TOFP16_EN=1, scale=1)
  if ew_op is Ops.FDIV:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 1)
  else:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
  # FDIV: MRDMA_FP16TOFP32_EN=0 (bit 3 clear) — division runs in fp16, no fp32 conversion
  rdma_fmc = 0x17849 if ew_op is not Ops.FDIV else 0x17841
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_fmc)
  emitter_pc_op_en(cmds, 12)
  # PC chaining is sensitive to register order. Keep the branch-oriented emission above for
  # reference, then canonicalize to the sequence proven by multicore_elementwise.py.
  reg_order = {(t, r): i for i, (t, r) in enumerate((
    (_T_DPU, rk.REG_DPU_S_POINTER), (_T_DPU, rk.REG_DPU_FEATURE_MODE_CFG), (_T_DPU, rk.REG_DPU_DATA_FORMAT),
    (_T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH), (_T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT), (_T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR),
    (_T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL), (_T_DPU, rk.REG_DPU_BS_CFG), (_T_DPU, rk.REG_DPU_BS_ALU_CFG),
    (_T_DPU, rk.REG_DPU_BS_MUL_CFG), (_T_DPU, rk.REG_DPU_BS_OW_CFG), (_T_DPU, rk.REG_DPU_WDMA_SIZE_0),
    (_T_DPU, rk.REG_DPU_WDMA_SIZE_1), (_T_DPU, rk.REG_DPU_BN_CFG), (_T_DPU, rk.REG_DPU_BN_MUL_CFG),
    (_T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE), (_T_DPU, rk.REG_DPU_EW_CFG), (_T_DPU, rk.REG_DPU_OUT_CVT_SCALE),
    (_T_DPU, rk.REG_DPU_SURFACE_ADD),
    (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER), (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH),
    (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT), (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL),
    (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG), (_T_DPU, rk.REG_DPU_DST_BASE_ADDR),
    (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR), (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR),
    (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG), (_T_PC, rk.REG_PC_OPERATION_ENABLE)))}
  ordered = sorted(enumerate(cmds), key=lambda x: (reg_order.get((x[1].target, x[1].reg), len(reg_order)), x[0]))
  old_to_new = {old: new for new, (old, _) in enumerate(ordered)}
  cmds = [cmd for _, cmd in ordered]
  relocs = [RKReloc(old_to_new[r.word_index], r.globals_slot, r.addend, r.shift, r.mask, r.field_shift) for r in relocs]
  return tuple(c.pack() for c in cmds), RKTask(0x18, 0x300, 4, "dpu", layout, plan.out_slot, is_copy=is_copy, is_fill=is_fill,
                                               fp32_inputs=plan.fp32_inputs, fp32_output=plan.fp32_output), tuple(relocs)

def _emit_dpu_lut(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """DPU LUT op (EXP2). Register sequence from ref/rk3588/experimental/kernel_6_18/silu.py.
  Uses LUT table lookup with BN_MUL index scaling and OUT_CVT FP32→FP16 conversion.
  The DPU LUT always processes a full 16x8 tile (128 elements) regardless of logical size."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  sink = plan.sink
  store = _store_node(sink)
  assert store is not None, "dpu_lut: no STORE node"
  val = _unwrap(store.src[1])
  total = 1
  for s in _shape_of_store(sink): total *= s
  lut_result = _try_lut(val)
  assert lut_result is not None, "dpu_lut: no LUT slot"
  lut_slot = lut_result[0]
  input_scale = plan.input_scale  # from classifier (for EXP2(MUL(x, CONST)) etc.)
  output_scale_factor = plan.output_scale  # from classifier (for MUL(LOG2(x), CONST) etc.)
  lut_op = plan.lut_op  # which LUT builder to use
  layout = (total,)
  # DPU LUT: use dw for width (like regular DPU), channel=7 (8 channels)
  dw = (total + 7) // 8 - 1
  width = dw
  # DST_SURF_STRIDE = (width+1) * (channel+1) * 2 bytes
  surf_stride = (width + 1) * 8 * 2
  # --- LUT table fill (513 entries × 2 tables) ---
  if lut_op is Ops.EXP2:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_exp2_lut(input_scale)
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # HYBRID_PRIORITY, OFLOW_PRIORITY, LO_LE_MUX=2
  elif lut_op is _LUT_EXP_CORRECTION:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_exp_correction_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_HARDSWISH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_hardswish_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_HARDSWISH_CORRECTION:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_hardswish_correction_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is Ops.LOG2:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_log2_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif lut_op is Ops.SIN:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sin_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif lut_op is Ops.SQRT:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sqrt_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif lut_op is Ops.RECIPROCAL:
    # RECIPROCAL(SQRT(x)) → RSQRT LUT
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_rsqrt_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif lut_op is _LUT_SIGMOID:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sigmoid_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ROUNDOFF:
    lut = [0 if i % 2 == 0 else 1 << 14 for i in range(_LUT_SIZE)] * 2
    bn_mul_operand, minus_exp = 0, 0
    lut_le_start, lut_lo_end = 0x00000000, 0x44800000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  else:
    raise AssertionError(f"dpu_lut: no builder for {lut_op}")
  # Apply output_scale_factor via OUT_CVT_SCALE (Q15 fixed-point) — see below
  for table_id, base in ((0, 0), (1, _LUT_SIZE)):
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_ACCESS_CFG,
      (1 << 17) | (table_id << 16))  # LUT_ACCESS_TYPE=1, TABLE_ID, ADDR=0
    for i in range(_LUT_SIZE):
      v = int(lut[base + i])
      data = v & 0xFFFF
      if v < 0: data |= 0xFFFF0000  # sign-extend negative values in RESERVED_0
      emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_ACCESS_DATA, data)
  # --- S_POINTER (PP_CLEAR) — EXECUTER_PP_CLEAR=bit5, POINTER_PP_CLEAR=bit4 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_S_POINTER, (1 << 5) | (1 << 4))
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, (1 << 5) | (1 << 4))
  # --- FEATURE_MODE_CFG: burst_len=15, output_mode=2, flying_mode=1 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1) | 1)
  # --- DATA_FORMAT: fp16 in/out/proc ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, (2 << 26) | (2 << 29) | 2)
  # --- DST_BASE_ADDR (relocated) ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.out_slot)
  # --- DST_SURF_STRIDE = (width+1) * 8 * 2 bytes per surface ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_SURF_STRIDE, surf_stride)
  # --- DATA_CUBE_WIDTH/CHANNEL — always 16x8 tile ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, (7 << 16) | 7)
  # --- BS_CFG: all bypassed ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1)
  # --- BS_OW_CFG: OD_BYPASS=1 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_OW_CFG, (1 << 1))
  # --- WDMA_SIZE_0/1 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_0, 7)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_1, width)
  if lut_op is _LUT_ROUNDOFF:
    # Algorithm 23: bypass BN and use LUT interpolation directly as fp16 output.
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, 0x53)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 1))
  else:
    # --- BN_CFG: BN_ALU_ALGO=2, BN_RELU_BYPASS=1 ---
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, (2 << 16) | (1 << 6))
    # --- BN_ALU_CFG ---
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000)
    # --- BN_MUL_CFG: fp16(index_scale) — LUT builder already accounts for input_scale ---
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_MUL_CFG, (bn_mul_operand & 0xFFFF) << 16)
    # --- EW_CFG: relu_bypass=1, op_cvt_bypass=1, op_bypass=1 (LUT mode) ---
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 1))
    # --- EW_CVT_SCALE_VALUE: EW_OP_CVT_SCALE=1 ---
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)
  # --- OUT_CVT_SCALE: FP32TOFP16_EN=1, scale=output_scale_factor (as Q15 fixed-point) ---
  # --- OUT_CVT_SHIFT: MINUS_EXP (FP16 float division by 2^minus_exp) + OUT_CVT_SHIFT (integer right shift) ---
  if lut_op is _LUT_ROUNDOFF:
    pass
  elif output_scale_factor != 1.0:
    # Use Q15 fixed-point: scale = factor * 32768, shift = 15
    scale_q15 = int(round(output_scale_factor * 32768)) & 0xFFFF
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | scale_q15)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SHIFT, (minus_exp << 12) | 15)
  else:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SHIFT, (minus_exp << 12))
  # --- SURFACE_ADD = 2 * surf_stride ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_SURFACE_ADD, 2 * surf_stride)
  # --- unknown reg 0x40c4 = 0 (from silu.py) ---
  emitter_emit(cmds, _T_DPU, 0x40c4, 0)
  # --- LUT config registers ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_CFG, lut_cfg)
  index_select = 14 if lut_op is _LUT_ROUNDOFF else 5
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_INFO, (index_select << 16) | (index_select << 8))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LE_START, lut_le_start)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LE_END, 0x44000000 if lut_op is _LUT_ROUNDOFF else 0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_START, 0x44000000 if lut_op is _LUT_ROUNDOFF else 0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_END, lut_lo_end)
  if lut_op is _LUT_ROUNDOFF:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE, 23107)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT, 22)
  elif lut_op in (_LUT_EXP_CORRECTION, _LUT_HARDSWISH, _LUT_HARDSWISH_CORRECTION):
    # These LUTs use flat endpoint values; their staged epilogues handle the
    # behavior outside the table domain.
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 0)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 0)
  else:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 16434 << 16)  # OFLOW_SCALE=16434
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 13 << 5)  # OFLOW_SHIFT=13
  # --- RDMA config ---
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7)
  # --- RDMA SRC_BASE_ADDR (relocated — input) ---
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, lut_slot)
  # --- ERDMA_CFG: ERDMA_DISABLE=1 ---
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1)
  # --- RDMA FEATURE_MODE_CFG: IN_PRECISION=2, BURST_LEN=15, PROC_PRECISION=2, MRDMA_FP16TOFP32_EN=1, FLYING_MODE=1 ---
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849)
  # --- RDMA_WEIGHT: all weights=1 ---
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101)
  # --- PC_OPERATION_ENABLE ---
  emitter_pc_op_en(cmds, 12)
  return tuple(c.pack() for c in cmds), RKTask(0x18, 0x300, 4, "dpu_lut", layout, plan.out_slot,
                                               fp32_inputs=plan.fp32_inputs, fp32_output=plan.fp32_output), tuple(relocs)

def _emit_cmac(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """CNA+CORE matmul or sum. A=(M,K), B=(K,N), output=(M,N) FP32→FP16. Transforms in __call__."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  sink = plan.sink
  reduce = _reduce_node(sink)
  assert reduce is not None, "cmac: no REDUCE node"
  body = _reduce_body(reduce)
  is_sum = body.op is Ops.INDEX or (body.op is Ops.MUL and _try_sum(sink, reduce) is not None)  # sum or scaled sum
  cv = 1.0
  if is_sum:
    sum_info = _try_sum(sink, reduce)
    assert sum_info is not None, "cmac: sum classification failed"
    input_slot, M_sum, N_sum, _, cv = sum_info
    a_slot, b_slot = (_CONST_SLOT, input_slot) if M_sum == 1 else (input_slot, _CONST_SLOT)
  else:
    a_idx_node, b_idx_node = _unwrap(body.src[0]), _unwrap(body.src[1])
    a_slot, b_slot = a_idx_node.src[0].buf_uop.arg.slot, b_idx_node.src[0].buf_uop.arg.slot
  out_shape = _shape_of_store(sink)
  if is_sum: M, N = M_sum, N_sum
  elif len(out_shape) == 2: M, N = int(out_shape[0]), int(out_shape[1])
  else:  # GEMV: vector is A (M=1) or B (N=1)
    M, N = (1, int(out_shape[0])) if _is_1d_index(a_idx_node.src[1], "REDUCE") else (int(out_shape[0]), 1)  # type: ignore[union-attr]
  K = _reduce_extent(reduce)
  if K < 0: raise RuntimeError("cmac: K must be compile-time constant")
  if not is_sum:
    # Detect transposed pattern (1x1 conv): A has REDUCE outer, B has LOOP outer
    if len(out_shape) == 2 and _is_2d_index(a_idx_node.src[1], "REDUCE", "LOOP", N) and \
       _is_2d_index(b_idx_node.src[1], "LOOP", "REDUCE", K):
      a_slot, b_slot = b_slot, a_slot  # swap: hardware expects A=(M,K), B=(K,N)
  # NPU geometry constants from gemm.py
  CBUF_BANK_SIZE = 256 * 128  # 32 KiB
  RK_CBUF_BANKS = 12
  MIN_CHANNEL_TILE = 32
  RK_LINE_STRIDE_GROUP_CAP = 13
  # layout: align K and N to 32
  aligned_k = max(MIN_CHANNEL_TILE, round_up(K, MIN_CHANNEL_TILE))
  align_out = max(MIN_CHANNEL_TILE, round_up(N, MIN_CHANNEL_TILE))
  align_in = max(aligned_k, align_out)
  eff_k = align_in if align_in != aligned_k else K
  input_row_bytes = align_in * 2
  # feature grains and line stride from gemm.py
  even_rows_per_two_banks = (ceildiv(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
  feature_grains = max(80, even_rows_per_two_banks)
  line_stride = 4 * min(ceildiv(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
  notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
  data_banks = max(1, ceildiv(M * input_row_bytes, CBUF_BANK_SIZE))
  wt_banks = RK_CBUF_BANKS - data_banks
  if data_banks > RK_CBUF_BANKS-1 or input_row_bytes*align_out > wt_banks*CBUF_BANK_SIZE:
    raise RuntimeError("RKPLAN_REJECT:cmac_exceeds_cbuf")
  # --- exact register sequence from gemm.py make_gemm_regs ---
  # 1. DPU S_POINTER
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  # 2. CNA CONV_CON1: IN_PRECISION=2, PROC_PRECISION=2, GROUP_LINE_OFF=1
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CONV_CON1, (2<<4)|(2<<7)|(1<<29))
  # 3. CNA CONV_CON2: FEATURE_GRAINS
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CONV_CON2, (feature_grains << 4))
  # 4. CNA CONV_CON3: CONV_Y_STRIDE=1, CONV_X_STRIDE=1
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CONV_CON3, (1<<3)|1)
  # 5-8. CNA data sizes
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE0, (1<<16)|M)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE1, ((align_in-1)<<16)|align_in)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE2, 1)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE3, M)
  # 9-11. CNA weight sizes
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_WEIGHT_SIZE0, input_row_bytes * align_out)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_WEIGHT_SIZE1, input_row_bytes)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_WEIGHT_SIZE2, (1<<24)|(1<<16)|align_out)
  # 12-13. CNA CBUF config
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CBUF_CON0, ((RK_CBUF_BANKS-data_banks)<<4)|data_banks)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CBUF_CON1, ceildiv(align_in, MIN_CHANNEL_TILE))
  # 14-18. CNA CVT config
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_CVT_CON0, (1<<3)|(1<<1)|1)
  for r in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4): emitter_emit(cmds, _T_CNA, r, 1<<16)
  # 19. CNA FEATURE_DATA_ADDR (relocated — input A)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0)
  emitter_reloc(cmds, relocs, a_slot)
  # 20-24. CNA DMA config
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DMA_CON0, (15<<16)|15)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DMA_CON1, line_stride)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DMA_CON2, 0)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_FC_DATA_SIZE0, (1<<16)|M)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_FC_DATA_SIZE1, align_in)
  # 25. CNA DCOMP_ADDR0 (relocated — weight B, NO 0x4000 offset)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DCOMP_ADDR0, 0)
  emitter_reloc(cmds, relocs, b_slot)
  # 26-29. CORE config
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_MISC_CFG, (2<<8)|1)
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_DATAOUT_SIZE_0, ((M-1)<<16)|0)
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out-1)
  emitter_emit(cmds, _T_CORE, 0x3030, 0)  # CORE_RESERVED_3030
  # 30-45. DPU output config (FP32 output: OUT_PRECISION=5, size_e=3)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15<<5)|(2<<1))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, (5<<29)|(2<<26)|2)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.out_slot)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_SURF_STRIDE, (1<<4))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, M-1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch_val<<16)|notch_val)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out-1)<<16)|(align_out-1))
  # BS/BN epilogue fusion: configure BS for relu or scale after CMAC reduce.
  # BS flow: CORE(FP32) → DPU CVT(FP32→FP16) → BS → BN → EW → WDMA
  if plan.epilogue == "relu":
    # BS enabled, ReLU enabled (not bypassed), MUL bypassed, ALU bypassed
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<4)|(1<<1))  # 0x12
  elif plan.epilogue == "scale":
    # BS enabled, ReLU bypassed, MUL enabled (not bypassed), ALU bypassed
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<1))  # 0x42
    # BS_MUL_OPERAND: fp16 scale at bits 16-31
    fp16_scale = struct.unpack('<H', struct.pack('<e', plan.epilogue_scale))[0]
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_MUL_CFG, fp16_scale << 16)
  else:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<4)|(1<<1)|1)  # 0x53 all bypassed
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_OW_CFG, (3<<8)|(3<<5)|(3<<2)|(1<<1))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_0, align_out-1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_1, ((M-1)<<16)|0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, (1<<6)|(1<<4)|(1<<1)|1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, (1<<9)|(1<<8)|(1<<7)|(1<<1)|1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0)  # FP32 output, no conversion
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_SURFACE_ADD, (4<<4))
  # 46. PC_OPERATION_ENABLE: CNA+CORE+DPU (reserved_0=6, op_en=1)
  emitter_emit(cmds, _T_PC, rk.REG_PC_OPERATION_ENABLE, (6<<1)|1)
  layout = (M, N, K, align_in, align_out)
  return tuple(c.pack() for c in cmds), RKTask(0xd, 0x300, 0, "cmac", layout, plan.out_slot, const_val=cv,
                                               fp32_inputs=plan.fp32_inputs, fp32_output=plan.fp32_output), tuple(relocs)

def _emit_ppu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """PPU globalmax. Input (H,W,C) fp16 → (C,) fp16. Raw register values per pool.py.
  PPU processes channels in groups of 8 for FP16; C is padded to a multiple of 8."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  sink = plan.sink
  reduce = _reduce_node(sink)
  assert reduce is not None, "ppu: no REDUCE node"
  out_shape = _shape_of_store(sink)
  channels = int(out_shape[0])
  K = _reduce_extent(reduce)
  if K < 0: raise RuntimeError("ppu: reduce extent must be compile-time constant")
  chan_padded = ((channels + 7) // 8) * 8  # PPU processes 8 channels per group for FP16
  split = _ppu_split_k(K)
  if split is None: raise RuntimeError(f"ppu: cannot split K={K} into (in_h,in_w) with both 2-16")
  in_h, in_w = split
  in_w_field, in_h_field, channel_field = in_w - 1, in_h - 1, chan_padded - 1
  width_stride = in_w * chan_padded * 2  # bytes
  # --- exact register sequence from pool.py pooling_regs("globalmax") ---
  # All values are raw 32-bit, written directly (no _f field shifting)
  # 1-2. S_POINTER
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  # 3-5. PPU input cube (zero-based)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, in_w_field)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, in_h_field)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, channel_field)
  # 6-8. PPU output cube (global: 0,0,7)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_OUT_WIDTH, 0)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_OUT_HEIGHT, 0)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, channel_field)
  # 9. OPERATION_MODE_CFG: flying=1, pooling_method=1 (MAX)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_OPERATION_MODE_CFG, (1<<4)|1)
  # 10. POOLING_KERNEL_CFG: (s_h<<20)|(s_w<<16)|(k_h<<8)|k_w — global: kernel=stride=full input
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (in_h_field<<20)|(in_w_field<<16)|(in_h_field<<8)|in_w_field)
  # 11. DST_BASE_ADDR (relocated — pool.py writes (output_dma // 16) << 4)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DST_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.out_slot, shift=4, mask=0xFFFFFFF, field_shift=4)
  # 12. DST_SURF_STRIDE (raw value — pool.py writes dst_surf_stride directly)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DST_SURF_STRIDE, 1)
  # 13. DATA_FORMAT (raw — pool.py writes (index_add << 16) | 2; autogen INDEX_ADD shift=4 is WRONG)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_DATA_FORMAT, (1 << 16) | 2)
  # 14. MISC_CTRL (raw — pool.py writes 3 for burst_len)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_MISC_CTRL, 3)
  # 15-17. RDMA input cube (zero-based)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH, in_w_field)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, in_h_field)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, channel_field)
  # 18. RDMA SRC_BASE_ADDR (relocated — pool.py writes raw input_dma)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.in_slots[0])
  # 19-20. RDMA strides (raw bytes — pool.py writes width_stride/src_surf_stride directly;
  #   hardware field at shift=4 divides by 16 to get 16-byte units)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, width_stride)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, width_stride * in_h)
  # 21. RDMA DATA_FORMAT (raw — pool.py writes 2 for fp16)
  emitter_emit(cmds, _T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_DATA_FORMAT, 2)
  # 22. RDMA OPERATION_ENABLE (raw — pool.py writes 1)
  # NOTE: autogen has wrong address 0x7008; correct is 0x7038 (see ref/rk3588/experimental/pool.py)
  emitter_emit(cmds, _T_PPU_RDMA, 0x7038, 1)
  # 23-24. RECIP_KERNEL_WIDTH/HEIGHT (raw — pool.py writes 0 for max pool, 30720 for avg)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_RECIP_KERNEL_WIDTH, 0)
  emitter_emit(cmds, _T_PPU, rk.REG_PPU_RECIP_KERNEL_HEIGHT, 0)
  # 25. PC_OPERATION_ENABLE: PPU-only mode (reserved_0=48, op_en=0 — pool.py uses <<1 without |1)
  emitter_pc_op_en(cmds, 48)
  return tuple(c.pack() for c in cmds), RKTask(0x60, 0xc00, 1, "ppu", (in_h, in_w, channels, chan_padded), plan.out_slot,
                                               fp32_inputs=plan.fp32_inputs, fp32_output=plan.fp32_output), tuple(relocs)

# NOTE: _emit_ppu_pool2d was an attempt at PPU windowed pooling for max_pool2d.
# The PPU requires HWC layout with C=8 (padded to atom size), but tinygrad uses NCHW.
# For C=1, NCHW and HWC layouts differ, and the PPU's atom-based RDMA reads 8 channels
# per pixel, producing wrong results with NCHW data. This needs a data layout transform
# (multi-task) to pack data as HWC with C=8 padding before the PPU can process it.
#
# def _flatten_add(u: UOp) -> list[UOp]:
#   """Flatten an ADD tree into a list of leaf terms."""
#   if u.op is Ops.ADD: return _flatten_add(u.src[0]) + _flatten_add(u.src[1])
#   return [u]
#
# def _try_pool2d(sink, reduce):
#   """Detect windowed max_pool2d: REDUCE(MAX, INDEX(input, idx)) with 2 REDUCE ranges.
#   For C=1 NCHW, idx = reduce_h*W + out_h*sh*W + out_w*sw + reduce_w.
#   Returns (input_slot, in_h, in_w, kh, kw, sh, sw, out_h, out_w) or None."""
#   ...
#
# def _emit_ppu_pool2d(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
#   ...

# ---- emit_rk dispatcher (§2.3 API) ----
def emit_rk(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """Dispatch emission by plan.kind. Returns (cmds, task, relocs)."""
  return {"dpu":_emit_dpu, "dpu_lut":_emit_dpu_lut, "cmac":_emit_cmac, "ppu":_emit_ppu}[plan.kind](plan)

# ---- codec ----
_RK_MAGIC = 0x524b494d  # "RKIM"
_RK_KINDS = ("dpu", "dpu_lut", "cmac", "ppu")

def encode_rk(cmds: tuple[int,...], task: RKTask, relocs: tuple[RKReloc,...]) -> bytes:
  """Pack commands, task metadata, and relocations into deterministic bytes (§2.3).
  Version 3 adds fp32_inputs/fp32_output after const_val."""
  n_cmds, n_relocs, n_layout = len(cmds), len(relocs), len(task.layout)
  kind_layout = (_RK_KINDS.index(task.kind) << 24) | ((1 if task.is_copy else 0) << 23) | ((1 if task.is_fill else 0) << 22) | n_layout
  header = struct.pack("<IIIIIIIIi", _RK_MAGIC, 3, n_cmds, n_relocs, task.enable_mask, task.int_mask, task.op_idx, kind_layout, task.out_slot)
  reloc_bytes = struct.pack(f"<{n_relocs*6}I", *[v for r in relocs for v in (r.word_index, r.globals_slot, r.addend, r.shift, r.mask, r.field_shift)])
  layout = struct.pack(f"<{n_layout}i", *task.layout) if task.layout else b""
  const_bytes = struct.pack("<f", task.const_val)
  # fp32_mask: bit 7 = fp32_output, bits 0-6 = fp32_inputs mask (bit N = slot N is fp32)
  fp32_mask = (1 << 7 if task.fp32_output else 0) | sum(1 << s for s in task.fp32_inputs if s < 7)
  fp32_bytes = struct.pack("<B", fp32_mask)
  return header + struct.pack(f"<{n_cmds}Q", *cmds) + reloc_bytes + layout + const_bytes + fp32_bytes

def decode_rk(data: bytes) -> tuple[list[int], RKTask, list[RKReloc]]:
  """Decode bytes into (cmds, task, relocs). Raises on bad magic, version, truncated tables, or out-of-range relocations."""
  if len(data) < 36: raise RuntimeError("RKImage: truncated header")
  magic, ver, n_cmds, n_relocs, emask, imask, op_idx, kl, out_slot = struct.unpack_from("<IIIIIIIIi", data, 0)
  if magic != _RK_MAGIC: raise RuntimeError("RKImage: bad magic")
  if ver not in (2, 3): raise RuntimeError(f"RKImage: unsupported version {ver}")
  ki = (kl >> 24) & 0xFF
  if ki >= len(_RK_KINDS): raise RuntimeError(f"RKImage: invalid kind index {ki}")
  is_copy = bool((kl >> 23) & 1)
  is_fill = bool((kl >> 22) & 1)
  n_layout = kl & 0x3FFFFF
  off = 36
  if off + n_cmds * 8 > len(data): raise RuntimeError("RKImage: truncated commands")
  cmds = list(struct.unpack_from(f"<{n_cmds}Q", data, off))
  off += n_cmds * 8
  if off + n_relocs * 24 > len(data): raise RuntimeError("RKImage: truncated relocations")
  relocs = []
  for i in range(n_relocs):
    wi, gs, add, sh, mask, fsh = struct.unpack_from("<IIIIII", data, off + i * 24)
    if wi >= n_cmds: raise RuntimeError(f"RKImage: reloc word_index {wi} out of range ({n_cmds} cmds)")
    relocs.append(RKReloc(wi, gs, add, sh, mask, fsh))
  off += n_relocs * 24
  if n_layout and off + n_layout * 4 > len(data): raise RuntimeError("RKImage: truncated layout")
  layout = struct.unpack_from(f"<{n_layout}i", data, off) if n_layout else ()
  off += n_layout * 4
  # v2: const_val optional; v3: const_val (4 bytes) + fp32_mask (1 byte) always present
  if ver >= 3:
    const_val = struct.unpack_from("<f", data, off)[0] if off + 4 <= len(data) else 1.0
    off += 4
    fp32_mask = struct.unpack_from("<B", data, off)[0] if off < len(data) else 0
    fp32_output = bool(fp32_mask & (1 << 7))
    fp32_inputs = tuple(s for s in range(7) if fp32_mask & (1 << s))
  else:
    const_val = struct.unpack_from("<f", data, off)[0] if off + 4 <= len(data) else 1.0
    fp32_output, fp32_inputs = False, ()
  return cmds, RKTask(emask, imask, op_idx, _RK_KINDS[ki], layout, out_slot, is_copy, is_fill,
                      const_val=const_val, fp32_inputs=fp32_inputs, fp32_output=fp32_output), relocs

def _encode_one_task(task: RKTask, cmds: tuple[int,...], relocs: tuple[RKReloc,...]) -> bytes:
  """Encode a single task's cmds + relocs + metadata (no header)."""
  n_cmds, n_relocs, n_layout = len(cmds), len(relocs), len(task.layout)
  reloc_bytes = struct.pack(f"<{n_relocs*6}I", *[v for r in relocs for v in (r.word_index, r.globals_slot, r.addend, r.shift, r.mask, r.field_shift)])
  layout = struct.pack(f"<{n_layout}i", *task.layout) if task.layout else b""
  const_bytes = struct.pack("<f", task.const_val)
  fp32_mask = (1 << 7 if task.fp32_output else 0) | sum(1 << s for s in task.fp32_inputs if s < 7)
  fp32_bytes = struct.pack("<B", fp32_mask)
  kind_layout = (_RK_KINDS.index(task.kind) << 24) | ((1 if task.is_copy else 0) << 23) | ((1 if task.is_fill else 0) << 22) | n_layout
  dtype_flags = int(task.int32_output) | (int(task.uint8_output) << 1) | (int(task.bool_output) << 2) | (int(task.trunc_output) << 3) | \
    sum(1 << (8+s) for s in task.bool_inputs if s < 16) | sum(1 << (24+s) for s in task.int32_inputs if s < 7)
  input_flags = sum(1 << s for s in task.broadcast_inputs if s < 16) | \
    sum(1 << (16+s) for s in task.comparison_inputs if s < 16)
  task_header = struct.pack("<IIIIIIIIi", n_cmds, n_relocs, task.enable_mask, task.int_mask, task.op_idx, kind_layout,
                            task.out_slot, input_flags, dtype_flags)
  return task_header + struct.pack(f"<{n_cmds}Q", *cmds) + reloc_bytes + layout + const_bytes + fp32_bytes

def _decode_one_task(data: bytes, off: int) -> tuple[list[int], RKTask, list[RKReloc], int]:
  """Decode a single task from offset. Returns (cmds, task, relocs, new_offset)."""
  n_cmds, n_relocs, emask, imask, op_idx, kl, out_slot, input_flags, dtype_flags = struct.unpack_from("<IIIIIIIIi", data, off)
  off += 36
  ki = (kl >> 24) & 0xFF
  is_copy = bool((kl >> 23) & 1)
  is_fill = bool((kl >> 22) & 1)
  n_layout = kl & 0x3FFFFF
  cmds = list(struct.unpack_from(f"<{n_cmds}Q", data, off))
  off += n_cmds * 8
  relocs = []
  for i in range(n_relocs):
    wi, gs, add, sh, mask, fsh = struct.unpack_from("<IIIIII", data, off + i * 24)
    if gs >= 1 << 31: gs -= 1 << 32
    relocs.append(RKReloc(wi, gs, add, sh, mask, fsh))
  off += n_relocs * 24
  layout = struct.unpack_from(f"<{n_layout}i", data, off) if n_layout else ()
  off += n_layout * 4
  const_val = struct.unpack_from("<f", data, off)[0] if off + 4 <= len(data) else 1.0
  off += 4
  fp32_mask = struct.unpack_from("<B", data, off)[0] if off < len(data) else 0
  off += 1
  fp32_output = bool(fp32_mask & (1 << 7))
  fp32_inputs = tuple(s for s in range(7) if fp32_mask & (1 << s))
  bool_inputs = tuple(s for s in range(16) if dtype_flags & (1 << (8+s)))
  int32_inputs = tuple(s for s in range(7) if dtype_flags & (1 << (24+s)))
  broadcast_inputs = tuple(s for s in range(16) if input_flags & (1 << s))
  comparison_inputs = tuple(s for s in range(16) if input_flags & (1 << (16+s)))
  return cmds, RKTask(emask, imask, op_idx, _RK_KINDS[ki], layout, out_slot, is_copy, is_fill,
                      const_val=const_val, fp32_inputs=fp32_inputs, fp32_output=fp32_output,
                      bool_inputs=bool_inputs, int32_inputs=int32_inputs, broadcast_inputs=broadcast_inputs,
                      comparison_inputs=comparison_inputs,
                      int32_output=bool(dtype_flags & 1), uint8_output=bool(dtype_flags & 2),
                      bool_output=bool(dtype_flags & 4), trunc_output=bool(dtype_flags & 8)), relocs, off

def encode_rk_multi(subtasks: tuple[RKSubTask, ...]) -> bytes:
  """Pack multiple tasks into one image (version 4 = PC chain).
  Header: magic, version=4, n_tasks. Then each task encoded back-to-back."""
  n_tasks = len(subtasks)
  header = struct.pack("<III", _RK_MAGIC, 4, n_tasks)
  body = b""
  for st in subtasks:
    body += _encode_one_task(st.task, st.cmds, st.relocs)
  return header + body

def decode_rk_multi(data: bytes) -> list[RKSubTask]:
  """Decode a multi-task image (version 4). Returns list of RKSubTask."""
  if len(data) < 12: raise RuntimeError("RKImage: truncated multi-task header")
  magic, ver, n_tasks = struct.unpack_from("<III", data, 0)
  if magic != _RK_MAGIC: raise RuntimeError("RKImage: bad magic")
  if ver != 4: raise RuntimeError(f"RKImage: expected version 4, got {ver}")
  off = 12
  result = []
  for _ in range(n_tasks):
    cmds, task, relocs, off = _decode_one_task(data, off)
    result.append(RKSubTask(tuple(cmds), task, tuple(relocs)))
  return result

# ---- the native_program hook ----
def build_native_program(sink: UOp) -> UOp|None:
  """Classify and build a PROGRAM(SINK, LINEAR(INS...)). Raises RKPLAN_REJECT:<reason>
  if unsupported (no fallback per §15). Raises if a classified kernel fails emission."""
  # Pre-classification rewrite: MUL(a, RECIPROCAL(b)) → FDIV(a, b)
  sink = graph_rewrite(sink, _pm_fdiv, name="rk fdiv decomp")
  if (cast_tasks := _try_cast_subtasks(sink)) is not None: return build_native_program_multi(sink, cast_tasks)
  if (fill_tasks := _try_typed_fill_subtasks(sink)) is not None: return build_native_program_multi(sink, fill_tasks)
  if (round_tasks := _try_round_subtasks(sink)) is not None: return build_native_program_multi(sink, round_tasks)
  if (sign_tasks := _try_sign_subtasks(sink)) is not None: return build_native_program_multi(sink, sign_tasks)
  if (inf_div_tasks := _try_inf_div_subtasks(sink)) is not None: return build_native_program_multi(sink, inf_div_tasks)
  if (hardsigmoid_tasks := _try_hardsigmoid_subtasks(sink)) is not None: return build_native_program_multi(sink, hardsigmoid_tasks)
  if (hardswish_tasks := _try_hardswish_subtasks(sink)) is not None: return build_native_program_multi(sink, hardswish_tasks)
  if (tanh_tasks := _try_tanh_saturation_subtasks(sink)) is not None: return build_native_program_multi(sink, tanh_tasks)
  if (quick_gelu_tasks := _try_quick_gelu_saturation_subtasks(sink)) is not None: return build_native_program_multi(sink, quick_gelu_tasks)
  if (comparison_tasks := _try_comparison_subtasks(sink)) is not None: return build_native_program_multi(sink, comparison_tasks)
  if (exp_correction_tasks := _try_exp_correction_subtasks(sink)) is not None:
    return build_native_program_multi(sink, exp_correction_tasks)
  if (sigmoid_tasks := _try_sigmoid_special_subtasks(sink)) is not None: return build_native_program_multi(sink, sigmoid_tasks)
  if (exp2_tasks := _try_exp2_special_subtasks(sink)) is not None: return build_native_program_multi(sink, exp2_tasks)
  if (log2_tasks := _try_log2_special_subtasks(sink)) is not None: return build_native_program_multi(sink, log2_tasks)
  if (rsqrt_tasks := _try_rsqrt_special_subtasks(sink)) is not None: return build_native_program_multi(sink, rsqrt_tasks)
  if (sqrt_tasks := _try_sqrt_special_subtasks(sink)) is not None: return build_native_program_multi(sink, sqrt_tasks)
  if (abs_tasks := _try_abs_subtasks(sink)) is not None: return build_native_program_multi(sink, abs_tasks)
  plan = plan_rk(sink)
  if isinstance(plan, str):
    # Preserve directly recognizable single-stage forms such as abs(x) before
    # expanding MUL(WHERE(...)) into the general arithmetic-WHERE representation.
    sink = graph_rewrite(sink, _pm_where_mul, name="rk distribute where mul")
    plan = plan_rk(sink)
  if isinstance(plan, str):
    if (where_tasks := _try_where_subtasks(sink)) is not None: return build_native_program_multi(sink, where_tasks)
    if (elementwise_tasks := _try_elementwise_subtasks(sink)) is not None: return build_native_program_multi(sink, elementwise_tasks)
    raise RuntimeError(plan)  # reject — preserve reason, no fallback
  cmds, task, relocs = emit_rk(plan)
  # each INS carries an int (packed command), RKReloc, or RKTask as its arg
  ins_args = [task] + list(cmds) + list(relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))

def build_native_program_multi(sink: UOp, subtasks: tuple[RKSubTask, ...]) -> UOp|None:
  """Build a multi-task PROGRAM for PC chain. The first INS carries a tuple of RKSubTask,
  followed by all cmds and relocs from all subtasks (flattened)."""
  ins_args: list = [subtasks]  # first INS carries the subtask list
  for st in subtasks:
    ins_args.extend(st.cmds)
    ins_args.extend(st.relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))
