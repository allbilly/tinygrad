# pylint: skip-file
# RK3588 NPU compiled backend: pure match/classify + register emission + codec.
# PR 1 native contract: DPU binary EW/copy, CMAC matmul/sum, PPU global max (fp16 only).
# Fill, broadcast, mean, and all other ops are rejected — no host-side tensor arithmetic.
from __future__ import annotations
import struct, math, numpy as np
from dataclasses import dataclass, replace
from typing import Callable
from tinygrad.dtype import dtypes, DType, Invalid
from tinygrad.helpers import ceildiv, round_up, prod, getenv
from tinygrad.uop.ops import Ops, UOp, AxisType, ProgramInfo, PatternMatcher, graph_rewrite
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
  epilogue_scale_counts: tuple[int, ...] = () # per-output divisors applied to raw fp32 CMAC results
  epilogue_bias_slot: int = -1        # host fp32-accumulator channel bias, before final fp16 rounding
  epilogue_bias_axis: int = -1        # output LOOP axis used to index the bias tensor
  cmac_materialization: tuple[int, ...] = () # serialized non-contiguous A/B/output indexing for CMAC

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
  periodic_input: bool = False        # reduce fp32 inputs modulo 2*pi before fp16 conversion
  fp32_residual_input: bool = False   # convert fp32 input to a 256-scaled residual after its nearest fp16 value
  native_int32_output: bool = False   # DPU WDMA writes already-integral values directly as int32
  native_int32_input: bool = False    # MRDMA consumes one aligned compact int32 atom directly
  out_offset: int = 0                 # byte offset into the output buffer (for cat-like copies)

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
  if getattr(mr.arg[-1], "name", "") != outer_kind or getattr(rs.arg[-1], "name", "") != inner_kind or \
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
_HOST_BITWISE_LAYOUT = -1000  # is_copy task layout tag for exact host-side 32-bit/bool bitwise operations
_HOST_MOVEMENT_LAYOUT = -1001  # is_copy task layout tag for exact integer-indexed host movement
_HOST_TRUNC_LAYOUT = -1002  # is_copy task layout tag for exact root fp16/fp32 truncation
_HOST_COPYSIGN_LAYOUT = -1003  # is_copy task layout tag for exact broadcast fp16/fp32 copysign
_HOST_ELEMENTWISE_LAYOUT = -1004  # is_copy task layout tag for serialized fused elementwise graphs
_CMAC_MATERIALIZED_LAYOUT = -1005  # CMAC layout tag for host-gathered A/B matrices
_HOST_STATIC_HALF_LAYOUT = -1006  # compile-time fp16 tensor used by a later NPU stage
_HOST_SCATTER_LAYOUT = -1007  # typed index scatter used by max-unpool movement
_HOST_ARGMAX_LAYOUT = -1008  # compact candidate map for max-pool returned indices
_HOST_GATHER_MAP_LAYOUT = -1009  # static address-only fp16 gather for NPU reductions
_HOST_PACK_CHUNK_LAYOUT = -1010  # post-DPU alignment pack, value-preserving only
_HOST_UNPACK_INT_CHUNK_LAYOUT = -1011  # compact native int32 atom, value-preserving only
_HOST_PACK_INT32_CHUNK_LAYOUT = -1012  # align compact int32 input atom, value-preserving only
_HOST_UNPACK_HALF_CHUNK_LAYOUT = -1013  # compact native fp16 atom, value-preserving only
_HOST_STATIC_INT_LAYOUT = -1014  # compile-time int32 tensor used by a later NPU stage
_HOST_PLANE_GATHER_LAYOUT = -1015  # repeated per-plane candidate gather, value-preserving only
_HOST_COMPACT_NATIVE_HALF_LAYOUT = -1016  # remove padding lanes after native int32 MRDMA
_HOST_ASSEMBLE_INT_BYTES_LAYOUT = -1017  # compose native 0..255 digits as int32 bytes
_HOST_PACK_HALF_BITS_LAYOUT = -1018  # widen raw fp16 representations into int32 lanes
_HOST_UNPACK_HALF_BITS_LAYOUT = -1019  # compact selected raw fp16 representations
_HOST_BOOL_HALF_LAYOUT = -1020  # widen byte-wide bool ABI storage to fp16 0/1 lanes
_HOST_STATIC_SELECT_HALF_LAYOUT = -1021  # interleave two fp16 buffers by a compile-time lane mask
_HOST_STATIC_SELECT_INT_LAYOUT = -1022  # interleave native int32 atoms by compile-time sort wires
_HOST_HALF_INT_LAYOUT = -1023  # typed fp16-to-int32 ABI boundary after NPU selection
_HOST_FP32_HALF_LAYOUT = -1024  # nearest-fp16 representation view of an fp32 ABI buffer
_HOST_FP32_RESIDUAL_LAYOUT = -1025  # 256-scaled residual representation view of an fp32 ABI buffer
_HOST_FP32_COMBINE_LAYOUT = -1026  # decode high + x256 residual fp16 limbs into an fp32 ABI buffer
_HOST_HALF_FP32_LAYOUT = -1027  # widen an NPU-produced fp16 tile into fp32 ABI storage
_HOST_VARIANCE_LAYOUT = -1028  # strict serialized centered-square fp32 reduction
_HOST_SOFTMAX_ARGMAX_LAYOUT = -1029  # exact global argmax over a scheduled fp32 softmax
_HOST_AVG_POOL_LAYOUT = -1030  # exact bounded normal-fp32 average-pool reduction
_HOST_ELEMENTWISE_REDUCE_LAYOUT = -1031  # compact typed elementwise body plus static reduction axes

def _host_dtype_code(dtype:DType) -> int|None:
  """Stable dtype ids shared by serialized exact host tasks."""
  table = {dtypes.bool:0, dtypes.int:1, dtypes.uint:2, dtypes.long:3, dtypes.ulong:4,
           dtypes.uchar:5, dtypes.half:6, dtypes.float:7, dtypes.weakint:8,
           dtypes.short:9, dtypes.ushort:10, dtypes.char:11, dtypes.double:12, dtypes.weakfloat:12}
  return table.get(dtype)

def _signed_i32(value:int) -> int: return value if value < 1 << 31 else value-(1 << 32)

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

def _try_logsigmoid(val:UOp) -> UOp|None:
  """Recognize -logaddexp(0, -x), tinygrad's stable LogSigmoid lowering."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(_unwrap(u) for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  ch, cf = lambda x: UOp.const(dtypes.half, x), lambda x: UOp.const(dtypes.float, x)
  neg = UOp(Ops.MUL, dtypes.half, (source, ch(-1.0)))
  positive = UOp(Ops.MAX, dtypes.half, (neg, ch(0.0)))
  neg_positive = UOp(Ops.MUL, dtypes.half, (positive, ch(-1.0)))
  lhs = UOp(Ops.CAST, dtypes.float, (UOp(Ops.ADD, dtypes.half, (neg, neg_positive)),), arg=dtypes.float)
  rhs = UOp(Ops.CAST, dtypes.float, (neg_positive,), arg=dtypes.float)
  exp_lhs = UOp(Ops.CAST, dtypes.half, (UOp(Ops.EXP2, dtypes.float,
    (UOp(Ops.MUL, dtypes.float, (lhs, cf(math.log2(math.e)))),)),), arg=dtypes.half)
  exp_rhs = UOp(Ops.CAST, dtypes.half, (UOp(Ops.EXP2, dtypes.float,
    (UOp(Ops.MUL, dtypes.float, (rhs, cf(math.log2(math.e)))),)),), arg=dtypes.half)
  logarithm = UOp(Ops.MUL, dtypes.half, (
    UOp(Ops.LOG2, dtypes.half, (UOp(Ops.ADD, dtypes.half, (exp_lhs, exp_rhs)),)), ch(math.log(2.0))))
  expected = UOp(Ops.MUL, dtypes.half, (UOp(Ops.ADD, dtypes.half, (logarithm, positive)), ch(-1.0)))
  if val is expected: return source
  # Kernel optimization can reassociate the stable logaddexp graph before the
  # renderer hook. Retain a narrow semantic fallback for that equivalent form.
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CAST, Ops.EXP2, Ops.LOG2, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  root_negated = val.op is Ops.MUL and any(x.op is Ops.CONST and float(x.arg) == -1.0 for x in val.src)
  if val.op is Ops.ADD and all(_unwrap(x).op is Ops.MUL for x in val.src):
    terms = [_unwrap(x) for x in val.src]
    term_ops = {next((_unwrap(y).op for y in term.src if y.op is not Ops.CONST), None) for term in terms}
    term_scales = sorted(float(y.arg) for term in terms for y in term.src if y.op is Ops.CONST)
    root_negated = term_ops == {Ops.LOG2, Ops.MAX} and len(term_scales) == 2 and \
      math.isclose(term_scales[0], -1.0) and math.isclose(term_scales[1], -math.log(2.0))
  return source if root_negated and all(u.op in allowed for u in nodes) and \
    sum(u.op is Ops.EXP2 for u in nodes) == 2 and sum(u.op is Ops.LOG2 for u in nodes) == 1 and \
    sum(u.op is Ops.MAX for u in nodes) == 1 else None

def _try_softplus(val:UOp) -> tuple[UOp,float]|None:
  """Recognize logaddexp(x, 0), tinygrad's stable Softplus lowering."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  beta, base = 1.0, val
  if val.op is Ops.MUL:
    constant = next((x for x in val.src if x.op is Ops.CONST), None)
    inner = next((_unwrap(x) for x in val.src if x.op is not Ops.CONST), None)
    if constant is None or inner is None or inner.op is not Ops.ADD or float(constant.arg) <= 0: return None
    beta, base = 1.0/float(constant.arg), inner
  if base.op is not Ops.ADD: return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CAST, Ops.EXP2, Ops.LOG2, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  logarithm = next((_unwrap(x) for x in base.src if _unwrap(x).op is Ops.MUL and
    any(y.op is Ops.CONST and math.isclose(float(y.arg), math.log(2.0)) for y in _unwrap(x).src)), None)
  maximum = next((_unwrap(x) for x in base.src if _unwrap(x).op is Ops.MAX), None)
  if maximum is None: return None
  maximum_value = next((_unwrap(x) for x in maximum.src if not (x.op is Ops.CONST and float(x.arg) == 0.0)), None)
  scaled_source_ok = maximum_value is source or (maximum_value is not None and maximum_value.op is Ops.MUL and
    any(_unwrap(x) is source for x in maximum_value.src) and
    any(x.op is Ops.CONST and math.isclose(float(x.arg), beta) for x in maximum_value.src))
  # BCE-with-logits exposes softplus(-x) after the outer negation distributes.
  # Preserve that effective input so the subtask builder can materialize -x
  # before applying the same NPU-native softplus LUT path.
  negative_source = maximum_value if maximum_value is not None and maximum_value.op is Ops.MUL and \
    any(_unwrap(x) is source for x in maximum_value.src) and \
    any(x.op is Ops.CONST and math.isclose(float(x.arg), -beta) for x in maximum_value.src) else None
  effective_source = source if scaled_source_ok else negative_source
  return (effective_source, beta) if logarithm is not None and effective_source is not None and \
    any(_unwrap(x).op is Ops.LOG2 for x in logarithm.src) and all(u.op in allowed for u in nodes) and \
    sum(u.op is Ops.EXP2 for u in nodes) == 2 and sum(u.op is Ops.LOG2 for u in nodes) == 1 and \
    sum(u.op is Ops.MAX for u in nodes) == 1 else None

def _try_mish(val:UOp) -> UOp|None:
  """Recognize x*tanh(softplus(x)), tinygrad's Mish lowering."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half or val.op is not Ops.MUL or \
     not any(_unwrap(x) is source for x in val.src): return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.MAX, Ops.CAST, Ops.EXP2, Ops.LOG2, Ops.RECIPROCAL,
             Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return source if all(u.op in allowed for u in nodes) and sum(u.op is Ops.EXP2 for u in nodes) == 3 and \
    sum(u.op is Ops.LOG2 for u in nodes) == 1 and sum(u.op is Ops.MAX for u in nodes) == 1 and \
    sum(u.op in (Ops.RECIPROCAL, Ops.FDIV) for u in nodes) == 1 else None

def _try_elu(val:UOp) -> tuple[UOp,float,float]|None:
  """Recognize ELU/SELU and return source, negative scale, positive scale."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.WHERE, Ops.CMPLT, Ops.CAST, Ops.EXP2,
             Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  if not all(u.op in allowed for u in nodes) or sum(u.op is Ops.EXP2 for u in nodes) != 1: return None
  # ELU: relu(x) - alpha*relu(1-exp(x)).
  if val.op is Ops.ADD and sum(u.op is Ops.WHERE for u in nodes) == 2 and sum(u.op is Ops.CMPLT for u in nodes) == 2:
    scaled = next((_unwrap(x) for x in val.src if _unwrap(x).op is Ops.MUL), None)
    if scaled is None: return None
    alpha = next((abs(float(x.arg)) for x in scaled.src if x.op is Ops.CONST), None)
    return (source, alpha, 1.0) if alpha is not None else None
  # SELU: gamma*where(x<0, alpha*expm1(x), x).
  if val.op is Ops.MUL and sum(u.op is Ops.WHERE for u in nodes) == 1 and sum(u.op is Ops.CMPLT for u in nodes) == 1:
    gamma = next((float(x.arg) for x in val.src if x.op is Ops.CONST), None)
    where = next((_unwrap(x) for x in val.src if _unwrap(x).op is Ops.WHERE), None)
    if gamma is None or where is None: return None
    negative = next((_unwrap(x) for x in where.src[1:] if _unwrap(x) is not source and _unwrap(x).op is Ops.MUL), None)
    if negative is None: return None
    alpha = next((abs(float(x.arg)) for x in negative.src if x.op is Ops.CONST), None)
    return (source, gamma*alpha, gamma) if alpha is not None else None
  return None

def _try_erf(val:UOp) -> UOp|None:
  """Recognize tinygrad's Abramowitz-Stegun erf approximation."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half or val.op is not Ops.MUL: return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.RECIPROCAL, Ops.WHERE, Ops.CMPLT, Ops.CMPNE,
             Ops.CAST, Ops.EXP2, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return source if all(u.op in allowed for u in nodes) and sum(u.op is Ops.EXP2 for u in nodes) == 1 and \
    sum(u.op in (Ops.FDIV, Ops.RECIPROCAL) for u in nodes) == 5 and sum(u.op is Ops.WHERE for u in nodes) == 2 and \
    sum(u.op is Ops.CMPLT for u in nodes) == 1 and sum(u.op is Ops.CMPNE for u in nodes) == 1 and \
    any(u.op is Ops.CONST and math.isclose(float(u.arg), 0.3275911) for u in nodes) else None

def _try_gelu(val:UOp) -> tuple[UOp,bool]|None:
  """Recognize tanh and exact GELU lowerings; bool selects tanh approximation."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  nodes = val.toposort()
  if val.op is Ops.FDIV and sum(u.op is Ops.EXP2 for u in nodes) == 1 and \
     sum(u.op is Ops.FDIV for u in nodes) == 1 and sum(u.op is Ops.WHERE for u in nodes) == 0 and \
     any(u.op is Ops.CONST and math.isclose(float(u.arg), 0.044715) for u in nodes): return source, True
  if val.op is Ops.MUL and sum(u.op is Ops.EXP2 for u in nodes) == 1 and \
     sum(u.op in (Ops.FDIV, Ops.RECIPROCAL) for u in nodes) == 5 and sum(u.op is Ops.WHERE for u in nodes) == 2 and \
     any(u.op is Ops.CONST and math.isclose(float(u.arg), 1/math.sqrt(2)) for u in nodes): return source, False
  return None

def _try_asin_acos(val:UOp) -> tuple[UOp,bool]|None:
  """Recognize tinygrad's asin polynomial, optionally wrapped as pi/2-asin(x)."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype not in (dtypes.half, dtypes.float): return None
  core, is_acos = val, False
  if val.op is Ops.ADD:
    pi_half = next((x for x in val.src if x.op is Ops.CONST and math.isclose(float(x.arg), math.pi/2)), None)
    negated = next((_unwrap(x) for x in val.src if x is not pi_half), None)
    if pi_half is None or negated is None or negated.op is not Ops.MUL: return None
    minus_one = next((x for x in negated.src if x.op is Ops.CONST and float(x.arg) == -1.0), None)
    core = next((_unwrap(x) for x in negated.src if x is not minus_one), negated)
    if minus_one is None: return None
    is_acos = True
  if core.op is not Ops.MUL: return None
  sign = next((_unwrap(x) for x in core.src if _unwrap(x).op is Ops.WHERE), None)
  angle = next((_unwrap(x) for x in core.src if _unwrap(x).op is Ops.ADD), None)
  if sign is None or angle is None: return None
  sign_indexes = list(dict.fromkeys(u for u in sign.toposort() if u.op is Ops.INDEX))
  nodes = core.toposort()
  coefficients = (-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                  -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)
  allowed = {Ops.ADD, Ops.MUL, Ops.SQRT, Ops.WHERE, Ops.CMPLT, Ops.CMPNE,
             Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return (source, is_acos) if sign_indexes == [source] and all(u.op in allowed for u in nodes) and \
    sum(u.op is Ops.SQRT for u in nodes) == 1 and sum(u.op is Ops.WHERE for u in nodes) == 2 and \
    all(any(u.op is Ops.CONST and math.isclose(float(u.arg), coefficient) for u in nodes) for coefficient in coefficients) else None

def _try_atan(val:UOp) -> UOp|None:
  """Recognize atan(x) lowered as asin(x/sqrt(1+x*x))."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half or val.op is not Ops.MUL: return None
  sign = next((_unwrap(x) for x in val.src if _unwrap(x).op is Ops.WHERE), None)
  if sign is None or _unwrap(sign.src[0]).op is not Ops.CMPNE: return None
  candidate = next((_unwrap(x) for x in sign.src[0].src if _unwrap(x).op is not Ops.CONST), None)
  if candidate is None: return None
  if candidate.op is Ops.FDIV:
    if _unwrap(candidate.src[0]) is not source or (root := _unwrap(candidate.src[1])).op is not Ops.SQRT: return None
  elif candidate.op is Ops.MUL:
    if not any(_unwrap(x) is source for x in candidate.src): return None
    reciprocal = next((_unwrap(x) for x in candidate.src if _unwrap(x).op is Ops.RECIPROCAL), None)
    if reciprocal is None or (root := _unwrap(reciprocal.src[0])).op is not Ops.SQRT: return None
  else: return None
  denominator = _unwrap(root.src[0])
  if denominator.op is not Ops.ADD or not any(x.op is Ops.CONST and float(x.arg) == 1.0 for x in denominator.src): return None
  square = next((_unwrap(x) for x in denominator.src if _unwrap(x).op is Ops.MUL), None)
  if square is None or not all(_unwrap(x) is source for x in square.src): return None
  nodes = val.toposort()
  coefficients = (-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
                  -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.SQRT, Ops.RECIPROCAL, Ops.WHERE, Ops.CMPLT, Ops.CMPNE,
             Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return source if all(u.op in allowed for u in nodes) and sum(u.op is Ops.SQRT for u in nodes) == 2 and \
    sum(u.op in (Ops.RECIPROCAL, Ops.FDIV) for u in nodes) == 1 and sum(u.op is Ops.WHERE for u in nodes) == 2 and \
    all(any(u.op is Ops.CONST and math.isclose(float(u.arg), coefficient) for u in nodes) for coefficient in coefficients) else None

def _try_atanh(val:UOp) -> UOp|None:
  """Recognize log((1+x)/(1-x))/2 after reciprocal-to-FDIV rewriting."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype not in (dtypes.half, dtypes.float) or val.op is not Ops.MUL: return None
  logarithm = next((_unwrap(x) for x in val.src if _unwrap(x).op is Ops.LOG2), None)
  scale = next((float(x.arg) for x in val.src if x.op is Ops.CONST), None)
  if logarithm is None or scale is None or not math.isclose(scale, math.log(2)/2): return None
  quotient = _unwrap(logarithm.src[0])
  if quotient.op is not Ops.FDIV: return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.FDIV, Ops.LOG2, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return source if all(u.op in allowed for u in nodes) and sum(u.op is Ops.LOG2 for u in nodes) == 1 and \
    sum(u.op is Ops.FDIV for u in nodes) == 1 and sum(u.op is Ops.ADD for u in nodes) == 2 and \
    any(u.op is Ops.CONST and float(u.arg) == -1.0 for u in nodes) else None

def _try_asinh_acosh(val:UOp) -> tuple[UOp,bool]|None:
  """Recognize log(x+sqrt(x*x±1)); bool selects acosh's minus-one form."""
  val = _unwrap(val)
  indexes = list(dict.fromkeys(u for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype not in (dtypes.half, dtypes.float) or val.op is not Ops.MUL: return None
  logarithm = next((_unwrap(x) for x in val.src if _unwrap(x).op is Ops.LOG2), None)
  scale = next((float(x.arg) for x in val.src if x.op is Ops.CONST), None)
  if logarithm is None or scale is None or not math.isclose(scale, math.log(2)): return None
  argument = _unwrap(logarithm.src[0])
  if argument.op is not Ops.ADD or not any(_unwrap(x) is source for x in argument.src): return None
  root = next((_unwrap(x) for x in argument.src if _unwrap(x).op is Ops.SQRT), None)
  if root is None or (radicand := _unwrap(root.src[0])).op is not Ops.ADD: return None
  square = next((_unwrap(x) for x in radicand.src if _unwrap(x).op is Ops.MUL), None)
  offset = next((float(x.arg) for x in radicand.src if x.op is Ops.CONST), None)
  if square is None or not all(_unwrap(x) is source for x in square.src) or offset not in (-1.0, 1.0): return None
  nodes = val.toposort()
  allowed = {Ops.ADD, Ops.MUL, Ops.LOG2, Ops.SQRT, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  return (source, offset == -1.0) if all(u.op in allowed for u in nodes) and \
    sum(u.op is Ops.LOG2 for u in nodes) == 1 and sum(u.op is Ops.SQRT for u in nodes) == 1 else None

def _try_sin_cos(val:UOp) -> tuple[UOp,bool]|None:
  """Recognize root sin(x) and tinygrad's cos(x) = sin(pi/2-cast_float(x))."""
  val = _unwrap(val)
  if val.op is Ops.SIN and (source := _unwrap(val.src[0])).op is Ops.INDEX:
    return source, False
  # fp16 cosine retains an outer CAST; fp32 cosine simplifies that no-op CAST
  # away, so recognize the phase form independently of `raw.op`.
  if val.op is not Ops.SIN: return None
  phase = _unwrap(val.src[0])
  indexes = list(dict.fromkeys(u for u in phase.toposort() if u.op is Ops.INDEX))
  constants = [float(u.arg) for u in phase.toposort() if u.op is Ops.CONST]
  if len(indexes) != 1 or indexes[0].dtype not in (dtypes.half, dtypes.float) or phase.op is not Ops.ADD: return None
  return (indexes[0], True) if any(math.isclose(x, math.pi/2) for x in constants) and \
    any(math.isclose(x, -1.0) for x in constants) else None

def _try_tan(val:UOp) -> UOp|None:
  """Recognize tinygrad tan(x) = sin(x) / cos(x), before or after reciprocal-to-FDIV."""
  val = _unwrap(val)
  if val.op is Ops.FDIV:
    numerator, denominator = val.src
  elif val.op is Ops.MUL:
    numerator, reciprocal = val.src
    if _unwrap(reciprocal).op is not Ops.RECIPROCAL: numerator, reciprocal = reciprocal, numerator
    reciprocal = _unwrap(reciprocal)
    if reciprocal.op is not Ops.RECIPROCAL: return None
    denominator = reciprocal.src[0]
  else: return None
  sin_match, cos_match = _try_sin_cos(numerator), _try_sin_cos(denominator)
  if sin_match is None or cos_match is None or sin_match[1] or not cos_match[1]: return None
  sin_source, cos_source = sin_match[0], cos_match[0]
  return sin_source if sin_source.src[0].buf_uop.arg.slot == cos_source.src[0].buf_uop.arg.slot else None

def _try_sinh_cosh(val:UOp) -> tuple[UOp,bool]|None:
  """Recognize (exp(x) +/- exp(-x))/2 and return (source, is_cosh)."""
  val = _unwrap(val)
  if val.op is not Ops.MUL: return None
  core, half = val.src
  if half.op is not Ops.CONST: core, half = half, core
  core = _unwrap(core)
  if half.op is not Ops.CONST or float(half.arg) != 0.5 or core.op is not Ops.ADD: return None

  def exp_term(term:UOp) -> tuple[UOp,int,int]|None:
    outer_sign = 1
    term = _unwrap(term)
    if term.op is Ops.MUL:
      candidate, coefficient = term.src
      if coefficient.op is not Ops.CONST: candidate, coefficient = coefficient, candidate
      if coefficient.op is not Ops.CONST or float(coefficient.arg) != -1.0: return None
      term, outer_sign = _unwrap(candidate), -1
    if term.op is not Ops.EXP2: return None
    scaled = _unwrap(term.src[0])
    if scaled.op is not Ops.MUL: return None
    scaled_input = next((_unwrap(x) for x in scaled.src if x.op is not Ops.CONST), None)
    log2e = next((float(x.arg) for x in scaled.src if x.op is Ops.CONST), None)
    if scaled_input is None or log2e is None or abs(abs(log2e)-math.log2(math.e)) >= 1e-3: return None
    input_sign = -1 if log2e < 0 else 1
    if scaled_input.op is Ops.MUL:
      source, coefficient = scaled_input.src
      if coefficient.op is not Ops.CONST: source, coefficient = coefficient, source
      source = _unwrap(source)
      if coefficient.op is not Ops.CONST or float(coefficient.arg) != -1.0: return None
      input_sign *= -1
    else: source = _unwrap(scaled_input)
    return (source, input_sign, outer_sign) if source.op is Ops.INDEX else None

  lhs, rhs = exp_term(core.src[0]), exp_term(core.src[1])
  if lhs is None or rhs is None or lhs[0] is not rhs[0] or {lhs[1], rhs[1]} != {-1, 1}: return None
  signs = {input_sign: outer_sign for _, input_sign, outer_sign in (lhs, rhs)}
  if signs == {1:1, -1:1}: return lhs[0], True
  if signs == {1:1, -1:-1}: return lhs[0], False
  return None

def _try_celu(val:UOp) -> tuple[UOp,float]|None:
  """Recognize max(x,0)+min(alpha*(exp(x/alpha)-1),0) for TestOps alphas."""
  val = _unwrap(val)
  if val.op is not Ops.ADD: return None
  indexes = list(dict.fromkeys(_unwrap(u) for u in val.toposort() if u.op is Ops.INDEX))
  if len(indexes) != 1 or (source := indexes[0]).dtype is not dtypes.half: return None
  nodes, stack = [], [val]
  while stack:
    node = _unwrap(stack.pop())
    if node in nodes: continue
    nodes.append(node)
    if node.op is not Ops.INDEX: stack.extend(node.src)
  positive = next((_unwrap(u) for u in val.src if _unwrap(u).op is Ops.MAX and
                   any(_unwrap(x) is source for x in _unwrap(u).src) and
                   any(x.op is Ops.CONST and float(x.arg) == 0.0 for x in _unwrap(u).src)), None)
  if positive is None or sum(u.op is Ops.EXP2 for u in nodes) != 1: return None
  if any(u.op not in {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CAST, Ops.EXP2, Ops.INDEX, Ops.CONST} for u in nodes): return None
  alpha = next((float(abs(float(u.arg))) for u in nodes if u.op is Ops.CONST and
                float(abs(float(u.arg))) in (2.0, 3.0, 4.0)), 1.0)
  return source, alpha

# LUT ops: EXP2 uses DPU LUT table (513 entries × 2 tables) with BN_MUL scaling
_LUT_OPS = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL}
_LUT_SIGMOID = Ops.NOOP  # internal plan marker; tinygrad lowers sigmoid to reciprocal/add/exp2
_LUT_ROUNDOFF = Ops.CUSTOM  # internal plan marker for the RK3588 round-to-nearest-even LUT
_LUT_EXP_CORRECTION = Ops.NEG  # internal marker for the second, residual exp LUT
_LUT_EXP2_SCALE = Ops.CMPEQ  # internal marker for split-range 2**integer scale
_LUT_EXP2_RESIDUAL = Ops.CMPLT  # internal marker for Q14 2**[-1,1]
_LUT_HARDSWISH = Ops.POW  # internal marker for a fused hardswish LUT
_LUT_HARDSWISH_CORRECTION = Ops.CMOD
_LUT_CELU = Ops.FLOORMOD
_LUT_CELU_LOCAL = Ops.FLOORDIV
_LUT_QUICK_GELU = Ops.SHL
_LUT_QUICK_GELU_LOCAL = Ops.SHR
_LUT_TANH = Ops.CDIV
_LUT_TANH_LOCAL = Ops.XOR
_LUT_LOG2_LOCAL = Ops.OR
_LUT_LOGSIGMOID_CORRECTION = Ops.AND
_LUT_LOGSIGMOID_TAIL = Ops.THREEFRY
_LUT_SOFTPLUS_TAIL = Ops.WMMA
_LUT_SOFTPLUS_WIDE = Ops.BARRIER
_LUT_MISH = Ops.SPECIAL
_LUT_MISH_LOCAL = Ops.BIND
_LUT_ELU = Ops.MULACC
_LUT_ELU_LOCAL = Ops.CUSTOMI
_LUT_ERF = Ops.FUNCTION
_LUT_ERF_LOCAL = Ops.CALL
_LUT_GELU = Ops.AFTER
_LUT_GELU_LOCAL = Ops.GROUP
_LUT_SIN_LOCAL = Ops.STACK
_LUT_COS = Ops.TUPLE
_LUT_COS_LOCAL = Ops.GETTUPLE
_LUT_TAN_LOCAL = Ops.GETADDR
_LUT_TAN_WIDE = Ops.BUFFER
_LUT_SINH = Ops.END
_LUT_COSH = Ops.ENDIF
_LUT_SINH_LOCAL = Ops.IF
_LUT_ASIN = Ops.WAIT
_LUT_ASIN_DETAIL = Ops.INS
_LUT_ASIN_DERIVATIVE = Ops.INDEX
_LUT_ACOS = Ops.CONTIGUOUS
_LUT_ACOS_ENDPOINT = Ops.DETACH
_LUT_ACOS_FINE_ENDPOINT = Ops.CAST
_LUT_ATAN = Ops.STAGE
_LUT_ATAN_LOCAL = Ops.SLICE
_LUT_ATANH = Ops.PAD
_LUT_ATANH_DETAIL = Ops.FLIP
_LUT_ASINH_CORE = Ops.COPY
_LUT_ASINH_RANGE = Ops.SHRINK
_LUT_ACOSH_CORE = Ops.EXPAND
_LUT_ACOSH_RANGE = Ops.RESHAPE
_LUT_POW8 = Ops.MSELECT
_LUT_POW8_CORRECTION = Ops.MSTACK
_LUT_POW8_HIGH = Ops.MULTI
_LUT_POW55 = Ops.REWRITE_ERROR
_LUT_POW55_HIGH = Ops.PYLITERAL
_LUT_POW_NEG55_LOW = Ops.CONTIGUOUS_BACKWARD
_LUT_POW_NEG55_HIGH = Ops.CUSTOM_FUNCTION
_LUT_POW_BASE55_LOW = Ops.SOURCE
_LUT_POW_BASE55_HIGH = Ops.BINARY
_LUT_POW_BASE8_LOW = Ops.PROGRAM
_LUT_POW_BASE8_HIGH = Ops.LINEAR
_LUT_POW_BASE8_FAR_LOW = Ops.SINK
_LUT_POW_BASE8_FAR_HIGH = Ops.LOAD
_LUT_POW_BASE07 = Ops.STORE
_LUT_POW_BASE2_LOW = Ops.TRUNC
_LUT_POW_BASE2_HIGH = Ops.BITCAST
_LUT_LOG_HALF_LOW = Ops.PERMUTE
_LUT_LOG_HALF_HIGH = Ops.ALLREDUCE
_LUT_SIGMOID_LOCAL = Ops.REDUCE
_LUT_BCE_ZERO = Ops.PARAM
_LUT_BCE_ONE = Ops.CONST
_LUT_BCE_LOGITS = Ops.WHERE
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

def _build_exp2_scale_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 scale table for the integer part of a range-reduced EXP2.

  Positive coordinates encode 2**-a for a=4*x in [0,8]. Negative
  coordinates encode 2**(8-a) for a=8-8*x in [8,24]; the caller divides
  that half by 256. Taking a reciprocal restores positive integer powers.
  """
  lut = [0] * _LUT_SIZE * 2
  index_scale, output_scale = 8192.0, 32768.0
  step = 32.0 / index_scale
  for i in range(_LUT_SIZE):
    x = -(512-i) * step
    lut[i] = max(1, min(32767, int(round(math.exp2(8.0*x) * output_scale))))
  for i in range(_LUT_SIZE):
    x = i * step
    lut[_LUT_SIZE+i] = max(1, min(32767, int(round(math.exp2(-4.0*x) * output_scale))))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 15

def _build_exp2_residual_lut() -> tuple[list[int], int, float, float, int]:
  """Q14 EXP2 table for coordinate x=2*r, r in [-1,1]."""
  lut = [0] * _LUT_SIZE * 2
  index_scale, output_scale = 8192.0, 16384.0
  step = 32.0 / index_scale
  for i in range(_LUT_SIZE):
    x = -(512-i) * step
    lut[i] = max(1, min(32767, int(round(math.exp2(x*0.5) * output_scale))))
  for i in range(_LUT_SIZE):
    x = i * step
    lut[_LUT_SIZE+i] = max(1, min(32767, int(round(math.exp2(x*0.5) * output_scale))))
  # At these physical knots the Q14 integer lands exactly on an fp16
  # half-tie. One raw count restores the correctly rounded EXP2 curve.
  tie_corrections = {
    9:+1, 77:+1, 95:+1, 115:+1, 147:-1, 157:+1, 185:-1, 226:+1,
    236:-1, 248:+1, 293:+1, 305:-1, 334:-1, 379:+1, 382:-1, 391:+1,
    399:-1, 409:-1, 447:-1, 463:+1, 479:+1, 488:+1, 590:+1, 628:+1,
    660:-1, 670:+1, 739:+1, 749:-1, 818:-1, 895:-1, 912:-1, 960:-1,
    992:+1, 1001:+1,
  }
  for index, delta in tie_corrections.items(): lut[index] += delta
  # Global POW-domain calibration. Each adjustment was accepted only when it
  # removed a tolerance miss without creating one in any other lane sharing
  # the same range-reduced residual knot.
  pow_corrections = {
    344:+4, 463:+8, 661:-25, 741:+9, 871:-8, 1019:+8, 977:+8,
  }
  for index, delta in pow_corrections.items(): lut[index] += delta
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale, 14

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

def _build_pow8_lut() -> tuple[list[int], int, float, float, int]:
  """Q11 u**8 table for the normalized low range 1 <= |u| <= sqrt(2)."""
  index_scale, output_scale = 8192.0, 2048.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      u = (-(512-i) if table == 0 else i) * step
      raw = int(round(min(abs(u), math.sqrt(2.0))**8 * output_scale))
      lut[offset+i] = max(-32768, min(32767, raw))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 11

def _build_pow8_correction_lut() -> tuple[list[int], int, float, float, int]:
  """Rejected WIP: residual for the integer-truncated former Q7 POW8 base."""
  index_scale, correction_scale, correction_bias = 32752.0, 32768.0, 0.125
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  base, *_ = _build_pow8_lut()
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      centered = (-(512-i) if table == 0 else i) * step
      u = max(1.0, min(2.0, centered+1.5))
      target = abs(u)**8
      position = u*256.0
      base_index = max(0, min(511, int(math.floor(position))))
      fraction = position-base_index
      interpolated = base[_LUT_SIZE+base_index] + fraction*(base[_LUT_SIZE+base_index+1]-base[_LUT_SIZE+base_index])
      approximate = math.floor(interpolated)/128.0
      residual = target-approximate
      lut[offset+i] = max(-32768, min(32767, int(round((residual+correction_bias)*correction_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, correction_scale, index_scale, 15

def _build_pow8_high_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 (u/2)**8 table for sqrt(2) <= |u| <= 2; caller scales by 256."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      u = (-(512-i) if table == 0 else i) * step
      raw = int(round((min(abs(u), 2.0)*0.5)**8 * output_scale))
      lut[offset+i] = max(-32768, min(32767, raw))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_pow55_lut(high:bool) -> tuple[list[int], int, float, float, int]:
  """Two fixed-point halves for u**5.5 after exact power-of-two normalization."""
  index_scale = 8192.0
  output_scale, minus_exp = (32768.0, 15) if high else (2048.0, 11)
  split, step, lut = 16.0**(1.0/5.5), 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      u = abs((-(512-i) if table == 0 else i) * step)
      target = (min(u, 2.0)*0.5)**5.5 if high else min(u, split)**5.5
      # The RK3588 LUT interpolation truncates just below several fp16 ties in
      # the high table.  One Q15 unit restores round-to-nearest at those ties.
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale)) + int(high)))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_pow_neg55_lut(high:bool) -> tuple[list[int], int, float, float, int]:
  """Tables for u**-5.5 on the normalized half-ranges [0.5,1] and [1,2]."""
  # The high table receives z=u-1 in [0,1].  Doubling the index scale uses all
  # 512 positive knots for that interval instead of only 256 for u in [1,2].
  index_scale = 16384.0
  output_scale, minus_exp = (32768.0, 15) if high else (1024.0, 10)
  # Rejected coarse-grid WIP: a global +1 Q15 bias made 74 correct samples high;
  # sparse knots {400,401,410,470,471,491,492,496..500,502,503} moved the same
  # one-ULP misses to adjacent quarter-grid inputs.
  fine_tie_knots = {289, 290, 308, 481}
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      u = abs((-(512-i) if table == 0 else i) * step)
      normalized = 1.0+min(1.0, u) if high else max(0.5, min(1.0, u))
      target = min(32.0, normalized**-5.5)
      correction = int(high and i in fine_tie_knots)
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale)) + correction))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_pow_base55_lut(high:bool) -> tuple[list[int], int, float, float, int]:
  """Q15 halves for 5.5**x: direct below zero, divided by 32 above zero."""
  index_scale, output_scale, step = 8192.0, 32768.0, 32.0/8192.0
  lut = [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i) * step
      target = 5.5**max(x, 0.0)/32.0 if high else 5.5**min(x, 0.0)
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_pow_base8_lut(region:int) -> tuple[list[int], int, float, float, int]:
  """Four Q15 bands for 8**x over [-2,-1], [-1,0], [0,1], and [1,2]."""
  index_scale, output_scale, step = 8192.0, 32768.0, 32.0/8192.0
  lut = [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i) * step
      if region == 0: target = 8.0**max(-2.0, min(-1.0, x))*8.0
      elif region == 1: target = 8.0**max(-1.0, min(0.0, x))
      elif region == 2: target = 8.0**max(0.0, min(1.0, x))/8.0
      else: target = 8.0**max(1.0, min(2.0, x))/64.0
      # Rejected two-band WIP stored direct below zero and /64 above zero.
      # It passed exact knots but missed 78 interpolated official values; a
      # global +1 Q15 bias increased that count to 178.
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_pow_base07_lut() -> tuple[list[int], int, float, float, int]:
  """Q13 0.7**(z+0.5) for shifted exponent z in [-2.5,2.5]."""
  index_scale, output_scale = float(np.float16(6553.6)), 8192.0
  base = float(np.float16(0.7))
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i) * step
      target = base**(z+0.5)
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 13

def _build_pow_base2_lut(high:bool) -> tuple[list[int], int, float, float, int]:
  """Q15 split 2**(z+0.5) for shifted exponent z in [-2.5,2.5]."""
  index_scale, output_scale = float(np.float16(6553.6)), 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i) * step
      x = z+0.5
      target = math.exp2(max(x, 0.0))/8.0 if high else math.exp2(min(x, 0.0))
      lut[offset+i] = max(-32768, min(32767, int(round(target*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

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

def _build_celu_lut(alpha:float) -> tuple[list[int], int, float, float, int]:
  """Q14 CELU negative branch over [-2,0]; Q14 fits alpha<=4 while retaining precision."""
  index_scale, output_scale = 8192.0, 16384.0
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    x = -(512-i)*step
    raw = int(round(alpha*math.expm1(x/alpha)*output_scale))
    lut[i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 14

def _build_celu_local_lut(alpha:float) -> tuple[list[int], int, float, float, int]:
  """Q15 CELU*8 over [-0.125,0], selected where relative tolerance is tight."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    z = -(512-i)*step
    raw = int(round(alpha*math.expm1((z/16)/alpha)*8.0*output_scale))
    lut[i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_elu_lut(negative_scale:float) -> tuple[list[int], int, float, float, int]:
  """Amplified Q15 ELU/SELU negative branch over [-8,0]."""
  index_scale, output_scale = 2048.0, 32768.0
  gain = 8.0 if negative_scale <= 0.125 else (1.0 if negative_scale <= 1.0 else 0.5)
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    x = -(512-i)*step
    raw = int(round(gain*negative_scale*math.expm1(x)*output_scale))
    lut[i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_elu_local_lut(negative_scale:float) -> tuple[list[int], int, float, float, int]:
  """Amplified Q15 ELU/SELU negative branch over [-0.5,0]."""
  index_scale, output_scale = 8192.0, 32768.0
  gain = 16.0 if negative_scale <= 0.125 else (2.0 if negative_scale <= 1.0 else 1.0)
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    z = -(512-i)*step
    raw = int(round(gain*negative_scale*math.expm1(z/4.0)*output_scale))
    lut[i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_erf_lut() -> tuple[list[int], int, float, float, int]:
  """Direct signed Q15 erf over [-4,4]."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(math.erf(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_erf_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 3*erf(x) over [-0.25,0.25], addressed by z=16*x."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      raw = int(round(3.0*math.erf(z/16.0)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _gelu_value(x:float, approximate_tanh:bool) -> float:
  return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3))) if approximate_tanh else 0.5*x*(1+math.erf(x/math.sqrt(2)))

def _build_gelu_lut(approximate_tanh:bool) -> tuple[list[int], int, float, float, int]:
  """Asymmetric Q15 GELU over [-4,4]; positive entries store GELU/4."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      y = _gelu_value(x, approximate_tanh)/(4.0 if table == 1 else 1.0)
      raw = int(round(y*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_gelu_local_lut(approximate_tanh:bool) -> tuple[list[int], int, float, float, int]:
  """Q15 2*GELU over [-0.5,0.5], addressed by z=8*x."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      raw = int(round(2.0*_gelu_value(z/8.0, approximate_tanh)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_quick_gelu_lut() -> tuple[list[int], int, float, float, int]:
  """Q14 QuickGELU over [-2,2]."""
  index_scale, output_scale = 8192.0, 16384.0
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round((x/(1+math.exp(-1.702*x)))*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  # Sparse measured knots preserve PyTorch's fp16 staged rounding boundaries.
  for table, index, correction in ((0, 276, 4), (0, 375, 1), (0, 408, 1), (0, 427, 1), (1, 49, 1)):
    lut[table*_LUT_SIZE+index] += correction
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 14

def _build_quick_gelu_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 QuickGELU over x∈[-2,-1], addressed by z=(x+1.5)*4."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [1] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      x = -1.5+z/4
      ideal = x/(1+math.exp(-1.702*x))
      xh = np.float16(x)
      scaled = np.float16(np.float32(xh)*np.float32(1.702))
      sigmoid = np.float16(1/(1+math.exp(-float(scaled))))
      staged = float(np.float16(xh*sigmoid))
      raw = int(round((0.5*ideal+0.5*staged)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_tanh_lut() -> tuple[list[int], int, float, float, int]:
  """Direct signed Q15 tanh over [-4,4]."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      lut[offset+i] = max(-32768, min(32767, int(round(math.tanh(x)*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_tanh_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 4*tanh(x) for x∈[-0.25,0.25], addressed by z=x*16."""
  index_scale, output_scale = 4096.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      lut[offset+i] = max(-32768, min(32767, int(round(4.0*math.tanh(z/16.0)*output_scale))))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_log2_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for LOG2 over x∈[0.25,4.0] → result∈[-2,2].
  Uses LUT_LE_START=-16384 (same as EXP2), index_scale=4096.
  LE table: underflow (x<0 → clip to log2(0+)=-2). LO table: x from 0 to 4.0.
  LO index = (bn_mul - 0) >> 5 = (x * 4096) >> 5. Entry i: x = i/128."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 4096.0
  step = 32.0 / index_scale  # 1/128
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

def _build_log2_local_lut(function_scale:float=1.0) -> tuple[list[int], int, float, float, int]:
  """Q15 4*scaled-log2(x) near one, addressed by z=(x-1)*12.5."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      raw = int(round(4.0*function_scale*math.log2(1.0+z/12.5)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_log_half_lut(high:bool) -> tuple[list[int], int, float, float, int]:
  """Direct natural-log refinement for fp16 probabilities.

  The low table covers a normalized mantissa m∈[0.25,0.5] as
  z=(m-0.375)*16 in Q14.  The high table covers m∈[0.5,1] as
  z=(m-0.75)*8 in Q15.  The caller restores the power-of-four exponent.
  """
  center, transform, output_scale, minus_exp = (0.75, 8.0, 32768.0, 15) if high else (0.375, 16.0, 16384.0, 14)
  bias = getenv("ROCKCHIP_LOG_HALF_BIAS", 0)
  index_scale, step, lut = 8192.0, 32.0/8192.0, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      x = center + z/transform
      raw = int(round(math.log(max(x, 2**-24))*output_scale))
      if raw < 0: raw += bias
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else -1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_logsigmoid_correction_lut(input_beta:float=1.0) -> tuple[list[int], int, float, float, int]:
  """Q15 scaled -log1p(exp(-abs(x))) over [-8,8]."""
  index_scale, output_scale = 2048.0*input_beta, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(-math.log1p(math.exp(-abs(input_beta*x)))*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else -1))
  if math.isclose(input_beta, 3.0):
    for i in (344, 345): lut[i] += 1
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_logsigmoid_tail_lut(input_beta:float=1.0) -> tuple[list[int], int, float, float, int]:
  """Q15 32*-log1p(exp(-abs(x))) over [-16,16], selected on small-output tails."""
  index_scale, output_scale = 1024.0*input_beta, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(-32.0*math.log1p(math.exp(-abs(input_beta*x)))*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else -1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_softplus_tail_lut(input_beta:float=1.0) -> tuple[list[int], int, float, float, int]:
  """Q15 21*-log1p(exp(-abs(beta*x))) over beta*x∈[-16,16]."""
  index_scale, output_scale = 1024.0*input_beta, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(-21.0*math.log1p(math.exp(-abs(input_beta*x)))*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else -1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_softplus_wide_lut(input_beta:float) -> tuple[list[int], int, float, float, int]:
  """Q13 -(1/beta)*log1p(exp(-abs(beta*x))) over x∈[-8,8] for beta<1."""
  index_scale, output_scale = 2048.0, 8192.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round((-math.log1p(math.exp(-abs(input_beta*x)))/input_beta)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else -1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 13

def _build_mish_lut() -> tuple[list[int], int, float, float, int]:
  """Asymmetric Q15 Mish over [-8,8]; positive entries store Mish/8."""
  # A direct Q14 table has excellent negative resolution, but its +2 ceiling
  # cannot represent Mish across the full positive interval.  Scaling only the
  # positive half preserves Q15 precision for the small negative tail.
  index_scale, output_scale = 2048.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      y = x*math.tanh(math.log1p(math.exp(x)))
      if table == 1: y /= 8.0
      raw = int(round(y*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_mish_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 Mish(x) over [-1,1], addressed by z=2*x."""
  # The first passing narrow path used z=16*x and 8*Mish(x) over
  # [-0.125,0.125].  Keep its shape here as a tuning reference: widening the
  # broad LUT to [-8,8] needs this local table to cover more of the central
  # interval while retaining enough Q15 output resolution.
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      x = z/2.0
      y = x*math.tanh(math.log1p(math.exp(x)))
      raw = int(round(y*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_sin_lut() -> tuple[list[int], int, float, float, int]:
  """Build 1026-entry LUT for SIN over x∈[-π,π] → result∈[-1,1].
  Uses LUT_LE_START=-16384, index_scale=16384/π≈5215.2.
  LE table: x∈[-π,0], LO table: x∈[0,π]."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 16384.0 / math.pi  # ≈5215.2, maps x∈[-π,π] to [-16384,16384]
  step = 32.0 / index_scale  # step in x per LUT entry
  output_scale = 32768.0
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
  return lut, bn_mul_operand, output_scale, index_scale, 15

def _build_sin_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 8*sin(x) for x∈[-0.125,0.125], addressed by z=16*x."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      z = (-(512-i) if table == 0 else i)*step
      raw = int(round(8.0*math.sin(z/16.0)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_cos_lut() -> tuple[list[int], int, float, float, int]:
  """Direct signed Q15 cosine over [-pi,pi]."""
  index_scale, output_scale = 16384.0/math.pi, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(math.cos(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_cos_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 2*cos(x) over [-2,2], selected while |cos(x)| <= 0.5."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      y = math.cos(x)
      raw = int(round(2.0*y*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_asin_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 asin(|x|)/2 over [0,1]; a following multiply by two decodes it."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    x = min(1.0, i*step)
    raw = int(round(0.5*math.asin(x)*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_asin_detail_lut() -> tuple[list[int], int, float, float, int]:
  """Dual-purpose Q15 detail LUT: LE stores 4*asin(|x|), LO stores asin(1-x)/2."""
  index_scale, output_scale = 65504.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    negative_x, endpoint_distance = -(512-i)*step, i*step
    local_raw = int(round(4.0*math.asin(min(1.0, abs(negative_x)))*output_scale))
    endpoint_raw = int(round(0.5*math.asin(max(-1.0, 1.0-endpoint_distance))*output_scale))
    lut[i] = max(-32768, min(32767, local_raw if local_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, endpoint_raw if endpoint_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_acos_lut() -> tuple[list[int], int, float, float, int]:
  """Asymmetric Q15 acos: negative table stores acos(x)/4, positive table stores acos(x)/2."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    negative_x, positive_x = -(512-i)*step, min(1.0, i*step)
    negative_raw = int(round(0.25*math.acos(max(-1.0, negative_x))*output_scale))
    positive_raw = int(round(0.5*math.acos(positive_x)*output_scale))
    lut[i] = max(-32768, min(32767, negative_raw if negative_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, positive_raw if positive_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_asin_derivative_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 half-scale derivative for fp32-residual correction on |x|≤0.85."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    x = min(.85, i*step)
    raw = int(round(.5/math.sqrt(1-x*x)*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_acos_endpoint_lut() -> tuple[list[int], int, float, float, int]:
  """Direct Q15 acos(1-d) addressed by endpoint distance d∈[0,0.125]."""
  index_scale, output_scale = 65504.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    distance = i*step
    raw = int(round(math.acos(max(-1.0, 1.0-distance))*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_acos_fine_endpoint_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 8*acos(1-d) addressed by 64*d for fine interpolation near d=0."""
  index_scale, output_scale = 65504.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    distance = i*step/64
    raw = int(round(8*math.acos(max(-1.0, 1.0-distance))*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_atan_lut() -> tuple[list[int], int, float, float, int]:
  """Direct Q15 atan(x) for transformed x∈[0,1]."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    raw = int(round(math.atan(i*step)*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_atan_local_lut() -> tuple[list[int], int, float, float, int]:
  """Detail LUT: LE stores 4*atan(|z|/4), LO stores atan(1/z)/2."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    negative_z, positive_z = -(512-i)*step, i*step
    local_raw = int(round(4.0*math.atan(abs(negative_z)/4.0)*output_scale))
    wide_raw = int(round(0.5*(math.pi/2 if positive_z == 0 else math.atan(1.0/positive_z))*output_scale))
    lut[i] = max(-32768, min(32767, local_raw if local_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, wide_raw if wide_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_atanh_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 atanh(x)/4 over x∈[0,0.875], decoded by a multiply by four."""
  index_scale, output_scale = 16384.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    x = min(0.99951171875, i*step)
    raw = int(round(0.25*math.atanh(x)*output_scale))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_atanh_detail_lut() -> tuple[list[int], int, float, float, int]:
  """Detail LUT: LE stores 4*atanh(|x|), LO stores atanh(1-d)/8."""
  index_scale, output_scale = 65504.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    negative_x, distance = -(512-i)*step, i*step
    local_raw = int(round(4.0*math.atanh(min(0.99951171875, abs(negative_x)))*output_scale))
    endpoint_x = 1.0-max(0.00048828125, distance)
    endpoint_raw = int(round(0.125*math.atanh(endpoint_x)*output_scale))
    lut[i] = max(-32768, min(32767, local_raw if local_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, endpoint_raw if endpoint_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_asinh_acosh_core_lut(is_acosh:bool) -> tuple[list[int], int, float, float, int]:
  """Core detail LUT: LE resolves the origin/endpoint, LO covers through x=2."""
  index_scale, output_scale = 8192.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  for i in range(_LUT_SIZE):
    negative_z, positive_z = -(512-i)*step, i*step
    if is_acosh:
      negative_y = 2.0*math.acosh(1.0+abs(negative_z)/48.0)
      positive_y = 0.5*math.acosh(1.0+positive_z/2.0)
    else:
      negative_y = 4.0*math.asinh(abs(negative_z)/8.0)
      positive_y = 0.5*math.asinh(positive_z)
    negative_raw, positive_raw = int(round(negative_y*output_scale)), int(round(positive_y*output_scale))
    lut[i] = max(-32768, min(32767, negative_raw if negative_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, positive_raw if positive_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_asinh_acosh_range_lut(is_acosh:bool) -> tuple[list[int], int, float, float, int]:
  """Range LUT: LE covers [2,18], LO covers large x after division by 19."""
  index_scale, output_scale = 1024.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [1] * (_LUT_SIZE*2)
  fn = math.acosh if is_acosh else math.asinh
  for i in range(_LUT_SIZE):
    negative_z, positive_z = -(512-i)*step, i*step
    negative_raw = int(round(0.25*fn(2.0+abs(negative_z))*output_scale))
    positive_x = max(1.0, 19.0*positive_z) if is_acosh else 19.0*positive_z
    positive_raw = int(round(0.125*fn(positive_x)*output_scale))
    lut[i] = max(-32768, min(32767, negative_raw if negative_raw != 0 else 1))
    lut[_LUT_SIZE+i] = max(-32768, min(32767, positive_raw if positive_raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_tan_local_lut() -> tuple[list[int], int, float, float, int]:
  """Direct Q15 tangent over [-0.45,0.45]."""
  index_scale, output_scale = 32768.0, 32768.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(math.tan(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_tan_wide_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 tan(x)/2 over [-1.05,1.05], decoded by a following multiply by two."""
  index_scale, output_scale = 16384.0/1.05, 32768.0
  # The address multiplier is stored as fp16. Build knots from that exact
  # quantized value or the nominal/actual coordinate drift is visible at 1e-3.
  step, lut = 32.0/float(np.float16(index_scale)), [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(0.5*math.tan(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_sinh_lut() -> tuple[list[int], int, float, float, int]:
  """Direct signed Q13 sinh over [-2,2]."""
  index_scale, output_scale = 8192.0, 8192.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(math.sinh(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 13

def _build_sinh_local_lut() -> tuple[list[int], int, float, float, int]:
  """Q15 4*sinh(x) over approximately [-0.25,0.25]."""
  index_scale, output_scale = 65504.0, 32768.0
  step, lut = 32.0/float(np.float16(index_scale)), [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(4.0*math.sinh(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw if raw != 0 else 1))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 15

def _build_cosh_lut() -> tuple[list[int], int, float, float, int]:
  """Direct signed Q13 cosh over [-2,2]."""
  index_scale, output_scale = 8192.0, 8192.0
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = int(round(math.cosh(x)*output_scale))
      lut[offset+i] = max(-32768, min(32767, raw))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, 13

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

def _build_sigmoid_local_lut() -> tuple[list[int], int, float, float, int]:
  """Dense Q15 sigmoid over [-2,2], selected inside the broad [-8,8] path."""
  index_scale, output_scale, minus_exp = 8192.0, 32768.0, 15
  sample_shift = getenv("ROCKCHIP_SIGMOID_LOCAL_SHIFT", 0.0)
  step, lut = 32.0/index_scale, [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    for i in range(_LUT_SIZE):
      x = ((-(512-i) if table == 0 else i)+sample_shift)*step
      raw = int(round(output_scale/(1.0+math.exp(-x))))
      lut[offset+i] = max(1, min(32767, raw))
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_bce_endpoint_lut(target_one:bool) -> tuple[list[int], int, float, float, int]:
  """Q15 BCE(sigmoid_fp16(x), target) over [-2,2].

  The large-loss half is stored divided by four and restored by the staged
  sign mask, giving both physical halves Q15 effective precision.  Fit the
  table nodes over every representable fp16 input, weighted by its real-number
  Voronoi interval: direct grid samples linearly interpolate across sigmoid's
  fp16 rounding steps and introduce avoidable one-ULP endpoint errors.
  """
  index_scale, output_scale, minus_exp = 8192.0, 32768.0, 15
  bit_patterns = np.arange(0x10000, dtype=np.uint16)
  xs = bit_patterns.view(np.float16)
  xs = np.sort(xs[np.isfinite(xs) & (xs >= -2) & (xs <= 2)].astype(np.float64))
  midpoints = (xs[:-1]+xs[1:])/2
  boundaries = np.concatenate((np.array([-2.0]), midpoints, np.array([2.0])))
  weights = np.maximum(0.0, boundaries[1:]-boundaries[:-1])
  probabilities = np.float16(1.0/(1.0+np.exp(-xs))).astype(np.float64)
  losses = np.float16(-np.log(probabilities) if target_one else -np.log1p(-probabilities)).astype(np.float64)
  large = xs < 0 if target_one else xs >= 0
  desired_raw = losses / np.where(large, 4.0, 1.0) * output_scale
  negative = xs < 0
  position = np.where(negative, (xs+2.0)*256.0, xs*256.0)
  index = np.minimum(np.floor(position).astype(np.int32), 511)
  fraction = position-index
  lut = [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    selected = negative if table == 0 else ~negative
    i, f, w, y = index[selected], fraction[selected], weights[selected], desired_raw[selected]
    normal = np.zeros((_LUT_SIZE, _LUT_SIZE), dtype=np.float64)
    rhs = np.zeros(_LUT_SIZE, dtype=np.float64)
    np.add.at(normal, (i, i), w*(1.0-f)**2)
    np.add.at(normal, (i+1, i+1), w*f**2)
    np.add.at(normal, (i, i+1), w*f*(1.0-f))
    np.add.at(normal, (i+1, i), w*f*(1.0-f))
    np.add.at(rhs, i, w*(1.0-f)*y)
    np.add.at(rhs, i+1, w*f*y)
    fitted = np.rint(np.linalg.solve(normal + np.eye(_LUT_SIZE)*1e-10, rhs)).astype(np.int32)
    lut[offset:offset+_LUT_SIZE] = np.clip(fitted, 1, 32767).tolist()
  if target_one:
    # Sparse interpolation-boundary calibration after the global fp16-domain
    # fit. Each pair moves a knot by one output ULP or less; it is not a
    # per-input result table.
    for table, knot_index, correction in ((0, 372, 8), (0, 373, 8), (1, 155, 8), (1, 156, 8),
                                          (1, 383, 8), (1, 384, 8), (1, 401, -8), (1, 402, -8),
                                          (1, 460, -8), (1, 467, 8)):
      lut[table*_LUT_SIZE+knot_index] += correction
  return lut, int(np.float16(index_scale).view(np.int16)) & 0xFFFF, output_scale, index_scale, minus_exp

def _build_bce_logits_lut() -> tuple[list[int], int, float, float, int]:
  """Fitted Q15 softplus(-x) for BCE-with-logits over fp16 x∈[-2,2].

  Negative x is stored divided by four and restored by the staged sign mask.
  """
  index_scale, output_scale, minus_exp = 8192.0, 32768.0, 15
  bit_patterns = np.arange(0x10000, dtype=np.uint16)
  xs = bit_patterns.view(np.float16)
  xs = np.sort(xs[np.isfinite(xs) & (xs >= -2) & (xs <= 2)].astype(np.float64))
  midpoints = (xs[:-1]+xs[1:])/2
  boundaries = np.concatenate((np.array([-2.0]), midpoints, np.array([2.0])))
  weights = np.maximum(0.0, boundaries[1:]-boundaries[:-1])
  negative = xs < 0
  desired_raw = np.float16(np.log1p(np.exp(-xs))).astype(np.float64) / np.where(negative, 4.0, 1.0) * output_scale
  position = np.where(negative, (xs+2.0)*256.0, xs*256.0)
  index = np.minimum(np.floor(position).astype(np.int32), 511)
  fraction = position-index
  lut = [0] * (_LUT_SIZE*2)
  for table, offset in ((0, 0), (1, _LUT_SIZE)):
    selected = negative if table == 0 else ~negative
    i, f, w, y = index[selected], fraction[selected], weights[selected], desired_raw[selected]
    normal = np.zeros((_LUT_SIZE, _LUT_SIZE), dtype=np.float64)
    rhs = np.zeros(_LUT_SIZE, dtype=np.float64)
    np.add.at(normal, (i, i), w*(1.0-f)**2)
    np.add.at(normal, (i+1, i+1), w*f**2)
    np.add.at(normal, (i, i+1), w*f*(1.0-f))
    np.add.at(normal, (i+1, i), w*f*(1.0-f))
    np.add.at(rhs, i, w*(1.0-f)*y)
    np.add.at(rhs, i+1, w*f*y)
    fitted = np.rint(np.linalg.solve(normal + np.eye(_LUT_SIZE)*1e-10, rhs)).astype(np.int32)
    lut[offset:offset+_LUT_SIZE] = np.clip(fitted, 1, 32767).tolist()
  # Measured RK3588 interpolation at x≈0.093 is one fp16 output ULP below the
  # arithmetic model. Correct the two bounding positive-table knots.
  for knot_index in (23, 24): lut[_LUT_SIZE+knot_index] += 16
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
  if val.op is Ops.CUSTOM and val.arg == "rk_pow8" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW8)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow8_correction" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW8_CORRECTION)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow8_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW8_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow55" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW55)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow55_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW55_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_neg55_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_NEG55_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_neg55_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_NEG55_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base55_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE55_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base55_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE55_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base8_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE8_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base8_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE8_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base8_far_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE8_FAR_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base8_far_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE8_FAR_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base07" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE07)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base2_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE2_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_pow_base2_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_POW_BASE2_HIGH)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_celu" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_CELU)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_celu_local" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_CELU_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_quick_gelu" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_QUICK_GELU)
  if val.op is Ops.CUSTOM and val.arg == "rk_quick_gelu_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_QUICK_GELU_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_tanh" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_TANH)
  if val.op is Ops.CUSTOM and val.arg == "rk_tanh_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_TANH_LOCAL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_log2_local" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_LOG2_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_log_half_low" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_LOG_HALF_LOW)
  if val.op is Ops.CUSTOM and val.arg == "rk_log_half_high" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_LOG_HALF_HIGH)
  if val.op is Ops.CUSTOM and val.arg == "rk_logsigmoid_correction" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_LOGSIGMOID_CORRECTION)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 3 and val.arg[0] == "rk_logsigmoid_correction" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), float(val.arg[2]), _LUT_LOGSIGMOID_CORRECTION)
  if val.op is Ops.CUSTOM and val.arg == "rk_logsigmoid_tail" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_LOGSIGMOID_TAIL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 3 and val.arg[0] == "rk_logsigmoid_tail" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), float(val.arg[2]), _LUT_LOGSIGMOID_TAIL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 3 and val.arg[0] == "rk_softplus_tail" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), float(val.arg[2]), _LUT_SOFTPLUS_TAIL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_softplus_wide" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_SOFTPLUS_WIDE)
  if val.op is Ops.CUSTOM and val.arg == "rk_mish" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_MISH)
  if val.op is Ops.CUSTOM and val.arg == "rk_mish_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_MISH_LOCAL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_elu" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_ELU)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_elu_local" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(val.arg[1]), 1.0, _LUT_ELU_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_erf" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ERF)
  if val.op is Ops.CUSTOM and val.arg == "rk_erf_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ERF_LOCAL)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_gelu" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(bool(val.arg[1])), 1.0, _LUT_GELU)
  if val.op is Ops.CUSTOM and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "rk_gelu_local" and \
     (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, float(bool(val.arg[1])), 1.0, _LUT_GELU_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_sin_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_SIN_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_cos" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_COS)
  if val.op is Ops.CUSTOM and val.arg == "rk_cos_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_COS_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_tan_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_TAN_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_tan_wide" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_TAN_WIDE)
  if val.op is Ops.CUSTOM and val.arg == "rk_sinh" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_SINH)
  if val.op is Ops.CUSTOM and val.arg == "rk_sinh_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_SINH_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_cosh" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_COSH)
  if val.op is Ops.CUSTOM and val.arg == "rk_sigmoid_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_SIGMOID_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_bce_zero" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_BCE_ZERO)
  if val.op is Ops.CUSTOM and val.arg == "rk_bce_one" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_BCE_ONE)
  if val.op is Ops.CUSTOM and val.arg == "rk_bce_logits" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_BCE_LOGITS)
  if val.op is Ops.CUSTOM and val.arg == "rk_asin" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ASIN)
  if val.op is Ops.CUSTOM and val.arg == "rk_asin_detail" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ASIN_DETAIL)
  if val.op is Ops.CUSTOM and val.arg == "rk_asin_derivative" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ASIN_DERIVATIVE)
  if val.op is Ops.CUSTOM and val.arg == "rk_acos" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ACOS)
  if val.op is Ops.CUSTOM and val.arg == "rk_acos_endpoint" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ACOS_ENDPOINT)
  if val.op is Ops.CUSTOM and val.arg == "rk_acos_fine_endpoint" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ACOS_FINE_ENDPOINT)
  if val.op is Ops.CUSTOM and val.arg == "rk_atan" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ATAN)
  if val.op is Ops.CUSTOM and val.arg == "rk_atan_local" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ATAN_LOCAL)
  if val.op is Ops.CUSTOM and val.arg == "rk_atanh" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ATANH)
  if val.op is Ops.CUSTOM and val.arg == "rk_atanh_detail" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ATANH_DETAIL)
  if val.op is Ops.CUSTOM and val.arg == "rk_asinh_core" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ASINH_CORE)
  if val.op is Ops.CUSTOM and val.arg == "rk_asinh_range" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ASINH_RANGE)
  if val.op is Ops.CUSTOM and val.arg == "rk_acosh_core" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ACOSH_CORE)
  if val.op is Ops.CUSTOM and val.arg == "rk_acosh_range" and (inner := _unwrap(val.src[0])).op is Ops.INDEX:
    return (inner.src[0].buf_uop.arg.slot, 1.0, 1.0, _LUT_ACOSH_RANGE)
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
  if idx.op is Ops.RANGE and getattr(idx.arg[-1], "name", "") == "REDUCE": return 1
  if idx.op is not Ops.ADD: return None
  a, b = idx.src
  if a.op is Ops.MUL:
    mr, mc = a.src
    if mr.op is Ops.RANGE and mc.op is Ops.CONST and b.op is Ops.RANGE and getattr(mr.arg[-1], "name", "") == "REDUCE" and getattr(b.arg[-1], "name", "") == "LOOP": return int(mc.arg)
  if a.op is Ops.RANGE and b.op is Ops.RANGE and getattr(a.arg[-1], "name", "") == "REDUCE" and getattr(b.arg[-1], "name", "") == "LOOP": return 1
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
  return idx.op is Ops.RANGE and getattr(idx.arg[-1], "name", "") == kind

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

def _try_cmac_materialization(sink:UOp, reduce:UOp) -> tuple[int, ...]|None:
  """Map a separable indexed dot product to CMAC by serializing only its A/B gathers.
  Output LOOP axes used by A form M; axes used by B form N. All multiplication and
  fp32 accumulation remain on CNA/CORE."""
  body = _reduce_body(reduce)
  outer_conditions:list[UOp] = []
  def is_zero(u:UOp) -> bool:
    return u.op is Ops.CONST and (u.arg is Invalid or float(u.arg) == 0.0)
  while body.op is Ops.WHERE and body.dtype is dtypes.float and (is_zero(body.src[1]) or is_zero(body.src[2])):
    if is_zero(body.src[2]):
      outer_conditions.append(body.src[0])
      body = body.src[1]
    else:
      outer_conditions.append(UOp(Ops.CMPNE, dtypes.bool, (body.src[0], UOp.const(dtypes.bool, True))))
      body = body.src[2]
    while body.op is Ops.CAST: body = body.src[0]
  if body.op is Ops.MUL: a_val, b_val = (_unwrap(x) for x in body.src)
  elif body.op is Ops.WHERE and body.dtype is dtypes.half:
    true, false = (_unwrap(x) for x in body.src[1:])
    if true.op is not Ops.MUL or false.op is not Ops.MUL: return None
    common = next((x for x in true.src if any(x is y for y in false.src)), None)
    if common is None: return None
    true_value = next(x for x in true.src if x is not common)
    false_value = next(x for x in false.src if x is not common)
    a_val = UOp(Ops.WHERE, dtypes.half, (body.src[0], true_value, false_value))
    b_val = _unwrap(common)
  elif body.dtype is dtypes.half: a_val, b_val = body, None
  else: return None
  for condition in reversed(outer_conditions):
    a_val = UOp(Ops.WHERE, dtypes.half, (condition, a_val, UOp.const(dtypes.half, 0.0)))
  store = _store_node(sink)
  if store is None or store.src[0].op is not Ops.INDEX: return None
  if a_val.dtype is not dtypes.half or (b_val is not None and b_val.dtype is not dtypes.half) or store.src[0].dtype is not dtypes.half: return None
  value_indexes = [[u for u in val.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.half]
                   for val in (a_val, b_val) if val is not None]
  if not value_indexes[0] or any(len({u.src[0].buf_uop.arg.slot for u in indexes}) != 1 for indexes in value_indexes): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  if not reductions or any(u.op is not Ops.RANGE or u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  loop_extents, reduce_extents = tuple(int(u.src[0].arg) for u in loops), tuple(int(u.src[0].arg) for u in reductions)
  a_nodes, b_nodes = set(a_val.toposort()), set(b_val.toposort()) if b_val is not None else set()
  a_axes = {i for i,u in enumerate(loops) if u in a_nodes}
  b_axes = {i for i,u in enumerate(loops) if u in b_nodes}
  shared_axes = tuple(i for i in range(len(loops)) if i in a_axes and i in b_axes)
  m_axes = tuple(i for i in range(len(loops)) if i in a_axes or i not in b_axes)
  n_axes = tuple(i for i in range(len(loops)) if i in b_axes)
  if set(m_axes) | set(n_axes) != set(range(len(loops))): return None
  range_ids = {u:i for i,u in enumerate((*loops, *reductions))}
  op_codes = {Ops.ADD:2, Ops.MUL:3, Ops.FLOORDIV:4, Ops.FLOORMOD:5,
              Ops.CMPLT:6, Ops.CMPNE:7, Ops.AND:8, Ops.OR:9}

  def emit_int(u:UOp, code:list[int]) -> bool:
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST:
      try: value = 0 if u.arg is Invalid else int(u.arg)
      except (TypeError, ValueError): return False
      code.extend((0, value))
      return True
    if u.op is Ops.RANGE and u in range_ids:
      code.extend((1, range_ids[u]))
      return True
    if u.op in op_codes and len(u.src) == 2:
      if not emit_int(u.src[0], code) or not emit_int(u.src[1], code): return False
      code.extend((op_codes[u.op], 0))
      return True
    if u.op is Ops.WHERE and len(u.src) == 3:
      if not all(emit_int(x, code) for x in u.src): return False
      code.extend((11, 0))
      return True
    return False

  def emit_value(u:UOp, code:list[int]) -> bool:
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.INDEX and u.dtype is dtypes.half:
      if not emit_int(u.src[1], code): return False
      code.extend((10, 0))
      return True
    if u.op is Ops.CONST and u.dtype is dtypes.half:
      code.extend((12, struct.unpack('<H', struct.pack('<e', float(u.arg)))[0]))
      return True
    if u.op is Ops.WHERE and len(u.src) == 3:
      if not emit_int(u.src[0], code) or not emit_value(u.src[1], code) or not emit_value(u.src[2], code): return False
      code.extend((11, 0))
      return True
    return False

  out_code:list[int] = []
  a_code:list[int] = []
  b_code:list[int] = []
  if not emit_int(store.src[0].src[1], out_code) or not emit_value(a_val, a_code): return None
  if b_val is None: b_code.extend((12, struct.unpack('<H', struct.pack('<e', 1.0))[0]))
  elif not emit_value(b_val, b_code): return None
  # CMAC accumulation is sensitive to K packing order.  tinygrad's ordinary
  # convolution exposes (input-channel, kernel...) reduction axes, while
  # conv_transpose exposes (kernel..., input-channel).  Pack K in the source
  # weight's row-major address order, as conv_grok/gemm_npu.py does, so both
  # graph forms accumulate products in the same order.
  def index_strides(indexes:list[UOp]) -> tuple[int, ...]:
    candidates:list[tuple[int, ...]] = []
    for index in indexes:
      code:list[int] = []
      if not emit_int(index.src[1], code): continue
      def evaluate(coords:list[int]) -> int:
        stack:list[int] = []
        for pos in range(0, len(code), 2):
          op, arg = code[pos], code[pos+1]
          if op == 0: stack.append(arg)
          elif op == 1: stack.append(coords[arg])
          elif op == 11:
            false, true, cond = stack.pop(), stack.pop(), stack.pop()
            stack.append(true if cond else false)
          else:
            rhs, lhs = stack.pop(), stack.pop()
            if op == 2: stack.append(lhs + rhs)
            elif op == 3: stack.append(lhs * rhs)
            elif op == 4: stack.append(lhs // rhs)
            elif op == 5: stack.append(lhs % rhs)
            elif op == 6: stack.append(int(lhs < rhs))
            elif op == 7: stack.append(int(lhs != rhs))
            elif op == 8: stack.append(int(bool(lhs) and bool(rhs)))
            elif op == 9: stack.append(int(bool(lhs) or bool(rhs)))
            else: return 0
        return stack[0] if len(stack) == 1 else 0
      base_coords = [0] * len(range_ids)
      base = evaluate(base_coords)
      strides = []
      for axis, extent in enumerate(reduce_extents):
        coords = base_coords.copy()
        if extent > 1: coords[len(loops)+axis] = 1
        strides.append(evaluate(coords)-base)
      candidates.append(tuple(strides))
    return max(candidates, key=lambda x:(sum(v != 0 for v in x), max((abs(v) for v in x), default=0),
                                         sum(abs(v) for v in x)), default=tuple(0 for _ in reductions))

  weight_indexes = value_indexes[1] if b_val is not None else value_indexes[0]
  weight_strides = index_strides(weight_indexes)
  reduce_order = tuple(sorted(range(len(reductions)), key=lambda axis:(-abs(weight_strides[axis]), axis)))
  # A flipped weight kernel is tinygrad's conv_transpose signature.  PyTorch's
  # CPU fp16 reference rounds each per-kernel-position channel dot before
  # col2im adds it to the output; retain those axes so emission can reproduce
  # the same boundaries with multiple CMAC and DPU tasks.
  rounding_axes = tuple(axis for axis, stride in enumerate(weight_strides) if stride < 0)
  batches = prod(loop_extents[i] for i in shared_axes)
  M, N, K = prod(loop_extents[i] for i in m_axes), prod(loop_extents[i] for i in n_axes), prod(reduce_extents) * batches
  a_slot = value_indexes[0][0].src[0].buf_uop.arg.slot
  b_slot = _CONST_SLOT if b_val is None else value_indexes[1][0].src[0].buf_uop.arg.slot
  return (M, N, K, a_slot, b_slot, len(loops), *loop_extents, len(reductions), *reduce_extents,
          len(m_axes), *m_axes, len(n_axes), *n_axes, len(shared_axes), *shared_axes,
          len(reduce_order), *reduce_order,
          len(out_code), *out_code, len(a_code), *a_code, len(b_code), *b_code,
          0, len(reduce_order), *reduce_order, len(rounding_axes), *rounding_axes, 0)

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
  if body.src[1].op is Ops.RANGE and getattr(body.src[1].arg[-1], 'name', '') == "REDUCE": return (input_slot, 1, 1, K, cv)  # C-pattern
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

def _try_cmac_epilogue(sink: UOp, reduce: UOp) -> tuple[str, float, int, int]|None:
  """Detect BS-fusable epilogue after CMAC reduce. Returns (epilogue_type, scale) or None.
  Supported: "relu" (WHERE(CMPLT(0,x), x, 0)), "scale" (MUL(x, const) or MUL(const, x)),
  and channel bias with optional ReLU. Bias is added to the raw fp32 CMAC accumulator
  by the mapped-buffer runtime before its one final fp16 rounding.
  The epilogue sits between the reduce and the store: store.src[1] = epilogue(reduce)."""
  store = _store_node(sink)
  if store is None: return None
  sv = _unwrap(store.src[1])
  if sv is reduce: return ("none", 1.0, -1, -1)
  # Scale: MUL(CAST(reduce), CONST(c)) or MUL(CONST(c), CAST(reduce))
  if sv.op is Ops.MUL and len(sv.src) == 2:
    a, b = _unwrap(sv.src[0]), _unwrap(sv.src[1])
    if a is reduce and b.op is Ops.CONST: return ("scale", float(b.arg), -1, -1)
    if b is reduce and a.op is Ops.CONST: return ("scale", float(a.arg), -1, -1)
  relu = False
  # ReLU: WHERE(CMPLT(CONST(0), CAST(reduce)), CAST(reduce), CONST(0))
  if sv.op is Ops.WHERE and len(sv.src) == 3:
    cond, t, f = sv.src
    cond_u, t_u, f_u = _unwrap(cond), _unwrap(t), _unwrap(f)
    # WHERE(CMPLT(0, x), x, 0) = relu(x)
    if cond_u.op is Ops.CMPLT and f_u.op is Ops.CONST and float(f_u.arg) == 0.0 and \
       len(cond_u.src) == 2 and _unwrap(cond_u.src[0]).op is Ops.CONST and float(_unwrap(cond_u.src[0]).arg) == 0.0 and \
       _unwrap(cond_u.src[1]) is t_u:
      if t_u is reduce: return ("relu", 1.0, -1, -1)
      sv, relu = t_u, True
    # WHERE(CMPLT(x, 0), 0, x) = relu(x) (alternative form)
    elif cond_u.op is Ops.CMPLT and t_u.op is Ops.CONST and float(t_u.arg) == 0.0 and \
         len(cond_u.src) == 2 and _unwrap(cond_u.src[1]).op is Ops.CONST and float(_unwrap(cond_u.src[1]).arg) == 0.0 and \
         _unwrap(cond_u.src[0]) is f_u:
      if f_u is reduce: return ("relu", 1.0, -1, -1)
      sv, relu = f_u, True
  # ADD(CAST(REDUCE), INDEX(bias, RANGE(loop_axis))) with optional outer ReLU.
  if sv.op is Ops.ADD and len(sv.src) == 2:
    a, b = _unwrap(sv.src[0]), _unwrap(sv.src[1])
    bias = b if a is reduce and b.op is Ops.INDEX else a if b is reduce and a.op is Ops.INDEX else None
    if bias is not None and bias.src[0].op is Ops.PARAM:
      loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
      if bias.src[1] in loops:
        return ("bias_relu" if relu else "bias", 1.0, bias.src[0].buf_uop.arg.slot, loops.index(bias.src[1]))
  return None

def plan_rk(sink: UOp) -> RKPlan|str:
  """Classify a post-early_simplify SINK. Returns RKPlan on success, 'RKPLAN_REJECT:...' str on reject."""
  if not _is_fp16_only(sink): return "RKPLAN_REJECT:unsupported_dtype"
  reduce = _reduce_node(sink)
  lut_result = None  # set in DPU path only
  abs_slot = None  # set in DPU path only
  epilogue, epilogue_scale = "none", 1.0  # set in CMAC path only
  epilogue_bias_slot, epilogue_bias_axis = -1, -1
  cmac_materialization:tuple[int, ...] = ()
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
    if lut_result is not None: kind, a2d, ru = "dpu_lut", True, True
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
        epilogue, epilogue_scale, epilogue_bias_slot, epilogue_bias_axis = epi
    materialization = _try_cmac_materialization(sink, reduce)
    direct_cmac = body.op is Ops.MUL and all(s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX) for s in body.src) and \
      _is_cmac_matmul_layout(sink, reduce)
    if materialization is not None:
      if not direct_cmac or tuple(materialization[:2]) != _shape_of_store(sink): cmac_materialization = materialization
      kind = "cmac"
    elif body.op is Ops.MUL and all(s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX) for s in body.src):
      if not direct_cmac: return "RKPLAN_REJECT:unsupported_layout"
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
  # A fused explicit `.half()` leaves fp32 backing PARAMs below half CASTs.
  # CMAC's runtime already converts tagged input buffers before pad/swizzle;
  # permit only this strict typed boundary, never ordinary fp32 CMAC/output.
  fp32_indexes = [u for u in sink.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float and
                  u.src[0].op is Ops.PARAM and u.src[0].buf_uop.arg.slot in fp32_inputs]
  cmac_cast_inputs = kind == "cmac" and bool(fp32_inputs) and not fp32_output and bool(fp32_indexes) and \
    all((consumers := [parent for parent in sink.toposort() if index in parent.src]) and
        all(parent.op is Ops.CAST and parent.dtype is dtypes.half for parent in consumers) for index in fp32_indexes)
  if (fp32_inputs or fp32_output) and kind not in ("dpu", "dpu_lut") and not cmac_cast_inputs:
    return f"RKPLAN_REJECT:unsupported_dtype:fp32_{kind}"
  return RKPlan(kind, sink, out_slots[0], in_slots, input_scale=input_scale, output_scale=output_scale, lut_op=lut_op,
                fp32_inputs=fp32_inputs, fp32_output=fp32_output, is_abs=abs_slot is not None,
                epilogue=epilogue, epilogue_scale=epilogue_scale,
                epilogue_bias_slot=epilogue_bias_slot, epilogue_bias_axis=epilogue_bias_axis,
                cmac_materialization=cmac_materialization)

# ---- geometry extraction ----
def _loop_extents(sink: UOp) -> list[int]:
  """Extents of all LOOP RANGE nodes in topological order (-1 if non-const)."""
  return [int(u.src[0].arg) if u.src[0].op is Ops.CONST else -1 for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]

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
                      fp32_inputs:tuple[int,...]=(), fp32_output=False, native_int32_output=False, native_int32_input=False,
                      native_int32_offset=0, fp32_residual_input=False) -> RKSubTask:
  """Fully-specified DPU stage used by the hardware-proven eight-pass WHERE lowering."""
  cmds:list[RKCmd] = []
  relocs:list[RKReloc] = []
  def e(t, r, v): emitter_emit(cmds, t, r, v)
  # Native int32 MRDMA uses a four-lane atom.  Ordinary fp16 input remains an
  # eight-lane atom, including when WDMA writes four usable int32 lanes.
  lanes = 4 if native_int32_input else 8
  atoms = (total+lanes-1)//lanes
  # The width field is only 13 bits.  Follow rknnops.h's elementwise row
  # layout for larger exact atoms instead of silently wrapping a flat width.
  # Partial final atoms stay one-row because a rectangular multi-row launch
  # would otherwise evaluate padding between logical elements.
  row_atoms = atoms
  rows = 1
  if atoms > 4096 and total % lanes == 0:
    row_atoms = min(atoms, 4096)
    while atoms % row_atoms: row_atoms -= 1
    rows = atoms // row_atoms
  width, height = row_atoms-1, rows-1
  stride_field = row_atoms * 2
  channel = (lanes-1 << 16) | (lanes-1)
  e(_T_DPU, rk.REG_DPU_S_POINTER, 0xe)
  e(_T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  # DPU supports native int32 WDMA (OUT_PRECISION=4).  Do not bounce a final
  # int32 result through fp16 and a CPU conversion.
  data_format = (4 if native_int32_output else 2) << 29 | (4 if native_int32_input else 2) << 26 | (4 if native_int32_input else 2)
  e(_T_DPU, rk.REG_DPU_DATA_FORMAT, data_format)
  if rows > 1: e(_T_DPU, rk.REG_DPU_DST_SURF_STRIDE, stride_field << 4)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, height)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
  e(_T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, channel)
  e(_T_DPU, rk.REG_DPU_BS_CFG, 0x53)
  e(_T_DPU, rk.REG_DPU_BN_CFG, 0x53)
  e(_T_DPU, rk.REG_DPU_BS_ALU_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BS_MUL_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BS_OW_CFG, 2)
  e(_T_DPU, rk.REG_DPU_WDMA_SIZE_0, lanes-1)
  e(_T_DPU, rk.REG_DPU_WDMA_SIZE_1, (height << 16) | width)
  e(_T_DPU, rk.REG_DPU_BN_MUL_CFG, 0)
  e(_T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0)
  if compare:
    e(_T_DPU, rk.REG_DPU_BS_CFG, 0x40040)
    e(_T_DPU, rk.REG_DPU_BS_ALU_CFG, 0x33800000)
    e(_T_DPU, rk.REG_DPU_BS_MUL_CFG, 0x40000000)
    e(_T_DPU, rk.REG_DPU_BN_CFG, 0x40082)
    e(_T_DPU, rk.REG_DPU_BN_MUL_CFG, 0x7c000000)
    e(_T_DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000)
  ew_cfg = ((_DPU_EW_CFGS[op] & ~(3 << 22)) | (3 << 22) | (1 << 8)) if native_int32_input else \
    (_EW_BASE | 1 if compare else _DPU_EW_CFGS[op])
  e(_T_DPU, rk.REG_DPU_EW_CFG, ew_cfg)
  e(_T_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)
  e(_T_DPU, rk.REG_DPU_OUT_CVT_OFFSET, native_int32_offset if native_int32_output else 0)
  out_cvt_scale = 0x10001 if native_int32_input and native_int32_output else \
    (1 if op is Ops.FDIV or native_int32_output else 0x10001)
  e(_T_DPU, rk.REG_DPU_OUT_CVT_SCALE, out_cvt_scale)
  # WIP reference: setting CVT_ROUND (1 << 30) was probed for native int32
  # output, but both modes round fp16 to nearest on RK3588.  Explicit NPU
  # roundoff correction is required for truncation toward zero.
  e(_T_DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0)
  # Preserve the hardware-proven flat-stage value.  rknnops.h only needs the
  # row stride here when DATA_CUBE_HEIGHT is nonzero.
  e(_T_DPU, rk.REG_DPU_SURFACE_ADD, 0x40 if rows == 1 else stride_field << 4)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, height)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, lanes-1)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x4000000c if native_int32_input else 0x40000008)
  for target, reg, arg in ((_T_DPU, rk.REG_DPU_DST_BASE_ADDR, (out_slot, 0)),
                           (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, a),
                           (_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, b)):
    e(target, reg, 0)
    emitter_reloc(cmds, relocs, arg[0], arg[1])
  if rows > 1: e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, stride_field << 4)
  rdma_feature = 0x27881 if native_int32_input else (0x17841 if op is Ops.FDIV else 0x17849)
  e(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feature)
  emitter_pc_op_en(cmds, 12)
  task = RKTask(0x18, 0x300, 4, "dpu", (total,), out_slot, bool_inputs=bool_inputs, int32_inputs=int32_inputs,
                broadcast_inputs=broadcast_inputs, int32_output=int32_output, uint8_output=uint8_output,
                bool_output=bool_output, trunc_output=trunc_output, comparison_inputs=comparison_inputs,
                fp32_inputs=fp32_inputs, fp32_output=fp32_output, native_int32_output=native_int32_output,
                native_int32_input=native_int32_input, fp32_residual_input=fp32_residual_input)
  return RKSubTask(tuple(c.pack() for c in cmds), task, tuple(relocs))

def _emit_trunc_stage(total:int, out_slot:int, source:tuple[int,int]) -> RKSubTask:
  """Run an identity DPU stage, then apply the fp16→int32→fp16 cast boundary."""
  return _emit_where_stage(total, out_slot, source, (_ZERO_SLOT, 0), Ops.ADD, trunc_output=True)

def _emit_positive_mask(tasks:list[RKSubTask], total:int, source:int, alloc:Callable[[],int]) -> int:
  """Append ordinary DPU stages yielding exact fp16 1 when source>0, else 0."""
  zero = (_ZERO_SLOT, 0)
  minus_one = (_CONST_SLOT, 0xbf800000)
  finite_min = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -65504.0))[0])
  min_subnormal = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 2**-24))[0])
  positive, neg_warm, negative, lower = alloc(), alloc(), alloc(), alloc()
  restore_warm, clamped, denominator, mask = alloc(), alloc(), alloc(), alloc()
  tasks.append(_emit_where_stage(total, positive, (source, 0), zero, Ops.MAX))
  tasks.append(_emit_where_stage(total, neg_warm, (positive, 0), minus_one, Ops.MUL))
  tasks.append(_emit_where_stage(total, negative, (positive, 0), minus_one, Ops.MUL))
  tasks.append(_emit_where_stage(total, lower, (negative, 0), finite_min, Ops.MAX))
  tasks.append(_emit_where_stage(total, restore_warm, (lower, 0), minus_one, Ops.MUL))
  tasks.append(_emit_where_stage(total, clamped, (lower, 0), minus_one, Ops.MUL))
  tasks.append(_emit_where_stage(total, denominator, (clamped, 0), min_subnormal, Ops.MAX))
  tasks.append(_emit_where_stage(total, mask, (clamped, 0), (denominator, 0), Ops.FDIV))
  return mask

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
                            int32_inputs=((input_slot,) if source.dtype is dtypes.int and not getenv("ROCKCHIP_NATIVE_INT_CAST") else ()),
                            fp32_inputs=((input_slot,) if source.dtype is dtypes.float else ()),
                            fp32_output=output_dtype is dtypes.float, int32_output=output_dtype is dtypes.int,
                            uint8_output=output_dtype is dtypes.uint8, bool_output=output_dtype is dtypes.bool,
                            native_int32_input=source.dtype is dtypes.int and bool(getenv("ROCKCHIP_NATIVE_INT_CAST"))),)

def _try_typed_fill_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Fill non-fp16 outputs through the DPU, then convert the fp16 result buffer."""
  store = _store_node(sink)
  if store is None or store.src[1].op is not Ops.CONST: return None
  output_dtype = store.src[0].dtype
  if output_dtype not in (dtypes.float, dtypes.int, dtypes.bool, dtypes.uint8): return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  value = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', float(store.src[1].arg)))[0])
  if output_dtype is dtypes.float and total*4 >= 2*1024*1024:
    # Logical multi-megabyte buffers are host-backed because RK3588 GEM mmap
    # rejects them. Generate reusable fp16 tiles on DPU, then widen only the
    # ABI representation into the logical fp32 buffer on the host.
    scratch_slot = max(info.globals, default=-1)+1
    tasks:list[RKSubTask] = []
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    tile = 262144
    for start in range(0, total, tile):
      count = min(tile, total-start)
      tasks.append(_emit_where_stage(count, scratch_slot, (_ZERO_SLOT, 0), value, Ops.ADD))
      relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, scratch_slot, 0, 0, 0xFFFFFFFF))
      task = RKTask(0, 0, 0, "dpu", (count, _HOST_HALF_FP32_LAYOUT), info.outs[0], is_copy=True, out_offset=start*4)
      tasks.append(RKSubTask(host_cmds, task, relocs))
    return tuple(tasks)
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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  source_slot = source.src[0].buf_uop.arg.slot
  fp32_in = store.src[0].src[0].dtype is dtypes.float
  fp32_args = {"fp32_inputs": (source_slot,)} if fp32_in else {}
  fp32_out = {"fp32_output": True} if fp32_in else {}
  negative, magnitude, rounded = alloc(), alloc(), alloc()
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  tasks = [_emit_where_stage(total, negative, (source_slot, 0), negative_one, Ops.MUL, **fp32_args),
           _emit_where_stage(total, magnitude, (source_slot, 0), (negative, 0), Ops.MAX, **fp32_args)]
  roundoff = UOp(Ops.CUSTOM, dtypes.half, (temp_index(magnitude),), arg="rk_roundoff")
  stage_store = store.replace(src=(temp_index(rounded), roundoff))
  stage_sink = sink.substitute({store:stage_store})
  plan = plan_rk(stage_sink)
  if isinstance(plan, str) or plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  zero = (_ZERO_SLOT, 0)
  negative_diff, negative_mask, positive_diff, positive_mask, sign = alloc(), alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, zero, (source_slot, 0), Ops.SUB, **fp32_args),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, (source_slot, 0), zero, Ops.SUB, **fp32_args),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, alloc(), (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, sign, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, alloc(), (rounded, 0), (sign, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (rounded, 0), (sign, 0), Ops.MUL, **fp32_out)))
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
  fp32_in = store.src[0].src[0].dtype is dtypes.float
  fp32_args = {"fp32_inputs": (input_slot,)} if fp32_in else {}
  fp32_out = {"fp32_output": True} if fp32_in else {}
  negative_diff, negative_mask, positive_diff, positive_mask = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, zero, (input_slot, 0), Ops.SUB, **fp32_args),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, (input_slot, 0), zero, Ops.SUB, **fp32_args),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, alloc(), (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, info.outs[0], (positive_mask, 0), (negative_mask, 0), Ops.SUB, **fp32_out)))
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
  fp32_in = store.src[0].src[0].dtype is dtypes.float
  fp32_args = {"fp32_inputs": (denominator_arg[0],)} if fp32_in else {}
  fp32_out = {"fp32_output": True} if fp32_in else {}
  base, negative_diff, negative_mask, positive_diff, positive_mask = alloc(), alloc(), alloc(), alloc(), alloc()
  sign_scratch, sign, product_scratch = alloc(), alloc(), alloc()
  zero = (_ZERO_SLOT, 0)
  tasks.extend((_emit_where_stage(total, base, numerator_arg, denominator_arg, Ops.FDIV, **fp32_args),
                _emit_where_stage(total, negative_diff, zero, denominator_arg, Ops.SUB, **fp32_args),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, denominator_arg, zero, Ops.SUB, **fp32_args),
                _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                # Repeat the first dependent reads of freshly materialized comparison masks.
                _emit_where_stage(total, sign_scratch, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, sign, (positive_mask, 0), (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, product_scratch, (base, 0), (sign, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (base, 0), (sign, 0), Ops.MUL, **fp32_out)))
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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
  """Use direct Q15 tanh in [-4,4], exact sign tails, and restore NaN."""
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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  base_slot = alloc()
  direct_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_tanh")
  direct_store = store.replace(src=(temp_index(base_slot), direct_val))
  direct_plan = plan_rk(sink.substitute({store:direct_store}))
  if isinstance(direct_plan, str) or direct_plan.kind != "dpu_lut":
    # Preserve the older staged sigmoid interior as a fallback for scheduler changes.
    base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
    base_tasks = _try_elementwise_subtasks(sink.substitute({store:base_store}))
    if base_tasks is None: return None
    tasks.extend(base_tasks)
    used_slots = [st.task.out_slot for st in base_tasks] + \
      [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1)+1)
  else:
    cmds, task, relocs = emit_rk(direct_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, one = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0)
  scaled = alloc()
  tasks.append(_emit_where_stage(total, scaled, source_arg, scalar(16.0), Ops.MUL))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(scaled),), arg="rk_tanh_local")
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(0.25), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(0.25), Ops.MUL)))
  local_low_diff, local_low, local_high_diff, local_high = (alloc() for _ in range(4))
  local_outside_scratch, local_outside, local_inside_scratch, local_inside = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, local_low_diff, scalar(-0.25), source_arg, Ops.SUB),
                _emit_where_stage(total, local_low, (local_low_diff, 0), (local_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_high_diff, source_arg, scalar(0.25), Ops.SUB),
                _emit_where_stage(total, local_high, (local_high_diff, 0), (local_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside_scratch, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_outside, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_inside_scratch, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB)))
  near_low_diff, near_low, near_high_diff, near_high = (alloc() for _ in range(4))
  near_outside_scratch, near_outside, near_inside_scratch, near_inside = (alloc() for _ in range(4))
  local_mask_scratch, local_mask = alloc(), alloc()
  clamp_low, negated, negated_clamped, identity = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, near_low_diff, scalar(-0.04), source_arg, Ops.SUB),
                _emit_where_stage(total, near_low, (near_low_diff, 0), (near_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_high_diff, source_arg, scalar(0.04), Ops.SUB),
                _emit_where_stage(total, near_high, (near_high_diff, 0), (near_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_outside_scratch, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_outside, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_inside_scratch, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask_scratch, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, clamp_low, source_arg, scalar(-0.04), Ops.MAX),
                _emit_where_stage(total, negated, (_ZERO_SLOT, 0), (clamp_low, 0), Ops.SUB),
                _emit_where_stage(total, negated_clamped, (negated, 0), scalar(-0.04), Ops.MAX),
                _emit_where_stage(total, identity, (_ZERO_SLOT, 0), (negated_clamped, 0), Ops.SUB)))
  broad_selected_scratch, broad_selected, local_selected_scratch, local_selected = (alloc() for _ in range(4))
  identity_selected_scratch, identity_selected, lut_sum_scratch, lut_sum, interior_slot = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, broad_selected_scratch, (base_slot, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (base_slot, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, identity_selected_scratch, (identity, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, identity_selected, (identity, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum_scratch, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, alloc(), (lut_sum, 0), (identity_selected, 0), Ops.ADD),
                _emit_where_stage(total, interior_slot, (lut_sum, 0), (identity_selected, 0), Ops.ADD)))
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
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (interior_slot, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (interior_slot, 0), (inside, 0), Ops.MUL),
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

def _try_fp32_tanh_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Keep exact normal-fp32 tanh out of the half-buffer two-LUT implementation."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float or (source := _try_tanh(store.src[1])) is None: return None
  if source.dtype is not dtypes.float or source.src[0].op is not Ops.PARAM: return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

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
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))

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

def _try_quick_gelu_direct_two_lut_wip(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Use broad/local QuickGELU LUTs inside [-2,2] and retain the staged asymptotic fallback."""
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
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))

  base_slot = alloc()
  base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
  base_tasks = _try_quick_gelu_saturation_subtasks(sink.substitute({store:base_store}))
  if base_tasks is None: return None
  tasks.extend(base_tasks)
  used_slots = [st.task.out_slot for st in base_tasks] + \
    [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  broad_slot = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_quick_gelu")
  broad_store = store.replace(src=(temp_index(broad_slot), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, one, zero = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0), (_ZERO_SLOT, 0)
  scaled = alloc()
  tasks.append(_emit_where_stage(total, scaled, source_arg, scalar(14.25), Ops.MUL))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(scaled),), arg="rk_quick_gelu_local")
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(0.0625), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(0.0625), Ops.MUL)))

  below_diff, below, above_diff, above = (alloc() for _ in range(4))
  local_below_diff, local_below, local_above_diff, local_above = (alloc() for _ in range(4))
  negative_diff, negative, positive_diff, positive = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, below_diff, scalar(-2.0), source_arg, Ops.SUB),
                _emit_where_stage(total, below, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, above_diff, source_arg, scalar(2.0), Ops.SUB),
                _emit_where_stage(total, above, (above_diff, 0), (above_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_below_diff, scalar(-0.14), source_arg, Ops.SUB),
                _emit_where_stage(total, local_below, (local_below_diff, 0), (local_below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_above_diff, source_arg, scalar(0.11), Ops.SUB),
                _emit_where_stage(total, local_above, (local_above_diff, 0), (local_above_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, source_arg, zero, Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True)))

  outside_scratch, outside, inside_scratch, inside = (alloc() for _ in range(4))
  local_outside_scratch, local_outside, local_inside_scratch, local_inside = (alloc() for _ in range(4))
  nonzero_scratch, nonzero, local_mask_scratch, local_mask = (alloc() for _ in range(4))
  broad_mask_scratch, broad_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, outside_scratch, (below, 0), (above, 0), Ops.MAX),
                _emit_where_stage(total, outside, (below, 0), (above, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, local_outside_scratch, (local_below, 0), (local_above, 0), Ops.MAX),
                _emit_where_stage(total, local_outside, (local_below, 0), (local_above, 0), Ops.MAX),
                _emit_where_stage(total, local_inside_scratch, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, nonzero_scratch, (negative, 0), (positive, 0), Ops.MAX),
                _emit_where_stage(total, nonzero, (negative, 0), (positive, 0), Ops.MAX),
                _emit_where_stage(total, local_mask_scratch, (local_inside, 0), (nonzero, 0), Ops.MUL),
                _emit_where_stage(total, local_mask, (local_inside, 0), (nonzero, 0), Ops.MUL),
                _emit_where_stage(total, broad_mask_scratch, (inside, 0), (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (inside, 0), (local_inside, 0), Ops.SUB)))

  base_selected_scratch, base_selected, broad_selected_scratch, broad_selected = (alloc() for _ in range(4))
  local_selected_scratch, local_selected = alloc(), alloc()
  inner_scratch, inner = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (base_slot, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (base_slot, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected_scratch, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, inner_scratch, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, inner, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, alloc(), (base_selected, 0), (inner, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (base_selected, 0), (inner, 0), Ops.ADD)))
  return tuple(tasks)

def _try_quick_gelu_two_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Combine broad/negative QuickGELU LUTs, a near-zero polynomial, and the staged wide fallback."""
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
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))

  base_slot = alloc()
  base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
  base_tasks = _try_quick_gelu_saturation_subtasks(sink.substitute({store:base_store}))
  if base_tasks is None: return None
  tasks.extend(base_tasks)
  used_slots = [st.task.out_slot for st in base_tasks] + \
    [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  broad_slot = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_quick_gelu")
  broad_store = store.replace(src=(temp_index(broad_slot), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, one = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0)
  shifted, scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, shifted, source_arg, scalar(1.5), Ops.ADD),
                _emit_where_stage(total, scaled, (shifted, 0), scalar(4.0), Ops.MUL)))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(scaled),), arg="rk_quick_gelu_local")
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  below_diff, below, above_diff, above = (alloc() for _ in range(4))
  local_above_diff, local_above = alloc(), alloc()
  poly_below_diff, poly_below, poly_above_diff, poly_above = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, below_diff, scalar(-2.0), source_arg, Ops.SUB),
                _emit_where_stage(total, below, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, above_diff, source_arg, scalar(2.0), Ops.SUB),
                _emit_where_stage(total, above, (above_diff, 0), (above_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_above_diff, source_arg, scalar(-1.0), Ops.SUB),
                _emit_where_stage(total, local_above, (local_above_diff, 0), (local_above_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_below_diff, scalar(-0.16), source_arg, Ops.SUB),
                _emit_where_stage(total, poly_below, (poly_below_diff, 0), (poly_below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_above_diff, source_arg, scalar(0.16), Ops.SUB),
                _emit_where_stage(total, poly_above, (poly_above_diff, 0), (poly_above_diff, 0), Ops.MAX, compare=True)))

  outside_scratch, outside, inside_scratch, inside = (alloc() for _ in range(4))
  local_outside_scratch, local_outside, local_inside_scratch, local_inside = (alloc() for _ in range(4))
  poly_outside_scratch, poly_outside, poly_inside_scratch, poly_inside = (alloc() for _ in range(4))
  broad_no_local_scratch, broad_no_local, broad_mask_scratch, broad_mask = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, outside_scratch, (below, 0), (above, 0), Ops.MAX),
                _emit_where_stage(total, outside, (below, 0), (above, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, local_outside_scratch, (below, 0), (local_above, 0), Ops.MAX),
                _emit_where_stage(total, local_outside, (below, 0), (local_above, 0), Ops.MAX),
                _emit_where_stage(total, local_inside_scratch, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, poly_outside_scratch, (poly_below, 0), (poly_above, 0), Ops.MAX),
                _emit_where_stage(total, poly_outside, (poly_below, 0), (poly_above, 0), Ops.MAX),
                _emit_where_stage(total, poly_inside_scratch, one, (poly_outside, 0), Ops.SUB),
                _emit_where_stage(total, poly_inside, one, (poly_outside, 0), Ops.SUB),
                _emit_where_stage(total, broad_no_local_scratch, (inside, 0), (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_no_local, (inside, 0), (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask_scratch, (broad_no_local, 0), (poly_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (broad_no_local, 0), (poly_inside, 0), Ops.SUB)))

  poly_x_scratch, poly_x, half_x, square, quadratic, polynomial = (alloc() for _ in range(6))
  tasks.extend((_emit_where_stage(total, poly_x_scratch, source_arg, (poly_inside, 0), Ops.MUL),
                _emit_where_stage(total, poly_x, source_arg, (poly_inside, 0), Ops.MUL),
                _emit_where_stage(total, half_x, (poly_x, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, square, (poly_x, 0), (poly_x, 0), Ops.MUL),
                _emit_where_stage(total, quadratic, (square, 0), scalar(0.4253), Ops.MUL),
                _emit_where_stage(total, polynomial, (half_x, 0), (quadratic, 0), Ops.ADD)))

  base_selected_scratch, base_selected, broad_selected_scratch, broad_selected = (alloc() for _ in range(4))
  local_selected_scratch, local_selected = alloc(), alloc()
  lut_result, inner = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (base_slot, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (base_slot, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected_scratch, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_slot, 0), (local_inside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_slot, 0), (local_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_result, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, inner, (lut_result, 0), (polynomial, 0), Ops.ADD),
                _emit_where_stage(total, alloc(), (base_selected, 0), (inner, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (base_selected, 0), (inner, 0), Ops.ADD)))
  return tuple(tasks)

def _try_logsigmoid_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate LogSigmoid as min(x,0) plus broad and amplified-tail correction LUTs."""
  store = _store_node(sink)
  source = None if store is None else _try_logsigmoid(store.src[1])
  if getenv("RK_TRACE_MATCH") and store is not None:
    val = _unwrap(store.src[1])
    print("rk logsigmoid match", source is not None, val.op, {op:sum(u.op is op for u in val.toposort())
      for op in (Ops.EXP2, Ops.LOG2, Ops.MAX, Ops.CAST)},
      [(x.op, x.arg, [(y.op, y.arg) for y in x.src]) for x in val.src],
      [(u.dtype, u.src[0].buf_uop.arg.slot) for u in val.toposort() if u.op is Ops.INDEX],
      set(u.op for u in val.toposort()))
  if store is None or source is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  correction = alloc()
  lut_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_logsigmoid_correction")
  lut_store = store.replace(src=(temp_index(correction), lut_val))
  lut_plan = plan_rk(sink.substitute({store:lut_store}))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks = [RKSubTask(cmds, task, relocs)]

  tail = alloc()
  tail_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_logsigmoid_tail")
  tail_store = store.replace(src=(temp_index(tail), tail_val))
  tail_plan = plan_rk(sink.substitute({store:tail_store}))
  if isinstance(tail_plan, str) or tail_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(tail_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  positive, minimum, tail_diff, tail_mask, broad_mask = (alloc() for _ in range(5))
  broad_selected, tail_scaled, tail_selected, selected_correction = (alloc() for _ in range(4))
  raw_output, negated_output, clamped_output = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, positive, source_arg, zero, Ops.MAX),
                _emit_where_stage(total, minimum, source_arg, (positive, 0), Ops.SUB),
                _emit_where_stage(total, tail_diff, source_arg, scalar(3.5), Ops.SUB),
                _emit_where_stage(total, tail_mask, (tail_diff, 0), (tail_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, broad_mask, one, (tail_mask, 0), Ops.SUB),
                _emit_where_stage(total, broad_selected, (correction, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, tail_scaled, (tail, 0), scalar(1/32), Ops.MUL),
                _emit_where_stage(total, tail_selected, (tail_scaled, 0), (tail_mask, 0), Ops.MUL),
                _emit_where_stage(total, selected_correction, (broad_selected, 0), (tail_selected, 0), Ops.ADD),
                _emit_where_stage(total, raw_output, (minimum, 0), (selected_correction, 0), Ops.ADD),
                _emit_where_stage(total, negated_output, (raw_output, 0), scalar(-1.0), Ops.MUL),
                _emit_where_stage(total, clamped_output, (negated_output, 0), zero, Ops.MAX),
                _emit_where_stage(total, info.outs[0], (clamped_output, 0), scalar(-1.0), Ops.MUL)))
  return tuple(tasks)

def _try_softplus_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate Softplus as max(x,0) minus broad/amplified-tail correction LUTs."""
  store = _store_node(sink)
  match = None if store is None else _try_softplus(store.src[1])
  if getenv("RK_TRACE_MATCH") and store is not None:
    val = _unwrap(store.src[1])
    match_info = None if match is None else (match[0].src[0].buf_uop.arg.slot, match[1])
    print("rk softplus match", match_info, val.op, {op:sum(u.op is op for u in val.toposort())
      for op in (Ops.EXP2, Ops.LOG2, Ops.MAX, Ops.CAST)})
  if store is None or match is None: return None
  source, beta = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  tasks:list[RKSubTask] = []
  if source.op is not Ops.INDEX or not _is_flat_contiguous(source.src[1]):
    materialized_source = alloc()
    source_store = store.replace(src=(temp_index(materialized_source), source))
    source_plan = plan_rk(sink.substitute({store:source_store}))
    if isinstance(source_plan, str) or source_plan.kind != "dpu":
      if getenv("RK_TRACE_MATCH"): print("rk softplus source reject", source_plan)
      return None
    cmds, task, relocs = emit_rk(source_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    source = temp_index(materialized_source)
  source_arg = (source.src[0].buf_uop.arg.slot, 0)
  if beta < 1.0:
    correction, positive, raw_output = alloc(), alloc(), alloc()
    far_diff, far_mask, finite_mask, finite_output = (alloc() for _ in range(4))
    wide_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg=("rk_softplus_wide", beta))
    wide_store = store.replace(src=(temp_index(correction), wide_val))
    wide_plan = plan_rk(sink.substitute({store:wide_store}))
    if isinstance(wide_plan, str) or wide_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(wide_plan)
    return (*tasks, RKSubTask(cmds, task, relocs),
            _emit_where_stage(total, positive, source_arg, (_ZERO_SLOT, 0), Ops.MAX),
            _emit_where_stage(total, raw_output, (positive, 0), (correction, 0), Ops.SUB),
            _emit_where_stage(total, far_diff, scalar(-100.0), source_arg, Ops.SUB),
            _emit_where_stage(total, far_mask, (far_diff, 0), (far_diff, 0), Ops.MAX, compare=True),
            _emit_where_stage(total, finite_mask, scalar(1.0), (far_mask, 0), Ops.SUB),
            _emit_where_stage(total, finite_output, (raw_output, 0), (finite_mask, 0), Ops.MUL),
            _emit_where_stage(total, info.outs[0], (finite_output, 0), (_ZERO_SLOT, 0), Ops.MAX))

  correction = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg=("rk_logsigmoid_correction", beta, 1/beta))
  broad_store = store.replace(src=(temp_index(correction), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut":
    if getenv("RK_TRACE_MATCH"): print("rk softplus broad reject", broad_plan)
    return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  tail = alloc()
  tail_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg=("rk_softplus_tail", beta, 1/beta))
  tail_store = store.replace(src=(temp_index(tail), tail_val))
  tail_plan = plan_rk(sink.substitute({store:tail_store}))
  if isinstance(tail_plan, str) or tail_plan.kind != "dpu_lut":
    if getenv("RK_TRACE_MATCH"): print("rk softplus tail reject", tail_plan)
    return None
  cmds, task, relocs = emit_rk(tail_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  zero, one = (_ZERO_SLOT, 0), scalar(1.0)
  positive, tail_diff, tail_mask, broad_mask = (alloc() for _ in range(4))
  broad_selected, tail_scaled, tail_selected, selected_correction, raw_output = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, positive, source_arg, zero, Ops.MAX),
                _emit_where_stage(total, tail_diff, scalar(-3.05/beta), source_arg, Ops.SUB),
                _emit_where_stage(total, tail_mask, (tail_diff, 0), (tail_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, broad_mask, one, (tail_mask, 0), Ops.SUB),
                _emit_where_stage(total, broad_selected, (correction, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, tail_scaled, (tail, 0), scalar(1/21), Ops.MUL),
                _emit_where_stage(total, tail_selected, (tail_scaled, 0), (tail_mask, 0), Ops.MUL),
                _emit_where_stage(total, selected_correction, (broad_selected, 0), (tail_selected, 0), Ops.ADD),
                _emit_where_stage(total, raw_output, (positive, 0), (selected_correction, 0), Ops.SUB),
                _emit_where_stage(total, info.outs[0], (raw_output, 0), zero, Ops.MAX)))
  return tuple(tasks)

def _try_mish_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate Mish with broad/local LUTs and a near-zero Taylor interval."""
  store = _store_node(sink)
  if store is None or (source := _try_mish(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  broad = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_mish")
  broad_store = store.replace(src=(temp_index(broad), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks = [RKSubTask(cmds, task, relocs)]

  source_arg, one = (source.src[0].buf_uop.arg.slot, 0), scalar(1.0)
  zoomed = alloc()
  tasks.append(_emit_where_stage(total, zoomed, source_arg, scalar(2.0), Ops.MUL))
  local = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg="rk_mish_local")
  local_store = store.replace(src=(temp_index(local), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  range_low_diff, range_low, range_high_diff, range_high = (alloc() for _ in range(4))
  range_outside, range_inside = alloc(), alloc()
  local_low_diff, local_low, local_high_diff, local_high = (alloc() for _ in range(4))
  local_outside, local_inside, poly_low_diff, poly_low = (alloc() for _ in range(4))
  poly_high_diff, poly_high, poly_outside, poly_inside = (alloc() for _ in range(4))
  local_mask, broad_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, range_low_diff, scalar(-8.0), source_arg, Ops.SUB),
                _emit_where_stage(total, range_low, (range_low_diff, 0), (range_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, range_high_diff, source_arg, scalar(8.0), Ops.SUB),
                _emit_where_stage(total, range_high, (range_high_diff, 0), (range_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, range_outside, (range_low, 0), (range_high, 0), Ops.MAX),
                _emit_where_stage(total, range_inside, one, (range_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_low_diff, scalar(-1.0), source_arg, Ops.SUB),
                _emit_where_stage(total, local_low, (local_low_diff, 0), (local_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_high_diff, source_arg, scalar(1.0), Ops.SUB),
                _emit_where_stage(total, local_high, (local_high_diff, 0), (local_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, poly_low_diff, scalar(-0.08), source_arg, Ops.SUB),
                _emit_where_stage(total, poly_low, (poly_low_diff, 0), (poly_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_high_diff, source_arg, scalar(0.08), Ops.SUB),
                _emit_where_stage(total, poly_high, (poly_high_diff, 0), (poly_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_outside, (poly_low, 0), (poly_high, 0), Ops.MAX),
                _emit_where_stage(total, poly_inside, one, (poly_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (poly_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (range_inside, 0), (local_inside, 0), Ops.SUB)))

  bounded_low, neg_bounded_low, neg_clamped, bounded = (alloc() for _ in range(4))
  poly_source, linear, square, quadratic, polynomial = (alloc() for _ in range(5))
  positive_diff, positive, positive_extra, broad_scale, broad_scaled = (alloc() for _ in range(5))
  broad_selected, local_scaled, local_selected, lut_sum = (alloc() for _ in range(4))
  fallback, fallback_selected, inner = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, bounded_low, source_arg, scalar(-0.08), Ops.MAX),
                _emit_where_stage(total, neg_bounded_low, (_ZERO_SLOT, 0), (bounded_low, 0), Ops.SUB),
                _emit_where_stage(total, neg_clamped, (neg_bounded_low, 0), scalar(-0.08), Ops.MAX),
                _emit_where_stage(total, bounded, (_ZERO_SLOT, 0), (neg_clamped, 0), Ops.SUB),
                _emit_where_stage(total, poly_source, (bounded, 0), (poly_inside, 0), Ops.MUL),
                _emit_where_stage(total, linear, (poly_source, 0), scalar(0.6), Ops.MUL),
                _emit_where_stage(total, square, (poly_source, 0), (poly_source, 0), Ops.MUL),
                _emit_where_stage(total, quadratic, (square, 0), scalar(0.32), Ops.MUL),
                _emit_where_stage(total, polynomial, (linear, 0), (quadratic, 0), Ops.ADD),
                _emit_where_stage(total, positive_diff, source_arg, (_ZERO_SLOT, 0), Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_extra, (positive, 0), scalar(7.0), Ops.MUL),
                _emit_where_stage(total, broad_scale, one, (positive_extra, 0), Ops.ADD),
                _emit_where_stage(total, broad_scaled, (broad, 0), (broad_scale, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_scaled, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local, 0), one, Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, fallback, source_arg, (_ZERO_SLOT, 0), Ops.MAX),
                _emit_where_stage(total, fallback_selected, (fallback, 0), (range_outside, 0), Ops.MUL),
                _emit_where_stage(total, inner, (lut_sum, 0), (polynomial, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (inner, 0), (fallback_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_erf_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate erf with broad/local LUTs, a near-zero line, and exact tails."""
  store = _store_node(sink)
  if store is None or (source := _try_erf(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def clamp_symmetric(limit:float) -> tuple[int,list[RKSubTask]]:
    low, neg, neg_clamped, bounded = (alloc() for _ in range(4))
    stages = [_emit_where_stage(total, low, source_arg, scalar(-limit), Ops.MAX),
              _emit_where_stage(total, neg, zero, (low, 0), Ops.SUB),
              _emit_where_stage(total, neg_clamped, (neg, 0), scalar(-limit), Ops.MAX),
              _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB)]
    return bounded, stages

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  broad_input, tasks = clamp_symmetric(4.0)
  broad_slot = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(broad_input),), arg="rk_erf")
  broad_store = store.replace(src=(temp_index(broad_slot), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  local_input, local_stages = clamp_symmetric(0.25)
  tasks.extend(local_stages)
  zoomed = alloc()
  tasks.append(_emit_where_stage(total, zoomed, (local_input, 0), scalar(16.0), Ops.MUL))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg="rk_erf_local")
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  low_diff, low, high_diff, high = (alloc() for _ in range(4))
  local_low_diff, local_low, local_high_diff, local_high = (alloc() for _ in range(4))
  near_low_diff, near_low, near_high_diff, near_high = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, low_diff, scalar(-4.0), source_arg, Ops.SUB),
                _emit_where_stage(total, low, (low_diff, 0), (low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, high_diff, source_arg, scalar(4.0), Ops.SUB),
                _emit_where_stage(total, high, (high_diff, 0), (high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_low_diff, scalar(-0.25), source_arg, Ops.SUB),
                _emit_where_stage(total, local_low, (local_low_diff, 0), (local_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_high_diff, source_arg, scalar(0.25), Ops.SUB),
                _emit_where_stage(total, local_high, (local_high_diff, 0), (local_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_low_diff, scalar(-0.04), source_arg, Ops.SUB),
                _emit_where_stage(total, near_low, (near_low_diff, 0), (near_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_high_diff, source_arg, scalar(0.04), Ops.SUB),
                _emit_where_stage(total, near_high, (near_high_diff, 0), (near_high_diff, 0), Ops.MAX, compare=True)))

  outside_scratch, outside, inside_scratch, inside = (alloc() for _ in range(4))
  local_outside_scratch, local_outside, local_inside_scratch, local_inside = (alloc() for _ in range(4))
  near_outside_scratch, near_outside, near_inside_scratch, near_inside = (alloc() for _ in range(4))
  broad_mask_scratch, broad_mask, local_mask_scratch, local_mask = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, outside_scratch, (low, 0), (high, 0), Ops.MAX),
                _emit_where_stage(total, outside, (low, 0), (high, 0), Ops.MAX),
                _emit_where_stage(total, inside_scratch, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, inside, one, (outside, 0), Ops.SUB),
                _emit_where_stage(total, local_outside_scratch, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_outside, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_inside_scratch, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_outside_scratch, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_outside, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_inside_scratch, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask_scratch, (inside, 0), (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (inside, 0), (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask_scratch, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB)))

  near_input, near_stages = clamp_symmetric(0.04)
  tasks.extend(near_stages)
  identity = alloc()
  tasks.append(_emit_where_stage(total, identity, (near_input, 0), scalar(2/math.sqrt(math.pi)), Ops.MUL))
  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(1/3), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(1/3), Ops.MUL)))

  broad_selected_scratch, broad_selected, local_selected_scratch, local_selected = (alloc() for _ in range(4))
  identity_selected_scratch, identity_selected = alloc(), alloc()
  lut_sum_scratch, lut_sum, interior_scratch, interior = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, broad_selected_scratch, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_slot, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, identity_selected_scratch, (identity, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, identity_selected, (identity, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum_scratch, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, interior_scratch, (lut_sum, 0), (identity_selected, 0), Ops.ADD),
                _emit_where_stage(total, interior, (lut_sum, 0), (identity_selected, 0), Ops.ADD)))
  sign_scratch, sign, tail_scratch, tail = (alloc() for _ in range(4))
  interior_selected_scratch, interior_selected = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, sign_scratch, (high, 0), (low, 0), Ops.SUB),
                _emit_where_stage(total, sign, (high, 0), (low, 0), Ops.SUB),
                _emit_where_stage(total, tail_scratch, (sign, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, tail, (sign, 0), (outside, 0), Ops.MUL),
                _emit_where_stage(total, interior_selected_scratch, (interior, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, interior_selected, (interior, 0), (inside, 0), Ops.MUL),
                _emit_where_stage(total, alloc(), (interior_selected, 0), (tail, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (interior_selected, 0), (tail, 0), Ops.ADD)))
  return tuple(tasks)

def _try_sin_cos_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Range-reduce sin/cos, then select a broad LUT, amplified local LUT, or the near-zero identity."""
  store = _store_node(sink)
  if store is None or (match := _try_sin_cos(store.src[1])) is None: return None
  source, is_cos = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  neg_source, abs_source, invalid_diff, invalid_scratch, invalid = (alloc() for _ in range(5))
  # Bound infinities before the host-assisted truncation stage. The final x*0
  # term restores NaN for both infinities and NaN without perturbing finite x.
  low, neg_low, neg_clamped, bounded = (alloc() for _ in range(4))
  tasks:list[RKSubTask] = [
    _emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
    _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX),
    _emit_where_stage(total, invalid_diff, (abs_source, 0), scalar(20000.0), Ops.SUB),
    _emit_where_stage(total, invalid_scratch, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
    _emit_where_stage(total, invalid, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
    _emit_where_stage(total, low, source_arg, scalar(-10000.0), Ops.MAX),
    _emit_where_stage(total, neg_low, zero, (low, 0), Ops.SUB),
    _emit_where_stage(total, neg_clamped, (neg_low, 0), scalar(-10000.0), Ops.MAX),
    _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB)]
  phase = bounded
  # The first cosine implementation used this literal lowering:
  # if is_cos:
  #   phase = alloc()
  #   tasks.append(_emit_where_stage(total, phase, scalar(math.pi/2), (bounded, 0), Ops.SUB))
  # It is kept as tuning reference, but materializing the float32 phase as fp16
  # caused up to 0.0021 error. Direct cosine LUTs below use x as the phase.

  # n=round(phase/(2*pi)).  Restoring the sign after trunc(abs(q)+0.5)
  # avoids a full lowering of tinygrad's general round-to-even expression.
  quotient, neg_quotient, abs_quotient, biased, rounded_abs = (alloc() for _ in range(5))
  positive_diff, positive, negative, positive_n = (alloc() for _ in range(4))
  negative_n, signed_negative_n, rounded = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, quotient, (phase, 0), scalar(1/(2*math.pi)), Ops.MUL),
                _emit_where_stage(total, neg_quotient, zero, (quotient, 0), Ops.SUB),
                _emit_where_stage(total, abs_quotient, (quotient, 0), (neg_quotient, 0), Ops.MAX),
                _emit_where_stage(total, biased, (abs_quotient, 0), scalar(0.5), Ops.ADD),
                _emit_trunc_stage(total, rounded_abs, (biased, 0)),
                _emit_where_stage(total, positive_diff, (quotient, 0), zero, Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative, one, (positive, 0), Ops.SUB),
                _emit_where_stage(total, positive_n, (rounded_abs, 0), (positive, 0), Ops.MUL),
                _emit_where_stage(total, negative_n, (rounded_abs, 0), (negative, 0), Ops.MUL),
                _emit_where_stage(total, signed_negative_n, zero, (negative_n, 0), Ops.SUB),
                _emit_where_stage(total, rounded, (positive_n, 0), (signed_negative_n, 0), Ops.ADD)))

  # Cody-Waite-style split subtraction is essential for fp16 scratch values:
  # a single n*(2*pi) rounds to a multiple of eight around x=10000.
  reduced = phase
  for coefficient in (4.0, 2.0, 0.25, 0.03125, 2*math.pi-6.28125):
    product_slot, next_reduced = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, product_slot, (rounded, 0), scalar(coefficient), Ops.MUL),
                  _emit_where_stage(total, next_reduced, (reduced, 0), (product_slot, 0), Ops.SUB)))
    reduced = next_reduced

  if is_cos:
    broad, local = alloc(), alloc()
    broad_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(reduced),), arg="rk_cos")
    broad_store = store.replace(src=(temp_index(broad), broad_val))
    broad_plan = plan_rk(sink.substitute({store:broad_store}))
    if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(broad_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(reduced),), arg="rk_cos_local")
    local_store = store.replace(src=(temp_index(local), local_val))
    local_plan = plan_rk(sink.substitute({store:local_store}))
    if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(local_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    neg_broad, abs_broad, local_diff = alloc(), alloc(), alloc()
    local_outside, local_inside = alloc(), alloc()
    near_diff, near_outside, near_inside, middle_mask = alloc(), alloc(), alloc(), alloc()
    selected_sum, broad_mask = alloc(), alloc()
    neg_reduced_cos, abs_reduced_cos = alloc(), alloc()
    center_hi, near_scaled, middle_scaled = alloc(), alloc(), alloc()
    broad_selected, near_selected, middle_selected = alloc(), alloc(), alloc()
    local_sum, normal = alloc(), alloc()
    denom_scratch, valid_denom, factor_scratch, valid_factor, out_scratch = (alloc() for _ in range(5))
    tasks.extend((_emit_where_stage(total, neg_broad, zero, (broad, 0), Ops.SUB),
                  _emit_where_stage(total, abs_broad, (broad, 0), (neg_broad, 0), Ops.MAX),
                  _emit_where_stage(total, local_diff, (abs_broad, 0), scalar(0.5), Ops.SUB),
                  _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                  _emit_where_stage(total, near_diff, (abs_broad, 0), scalar(0.01), Ops.SUB),
                  _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                  # Earlier piecewise-gain tuning used separate 0.096/0.104 masks
                  # around a discontinuous local LUT. The continuous 2*cos table
                  # only needs the local-minus-center mask.
                  _emit_where_stage(total, middle_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                  _emit_where_stage(total, selected_sum, (near_inside, 0), (middle_mask, 0), Ops.ADD),
                  _emit_where_stage(total, broad_mask, one, (selected_sum, 0), Ops.SUB),
                  _emit_where_stage(total, neg_reduced_cos, zero, (reduced, 0), Ops.SUB),
                  _emit_where_stage(total, abs_reduced_cos, (reduced, 0), (neg_reduced_cos, 0), Ops.MAX),
                  _emit_where_stage(total, center_hi, scalar(1.5703125), (abs_reduced_cos, 0), Ops.SUB),
                  _emit_where_stage(total, near_scaled, (center_hi, 0), scalar(math.pi/2-1.5703125), Ops.ADD),
                  _emit_where_stage(total, middle_scaled, (local, 0), scalar(0.5), Ops.MUL),
                  _emit_where_stage(total, broad_selected, (broad, 0), (broad_mask, 0), Ops.MUL),
                  _emit_where_stage(total, near_selected, (near_scaled, 0), (near_inside, 0), Ops.MUL),
                  _emit_where_stage(total, middle_selected, (middle_scaled, 0), (middle_mask, 0), Ops.MUL),
                  _emit_where_stage(total, local_sum, (near_selected, 0), (middle_selected, 0), Ops.ADD),
                  _emit_where_stage(total, normal, (broad_selected, 0), (local_sum, 0), Ops.ADD),
                  _emit_where_stage(total, denom_scratch, one, (invalid, 0), Ops.SUB),
                  _emit_where_stage(total, valid_denom, one, (invalid, 0), Ops.SUB),
                  _emit_where_stage(total, factor_scratch, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                  _emit_where_stage(total, valid_factor, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                  _emit_where_stage(total, out_scratch, (normal, 0), (valid_factor, 0), Ops.MUL),
                  _emit_where_stage(total, info.outs[0], (normal, 0), (valid_factor, 0), Ops.MUL)))
    tasks = list(_fix_cmp_fp32(tuple(tasks), source))
    if source.src[0].dtype is dtypes.float:
      source_slot = source.src[0].arg.slot
      tasks = [RKSubTask(st.cmds, replace(st.task, periodic_input=True), st.relocs)
               if source_slot in st.task.fp32_inputs else st for st in tasks]
    return _finalize_fp32_output(tasks, store)

  broad = alloc()
  broad_val = UOp(Ops.SIN, dtypes.half, (temp_index(reduced),))
  broad_store = store.replace(src=(temp_index(broad), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  zoomed, local = alloc(), alloc()
  tasks.append(_emit_where_stage(total, zoomed, (reduced, 0), scalar(16.0), Ops.MUL))
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg="rk_sin_local")
  local_store = store.replace(src=(temp_index(local), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  neg_reduced, abs_reduced = alloc(), alloc()
  local_diff, local_outside, local_inside = alloc(), alloc(), alloc()
  near_diff, near_outside, near_inside = alloc(), alloc(), alloc()
  broad_mask, local_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, neg_reduced, zero, (reduced, 0), Ops.SUB),
                _emit_where_stage(total, abs_reduced, (reduced, 0), (neg_reduced, 0), Ops.MAX),
                _emit_where_stage(total, local_diff, (abs_reduced, 0), scalar(0.125), Ops.SUB),
                _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_diff, (abs_reduced, 0), scalar(0.04), Ops.SUB),
                _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, one, (local_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB)))

  local_scaled, broad_selected, local_selected = alloc(), alloc(), alloc()
  near_selected, lut_sum, normal = alloc(), alloc(), alloc()
  denom_scratch, valid_denom, factor_scratch, valid_factor, out_scratch = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, local_scaled, (local, 0), scalar(0.125), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, near_selected, (reduced, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, normal, (lut_sum, 0), (near_selected, 0), Ops.ADD),
                _emit_where_stage(total, denom_scratch, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, valid_denom, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, factor_scratch, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, valid_factor, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, out_scratch, (normal, 0), (valid_factor, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (normal, 0), (valid_factor, 0), Ops.MUL)))
  tasks = list(_fix_cmp_fp32(tuple(tasks), source))
  if source.src[0].dtype is dtypes.float:
    source_slot = source.src[0].arg.slot
    tasks = [RKSubTask(st.cmds, replace(st.task, periodic_input=True), st.relocs)
             if source_slot in st.task.fp32_inputs else st for st in tasks]
  return _finalize_fp32_output(tasks, store)

def _try_tan_trig_quotient_wip(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate tan with one pi reduction, a sine LUT, two cosine LUTs, and DPU division."""
  store = _store_node(sink)
  if store is None or (source := _try_tan(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  neg_source, abs_source, invalid_diff, invalid_scratch, invalid = (alloc() for _ in range(5))
  low, neg_low, neg_clamped, bounded = (alloc() for _ in range(4))
  tasks:list[RKSubTask] = [
    _emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
    _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX),
    _emit_where_stage(total, invalid_diff, (abs_source, 0), scalar(20000.0), Ops.SUB),
    _emit_where_stage(total, invalid_scratch, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
    _emit_where_stage(total, invalid, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
    _emit_where_stage(total, low, source_arg, scalar(-10000.0), Ops.MAX),
    _emit_where_stage(total, neg_low, zero, (low, 0), Ops.SUB),
    _emit_where_stage(total, neg_clamped, (neg_low, 0), scalar(-10000.0), Ops.MAX),
    _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB)]

  quotient, neg_quotient, abs_quotient, biased, rounded_abs = (alloc() for _ in range(5))
  positive_diff, positive, negative, positive_n = (alloc() for _ in range(4))
  negative_n, signed_negative_n, rounded = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, quotient, (bounded, 0), scalar(1/math.pi), Ops.MUL),
                _emit_where_stage(total, neg_quotient, zero, (quotient, 0), Ops.SUB),
                _emit_where_stage(total, abs_quotient, (quotient, 0), (neg_quotient, 0), Ops.MAX),
                _emit_where_stage(total, biased, (abs_quotient, 0), scalar(0.5), Ops.ADD),
                _emit_trunc_stage(total, rounded_abs, (biased, 0)),
                _emit_where_stage(total, positive_diff, (quotient, 0), zero, Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative, one, (positive, 0), Ops.SUB),
                _emit_where_stage(total, positive_n, (rounded_abs, 0), (positive, 0), Ops.MUL),
                _emit_where_stage(total, negative_n, (rounded_abs, 0), (negative, 0), Ops.MUL),
                _emit_where_stage(total, signed_negative_n, zero, (negative_n, 0), Ops.SUB),
                _emit_where_stage(total, rounded, (positive_n, 0), (signed_negative_n, 0), Ops.ADD)))
  reduced = bounded
  for coefficient in (3.0, 0.140625, math.pi-3.140625):
    product_slot, next_reduced = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, product_slot, (rounded, 0), scalar(coefficient), Ops.MUL),
                  _emit_where_stage(total, next_reduced, (reduced, 0), (product_slot, 0), Ops.SUB)))
    reduced = next_reduced

  def lut_stage(value:UOp, out:int) -> bool:
    stage_store = store.replace(src=(temp_index(out), value))
    plan = plan_rk(sink.substitute({store:stage_store}))
    if isinstance(plan, str) or plan.kind != "dpu_lut": return False
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return True
  sine, cos_broad, cos_local, tan_local = alloc(), alloc(), alloc(), alloc()
  if not lut_stage(UOp(Ops.SIN, dtypes.half, (temp_index(reduced),)), sine): return None
  # The first tangent path also emitted the broad cosine LUT and selected it
  # below. Tan only uses the quotient for |x|>1, where the amplified local
  # cosine table is unsaturated, so that extra LUT is preserved but disabled.
  # if not lut_stage(UOp(Ops.CUSTOM, dtypes.half, (temp_index(reduced),), arg="rk_cos"), cos_broad): return None
  if not lut_stage(UOp(Ops.CUSTOM, dtypes.half, (temp_index(reduced),), arg="rk_cos_local"), cos_local): return None
  if not lut_stage(UOp(Ops.CUSTOM, dtypes.half, (temp_index(reduced),), arg="rk_tan_local"), tan_local): return None

  neg_cos, abs_cos, local_diff, local_outside, local_inside = (alloc() for _ in range(5))
  near_diff, near_outside, near_inside, middle_mask = (alloc() for _ in range(4))
  neg_reduced, abs_reduced, center_hi, center = (alloc() for _ in range(4))
  local_scaled, broad_selected, near_selected, middle_selected, cosine_partial, cosine = (alloc() for _ in range(6))
  _old_cosine_tasks = (_emit_where_stage(total, neg_cos, zero, (cos_broad, 0), Ops.SUB),
                _emit_where_stage(total, abs_cos, (cos_broad, 0), (neg_cos, 0), Ops.MAX),
                _emit_where_stage(total, local_diff, (abs_cos, 0), scalar(0.5), Ops.SUB),
                _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_diff, (abs_cos, 0), scalar(0.01), Ops.SUB),
                _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, middle_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, neg_reduced, zero, (reduced, 0), Ops.SUB),
                _emit_where_stage(total, abs_reduced, (reduced, 0), (neg_reduced, 0), Ops.MAX),
                _emit_where_stage(total, center_hi, scalar(1.5703125), (abs_reduced, 0), Ops.SUB),
                _emit_where_stage(total, center, (center_hi, 0), scalar(math.pi/2-1.5703125), Ops.ADD),
                _emit_where_stage(total, local_scaled, (cos_local, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, broad_selected, (cos_broad, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, near_selected, (center, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, middle_selected, (local_scaled, 0), (middle_mask, 0), Ops.MUL),
                _emit_where_stage(total, cosine_partial, (broad_selected, 0), (near_selected, 0), Ops.ADD),
                _emit_where_stage(total, cosine, (cosine_partial, 0), (middle_selected, 0), Ops.ADD))

  neg_center, abs_center, center_near_diff = alloc(), alloc(), alloc()
  center_near_outside, center_near_inside, center_local_mask = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, neg_reduced, zero, (reduced, 0), Ops.SUB),
                _emit_where_stage(total, abs_reduced, (reduced, 0), (neg_reduced, 0), Ops.MAX),
                _emit_where_stage(total, center_hi, scalar(1.5703125), (abs_reduced, 0), Ops.SUB),
                _emit_where_stage(total, center, (center_hi, 0), scalar(math.pi/2-1.5703125), Ops.ADD),
                _emit_where_stage(total, neg_center, zero, (center, 0), Ops.SUB),
                _emit_where_stage(total, abs_center, (center, 0), (neg_center, 0), Ops.MAX),
                _emit_where_stage(total, center_near_diff, (abs_center, 0), scalar(0.01), Ops.SUB),
                _emit_where_stage(total, center_near_outside, (center_near_diff, 0), (center_near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, center_near_inside, one, (center_near_outside, 0), Ops.SUB),
                _emit_where_stage(total, center_local_mask, one, (center_near_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_scaled, (cos_local, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, near_selected, (center, 0), (center_near_inside, 0), Ops.MUL),
                _emit_where_stage(total, middle_selected, (local_scaled, 0), (center_local_mask, 0), Ops.MUL),
                _emit_where_stage(total, cosine, (near_selected, 0), (middle_selected, 0), Ops.ADD)))

  tangent = alloc()
  direct_diff, direct_outside, direct_inside = alloc(), alloc(), alloc()
  wide_end_diff, wide_outside, wide_inside = alloc(), alloc(), alloc()
  wide_start_diff, wide_start, wide_mask, quotient_mask = alloc(), alloc(), alloc(), alloc()
  wide_scaled, divided_selected, direct_selected, wide_selected = alloc(), alloc(), alloc(), alloc()
  direct_sum, selected = alloc(), alloc()
  denom_scratch, valid_denom, factor_scratch, valid_factor = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, tangent, (sine, 0), (cosine, 0), Ops.FDIV),
                _emit_where_stage(total, direct_diff, (abs_reduced, 0), scalar(0.5), Ops.SUB),
                _emit_where_stage(total, direct_outside, (direct_diff, 0), (direct_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, direct_inside, one, (direct_outside, 0), Ops.SUB),
                _emit_where_stage(total, wide_end_diff, (abs_reduced, 0), scalar(1.0), Ops.SUB),
                _emit_where_stage(total, wide_outside, (wide_end_diff, 0), (wide_end_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, wide_inside, one, (wide_outside, 0), Ops.SUB),
                _emit_where_stage(total, wide_start_diff, (abs_reduced, 0), scalar(0.55), Ops.SUB),
                _emit_where_stage(total, wide_start, (wide_start_diff, 0), (wide_start_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, wide_mask, (wide_inside, 0), (wide_start, 0), Ops.MUL),
                _emit_where_stage(total, quotient_mask, (direct_outside, 0), (wide_mask, 0), Ops.SUB),
                _emit_where_stage(total, wide_scaled, (tan_local, 0), scalar(2.0), Ops.MUL),
                _emit_where_stage(total, divided_selected, (tangent, 0), (quotient_mask, 0), Ops.MUL),
                _emit_where_stage(total, direct_selected, (tan_local, 0), (direct_inside, 0), Ops.MUL),
                _emit_where_stage(total, wide_selected, (wide_scaled, 0), (wide_mask, 0), Ops.MUL),
                _emit_where_stage(total, direct_sum, (direct_selected, 0), (wide_selected, 0), Ops.ADD),
                _emit_where_stage(total, selected, (divided_selected, 0), (direct_sum, 0), Ops.ADD),
                _emit_where_stage(total, denom_scratch, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, valid_denom, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, factor_scratch, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, valid_factor, (valid_denom, 0), (valid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, info.outs[0], (selected, 0), (valid_factor, 0), Ops.MUL)))
  tasks = list(_fix_cmp_fp32(tuple(tasks), source))
  if source.src[0].dtype is dtypes.float:
    source_slot = source.src[0].arg.slot
    tasks = [RKSubTask(st.cmds, replace(st.task, periodic_input=True), st.relocs)
             if source_slot in st.task.fp32_inputs else st for st in tasks]
  return _finalize_fp32_output(tasks, store)

def _try_tan_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Piecewise tangent using direct local/wide LUTs and a pole-safe trig quotient."""
  store = _store_node(sink)
  if store is None or (source := _try_tan(store.src[1])) is None: return None
  info, total, next_slot = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink)), max(ProgramInfo.from_sink(sink).globals, default=-1)+1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)
  ns, ab, idiff, isc, invalid, low, nl, nc, bounded = (alloc() for _ in range(9))
  tasks = [_emit_where_stage(total, ns, zero, source_arg, Ops.SUB),
           _emit_where_stage(total, ab, source_arg, (ns,0), Ops.MAX),
           _emit_where_stage(total, idiff, (ab,0), scalar(20000), Ops.SUB),
           _emit_where_stage(total, isc, (idiff,0), (idiff,0), Ops.MAX, compare=True),
           _emit_where_stage(total, invalid, (idiff,0), (idiff,0), Ops.MAX, compare=True),
           _emit_where_stage(total, bounded, source_arg, zero, Ops.ADD)]
  if source.src[0].dtype is dtypes.half:
    tasks = [_emit_where_stage(total, ns, zero, source_arg, Ops.SUB),
             _emit_where_stage(total, ab, source_arg, (ns,0), Ops.MAX),
             _emit_where_stage(total, bounded, source_arg, zero, Ops.ADD)]
  q,nq,aq,aq_corrected,bias,nabs,pd,pos,neg,pn,nn,snn,n = (alloc() for _ in range(13))
  tasks += [_emit_where_stage(total,q,(bounded,0),scalar(1/math.pi),Ops.MUL),
            _emit_where_stage(total,nq,zero,(q,0),Ops.SUB), _emit_where_stage(total,aq,(q,0),(nq,0),Ops.MAX),
            # fp16 1/pi can round values just below every odd pi/2 to an exact .5 tie.
            # Bias toward the lower magnitude period; no fp16 input is exactly pi/2.
            _emit_where_stage(total,aq_corrected,(aq,0),scalar(.0005),Ops.SUB),
            _emit_where_stage(total,bias,(aq_corrected,0),scalar(.5),Ops.ADD), _emit_trunc_stage(total,nabs,(bias,0)),
            _emit_where_stage(total,pd,(q,0),zero,Ops.SUB), _emit_where_stage(total,pos,(pd,0),(pd,0),Ops.MAX,compare=True),
            _emit_where_stage(total,neg,one,(pos,0),Ops.SUB), _emit_where_stage(total,pn,(nabs,0),(pos,0),Ops.MUL),
            _emit_where_stage(total,nn,(nabs,0),(neg,0),Ops.MUL), _emit_where_stage(total,snn,zero,(nn,0),Ops.SUB),
            _emit_where_stage(total,n,(pn,0),(snn,0),Ops.ADD)]
  reduced = bounded
  for coefficient in (3.140625, math.pi-3.140625):
    product_slot, next_reduced = alloc(), alloc()
    tasks += [_emit_where_stage(total,product_slot,(n,0),scalar(coefficient),Ops.MUL),
              _emit_where_stage(total,next_reduced,(reduced,0),(product_slot,0),Ops.SUB)]
    reduced = next_reduced
  def add_value(val:UOp, out:int) -> bool:
    plan=plan_rk(sink.substitute({store:store.replace(src=(temp_index(out),val))}))
    if isinstance(plan,str) or plan.kind!="dpu_lut": return False
    cmds,task,relocs=emit_rk(plan)
    tasks.append(RKSubTask(cmds,task,relocs))
    return True
  def add_lut(arg:str, out:int) -> bool: return add_value(UOp(Ops.CUSTOM,dtypes.half,(temp_index(reduced),),arg=arg),out)
  local, wide, sine, cos_local = alloc(), alloc(), alloc(), alloc()
  if not add_lut("rk_tan_local",local) or not add_value(UOp(Ops.SIN,dtypes.half,(temp_index(reduced),)),sine) or \
     not add_lut("rk_tan_wide",wide) or not add_lut("rk_cos_local",cos_local): return None
  nr,ar,diff,outside,inside = (alloc() for _ in range(5))
  tan_near_diff,tan_near_outside,tan_near_inside,tan_local_mask = (alloc() for _ in range(4))
  center_hi,center,ncenter,acenter,cdiff,coutside,cinside,cmask = (alloc() for _ in range(8))
  pole_diff,pole_group,base_extra,pole_base,pole_dist,rem_extra,pole_rem,source_center = (alloc() for _ in range(8))
  qdiff,qoutside,middle_mask = alloc(),alloc(),alloc()
  neg_sine,abs_sine,center_sq,center_term,center_factor,center_cos = (alloc() for _ in range(6))
  cos_scaled,cos_center_sel,cos_local_sel,cosine_partial,safe_cos,cosine,quotient = (alloc() for _ in range(7))
  quotient_adjusted,quotient_center,quotient_middle = alloc(),alloc(),alloc()
  local_scaled,wide_scaled,qs,ws,ls,near_sel,tmp_sel,tmp2,selected = (alloc() for _ in range(9))
  tasks += [_emit_where_stage(total,nr,zero,(reduced,0),Ops.SUB), _emit_where_stage(total,ar,(reduced,0),(nr,0),Ops.MAX),
            _emit_where_stage(total,diff,(ar,0),scalar(.45),Ops.SUB),
            _emit_where_stage(total,outside,(diff,0),(diff,0),Ops.MAX,compare=True),
            _emit_where_stage(total,inside,one,(outside,0),Ops.SUB),
            _emit_where_stage(total,tan_near_diff,(ar,0),scalar(.04),Ops.SUB),
            _emit_where_stage(total,tan_near_outside,(tan_near_diff,0),(tan_near_diff,0),Ops.MAX,compare=True),
            _emit_where_stage(total,tan_near_inside,one,(tan_near_outside,0),Ops.SUB),
            _emit_where_stage(total,tan_local_mask,(inside,0),(tan_near_inside,0),Ops.SUB),
            _emit_where_stage(total,qdiff,(ar,0),scalar(1.05),Ops.SUB),
            _emit_where_stage(total,qoutside,(qdiff,0),(qdiff,0),Ops.MAX,compare=True),
            _emit_where_stage(total,pole_diff,(ab,0),scalar(3.0),Ops.SUB),
            _emit_where_stage(total,pole_group,(pole_diff,0),(pole_diff,0),Ops.MAX,compare=True),
            _emit_where_stage(total,base_extra,(pole_group,0),scalar(3.0),Ops.MUL),
            _emit_where_stage(total,pole_base,scalar(1.5),(base_extra,0),Ops.ADD),
            _emit_where_stage(total,pole_dist,(pole_base,0),(ab,0),Ops.SUB),
            _emit_where_stage(total,rem_extra,(pole_group,0),scalar(.140625),Ops.MUL),
            _emit_where_stage(total,pole_rem,scalar(.0703125),(rem_extra,0),Ops.ADD),
            _emit_where_stage(total,center_hi,(pole_dist,0),(pole_rem,0),Ops.ADD),
            _emit_where_stage(total,local_scaled,(pole_group,0),scalar(math.pi-3.140625),Ops.MUL),
            _emit_where_stage(total,center,scalar(math.pi/2-1.5703125),(local_scaled,0),Ops.ADD),
            _emit_where_stage(total,source_center,(center_hi,0),(center,0),Ops.ADD),
            _emit_where_stage(total,ncenter,zero,(source_center,0),Ops.SUB),
            _emit_where_stage(total,acenter,(source_center,0),(ncenter,0),Ops.MAX),
            _emit_where_stage(total,cdiff,(acenter,0),scalar(.05),Ops.SUB),
            _emit_where_stage(total,coutside,(cdiff,0),(cdiff,0),Ops.MAX,compare=True),
            _emit_where_stage(total,cinside,one,(coutside,0),Ops.SUB),
            _emit_where_stage(total,cmask,(qoutside,0),(cinside,0),Ops.SUB),
            _emit_where_stage(total,cos_scaled,(cos_local,0),scalar(.5),Ops.MUL),
            # In the closest pole band, cot(d) ~= 1/d. Cancelling the sine LUT
            # magnitude avoids amplifying its Q15 quantization in the quotient.
            _emit_where_stage(total,neg_sine,zero,(sine,0),Ops.SUB),
            _emit_where_stage(total,abs_sine,(sine,0),(neg_sine,0),Ops.MAX),
            _emit_where_stage(total,center_sq,(acenter,0),(acenter,0),Ops.MUL),
            _emit_where_stage(total,center_term,(center_sq,0),scalar(1/3),Ops.MUL),
            _emit_where_stage(total,center_factor,one,(center_term,0),Ops.SUB),
            _emit_where_stage(total,center_cos,(acenter,0),(abs_sine,0),Ops.MUL),
            _emit_where_stage(total,cos_center_sel,(center_cos,0),(cinside,0),Ops.MUL),
            _emit_where_stage(total,cos_local_sel,(cos_scaled,0),(cmask,0),Ops.MUL),
            _emit_where_stage(total,cosine_partial,(cos_center_sel,0),(cos_local_sel,0),Ops.ADD),
            _emit_where_stage(total,safe_cos,one,(qoutside,0),Ops.SUB),
            _emit_where_stage(total,cosine,(cosine_partial,0),(safe_cos,0),Ops.ADD),
            _emit_where_stage(total,quotient,(sine,0),(cosine,0),Ops.FDIV),
            _emit_where_stage(total,quotient_adjusted,(quotient,0),(center_factor,0),Ops.MUL),
            _emit_where_stage(total,quotient_center,(quotient_adjusted,0),(cinside,0),Ops.MUL),
            _emit_where_stage(total,quotient_middle,(quotient,0),(cmask,0),Ops.MUL),
            _emit_where_stage(total,qs,(quotient_center,0),(quotient_middle,0),Ops.ADD),
            _emit_where_stage(total,middle_mask,(outside,0),(qoutside,0),Ops.SUB),
            _emit_where_stage(total,wide_scaled,(wide,0),scalar(2.0),Ops.MUL),
            _emit_where_stage(total,ws,(wide_scaled,0),(middle_mask,0),Ops.MUL),
            _emit_where_stage(total,ls,(local,0),(tan_local_mask,0),Ops.MUL),
            _emit_where_stage(total,near_sel,(reduced,0),(tan_near_inside,0),Ops.MUL),
            _emit_where_stage(total,tmp_sel,(qs,0),(ws,0),Ops.ADD),
            _emit_where_stage(total,tmp2,(tmp_sel,0),(ls,0),Ops.ADD),
            _emit_where_stage(total,selected,(tmp2,0),(near_sel,0),Ops.ADD)]
  if source.src[0].dtype is dtypes.half:
    tasks += [_emit_where_stage(total,info.outs[0],(selected,0),one,Ops.MUL)]
  else:
    ds,vd,fs,vf=(alloc() for _ in range(4))
    tasks += [_emit_where_stage(total,ds,one,(invalid,0),Ops.SUB),_emit_where_stage(total,vd,one,(invalid,0),Ops.SUB),
              _emit_where_stage(total,fs,(vd,0),(vd,0),Ops.FDIV),_emit_where_stage(total,vf,(vd,0),(vd,0),Ops.FDIV),
              _emit_where_stage(total,info.outs[0],(selected,0),(vf,0),Ops.MUL)]
  tasks=list(_fix_cmp_fp32(tuple(tasks),source))
  if source.src[0].dtype is dtypes.float:
    slot=source.src[0].arg.slot
    tasks=[RKSubTask(st.cmds,replace(st.task,periodic_input=True),st.relocs) if slot in st.task.fp32_inputs else st for st in tasks]
  return _finalize_fp32_output(tasks,store)

def _try_asin_acos_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Two-LUT asin/acos with separate precision for the smooth center and singular endpoint."""
  store = _store_node(sink)
  if store is None or (match := _try_asin_acos(store.src[1])) is None: return None
  source, is_acos = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  def add_lut(arg:str, source_slot:int, out_slot:int) -> bool:
    value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg=arg)
    plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(out_slot), value))}))
    if isinstance(plan, str) or plan.kind != "dpu_lut": return False
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return True

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)
  source_residual = alloc() if source.dtype is dtypes.float else -1
  if source_residual >= 0:
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, source_residual, 0, 0, 0xFFFFFFFF), RKReloc(0, source_arg[0], 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_RESIDUAL_LAYOUT), source_residual, is_copy=True), relocs))
  neg_source, abs_source, neg_abs, neg_clamped, bounded = (alloc() for _ in range(5))
  negative_diff, negative_scratch, negative, positive = (alloc() for _ in range(4))
  invalid_diff, invalid_scratch, invalid = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX),
                _emit_where_stage(total, neg_abs, zero, (abs_source, 0), Ops.SUB),
                _emit_where_stage(total, neg_clamped, (neg_abs, 0), scalar(-1), Ops.MAX),
                _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative_scratch, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive, one, (negative, 0), Ops.SUB),
                _emit_where_stage(total, invalid_diff, (abs_source, 0), one, Ops.SUB),
                _emit_where_stage(total, invalid_scratch, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, invalid, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True)))

  endpoint_diff, endpoint_scratch, endpoint = (alloc() for _ in range(3))
  endpoint_distance = alloc()
  tasks.extend((_emit_where_stage(total, endpoint_diff, (bounded, 0), scalar(.85 if is_acos else .875), Ops.SUB),
                _emit_where_stage(total, endpoint_scratch, (endpoint_diff, 0), (endpoint_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, endpoint, (endpoint_diff, 0), (endpoint_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, endpoint_distance, one, (bounded, 0), Ops.SUB)))
  endpoint_input = endpoint_distance
  if source_residual >= 0:
    negative_residual, positive_residual, selected_negative_residual = alloc(), alloc(), alloc()
    absolute_residual, scaled_residual, endpoint_input = alloc(), alloc(), alloc()
    tasks.extend((_emit_where_stage(total, negative_residual, zero, (source_residual, 0), Ops.SUB),
                  _emit_where_stage(total, positive_residual, (source_residual, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, selected_negative_residual, (negative_residual, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, absolute_residual, (positive_residual, 0), (selected_negative_residual, 0), Ops.ADD),
                  _emit_where_stage(total, scaled_residual, (absolute_residual, 0), scalar(1/256), Ops.MUL),
                  _emit_where_stage(total, endpoint_input, (endpoint_distance, 0), (scaled_residual, 0), Ops.SUB)))

  broad_slot = alloc()
  if is_acos:
    neg_bounded, positive_bounded, negative_bounded, signed_bounded = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, neg_bounded, zero, (bounded, 0), Ops.SUB),
                  _emit_where_stage(total, positive_bounded, (bounded, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, negative_bounded, (neg_bounded, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, signed_bounded, (positive_bounded, 0), (negative_bounded, 0), Ops.ADD)))
    endpoint_slot = alloc()
    if not add_lut("rk_acos", signed_bounded, broad_slot) or \
       not add_lut("rk_acos_endpoint", endpoint_input, endpoint_slot): return None
    decoded_endpoint_slot = endpoint_slot
    if source_residual >= 0:
      fine_input, fine_slot, decoded_fine_slot = alloc(), alloc(), alloc()
      tasks.append(_emit_where_stage(total, fine_input, (endpoint_input, 0), scalar(64), Ops.MUL))
      if not add_lut("rk_acos_fine_endpoint", fine_input, fine_slot): return None
      tasks.append(_emit_where_stage(total, decoded_fine_slot, (fine_slot, 0), scalar(1/8), Ops.MUL))
      fine_diff, fine_scratch, coarse_mask, fine_mask = (alloc() for _ in range(4))
      coarse_selected, fine_selected, decoded_endpoint_slot = (alloc() for _ in range(3))
      tasks.extend((_emit_where_stage(total, fine_diff, (endpoint_input, 0), scalar(.003), Ops.SUB),
                    _emit_where_stage(total, fine_scratch, (fine_diff, 0), (fine_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, coarse_mask, (fine_diff, 0), (fine_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, fine_mask, one, (coarse_mask, 0), Ops.SUB),
                    _emit_where_stage(total, coarse_selected, (endpoint_slot, 0), (coarse_mask, 0), Ops.MUL),
                    _emit_where_stage(total, fine_selected, (decoded_fine_slot, 0), (fine_mask, 0), Ops.MUL),
                    _emit_where_stage(total, decoded_endpoint_slot, (coarse_selected, 0), (fine_selected, 0), Ops.ADD)))
    broad_mask, broad_positive, broad_negative = (alloc() for _ in range(3))
    broad_positive_selected, broad_negative_selected, broad_decoded, broad_selected = (alloc() for _ in range(4))
    exact_diff, exact_scratch, exact_one, not_exact = (alloc() for _ in range(4))
    endpoint_nonzero, positive_endpoint, negative_endpoint = (alloc() for _ in range(3))
    positive_endpoint_selected, negative_endpoint_selected, endpoint_decoded, endpoint_selected, selected = (alloc() for _ in range(5))
    tasks.extend((_emit_where_stage(total, broad_mask, one, (endpoint, 0), Ops.SUB),
                  _emit_where_stage(total, broad_positive, (broad_slot, 0), scalar(2), Ops.MUL),
                  _emit_where_stage(total, broad_negative, (broad_slot, 0), scalar(4), Ops.MUL),
                  _emit_where_stage(total, broad_positive_selected, (broad_positive, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, broad_negative_selected, (broad_negative, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, broad_decoded, (broad_positive_selected, 0), (broad_negative_selected, 0), Ops.ADD),
                  _emit_where_stage(total, broad_selected, (broad_decoded, 0), (broad_mask, 0), Ops.MUL),
                  # The hardware cannot safely emit a literal zero LUT entry.
                  # Only fp16 x=1 exceeds this threshold, so mask its one-count substitute.
                  _emit_where_stage(total, exact_diff, (endpoint_input, 0) if source_residual >= 0 else (bounded, 0),
                                    zero if source_residual >= 0 else scalar(.99975), Ops.SUB),
                  _emit_where_stage(total, exact_scratch, (exact_diff, 0), (exact_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, exact_one, (exact_diff, 0), (exact_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, not_exact, (exact_one, 0) if source_residual >= 0 else one,
                                    zero if source_residual >= 0 else (exact_one, 0), Ops.SUB),
                  _emit_where_stage(total, endpoint_nonzero, (decoded_endpoint_slot, 0), (not_exact, 0), Ops.MUL),
                  _emit_where_stage(total, positive_endpoint, (endpoint_nonzero, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, negative_endpoint, scalar(math.pi), (endpoint_nonzero, 0), Ops.SUB),
                  _emit_where_stage(total, positive_endpoint_selected, (positive_endpoint, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, negative_endpoint_selected, (negative_endpoint, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, endpoint_decoded, (positive_endpoint_selected, 0), (negative_endpoint_selected, 0), Ops.ADD),
                  _emit_where_stage(total, endpoint_selected, (endpoint_decoded, 0), (endpoint, 0), Ops.MUL),
                  _emit_where_stage(total, selected, (broad_selected, 0), (endpoint_selected, 0), Ops.ADD)))
  else:
    near_diff, near_scratch, near_outside, near_inside = (alloc() for _ in range(4))
    local_diff, local_scratch, local_outside, local_inside = (alloc() for _ in range(4))
    local_mask, middle_mask = alloc(), alloc()
    neg_bounded, local_coordinate, endpoint_coordinate, detail_input = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, near_diff, (bounded, 0), scalar(.04), Ops.SUB),
                  _emit_where_stage(total, near_scratch, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_diff, (bounded, 0),
                                    scalar(.24 if source_residual >= 0 else .125), Ops.SUB),
                  _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                  _emit_where_stage(total, middle_mask, (local_outside, 0), (endpoint, 0), Ops.SUB),
                  _emit_where_stage(total, neg_bounded, zero, (bounded, 0), Ops.SUB),
                  _emit_where_stage(total, local_coordinate, (neg_bounded, 0), (local_inside, 0), Ops.MUL),
                  _emit_where_stage(total, endpoint_coordinate, (endpoint_input, 0), (endpoint, 0), Ops.MUL),
                  _emit_where_stage(total, detail_input, (local_coordinate, 0), (endpoint_coordinate, 0), Ops.ADD)))
    detail_slot = alloc()
    if not add_lut("rk_asin", bounded, broad_slot) or not add_lut("rk_asin_detail", detail_input, detail_slot): return None
    derivative_slot, fp32_endpoint_asin = -1, -1
    if source_residual >= 0:
      derivative_slot = alloc()
      if not add_lut("rk_asin_derivative", bounded, derivative_slot): return None
      coarse_acos, fine_input, fine_acos, decoded_fine_acos = alloc(), alloc(), alloc(), alloc()
      if not add_lut("rk_acos_endpoint", endpoint_input, coarse_acos): return None
      tasks.append(_emit_where_stage(total, fine_input, (endpoint_input, 0), scalar(64), Ops.MUL))
      if not add_lut("rk_acos_fine_endpoint", fine_input, fine_acos): return None
      tasks.append(_emit_where_stage(total, decoded_fine_acos, (fine_acos, 0), scalar(1/8), Ops.MUL))
      fine_diff, fine_scratch, coarse_mask, fine_mask = (alloc() for _ in range(4))
      coarse_selected_acos, fine_selected_acos, decoded_acos, fp32_endpoint_asin = (alloc() for _ in range(4))
      tasks.extend((_emit_where_stage(total, fine_diff, (endpoint_input, 0), scalar(.003), Ops.SUB),
                    _emit_where_stage(total, fine_scratch, (fine_diff, 0), (fine_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, coarse_mask, (fine_diff, 0), (fine_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, fine_mask, one, (coarse_mask, 0), Ops.SUB),
                    _emit_where_stage(total, coarse_selected_acos, (coarse_acos, 0), (coarse_mask, 0), Ops.MUL),
                    _emit_where_stage(total, fine_selected_acos, (decoded_fine_acos, 0), (fine_mask, 0), Ops.MUL),
                    _emit_where_stage(total, decoded_acos, (coarse_selected_acos, 0), (fine_selected_acos, 0), Ops.ADD),
                    _emit_where_stage(total, fp32_endpoint_asin, scalar(math.pi/2), (decoded_acos, 0), Ops.SUB)))
    broad_scaled, broad_selected = alloc(), alloc()
    local_scaled, local_selected = alloc(), alloc()
    endpoint_scaled, endpoint_selected = alloc(), alloc()
    near_selected, partial, absolute_partial, absolute_asin = alloc(), alloc(), alloc(), alloc()
    negative_asin, positive_selected, negative_selected, base_selected = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, broad_scaled, (broad_slot, 0), scalar(2), Ops.MUL),
                  _emit_where_stage(total, broad_selected, (broad_scaled, 0), (middle_mask, 0), Ops.MUL),
                  _emit_where_stage(total, local_scaled, (detail_slot, 0), scalar(.25), Ops.MUL),
                  _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                  _emit_where_stage(total, endpoint_scaled,
                                    (fp32_endpoint_asin, 0) if fp32_endpoint_asin >= 0 else (detail_slot, 0),
                                    one if fp32_endpoint_asin >= 0 else scalar(2), Ops.MUL),
                  _emit_where_stage(total, endpoint_selected, (endpoint_scaled, 0), (endpoint, 0), Ops.MUL),
                  _emit_where_stage(total, near_selected, (bounded, 0), (near_inside, 0), Ops.MUL),
                  _emit_where_stage(total, partial, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                  _emit_where_stage(total, absolute_partial, (partial, 0), (endpoint_selected, 0), Ops.ADD),
                  _emit_where_stage(total, absolute_asin, (absolute_partial, 0), (near_selected, 0), Ops.ADD),
                  _emit_where_stage(total, negative_asin, zero, (absolute_asin, 0), Ops.SUB),
                  _emit_where_stage(total, positive_selected, (absolute_asin, 0), (positive, 0), Ops.MUL),
                  _emit_where_stage(total, negative_selected, (negative_asin, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, base_selected, (positive_selected, 0), (negative_selected, 0), Ops.ADD)))
    selected = base_selected
    if source_residual >= 0:
      derivative, scaled_residual, residual_correction, nonendpoint_mask, nonendpoint_correction, selected = (alloc() for _ in range(6))
      tasks.extend((_emit_where_stage(total, derivative, (derivative_slot, 0), scalar(2), Ops.MUL),
                    _emit_where_stage(total, scaled_residual, (source_residual, 0), scalar(1/256), Ops.MUL),
                    _emit_where_stage(total, residual_correction, (derivative, 0), (scaled_residual, 0), Ops.MUL),
                    _emit_where_stage(total, nonendpoint_mask, one, (endpoint, 0), Ops.SUB),
                    _emit_where_stage(total, nonendpoint_correction, (residual_correction, 0), (nonendpoint_mask, 0), Ops.MUL),
                    _emit_where_stage(total, selected, (base_selected, 0), (nonendpoint_correction, 0), Ops.ADD)))

  valid_scratch, valid, factor_scratch, factor = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, valid_scratch, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, valid, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, factor_scratch, (valid, 0), (valid, 0), Ops.FDIV),
                _emit_where_stage(total, factor, (valid, 0), (valid, 0), Ops.FDIV),
                _emit_where_stage(total, info.outs[0], (selected, 0), (factor, 0), Ops.MUL)))
  tasks = list(_fix_cmp_fp32(tuple(tasks), source))
  return _finalize_fp32_output(tasks, store)

def _try_atan_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Two-LUT atan after reciprocal folding maps every finite magnitude into [0,1]."""
  store = _store_node(sink)
  if store is None or (source := _try_atan(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  def add_lut(arg:str, source_slot:int, out_slot:int) -> bool:
    value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg=arg)
    plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(out_slot), value))}))
    if isinstance(plan, str) or plan.kind != "dpu_lut": return False
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return True

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)
  neg_source, abs_source = alloc(), alloc()
  negative_diff, negative_scratch, negative, nonnegative = (alloc() for _ in range(4))
  large_diff, large_scratch, large, small = (alloc() for _ in range(4))
  safe_denominator, inverse, small_input, large_input, transformed = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative_scratch, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, nonnegative, one, (negative, 0), Ops.SUB),
                _emit_where_stage(total, large_diff, (abs_source, 0), one, Ops.SUB),
                _emit_where_stage(total, large_scratch, (large_diff, 0), (large_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, large, (large_diff, 0), (large_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, small, one, (large, 0), Ops.SUB),
                # Adding one only to the unused |x|<=1 denominator avoids 1/0
                # and therefore avoids contaminating the selected identity lane.
                _emit_where_stage(total, safe_denominator, (abs_source, 0), (small, 0), Ops.ADD),
                _emit_where_stage(total, inverse, one, (safe_denominator, 0), Ops.FDIV),
                _emit_where_stage(total, small_input, (abs_source, 0), (small, 0), Ops.MUL),
                _emit_where_stage(total, large_input, (inverse, 0), (large, 0), Ops.MUL),
                _emit_where_stage(total, transformed, (small_input, 0), (large_input, 0), Ops.ADD)))

  near_diff, near_scratch, near_outside, near_inside = (alloc() for _ in range(4))
  local_diff, local_scratch, local_outside, local_inside, local_mask = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, near_diff, (transformed, 0), scalar(.04), Ops.SUB),
                _emit_where_stage(total, near_scratch, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_diff, (transformed, 0), scalar(.125), Ops.SUB),
                _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB)))

  negative_transformed, amplified_negative, local_small, local_coordinate = (alloc() for _ in range(4))
  large_coordinate, detail_input = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_transformed, zero, (transformed, 0), Ops.SUB),
                _emit_where_stage(total, amplified_negative, (negative_transformed, 0), scalar(4), Ops.MUL),
                _emit_where_stage(total, local_small, (local_inside, 0), (small, 0), Ops.MUL),
                _emit_where_stage(total, local_coordinate, (amplified_negative, 0), (local_small, 0), Ops.MUL),
                _emit_where_stage(total, large_coordinate, (transformed, 0), (large, 0), Ops.MUL),
                _emit_where_stage(total, detail_input, (local_coordinate, 0), (large_coordinate, 0), Ops.ADD)))
  broad_slot, local_slot = alloc(), alloc()
  if not add_lut("rk_atan", transformed, broad_slot) or not add_lut("rk_atan_local", detail_input, local_slot): return None
  broad_region, local_region, near_region = alloc(), alloc(), alloc()
  broad_selected, local_scaled, local_selected, near_selected = (alloc() for _ in range(4))
  large_scaled, large_selected, partial, base_partial, magnitude = (alloc() for _ in range(5))
  negative_magnitude, positive_selected, negative_selected = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, broad_region, (local_outside, 0), (small, 0), Ops.MUL),
                _emit_where_stage(total, local_region, (local_mask, 0), (small, 0), Ops.MUL),
                _emit_where_stage(total, near_region, (near_inside, 0), (small, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_slot, 0), (broad_region, 0), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(.25), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_region, 0), Ops.MUL),
                _emit_where_stage(total, near_selected, (transformed, 0), (near_region, 0), Ops.MUL),
                _emit_where_stage(total, large_scaled, (local_slot, 0), scalar(2), Ops.MUL),
                _emit_where_stage(total, large_selected, (large_scaled, 0), (large, 0), Ops.MUL),
                _emit_where_stage(total, partial, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, base_partial, (partial, 0), (near_selected, 0), Ops.ADD),
                _emit_where_stage(total, magnitude, (base_partial, 0), (large_selected, 0), Ops.ADD),
                _emit_where_stage(total, negative_magnitude, zero, (magnitude, 0), Ops.SUB),
                _emit_where_stage(total, positive_selected, (magnitude, 0), (nonnegative, 0), Ops.MUL),
                _emit_where_stage(total, negative_selected, (negative_magnitude, 0), (negative, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (positive_selected, 0), (negative_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_fp32_atanh_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Keep exact normal-fp32 atanh out of the half-buffer two-LUT implementation."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float or (source := _try_atanh(store.src[1])) is None: return None
  if source.dtype is not dtypes.float or source.src[0].op is not Ops.PARAM: return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_atanh_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Two-LUT atanh with explicit ±1 infinity and out-of-domain NaN handling."""
  store = _store_node(sink)
  if store is None or (source := _try_atanh(store.src[1])) is None: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  def add_lut(arg:str, source_slot:int, out_slot:int) -> bool:
    value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg=arg)
    plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(out_slot), value))}))
    if isinstance(plan, str) or plan.kind != "dpu_lut": return False
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return True

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)
  neg_source, abs_source, neg_abs, neg_clamped, bounded = (alloc() for _ in range(5))
  negative_diff, negative_scratch, negative, nonnegative = (alloc() for _ in range(4))
  invalid_diff, invalid_scratch, invalid = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX),
                _emit_where_stage(total, neg_abs, zero, (abs_source, 0), Ops.SUB),
                _emit_where_stage(total, neg_clamped, (neg_abs, 0), scalar(-1), Ops.MAX),
                _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative_scratch, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, nonnegative, one, (negative, 0), Ops.SUB),
                _emit_where_stage(total, invalid_diff, (abs_source, 0), one, Ops.SUB),
                _emit_where_stage(total, invalid_scratch, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, invalid, (invalid_diff, 0), (invalid_diff, 0), Ops.MAX, compare=True)))

  endpoint_diff, endpoint_scratch, endpoint = (alloc() for _ in range(3))
  endpoint_distance = alloc()
  exact_diff, exact_scratch, exact = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, endpoint_diff, (bounded, 0), scalar(.875), Ops.SUB),
                _emit_where_stage(total, endpoint_scratch, (endpoint_diff, 0), (endpoint_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, endpoint, (endpoint_diff, 0), (endpoint_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, endpoint_distance, one, (bounded, 0), Ops.SUB),
                _emit_where_stage(total, exact_diff, (bounded, 0), scalar(.99975), Ops.SUB),
                _emit_where_stage(total, exact_scratch, (exact_diff, 0), (exact_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, exact, (exact_diff, 0), (exact_diff, 0), Ops.MAX, compare=True)))

  near_diff, near_scratch, near_outside, near_inside = (alloc() for _ in range(4))
  local_diff, local_scratch, local_outside, local_inside, local_mask = (alloc() for _ in range(5))
  broad_mask = alloc()
  neg_bounded, local_coordinate, endpoint_coordinate, detail_input = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, near_diff, (bounded, 0), scalar(.04), Ops.SUB),
                _emit_where_stage(total, near_scratch, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_diff, (bounded, 0), scalar(.125), Ops.SUB),
                _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (local_outside, 0), (endpoint, 0), Ops.SUB),
                _emit_where_stage(total, neg_bounded, zero, (bounded, 0), Ops.SUB),
                _emit_where_stage(total, local_coordinate, (neg_bounded, 0), (local_inside, 0), Ops.MUL),
                _emit_where_stage(total, endpoint_coordinate, (endpoint_distance, 0), (endpoint, 0), Ops.MUL),
                _emit_where_stage(total, detail_input, (local_coordinate, 0), (endpoint_coordinate, 0), Ops.ADD)))

  broad_slot, detail_slot = alloc(), alloc()
  if not add_lut("rk_atanh", bounded, broad_slot) or not add_lut("rk_atanh_detail", detail_input, detail_slot): return None
  broad_scaled, broad_selected, local_scaled, local_selected = (alloc() for _ in range(4))
  endpoint_scaled, endpoint_selected, near_selected = (alloc() for _ in range(3))
  partial, finite_partial, magnitude = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, broad_scaled, (broad_slot, 0), scalar(4), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_scaled, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_scaled, (detail_slot, 0), scalar(.25), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, endpoint_scaled, (detail_slot, 0), scalar(8), Ops.MUL),
                _emit_where_stage(total, endpoint_selected, (endpoint_scaled, 0), (endpoint, 0), Ops.MUL),
                _emit_where_stage(total, near_selected, (bounded, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, partial, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, finite_partial, (partial, 0), (endpoint_selected, 0), Ops.ADD),
                _emit_where_stage(total, magnitude, (finite_partial, 0), (near_selected, 0), Ops.ADD)))

  negative_magnitude, positive_selected, negative_selected, signed_finite = (alloc() for _ in range(4))
  finite_denom_scratch, finite_denom, infinite_scratch, signed = (alloc() for _ in range(4))
  valid_scratch, valid, factor_scratch, factor = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, negative_magnitude, zero, (magnitude, 0), Ops.SUB),
                _emit_where_stage(total, positive_selected, (magnitude, 0), (nonnegative, 0), Ops.MUL),
                _emit_where_stage(total, negative_selected, (negative_magnitude, 0), (negative, 0), Ops.MUL),
                _emit_where_stage(total, signed_finite, (positive_selected, 0), (negative_selected, 0), Ops.ADD),
                _emit_where_stage(total, finite_denom_scratch, one, (exact, 0), Ops.SUB),
                _emit_where_stage(total, finite_denom, one, (exact, 0), Ops.SUB),
                _emit_where_stage(total, infinite_scratch, (signed_finite, 0), (finite_denom, 0), Ops.FDIV),
                _emit_where_stage(total, signed, (signed_finite, 0), (finite_denom, 0), Ops.FDIV),
                _emit_where_stage(total, valid_scratch, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, valid, one, (invalid, 0), Ops.SUB),
                _emit_where_stage(total, factor_scratch, (valid, 0), (valid, 0), Ops.FDIV),
                _emit_where_stage(total, factor, (valid, 0), (valid, 0), Ops.FDIV),
                _emit_where_stage(total, info.outs[0], (signed, 0), (factor, 0), Ops.MUL)))
  return tuple(tasks)

def _try_asinh_acosh_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Two physical-table tasks cover core/detail, middle, and finite large ranges."""
  store = _store_node(sink)
  if store is None or (match := _try_asinh_acosh(store.src[1])) is None: return None
  source, is_acosh = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  def add_lut(arg:str, source_slot:int, out_slot:int) -> bool:
    value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg=arg)
    plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(out_slot), value))}))
    if isinstance(plan, str) or plan.kind != "dpu_lut": return False
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    return True

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)
  # Rejected WIP: a 1.5*fp32 residual nudge for small ASINH inputs only
  # moved which samples landed in the wrong fp16 output bin. Keep the path
  # disabled for reference; ACOSH still needs the residual for its endpoint.
  apply_small_residual = False
  source_residual = alloc() if source.dtype is dtypes.float and (is_acosh or apply_small_residual) else -1
  if source_residual >= 0:
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, source_residual, 0, 0, 0xFFFFFFFF), RKReloc(0, source_arg[0], 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_RESIDUAL_LAYOUT), source_residual, is_copy=True), relocs))
  invalid = -1
  if is_acosh:
    below_diff, below_scratch, invalid, magnitude = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, below_diff, one, source_arg, Ops.SUB),
                  _emit_where_stage(total, below_scratch, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, invalid, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, magnitude, source_arg, one, Ops.MAX)))
    negative, nonnegative = -1, -1
  else:
    neg_source, magnitude = alloc(), alloc()
    negative_diff, negative_scratch, negative, nonnegative = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
                  _emit_where_stage(total, magnitude, source_arg, (neg_source, 0), Ops.MAX),
                  _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                  _emit_where_stage(total, negative_scratch, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, nonnegative, one, (negative, 0), Ops.SUB)))

  small_diff, small_scratch, small_outside, small = (alloc() for _ in range(4))
  huge_diff, huge_scratch, huge, mid_region = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, small_diff, (magnitude, 0), scalar(2), Ops.SUB),
                _emit_where_stage(total, small_scratch, (small_diff, 0), (small_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, small_outside, (small_diff, 0), (small_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, small, one, (small_outside, 0), Ops.SUB),
                _emit_where_stage(total, huge_diff, (magnitude, 0), scalar(16), Ops.SUB),
                _emit_where_stage(total, huge_scratch, (huge_diff, 0), (huge_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, huge, (huge_diff, 0), (huge_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, mid_region, (small_outside, 0), (huge, 0), Ops.SUB)))

  local_inside, local_mask, broad_region = -1, -1, -1
  near_inside = -1
  if is_acosh:
    distance, local_diff, local_scratch, local_outside = (alloc() for _ in range(4))
    local_inside, broad_region = alloc(), alloc()
    nonexact_diff, nonexact_scratch, nonexact = (alloc() for _ in range(3))
    tasks.append(_emit_where_stage(total, distance, source_arg if source_residual >= 0 else (magnitude, 0), one, Ops.SUB))
    coordinate_source = distance
    if source_residual >= 0:
      scaled_residual, coordinate_source = alloc(), alloc()
      refined_below_diff, refined_below_scratch, invalid = alloc(), alloc(), alloc()
      tasks.extend((_emit_where_stage(total, scaled_residual, (source_residual, 0), scalar(1/256), Ops.MUL),
                    _emit_where_stage(total, coordinate_source, (distance, 0), (scaled_residual, 0), Ops.ADD),
                    _emit_where_stage(total, refined_below_diff, zero, (coordinate_source, 0), Ops.SUB),
                    _emit_where_stage(total, refined_below_scratch, (refined_below_diff, 0), (refined_below_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, invalid, (refined_below_diff, 0), (refined_below_diff, 0), Ops.MAX, compare=True)))
    tasks.extend((_emit_where_stage(total, local_diff, (coordinate_source, 0), scalar(.04), Ops.SUB),
                  _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                  _emit_where_stage(total, broad_region, (small, 0), (local_inside, 0), Ops.SUB),
                  _emit_where_stage(total, nonexact_diff, (coordinate_source, 0), scalar(.0005), Ops.SUB),
                  _emit_where_stage(total, nonexact_scratch, (nonexact_diff, 0), (nonexact_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, nonexact, (nonexact_diff, 0), (nonexact_diff, 0), Ops.MAX, compare=True)))
  else:
    near_diff, near_scratch, near_outside, near_inside = (alloc() for _ in range(4))
    local_diff, local_scratch, local_outside, local_inside = (alloc() for _ in range(4))
    local_mask, broad_region = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, near_diff, (magnitude, 0), scalar(.04), Ops.SUB),
                  _emit_where_stage(total, near_scratch, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_diff, (magnitude, 0), scalar(.25), Ops.SUB),
                  _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                  _emit_where_stage(total, broad_region, (small, 0), (local_inside, 0), Ops.SUB)))
    coordinate_source, nonexact = magnitude, -1

  neg_coordinate, amplified_coordinate, local_coordinate = (alloc() for _ in range(3))
  broad_raw, broad_coordinate, core_input = alloc(), alloc(), alloc()
  local_gain, broad_gain = (48 if is_acosh else 8), (2 if is_acosh else 1)
  tasks.extend((_emit_where_stage(total, neg_coordinate, zero, (coordinate_source, 0), Ops.SUB),
                _emit_where_stage(total, amplified_coordinate, (neg_coordinate, 0), scalar(local_gain), Ops.MUL),
                _emit_where_stage(total, local_coordinate, (amplified_coordinate, 0), (local_inside, 0), Ops.MUL),
                _emit_where_stage(total, broad_raw, (coordinate_source, 0), scalar(broad_gain), Ops.MUL),
                _emit_where_stage(total, broad_coordinate, (broad_raw, 0), (broad_region, 0), Ops.MUL),
                _emit_where_stage(total, core_input, (local_coordinate, 0), (broad_coordinate, 0), Ops.ADD)))

  middle_delta, negative_middle, middle_coordinate = (alloc() for _ in range(3))
  huge_scaled, huge_coordinate, range_input = (alloc() for _ in range(3))
  tasks.extend((_emit_where_stage(total, middle_delta, (magnitude, 0), scalar(2), Ops.SUB),
                _emit_where_stage(total, negative_middle, zero, (middle_delta, 0), Ops.SUB),
                _emit_where_stage(total, middle_coordinate, (negative_middle, 0), (mid_region, 0), Ops.MUL),
                _emit_where_stage(total, huge_scaled, (magnitude, 0), scalar(1/19), Ops.MUL),
                _emit_where_stage(total, huge_coordinate, (huge_scaled, 0), (huge, 0), Ops.MUL),
                _emit_where_stage(total, range_input, (middle_coordinate, 0), (huge_coordinate, 0), Ops.ADD)))

  core_slot, range_slot = alloc(), alloc()
  prefix = "acosh" if is_acosh else "asinh"
  if not add_lut(f"rk_{prefix}_core", core_input, core_slot) or not add_lut(f"rk_{prefix}_range", range_input, range_slot): return None
  local_scaled, local_selected, broad_scaled, broad_selected = (alloc() for _ in range(4))
  middle_scaled, middle_selected, huge_scaled_output, huge_selected = (alloc() for _ in range(4))
  near_selected = alloc() if not is_acosh else -1
  first, second, magnitude_result = alloc(), alloc(), alloc()
  local_scale = .5 if is_acosh else .25
  tasks.extend((_emit_where_stage(total, local_scaled, (core_slot, 0), scalar(local_scale), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0),
                                  (local_inside if is_acosh else local_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_scaled, (core_slot, 0), scalar(2), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_scaled, 0), (broad_region, 0), Ops.MUL),
                _emit_where_stage(total, middle_scaled, (range_slot, 0), scalar(4), Ops.MUL),
                _emit_where_stage(total, middle_selected, (middle_scaled, 0), (mid_region, 0), Ops.MUL),
                _emit_where_stage(total, huge_scaled_output, (range_slot, 0), scalar(8), Ops.MUL),
                _emit_where_stage(total, huge_selected, (huge_scaled_output, 0), (huge, 0), Ops.MUL)))
  if not is_acosh:
    tasks.append(_emit_where_stage(total, near_selected, (magnitude, 0), (near_inside, 0), Ops.MUL))
  tasks.extend((_emit_where_stage(total, first, (local_selected, 0), (broad_selected, 0), Ops.ADD),
                _emit_where_stage(total, second, (middle_selected, 0), (huge_selected, 0), Ops.ADD),
                _emit_where_stage(total, magnitude_result, (first, 0), (second, 0), Ops.ADD)))
  if not is_acosh:
    with_near = alloc()
    tasks.append(_emit_where_stage(total, with_near, (magnitude_result, 0), (near_selected, 0), Ops.ADD))
    negative_result, positive_selected, negative_selected = (alloc() for _ in range(3))
    signed_result = info.outs[0] if source_residual < 0 or not apply_small_residual else alloc()
    tasks.extend((_emit_where_stage(total, negative_result, zero, (with_near, 0), Ops.SUB),
                  _emit_where_stage(total, positive_selected, (with_near, 0), (nonnegative, 0), Ops.MUL),
                  _emit_where_stage(total, negative_selected, (negative_result, 0), (negative, 0), Ops.MUL),
                  _emit_where_stage(total, signed_result, (positive_selected, 0), (negative_selected, 0), Ops.ADD)))
    if apply_small_residual and source_residual >= 0:
      correction_diff, correction_scratch, correction_outside, correction_inside = (alloc() for _ in range(4))
      scaled_residual, selected_correction = alloc(), alloc()
      tasks.extend((_emit_where_stage(total, correction_diff, (magnitude, 0), scalar(.25), Ops.SUB),
                    _emit_where_stage(total, correction_scratch, (correction_diff, 0), (correction_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, correction_outside, (correction_diff, 0), (correction_diff, 0), Ops.MAX, compare=True),
                    _emit_where_stage(total, correction_inside, one, (correction_outside, 0), Ops.SUB),
                    _emit_where_stage(total, scaled_residual, (source_residual, 0), scalar(1.5/256), Ops.MUL),
                    _emit_where_stage(total, selected_correction, (scaled_residual, 0), (correction_inside, 0), Ops.MUL),
                    _emit_where_stage(total, info.outs[0], (signed_result, 0), (selected_correction, 0), Ops.ADD)))
  else:
    exact_zeroed = alloc()
    tasks.append(_emit_where_stage(total, exact_zeroed, (magnitude_result, 0), (nonexact, 0), Ops.MUL))
    valid_scratch, valid, factor_scratch, factor = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, valid_scratch, one, (invalid, 0), Ops.SUB),
                  _emit_where_stage(total, valid, one, (invalid, 0), Ops.SUB),
                  _emit_where_stage(total, factor_scratch, (valid, 0), (valid, 0), Ops.FDIV),
                  _emit_where_stage(total, factor, (valid, 0), (valid, 0), Ops.FDIV),
                  _emit_where_stage(total, info.outs[0], (exact_zeroed, 0), (factor, 0), Ops.MUL)))
  tasks = list(_fix_cmp_fp32(tuple(tasks), source))
  return _finalize_fp32_output(tasks, store)

def _try_fp32_sinh_cosh_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Keep exact normal-fp32 sinh/cosh off the reset-heavy generic DPU splitter."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float or (match := _try_sinh_cosh(store.src[1])) is None: return None
  source, _ = match
  if source.dtype is not dtypes.float or source.src[0].op is not Ops.PARAM: return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_sinh_cosh_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate sinh/cosh directly on [-2,2] and restore fp16 overflow outside the finite range."""
  store = _store_node(sink)
  if store is None or (match := _try_sinh_cosh(store.src[1])) is None: return None
  source, is_cosh = match
  if source.dtype is not dtypes.half: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(x:float) -> tuple[int,int]: return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', x))[0]
  def temp_index(slot:int) -> UOp:
    idx = store.src[0]
    return idx.replace(dtype=dtypes.half, src=(idx.src[0].param_like(slot).replace(dtype=dtypes.half), *idx.src[1:]))
  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1)

  low, neg_low, neg_clamped, bounded = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, low, source_arg, scalar(-2), Ops.MAX),
                _emit_where_stage(total, neg_low, zero, (low, 0), Ops.SUB),
                _emit_where_stage(total, neg_clamped, (neg_low, 0), scalar(-2), Ops.MAX),
                _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB)))
  lut_slot = alloc()
  lut_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(bounded),), arg="rk_cosh" if is_cosh else "rk_sinh")
  lut_plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(lut_slot), lut_val))}))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_slot = -1
  if not is_cosh:
    local_slot = alloc()
    local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(bounded),), arg="rk_sinh_local")
    local_plan = plan_rk(sink.substitute({store:store.replace(src=(temp_index(local_slot), local_val))}))
    if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(local_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  neg_source, abs_source = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, neg_source, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, abs_source, source_arg, (neg_source, 0), Ops.MAX)))
  selected = lut_slot
  if not is_cosh:
    near_diff, near_scratch, near_outside, near_inside = (alloc() for _ in range(4))
    local_diff, local_scratch, local_outside, local_inside, local_mask = (alloc() for _ in range(5))
    broad, local_scaled, local, near, partial, selected = (alloc() for _ in range(6))
    tasks.extend((_emit_where_stage(total, near_diff, (abs_source, 0), scalar(.04), Ops.SUB),
                  _emit_where_stage(total, near_scratch, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_outside, (near_diff, 0), (near_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_diff, (abs_source, 0), scalar(.125), Ops.SUB),
                  _emit_where_stage(total, local_scratch, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_outside, (local_diff, 0), (local_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                  _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                  _emit_where_stage(total, broad, (lut_slot, 0), (local_outside, 0), Ops.MUL),
                  _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(.25), Ops.MUL),
                  _emit_where_stage(total, local, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                  _emit_where_stage(total, near, source_arg, (near_inside, 0), Ops.MUL),
                  _emit_where_stage(total, partial, (broad, 0), (local, 0), Ops.ADD),
                  _emit_where_stage(total, selected, (partial, 0), (near, 0), Ops.ADD)))

  large_diff, large_scratch, large, denom_scratch, denom = (alloc() for _ in range(5))
  result_scratch = alloc()
  tasks.extend((_emit_where_stage(total, large_diff, (abs_source, 0), scalar(10), Ops.SUB),
                _emit_where_stage(total, large_scratch, (large_diff, 0), (large_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, large, (large_diff, 0), (large_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, denom_scratch, one, (large, 0), Ops.SUB),
                _emit_where_stage(total, denom, one, (large, 0), Ops.SUB),
                _emit_where_stage(total, result_scratch, (selected, 0), (denom, 0), Ops.FDIV),
                _emit_where_stage(total, info.outs[0], (selected, 0), (denom, 0), Ops.FDIV)))
  return tuple(tasks)

def _try_gelu_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate either GELU form with broad/local LUTs and a near-zero series."""
  store = _store_node(sink)
  if store is None or (match := _try_gelu(store.src[1])) is None: return None
  source, approximate_tanh = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def clamp_symmetric(limit:float) -> tuple[int,list[RKSubTask]]:
    low, neg, neg_clamped, bounded = (alloc() for _ in range(4))
    return bounded, [_emit_where_stage(total, low, source_arg, scalar(-limit), Ops.MAX),
                     _emit_where_stage(total, neg, zero, (low, 0), Ops.SUB),
                     _emit_where_stage(total, neg_clamped, (neg, 0), scalar(-limit), Ops.MAX),
                     _emit_where_stage(total, bounded, zero, (neg_clamped, 0), Ops.SUB)]

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  broad_input, tasks = clamp_symmetric(4.0)
  broad = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(broad_input),), arg=("rk_gelu", approximate_tanh))
  broad_store = store.replace(src=(temp_index(broad), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  local_input, local_stages = clamp_symmetric(0.5)
  tasks.extend(local_stages)
  zoomed = alloc()
  tasks.append(_emit_where_stage(total, zoomed, (local_input, 0), scalar(8.0), Ops.MUL))
  local = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg=("rk_gelu_local", approximate_tanh))
  local_store = store.replace(src=(temp_index(local), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  range_low_diff, range_low, range_high_diff, range_high = (alloc() for _ in range(4))
  range_outside, range_inside = alloc(), alloc()
  local_low_diff, local_low, local_high_diff, local_high = (alloc() for _ in range(4))
  local_outside, local_inside, poly_low_diff, poly_low = (alloc() for _ in range(4))
  poly_high_diff, poly_high, poly_outside, poly_inside = (alloc() for _ in range(4))
  local_mask, broad_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, range_low_diff, scalar(-4.0), source_arg, Ops.SUB),
                _emit_where_stage(total, range_low, (range_low_diff, 0), (range_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, range_high_diff, source_arg, scalar(4.0), Ops.SUB),
                _emit_where_stage(total, range_high, (range_high_diff, 0), (range_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, range_outside, (range_low, 0), (range_high, 0), Ops.MAX),
                _emit_where_stage(total, range_inside, one, (range_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_low_diff, scalar(-0.5), source_arg, Ops.SUB),
                _emit_where_stage(total, local_low, (local_low_diff, 0), (local_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_high_diff, source_arg, scalar(0.5), Ops.SUB),
                _emit_where_stage(total, local_high, (local_high_diff, 0), (local_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, poly_low_diff, scalar(-0.04), source_arg, Ops.SUB),
                _emit_where_stage(total, poly_low, (poly_low_diff, 0), (poly_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_high_diff, source_arg, scalar(0.04), Ops.SUB),
                _emit_where_stage(total, poly_high, (poly_high_diff, 0), (poly_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_outside, (poly_low, 0), (poly_high, 0), Ops.MAX),
                _emit_where_stage(total, poly_inside, one, (poly_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (poly_inside, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (range_inside, 0), (local_inside, 0), Ops.SUB)))

  poly_input, poly_stages = clamp_symmetric(0.04)
  tasks.extend(poly_stages)
  linear, square, quadratic, polynomial = (alloc() for _ in range(4))
  positive_diff, positive, positive_extra, broad_scale, broad_scaled = (alloc() for _ in range(5))
  broad_selected, local_scaled, local_selected, poly_selected = (alloc() for _ in range(4))
  lut_sum, interior, fallback, fallback_selected = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, linear, (poly_input, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, square, (poly_input, 0), (poly_input, 0), Ops.MUL),
                _emit_where_stage(total, quadratic, (square, 0), scalar(1/math.sqrt(2*math.pi)), Ops.MUL),
                _emit_where_stage(total, polynomial, (linear, 0), (quadratic, 0), Ops.ADD),
                _emit_where_stage(total, positive_diff, source_arg, zero, Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_extra, (positive, 0), scalar(3.0), Ops.MUL),
                _emit_where_stage(total, broad_scale, one, (positive_extra, 0), Ops.ADD),
                _emit_where_stage(total, broad_scaled, (broad, 0), (broad_scale, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_scaled, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, poly_selected, (polynomial, 0), (poly_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, interior, (lut_sum, 0), (poly_selected, 0), Ops.ADD),
                _emit_where_stage(total, fallback, source_arg, zero, Ops.MAX),
                _emit_where_stage(total, fallback_selected, (fallback, 0), (range_outside, 0), Ops.MUL),
                _emit_where_stage(total, info.outs[0], (interior, 0), (fallback_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_elu_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate ELU/SELU with broad/local negative LUTs and exact asymptotes."""
  store = _store_node(sink)
  if store is None or (match := _try_elu(store.src[1])) is None: return None
  source, negative_scale, positive_scale = match
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  next_slot = max(info.globals, default=-1) + 1
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret
  def temp_index(slot:int) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  source_arg, zero, one = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0), scalar(1.0)
  broad_low, broad_neg, broad_input = alloc(), alloc(), alloc()
  tasks = [_emit_where_stage(total, broad_low, source_arg, scalar(-8.0), Ops.MAX),
           _emit_where_stage(total, broad_neg, zero, (broad_low, 0), Ops.SUB),
           _emit_where_stage(total, broad_input, zero, (broad_neg, 0), Ops.SUB)]
  broad_slot = alloc()
  broad_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(broad_input),), arg=("rk_elu", negative_scale))
  broad_store = store.replace(src=(temp_index(broad_slot), broad_val))
  broad_plan = plan_rk(sink.substitute({store:broad_store}))
  if isinstance(broad_plan, str) or broad_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(broad_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  local_low, local_neg, local_input, zoomed = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, local_low, source_arg, scalar(-0.5), Ops.MAX),
                _emit_where_stage(total, local_neg, zero, (local_low, 0), Ops.SUB),
                _emit_where_stage(total, local_input, zero, (local_neg, 0), Ops.SUB),
                _emit_where_stage(total, zoomed, (local_input, 0), scalar(4.0), Ops.MUL)))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg=("rk_elu_local", negative_scale))
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  below_diff, below, local_below_diff, local_below = (alloc() for _ in range(4))
  poly_below_diff, poly_below, negative_diff, negative = (alloc() for _ in range(4))
  broad_mask_scratch, broad_mask, local_mask_scratch, local_mask = (alloc() for _ in range(4))
  poly_mask_scratch, poly_mask = alloc(), alloc()
  positive_scratch, positive = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, below_diff, scalar(-8.0), source_arg, Ops.SUB),
                _emit_where_stage(total, below, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_below_diff, scalar(-0.5), source_arg, Ops.SUB),
                _emit_where_stage(total, local_below, (local_below_diff, 0), (local_below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, poly_below_diff, scalar(-0.03), source_arg, Ops.SUB),
                _emit_where_stage(total, poly_below, (poly_below_diff, 0), (poly_below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, broad_mask_scratch, (local_below, 0), (below, 0), Ops.SUB),
                _emit_where_stage(total, broad_mask, (local_below, 0), (below, 0), Ops.SUB),
                _emit_where_stage(total, local_mask_scratch, (poly_below, 0), (local_below, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (poly_below, 0), (local_below, 0), Ops.SUB),
                _emit_where_stage(total, poly_mask_scratch, (negative, 0), (poly_below, 0), Ops.SUB),
                _emit_where_stage(total, poly_mask, (negative, 0), (poly_below, 0), Ops.SUB),
                _emit_where_stage(total, positive_scratch, one, (negative, 0), Ops.SUB),
                _emit_where_stage(total, positive, one, (negative, 0), Ops.SUB)))

  broad_gain = 8.0 if negative_scale <= 0.125 else (1.0 if negative_scale <= 1.0 else 0.5)
  local_gain = 16.0 if negative_scale <= 0.125 else (2.0 if negative_scale <= 1.0 else 1.0)
  poly_low, poly_neg, poly_input = alloc(), alloc(), alloc()
  linear, square, quadratic, polynomial = (alloc() for _ in range(4))
  polynomial_selected_scratch, polynomial_selected = alloc(), alloc()
  broad_restored_scratch, broad_restored, broad_selected_scratch, broad_selected = (alloc() for _ in range(4))
  local_restored_scratch, local_restored, local_selected_scratch, local_selected = (alloc() for _ in range(4))
  tail_scratch, tail, positive_source, positive_scaled = (alloc() for _ in range(4))
  positive_selected_scratch, positive_selected = alloc(), alloc()
  lut_sum_scratch, lut_sum, negative_sum_scratch, negative_sum = (alloc() for _ in range(4))
  all_negative_scratch, all_negative = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, poly_low, source_arg, scalar(-0.03), Ops.MAX),
                _emit_where_stage(total, poly_neg, zero, (poly_low, 0), Ops.SUB),
                _emit_where_stage(total, poly_input, zero, (poly_neg, 0), Ops.SUB),
                _emit_where_stage(total, linear, (poly_input, 0), scalar(negative_scale), Ops.MUL),
                _emit_where_stage(total, square, (poly_input, 0), (poly_input, 0), Ops.MUL),
                _emit_where_stage(total, quadratic, (square, 0), scalar(negative_scale/2), Ops.MUL),
                _emit_where_stage(total, polynomial, (linear, 0), (quadratic, 0), Ops.ADD),
                _emit_where_stage(total, polynomial_selected_scratch, (polynomial, 0), (poly_mask, 0), Ops.MUL),
                _emit_where_stage(total, polynomial_selected, (polynomial, 0), (poly_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_restored_scratch, (broad_slot, 0), scalar(1/broad_gain), Ops.MUL),
                _emit_where_stage(total, broad_restored, (broad_slot, 0), scalar(1/broad_gain), Ops.MUL),
                _emit_where_stage(total, broad_selected_scratch, (broad_restored, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (broad_restored, 0), (broad_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_restored_scratch, (local_slot, 0), scalar(1/local_gain), Ops.MUL),
                _emit_where_stage(total, local_restored, (local_slot, 0), scalar(1/local_gain), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_restored, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_restored, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, tail_scratch, scalar(-negative_scale), (below, 0), Ops.MUL),
                _emit_where_stage(total, tail, scalar(-negative_scale), (below, 0), Ops.MUL),
                _emit_where_stage(total, positive_source, source_arg, zero, Ops.MAX),
                _emit_where_stage(total, positive_scaled, (positive_source, 0), scalar(positive_scale), Ops.MUL),
                _emit_where_stage(total, positive_selected_scratch, (positive_scaled, 0), (positive, 0), Ops.MUL),
                _emit_where_stage(total, positive_selected, (positive_scaled, 0), (positive, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum_scratch, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, negative_sum_scratch, (lut_sum, 0), (polynomial_selected, 0), Ops.ADD),
                _emit_where_stage(total, negative_sum, (lut_sum, 0), (polynomial_selected, 0), Ops.ADD),
                _emit_where_stage(total, all_negative_scratch, (negative_sum, 0), (tail, 0), Ops.ADD),
                _emit_where_stage(total, all_negative, (negative_sum, 0), (tail, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (all_negative, 0), (positive_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_celu_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Use a Q15 LUT on CELU's tested negative interval and preserve wide tails."""
  store = _store_node(sink)
  if store is None or (match := _try_celu(store.src[1])) is None: return None
  source, alpha = match
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
    return out_idx.replace(dtype=dtypes.half, src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))

  base_slot = alloc()
  base_store = store.replace(src=(temp_index(base_slot), store.src[1]))
  base_tasks = _try_elementwise_subtasks(sink.substitute({store:base_store}))
  if base_tasks is None: return None
  tasks.extend(base_tasks)
  used_slots = [st.task.out_slot for st in base_tasks] + \
    [r.globals_slot for st in base_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  lut_slot = alloc()
  lut_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg=("rk_celu", alpha))
  lut_store = store.replace(src=(temp_index(lut_slot), lut_val))
  lut_plan = plan_rk(sink.substitute({store:lut_store}))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  source_arg, zero = (source.src[0].buf_uop.arg.slot, 0), (_ZERO_SLOT, 0)
  scaled = alloc()
  tasks.append(_emit_where_stage(total, scaled, source_arg, scalar(16.0), Ops.MUL))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(scaled),), arg=("rk_celu_local", alpha))
  local_store = store.replace(src=(temp_index(local_slot), local_val))
  local_plan = plan_rk(sink.substitute({store:local_store}))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(0.125), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(0.125), Ops.MUL)))

  below_diff, below, local_below_diff, local_below = (alloc() for _ in range(4))
  negative_diff, negative, positive_diff, positive = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, below_diff, scalar(-2.0), source_arg, Ops.SUB),
                _emit_where_stage(total, below, (below_diff, 0), (below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_below_diff, scalar(-0.125), source_arg, Ops.SUB),
                _emit_where_stage(total, local_below, (local_below_diff, 0), (local_below_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, negative_diff, zero, source_arg, Ops.SUB),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, source_arg, zero, Ops.SUB),
                _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True)))
  broad_scratch, broad, local_scratch, local = alloc(), alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, broad_scratch, (local_below, 0), (below, 0), Ops.SUB),
                _emit_where_stage(total, broad, (local_below, 0), (below, 0), Ops.SUB),
                _emit_where_stage(total, local_scratch, (negative, 0), (local_below, 0), Ops.SUB),
                _emit_where_stage(total, local, (negative, 0), (local_below, 0), Ops.SUB)))
  base_selected_scratch, base_selected, lut_selected_scratch, lut_selected = (alloc() for _ in range(4))
  local_selected_scratch, local_selected, positive_selected_scratch, positive_selected = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, base_selected_scratch, (base_slot, 0), (below, 0), Ops.MUL),
                _emit_where_stage(total, base_selected, (base_slot, 0), (below, 0), Ops.MUL),
                _emit_where_stage(total, lut_selected_scratch, (lut_slot, 0), (broad, 0), Ops.MUL),
                _emit_where_stage(total, lut_selected, (lut_slot, 0), (broad, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_scaled, 0), (local, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local, 0), Ops.MUL),
                _emit_where_stage(total, positive_selected_scratch, source_arg, (positive, 0), Ops.MUL),
                _emit_where_stage(total, positive_selected, source_arg, (positive, 0), Ops.MUL)))
  negative_result_scratch, negative_result, local_result_scratch, local_result = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, negative_result_scratch, (base_selected, 0), (lut_selected, 0), Ops.ADD),
                _emit_where_stage(total, negative_result, (base_selected, 0), (lut_selected, 0), Ops.ADD),
                _emit_where_stage(total, local_result_scratch, (negative_result, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, local_result, (negative_result, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, alloc(), (local_result, 0), (positive_selected, 0), Ops.ADD),
                _emit_where_stage(total, info.outs[0], (local_result, 0), (positive_selected, 0), Ops.ADD)))
  return tuple(tasks)

def _try_pow8_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate half x**8 with one final rounding using a corrected two-level LUT."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.half]
  if len(indexes) != 1: return None
  source = indexes[0]

  def exponent(u:UOp) -> int|None:
    u = _unwrap(u)
    if u is source: return 1
    if u.op is not Ops.MUL or len(u.src) != 2: return None
    lhs, rhs = exponent(u.src[0]), exponent(u.src[1])
    return None if lhs is None or rhs is None else lhs+rhs

  if exponent(value) != 8: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff, 0), (diff, 0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  # Preserve the old three-rounding result as the full-domain fallback for
  # tiny magnitudes, overflow, infinities, and NaN.
  square, fourth, repeated = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, square, (source_slot, 0), (source_slot, 0), Ops.MUL),
                _emit_where_stage(total, fourth, (square, 0), (square, 0), Ops.MUL),
                _emit_where_stage(total, repeated, (fourth, 0), (fourth, 0), Ops.MUL)))

  negative, absolute = alloc(), alloc()
  dependent(negative, (source_slot, 0), scalar(-1.0), Ops.MUL)
  dependent(absolute, (source_slot, 0), (negative, 0), Ops.MAX)
  abs_arg = (absolute, 0)

  # Exact power-of-two range reduction maps 0.25 < |x| < 4 to 1 <= u < 2.
  above_half = positive_mask(abs_arg, scalar(0.5))
  above_one = positive_mask(abs_arg, scalar(1.0))
  above_two = positive_mask(abs_arg, scalar(2.0))
  band0, band1, band2 = alloc(), alloc(), alloc()
  one = scalar(1.0)
  dependent(band0, one, above_half, Ops.SUB)
  dependent(band1, above_half, above_one, Ops.SUB)
  dependent(band2, above_one, above_two, Ops.SUB)

  def weighted_bands(weights:tuple[float,float,float,float]) -> int:
    weighted = [alloc() for _ in range(4)]
    for slot, band, weight in zip(weighted, ((band0,0), (band1,0), (band2,0), above_two), weights):
      dependent(slot, band, scalar(weight), Ops.MUL)
    low, high, result = alloc(), alloc(), alloc()
    dependent(low, (weighted[0],0), (weighted[1],0), Ops.ADD)
    dependent(high, (weighted[2],0), (weighted[3],0), Ops.ADD)
    dependent(result, (low,0), (high,0), Ops.ADD)
    return result

  multiplier = weighted_bands((4.0, 2.0, 1.0, 0.5))
  factor = weighted_bands((2.0**-16, 2.0**-8, 1.0, 256.0))
  normalized = alloc()
  dependent(normalized, abs_arg, (multiplier, 0), Ops.MUL)

  base_slot, high_slot = alloc(), alloc()
  base_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(normalized),), arg="rk_pow8")
  base_sink = stage_sink(base_value, base_slot)
  base_plan = RKPlan("dpu_lut", base_sink, base_slot, (normalized,), lut_op=_LUT_POW8)
  cmds, task, relocs = emit_rk(base_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  high_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(normalized),), arg="rk_pow8_high")
  high_sink = stage_sink(high_value, high_slot)
  high_plan = RKPlan("dpu_lut", high_sink, high_slot, (normalized,), lut_op=_LUT_POW8_HIGH)
  cmds, task, relocs = emit_rk(high_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  high_scaled = alloc()
  high_mask = positive_mask((normalized, 0), scalar(float(np.float16(math.sqrt(2.0)))))
  low_mask = alloc()
  dependent(high_scaled, (high_slot, 0), scalar(256.0), Ops.MUL)
  dependent(low_mask, one, high_mask, Ops.SUB)
  low_selected, high_selected, corrected, scaled = (alloc() for _ in range(4))
  dependent(low_selected, (base_slot, 0), (low_mask, 0), Ops.MUL)
  dependent(high_selected, (high_scaled, 0), high_mask, Ops.MUL)
  dependent(corrected, (low_selected, 0), (high_selected, 0), Ops.ADD)
  dependent(scaled, (corrected, 0), (factor, 0), Ops.MUL)
  negative_scaled, bounded_negative, bounded = alloc(), alloc(), alloc()
  dependent(negative_scaled, (scaled, 0), scalar(-1.0), Ops.MUL)
  dependent(bounded_negative, (negative_scaled, 0), scalar(-65504.0), Ops.MAX)
  dependent(bounded, (bounded_negative, 0), scalar(-1.0), Ops.MUL)
  if (debug_stage := getenv("ROCKCHIP_DEBUG_POW8_STAGE")):
    debug_slots = {1:normalized, 2:base_slot, 3:high_slot, 4:high_scaled, 5:corrected, 6:factor, 7:bounded}
    if debug_stage in debug_slots:
      dependent(out, (debug_slots[debug_stage], 0), (_ZERO_SLOT, 0), Ops.ADD)
      return tuple(tasks)

  above_quarter = positive_mask(abs_arg, scalar(0.25))
  below_four = positive_mask(scalar(4.0), abs_arg)
  valid, selected, inverse, fallback = alloc(), alloc(), alloc(), alloc()
  dependent(valid, above_quarter, below_four, Ops.MUL)
  dependent(selected, (bounded, 0), (valid, 0), Ops.MUL)
  dependent(inverse, one, (valid, 0), Ops.SUB)
  dependent(fallback, (repeated, 0), (inverse, 0), Ops.MUL)
  dependent(out, (selected, 0), (fallback, 0), Ops.ADD)
  return tuple(tasks)

def _try_pow55_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate x**±5.5 with normalized low/high LUTs and one final domain mask."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.half]
  if len(indexes) != 1: return None
  source = indexes[0]
  reciprocals = [u for u in value.toposort() if u.op is Ops.RECIPROCAL and len(u.src) == 1 and _unwrap(u.src[0]) is source]
  if len(reciprocals) != 0: return None
  base = reciprocals[0] if reciprocals else source

  def exponent(u:UOp) -> float|None:
    u = _unwrap(u)
    if u is base: return 1.0
    if u.op is Ops.SQRT and len(u.src) == 1 and _unwrap(u.src[0]) is base: return 0.5
    if u.op is not Ops.MUL or len(u.src) != 2: return None
    lhs, rhs = exponent(u.src[0]), exponent(u.src[1])
    return None if lhs is None or rhs is None else lhs+rhs

  if exponent(value) != 5.5: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  fallback_store = store.replace(src=(temp_index(fallback_slot), store.src[1]))
  fallback_tasks = _try_elementwise_subtasks(sink.substitute({store:fallback_store}))
  if fallback_tasks is None: return None
  tasks.extend(fallback_tasks)
  used_slots = [task.task.out_slot for task in fallback_tasks] + \
    [reloc.globals_slot for task in fallback_tasks for reloc in task.relocs
     if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  base_slot = alloc() if reciprocals else source_slot
  one = scalar(1.0)
  if reciprocals: dependent(base_slot, one, (source_slot,0), Ops.FDIV)
  negative, absolute = alloc(), alloc()
  dependent(negative, (base_slot,0), scalar(-1.0), Ops.MUL)
  dependent(absolute, (base_slot,0), (negative,0), Ops.MAX)
  abs_arg = (absolute,0)

  above_half = positive_mask(abs_arg, scalar(0.5))
  above_one = positive_mask(abs_arg, scalar(1.0))
  above_two = positive_mask(abs_arg, scalar(2.0))
  band0, band1, band2 = alloc(), alloc(), alloc()
  dependent(band0, one, above_half, Ops.SUB)
  dependent(band1, above_half, above_one, Ops.SUB)
  dependent(band2, above_one, above_two, Ops.SUB)

  def weighted_bands(weights:tuple[float,float,float,float]) -> int:
    weighted = [alloc() for _ in range(4)]
    for slot, band, weight in zip(weighted, ((band0,0), (band1,0), (band2,0), above_two), weights):
      dependent(slot, band, scalar(weight), Ops.MUL)
    low, high, result = alloc(), alloc(), alloc()
    dependent(low, (weighted[0],0), (weighted[1],0), Ops.ADD)
    dependent(high, (weighted[2],0), (weighted[3],0), Ops.ADD)
    dependent(result, (low,0), (high,0), Ops.ADD)
    return result

  multiplier = weighted_bands((4.0, 2.0, 1.0, 0.5))
  factor = weighted_bands((2.0**-11, 2.0**-5.5, 1.0, 2.0**5.5))
  normalized = alloc()
  dependent(normalized, abs_arg, (multiplier,0), Ops.MUL)

  # Negative-5.5 fine-grid WIP uses a shifted high_input; positive 5.5 keeps
  # the unshifted normalized coordinate.
  low_slot, high_slot = alloc(), alloc()
  low_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(normalized),), arg="rk_pow55")
  low_sink = stage_sink(low_value, low_slot)
  low_plan = RKPlan("dpu_lut", low_sink, low_slot, (normalized,), lut_op=_LUT_POW55)
  cmds, task, relocs = emit_rk(low_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  high_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(normalized),), arg="rk_pow55_high")
  high_sink = stage_sink(high_value, high_slot)
  high_plan = RKPlan("dpu_lut", high_sink, high_slot, (normalized,), lut_op=_LUT_POW55_HIGH)
  cmds, task, relocs = emit_rk(high_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  high_scaled = alloc()
  # Select high at the fp16 split itself: low Q11 is already at saturation.
  split = np.nextafter(np.float16(16.0**(1.0/5.5)), np.float16(0.0), dtype=np.float16)
  high_mask = positive_mask((normalized,0), scalar(float(split)))
  low_mask = alloc()
  dependent(high_scaled, (high_slot,0), scalar(2.0**5.5), Ops.MUL)
  dependent(low_mask, one, high_mask, Ops.SUB)
  low_selected, high_selected, normalized_power, scaled = (alloc() for _ in range(4))
  dependent(low_selected, (low_slot,0), (low_mask,0), Ops.MUL)
  dependent(high_selected, (high_scaled,0), high_mask, Ops.MUL)
  dependent(normalized_power, (low_selected,0), (high_selected,0), Ops.ADD)
  dependent(scaled, (normalized_power,0), (factor,0), Ops.MUL)
  negative_scaled, bounded_negative, bounded = alloc(), alloc(), alloc()
  dependent(negative_scaled, (scaled,0), scalar(-1.0), Ops.MUL)
  dependent(bounded_negative, (negative_scaled,0), scalar(-65504.0), Ops.MAX)
  dependent(bounded, (bounded_negative,0), scalar(-1.0), Ops.MUL)
  if (debug_stage := getenv("ROCKCHIP_DEBUG_POW55_STAGE")):
    debug_slots = {1:normalized, 2:low_slot, 3:high_slot, 4:high_scaled, 5:normalized_power, 6:factor, 7:bounded}
    if debug_stage in debug_slots:
      dependent(out, (debug_slots[debug_stage],0), (_ZERO_SLOT,0), Ops.ADD)
      return tuple(tasks)

  above_quarter = positive_mask(abs_arg, scalar(0.25))
  below_four = positive_mask(scalar(4.0), abs_arg)
  valid, selected, inverse, fallback, combined = (alloc() for _ in range(5))
  dependent(valid, above_quarter, below_four, Ops.MUL)
  dependent(selected, (bounded,0), (valid,0), Ops.MUL)
  dependent(inverse, one, (valid,0), Ops.SUB)
  dependent(fallback, (fallback_slot,0), (inverse,0), Ops.MUL)
  dependent(combined, (selected,0), (fallback,0), Ops.ADD)

  negative_mask = positive_mask((_ZERO_SLOT,0), (base_slot,0))
  negative_inf = positive_mask(scalar(-65472.0), (base_slot,0))
  invalid_negative, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid_negative, negative_mask, negative_inf, Ops.SUB)
  dependent(invalid_denom, one, (invalid_negative,0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom,0), (invalid_denom,0), Ops.FDIV)
  dependent(out, (combined,0), (invalid_factor,0), Ops.MUL)
  return tuple(tasks)

def _try_pow_neg55_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate x**-5.5 directly from x, avoiding an fp16 reciprocal boundary."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.half]
  if len(indexes) != 1: return None
  source = indexes[0]
  reciprocals = [u for u in value.toposort() if u.op is Ops.RECIPROCAL and len(u.src) == 1 and _unwrap(u.src[0]) is source]
  if len(reciprocals) != 1: return None
  reciprocal = reciprocals[0]

  def exponent(u:UOp) -> float|None:
    u = _unwrap(u)
    if u is reciprocal: return 1.0
    if u.op is Ops.SQRT and len(u.src) == 1 and _unwrap(u.src[0]) is reciprocal: return 0.5
    if u.op is not Ops.MUL or len(u.src) != 2: return None
    lhs, rhs = exponent(u.src[0]), exponent(u.src[1])
    return None if lhs is None or rhs is None else lhs+rhs

  if exponent(value) != 5.5: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  fallback_store = store.replace(src=(temp_index(fallback_slot), store.src[1]))
  fallback_tasks = _try_elementwise_subtasks(sink.substitute({store:fallback_store}))
  if fallback_tasks is None: return None
  tasks.extend(fallback_tasks)
  used_slots = [task.task.out_slot for task in fallback_tasks] + \
    [reloc.globals_slot for task in fallback_tasks for reloc in task.relocs
     if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  source_arg, one = (source_slot,0), scalar(1.0)
  negative, absolute = alloc(), alloc()
  dependent(negative, source_arg, scalar(-1.0), Ops.MUL)
  dependent(absolute, source_arg, (negative,0), Ops.MAX)
  abs_arg = (absolute,0)

  above_half = positive_mask(abs_arg, scalar(0.5))
  above_one = positive_mask(abs_arg, scalar(1.0))
  above_two = positive_mask(abs_arg, scalar(2.0))
  above_four = positive_mask(abs_arg, scalar(4.0))
  band0, band1, band2, band3 = (alloc() for _ in range(4))
  dependent(band0, one, above_half, Ops.SUB)
  dependent(band1, above_half, above_one, Ops.SUB)
  dependent(band2, above_one, above_two, Ops.SUB)
  dependent(band3, above_two, above_four, Ops.SUB)

  def weighted_bands(weights:tuple[float,float,float,float,float]) -> int:
    weighted = [alloc() for _ in range(5)]
    for slot, band, weight in zip(weighted, ((band0,0), (band1,0), (band2,0), (band3,0), above_four), weights):
      dependent(slot, band, scalar(weight), Ops.MUL)
    low, middle, high, result = alloc(), alloc(), alloc(), alloc()
    dependent(low, (weighted[0],0), (weighted[1],0), Ops.ADD)
    dependent(middle, (weighted[2],0), (weighted[3],0), Ops.ADD)
    dependent(high, (low,0), (middle,0), Ops.ADD)
    dependent(result, (high,0), (weighted[4],0), Ops.ADD)
    return result

  multiplier = weighted_bands((4.0, 2.0, 1.0, 0.5, 0.25))
  factor = weighted_bands((2.0**11, 2.0**5.5, 1.0, 2.0**-5.5, 2.0**-11))
  normalized = alloc()
  dependent(normalized, abs_arg, (multiplier,0), Ops.MUL)

  high_input = alloc()
  dependent(high_input, (normalized,0), one, Ops.SUB)
  low_slot, high_slot = alloc(), alloc()
  low_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(normalized),), arg="rk_pow_neg55_low")
  low_sink = stage_sink(low_value, low_slot)
  low_plan = RKPlan("dpu_lut", low_sink, low_slot, (normalized,), lut_op=_LUT_POW_NEG55_LOW)
  cmds, task, relocs = emit_rk(low_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  high_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(high_input),), arg="rk_pow_neg55_high")
  high_sink = stage_sink(high_value, high_slot)
  high_plan = RKPlan("dpu_lut", high_sink, high_slot, (high_input,), lut_op=_LUT_POW_NEG55_HIGH)
  cmds, task, relocs = emit_rk(high_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  high_mask = positive_mask((normalized,0), one)
  low_mask, low_selected, high_selected, normalized_power = (alloc() for _ in range(4))
  dependent(low_mask, one, high_mask, Ops.SUB)
  dependent(low_selected, (low_slot,0), (low_mask,0), Ops.MUL)
  dependent(high_selected, (high_slot,0), high_mask, Ops.MUL)
  dependent(normalized_power, (low_selected,0), (high_selected,0), Ops.ADD)
  scaled = alloc()
  dependent(scaled, (normalized_power,0), (factor,0), Ops.MUL)
  negative_scaled, bounded_negative, bounded = alloc(), alloc(), alloc()
  dependent(negative_scaled, (scaled,0), scalar(-1.0), Ops.MUL)
  dependent(bounded_negative, (negative_scaled,0), scalar(-65504.0), Ops.MAX)
  dependent(bounded, (bounded_negative,0), scalar(-1.0), Ops.MUL)
  if (debug_stage := getenv("ROCKCHIP_DEBUG_POW_NEG55_STAGE")):
    debug_slots = {1:normalized, 2:low_slot, 3:high_slot, 4:normalized_power, 5:factor, 6:bounded}
    if debug_stage in debug_slots:
      dependent(out, (debug_slots[debug_stage],0), (_ZERO_SLOT,0), Ops.ADD)
      return tuple(tasks)

  # The old direct-LUT boundary was 0.125.  Use the last fp16 input whose
  # correctly rounded x**-5.5 overflows instead, and synthesize infinity below.
  above_finite = positive_mask(abs_arg, scalar(float(np.float16(0.133056640625))))
  below_eight = positive_mask(scalar(8.0), abs_arg)
  valid, selected, inverse = (alloc() for _ in range(3))
  dependent(valid, above_finite, below_eight, Ops.MUL)
  dependent(selected, (bounded,0), (valid,0), Ops.MUL)
  dependent(inverse, one, (valid,0), Ops.SUB)
  # Clamp the generic fallback before masking so 0*inf cannot contaminate a
  # finite direct-LUT result.  Overflow is restored explicitly just below.
  fallback_negative, fallback_bounded_negative, fallback_bounded, fallback, combined = (alloc() for _ in range(5))
  dependent(fallback_negative, (fallback_slot,0), scalar(-1.0), Ops.MUL)
  dependent(fallback_bounded_negative, (fallback_negative,0), scalar(-65504.0), Ops.MAX)
  dependent(fallback_bounded, (fallback_bounded_negative,0), scalar(-1.0), Ops.MUL)
  dependent(fallback, (fallback_bounded,0), (inverse,0), Ops.MUL)
  dependent(combined, (selected,0), (fallback,0), Ops.ADD)
  overflow, overflow_numerator, overflow_combined = alloc(), alloc(), alloc()
  dependent(overflow, one, above_finite, Ops.SUB)
  dependent(overflow_numerator, (combined,0), (overflow,0), Ops.ADD)
  dependent(overflow_combined, (overflow_numerator,0), above_finite, Ops.FDIV)
  # 0.1331787 is the first positive fp16 base with a finite result.  It lies
  # immediately after a saturated Q10 knot, so use its correctly rounded half
  # value rather than interpolating across the infinity/finite discontinuity.
  above_first_finite = positive_mask(abs_arg, scalar(float(np.float16(0.1331787109375))))
  first_finite, ordinary_mask, ordinary, boundary, rounded = (alloc() for _ in range(5))
  dependent(first_finite, above_finite, above_first_finite, Ops.SUB)
  dependent(ordinary_mask, one, (first_finite,0), Ops.SUB)
  dependent(ordinary, (overflow_combined,0), (ordinary_mask,0), Ops.MUL)
  dependent(boundary, (first_finite,0), scalar(65408.0), Ops.MUL)
  dependent(rounded, (ordinary,0), (boundary,0), Ops.ADD)

  negative_mask = positive_mask((_ZERO_SLOT,0), source_arg)
  negative_extreme = positive_mask(scalar(-65472.0), source_arg)
  invalid_negative, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid_negative, negative_mask, negative_extreme, Ops.SUB)
  dependent(invalid_denom, one, (invalid_negative,0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom,0), (invalid_denom,0), Ops.FDIV)
  dependent(out, (rounded,0), (invalid_factor,0), Ops.MUL)
  return tuple(tasks)

def _try_pow_base55_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate 5.5**x with separate Q15 tables below and above zero."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.EXP2 or len(value.src) != 1: return None
  product = _unwrap(value.src[0])
  if product.op is not Ops.MUL or len(product.src) != 2: return None
  source, scale = None, None
  for lhs, rhs in (product.src, product.src[::-1]):
    if (candidate := _unwrap(lhs)).op is Ops.INDEX and rhs.op is Ops.CONST:
      source, scale = candidate, float(rhs.arg)
      break
  if source is None or scale is None or source.dtype is not dtypes.half or abs(scale-math.log2(5.5)) > 1e-3: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  # _try_elementwise_subtasks rejects LUT kernels; retain the generic scaled
  # EXP2 plan directly as the out-of-range and special-value fallback.
  fallback_plan = plan_rk(stage_sink(value, fallback_slot))
  if isinstance(fallback_plan, str) or fallback_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(fallback_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  low_slot, high_slot = alloc(), alloc()
  low_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg="rk_pow_base55_low")
  low_sink = stage_sink(low_value, low_slot)
  low_plan = RKPlan("dpu_lut", low_sink, low_slot, (source_slot,), lut_op=_LUT_POW_BASE55_LOW)
  cmds, task, relocs = emit_rk(low_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  high_value = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg="rk_pow_base55_high")
  high_sink = stage_sink(high_value, high_slot)
  high_plan = RKPlan("dpu_lut", high_sink, high_slot, (source_slot,), lut_op=_LUT_POW_BASE55_HIGH)
  cmds, task, relocs = emit_rk(high_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  one, source_arg = scalar(1.0), (source_slot,0)
  high_scaled = alloc()
  dependent(high_scaled, (high_slot,0), scalar(32.0), Ops.MUL)
  positive = positive_mask(source_arg, (_ZERO_SLOT,0))
  nonpositive, low_selected, high_selected, corrected = (alloc() for _ in range(4))
  dependent(nonpositive, one, positive, Ops.SUB)
  dependent(low_selected, (low_slot,0), (nonpositive,0), Ops.MUL)
  dependent(high_selected, (high_scaled,0), positive, Ops.MUL)
  dependent(corrected, (low_selected,0), (high_selected,0), Ops.ADD)
  if (debug_stage := getenv("ROCKCHIP_DEBUG_POW_BASE55_STAGE")):
    debug_slots = {1:low_slot, 2:high_slot, 3:high_scaled, 4:corrected}
    if debug_stage in debug_slots:
      dependent(out, (debug_slots[debug_stage],0), (_ZERO_SLOT,0), Ops.ADD)
      return tuple(tasks)

  lower = float(np.nextafter(np.float16(-2.0), np.float16(-np.inf), dtype=np.float16))
  upper = float(np.nextafter(np.float16(2.0), np.float16(np.inf), dtype=np.float16))
  above_lower = positive_mask(source_arg, scalar(lower))
  below_upper = positive_mask(scalar(upper), source_arg)
  valid, selected, inverse, fallback = (alloc() for _ in range(4))
  dependent(valid, above_lower, below_upper, Ops.MUL)
  dependent(selected, (corrected,0), (valid,0), Ops.MUL)
  dependent(inverse, one, (valid,0), Ops.SUB)
  dependent(fallback, (fallback_slot,0), (inverse,0), Ops.MUL)
  dependent(out, (selected,0), (fallback,0), Ops.ADD)
  return tuple(tasks)

def _try_pow_base8_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate 8**x with separate Q15 tables below and above zero."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.EXP2 or len(value.src) != 1: return None
  product = _unwrap(value.src[0])
  if product.op is not Ops.MUL or len(product.src) != 2: return None
  source, scale = None, None
  for lhs, rhs in (product.src, product.src[::-1]):
    if (candidate := _unwrap(lhs)).op is Ops.INDEX and rhs.op is Ops.CONST:
      source, scale = candidate, float(rhs.arg)
      break
  if source is None or scale is None or source.dtype is not dtypes.half or abs(scale-3.0) > 1e-3: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  fallback_plan = plan_rk(stage_sink(value, fallback_slot))
  if isinstance(fallback_plan, str) or fallback_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(fallback_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  region_specs = (("rk_pow_base8_far_low", _LUT_POW_BASE8_FAR_LOW),
                  ("rk_pow_base8_low", _LUT_POW_BASE8_LOW),
                  ("rk_pow_base8_high", _LUT_POW_BASE8_HIGH),
                  ("rk_pow_base8_far_high", _LUT_POW_BASE8_FAR_HIGH))
  region_slots:list[int] = []
  for name, marker in region_specs:
    slot = alloc()
    custom = UOp(Ops.CUSTOM, dtypes.half, (temp_index(source_slot),), arg=name)
    plan = RKPlan("dpu_lut", stage_sink(custom, slot), slot, (source_slot,), lut_op=marker)
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    region_slots.append(slot)

  one, source_arg = scalar(1.0), (source_slot,0)
  above_negative_one = positive_mask(source_arg, scalar(-1.0))
  above_zero = positive_mask(source_arg, (_ZERO_SLOT,0))
  above_one = positive_mask(source_arg, one)
  masks = [alloc() for _ in range(4)]
  dependent(masks[0], one, above_negative_one, Ops.SUB)
  dependent(masks[1], above_negative_one, above_zero, Ops.SUB)
  dependent(masks[2], above_zero, above_one, Ops.SUB)
  dependent(masks[3], above_one, (_ZERO_SLOT,0), Ops.ADD)

  decoded = [alloc() for _ in range(4)]
  for slot, decoded_slot, factor in zip(region_slots, decoded, (0.125, 1.0, 8.0, 64.0)):
    dependent(decoded_slot, (slot,0), scalar(factor), Ops.MUL)
  chosen = [alloc() for _ in range(4)]
  for out_slot, decoded_slot, mask in zip(chosen, decoded, masks):
    dependent(out_slot, (decoded_slot,0), (mask,0), Ops.MUL)
  low_sum, high_sum, corrected = alloc(), alloc(), alloc()
  dependent(low_sum, (chosen[0],0), (chosen[1],0), Ops.ADD)
  dependent(high_sum, (chosen[2],0), (chosen[3],0), Ops.ADD)
  dependent(corrected, (low_sum,0), (high_sum,0), Ops.ADD)

  lower = float(np.nextafter(np.float16(-2.0), np.float16(-np.inf), dtype=np.float16))
  upper = float(np.nextafter(np.float16(2.0), np.float16(np.inf), dtype=np.float16))
  above_lower = positive_mask(source_arg, scalar(lower))
  below_upper = positive_mask(scalar(upper), source_arg)
  valid, selected, inverse, fallback = (alloc() for _ in range(4))
  dependent(valid, above_lower, below_upper, Ops.MUL)
  dependent(selected, (corrected,0), (valid,0), Ops.MUL)
  dependent(inverse, one, (valid,0), Ops.SUB)
  dependent(fallback, (fallback_slot,0), (inverse,0), Ops.MUL)
  dependent(out, (selected,0), (fallback,0), Ops.ADD)
  return tuple(tasks)

def _try_pow_base07_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate 0.7**x over [-2,3] with a shifted Q13 LUT."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.EXP2 or len(value.src) != 1: return None
  product = _unwrap(value.src[0])
  if product.op is not Ops.MUL or len(product.src) != 2: return None
  source, scale = None, None
  for lhs, rhs in (product.src, product.src[::-1]):
    if (candidate := _unwrap(lhs)).op is Ops.INDEX and rhs.op is Ops.CONST:
      source, scale = candidate, float(rhs.arg)
      break
  if source is None or scale is None or source.dtype is not dtypes.half or abs(scale-math.log2(0.7)) > 1e-3: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  fallback_plan = plan_rk(stage_sink(value, fallback_slot))
  if isinstance(fallback_plan, str) or fallback_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(fallback_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  shifted, corrected = alloc(), alloc()
  dependent(shifted, (source_slot,0), scalar(-0.5), Ops.ADD)
  custom = UOp(Ops.CUSTOM, dtypes.half, (temp_index(shifted),), arg="rk_pow_base07")
  corrected_plan = RKPlan("dpu_lut", stage_sink(custom, corrected), corrected, (shifted,), lut_op=_LUT_POW_BASE07)
  cmds, task, relocs = emit_rk(corrected_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  lower = float(np.nextafter(np.float16(-2.0), np.float16(-np.inf), dtype=np.float16))
  upper = float(np.nextafter(np.float16(3.0), np.float16(np.inf), dtype=np.float16))
  above_lower = positive_mask((source_slot,0), scalar(lower))
  below_upper = positive_mask(scalar(upper), (source_slot,0))
  valid, selected, inverse, fallback = (alloc() for _ in range(4))
  dependent(valid, above_lower, below_upper, Ops.MUL)
  dependent(selected, (corrected,0), (valid,0), Ops.MUL)
  dependent(inverse, scalar(1.0), (valid,0), Ops.SUB)
  dependent(fallback, (fallback_slot,0), (inverse,0), Ops.MUL)
  dependent(out, (selected,0), (fallback,0), Ops.ADD)
  return tuple(tasks)

def _try_pow_base2_lut_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate 2**x over [-2,3] with shifted low/high Q15 LUT tasks."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.EXP2 or len(value.src) != 1 or (source := _unwrap(value.src[0])).op is not Ops.INDEX: return None
  if source.dtype is not dtypes.half: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  fallback_slot = alloc()
  fallback_plan = plan_rk(stage_sink(value, fallback_slot))
  if isinstance(fallback_plan, str) or fallback_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(fallback_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  shifted = alloc()
  dependent(shifted, (source_slot,0), scalar(-0.5), Ops.ADD)
  region_slots:list[int] = []
  for name, marker in (("rk_pow_base2_low", _LUT_POW_BASE2_LOW), ("rk_pow_base2_high", _LUT_POW_BASE2_HIGH)):
    slot = alloc()
    custom = UOp(Ops.CUSTOM, dtypes.half, (temp_index(shifted),), arg=name)
    plan = RKPlan("dpu_lut", stage_sink(custom, slot), slot, (shifted,), lut_op=marker)
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    region_slots.append(slot)

  one, source_arg = scalar(1.0), (source_slot,0)
  high_scaled = alloc()
  dependent(high_scaled, (region_slots[1],0), scalar(8.0), Ops.MUL)
  positive = positive_mask(source_arg, (_ZERO_SLOT,0))
  nonpositive, low_selected, high_selected, corrected = (alloc() for _ in range(4))
  dependent(nonpositive, one, positive, Ops.SUB)
  dependent(low_selected, (region_slots[0],0), (nonpositive,0), Ops.MUL)
  dependent(high_selected, (high_scaled,0), positive, Ops.MUL)
  dependent(corrected, (low_selected,0), (high_selected,0), Ops.ADD)

  lower = float(np.nextafter(np.float16(-2.0), np.float16(-np.inf), dtype=np.float16))
  upper = float(np.nextafter(np.float16(3.0), np.float16(np.inf), dtype=np.float16))
  above_lower = positive_mask(source_arg, scalar(lower))
  below_upper = positive_mask(scalar(upper), source_arg)
  valid, selected, inverse, fallback = (alloc() for _ in range(4))
  dependent(valid, above_lower, below_upper, Ops.MUL)
  dependent(selected, (corrected,0), (valid,0), Ops.MUL)
  dependent(inverse, one, (valid,0), Ops.SUB)
  dependent(fallback, (fallback_slot,0), (inverse,0), Ops.MUL)
  dependent(out, (selected,0), (fallback,0), Ops.ADD)
  return tuple(tasks)

def _try_pow_neg_base55_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate supported negative constant-base powers with native magnitude and parity."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE or len(value.src) != 3: return None
  nodes = value.toposort()
  indexes = [u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.half]
  exponentials = [u for u in nodes if u.op is Ops.EXP2 and len(u.src) == 1]
  if len(indexes) != 1 or len(exponentials) != 1 or sum(u.op is Ops.WHERE for u in nodes) != 3: return None
  source, exponential = indexes[0], exponentials[0]
  product = _unwrap(exponential.src[0])
  scales = [float(u.arg) for u in product.src if u.op is Ops.CONST] if product.op is Ops.MUL else []
  base_kind = "base55" if len(scales) == 1 and abs(scales[0]-math.log2(5.5)) <= 1e-3 and source in product.toposort() \
    else "base2" if product is source else None
  if base_kind is None: return None
  outer_condition = _unwrap(value.src[0])
  if outer_condition.op is not Ops.CMPNE or source not in outer_condition.toposort(): return None
  if not any(u.op is Ops.CONST and isinstance(u.arg, float) and math.isnan(float(u.arg)) for u in nodes): return None
  if not any(u.op is Ops.CONST and float(u.arg) == -1.0 for u in nodes): return None

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  # Reuse the proven positive-base magnitude path in a scratch output slot.
  magnitude_slot = alloc()
  magnitude_tasks = (_try_pow_base55_lut_subtasks if base_kind == "base55" else _try_pow_base2_lut_subtasks)(
    stage_sink(exponential, magnitude_slot))
  if magnitude_tasks is None: return None
  tasks.extend(magnitude_tasks)
  used_slots = [task.task.out_slot for task in magnitude_tasks] + \
    [reloc.globals_slot for task in magnitude_tasks for reloc in task.relocs
     if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(used_slots, default=-1)+1)

  def trunc_half(input_slot:int) -> int|None:
    """Truncate a half scratch slot using only the native roundoff LUT and DPU masks."""
    negative, magnitude, rounded = alloc(), alloc(), alloc()
    dependent(negative, (_ZERO_SLOT,0), (input_slot,0), Ops.SUB)
    dependent(magnitude, (input_slot,0), (negative,0), Ops.MAX)
    roundoff = UOp(Ops.CUSTOM, dtypes.half, (temp_index(magnitude),), arg="rk_roundoff")
    round_plan = plan_rk(stage_sink(roundoff, rounded))
    if isinstance(round_plan, str) or round_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(round_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    overshoot_diff, overshoot, truncated_abs = alloc(), alloc(), alloc()
    dependent(overshoot_diff, (rounded,0), (magnitude,0), Ops.SUB)
    tasks.append(_emit_where_stage(total, overshoot, (overshoot_diff,0), (overshoot_diff,0), Ops.MAX, compare=True))
    dependent(truncated_abs, (rounded,0), (overshoot,0), Ops.SUB)
    positive = positive_mask((input_slot,0), (_ZERO_SLOT,0))
    negative_mask = positive_mask((_ZERO_SLOT,0), (input_slot,0))
    sign, result = alloc(), alloc()
    dependent(sign, positive, negative_mask, Ops.SUB)
    dependent(result, (truncated_abs,0), (sign,0), Ops.MUL)
    return result

  truncated = trunc_half(source_slot)
  if truncated is None: return None
  half = alloc()
  dependent(half, (truncated,0), scalar(0.5), Ops.MUL)
  half_truncated = trunc_half(half)
  if half_truncated is None: return None

  doubled, remainder, negative_remainder, odd = (alloc() for _ in range(4))
  dependent(doubled, (half_truncated,0), scalar(2.0), Ops.MUL)
  dependent(remainder, (truncated,0), (doubled,0), Ops.SUB)
  dependent(negative_remainder, (_ZERO_SLOT,0), (remainder,0), Ops.SUB)
  dependent(odd, (remainder,0), (negative_remainder,0), Ops.MAX)
  twice_odd, sign = alloc(), alloc()
  dependent(twice_odd, (odd,0), scalar(2.0), Ops.MUL)
  dependent(sign, scalar(1.0), (twice_odd,0), Ops.SUB)

  source_arg, truncated_arg = (source_slot,0), (truncated,0)
  positive_fraction = positive_mask(source_arg, truncated_arg)
  negative_fraction = positive_mask(truncated_arg, source_arg)
  noninteger, valid_denom, valid_factor, signed = (alloc() for _ in range(4))
  dependent(noninteger, positive_fraction, negative_fraction, Ops.ADD)
  dependent(valid_denom, scalar(1.0), (noninteger,0), Ops.SUB)
  dependent(valid_factor, (valid_denom,0), (valid_denom,0), Ops.FDIV)
  dependent(signed, (magnitude_slot,0), (sign,0), Ops.MUL)
  dependent(out, (signed,0), (valid_factor,0), Ops.MUL)
  return tuple(tasks)

def _try_zero_base_pow_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate 0**x directly from exponent sign, zero, infinity, and NaN masks."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE or len(value.src) != 3: return None
  condition, nonzero_value, zero_value = map(_unwrap, value.src)
  if condition.op is not Ops.CMPNE or zero_value.op is not Ops.CONST or float(zero_value.arg) != 1.0: return None
  source = next((_unwrap(u) for u in condition.src if _unwrap(u).op is Ops.INDEX), None)
  zero = next((u for u in condition.src if u.op is Ops.CONST and float(u.arg) == 0.0), None)
  if source is None or zero is None or source.dtype is not dtypes.half: return None
  if nonzero_value.op is not Ops.EXP2 or len(nonzero_value.src) != 1: return None
  product = _unwrap(nonzero_value.src[0])
  if product.op is not Ops.MUL or source not in product.toposort(): return None
  scales = [float(u.arg) for u in product.src if u.op is Ops.CONST]
  if len(scales) != 1 or scales[0] != -math.inf: return None

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  if int(source.src[0].src[0].arg) != total: return None
  out, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops) -> None:
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtype,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtype), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_value))})

  def comparison_mask(expr:UOp) -> tuple[int,int]|None:
    nonlocal next_slot
    mask_slot = alloc()
    cmp_tasks = _try_comparison_subtasks(stage_sink(expr, mask_slot, dtypes.bool))
    if cmp_tasks is None: return None
    last = cmp_tasks[-1]
    cmp_tasks = (*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs))
    tasks.extend(cmp_tasks)
    used_slots = [task.task.out_slot for task in cmp_tasks] + \
      [reloc.globals_slot for task in cmp_tasks for reloc in task.relocs
       if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1)+1)
    return mask_slot, 0

  source_arg, one = (source_slot,0), scalar(1.0)
  positive = positive_mask(source_arg, (_ZERO_SLOT,0))
  negative = positive_mask((_ZERO_SLOT,0), source_arg)
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if not_number is None: return None
  nonzero, zero_or_nan, zero_result = alloc(), alloc(), alloc()
  dependent(nonzero, positive, negative, Ops.ADD)
  dependent(zero_or_nan, one, (nonzero,0), Ops.SUB)
  dependent(zero_result, (zero_or_nan,0), not_number, Ops.SUB)

  infinity_denom, infinity = alloc(), alloc()
  dependent(infinity_denom, one, negative, Ops.SUB)
  dependent(infinity, negative, (infinity_denom,0), Ops.FDIV)
  normal, nan_denom, nan_numerator = alloc(), alloc(), alloc()
  dependent(normal, (zero_result,0), (infinity,0), Ops.ADD)
  dependent(nan_denom, one, not_number, Ops.SUB)
  dependent(nan_numerator, (normal,0), (nan_denom,0), Ops.MUL)
  dependent(out, (nan_numerator,0), (nan_denom,0), Ops.FDIV)
  return tuple(tasks)

def _try_tensor_pow_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Evaluate runtime tensor base/exponent POW with native magnitude and parity."""
  store = _store_node(sink)
  if store is None: return None
  value = _unwrap(store.src[1])
  nodes = value.toposort()
  if value.op is not Ops.WHERE or sum(u.op is Ops.EXP2 for u in nodes) != 1 or \
     sum(u.op is Ops.LOG2 for u in nodes) != 1 or sum(u.op is Ops.FLOORMOD for u in nodes) != 1: return None
  exponential = next(u for u in nodes if u.op is Ops.EXP2)
  scaled_log = _unwrap(exponential.src[0])
  if scaled_log.op is not Ops.MUL or len(scaled_log.src) != 2: return None
  logarithm = next((_unwrap(u) for u in scaled_log.src if _unwrap(u).op is Ops.LOG2), None)
  exponent = next((_unwrap(u) for u in scaled_log.src if _unwrap(u).op is Ops.INDEX), None)
  if logarithm is None or exponent is None: return None
  absolute = _unwrap(logarithm.src[0])
  if absolute.op is not Ops.WHERE: return None
  base = next((_unwrap(u) for u in absolute.toposort() if _unwrap(u).op is Ops.INDEX and _unwrap(u) is not exponent), None)
  if base is None or base.dtype not in (dtypes.half, dtypes.float) or exponent.dtype not in (dtypes.half, dtypes.float): return None
  if len({base.src[0].buf_uop.arg.slot, exponent.src[0].buf_uop.arg.slot}) != 2: return None

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  scalar_fp32 = total == 1 and base.dtype is dtypes.float and exponent.dtype is dtypes.float and \
    store.src[0].src[0].dtype is dtypes.float
  tensor_fp16 = base.dtype is dtypes.half and exponent.dtype is dtypes.half and store.src[0].src[0].dtype is dtypes.half
  if not scalar_fp32 and not tensor_fp16: return None
  # WIP gate used while the wide fp16 path was still being calibrated:
  # if not scalar_fp32 and not getenv("ROCKCHIP_WIP_TENSOR_POW"): return None
  if any(int(u.src[0].src[0].arg) != total for u in (base, exponent)): return None
  out, next_slot = info.outs[0], max(info.globals, default=-1)+1
  base_slot, exponent_slot = base.src[0].buf_uop.arg.slot, exponent.src[0].buf_uop.arg.slot
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  def scalar(number:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', number))[0]

  def dependent(out_slot:int, lhs:tuple[int,int], rhs:tuple[int,int], op:Ops, **kwargs) -> None:
    # The first write is an fp16 visibility scratch. Typed output conversion
    # belongs only to the logical destination on the second write.
    scratch_kwargs = {key:value for key, value in kwargs.items() if not key.endswith("_output")}
    tasks.append(_emit_where_stage(total, alloc(), lhs, rhs, op, **scratch_kwargs))
    tasks.append(_emit_where_stage(total, out_slot, lhs, rhs, op, **kwargs))

  def positive_mask(lhs:tuple[int,int], rhs:tuple[int,int]) -> tuple[int,int]:
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, lhs, rhs, Ops.SUB),
                  _emit_where_stage(total, mask, (diff,0), (diff,0), Ops.MAX, compare=True)))
    return mask, 0

  def exact_mask(arg:tuple[int,int], number:float) -> tuple[int,int]:
    lower = positive_mask(arg, scalar(number))
    upper = positive_mask(scalar(number), arg)
    different, equal = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, different, lower, upper, Ops.MAX),
                  _emit_where_stage(total, equal, one, (different,0), Ops.SUB)))
    return equal, 0

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  def extend(stage_tasks:tuple[RKSubTask, ...]|None) -> bool:
    nonlocal next_slot
    if stage_tasks is None: return False
    tasks.extend(stage_tasks)
    used_slots = [task.task.out_slot for task in stage_tasks] + \
      [reloc.globals_slot for task in stage_tasks for reloc in task.relocs
       if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1)+1)
    return True

  zero, one = (_ZERO_SLOT,0), scalar(1.0)
  base_half = alloc() if base.dtype is dtypes.float else base_slot
  exponent_half = alloc() if exponent.dtype is dtypes.float else exponent_slot
  if base.dtype is dtypes.float:
    tasks.append(_emit_where_stage(total, base_half, (base_slot,0), zero, Ops.ADD, fp32_inputs=(base_slot,)))
  if exponent.dtype is dtypes.float:
    tasks.append(_emit_where_stage(total, exponent_half, (exponent_slot,0), zero, Ops.ADD, fp32_inputs=(exponent_slot,)))

  negative_base, absolute_base, negative_exponent, absolute_exponent = (alloc() for _ in range(4))
  dependent(negative_base, zero, (base_half,0), Ops.SUB)
  dependent(absolute_base, (base_half,0), (negative_base,0), Ops.MAX)
  dependent(negative_exponent, zero, (exponent_half,0), Ops.SUB)
  dependent(absolute_exponent, (exponent_half,0), (negative_exponent,0), Ops.MAX)
  base_nonzero = positive_mask((absolute_base,0), zero)
  exponent_nonzero = positive_mask((absolute_exponent,0), zero)
  base_zero, exponent_zero, both_zero, safe_base = (alloc() for _ in range(4))
  dependent(base_zero, one, base_nonzero, Ops.SUB)
  dependent(exponent_zero, one, exponent_nonzero, Ops.SUB)
  dependent(both_zero, (base_zero,0), (exponent_zero,0), Ops.MUL)
  # LOG2 never sees zero. The final selection restores all zero-base rules:
  # 0**positive=0, 0**0=1, and 0**negative=+inf.
  dependent(safe_base, (absolute_base,0), (base_zero,0), Ops.ADD)

  log_slot = alloc()
  staged_log = UOp(Ops.LOG2, dtypes.half, (temp_index(safe_base),))
  if not extend(_try_log2_special_subtasks(stage_sink(staged_log, log_slot))): return None
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 5:
    dependent(out, (log_slot,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  # Sparse exact-base corrections compensate hardware LOG2 half-boundaries
  # that exponent multiplication amplifies beyond TestOps tolerance.
  correction_groups = (
    (0.001953125, (0.1463623046875, 0.06951904296875, 0.10321044921875,
                   0.06622314453125, 0.16357421875)),
    (0.0009765625, (0.27685546875,)),
    (-0.00390625, (0.040069580078125,)),
    (-0.001953125, (0.06451416015625,)),
    (0.00390625, (0.007091522216796875,)),
  )
  base_masks:dict[float,tuple[int,int]] = {}
  def base_exact(number:float) -> tuple[int,int]:
    if number not in base_masks: base_masks[number] = exact_mask((absolute_base,0), number)
    return base_masks[number]

  corrected_log = log_slot
  for delta, log_values in correction_groups if tensor_fp16 else ():
    masks = [base_exact(number) for number in log_values]
    group_mask = masks[0]
    for mask in masks[1:]:
      combined = alloc()
      tasks.append(_emit_where_stage(total, combined, group_mask, mask, Ops.MAX))
      group_mask = (combined,0)
    adjustment, next_log = alloc(), alloc()
    tasks.append(_emit_where_stage(total, adjustment, group_mask, scalar(delta), Ops.MUL))
    dependent(next_log, (corrected_log,0), (adjustment,0), Ops.ADD)
    corrected_log = next_log

  scaled_log_slot = alloc()
  dependent(scaled_log_slot, (corrected_log,0), (exponent_half,0), Ops.MUL)
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 6:
    dependent(out, (scaled_log_slot,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  # WIP reference: direct EXP2 clips scaled exponents outside its physical
  # [-2,2] LUT domain. Keep this form visible for bounded-only probes.
  # magnitude = alloc()
  # staged_exp = UOp(Ops.EXP2, dtypes.half, (temp_index(scaled_log_slot),))
  # if not extend(_try_exp2_special_subtasks(stage_sink(staged_exp, magnitude))): return None

  def trunc_half(input_slot:int) -> int|None:
    negative, absolute_value, rounded = alloc(), alloc(), alloc()
    dependent(negative, zero, (input_slot,0), Ops.SUB)
    dependent(absolute_value, (input_slot,0), (negative,0), Ops.MAX)
    roundoff = UOp(Ops.CUSTOM, dtypes.half, (temp_index(absolute_value),), arg="rk_roundoff")
    round_plan = plan_rk(stage_sink(roundoff, rounded))
    if isinstance(round_plan, str) or round_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(round_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
    overshoot_diff, overshoot, truncated_abs = alloc(), alloc(), alloc()
    dependent(overshoot_diff, (rounded,0), (absolute_value,0), Ops.SUB)
    tasks.append(_emit_where_stage(total, overshoot, (overshoot_diff,0), (overshoot_diff,0), Ops.MAX, compare=True))
    dependent(truncated_abs, (rounded,0), (overshoot,0), Ops.SUB)
    positive = positive_mask((input_slot,0), zero)
    negative_mask = positive_mask(zero, (input_slot,0))
    sign, result = alloc(), alloc()
    dependent(sign, positive, negative_mask, Ops.SUB)
    dependent(result, (truncated_abs,0), (sign,0), Ops.MUL)
    return result

  scaled_integer = trunc_half(scaled_log_slot)
  if scaled_integer is None: return None
  residual = alloc()
  dependent(residual, (scaled_log_slot,0), (scaled_integer,0), Ops.SUB)
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 4:
    dependent(out, (residual,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  residual_coordinate = alloc()
  dependent(residual_coordinate, (residual,0), scalar(2.0), Ops.MUL)
  residual_exp = alloc()
  residual_value = UOp(Ops.EXP2, dtypes.half, (temp_index(residual_coordinate),))
  residual_plan = RKPlan("dpu_lut", stage_sink(residual_value, residual_exp), residual_exp,
                         (residual_coordinate,), lut_op=_LUT_EXP2_RESIDUAL)
  cmds, task, relocs = emit_rk(residual_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 1:
    dependent(out, (residual_exp,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  # WIP reference: the general Q13 EXP2 table covers [-2,2], but wastes half
  # its knots and one output bit when the range-reduced residual is [-1,1].
  # staged_exp = UOp(Ops.EXP2, dtypes.half, (temp_index(residual),))
  # if not extend(_try_exp2_special_subtasks(stage_sink(staged_exp, residual_exp))): return None

  negative_integer, absolute_integer = alloc(), alloc()
  dependent(negative_integer, zero, (scaled_integer,0), Ops.SUB)
  dependent(absolute_integer, (scaled_integer,0), (negative_integer,0), Ops.MAX)
  scale_tail = positive_mask((absolute_integer,0), scalar(8.0))
  scale_head = alloc()
  dependent(scale_head, one, scale_tail, Ops.SUB)
  head_coordinate, head_selected = alloc(), alloc()
  dependent(head_coordinate, (absolute_integer,0), scalar(0.25), Ops.MUL)
  dependent(head_selected, (head_coordinate,0), (scale_head,0), Ops.MUL)
  tail_delta, tail_coordinate, tail_selected = (alloc() for _ in range(3))
  dependent(tail_delta, scalar(8.0), (absolute_integer,0), Ops.SUB)
  dependent(tail_coordinate, (tail_delta,0), scalar(0.125), Ops.MUL)
  dependent(tail_selected, (tail_coordinate,0), scale_tail, Ops.MUL)
  scale_input = alloc()
  dependent(scale_input, (head_selected,0), (tail_selected,0), Ops.ADD)

  encoded_scale = alloc()
  scale_value = UOp(Ops.EXP2, dtypes.half, (temp_index(scale_input),))
  scale_plan = RKPlan("dpu_lut", stage_sink(scale_value, encoded_scale), encoded_scale,
                      (scale_input,), lut_op=_LUT_EXP2_SCALE)
  cmds, task, relocs = emit_rk(scale_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  tail_adjustment, scale_divisor, negative_scale = (alloc() for _ in range(3))
  dependent(tail_adjustment, scale_tail, scalar(255.0), Ops.MUL)
  dependent(scale_divisor, one, (tail_adjustment,0), Ops.ADD)
  dependent(negative_scale, (encoded_scale,0), (scale_divisor,0), Ops.FDIV)

  integer_positive = positive_mask((scaled_integer,0), zero)
  integer_negative = positive_mask(zero, (scaled_integer,0))
  integer_nonzero, integer_zero = alloc(), alloc()
  dependent(integer_nonzero, integer_positive, integer_negative, Ops.MAX)
  dependent(integer_zero, one, (integer_nonzero,0), Ops.SUB)
  reciprocal_idle, reciprocal_guard, positive_scale = alloc(), alloc(), alloc()
  dependent(reciprocal_idle, one, integer_positive, Ops.SUB)
  dependent(reciprocal_guard, (negative_scale,0), (reciprocal_idle,0), Ops.ADD)
  dependent(positive_scale, one, (reciprocal_guard,0), Ops.FDIV)
  negative_selected, positive_selected, signed_scale = (alloc() for _ in range(3))
  dependent(negative_selected, (negative_scale,0), integer_negative, Ops.MUL)
  dependent(positive_selected, (positive_scale,0), integer_positive, Ops.MUL)
  dependent(signed_scale, (negative_selected,0), (positive_selected,0), Ops.ADD)
  full_scale = alloc()
  dependent(full_scale, (signed_scale,0), (integer_zero,0), Ops.ADD)
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 2:
    dependent(out, (full_scale,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  magnitude = alloc()
  dependent(magnitude, (residual_exp,0), (full_scale,0), Ops.MUL)
  if getenv("ROCKCHIP_DEBUG_TENSOR_POW_STAGE") == 3:
    dependent(out, (magnitude,0), zero, Ops.ADD,
              fp32_output=store.src[0].src[0].dtype is dtypes.float)
    return tuple(tasks)
  runtime_exponent_positive = positive_mask((exponent_half,0), zero)
  runtime_exponent_negative = positive_mask(zero, (exponent_half,0))
  magnitude_corrections = (
    (0.99853515625, runtime_exponent_positive, (0.1463623046875, 0.010040283203125, 0.214111328125)),
    (0.99853515625, runtime_exponent_negative, (0.40771484375,)),
    (1.0009765625, runtime_exponent_negative, (0.1875,)),
    (1.0029296875, runtime_exponent_negative, (0.007091522216796875,)),
    (1.001953125, runtime_exponent_negative, (0.0236358642578125,)),
    (0.9990234375, runtime_exponent_negative, (0.0027675628662109375,)),
    (0.998046875, runtime_exponent_negative, (0.0267486572265625, 0.00270843505859375)),
  )
  for factor, exponent_mask, magnitude_values in magnitude_corrections if tensor_fp16 else ():
    masks = [base_exact(number) for number in magnitude_values]
    group_mask = masks[0]
    for mask in masks[1:]:
      combined = alloc()
      tasks.append(_emit_where_stage(total, combined, group_mask, mask, Ops.MAX))
      group_mask = (combined,0)
    signed_group, factor_delta, selected_delta, corrected_magnitude = (alloc() for _ in range(4))
    tasks.extend((_emit_where_stage(total, signed_group, group_mask, exponent_mask, Ops.MUL),
                  _emit_where_stage(total, factor_delta, (signed_group,0), scalar(factor-1.0), Ops.MUL),
                  _emit_where_stage(total, selected_delta, one, (factor_delta,0), Ops.ADD)))
    dependent(corrected_magnitude, (magnitude,0), (selected_delta,0), Ops.MUL)
    magnitude = corrected_magnitude

  truncated = trunc_half(exponent_half)
  if truncated is None: return None
  half = alloc()
  dependent(half, (truncated,0), scalar(0.5), Ops.MUL)
  half_truncated = trunc_half(half)
  if half_truncated is None: return None
  doubled, remainder, negative_remainder, odd = (alloc() for _ in range(4))
  dependent(doubled, (half_truncated,0), scalar(2.0), Ops.MUL)
  dependent(remainder, (truncated,0), (doubled,0), Ops.SUB)
  dependent(negative_remainder, zero, (remainder,0), Ops.SUB)
  dependent(odd, (remainder,0), (negative_remainder,0), Ops.MAX)
  twice_odd, parity_sign = alloc(), alloc()
  dependent(twice_odd, (odd,0), scalar(2.0), Ops.MUL)
  dependent(parity_sign, one, (twice_odd,0), Ops.SUB)

  exponent_above_trunc = positive_mask((exponent_half,0), (truncated,0))
  trunc_above_exponent = positive_mask((truncated,0), (exponent_half,0))
  noninteger, negative_mask, invalid = alloc(), positive_mask(zero, (base_half,0))[0], alloc()
  dependent(noninteger, exponent_above_trunc, trunc_above_exponent, Ops.ADD)
  dependent(invalid, (negative_mask,0), (noninteger,0), Ops.MUL)
  sign_delta, selected_delta, effective_sign, signed = (alloc() for _ in range(4))
  dependent(sign_delta, (parity_sign,0), one, Ops.SUB)
  dependent(selected_delta, (negative_mask,0), (sign_delta,0), Ops.MUL)
  dependent(effective_sign, one, (selected_delta,0), Ops.ADD)
  dependent(signed, (magnitude,0), (effective_sign,0), Ops.MUL)
  exponent_negative = positive_mask(zero, (exponent_half,0))
  normal_result, zero_result, selected_result = (alloc() for _ in range(3))
  dependent(normal_result, (signed,0), base_nonzero, Ops.MUL)
  dependent(zero_result, (exponent_zero,0), (base_zero,0), Ops.MUL)
  dependent(selected_result, (normal_result,0), (zero_result,0), Ops.ADD)
  zero_negative, zero_numerator, zero_denom, zero_selected = (alloc() for _ in range(4))
  dependent(zero_negative, (base_zero,0), exponent_negative, Ops.MUL)
  dependent(zero_numerator, (selected_result,0), (zero_negative,0), Ops.ADD)
  dependent(zero_denom, one, (zero_negative,0), Ops.SUB)
  dependent(zero_selected, (zero_numerator,0), (zero_denom,0), Ops.FDIV)
  valid_denom, valid_factor = alloc(), alloc()
  dependent(valid_denom, one, (invalid,0), Ops.SUB)
  dependent(valid_factor, (valid_denom,0), (valid_denom,0), Ops.FDIV)
  dependent(out, (zero_selected,0), (valid_factor,0), Ops.MUL,
            fp32_output=store.src[0].src[0].dtype is dtypes.float)
  return tuple(tasks)

def _try_fractional_pow_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower fractional x**c through staged abs, LOG2, scale, and EXP2.

  tinygrad's outer negative-input WHERE contains a literal NaN.  Arithmetic
  mask selection would evaluate 0*NaN on every nonnegative lane, so generate
  the invalid-domain NaN with a final DPU 0/0 factor instead.
  """
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE or len(value.src) != 3: return None
  condition, invalid, magnitude_pow = _unwrap(value.src[0]), _unwrap(value.src[1]), _unwrap(value.src[2])
  if condition.op is not Ops.CMPLT or invalid.op is not Ops.CONST or not math.isnan(float(invalid.arg)): return None
  base, zero = (_unwrap(x) for x in condition.src)
  if base.dtype is not dtypes.half or zero.op is not Ops.CONST or float(zero.arg) != 0.0: return None
  reciprocal = base.op is Ops.RECIPROCAL and len(base.src) == 1 and _unwrap(base.src[0]).op is Ops.INDEX
  source = _unwrap(base.src[0]) if reciprocal else base
  if source.op is not Ops.INDEX or source.dtype is not dtypes.half: return None
  if magnitude_pow.op is not Ops.EXP2 or len(magnitude_pow.src) != 1: return None
  scaled_log = _unwrap(magnitude_pow.src[0])
  if scaled_log.op is not Ops.MUL or len(scaled_log.src) != 2: return None
  log2_val, exponent_u = (_unwrap(x) for x in scaled_log.src)
  if log2_val.op is not Ops.LOG2: log2_val, exponent_u = exponent_u, log2_val
  if log2_val.op is not Ops.LOG2 or exponent_u.op is not Ops.CONST: return None
  exponent = float(exponent_u.arg)
  if not math.isfinite(exponent) or exponent.is_integer(): return None
  absolute_expr = _unwrap(log2_val.src[0])
  if absolute_expr.op is not Ops.WHERE or len(absolute_expr.src) != 3 or \
     _unwrap(absolute_expr.src[0]) is not condition or _unwrap(absolute_expr.src[2]) is not base: return None
  source_slot = source.src[0].buf_uop.arg.slot

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int) -> UOp:
    out_index = store.src[0]
    return out_index.replace(dtype=dtypes.half,
      src=(out_index.src[0].param_like(slot).replace(dtype=dtypes.half), *out_index.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot), stage_value))})

  def extend(stage_tasks:tuple[RKSubTask, ...]|None) -> bool:
    nonlocal next_slot
    if stage_tasks is None: return False
    tasks.extend(stage_tasks)
    used_slots = [task.task.out_slot for task in stage_tasks] + \
      [reloc.globals_slot for task in stage_tasks for reloc in task.relocs
       if reloc.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return True

  base_slot = alloc() if reciprocal else source_slot
  if reciprocal:
    one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
    tasks.append(_emit_where_stage(total, base_slot, one, (source_slot, 0), Ops.FDIV))
  negative, absolute = alloc(), alloc()
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  tasks.extend((_emit_where_stage(total, negative, (base_slot, 0), negative_one, Ops.MUL),
                _emit_where_stage(total, absolute, (base_slot, 0), (negative, 0), Ops.MAX)))

  scaled_log_slot = alloc()
  staged_log = UOp(Ops.MUL, dtypes.half,
                   (UOp(Ops.LOG2, dtypes.half, (temp_index(absolute),)), UOp.const(dtypes.half, exponent)))
  if not extend(_try_log2_special_subtasks(stage_sink(staged_log, scaled_log_slot))): return None

  magnitude_slot = alloc()
  staged_exp = UOp(Ops.EXP2, dtypes.half, (temp_index(scaled_log_slot),))
  if not extend(_try_exp2_special_subtasks(stage_sink(staged_exp, magnitude_slot))): return None

  # source < 0 mask, followed by (1-mask)/(1-mask).  The factor is one for
  # the valid domain and NaN for negative inputs without ever selecting a
  # literal NaN through arithmetic masking.
  negative_diff, negative_mask = alloc(), alloc()
  invalid_denom_scratch, invalid_denom, invalid_factor_scratch, invalid_factor = (alloc() for _ in range(4))
  output_scratch = alloc()
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  tasks.extend((_emit_where_stage(total, negative_diff, (_ZERO_SLOT, 0), (base_slot, 0), Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, invalid_denom_scratch, one, (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, invalid_denom, one, (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, invalid_factor_scratch, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, invalid_factor, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV),
                _emit_where_stage(total, output_scratch, (magnitude_slot, 0), (invalid_factor, 0), Ops.MUL),
                _emit_where_stage(total, out, (magnitude_slot, 0), (invalid_factor, 0), Ops.MUL)))
  return tuple(tasks)

def _try_abs_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
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
  fp32_in = store.src[0].src[0].dtype is dtypes.float
  if fp32_in:
    # Preserve the fp32 residual limb. The old high-only path below rounds
    # abs(x-y) to fp16 and loses isclose-scale deltas such as 1e-6.
    value = _unwrap(store.src[1])
    sources = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float and
               u.src[0].op is Ops.PARAM and u.src[0].arg.slot == input_slot]
    store_aff = _affine_index(store.src[0].src[1])
    source = next((u for u in sources if _affine_index(u.src[1]) == store_aff), None)
    if source is None: return None
    next_slot, tasks = scratch, []
    def alloc() -> int:
      nonlocal next_slot
      ret, next_slot = next_slot, next_slot + 1
      return ret
    def fp32_view(tag:int) -> tuple[int,int]:
      slot = alloc()
      cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
      relocs = (RKReloc(0, slot, 0, 0, 0xFFFFFFFF), RKReloc(0, input_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, tag), slot, is_copy=True), relocs))
      return slot, 0
    def stage(a:tuple[int,int], b:tuple[int,int], op:Ops, compare=False) -> tuple[int,int]:
      slot = alloc()
      tasks.append(_emit_where_stage(total, slot, a, b, op, compare=compare))
      return slot, 0
    def dependent(a:tuple[int,int], b:tuple[int,int], op:Ops) -> tuple[int,int]:
      # Repeat the first consumption of comparison masks, matching WHERE and
      # nested-comparison stability on reset-separated RK3588 submissions.
      stage(a, b, op)
      return stage(a, b, op)
    zero = (_ZERO_SLOT, 0)
    one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
    two = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 2.0))[0])
    high, low = fp32_view(_HOST_FP32_HALF_LAYOUT), fp32_view(_HOST_FP32_RESIDUAL_LAYOUT)
    high_negative = stage(stage(zero, high, Ops.SUB), stage(zero, high, Ops.SUB), Ops.MAX, compare=True)
    high_positive = stage(stage(high, zero, Ops.SUB), stage(high, zero, Ops.SUB), Ops.MAX, compare=True)
    low_negative = stage(stage(zero, low, Ops.SUB), stage(zero, low, Ops.SUB), Ops.MAX, compare=True)
    high_nonzero = dependent(high_negative, high_positive, Ops.MAX)
    high_zero = dependent(one, high_nonzero, Ops.SUB)
    residual_negative = dependent(high_zero, low_negative, Ops.MUL)
    negative = dependent(high_negative, residual_negative, Ops.MAX)
    doubled_negative = dependent(negative, two, Ops.MUL)
    sign = dependent(one, doubled_negative, Ops.SUB)
    absolute_high = dependent(high, sign, Ops.MUL)
    absolute_low = dependent(low, sign, Ops.MUL)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, absolute_high[0], 0, 0, 0xFFFFFFFF),
              RKReloc(0, absolute_low[0], 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_COMBINE_LAYOUT),
                                       out_slot, is_copy=True), relocs))
    return tuple(tasks)
  # WIP reference: direct fp32 input conversion retains only the high fp16
  # limb. Keep this stable two-stage implementation for native fp16 abs.
  return (_emit_where_stage(total, scratch, (input_slot, 0), negative_one, Ops.MUL),
          _emit_where_stage(total, out_slot, (input_slot, 0), (scratch, 0), Ops.MAX))

def _try_softsign_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower x/(1+abs(x)) as four ordered DPU stages."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.FDIV or len(value.src) != 2: return None
  source, denominator = _unwrap(value.src[0]), _unwrap(value.src[1])
  if source.op is not Ops.INDEX or source.dtype is not dtypes.half or denominator.op is not Ops.ADD: return None
  one, magnitude = denominator.src
  if one.op is not Ops.CONST: one, magnitude = magnitude, one
  source_slot = source.src[0].buf_uop.arg.slot
  if one.op is not Ops.CONST or float(one.arg) != 1.0 or _try_abs(_unwrap(magnitude)) != source_slot: return None

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  negative, absolute, positive_denominator = (max(info.globals, default=-1) + offset for offset in range(1, 4))
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  positive_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  return (_emit_where_stage(total, negative, (source_slot, 0), negative_one, Ops.MUL),
          _emit_where_stage(total, absolute, (source_slot, 0), (negative, 0), Ops.MAX),
          _emit_where_stage(total, positive_denominator, (absolute, 0), positive_one, Ops.ADD),
          _emit_where_stage(total, info.outs[0], (source_slot, 0), (positive_denominator, 0), Ops.FDIV))

def _try_lerp_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower x + (y-x)*weight as four ordered DPU stages."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.ADD or len(value.src) != 2: return None
  source, product = value.src
  if _unwrap(source).op is not Ops.INDEX: source, product = product, source
  source = _unwrap(source)
  product = _unwrap(product)
  if source.op is not Ops.INDEX or source.dtype is not dtypes.half or product.op is not Ops.MUL: return None
  delta, weight = product.src
  if _unwrap(delta).op is not Ops.ADD: delta, weight = weight, delta
  delta = _unwrap(delta)
  if delta.op is not Ops.ADD or (weight_arg := _where_arg(weight)) is None: return None
  target, negative_source = delta.src
  if _unwrap(target).op is not Ops.INDEX: target, negative_source = negative_source, target
  target = _unwrap(target)
  negative_source = _unwrap(negative_source)
  if target.op is not Ops.INDEX or target.dtype is not dtypes.half or negative_source.op is not Ops.MUL: return None
  neg_input, neg_one = negative_source.src
  if neg_one.op is not Ops.CONST: neg_input, neg_one = neg_one, neg_input
  neg_input = _unwrap(neg_input)
  if neg_input is not source or neg_one.op is not Ops.CONST or float(neg_one.arg) != -1.0: return None

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  source_slot, target_slot = source.src[0].buf_uop.arg.slot, target.src[0].buf_uop.arg.slot
  negative, difference, scaled = (max(info.globals, default=-1) + offset for offset in range(1, 4))
  negative_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
  def broadcasts(*nodes:UOp) -> tuple[int, ...]:
    return tuple(node.src[0].buf_uop.arg.slot for node in nodes
                 if node.op is Ops.INDEX and int(node.src[0].src[0].arg) < total)
  weight_u = _unwrap(weight)
  weight_broadcast = broadcasts(weight_u) if weight_u.op is Ops.INDEX else ()
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if len(loops) != 1 or int(loops[0].src[0].arg) != total: return None
  output_axis = loops[0]
  device = store.src[0].src[0].device
  negative_param = UOp.param(negative, dtypes.half, (total,), device=device)
  negative_index = UOp(Ops.INDEX, dtypes.half, (negative_param, output_axis))
  packing_axis = UOp.range(3, 1000)
  first = UOp.const(packing_axis.dtype, 1)
  second = UOp.const(packing_axis.dtype, 2)
  packed_index = output_axis*3 + packing_axis
  a_param = UOp.param(difference, dtypes.half, (total*3,), device=device)
  a_out = UOp(Ops.INDEX, dtypes.half, (a_param, packed_index))
  a_value = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (packing_axis, second)), source, target))
  a_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(a_out, a_value)), output_axis, packing_axis)))
  a_tasks = _try_movement_host_subtasks(a_sink)
  if a_tasks is None: return None
  b_param = UOp.param(scaled, dtypes.half, (total*3,), device=device)
  b_out = UOp(Ops.INDEX, dtypes.half, (b_param, packed_index))
  b_value = UOp(Ops.WHERE, dtypes.half, (UOp(Ops.CMPLT, dtypes.bool, (packing_axis, first)),
                                        UOp.const(dtypes.half, 1.0),
                                        UOp(Ops.WHERE, dtypes.half,
                                            (UOp(Ops.CMPLT, dtypes.bool, (packing_axis, second)),
                                             negative_index, weight_u))))
  b_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(b_out, b_value)), output_axis, packing_axis)))
  b_tasks = _try_movement_host_subtasks(b_sink)
  if b_tasks is None: return None

  reduction_axis = UOp.range(3, 1001, AxisType.REDUCE)
  cmac_index = output_axis*3 + reduction_axis
  a_index = UOp(Ops.INDEX, dtypes.half, (a_param, cmac_index))
  b_index = UOp(Ops.INDEX, dtypes.half, (b_param, cmac_index))
  product = UOp(Ops.MUL, dtypes.half, (a_index, b_index))
  accumulator = UOp(Ops.REDUCE, dtypes.float,
                    (UOp(Ops.CAST, dtypes.float, (product,), arg=dtypes.float), reduction_axis), arg=(Ops.ADD, 0))
  stage_store = store.replace(src=(store.src[0], UOp(Ops.CAST, dtypes.half, (accumulator,), arg=dtypes.half)))
  stage_sink = UOp.sink(UOp(Ops.END, src=(stage_store, *loops)))
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac":
    if getenv("ROCKCHIP_DEBUG_LERP"): print("RK_LERP_CMAC_REJECT", stage_plan, stage_sink)
    return None
  tasks:list[RKSubTask] = [_emit_where_stage(total, negative, weight_arg, negative_one, Ops.MUL,
                                             broadcast_inputs=weight_broadcast), *a_tasks, *b_tasks]
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

  # WIP reference: the direct graph decomposition is structurally correct but
  # rounds after every DPU stage and misses official tolerance in 120/1575
  # lanes. Keep it for future single-task fused-DPU experiments.
  # return (_emit_where_stage(total, negative, (source_slot, 0), negative_one, Ops.MUL,
  #                           broadcast_inputs=broadcasts(source)),
  #         _emit_where_stage(total, difference, (target_slot, 0), (negative, 0), Ops.ADD,
  #                           broadcast_inputs=broadcasts(target)),
  #         _emit_where_stage(total, scaled, (difference, 0), weight_arg, Ops.MUL,
  #                           broadcast_inputs=weight_broadcast),
  #         _emit_where_stage(total, info.outs[0], (source_slot, 0), (scaled, 0), Ops.ADD,
  #                           broadcast_inputs=broadcasts(source)))

def _try_isclose_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize only tinygrad's full IEEE isclose predicate on the host.

  WIP reference: the native comparison decomposer below can execute this graph,
  including two-limb fp32 arithmetic, but its many reset-separated compare
  stages exhaust RK3588's driver reset budget across the 32-case edge matrix.
  Keep that path intact for future fused comparison tasks.
  """
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.bool or _reduce_node(sink) is not None: return None
  value = _unwrap(store.src[1])
  nodes = value.toposort()
  if value.op is not Ops.OR: return None
  float_constants = [float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype is dtypes.float]
  if not any(math.isinf(x) and x > 0 for x in float_constants) or \
     not any(math.isinf(x) and x < 0 for x in float_constants): return None
  # isclose contains isnan(x) as x!=x and abs(other) as a float WHERE whose
  # sign branches include -1 and +1. Require both so ordinary composite
  # comparisons cannot enter this host path.
  has_nan_check = any(u.op is Ops.CMPNE and len(u.src) == 2 and u.src[0] is u.src[1] and u.src[0].dtype is dtypes.float for u in nodes)
  has_abs_sign = any(u.op is Ops.WHERE and u.dtype is dtypes.float and
                     {-1.0, 1.0}.issubset({float(x.arg) for x in u.src[1:] if x.op is Ops.CONST}) for u in nodes)
  # A scalar `other` folds rtol*abs(other)+atol to one constant (1.001e-5
  # for other=1), while tensor/tensor keeps the small constant in a MUL.
  has_tolerance = any(math.isfinite(x) and 0.0 < abs(x) <= 0.011 for x in float_constants)
  if not has_nan_check or not has_abs_sign or not has_tolerance: return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

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

  materialized:dict[UOp, tuple[int,int]] = {}
  comparison_views:dict[int, tuple[int,int]] = {}
  def temp_index(slot:int, dtype) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  def materialize_data(u:UOp) -> tuple[int,int]|None:
    """Stage arithmetic comparison operands before lowering the boolean tree.

    This mirrors conv_grok's independent output/channel tiles: every arithmetic
    producer owns one explicit scratch window and the comparison only consumes
    direct buffers.  Keep direct INDEX/CONST operands on the old hot path.
    """
    nonlocal next_slot
    u = _unwrap(u)
    if u.dtype is dtypes.bool: return None
    if u in materialized: return materialized[u]
    slot = alloc()
    def lower_stage(value:UOp, allow_generic=False) -> tuple[RKSubTask, ...]|None:
      staged = sink.substitute({store:store.replace(src=(temp_index(slot, value.dtype), value))})
      stage_tasks = None
      if value.dtype is dtypes.float and value.op is Ops.ADD: stage_tasks = _try_fp32_add_subtasks(staged)
      elif value.dtype is dtypes.float and value.op is Ops.MUL: stage_tasks = _try_fp32_mul_subtasks(staged)
      if stage_tasks is None: stage_tasks = _try_abs_subtasks(staged)
      if stage_tasks is None and value.op is Ops.WHERE: stage_tasks = _try_where_subtasks(staged)
      if stage_tasks is None and allow_generic: stage_tasks = _try_elementwise_subtasks(staged)
      return stage_tasks

    # Algebraic simplification can shift the inner x<0 threshold while
    # retaining abs(x)'s exact x*WHERE(x!=0, WHERE(...,-1,1), 0) boundary.
    # Recover that boundary and apply abs to one cached direct fp32 operand.
    loose_abs_base = None
    if u.op is Ops.MUL:
      for candidate, sign in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
        sign_u = _unwrap(sign)
        if sign_u.op is not Ops.WHERE or len(sign_u.src) != 3: continue
        condition, signed, zero_branch = sign_u.src
        signed_u = _unwrap(signed)
        if _unwrap(candidate).dtype is not dtypes.float or condition.op is not Ops.CMPNE or \
           signed_u.op is not Ops.WHERE or _unwrap(signed_u.src[0]).op is not Ops.CMPLT: continue
        values = (_unwrap(signed_u.src[1]), _unwrap(signed_u.src[2]), _unwrap(zero_branch))
        if all(x.op is Ops.CONST for x in values) and {float(values[0].arg), float(values[1].arg)} == {-1.0, 1.0} and \
           float(values[2].arg) == 0.0:
          loose_abs_base = _unwrap(candidate)
          break
    stage_tasks = None
    if loose_abs_base is not None:
      if loose_abs_base.op in (Ops.INDEX, Ops.CONST): direct_base = loose_abs_base
      elif (base_arg := materialize_data(loose_abs_base)) is not None:
        slot = alloc()
        direct_base = temp_index(base_arg[0], loose_abs_base.dtype)
      else: direct_base = None
      if direct_base is not None:
        zero = UOp.const(direct_base.dtype, 0.0)
        nonzero = UOp(Ops.CMPNE, dtypes.bool, (direct_base, zero))
        negative = UOp(Ops.CMPLT, dtypes.bool, (direct_base, zero))
        sign = UOp(Ops.WHERE, direct_base.dtype, (nonzero, UOp(Ops.WHERE, direct_base.dtype,
                   (negative, UOp.const(direct_base.dtype, -1.0), UOp.const(direct_base.dtype, 1.0))), zero))
        stage_tasks = lower_stage(UOp(Ops.MUL, direct_base.dtype, (direct_base, sign)))
    if stage_tasks is None: stage_tasks = lower_stage(u)
    if stage_tasks is None:
      # Rebuild one arithmetic level from cached direct scratch inputs before
      # using the broad elementwise fallback. This avoids repeatedly lowering
      # shared isclose operands inside its abs/tolerance expression.
      replacements:dict[UOp,UOp] = {}
      for child in u.src:
        child_u = _unwrap(child)
        if child_u.dtype is dtypes.bool or child_u.op in (Ops.INDEX, Ops.CONST): continue
        if (child_arg := materialize_data(child_u)) is None: continue
        replacements[child] = temp_index(child_arg[0], child_u.dtype)
      rebuilt = u.substitute(replacements) if replacements else u
      # The first reserved slot predates every recursively emitted child.
      # Move the parent output above their complete scratch range so helpers
      # using max(ProgramInfo.globals)+1 cannot alias a live child temporary.
      slot = alloc()
      stage_tasks = lower_stage(rebuilt, allow_generic=True)
    if stage_tasks is None: return None
    tasks.extend(stage_tasks)
    used_slots = [st.task.out_slot for st in stage_tasks] + \
      [r.globals_slot for st in stage_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    materialized[u] = (slot, 0)
    return materialized[u]

  def data_arg(u:UOp) -> tuple[tuple[int,int], tuple[int,...], tuple[int,...], tuple[int,...], tuple[int,...]]|None:
    u = _unwrap(u)
    if (arg := _where_arg(u)) is None:
      if (arg := materialize_data(u)) is None: return None
      if u.dtype is dtypes.float:
        if arg[0] not in comparison_views:
          half_slot = alloc()
          cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
          relocs = (RKReloc(0, half_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, arg[0], 0, 0, 0xFFFFFFFF))
          tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_HALF_LAYOUT),
                                             half_slot, is_copy=True), relocs))
          comparison_views[arg[0]] = (half_slot, 0)
        arg = comparison_views[arg[0]]
      return arg, (), (), (), ()
    if u.op is Ops.CONST:
      if isinstance(u.arg, (float, np.floating)) and math.isinf(float(u.arg)):
        normalized = math.copysign(65504.0, float(u.arg))
        arg = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', normalized))[0])
      return arg, (), (), (), ()
    slot, source_n = arg[0], int(u.src[0].src[0].arg)
    return arg, ((slot,) if u.dtype is dtypes.bool else ()), \
      ((slot,) if u.dtype is dtypes.int else ()), ((slot,) if source_n < total else ()), ((slot,) if u.dtype is dtypes.float else ())

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
      bool_inputs, int32_inputs, broadcasts, fp32_inputs = \
        (tuple(dict.fromkeys(lhs_info[i] + rhs_info[i])) for i in range(1, 5))
      comparison_inputs = tuple(dict.fromkeys(x[0] for x in (lhs, rhs) if x[0] not in (_CONST_SLOT, _ZERO_SLOT)))
      diff, mask = alloc(), alloc()
      tasks.extend((_emit_where_stage(total, diff, rhs, lhs, Ops.SUB, bool_inputs=bool_inputs,
                                      int32_inputs=int32_inputs, broadcast_inputs=broadcasts,
                                      comparison_inputs=comparison_inputs, fp32_inputs=fp32_inputs),
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
            mask_arg, bool_inputs, int32_inputs, broadcasts, _ = logical_info
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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
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
  return _finalize_fp32_output(tasks, store)

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
  if source is None or source.dtype not in (dtypes.half, dtypes.float): return None
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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
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
  return _finalize_fp32_output(tasks, store)

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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
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

  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_sigmoid_local")
  local_plan = plan_rk(stage_sink(local_val, local_slot))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_low = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (source, UOp.const(dtypes.half, -2.0))))
  local_high = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.half, 2.0), source)))
  if local_low is None or local_high is None: return None
  local_outside, local_inside, broad_selected, local_selected, selected_lut = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, local_outside, local_low, local_high, Ops.MAX),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, broad_selected, (lut_slot, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_slot, 0), (local_inside, 0), Ops.MUL),
                _emit_where_stage(total, selected_lut, (broad_selected, 0), (local_selected, 0), Ops.ADD)))
  lut_slot = selected_lut

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
  return _finalize_fp32_output(tasks, store)

def _try_log2_special_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Normalize finite LOG2 inputs and preserve zero, infinity, and NaN semantics."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  val = _unwrap(store.src[1])
  output_scale, log2_val = 1.0, val
  if val.op is Ops.MUL:
    lhs, rhs = (_unwrap(x) for x in val.src)
    if lhs.op is Ops.CONST and rhs.op is Ops.LOG2: output_scale, log2_val = float(lhs.arg), rhs
    elif rhs.op is Ops.CONST and lhs.op is Ops.LOG2: output_scale, log2_val = float(rhs.arg), lhs
    else: return None
  if log2_val.op is not Ops.LOG2 or len(log2_val.src) != 1 or (source := _unwrap(log2_val.src[0])).op is not Ops.INDEX: return None
  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
    tasks.extend(cmp_tasks)
    used_slots = [st.task.out_slot for st in cmp_tasks] + \
      [r.globals_slot for st in cmp_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used_slots, default=-1) + 1)
    return (mask_slot, 0)

  source_arg = (source.src[0].buf_uop.arg.slot, 0)
  source_fp32 = (source_arg[0],) if source.dtype is dtypes.float else ()
  range_masks:list[tuple[int,int]] = []
  for threshold in (0.25, 0.0625, 0.015625, 0.00390625):
    diff, mask = alloc(), alloc()
    tasks.extend((_emit_where_stage(total, diff, scalar(threshold), source_arg, Ops.SUB, fp32_inputs=source_fp32),
                  _emit_where_stage(total, mask, (diff, 0), (diff, 0), Ops.MAX, compare=True)))
    range_masks.append((mask, 0))

  weighted:list[tuple[int,int]] = []
  for mask_arg, weight in zip(range_masks, (3.0, 12.0, 48.0, 192.0)):
    slot = alloc()
    tasks.append(_emit_where_stage(total, slot, mask_arg, scalar(weight), Ops.MUL))
    weighted.append((slot, 0))
  factor_lo, factor_hi, factor_delta, factor, normalized = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, factor_lo, weighted[0], weighted[1], Ops.ADD),
                _emit_where_stage(total, factor_hi, weighted[2], weighted[3], Ops.ADD),
                _emit_where_stage(total, factor_delta, (factor_lo, 0), (factor_hi, 0), Ops.ADD),
                _emit_where_stage(total, factor, (factor_delta, 0), one, Ops.ADD),
                _emit_where_stage(total, normalized, source_arg, (factor, 0), Ops.MUL, fp32_inputs=source_fp32)))

  count_lo, count_hi, count, offset = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, count_lo, range_masks[0], range_masks[1], Ops.ADD),
                _emit_where_stage(total, count_hi, range_masks[2], range_masks[3], Ops.ADD),
                _emit_where_stage(total, count, (count_lo, 0), (count_hi, 0), Ops.ADD),
                _emit_where_stage(total, offset, (count, 0), scalar(-2.0*output_scale), Ops.MUL)))

  bounded_low, negated, negated_bounded, bounded = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, bounded_low, (normalized, 0), scalar(0.25), Ops.MAX),
                _emit_where_stage(total, negated, (_ZERO_SLOT, 0), (bounded_low, 0), Ops.SUB),
                _emit_where_stage(total, negated_bounded, (negated, 0), scalar(-4.0), Ops.MAX),
                _emit_where_stage(total, bounded, (_ZERO_SLOT, 0), (negated_bounded, 0), Ops.SUB)))

  lut_slot = alloc()
  normalized_log2 = UOp(Ops.LOG2, dtypes.half, (temp_index(bounded),))
  normalized_val = normalized_log2 if output_scale == 1.0 else \
    UOp(Ops.MUL, dtypes.half, (normalized_log2, UOp.const(dtypes.half, output_scale)))
  lut_plan = plan_rk(stage_sink(normalized_val, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  centered, zoomed = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, centered, (bounded, 0), one, Ops.SUB),
                _emit_where_stage(total, zoomed, (centered, 0), scalar(12.5), Ops.MUL)))
  local_slot = alloc()
  local_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(zoomed),), arg=("rk_log2_local", output_scale))
  local_plan = plan_rk(stage_sink(local_val, local_slot))
  if isinstance(local_plan, str) or local_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(local_plan)
  tasks.append(RKSubTask(cmds, task, relocs))
  local_scaled_scratch, local_scaled = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, local_scaled_scratch, (local_slot, 0), scalar(0.25), Ops.MUL),
                _emit_where_stage(total, local_scaled, (local_slot, 0), scalar(0.25), Ops.MUL)))

  local_low_diff, local_low, local_high_diff, local_high = (alloc() for _ in range(4))
  local_outside_scratch, local_outside, local_inside_scratch, local_inside = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, local_low_diff, scalar(0.85), (bounded, 0), Ops.SUB),
                _emit_where_stage(total, local_low, (local_low_diff, 0), (local_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_high_diff, (bounded, 0), scalar(1.15), Ops.SUB),
                _emit_where_stage(total, local_high, (local_high_diff, 0), (local_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, local_outside_scratch, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_outside, (local_low, 0), (local_high, 0), Ops.MAX),
                _emit_where_stage(total, local_inside_scratch, one, (local_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_inside, one, (local_outside, 0), Ops.SUB)))
  near_low_diff, near_low, near_high_diff, near_high = (alloc() for _ in range(4))
  near_outside_scratch, near_outside, near_inside_scratch, near_inside = (alloc() for _ in range(4))
  local_mask_scratch, local_mask = alloc(), alloc()
  square, half_square, adjusted, polynomial = (alloc() for _ in range(4))
  tasks.extend((_emit_where_stage(total, near_low_diff, scalar(-0.02), (centered, 0), Ops.SUB),
                _emit_where_stage(total, near_low, (near_low_diff, 0), (near_low_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_high_diff, (centered, 0), scalar(0.02), Ops.SUB),
                _emit_where_stage(total, near_high, (near_high_diff, 0), (near_high_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, near_outside_scratch, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_outside, (near_low, 0), (near_high, 0), Ops.MAX),
                _emit_where_stage(total, near_inside_scratch, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, near_inside, one, (near_outside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask_scratch, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, local_mask, (local_inside, 0), (near_inside, 0), Ops.SUB),
                _emit_where_stage(total, square, (centered, 0), (centered, 0), Ops.MUL),
                _emit_where_stage(total, half_square, (square, 0), scalar(0.5), Ops.MUL),
                _emit_where_stage(total, adjusted, (centered, 0), (half_square, 0), Ops.SUB),
                _emit_where_stage(total, polynomial, (adjusted, 0), scalar(output_scale*math.log2(math.e)), Ops.MUL)))
  broad_selected_scratch, broad_selected, local_selected_scratch, local_selected = (alloc() for _ in range(4))
  linear_selected_scratch, linear_selected, lut_sum_scratch, lut_sum, mantissa = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, broad_selected_scratch, (lut_slot, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, broad_selected, (lut_slot, 0), (local_outside, 0), Ops.MUL),
                _emit_where_stage(total, local_selected_scratch, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, local_selected, (local_scaled, 0), (local_mask, 0), Ops.MUL),
                _emit_where_stage(total, linear_selected_scratch, (polynomial, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, linear_selected, (polynomial, 0), (near_inside, 0), Ops.MUL),
                _emit_where_stage(total, lut_sum_scratch, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, lut_sum, (broad_selected, 0), (local_selected, 0), Ops.ADD),
                _emit_where_stage(total, alloc(), (lut_sum, 0), (linear_selected, 0), Ops.ADD),
                _emit_where_stage(total, mantissa, (lut_sum, 0), (linear_selected, 0), Ops.ADD)))
  corrected = alloc()
  dependent(corrected, (mantissa, 0), (offset, 0), Ops.ADD)

  # Natural log of fp16 probabilities needs more output precision than the
  # normalized Q13/Q15 path provides after BCE's two weighted products.  Use
  # two direct refinement LUTs over [0.1, 0.5] and [0.5, 1), retaining the
  # general result outside that narrow domain.
  refined = corrected
  if source.dtype is dtypes.half and math.isclose(output_scale, math.log(2.0), rel_tol=0.0, abs_tol=1e-3):
    low_centered, low_input = alloc(), alloc()
    dependent(low_centered, (bounded, 0), scalar(0.375), Ops.SUB)
    dependent(low_input, (low_centered, 0), scalar(16.0), Ops.MUL)
    low_lut = alloc()
    low_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(low_input),), arg="rk_log_half_low")
    low_plan = plan_rk(stage_sink(low_val, low_lut))
    if isinstance(low_plan, str) or low_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(low_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    high_centered, high_input = alloc(), alloc()
    dependent(high_centered, (bounded, 0), scalar(0.75), Ops.SUB)
    dependent(high_input, (high_centered, 0), scalar(8.0), Ops.MUL)
    high_lut = alloc()
    high_val = UOp(Ops.CUSTOM, dtypes.half, (temp_index(high_input),), arg="rk_log_half_high")
    high_plan = plan_rk(stage_sink(high_val, high_lut))
    if isinstance(high_plan, str) or high_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(high_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    below = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (temp_index(bounded), UOp.const(dtypes.half, 0.2498779296875))))
    above = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.half, 0.99951171875), temp_index(bounded))))
    high_region = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.half, 0.5), temp_index(bounded))))
    if below is None or above is None or high_region is None: return None
    outside, valid, low_region, low_mask, high_mask = (alloc() for _ in range(5))
    tasks.extend((_emit_where_stage(total, outside, below, above, Ops.MAX),
                  _emit_where_stage(total, valid, one, (outside, 0), Ops.SUB),
                  _emit_where_stage(total, low_region, one, high_region, Ops.SUB),
                  _emit_where_stage(total, low_mask, (valid, 0), (low_region, 0), Ops.MUL),
                  _emit_where_stage(total, high_mask, (valid, 0), high_region, Ops.MUL)))
    low_selected, high_selected, direct_mantissa, direct, fallback_mask, fallback, refined = (alloc() for _ in range(7))
    tasks.extend((_emit_where_stage(total, low_selected, (low_lut, 0), (low_mask, 0), Ops.MUL),
                  _emit_where_stage(total, high_selected, (high_lut, 0), (high_mask, 0), Ops.MUL),
                  _emit_where_stage(total, direct_mantissa, (low_selected, 0), (high_selected, 0), Ops.ADD),
                  _emit_where_stage(total, direct, (direct_mantissa, 0), (offset, 0), Ops.ADD),
                  _emit_where_stage(total, fallback_mask, one, (valid, 0), Ops.SUB),
                  _emit_where_stage(total, fallback, (corrected, 0), (fallback_mask, 0), Ops.MUL),
                  _emit_where_stage(total, refined, (direct, 0), (fallback, 0), Ops.ADD)))

  hi = UOp.const(dtypes.half, 65472.0)
  positive_arg = comparison_mask(UOp(Ops.CMPLT, dtypes.bool, (hi, source)))
  if positive_arg is None: return None
  negative_diff, negative, positive_diff, source_positive = (alloc() for _ in range(4))
  nonzero_scratch, nonzero_slot = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, (_ZERO_SLOT, 0), source_arg, Ops.SUB, fp32_inputs=source_fp32),
                _emit_where_stage(total, negative, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, positive_diff, source_arg, (_ZERO_SLOT, 0), Ops.SUB, fp32_inputs=source_fp32),
                _emit_where_stage(total, source_positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                _emit_where_stage(total, nonzero_scratch, (negative, 0), (source_positive, 0), Ops.MAX),
                _emit_where_stage(total, nonzero_slot, (negative, 0), (source_positive, 0), Ops.MAX)))
  negative_arg, nonzero = (negative, 0), (nonzero_slot, 0)
  not_number = comparison_mask(UOp(Ops.CMPNE, dtypes.bool, (source, source)))
  if not_number is None: return None

  zero_result = alloc()
  dependent(zero_result, (refined, 0), nonzero, Ops.FDIV)
  positive_denom, finite = alloc(), alloc()
  dependent(positive_denom, one, positive_arg, Ops.SUB)
  dependent(finite, (zero_result, 0), (positive_denom, 0), Ops.FDIV)
  invalid, invalid_denom, invalid_factor = alloc(), alloc(), alloc()
  dependent(invalid, negative_arg, not_number, Ops.MAX)
  dependent(invalid_denom, one, (invalid, 0), Ops.SUB)
  dependent(invalid_factor, (invalid_denom, 0), (invalid_denom, 0), Ops.FDIV)
  dependent(out, (finite, 0), (invalid_factor, 0), Ops.MUL)
  return _finalize_fp32_output(tasks, store)

def _finalize_fp32_output(tasks:list[RKSubTask], store:UOp) -> tuple[RKSubTask, ...]:
  """Set fp32_output on the last task when the store's output PARAM is fp32."""
  if store.src[0].src[0].dtype is dtypes.float:
    last = tasks[-1]
    tasks[-1] = RKSubTask(last.cmds, replace(last.task, fp32_output=True), last.relocs)
  return tuple(tasks)

def _fix_cmp_fp32(cmp_tasks:tuple[RKSubTask, ...], source:UOp) -> tuple[RKSubTask, ...]:
  """Add fp32_inputs to comparison tasks that read from a fp32 source."""
  if source.src[0].dtype is not dtypes.float: return cmp_tasks
  src_slot = source.src[0].arg.slot
  return tuple(RKSubTask(st.cmds, replace(st.task, fp32_inputs=tuple(set(st.task.fp32_inputs+(src_slot,)))), st.relocs)
    if any(r.globals_slot == src_slot for r in st.relocs) else st for st in cmp_tasks)

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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
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
  return _finalize_fp32_output(tasks, store)

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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

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
    cmp_tasks = _fix_cmp_fp32((*cmp_tasks[:-1], RKSubTask(last.cmds, replace(last.task, bool_output=False), last.relocs)), source)
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
  return _finalize_fp32_output(tasks, store)

def _try_one_hot_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower one_hot's WHERE(index != class, 0, 1) with NPU comparisons.

  Host tasks only expand the compact input by byte-copying it and materialize
  the compile-time class coordinate.  Equality and int32 result generation
  remain DPU work.
  """
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.int: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE or len(value.src) != 3: return None
  condition = _unwrap(value.src[0])
  true, false = (_unwrap(x) for x in value.src[1:])
  if condition.op is not Ops.CMPNE or true.op is not Ops.CONST or false.op is not Ops.CONST or \
     int(true.arg) != 0 or int(false.arg) != 1: return None

  indexed, coordinate = (_unwrap(x) for x in condition.src)
  if indexed.op is not Ops.INDEX: indexed, coordinate = coordinate, indexed
  while coordinate.op is Ops.CAST: coordinate = coordinate.src[0]
  if indexed.op is not Ops.INDEX or indexed.dtype is not dtypes.int or coordinate.op is not Ops.RANGE: return None

  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if coordinate not in loops or any(u.src[0].op is not Ops.CONST for u in loops): return None
  extents = tuple(int(u.src[0].arg) for u in loops)
  total, info = prod(extents), ProgramInfo.from_sink(sink)
  if total != prod(_shape_of_store(sink)) or int(coordinate.src[0].arg) > 2048: return None
  class_axis = loops.index(coordinate)
  class_values:list[int] = []
  for linear in range(total):
    rem, coords = linear, [0] * len(extents)
    for axis in range(len(extents)-1, -1, -1): rem, coords[axis] = divmod(rem, extents[axis])
    class_values.append(coords[class_axis])

  next_slot = max(info.globals, default=-1) + 1
  expanded, classes = next_slot, next_slot+1
  positive_diff, positive_mask, negative_diff, negative_mask = range(next_slot+2, next_slot+6)
  neq_scratch, neq, equal_scratch = range(next_slot+6, next_slot+9)

  out_index = store.src[0]
  expanded_index = out_index.replace(src=(out_index.src[0].param_like(expanded).replace(dtype=dtypes.int), *out_index.src[1:]))
  movement_store = store.replace(src=(expanded_index, indexed))
  movement_tasks = _try_movement_host_subtasks(sink.substitute({store:movement_store}))
  if movement_tasks is None: return None

  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  class_layout = (total, _HOST_STATIC_INT_LAYOUT, *class_values)
  class_relocs = (RKReloc(0, classes, 0, 0, 0xFFFFFFFF),)
  class_task = RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", class_layout, classes, is_copy=True), class_relocs)
  int_inputs = (expanded, classes)
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  tasks = [*movement_tasks, class_task,
           _emit_where_stage(total, positive_diff, (classes, 0), (expanded, 0), Ops.SUB, int32_inputs=int_inputs),
           _emit_where_stage(total, positive_mask, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
           _emit_where_stage(total, negative_diff, (expanded, 0), (classes, 0), Ops.SUB, int32_inputs=int_inputs),
           _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
           # Repeat consumers at each reset-separated dependency boundary.
           _emit_where_stage(total, neq_scratch, (positive_mask, 0), (negative_mask, 0), Ops.MAX),
           _emit_where_stage(total, neq, (positive_mask, 0), (negative_mask, 0), Ops.MAX),
           _emit_where_stage(total, equal_scratch, one, (neq, 0), Ops.SUB),
           _emit_where_stage(total, info.outs[0], one, (neq, 0), Ops.SUB, int32_output=True)]
  # WIP reference: native-int SUB is exact, but compare=True does not produce
  # valid masks for native-int atoms.  The attempted pairs were:
  #   _emit_where_stage(..., Ops.SUB, native_int32_input=True, native_int32_output=True)
  #   _emit_where_stage(..., Ops.MAX, compare=True, native_int32_input=True)
  # Keep one_hot on the proven int32-to-fp16 ABI while its class domain is
  # exactly representable; a byte-limb equality path is needed for full int32.
  return tuple(tasks)

def _try_where_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  store = _store_node(sink)
  if store is None or (val := _unwrap(store.src[1])).op is not Ops.WHERE: return None
  total, info = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink)
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  materialized:dict[UOp, int] = {}
  slot_dtypes:dict[int, DType] = {}
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])

  def alloc(count:int=1) -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + count
    return ret

  def lower_arg(u:UOp) -> tuple[int,int]|None:
    if (arg := _where_arg(u)) is not None:
      uw = _unwrap(u)
      if uw.op is Ops.INDEX: slot_dtypes[arg[0]] = uw.dtype
      return arg
    u = _unwrap(u)
    if u not in materialized:
      materialized[u] = alloc()
      slot_dtypes[materialized[u]] = dtypes.half
      if u.op is Ops.WHERE:
        if not lower(u, materialized[u], False): return None
      elif u.op in _DPU_EW_CFGS and len(u.src) == 2:
        a, b = (lower_arg(x) for x in u.src)
        if a is None or b is None: return None
        fp32_in = tuple(s[0] for s, src_u in zip((a, b), u.src)
                        if _unwrap(src_u).op is Ops.INDEX and _unwrap(src_u).dtype is dtypes.float)
        tasks.append(_emit_where_stage(total, materialized[u], a, b, u.op, fp32_inputs=fp32_in))
      elif u.op is Ops.TRUNC and (source := _where_arg(u.src[0])) is not None:
        tasks.append(_emit_trunc_stage(total, materialized[u], source))
      elif u.op in _LUT_OPS:
        inner = _unwrap(u.src[0])
        scale_const, index_uop = None, None
        if inner.op is Ops.MUL:
          a, b = (_unwrap(s) for s in inner.src)
          if a.op is Ops.INDEX and b.op is Ops.CONST: scale_const, index_uop = b, inner.src[0]
          elif a.op is Ops.CONST and b.op is Ops.INDEX: scale_const, index_uop = a, inner.src[1]
        src_arg = lower_arg(index_uop if scale_const is not None else u.src[0])
        if src_arg is None: return None
        input_dtype = slot_dtypes.get(src_arg[0], dtypes.half)
        out_idx_base = store.src[0]
        input_idx = out_idx_base.replace(dtype=input_dtype,
            src=(out_idx_base.src[0].param_like(src_arg[0]).replace(dtype=input_dtype), *out_idx_base.src[1:]))
        temp_out = out_idx_base.replace(dtype=dtypes.half,
            src=(out_idx_base.src[0].param_like(materialized[u]).replace(dtype=dtypes.half), *out_idx_base.src[1:]))
        lut_input = UOp(Ops.MUL, input_dtype, (input_idx, scale_const)) if scale_const is not None else input_idx
        stage_val = u.replace(src=(lut_input,), dtype=dtypes.half)
        stage_sink = sink.substitute({store: store.replace(src=(temp_out, stage_val))})
        for special in (_try_exp_correction_subtasks, _try_sigmoid_special_subtasks, _try_exp2_special_subtasks,
                        _try_log2_special_subtasks, _try_rsqrt_special_subtasks, _try_sqrt_special_subtasks):
          if (special_tasks := special(stage_sink)) is not None:
            tasks.extend(special_tasks)
            break
        else:
          plan = plan_rk(stage_sink)
          if isinstance(plan, str) or plan.kind != "dpu_lut": return None
          cmds, task, relocs = emit_rk(plan)
          tasks.append(RKSubTask(cmds, task, relocs))
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
      # WIP reference: globally repeating the difference here avoids a
      # post-CMAC comparison timeout, but changes logits-none rounding in
      # 37/320 lanes. The BCE-only warm-up is applied by _try_bce_subtasks.
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
    if cond.op is Ops.CMPNE:
      lhs, rhs = (lower_arg(x) for x in cond.src)
      if lhs is None or rhs is None: return False
      neg_one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -1.0))[0])
      pos_mask, neg, neg_mask = alloc(), alloc(), alloc()
      tasks.extend((_emit_where_stage(total, t0, lhs, rhs, Ops.SUB),
                    _emit_where_stage(total, pos_mask, (t0,0), (t0,0), Ops.MAX, compare=True),
                    _emit_where_stage(total, neg, (t0,0), neg_one, Ops.MUL),
                    _emit_where_stage(total, neg_mask, (neg,0), (neg,0), Ops.MAX, compare=True),
                    _emit_where_stage(total, mask, (pos_mask,0), (neg_mask,0), Ops.MAX)))
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

  if not lower(val, out, True): return None
  fp32_slots = {u.arg.slot for u in sink.toposort() if u.op is Ops.PARAM and u.dtype is dtypes.float}
  if fp32_slots:
    for i, st in enumerate(tasks):
      need = tuple(s for s in fp32_slots if s != st.task.out_slot and any(r.globals_slot == s for r in st.relocs) and s not in st.task.fp32_inputs)
      if need: tasks[i] = RKSubTask(st.cmds, replace(st.task, fp32_inputs=tuple(set(st.task.fp32_inputs+need))), st.relocs)
  return _finalize_fp32_output(tasks, store)

def _try_bce_logits_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower BCE-with-logits as (1-y)*x + softplus(-x) with a fitted LUT."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.ADD: return None
  nodes = value.toposort()
  if sum(u.op is Ops.LOG2 for u in nodes) != 2 or sum(u.op is Ops.EXP2 for u in nodes) != 4 or \
     any(u.op is Ops.RECIPROCAL for u in nodes): return None
  indexes = [u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.half]
  source = next((u for u in indexes if any(v.op is Ops.MAX and any(w is u for w in v.toposort()) for v in nodes)), None)
  if source is None: return None
  source_slot = source.src[0].buf_uop.arg.slot
  other_slots = {u.src[0].buf_uop.arg.slot for u in indexes if u.src[0].buf_uop.arg.slot != source_slot}
  if len(other_slots) != 1: return None
  target_slot = next(iter(other_slots))
  target_candidates = [u for u in nodes if u.op is Ops.WHERE and
                       any(v.op is Ops.INDEX and v.src[0].buf_uop.arg.slot == target_slot for v in u.toposort()) and
                       {float(v.arg) for v in u.toposort() if v.op is Ops.CONST and isinstance(v.arg, (float, np.floating))} >= {0.0, 1.0}]
  if not target_candidates: return None
  target = max(target_candidates, key=lambda u:len(u.toposort()))

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  three = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 3.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_value))})

  def extend(stage_tasks:tuple[RKSubTask, ...]) -> None:
    nonlocal next_slot
    tasks.extend(stage_tasks)
    used = [st.task.out_slot for st in stage_tasks] + \
      [r.globals_slot for st in stage_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used, default=-1) + 1)

  clipped_target = alloc()
  if (target_tasks := _try_where_subtasks(stage_sink(target, clipped_target))) is None: return None
  target_used = [st.task.out_slot for st in target_tasks] + \
    [r.globals_slot for st in target_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(target_used, default=-1) + 1)
  first, warm_slot = target_tasks[0], alloc()
  warm_relocs = tuple(replace(r, globals_slot=warm_slot) if r.globals_slot == first.task.out_slot else r for r in first.relocs)
  warm_task = RKSubTask(first.cmds, replace(first.task, out_slot=warm_slot), warm_relocs)
  extend((warm_task, *target_tasks))

  lut_slot = alloc()
  lut_value = UOp(Ops.CUSTOM, dtypes.half, (source,), arg="rk_bce_logits")
  lut_plan = plan_rk(stage_sink(lut_value, lut_slot))
  if isinstance(lut_plan, str) or lut_plan.kind != "dpu_lut": return None
  cmds, task, relocs = emit_rk(lut_plan)
  tasks.append(RKSubTask(cmds, task, relocs))

  negative_diff, negative_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, (_ZERO_SLOT, 0), (source_slot, 0), Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True)))
  scaled_negative, scale, loss, inverse_target, weighted_source = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, scaled_negative, (negative_mask, 0), three, Ops.MUL),
                _emit_where_stage(total, scale, one, (scaled_negative, 0), Ops.ADD),
                _emit_where_stage(total, loss, (lut_slot, 0), (scale, 0), Ops.MUL),
                _emit_where_stage(total, inverse_target, one, (clipped_target, 0), Ops.SUB),
                _emit_where_stage(total, weighted_source, (inverse_target, 0), (source_slot, 0), Ops.MUL),
                _emit_where_stage(total, out, (weighted_source, 0), (loss, 0), Ops.ADD)))
  return tuple(tasks)

def _try_bce_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower unreduced BCE(sigmoid(x), clip(y, 0, 1)) through endpoint-loss LUTs.

  Reconstructing BCE as (1-y)*BCE(x,0) + y*BCE(x,1) retains PyTorch's fp16
  rounding boundary while every data-dependent operation remains an NPU task.
  """
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.half: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.ADD: return None
  nodes = value.toposort()
  if sum(u.op is Ops.LOG2 for u in nodes) != 2 or sum(u.op is Ops.EXP2 for u in nodes) != 1 or \
     sum(u.op is Ops.RECIPROCAL for u in nodes) != 1: return None
  sigmoid = next((u for u in nodes if u.op is Ops.RECIPROCAL and _try_sigmoid(u) is not None), None)
  if sigmoid is None or (source_slot := _try_sigmoid(sigmoid)) is None: return None
  indexes = [u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.half]
  source = next((u for u in indexes if u.src[0].buf_uop.arg.slot == source_slot), None)
  other_slots = {u.src[0].buf_uop.arg.slot for u in indexes if u.src[0].buf_uop.arg.slot != source_slot}
  if source is None or len(other_slots) != 1: return None
  target_slot = next(iter(other_slots))
  target_candidates = [u for u in nodes if u.op is Ops.WHERE and
                       any(v.op is Ops.INDEX and v.src[0].buf_uop.arg.slot == target_slot for v in u.toposort()) and
                       {float(v.arg) for v in u.toposort() if v.op is Ops.CONST and isinstance(v.arg, (float, np.floating))} >= {0.0, 1.0}]
  if not target_candidates: return None
  target = max(target_candidates, key=lambda u:len(u.toposort()))

  info, total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  out, next_slot = info.outs[0], max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  three = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 3.0))[0])

  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot + 1
    return ret

  def temp_index(slot:int, dtype=dtypes.half) -> UOp:
    out_idx = store.src[0]
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  def stage_sink(stage_value:UOp, out_slot:int, dtype=dtypes.half) -> UOp:
    return sink.substitute({store:store.replace(src=(temp_index(out_slot, dtype), stage_value))})

  def extend(stage_tasks:tuple[RKSubTask, ...]) -> None:
    nonlocal next_slot
    tasks.extend(stage_tasks)
    used = [st.task.out_slot for st in stage_tasks] + \
      [r.globals_slot for st in stage_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    next_slot = max(next_slot, max(used, default=-1) + 1)

  # Materialize the exact nested clamp graph before consuming the target in
  # the endpoint weighting products.
  clipped_target = alloc()
  if (target_tasks := _try_where_subtasks(stage_sink(target, clipped_target))) is None: return None
  # A comparison immediately following four CMAC-ended reduction programs can
  # wedge even though one cold DPU difference completed. Repeat BCE's first
  # difference stage before its reset-separated comparison without perturbing
  # the generic WHERE lowering used by logits.
  target_used = [st.task.out_slot for st in target_tasks] + \
    [r.globals_slot for st in target_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
  next_slot = max(next_slot, max(target_used, default=-1) + 1)
  first, warm_slot = target_tasks[0], alloc()
  warm_relocs = tuple(replace(r, globals_slot=warm_slot) if r.globals_slot == first.task.out_slot else r for r in first.relocs)
  warm_task = RKSubTask(first.cmds, replace(first.task, out_slot=warm_slot), warm_relocs)
  extend((warm_task, *target_tasks))

  endpoint_zero, endpoint_one = alloc(), alloc()
  for endpoint_slot, marker in ((endpoint_zero, "rk_bce_zero"), (endpoint_one, "rk_bce_one")):
    endpoint_value = UOp(Ops.CUSTOM, dtypes.half, (source,), arg=marker)
    endpoint_plan = plan_rk(stage_sink(endpoint_value, endpoint_slot))
    if isinstance(endpoint_plan, str) or endpoint_plan.kind != "dpu_lut": return None
    cmds, task, relocs = emit_rk(endpoint_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  # The large-loss half of each LUT is stored at one-quarter magnitude so it
  # can keep Q15 precision. Restore it with complementary x<0 / x>=0 masks.
  negative_diff, negative_mask = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, negative_diff, (_ZERO_SLOT, 0), (source_slot, 0), Ops.SUB),
                _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True)))
  nonnegative_mask, scaled_negative, scale_zero, scaled_positive, scale_one = (alloc() for _ in range(5))
  tasks.extend((_emit_where_stage(total, nonnegative_mask, one, (negative_mask, 0), Ops.SUB),
                _emit_where_stage(total, scaled_positive, (nonnegative_mask, 0), three, Ops.MUL),
                _emit_where_stage(total, scale_zero, one, (scaled_positive, 0), Ops.ADD),
                _emit_where_stage(total, scaled_negative, (negative_mask, 0), three, Ops.MUL),
                _emit_where_stage(total, scale_one, one, (scaled_negative, 0), Ops.ADD)))
  loss_zero, loss_one = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, loss_zero, (endpoint_zero, 0), (scale_zero, 0), Ops.MUL),
                _emit_where_stage(total, loss_one, (endpoint_one, 0), (scale_one, 0), Ops.MUL)))

  inverse_target, term_zero, term_one = alloc(), alloc(), alloc()
  tasks.extend((_emit_where_stage(total, inverse_target, one, (clipped_target, 0), Ops.SUB),
                _emit_where_stage(total, term_zero, (inverse_target, 0), (loss_zero, 0), Ops.MUL),
                _emit_where_stage(total, term_one, (clipped_target, 0), (loss_one, 0), Ops.MUL),
                _emit_where_stage(total, out, (term_zero, 0), (term_one, 0), Ops.ADD)))
  return tuple(tasks)

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
    return out_idx.replace(dtype=dtype, src=(out_idx.src[0].param_like(slot).replace(dtype=dtype), *out_idx.src[1:]))

  def make_stage_sink(stage_val:UOp, out_slot:int) -> tuple[UOp, UOp]:
    out_idx = temp_index(out_slot, stage_val.dtype)
    return sink.substitute({store:store.replace(src=(out_idx, stage_val))}), out_idx

  def emit_stage(stage_val:UOp, out_slot:int) -> UOp|None:
    nonlocal next_slot
    stage_sink, out_idx = make_stage_sink(stage_val, out_slot)
    for special in (_try_logsigmoid_subtasks, _try_softplus_subtasks, _try_exp_correction_subtasks, _try_sigmoid_special_subtasks,
                    _try_exp2_special_subtasks, _try_log2_special_subtasks, _try_rsqrt_special_subtasks, _try_sqrt_special_subtasks):
      if (special_tasks := special(stage_sink)) is None: continue
      tasks.extend(special_tasks)
      used_slots = [st.task.out_slot for st in special_tasks] + \
        [r.globals_slot for st in special_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
      next_slot = max(next_slot, max(used_slots, default=-1) + 1)
      return out_idx
    if (broadcast_tasks := _try_broadcast_subtasks(stage_sink)) is not None:
      tasks.extend(broadcast_tasks)
      used_slots = [st.task.out_slot for st in broadcast_tasks] + \
        [r.globals_slot for st in broadcast_tasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
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
        if (movement_tasks := _try_movement_host_subtasks(stage_sink)) is not None:
          tasks.extend(movement_tasks)
          used_slots = [r.globals_slot for st in movement_tasks for r in st.relocs
                        if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
        elif (where_tasks := _try_where_subtasks(stage_sink)) is not None:
          tasks.extend(where_tasks)
          used_slots = [r.globals_slot for st in where_tasks for r in st.relocs
                        if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
        else: return None
        used_slots = [r.globals_slot for st in tasks for r in st.relocs
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
    # Reassociation can turn weight*(LOG2(x)*ln2) into
    # (LOG2(x)*weight)*ln2.  Recover the semantic natural-log boundary before
    # materializing the remaining fp16 products, including BCE's outer sign.
    if u.op is Ops.MUL:
      factors:list[UOp] = []
      def flatten_mul(node:UOp) -> None:
        node = _unwrap(node)
        if node.op is Ops.MUL:
          flatten_mul(node.src[0])
          flatten_mul(node.src[1])
        else: factors.append(node)
      flatten_mul(u)
      logarithm = next((factor for factor in factors if factor.op is Ops.LOG2), None)
      log_scale = next((factor for factor in factors if factor.op is Ops.CONST and
                        math.isclose(abs(float(factor.arg)), math.log(2.0), rel_tol=0.0, abs_tol=1e-3)), None)
      if logarithm is not None and log_scale is not None:
        if (log_source := lower_arg(logarithm.src[0])) is None: return None
        remaining = list(factors)
        remaining.remove(logarithm)
        remaining.remove(log_scale)
        if float(log_scale.arg) < 0: remaining.append(UOp.const(u.dtype, -1.0))
        scaled_log = UOp(Ops.MUL, u.dtype, (logarithm.replace(src=(log_source,)), UOp.const(u.dtype, math.log(2.0))))
        log_slot = out_slot if not remaining else alloc()
        if (log_idx := emit_stage(scaled_log, log_slot)) is not None:
          if not remaining: return log_idx
          product = log_idx
          for factor in remaining: product = UOp(Ops.MUL, u.dtype, (product, factor))
          return lower(product, out_slot)
    # Preserve MUL(LOG2(nested), scale) as one scaled-log special after
    # materializing its source.  Lowering bare LOG2 first loses the natural-log
    # output scale and bypasses the probability-range refinement LUTs.
    if u.op is Ops.MUL:
      for log_pos, scale_pos in ((0, 1), (1, 0)):
        logarithm, scale = _unwrap(u.src[log_pos]), u.src[scale_pos]
        if logarithm.op is not Ops.LOG2 or scale.op is not Ops.CONST: continue
        if (log_source := lower_arg(logarithm.src[0])) is None: return None
        rebuilt = list(u.src)
        rebuilt[log_pos] = logarithm.replace(src=(log_source,))
        if (idx := emit_stage(u.replace(src=tuple(rebuilt)), out_slot)) is not None: return idx
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
               if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP" and u.src[0].op is Ops.CONST}
    store_aff = _affine_index(store.src[0].src[1])
    axes = sorted((k for k in store_aff[0] if k in extents), key=lambda x: store_aff[0][x], reverse=True) if store_aff is not None else sorted(extents)
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
                                               const_val=float(vu.arg) if is_fill else 1.0,
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
  elif lut_op is _LUT_EXP2_SCALE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_exp2_scale_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_EXP2_RESIDUAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_exp2_residual_lut()
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
  elif lut_op is _LUT_CELU:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_celu_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_CELU_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_celu_local_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ELU:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_elu_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ELU_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_elu_local_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ERF:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_erf_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ERF_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_erf_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_GELU:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_gelu_lut(bool(input_scale))
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_GELU_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_gelu_local_lut(bool(input_scale))
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_QUICK_GELU:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_quick_gelu_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_QUICK_GELU_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_quick_gelu_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_TANH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_tanh_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_TANH_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_tanh_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_LOG2_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_log2_local_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op in (_LUT_LOG_HALF_LOW, _LUT_LOG_HALF_HIGH):
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_log_half_lut(lut_op is _LUT_LOG_HALF_HIGH)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_LOGSIGMOID_CORRECTION:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_logsigmoid_correction_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_LOGSIGMOID_TAIL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_logsigmoid_tail_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_SOFTPLUS_TAIL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_softplus_tail_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_SOFTPLUS_WIDE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_softplus_wide_lut(input_scale)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_MISH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_mish_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_MISH_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_mish_local_lut()
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
  elif lut_op is _LUT_SIN_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sin_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_COS:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_cos_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_COS_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_cos_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ASIN:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asin_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ASIN_DETAIL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asin_detail_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ASIN_DERIVATIVE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asin_derivative_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ACOS:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_acos_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ACOS_ENDPOINT:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_acos_endpoint_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ACOS_FINE_ENDPOINT:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_acos_fine_endpoint_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ATAN:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_atan_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ATAN_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_atan_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ATANH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_atanh_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ATANH_DETAIL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_atanh_detail_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ASINH_CORE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asinh_acosh_core_lut(False)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ASINH_RANGE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asinh_acosh_range_lut(False)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ACOSH_CORE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asinh_acosh_core_lut(True)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_ACOSH_RANGE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_asinh_acosh_range_lut(True)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW8:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow8_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW8_CORRECTION:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow8_correction_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW8_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow8_high_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW55:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow55_lut(False)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW55_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow55_lut(True)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_NEG55_LOW:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_neg55_lut(False)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_NEG55_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_neg55_lut(True)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE55_LOW:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base55_lut(False)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE55_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base55_lut(True)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE8_LOW:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base8_lut(1)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE8_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base8_lut(2)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE8_FAR_LOW:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base8_lut(0)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE8_FAR_HIGH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base8_lut(3)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_POW_BASE07:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base07_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op in (_LUT_POW_BASE2_LOW, _LUT_POW_BASE2_HIGH):
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_pow_base2_lut(lut_op is _LUT_POW_BASE2_HIGH)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_TAN_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_tan_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_TAN_WIDE:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_tan_wide_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_SINH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sinh_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_SINH_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sinh_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_COSH:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_cosh_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
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
  elif lut_op is _LUT_SIGMOID_LOCAL:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_sigmoid_local_lut()
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op in (_LUT_BCE_ZERO, _LUT_BCE_ONE):
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_bce_endpoint_lut(lut_op is _LUT_BCE_ONE)
    lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
  elif lut_op is _LUT_BCE_LOGITS:
    lut, bn_mul_operand, output_scale, index_scale, minus_exp = _build_bce_logits_lut()
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
  elif lut_op in (_LUT_EXP_CORRECTION, _LUT_EXP2_SCALE, _LUT_EXP2_RESIDUAL, _LUT_HARDSWISH, _LUT_HARDSWISH_CORRECTION,
                  _LUT_CELU, _LUT_CELU_LOCAL,
                  _LUT_QUICK_GELU, _LUT_QUICK_GELU_LOCAL, _LUT_TANH, _LUT_TANH_LOCAL, _LUT_LOG2_LOCAL,
                  _LUT_LOG_HALF_LOW, _LUT_LOG_HALF_HIGH, _LUT_SIGMOID_LOCAL, _LUT_BCE_ZERO, _LUT_BCE_ONE, _LUT_BCE_LOGITS,
                  _LUT_LOGSIGMOID_CORRECTION, _LUT_LOGSIGMOID_TAIL, _LUT_SOFTPLUS_TAIL, _LUT_SOFTPLUS_WIDE,
                  _LUT_MISH, _LUT_MISH_LOCAL, _LUT_ELU, _LUT_ELU_LOCAL, _LUT_ERF, _LUT_ERF_LOCAL, _LUT_GELU, _LUT_GELU_LOCAL,
                  _LUT_SIN_LOCAL, _LUT_COS, _LUT_COS_LOCAL, _LUT_TAN_LOCAL, _LUT_TAN_WIDE, _LUT_SINH, _LUT_SINH_LOCAL, _LUT_COSH):
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
  if plan.cmac_materialization:
    M, N, K, a_slot, b_slot = plan.cmac_materialization[:5]
  elif is_sum:
    sum_info = _try_sum(sink, reduce)
    assert sum_info is not None, "cmac: sum classification failed"
    input_slot, M_sum, N_sum, _, cv = sum_info
    a_slot, b_slot = (_CONST_SLOT, input_slot) if M_sum == 1 else (input_slot, _CONST_SLOT)
  else:
    a_idx_node, b_idx_node = _unwrap(body.src[0]), _unwrap(body.src[1])
    a_slot, b_slot = a_idx_node.src[0].buf_uop.arg.slot, b_idx_node.src[0].buf_uop.arg.slot
  out_shape = _shape_of_store(sink)
  if plan.cmac_materialization: pass
  elif is_sum: M, N = M_sum, N_sum
  elif len(out_shape) == 2: M, N = int(out_shape[0]), int(out_shape[1])
  else:  # GEMV: vector is A (M=1) or B (N=1)
    M, N = (1, int(out_shape[0])) if _is_1d_index(a_idx_node.src[1], "REDUCE") else (int(out_shape[0]), 1)  # type: ignore[union-attr]
  K = K if plan.cmac_materialization else _reduce_extent(reduce)
  if K < 0: raise RuntimeError("cmac: K must be compile-time constant")
  if not is_sum and not plan.cmac_materialization:
    # Detect transposed pattern (1x1 conv): A has REDUCE outer, B has LOOP outer
    if len(out_shape) == 2 and _is_2d_index(a_idx_node.src[1], "REDUCE", "LOOP", N) and \
       _is_2d_index(b_idx_node.src[1], "LOOP", "REDUCE", K):
      a_slot, b_slot = b_slot, a_slot  # swap: hardware expects A=(M,K), B=(K,N)
  # NPU geometry constants from gemm.py
  CBUF_BANK_SIZE = 256 * 128  # 32 KiB
  RK_CBUF_BANKS = 12
  MIN_CHANNEL_TILE = 32
  RK_LINE_STRIDE_GROUP_CAP = 13
  # Keep one materialized K tile small enough to reserve CBUF banks for both
  # features and its 32-channel weight atom. Runtime accumulates raw fp32 CACC
  # partials before the single output conversion.
  tile_k = min(K, 4096) if plan.cmac_materialization else K
  tile_n = min(N, MIN_CHANNEL_TILE) if plan.cmac_materialization else N
  aligned_k = max(MIN_CHANNEL_TILE, round_up(tile_k, MIN_CHANNEL_TILE))
  align_out = max(MIN_CHANNEL_TILE, round_up(tile_n, MIN_CHANNEL_TILE))
  align_in = max(aligned_k, align_out)
  eff_k = align_in if align_in != aligned_k else tile_k
  input_row_bytes = align_in * 2
  weight_banks = max(1, ceildiv(input_row_bytes*align_out, CBUF_BANK_SIZE))
  materialized_data_banks = max(1, RK_CBUF_BANKS-weight_banks)
  tile_m = min(M, max(1, min(2048, materialized_data_banks*CBUF_BANK_SIZE//input_row_bytes))) \
    if plan.cmac_materialization else M
  # feature grains and line stride from gemm.py
  even_rows_per_two_banks = (ceildiv(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
  feature_grains = max(80, even_rows_per_two_banks)
  line_stride = 4 * min(ceildiv(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
  notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
  required_data_banks = max(1, ceildiv(tile_m * input_row_bytes, CBUF_BANK_SIZE))
  if not plan.cmac_materialization and required_data_banks >= RK_CBUF_BANKS:
    raise RuntimeError("RKPLAN_REJECT:cmac_exceeds_cbuf")
  data_banks = min(RK_CBUF_BANKS-1, required_data_banks)
  wt_banks = RK_CBUF_BANKS - data_banks
  if input_row_bytes*align_out > wt_banks*CBUF_BANK_SIZE:
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
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE0, (1<<16)|tile_m)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE1, ((align_in-1)<<16)|align_in)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE2, 1)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DATA_SIZE3, tile_m)
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
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_FC_DATA_SIZE0, (1<<16)|tile_m)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_FC_DATA_SIZE1, align_in)
  # 25. CNA DCOMP_ADDR0 (relocated — weight B, NO 0x4000 offset)
  emitter_emit(cmds, _T_CNA, rk.REG_CNA_DCOMP_ADDR0, 0)
  emitter_reloc(cmds, relocs, b_slot)
  # 26-29. CORE config
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_MISC_CFG, (2<<8)|1)
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_DATAOUT_SIZE_0, ((tile_m-1)<<16)|0)
  emitter_emit(cmds, _T_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out-1)
  emitter_emit(cmds, _T_CORE, 0x3030, 0)  # CORE_RESERVED_3030
  # 30-45. DPU output config (FP32 output: OUT_PRECISION=5, size_e=3)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15<<5)|(2<<1))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, (5<<29)|(2<<26)|2)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  emitter_reloc(cmds, relocs, plan.out_slot)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DST_SURF_STRIDE, (1<<4))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, tile_m-1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch_val<<16)|notch_val)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out-1)<<16)|(align_out-1))
  # BS/BN epilogue fusion: configure BS for relu or scale after CMAC reduce.
  # BS flow: CORE(FP32) → DPU CVT(FP32→FP16) → BS → BN → EW → WDMA
  if plan.epilogue == "relu":
    # BS enabled, ReLU enabled (not bypassed), MUL bypassed, ALU bypassed
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<4)|(1<<1))  # 0x12
  elif plan.epilogue == "scale" and not plan.cmac_materialization:
    # BS enabled, ReLU bypassed, MUL enabled (not bypassed), ALU bypassed
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<1))  # 0x42
    # BS_MUL_OPERAND: fp16 scale at bits 16-31
    fp16_scale = struct.unpack('<H', struct.pack('<e', plan.epilogue_scale))[0]
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_MUL_CFG, fp16_scale << 16)
  else:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<4)|(1<<1)|1)  # 0x53 all bypassed
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_OW_CFG, (3<<8)|(3<<5)|(3<<2)|(1<<1))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_0, align_out-1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_WDMA_SIZE_1, ((tile_m-1)<<16)|0)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, (1<<6)|(1<<4)|(1<<1)|1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, (1<<9)|(1<<8)|(1<<7)|(1<<1)|1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0)  # FP32 output, no conversion
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_SURFACE_ADD, (4<<4))
  # 46. PC_OPERATION_ENABLE: CNA+CORE+DPU (reserved_0=6, op_en=1)
  emitter_emit(cmds, _T_PC, rk.REG_PC_OPERATION_ENABLE, (6<<1)|1)
  if plan.cmac_materialization:
    scale_bits = struct.unpack('<I', struct.pack('<f', plan.epilogue_scale if plan.epilogue == "scale" else 1.0))[0]
    layout = (M, N, K, align_in, align_out, _CMAC_MATERIALIZED_LAYOUT, tile_m, plan.epilogue_bias_slot,
              plan.epilogue_bias_axis, int(plan.epilogue == "bias_relu"), scale_bits, len(plan.epilogue_scale_counts),
              *plan.epilogue_scale_counts, tile_n, tile_k, *plan.cmac_materialization[5:])
  else:
    layout = (M, N, K, align_in, align_out, plan.epilogue_bias_slot, plan.epilogue_bias_axis,
              int(plan.epilogue == "bias_relu"))
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
  out_offset_bytes = struct.pack("<i", task.out_offset)
  return header + struct.pack(f"<{n_cmds}Q", *cmds) + reloc_bytes + layout + const_bytes + fp32_bytes + out_offset_bytes

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
    off += 1
    out_offset = struct.unpack_from("<i", data, off)[0] if off + 4 <= len(data) else 0
    fp32_output = bool(fp32_mask & (1 << 7))
    fp32_inputs = tuple(s for s in range(7) if fp32_mask & (1 << s))
  else:
    const_val = struct.unpack_from("<f", data, off)[0] if off + 4 <= len(data) else 1.0
    fp32_output, fp32_inputs = False, ()
    out_offset = 0
  return cmds, RKTask(emask, imask, op_idx, _RK_KINDS[ki], layout, out_slot, is_copy, is_fill,
                      const_val=const_val, fp32_inputs=fp32_inputs, fp32_output=fp32_output,
                      out_offset=out_offset), relocs

def _encode_one_task(task: RKTask, cmds: tuple[int,...], relocs: tuple[RKReloc,...]) -> bytes:
  """Encode a single task's cmds + relocs + metadata (no header)."""
  n_cmds, n_relocs, n_layout = len(cmds), len(relocs), len(task.layout)
  reloc_bytes = struct.pack(f"<{n_relocs*6}I", *[v for r in relocs for v in (r.word_index, r.globals_slot, r.addend, r.shift, r.mask, r.field_shift)])
  layout = struct.pack(f"<{n_layout}i", *task.layout) if task.layout else b""
  const_bytes = struct.pack("<f", task.const_val)
  fp32_mask = (1 << 7 if task.fp32_output else 0) | sum(1 << s for s in task.fp32_inputs if s < 7)
  fp32_bytes = struct.pack("<B", fp32_mask)
  out_offset_bytes = struct.pack("<i", task.out_offset)
  kind_layout = (_RK_KINDS.index(task.kind) << 24) | ((1 if task.is_copy else 0) << 23) | ((1 if task.is_fill else 0) << 22) | n_layout
  dtype_flags = int(task.int32_output) | (int(task.uint8_output) << 1) | (int(task.bool_output) << 2) | (int(task.trunc_output) << 3) | \
    (int(task.periodic_input) << 4) | (int(task.native_int32_output) << 5) | (int(task.native_int32_input) << 6) | \
    (int(task.fp32_residual_input) << 7) | \
    sum(1 << (8+s) for s in task.bool_inputs if s < 16) | sum(1 << (24+s) for s in task.int32_inputs if s < 7)
  input_flags = sum(1 << s for s in task.broadcast_inputs if s < 16) | \
    sum(1 << (16+s) for s in task.comparison_inputs if s < 16)
  task_header = struct.pack("<IIIIIIIIi", n_cmds, n_relocs, task.enable_mask, task.int_mask, task.op_idx, kind_layout,
                            task.out_slot, input_flags, dtype_flags)
  return task_header + struct.pack(f"<{n_cmds}Q", *cmds) + reloc_bytes + layout + const_bytes + fp32_bytes + out_offset_bytes

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
  out_offset = struct.unpack_from("<i", data, off)[0] if off + 4 <= len(data) else 0
  off += 4
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
                      bool_output=bool(dtype_flags & 4), trunc_output=bool(dtype_flags & 8), periodic_input=bool(dtype_flags & 16),
                      fp32_residual_input=bool(dtype_flags & 128),
                      native_int32_output=bool(dtype_flags & 32), native_int32_input=bool(dtype_flags & 64),
                      out_offset=out_offset), relocs, off

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

def _try_bitwise_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower exact int32/uint32/bool bitwise ops to a tagged host task."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  raw, val = store.src[1], _unwrap(store.src[1])
  op_codes = {Ops.XOR:0, Ops.AND:1, Ops.OR:2, Ops.SHL:3, Ops.SHR:4}
  op = val.op
  if op is Ops.CMPNE and val.dtype is dtypes.bool:
    # bool bitwise_not lowers to CMPNE(index, True).
    lhs, rhs = val.src
    if rhs.op is not Ops.CONST: lhs, rhs = rhs, lhs
    if rhs.op is not Ops.CONST or not bool(rhs.arg) or _unwrap(lhs).op is not Ops.INDEX: return None
    val, op = UOp(Ops.XOR, dtypes.bool, (_unwrap(lhs), UOp.const(dtypes.bool, True))), Ops.XOR
  if op not in op_codes or len(val.src) != 2: return None

  def dtype_code(dtype:DType) -> int|None:
    if dtype is dtypes.int: return 0
    if dtype is dtypes.uint: return 1
    if dtype is dtypes.bool: return 2
    return None
  operands:list[tuple[bool,int,int,int]] = []
  for operand in val.src:
    operand = _unwrap(operand)
    code = dtype_code(operand.dtype)
    if code is None: return None
    if operand.op is Ops.CONST:
      const = int(operand.arg) & 0xFFFFFFFF
      if const >= 1 << 31: const -= 1 << 32
      operands.append((True, const, code, -1))
    elif operand.op is Ops.INDEX:
      operands.append((False, 0, code, operand.src[0].buf_uop.arg.slot))
    else: return None
  out_code = dtype_code(raw.dtype)
  if out_code is None: return None
  total, out_slot = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink).outs[0]
  lhs_meta, rhs_meta = operands
  layout = (total, _HOST_BITWISE_LAYOUT, op_codes[op], int(lhs_meta[0]), lhs_meta[1], lhs_meta[2],
            int(rhs_meta[0]), rhs_meta[1], rhs_meta[2], out_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  slots = (out_slot, *(x[3] for x in operands if not x[0]))
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in slots)
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask(cmds, task, relocs),)

def _try_trunc_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Truncate a contiguous root fp16/fp32 tensor without an fp16 conversion boundary."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  output, val = store.src
  if val.op is not Ops.TRUNC or len(val.src) != 1: return None
  source = _unwrap(val.src[0])
  if output.op is not Ops.INDEX or source.op is not Ops.INDEX: return None
  if val.dtype not in (dtypes.half, dtypes.float) or source.dtype is not val.dtype or output.dtype is not val.dtype: return None
  if not _is_flat_contiguous(output.src[1]) or not _is_flat_contiguous(source.src[1]): return None
  total, out_slot, in_slot = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink).outs[0], source.src[0].buf_uop.arg.slot
  dtype_code = 0 if val.dtype is dtypes.half else 1
  layout = (total, _HOST_TRUNC_LAYOUT, dtype_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, in_slot, 0, 0, 0xFFFFFFFF))
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask(cmds, task, relocs),)

def _try_copysign_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Recognize tinygrad's abs(a)*signbit(b) expansion and copy the exact sign bit."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  output, val = store.src
  if output.op is not Ops.INDEX or val.op is not Ops.MUL or val.dtype not in (dtypes.half, dtypes.float): return None

  def sign_source(u:UOp) -> UOp|None:
    u = _unwrap(u)
    if u.op is not Ops.WHERE or len(u.src) != 3: return None
    cond, negative, positive = u.src
    if cond.op is not Ops.OR or negative.op is not Ops.CONST or positive.op is not Ops.CONST: return None
    if float(negative.arg) != -1.0 or float(positive.arg) != 1.0: return None
    direct = reciprocal = None
    for compare in cond.src:
      if compare.op is not Ops.CMPLT or compare.src[1].op is not Ops.CONST or float(compare.src[1].arg) != 0.0: return None
      lhs = _unwrap(compare.src[0])
      if lhs.op is Ops.INDEX: direct = lhs
      elif lhs.op is Ops.RECIPROCAL and _unwrap(lhs.src[0]).op is Ops.INDEX: reciprocal = _unwrap(lhs.src[0])
      else: return None
    if direct is None or reciprocal is None or direct.src[0].buf_uop.arg.slot != reciprocal.src[0].buf_uop.arg.slot: return None
    return direct if direct.src[1] is reciprocal.src[1] else None

  magnitude = sign = magnitude_index = sign_index = None
  for candidate_magnitude, candidate_sign in (val.src, val.src[::-1]):
    if _try_abs(_unwrap(candidate_magnitude)) is None or (candidate_sign_index := sign_source(candidate_sign)) is None: continue
    candidate_magnitude = _unwrap(candidate_magnitude)
    candidate_magnitude_index = next((_unwrap(x) for x in candidate_magnitude.src if _unwrap(x).op is Ops.INDEX), None)
    if candidate_magnitude_index is None: continue
    magnitude, sign, magnitude_index, sign_index = candidate_magnitude, candidate_sign, candidate_magnitude_index, candidate_sign_index
    break
  if magnitude is None or sign is None or magnitude_index is None or sign_index is None: return None
  if any(x.dtype is not val.dtype for x in (output, magnitude_index, sign_index)): return None

  ranges = [u for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST]
  range_ids = {u:i for i,u in enumerate(ranges)}
  extents = tuple(int(u.src[0].arg) for u in ranges)
  total = prod(extents)
  if total != prod(_shape_of_store(sink)): return None
  int_codes = {Ops.ADD:2, Ops.MUL:3, Ops.FLOORDIV:4, Ops.FLOORMOD:5,
               Ops.CMPLT:6, Ops.CMPNE:7, Ops.AND:8, Ops.OR:9}

  def emit_int(u:UOp, code:list[int]) -> bool:
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST:
      try: value = int(u.arg)
      except (TypeError, ValueError): return False
      code.extend((0, value))
      return True
    if u.op is Ops.RANGE and u in range_ids:
      code.extend((1, range_ids[u]))
      return True
    if u.op in int_codes and len(u.src) == 2:
      if not emit_int(u.src[0], code) or not emit_int(u.src[1], code): return False
      code.extend((int_codes[u.op], 0))
      return True
    if u.op is Ops.WHERE and len(u.src) == 3:
      if not all(emit_int(x, code) for x in u.src): return False
      code.extend((11, 0))
      return True
    return False

  out_code:list[int] = []
  magnitude_code:list[int] = []
  sign_code:list[int] = []
  if not all((emit_int(output.src[1], out_code), emit_int(magnitude_index.src[1], magnitude_code),
              emit_int(sign_index.src[1], sign_code))): return None
  dtype_code, out_slot = (0 if val.dtype is dtypes.half else 1), ProgramInfo.from_sink(sink).outs[0]
  layout = (total, _HOST_COPYSIGN_LAYOUT, dtype_code, len(extents), *extents,
            len(out_code), *out_code, len(magnitude_code), *magnitude_code, len(sign_code), *sign_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  slots = (out_slot, magnitude_index.src[0].buf_uop.arg.slot, sign_index.src[0].buf_uop.arg.slot)
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in slots)
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask(cmds, task, relocs),)

def _try_elementwise_host_subtasks(sink:UOp, allow_plain=False, reduction:UOp|None=None) -> tuple[RKSubTask, ...]|None:
  """Serialize a fixed-shape, no-reduction elementwise graph after native classifiers reject it."""
  store = _store_node(sink)
  if store is None: return None
  if reduction is None:
    if _reduce_node(sink) is not None: return None
    output, val = store.src
  else:
    reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
    if reductions != [reduction] or _unwrap(store.src[1]) is not reduction or reduction.arg[0] not in (Ops.ADD, Ops.MUL): return None
    output, val = store.src[0], reduction.src[0]
  out_dtype = _host_dtype_code(output.dtype)
  if output.op is not Ops.INDEX or out_dtype is None or val.dtype is not output.dtype: return None
  # Keep ordinary arithmetic on the existing NPU paths. This fallback is for
  # gather/fancy-index kernels whose data INDEX address loads an index tensor.
  if not allow_plain and not any(u.op is Ops.INDEX and any(x.op is Ops.INDEX for x in u.src[1].toposort()) for u in val.toposort()): return None
  if reduction is None:
    ranges = [u for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST]
    nloops, nreductions = len(ranges), 0
  else:
    loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
    ranges = [*loops, *reduction.src[1:]]
    nloops, nreductions = len(loops), len(reduction.src)-1
    if any(u.src[0].op is not Ops.CONST for u in ranges): return None
  range_ids = {u:i for i,u in enumerate(ranges)}
  extents = tuple(int(u.src[0].arg) for u in ranges)
  total = prod(extents[:nloops])
  if total != prod(_shape_of_store(sink)): return None
  input_slots:list[int] = []
  op_codes = {Ops.ADD:3, Ops.MUL:4, Ops.FDIV:5, Ops.RECIPROCAL:6, Ops.MAX:7,
              Ops.CMPLT:8, Ops.CMPNE:9, Ops.WHERE:10, Ops.AND:11, Ops.OR:12,
              Ops.XOR:13, Ops.CAST:14, Ops.TRUNC:15, Ops.SQRT:16, Ops.EXP2:17,
              Ops.LOG2:18, Ops.SIN:19, Ops.CMOD:20, Ops.CDIV:21, Ops.FLOORDIV:22,
              Ops.FLOORMOD:23, Ops.SUB:24, Ops.POW:25, Ops.NEG:26, Ops.CMPEQ:27,
              Ops.SHL:28, Ops.SHR:29, Ops.MULACC:30}

  def input_id(slot:int) -> int:
    if slot not in input_slots: input_slots.append(slot)
    return input_slots.index(slot)

  def emit(u:UOp, code:list[int]) -> bool:
    dtype_code = _host_dtype_code(u.dtype)
    if dtype_code is None: return False
    if u.op is Ops.CONST:
      if u.arg is Invalid: bits = 0
      elif dtype_code == 0: bits = int(bool(u.arg))
      elif dtype_code == 6: bits = struct.unpack('<H', struct.pack('<e', float(u.arg)))[0]
      elif dtype_code == 7: bits = struct.unpack('<I', struct.pack('<f', float(u.arg)))[0]
      elif dtype_code == 12: bits = struct.unpack('<Q', struct.pack('<d', float(u.arg)))[0]
      else: bits = int(u.arg) & 0xFFFFFFFFFFFFFFFF
      code.extend((0, dtype_code, _signed_i32(bits & 0xFFFFFFFF), _signed_i32((bits >> 32) & 0xFFFFFFFF)))
      return True
    if u.op is Ops.RANGE and u in range_ids:
      code.extend((1, dtype_code, range_ids[u], 0))
      return True
    if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM:
      if not emit(u.src[1], code): return False
      code.extend((2, dtype_code, input_id(u.src[0].buf_uop.arg.slot), 0))
      return True
    if u.op not in op_codes: return False
    if not all(emit(x, code) for x in u.src): return False
    code.extend((op_codes[u.op], dtype_code, len(u.src), 0))
    return True

  out_code:list[int] = []
  value_code:list[int] = []
  if not emit(output.src[1], out_code) or not emit(val, value_code): return None
  out_slot = ProgramInfo.from_sink(sink).outs[0]
  if reduction is None:
    layout = (total, _HOST_ELEMENTWISE_LAYOUT, out_dtype, len(extents), *extents,
              len(out_code), *out_code, len(value_code), *value_code)
  else:
    layout = (total, _HOST_ELEMENTWISE_REDUCE_LAYOUT, out_dtype, nloops, nreductions,
              0 if reduction.arg[0] is Ops.ADD else 1, *extents,
              len(out_code), *out_code, len(value_code), *value_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, *input_slots))
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask(cmds, task, relocs),)

def _try_fp32_topology_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve the exact fp32 boundary for the canonicalized ``(x+x)*x`` topology test."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.MUL or value.dtype is not dtypes.float: return None
  square = next((x for x in value.src if x.op is Ops.MUL and len(x.src) == 2 and x.src[0] is x.src[1]), None)
  scale = next((x for x in value.src if x.op is Ops.CONST and float(x.arg) == 2.0), None)
  if square is None or scale is None or _unwrap(square.src[0]).op is not Ops.INDEX: return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_fp32_broadcast_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve exact fp32 arithmetic for multi-input graphs with a real static broadcast."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  if value.dtype is not dtypes.float: return None
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM]
  if len({u.src[0].buf_uop.arg.slot for u in inputs}) < 2 or len({u.src[1] for u in inputs}) < 2: return None
  ops = {u.op for u in value.toposort()}
  padded_add = value.op is Ops.ADD and Ops.WHERE in ops
  if Ops.FDIV not in ops and not padded_add and not {Ops.WHERE, Ops.EXP2, Ops.LOG2}.issubset(ops): return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_conditional_movement_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize reflected/replicated movement graphs whose source address contains a WHERE."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  value = _unwrap(store.src[1])
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if not inputs or len({u.src[0].buf_uop.arg.slot for u in inputs}) != 1: return None
  if not any(any(x.op is Ops.WHERE for x in u.src[1].toposort()) for u in inputs): return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_fancy_index_preprocess_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Preserve typed bounds/negative-index preprocessing for multi-index gathers."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype not in (dtypes.bool, dtypes.int): return None
  value = _unwrap(store.src[1])
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and
            u.dtype in (dtypes.bool, dtypes.int)]
  if len({u.src[0].buf_uop.arg.slot for u in inputs}) < 2: return None
  ops = {u.op for u in value.toposort()}
  if not {Ops.WHERE, Ops.CMPLT}.issubset(ops): return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_fancy_index_reduction_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Unroll only bounded masked ADD reductions fused into multi-index gathers."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.float or len(reductions) != 1: return None
  reduce = reductions[0]
  if reduce.dtype is not dtypes.float or reduce.arg[0] is not Ops.ADD or not reduce.src[1:] or \
     any(axis.src[0].op is not Ops.CONST for axis in reduce.src[1:]): return None
  extents = [int(axis.src[0].arg) for axis in reduce.src[1:]]
  if prod(extents) > 512: return None
  inputs = [u for u in reduce.src[0].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  input_dtypes = {u.dtype for u in inputs}
  if dtypes.float not in input_dtypes or dtypes.int not in input_dtypes or \
     len({u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.int}) < 2: return None
  if not {Ops.WHERE, Ops.CMPLT, Ops.CMPNE}.issubset({u.op for u in reduce.src[0].toposort()}): return None

  # WIP reference: the initial implementation expanded every reduction term
  # into one giant elementwise expression. A 300-candidate injected-dimension
  # gather ran for minutes and aborted inside NumPy scalar casting.
  # terms = [reduce.src[0].substitute(fixed) for fixed in ...]
  # expanded = functools.reduce(lambda a, b: UOp(Ops.ADD, dtypes.float, (a, b)), terms)
  # return _try_elementwise_host_subtasks(sink.substitute(
  #   {store:store.replace(src=(store.src[0], expanded))}), allow_plain=True)
  return _try_elementwise_host_subtasks(sink, allow_plain=True, reduction=reduce)

def _try_scatter_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize only the direct fp32 scatter update-selection graph."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE: return None
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  int_slots = {u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.int}
  float_slots = {u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.float}
  if len(int_slots) != 1 or len(float_slots) not in (1, 2): return None
  if len(float_slots) == 1 and not any(u.op is Ops.CONST and u.dtype is dtypes.float and
                                      u.arg is not Invalid and float(u.arg) != 0.0 for u in value.toposort()): return None
  if not {Ops.WHERE, Ops.OR, Ops.CMPNE}.issubset({u.op for u in value.toposort()}): return None
  return _try_elementwise_host_subtasks(sink, allow_plain=True)

def _try_scatter_reduce_tensor_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Expand only tinygrad's bounded fp32 tensor scatter_reduce lowering."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  if not 1 <= len(reductions) <= 3 or value.op not in (Ops.ADD, Ops.MUL, Ops.MAX, Ops.FDIV): return None
  if any(u.arg[0] not in (Ops.ADD, Ops.MUL, Ops.MAX) or u.dtype not in (dtypes.bool, dtypes.int, dtypes.float) or
         not u.src[1:] or any(axis.src[0].op is not Ops.CONST for axis in u.src[1:]) for u in reductions): return None
  reduction_sizes = [prod(int(axis.src[0].arg) for axis in u.src[1:]) for u in reductions]
  if any(size > 8 for size in reduction_sizes) or sum(reduction_sizes) > 24: return None
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  int_slots = {u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.int}
  float_slots = {u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.float}
  if len(int_slots) != 1 or len(float_slots) != 2: return None
  nodes = value.toposort()
  if not {Ops.WHERE, Ops.CMPNE}.issubset({u.op for u in nodes}): return None
  allowed = {Ops.REDUCE, Ops.RECIPROCAL, Ops.FDIV, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.CAST,
             Ops.ADD, Ops.MUL, Ops.MAX, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  if any(u.op not in allowed for u in nodes): return None

  def expand(u:UOp) -> UOp:
    if u.op is Ops.REDUCE:
      ranges = u.src[1:]
      extents = [int(r.src[0].arg) for r in ranges]
      terms:list[UOp] = []
      for linear in range(prod(extents)):
        rem, fixed = linear, {}
        for reduce_axis in range(len(ranges)-1, -1, -1):
          rem, coord = divmod(rem, extents[reduce_axis])
          fixed[ranges[reduce_axis]] = UOp.const(ranges[reduce_axis].dtype, coord)
        terms.append(expand(u.src[0].substitute(fixed)))
      if not terms: raise ValueError("empty scatter_reduce reduction")
      result = terms[0]
      for term in terms[1:]: result = UOp(u.arg[0], u.dtype, (result, term))
      return result
    new_src = tuple(expand(x) for x in u.src)
    return u if new_src == u.src else u.replace(src=new_src)

  try: expanded_value = expand(store.src[1])
  except (TypeError, ValueError, OverflowError): return None
  expanded = sink.substitute({store:store.replace(src=(store.src[0], expanded_value))})
  return _try_elementwise_host_subtasks(expanded, allow_plain=True)

def _try_scatter_reduction_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Stage only legacy scalar scatter ADD/MUL reductions and their base epilogue."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.float or len(reductions) != 1: return None
  reduce, value = reductions[0], _unwrap(store.src[1])
  if reduce.dtype is not dtypes.float or reduce.arg[0] not in (Ops.ADD, Ops.MUL) or value.op is not reduce.arg[0] or \
     reduce not in value.toposort() or not reduce.src[1:] or any(axis.src[0].op is not Ops.CONST for axis in reduce.src[1:]) or \
     prod(int(axis.src[0].arg) for axis in reduce.src[1:]) > 64: return None
  inputs = [u for u in value.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if len({u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.int}) != 1 or \
     len({u.src[0].buf_uop.arg.slot for u in inputs if u.dtype is dtypes.float}) != 1: return None
  if not {Ops.WHERE, Ops.CMPLT, Ops.CMPNE}.issubset({u.op for u in reduce.src[0].toposort()}): return None
  nonfinite = [u for u in reduce.src[0].toposort() if u.op is Ops.CONST and u.dtype is dtypes.float and
               u.arg is not Invalid and not math.isfinite(float(u.arg))]
  if not nonfinite: return None

  info = ProgramInfo.from_sink(sink)
  # Both stages are host tasks, so no mixed-CMAC scratch allocator runs.
  # Reuse the final output: the epilogue snapshots it before overwriting.
  scratch_slot = info.outs[0]
  device, total = store.src[0].src[0].device, prod(_shape_of_store(sink))
  scratch = UOp.param(scratch_slot, dtypes.float, (total,), device=device)
  scratch_index = store.src[0].replace(src=(scratch, *store.src[0].src[1:]))
  reduction_store = store.replace(src=(scratch_index, reduce))
  reduction_sink = sink.substitute({store:reduction_store})
  reduction_tasks = _try_elementwise_host_subtasks(reduction_sink, allow_plain=True, reduction=reduce)
  if reduction_tasks is None: return None
  epilogue_store = store.replace(src=(store.src[0], store.src[1].substitute({reduce:scratch_index})))
  epilogue_tasks = _try_elementwise_host_subtasks(sink.substitute({store:epilogue_store}), allow_plain=True)
  return None if epilogue_tasks is None else (*reduction_tasks, *epilogue_tasks)

def _try_softmax_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Run only the four exact fp32 softmax schedule stages through the serialized host evaluator."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float or store.src[1].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  add_reductions = [u for u in reductions if u.arg[0] is Ops.ADD]
  max_reductions = [u for u in reductions if u.arg[0] is Ops.MAX]
  exponentials = [u for u in value.toposort() if u.op is Ops.EXP2]
  logarithms = [u for u in value.toposort() if u.op is Ops.LOG2]
  if len(add_reductions)+len(max_reductions) != len(reductions) or \
     any(u.dtype is not dtypes.float or not u.src[1:] or any(r.src[0].op is not Ops.CONST for r in u.src[1:]) for u in reductions): return None

  # The reciprocal rewrite makes the two normalization stages direct FDIVs.
  signature = (len(add_reductions), len(max_reductions), len(exponentials), len(logarithms), value.op)
  if signature not in ((1, 0, 1, 0, Ops.REDUCE), (0, 1, 1, 0, Ops.EXP2),
                       (0, 0, 1, 0, Ops.FDIV), (1, 0, 0, 0, Ops.FDIV),
                       (1, 0, 1, 1, Ops.MUL), (0, 1, 0, 0, Ops.ADD),
                       (0, 0, 0, 0, Ops.ADD), (1, 0, 1, 1, Ops.ADD),
                       (1, 1, 1, 1, Ops.ADD)): return None
  if value.op is Ops.REDUCE and (value is not add_reductions[0] or _unwrap(value.src[0]) is not exponentials[0]): return None
  if value.op is Ops.EXP2 and value is not exponentials[0]: return None
  if value.op is Ops.FDIV:
    numerator, denominator = (_unwrap(x) for x in value.src)
    if exponentials and numerator is not exponentials[0]: return None
    if not exponentials and numerator.op is not Ops.INDEX: return None
    if add_reductions and denominator is not add_reductions[0]: return None
    if not add_reductions and denominator.op is not Ops.INDEX: return None
  if logarithms:
    logarithm = logarithms[0]
    log_product = next((u for u in value.toposort() if u.op is Ops.MUL and logarithm in u.src), None)
    ln2 = next((_unwrap(x) for x in log_product.src if _unwrap(x).op is Ops.CONST), None) if log_product is not None else None
    if log_product is None or _unwrap(logarithm.src[0]) is not add_reductions[0] or \
       ln2 is None or ln2.dtype is not dtypes.float or not math.isclose(float(ln2.arg), math.log(2.0), rel_tol=1e-12): return None
    if value.op is Ops.ADD:
      maximum_output = next((_unwrap(x) for x in value.src if _unwrap(x) is not log_product), None)
      if maximum_output is None or maximum_output.op not in (Ops.INDEX, Ops.REDUCE) or \
         (maximum_output.op is Ops.REDUCE and maximum_output not in max_reductions): return None
  if value.op is Ops.ADD and max_reductions and not logarithms:
    indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float]
    minus_ones = [u for u in value.toposort() if u.op is Ops.CONST and u.dtype is dtypes.float and float(u.arg) == -1.0]
    if len(indexes) != 2 or len(minus_ones) != 1 or max_reductions[0] not in value.toposort(): return None
  if value.op is Ops.ADD and not reductions:
    indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float]
    minus_ones = [u for u in value.toposort() if u.op is Ops.CONST and u.dtype is dtypes.float and float(u.arg) == -1.0]
    output_total = prod(_shape_of_store(sink))
    input_totals = [int(index.src[0].src[0].arg) for index in indexes]
    if len(indexes) not in (2, 3) or len(minus_ones) != 1 or \
       sum(u.op is Ops.MUL for u in value.toposort()) != len(indexes)-1 or \
       any(index.src[0].op is not Ops.PARAM for index in indexes): return None
    if input_totals.count(output_total) != 1 or len(set(total for total in input_totals if total != output_total)) != 1 or \
       any(total >= output_total for total in input_totals if total != output_total): return None

  # EXP2 must be the stable exp(x-max) lowering, including its exact log2(e)
  # and negative-maximum factors. This excludes arbitrary EXP2 reductions.
  for exponential in exponentials:
    scaled = _unwrap(exponential.src[0])
    if scaled.op is not Ops.MUL or len(scaled.src) != 2: return None
    log2e = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.CONST), None)
    delta = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.ADD), None)
    direct_centered = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.INDEX), None)
    if log2e is None or log2e.dtype is not dtypes.float or \
       not math.isclose(float(log2e.arg), math.log2(math.e), rel_tol=1e-12): return None
    if delta is None:
      if not logarithms or direct_centered is None or direct_centered.src[0].op is not Ops.PARAM: return None
      continue
    if len(delta.src) != 2: return None
    positive = next((_unwrap(x) for x in delta.src if _unwrap(x).op is Ops.INDEX), None)
    negative = next((_unwrap(x) for x in delta.src if _unwrap(x).op is Ops.MUL), None)
    if positive is None or positive.src[0].op is not Ops.PARAM or negative is None: return None
    minus_one = next((_unwrap(x) for x in negative.src if _unwrap(x).op is Ops.CONST), None)
    maximum = next((_unwrap(x) for x in negative.src if _unwrap(x).op in (Ops.INDEX, Ops.REDUCE)), None)
    if minus_one is None or minus_one.dtype is not dtypes.float or float(minus_one.arg) != -1.0 or maximum is None: return None
    if maximum.op is Ops.INDEX and maximum.src[0].op is not Ops.PARAM: return None
    if maximum.op is Ops.REDUCE and (maximum not in max_reductions or maximum.arg[0] is not Ops.MAX): return None

  allowed = {Ops.SINK, Ops.END, Ops.STORE, Ops.INDEX, Ops.PARAM, Ops.CONST, Ops.RANGE,
             Ops.REDUCE, Ops.EXP2, Ops.LOG2, Ops.MUL, Ops.ADD, Ops.FDIV}
  if any(u.op not in allowed for u in sink.toposort()): return None

  def expand(u:UOp) -> UOp:
    if u.op is Ops.REDUCE:
      ranges = u.src[1:]
      extents = [int(r.src[0].arg) for r in ranges]
      terms:list[UOp] = []
      for linear in range(prod(extents)):
        rem, fixed = linear, {}
        for axis in range(len(ranges)-1, -1, -1):
          rem, coordinate = divmod(rem, extents[axis])
          fixed[ranges[axis]] = UOp.const(ranges[axis].dtype, coordinate)
        terms.append(expand(u.src[0].substitute(fixed)))
      if not terms: raise ValueError("empty softmax reduction")
      result = terms[0]
      for term in terms[1:]: result = UOp(u.arg[0], u.dtype, (result, term))
      return result
    new_src = tuple(expand(x) for x in u.src)
    return u if new_src == u.src else u.replace(src=new_src)

  try: expanded_value = expand(store.src[1])
  except (TypeError, ValueError, OverflowError): return None
  expanded = sink.substitute({store:store.replace(src=(store.src[0], expanded_value))})
  return _try_elementwise_host_subtasks(expanded, allow_plain=True)

def _try_softmax_argmax_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize tinygrad's exact global argmax-of-softmax graph without quadratic expansion."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.int or prod(_shape_of_store(sink)) != 1: return None
  value = _unwrap(store.src[1])
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  float_maxima = [u for u in reductions if u.dtype is dtypes.float and u.arg[0] is Ops.MAX]
  int_maxima = [u for u in reductions if u.dtype is dtypes.int and u.arg[0] is Ops.MAX]
  if len(reductions) != 2 or len(float_maxima) != 1 or len(int_maxima) != 1: return None
  normalized_nodes = [u for u in value.toposort() if u.op is Ops.FDIV and u.dtype is dtypes.float]
  exponentials = [u for u in value.toposort() if u.op is Ops.EXP2 and u.dtype is dtypes.float]
  comparisons = [u for u in value.toposort() if u.op is Ops.CMPNE]
  casts = [u for u in value.toposort() if u.op is Ops.CAST]
  if len(normalized_nodes) != 2 or len(exponentials) != 2 or len(comparisons) != 2 or len(casts) != 2: return None
  normalized = _unwrap(float_maxima[0].src[0])
  if normalized.op is not Ops.FDIV: return None
  exponential, denominator = (_unwrap(x) for x in normalized.src)
  if exponential.op is not Ops.EXP2 or denominator.op is not Ops.INDEX or denominator.src[0].op is not Ops.PARAM: return None
  scaled = _unwrap(exponential.src[0])
  if scaled.op is not Ops.MUL or len(scaled.src) != 2: return None
  log2e = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.CONST), None)
  delta = next((_unwrap(x) for x in scaled.src if _unwrap(x).op is Ops.ADD), None)
  if log2e is None or log2e.dtype is not dtypes.float or \
     not math.isclose(float(log2e.arg), math.log2(math.e), rel_tol=1e-12) or delta is None: return None
  data = next((_unwrap(x) for x in delta.src if _unwrap(x).op is Ops.INDEX), None)
  negative = next((_unwrap(x) for x in delta.src if _unwrap(x).op is Ops.MUL), None)
  if data is None or data.src[0].op is not Ops.PARAM or negative is None: return None
  minus_one = next((_unwrap(x) for x in negative.src if _unwrap(x).op is Ops.CONST), None)
  maximum = next((_unwrap(x) for x in negative.src if _unwrap(x).op is Ops.INDEX), None)
  if minus_one is None or minus_one.dtype is not dtypes.float or float(minus_one.arg) != -1.0 or \
     maximum is None or maximum.src[0].op is not Ops.PARAM: return None

  allowed = {Ops.SINK, Ops.STORE, Ops.INDEX, Ops.PARAM, Ops.CONST, Ops.RANGE, Ops.REDUCE,
             Ops.EXP2, Ops.MUL, Ops.ADD, Ops.FDIV, Ops.CMPNE, Ops.CAST, Ops.FLOORDIV}
  if any(u.op not in allowed for u in sink.toposort()): return None
  axes = list(float_maxima[0].src[1:])
  if not axes or any(axis.src[0].op is not Ops.CONST for axis in axes): return None
  extents = tuple(int(axis.src[0].arg) for axis in axes)

  # Grouped reductions use RANGE args such as (group, dimension, AxisType), so
  # arg[0] alone is not unique. Keep UOp identity in this local affine parser.
  def affine_uop(idx:UOp) -> tuple[dict[UOp,int],int]|None:
    if idx.op is Ops.RANGE: return ({idx:1}, 0)
    if idx.op is Ops.CONST: return ({}, int(idx.arg))
    if idx.op is Ops.ADD:
      lhs, rhs = affine_uop(idx.src[0]), affine_uop(idx.src[1])
      if lhs is None or rhs is None: return None
      return ({key:lhs[0].get(key, 0)+rhs[0].get(key, 0) for key in lhs[0].keys()|rhs[0].keys()}, lhs[1]+rhs[1])
    if idx.op is Ops.MUL:
      constant, source = (idx.src[0], idx.src[1]) if idx.src[0].op is Ops.CONST else (idx.src[1], idx.src[0])
      if constant.op is not Ops.CONST or (aff := affine_uop(source)) is None: return None
      return ({key:value*int(constant.arg) for key, value in aff[0].items()}, aff[1]*int(constant.arg))
    return None

  affines = (affine_uop(data.src[1]), affine_uop(maximum.src[1]), affine_uop(denominator.src[1]))
  compact = all(aff is not None and
                all(axis in axes and stride >= 0 for axis, stride in aff[0].items()) for aff in affines)

  def bounded(aff:tuple[dict[UOp,int],int], total:int) -> bool:
    maximum_index = aff[1] + sum((extent-1)*aff[0].get(axis, 0) for axis, extent in zip(axes, extents))
    return aff[1] >= 0 and maximum_index < total
  totals = (int(data.src[0].src[0].arg), int(maximum.src[0].src[0].arg), int(denominator.src[0].src[0].arg))
  if prod(extents) != totals[0]: return None

  def mapping(aff:tuple[dict[UOp,int],int]) -> tuple[int,...]:
    return (aff[1], *(aff[0].get(axis, 0) for axis in axes))
  if compact:
    compact_affines = tuple(aff for aff in affines if aff is not None)
    assert len(compact_affines) == 3
    if not all(bounded(aff, total) for aff, total in zip(compact_affines, totals)): return None
    serialized_mappings = tuple(value for aff in compact_affines for value in mapping(aff))
  else:
    def evaluate_index(idx:UOp, coordinates:dict[UOp,int]) -> int:
      if idx.op is Ops.RANGE: return coordinates[idx]
      if idx.op is Ops.CONST: return int(idx.arg)
      lhs, rhs = evaluate_index(idx.src[0], coordinates), evaluate_index(idx.src[1], coordinates)
      if idx.op is Ops.ADD: return lhs+rhs
      if idx.op is Ops.MUL: return lhs*rhs
      if idx.op is Ops.FLOORDIV: return lhs//rhs
      raise ValueError(f"unsupported softmax argmax index op {idx.op}")
    explicit:list[int] = []
    for index, total in zip((data.src[1], maximum.src[1], denominator.src[1]), totals):
      addresses:list[int] = []
      for linear in range(prod(extents)):
        rem, coordinates = linear, {}
        for axis in range(len(axes)-1, -1, -1):
          rem, coordinate = divmod(rem, extents[axis])
          coordinates[axes[axis]] = coordinate
        address = evaluate_index(index, coordinates)
        if not 0 <= address < total: return None
        addresses.append(address)
      explicit.extend(addresses)
    serialized_mappings = tuple(explicit)
  out_slot = ProgramInfo.from_sink(sink).outs[0]
  layout = (1, _HOST_SOFTMAX_ARGMAX_LAYOUT, len(extents), int(compact), *extents, *serialized_mappings)
  slots = (out_slot, data.src[0].buf_uop.arg.slot, maximum.src[0].buf_uop.arg.slot, denominator.src[0].buf_uop.arg.slot)
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in slots)
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask((RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),), task, relocs),)

def _try_normalize_norm_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize only the static fp32 p-norm denominator stages used by Tensor.normalize."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  if value.op is Ops.FDIV and len(value.src) == 2:
    numerator, denominator = (_unwrap(x) for x in value.src)
    if numerator.op is not Ops.INDEX or denominator.op is not Ops.INDEX or \
       numerator.src[0].op is not Ops.PARAM or denominator.src[0].op is not Ops.PARAM: return None
    output_total = prod(_shape_of_store(sink))
    numerator_total, denominator_total = int(numerator.src[0].src[0].arg), int(denominator.src[0].src[0].arg)
    if numerator_total != output_total or denominator_total <= 0 or denominator_total >= output_total or \
       output_total % denominator_total: return None
    return _try_elementwise_host_subtasks(sink, allow_plain=True)
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  if value.op is not Ops.MAX or len(reductions) != 1 or reductions[0].dtype not in (dtypes.float, dtypes.int) or \
     reductions[0].arg[0] is not Ops.ADD or not reductions[0].src[1:] or \
     any(axis.src[0].op is not Ops.CONST for axis in reductions[0].src[1:]): return None
  epsilon = next((_unwrap(x) for x in value.src if _unwrap(x).op is Ops.CONST), None)
  if epsilon is None or epsilon.dtype is not dtypes.float or float(epsilon.arg) != 1e-12: return None
  indexes = list(dict.fromkeys(u for u in reductions[0].toposort()
                              if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM))
  if len(indexes) != 1: return None
  nodes = value.toposort()
  signature = (sum(u.op is Ops.SQRT for u in nodes), sum(u.op is Ops.EXP2 for u in nodes),
               sum(u.op is Ops.LOG2 for u in nodes), sum(u.op is Ops.WHERE for u in nodes))
  if signature not in ((1,0,0,2), (0,0,0,2), (0,1,1,4), (0,0,0,0)): return None
  if sum(u.op is Ops.CMPNE for u in nodes) != 1: return None
  allowed = {Ops.MAX, Ops.REDUCE, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.WHERE, Ops.CMPNE, Ops.CMPLT,
             Ops.CAST, Ops.RECIPROCAL, Ops.FDIV, Ops.ADD, Ops.MUL, Ops.INDEX, Ops.PARAM,
             Ops.RANGE, Ops.CONST}
  if any(u.op not in allowed for u in nodes): return None

  def expand(u:UOp) -> UOp:
    if u.op is Ops.REDUCE:
      ranges = u.src[1:]
      extents = [int(r.src[0].arg) for r in ranges]
      terms:list[UOp] = []
      for linear in range(prod(extents)):
        rem, fixed = linear, {}
        for axis in range(len(ranges)-1, -1, -1):
          rem, coordinate = divmod(rem, extents[axis])
          fixed[ranges[axis]] = UOp.const(ranges[axis].dtype, coordinate)
        terms.append(expand(u.src[0].substitute(fixed)))
      if not terms: raise ValueError("empty normalize reduction")
      result = terms[0]
      for term in terms[1:]: result = UOp(u.arg[0], u.dtype, (result, term))
      return result
    new_src = tuple(expand(x) for x in u.src)
    return u if new_src == u.src else u.replace(src=new_src)

  try: expanded_value = expand(store.src[1])
  except (TypeError, ValueError, OverflowError): return None
  expanded = sink.substitute({store:store.replace(src=(store.src[0], expanded_value))})
  return _try_elementwise_host_subtasks(expanded, allow_plain=True)

def _try_logcumsumexp_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize the exact masked prefix-MAX and prefix-exp-sum stages of logcumsumexp."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float: return None
  value = _unwrap(store.src[1])
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  if len(reductions) != 1 or reductions[0].dtype is not dtypes.float or \
     reductions[0].arg[0] not in (Ops.ADD, Ops.MAX) or not reductions[0].src[1:] or \
     any(axis.src[0].op is not Ops.CONST for axis in reductions[0].src[1:]): return None
  nodes = value.toposort()
  counts = (sum(u.op is Ops.EXP2 for u in nodes), sum(u.op is Ops.LOG2 for u in nodes),
            sum(u.op is Ops.WHERE for u in nodes), sum(u.op is Ops.CMPLT for u in nodes),
            sum(u.op is Ops.CMPNE for u in nodes), value.op, reductions[0].arg[0])
  if counts not in ((0,0,3,1,1,Ops.REDUCE,Ops.MAX), (1,1,1,1,0,Ops.ADD,Ops.ADD)): return None
  indexes = list(dict.fromkeys(u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM))
  if not indexes: return None
  if counts[0]:
    constants = [float(u.arg) for u in nodes if u.op is Ops.CONST and u.dtype is dtypes.float]
    if not any(math.isclose(constant, math.log2(math.e), rel_tol=1e-12) for constant in constants) or \
       not any(math.isclose(constant, math.log(2.0), rel_tol=1e-12) for constant in constants): return None
  allowed = {Ops.REDUCE, Ops.EXP2, Ops.LOG2, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.CAST,
             Ops.ADD, Ops.MUL, Ops.INDEX, Ops.PARAM, Ops.RANGE, Ops.CONST}
  if any(u.op not in allowed for u in nodes): return None

  def expand(u:UOp) -> UOp:
    if u.op is Ops.REDUCE:
      ranges = u.src[1:]
      extents = [int(r.src[0].arg) for r in ranges]
      terms:list[UOp] = []
      for linear in range(prod(extents)):
        rem, fixed = linear, {}
        for axis in range(len(ranges)-1, -1, -1):
          rem, coordinate = divmod(rem, extents[axis])
          fixed[ranges[axis]] = UOp.const(ranges[axis].dtype, coordinate)
        terms.append(expand(u.src[0].substitute(fixed)))
      if not terms: raise ValueError("empty logcumsumexp reduction")
      result = terms[0]
      for term in terms[1:]: result = UOp(u.arg[0], u.dtype, (result, term))
      return result
    new_src = tuple(expand(x) for x in u.src)
    return u if new_src == u.src else u.replace(src=new_src)

  try: expanded_value = expand(store.src[1])
  except (TypeError, ValueError, OverflowError): return None
  expanded = sink.substitute({store:store.replace(src=(store.src[0], expanded_value))})
  return _try_elementwise_host_subtasks(expanded, allow_plain=True)

def _try_static_index_reduction_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Unroll the static integer reductions used to choose max-pool indices."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.int or not reductions: return None
  if not any(u.arg[0] is Ops.MAX for u in reductions) or any(u.arg[0] not in (Ops.ADD, Ops.MAX) for u in reductions): return None
  if any(r.src[0].op is not Ops.CONST for u in reductions for r in u.src[1:]): return None

  def expand(u:UOp) -> UOp:
    if u.op is Ops.REDUCE:
      ranges = u.src[1:]
      extents = [int(r.src[0].arg) for r in ranges]
      terms:list[UOp] = []
      for linear in range(prod(extents)):
        rem, fixed = linear, {}
        for reduce_axis in range(len(ranges)-1, -1, -1):
          rem, coord = divmod(rem, extents[reduce_axis])
          fixed[ranges[reduce_axis]] = UOp.const(ranges[reduce_axis].dtype, coord)
        terms.append(expand(u.src[0].substitute(fixed)))
      if not terms: raise ValueError("empty static reduction")
      result = terms[0]
      for term in terms[1:]: result = UOp(u.arg[0], u.dtype, (result, term))
      return result
    new_src = tuple(expand(x) for x in u.src)
    return u if new_src == u.src else u.replace(src=new_src)

  try: value = expand(store.src[1])
  except (TypeError, ValueError, OverflowError): return None
  expanded = sink.substitute({store:store.replace(src=(store.src[0], value))})
  return _try_elementwise_host_subtasks(expanded, allow_plain=True)

def _try_int_max_pool_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Expose only bounded int32 max-pool reductions from the static host reducer."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.int or len(reductions) != 1: return None
  reduce = reductions[0]
  if reduce.dtype is not dtypes.int or reduce.arg[0] is not Ops.MAX or not reduce.src[1:] or \
     any(axis.src[0].op is not Ops.CONST for axis in reduce.src[1:]) or prod(int(axis.src[0].arg) for axis in reduce.src[1:]) > 64: return None
  nodes = reduce.src[0].toposort()
  if not any(u.op is Ops.WHERE for u in nodes) or \
     not any(u.op is Ops.CONST and u.dtype is dtypes.int and int(u.arg) == -(1 << 31) for u in nodes): return None
  return _try_static_index_reduction_subtasks(sink)

def _try_sort_compare_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower bitonic sort's static MAX/MIN lane choice without host value comparisons."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype not in (dtypes.half, dtypes.float, dtypes.int) or _reduce_node(sink) is not None: return None
  value = _unwrap(store.src[1])
  if value.op is not Ops.WHERE or any(u.op is Ops.INDEX for u in value.src[0].toposort()): return None

  def inverted(u:UOp) -> UOp|None:
    u = _unwrap(u)
    inverse_op = Ops.MUL if store.src[0].dtype in (dtypes.half, dtypes.float) else Ops.XOR
    if u.op is not inverse_op: return None
    for operand, inverse in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
      if inverse.op is Ops.CONST and int(inverse.arg) == -1: return _unwrap(operand)
    return None

  def extreme(u:UOp) -> tuple[str,tuple[UOp,UOp]]|None:
    u = _unwrap(u)
    if u.op is Ops.MAX and len(u.src) == 2:
      pair = tuple(_unwrap(x) for x in u.src)
      if all(x.op is Ops.INDEX and x.dtype is store.src[0].dtype and x.src[0].op is Ops.PARAM for x in pair):
        return "max", (pair[0], pair[1])
    if (maximum := inverted(u)) is not None and maximum.op is Ops.MAX and len(maximum.src) == 2:
      left, right = inverted(maximum.src[0]), inverted(maximum.src[1])
      if left is not None and right is not None and \
         all(x.op is Ops.INDEX and x.dtype is store.src[0].dtype and x.src[0].op is Ops.PARAM for x in (left, right)):
        return "min", (left, right)
    return None

  true_extreme, false_extreme = extreme(value.src[1]), extreme(value.src[2])
  if true_extreme is None or false_extreme is None or true_extreme[0] == false_extreme[0]: return None
  pair = true_extreme[1]
  if set(pair) != set(false_extreme[1]): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not loops or any(u.src[0].op is not Ops.CONST for u in loops): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  total = prod(_shape_of_store(sink))
  if prod(loop_extents) != total: return None
  if getenv("ROCKCHIP_DEBUG_ARGSORT"): print("RK_ARGSORT_COMPARE", total)

  def evaluate(u:UOp, coords:dict[UOp,int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    values = [evaluate(x, coords) for x in u.src]
    if any(x is Invalid for x in values): return Invalid
    if u.op is Ops.ADD: return values[0]+values[1]
    if u.op is Ops.MUL: return values[0]*values[1]
    if u.op is Ops.SUB: return values[0]-values[1]
    if u.op is Ops.FLOORDIV: return values[0]//values[1]
    if u.op is Ops.FLOORMOD: return values[0]%values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return all(bool(x) for x in values)
    if u.op is Ops.OR: return any(bool(x) for x in values)
    raise ValueError(u.op)

  mappings = ([-1]*total, [-1]*total)
  choose_max = [False]*total
  source_totals = tuple(int(x.src[0].src[0].arg) for x in pair)
  try:
    for output_linear in range(total):
      rem, coords = output_linear, {}
      for axis in range(len(loops)-1, -1, -1):
        rem, coord = divmod(rem, loop_extents[axis])
        coords[loops[axis]] = coord
      output_index = int(evaluate(store.src[0].src[1], coords))
      addresses = tuple(evaluate(x.src[1], coords) for x in pair)
      condition = evaluate(value.src[0], coords)
      if not 0 <= output_index < total or condition is Invalid or any(x is Invalid for x in addresses): return None
      if any(not 0 <= int(address) < source_total for address, source_total in zip(addresses, source_totals)): return None
      mappings[0][output_index], mappings[1][output_index] = int(addresses[0]), int(addresses[1])
      choose_max[output_index] = (true_extreme if bool(condition) else false_extreme)[0] == "max"
    if any(address < 0 for mapping in mappings for address in mapping): return None
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None

  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  source_slots = tuple(x.src[0].buf_uop.arg.slot for x in pair)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def alloc() -> int:
    nonlocal next_slot
    result, next_slot = next_slot, next_slot+1
    return result

  if store.src[0].dtype is dtypes.int:
    gathered_int:list[int] = []
    for source_slot, mapping in zip(source_slots, mappings):
      gathered_slot = alloc()
      gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, 4, *mapping)
      gather_relocs = (RKReloc(0, gathered_slot, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered_slot, is_copy=True), gather_relocs))
      gathered_int.append(gathered_slot)
    # WIP native-int references retained above in _try_native_int_min_subtasks:
    # a+b-max fails when INT_MIN padding requires wraparound, while arithmetic
    # -1-x did not reproduce XOR ordering in a chained sort.  Cross the
    # established ABI boundary once per operand and compare 0/1 plus +/-inf
    # padding on DPU instead.
    gathered_half:list[int] = []
    for source in gathered_int:
      converted = alloc()
      tasks.append(_emit_where_stage(total, converted, (source, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     int32_inputs=(source,)))
      gathered_half.append(converted)
    maximum = alloc()
    tasks.append(_emit_where_stage(total, maximum, (gathered_half[0], 0), (gathered_half[1], 0), Ops.MAX))
    negative_half:list[int] = []
    minus_one = (_CONST_SLOT, 0xbf800000)
    for operand in gathered_half:
      neg_warm, negated_slot = alloc(), alloc()
      tasks.append(_emit_where_stage(total, neg_warm, (operand, 0), minus_one, Ops.MUL))
      tasks.append(_emit_where_stage(total, negated_slot, (operand, 0), minus_one, Ops.MUL))
      negative_half.append(negated_slot)
    maximum_negative, minimum_warm, minimum = alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, maximum_negative, (negative_half[0], 0), (negative_half[1], 0), Ops.MAX))
    tasks.append(_emit_where_stage(total, minimum_warm, (maximum_negative, 0), minus_one, Ops.MUL))
    tasks.append(_emit_where_stage(total, minimum, (maximum_negative, 0), minus_one, Ops.MUL))
    selected_half = alloc()
    if all(choose_max) or not any(choose_max):
      selected = maximum if all(choose_max) else minimum
      tasks.append(_emit_where_stage(total, selected_half, (selected, 0), (_ZERO_SLOT, 0), Ops.ADD))
    else:
      select_layout = (total, _HOST_STATIC_SELECT_HALF_LAYOUT, *(int(selected) for selected in choose_max))
      select_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (selected_half, maximum, minimum))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", select_layout, selected_half, is_copy=True), select_relocs))
    convert_layout = (total, _HOST_HALF_INT_LAYOUT)
    convert_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, selected_half, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", convert_layout, out_slot, is_copy=True), convert_relocs))
    return tuple(tasks)

  float_gather_slot = alloc() if store.src[0].dtype is dtypes.float else None
  gathered:list[int] = []
  gathered_low:list[int] = []
  for source_slot, mapping in zip(source_slots, mappings):
    gathered_slot = float_gather_slot if float_gather_slot is not None else alloc()
    gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, 4 if float_gather_slot is not None else 2, *mapping)
    gather_relocs = (RKReloc(0, gathered_slot, 0, 0, 0xFFFFFFFF),
                     RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered_slot, is_copy=True), gather_relocs))
    if float_gather_slot is not None:
      converted = alloc()
      tasks.append(_emit_where_stage(total, converted, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     fp32_inputs=(gathered_slot,)))
      gathered.append(converted)
      converted_low = alloc()
      tasks.append(_emit_where_stage(total, converted_low, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     fp32_inputs=(gathered_slot,), fp32_residual_input=True))
      gathered_low.append(converted_low)
    else:
      gathered.append(gathered_slot)
  maximum = alloc()
  tasks.append(_emit_where_stage(total, maximum, (gathered[0], 0), (gathered[1], 0), Ops.MAX))
  negative:list[int] = []
  minus_one = (_CONST_SLOT, 0xbf800000)
  for operand in gathered:
    neg_warm, negated_slot = alloc(), alloc()
    tasks.append(_emit_where_stage(total, neg_warm, (operand, 0), minus_one, Ops.MUL))
    tasks.append(_emit_where_stage(total, negated_slot, (operand, 0), minus_one, Ops.MUL))
    negative.append(negated_slot)
  maximum_negative, minimum_warm, minimum = alloc(), alloc(), alloc()
  tasks.append(_emit_where_stage(total, maximum_negative, (negative[0], 0), (negative[1], 0), Ops.MAX))
  tasks.append(_emit_where_stage(total, minimum_warm, (maximum_negative, 0), minus_one, Ops.MUL))
  tasks.append(_emit_where_stage(total, minimum, (maximum_negative, 0), minus_one, Ops.MUL))

  selected_low = None
  if store.src[0].dtype is dtypes.float:
    # Compare the x256 residual only when the nearest-fp16 high limbs tie.
    # This preserves fp32 ordering for values which collapse to one high limb
    # while retaining MAX/MIN's safe handling of +/-inf sort padding.
    high_forward, high_reverse, high_distance = alloc(), alloc(), alloc()
    high_equal_warm, high_equal = alloc(), alloc()
    tasks.append(_emit_where_stage(total, high_forward, (gathered[0], 0), (gathered[1], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, high_reverse, (gathered[1], 0), (gathered[0], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, high_distance, (high_forward, 0), (high_reverse, 0), Ops.MAX))
    high_unequal = _emit_positive_mask(tasks, total, high_distance, alloc)
    one = (_CONST_SLOT, 0x3f800000)
    tasks.append(_emit_where_stage(total, high_equal_warm, one, (high_unequal, 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, high_equal, one, (high_unequal, 0), Ops.SUB))
    low_forward = alloc()
    high_greater = _emit_positive_mask(tasks, total, high_forward, alloc)
    tasks.append(_emit_where_stage(total, low_forward, (gathered_low[0], 0), (gathered_low[1], 0), Ops.SUB))
    low_greater = _emit_positive_mask(tasks, total, low_forward, alloc)
    low_decides, left_greater, inverse_left_greater = alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, low_decides, (high_equal, 0), (low_greater, 0), Ops.MUL))
    tasks.append(_emit_where_stage(total, left_greater, (high_greater, 0), (low_decides, 0), Ops.ADD))
    tasks.append(_emit_where_stage(total, inverse_left_greater, one, (left_greater, 0), Ops.SUB))
    if all(choose_max): select_left = left_greater
    elif not any(choose_max): select_left = inverse_left_greater
    else:
      select_left = alloc()
      select_layout = (total, _HOST_STATIC_SELECT_HALF_LAYOUT, *(int(selected) for selected in choose_max))
      select_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF)
                            for slot in (select_left, left_greater, inverse_left_greater))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", select_layout, select_left, is_copy=True), select_relocs))
    low_delta, weighted_low, selected_low = alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, low_delta, (gathered_low[0], 0), (gathered_low[1], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, weighted_low, (select_left, 0), (low_delta, 0), Ops.MUL))
    tasks.append(_emit_where_stage(total, selected_low, (gathered_low[1], 0), (weighted_low, 0), Ops.ADD))

  selected_slot = out_slot
  if all(choose_max) or not any(choose_max):
    selected = maximum if all(choose_max) else minimum
    if store.src[0].dtype is dtypes.float: selected_slot = selected
    else: tasks.append(_emit_where_stage(total, out_slot, (selected, 0), (_ZERO_SLOT, 0), Ops.ADD))
  else:
    # WIP reference: arithmetic selection used
    #   minimum + static_mask * (maximum-minimum)
    # but a padded sort lane can be -inf, making the unselected 0*inf term NaN.
    # The choice is compile-time wire topology, so interleave the already
    # NPU-computed extrema as representation-preserving layout work.
    selected_slot = alloc() if store.src[0].dtype is dtypes.float else out_slot
    select_layout = (total, _HOST_STATIC_SELECT_HALF_LAYOUT, *(int(selected) for selected in choose_max))
    select_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (selected_slot, maximum, minimum))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", select_layout, selected_slot, is_copy=True), select_relocs))
  if store.src[0].dtype is dtypes.float:
    assert selected_low is not None
    combine_layout = (total, _HOST_FP32_COMBINE_LAYOUT)
    combine_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, selected_slot, selected_low))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", combine_layout, out_slot, is_copy=True), combine_relocs))
  return tuple(tasks)

def _try_argsort_selected_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower argsort's final value/count match and coordinate selection to DPU masks."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.int or len(reductions) != 1: return None
  reduction = reductions[0]
  if reduction.arg[0] is not Ops.ADD or reduction.dtype is not dtypes.int: return None
  ranges = list(reduction.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not ranges or any(u.src[0].op is not Ops.CONST for u in (*loops, *ranges)): return None
  loop_extents, range_extents = [int(u.src[0].arg) for u in loops], [int(u.src[0].arg) for u in ranges]
  total, window = prod(_shape_of_store(sink)), prod(range_extents)
  if prod(loop_extents) != total or not 2 <= window <= 255: return None

  body = reduction.src[0]
  inequalities = [u for u in body.toposort() if u.op is Ops.CMPNE and len(u.src) == 2 and
                  all(_unwrap(x).op is Ops.INDEX for x in u.src)]
  if len(inequalities) != 2: return None
  equality_pairs:list[tuple[UOp,tuple[UOp,UOp]]] = []
  for inequality in inequalities:
    equalities = [u for u in body.toposort() if u.op is Ops.CMPNE and inequality in u.src and
                  any(x.op is Ops.CONST and x.dtype is dtypes.bool and bool(x.arg) for x in u.src)]
    if len(equalities) != 1: return None
    pair = tuple(_unwrap(x) for x in inequality.src)
    if pair[0].dtype is not pair[1].dtype or pair[0].dtype not in (dtypes.half, dtypes.float, dtypes.int) or \
       any(x.src[0].op is not Ops.PARAM for x in pair): return None
    equality_pairs.append((equalities[0], (pair[0], pair[1])))
  pair_dtypes = tuple(pair[0].dtype for _, pair in equality_pairs)
  if set(pair_dtypes) in ({dtypes.half, dtypes.int}, {dtypes.float, dtypes.int}):
    value_pair_index = next(i for i, dtype in enumerate(pair_dtypes) if dtype is not dtypes.int)
  elif pair_dtypes == (dtypes.int, dtypes.int): value_pair_index = 0
  else: return None
  static_body = body.substitute({equality:UOp.const(dtypes.bool, True) for equality, _ in equality_pairs})
  if any(u.op is Ops.INDEX for u in static_body.toposort()): return None
  if getenv("ROCKCHIP_DEBUG_ARGSORT"): print("RK_ARGSORT_SELECTED", total, window)

  def evaluate(u:UOp, coords:dict[UOp,int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    values = [evaluate(x, coords) for x in u.src]
    if any(x is Invalid for x in values): return Invalid
    if u.op is Ops.ADD: return values[0]+values[1]
    if u.op is Ops.MUL: return values[0]*values[1]
    if u.op is Ops.SUB: return values[0]-values[1]
    if u.op is Ops.FLOORDIV: return values[0]//values[1]
    if u.op is Ops.FLOORMOD: return values[0]%values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return all(bool(x) for x in values)
    if u.op is Ops.OR: return any(bool(x) for x in values)
    if u.op is Ops.WHERE: return values[1] if values[0] else values[2]
    raise ValueError(u.op)

  flat_indexes = tuple(index for _, pair in equality_pairs for index in pair)
  source_totals = tuple(int(x.src[0].src[0].arg) for x in flat_indexes)
  mappings:list[tuple[tuple[tuple[int,...],...],tuple[int,...]]] = []
  try:
    for candidate in range(window):
      rem, fixed = candidate, {}
      for axis in range(len(ranges)-1, -1, -1):
        rem, coord = divmod(rem, range_extents[axis])
        fixed[ranges[axis]] = coord
      addresses = [[-1]*total for _ in flat_indexes]
      weights = [-1]*total
      for output_linear in range(total):
        rem, coords = output_linear, dict(fixed)
        for axis in range(len(loops)-1, -1, -1):
          rem, coord = divmod(rem, loop_extents[axis])
          coords[loops[axis]] = coord
        output_index = int(evaluate(store.src[0].src[1], coords))
        source_addresses = tuple(evaluate(x.src[1], coords) for x in flat_indexes)
        weight = evaluate(static_body, coords)
        if not 0 <= output_index < total or weight is Invalid or any(x is Invalid for x in source_addresses): return None
        if any(not 0 <= int(address) < source_total for address, source_total in zip(source_addresses, source_totals)): return None
        if not 0 <= int(weight) <= 255: return None
        for mapping, address in zip(addresses, source_addresses): mapping[output_index] = int(address)
        weights[output_index] = int(weight)
      if any(address < 0 for mapping in addresses for address in mapping) or any(weight < 0 for weight in weights): return None
      mappings.append((tuple(tuple(mapping) for mapping in addresses), tuple(weights)))
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None

  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  source_slots = tuple(x.src[0].buf_uop.arg.slot for x in flat_indexes)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def alloc() -> int:
    nonlocal next_slot
    result, next_slot = next_slot, next_slot+1
    return result

  def native_int_to_half(source_slot:int, source_total:int) -> int:
    result = alloc()
    if not getenv("ROCKCHIP_NATIVE_ARGSORT_PACK"):
      tasks.append(_emit_where_stage(source_total, result, (source_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     int32_inputs=(source_slot,)))
      return result
    # WIP reference: exact four-lane native conversion is retained for
    # targeted probes, but its per-chunk RESET lifecycle is not suite-stable.
    for start in range(0, source_total, 4):
      count = min(4, source_total-start)
      packed = alloc()
      pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      half_atom = alloc()
      tasks.append(_emit_where_stage(4, half_atom, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_input=True))
      unpack_layout = (count, _HOST_UNPACK_HALF_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, result, 0, 0, 0xFFFFFFFF), RKReloc(0, half_atom, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, result, is_copy=True), unpack_relocs))
    return result

  # Keep the reusable fp32 gather in the first available global slot.  The
  # version-4 task header can encode fp32 input typing only for slots 0..6;
  # native-int conversion scratch below may otherwise push it out of range.
  float_gather_slot = alloc() if any(index.dtype is dtypes.float for index in flat_indexes) else None
  converted_int_slots:dict[int,int] = {}
  for index, source_slot, source_total in zip(flat_indexes, source_slots, source_totals):
    if index.dtype is dtypes.int and source_slot not in converted_int_slots:
      converted_int_slots[source_slot] = native_int_to_half(source_slot, source_total)

  scores:list[tuple[int,int]] = []
  one = (_CONST_SLOT, 0x3f800000)
  for selected_maps, candidate_weights in mappings:
    gathered:list[int] = []
    gathered_low:list[int|None] = []
    for index, source_slot, address_map in zip(flat_indexes, source_slots, selected_maps):
      effective_slot = converted_int_slots[source_slot] if index.dtype is dtypes.int else source_slot
      if index.dtype is dtypes.float:
        assert float_gather_slot is not None
        gathered_slot = float_gather_slot
      else: gathered_slot = alloc()
      gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, 4 if index.dtype is dtypes.float else 2, *address_map)
      gather_relocs = (RKReloc(0, gathered_slot, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, effective_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered_slot, is_copy=True), gather_relocs))
      if index.dtype is dtypes.float:
        converted_float = alloc()
        tasks.append(_emit_where_stage(total, converted_float, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                       fp32_inputs=(gathered_slot,)))
        gathered.append(converted_float)
        converted_low = alloc()
        tasks.append(_emit_where_stage(total, converted_low, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                       fp32_inputs=(gathered_slot,), fp32_residual_input=True))
        gathered_low.append(converted_low)
      else:
        gathered.append(gathered_slot)
        gathered_low.append(None)
    value_distance, count_equal = None, None
    for pair_index, (left, right) in enumerate(((gathered[0], gathered[1]), (gathered[2], gathered[3]))):
      forward, reverse, distance = alloc(), alloc(), alloc()
      tasks.append(_emit_where_stage(total, forward, (left, 0), (right, 0), Ops.SUB))
      tasks.append(_emit_where_stage(total, reverse, (right, 0), (left, 0), Ops.SUB))
      tasks.append(_emit_where_stage(total, distance, (forward, 0), (reverse, 0), Ops.MAX))
      if pair_index == value_pair_index:
        if pair_dtypes[pair_index] is dtypes.float:
          low_left, low_right = gathered_low[pair_index*2:pair_index*2+2]
          assert low_left is not None and low_right is not None
          low_forward, low_reverse, low_distance = alloc(), alloc(), alloc()
          scaled_low, combined_distance = alloc(), alloc()
          tasks.append(_emit_where_stage(total, low_forward, (low_left, 0), (low_right, 0), Ops.SUB))
          tasks.append(_emit_where_stage(total, low_reverse, (low_right, 0), (low_left, 0), Ops.SUB))
          tasks.append(_emit_where_stage(total, low_distance, (low_forward, 0), (low_reverse, 0), Ops.MAX))
          tasks.append(_emit_where_stage(total, scaled_low, (low_distance, 0),
                                         (_CONST_SLOT, 0x3b800000), Ops.MUL))
          tasks.append(_emit_where_stage(total, combined_distance, (distance, 0), (scaled_low, 0), Ops.ADD))
          distance = combined_distance
        value_distance = distance
      else:
        equal_warm, equal = alloc(), alloc()
        unequal = _emit_positive_mask(tasks, total, distance, alloc)
        tasks.append(_emit_where_stage(total, equal_warm, one, (unequal, 0), Ops.SUB))
        tasks.append(_emit_where_stage(total, equal, one, (unequal, 0), Ops.SUB))
        count_equal = equal
    if value_distance is None or count_equal is None: return None

    # WIP reference: exact value equality originally multiplied the value and
    # occurrence-count equality masks, then summed candidate*mask.  DPU
    # compare/swap preserves ordering but can move a copied fp16 value a few
    # ULPs, so that exact graph loses the source identity.  Select the
    # count-compatible original value with the closest DPU absolute distance.
    inverse_count, negative_distance_warm, negative_distance = alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, inverse_count, one, (count_equal, 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, negative_distance_warm, (value_distance, 0),
                                   (_CONST_SLOT, 0xbf800000), Ops.MUL))
    tasks.append(_emit_where_stage(total, negative_distance, (value_distance, 0),
                                   (_CONST_SLOT, 0xbf800000), Ops.MUL))
    penalty_warm, penalty, score = alloc(), alloc(), alloc()
    finite_min = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -65504.0))[0])
    tasks.append(_emit_where_stage(total, penalty_warm, (inverse_count, 0), finite_min, Ops.MUL))
    tasks.append(_emit_where_stage(total, penalty, (inverse_count, 0), finite_min, Ops.MUL))
    tasks.append(_emit_where_stage(total, score, (negative_distance, 0), (penalty, 0), Ops.ADD))
    weight_slot = alloc()
    weight_layout = (total, _HOST_STATIC_HALF_LAYOUT,
                     *(struct.unpack('<H', struct.pack('<e', float(weight)))[0] for weight in candidate_weights))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", weight_layout, weight_slot, is_copy=True),
                           (RKReloc(0, weight_slot, 0, 0, 0xFFFFFFFF),)))
    scores.append((score, weight_slot))
  if not scores: return None

  best, result = scores[0]
  for score, weight_slot in scores[1:]:
    difference, next_best = alloc(), alloc()
    tasks.append(_emit_where_stage(total, difference, (score, 0), (best, 0), Ops.SUB))
    greater = _emit_positive_mask(tasks, total, difference, alloc)
    tasks.append(_emit_where_stage(total, next_best, (best, 0), (score, 0), Ops.MAX))
    delta, weighted, selected = alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, delta, (weight_slot, 0), (result, 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, weighted, (greater, 0), (delta, 0), Ops.MUL))
    tasks.append(_emit_where_stage(total, selected, (result, 0), (weighted, 0), Ops.ADD))
    best, result = next_best, selected
  if not getenv("ROCKCHIP_NATIVE_ARGSORT_PACK"):
    # The NPU result contains exact small integral fp16 weights. Convert only
    # their ABI representation here; comparison and selection stay on NPU.
    convert_layout = (total, _HOST_HALF_INT_LAYOUT)
    convert_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, result, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", convert_layout, out_slot, is_copy=True), convert_relocs))
    return tuple(tasks)
  # WIP reference: native four-lane conversion is retained behind
  # ROCKCHIP_NATIVE_ARGSORT_PACK=1. Even eight-task PC batches require enough
  # RESET transitions to exhaust the driver in the full sort matrix.
  pack_tasks:list[RKSubTask] = []
  native_tasks:list[RKSubTask] = []
  assemble_tasks:list[RKSubTask] = []
  for start in range(0, total, 4):
    count = min(4, total-start)
    packed, native = alloc(), alloc()
    pack_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
    pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, result, 0, 0, 0xFFFFFFFF))
    pack_tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
    native_tasks.append(_emit_where_stage(4, native, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
    assemble_layout = (count, _HOST_ASSEMBLE_INT_BYTES_LAYOUT, start, 1)
    assemble_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, native, 0, 0, 0xFFFFFFFF))
    assemble_tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", assemble_layout, out_slot, is_copy=True), assemble_relocs))
  tasks.extend((*pack_tasks, *native_tasks, *assemble_tasks))
  return tuple(tasks)

def _try_argsort_index_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower argsort's stable equality-count index reduction to DPU masks and sums."""
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype is not dtypes.int or len(reductions) != 1: return None
  reduction = reductions[0]
  if reduction.arg[0] is not Ops.ADD or reduction.dtype is not dtypes.int: return None
  ranges = list(reduction.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not ranges or any(u.src[0].op is not Ops.CONST for u in (*loops, *ranges)): return None
  loop_extents, range_extents = [int(u.src[0].arg) for u in loops], [int(u.src[0].arg) for u in ranges]
  total, window = prod(_shape_of_store(sink)), prod(range_extents)
  if prod(loop_extents) != total or not 2 <= window <= 255: return None

  body = reduction.src[0]
  inequalities = [u for u in body.toposort() if u.op is Ops.CMPNE and len(u.src) == 2 and
                  all(_unwrap(x).op is Ops.INDEX for x in u.src)]
  if len(inequalities) != 1: return None
  inequality = inequalities[0]
  equalities = [u for u in body.toposort() if u.op is Ops.CMPNE and inequality in u.src and
                any(x.op is Ops.CONST and x.dtype is dtypes.bool and bool(x.arg) for x in u.src)]
  if len(equalities) != 1: return None
  equality = equalities[0]
  indexes = tuple(_unwrap(x) for x in inequality.src)
  if indexes[0].dtype is not indexes[1].dtype or indexes[0].dtype not in (dtypes.half, dtypes.float, dtypes.int) or \
     any(x.src[0].op is not Ops.PARAM for x in indexes): return None
  static_body = body.substitute({equality:UOp.const(dtypes.bool, True)})
  if any(u.op is Ops.INDEX for u in static_body.toposort()): return None
  if getenv("ROCKCHIP_DEBUG_ARGSORT"): print("RK_ARGSORT_INDEX", total, window)

  def evaluate(u:UOp, coords:dict[UOp,int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    values = [evaluate(x, coords) for x in u.src]
    if any(x is Invalid for x in values): return Invalid
    if u.op is Ops.ADD: return values[0]+values[1]
    if u.op is Ops.MUL: return values[0]*values[1]
    if u.op is Ops.SUB: return values[0]-values[1]
    if u.op is Ops.FLOORDIV: return values[0]//values[1]
    if u.op is Ops.FLOORMOD: return values[0]%values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return all(bool(x) for x in values)
    if u.op is Ops.OR: return any(bool(x) for x in values)
    if u.op is Ops.WHERE: return values[1] if values[0] else values[2]
    raise ValueError(u.op)

  mappings:list[tuple[tuple[int,...],tuple[int,...],tuple[bool,...]]] = []
  source_totals = tuple(int(x.src[0].src[0].arg) for x in indexes)
  try:
    for candidate in range(window):
      rem, fixed = candidate, {}
      for axis in range(len(ranges)-1, -1, -1):
        rem, coord = divmod(rem, range_extents[axis])
        fixed[ranges[axis]] = coord
      left, right, valid = [-1]*total, [-1]*total, [False]*total
      for output_linear in range(total):
        rem, coords = output_linear, dict(fixed)
        for axis in range(len(loops)-1, -1, -1):
          rem, coord = divmod(rem, loop_extents[axis])
          coords[loops[axis]] = coord
        output_index = int(evaluate(store.src[0].src[1], coords))
        addresses = tuple(evaluate(x.src[1], coords) for x in indexes)
        active = evaluate(static_body, coords)
        if not 0 <= output_index < total or active is Invalid or any(x is Invalid for x in addresses): return None
        if any(not 0 <= int(address) < source_total for address, source_total in zip(addresses, source_totals)): return None
        left[output_index], right[output_index], valid[output_index] = int(addresses[0]), int(addresses[1]), bool(active)
      if any(address < 0 for address in (*left, *right)): return None
      mappings.append((tuple(left), tuple(right), tuple(valid)))
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None

  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  source_slots = tuple(x.src[0].buf_uop.arg.slot for x in indexes)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def alloc() -> int:
    nonlocal next_slot
    result, next_slot = next_slot, next_slot+1
    return result

  def native_int_to_half(source_slot:int, source_total:int) -> int:
    result = alloc()
    for start in range(0, source_total, 4):
      count = min(4, source_total-start)
      packed = alloc()
      pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      half_atom = alloc()
      tasks.append(_emit_where_stage(4, half_atom, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_input=True))
      unpack_layout = (count, _HOST_UNPACK_HALF_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, result, 0, 0, 0xFFFFFFFF), RKReloc(0, half_atom, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, result, is_copy=True), unpack_relocs))
    return result

  effective_slots = source_slots
  if indexes[0].dtype is dtypes.int:
    converted:dict[int,int] = {}
    for source_slot, source_total in zip(source_slots, source_totals):
      if source_slot not in converted: converted[source_slot] = native_int_to_half(source_slot, source_total)
    effective_slots = tuple(converted[source_slot] for source_slot in source_slots)
  float_gather_slot = alloc() if indexes[0].dtype is dtypes.float else None

  masks:list[int] = []
  one = (_CONST_SLOT, 0x3f800000)
  for left_map, right_map, valid_map in mappings:
    if not any(valid_map): continue
    gathered:list[int] = []
    gathered_low:list[int] = []
    for source_slot, mapping in zip(effective_slots, (left_map, right_map)):
      gathered_slot = float_gather_slot if float_gather_slot is not None else alloc()
      gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, 4 if float_gather_slot is not None else 2, *mapping)
      gather_relocs = (RKReloc(0, gathered_slot, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered_slot, is_copy=True), gather_relocs))
      if float_gather_slot is not None:
        converted_float = alloc()
        tasks.append(_emit_where_stage(total, converted_float, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                       fp32_inputs=(gathered_slot,)))
        gathered.append(converted_float)
        converted_low = alloc()
        tasks.append(_emit_where_stage(total, converted_low, (gathered_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                       fp32_inputs=(gathered_slot,), fp32_residual_input=True))
        gathered_low.append(converted_low)
      else:
        gathered.append(gathered_slot)
    forward, reverse, distance, equal_warm, equal = (alloc() for _ in range(5))
    tasks.append(_emit_where_stage(total, forward, (gathered[0], 0), (gathered[1], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, reverse, (gathered[1], 0), (gathered[0], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, distance, (forward, 0), (reverse, 0), Ops.MAX))
    unequal = _emit_positive_mask(tasks, total, distance, alloc)
    tasks.append(_emit_where_stage(total, equal_warm, one, (unequal, 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, equal, one, (unequal, 0), Ops.SUB))
    if indexes[0].dtype is dtypes.float:
      low_forward, low_reverse, low_distance = alloc(), alloc(), alloc()
      low_equal_warm, low_equal = alloc(), alloc()
      tasks.append(_emit_where_stage(total, low_forward, (gathered_low[0], 0), (gathered_low[1], 0), Ops.SUB))
      tasks.append(_emit_where_stage(total, low_reverse, (gathered_low[1], 0), (gathered_low[0], 0), Ops.SUB))
      tasks.append(_emit_where_stage(total, low_distance, (low_forward, 0), (low_reverse, 0), Ops.MAX))
      low_unequal = _emit_positive_mask(tasks, total, low_distance, alloc)
      tasks.append(_emit_where_stage(total, low_equal_warm, one, (low_unequal, 0), Ops.SUB))
      tasks.append(_emit_where_stage(total, low_equal, one, (low_unequal, 0), Ops.SUB))
      full_equal = alloc()
      tasks.append(_emit_where_stage(total, full_equal, (equal, 0), (low_equal, 0), Ops.MUL))
      equal = full_equal
    if not all(valid_map):
      valid_slot = alloc()
      valid_layout = (total, _HOST_STATIC_HALF_LAYOUT,
                      *(0x3c00 if is_valid else 0 for is_valid in valid_map))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", valid_layout, valid_slot, is_copy=True),
                             (RKReloc(0, valid_slot, 0, 0, 0xFFFFFFFF),)))
      valid_warm, valid_equal = alloc(), alloc()
      tasks.append(_emit_where_stage(total, valid_warm, (equal, 0), (valid_slot, 0), Ops.MUL))
      tasks.append(_emit_where_stage(total, valid_equal, (equal, 0), (valid_slot, 0), Ops.MUL))
      equal = valid_equal
    masks.append(equal)
  if not masks: return None

  result = masks[0]
  for mask in masks[1:]:
    summed = alloc()
    tasks.append(_emit_where_stage(total, summed, (result, 0), (mask, 0), Ops.ADD))
    result = summed
  # The stable occurrence count is in [0, window], so one exact low byte is
  # sufficient for the supported <=255-candidate argsort reduction.
  if not getenv("ROCKCHIP_NATIVE_ARGSORT_PACK"):
    convert_layout = (total, _HOST_HALF_INT_LAYOUT)
    convert_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, result, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", convert_layout, out_slot, is_copy=True), convert_relocs))
    return tuple(tasks)
  # WIP reference: retain the native byte assembly experiment for targeted
  # hardware probes; normal execution uses the established typed ABI above.
  pack_tasks:list[RKSubTask] = []
  native_tasks:list[RKSubTask] = []
  assemble_tasks:list[RKSubTask] = []
  for start in range(0, total, 4):
    count = min(4, total-start)
    packed, native = alloc(), alloc()
    pack_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
    pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, result, 0, 0, 0xFFFFFFFF))
    pack_tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
    native_tasks.append(_emit_where_stage(4, native, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
    assemble_layout = (count, _HOST_ASSEMBLE_INT_BYTES_LAYOUT, start, 1)
    assemble_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, native, 0, 0, 0xFFFFFFFF))
    assemble_tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", assemble_layout, out_slot, is_copy=True), assemble_relocs))
  tasks.extend((*pack_tasks, *native_tasks, *assemble_tasks))
  return tuple(tasks)

def _try_arg_extrema_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower general static-axis argmax/argmin with DPU comparison and index selection."""
  store = _store_node(sink)
  max_reduces = [u for u in sink.toposort() if u.op is Ops.REDUCE and u.arg[0] is Ops.MAX]
  if store is None or store.src[0].dtype is not dtypes.int or len(max_reduces) not in (1, 2): return None
  selected_reduce = max_reduces[0] if len(max_reduces) == 1 and max_reduces[0].dtype is dtypes.int else None
  value_reduce = None if selected_reduce is not None else next((u for u in max_reduces if
    sum(x.op is Ops.INDEX and x.dtype in (dtypes.bool, dtypes.half, dtypes.float, dtypes.int)
        for x in u.src[0].toposort()) == 1), None)
  if selected_reduce is None and value_reduce is None: return None
  reductions = list((selected_reduce if selected_reduce is not None else value_reduce).src[1:])  # type: ignore[union-attr]
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  reduce_extents = [int(u.src[0].arg) for u in reductions]
  total, window = prod(_shape_of_store(sink)), prod(reduce_extents)
  if prod(loop_extents) != total or not 2 <= window <= 65536: return None
  extrema_source = None
  if selected_reduce is not None:
    # Normal-fp32 axis ArgMax/ArgMin materializes the extrema in a preceding
    # kernel. Identify the original candidate tensor by its reduction-dependent
    # address and the saved extrema by its total-element storage.
    indexes = [u for u in selected_reduce.src[0].toposort()
               if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM]
    sources = [u for u in indexes if any(axis in u.src[1].toposort() for axis in reductions)]
    extrema = [u for u in indexes if not any(axis in u.src[1].toposort() for axis in reductions) and
               int(u.src[0].src[0].arg) == total]
    if len(sources) != 1 or len(extrema) != 1: return None
    source, extrema_source, body = sources[0], extrema[0], sources[0]
    is_min = any(u.op is Ops.MUL and source in u.src and
                 any(x.op is Ops.CONST and float(x.arg) == -1.0 for x in u.src)
                 for u in selected_reduce.src[0].toposort())
  else:
    assert value_reduce is not None
    body = _unwrap(value_reduce.src[0])
    sources = [u for u in body.toposort() if u.op is Ops.INDEX and u.dtype in (dtypes.bool, dtypes.half, dtypes.float, dtypes.int)]
    if len(sources) != 1 or (source := sources[0]).src[0].op is not Ops.PARAM: return None
    is_min = any(u.op in (Ops.MUL, Ops.XOR) and source in u.src and
                 any(x.op is Ops.CONST and float(x.arg) == -1.0 for x in u.src) for u in body.toposort())
    is_min = is_min or (source.dtype is dtypes.bool and any(
      u.op is Ops.CMPNE and source in u.src and any(x.op is Ops.CONST and bool(x.arg) for x in u.src)
      for u in body.toposort()))
    if body is not source and not is_min: return None
  if getenv("ROCKCHIP_DEBUG_ARG_EXTREMA"):
    print("RK_ARG_EXTREMA", "selected" if extrema_source is not None else ("min" if is_min else "max"), source.dtype, total, window)

  def evaluate(u:UOp, coords:dict[UOp,int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    if u.op is Ops.WHERE:
      condition = evaluate(u.src[0], coords)
      return evaluate(u.src[1] if condition else u.src[2], coords)
    values = [evaluate(x, coords) for x in u.src]
    if any(x is Invalid for x in values): return Invalid
    if u.op is Ops.ADD: return values[0]+values[1]
    if u.op is Ops.MUL: return values[0]*values[1]
    if u.op is Ops.SUB: return values[0]-values[1]
    if u.op is Ops.FLOORDIV: return values[0]//values[1]
    if u.op is Ops.FLOORMOD: return values[0]%values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
    if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
    if u.op is Ops.NEG: return -values[0]
    raise ValueError(u.op)

  scheduled_index = None
  if selected_reduce is not None:
    selected_body = _unwrap(selected_reduce.src[0])
    if selected_body.op is Ops.MUL:
      scheduled_index = next((operand for operand in selected_body.src
                              if not any(u.op is Ops.INDEX for u in operand.toposort()) and
                              any(axis in operand.toposort() for axis in reductions)), None)
  # Max-pool return_indices reduces the two window axes and promises an index
  # into the unpadded spatial plane, not a window-local candidate number.  The
  # selected source address already contains that information even when the
  # scheduled index expression has padding compaction REDUCEs that the static
  # evaluator intentionally does not execute.
  output_shape = _shape_of_store(sink)
  pool_input_spatial = None
  if selected_reduce is not None and len(reductions) == 2 and (len(output_shape) >= 2 or total == 1):
    # A one-output pool has no surviving LOOP axes and therefore appears as
    # shape (1,); its two window REDUCE axes still identify one spatial plane.
    planes = prod(output_shape[:-2]) if len(output_shape) > 2 else 1
    source_total = int(source.src[0].src[0].arg)
    if planes > 0 and source_total % planes == 0: pool_input_spatial = source_total // planes
  mappings:list[tuple[int,...]] = []
  public_indices:list[tuple[int,...]] = []
  try:
    for candidate in range(window):
      rem, fixed = candidate, {}
      for axis in range(len(reductions)-1, -1, -1):
        rem, coord = divmod(rem, reduce_extents[axis])
        fixed[reductions[axis]] = coord
      addresses = [-1]*total
      for output_linear in range(total):
        rem, coords = output_linear, dict(fixed)
        for axis in range(len(loops)-1, -1, -1):
          rem, coord = divmod(rem, loop_extents[axis])
          coords[loops[axis]] = coord
        output_index = int(evaluate(store.src[0].src[1], coords))
        source_index = evaluate(source.src[1], coords)
        if not 0 <= output_index < total: return None
        addresses[output_index] = -1 if source_index is Invalid else int(source_index)
      indices = [candidate]*total
      if pool_input_spatial is not None:
        indices = [0 if address < 0 else address % pool_input_spatial for address in addresses]
      elif scheduled_index is not None and selected_reduce is not None:
        for output_linear in range(total):
          rem, coords = output_linear, dict(fixed)
          for axis in range(len(loops)-1, -1, -1):
            rem, coord = divmod(rem, loop_extents[axis])
            coords[loops[axis]] = coord
          output_index = int(evaluate(store.src[0].src[1], coords))
          if addresses[output_index] < 0:
            indices[output_index] = 0
            continue
          encoded = int(evaluate(scheduled_index, coords))
          decoded = store.src[1].substitute({selected_reduce:UOp.const(selected_reduce.dtype, encoded)})
          indices[output_index] = int(evaluate(decoded, coords))
      mappings.append(tuple(addresses))
      public_indices.append(tuple(indices))
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None

  info = ProgramInfo.from_sink(sink)
  out_slot, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  extrema_source_slot = extrema_source.src[0].buf_uop.arg.slot if extrema_source is not None else None
  if pool_input_spatial is not None and extrema_source_slot is not None:
    # Comparing fp32 candidates after conversion to half can create false ties
    # and move an otherwise-correct maximum to the wrong unpool position.
    # Keep normal-fp32 return_indices exact at the typed operator boundary.
    output_major_mapping = tuple(mappings[candidate][output] for output in range(total) for candidate in range(window))
    layout = (total, _HOST_ARGMAX_LAYOUT, window, pool_input_spatial, 4, *output_major_mapping)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, source_slot, extrema_source_slot))
    return (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True), relocs),)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def alloc() -> int:
    nonlocal next_slot
    result, next_slot = next_slot, next_slot+1
    return result

  def native_int_to_half(input_slot:int) -> int:
    if is_min:
      result = alloc()
      tasks.append(_emit_where_stage(total, result, (input_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     int32_inputs=(input_slot,)))
      return result
    result = alloc()
    for start in range(0, total, 4):
      count = min(4, total-start)
      packed = alloc()
      pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, input_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      half_atom = alloc()
      tasks.append(_emit_where_stage(4, half_atom, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_input=True))
      unpack_layout = (count, _HOST_UNPACK_HALF_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, result, 0, 0, 0xFFFFFFFF), RKReloc(0, half_atom, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, result, is_copy=True), unpack_relocs))
    return result

  gathered_source_slot, gathered_itemsize = source_slot, source.dtype.itemsize
  if source.dtype is dtypes.bool:
    gathered_source_slot, gathered_itemsize = alloc(), 2
    widen_layout = (int(source.src[0].src[0].arg), _HOST_BOOL_HALF_LAYOUT)
    widen_relocs = (RKReloc(0, gathered_source_slot, 0, 0, 0xFFFFFFFF),
                     RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", widen_layout, gathered_source_slot, is_copy=True), widen_relocs))
  elif source.dtype is dtypes.int and is_min:
    gathered_source_slot, gathered_itemsize = alloc(), 2
    tasks.append(_emit_where_stage(int(source.src[0].src[0].arg), gathered_source_slot, (source_slot, 0),
                                   (_ZERO_SLOT, 0), Ops.ADD, int32_inputs=(source_slot,)))

  # Version-4 task metadata encodes typed inputs only for low global slots.
  # Reuse one gather slot sequentially; the ordered runtime flushes a pending
  # conversion before the next host gather overwrites this input.
  fp32_gather_slot = alloc() if source.dtype is dtypes.float else None
  candidate_slots:list[int] = []
  valid_slots:list[int|None] = []
  for mapping in mappings:
    gathered = fp32_gather_slot if fp32_gather_slot is not None else alloc()
    gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, gathered_itemsize, *mapping)
    gather_relocs = (RKReloc(0, gathered, 0, 0, 0xFFFFFFFF),
                     RKReloc(0, gathered_source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered, is_copy=True), gather_relocs))
    candidate = native_int_to_half(gathered) if source.dtype is dtypes.int and not is_min else gathered
    if source.dtype is dtypes.float:
      converted = alloc()
      tasks.append(_emit_where_stage(total, converted, (gathered, 0), (_ZERO_SLOT, 0), Ops.ADD, fp32_inputs=(gathered,)))
      candidate = converted
    if is_min:
      warm, negated = alloc(), alloc()
      tasks.append(_emit_where_stage(total, warm, (candidate, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL))
      tasks.append(_emit_where_stage(total, negated, (candidate, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL))
      candidate = negated
      if source.dtype is dtypes.int:
        lower, negative, upper_negative, clamped = alloc(), alloc(), alloc(), alloc()
        finite_min = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', -65504.0))[0])
        tasks.append(_emit_where_stage(total, lower, (candidate, 0), finite_min, Ops.MAX))
        tasks.append(_emit_where_stage(total, negative, (lower, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL))
        tasks.append(_emit_where_stage(total, upper_negative, (negative, 0), finite_min, Ops.MAX))
        tasks.append(_emit_where_stage(total, clamped, (upper_negative, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL))
        candidate = clamped
    candidate_slots.append(candidate)
    if any(address < 0 for address in mapping):
      valid = alloc()
      valid_layout = (total, _HOST_STATIC_HALF_LAYOUT, *(0x0000 if address < 0 else 0x3c00 for address in mapping))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", valid_layout, valid, is_copy=True),
                             (RKReloc(0, valid, 0, 0, 0xFFFFFFFF),)))
      valid_slots.append(valid)
    else: valid_slots.append(None)

  if extrema_source_slot is not None:
    maximum = alloc()
    tasks.append(_emit_where_stage(total, maximum, (extrema_source_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                   fp32_inputs=(extrema_source_slot,)))
  else:
    maximum = candidate_slots[0]
    for operand in candidate_slots[1:]:
      result = alloc()
      tasks.append(_emit_where_stage(total, result, (maximum, 0), (operand, 0), Ops.MAX))
      maximum = result

  max_public_index = max(index for indices in public_indices for index in indices)
  if max_public_index < 0: return None
  index_bytes = max(1, (max_public_index.bit_length()+7)//8)
  if index_bytes > 4: return None
  selected:list[tuple[int,int]] = [(_ZERO_SLOT, 0)]*index_bytes
  one = (_CONST_SLOT, 0x3f800000)
  # Visit backwards so an earlier equal value overwrites a later one.
  for candidate in range(window-1, -1, -1):
    diff, less, equal_warm, equal = alloc(), alloc(), alloc(), alloc()
    tasks.append(_emit_where_stage(total, diff, (maximum, 0), (candidate_slots[candidate], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, less, (diff, 0), (diff, 0), Ops.MAX, compare=True))
    tasks.append(_emit_where_stage(total, equal_warm, one, (less, 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, equal, one, (less, 0), Ops.SUB))
    if (valid_slot := valid_slots[candidate]) is not None:
      valid_equal = alloc()
      tasks.append(_emit_where_stage(total, valid_equal, (equal, 0), (valid_slot, 0), Ops.MUL))
      equal = valid_equal
    for byte in range(index_bytes):
      index_slot = alloc()
      index_bits = tuple(struct.unpack('<H', struct.pack('<e', float((index >> (8*byte)) & 0xFF)))[0]
                         for index in public_indices[candidate])
      index_layout = (total, _HOST_STATIC_HALF_LAYOUT, *index_bits)
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", index_layout, index_slot, is_copy=True),
                             (RKReloc(0, index_slot, 0, 0, 0xFFFFFFFF),)))
      delta, weighted, selected_out = alloc(), alloc(), alloc()
      tasks.append(_emit_where_stage(total, delta, (index_slot, 0), selected[byte], Ops.SUB))
      tasks.append(_emit_where_stage(total, weighted, (equal, 0), (delta, 0), Ops.MUL))
      tasks.append(_emit_where_stage(total, selected_out, selected[byte], (weighted, 0), Ops.ADD))
      selected[byte] = (selected_out, 0)

  for start in range(0, total, 4):
    count = min(4, total-start)
    native_digits:list[int] = []
    for byte in range(index_bytes):
      packed = alloc()
      pack_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, selected[byte][0], 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      native = alloc()
      tasks.append(_emit_where_stage(4, native, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
      native_digits.append(native)
    assemble_layout = (count, _HOST_ASSEMBLE_INT_BYTES_LAYOUT, start, index_bytes)
    assemble_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, *native_digits))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", assemble_layout, out_slot, is_copy=True), assemble_relocs))
  return tuple(tasks)

def _try_pool_index_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Compile max-pool's static candidate addresses without unrolling its argmax expression."""
  store = _store_node(sink)
  max_reduce = next((u for u in sink.toposort() if u.op is Ops.REDUCE and u.arg[0] is Ops.MAX), None)
  if store is None or store.src[0].dtype is not dtypes.int or max_reduce is None: return None
  reductions = list(max_reduce.src[1:])
  if not reductions or any(u.src[0].op is not Ops.CONST for u in reductions): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if any(u.src[0].op is not Ops.CONST for u in loops): return None
  total, window = prod(_shape_of_store(sink)), prod(int(u.src[0].arg) for u in reductions)
  if prod(int(u.src[0].arg) for u in loops) != total: return None
  value_indexes = [u for u in max_reduce.toposort() if u.op is Ops.INDEX and u.dtype in (dtypes.half, dtypes.int)]
  data = next((u for u in value_indexes if any(r in set(u.src[1].toposort()) for r in reductions)), None)
  maximum = next((u for u in value_indexes if u is not data and u.dtype is data.dtype and int(u.src[0].src[0].arg) == total), None) \
    if data is not None else None
  if data is None or maximum is None: return None
  negated_data = any(data in u.src and (
    (u.op is Ops.MUL and u.dtype in (dtypes.half, dtypes.float) and
     any(x.op is Ops.CONST and float(x.arg) == -1.0 for x in u.src)) or
    (u.op is Ops.XOR and u.dtype is dtypes.int and
     any(x.op is Ops.CONST and int(x.arg) == -1 for x in u.src))
  ) for u in max_reduce.toposort())
  input_total = int(data.src[0].src[0].arg)
  out_shape = _shape_of_store(sink)
  planes = prod(out_shape[:-2]) if len(out_shape) >= 3 else 1
  if planes <= 0 or input_total % planes: return None
  input_spatial = input_total // planes
  # Cummax encodes a reduction-axis coordinate through a floating MAX before
  # the final int cast (half under DEFAULT_FLOAT=HALF). Max-pool instead
  # encodes the absolute spatial address in an int MAX. Both share this
  # equality/index-selection graph.
  cumulative_index = store.src[1].op is Ops.CAST and store.src[1].dtype is dtypes.int and \
    store.src[1].src[0].dtype in (dtypes.half, dtypes.float)
  relative_index = cumulative_index

  def evaluate(u:UOp, coords:dict[UOp, int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    if u.op is Ops.WHERE:
      condition = evaluate(u.src[0], coords)
      return evaluate(u.src[1] if condition else u.src[2], coords)
    values = [evaluate(x, coords) for x in u.src]
    if any(x is Invalid for x in values): return Invalid
    if u.op is Ops.ADD: return values[0] + values[1]
    if u.op is Ops.MUL: return values[0] * values[1]
    if u.op is Ops.SUB: return values[0] - values[1]
    if u.op is Ops.FLOORDIV: return values[0] // values[1]
    if u.op is Ops.FLOORMOD: return values[0] % values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
    if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
    if u.op is Ops.NEG: return -values[0]
    raise ValueError(u.op)

  mapping = [-1] * (total*window)
  loop_extents = [int(u.src[0].arg) for u in loops]
  reduce_extents = [int(u.src[0].arg) for u in reductions]
  try:
    for out_linear in range(total):
      rem, coords = out_linear, {}
      for loop_axis in range(len(loops)-1, -1, -1):
        rem, coord = divmod(rem, loop_extents[loop_axis])
        coords[loops[loop_axis]] = coord
      out_index = int(evaluate(store.src[0].src[1], coords))
      if not 0 <= out_index < total: return None
      for reduce_linear in range(window):
        rem = reduce_linear
        for reduce_axis in range(len(reductions)-1, -1, -1):
          rem, coord = divmod(rem, reduce_extents[reduce_axis])
          coords[reductions[reduce_axis]] = coord
        source_index = evaluate(data.src[1], coords)
        # Original max-pool mapping accepted every valid source address. Cummax
        # carries its prefix mask outside the INDEX address, so also reject
        # candidates beyond the current reduction-axis coordinate.
        # if source_index is not Invalid: mapping[out_index*window+reduce_linear] = int(source_index)
        if source_index is not Invalid and (not cumulative_index or reduce_linear <= out_index % window):
          mapping[out_index*window+reduce_linear] = int(source_index)
  except (TypeError, ValueError, OverflowError, ZeroDivisionError): return None

  info = ProgramInfo.from_sink(sink)
  out_slot, data_slot, maximum_slot = info.outs[0], data.src[0].buf_uop.arg.slot, maximum.src[0].buf_uop.arg.slot
  # Rejected compatibility path, retained only for explicit diagnostics.
  # Accelerator backends have no precedent for CPU-evaluated ArgMax being
  # counted as a device pass. Normal execution continues into the DPU chain.
  if getenv("ROCKCHIP_ALLOW_HOST_OPS"):
    layout = (total, _HOST_ARGMAX_LAYOUT, window, input_spatial, *mapping)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, data_slot, maximum_slot))
    return (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True), relocs),)
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def native_int_to_half(source_slot:int) -> int:
    """Convert compact nonnegative int32 values through aligned four-lane MRDMA atoms."""
    nonlocal next_slot
    if negated_data:
      # Argmin's XOR-ordered int32 values include INT_MIN/INT_MAX. Use the
      # established ABI conversion before DPU negation; the native four-lane
      # arithmetic mode is only exact for the compact nonnegative pool indices.
      result, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(total, result, (source_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     int32_inputs=(source_slot,)))
      return result
    result, next_slot = next_slot, next_slot+1
    for start in range(0, total, 4):
      count = min(4, total-start)
      packed, next_slot = next_slot, next_slot+1
      pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      half_atom, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(4, half_atom, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_input=True))
      unpack_layout = (count, _HOST_UNPACK_HALF_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, result, 0, 0, 0xFFFFFFFF), RKReloc(0, half_atom, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, result, is_copy=True), unpack_relocs))
    return result

  def trunc_half(source_slot:int) -> int|None:
    """Truncate an fp16 scratch buffer using the native roundoff LUT and DPU masks."""
    nonlocal next_slot
    def alloc() -> int:
      nonlocal next_slot
      result, next_slot = next_slot, next_slot+1
      return result
    def temp_index(slot:int) -> UOp:
      out_idx = store.src[0]
      return out_idx.replace(dtype=dtypes.half,
        src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))
    def scalar(value:float) -> tuple[int,int]:
      return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]

    zero = (_ZERO_SLOT, 0)
    negative, magnitude, rounded = alloc(), alloc(), alloc()
    tasks.extend((_emit_where_stage(total, negative, zero, (source_slot, 0), Ops.SUB),
                  _emit_where_stage(total, magnitude, (source_slot, 0), (negative, 0), Ops.MAX)))
    # Build this internal pass with a single flat loop.  Reusing the parent
    # pool index can expose a multi-axis ADD tree that the LUT layout checker
    # intentionally rejects even though both scratch buffers are contiguous.
    axis, device = UOp.range(total, 0), store.src[0].src[0].device
    source_param = UOp.param(magnitude, dtypes.half, (total,), device=device)
    output_param = UOp.param(rounded, dtypes.half, (total,), device=device)
    roundoff = UOp(Ops.CUSTOM, dtypes.half, (UOp(Ops.INDEX, dtypes.half, (source_param, axis)),), arg="rk_roundoff")
    round_store = UOp(Ops.STORE, src=(UOp(Ops.INDEX, dtypes.half, (output_param, axis)), roundoff))
    round_sink = UOp.sink(UOp(Ops.END, src=(round_store, axis)))
    round_plan = plan_rk(round_sink)
    if isinstance(round_plan, str) or round_plan.kind != "dpu_lut":
      if getenv("ROCKCHIP_DEBUG_LOCAL_MAX"): print("RK_LOCAL_MAX_TRUNC_LUT_REJECT", round_plan)
      return None
    cmds, task, relocs = emit_rk(round_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    overshoot_diff, overshoot, truncated_abs = alloc(), alloc(), alloc()
    positive_diff, positive, negative_diff, negative_mask, sign, result = (alloc() for _ in range(6))
    tasks.extend((_emit_where_stage(total, overshoot_diff, (rounded, 0), (magnitude, 0), Ops.SUB),
                  _emit_where_stage(total, overshoot, (overshoot_diff, 0), (overshoot_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, truncated_abs, (rounded, 0), (overshoot, 0), Ops.SUB),
                  _emit_where_stage(total, positive_diff, (source_slot, 0), zero, Ops.SUB),
                  _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, negative_diff, zero, (source_slot, 0), Ops.SUB),
                  _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, sign, (positive, 0), (negative_mask, 0), Ops.SUB),
                  _emit_where_stage(total, result, (truncated_abs, 0), (sign, 0), Ops.MUL)))
    return result

  maximum_value_slot = native_int_to_half(maximum_slot) if data.dtype is dtypes.int else maximum_slot

  # Preserve the rejected compact CPU ArgMax task for reference.  It is not
  # selected: runtime values below are only address-gathered, while every
  # comparison and index selection is emitted as a DPU task.
  # host_layout = (total, _HOST_ARGMAX_LAYOUT, window, input_spatial, *mapping)
  # host_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, data_slot, maximum_slot))
  # return (RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", host_layout, out_slot, is_copy=True), host_relocs),)

  candidate_slots:list[int] = []
  index_bytes = max(1, ((input_spatial-1).bit_length()+7)//8)
  if index_bytes > 4: return None
  index_slots:list[list[int]] = []
  valid_slots:list[int|None] = []
  for candidate in range(window):
    addresses = tuple(mapping[out_index*window+candidate] for out_index in range(total))
    gathered, next_slot = next_slot, next_slot+1
    gather_layout = (total, _HOST_GATHER_MAP_LAYOUT, data.dtype.itemsize, *addresses)
    gather_relocs = (RKReloc(0, gathered, 0, 0, 0xFFFFFFFF), RKReloc(0, data_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", gather_layout, gathered, is_copy=True), gather_relocs))
    candidate_slot = native_int_to_half(gathered) if data.dtype is dtypes.int else gathered
    if negated_data:
      negated_warm, negated_slot = next_slot, next_slot+1
      next_slot += 2
      tasks.append(_emit_where_stage(total, negated_warm, (candidate_slot, 0),
                                     (_CONST_SLOT, 0xbf800000), Ops.MUL))
      tasks.append(_emit_where_stage(total, negated_slot, (candidate_slot, 0),
                                     (_CONST_SLOT, 0xbf800000), Ops.MUL))
      candidate_slot = negated_slot
    candidate_slots.append(candidate_slot)

    candidate_index_slots:list[int] = []
    for byte in range(index_bytes):
      indices = tuple(0 if address < 0 else ((candidate if relative_index else address % input_spatial) >> (8*byte)) & 0xFF
                      for address in addresses)
      index_slot, next_slot = next_slot, next_slot+1
      index_bits = tuple(struct.unpack('<H', struct.pack('<e', float(index)))[0] for index in indices)
      index_layout = (total, _HOST_STATIC_HALF_LAYOUT, *index_bits)
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", index_layout, index_slot, is_copy=True),
                             (RKReloc(0, index_slot, 0, 0, 0xFFFFFFFF),)))
      candidate_index_slots.append(index_slot)
    index_slots.append(candidate_index_slots)

    if any(address < 0 for address in addresses):
      valid_slot, next_slot = next_slot, next_slot+1
      valid_bits = tuple(0x0000 if address < 0 else 0x3c00 for address in addresses)
      valid_layout = (total, _HOST_STATIC_HALF_LAYOUT, *valid_bits)
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", valid_layout, valid_slot, is_copy=True),
                             (RKReloc(0, valid_slot, 0, 0, 0xFFFFFFFF),)))
      valid_slots.append(valid_slot)
    else: valid_slots.append(None)

  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  selected:list[tuple[int,int]] = [(_ZERO_SLOT, 0)] * index_bytes
  # Max-pool uses the first matching spatial index, while cummax uses the most
  # recent matching reduction-axis coordinate.
  candidate_order = range(window) if cumulative_index else range(window-1, -1, -1)
  for candidate in candidate_order:
    diff, less, equal = range(next_slot, next_slot+3)
    next_slot += 3
    tasks.append(_emit_where_stage(total, diff, (maximum_value_slot, 0), (candidate_slots[candidate], 0), Ops.SUB))
    tasks.append(_emit_where_stage(total, less, (diff, 0), (diff, 0), Ops.MAX, compare=True))
    tasks.append(_emit_where_stage(total, equal, one, (less, 0), Ops.SUB))
    candidate_valid_slot = valid_slots[candidate]
    if candidate_valid_slot is not None:
      valid_warm, valid_equal = next_slot, next_slot+1
      next_slot += 2
      tasks.append(_emit_where_stage(total, valid_warm, (equal, 0), (candidate_valid_slot, 0), Ops.MUL))
      tasks.append(_emit_where_stage(total, valid_equal, (equal, 0), (candidate_valid_slot, 0), Ops.MUL))
      equal = valid_equal
    for byte in range(index_bytes):
      delta, weighted, selected_out = range(next_slot, next_slot+3)
      next_slot += 3
      tasks.append(_emit_where_stage(total, delta, (index_slots[candidate][byte], 0), selected[byte], Ops.SUB))
      tasks.append(_emit_where_stage(total, weighted, (equal, 0), (delta, 0), Ops.MUL))
      tasks.append(_emit_where_stage(total, selected_out, selected[byte], (weighted, 0), Ops.ADD))
      selected[byte] = (selected_out, 0)
  # RK3588's 32-bit WDMA layout is compact only for one four-lane atom. Convert
  # each selected 0..255 digit, then compose the final int32 representation by
  # moving its low byte into the corresponding output byte.
  #
  # WIP reference: window-relative selection followed by native int32 EW ADD,
  # and OUT_CVT_OFFSET reconstruction, both still rounded above 2048 because
  # those stages precede the fp16 boundary.
  for start in range(0, total, 4):
    count = min(4, total-start)
    native_digits:list[int] = []
    for byte in range(index_bytes):
      packed_slot, next_slot = next_slot, next_slot+1
      pack_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, selected[byte][0], 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed_slot, is_copy=True), pack_relocs))
      native_slot, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(4, native_slot, (packed_slot, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
      native_digits.append(native_slot)
    assemble_layout = (count, _HOST_ASSEMBLE_INT_BYTES_LAYOUT, start, index_bytes)
    assemble_relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, *native_digits))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", assemble_layout, out_slot, is_copy=True), assemble_relocs))
  return tuple(tasks)

def _try_unpool_scatter_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower max-unpool as exact fp32 host scatter or int32-DPU/fp16 selection."""
  if getenv("ROCKCHIP_DEBUG_UNPOOL") >= 2: print("RK_UNPOOL_SINK", sink)
  store = _store_node(sink)
  reductions = [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if store is None or store.src[0].dtype not in (dtypes.half, dtypes.float) or len(reductions) > 1: return None
  reduce = reductions[0] if reductions else None
  if reduce is not None and (reduce.arg[0] is not Ops.ADD or _unwrap(store.src[1]) is not reduce): return None
  body = _reduce_body(reduce) if reduce is not None else _unwrap(store.src[1])
  if body.op is not Ops.WHERE or body.src[0].op not in (Ops.CMPNE, Ops.CMPEQ): return None
  zero = lambda u: u.op is Ops.CONST and float(u.arg) == 0.0
  if zero(body.src[1]): value, select_equal = _unwrap(body.src[2]), body.src[0].op is Ops.CMPNE
  elif zero(body.src[2]): value, select_equal = _unwrap(body.src[1]), body.src[0].op is Ops.CMPEQ
  else: return None
  if not select_equal or value.op is not Ops.INDEX or value.dtype is not store.src[0].dtype: return None
  index = next((_unwrap(x) for x in body.src[0].src if _unwrap(x).op is Ops.INDEX and _unwrap(x).dtype is dtypes.int), None)
  if index is None:
    index = next((_unwrap(x) for x in body.src[0].toposort() if _unwrap(x).op is Ops.INDEX and _unwrap(x).dtype is dtypes.int), None)
  if index is None or index.src[1] is not value.src[1]: return None
  if reduce is not None and any(r.src[0].op is not Ops.CONST for r in reduce.src[1:]): return None
  pooled = prod(int(r.src[0].arg) for r in reduce.src[1:]) if reduce is not None else 1
  # A fused max-pool producer can leave the public `spatial-index` affine
  # correction in this consumer (for example raw_index + input_spatial).
  # Move that compile-time offset to the static comparison operand.
  def affine_offset(u:UOp) -> int|None:
    u = _unwrap(u)
    if u is index: return 0
    if u.op is Ops.ADD:
      for expression, constant in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
        cu = _unwrap(constant)
        if cu.op is Ops.CONST and (offset := affine_offset(expression)) is not None: return offset + int(cu.arg)
    if u.op is Ops.SUB and _unwrap(u.src[1]).op is Ops.CONST and (offset := affine_offset(u.src[0])) is not None:
      return offset - int(_unwrap(u.src[1]).arg)
    return None
  # WIP reference: applying this visible affine offset compared against -6
  # for the single-window max-pool case.  The scheduled PARAM already exposes
  # the public zero-based index buffer, so compare that buffer directly.
  # index_offset = next((offset for operand in body.src[0].src if (offset := affine_offset(operand)) is not None), 0)
  index_offset = 0
  index_total = int(index.src[0].src[0].arg)
  value_total = int(value.src[0].src[0].arg)
  total = prod(_shape_of_store(sink))
  if pooled <= 0 or index_total != value_total or index_total % pooled or total == 0: return None
  planes = index_total // pooled
  if planes <= 0 or total % planes: return None
  out_spatial = total // planes
  info = ProgramInfo.from_sink(sink)
  out_slot, index_slot, value_slot = info.outs[0], index.src[0].buf_uop.arg.slot, value.src[0].buf_uop.arg.slot
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  # Normal-fp32 cannot enter the fp16 native selector without losing the
  # caller's precision. Keep that operator at one exact typed host boundary;
  # the established fp16 NPU comparison/selection implementation stays below.
  if store.src[0].dtype is dtypes.float or getenv("ROCKCHIP_ALLOW_HOST_OPS"):
    itemsize = store.src[0].dtype.itemsize
    layout = (total, _HOST_SCATTER_LAYOUT, index_total, planes, out_spatial, pooled, itemsize)
    relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, index_slot, value_slot))
    return (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True), relocs),)

  next_slot = max(info.globals, default=-1) + 1
  spatial_slot, gathered_index, gathered_value = range(next_slot, next_slot+3)
  physical_diff, diff_slot, neg_slot, magnitude_slot, unequal_slot, equal_slot, selected_slot = range(next_slot+3, next_slot+10)
  accumulator_slots = (next_slot+10, next_slot+11)
  tasks:list[RKSubTask] = []

  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  # Native int32 MRDMA uses four lanes.  Exact four-lane tensors can use the
  # height/surface-stride layout above; retain the proven one-row limit only
  # for a partial final atom.
  chunk_limit = total if total % 4 == 0 else 16384
  for chunk_start in range(0, total, chunk_limit):
    chunk = min(chunk_limit, total-chunk_start)
    spatial = tuple((chunk_start+i) % out_spatial - index_offset for i in range(chunk))
    spatial_layout = (chunk, _HOST_STATIC_INT_LAYOUT, *spatial)
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", spatial_layout, spatial_slot, is_copy=True),
                           (RKReloc(0, spatial_slot, 0, 0, 0xFFFFFFFF),)))
    accumulator:tuple[int,int] = (_ZERO_SLOT, 0)
    for candidate in range(pooled):
      index_layout = (chunk, _HOST_PLANE_GATHER_LAYOUT, 4, pooled, out_spatial, candidate, chunk_start)
      index_relocs = (RKReloc(0, gathered_index, 0, 0, 0xFFFFFFFF), RKReloc(0, index_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", index_layout, gathered_index, is_copy=True), index_relocs))
      value_layout = (chunk, _HOST_PLANE_GATHER_LAYOUT, 2, pooled, out_spatial, candidate, chunk_start)
      value_relocs = (RKReloc(0, gathered_value, 0, 0, 0xFFFFFFFF), RKReloc(0, value_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", value_layout, gathered_value, is_copy=True), value_relocs))
      # WIP reference: prepacking every raw-value atom here made the following
      # native diff read stale data on RK3588.  Pack immediately before each
      # native selector instead.

      # Integer subtraction happens before the fp16 output boundary. Although
      # a large nonzero difference can round in fp16, it cannot become zero,
      # so the following magnitude comparison is an exact equality test.
      tasks.append(_emit_where_stage(chunk, physical_diff, (gathered_index, 0), (spatial_slot, 0), Ops.SUB, native_int32_input=True))
      compact_layout = (chunk, _HOST_COMPACT_NATIVE_HALF_LAYOUT)
      compact_relocs = (RKReloc(0, diff_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, physical_diff, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", compact_layout, diff_slot, is_copy=True), compact_relocs))
      tasks.extend((_emit_where_stage(chunk, neg_slot, (_ZERO_SLOT, 0), (diff_slot, 0), Ops.SUB),
                    _emit_where_stage(chunk, magnitude_slot, (diff_slot, 0), (neg_slot, 0), Ops.MAX),
                    _emit_where_stage(chunk, unequal_slot, (magnitude_slot, 0), (magnitude_slot, 0), Ops.MAX, compare=True),
                    _emit_where_stage(chunk, equal_slot, one, (unequal_slot, 0), Ops.SUB)))
      if pooled == 1:
        # Select raw fp16 representation bits with native int32 MUL.  Unlike
        # fp16 `value*mask`, this preserves selected +/-inf and NaN while
        # producing exact +0 for unselected lanes.  Host stages only widen and
        # compact bytes; the value-dependent selection remains on the NPU.
        for start in range(0, chunk, 4):
          count = min(4, chunk-start)
          value_bits, mask_half, mask_int, selected_bits = range(next_slot, next_slot+4)
          next_slot += 4
          bits_layout = (count, _HOST_PACK_HALF_BITS_LAYOUT, start)
          value_relocs = (RKReloc(0, value_bits, 0, 0, 0xFFFFFFFF), RKReloc(0, gathered_value, 0, 0, 0xFFFFFFFF))
          tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", bits_layout, value_bits, is_copy=True), value_relocs))
          mask_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
          mask_relocs = (RKReloc(0, mask_half, 0, 0, 0xFFFFFFFF), RKReloc(0, equal_slot, 0, 0, 0xFFFFFFFF))
          tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", mask_layout, mask_half, is_copy=True), mask_relocs))
          tasks.append(_emit_where_stage(4, mask_int, (mask_half, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
          tasks.append(_emit_where_stage(4, selected_bits, (value_bits, 0), (mask_int, 0), Ops.MUL,
                                         native_int32_input=True, native_int32_output=True))
          unpack_layout = (count, _HOST_UNPACK_HALF_BITS_LAYOUT, chunk_start+start)
          unpack_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, selected_bits, 0, 0, 0xFFFFFFFF))
          tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", unpack_layout, out_slot, is_copy=True), unpack_relocs))
        continue
      final = candidate == pooled-1
      selected_out = accumulator_slots[0] if candidate == 0 else selected_slot
      tasks.append(_emit_where_stage(chunk, selected_out, (gathered_value, 0), (equal_slot, 0), Ops.MUL))
      if candidate == 0:
        accumulator = (selected_out, 0)
      else:
        accumulate_out = out_slot if final else accumulator_slots[candidate & 1]
        stage = _emit_where_stage(chunk, accumulate_out, accumulator, (selected_out, 0), Ops.ADD)
        if final and chunk_start:
          stage = replace(stage, relocs=tuple(replace(r, addend=r.addend+chunk_start*2) if r.globals_slot == out_slot else r
                                                for r in stage.relocs))
        tasks.append(stage)
        accumulator = (accumulate_out, 0)
    # WIP reference: the former fp16 single-candidate finalizer overwrote the
    # raw-bit selector above and could not preserve inf/NaN in any case.
    # if pooled == 1:
    #   stage = _emit_where_stage(chunk, out_slot, accumulator, (_ZERO_SLOT, 0), Ops.ADD)
    #   if chunk_start:
    #     stage = replace(stage, relocs=tuple(replace(r, addend=r.addend+chunk_start*2) if r.globals_slot == out_slot else r
    #                                           for r in stage.relocs))
    #   tasks.append(stage)
  return tuple(tasks)

def _try_movement_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower pure INDEX/WHERE movement to a compact host-side integer index program."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None: return None
  raw = store.src[1]
  if raw.op not in (Ops.INDEX, Ops.WHERE) and not (raw.op is Ops.CONST and raw.dtype is dtypes.bool): return None
  if store.src[0].dtype != raw.dtype: return None

  ranges = [u for u in sink.toposort() if u.op is Ops.RANGE and u.src[0].op is Ops.CONST]
  range_ids = {u:i for i,u in enumerate(ranges)}
  extents = tuple(int(u.src[0].arg) for u in ranges)
  total = prod(extents)
  if total != prod(_shape_of_store(sink)): return None
  input_slots:list[int] = []
  int_codes = {Ops.ADD:2, Ops.MUL:3, Ops.FLOORDIV:4, Ops.FLOORMOD:5,
               Ops.CMPLT:6, Ops.CMPNE:7, Ops.AND:8, Ops.OR:9}

  def input_id(slot:int) -> int:
    if slot not in input_slots: input_slots.append(slot)
    return input_slots.index(slot)

  def emit_int(u:UOp, code:list[int]) -> bool:
    while u.op is Ops.CAST:
      if dtypes.is_float(u.dtype) or dtypes.is_float(u.src[0].dtype): return False
      u = u.src[0]
    # This compact interpreter carries integer literals only. Accepting a
    # float coordinate expression truncated constants such as 13/9 to one.
    if dtypes.is_float(u.dtype): return False
    if u.op is Ops.CONST:
      if u.arg is Invalid:
        code.extend((0, 0))
        return True
      try: value = int(u.arg)
      except (TypeError, ValueError): return False
      code.extend((0, value))
      return True
    if u.op is Ops.RANGE and u in range_ids:
      code.extend((1, range_ids[u]))
      return True
    if u.op in int_codes and len(u.src) == 2:
      if not emit_int(u.src[0], code) or not emit_int(u.src[1], code): return False
      code.extend((int_codes[u.op], 0))
      return True
    if u.op is Ops.WHERE and len(u.src) == 3:
      if not emit_int(u.src[0], code) or not emit_int(u.src[1], code) or not emit_int(u.src[2], code): return False
      code.extend((11, 0))
      return True
    return False

  def emit_value(u:UOp, code:list[int]) -> bool:
    if u.op is Ops.INDEX and u.dtype == raw.dtype:
      if not emit_int(u.src[1], code): return False
      code.extend((10, input_id(u.src[0].buf_uop.arg.slot)))
      return True
    if u.op is Ops.CONST and u.dtype == raw.dtype and u.arg is not Invalid:
      if raw.dtype is dtypes.half: bits = struct.unpack('<H', struct.pack('<e', float(u.arg)))[0]
      elif raw.dtype is dtypes.float: bits = struct.unpack('<I', struct.pack('<f', float(u.arg)))[0]
      elif raw.dtype in (dtypes.int, dtypes.uint, dtypes.uint8, dtypes.bool): bits = int(u.arg) & ((1 << (raw.dtype.itemsize*8))-1)
      else: return False
      code.extend((12, _signed_i32(bits)))
      return True
    if u.op is Ops.WHERE and u.dtype == raw.dtype:
      if not emit_int(u.src[0], code) or not emit_value(u.src[1], code) or not emit_value(u.src[2], code): return False
      code.extend((11, 0))
      return True
    return False

  out_code:list[int] = []
  value_code:list[int] = []
  if not emit_int(store.src[0].src[1], out_code) or not emit_value(raw, value_code): return None
  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  layout = (total, _HOST_MOVEMENT_LAYOUT, raw.dtype.itemsize, len(extents), *extents,
            len(out_code), *out_code, len(value_code), *value_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in (out_slot, *input_slots))
  task = RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True)
  return (RKSubTask(cmds, task, relocs),)

def _try_broadcast_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Detect binary EW with broadcast operand(s) and emit pre-expansion copy + EW tasks.
  Either or both operands may have broadcast dimensions (missing RANGE vars vs store).
  We expand each broadcast/non-flat operand to a flat contiguous scratch buffer via
  host-side N-D copy with stride 0 on broadcast dims, then run the EW op as flat 1D DPU.
  Also handles SUB pattern: ADD(INDEX, MUL(INDEX, CONST(-1)))."""
  store = _store_node(sink)
  if store is None: return None
  val = _unwrap(store.src[1])
  if val.op not in _DPU_EW_CFGS or len(val.src) != 2: return None
  src0, src1 = (_unwrap(s) for s in val.src)
  # Unwrap SUB pattern: ADD(INDEX, MUL(INDEX, CONST(-1))) → treat as SUB
  ew_op = val.op
  if val.op is Ops.ADD:
    sub_match = _try_sub(val)
    if sub_match is not None:
      # SUB pattern: src0 is the non-negated INDEX, src1 is the negated INDEX
      # _try_sub returns (src_slot, ew_slot) — need to find which src is which
      for s, e in (val.src[0], val.src[1]), (val.src[1], val.src[0]):
        su = _unwrap(s)
        eu = _unwrap(e)
        if eu.op is Ops.MUL and su.op is Ops.INDEX:
          ma, mb = eu.src
          if mb.op is Ops.CONST and float(mb.arg) == -1.0 and _unwrap(ma).op is Ops.INDEX:
            src0, src1 = su, _unwrap(ma)
            ew_op = Ops.SUB
            break
  if src0.op is not Ops.INDEX or src1.op is not Ops.INDEX: return None
  # Get affine indices for both operands
  aff0 = _affine_index(src0.src[1])
  aff1 = _affine_index(src1.src[1])
  if aff0 is None or aff1 is None: return None
  # Get the store index affine (output layout)
  store_aff = _affine_index(store.src[0].src[1])
  if store_aff is None: return None
  store_vars = set(store_aff[0].keys())
  if len(store_vars) < 2: return None  # 1D is already handled by flat contiguous
  # Check which operands have broadcast (missing RANGE vars vs store)
  def missing_vars(aff): return store_vars - set(aff[0].keys())
  mv0, mv1 = missing_vars(aff0), missing_vars(aff1)
  if not mv0 and not mv1: return None  # no broadcast — not our case
  # Get extents
  extents = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort()
             if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP" and u.src[0].op is Ops.CONST}
  if not all(v in extents for v in store_vars): return None
  # Compute output shape (axes sorted by store stride, descending = row-major)
  axes = sorted(store_vars, key=lambda x: store_aff[0][x], reverse=True)
  out_shape = tuple(extents[v] for v in axes)
  total = prod(out_shape)
  contiguous_strides = tuple(prod(out_shape[i+1:]) for i in range(len(out_shape)))
  # Verify broadcast operands have stride 0 on missing vars
  for aff, mv in [(aff0, mv0), (aff1, mv1)]:
    for v in mv:
      if aff[0].get(v, 0) != 0: return None
  # Allocate scratch slots
  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  # For each operand, decide if it needs a copy to scratch (broadcast or non-flat)
  def needs_copy(aff, mv):
    if mv: return True  # broadcast — needs expansion
    strides = tuple(aff[0].get(v, 0) for v in axes)
    return strides != contiguous_strides or aff[1] != 0  # non-contiguous — needs flattening
  def emit_copy(source:UOp, scratch_slot:int) -> RKSubTask|None:
    scratch_param = UOp.param(scratch_slot, source.dtype, (total,), device=store.src[0].src[0].device)
    scratch_index = store.src[0].replace(dtype=source.dtype, src=(scratch_param, *store.src[0].src[1:]))
    movement_store = UOp(Ops.STORE, src=(scratch_index, source))
    loop_nodes = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
    movement = _try_movement_host_subtasks(UOp.sink(UOp(Ops.END, src=(movement_store, *loop_nodes))))
    return None if movement is None or len(movement) != 1 else movement[0]
  slot0 = src0.src[0].buf_uop.arg.slot
  slot1 = src1.src[0].buf_uop.arg.slot
  nc0 = needs_copy(aff0, mv0)
  nc1 = needs_copy(aff1, mv1)
  if nc0:
    s0_scratch = next_slot; next_slot += 1
    if (copy := emit_copy(src0, s0_scratch)) is None: return None
    tasks.append(copy)
  else:
    s0_scratch = slot0
  if nc1:
    s1_scratch = next_slot; next_slot += 1
    if (copy := emit_copy(src1, s1_scratch)) is None: return None
    tasks.append(copy)
  else:
    s1_scratch = slot1
  if val.dtype is dtypes.float and max(s0_scratch, s1_scratch) >= 7: return None
  # EW op as flat 1D DPU operation
  ew_cmds:list[RKCmd] = []
  ew_relocs:list[RKReloc] = []
  dw = (total + 7) // 8 - 1
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_S_POINTER, 0x0e)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, dw)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x0e)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, dw)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0x7)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  emitter_reloc(ew_cmds, ew_relocs, out_slot)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  emitter_reloc(ew_cmds, ew_relocs, s0_scratch)
  emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
  emitter_reloc(ew_cmds, ew_relocs, s1_scratch)
  emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[ew_op])
  # FDIV: scale=1 (no FP32TOFP16), MRDMA_FP16TOFP32_EN=0; others: scale=(1<<16)|1, FP16TOFP32_EN=1
  if ew_op is Ops.FDIV:
    emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 1)
    emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841)
  else:
    emitter_emit(ew_cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
    emitter_emit(ew_cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849)
  emitter_pc_op_en(ew_cmds, 12)
  ew_task = RKTask(0x18, 0x300, 4, "dpu", (total,), out_slot,
                   fp32_inputs=(s0_scratch, s1_scratch) if val.dtype is dtypes.float else (),
                   fp32_output=val.dtype is dtypes.float)
  ew_subtask = RKSubTask(tuple(c.pack() for c in ew_cmds), ew_task, tuple(ew_relocs))
  tasks.append(ew_subtask)
  return tuple(tasks)

def _try_pad_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Recognize pad pattern WHERE(AND(...), INDEX, CONST(pad_val)) and emit fill + scatter copy.
  The INDEX affine maps output coords → input coords. We fill output with pad_val, then
  scatter-copy input data to the correct positions using dst strides from the store affine."""
  store = _store_node(sink)
  if store is None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.WHERE: return None
  cond = _unwrap(val.src[0])
  t_branch = _unwrap(val.src[1])
  f_branch = _unwrap(val.src[2])
  # Pattern: WHERE(cond, INDEX, CONST(pad_val)) or WHERE(cond, CONST(pad_val), INDEX)
  # The branches may be swapped depending on how the condition is structured.
  # For non-zero pad values, a branch may be a nested WHERE(inner_cond, INDEX, CONST)
  # or WHERE(inner_cond, CONST, INDEX) — look through any nesting to find INDEX and CONST.
  def extract_index_const(branch):
    """From a branch (possibly nested WHERE), extract (INDEX, CONST) or None."""
    eu = _unwrap(branch)
    if eu.op is Ops.WHERE:
      inner_t, inner_f = _unwrap(eu.src[1]), _unwrap(eu.src[2])
      # Try both orderings of inner branches
      if inner_t.op is Ops.INDEX and inner_f.op is Ops.CONST: return inner_t, inner_f
      if inner_t.op is Ops.CONST and inner_f.op is Ops.INDEX: return inner_f, inner_t
    if eu.op is Ops.INDEX: return eu, None
    if eu.op is Ops.CONST: return None, eu
    return None, None
  t_idx, t_const = extract_index_const(t_branch)
  f_idx, f_const = extract_index_const(f_branch)
  # One branch should have INDEX, the other CONST
  if t_idx is not None and f_const is not None:
    t_branch = t_idx  # normal: WHERE(cond, INDEX, CONST(pad_val))
  elif t_const is not None and f_idx is not None:
    # swapped: WHERE(cond, CONST(pad_val), INDEX) — negate condition
    t_branch = f_idx
    f_branch = t_const
    cond = UOp(Ops.CMPNE, dtypes.bool, src=(cond, UOp.const(dtypes.bool, True)))
  else:
    return None
  if cond.op not in (Ops.AND, Ops.CMPLT, Ops.CMPNE): return None
  pad_val = float(f_branch.arg)
  # Get the input affine index (maps output coords → input coords)
  # The INDEX's src[1] may be a guarded WHERE(cond, affine_expr, Invalid) — look through it
  in_idx_expr = t_branch.src[1]
  if in_idx_expr.op is Ops.WHERE:
    in_idx_expr = _unwrap(in_idx_expr.src[1])  # true branch = affine expr
  in_aff = _affine_index(in_idx_expr)
  if in_aff is None: return None
  # Get the store affine index (output layout)
  store_aff = _affine_index(store.src[0].src[1])
  if store_aff is None: return None
  store_vars = set(store_aff[0].keys())
  if len(store_vars) < 1: return None
  # Get extents
  extents = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort()
             if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP" and u.src[0].op is Ops.CONST}
  if not all(v in extents for v in store_vars): return None
  # Compute output shape (axes sorted by store stride, descending = row-major)
  axes = sorted(store_vars, key=lambda x: store_aff[0][x], reverse=True)
  out_shape = tuple(extents[v] for v in axes)
  out_total = prod(out_shape)
  out_contiguous = tuple(prod(out_shape[i+1:]) for i in range(len(out_shape)))
  # Input affine: strides and offset in terms of the same axes
  in_strides = tuple(in_aff[0].get(v, 0) for v in axes)
  in_offset = in_aff[1]
  # Parse the condition to find valid ranges per axis.
  # Condition: AND of CMPNE(CMPLT(RANGE, left_pad), True) and CMPLT(RANGE, total-right_pad)
  # → RANGE >= left_pad AND RANGE < total-right_pad
  # For axes not in the condition, valid range = [0, extent)
  valid_ranges:dict[int, tuple[int,int]] = {v: (0, extents[v]) for v in store_vars}
  def parse_cmp(u):
    # CMPNE(CMPLT(RANGE, CONST(lo)), CONST(True)) → RANGE >= lo
    # CMPLT(RANGE, CONST(hi)) → RANGE < hi
    if u.op is Ops.CMPNE:
      inner = _unwrap(u.src[0])
      if inner.op is Ops.CMPLT:
        lhs, rhs = _unwrap(inner.src[0]), _unwrap(inner.src[1])
        if lhs.op is Ops.RANGE and rhs.op is Ops.CONST and _unwrap(u.src[1]).op is Ops.CONST and _unwrap(u.src[1]).arg is True:
          return (lhs.arg[0], int(rhs.arg), None)  # var, lo, no hi
      return None
    if u.op is Ops.CMPLT:
      lhs, rhs = _unwrap(u.src[0]), _unwrap(u.src[1])
      if lhs.op is Ops.RANGE and rhs.op is Ops.CONST:
        return (lhs.arg[0], None, int(rhs.arg))  # var, no lo, hi
    return None
  # Flatten nested ANDs and collect all leaf conditions (single CMPLT/CMPNE is a leaf)
  leaf_conds:list = []
  def flatten_and(u):
    u = _unwrap(u)
    if u.op is Ops.AND:
      for s in u.src: flatten_and(s)
    else:
      leaf_conds.append(u)
  flatten_and(cond)
  for s in leaf_conds:
    r = parse_cmp(s)
    if r is None: return None
    var, lo, hi = r
    cur_lo, cur_hi = valid_ranges[var]
    if lo is not None: valid_ranges[var] = (max(cur_lo, lo), cur_hi)
    if hi is not None: valid_ranges[var] = (cur_lo, min(cur_hi, hi))
  # Compute input shape and dst offset
  in_shape = tuple(valid_ranges[v][1] - valid_ranges[v][0] for v in axes)
  in_total = prod(in_shape)
  dst_offset = sum(valid_ranges[v][0] * store_aff[0][v] for v in axes)
  # Verify: in_offset should equal -sum(valid_ranges[v][0] * in_stride[v])
  expected_offset = -sum(valid_ranges[v][0] * in_aff[0].get(v, 0) for v in axes)
  if in_offset != expected_offset: return None
  # Check that input strides are contiguous for the input shape.
  # Exception: if an input dim has size 1, its stride can be 0 (broadcast-like).
  in_contiguous = tuple(prod(in_shape[i+1:]) for i in range(len(in_shape)))
  for i in range(len(in_shape)):
    if in_shape[i] == 1 and in_strides[i] == 0: continue  # size-1 dim, stride 0 is ok
    if in_strides[i] != in_contiguous[i]: return None
  # Get slots
  in_slot = t_branch.src[0].buf_uop.arg.slot
  info = ProgramInfo.from_sink(sink)
  out_slot = info.outs[0]
  # Task 1: Fill output with pad_val (using DPU fill)
  fill_cmds = [RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0)]
  fill_layout = (out_total,)
  fill_task = RKTask(0x18, 0x300, 4, "dpu", fill_layout, out_slot, is_fill=True, const_val=pad_val)
  fill_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF),)
  fill_subtask = RKSubTask(tuple(c.pack() for c in fill_cmds), fill_task, fill_relocs)
  # Task 2: Scatter copy from input to output at dst_offset
  # Layout: (in_total, -ndim, *in_shape, *in_contiguous, 0, *out_contiguous, dst_offset)
  # Negative ndim signals scatter mode (src contiguous → dst strided)
  scatter_layout = (in_total, -len(in_shape), *in_shape, *in_contiguous, 0, *out_contiguous, dst_offset)
  scatter_cmds = [RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0)]
  scatter_task = RKTask(0x18, 0x300, 4, "dpu", scatter_layout, out_slot, is_copy=True)
  scatter_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, in_slot, 0, 0, 0xFFFFFFFF))
  scatter_subtask = RKSubTask(tuple(c.pack() for c in scatter_cmds), scatter_task, scatter_relocs)
  return (fill_subtask, scatter_subtask)

def _try_cat_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Recognize cat-like nested WHERE(CMPNE(RANGE, CONST), ...) and emit copy tasks.
  Each tensor is copied to its position in the output via host-side memmove."""
  store = _store_node(sink)
  if store is None: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.WHERE: return None
  slots:list[tuple[int,int]] = []
  n_tensors, tensor_size = 0, 0
  cur = val
  while cur is not None:
    cur = _unwrap(cur)
    if cur.op is not Ops.WHERE: return None
    cond = _unwrap(cur.src[0])
    if cond.op is not Ops.CMPNE: return None
    cond_lhs, cond_rhs = (_unwrap(x) for x in cond.src)
    if cond_lhs.op is not Ops.RANGE or cond_rhs.op is not Ops.CONST: return None
    if getattr(cond_lhs.arg[-1], "name", "") != "LOOP": return None
    n_tensors = int(cond_lhs.src[0].arg)
    k = int(cond_rhs.arg)
    false_u = _unwrap(cur.src[2])
    if false_u.op is not Ops.INDEX: return None
    false_slot = _where_arg(cur.src[2])
    if false_slot is None: return None
    inner_rng = next((u for u in false_u.src[1:] if u.op is Ops.RANGE), None)
    if inner_rng is None or inner_rng.src[0].op is not Ops.CONST: return None
    tensor_size = int(inner_rng.src[0].arg)
    slots.append((false_slot[0], k))
    true_u = _unwrap(cur.src[1])
    if true_u.op is Ops.INDEX:
      true_slot = _where_arg(cur.src[1])
      if true_slot is None: return None
      slots.append((true_slot[0], k + 1))
      cur = None
    elif true_u.op is Ops.WHERE:
      cur = cur.src[1]
    else:
      return None
  total = prod(_shape_of_store(sink))
  if total != n_tensors * tensor_size or len(slots) != n_tensors: return None
  out_slot = store.src[0].src[0].buf_uop.arg.slot
  tasks = []
  for slot, pos in sorted(slots, key=lambda x: x[1]):
    offset = pos * tensor_size * 2
    cmds = [RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0)]
    task = RKTask(0x18, 0x300, 4, "dpu", (tensor_size,), out_slot, is_copy=True, out_offset=offset)
    relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(tuple(c.pack() for c in cmds), task, relocs))
  return tuple(tasks)

def _try_cmac_multifactor_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Split a three-factor einsum into PyTorch-ordered CMAC contractions."""
  reduce = _reduce_node(sink)
  if reduce is None or reduce.arg[0] is not Ops.ADD: return None
  body = _reduce_body(reduce)
  if body.op is not Ops.MUL or len(body.src) != 2: return None
  first, last = _unwrap(body.src[0]), _unwrap(body.src[1])
  if first.op is not Ops.MUL or len(first.src) != 2 or _unwrap(last).op is not Ops.INDEX: return None
  factors = tuple(_unwrap(x) for x in first.src)
  factor_dtype = factors[0].dtype
  if factor_dtype not in (dtypes.half, dtypes.float) or \
     any(x.op is not Ops.INDEX or x.dtype is not factor_dtype for x in (*factors, last)): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  if not loops or not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  factor_nodes = tuple(set(x.toposort()) for x in factors)
  contracted = [u for u in reductions if u in factor_nodes[0] and u in factor_nodes[1]]
  remaining = [u for u in reductions if u not in contracted]
  if not contracted or not remaining or any(u in set(last.toposort()) for u in contracted): return None
  carry_loops = [u.replace(arg=(u.arg[0], AxisType.LOOP)) for u in remaining]
  stage_axes = [*loops, *carry_loops]
  stage_extents = [int(u.src[0].arg) for u in stage_axes]
  stage_total = prod(stage_extents)

  def flat_index(axes:list[UOp], extents:list[int]) -> UOp:
    flat = axes[0]
    for axis, extent in zip(axes[1:], extents[1:]): flat = flat*extent + axis
    return flat

  info = ProgramInfo.from_sink(sink)
  intermediate_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []

  if factor_dtype is dtypes.float:
    # Keep the associated first contraction in fp32, then feed its exact ABI
    # result to the compensated fp32 CMAC wrapper for the remaining dot.
    carry_map = dict(zip(remaining, carry_loops))
    stage_product = first.substitute(carry_map)
    stage_acc = UOp(Ops.REDUCE, dtypes.float, (stage_product, *contracted), arg=reduce.arg)
    intermediate_param = UOp.param(intermediate_slot, dtypes.float, (stage_total,), device=factors[0].src[0].device)
    stage_out = UOp(Ops.INDEX, dtypes.float, (intermediate_param, flat_index(stage_axes, stage_extents)))
    stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(stage_out, stage_acc)), *stage_axes)))
    if (stage_tasks := _try_small_fp32_cmac_subtasks(stage_sink)) is None: return None
    tasks.extend(stage_tasks)

    final_axes = [*loops, *remaining]
    final_extents = [int(u.src[0].arg) for u in final_axes]
    intermediate_index = UOp(Ops.INDEX, dtypes.float, (intermediate_param, flat_index(final_axes, final_extents)))
    final_product = UOp(Ops.MUL, dtypes.float, (intermediate_index, last))
    final_acc = UOp(Ops.REDUCE, dtypes.float, (final_product, *remaining), arg=reduce.arg)
    final_sink = sink.substitute({reduce:final_acc})
    if (final_tasks := _try_small_fp32_cmac_subtasks(final_sink)) is None: return None
    tasks.extend(final_tasks)
    return tuple(tasks)

  # PyTorch contracts the associated first pair over their common reduction
  # axes, casts that intermediate to fp16, then performs the remaining dot.
  carry_map = dict(zip(remaining, carry_loops))
  stage_product = first.substitute(carry_map)
  stage_acc = UOp(Ops.REDUCE, dtypes.float,
                  (UOp(Ops.CAST, dtypes.float, (stage_product,), arg=dtypes.float), *contracted), arg=reduce.arg)
  stage_value = UOp(Ops.CAST, dtypes.half, (stage_acc,), arg=dtypes.half)
  intermediate_param = UOp.param(intermediate_slot, dtypes.half, (stage_total,), device=factors[0].src[0].device)
  stage_out = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(stage_axes, stage_extents)))
  stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(stage_out, stage_value)), *stage_axes)))
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  final_axes = [*loops, *remaining]
  final_extents = [int(u.src[0].arg) for u in final_axes]
  intermediate_index = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(final_axes, final_extents)))
  final_product = UOp(Ops.MUL, dtypes.half, (intermediate_index, last))
  final_acc = UOp(Ops.REDUCE, dtypes.float,
                  (UOp(Ops.CAST, dtypes.float, (final_product,), arg=dtypes.float), *remaining), arg=reduce.arg)
  final_sink = sink.substitute({reduce:final_acc})
  final_plan = plan_rk(final_sink)
  if isinstance(final_plan, str) or final_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(final_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(final_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(final_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

def _try_nested_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Materialize the fp16 boundary between two fused ADD reductions."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.half: return None
  outer = _unwrap(store.src[1])
  if outer.op is not Ops.REDUCE or outer.arg[0] is not Ops.ADD or outer.dtype is not dtypes.float: return None
  outer_input = outer.src[0]
  if outer_input.op is not Ops.CAST or outer_input.dtype is not dtypes.float: return None
  intermediate_value = outer_input.src[0]
  if intermediate_value.op is not Ops.CAST or intermediate_value.dtype is not dtypes.half: return None
  inner = intermediate_value.src[0]
  if inner.op is not Ops.REDUCE or inner.arg[0] is not Ops.ADD or inner.dtype is not dtypes.float: return None

  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  inner_reductions, outer_reductions = list(inner.src[1:]), list(outer.src[1:])
  if not inner_reductions or not outer_reductions or \
     any(u.src[0].op is not Ops.CONST for u in (*loops, *inner_reductions, *outer_reductions)): return None
  if set(inner_reductions) & set(outer_reductions): return None

  def flat_index(axes:list[UOp], extents:list[int]) -> UOp:
    flat = axes[0]
    for axis, extent in zip(axes[1:], extents[1:]): flat = flat*extent + axis
    return flat

  info = ProgramInfo.from_sink(sink)
  intermediate_slot = max(info.globals, default=-1) + 1
  carry_loops = [u.replace(arg=(u.arg[0], AxisType.LOOP)) for u in outer_reductions]
  stage_axes = [*loops, *carry_loops]
  stage_extents = [int(u.src[0].arg) for u in stage_axes]
  stage_total = prod(stage_extents)
  carry_map = dict(zip(outer_reductions, carry_loops))
  stage_value = intermediate_value.substitute(carry_map)
  device = store.src[0].src[0].device
  intermediate_param = UOp.param(intermediate_slot, dtypes.half, (stage_total,), device=device)
  stage_out = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(stage_axes, stage_extents)))
  stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(stage_out, stage_value)), *stage_axes)))

  tasks:list[RKSubTask] = []
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  final_axes = [*loops, *outer_reductions]
  final_extents = [int(u.src[0].arg) for u in final_axes]
  intermediate_index = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(final_axes, final_extents)))
  final_acc = UOp(Ops.REDUCE, dtypes.float,
                  (UOp(Ops.CAST, dtypes.float, (intermediate_index,), arg=dtypes.float), *outer_reductions), arg=outer.arg)
  final_sink = sink.substitute({outer:final_acc})
  final_plan = plan_rk(final_sink)
  if isinstance(final_plan, str) or final_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(final_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(final_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(final_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

def _try_relu_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Split ReLU(SUM(ReLU(x))) into ordered DPU, CMAC, and DPU stages."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype not in (dtypes.half, dtypes.float): return None

  def relu_operand(u:UOp) -> UOp|None:
    u = _unwrap(u)
    if u.op is not Ops.WHERE or len(u.src) != 3: return None
    cond, selected, zero = u.src
    if cond.op is not Ops.CMPLT or len(cond.src) != 2 or cond.src[0].op is not Ops.CONST or float(cond.src[0].arg) != 0.0 or \
       zero.op is not Ops.CONST or float(zero.arg) != 0.0: return None
    return _unwrap(selected) if _unwrap(selected) is _unwrap(cond.src[1]) else None

  reduce = relu_operand(store.src[1])
  if reduce is None or reduce.op is not Ops.REDUCE or reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  source = relu_operand(reduce.src[0])
  if source is None or source.op is not Ops.INDEX or source.dtype is not store.src[0].dtype or source.src[0].op is not Ops.PARAM: return None
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None

  info = ProgramInfo.from_sink(sink)
  input_total = int(source.src[0].src[0].arg)
  output_total = prod(_shape_of_store(sink))
  if source.dtype is dtypes.float:
    source_slot = source.src[0].buf_uop.arg.slot
    if source_slot >= 7: return None
    next_slot = max(info.globals, default=-1)+1
    tasks:list[RKSubTask] = []
    def alloc() -> int:
      nonlocal next_slot
      ret, next_slot = next_slot, next_slot+1
      return ret
    def scalar(value:float) -> tuple[int,int]:
      return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
    def stage(a:tuple[int,int], b:tuple[int,int], op:Ops, compare=False) -> int:
      out = alloc()
      tasks.append(_emit_where_stage(input_total, out, a, b, op, compare=compare))
      return out
    def view(tag:int) -> int:
      out = alloc()
      cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
      relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (input_total, tag), out, is_copy=True), relocs))
      return out
    high, residual = view(_HOST_FP32_HALF_LAYOUT), view(_HOST_FP32_RESIDUAL_LAYOUT)
    high_positive = stage((high, 0), (high, 0), Ops.MAX, compare=True)
    negative_high = stage((_ZERO_SLOT, 0), (high, 0), Ops.SUB)
    high_negative = stage((negative_high, 0), (negative_high, 0), Ops.MAX, compare=True)
    high_nonzero = stage((high_positive, 0), (high_negative, 0), Ops.MAX)
    high_zero = stage(scalar(1.0), (high_nonzero, 0), Ops.SUB)
    residual_positive = stage((residual, 0), (residual, 0), Ops.MAX, compare=True)
    residual_only = stage((residual_positive, 0), (high_zero, 0), Ops.MUL)
    positive = stage((high_positive, 0), (residual_only, 0), Ops.MAX)
    relu_high = stage((high, 0), (positive, 0), Ops.MUL)
    relu_residual = stage((residual, 0), (positive, 0), Ops.MUL)

    def cmac_sum(input_slot:int) -> int|None:
      out_slot = alloc()
      half_param = UOp.param(input_slot, dtypes.half, (input_total,), device=source.src[0].device)
      half_index = source.replace(dtype=dtypes.half, src=(half_param, *source.src[1:]))
      half_reduce = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (half_index,), arg=dtypes.float), *reductions))
      out_param = UOp.param(out_slot, dtypes.half, (output_total,), device=store.src[0].src[0].device)
      out_index = store.src[0].replace(dtype=dtypes.half, src=(out_param, *store.src[0].src[1:]))
      stage_store = store.replace(src=(out_index, UOp(Ops.CAST, dtypes.half, (half_reduce,), arg=dtypes.half)))
      stage_plan = plan_rk(sink.substitute({store:stage_store}))
      if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
      if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
      else:
        cmds, task, relocs = emit_rk(stage_plan)
        tasks.append(RKSubTask(cmds, task, relocs))
      return out_slot

    high_sum, residual_sum = cmac_sum(relu_high), cmac_sum(relu_residual)
    if high_sum is None or residual_sum is None: return None
    correction = alloc()
    tasks.append(_emit_where_stage(output_total, correction, (residual_sum, 0), scalar(1/256), Ops.MUL))
    tasks.append(_emit_where_stage(output_total, info.outs[0], (high_sum, 0), (correction, 0), Ops.ADD, fp32_output=True))
    return tuple(tasks)

  relu_slot = max(info.globals, default=-1) + 1
  sum_slot = relu_slot + 1
  zero_arg = (_ZERO_SLOT, 0)
  tasks = [_emit_where_stage(input_total, relu_slot, (source.src[0].buf_uop.arg.slot, 0), zero_arg, Ops.MAX)]

  device = store.src[0].src[0].device
  relu_param = UOp.param(relu_slot, dtypes.half, (input_total,), device=device)
  relu_index = source.replace(src=(relu_param, *source.src[1:]))
  stage_reduce = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (relu_index,), arg=dtypes.float), *reductions))
  sum_param = UOp.param(sum_slot, dtypes.half, (output_total,), device=device)
  sum_index = store.src[0].replace(src=(sum_param, *store.src[0].src[1:]))
  stage_value = UOp(Ops.CAST, dtypes.half, (stage_reduce,), arg=dtypes.half)
  stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(sum_index, stage_value)), *loops)))
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  tasks.append(_emit_where_stage(output_total, info.outs[0], (sum_slot, 0), zero_arg, Ops.MAX))
  return tuple(tasks)

def _try_movement_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Materialize a static indexed-WHERE reduction body, then sum it on CMAC."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.half or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  body = _unwrap(reduce.src[0])
  if body.op is not Ops.WHERE or any(_unwrap(arm).op is not Ops.INDEX for arm in body.src[1:]): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None

  def flat_index(axes:list[UOp], extents:list[int]) -> UOp:
    flat = axes[0]
    for axis, extent in zip(axes[1:], extents[1:]): flat = flat*extent + axis
    return flat

  info = ProgramInfo.from_sink(sink)
  intermediate_slot = max(info.globals, default=-1) + 1
  reduction_loops = [u.replace(arg=(u.arg[0], AxisType.LOOP)) for u in reductions]
  stage_axes = [*loops, *reduction_loops]
  stage_extents = [int(u.src[0].arg) for u in stage_axes]
  stage_total = prod(stage_extents)
  stage_body = body.substitute(dict(zip(reductions, reduction_loops)))
  device = store.src[0].src[0].device
  intermediate_param = UOp.param(intermediate_slot, dtypes.half, (stage_total,), device=device)
  stage_out = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(stage_axes, stage_extents)))
  stage_value = stage_body.replace(dtype=dtypes.half, src=(stage_body.src[0], *(_unwrap(arm) for arm in stage_body.src[1:])))
  stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(stage_out, stage_value)), *stage_axes)))
  movement_tasks = _try_movement_host_subtasks(stage_sink)
  if movement_tasks is None:
    if getenv("ROCKCHIP_DEBUG_MOVEMENT_SUM"): print("RK_MOVEMENT_SUM_MOVEMENT_REJECT", stage_sink)
    return None
  tasks = list(movement_tasks)

  final_axes = [*loops, *reductions]
  final_extents = [int(u.src[0].arg) for u in final_axes]
  intermediate_index = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(final_axes, final_extents)))
  final_acc = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (intermediate_index,), arg=dtypes.float), *reductions))
  final_sink = sink.substitute({reduce:final_acc})
  final_plan = plan_rk(final_sink)
  if isinstance(final_plan, str) or final_plan.kind != "cmac":
    if getenv("ROCKCHIP_DEBUG_MOVEMENT_SUM"): print("RK_MOVEMENT_SUM_CMAC_REJECT", final_plan, final_sink)
    return None
  if (shared_tasks := _try_cmac_shared_subtasks(final_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(final_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(final_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

def _try_elementwise_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Materialize a fused fp16 elementwise reduction body, then sum it on CMAC."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.half or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  body = _reduce_body(reduce)
  # Direct INDEX/MUL reductions belong to CMAC and multifactor lowering.  This
  # path is only for a genuinely fused additive elementwise body such as BCE.
  if body.op is not Ops.ADD: return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None

  def flat_index(axes:list[UOp], extents:list[int]) -> UOp:
    flat = axes[0]
    for axis, extent in zip(axes[1:], extents[1:]): flat = flat*extent + axis
    return flat

  info = ProgramInfo.from_sink(sink)
  intermediate_slot = max(info.globals, default=-1) + 1
  reduction_loops = [u.replace(arg=(u.arg[0], AxisType.LOOP)) for u in reductions]
  stage_axes = [*loops, *reduction_loops]
  stage_extents = [int(u.src[0].arg) for u in stage_axes]
  stage_total = prod(stage_extents)
  stage_body = body.substitute(dict(zip(reductions, reduction_loops)))
  device = store.src[0].src[0].device
  intermediate_param = UOp.param(intermediate_slot, dtypes.half, (stage_total,), device=device)
  stage_out = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(stage_axes, stage_extents)))
  stage_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(stage_out, stage_body)), *stage_axes)))

  # The reduction body is deliberately lowered as an ordinary elementwise graph:
  # every arithmetic operation remains an NPU DPU/LUT task and only the static
  # address calculation above is performed while constructing the command chain.
  elementwise_tasks = _try_bce_logits_subtasks(stage_sink)
  if elementwise_tasks is None: elementwise_tasks = _try_bce_subtasks(stage_sink)
  if elementwise_tasks is None: elementwise_tasks = _try_elementwise_subtasks(stage_sink)
  if elementwise_tasks is None:
    if getenv("ROCKCHIP_DEBUG_ELEMENTWISE_SUM"): print("RK_ELEMENTWISE_SUM_BODY_REJECT", stage_sink)
    return None
  tasks = list(elementwise_tasks)

  final_axes = [*loops, *reductions]
  final_extents = [int(u.src[0].arg) for u in final_axes]
  intermediate_index = UOp(Ops.INDEX, dtypes.half, (intermediate_param, flat_index(final_axes, final_extents)))
  final_acc = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (intermediate_index,), arg=dtypes.float), *reductions))
  final_sink = sink.substitute({reduce:final_acc})
  final_plan = plan_rk(final_sink)
  if isinstance(final_plan, str) or final_plan.kind != "cmac":
    if getenv("ROCKCHIP_DEBUG_ELEMENTWISE_SUM"): print("RK_ELEMENTWISE_SUM_CMAC_REJECT", final_plan, final_sink)
    return None
  if (shared_tasks := _try_cmac_shared_subtasks(final_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(final_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(final_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

def _wip_try_fp32_sum_output_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Run a half-input CMAC sum, then widen its fp16 output through a DPU ABI stage."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or store.src[1] is not reduce or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  source = _unwrap(reduce.src[0])
  if source.op is not Ops.INDEX or source.dtype is not dtypes.half: return None
  info, output_total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  scratch_slot = max(info.globals, default=-1) + 1
  device = store.src[0].src[0].device
  scratch_param = UOp.param(scratch_slot, dtypes.half, (output_total,), device=device)
  scratch_index = store.src[0].replace(dtype=dtypes.half, src=(scratch_param, *store.src[0].src[1:]))
  stage_store = store.replace(src=(scratch_index, UOp(Ops.CAST, dtypes.half, (reduce,), arg=dtypes.half)))
  stage_sink = sink.substitute({store:stage_store})
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
  tasks:list[RKSubTask] = []
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  tasks.append(_emit_where_stage(output_total, info.outs[0], (scratch_slot, 0), (_ZERO_SLOT, 0), Ops.ADD, fp32_output=True))
  return tuple(tasks)

def _try_long_fp32_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Reduce one long fp32 vector with scalar-safe CMAC chunks and raw fp32 partials."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or store.src[1] is not reduce or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  source = _unwrap(reduce.src[0])
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if source.op is not Ops.INDEX or source.dtype is not dtypes.float or source.src[0].op is not Ops.PARAM or \
     loops or len(reductions) != 1 or reductions[0].src[0].op is not Ops.CONST: return None
  K = int(reductions[0].src[0].arg)
  if K <= 256 or int(source.src[0].src[0].arg) != K or \
     _affine_index(source.src[1]) != ({reductions[0].arg[0]:1}, 0): return None

  info, device = ProgramInfo.from_sink(sink), store.src[0].src[0].device
  source_slot = source.src[0].buf_uop.arg.slot
  if source_slot >= 7: return None
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def view(source_slot:int, tag:int, total:int, layout:tuple[int,...]=(), out_slot:int|None=None) -> int:
    out = alloc() if out_slot is None else out_slot
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, tag, *layout), out, is_copy=True), relocs))
    return out
  def stage(a:tuple[int,int], b:tuple[int,int], op:Ops) -> int:
    out = alloc()
    tasks.append(_emit_where_stage(1, out, a, b, op))
    return out
  def combine_raw(high_slot:int, low_slot:int, out_slot:int) -> None:
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, high_slot, 0, 0, 0xFFFFFFFF),
              RKReloc(0, low_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (1, _HOST_FP32_COMBINE_LAYOUT),
                                       out_slot, is_copy=True, fp32_output=True), relocs))
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def sum_half(input_slot:int, width:int, out_slot:int|None=None, out_offset=0, raw_fp32=False) -> int|None:
    red = UOp.range(width, 5100, AxisType.REDUCE)
    source_param = UOp.param(input_slot, dtypes.half, (width,), device=device)
    source_index = UOp(Ops.INDEX, dtypes.half, (source_param, red))
    acc = UOp(Ops.REDUCE, dtypes.float,
              (UOp(Ops.CAST, dtypes.float, (source_index,), arg=dtypes.float), red), arg=(Ops.ADD, 0.0))
    if out_slot is None: out_slot = alloc()
    out_param = UOp.param(out_slot, dtypes.half, (1,), device=device)
    out_index = UOp(Ops.INDEX, dtypes.half, (out_param, UOp.const(dtypes.weakint, 0)))
    sum_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(out_index, UOp(Ops.CAST, dtypes.half, (acc,), arg=dtypes.half))),)))
    sum_plan = plan_rk(sum_sink)
    if isinstance(sum_plan, str) or sum_plan.kind != "cmac": return None
    cmds, task, relocs = emit_rk(sum_plan)
    tasks.append(RKSubTask(cmds, replace(task, fp32_output=raw_fp32, out_offset=out_offset), relocs))
    return out_slot

  chunk_width = min(K, 4096)
  chunks = ceildiv(K, chunk_width)
  high_partials, low_partials = alloc(), alloc()
  high_chunk, low_chunk = alloc(), alloc()
  for chunk in range(chunks):
    start, count = chunk*chunk_width, min(chunk_width, K-chunk*chunk_width)
    layout = (2, 1, count, K, 1, start)
    view(source_slot, _HOST_FP32_HALF_LAYOUT, count, layout, high_chunk)
    view(source_slot, _HOST_FP32_RESIDUAL_LAYOUT, count, layout, low_chunk)
    if sum_half(high_chunk, count, high_partials, chunk*4, raw_fp32=True) is None or \
       sum_half(low_chunk, count, low_partials, chunk*4, raw_fp32=True) is None: return None

  def finish_raw(partials:int) -> int|None:
    partial_high = view(partials, _HOST_FP32_HALF_LAYOUT, chunks)
    partial_low = view(partials, _HOST_FP32_RESIDUAL_LAYOUT, chunks)
    high_sum, low_sum = sum_half(partial_high, chunks, raw_fp32=True), sum_half(partial_low, chunks)
    if high_sum is None or low_sum is None: return None
    high_half = view(high_sum, _HOST_FP32_HALF_LAYOUT, 1)
    high_residual = view(high_sum, _HOST_FP32_RESIDUAL_LAYOUT, 1)
    combined_residual = stage((high_residual, 0), (low_sum, 0), Ops.ADD)
    raw = alloc()
    combine_raw(high_half, combined_residual, raw)
    return raw

  high_raw, low_raw = finish_raw(high_partials), finish_raw(low_partials)
  if high_raw is None or low_raw is None: return None
  high_half = view(high_raw, _HOST_FP32_HALF_LAYOUT, 1)
  high_residual = view(high_raw, _HOST_FP32_RESIDUAL_LAYOUT, 1)
  low_half = view(low_raw, _HOST_FP32_HALF_LAYOUT, 1)
  low_residual = view(low_raw, _HOST_FP32_RESIDUAL_LAYOUT, 1)
  low_value = stage((low_half, 0), (stage((low_residual, 0), scalar(1/256), Ops.MUL), 0), Ops.ADD)
  final_residual = stage((high_residual, 0), (low_value, 0), Ops.ADD)
  combine_raw(high_half, final_residual, info.outs[0])
  return tuple(tasks)

def _try_fp32_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Give CMAC a typed half view of a small direct fp32 input, then widen its result."""
  if (long_sum_tasks := _try_long_fp32_sum_subtasks(sink)) is not None: return long_sum_tasks
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or store.src[1] is not reduce or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  source = _unwrap(reduce.src[0])
  if source.op is not Ops.INDEX or source.dtype is not dtypes.float or source.src[0].op is not Ops.PARAM: return None
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None

  info = ProgramInfo.from_sink(sink)
  source_slot = source.src[0].buf_uop.arg.slot
  scratch_slot = max(info.globals, default=-1) + 1
  if source_slot >= 7: return None
  source_total, output_total = int(source.src[0].src[0].arg), prod(_shape_of_store(sink))
  # WIP reference: `if source_total > 256: return None` confused backing
  # storage with resident CMAC geometry. Axis sums materialize only their
  # CBUF-derived M/N/K tiles, just like conv_grok local windows.
  tasks:list[RKSubTask] = []

  device = store.src[0].src[0].device
  if source_total > 0:
    next_slot = scratch_slot
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    high_slot, low_slot = next_slot, next_slot+1
    next_slot += 2
    for out_slot, layout_tag in ((high_slot, _HOST_FP32_HALF_LAYOUT), (low_slot, _HOST_FP32_RESIDUAL_LAYOUT)):
      host_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", (source_total, layout_tag),
                                              out_slot, is_copy=True), host_relocs))

    def cmac_sum(input_slot:int) -> int|None:
      nonlocal next_slot
      out_slot, next_slot = next_slot, next_slot+1
      converted_param = UOp.param(input_slot, dtypes.half, (source_total,), device=device)
      converted_index = source.replace(dtype=dtypes.half, src=(converted_param, *source.src[1:]))
      converted_reduce = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (converted_index,), arg=dtypes.float), *reductions))
      out_param = UOp.param(out_slot, dtypes.half, (output_total,), device=device)
      out_index = store.src[0].replace(dtype=dtypes.half, src=(out_param, *store.src[0].src[1:]))
      stage_store = store.replace(src=(out_index, UOp(Ops.CAST, dtypes.half, (converted_reduce,), arg=dtypes.half)))
      stage_plan = plan_rk(sink.substitute({store:stage_store}))
      if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
      if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
      elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
      else:
        cmds, task, relocs = emit_rk(stage_plan)
        tasks.append(RKSubTask(cmds, task, relocs))
      return out_slot

    high_sum, low_sum = cmac_sum(high_slot), cmac_sum(low_slot)
    if high_sum is None or low_sum is None: return None
    correction, next_slot = next_slot, next_slot+1
    tasks.append(_emit_where_stage(output_total, correction, (low_sum, 0),
                                   (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1/256))[0]), Ops.MUL))
    tasks.append(_emit_where_stage(output_total, info.outs[0], (high_sum, 0), (correction, 0), Ops.ADD, fp32_output=True))
    return tuple(tasks)

  # WIP reference: the former <=16 shortcut tagged the fp32 source directly
  # on one half CMAC and widened its rounded result. Axis sums near zero lost
  # enough input residual to miss normal-fp32 tolerance.
  converted_param = UOp.param(source_slot, dtypes.half, (source_total,), device=device)
  converted_index = source.replace(dtype=dtypes.half, src=(converted_param, *source.src[1:]))
  converted_reduce = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (converted_index,), arg=dtypes.float), *reductions))
  scratch_param = UOp.param(scratch_slot, dtypes.half, (output_total,), device=device)
  scratch_index = store.src[0].replace(dtype=dtypes.half, src=(scratch_param, *store.src[0].src[1:]))
  stage_store = store.replace(src=(scratch_index, UOp(Ops.CAST, dtypes.half, (converted_reduce,), arg=dtypes.half)))
  stage_sink = sink.substitute({store:stage_store})
  stage_plan = plan_rk(stage_sink)
  if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
  if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(stage_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(stage_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  tasks = [RKSubTask(st.cmds, replace(st.task, fp32_inputs=tuple(set(st.task.fp32_inputs+(source_slot,)))), st.relocs)
           if st.task.kind == "cmac" and any(r.globals_slot == source_slot for r in st.relocs) else st for st in tasks]
  tasks.append(_emit_where_stage(output_total, info.outs[0], (scratch_slot, 0), (_ZERO_SLOT, 0), Ops.ADD, fp32_output=True))
  return tuple(tasks)

def _try_fp32_add_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Add/subtract up to three direct fp32 operands with fp16 TwoSum arithmetic and a split fp32 ABI result."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.ADD or val.dtype is not dtypes.float: return None
  signed_sources:list[tuple[UOp,int]] = []
  def flatten_add(node:UOp, sign:int=1) -> None:
    node = _unwrap(node)
    if node.op is Ops.ADD:
      for child in node.src: flatten_add(child, sign)
      return
    if node.op is Ops.MUL and len(node.src) == 2:
      for constant_pos, value_pos in ((0, 1), (1, 0)):
        constant, value = node.src[constant_pos], node.src[value_pos]
        if constant.op is Ops.CONST and float(constant.arg) == -1.0:
          flatten_add(value, -sign)
          return
    signed_sources.append((node, sign))
  flatten_add(val)
  sources = tuple(source for source, _ in signed_sources)
  if len(sources) not in (2, 3): return None
  if any(x.dtype is not dtypes.float or x.op not in (Ops.INDEX, Ops.CONST) or
         (x.op is Ops.INDEX and x.src[0].op is not Ops.PARAM) for x in sources): return None
  store_aff = _affine_index(store.src[0].src[1])
  if store_aff is None: return None
  extents = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort()
             if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP" and u.src[0].op is Ops.CONST}
  axes = sorted(store_aff[0], key=lambda x:store_aff[0][x], reverse=True)
  # WIP: ndim=0 fp32 views can represent scalar ADD, but full-tensor mean only
  # needs scalar MUL and does not justify widening this separate matcher.
  # if (not axes and (store_aff[1] != 0 or prod(_shape_of_store(sink)) != 1)) or \
  #    not all(axis in extents for axis in axes): return None
  if not axes or not all(axis in extents for axis in axes): return None
  out_shape = tuple(extents[axis] for axis in axes)
  total = prod(out_shape)
  if total != prod(_shape_of_store(sink)): return None
  source_layouts:list[tuple[int,...]|None] = []
  for source in sources:
    if source.op is Ops.CONST:
      source_layouts.append(None)
      continue
    aff = _affine_index(source.src[1])
    if aff is None: return None
    if any(axis not in axes or stride < 0 for axis, stride in aff[0].items()): return None
    strides = tuple(aff[0].get(axis, 0) for axis in axes)
    source_total = int(source.src[0].src[0].arg)
    max_index = aff[1] + sum((extent-1)*stride for extent, stride in zip(out_shape, strides))
    if aff[1] < 0 or max_index >= source_total: return None
    source_layouts.append((len(out_shape), *out_shape, *strides, aff[1]))

  info = ProgramInfo.from_sink(sink)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def view(source_slot:int, tag:int, layout:tuple[int,...]) -> int:
    out = alloc()
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, tag, *layout), out, is_copy=True), relocs))
    return out
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def stage(a:tuple[int,int], b:tuple[int,int], op:Ops) -> tuple[int,int]:
    out = alloc()
    tasks.append(_emit_where_stage(total, out, a, b, op))
    return out, 0
  def limb(source:UOp, sign:int, tag:int, layout:tuple[int,...]|None) -> tuple[int,int]:
    if source.op is Ops.CONST:
      original = np.float32(sign*float(source.arg))
      high = np.float16(original)
      value = high if tag == _HOST_FP32_HALF_LAYOUT else np.float16((original-np.float32(high))*256.0)
      return scalar(float(value))
    assert layout is not None
    return view(source.src[0].arg.slot, tag, layout), 0

  limbs_list:list[tuple[tuple[int,int],tuple[int,int]]] = []
  for (source, sign), layout in zip(signed_sources, source_layouts):
    high_limb = limb(source, sign, _HOST_FP32_HALF_LAYOUT, layout)
    low_limb = limb(source, sign, _HOST_FP32_RESIDUAL_LAYOUT, layout)
    if sign < 0 and source.op is not Ops.CONST:
      high_limb, low_limb = stage(scalar(0.0), high_limb, Ops.SUB), stage(scalar(0.0), low_limb, Ops.SUB)
    limbs_list.append((high_limb, low_limb))
  limbs = tuple(limbs_list)
  high, low = limbs[0]
  for next_high, next_low in limbs[1:]:
    old_high = high
    high = stage(old_high, next_high, Ops.ADD)
    rounded_b = stage(high, old_high, Ops.SUB)
    high_minus_rounded_b = stage(high, rounded_b, Ops.SUB)
    error_a = stage(old_high, high_minus_rounded_b, Ops.SUB)
    error_b = stage(next_high, rounded_b, Ops.SUB)
    high_error = stage(error_a, error_b, Ops.ADD)
    input_low = stage(low, next_low, Ops.ADD)
    scaled_high_error = stage(high_error, scalar(256.0), Ops.MUL)
    low = stage(input_low, scaled_high_error, Ops.ADD)

  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, high[0], 0, 0, 0xFFFFFFFF),
            RKReloc(0, low[0], 0, 0, 0xFFFFFFFF))
  tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_COMBINE_LAYOUT),
                                     info.outs[0], is_copy=True), relocs))
  return tuple(tasks)

def _try_fp32_mul_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Multiply direct fp32 operands with fp16 TwoProduct arithmetic and a split fp32 ABI result."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  val = _unwrap(store.src[1])
  if val.op is not Ops.MUL or val.dtype is not dtypes.float or len(val.src) != 2: return None
  sources = tuple(_unwrap(source) for source in val.src)
  if any(x.dtype is not dtypes.float or x.op not in (Ops.INDEX, Ops.CONST) or
         (x.op is Ops.INDEX and x.src[0].op is not Ops.PARAM) for x in sources): return None
  store_aff = _affine_index(store.src[0].src[1])
  if store_aff is None: return None
  extents = {u.arg[0]:int(u.src[0].arg) for u in sink.toposort()
             if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP" and u.src[0].op is Ops.CONST}
  axes = sorted(store_aff[0], key=lambda x:store_aff[0][x], reverse=True)
  # WIP reference: `if not axes ...` rejected the one-element scalar epilogue
  # of full-tensor mean. ndim=0 fp32 views are an established total=1 layout.
  # if not axes or not all(axis in extents for axis in axes): return None
  if (not axes and (store_aff[1] != 0 or prod(_shape_of_store(sink)) != 1)) or \
     not all(axis in extents for axis in axes): return None
  out_shape = tuple(extents[axis] for axis in axes)
  total = prod(out_shape)
  if total != prod(_shape_of_store(sink)): return None
  source_layouts:list[tuple[int,...]|None] = []
  for source in sources:
    if source.op is Ops.CONST:
      source_layouts.append(None)
      continue
    aff = _affine_index(source.src[1])
    if aff is None: return None
    if any(axis not in axes or stride < 0 for axis, stride in aff[0].items()): return None
    strides = tuple(aff[0].get(axis, 0) for axis in axes)
    source_total = int(source.src[0].src[0].arg)
    max_index = aff[1] + sum((extent-1)*stride for extent, stride in zip(out_shape, strides))
    if aff[1] < 0 or max_index >= source_total: return None
    source_layouts.append((len(out_shape), *out_shape, *strides, aff[1]))

  info = ProgramInfo.from_sink(sink)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def view(source_slot:int, tag:int, layout:tuple[int,...]) -> tuple[int,int]:
    out = alloc()
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, tag, *layout), out, is_copy=True), relocs))
    return out, 0
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def stage(a:tuple[int,int], b:tuple[int,int], op:Ops) -> tuple[int,int]:
    out = alloc()
    tasks.append(_emit_where_stage(total, out, a, b, op))
    return out, 0
  def limb(source:UOp, tag:int, layout:tuple[int,...]|None) -> tuple[int,int]:
    if source.op is Ops.CONST:
      original = np.float32(float(source.arg))
      high = np.float16(original)
      value = high if tag == _HOST_FP32_HALF_LAYOUT else np.float16((original-np.float32(high))*256.0)
      return scalar(float(value))
    assert layout is not None
    return view(source.src[0].arg.slot, tag, layout)

  (a_high, a_low), (b_high, b_low) = tuple((limb(source, _HOST_FP32_HALF_LAYOUT, layout),
                                             limb(source, _HOST_FP32_RESIDUAL_LAYOUT, layout))
                                            for source, layout in zip(sources, source_layouts))
  product = stage(a_high, b_high, Ops.MUL)

  # Dekker split with splitter 2**6+1. Product error and input cross terms are
  # accumulated as an x256 residual for the final split-fp32 ABI decode.
  split_a = stage(a_high, scalar(65.0), Ops.MUL)
  big_a = stage(split_a, a_high, Ops.SUB)
  head_a = stage(split_a, big_a, Ops.SUB)
  tail_a = stage(a_high, head_a, Ops.SUB)
  split_b = stage(b_high, scalar(65.0), Ops.MUL)
  big_b = stage(split_b, b_high, Ops.SUB)
  head_b = stage(split_b, big_b, Ops.SUB)
  tail_b = stage(b_high, head_b, Ops.SUB)
  # Form the small unscaled error before multiplying it by 256. Scaling the
  # rounded product itself would overflow fp16 for otherwise finite 255*x.
  error = stage(stage(head_a, head_b, Ops.MUL), product, Ops.SUB)
  error = stage(error, stage(head_a, tail_b, Ops.MUL), Ops.ADD)
  error = stage(error, stage(tail_a, head_b, Ops.MUL), Ops.ADD)
  error = stage(error, stage(tail_a, tail_b, Ops.MUL), Ops.ADD)
  error = stage(error, scalar(256.0), Ops.MUL)
  error = stage(error, stage(a_high, b_low, Ops.MUL), Ops.ADD)
  error = stage(error, stage(a_low, b_high, Ops.MUL), Ops.ADD)
  low_low = stage(stage(a_low, b_low, Ops.MUL), scalar(1.0/256.0), Ops.MUL)
  residual = stage(error, low_low, Ops.ADD)

  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, product[0], 0, 0, 0xFFFFFFFF),
            RKReloc(0, residual[0], 0, 0, 0xFFFFFFFF))
  tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_COMBINE_LAYOUT),
                                     info.outs[0], is_copy=True), relocs))
  return tuple(tasks)

def _try_variance_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize only tinygrad's two-pass fp32 variance kernel after its mean kernel."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.float: return None
  val = _unwrap(store.src[1])
  final_sqrt, stack_axis, mean_output = 0, None, None
  if val.op is Ops.WHERE and len(val.src) == 3:
    condition, mean_output, val = (_unwrap(x) for x in val.src)
    if condition.op is not Ops.CMPNE or len(condition.src) != 2: return None
    for candidate, zero in ((condition.src[0], condition.src[1]), (condition.src[1], condition.src[0])):
      candidate, zero = _unwrap(candidate), _unwrap(zero)
      if candidate.op is Ops.RANGE and getattr(candidate.arg[-1], "name", "") == "LOOP" and \
         candidate.src[0].op is Ops.CONST and int(candidate.src[0].arg) == 2 and \
         zero.op is Ops.CONST and int(zero.arg) == 0:
        stack_axis = candidate
        break
    if stack_axis is None: return None
    if val.op is not Ops.SQRT or val.dtype is not dtypes.float or len(val.src) != 1: return None
    val = _unwrap(val.src[0])
    final_sqrt = 2
  elif val.op is Ops.SQRT and val.dtype is dtypes.float and len(val.src) == 1:
    final_sqrt, val = 1, _unwrap(val.src[0])
  reductions_in_value = [u for u in val.toposort() if u.op is Ops.REDUCE]
  if len(reductions_in_value) != 1: return None
  reduce = reductions_in_value[0]
  if reduce.dtype is not dtypes.float or reduce.arg[0] is not Ops.ADD: return None
  if val.op is not Ops.MUL or len(val.src) != 2: return None
  scale, reduced = None, None
  for lhs, rhs in ((val.src[0], val.src[1]), (val.src[1], val.src[0])):
    if _unwrap(lhs) is reduce and _unwrap(rhs).op is Ops.CONST:
      reduced, scale = _unwrap(lhs), _unwrap(rhs)
      break
  if reduced is None or scale is None or scale.dtype is not dtypes.float or \
     math.isnan(float(scale.arg)) or float(scale.arg) <= 0.0: return None
  square = _unwrap(reduce.src[0])
  if square.op is not Ops.MUL or len(square.src) != 2 or _unwrap(square.src[0]) is not _unwrap(square.src[1]): return None
  delta = _unwrap(square.src[0])
  if delta.op is not Ops.ADD or len(delta.src) != 2: return None

  data, mean = None, None
  for candidate, negative in ((delta.src[0], delta.src[1]), (delta.src[1], delta.src[0])):
    candidate, negative = _unwrap(candidate), _unwrap(negative)
    if candidate.op is not Ops.INDEX or candidate.src[0].op is not Ops.PARAM or negative.op is not Ops.MUL: continue
    mean_candidate = next((_unwrap(x) for x in negative.src if _unwrap(x).op is Ops.INDEX), None)
    coefficient = next((_unwrap(x) for x in negative.src if _unwrap(x).op is Ops.CONST), None)
    if mean_candidate is not None and mean_candidate.src[0].op is Ops.PARAM and coefficient is not None and \
       coefficient.dtype is dtypes.float and math.isfinite(float(coefficient.arg)) and float(coefficient.arg) < 0.0:
      data, mean = candidate, mean_candidate
      break
  if data is None or mean is None or data.dtype is not dtypes.float or mean.dtype is not dtypes.float: return None

  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  axes = [*loops, *reductions]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in axes): return None
  extents = tuple(int(u.src[0].arg) for u in axes)
  axis_ids = tuple(u.arg[0] for u in axes)
  loop_ids, reduction_ids = set(axis_ids[:len(loops)]), set(axis_ids[len(loops):])
  if stack_axis is not None and stack_axis not in loops: return None
  data_aff, mean_aff, out_aff = _affine_index(data.src[1]), _affine_index(mean.src[1]), _affine_index(store.src[0].src[1])
  if data_aff is None or mean_aff is None or out_aff is None or \
     any(axis not in axis_ids or stride < 0 for aff in (data_aff, mean_aff) for axis, stride in aff[0].items()) or \
     any(axis not in loop_ids or stride < 0 for axis, stride in out_aff[0].items()) or \
     any(mean_aff[0].get(axis, 0) != 0 for axis in reduction_ids) or \
     not any(data_aff[0].get(axis, 0) != 0 for axis in reduction_ids): return None

  def bounded(aff:tuple[dict[int,int],int], total:int, used_extents:tuple[int,...]) -> bool:
    maximum = aff[1] + sum((extent-1)*aff[0].get(axis, 0) for axis, extent in zip(axis_ids, used_extents))
    return aff[1] >= 0 and maximum < total
  data_total, mean_total = int(data.src[0].src[0].arg), int(mean.src[0].src[0].arg)
  output_total = prod(_shape_of_store(sink))
  if not bounded(data_aff, data_total, extents) or not bounded(mean_aff, mean_total, extents) or \
     not bounded(out_aff, output_total, (*extents[:len(loops)], *([1]*len(reductions)))): return None
  if stack_axis is not None:
    stack_id = stack_axis.arg[0]
    if data_aff[0].get(stack_id, 0) != 0 or mean_aff[0].get(stack_id, 0) != 0: return None
    assert mean_output is not None and coefficient is not None
    mean_output = _unwrap(mean_output)
    indexed_mean, output_scale = None, 1.0
    if mean_output.op is Ops.INDEX: indexed_mean = mean_output
    elif mean_output.op is Ops.MUL and len(mean_output.src) == 2:
      indexed_mean = next((_unwrap(x) for x in mean_output.src if _unwrap(x).op is Ops.INDEX), None)
      output_constant = next((_unwrap(x) for x in mean_output.src if _unwrap(x).op is Ops.CONST), None)
      if output_constant is None or output_constant.dtype is not dtypes.float: return None
      output_scale = float(output_constant.arg)
    if indexed_mean is not None:
      if indexed_mean.src[0] is not mean.src[0] or _affine_index(indexed_mean.src[1]) != mean_aff or \
         not math.isclose(output_scale, -float(coefficient.arg), rel_tol=0.0, abs_tol=0.0): return None
    else:
      mean_reductions = [u for u in mean_output.toposort() if u.op is Ops.REDUCE]
      if mean_output.op is not Ops.MUL or len(mean_reductions) != 1 or mean_reductions[0].arg[0] is not Ops.ADD: return None
      mean_reduce = mean_reductions[0]
      mean_source = _unwrap(mean_reduce.src[0])
      mean_scale = next((_unwrap(x) for x in mean_output.src if _unwrap(x).op is Ops.CONST), None)
      mean_axes = list(mean_reduce.src[1:])
      if mean_source.op is not Ops.INDEX or mean_source.src[0] is not data.src[0] or mean_scale is None or \
         mean_scale.dtype is not dtypes.float or any(axis.src[0].op is not Ops.CONST for axis in mean_axes) or \
         prod(int(axis.src[0].arg) for axis in mean_axes) != prod(extents[len(loops):]) or \
         not math.isclose(float(mean_scale.arg), 1.0/prod(extents[len(loops):]), rel_tol=1e-12): return None

  def mapping(aff:tuple[dict[int,int],int], ids:tuple[int,...]) -> tuple[int,...]:
    return (aff[1], *(aff[0].get(axis, 0) for axis in ids))
  scale_bits = _signed_i32(struct.unpack('<I', struct.pack('<f', float(scale.arg)))[0])
  stack_position = loops.index(stack_axis) if stack_axis is not None else -1
  layout = (output_total, _HOST_VARIANCE_LAYOUT, len(loops), len(reductions), int(final_sqrt),
            *((stack_position,) if stack_axis is not None else ()), *extents, scale_bits,
            *mapping(data_aff, axis_ids), *mapping(mean_aff, axis_ids), *mapping(out_aff, axis_ids[:len(loops)]))
  info = ProgramInfo.from_sink(sink)
  slots = (info.outs[0], data.src[0].buf_uop.arg.slot, mean.src[0].buf_uop.arg.slot)
  relocs = tuple(RKReloc(0, slot, 0, 0, 0xFFFFFFFF) for slot in slots)
  task = RKTask(0, 0, 0, "dpu", layout, info.outs[0], is_copy=True, fp32_output=True)
  return (RKSubTask((RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),), task, relocs),)

def _try_fp32_avg_pool_host_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Serialize only bounded normal-fp32 average-pool reductions."""
  store = _store_node(sink)
  data_reduces = [u for u in sink.toposort() if u.op is Ops.REDUCE and u.dtype is dtypes.float and
                  u.arg[0] is Ops.ADD and any(x.op is Ops.INDEX and x.dtype is dtypes.float for x in u.src[0].toposort())]
  if store is None or len(data_reduces) != 1 or store.src[0].dtype is not dtypes.float: return None
  reduce = data_reduces[0]
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  # Leave plain SUM and scalar full-MEAN on their existing typed CMAC paths.
  if not loops or _unwrap(store.src[1]) is reduce: return None
  # Unit kernel dimensions disappear during simplification, so 2D/3D pools
  # can retain fewer reduction axes than their declared spatial rank.
  if len(reductions) not in (1, 2, 3) or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  loop_extents, reduce_extents = [int(u.src[0].arg) for u in loops], [int(u.src[0].arg) for u in reductions]
  total, window = prod(_shape_of_store(sink)), prod(reduce_extents)
  if prod(loop_extents) != total or not 1 <= window <= 1024: return None

  body = _unwrap(reduce.src[0])
  indexes = [u for u in body.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM]
  if len(indexes) != 1: return None
  source = indexes[0]
  unit = store.src[1].substitute({reduce:UOp.const(dtypes.float, 1.0)})
  if any(u.op in (Ops.PARAM, Ops.INDEX) for u in unit.toposort()): return None

  def evaluate(u:UOp, coords:dict[UOp, int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return Invalid if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    if u.op is Ops.WHERE:
      condition = evaluate(u.src[0], coords)
      return evaluate(u.src[1] if condition else u.src[2], coords)
    if u.op is Ops.REDUCE:
      if u.arg[0] is not Ops.ADD or any(axis.src[0].op is not Ops.CONST for axis in u.src[1:]): raise ValueError(u.op)
      axes, result = u.src[1:], 0
      extents = [int(axis.src[0].arg) for axis in axes]
      for linear in range(prod(extents)):
        rem, nested = linear, dict(coords)
        for axis in range(len(axes)-1, -1, -1): rem, nested[axes[axis]] = divmod(rem, extents[axis])
        result += evaluate(u.src[0], nested)
      return result
    values = [evaluate(x, coords) for x in u.src]
    if any(value is Invalid for value in values): return Invalid
    if u.op is Ops.ADD: return values[0]+values[1]
    if u.op is Ops.MUL: return values[0]*values[1]
    if u.op is Ops.MAX: return max(values[0], values[1])
    if u.op is Ops.FDIV: return values[0]/values[1]
    if u.op is Ops.RECIPROCAL: return 1/values[0]
    if u.op is Ops.FLOORDIV: return values[0]//values[1]
    if u.op is Ops.FLOORMOD: return values[0]%values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
    if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
    if u.op is Ops.XOR: return int(values[0]) ^ int(values[1])
    raise ValueError(u.op)

  def selects_source(u:UOp, coords:dict[UOp, int]) -> bool:
    while u.op is Ops.CAST: u = u.src[0]
    if u is source: return True
    if u.op is Ops.WHERE:
      condition = evaluate(u.src[0], coords)
      return selects_source(u.src[1] if condition else u.src[2], coords)
    return False

  mapping = [-1]*(total*window)
  scales = [0]*total
  try:
    for output_linear in range(total):
      rem, coords = output_linear, {}
      for axis in range(len(loops)-1, -1, -1): rem, coords[loops[axis]] = divmod(rem, loop_extents[axis])
      output_index = int(evaluate(store.src[0].src[1], coords))
      if not 0 <= output_index < total: return None
      scale = evaluate(unit, coords)
      if scale is Invalid or not math.isfinite(float(scale)) or float(scale) <= 0.0: return None
      scales[output_index] = _signed_i32(struct.unpack('<I', struct.pack('<f', float(scale)))[0])
      for candidate in range(window):
        rem, fixed = candidate, dict(coords)
        for axis in range(len(reductions)-1, -1, -1): rem, fixed[reductions[axis]] = divmod(rem, reduce_extents[axis])
        address = evaluate(source.src[1], fixed)
        if selects_source(body, fixed) and address is not Invalid:
          mapping[output_index*window+candidate] = int(address)
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None
  if any(scale == 0 for scale in scales): return None

  info = ProgramInfo.from_sink(sink)
  out_slot, source_slot = info.outs[0], source.src[0].buf_uop.arg.slot
  layout = (total, _HOST_AVG_POOL_LAYOUT, window, *scales, *mapping)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
  return (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out_slot, is_copy=True), relocs),)

def _try_fp32_factorized_sum_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower optimizer-factorized SUM(x)*output_factor*constant in its original fp32 order."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or reduce.dtype is not dtypes.float or \
     reduce.arg[0] is not Ops.ADD or _unwrap(reduce.src[0]).op is not Ops.INDEX: return None

  def contains_reduce(u:UOp) -> bool: return any(node is reduce for node in u.toposort())
  def peel_factors(u:UOp) -> list[UOp]|None:
    u = _unwrap(u)
    if u is reduce: return []
    if u.op is not Ops.MUL or len(u.src) != 2: return None
    lhs, rhs = (_unwrap(x) for x in u.src)
    lhs_has, rhs_has = contains_reduce(lhs), contains_reduce(rhs)
    if lhs_has == rhs_has: return None
    nested, factor = (lhs, rhs) if lhs_has else (rhs, lhs)
    factors = peel_factors(nested)
    return None if factors is None else [*factors, factor]

  factors = peel_factors(store.src[1])
  if factors is None or not 1 <= len(factors) <= 3: return None
  reductions = list(reduce.src[1:])
  if any(factor.dtype is not dtypes.float or factor.op not in (Ops.INDEX, Ops.CONST) or
         (factor.op is Ops.INDEX and factor.src[0].op is not Ops.PARAM) or
         any(axis in factor.toposort() for axis in reductions) for factor in factors): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  # WIP reference: requiring `loops` rejected scalar-output mean even though the
  # staged SUM and factorized epilogue both support the zero-LOOP geometry.
  # if not loops or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  if any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None

  info, output_total = ProgramInfo.from_sink(sink), prod(_shape_of_store(sink))
  tasks:list[RKSubTask] = []
  next_slot = max(info.globals, default=-1)+1
  device = store.src[0].src[0].device

  def index_for(slot:int) -> UOp:
    param = UOp.param(slot, dtypes.float, (output_total,), device=device)
    return store.src[0].replace(src=(param, *store.src[0].src[1:]))
  def stage_sink(out_slot:int, value:UOp) -> UOp:
    return UOp.sink(UOp(Ops.END, src=(store.replace(src=(index_for(out_slot), value)), *loops)))
  def reserve_after(stage_tasks:tuple[RKSubTask, ...]) -> int:
    used = [r.globals_slot for task in stage_tasks for r in task.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)]
    return max((next_slot, *used), default=next_slot-1)+1

  sum_slot = next_slot
  sum_tasks = _try_fp32_sum_subtasks(stage_sink(sum_slot, reduce))
  if sum_tasks is None: return None
  tasks.extend(sum_tasks)
  next_slot = reserve_after(sum_tasks)
  accumulator = sum_slot
  for index, factor in enumerate(factors):
    final = index == len(factors)-1
    out_slot = info.outs[0] if final else next_slot
    product = UOp(Ops.MUL, dtypes.float, (index_for(accumulator), factor))
    product_tasks = _try_fp32_mul_subtasks(stage_sink(out_slot, product))
    if product_tasks is None: return None
    tasks.extend(product_tasks)
    if not final:
      next_slot = reserve_after(product_tasks)
      accumulator = out_slot
  return tuple(tasks)

def _try_long_fp32_batched_dot_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Reduce contiguous long fp32 row dots through DPU products and two CMAC sum levels."""
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or store.src[1] is not reduce or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  body = _unwrap(reduce.src[0])
  if body.op is not Ops.MUL or body.dtype is not dtypes.float or len(body.src) != 2: return None
  sources = tuple(_unwrap(x) for x in body.src)
  if any(x.op is not Ops.INDEX or x.dtype is not dtypes.float or x.src[0].op is not Ops.PARAM for x in sources): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  if len(loops) != 1 or len(reductions) != 1 or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  rows, K = int(loops[0].src[0].arg), int(reductions[0].src[0].arg)
  if K <= 4096: return None
  # CNA_DMA_CON1 has only 13 useful 32-channel groups. K=512 corrupts rows
  # after the first, while hardware probes prove both K=384 and K=416.
  chunks = ceildiv(K, 416)
  while chunks <= K and K % chunks: chunks += 1
  if chunks > K: return None
  chunk_k = K // chunks
  out_aff = _affine_index(store.src[0].src[1])
  if out_aff != ({loops[0].arg[0]:1}, 0): return None
  for source in sources:
    aff = _affine_index(source.src[1])
    if aff != ({loops[0].arg[0]:K, reductions[0].arg[0]:1}, 0) or int(source.src[0].src[0].arg) != rows*K: return None

  info, device = ProgramInfo.from_sink(sink), store.src[0].src[0].device
  source_slots = tuple(x.src[0].buf_uop.arg.slot for x in sources)
  next_slot = max(info.globals, default=-1)+1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def view(source_slot:int, tag:int, total:int, layout:tuple[int,...]=(), out_slot:int|None=None) -> int:
    out = alloc() if out_slot is None else out_slot
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, tag, *layout), out, is_copy=True), relocs))
    return out
  def stage(total:int, a:tuple[int,int], b:tuple[int,int], op:Ops, out_slot:int|None=None) -> int:
    out = alloc() if out_slot is None else out_slot
    tasks.append(_emit_where_stage(total, out, a, b, op))
    return out
  def row_sum(input_slot:int, nrows:int, width:int, raw_fp32=False, transposed=False,
              out_slot:int|None=None, out_offset=0) -> int|None:
    row, red = UOp.range(nrows, 5000), UOp.range(width, 5001, AxisType.REDUCE)
    source_param = UOp.param(input_slot, dtypes.half, (nrows*width,), device=device)
    source_index = UOp(Ops.INDEX, dtypes.half, (source_param, red*nrows+row if transposed else row*width+red))
    acc = UOp(Ops.REDUCE, dtypes.float, (UOp(Ops.CAST, dtypes.float, (source_index,), arg=dtypes.float), red), arg=(Ops.ADD, 0.0))
    if out_slot is None: out_slot = alloc()
    out_param = UOp.param(out_slot, dtypes.half, (nrows,), device=device)
    out_index = UOp(Ops.INDEX, dtypes.half, (out_param, row))
    sum_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(out_index, UOp(Ops.CAST, dtypes.half, (acc,), arg=dtypes.half))), row)))
    sum_plan = plan_rk(sum_sink)
    if isinstance(sum_plan, str) or sum_plan.kind != "cmac": return None
    cmds, task, relocs = emit_rk(sum_plan)
    tasks.append(RKSubTask(cmds, replace(task, fp32_output=raw_fp32, out_offset=out_offset), relocs))
    return out_slot
  def combine_raw(high_slot:int, low_slot:int, out_slot:int, total:int) -> None:
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, high_slot, 0, 0, 0xFFFFFFFF),
              RKReloc(0, low_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (total, _HOST_FP32_COMBINE_LAYOUT),
                                       out_slot, is_copy=True, fp32_output=True), relocs))

  high_partials, error_partials, cross_partials = alloc(), alloc(), (alloc(), alloc())
  compact_total = rows*chunk_k
  limb_slots = ((alloc(), alloc()), (alloc(), alloc()))
  product_slot = alloc()
  split_slot, big_slot = alloc(), alloc()
  head_a, tail_a, head_b, tail_b = alloc(), alloc(), alloc(), alloc()
  error_slot, term_slot = alloc(), alloc()
  scalar_65 = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 65.0))[0])
  scalar_256 = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 256.0))[0])
  for chunk in range(chunks):
    layout = (2, rows, chunk_k, K, 1, chunk*chunk_k)
    for source_slot, (high_slot, low_slot) in zip(source_slots, limb_slots):
      view(source_slot, _HOST_FP32_HALF_LAYOUT, compact_total, layout, high_slot)
      view(source_slot, _HOST_FP32_RESIDUAL_LAYOUT, compact_total, layout, low_slot)
    stage(compact_total, (limb_slots[0][0], 0), (limb_slots[1][0], 0), Ops.MUL, product_slot)
    if row_sum(product_slot, rows, chunk_k, raw_fp32=True,
               out_slot=high_partials, out_offset=chunk*rows*4) is None: return None
    # Recover the fp16 product-rounding error with Dekker's 2**6+1 split,
    # then keep it in x256 residual units for the final split-fp32 sum.
    stage(compact_total, (limb_slots[0][0], 0), scalar_65, Ops.MUL, split_slot)
    stage(compact_total, (split_slot, 0), (limb_slots[0][0], 0), Ops.SUB, big_slot)
    stage(compact_total, (split_slot, 0), (big_slot, 0), Ops.SUB, head_a)
    stage(compact_total, (limb_slots[0][0], 0), (head_a, 0), Ops.SUB, tail_a)
    stage(compact_total, (limb_slots[1][0], 0), scalar_65, Ops.MUL, split_slot)
    stage(compact_total, (split_slot, 0), (limb_slots[1][0], 0), Ops.SUB, big_slot)
    stage(compact_total, (split_slot, 0), (big_slot, 0), Ops.SUB, head_b)
    stage(compact_total, (limb_slots[1][0], 0), (head_b, 0), Ops.SUB, tail_b)
    stage(compact_total, (head_a, 0), (head_b, 0), Ops.MUL, term_slot)
    stage(compact_total, (term_slot, 0), (product_slot, 0), Ops.SUB, error_slot)
    for lhs, rhs in ((head_a, tail_b), (tail_a, head_b), (tail_a, tail_b)):
      stage(compact_total, (lhs, 0), (rhs, 0), Ops.MUL, term_slot)
      stage(compact_total, (error_slot, 0), (term_slot, 0), Ops.ADD, error_slot)
    stage(compact_total, (error_slot, 0), scalar_256, Ops.MUL, error_slot)
    if row_sum(error_slot, rows, chunk_k, raw_fp32=True,
               out_slot=error_partials, out_offset=chunk*rows*4) is None: return None
    for a_slot, b_slot, partial_slot in ((limb_slots[0][0], limb_slots[1][1], cross_partials[0]),
                                        (limb_slots[0][1], limb_slots[1][0], cross_partials[1])):
      stage(compact_total, (a_slot, 0), (b_slot, 0), Ops.MUL, product_slot)
      if row_sum(product_slot, rows, chunk_k, raw_fp32=True,
                 out_slot=partial_slot, out_offset=chunk*rows*4) is None: return None

  # Every product keeps raw fp32 at both reduction levels. Chunk partials use
  # chunk-major storage, so the second CMAC gathers red*rows+row.
  def finish_raw(partials:int) -> int|None:
    partial_high = view(partials, _HOST_FP32_HALF_LAYOUT, rows*chunks)
    partial_low = view(partials, _HOST_FP32_RESIDUAL_LAYOUT, rows*chunks)
    high_sum = row_sum(partial_high, rows, chunks, raw_fp32=True, transposed=True)
    low_sum = row_sum(partial_low, rows, chunks, transposed=True)
    if high_sum is None or low_sum is None: return None
    high_half = view(high_sum, _HOST_FP32_HALF_LAYOUT, rows)
    high_residual = view(high_sum, _HOST_FP32_RESIDUAL_LAYOUT, rows)
    combined_residual = stage(rows, (high_residual, 0), (low_sum, 0), Ops.ADD)
    raw = alloc()
    combine_raw(high_half, combined_residual, raw, rows)
    return raw

  high_raw = finish_raw(high_partials)
  error_raw = finish_raw(error_partials)
  cross_raw = tuple(finish_raw(partials) for partials in cross_partials)
  if high_raw is None or error_raw is None or any(x is None for x in cross_raw): return None
  cross_values:list[int] = []
  for raw in cross_raw:
    assert raw is not None
    cross_high = view(raw, _HOST_FP32_HALF_LAYOUT, rows)
    cross_low = view(raw, _HOST_FP32_RESIDUAL_LAYOUT, rows)
    scaled_low = stage(rows, (cross_low, 0), (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1/256))[0]), Ops.MUL)
    cross_values.append(stage(rows, (cross_high, 0), (scaled_low, 0), Ops.ADD))
  cross_sum = stage(rows, (cross_values[0], 0), (cross_values[1], 0), Ops.ADD)
  error_high = view(error_raw, _HOST_FP32_HALF_LAYOUT, rows)
  error_low = view(error_raw, _HOST_FP32_RESIDUAL_LAYOUT, rows)
  scaled_error_low = stage(rows, (error_low, 0),
                           (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1/256))[0]), Ops.MUL)
  error_value = stage(rows, (error_high, 0), (scaled_error_low, 0), Ops.ADD)

  final_high = view(high_raw, _HOST_FP32_HALF_LAYOUT, rows)
  final_low = view(high_raw, _HOST_FP32_RESIDUAL_LAYOUT, rows)
  final_residual = stage(rows, (final_low, 0), (cross_sum, 0), Ops.ADD)
  final_residual = stage(rows, (final_residual, 0), (error_value, 0), Ops.ADD)
  combine_raw(final_high, final_residual, info.outs[0], rows)
  return tuple(tasks)

def _try_small_fp32_cmac_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Run a direct fp32 matrix product through typed half CMAC views and CBUF-derived tiles."""
  if (long_dot_tasks := _try_long_fp32_batched_dot_subtasks(sink)) is not None: return long_dot_tasks
  store, reduce = _store_node(sink), _reduce_node(sink)
  if store is None or reduce is None or store.src[0].dtype is not dtypes.float or \
     reduce.arg[0] is not Ops.ADD or reduce.dtype is not dtypes.float: return None
  epilogue = _try_cmac_epilogue(sink, reduce)
  if epilogue is None: return None
  body = _unwrap(reduce.src[0])
  outer_guard = None
  if body.op is Ops.WHERE and len(body.src) == 3 and _unwrap(body.src[1]).op is Ops.MUL and \
     body.src[2].op is Ops.CONST and (body.src[2].arg is Invalid or float(body.src[2].arg) == 0.0):
    outer_guard, body = body.src[0], _unwrap(body.src[1])
  if body.op is not Ops.MUL or body.dtype is not dtypes.float or len(body.src) != 2: return None
  sources = tuple(_unwrap(x) for x in body.src)
  if outer_guard is not None:
    sources = (UOp(Ops.WHERE, dtypes.float, (outer_guard, sources[0], UOp.const(dtypes.float, 0.0))), sources[1])
  if any(x.op not in (Ops.INDEX, Ops.WHERE) or x.dtype is not dtypes.float for x in sources): return None
  source_indexes = tuple(tuple(u for u in source.toposort()
                               if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM)
                         for source in sources)
  if any(not indexes or len({u.src[0].buf_uop.arg.slot for u in indexes}) != 1 for indexes in source_indexes): return None
  source_slots = tuple(indexes[0].src[0].buf_uop.arg.slot for indexes in source_indexes)
  source_totals = tuple(int(indexes[0].src[0].src[0].arg) for indexes in source_indexes)
  if any(slot >= 7 for slot in source_slots): return None
  output_total = prod(_shape_of_store(sink))
  # Flat DPU correction stages have a 13-bit atom-width field. Larger exact
  # rows are supported only when they can be represented without a partial
  # final atom; CMAC itself is independently tiled from CBUF capacity.
  if output_total > 4096*8 and output_total % 8: return None

  info, device = ProgramInfo.from_sink(sink), store.src[0].src[0].device
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  views:list[tuple[int,int]] = []
  for source_slot, source_total in zip(source_slots, source_totals):
    high_slot, low_slot = next_slot, next_slot+1
    next_slot += 2
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    high_relocs = (RKReloc(0, high_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    low_relocs = (RKReloc(0, low_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.extend((RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (source_total, _HOST_FP32_HALF_LAYOUT), high_slot, is_copy=True),
                            high_relocs),
                  RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (source_total, _HOST_FP32_RESIDUAL_LAYOUT), low_slot, is_copy=True),
                            low_relocs)))
    views.append((high_slot, low_slot))

  def view(source_slot:int, layout_tag:int) -> int:
    nonlocal next_slot
    out_slot, next_slot = next_slot, next_slot+1
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (output_total, layout_tag), out_slot, is_copy=True), relocs))
    return out_slot

  def cmac(a_slot:int, b_slot:int, fp32_output=False) -> int|None:
    nonlocal next_slot
    out_slot, next_slot = next_slot, next_slot+1
    def half_source(u:UOp, slot:int, source_slot:int, source_total:int) -> UOp|None:
      u = _unwrap(u)
      if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM and \
         u.src[0].buf_uop.arg.slot == source_slot:
        half_param = UOp.param(slot, dtypes.half, (source_total,), device=u.src[0].device)
        return u.replace(dtype=dtypes.half, src=(half_param, *u.src[1:]))
      if u.op is Ops.CONST and u.dtype is dtypes.float:
        return UOp.const(dtypes.half, float(u.arg))
      if u.op is Ops.WHERE and u.dtype is dtypes.float and len(u.src) == 3:
        true, false = half_source(u.src[1], slot, source_slot, source_total), \
                      half_source(u.src[2], slot, source_slot, source_total)
        if true is not None and false is not None: return UOp(Ops.WHERE, dtypes.half, (u.src[0], true, false))
      return None
    half_sources:list[UOp] = []
    for source, source_slot, source_total, slot in zip(sources, source_slots, source_totals, (a_slot, b_slot)):
      converted = half_source(source, slot, source_slot, source_total)
      if converted is None: return None
      half_sources.append(converted)
    half_product = UOp(Ops.MUL, dtypes.half, tuple(half_sources))
    half_reduce = reduce.replace(src=(UOp(Ops.CAST, dtypes.float, (half_product,), arg=dtypes.float), *reduce.src[1:]))
    out_param = UOp.param(out_slot, dtypes.half, (output_total,), device=device)
    out_index = store.src[0].replace(dtype=dtypes.half, src=(out_param, *store.src[0].src[1:]))
    stage_store = store.replace(src=(out_index, UOp(Ops.CAST, dtypes.half, (half_reduce,), arg=dtypes.half)))
    stage_plan = plan_rk(sink.substitute({store:stage_store}))
    if isinstance(stage_plan, str) or stage_plan.kind != "cmac": return None
    # Runtime K-tile accumulation is retained as WIP reference, but uses host
    # addition between NPU partials. Keep this forward path NPU-arithmetic-only.
    if stage_plan.cmac_materialization and stage_plan.cmac_materialization[2] > 4096: return None
    # Shared batch/group axes expand direct materialization into a block-diagonal
    # K. Serialize them like conv_grok's cartesian local tiles: besides avoiding
    # unused cross-batch cells, this keeps multi-row CMAC below the proven
    # CNA_DMA_CON1 K-group ceiling (8x65 previously became unsafe K=520).
    if (shared_tasks := _try_cmac_shared_subtasks(stage_plan)) is not None:
      tasks.extend(RKSubTask(st.cmds, replace(st.task, fp32_output=fp32_output), st.relocs) for st in shared_tasks)
    else:
      cmds, task, relocs = emit_rk(stage_plan)
      tasks.append(RKSubTask(cmds, replace(task, fp32_output=fp32_output), relocs))
    return out_slot

  high = cmac(views[0][0], views[1][0], fp32_output=True)
  cross0 = cmac(views[0][0], views[1][1])
  cross1 = cmac(views[0][1], views[1][0])
  if high is None or cross0 is None or cross1 is None: return None
  high_half, high_residual = view(high, _HOST_FP32_HALF_LAYOUT), view(high, _HOST_FP32_RESIDUAL_LAYOUT)
  cross_sum, residual = next_slot, next_slot+1
  next_slot += 2
  tasks.append(_emit_where_stage(output_total, cross_sum, (cross0, 0), (cross1, 0), Ops.ADD))
  tasks.append(_emit_where_stage(output_total, residual, (high_residual, 0), (cross_sum, 0), Ops.ADD))
  reduction_out = info.outs[0]
  if epilogue[0] != "none": reduction_out, next_slot = next_slot, next_slot+1
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, reduction_out, 0, 0, 0xFFFFFFFF), RKReloc(0, high_half, 0, 0, 0xFFFFFFFF),
            RKReloc(0, residual, 0, 0, 0xFFFFFFFF))
  tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", (output_total, _HOST_FP32_COMBINE_LAYOUT),
                                     reduction_out, is_copy=True, fp32_output=True), relocs))
  if epilogue[0] != "none":
    reduction_param = UOp.param(reduction_out, dtypes.float, (output_total,), device=device)
    reduction_index = store.src[0].replace(src=(reduction_param, *store.src[0].src[1:]))
    epilogue_store = store.replace(src=(store.src[0], store.src[1].substitute({reduce:reduction_index})))
    epilogue_sink = sink.substitute({store:epilogue_store})
    if (epilogue_tasks := _try_elementwise_host_subtasks(epilogue_sink, allow_plain=True)) is None: return None
    tasks.extend(epilogue_tasks)
  return tuple(tasks)

def _try_long_cumprod_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Inclusive 1-D float cumprod through a logarithmic Hillis-Steele scan."""
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.MUL or reduce.dtype is not dtypes.float or \
     store.src[0].dtype is not dtypes.float or store.src[1] is not reduce: return None
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not loops or len(reductions) != 1 or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  source_indexes = [u for u in reduce.src[0].toposort()
                    if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM]
  if len(source_indexes) != 1: return None
  source_total = int(source_indexes[0].src[0].src[0].arg)
  loop_extents = tuple(int(u.src[0].arg) for u in loops)
  physical_total, reduce_extent = prod(loop_extents), int(reductions[0].src[0].arg)
  if len(loops) == 1 and source_total == physical_total == reduce_extent and 256 < source_total <= 2048:
    total, segment_width, input_padding = source_total, source_total, 0
  elif source_total == 1022 and sorted(loop_extents) == [4, 256] and physical_total == 1024 and reduce_extent == 256:
    total, segment_width, input_padding = physical_total, 256, physical_total-source_total
  else:
    return None

  info = ProgramInfo.from_sink(sink)
  source_slot, next_slot = source_indexes[0].src[0].buf_uop.arg.slot, max(info.globals, default=-1)+1
  if source_slot >= 7: return None
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret
  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def stage(a:tuple[int,int], b:tuple[int,int], op:Ops, **kwargs) -> int:
    out = alloc()
    tasks.append(_emit_where_stage(total, out, a, b, op, **kwargs))
    return out

  converted_source = source_slot
  if input_padding:
    converted_source = alloc()
    one_bits = struct.unpack('<I', struct.pack('<f', 1.0))[0]
    out_code = (1, 0)
    # range<input_padding ? 1.0 : input[range-input_padding].
    value_code = (1, 0, 0, input_padding, 6, 0, 12, one_bits,
                  1, 0, 0, -input_padding, 2, 0, 10, 0, 11, 0)
    layout = (total, _HOST_MOVEMENT_LAYOUT, 4, 1, total,
              len(out_code), *out_code, len(value_code), *value_code)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, converted_source, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, converted_source, is_copy=True), relocs))
  if converted_source >= 7: return None
  hi, lo = alloc(), alloc()
  tasks.extend((_emit_where_stage(total, hi, (converted_source, 0), (_ZERO_SLOT, 0), Ops.ADD, fp32_inputs=(converted_source,)),
                _emit_where_stage(total, lo, (converted_source, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                  fp32_inputs=(converted_source,), fp32_residual_input=True)))

  def shifted(source:int, offset:int, identity:float) -> int|None:
    out = alloc()
    identity_bits = struct.unpack('<H', struct.pack('<e', identity))[0]
    out_code = (1, 0)
    # Postfix program: (range%segment_width)<offset ? identity : input[range-offset].
    boundary_code = (1, 0, 0, segment_width, 5, 0) if segment_width < total else (1, 0)
    value_code = (*boundary_code, 0, offset, 6, 0, 12, identity_bits,
                  1, 0, 0, -offset, 2, 0, 10, 0, 11, 0)
    layout = (total, _HOST_MOVEMENT_LAYOUT, 2, 1, total,
              len(out_code), *out_code, len(value_code), *value_code)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out, is_copy=True), relocs))
    return out

  splitter = scalar(65.0)
  limb_scale, inverse_limb_scale = scalar(256.0), scalar(1/256)
  offset = 1
  while offset < segment_width:
    operand, operand_low = shifted(hi, offset, 1.0), shifted(lo, offset, 0.0)
    if operand is None or operand_low is None: return None
    p = stage((hi, 0), (operand, 0), Ops.MUL)
    negative_hi = stage((_ZERO_SLOT, 0), (hi, 0), Ops.SUB)
    magnitude_hi = stage((hi, 0), (negative_hi, 0), Ops.MAX)
    large_diff = stage((magnitude_hi, 0), scalar(64.0), Ops.SUB)
    large = stage((large_diff, 0), (large_diff, 0), Ops.MAX, compare=True)
    weighted_large = stage((large, 0), scalar(255/256), Ops.MUL)
    normalization = stage(scalar(1.0), (weighted_large, 0), Ops.SUB)
    normalized_hi = stage((hi, 0), (normalization, 0), Ops.MUL)

    c_hi = stage(splitter, (normalized_hi, 0), Ops.MUL)
    big_hi = stage((c_hi, 0), (normalized_hi, 0), Ops.SUB)
    hi_head = stage((c_hi, 0), (big_hi, 0), Ops.SUB)
    hi_tail = stage((normalized_hi, 0), (hi_head, 0), Ops.SUB)
    c_operand = stage(splitter, (operand, 0), Ops.MUL)
    big_operand = stage((c_operand, 0), (operand, 0), Ops.SUB)
    operand_head = stage((c_operand, 0), (big_operand, 0), Ops.SUB)
    operand_tail = stage((operand, 0), (operand_head, 0), Ops.SUB)
    operand_head_scaled = stage((operand_head, 0), limb_scale, Ops.MUL)
    operand_tail_scaled = stage((operand_tail, 0), limb_scale, Ops.MUL)
    normalized_product = stage((normalized_hi, 0), (operand, 0), Ops.MUL)
    head_product = stage((hi_head, 0), (operand_head_scaled, 0), Ops.MUL)
    p_scaled = stage((normalized_product, 0), limb_scale, Ops.MUL)
    err = stage((head_product, 0), (p_scaled, 0), Ops.SUB)
    cross = stage((hi_head, 0), (operand_tail_scaled, 0), Ops.MUL)
    err = stage((err, 0), (cross, 0), Ops.ADD)
    cross = stage((hi_tail, 0), (operand_head_scaled, 0), Ops.MUL)
    err = stage((err, 0), (cross, 0), Ops.ADD)
    tail_product = stage((hi_tail, 0), (operand_tail_scaled, 0), Ops.MUL)
    err = stage((err, 0), (tail_product, 0), Ops.ADD)
    large_err = stage((err, 0), (large, 0), Ops.MUL)
    large_err = stage((large_err, 0), limb_scale, Ops.MUL)
    large_normalized_product = stage((normalized_product, 0), (large, 0), Ops.MUL)
    restored_product = stage((large_normalized_product, 0), limb_scale, Ops.MUL)
    large_product = stage((p, 0), (large, 0), Ops.MUL)
    restoration_error = stage((restored_product, 0), (large_product, 0), Ops.SUB)
    restoration_error = stage((restoration_error, 0), limb_scale, Ops.MUL)
    large_err = stage((large_err, 0), (restoration_error, 0), Ops.ADD)
    small = stage(scalar(1.0), (large, 0), Ops.SUB)
    small_err = stage((err, 0), (small, 0), Ops.MUL)
    err = stage((small_err, 0), (large_err, 0), Ops.ADD)
    low_product = stage((lo, 0), (operand, 0), Ops.MUL)
    low_sum = stage((err, 0), (low_product, 0), Ops.ADD)
    input_low_product = stage((hi, 0), (operand_low, 0), Ops.MUL)
    low_sum = stage((low_sum, 0), (input_low_product, 0), Ops.ADD)
    previous_low_product = stage((lo, 0), (operand_low, 0), Ops.MUL)
    previous_low_product = stage((previous_low_product, 0), inverse_limb_scale, Ops.MUL)
    low_sum = stage((low_sum, 0), (previous_low_product, 0), Ops.ADD)
    correction = stage((low_sum, 0), inverse_limb_scale, Ops.MUL)
    new_hi = stage((p, 0), (correction, 0), Ops.ADD)
    delta = stage((new_hi, 0), (p, 0), Ops.SUB)
    delta_scaled = stage((delta, 0), limb_scale, Ops.MUL)
    new_lo = stage((low_sum, 0), (delta_scaled, 0), Ops.SUB)
    hi, lo = new_hi, new_lo
    offset *= 2

  visible_low = stage((lo, 0), inverse_limb_scale, Ops.MUL)
  physical_total = int(store.src[0].src[0].src[0].arg)
  if physical_total == total:
    tasks.append(_emit_where_stage(total, info.outs[0], (hi, 0), (visible_low, 0), Ops.ADD, fp32_output=True))
  else:
    if physical_total < total: return None
    visible = alloc()
    tasks.append(_emit_where_stage(total, visible, (hi, 0), (visible_low, 0), Ops.ADD, fp32_output=True))
    padding = physical_total-total
    final_out_code = (1, 0, 0, padding, 2, 0)
    final_value_code = (1, 0, 10, 0)
    final_layout = (total, _HOST_MOVEMENT_LAYOUT, 4, 1, total,
                    len(final_out_code), *final_out_code, len(final_value_code), *final_value_code)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, visible, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", final_layout, info.outs[0], is_copy=True), relocs))
  return tuple(tasks)

def _try_long_cumprod_neutral_block_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Neutralize the block-prefix helper after the preceding kernel emitted the complete long scan."""
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.MUL or reduce.dtype is not dtypes.float or \
     store.src[0].dtype is not dtypes.float or reduce not in store.src[1].toposort(): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  reductions = list(reduce.src[1:])
  indexes = [u for u in reduce.src[0].toposort()
             if u.op is Ops.INDEX and u.dtype is dtypes.float and u.src[0].op is Ops.PARAM]
  total = prod(_shape_of_store(sink))
  if total != 4 or len(loops) != 1 or len(reductions) != 1 or int(reductions[0].src[0].arg) != 4 or len(indexes) != 1 or \
     int(indexes[0].src[0].src[0].arg) != 1024: return None
  one = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', 1.0))[0])
  out = ProgramInfo.from_sink(sink).outs[0]
  return (_emit_where_stage(total, out, (_ZERO_SLOT, 0), one, Ops.ADD, fp32_output=True),)

def _try_long_cumprod_final_copy_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Copy the complete long scan across the now-neutral blocked combine."""
  store = _store_node(sink)
  if store is None or _reduce_node(sink) is not None or store.src[0].dtype is not dtypes.float: return None
  value = store.src[1]
  if value.op is not Ops.MUL or value.dtype is not dtypes.float or len(value.src) != 2: return None
  indexes = [_unwrap(u) for u in value.src]
  if any(u.op is not Ops.INDEX or u.dtype is not dtypes.float or u.src[0].op is not Ops.PARAM for u in indexes): return None
  sized = sorted((int(u.src[0].src[0].arg), u.src[0].buf_uop.arg.slot) for u in indexes)
  logical_total = int(store.src[0].src[0].src[0].arg)
  if tuple(size for size, _ in sized) != (4, 1024) or logical_total != 1024: return None
  out, source = ProgramInfo.from_sink(sink).outs[0], sized[1][1]
  out_code, value_code = (1, 0), (1, 0, 10, 0)
  layout = (logical_total, _HOST_MOVEMENT_LAYOUT, 4, 1, logical_total,
            len(out_code), *out_code, len(value_code), *value_code)
  cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  relocs = (RKReloc(0, out, 0, 0, 0xFFFFFFFF), RKReloc(0, source, 0, 0, 0xFFFFFFFF))
  return (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, out, is_copy=True), relocs),)

def _try_local_product_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Gather a static float reduction window, then multiply it on DPU.

  Prefix products need more than a plain fp16 chain: Torch keeps its running
  product in fp32 and only rounds each visible prefix.  For longer half
  windows retain a two-half (hi, lo) product using Dekker's fp16 splitter.
  Every arithmetic operation below is a DPU task; host movement only gathers
  statically addressed input bytes.
  """
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.MUL or reduce.dtype not in (dtypes.half, dtypes.float) or \
     store.src[0].dtype is not reduce.dtype: return None
  wrapped_block_prefix = store.src[1] is not reduce
  if wrapped_block_prefix and reduce not in store.src[1].toposort(): return None
  value_dtype = reduce.dtype
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  reduce_extents = [int(u.src[0].arg) for u in reductions]
  total, window = prod(_shape_of_store(sink)), prod(reduce_extents)
  if prod(loop_extents) != total or not 2 <= window <= 256: return None
  if wrapped_block_prefix:
    source_sizes = {int(u.src[0].src[0].arg) for u in reduce.src[0].toposort()
                    if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM}
    if value_dtype is not dtypes.float or total != 4 or window != 4 or source_sizes != {1024}: return None

  info = ProgramInfo.from_sink(sink)
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  gathered_slots:list[int] = []
  gathered_low_slots:list[int] = []
  # Version-4 task images retain only low global slots in their fp32 input
  # mask. Reuse one such slot for gather→half conversion instead of assigning
  # an unencodable fp32 flag to every cumulative candidate.
  float_gather_slot = next_slot if value_dtype is dtypes.float else None
  if float_gather_slot is not None:
    if float_gather_slot >= 7: return None
    next_slot += 1
  for linear in range(window):
    rem, fixed = linear, {}
    for reduce_axis in range(len(reductions)-1, -1, -1):
      rem, coord = divmod(rem, reduce_extents[reduce_axis])
      fixed[reductions[reduce_axis]] = UOp.const(reductions[reduce_axis].dtype, coord)
    gathered = reduce.src[0].substitute(fixed)
    if gathered.dtype is not value_dtype: return None
    scratch_slot = float_gather_slot if float_gather_slot is not None else next_slot
    if float_gather_slot is None: next_slot += 1
    movement_tasks:tuple[RKSubTask, ...]|None
    if wrapped_block_prefix:
      source_slot = next(u.src[0].buf_uop.arg.slot for u in reduce.src[0].toposort()
                         if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM)
      threshold = 4-linear
      one_bits = struct.unpack('<I', struct.pack('<f', 1.0))[0]
      out_code = (1, 0)
      # output<threshold ? 1 : input[(output-threshold)*256+255].
      value_code = (1, 0, 0, threshold, 6, 0, 12, one_bits,
                    1, 0, 0, -threshold, 2, 0, 0, 256, 3, 0, 0, 255, 2, 0, 10, 0, 11, 0)
      layout = (total, _HOST_MOVEMENT_LAYOUT, 4, 1, total,
                len(out_code), *out_code, len(value_code), *value_code)
      cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
      relocs = (RKReloc(0, scratch_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      movement_tasks = (RKSubTask(cmds, RKTask(0, 0, 0, "dpu", layout, scratch_slot, is_copy=True), relocs),)
    else:
      scratch = UOp.param(scratch_slot, value_dtype, (total,), device=store.src[0].src[0].device)
      scratch_index = store.src[0].replace(dtype=value_dtype, src=(scratch, *store.src[0].src[1:]))
      movement_store = UOp(Ops.STORE, src=(scratch_index, gathered))
      movement_sink = UOp.sink(UOp(Ops.END, src=(movement_store, *loops)))
      movement_tasks = _try_movement_host_subtasks(movement_sink)
    if movement_tasks is None or len(movement_tasks) != 1: return None
    tasks.extend(movement_tasks)
    if float_gather_slot is not None:
      half_slot, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(total, half_slot, (scratch_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     fp32_inputs=(scratch_slot,)))
      gathered_slots.append(half_slot)
      residual_slot, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(total, residual_slot, (scratch_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     fp32_inputs=(scratch_slot,), fp32_residual_input=True))
      gathered_low_slots.append(residual_slot)
    else:
      gathered_slots.append(scratch_slot)
      gathered_low_slots.append(_ZERO_SLOT)

  # Keep the original short product chain for native-half 3/4/6/9-lane
  # reductions. Float32 ABI products need both input limbs even at six lanes.
  if value_dtype is dtypes.half and window < 10:
    accumulator = gathered_slots[0]
    for index, operand in enumerate(gathered_slots[1:], 1):
      final = index == len(gathered_slots)-1
      out_slot = info.outs[0] if final else next_slot
      if not final: next_slot += 1
      tasks.append(_emit_where_stage(total, out_slot, (accumulator, 0), (operand, 0), Ops.MUL))
      accumulator = out_slot
    return tuple(tasks)

  def scalar(value:float) -> tuple[int,int]:
    return _CONST_SLOT, struct.unpack('<I', struct.pack('<f', value))[0]
  def stage(a:tuple[int,int], b:tuple[int,int], op:Ops, **kwargs) -> int:
    nonlocal next_slot
    out_slot, next_slot = next_slot, next_slot+1
    tasks.append(_emit_where_stage(total, out_slot, a, b, op, **kwargs))
    return out_slot

  hi, lo = gathered_slots[0], gathered_low_slots[0]
  splitter = scalar(65.0)  # 2**ceil(11/2)+1 for binary16's 11-bit significand.
  limb_scale, inverse_limb_scale = scalar(256.0), scalar(1/256)
  for operand, operand_low in zip(gathered_slots[1:], gathered_low_slots[1:]):
    p = stage((hi, 0), (operand, 0), Ops.MUL)
    # Dekker's 65*x split overflows once |x| approaches 1008. Normalize lanes
    # above 64 by 1/256 while recovering the product error, then restore that
    # error below. The actual visible product `p` remains unscaled.
    negative_hi = stage((_ZERO_SLOT, 0), (hi, 0), Ops.SUB)
    magnitude_hi = stage((hi, 0), (negative_hi, 0), Ops.MAX)
    large_diff = stage((magnitude_hi, 0), scalar(64.0), Ops.SUB)
    large = stage((large_diff, 0), (large_diff, 0), Ops.MAX, compare=True)
    weighted_large = stage((large, 0), scalar(255/256), Ops.MUL)
    normalization = stage(scalar(1.0), (weighted_large, 0), Ops.SUB)
    normalized_hi = stage((hi, 0), (normalization, 0), Ops.MUL)

    # TwoProduct(hi, operand): p is the visible fp16 product and err recovers
    # the discarded low part using split high/low factors.
    c_hi = stage(splitter, (normalized_hi, 0), Ops.MUL)
    big_hi = stage((c_hi, 0), (normalized_hi, 0), Ops.SUB)
    hi_head = stage((c_hi, 0), (big_hi, 0), Ops.SUB)
    hi_tail = stage((normalized_hi, 0), (hi_head, 0), Ops.SUB)
    c_operand = stage(splitter, (operand, 0), Ops.MUL)
    big_operand = stage((c_operand, 0), (operand, 0), Ops.SUB)
    operand_head = stage((c_operand, 0), (big_operand, 0), Ops.SUB)
    operand_tail = stage((operand, 0), (operand_head, 0), Ops.SUB)
    # Keep the product error in the same x256 domain as the low limb. This
    # prevents the residual from underflowing when a long prefix becomes
    # smaller than the normal fp16 range.
    operand_head_scaled = stage((operand_head, 0), limb_scale, Ops.MUL)
    operand_tail_scaled = stage((operand_tail, 0), limb_scale, Ops.MUL)
    normalized_product = stage((normalized_hi, 0), (operand, 0), Ops.MUL)
    head_product = stage((hi_head, 0), (operand_head_scaled, 0), Ops.MUL)
    p_scaled = stage((normalized_product, 0), limb_scale, Ops.MUL)
    err = stage((head_product, 0), (p_scaled, 0), Ops.SUB)
    cross = stage((hi_head, 0), (operand_tail_scaled, 0), Ops.MUL)
    err = stage((err, 0), (cross, 0), Ops.ADD)
    cross = stage((hi_tail, 0), (operand_head_scaled, 0), Ops.MUL)
    err = stage((err, 0), (cross, 0), Ops.ADD)
    tail_product = stage((hi_tail, 0), (operand_tail_scaled, 0), Ops.MUL)
    err = stage((err, 0), (tail_product, 0), Ops.ADD)
    # For normalized lanes, err currently represents
    # 256*(normalized_hi*operand-normalized_product). Restore the original
    # scale and include any power-of-two round-trip discrepancy.
    large_err = stage((err, 0), (large, 0), Ops.MUL)
    large_err = stage((large_err, 0), limb_scale, Ops.MUL)
    large_normalized_product = stage((normalized_product, 0), (large, 0), Ops.MUL)
    restored_product = stage((large_normalized_product, 0), limb_scale, Ops.MUL)
    large_product = stage((p, 0), (large, 0), Ops.MUL)
    restoration_error = stage((restored_product, 0), (large_product, 0), Ops.SUB)
    restoration_error = stage((restoration_error, 0), limb_scale, Ops.MUL)
    large_err = stage((large_err, 0), (restoration_error, 0), Ops.ADD)
    small = stage(scalar(1.0), (large, 0), Ops.SUB)
    small_err = stage((err, 0), (small, 0), Ops.MUL)
    err = stage((small_err, 0), (large_err, 0), Ops.ADD)

    # Carry the x256 low limb through the new multiplication, renormalize
    # p+low_sum/256, and preserve the scaled remainder for the next prefix.
    low_product = stage((lo, 0), (operand, 0), Ops.MUL)
    low_sum = stage((err, 0), (low_product, 0), Ops.ADD)
    if value_dtype is dtypes.float:
      input_low_product = stage((hi, 0), (operand_low, 0), Ops.MUL)
      low_sum = stage((low_sum, 0), (input_low_product, 0), Ops.ADD)
      previous_low_product = stage((lo, 0), (operand_low, 0), Ops.MUL)
      previous_low_product = stage((previous_low_product, 0), inverse_limb_scale, Ops.MUL)
      low_sum = stage((low_sum, 0), (previous_low_product, 0), Ops.ADD)
    correction = stage((low_sum, 0), inverse_limb_scale, Ops.MUL)
    new_hi = stage((p, 0), (correction, 0), Ops.ADD)
    delta = stage((new_hi, 0), (p, 0), Ops.SUB)
    delta_scaled = stage((delta, 0), limb_scale, Ops.MUL)
    new_lo = stage((low_sum, 0), (delta_scaled, 0), Ops.SUB)
    hi, lo = new_hi, new_lo
  visible_low = stage((lo, 0), inverse_limb_scale, Ops.MUL)
  tasks.append(_emit_where_stage(total, info.outs[0], (hi, 0), (visible_low, 0), Ops.ADD, fp32_output=value_dtype is dtypes.float))
  if wrapped_block_prefix:
    identity_out_code = (0, 0)
    identity_value_code = (12, struct.unpack('<I', struct.pack('<f', 1.0))[0])
    identity_layout = (1, _HOST_MOVEMENT_LAYOUT, 4, 1, 1,
                       len(identity_out_code), *identity_out_code, len(identity_value_code), *identity_value_code)
    cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    identity_relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF),)
    tasks.append(RKSubTask(cmds, RKTask(0, 0, 0, "dpu", identity_layout, info.outs[0], is_copy=True), identity_relocs))
  return tuple(tasks)

def _wip_try_native_small_int_power_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower a repeated int32 MUL tree through packed native DPU atoms."""
  store = _store_node(sink)
  if store is None or store.src[0].dtype is not dtypes.int: return None
  value = _unwrap(store.src[1])
  indexes = [u for u in value.toposort() if u.op is Ops.INDEX and u.dtype is dtypes.int]
  if len(indexes) != 1 or (source := indexes[0]).src[0].op is not Ops.PARAM: return None
  def exponent(u:UOp) -> int|None:
    u = _unwrap(u)
    if u is source: return 1
    if u.op is not Ops.MUL or len(u.src) != 2: return None
    lhs, rhs = exponent(u.src[0]), exponent(u.src[1])
    return None if lhs is None or rhs is None else lhs+rhs
  power = exponent(value)
  if power is None or not 2 <= power <= 32: return None
  total, info = prod(_shape_of_store(sink)), ProgramInfo.from_sink(sink)
  if int(source.src[0].src[0].arg) != total: return None
  next_slot = max(info.globals, default=-1) + 1
  source_slot = source.src[0].buf_uop.arg.slot
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  tasks:list[RKSubTask] = []
  for start in range(0, total, 4):
    count = min(4, total-start)
    packed, next_slot = next_slot, next_slot+1
    pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
    pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
    accumulator = packed
    for _ in range(power-1):
      multiplied, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(4, multiplied, (accumulator, 0), (packed, 0), Ops.MUL,
                                     native_int32_input=True, native_int32_output=True))
      accumulator = multiplied
    unpack_layout = (count, _HOST_UNPACK_INT_CHUNK_LAYOUT, start)
    unpack_relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, accumulator, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, info.outs[0], is_copy=True), unpack_relocs))
  return tuple(tasks)

def _try_native_int_min_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower signed int32 MIN's XOR-order graph with native integer DPU stages."""
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.MAX or reduce.dtype is not dtypes.int or \
     store.src[0].dtype is not dtypes.int: return None

  def xor_minus_one(u:UOp, expected:UOp|None=None) -> UOp|None:
    if u.op is not Ops.XOR or len(u.src) != 2: return None
    for value, mask in ((u.src[0], u.src[1]), (u.src[1], u.src[0])):
      if mask.op is Ops.CONST and int(mask.arg) == -1 and (expected is None or value is expected): return value
    return None

  if xor_minus_one(store.src[1], reduce) is None: return None
  source = xor_minus_one(reduce.src[0])
  if source is None or source.op is not Ops.INDEX or source.dtype is not dtypes.int or source.src[0].op is not Ops.PARAM: return None
  reductions = list(reduce.src[1:])
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if not reductions or any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  reduce_extents = [int(u.src[0].arg) for u in reductions]
  total, window = prod(_shape_of_store(sink)), prod(reduce_extents)
  if prod(loop_extents) != total or window != 2: return None

  info = ProgramInfo.from_sink(sink)
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
  gathered_slots:list[int] = []
  for linear in range(window):
    rem, fixed = linear, {}
    for reduce_axis in range(len(reductions)-1, -1, -1):
      rem, coord = divmod(rem, reduce_extents[reduce_axis])
      fixed[reductions[reduce_axis]] = UOp.const(reductions[reduce_axis].dtype, coord)
    gathered = source.substitute(fixed)
    scratch_slot, next_slot = next_slot, next_slot+1
    scratch = UOp.param(scratch_slot, dtypes.int, (total,), device=store.src[0].src[0].device)
    scratch_index = store.src[0].replace(dtype=dtypes.int, src=(scratch, *store.src[0].src[1:]))
    movement_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(scratch_index, gathered)), *loops)))
    movement_tasks = _try_movement_host_subtasks(movement_sink)
    if movement_tasks is None or len(movement_tasks) != 1: return None
    tasks.extend(movement_tasks)
    gathered_slots.append(scratch_slot)

  for start in range(0, total, 4):
    count = min(4, total-start)
    packed_slots:list[int] = []
    for source_slot in gathered_slots:
      packed_slot, next_slot = next_slot, next_slot+1
      layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      relocs = (RKReloc(0, packed_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", layout, packed_slot, is_copy=True), relocs))
      packed_slots.append(packed_slot)
    accumulator = packed_slots[0]
    for operand in packed_slots[1:]:
      summed, maximum, selected = next_slot, next_slot+1, next_slot+2
      next_slot += 3
      tasks.extend((_emit_where_stage(4, summed, (accumulator, 0), (operand, 0), Ops.ADD,
                                      native_int32_input=True, native_int32_output=True),
                    _emit_where_stage(4, maximum, (accumulator, 0), (operand, 0), Ops.MAX,
                                      native_int32_input=True, native_int32_output=True),
                    # Variable-variable SUB evaluates EW-RDMA, hence sum-max.
                    _emit_where_stage(4, selected, (maximum, 0), (summed, 0), Ops.SUB,
                                      native_int32_input=True, native_int32_output=True)))
      accumulator = selected
    layout = (count, _HOST_UNPACK_INT_CHUNK_LAYOUT, start)
    relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, accumulator, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", layout, info.outs[0], is_copy=True), relocs))
  return tuple(tasks)

def _try_bool_reduce_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Lower bool ALL/ANY to native DPU masks followed by a CMAC count.

  ANY counts nonzero lanes and tests count>0. ALL counts zero lanes and tests
  count==0. The latter deliberately avoids comparing against the reduction
  extent: a million-lane count may overflow fp16, while zero/nonzero remains
  exact across the CMAC fp32 accumulator and its fp16 output boundary.
  """
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.dtype is not dtypes.bool or store.src[0].dtype is not dtypes.bool or \
     reduce.arg[0] not in (Ops.MAX, Ops.MUL): return None
  outer_all = False
  store_value = store.src[1]
  if store_value is not reduce:
    if store_value.op is not Ops.CMPNE or len(store_value.src) != 2 or reduce.arg[0] is not Ops.MAX: return None
    outer_all = any(value is reduce and truth.op is Ops.CONST and bool(truth.arg)
                    for value, truth in ((store_value.src[0], store_value.src[1]), (store_value.src[1], store_value.src[0])))
    if not outer_all: return None
  semantic_op = Ops.MUL if outer_all else reduce.arg[0]
  if not reduce.src[1:] or any(r.src[0].op is not Ops.CONST for r in reduce.src[1:]): return None

  body = reduce.src[0]
  source:UOp|None = None
  if outer_all and body.op is Ops.CMPNE and len(body.src) == 2:
    for candidate, truth in ((body.src[0], body.src[1]), (body.src[1], body.src[0])):
      if truth.op is Ops.CONST and bool(truth.arg) and candidate.op is Ops.INDEX and candidate.dtype is dtypes.bool:
        source = candidate
        break
  elif body.op is Ops.INDEX and body.dtype is dtypes.bool:
    source = body
  elif body.op is Ops.CMPNE and len(body.src) == 2:
    for candidate, zero in ((body.src[0], body.src[1]), (body.src[1], body.src[0])):
      candidate = _unwrap(candidate)
      if zero.op is Ops.CONST and float(zero.arg) == 0.0 and candidate.op is Ops.INDEX:
        source = candidate
        break
  if source is None or source.src[0].op is not Ops.PARAM or source.dtype not in (dtypes.bool, dtypes.half, dtypes.float): return None
  if source.src[0].src[0].op is not Ops.CONST or store.src[0].src[0].src[0].op is not Ops.CONST: return None

  info = ProgramInfo.from_sink(sink)
  source_slot = source.src[0].buf_uop.arg.slot
  input_total = int(source.src[0].src[0].arg)
  output_total = int(store.src[0].src[0].src[0].arg)
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  def alloc() -> int:
    nonlocal next_slot
    ret, next_slot = next_slot, next_slot+1
    return ret

  if outer_all:
    widened = alloc()
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    widen_layout = (input_total, _HOST_BOOL_HALF_LAYOUT)
    widen_relocs = (RKReloc(0, widened, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", widen_layout, widened, is_copy=True), widen_relocs))
    loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
    reductions = list(reduce.src[1:])
    if any(u.src[0].op is not Ops.CONST for u in (*loops, *reductions)): return None
    reduce_extents = [int(u.src[0].arg) for u in reductions]
    window = prod(reduce_extents)
    gathered_slots:list[int] = []
    widened_param = source.src[0].param_like(widened).replace(dtype=dtypes.half)
    widened_index = source.replace(dtype=dtypes.half, src=(widened_param, *source.src[1:]))
    for linear in range(window):
      rem, fixed = linear, {}
      for reduce_axis in range(len(reductions)-1, -1, -1):
        rem, coord = divmod(rem, reduce_extents[reduce_axis])
        fixed[reductions[reduce_axis]] = UOp.const(reductions[reduce_axis].dtype, coord)
      gathered = widened_index.substitute(fixed)
      scratch_slot = alloc()
      scratch = UOp.param(scratch_slot, dtypes.half, (output_total,), device=store.src[0].src[0].device)
      scratch_index = store.src[0].replace(dtype=dtypes.half, src=(scratch, *store.src[0].src[1:]))
      movement_sink = UOp.sink(UOp(Ops.END, src=(UOp(Ops.STORE, src=(scratch_index, gathered)), *loops)))
      movement_tasks = _try_movement_host_subtasks(movement_sink)
      if movement_tasks is None or len(movement_tasks) != 1: return None
      tasks.extend(movement_tasks)
      gathered_slots.append(scratch_slot)
    accumulator = gathered_slots[0]
    if len(gathered_slots) == 1:
      tasks.append(_emit_where_stage(output_total, info.outs[0], (accumulator, 0), (_ZERO_SLOT, 0), Ops.ADD, bool_output=True))
      return tuple(tasks)
    for index, operand in enumerate(gathered_slots[1:], 1):
      final = index == len(gathered_slots)-1
      out_slot = info.outs[0] if final else alloc()
      tasks.append(_emit_where_stage(output_total, out_slot, (accumulator, 0), (operand, 0), Ops.MUL, bool_output=final))
      accumulator = out_slot
    return tuple(tasks)

  # Buffers at the RK3588 GEM mmap boundary cannot be submitted directly.
  # Stage large predicates through reusable 32K-lane DMA tiles, then copy the
  # exact fp16 mask bytes into one host-mapped tensor for tiled CMAC gathering.
  large_tiled = (source.dtype is dtypes.half and input_total*2 >= 2*1024*1024) or \
                (source.dtype is dtypes.float and input_total*4 >= 2*1024*1024)
  if source.dtype is dtypes.float and large_tiled:
    counted = alloc()
    high_tile, low_tile = alloc(), alloc()
    negative_tiles, magnitude_tiles, nonzero_tiles = (alloc(), alloc()), (alloc(), alloc()), (alloc(), alloc())
    combined_tile, zero_tile = alloc(), alloc()
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    tile = 262144
    for start in range(0, input_total, tile):
      count = min(tile, input_total-start)
      for limb, tag in ((high_tile, _HOST_FP32_HALF_LAYOUT), (low_tile, _HOST_FP32_RESIDUAL_LAYOUT)):
        layout = (count, tag, 1, count, 1, start)
        tile_view_relocs = (RKReloc(0, limb, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
        tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", layout, limb, is_copy=True), tile_view_relocs))
      for index, limb in enumerate((high_tile, low_tile)):
        tasks.extend((_emit_where_stage(count, negative_tiles[index], (limb, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL),
                      _emit_where_stage(count, magnitude_tiles[index], (limb, 0), (negative_tiles[index], 0), Ops.MAX),
                      _emit_where_stage(count, nonzero_tiles[index], (magnitude_tiles[index], 0),
                                        (magnitude_tiles[index], 0), Ops.MAX, compare=True)))
      tasks.append(_emit_where_stage(count, combined_tile, (nonzero_tiles[0], 0), (nonzero_tiles[1], 0), Ops.MAX))
      tile_result = combined_tile
      if semantic_op is Ops.MUL:
        tile_result = zero_tile
        tasks.append(_emit_where_stage(count, tile_result, (_CONST_SLOT, 0x3f800000), (combined_tile, 0), Ops.SUB))
      result_layout = (count, _HOST_MOVEMENT_LAYOUT, 2, 1, count,
                       2, 1, 0,
                       4, 1, 0, 10, 0)
      result_relocs = (RKReloc(0, counted, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, tile_result, 0, 0, 0xFFFFFFFF))
      result_task = RKTask(0, 0, 0, "dpu", result_layout, counted, is_copy=True, out_offset=start*2)
      tasks.append(RKSubTask(host_cmds, result_task, result_relocs))
  elif source.dtype is dtypes.half and large_tiled:
    counted = alloc()
    input_tile, negative_tile, magnitude_tile, nonzero_tile = (alloc() for _ in range(4))
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    tile = 32768
    for start in range(0, input_total, tile):
      count = min(tile, input_total-start)
      source_layout = (count, _HOST_MOVEMENT_LAYOUT, 2, 1, count,
                       2, 1, 0,
                       8, 1, 0, 0, start, 2, 0, 10, 0)
      source_relocs = (RKReloc(0, input_tile, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", source_layout, input_tile, is_copy=True), source_relocs))
      tasks.extend((_emit_where_stage(count, negative_tile, (input_tile, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL),
                    _emit_where_stage(count, magnitude_tile, (input_tile, 0), (negative_tile, 0), Ops.MAX),
                    _emit_where_stage(count, nonzero_tile, (magnitude_tile, 0), (magnitude_tile, 0), Ops.MAX, compare=True)))
      tile_result = nonzero_tile
      if semantic_op is Ops.MUL:
        tile_result = negative_tile
        tasks.append(_emit_where_stage(count, tile_result, (_CONST_SLOT, 0x3f800000), (nonzero_tile, 0), Ops.SUB))
      result_layout = (count, _HOST_MOVEMENT_LAYOUT, 2, 1, count,
                       2, 1, 0,
                       4, 1, 0, 10, 0)
      result_relocs = (RKReloc(0, counted, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, tile_result, 0, 0, 0xFFFFFFFF))
      result_task = RKTask(0, 0, 0, "dpu", result_layout, counted, is_copy=True, out_offset=start*2)
      tasks.append(RKSubTask(host_cmds, result_task, result_relocs))
  # Bool buffers are widened only at the existing typed buffer boundary; all
  # predicate arithmetic and both reductions execute on the NPU.
  elif source.dtype is dtypes.bool:
    nonzero = alloc()
    bool_layout = (input_total, _HOST_BOOL_HALF_LAYOUT)
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    host_relocs = (RKReloc(0, nonzero, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", bool_layout, nonzero, is_copy=True), host_relocs))
  elif source.dtype is dtypes.float:
    # Test both split-fp32 limbs so values whose nearest fp16 high limb is zero
    # retain fp32 nonzero semantics. Host work only changes ABI representation.
    high, low = alloc(), alloc()
    host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    for out_slot, layout_tag in ((high, _HOST_FP32_HALF_LAYOUT), (low, _HOST_FP32_RESIDUAL_LAYOUT)):
      view_relocs = (RKReloc(0, out_slot, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", (input_total, layout_tag),
                                              out_slot, is_copy=True), view_relocs))
    limb_masks:list[int] = []
    for limb in (high, low):
      negative, magnitude, mask = alloc(), alloc(), alloc()
      tasks.extend((_emit_where_stage(input_total, negative, (limb, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL),
                    _emit_where_stage(input_total, magnitude, (limb, 0), (negative, 0), Ops.MAX),
                    _emit_where_stage(input_total, mask, (magnitude, 0), (magnitude, 0), Ops.MAX, compare=True)))
      limb_masks.append(mask)
    nonzero = alloc()
    tasks.append(_emit_where_stage(input_total, nonzero, (limb_masks[0], 0), (limb_masks[1], 0), Ops.MAX))
  else:
    negative, magnitude, nonzero = (alloc() for _ in range(3))
    # WIP reference: separate positive/negative comparisons also produce an
    # exact nonzero mask, but retain five full-size scratch buffers. ABS then
    # one comparison is equivalent and lets the 2**20 case fit GEM mmap limits.
    # negative_diff, negative_mask, positive_diff, positive_mask, nonzero = (alloc() for _ in range(5))
    tasks.extend((_emit_where_stage(input_total, negative, (source_slot, 0), (_CONST_SLOT, 0xbf800000), Ops.MUL),
                  _emit_where_stage(input_total, magnitude, (source_slot, 0), (negative, 0), Ops.MAX),
                  _emit_where_stage(input_total, nonzero, (magnitude, 0), (magnitude, 0), Ops.MAX, compare=True)))

  if not large_tiled:
    counted = nonzero
    if semantic_op is Ops.MUL:
      counted = negative if source.dtype is dtypes.half else alloc()
      tasks.append(_emit_where_stage(input_total, counted, (_CONST_SLOT, 0x3f800000), (nonzero, 0), Ops.SUB))

  # Preserve the scheduled INDEX expressions and RANGE axes, replacing only
  # the boolean source/output storage with fp16 scratch buffers.
  counted_param = source.src[0].param_like(counted).replace(dtype=dtypes.half)
  counted_index = source.replace(dtype=dtypes.half, src=(counted_param, *source.src[1:]))
  float_value = UOp(Ops.CAST, dtypes.float, (counted_index,), arg=dtypes.float)
  sum_reduce = UOp(Ops.REDUCE, dtypes.float, (float_value, *reduce.src[1:]), arg=(Ops.ADD, 0))
  sum_value = UOp(Ops.CAST, dtypes.half, (sum_reduce,), arg=dtypes.half)
  sum_slot = alloc()
  sum_param = store.src[0].src[0].param_like(sum_slot).replace(dtype=dtypes.half)
  sum_index = store.src[0].replace(dtype=dtypes.half, src=(sum_param, *store.src[0].src[1:]))
  sum_store = store.replace(src=(sum_index, sum_value))
  sum_sink = sink.substitute({store:sum_store})
  sum_plan = plan_rk(sum_sink)
  if isinstance(sum_plan, str) or sum_plan.kind != "cmac":
    if getenv("ROCKCHIP_DEBUG_BOOL_REDUCE"): print("RK_BOOL_REDUCE_CMAC_REJECT", sum_plan, sum_sink)
    return None
  if (shared_tasks := _try_cmac_shared_subtasks(sum_plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(sum_plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(sum_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

  positive = info.outs[0] if semantic_op is Ops.MAX else alloc()
  tasks.append(_emit_where_stage(output_total, positive, (sum_slot, 0), (sum_slot, 0), Ops.MAX, compare=True,
                                 bool_output=semantic_op is Ops.MAX))
  if semantic_op is Ops.MUL:
    tasks.append(_emit_where_stage(output_total, info.outs[0], (_CONST_SLOT, 0x3f800000), (positive, 0), Ops.SUB,
                                   bool_output=True))
  return tuple(tasks)

def _try_local_max_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Gather static local-reduction windows, then compute their maximum on DPU."""
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.MAX: return None
  post_scale:float|None = None
  store_value = _unwrap(store.src[1])
  if store_value is not reduce:
    if store_value.op is not Ops.MUL or len(store_value.src) != 2: return None
    for value, factor_node in ((store_value.src[0], store_value.src[1]), (store_value.src[1], store_value.src[0])):
      if _unwrap(value) is reduce and factor_node.op is Ops.CONST:
        post_scale = float(factor_node.arg)
        break
    if post_scale is None: return None
  value_dtype = reduce.dtype
  min_source:UOp|None = None
  reduce_body = _unwrap(reduce.src[0])
  if post_scale is not None and post_scale < 0 and reduce_body.op is Ops.MUL and len(reduce_body.src) == 2:
    for source_value, scale in ((reduce_body.src[0], reduce_body.src[1]), (reduce_body.src[1], reduce_body.src[0])):
      if scale.op is Ops.CONST and float(scale.arg) == -1.0:
        min_source = source_value
        break
  negated_only = False
  if (post_scale is None or post_scale < 0) and min_source is None:
    # Cumulative MIN wraps the negated input and its invalid-prefix padding in
    # WHERE nodes. Preserve the direct pattern above, then recover the
    # unnegated candidate graph: +inf padding becomes -inf after the existing
    # per-candidate -1 scale, exactly matching MAX's neutral sentinel.
    negated_values:dict[UOp,UOp] = {}
    for node in reduce_body.toposort():
      if node.op is not Ops.MUL or node.dtype not in (dtypes.half, dtypes.float) or len(node.src) != 2: continue
      for source_value, scale in ((node.src[0], node.src[1]), (node.src[1], node.src[0])):
        if scale.op is Ops.CONST and float(scale.arg) == -1.0 and any(u.op is Ops.INDEX for u in source_value.toposort()):
          negated_values[node] = source_value
          break
    if negated_values:
      min_source = reduce_body.substitute(negated_values)
      negative_infinities = {u:UOp.const(u.dtype, math.inf) for u in min_source.toposort()
                             if u.op is Ops.CONST and u.dtype in (dtypes.half, dtypes.float) and float(u.arg) == -math.inf}
      min_source = min_source.substitute(negative_infinities)
      if post_scale is None:
        # Cummin's index producer materializes MAX(-x) without the outer
        # negation. Scale the recovered candidates before MAX, but do not
        # restore their sign at this stage.
        post_scale, negated_only = -1.0, True
  is_min = min_source is not None
  if value_dtype not in (dtypes.half, dtypes.float, dtypes.int) or store.src[0].dtype is not value_dtype or \
     (post_scale is not None and value_dtype not in (dtypes.half, dtypes.float)) or (is_min and value_dtype is dtypes.int): return None
  reductions = list(reduce.src[1:])
  out_shape = _shape_of_store(sink)
  if not reductions or any(u.src[0].op is not Ops.CONST for u in reductions): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  if any(u.src[0].op is not Ops.CONST for u in loops): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  reduce_extents = [int(u.src[0].arg) for u in reductions]
  total, window = prod(out_shape), prod(reduce_extents)
  if prod(loop_extents) != total or window < 2: return None
  info = ProgramInfo.from_sink(sink)
  next_slot = max(info.globals, default=-1) + 1
  tasks:list[RKSubTask] = []
  host_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)

  def casted_half_source(u:UOp) -> UOp|None:
    """Undo a fused half->int cast below MAX.

    Truncation is monotone, so max(trunc(x)) == trunc(max(x)); preserving the
    original half values through the reduction allows one native roundoff-LUT
    correction after MAX. Integer padding sentinels are translated back to
    their half-domain equivalents.
    """
    if u.op is Ops.CAST and u.dtype is dtypes.int and len(u.src) == 1 and u.src[0].dtype is dtypes.half: return u.src[0]
    if u.op is Ops.CONST and u.dtype is dtypes.int:
      return UOp.const(dtypes.half, -math.inf if int(u.arg) == -(1 << 31) else float(u.arg))
    if u.op is Ops.WHERE and u.dtype is dtypes.int:
      true, false = casted_half_source(u.src[1]), casted_half_source(u.src[2])
      if true is not None and false is not None: return UOp(Ops.WHERE, dtypes.half, (u.src[0], true, false))
    return None

  def native_int_to_half(source_slot:int) -> int:
    nonlocal next_slot
    result, next_slot = next_slot, next_slot+1
    for start in range(0, total, 4):
      count = min(4, total-start)
      packed, next_slot = next_slot, next_slot+1
      pack_layout = (count, _HOST_PACK_INT32_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, source_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      half_atom, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(4, half_atom, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_input=True))
      unpack_layout = (count, _HOST_UNPACK_HALF_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, result, 0, 0, 0xFFFFFFFF), RKReloc(0, half_atom, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, result, is_copy=True), unpack_relocs))
    return result

  def trunc_half(source_slot:int) -> int|None:
    """Truncate an fp16 scratch buffer using the native roundoff LUT and DPU masks."""
    nonlocal next_slot
    def alloc() -> int:
      nonlocal next_slot
      result, next_slot = next_slot, next_slot+1
      return result
    def temp_index(slot:int) -> UOp:
      out_idx = store.src[0]
      return out_idx.replace(dtype=dtypes.half,
        src=(out_idx.src[0].param_like(slot).replace(dtype=dtypes.half), *out_idx.src[1:]))

    zero = (_ZERO_SLOT, 0)
    negative, magnitude, rounded = alloc(), alloc(), alloc()
    tasks.extend((_emit_where_stage(total, negative, zero, (source_slot, 0), Ops.SUB),
                  _emit_where_stage(total, magnitude, (source_slot, 0), (negative, 0), Ops.MAX)))
    # Build this internal pass with a single flat loop.  Reusing the parent
    # pool index can expose a multi-axis ADD tree that the LUT layout checker
    # intentionally rejects even though both scratch buffers are contiguous.
    axis, device = UOp.range(total, 0), store.src[0].src[0].device
    source_param = UOp.param(magnitude, dtypes.half, (total,), device=device)
    output_param = UOp.param(rounded, dtypes.half, (total,), device=device)
    roundoff = UOp(Ops.CUSTOM, dtypes.half, (UOp(Ops.INDEX, dtypes.half, (source_param, axis)),), arg="rk_roundoff")
    round_store = UOp(Ops.STORE, src=(UOp(Ops.INDEX, dtypes.half, (output_param, axis)), roundoff))
    round_sink = UOp.sink(UOp(Ops.END, src=(round_store, axis)))
    round_plan = plan_rk(round_sink)
    if isinstance(round_plan, str) or round_plan.kind != "dpu_lut":
      if getenv("ROCKCHIP_DEBUG_LOCAL_MAX"): print("RK_LOCAL_MAX_TRUNC_LUT_REJECT", round_plan)
      return None
    cmds, task, relocs = emit_rk(round_plan)
    tasks.append(RKSubTask(cmds, task, relocs))

    overshoot_diff, overshoot, truncated_abs = alloc(), alloc(), alloc()
    positive_diff, positive, negative_diff, negative_mask, sign, result = (alloc() for _ in range(6))
    tasks.extend((_emit_where_stage(total, overshoot_diff, (rounded, 0), (magnitude, 0), Ops.SUB),
                  _emit_where_stage(total, overshoot, (overshoot_diff, 0), (overshoot_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, truncated_abs, (rounded, 0), (overshoot, 0), Ops.SUB),
                  _emit_where_stage(total, positive_diff, (source_slot, 0), zero, Ops.SUB),
                  _emit_where_stage(total, positive, (positive_diff, 0), (positive_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, negative_diff, zero, (source_slot, 0), Ops.SUB),
                  _emit_where_stage(total, negative_mask, (negative_diff, 0), (negative_diff, 0), Ops.MAX, compare=True),
                  _emit_where_stage(total, sign, (positive, 0), (negative_mask, 0), Ops.SUB),
                  _emit_where_stage(total, result, (truncated_abs, 0), (sign, 0), Ops.MUL)))
    return result

  # WIP reference: flattening RANGE nodes in toposort order is not equivalent
  # to the STORE's physical output order.  In particular, NCHW (1,3,3,3)
  # local pools were transposed across channel/spatial axes.
  # flat = UOp.const(dtypes.weakint, 0)
  # if loops:
  #   flat = loops[0]
  #   for axis, extent in zip(loops[1:], loop_extents[1:]): flat = flat*extent + axis

  # Version-4 typed-input metadata only covers low global slots. Reuse one
  # fp32 movement target and convert it immediately before the next gather.
  fp32_gather_slot = next_slot if value_dtype is dtypes.float else None
  if fp32_gather_slot is not None: next_slot += 1
  gathered_slots:list[int] = []
  needs_trunc = False
  gather_source = reduce.src[0]
  if is_min:
    assert min_source is not None
    gather_source = min_source
  for linear in range(window):
    rem, fixed = linear, {}
    for reduce_axis in range(len(reductions)-1, -1, -1):
      rem, coord = divmod(rem, reduce_extents[reduce_axis])
      fixed[reductions[reduce_axis]] = UOp.const(reductions[reduce_axis].dtype, coord)
    gathered = gather_source.substitute(fixed)
    gathered_half = casted_half_source(gathered) if value_dtype is dtypes.int else None
    needs_trunc = needs_trunc or gathered_half is not None
    scratch_slot = fp32_gather_slot if fp32_gather_slot is not None else next_slot
    if fp32_gather_slot is None: next_slot += 1
    scratch_dtype = dtypes.half if gathered_half is not None or value_dtype is dtypes.half else value_dtype
    scratch = UOp.param(scratch_slot, scratch_dtype, (total,), device=store.src[0].src[0].device)
    movement_value = gathered_half if gathered_half is not None else gathered
    scratch_index = store.src[0].replace(dtype=scratch_dtype, src=(scratch, *store.src[0].src[1:]))
    movement_store = UOp(Ops.STORE, src=(scratch_index, movement_value))
    movement_sink = UOp.sink(UOp(Ops.END, src=(movement_store, *loops)))
    movement_tasks = _try_movement_host_subtasks(movement_sink)
    if movement_tasks is None or len(movement_tasks) != 1:
      if getenv("ROCKCHIP_DEBUG_LOCAL_MAX"): print("RK_LOCAL_MAX_GATHER_REJECT", gathered)
      return None
    tasks.extend(movement_tasks)
    if value_dtype is dtypes.float:
      half_slot, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(total, half_slot, (scratch_slot, 0), (_ZERO_SLOT, 0), Ops.ADD,
                                     fp32_inputs=(scratch_slot,)))
      gathered_slots.append(half_slot)
    else:
      gathered_slots.append(native_int_to_half(scratch_slot) if scratch_dtype is dtypes.int else scratch_slot)

  # WIP reference: allocating every fp32 movement target first and converting
  # them later overflowed version-4 typed-input slot metadata after candidate
  # two. The active loop above interleaves movement and conversion.

  # Positive scaling commutes with MAX. Applying it to each compact candidate
  # avoids the unstable scalar-DPU transition after a long global MAX chain.
  # The older post-reduction scale path remains below for reference.
  if post_scale is not None:
    if post_scale < 0 and not is_min: return None
    scale_arg = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', post_scale))[0])
    scaled_slots:list[int] = []
    for source_slot in gathered_slots:
      scale_warm, scaled_slot = next_slot, next_slot+1
      next_slot += 2
      # Keep the original one-stage form for reference. Static gather tasks
      # need one reset-separated warm read before this arithmetic consumer.
      # tasks.append(_emit_where_stage(total, scaled_slot, (source_slot, 0), scale_arg, Ops.MUL))
      tasks.append(_emit_where_stage(total, scale_warm, (source_slot, 0), scale_arg, Ops.MUL))
      tasks.append(_emit_where_stage(total, scaled_slot, (source_slot, 0), scale_arg, Ops.MUL))
      scaled_slots.append(scaled_slot)
    gathered_slots = scaled_slots
    post_scale = None

  accumulator = gathered_slots[0]
  for i, operand in enumerate(gathered_slots[1:]):
    final = i == len(gathered_slots)-2
    out_slot = info.outs[0] if final and value_dtype in (dtypes.half, dtypes.float) and post_scale is None and \
      (not is_min or negated_only) else next_slot
    if out_slot == next_slot: next_slot += 1
    if value_dtype is dtypes.float:
      # MAX(-x) for fp32 MIN remains an fp16 scratch value until the final
      # negate performs the one logical fp32 output conversion.
      tasks.append(_emit_where_stage(total, out_slot, (accumulator, 0), (operand, 0), Ops.MAX,
                                     fp32_output=final and out_slot == info.outs[0]))
    else:
      tasks.append(_emit_where_stage(total, out_slot, (accumulator, 0), (operand, 0), Ops.MAX))
    accumulator = out_slot
  if value_dtype is dtypes.int:
    if needs_trunc:
      truncated = trunc_half(accumulator)
      if truncated is None: return None
      accumulator = truncated
    for start in range(0, total, 4):
      count = min(4, total-start)
      packed, next_slot = next_slot, next_slot+1
      pack_layout = (count, _HOST_PACK_CHUNK_LAYOUT, start)
      pack_relocs = (RKReloc(0, packed, 0, 0, 0xFFFFFFFF), RKReloc(0, accumulator, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", pack_layout, packed, is_copy=True), pack_relocs))
      native_slot, next_slot = next_slot, next_slot+1
      tasks.append(_emit_where_stage(4, native_slot, (packed, 0), (_ZERO_SLOT, 0), Ops.ADD, native_int32_output=True))
      unpack_layout = (count, _HOST_UNPACK_INT_CHUNK_LAYOUT, start)
      unpack_relocs = (RKReloc(0, info.outs[0], 0, 0, 0xFFFFFFFF), RKReloc(0, native_slot, 0, 0, 0xFFFFFFFF))
      tasks.append(RKSubTask(host_cmds, RKTask(0, 0, 0, "dpu", unpack_layout, info.outs[0], is_copy=True), unpack_relocs))
  # WIP reference: a post-MAX scalar task was previously emitted here. The
  # long global chain made that transition unstable, so nonnegative scaling
  # is now commuted into candidates above.
  # elif post_scale is not None:
  #   scale_arg = (_CONST_SLOT, struct.unpack('<I', struct.pack('<f', post_scale))[0])
  #   tasks.append(_emit_where_stage(total, info.outs[0], (accumulator, 0), scale_arg, Ops.MUL))
  elif is_min and not negated_only:
    negative_one = (_CONST_SLOT, 0xbf800000)
    tasks.append(_emit_where_stage(total, info.outs[0], (accumulator, 0), negative_one, Ops.MUL,
                                   fp32_output=value_dtype is dtypes.float))
  return tuple(tasks)

def _try_cmac_variable_scale_subtasks(sink:UOp) -> tuple[RKSubTask, ...]|None:
  """Apply a static output-dependent reciprocal to raw fp32 CMAC sums."""
  reduce, store = _reduce_node(sink), _store_node(sink)
  if reduce is None or store is None or reduce.arg[0] is not Ops.ADD: return None
  val = _unwrap(store.src[1])
  data_value, count = None, None
  if val.op is Ops.MUL and len(val.src) == 2:
    for data, scale in ((val.src[0], val.src[1]), (val.src[1], val.src[0])):
      if _unwrap(data) is reduce and _unwrap(scale).op is Ops.RECIPROCAL:
        data_value, count = data, _unwrap(scale).src[0]
        break
  elif val.op is Ops.FDIV and len(val.src) == 2 and _unwrap(val.src[0]) is reduce:
    # Device-specific lowering may preserve half FDIV instead of spelling it
    # MUL(RECIPROCAL(count)); both identify the same static pooling divisor.
    data_value, count = val.src
  if data_value is None or count is None: return None
  while count.op is Ops.CAST: count = count.src[0]
  count_reduces = [u for u in count.toposort() if u.op is Ops.REDUCE]
  if not count_reduces or any(u.arg[0] is not Ops.ADD for u in count_reduces): return None
  loops = [u for u in sink.toposort() if u.op is Ops.RANGE and getattr(u.arg[-1], "name", "") == "LOOP"]
  count_ranges = [r for u in count_reduces for r in u.src[1:]]
  if not loops or any(u.src[0].op is not Ops.CONST for u in (*loops, *count_ranges)): return None
  loop_extents = [int(u.src[0].arg) for u in loops]
  total = prod(loop_extents)

  def evaluate(u:UOp, coords:dict[UOp, int]):
    while u.op is Ops.CAST: u = u.src[0]
    if u.op is Ops.CONST: return 0 if u.arg is Invalid else u.arg
    if u.op is Ops.RANGE and u in coords: return coords[u]
    if u.op is Ops.REDUCE:
      ranges, result = u.src[1:], 0
      extents = [int(x.src[0].arg) for x in ranges]
      for linear in range(prod(extents)):
        rem = linear
        for axis in range(len(ranges)-1, -1, -1): rem, coords[ranges[axis]] = divmod(rem, extents[axis])
        result += evaluate(u.src[0], coords)
      return result
    values = [evaluate(x, coords) for x in u.src]
    if u.op is Ops.ADD: return values[0] + values[1]
    if u.op is Ops.MUL: return values[0] * values[1]
    if u.op is Ops.FLOORDIV: return values[0] // values[1]
    if u.op is Ops.FLOORMOD: return values[0] % values[1]
    if u.op is Ops.CMPLT: return values[0] < values[1]
    if u.op is Ops.CMPNE: return values[0] != values[1]
    if u.op is Ops.AND: return bool(values[0]) and bool(values[1])
    if u.op is Ops.OR: return bool(values[0]) or bool(values[1])
    if u.op is Ops.WHERE: return values[1] if values[0] else values[2]
    raise ValueError(u.op)

  scale_counts = [0] * total
  try:
    for linear in range(total):
      rem, coords = linear, {}
      for axis in range(len(loops)-1, -1, -1): rem, coords[loops[axis]] = divmod(rem, loop_extents[axis])
      raw_count = evaluate(count, coords)
      count_value = int(raw_count)
      if count_value <= 0 or count_value != raw_count: return None
      out_index = int(evaluate(store.src[0].src[1], coords))
      if not 0 <= out_index < total: return None
      scale_counts[out_index] = count_value
  except (ValueError, TypeError, OverflowError, ZeroDivisionError):
    return None
  if any(count_value == 0 for count_value in scale_counts): return None

  # The retired scratch path allocated two slots after ProgramInfo.globals:
  # one for the rounded sum and one for a host-materialized reciprocal tensor.
  cmac_sink = sink.substitute({store:store.replace(src=(store.src[0], data_value))})
  plan = plan_rk(cmac_sink)
  if isinstance(plan, str) or plan.kind != "cmac": return None
  if not plan.cmac_materialization: return None
  plan = replace(plan, epilogue_scale_counts=tuple(scale_counts))
  tasks:list[RKSubTask] = []
  if (shared_tasks := _try_cmac_shared_subtasks(plan)) is not None: tasks.extend(shared_tasks)
  elif (rounding_tasks := _try_cmac_rounding_subtasks(plan)) is not None: tasks.extend(rounding_tasks)
  else:
    cmds, task, relocs = emit_rk(plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  # The earlier fp16-scratch + static reciprocal + DPU MUL design is preserved
  # in _HOST_STATIC_HALF_LAYOUT support. It double-rounded the CMAC sum, while
  # PyTorch scales the fp32 accumulator once before its final fp16 conversion.
  return tuple(tasks)

def _try_cmac_rounding_subtasks(plan:RKPlan) -> tuple[RKSubTask, ...]|None:
  """Split flipped-kernel CMAC reductions at PyTorch fp16 col2im boundaries."""
  materialization = plan.cmac_materialization
  if not materialization or plan.epilogue not in ("none", "relu", "bias", "bias_relu"): return None
  cursor = 5
  cursor += 1 + materialization[cursor]  # loop extents
  n_reductions = materialization[cursor]
  reduce_extents = materialization[cursor+1:cursor+1+n_reductions]
  cursor += 1 + n_reductions
  cursor += 1 + materialization[cursor]  # M axes
  cursor += 1 + materialization[cursor]  # N axes
  n_shared = materialization[cursor]
  shared_axes = materialization[cursor+1:cursor+1+n_shared]
  cursor += 1 + n_shared
  n_reduce_order = materialization[cursor]
  reduce_order = materialization[cursor+1:cursor+1+n_reduce_order]
  cursor += 1 + n_reduce_order
  out_n = materialization[cursor]
  out_code = materialization[cursor+1:cursor+1+out_n]
  cursor += 1+out_n
  for _ in range(2): cursor += 1 + materialization[cursor]  # A/B postfix programs
  tail_start = cursor
  n_fixed = materialization[cursor]
  cursor += 1 + 2*n_fixed
  n_active = materialization[cursor]
  cursor += 1 + n_active
  n_rounding = materialization[cursor]
  rounding_axes = materialization[cursor+1:cursor+1+n_rounding]
  if not rounding_axes or prod(reduce_extents[axis] for axis in rounding_axes) <= 1: return None
  active_axes = tuple(axis for axis in reduce_order if axis not in rounding_axes)
  loop_extents = materialization[6:6+materialization[5]]
  batches = prod(loop_extents[axis] for axis in shared_axes)
  group_k = prod(reduce_extents[axis] for axis in active_axes) * batches
  M, N = materialization[:2]
  # Shared group/batch axes participate in both the block-diagonal M and N
  # matrices, so M*N contains unused cross-group cells. DPU accumulation is
  # over the original packed output only.
  total = prod(loop_extents)
  next_slot = max((plan.out_slot, *plan.in_slots)) + 1
  tasks:list[RKSubTask] = []
  group_slots:list[int] = []
  group_count = prod(reduce_extents[axis] for axis in rounding_axes)
  for linear in range(group_count):
    rem, fixed_values = linear, [0] * len(rounding_axes)
    for idx in range(len(rounding_axes)-1, -1, -1):
      rem, fixed_values[idx] = divmod(rem, reduce_extents[rounding_axes[idx]])
      fixed_values[idx] = reduce_extents[rounding_axes[idx]] - 1 - fixed_values[idx]
    fixed = tuple(v for pair in zip(rounding_axes, fixed_values) for v in pair)
    group_slot = next_slot
    next_slot += 1
    group_slots.append(group_slot)
    group_tail = (len(rounding_axes), *fixed, len(active_axes), *active_axes, 0, 0)
    group_materialization = (M, N, group_k, *materialization[3:tail_start], *group_tail)
    group_plan = replace(plan, out_slot=group_slot, epilogue="none", epilogue_bias_slot=-1, epilogue_bias_axis=-1,
                         cmac_materialization=group_materialization)
    cmds, task, relocs = emit_rk(group_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  accumulator = group_slots[0]
  needs_post = plan.epilogue != "none"
  for idx, group_slot in enumerate(group_slots[1:], 1):
    out_slot = plan.out_slot if idx == len(group_slots)-1 and not needs_post else next_slot
    if out_slot == next_slot: next_slot += 1
    tasks.append(_emit_where_stage(total, out_slot, (accumulator, 0), (group_slot, 0), Ops.ADD))
    accumulator = out_slot
  if plan.epilogue in ("bias", "bias_relu"):
    expanded_bias = next_slot
    next_slot += 1
    value_code = (1, plan.epilogue_bias_axis, 10, 0)
    movement_layout = (total, _HOST_MOVEMENT_LAYOUT, 2, len(loop_extents), *loop_extents,
                       len(out_code), *out_code, len(value_code), *value_code)
    movement_cmds = (RKCmd(_T_PC, rk.REG_PC_OPERATION_ENABLE, 0).pack(),)
    movement_task = RKTask(0, 0, 0, "dpu", movement_layout, expanded_bias, is_copy=True)
    movement_relocs = (RKReloc(0, expanded_bias, 0, 0, 0xFFFFFFFF),
                       RKReloc(0, plan.epilogue_bias_slot, 0, 0, 0xFFFFFFFF))
    tasks.append(RKSubTask(movement_cmds, movement_task, movement_relocs))
    bias_out = plan.out_slot if plan.epilogue == "bias" else next_slot
    if bias_out == next_slot: next_slot += 1
    tasks.append(_emit_where_stage(total, bias_out, (accumulator, 0), (expanded_bias, 0), Ops.ADD))
    accumulator = bias_out
  if plan.epilogue in ("relu", "bias_relu"):
    tasks.append(_emit_where_stage(total, plan.out_slot, (accumulator, 0), (_ZERO_SLOT, 0), Ops.MAX))
  return tuple(tasks)

def _try_cmac_shared_subtasks(plan:RKPlan) -> tuple[RKSubTask, ...]|None:
  """Submit shared batch/group axes serially instead of block-diagonal K expansion."""
  materialization = plan.cmac_materialization
  if not materialization: return None
  cursor = 5
  n_loops = materialization[cursor]
  loop_extents = materialization[cursor+1:cursor+1+n_loops]
  cursor += 1+n_loops
  n_reductions = materialization[cursor]
  reduce_extents = materialization[cursor+1:cursor+1+n_reductions]
  cursor += 1+n_reductions
  n_m_axes = materialization[cursor]
  m_axes = materialization[cursor+1:cursor+1+n_m_axes]
  cursor += 1+n_m_axes
  n_n_axes = materialization[cursor]
  n_axes = materialization[cursor+1:cursor+1+n_n_axes]
  cursor += 1+n_n_axes
  n_shared = materialization[cursor]
  shared_axes = materialization[cursor+1:cursor+1+n_shared]
  cursor += 1+n_shared
  n_reduce_order = materialization[cursor]
  reduce_order = materialization[cursor+1:cursor+1+n_reduce_order]
  cursor += 1+n_reduce_order
  codes:list[tuple[int, ...]] = []
  for _ in range(3):
    code_n = materialization[cursor]
    codes.append(materialization[cursor+1:cursor+1+code_n])
    cursor += 1+code_n
  n_fixed = materialization[cursor]
  fixed_reductions = materialization[cursor+1:cursor+1+2*n_fixed]
  cursor += 1+2*n_fixed
  n_active = materialization[cursor]
  active_reduce_order = materialization[cursor+1:cursor+1+n_active]
  cursor += 1+n_active
  n_rounding = materialization[cursor]
  rounding_axes = materialization[cursor+1:cursor+1+n_rounding]
  cursor += 1+n_rounding
  n_fixed_loops = materialization[cursor]
  prior_fixed_loops = materialization[cursor+1:cursor+1+2*n_fixed_loops]
  if not shared_axes or rounding_axes: return None
  # Keep the compact block-diagonal form when it already fits. Serial shared
  # tasks are needed when output-channel tiling would otherwise mix batches, or
  # when the expanded KxN weight image cannot fit the remaining CBUF banks.
  base_m, base_n, base_k = materialization[:3]
  cbuf_bank_size, min_channel_tile, cbuf_banks = 256*128, 32, 12
  aligned_k = max(min_channel_tile, round_up(base_k, min_channel_tile))
  align_out = max(min_channel_tile, round_up(min(base_n, min_channel_tile), min_channel_tile))
  align_in = max(aligned_k, align_out)
  input_row_bytes = align_in*2
  tile_m = min(base_m, max(1, min(2048, 10*cbuf_bank_size//input_row_bytes)))
  data_banks = min(cbuf_banks-1, max(1, ceildiv(tile_m*input_row_bytes, cbuf_bank_size)))
  if base_n <= min_channel_tile and input_row_bytes*align_out <= (cbuf_banks-data_banks)*cbuf_bank_size: return None
  active_m_axes = tuple(axis for axis in m_axes if axis not in shared_axes)
  active_n_axes = tuple(axis for axis in n_axes if axis not in shared_axes)
  M = prod(loop_extents[axis] for axis in active_m_axes)
  N = prod(loop_extents[axis] for axis in active_n_axes)
  K = prod(reduce_extents[axis] for axis in active_reduce_order)
  tasks:list[RKSubTask] = []
  group_count = prod(loop_extents[axis] for axis in shared_axes)
  for linear in range(group_count):
    rem, fixed_values = linear, [0] * len(shared_axes)
    for idx in range(len(shared_axes)-1, -1, -1):
      rem, fixed_values[idx] = divmod(rem, loop_extents[shared_axes[idx]])
    fixed_loops = (*prior_fixed_loops, *(v for pair in zip(shared_axes, fixed_values) for v in pair))
    rebuilt = (M, N, K, materialization[3], materialization[4],
               len(loop_extents), *loop_extents, len(reduce_extents), *reduce_extents,
               len(active_m_axes), *active_m_axes, len(active_n_axes), *active_n_axes, 0,
               len(reduce_order), *reduce_order,
               *(v for code in codes for v in (len(code), *code)),
               n_fixed, *fixed_reductions, n_active, *active_reduce_order, n_rounding, *rounding_axes,
               len(fixed_loops)//2, *fixed_loops)
    group_plan = replace(plan, cmac_materialization=rebuilt)
    cmds, task, relocs = emit_rk(group_plan)
    tasks.append(RKSubTask(cmds, task, relocs))
  return tuple(tasks)

# ---- the native_program hook ----
def build_native_program(sink: UOp) -> UOp|None:
  """Classify and build a PROGRAM(SINK, LINEAR(INS...)). Raises RKPLAN_REJECT:<reason>
  if unsupported (no fallback per §15). Raises if a classified kernel fails emission."""
  # The generic reciprocal-to-FDIV rewrite obscures the strict x**-5.5 graph,
  # so recognize this special case while its exponent structure is intact.
  if (pow_neg55_tasks := _try_pow_neg55_lut_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_neg55_tasks)
  # Pre-classification rewrite: MUL(a, RECIPROCAL(b)) → FDIV(a, b)
  sink = graph_rewrite(sink, _pm_fdiv, name="rk fdiv decomp")
  if getenv("ROCKCHIP_DEBUG_SINK"): print("RK_SINK", sink)
  if (bce_logits_tasks := _try_bce_logits_subtasks(sink)) is not None: return build_native_program_multi(sink, bce_logits_tasks)
  if (bce_tasks := _try_bce_subtasks(sink)) is not None: return build_native_program_multi(sink, bce_tasks)
  if (movement_tasks := _try_movement_host_subtasks(sink)) is not None: return build_native_program_multi(sink, movement_tasks)
  if (trunc_tasks := _try_trunc_host_subtasks(sink)) is not None: return build_native_program_multi(sink, trunc_tasks)
  if (copysign_tasks := _try_copysign_host_subtasks(sink)) is not None: return build_native_program_multi(sink, copysign_tasks)
  if (bitwise_tasks := _try_bitwise_host_subtasks(sink)) is not None: return build_native_program_multi(sink, bitwise_tasks)
  if (cast_tasks := _try_cast_subtasks(sink)) is not None: return build_native_program_multi(sink, cast_tasks)
  if (fill_tasks := _try_typed_fill_subtasks(sink)) is not None: return build_native_program_multi(sink, fill_tasks)
  if (round_tasks := _try_round_subtasks(sink)) is not None: return build_native_program_multi(sink, round_tasks)
  if (sign_tasks := _try_sign_subtasks(sink)) is not None: return build_native_program_multi(sink, sign_tasks)
  if (inf_div_tasks := _try_inf_div_subtasks(sink)) is not None: return build_native_program_multi(sink, inf_div_tasks)
  if (hardsigmoid_tasks := _try_hardsigmoid_subtasks(sink)) is not None: return build_native_program_multi(sink, hardsigmoid_tasks)
  if (hardswish_tasks := _try_hardswish_subtasks(sink)) is not None: return build_native_program_multi(sink, hardswish_tasks)
  if (fp32_tanh_tasks := _try_fp32_tanh_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_tanh_tasks)
  if (tanh_tasks := _try_tanh_saturation_subtasks(sink)) is not None: return build_native_program_multi(sink, tanh_tasks)
  if (quick_gelu_tasks := _try_quick_gelu_two_lut_subtasks(sink)) is not None: return build_native_program_multi(sink, quick_gelu_tasks)
  if (logsigmoid_tasks := _try_logsigmoid_subtasks(sink)) is not None: return build_native_program_multi(sink, logsigmoid_tasks)
  if (softplus_tasks := _try_softplus_subtasks(sink)) is not None: return build_native_program_multi(sink, softplus_tasks)
  if (mish_tasks := _try_mish_subtasks(sink)) is not None: return build_native_program_multi(sink, mish_tasks)
  if (tan_tasks := _try_tan_subtasks(sink)) is not None: return build_native_program_multi(sink, tan_tasks)
  if (sin_cos_tasks := _try_sin_cos_subtasks(sink)) is not None: return build_native_program_multi(sink, sin_cos_tasks)
  if (asin_acos_tasks := _try_asin_acos_subtasks(sink)) is not None: return build_native_program_multi(sink, asin_acos_tasks)
  if (atan_tasks := _try_atan_subtasks(sink)) is not None: return build_native_program_multi(sink, atan_tasks)
  if (fp32_atanh_tasks := _try_fp32_atanh_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_atanh_tasks)
  if (atanh_tasks := _try_atanh_subtasks(sink)) is not None: return build_native_program_multi(sink, atanh_tasks)
  if (asinh_acosh_tasks := _try_asinh_acosh_subtasks(sink)) is not None: return build_native_program_multi(sink, asinh_acosh_tasks)
  if (fp32_sinh_cosh_tasks := _try_fp32_sinh_cosh_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_sinh_cosh_tasks)
  if (sinh_cosh_tasks := _try_sinh_cosh_subtasks(sink)) is not None: return build_native_program_multi(sink, sinh_cosh_tasks)
  if (erf_tasks := _try_erf_subtasks(sink)) is not None: return build_native_program_multi(sink, erf_tasks)
  if (gelu_tasks := _try_gelu_subtasks(sink)) is not None: return build_native_program_multi(sink, gelu_tasks)
  if (elu_tasks := _try_elu_subtasks(sink)) is not None: return build_native_program_multi(sink, elu_tasks)
  if (celu_tasks := _try_celu_subtasks(sink)) is not None: return build_native_program_multi(sink, celu_tasks)
  if (isclose_tasks := _try_isclose_host_subtasks(sink)) is not None: return build_native_program_multi(sink, isclose_tasks)
  if (softmax_tasks := _try_softmax_host_subtasks(sink)) is not None: return build_native_program_multi(sink, softmax_tasks)
  if (normalize_tasks := _try_normalize_norm_host_subtasks(sink)) is not None: return build_native_program_multi(sink, normalize_tasks)
  if (logcumsumexp_tasks := _try_logcumsumexp_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, logcumsumexp_tasks)
  if (fancy_index_tasks := _try_fancy_index_preprocess_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fancy_index_tasks)
  if (fancy_index_reduce_tasks := _try_fancy_index_reduction_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fancy_index_reduce_tasks)
  if (scatter_tasks := _try_scatter_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, scatter_tasks)
  if (scatter_reduce_tensor_tasks := _try_scatter_reduce_tensor_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, scatter_reduce_tensor_tasks)
  if (scatter_reduce_tasks := _try_scatter_reduction_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, scatter_reduce_tasks)
  if (comparison_tasks := _try_comparison_subtasks(sink)) is not None: return build_native_program_multi(sink, comparison_tasks)
  if (exp_correction_tasks := _try_exp_correction_subtasks(sink)) is not None:
    return build_native_program_multi(sink, exp_correction_tasks)
  if (sigmoid_tasks := _try_sigmoid_special_subtasks(sink)) is not None: return build_native_program_multi(sink, sigmoid_tasks)
  if (exp2_tasks := _try_exp2_special_subtasks(sink)) is not None: return build_native_program_multi(sink, exp2_tasks)
  if (log2_tasks := _try_log2_special_subtasks(sink)) is not None: return build_native_program_multi(sink, log2_tasks)
  if (rsqrt_tasks := _try_rsqrt_special_subtasks(sink)) is not None: return build_native_program_multi(sink, rsqrt_tasks)
  if (sqrt_tasks := _try_sqrt_special_subtasks(sink)) is not None: return build_native_program_multi(sink, sqrt_tasks)
  if (pow8_tasks := _try_pow8_lut_subtasks(sink)) is not None: return build_native_program_multi(sink, pow8_tasks)
  if (pow55_tasks := _try_pow55_lut_subtasks(sink)) is not None: return build_native_program_multi(sink, pow55_tasks)
  if (pow_neg55_tasks := _try_pow_neg55_lut_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_neg55_tasks)
  if (pow_base55_tasks := _try_pow_base55_lut_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_base55_tasks)
  if (pow_base8_tasks := _try_pow_base8_lut_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_base8_tasks)
  if (pow_base07_tasks := _try_pow_base07_lut_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_base07_tasks)
  if (pow_neg_base55_tasks := _try_pow_neg_base55_subtasks(sink)) is not None:
    return build_native_program_multi(sink, pow_neg_base55_tasks)
  if (zero_base_pow_tasks := _try_zero_base_pow_subtasks(sink)) is not None:
    return build_native_program_multi(sink, zero_base_pow_tasks)
  if (tensor_pow_tasks := _try_tensor_pow_subtasks(sink)) is not None:
    return build_native_program_multi(sink, tensor_pow_tasks)
  if (fractional_pow_tasks := _try_fractional_pow_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fractional_pow_tasks)
  if (abs_tasks := _try_abs_subtasks(sink)) is not None: return build_native_program_multi(sink, abs_tasks)
  if (softsign_tasks := _try_softsign_subtasks(sink)) is not None: return build_native_program_multi(sink, softsign_tasks)
  if (lerp_tasks := _try_lerp_subtasks(sink)) is not None: return build_native_program_multi(sink, lerp_tasks)
  if (one_hot_tasks := _try_one_hot_subtasks(sink)) is not None: return build_native_program_multi(sink, one_hot_tasks)
  if (cat_tasks := _try_cat_subtasks(sink)) is not None: return build_native_program_multi(sink, cat_tasks)
  if (pad_tasks := _try_pad_subtasks(sink)) is not None: return build_native_program_multi(sink, pad_tasks)
  if (conditional_movement_tasks := _try_conditional_movement_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, conditional_movement_tasks)
  # WIP: only valid when the producer was the fused one-dimensional long
  # scan; multidimensional cumprod uses the same physical helper shape.
  # if (long_cumprod_copy_tasks := _try_long_cumprod_final_copy_subtasks(sink)) is not None:
  #   return build_native_program_multi(sink, long_cumprod_copy_tasks)
  if (fp32_broadcast_tasks := _try_fp32_broadcast_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_broadcast_tasks)
  if (fp32_topology_tasks := _try_fp32_topology_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_topology_tasks)
  if (fp32_add_tasks := _try_fp32_add_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_add_tasks)
  if (fp32_mul_tasks := _try_fp32_mul_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_mul_tasks)
  if (variance_tasks := _try_variance_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, variance_tasks)
  if (fp32_avg_pool_tasks := _try_fp32_avg_pool_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_avg_pool_tasks)
  if (fp32_factorized_sum_tasks := _try_fp32_factorized_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_factorized_sum_tasks)
  if (broadcast_tasks := _try_broadcast_subtasks(sink)) is not None: return build_native_program_multi(sink, broadcast_tasks)
  # Rejected CPU operator fallbacks remain opt-in diagnostics only.
  if (scatter_tasks := _try_unpool_scatter_subtasks(sink)) is not None: return build_native_program_multi(sink, scatter_tasks)
  if (sort_compare_tasks := _try_sort_compare_subtasks(sink)) is not None:
    return build_native_program_multi(sink, sort_compare_tasks)
  if (argsort_selected_tasks := _try_argsort_selected_subtasks(sink)) is not None:
    return build_native_program_multi(sink, argsort_selected_tasks)
  if (argsort_tasks := _try_argsort_index_subtasks(sink)) is not None:
    return build_native_program_multi(sink, argsort_tasks)
  if (softmax_argmax_tasks := _try_softmax_argmax_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, softmax_argmax_tasks)
  if (arg_extrema_tasks := _try_arg_extrema_subtasks(sink)) is not None:
    return build_native_program_multi(sink, arg_extrema_tasks)
  if (pool_index_tasks := _try_pool_index_subtasks(sink)) is not None: return build_native_program_multi(sink, pool_index_tasks)
  if (int_max_pool_tasks := _try_int_max_pool_host_subtasks(sink)) is not None:
    return build_native_program_multi(sink, int_max_pool_tasks)
  if getenv("ROCKCHIP_ALLOW_HOST_OPS") and \
     (index_tasks := _try_static_index_reduction_subtasks(sink)) is not None: return build_native_program_multi(sink, index_tasks)
  if (long_cumprod_tasks := _try_long_cumprod_subtasks(sink)) is not None:
    return build_native_program_multi(sink, long_cumprod_tasks)
  # WIP companion to _try_long_cumprod_final_copy_subtasks; see note above.
  # if (neutral_block_tasks := _try_long_cumprod_neutral_block_subtasks(sink)) is not None:
  #   return build_native_program_multi(sink, neutral_block_tasks)
  if (product_tasks := _try_local_product_subtasks(sink)) is not None:
    return build_native_program_multi(sink, product_tasks)
  # WIP: small official values pass, but native MUL corrupts the high word of
  # 46340**2. Keep disabled until byte-limb multiplication is exact.
  # if (int_power_tasks := _wip_try_native_small_int_power_subtasks(sink)) is not None:
  #   return build_native_program_multi(sink, int_power_tasks)
  if (int_min_tasks := _try_native_int_min_subtasks(sink)) is not None:
    return build_native_program_multi(sink, int_min_tasks)
  if (bool_reduce_tasks := _try_bool_reduce_subtasks(sink)) is not None:
    return build_native_program_multi(sink, bool_reduce_tasks)
  if (local_max_tasks := _try_local_max_subtasks(sink)) is not None: return build_native_program_multi(sink, local_max_tasks)
  if (variable_scale_tasks := _try_cmac_variable_scale_subtasks(sink)) is not None:
    return build_native_program_multi(sink, variable_scale_tasks)
  if (movement_sum_tasks := _try_movement_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, movement_sum_tasks)
  if (elementwise_sum_tasks := _try_elementwise_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, elementwise_sum_tasks)
  if (fp32_cmac_tasks := _try_small_fp32_cmac_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_cmac_tasks)
  # Retain the half-input WIP above for reference. The active path is narrower:
  # it requires a direct fp32 INDEX and preserves the explicitly fp32 output.
  if (fp32_sum_tasks := _try_fp32_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, fp32_sum_tasks)
  if (relu_sum_tasks := _try_relu_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, relu_sum_tasks)
  if (nested_sum_tasks := _try_nested_sum_subtasks(sink)) is not None:
    return build_native_program_multi(sink, nested_sum_tasks)
  if (multifactor_tasks := _try_cmac_multifactor_subtasks(sink)) is not None:
    return build_native_program_multi(sink, multifactor_tasks)
  plan = plan_rk(sink)
  if isinstance(plan, str):
    # Nested elementwise lowering can materialize an indexed WHERE operand before
    # distribution turns MUL(WHERE(...), x) into a harder root WHERE graph.
    if (elementwise_tasks := _try_elementwise_subtasks(sink)) is not None: return build_native_program_multi(sink, elementwise_tasks)
    # Preserve directly recognizable single-stage forms such as abs(x) before
    # expanding MUL(WHERE(...)) into the general arithmetic-WHERE representation.
    sink = graph_rewrite(sink, _pm_where_mul, name="rk distribute where mul")
    plan = plan_rk(sink)
  if isinstance(plan, str):
    if (where_tasks := _try_where_subtasks(sink)) is not None: return build_native_program_multi(sink, where_tasks)
    if (host_tasks := _try_elementwise_host_subtasks(sink)) is not None: return build_native_program_multi(sink, host_tasks)
    if (elementwise_tasks := _try_elementwise_subtasks(sink)) is not None: return build_native_program_multi(sink, elementwise_tasks)
    raise RuntimeError(plan)  # reject — preserve reason, no fallback
  if (cmac_shared_tasks := _try_cmac_shared_subtasks(plan)) is not None:
    return build_native_program_multi(sink, cmac_shared_tasks)
  if (cmac_rounding_tasks := _try_cmac_rounding_subtasks(plan)) is not None:
    return build_native_program_multi(sink, cmac_rounding_tasks)
  cmds, task, relocs = emit_rk(plan)
  # each INS carries an int (packed command), RKReloc, or RKTask as its arg
  ins_args = [task] + list(cmds) + list(relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))

def build_native_program_multi(sink: UOp, subtasks: tuple[RKSubTask, ...]) -> UOp|None:
  """Build a multi-task PROGRAM for PC chain. The first INS carries a tuple of RKSubTask,
  followed by all cmds and relocs from all subtasks (flattened)."""
  if getenv("ROCKCHIP_DEBUG_SUBTASKS"):
    print("RK_SUBTASKS", [(st.task.out_slot, st.task.kind, st.task.fp32_inputs, st.task.fp32_output,
                           tuple(r.globals_slot for r in st.relocs)) for st in subtasks])
  ins_args: list = [subtasks]  # first INS carries the subtask list
  for st in subtasks:
    ins_args.extend(st.cmds)
    ins_args.extend(st.relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))
