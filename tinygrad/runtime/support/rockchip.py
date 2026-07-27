# pylint: skip-file
# RK3588 NPU compiled backend: pure match/classify + register emission + codec.
# PR 1 native contract: DPU binary EW/copy, CMAC matmul/sum, PPU global max (fp16 only).
# Fill, broadcast, mean, and all other ops are rejected — no host-side tensor arithmetic.
from __future__ import annotations
import struct, math, numpy as np
from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.helpers import ceildiv, round_up, prod
from tinygrad.uop.ops import Ops, UOp, ProgramInfo, GroupOp, PatternMatcher, graph_rewrite
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.upat import UPat

# Pre-classification rewrite: MUL(a, RECIPROCAL(b)) → FDIV(a, b) and MUL(RECIPROCAL(b), a) → FDIV(a, b)
# This lets the classifier see FDIV directly instead of nested MUL+RECIPROCAL
_pm_fdiv = PatternMatcher([
  (UPat.var("a") * UPat(Ops.RECIPROCAL, src=(UPat.var("b"),), name="r"), lambda a, b, r: UOp(Ops.FDIV, r.dtype, (a, b))),
  (UPat(Ops.RECIPROCAL, src=(UPat.var("b"),), name="r") * UPat.var("a"), lambda a, b, r: UOp(Ops.FDIV, r.dtype, (a, b))),
])

# target ids for emit_raw (rkt_get_target(reg) + 1, from rkt_registers.h)
# PC=0x80 is the value used by the working allbilly reference (autogen has 0x100, which is wrong for PC)
_T_DPU, _T_DPU_RDMA, _T_CNA, _T_CORE, _T_PPU, _T_PPU_RDMA, _T_PC = 0x1001, 0x2001, 0x201, 0x801, 0x4001, 0x8001, 0x81

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

# ---- classifier ----
def _is_fp16_only(sink: UOp) -> bool:
  """All tensor-carrying nodes must be fp16 (REDUCE may be fp32 for the NPU accumulator)."""
  return all(not ((u.op is Ops.PARAM and u.dtype is not dtypes.half) or (u.op is Ops.REDUCE and u.dtype not in (dtypes.half, dtypes.float)) or
                  (u.op is Ops.STORE and u.src[1].dtype not in (dtypes.half, dtypes.void))) for u in sink.toposort())

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
  if mr.arg[1].name != outer_kind or rs.arg[1].name != inner_kind or (stride >= 0 and int(mc.arg) != stride) or (stride < 0 and mr.arg[0] != 0): return None
  return (int(mr.src[0].arg) if mr.src and mr.src[0].op is Ops.CONST else 0, int(mc.arg), int(mc.arg))

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

# LUT ops: EXP2 uses DPU LUT table (513 entries × 2 tables) with BN_MUL scaling
_LUT_OPS = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT}
_LUT_SIZE = 513

def _build_exp2_lut() -> tuple[list[int], int, float, float]:
  """Build 1026-entry LUT for EXP2 over x∈[-2,2]. Returns (lut, bn_mul_operand, output_scale, index_scale).
  - index_scale: BN_MUL operand that maps x∈[-2,2] to int16 range [-16384,16384]
  - output_scale: converts exp2(x) to int16 range; OUT_CVT_SCALE=1 gives raw int16 as fp16."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 8192.0
  step = 32.0 / index_scale  # = 1/256
  output_scale = 8192.0  # exp2(x) * 8192 → int16 range
  # Table 0 (LE): covers negative x. Entry i: x = -(512-i)*step (from -2.0 to 0)
  for i in range(_LUT_SIZE):
    x = -(512 - i) * step
    y = math.exp2(x)
    lut[i] = max(-32768, min(32767, int(round(y * output_scale))))
  # Table 1 (LO): covers positive x. Entry i: x = i*step (from 0 to 2.0)
  for i in range(_LUT_SIZE):
    x = i * step
    y = math.exp2(x)
    lut[_LUT_SIZE + i] = max(-32768, min(32767, int(round(y * output_scale))))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale

def _build_log2_lut() -> tuple[list[int], int, float, float]:
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
  return lut, bn_mul_operand, output_scale, index_scale

def _build_sin_lut() -> tuple[list[int], int, float, float]:
  """Build 1026-entry LUT for SIN over x∈[-π,π] → result∈[-1,1].
  Uses LUT_LE_START=-16384, index_scale=16384/π≈5215.2.
  LE table: x∈[-π,0], LO table: x∈[0,π]."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 16384.0 / math.pi  # ≈5215.2, maps x∈[-π,π] to [-16384,16384]
  step = 32.0 / index_scale  # step in x per LUT entry
  output_scale = 8192.0  # maps [-1,1] to [-8192,8192]
  # LE table (table 0): covers bn_mul from -16384 to 0, i.e., x from -π to 0
  for i in range(_LUT_SIZE):
    x = -i * step  # i=0: x=0, i=512: x=-π
    y = math.sin(x)
    v = int(round(y * output_scale))
    if v == 0: v = 1  # avoid exact 0
    lut[i] = max(-32768, min(32767, v))
  # LO table (table 1): covers bn_mul from 0 to 16384, i.e., x from 0 to π
  for i in range(_LUT_SIZE):
    x = i * step  # i=0: x=0, i=512: x=π
    y = math.sin(x)
    v = int(round(y * output_scale))
    if v == 0: v = 1  # avoid exact 0
    lut[_LUT_SIZE + i] = max(-32768, min(32767, v))
  bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
  return lut, bn_mul_operand, output_scale, index_scale

def _build_sqrt_lut() -> tuple[list[int], int, float, float]:
  """Build 1026-entry LUT for SQRT over x∈[0,4] → result∈[0,2].
  Uses LUT_LE_START=-16384 (LE handles underflow), index_scale=4090.
  LE table: underflow (x<0 → clip to 0). LO table: x from 0 to ~4.01."""
  lut = [0] * _LUT_SIZE * 2
  index_scale = 4090.0  # avoid x=4.0 hitting LUT_LO_END=16384 exactly
  step = 32.0 / index_scale  # ≈ 1/127.8
  output_scale = 8192.0  # maps [0,2] to [0,16384]
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
  return lut, bn_mul_operand, output_scale, index_scale

def _try_lut(val: UOp) -> int|None:
  """EXP2(INDEX) → index_slot for DPU LUT op, or None."""
  if val.op not in _LUT_OPS: return None
  inner = _unwrap(val.src[0])
  if inner.op is not Ops.INDEX: return None
  return inner.src[0].buf_uop.arg.slot

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
    return (_is_2d_index(a_idx.src[1], "LOOP", "REDUCE", K) and _is_2d_index(b_idx.src[1], "REDUCE", "LOOP", N)) or \
           (_is_2d_index(a_idx.src[1], "REDUCE", "LOOP", N) and _is_2d_index(b_idx.src[1], "LOOP", "REDUCE", K))
  if len(out_shape) == 1:  # GEMV: (K,)@(K,D)→(D,) or (D,K)@(K,)→(D,)
    D = int(out_shape[0])
    if not _is_1d_index(store.src[0].src[1], "LOOP"): return False
    return (_is_1d_index(a_idx.src[1], "REDUCE") and _is_2d_index(b_idx.src[1], "REDUCE", "LOOP", D)) or \
           (_is_2d_index(a_idx.src[1], "LOOP", "REDUCE", K) and _is_1d_index(b_idx.src[1], "REDUCE"))
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
  # Reject post-reduce scalar MUL (mean) — no host-side scaling in PR1
  inner = _unwrap(store.src[1])
  if inner is not reduce: return None
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

def plan_rk(sink: UOp) -> RKPlan|str:
  """Classify a post-early_simplify SINK. Returns RKPlan on success, 'RKPLAN_REJECT:...' str on reject."""
  if not _is_fp16_only(sink): return "RKPLAN_REJECT:unsupported_dtype"
  reduce = _reduce_node(sink)
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
    lut_slot = _try_lut(val)
    # PR1 DPU contract: binary EW with two INDEX operands, scalar operand, DMA copy, or constant fill.
    # Broadcast and mean are rejected — no host-side tensor arithmetic.
    if lut_slot is not None: kind, a2d, ru = "dpu_lut", False, False
    elif sub_slots is not None: kind, a2d, ru = "dpu", False, False
    elif reciprocal is not None: kind, a2d, ru = "dpu", True, True
    elif scalar is not None or (val.op in _DPU_EW_CFGS and all(_unwrap(s).op is Ops.INDEX for s in val.src)): kind, a2d, ru = "dpu", True, True
    elif _unwrap(val).op is Ops.INDEX: kind, a2d, ru = "dpu", False, False
    elif _unwrap(val).op is Ops.CONST: kind, a2d, ru = "dpu", True, False
    else: return f"RKPLAN_REJECT:unsupported_op:{val.op if val.op not in _DPU_EW_CFGS else 'non_index_operand'}"
    if (r := _check_dpu_layout(sink, a2d, ru)): return r
  # R1 CMAC: REDUCE(ADD, MUL(INDEX, INDEX)) or REDUCE(ADD, INDEX) [sum via ones]
  elif reduce.arg[0] is Ops.ADD:
    body = _reduce_body(reduce)
    # Reject fused epilogue (bias ADD, ReLU, etc.) — PR2 handles BS/BN fusion
    store = _store_node(sink)
    if store is not None and _unwrap(store.src[1]).op is Ops.ADD and \
       any(s is reduce or (_unwrap(s) is reduce) for s in _unwrap(store.src[1]).src):
      return "RKPLAN_REJECT:unsupported_op:fused_epilogue"
    if body.op is Ops.MUL and all(s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX) for s in body.src):
      if not _is_cmac_matmul_layout(sink, reduce): return "RKPLAN_REJECT:unsupported_layout"
      kind = "cmac"
    elif body.op is Ops.INDEX:
      if _try_sum(sink, reduce) is not None: kind = "cmac"
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
    else: return f"RKPLAN_REJECT:unsupported_op:{body.op}"
  else: return f"RKPLAN_REJECT:unsupported_op:{reduce.arg[0]}"
  prog_info = ProgramInfo.from_sink(sink)
  out_slots = list(prog_info.outs)
  if len(out_slots) != 1: return f"RKPLAN_REJECT:unsupported_layout:{len(out_slots)}-outputs"
  in_slots = tuple(s for s in prog_info.globals if s != out_slots[0])
  return RKPlan(kind, sink, out_slots[0], in_slots)

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

def _emit_dpu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """DPU elementwise op (ADD/SUB/MUL/MAX), scalar operand, or DMA copy. Register sequence from elementwise.py."""
  cmds, relocs = [], []
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
  lut_slot = _try_lut(val)
  sub_slots, scalar = (None, None) if (is_copy or is_fill or lut_slot is not None) else (_try_sub(val), _try_scalar(val))
  reciprocal = None if (is_copy or is_fill or sub_slots is not None or scalar is not None or lut_slot is not None) else _try_reciprocal(val)
  layout = (total,)
  # track the EW op for FDIV-specific register emissions (OUT_CVT_SCALE, FP16TOFP32_EN=0)
  ew_op = Ops.SUB if sub_slots else (Ops.FDIV if reciprocal else (val.op if scalar else (Ops.ADD if is_fill else (None if is_copy else val.op))))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, dw)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, dw)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0)
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0x70007)
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
  else:
    emitter_reloc(cmds, relocs, _unwrap(val.src[0]).src[0].buf_uop.arg.slot)
    emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
    emitter_reloc(cmds, relocs, _unwrap(val.src[1]).src[0].buf_uop.arg.slot)
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, _DPU_EW_CFGS[val.op])
  # FDIV: emit OUT_CVT_SCALE=1 (output conversion scale for fp16 division result)
  if ew_op is Ops.FDIV:
    emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 1)
  # FDIV: MRDMA_FP16TOFP32_EN=0 (bit 3 clear) — division runs in fp16, no fp32 conversion
  rdma_fmc = 0x17849 if ew_op is not Ops.FDIV else 0x17841
  emitter_emit(cmds, _T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_fmc)
  emitter_pc_op_en(cmds, 12)
  return tuple(c.pack() for c in cmds), RKTask(0x18, 0x300, 4, "dpu", layout, plan.out_slot, is_copy=is_copy, is_fill=is_fill), tuple(relocs)

def _emit_dpu_lut(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """DPU LUT op (EXP2). Register sequence from ref/rk3588/experimental/kernel_6_18/silu.py.
  Uses LUT table lookup with BN_MUL index scaling and OUT_CVT FP32→FP16 conversion.
  The DPU LUT always processes a full 16x8 tile (128 elements) regardless of logical size."""
  cmds, relocs = [], []
  sink = plan.sink
  store = _store_node(sink)
  assert store is not None, "dpu_lut: no STORE node"
  val = _unwrap(store.src[1])
  total = 1
  for s in _shape_of_store(sink): total *= s
  lut_slot = _try_lut(val)
  assert lut_slot is not None, "dpu_lut: no LUT slot"
  layout = (total,)
  # DPU LUT: use dw for width (like regular DPU), channel=7 (8 channels)
  dw = (total + 7) // 8 - 1
  width = dw
  # DST_SURF_STRIDE = (width+1) * (channel+1) * 2 bytes
  surf_stride = (width + 1) * 8 * 2
  # --- LUT table fill (513 entries × 2 tables) ---
  if val.op is Ops.EXP2:
    lut, bn_mul_operand, output_scale, index_scale = _build_exp2_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # HYBRID_PRIORITY, OFLOW_PRIORITY, LO_LE_MUX=2
  elif val.op is Ops.LOG2:
    lut, bn_mul_operand, output_scale, index_scale = _build_log2_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif val.op is Ops.SIN:
    lut, bn_mul_operand, output_scale, index_scale = _build_sin_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  elif val.op is Ops.SQRT:
    lut, bn_mul_operand, output_scale, index_scale = _build_sqrt_lut()
    lut_le_start = 0xffffc000  # -16384
    lut_lo_end = 0x00004000    # 16384
    lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)  # same as EXP2: HYBRID, OFLOW, LO_LE_MUX=2
  else:
    raise AssertionError(f"dpu_lut: no builder for {val.op}")
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
  # --- BN_CFG: BN_ALU_ALGO=2, BN_RELU_BYPASS=1 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_CFG, (2 << 16) | (1 << 6))
  # --- BN_ALU_CFG ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000)
  # --- BN_MUL_CFG: fp16(index_scale) ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BN_MUL_CFG, (bn_mul_operand & 0xFFFF) << 16)
  # --- EW_CFG: relu_bypass=1, op_cvt_bypass=1, op_bypass=1 (LUT mode) ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 1))
  # --- EW_CVT_SCALE_VALUE: EW_OP_CVT_SCALE=1 ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)
  # --- OUT_CVT_SCALE: FP32TOFP16_EN=1, scale=1 ---
  # --- OUT_CVT_SHIFT: MINUS_EXP=13 (FP16 float division by 2^13) ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_OUT_CVT_SHIFT, (13 << 12))
  # --- SURFACE_ADD = 2 * surf_stride ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_SURFACE_ADD, 2 * surf_stride)
  # --- unknown reg 0x40c4 = 0 (from silu.py) ---
  emitter_emit(cmds, _T_DPU, 0x40c4, 0)
  # --- LUT config registers ---
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_CFG, lut_cfg)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_INFO, (5 << 16) | (5 << 8))
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LE_START, lut_le_start)
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_LUT_LO_END, lut_lo_end)
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
  return tuple(c.pack() for c in cmds), RKTask(0x18, 0x300, 4, "dpu_lut", layout, plan.out_slot), tuple(relocs)

def _emit_cmac(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """CNA+CORE matmul or sum. A=(M,K), B=(K,N), output=(M,N) FP32→FP16. Transforms in __call__."""
  cmds, relocs = [], []
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
  emitter_emit(cmds, _T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<4)|(1<<1)|1)
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
  return tuple(c.pack() for c in cmds), RKTask(0xd, 0x300, 0, "cmac", layout, plan.out_slot, const_val=cv), tuple(relocs)

def _emit_ppu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """PPU globalmax. Input (H,W,C) fp16 → (C,) fp16. Raw register values per pool.py.
  PPU processes channels in groups of 8 for FP16; C is padded to a multiple of 8."""
  cmds, relocs = [], []
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
  return tuple(c.pack() for c in cmds), RKTask(0x60, 0xc00, 1, "ppu", (in_h, in_w, channels, chan_padded), plan.out_slot), tuple(relocs)

# ---- emit_rk dispatcher (§2.3 API) ----
def emit_rk(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """Dispatch emission by plan.kind. Returns (cmds, task, relocs)."""
  return {"dpu":_emit_dpu, "dpu_lut":_emit_dpu_lut, "cmac":_emit_cmac, "ppu":_emit_ppu}[plan.kind](plan)

# ---- codec ----
_RK_MAGIC = 0x524b494d  # "RKIM"
_RK_KINDS = ("dpu", "dpu_lut", "cmac", "ppu")

def encode_rk(cmds: tuple[int,...], task: RKTask, relocs: tuple[RKReloc,...]) -> bytes:
  """Pack commands, task metadata, and relocations into deterministic bytes (§2.3)."""
  n_cmds, n_relocs, n_layout = len(cmds), len(relocs), len(task.layout)
  kind_layout = (_RK_KINDS.index(task.kind) << 24) | ((1 if task.is_copy else 0) << 23) | ((1 if task.is_fill else 0) << 22) | n_layout
  header = struct.pack("<IIIIIIIIi", _RK_MAGIC, 2, n_cmds, n_relocs, task.enable_mask, task.int_mask, task.op_idx, kind_layout, task.out_slot)
  reloc_bytes = struct.pack(f"<{n_relocs*6}I", *[v for r in relocs for v in (r.word_index, r.globals_slot, r.addend, r.shift, r.mask, r.field_shift)])
  layout = struct.pack(f"<{n_layout}i", *task.layout) if task.layout else b""
  const_bytes = struct.pack("<f", task.const_val) if task.const_val != 1.0 else b""
  return header + struct.pack(f"<{n_cmds}Q", *cmds) + reloc_bytes + layout + const_bytes

def decode_rk(data: bytes) -> tuple[list[int], RKTask, list[RKReloc]]:
  """Decode bytes into (cmds, task, relocs). Raises on bad magic, version, truncated tables, or out-of-range relocations."""
  if len(data) < 36: raise RuntimeError("RKImage: truncated header")
  magic, ver, n_cmds, n_relocs, emask, imask, op_idx, kl, out_slot = struct.unpack_from("<IIIIIIIIi", data, 0)
  if magic != _RK_MAGIC: raise RuntimeError("RKImage: bad magic")
  if ver != 2: raise RuntimeError(f"RKImage: unsupported version {ver}")
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
  const_val = struct.unpack_from("<f", data, off)[0] if off + 4 <= len(data) else 1.0
  return cmds, RKTask(emask, imask, op_idx, _RK_KINDS[ki], layout, out_slot, is_copy, is_fill, const_val=const_val), relocs

# ---- the native_program hook ----
def build_native_program(sink: UOp) -> UOp|None:
  """Classify and build a PROGRAM(SINK, LINEAR(INS...)). Raises RKPLAN_REJECT:<reason>
  if unsupported (no fallback per §15). Raises if a classified kernel fails emission."""
  # Pre-classification rewrite: MUL(a, RECIPROCAL(b)) → FDIV(a, b)
  sink = graph_rewrite(sink, _pm_fdiv, name="rk fdiv decomp")
  plan = plan_rk(sink)
  if isinstance(plan, str): raise RuntimeError(plan)  # reject — preserve reason, no fallback
  cmds, task, relocs = emit_rk(plan)
  # each INS carries an int (packed command), RKReloc, or RKTask as its arg
  ins_args = [task] + list(cmds) + list(relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))
