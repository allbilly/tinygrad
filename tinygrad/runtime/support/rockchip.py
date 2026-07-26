# pylint: skip-file
# RK3588 NPU compiled backend: pure match/classify + register emission + codec.
# PR 1 scope: one single-task fp16 path per compute family (DPU, CNA+CORE, PPU).
# See rockchip_plan.md §0 for the full scope and constraints.
from __future__ import annotations
import struct
from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, ProgramInfo
from tinygrad.runtime.autogen import rockchip as rk

# ---- register field helpers ----
# NOTE: _f helper removed — DPU and PPU emitters use raw register values directly.
# The autogen _f field shifts don't match hardware for stride and DATA_FORMAT registers.

# target ids for emit_raw (rkt_get_target(reg) + 1, from rkt_registers.h)
# PC=0x80 is the value used by the working allbilly reference (autogen has 0x100, which is wrong for PC)
_T_DPU, _T_DPU_RDMA, _T_CNA, _T_CORE, _T_PPU, _T_PPU_RDMA, _T_PC = 0x1001, 0x2001, 0x201, 0x801, 0x4001, 0x8001, 0x81

# DPU EW ALU algorithm values (from rknnops.h alu_case_*)
# NOTE: _DPU_EW_ALGO is kept for reference but not used — the emitter uses _EW_CFG instead

# ---- image format ----
@dataclass(frozen=True)
class RKCmd:
  """One register command word: (target<<48) | (value<<16) | reg."""
  target: int
  reg: int
  value: int
  def __repr__(self): return f"RK(t={self.target:#04x} r={self.reg:#06x} v={self.value:#010x})"
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
  def __repr__(self): return f"Reloc(w={self.word_index} g={self.globals_slot} +{self.addend:#x} >>{self.shift} &{self.mask:#x} <<{self.field_shift})"

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

# ---- classifier ----
def _is_fp16_only(sink: UOp) -> bool:
  """All tensor-carrying nodes must be fp16 (REDUCE may be fp32 for the NPU accumulator)."""
  for u in sink.toposort():
    if u.op is Ops.PARAM and u.dtype is not dtypes.half: return False
    if u.op is Ops.REDUCE and u.dtype not in (dtypes.half, dtypes.float): return False
    if u.op is Ops.STORE and u.src[1].dtype not in (dtypes.half, dtypes.void): return False
  return True

def _find_op(sink: UOp, op: Ops) -> UOp|None: return next((u for u in sink.toposort() if u.op is op), None)
def _reduce_node(sink: UOp) -> UOp|None: return _find_op(sink, Ops.REDUCE)
def _store_node(sink: UOp) -> UOp|None: return _find_op(sink, Ops.STORE)

def _unwrap(u: UOp) -> UOp: return u.src[0] if u.op is Ops.CAST else u
def _reduce_body(reduce: UOp) -> UOp: return _unwrap(reduce.src[0])

def _all_indexes(sink: UOp) -> list[UOp]:
  """Return all INDEX nodes in the sink in topological order."""
  return [u for u in sink.toposort() if u.op is Ops.INDEX]

def _is_flat_contiguous(idx: UOp) -> bool:
  """Single flat RANGE or CONST(0) — rejects broadcast, transpose, strided access."""
  return idx.op is Ops.RANGE or (idx.op is Ops.CONST and int(idx.arg) == 0)

def _is_ppu_channel_layout(idx: UOp, channels: int = 8) -> bool:
  """ADD(MUL(RANGE(REDUCE), CONST(ch)), RANGE(LOOP)) — channel-preserving layout."""
  if idx.op is not Ops.ADD: return False
  mul_side, rng_side = idx.src
  if mul_side.op is not Ops.MUL: return False
  mul_rng, mul_const = mul_side.src
  if mul_rng.op is not Ops.RANGE: return False
  if mul_const.op is not Ops.CONST or int(mul_const.arg) != channels: return False
  if rng_side.op is not Ops.RANGE: return False
  if mul_rng.arg[1].name != "REDUCE": return False
  if rng_side.arg[1].name != "LOOP": return False
  return True

def _is_cmac_matmul_layout(sink: UOp, reduce: UOp) -> bool:
  """Validate INDEX patterns match row-major matmul: out=(i,j), A=(i,k), B=(k,j).
  Checks structure and affine coefficients (CONST=N, K, N for out, A, B)."""
  body = _reduce_body(reduce)
  if body.op is not Ops.MUL: return False
  a_idx, b_idx = _unwrap(body.src[0]), _unwrap(body.src[1])
  if a_idx.op is not Ops.INDEX or b_idx.op is not Ops.INDEX: return False
  out_shape = _shape_of_store(sink)
  if len(out_shape) != 2: return False
  N, K = int(out_shape[1]), _reduce_extent(reduce)
  if K < 0: return False
  store = _store_node(sink)
  if store is None or store.src[0].op is not Ops.INDEX: return False
  return all([_is_2d_index(store.src[0].src[1], "LOOP", "LOOP", N),
              _is_2d_index(a_idx.src[1], "LOOP", "REDUCE", K),
              _is_2d_index(b_idx.src[1], "REDUCE", "LOOP", N)])

def _is_2d_index(idx: UOp, outer_kind: str, inner_kind: str, stride: int) -> bool:
  """Check idx is ADD(MUL(RANGE(outer_kind), CONST=stride), RANGE(inner_kind))."""
  if idx.op is not Ops.ADD: return False
  mul_side, rng_side = idx.src
  if mul_side.op is not Ops.MUL or rng_side.op is not Ops.RANGE: return False
  mul_rng, mul_const = mul_side.src
  if mul_rng.op is not Ops.RANGE or mul_const.op is not Ops.CONST: return False
  if int(mul_const.arg) != stride: return False
  return mul_rng.arg[1].name == outer_kind and rng_side.arg[1].name == inner_kind

def plan_rk(sink: UOp) -> RKPlan|str:
  """Classify a post-early_simplify SINK. Returns RKPlan on success, 'RKPLAN_REJECT:...' str on reject."""
  if not _is_fp16_only(sink): return "RKPLAN_REJECT:unsupported_dtype"
  reduce = _reduce_node(sink)
  # R3 DPU: no REDUCE, single STORE with ADD over two INDEXes (PR1: ADD only)
  if reduce is None:
    store = _store_node(sink)
    if store is None: return "RKPLAN_REJECT:no_add_mul_reduction"
    val = store.src[1]
    if val.op is not Ops.ADD:
      return f"RKPLAN_REJECT:unsupported_op:{val.op}"
    idx_count = sum(1 for s in val.src if s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX))
    if idx_count != 2:
      return f"RKPLAN_REJECT:unsupported_op:only-{idx_count}-index-inputs"
    for idx_node in _all_indexes(sink):
      if not _is_flat_contiguous(idx_node.src[1]):
        return f"RKPLAN_REJECT:unsupported_layout:{idx_node.src[1].op}"
    kind = "dpu"
  # R1 CMAC: REDUCE(ADD, MUL(INDEX, INDEX))
  elif reduce.arg[0] is Ops.ADD:
    body = _reduce_body(reduce)
    if body.op is Ops.MUL and all(s.op is Ops.INDEX or (s.op is Ops.CAST and s.src[0].op is Ops.INDEX) for s in body.src):
      if not _is_cmac_matmul_layout(sink, reduce):
        return "RKPLAN_REJECT:unsupported_layout"
      kind = "cmac"
    elif body.op is Ops.INDEX:
      return "RKPLAN_REJECT:no_add_mul_reduction"
    else:
      return f"RKPLAN_REJECT:unsupported_op:{body.op}"
  # R4 PPU: REDUCE(MAX, INDEX) — global max pool over (H,W,8) → (8,)
  elif reduce.arg[0] is Ops.MAX:
    body = _reduce_body(reduce)
    if body.op in (Ops.INDEX, Ops.CAST):
      out_shape = _shape_of_store(sink)
      K = _reduce_extent(reduce)
      store = _store_node(sink)
      if len(out_shape) != 1 or out_shape[0] != 8 or K < 4 or K % 2 != 0 or K > 32:
        return f"RKPLAN_REJECT:unsupported_layout:{out_shape}:{K}"
      if store is None or store.src[0].op is not Ops.INDEX or not _is_flat_contiguous(store.src[0].src[1]):
        return "RKPLAN_REJECT:unsupported_layout"
      for idx_node in _all_indexes(sink):
        if idx_node is store.src[0]: continue
        if not _is_ppu_channel_layout(idx_node.src[1], 8):
          return f"RKPLAN_REJECT:unsupported_layout:{idx_node.src[1].op}"
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
  extents = []
  for u in sink.toposort():
    if u.op is Ops.RANGE and u.arg[1].name == "LOOP":
      extent = u.src[0]
      if extent.op is Ops.CONST: extents.append(int(extent.arg))
      else: extents.append(-1)
  return extents

def _shape_of_store(sink: UOp) -> tuple[int, ...]:
  """Extract the output shape from the LOOP RANGE extents."""
  return tuple(_loop_extents(sink)) or (1,)

def _reduce_extent(reduce: UOp) -> int:
  """Get the reduction axis extent (K for matmul, N for max)."""
  rng = reduce.src[1]
  if rng.op is Ops.RANGE:
    extent = rng.src[0]
    if extent.op is Ops.CONST: return int(extent.arg)
  return -1

# ---- emitter ----
class _Emitter:
  def __init__(self):
    self.cmds: list[RKCmd] = []
    self.relocs: list[RKReloc] = []
  def emit(self, target, reg, value): self.cmds.append(RKCmd(target, reg, value))
  def reloc(self, globals_slot, addend=0, shift=0, mask=0xFFFFFFFF, field_shift=0):
    self.relocs.append(RKReloc(len(self.cmds)-1, globals_slot, addend, shift, mask, field_shift))
  def pc_op_en(self, reserved_0): self.emit(_T_PC, rk.REG_PC_OPERATION_ENABLE, (reserved_0 << 1))

# DPU EW_CFG for ADD (from ref/rk3588/examples/simple_add.py)
# data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1, ALU_ALGO=2
_EW_ADD = 0x108202c0

def _emit_dpu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """DPU elementwise ADD. 14-register sequence from simple_add.py."""
  e = _Emitter()
  sink = plan.sink
  store = _store_node(sink)
  assert store is not None, "dpu: no STORE node"
  val = store.src[1]
  # derive operand slots from INDEX operands — allows in-place a.assign(a+b)
  a_node, b_node = val.src
  a_slot, b_slot = _unwrap(a_node).src[0].buf_uop.arg.slot, _unwrap(b_node).src[0].buf_uop.arg.slot
  total = 1
  for s in _shape_of_store(sink): total *= s
  dw = (total + 7) // 8 - 1
  e.emit(_T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5)
  e.emit(_T_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002)
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007)
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, dw)
  e.emit(_T_DPU, rk.REG_DPU_EW_CFG, _EW_ADD)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, dw)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0x70007)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008)
  e.emit(_T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  e.reloc(plan.out_slot)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  e.reloc(a_slot)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0)
  e.reloc(b_slot)
  e.emit(_T_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849)
  e.pc_op_en(12)
  cmds = tuple(c.pack() for c in e.cmds)
  task = RKTask(0x18, 0x300, 4, "dpu", (total,), plan.out_slot)
  return cmds, task, tuple(e.relocs)

def _emit_cmac(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """CNA+CORE matmul. A=(M,K), B=(K,N), output=(M,N) FP32→FP16. Transforms in __call__."""
  e = _Emitter()
  sink = plan.sink
  reduce = _reduce_node(sink)
  assert reduce is not None, "cmac: no REDUCE node"
  body = _reduce_body(reduce)
  a_node, b_node = body.src[0], body.src[1]
  # derive A/B slots from INDEX operands — allows a@a (same buffer, two relocs)
  a_slot = _unwrap(a_node).src[0].buf_uop.arg.slot
  b_slot = _unwrap(b_node).src[0].buf_uop.arg.slot
  out_shape = _shape_of_store(sink)
  M, N = int(out_shape[0]), int(out_shape[1])
  K = _reduce_extent(reduce)
  if K < 0: raise RuntimeError("cmac: K must be compile-time constant")
  # NPU geometry constants from gemm.py
  FP16_BYTES = 2
  CBUF_ENTRY_BYTES = 128
  CBUF_ENTRIES_PER_BANK = 256
  RK_CBUF_BANKS = 12
  MIN_CHANNEL_TILE = 32
  CBUF_BANK_SIZE = CBUF_ENTRIES_PER_BANK * CBUF_ENTRY_BYTES  # 32 KiB
  RK_MIN_WIDE_FEATURE_GRAINS = 80
  RK_LINE_STRIDE_GROUP_CAP = 13
  # layout: align K and N to 32
  def _ceil_div(x, y): return (x + y - 1) // y
  def _align_up(x, a): return _ceil_div(x, a) * a
  aligned_k = max(MIN_CHANNEL_TILE, _align_up(K, MIN_CHANNEL_TILE))
  align_out = max(MIN_CHANNEL_TILE, _align_up(N, MIN_CHANNEL_TILE))
  align_in = max(aligned_k, align_out)
  eff_k = align_in if align_in != aligned_k else K
  input_row_bytes = align_in * FP16_BYTES
  # feature grains and line stride from gemm.py
  even_rows_per_two_banks = (_ceil_div(2 * CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
  feature_grains = max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)
  line_stride = 4 * min(_ceil_div(eff_k, MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
  notch_val = 8 * min(align_out // MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
  data_banks = max(1, _ceil_div(M * input_row_bytes, CBUF_BANK_SIZE))
  wt_banks = RK_CBUF_BANKS - data_banks
  if data_banks > RK_CBUF_BANKS-1 or input_row_bytes*align_out > wt_banks*CBUF_BANK_SIZE:
    raise RuntimeError("RKPLAN_REJECT:cmac_exceeds_cbuf")
  # --- exact register sequence from gemm.py make_gemm_regs ---
  # 1. DPU S_POINTER
  e.emit(_T_DPU, rk.REG_DPU_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  # 2. CNA CONV_CON1: IN_PRECISION=2, PROC_PRECISION=2, GROUP_LINE_OFF=1
  e.emit(_T_CNA, rk.REG_CNA_CONV_CON1, (2<<4)|(2<<7)|(1<<29))
  # 3. CNA CONV_CON2: FEATURE_GRAINS
  e.emit(_T_CNA, rk.REG_CNA_CONV_CON2, (feature_grains << 4))
  # 4. CNA CONV_CON3: CONV_Y_STRIDE=1, CONV_X_STRIDE=1
  e.emit(_T_CNA, rk.REG_CNA_CONV_CON3, (1<<3)|1)
  # 5-8. CNA data sizes
  e.emit(_T_CNA, rk.REG_CNA_DATA_SIZE0, (1<<16)|M)
  e.emit(_T_CNA, rk.REG_CNA_DATA_SIZE1, ((align_in-1)<<16)|align_in)
  e.emit(_T_CNA, rk.REG_CNA_DATA_SIZE2, 1)
  e.emit(_T_CNA, rk.REG_CNA_DATA_SIZE3, M)
  # 9-11. CNA weight sizes
  e.emit(_T_CNA, rk.REG_CNA_WEIGHT_SIZE0, input_row_bytes * align_out)
  e.emit(_T_CNA, rk.REG_CNA_WEIGHT_SIZE1, input_row_bytes)
  e.emit(_T_CNA, rk.REG_CNA_WEIGHT_SIZE2, (1<<24)|(1<<16)|align_out)
  # 12-13. CNA CBUF config
  e.emit(_T_CNA, rk.REG_CNA_CBUF_CON0, ((RK_CBUF_BANKS-data_banks)<<4)|data_banks)
  e.emit(_T_CNA, rk.REG_CNA_CBUF_CON1, _ceil_div(align_in, MIN_CHANNEL_TILE))
  # 14-18. CNA CVT config
  e.emit(_T_CNA, rk.REG_CNA_CVT_CON0, (1<<3)|(1<<1)|1)
  for r in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4): e.emit(_T_CNA, r, 1<<16)
  # 19. CNA FEATURE_DATA_ADDR (relocated — input A)
  e.emit(_T_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0)
  e.reloc(a_slot)
  # 20-24. CNA DMA config
  e.emit(_T_CNA, rk.REG_CNA_DMA_CON0, (15<<16)|15)
  e.emit(_T_CNA, rk.REG_CNA_DMA_CON1, line_stride)
  e.emit(_T_CNA, rk.REG_CNA_DMA_CON2, 0)
  e.emit(_T_CNA, rk.REG_CNA_FC_DATA_SIZE0, (1<<16)|M)
  e.emit(_T_CNA, rk.REG_CNA_FC_DATA_SIZE1, align_in)
  # 25. CNA DCOMP_ADDR0 (relocated — weight B, NO 0x4000 offset)
  e.emit(_T_CNA, rk.REG_CNA_DCOMP_ADDR0, 0)
  e.reloc(b_slot)
  # 26-29. CORE config
  e.emit(_T_CORE, rk.REG_CORE_MISC_CFG, (2<<8)|1)
  e.emit(_T_CORE, rk.REG_CORE_DATAOUT_SIZE_0, ((M-1)<<16)|0)
  e.emit(_T_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out-1)
  e.emit(_T_CORE, 0x3030, 0)  # CORE_RESERVED_3030
  # 30-45. DPU output config (FP32 output: OUT_PRECISION=5, size_e=3)
  e.emit(_T_DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15<<5)|(2<<1))
  e.emit(_T_DPU, rk.REG_DPU_DATA_FORMAT, (5<<29)|(2<<26)|2)
  e.emit(_T_DPU, rk.REG_DPU_DST_BASE_ADDR, 0)
  e.reloc(plan.out_slot)
  e.emit(_T_DPU, rk.REG_DPU_DST_SURF_STRIDE, (1<<4))
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0)
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, M-1)
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch_val<<16)|notch_val)
  e.emit(_T_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out-1)<<16)|(align_out-1))
  e.emit(_T_DPU, rk.REG_DPU_BS_CFG, (1<<6)|(1<<4)|(1<<1)|1)
  e.emit(_T_DPU, rk.REG_DPU_BS_OW_CFG, (3<<8)|(3<<5)|(3<<2)|(1<<1))
  e.emit(_T_DPU, rk.REG_DPU_WDMA_SIZE_0, align_out-1)
  e.emit(_T_DPU, rk.REG_DPU_WDMA_SIZE_1, ((M-1)<<16)|0)
  e.emit(_T_DPU, rk.REG_DPU_BN_CFG, (1<<6)|(1<<4)|(1<<1)|1)
  e.emit(_T_DPU, rk.REG_DPU_EW_CFG, (1<<9)|(1<<8)|(1<<7)|(1<<1)|1)
  e.emit(_T_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0)  # FP32 output, no conversion
  e.emit(_T_DPU, rk.REG_DPU_SURFACE_ADD, (4<<4))
  # 46. PC_OPERATION_ENABLE: CNA+CORE+DPU (reserved_0=6, op_en=1)
  e.emit(_T_PC, rk.REG_PC_OPERATION_ENABLE, (6<<1)|1)
  cmds = tuple(c.pack() for c in e.cmds)
  task = RKTask(0xd, 0x300, 0, "cmac", (M, N, K, align_in, align_out), plan.out_slot)
  return cmds, task, tuple(e.relocs)

def _emit_ppu(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """PPU globalmax. Input (H,W,8) fp16 → (8,) fp16. Raw register values per pool.py."""
  e = _Emitter()
  sink = plan.sink
  reduce = _reduce_node(sink)
  assert reduce is not None, "ppu: no REDUCE node"
  # geometry: input (in_h, in_w, 8) fp16, globalmax → (1, 1, 8) = (8,) fp16
  out_shape = _shape_of_store(sink)
  if len(out_shape) != 1 or out_shape[0] != 8:
    raise RuntimeError(f"ppu: PR1 only supports globalmax over (H,W,8) → (8,), got output shape {out_shape}")
  K = _reduce_extent(reduce)
  if K < 0: raise RuntimeError("ppu: reduce extent must be compile-time constant")
  # For PR1: input is (H, W, 8), reduce over H*W → output (8,)
  # The PPU hardware requires in_h >= 2 and in_w >= 2 (see pool.py line 243).
  # Split the flattened reduction K into a 2D shape (in_h, in_w) where in_h * in_w = K.
  POOL_CHANNELS = 8
  FP16_BYTES = 2
  if K < 4 or K % 2 != 0 or K > 32:
    raise RuntimeError(f"ppu: reduce extent K={K} must be even, >= 4, and <= 32 (PPU kernel field is 4 bits)")
  in_h = 2
  in_w = K // 2
  # zero-based field values (hardware expects N-1 for N elements)
  in_w_field = in_w - 1
  in_h_field = in_h - 1
  channel_field = POOL_CHANNELS - 1
  # globalmax: kernel = full input, stride = full input, output = 1x1x8
  k_h = in_h_field
  k_w = in_w_field
  s_h = in_h_field
  s_w = in_w_field
  dst_surf_stride = 1  # global: single output surface
  index_add = 1  # global: index_add=1
  width_stride = in_w * POOL_CHANNELS * FP16_BYTES  # bytes
  src_surf_stride = width_stride * in_h  # bytes
  # --- exact register sequence from pool.py pooling_regs("globalmax") ---
  # All values are raw 32-bit, written directly (no _f field shifting)
  # 1-2. S_POINTER
  e.emit(_T_PPU, rk.REG_PPU_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_S_POINTER, (1<<3)|(1<<2)|(1<<1))
  # 3-5. PPU input cube (zero-based)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, in_w_field)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, in_h_field)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, channel_field)
  # 6-8. PPU output cube (global: 0,0,7)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_OUT_WIDTH, 0)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_OUT_HEIGHT, 0)
  e.emit(_T_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, channel_field)
  # 9. OPERATION_MODE_CFG: flying=1, pooling_method=1 (MAX)
  e.emit(_T_PPU, rk.REG_PPU_OPERATION_MODE_CFG, (1<<4)|1)
  # 10. POOLING_KERNEL_CFG: (s_h<<20)|(s_w<<16)|(k_h<<8)|k_w
  e.emit(_T_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (s_h<<20)|(s_w<<16)|(k_h<<8)|k_w)
  # 11. DST_BASE_ADDR (relocated — pool.py writes (output_dma // 16) << 4)
  e.emit(_T_PPU, rk.REG_PPU_DST_BASE_ADDR, 0)
  e.reloc(plan.out_slot, shift=4, mask=0xFFFFFFF, field_shift=4)
  # 12. DST_SURF_STRIDE (raw value — pool.py writes dst_surf_stride directly)
  e.emit(_T_PPU, rk.REG_PPU_DST_SURF_STRIDE, dst_surf_stride)
  # 13. DATA_FORMAT (raw — pool.py writes (index_add << 16) | 2; autogen INDEX_ADD shift=4 is WRONG)
  e.emit(_T_PPU, rk.REG_PPU_DATA_FORMAT, (index_add << 16) | 2)
  # 14. MISC_CTRL (raw — pool.py writes 3 for burst_len)
  e.emit(_T_PPU, rk.REG_PPU_MISC_CTRL, 3)
  # 15-17. RDMA input cube (zero-based)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH, in_w_field)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, in_h_field)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, channel_field)
  # 18. RDMA SRC_BASE_ADDR (relocated — pool.py writes raw input_dma)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0)
  e.reloc(plan.in_slots[0])
  # 19-20. RDMA strides (raw bytes — pool.py writes width_stride/src_surf_stride directly;
  #   hardware field at shift=4 divides by 16 to get 16-byte units)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, width_stride)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, src_surf_stride)
  # 21. RDMA DATA_FORMAT (raw — pool.py writes 2 for fp16)
  e.emit(_T_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_DATA_FORMAT, 2)
  # 22. RDMA OPERATION_ENABLE (raw — pool.py writes 1)
  # NOTE: autogen has wrong address 0x7008; correct is 0x7038 (see ref/rk3588/experimental/pool.py)
  e.emit(_T_PPU_RDMA, 0x7038, 1)
  # 23-24. RECIP_KERNEL_WIDTH/HEIGHT (raw — pool.py writes 0 for max pool, 30720 for avg)
  e.emit(_T_PPU, rk.REG_PPU_RECIP_KERNEL_WIDTH, 0)
  e.emit(_T_PPU, rk.REG_PPU_RECIP_KERNEL_HEIGHT, 0)
  # 25. PC_OPERATION_ENABLE: PPU-only mode (reserved_0=48, op_en=0 — pool.py uses <<1 without |1)
  e.pc_op_en(48)
  cmds = tuple(c.pack() for c in e.cmds)
  task = RKTask(0x60, 0xc00, 1, "ppu", (in_h, in_w, POOL_CHANNELS), plan.out_slot)
  return cmds, task, tuple(e.relocs)

# ---- emit_rk dispatcher (§2.3 API) ----
def emit_rk(plan: RKPlan) -> tuple[tuple[int,...], RKTask, tuple[RKReloc,...]]:
  """Dispatch emission by plan.kind. Returns (cmds, task, relocs)."""
  return {"dpu":_emit_dpu, "cmac":_emit_cmac, "ppu":_emit_ppu}[plan.kind](plan)

# ---- codec ----
_RK_MAGIC = 0x524b494d  # "RKIM"
_RK_KINDS = ("dpu", "cmac", "ppu")

def encode_rk(cmds: tuple[int,...], task: RKTask, relocs: tuple[RKReloc,...]) -> bytes:
  """Pack commands, task metadata, and relocations into deterministic bytes (§2.3)."""
  n_cmds, n_relocs, n_layout = len(cmds), len(relocs), len(task.layout)
  kind_layout = (_RK_KINDS.index(task.kind) << 24) | n_layout
  header = struct.pack("<IIIIIIIIi", _RK_MAGIC, 2, n_cmds, n_relocs,
                       task.enable_mask, task.int_mask, task.op_idx, kind_layout, task.out_slot)
  cmd_bytes = struct.pack(f"<{n_cmds}Q", *cmds)
  reloc_bytes = struct.pack(f"<{n_relocs*6}I", *[v for r in relocs for v in (r.word_index, r.globals_slot, r.addend, r.shift, r.mask, r.field_shift)])
  layout = struct.pack(f"<{n_layout}i", *task.layout) if task.layout else b""
  return header + cmd_bytes + reloc_bytes + layout

def decode_rk(data: bytes) -> tuple[list[int], RKTask, list[RKReloc]]:
  """Decode bytes into (cmds, task, relocs). Raises on bad magic, version, truncated tables, or out-of-range relocations."""
  if len(data) < 36: raise RuntimeError("RKImage: truncated header")
  magic, ver, n_cmds, n_relocs, emask, imask, op_idx, kl, out_slot = struct.unpack_from("<IIIIIIIIi", data, 0)
  if magic != _RK_MAGIC: raise RuntimeError("RKImage: bad magic")
  if ver != 2: raise RuntimeError(f"RKImage: unsupported version {ver}")
  ki = (kl >> 24) & 0xFF
  if ki >= len(_RK_KINDS): raise RuntimeError(f"RKImage: invalid kind index {ki}")
  n_layout = kl & 0xFFFFFF
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
  return cmds, RKTask(emask, imask, op_idx, _RK_KINDS[ki], layout, out_slot), relocs

# ---- the native_program hook ----
def build_native_program(sink: UOp) -> UOp|None:
  """Classify and build a PROGRAM(SINK, LINEAR(INS...)). Raises RKPLAN_REJECT:<reason>
  if unsupported (no fallback per §15). Raises if a classified kernel fails emission."""
  plan = plan_rk(sink)
  if isinstance(plan, str): raise RuntimeError(plan)  # reject — preserve reason, no fallback
  cmds, task, relocs = emit_rk(plan)
  # each INS carries an int (packed command), RKReloc, or RKTask as its arg
  ins_task = [UOp(Ops.INS, arg=task)]
  ins_cmds = [UOp(Ops.INS, arg=c) for c in cmds]
  ins_relocs = [UOp(Ops.INS, arg=r) for r in relocs]
  lin = UOp(Ops.LINEAR, src=tuple(ins_task + ins_cmds + ins_relocs))
  prog_info = ProgramInfo.from_sink(sink)
  prg = UOp(Ops.PROGRAM, src=(sink, lin), arg=prog_info)
  return prg
