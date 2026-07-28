# pylint: skip-file
# NVDLA compiled backend: UOp classification + direct MMIO register emission.
# Port of the rockchip backend to NVDLA using direct /dev/mem MMIO writes.
# No KMD needed — registers are written directly to NVDLA's MMIO space.
# PR 1 scope: SDP elementwise (ADD/SUB/MUL/MAX), scalar operand, BDMA copy, constant fill.
# fp16 only. No LUT, no CMAC, no PDP in this first pass.
#
# The register command format is (offset, value) pairs, written directly to MMIO.
# This mirrors the rockchip (target, reg, value) model but without the target field
# since NVDLA's flat MMIO offset space already encodes the block routing.
#
# Register values are initially copied from the rockchip DPU (which is an NVDLA derivative).
# Values that differ between RK3588 and original NVDLA will be adjusted during VP testing.
from __future__ import annotations
import struct
from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, UOp, ProgramInfo
from tinygrad.runtime.autogen import nvdla as nv

# ---- sentinel slots (matching rockchip convention) ----
_CONST_SLOT = 0xFFFF  # sentinel globals_slot for scalar constant buffer
_ZERO_SLOT = 0xFFFD   # sentinel globals_slot for zero-filled input buffer (fill)

# ---- SDP EW config values ----
# Copied from rockchip _DPU_EW_CFGS (RK3588 is an NVDLA derivative).
# Base: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
# These may need adjustment for original NVDLA — see VP test results.
_EW_BASE = 0x108002c0
_SDP_EW_CFGS = {Ops.ADD: _EW_BASE | (2 << 16), Ops.SUB: _EW_BASE | (4 << 16),
                 Ops.MUL: _EW_BASE | (1 << 2) | (1 << 8), Ops.MAX: _EW_BASE,
                 Ops.FDIV: _EW_BASE | (3 << 16) | (1 << 8)}

# ---- command format: (offset, value) packed as a 64-bit qword ----
# offset in bits 0-31, value in bits 32-63. This gives 32 bits for each,
# sufficient for all NVDLA registers (max offset 0x10050, max value 0xFFFFFFFF).
@dataclass(frozen=True)
class NVDLACmd:
  """One MMIO register write: write `value` to MMIO base + `offset`."""
  offset: int
  value: int
  def pack(self) -> int: return ((self.value & 0xFFFFFFFF) << 32) | (self.offset & 0xFFFFFFFF)

@dataclass(frozen=True)
class NVDLAReloc:
  """Patch a command's value field with a buffer's physical address at runtime."""
  cmd_index: int      # index into the cmds list
  globals_slot: int   # which buffer to use (or _CONST_SLOT/_ZERO_SLOT)
  addend: int = 0     # added to the buffer address before patching
  mask: int = 0xFFFFFFFF  # mask for the address value
  field_shift: int = 0    # left shift within the 32-bit value field

@dataclass(frozen=True)
class NVDLATask:
  """Task metadata carried in the image."""
  kind: str             # "sdp" or "bdma"
  layout: tuple         # (total,) for SDP/BDMA
  out_slot: int
  in_slots: tuple[int, ...]
  is_copy: bool = False
  is_fill: bool = False
  const_val: float = 1.0
  ew_op: Ops = Ops.ADD
  fp32_inputs: tuple[int, ...] = ()
  fp32_output: bool = False

@dataclass(frozen=True)
class NVDLAImage:
  """Serialized NVDLA program: register commands + task metadata + relocs."""
  cmds: tuple[int, ...]    # packed (offset<<32 | value) qwords
  task: NVDLATask
  relocs: tuple[NVDLAReloc, ...]

# ---- classifier (adapted from rockchip plan_rk) ----
def _is_fp16_only(sink: UOp) -> bool:
  for u in sink.toposort():
    if u.op is Ops.INDEX and u.dtype is not None:
      if u.dtype not in (dtypes.half, dtypes.float): return False
  return True

def _store_node(sink: UOp) -> UOp|None: return next((u for u in sink.toposort() if u.op is Ops.STORE), None)
def _unwrap(u: UOp) -> UOp:
  if u.op is Ops.CAST and u.dtype == u.src[0].dtype: return u.src[0]
  return u

def _try_scalar(val: UOp) -> tuple[int, float, bool]|None:
  if val.op not in _SDP_EW_CFGS: return None
  srcs = [_unwrap(s) for s in val.src]
  idx_slots = [s.src[0].buf_uop.arg.slot for s in srcs if s.op is Ops.INDEX]
  consts = [float(s.arg) for s in srcs if s.op is Ops.CONST]
  if len(idx_slots) == 1 and len(consts) == 1:
    swap = srcs[1].op is Ops.INDEX
    return idx_slots[0], consts[0], swap
  return None

def _try_sub(val: UOp) -> tuple[int, int]|None:
  if val.op is not Ops.SUB: return None
  srcs = [_unwrap(s) for s in val.src]
  if all(s.op is Ops.INDEX for s in srcs):
    return srcs[0].src[0].buf_uop.arg.slot, srcs[1].src[0].buf_uop.arg.slot
  return None

def _loop_extents(sink: UOp) -> list[int]:
  return [int(u.src[0].arg) if u.src[0].op is Ops.CONST else -1
          for u in sink.toposort() if u.op is Ops.RANGE and u.arg[1].name == "LOOP"]

def _shape_of_store(sink: UOp) -> tuple[int, ...]:
  return tuple(_loop_extents(sink)) or (1,)

# ---- emitter helpers (mirror rockchip emitter_emit/emitter_reloc) ----
def _make_emitter():
  cmds:list[NVDLACmd] = []
  relocs:list[NVDLAReloc] = []
  def e(offset:int, value:int): cmds.append(NVDLACmd(offset, value))
  def r(globals_slot:int, addend:int=0, mask:int=0xFFFFFFFF, field_shift:int=0):
    relocs.append(NVDLAReloc(len(cmds)-1, globals_slot, addend, mask, field_shift))
  return cmds, relocs, e, r

# ---- planning ----
def plan_nvdla(sink: UOp) -> NVDLATask|str:
  """Classify a post-early_simplify SINK. Returns NVDLATask on success, 'NVDLA_REJECT:...' on reject."""
  if not _is_fp16_only(sink): return "NVDLA_REJECT:unsupported_dtype"
  store = _store_node(sink)
  if store is None: return "NVDLA_REJECT:no_store"
  val = _unwrap(store.src[1])
  total = prod(_shape_of_store(sink))
  prog_info = ProgramInfo.from_sink(sink)
  out_slots = list(prog_info.outs)
  if len(out_slots) != 1: return f"NVDLA_REJECT:unsupported_layout:{len(out_slots)}-outputs"
  out_slot = out_slots[0]
  in_slots = tuple(s for s in prog_info.globals if s != out_slot)
  fp32_param_slots = {u.arg.slot for u in sink.toposort() if u.op is Ops.PARAM and u.dtype is dtypes.float}
  fp32_inputs = tuple(s for s in in_slots if s in fp32_param_slots)
  fp32_output = out_slot in fp32_param_slots
  # DMA copy: val is a bare INDEX
  if val.op is Ops.INDEX:
    return NVDLATask("bdma", (total,), out_slot, (val.src[0].buf_uop.arg.slot,),
                     is_copy=True, fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  # Constant fill: val is a bare CONST
  if val.op is Ops.CONST:
    return NVDLATask("sdp", (total,), out_slot, in_slots, is_fill=True,
                     const_val=float(val.arg), ew_op=Ops.ADD,
                     fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  # Binary EW with two INDEX operands
  if val.op in _SDP_EW_CFGS and all(_unwrap(s).op is Ops.INDEX for s in val.src):
    srcs = [_unwrap(s) for s in val.src]
    slots = tuple(s.src[0].buf_uop.arg.slot for s in srcs)
    return NVDLATask("sdp", (total,), out_slot, slots, ew_op=val.op,
                     fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  # Scalar EW: INDEX * CONST
  scalar = _try_scalar(val)
  if scalar is not None:
    slot, const_val, _swap = scalar
    return NVDLATask("sdp", (total,), out_slot, (slot,), const_val=const_val,
                     ew_op=val.op, fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  # SUB with two INDEX operands
  sub_slots = _try_sub(val)
  if sub_slots is not None:
    return NVDLATask("sdp", (total,), out_slot, sub_slots, ew_op=Ops.SUB,
                     fp32_inputs=fp32_inputs, fp32_output=fp32_output)
  return f"NVDLA_REJECT:unsupported_op:{val.op}"

# ---- emission: build register command sequence ----
def _emit_sdp(task: NVDLATask) -> NVDLAImage:
  """Build SDP register command sequence for an elementwise op.
  Register sequence adapted from rockchip _emit_dpu, with NVDLA SDP register addresses.
  RK3588 DPU (0x4xxx) → NVDLA SDP (0x9xxx), DPU_RDMA (0x5xxx) → SDP_RDMA (0x8xxx).
  Register values are initially copied from rockchip; adjust during VP testing."""
  cmds, relocs, e, r = _make_emitter()
  total = task.layout[0]
  dw = (total + 7) // 8 - 1  # width in 8-element atoms minus 1
  # SDP configuration (was DPU in rockchip)
  e(nv.NVDLA_SDP_S_POINTER_0, 0x0e)                    # re-arm ping-pong pointers
  e(nv.NVDLA_SDP_D_FEATURE_MODE_CFG_0, 0x1e5)          # winograd off, pipe overlap, fp16
  e(nv.NVDLA_SDP_D_DATA_FORMAT_0, 0x48000002)          # fp16 output, pipe overlapped
  e(nv.NVDLA_SDP_D_DATA_CUBE_CHANNEL_0, 0x70007)       # 8 channels, atomic 8
  e(nv.NVDLA_SDP_D_DATA_CUBE_WIDTH_0, dw)
  e(nv.NVDLA_SDP_D_DATA_CUBE_HEIGHT_0, 0)
  # SDP RDMA configuration (was DPU_RDMA in rockchip)
  e(nv.NVDLA_SDP_RDMA_S_POINTER_0, 0x0e)
  e(nv.NVDLA_SDP_RDMA_D_DATA_CUBE_WIDTH_0, dw)
  e(nv.NVDLA_SDP_RDMA_D_DATA_CUBE_HEIGHT_0, 0)
  e(nv.NVDLA_SDP_RDMA_D_DATA_CUBE_CHANNEL_0, 0x7)
  e(nv.NVDLA_SDP_RDMA_D_ERDMA_CFG_0, 0x40000008)       # EW DMA enabled, fp16
  # Output address (reloc patches at submit time)
  e(nv.NVDLA_SDP_D_DST_BASE_ADDR_LOW_0, 0)
  r(task.out_slot)
  # Input/operand addresses + EW config
  if task.is_fill:
    # fill: src = zero buffer, ew = const buffer, EW = ADD (zero + const = fill)
    e(nv.NVDLA_SDP_RDMA_D_SRC_BASE_ADDR_LOW_0, 0)
    r(_ZERO_SLOT)
    e(nv.NVDLA_SDP_RDMA_D_EW_BASE_ADDR_LOW_0, 0)
    r(_CONST_SLOT, struct.unpack('<I', struct.pack('<f', task.const_val))[0])
    e(nv.NVDLA_SDP_D_DP_EW_CFG_0, _SDP_EW_CFGS[Ops.ADD])
  elif len(task.in_slots) == 2:
    # binary EW: src = input_a, ew = input_b
    e(nv.NVDLA_SDP_RDMA_D_SRC_BASE_ADDR_LOW_0, 0)
    r(task.in_slots[0])
    e(nv.NVDLA_SDP_RDMA_D_EW_BASE_ADDR_LOW_0, 0)
    r(task.in_slots[1])
    e(nv.NVDLA_SDP_D_DP_EW_CFG_0, _SDP_EW_CFGS[task.ew_op])
  elif len(task.in_slots) == 1:
    # scalar EW: src = input, ew = const
    e(nv.NVDLA_SDP_RDMA_D_SRC_BASE_ADDR_LOW_0, 0)
    r(task.in_slots[0])
    e(nv.NVDLA_SDP_RDMA_D_EW_BASE_ADDR_LOW_0, 0)
    r(_CONST_SLOT, struct.unpack('<I', struct.pack('<f', task.const_val))[0])
    e(nv.NVDLA_SDP_D_DP_EW_CFG_0, _SDP_EW_CFGS[task.ew_op])
  else:
    raise RuntimeError(f"nvdla: unsupported SDP slot layout: {task.in_slots}")
  # Output CVT (same as rockchip: FP32TOFP16_EN=1, scale=1)
  e(nv.NVDLA_SDP_D_CVT_SCALE_0, (1 << 16) | 1)
  # SDP RDMA feature mode (same as rockchip: 0x17849)
  e(nv.NVDLA_SDP_RDMA_D_FEATURE_MODE_CFG_0, 0x17849)
  # OP_ENABLE: kick off SDP_RDMA first, then SDP
  # (NVDLA has per-block OP_ENABLE, unlike rockchip's single PC_OPERATION_ENABLE)
  e(nv.NVDLA_SDP_RDMA_D_OP_ENABLE_0, 1)
  e(nv.NVDLA_SDP_D_OP_ENABLE_0, 1)
  return NVDLAImage(tuple(c.pack() for c in cmds), task, tuple(relocs))

def _emit_bdma(task: NVDLATask) -> NVDLAImage:
  """Build BDMA register command sequence for a copy operation.
  BDMA copies a contiguous block from src to dst in DRAM."""
  cmds, relocs, e, r = _make_emitter()
  total = task.layout[0]
  bytes_per_elem = 2  # fp16
  total_bytes = total * bytes_per_elem
  # BDMA: copy total_bytes from src to dst, one line, one surface
  e(nv.NVDLA_BDMA_CFG_SRC_ADDR_LOW_0, 0)
  r(task.in_slots[0])
  e(nv.NVDLA_BDMA_CFG_SRC_ADDR_HIGH_0, 0)
  e(nv.NVDLA_BDMA_CFG_DST_ADDR_LOW_0, 0)
  r(task.out_slot)
  e(nv.NVDLA_BDMA_CFG_DST_ADDR_HIGH_0, 0)
  e(nv.NVDLA_BDMA_CFG_LINE_0, total_bytes)        # bytes per line
  e(nv.NVDLA_BDMA_CFG_LINE_REPEAT_0, 1)           # 1 line
  e(nv.NVDLA_BDMA_CFG_SRC_LINE_0, total_bytes)    # source line stride
  e(nv.NVDLA_BDMA_CFG_DST_LINE_0, total_bytes)    # dest line stride
  e(nv.NVDLA_BDMA_CFG_SURF_REPEAT_0, 1)           # 1 surface
  e(nv.NVDLA_BDMA_CFG_SRC_SURF_0, total_bytes)    # source surface stride
  e(nv.NVDLA_BDMA_CFG_DST_SURF_0, total_bytes)    # dest surface stride
  e(nv.NVDLA_BDMA_CFG_CMD_0, 0x10001000)          # enable, group0, src/dst in MC
  e(nv.NVDLA_BDMA_CFG_OP_0, 1)                    # enable operation
  e(nv.NVDLA_BDMA_CFG_LAUNCH0_0, 1)               # launch group0
  e(nv.NVDLA_BDMA_CFG_LAUNCH1_0, 0)
  return NVDLAImage(tuple(c.pack() for c in cmds), task, tuple(relocs))

def emit_nvdla(task: NVDLATask) -> NVDLAImage:
  if task.kind == "sdp": return _emit_sdp(task)
  if task.kind == "bdma": return _emit_bdma(task)
  raise RuntimeError(f"nvdla: unknown task kind: {task.kind}")

# ---- codec: serialize/deserialize NVDLAImage to/from bytes ----
_NVDLA_MAGIC = 0x4E56444C  # "NVDA"
_NVDLA_VERSION = 1

def encode_nvdla(img: NVDLAImage) -> bytes:
  """Serialize an NVDLAImage into a bytes blob for TinyELF.lib."""
  blob = struct.pack("<II", _NVDLA_MAGIC, _NVDLA_VERSION)
  blob += struct.pack("<I", len(img.cmds))
  for cmd in img.cmds: blob += struct.pack("<Q", cmd)
  blob += struct.pack("<I", len(img.relocs))
  for rel in img.relocs:
    blob += struct.pack("<IIIII", rel.cmd_index, rel.globals_slot, rel.addend, rel.mask, rel.field_shift)
  t = img.task
  kind_bytes = t.kind.encode("ascii")
  blob += struct.pack("<B", len(kind_bytes)) + kind_bytes
  blob += struct.pack("<I", len(t.layout)) + struct.pack("<" + "I"*len(t.layout), *t.layout)
  blob += struct.pack("<IffB", t.out_slot, t.const_val, 0.0, 0)  # placeholder for remaining fields
  # TODO: serialize all task fields properly
  return blob

def decode_nvdla(data: bytes) -> NVDLAImage:
  """Deserialize an NVDLAImage from a bytes blob."""
  magic, ver = struct.unpack_from("<II", data, 0)
  if magic != _NVDLA_MAGIC: raise RuntimeError(f"NVDLAImage: bad magic {magic:#x}")
  if ver != _NVDLA_VERSION: raise RuntimeError(f"NVDLAImage: bad version {ver}")
  off = 8
  n_cmds = struct.unpack_from("<I", data, off)[0]
  off += 4
  cmds = tuple(struct.unpack_from("<Q", data, off + i*8)[0] for i in range(n_cmds))
  off += n_cmds * 8
  n_relocs = struct.unpack_from("<I", data, off)[0]
  off += 4
  relocs = []
  for _ in range(n_relocs):
    ci, gs, ad, mk, fs = struct.unpack_from("<IIIII", data, off)
    off += 20
    relocs.append(NVDLAReloc(ci, gs, ad, mk, fs))
  # Task deserialization (simplified — reads back the fields we serialized)
  kind_len = data[off]
  off += 1
  kind = data[off:off+kind_len].decode("ascii")
  off += kind_len
  layout_len = struct.unpack_from("<I", data, off)[0]
  off += 4
  layout = tuple(struct.unpack_from("<" + "I"*layout_len, data, off))
  off += layout_len * 4
  out_slot, const_val = struct.unpack_from("<If", data, off)
  off += 8
  # Reconstruct task with defaults for fields not yet serialized
  task = NVDLATask(kind, layout, out_slot, (), const_val=const_val)
  return NVDLAImage(cmds, task, tuple(relocs))

# ---- the native_program hook ----
def build_native_program(sink: UOp) -> UOp|None:
  """Classify and build a PROGRAM(SINK, LINEAR(INS...)).
  Returns None if unsupported (tinygrad falls back to default renderer)."""
  task = plan_nvdla(sink)
  if isinstance(task, str): return None
  img = emit_nvdla(task)
  ins_args = [img] + list(img.cmds) + list(img.relocs)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=a) for a in ins_args))
  return UOp(Ops.PROGRAM, src=(sink, lin), arg=ProgramInfo.from_sink(sink))
