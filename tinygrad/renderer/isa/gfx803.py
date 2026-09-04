# flake8: noqa: E702
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import auto

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, greg
from tinygrad.uop import FastEnum
from tinygrad.uop.ops import GroupOp, Ops, PatternMatcher, UOp, UPat

class GFX803Ops(FastEnum):
  # DEFINE is a fixed-register placeholder and FLAT_ADDR expands to multiple
  # real GCN3 instructions. The remaining values map one-to-one to hardware.
  DEFINE = auto(); REG_BUFFER_META = auto(); REG_BUFFER = auto(); SCRATCH_BUFFER_META = auto(); SCRATCH_BUFFER_ADDR = auto()
  LDS_BUFFER = auto(); SCRATCH_SETUP = auto(); SCRATCH_TEARDOWN = auto()
  SCRATCH_LOAD_B32 = auto(); SCRATCH_LOAD_U16 = auto(); SCRATCH_LOAD_S16 = auto(); SCRATCH_LOAD_U8 = auto()
  SCRATCH_LOAD_S8 = auto(); SCRATCH_LOAD_B64 = auto(); SCRATCH_STORE_B32 = auto(); SCRATCH_STORE_B16 = auto()
  SCRATCH_STORE_B8 = auto(); SCRATCH_STORE_B64 = auto()
  S_LOAD_B64 = auto(); V_MOV_B32 = auto(); FLAT_ADDR = auto(); LDS_ADDR = auto()
  FLAT_LOAD_B32 = auto(); FLAT_LOAD_U16 = auto(); FLAT_LOAD_S16 = auto(); FLAT_LOAD_U8 = auto(); FLAT_LOAD_S8 = auto()
  GATED_FLAT_LOAD_B32 = auto(); GATED_FLAT_LOAD_U16 = auto(); GATED_FLAT_LOAD_S16 = auto()
  GATED_FLAT_LOAD_U8 = auto(); GATED_FLAT_LOAD_S8 = auto()
  FLAT_STORE_B32 = auto(); FLAT_STORE_B16 = auto(); FLAT_STORE_B8 = auto()
  GATED_FLAT_STORE_B32 = auto(); GATED_FLAT_STORE_B16 = auto(); GATED_FLAT_STORE_B8 = auto()
  DS_LOAD_B32 = auto(); DS_LOAD_U16 = auto(); DS_LOAD_S16 = auto(); DS_LOAD_U8 = auto(); DS_LOAD_S8 = auto()
  DS_STORE_B32 = auto(); DS_STORE_B16 = auto(); DS_STORE_B8 = auto(); REG_STORE_B32 = auto()
  V_ADD_U32 = auto(); V_LSHLREV_B32 = auto(); V_LSHRREV_B32 = auto(); V_ASHRREV_I32 = auto()
  V_AND_B32 = auto(); V_OR_B32 = auto(); V_XOR_B32 = auto(); V_MUL_LO_U32 = auto(); V_MUL_HI_U32 = auto()
  V_BFE_I32 = auto(); V_BFE_U32 = auto(); V_ADD_F32 = auto(); V_MUL_F32 = auto(); V_MAX_F32 = auto()
  V_MAX_I32 = auto(); V_MAX_U32 = auto(); V_CVT_F32_F16 = auto(); V_CVT_F16_F32 = auto()
  V_CVT_F32_I32 = auto(); V_CVT_F32_U32 = auto(); V_CVT_I32_F32 = auto(); V_CVT_U32_F32 = auto()
  V_CVT_I16_F16 = auto(); V_CVT_U16_F16 = auto(); V_CVT_F16_I16 = auto(); V_CVT_F16_U16 = auto()
  V_TRUNC = auto(); V_RCP_F32 = auto(); V_RCP_IFLAG_F32 = auto(); V_SQRT_F32 = auto(); V_RSQ_F32 = auto()
  V_EXP2_F32 = auto(); V_LOG2_F32 = auto(); V_SIN = auto(); V_CMPLT = auto(); V_CMPEQ = auto(); V_CMPNE = auto()
  V_CNDMASK_B32 = auto(); V_CMP_GT_U32 = auto(); S_WAITCNT = auto(); S_BARRIER = auto(); LABEL = auto()
  S_CBRANCH_VCCNZ = auto(); S_ENDPGM = auto()

@dataclass(frozen=True)
class GFX803Instruction:
  data: bytes
  text: str
  register_counts: tuple[int, int, int]
  lds_size: int = 0
  scratch_size: int = 0

  def to_bytes(self) -> bytes: return self.data
  def __str__(self) -> str: return self.text

# The old linear allocator in this branch does not model overlapping wide
# registers. Keep 64-bit addresses and scalar values in disjoint VGPR banks.
KERNARG_PTR = Register("s[0:1]", 0, size=8)
WGID = tuple(Register(f"s{i}", i, size=4) for i in range(2, 5))
WIID = tuple(Register(f"v{i}", i, size=4) for i in range(3))
SCRATCH_PTR = Register("v3", 3, size=4)
SGPR64 = tuple(Register(f"s[{i}:{i+1}]", i, size=8) for i in range(6, 32, 2))
# Broadcasts and masked indexing keep many 64-bit flat addresses live at once.
# Keep 40 address pairs disjoint from scalar values because this allocator does not model pair overlap.
VGPR64 = tuple(Register(f"v[{i}:{i+1}]", i, size=8) for i in range(4, 84, 2))
VGPR32 = tuple(Register(f"v{i}", i, size=4) for i in range(84, 224))
REG_BUFFER_BASE = 224
REG_BUFFER_COUNT = 256 - REG_BUFFER_BASE
SCRATCH_RSRC_INDEX = 36

def _fixed_reg(dtype:DType, reg:Register) -> UOp: return UOp(Ops.INS, dtype, arg=GFX803Ops.DEFINE, tag=(reg,))
def _const_u32(value:int) -> UOp: return UOp.const(dtypes.uint32, value)

def _reg_buffer_offset(ctx:IselContext, x:UOp) -> int:
  buffers = sorted((u for u in ctx.uses if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG), key=lambda u: int(u.arg.slot))
  return sum(int(u.src[0].arg) for u in buffers[:buffers.index(x)])

def _reg_buffer_scratch_layout(ctx:IselContext) -> tuple[dict[UOp, int], int]:
  offsets:dict[UOp, int] = {}
  cursor = 0
  for buf in sorted((u for u in ctx.uses if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG), key=lambda u: int(u.arg.slot)):
    align = min(buf.dtype.itemsize, 8)
    cursor += (-cursor) % align
    offsets[buf] = cursor
    cursor += int(buf.src[0].arg) * buf.dtype.itemsize
  return offsets, cursor

def _reg_buffers_use_scratch(ctx:IselContext) -> bool:
  return sum(int(u.src[0].arg) for u in ctx.uses if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG) > REG_BUFFER_COUNT

def _abi(ctx:IselContext, x:UOp) -> UOp|None:
  if x.tag is True: return None
  if x.op is Ops.SPECIAL:
    dim = int(x.arg[-1])
    if x.arg.startswith("lidx"): out = _fixed_reg(x.dtype, WIID[dim])
    elif x.arg.startswith("gidx"):
      out = UOp(Ops.INS, x.dtype, (_fixed_reg(x.dtype, WGID[dim]),), GFX803Ops.V_MOV_B32)
    else: raise NotImplementedError(f"gfx803 SPECIAL {x.arg!r} is not lowered yet")
    return out.after(x.rtag())

  offset = sum(8 if u.addrspace is not AddrSpace.ALU else u.dtype.itemsize for u in ctx.func_args[:ctx.func_args.index(x)])
  kernarg = _fixed_reg(dtypes.uint64, KERNARG_PTR)
  load = UOp(Ops.INS, dtypes.uint64, (kernarg, _const_u32(offset)), GFX803Ops.S_LOAD_B64)
  # Keep the PARAM reachable for ELF kernarg metadata and preserve its place in
  # the schedule. The True tag prevents instruction selection from revisiting it.
  # Scalar kernargs arrive in SGPRs, while the current integer selector expects
  # varying ALU values in VGPRs. Broadcast the low dword before using one in
  # index arithmetic; pointer arguments keep the full loaded pair.
  value = UOp(Ops.INS, x.dtype, (load,), GFX803Ops.V_MOV_B32) if x.addrspace is AddrSpace.ALU else load
  return value.after(x.rtag())

def _global_index(ctx:IselContext, x:UOp, base:UOp, idx:UOp) -> UOp|None:
  deps:tuple[UOp, ...] = ()
  root = base
  while root.op is Ops.AFTER:
    deps += root.src[1:]
    root = root.src[0]
  # INDEX can be selected before its BUFFER source. Make the fixed-register vs
  # private-scratch decision here too instead of relying on BUFFER selection.
  if root.op is Ops.BUFFER and root.addrspace is AddrSpace.REG and _reg_buffers_use_scratch(ctx):
    offsets, total = _reg_buffer_scratch_layout(ctx)
    root = UOp(Ops.INS, root.dtype, (UOp.const(dtypes.int32, offsets[root]), root.src[0], UOp.const(dtypes.int32, total)),
               GFX803Ops.SCRATCH_BUFFER_META, True)
  is_reg_buffer = (root.op is Ops.INS and root.arg is GFX803Ops.REG_BUFFER_META) or \
                  (root.op is Ops.BUFFER and root.addrspace is AddrSpace.REG)
  is_scratch_buffer = root.op is Ops.INS and root.arg is GFX803Ops.SCRATCH_BUFFER_META
  is_lds_buffer = (root.op is Ops.INS and root.arg is GFX803Ops.LDS_BUFFER) or \
                  (root.op is Ops.BUFFER and root.addrspace is AddrSpace.LOCAL)
  if is_scratch_buffer:
    if idx.op is not Ops.CONST: raise NotImplementedError("gfx803 indirect scratch-buffer indexing is not implemented")
    byte_offset, size = int(root.src[0].arg), int(root.src[1].arg)
    elem = int(idx.arg)
    if not 0 <= elem < size: raise RuntimeError(f"unsupported gfx803 scratch buffer shape/index: {size=}, {elem=}")
    ref = UOp(Ops.INS, x.dtype, (root, UOp.const(dtypes.int32, byte_offset + elem * x.dtype.itemsize)), GFX803Ops.SCRATCH_BUFFER_ADDR)
    return ref.after(*deps)
  if is_reg_buffer:
    if idx.op is not Ops.CONST: raise NotImplementedError("gfx803 indirect register-buffer indexing is not implemented")
    reg_offset = int(root.src[0].arg) if root.op is Ops.INS else _reg_buffer_offset(ctx, root)
    size = int(root.src[1].arg) if root.op is Ops.INS else int(root.src[0].arg)
    elem = int(idx.arg)
    if not 0 <= elem < size: raise RuntimeError(f"unsupported gfx803 register buffer shape/index: {size=}, {elem=}")
    reg_idx = REG_BUFFER_BASE + reg_offset + elem
    if reg_idx >= REG_BUFFER_BASE + REG_BUFFER_COUNT:
      raise RuntimeError(f"gfx803 register buffers exceed the reserved {REG_BUFFER_COUNT} VGPRs")
    ref = UOp(Ops.INS, x.dtype, (root,), GFX803Ops.REG_BUFFER, (Register(f"v{reg_idx}", reg_idx, size=4),))
    return ref.after(*deps)
  if is_lds_buffer:
    return UOp(Ops.INS, dtypes.uint32, (idx, UOp.const(dtypes.int32, x.dtype.itemsize.bit_length() - 1), root, *deps), GFX803Ops.LDS_ADDR)
  if idx.op is Ops.CONST:
    byte_offset = int(idx.arg) * x.dtype.itemsize
    if not 0 <= byte_offset <= 0xffffffff: raise OverflowError(f"gfx803 global byte offset out of range: {byte_offset}")
    offset = _const_u32(byte_offset)
  else:
    shift = x.dtype.itemsize.bit_length() - 1
    offset = UOp(Ops.SHL, idx.dtype, (idx, UOp.const(idx.dtype, shift))) if shift else idx
  return UOp(Ops.INS, dtypes.uint64, (base, offset), GFX803Ops.FLAT_ADDR)

def _indexed_addrspace(addr:UOp) -> AddrSpace|None:
  root = addr
  while root.op is Ops.AFTER: root = root.src[0]
  if root.op is Ops.INS and root.arg is GFX803Ops.REG_BUFFER: return AddrSpace.REG
  if root.op is Ops.INS and root.arg is GFX803Ops.LDS_ADDR: return AddrSpace.LOCAL
  if root.op is not Ops.INDEX: return None
  root = root.src[0]
  while root.op is Ops.AFTER: root = root.src[0]
  if root.op is Ops.BUFFER: return root.addrspace
  if root.op is Ops.INS and root.arg is GFX803Ops.REG_BUFFER_META: return AddrSpace.REG
  if root.op is Ops.INS and root.arg is GFX803Ops.LDS_BUFFER: return AddrSpace.LOCAL
  return None

def _scratch_buffer_addr(addr:UOp) -> UOp|None:
  while addr.op is Ops.AFTER: addr = addr.src[0]
  return addr if addr.op is Ops.INS and addr.arg is GFX803Ops.SCRATCH_BUFFER_ADDR else None

def _scratch_buffer_offset(ctx:IselContext, addr:UOp) -> int|None:
  while addr.op is Ops.AFTER: addr = addr.src[0]
  if addr.op is Ops.INS and addr.arg is GFX803Ops.SCRATCH_BUFFER_ADDR: return int(addr.src[1].arg)
  if addr.op is not Ops.INDEX: return None
  root = addr.src[0]
  while root.op is Ops.AFTER: root = root.src[0]
  if root.op is Ops.INS and root.arg is GFX803Ops.SCRATCH_BUFFER_META:
    byte_offset, size = int(root.src[0].arg), int(root.src[1].arg)
  elif root.op is Ops.BUFFER and root.addrspace is AddrSpace.REG and _reg_buffers_use_scratch(ctx):
    offsets, _ = _reg_buffer_scratch_layout(ctx)
    byte_offset, size = offsets[root], int(root.src[0].arg)
  else: return None
  if addr.src[1].op is not Ops.CONST: raise NotImplementedError("gfx803 indirect scratch-buffer indexing is not implemented")
  elem = int(addr.src[1].arg)
  if not 0 <= elem < size: raise RuntimeError(f"unsupported gfx803 scratch buffer shape/index: {size=}, {elem=}")
  return byte_offset + elem * addr.dtype.itemsize

def _scratch_load_op(dtype:DType) -> GFX803Ops:
  if dtype.itemsize == 8: return GFX803Ops.SCRATCH_LOAD_B64
  if dtype is dtypes.int8: return GFX803Ops.SCRATCH_LOAD_S8
  if dtype.itemsize == 1: return GFX803Ops.SCRATCH_LOAD_U8
  if dtype is dtypes.int16: return GFX803Ops.SCRATCH_LOAD_S16
  if dtype.itemsize == 2: return GFX803Ops.SCRATCH_LOAD_U16
  return GFX803Ops.SCRATCH_LOAD_B32

def _scratch_store_op(dtype:DType) -> GFX803Ops:
  if dtype.itemsize == 8: return GFX803Ops.SCRATCH_STORE_B64
  if dtype.itemsize == 1: return GFX803Ops.SCRATCH_STORE_B8
  if dtype.itemsize == 2: return GFX803Ops.SCRATCH_STORE_B16
  return GFX803Ops.SCRATCH_STORE_B32

def _load(ctx:IselContext, x:UOp, addr:UOp, alt:UOp|None=None, gate:UOp|None=None) -> UOp|None:
  if x.dtype not in (dtypes.bool, *dtypes.int8s, *dtypes.int16s, *dtypes.int32s, dtypes.half, dtypes.float32):
    raise NotImplementedError(f"gfx803 load dtype {x.dtype} is not lowered yet")
  if (scratch_offset:=_scratch_buffer_offset(ctx, addr)) is not None:
    if gate is not None: raise NotImplementedError("gfx803 gated scratch-buffer loads are not implemented")
    return UOp(Ops.INS, x.dtype, (UOp.const(dtypes.int32, scratch_offset), addr), _scratch_load_op(x.dtype))
  if (addrspace:=_indexed_addrspace(addr)) is AddrSpace.REG:
    if gate is not None: raise NotImplementedError("gfx803 gated register loads are not implemented")
    return UOp(Ops.INS, x.dtype, (addr,), GFX803Ops.V_MOV_B32)
  if addrspace is AddrSpace.LOCAL:
    if gate is not None: raise NotImplementedError("gfx803 gated LDS loads are not implemented")
    op = GFX803Ops.DS_LOAD_U8 if x.dtype in (dtypes.bool, dtypes.uint8) else \
         GFX803Ops.DS_LOAD_S8 if x.dtype is dtypes.int8 else \
         GFX803Ops.DS_LOAD_U16 if x.dtype in (dtypes.half, dtypes.uint16) else \
         GFX803Ops.DS_LOAD_S16 if x.dtype is dtypes.int16 else GFX803Ops.DS_LOAD_B32
  else:
    plain, gated = (GFX803Ops.FLAT_LOAD_U8, GFX803Ops.GATED_FLAT_LOAD_U8) if x.dtype in (dtypes.bool, dtypes.uint8) else \
                   (GFX803Ops.FLAT_LOAD_S8, GFX803Ops.GATED_FLAT_LOAD_S8) if x.dtype is dtypes.int8 else \
                   (GFX803Ops.FLAT_LOAD_U16, GFX803Ops.GATED_FLAT_LOAD_U16) if x.dtype in (dtypes.half, dtypes.uint16) else \
                   (GFX803Ops.FLAT_LOAD_S16, GFX803Ops.GATED_FLAT_LOAD_S16) if x.dtype is dtypes.int16 else \
                   (GFX803Ops.FLAT_LOAD_B32, GFX803Ops.GATED_FLAT_LOAD_B32)
    op = gated if gate is not None else plain
  return UOp(Ops.INS, x.dtype, (addr, *((alt, gate) if gate is not None else ())), op)

def _store(ctx:IselContext, x:UOp, addr:UOp, value:UOp, gate:UOp|None=None) -> UOp|None:
  if value.dtype not in (dtypes.bool, *dtypes.int8s, *dtypes.int16s, *dtypes.int32s, dtypes.half, dtypes.float32):
    raise NotImplementedError(f"gfx803 store dtype {value.dtype} is not lowered yet")
  if value.op is Ops.CONST: value = UOp(Ops.INS, value.dtype, (value,), GFX803Ops.V_MOV_B32)
  if (scratch_offset:=_scratch_buffer_offset(ctx, addr)) is not None:
    if gate is not None: raise NotImplementedError("gfx803 gated scratch-buffer stores are not implemented")
    return UOp(Ops.INS, dtypes.void, (value, UOp.const(dtypes.int32, scratch_offset), addr), _scratch_store_op(value.dtype))
  if (addrspace:=_indexed_addrspace(addr)) is AddrSpace.REG:
    if gate is not None: raise NotImplementedError("gfx803 gated register stores are not implemented")
    return UOp(Ops.INS, dtypes.void, (addr, value), GFX803Ops.REG_STORE_B32)
  if addrspace is AddrSpace.LOCAL:
    if gate is not None: raise NotImplementedError("gfx803 gated LDS stores are not implemented")
    store_op = GFX803Ops.DS_STORE_B8 if value.dtype in (dtypes.bool, *dtypes.int8s) else \
               GFX803Ops.DS_STORE_B16 if value.dtype in (*dtypes.int16s, dtypes.half) else GFX803Ops.DS_STORE_B32
    return UOp(Ops.INS, dtypes.void, (addr, value), store_op)
  store_op = GFX803Ops.GATED_FLAT_STORE_B8 if gate is not None and value.dtype in (dtypes.bool, *dtypes.int8s) else \
             GFX803Ops.GATED_FLAT_STORE_B16 if gate is not None and value.dtype in (*dtypes.int16s, dtypes.half) else \
             GFX803Ops.GATED_FLAT_STORE_B32 if gate is not None else \
             GFX803Ops.FLAT_STORE_B8 if value.dtype in (dtypes.bool, *dtypes.int8s) else \
             GFX803Ops.FLAT_STORE_B16 if value.dtype in (*dtypes.int16s, dtypes.half) else GFX803Ops.FLAT_STORE_B32
  return UOp(Ops.INS, dtypes.void, (addr, value, *((gate,) if gate is not None else ())),
             store_op)

def _buffer(ctx:IselContext, x:UOp) -> UOp:
  if x.addrspace is AddrSpace.REG:
    offset = _reg_buffer_offset(ctx, x)
    if _reg_buffers_use_scratch(ctx):
      offsets, total = _reg_buffer_scratch_layout(ctx)
      return UOp(Ops.INS, x.dtype, (UOp.const(dtypes.int32, offsets[x]), x.src[0], UOp.const(dtypes.int32, total)),
                 GFX803Ops.SCRATCH_BUFFER_META, True)
    return UOp(Ops.INS, x.dtype, (UOp.const(dtypes.int32, offset), x.src[0]), GFX803Ops.REG_BUFFER_META, True)
  if x.addrspace is AddrSpace.LOCAL:
    return UOp(Ops.INS, x.dtype, (UOp.const(dtypes.int32, x.arg.slot), x.src[0]), GFX803Ops.LDS_BUFFER, True)
  raise RuntimeError(f"unexpected gfx803 buffer address space {x.addrspace}")

def _range(ctx:IselContext, x:UOp) -> UOp|None:
  return x.replace(tag=(ctx.vreg(VGPR32),)) if not isinstance(x.tag, tuple) else None

def _barrier(x:UOp) -> UOp: return UOp(Ops.INS, dtypes.void, x.src, GFX803Ops.S_BARRIER)

def _stack_adjust(x:UOp, base:UOp, size:UOp) -> UOp|None:
  if greg(base) != SCRATCH_PTR or size.op is not Ops.CONST: return None
  return UOp(Ops.INS, dtypes.void, (size,), GFX803Ops.SCRATCH_SETUP if x.op is Ops.SUB else GFX803Ops.SCRATCH_TEARDOWN)

def _alloc_vreg(ctx:IselContext, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void or x.arg in {GFX803Ops.REG_BUFFER_META, GFX803Ops.REG_BUFFER, GFX803Ops.SCRATCH_BUFFER_META,
                                        GFX803Ops.SCRATCH_BUFFER_ADDR, GFX803Ops.LDS_BUFFER} or \
     (isinstance(x.tag, tuple) and isinstance(x.tag[0], Register)): return None
  regs = SGPR64 if x.arg is GFX803Ops.S_LOAD_B64 else VGPR64 if x.arg is GFX803Ops.FLAT_ADDR else VGPR32
  return x.replace(tag=(ctx.vreg(regs),))

def _bfe(value:UOp, dtype:DType, width:int, signed:bool) -> UOp:
  return UOp(Ops.INS, dtype, (value, _const_u32(0), _const_u32(width)), GFX803Ops.V_BFE_I32 if signed else GFX803Ops.V_BFE_U32)

def _normalize_int(value:UOp, dtype:DType) -> UOp:
  return value if dtype is dtypes.bool or dtype.itemsize == 4 else _bfe(value, dtype, dtype.bitsize, dtype in dtypes.sints)

def _int_add(x:UOp, a:UOp, b:UOp) -> UOp:
  return _normalize_int(UOp(Ops.INS, x.dtype, (b, a) if b.op is Ops.CONST else (a, b), GFX803Ops.V_ADD_U32), x.dtype)

def _int_mul(x:UOp, a:UOp, b:UOp) -> UOp:
  const:UOp
  value:UOp
  if a.op is Ops.CONST: const, value = a, b
  elif b.op is Ops.CONST: const, value = b, a
  else: result = UOp(Ops.INS, x.dtype, (a, b), GFX803Ops.V_MUL_LO_U32)
  if a.op is Ops.CONST or b.op is Ops.CONST:
    c = int(const.arg)
    if c > 0 and c & (c - 1) == 0:
      result = UOp(Ops.INS, x.dtype, (UOp.const(x.dtype, c.bit_length() - 1), value), GFX803Ops.V_LSHLREV_B32)
    else:
      if not -16 <= c <= 64: const = UOp(Ops.INS, x.dtype, (const,), GFX803Ops.V_MOV_B32)
      result = UOp(Ops.INS, x.dtype, (value, const), GFX803Ops.V_MUL_LO_U32)
  return _normalize_int(result, x.dtype)

def _bitwise(x:UOp, a:UOp, b:UOp) -> UOp:
  if b.op is Ops.CONST: a, b = b, a
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  result = UOp(Ops.INS, x.dtype, (a, b), {Ops.AND:GFX803Ops.V_AND_B32, Ops.OR:GFX803Ops.V_OR_B32, Ops.XOR:GFX803Ops.V_XOR_B32}[x.op])
  return _normalize_int(result, x.dtype)

def _bitcast(x:UOp, value:UOp) -> UOp|None:
  if x.dtype.itemsize != value.dtype.itemsize or x.dtype.itemsize > 4: return None
  moved = UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_MOV_B32)
  return _normalize_int(moved, x.dtype) if x.dtype in (*dtypes.int8s, *dtypes.int16s) else moved

def _shift(x:UOp, value:UOp, shift:UOp) -> UOp:
  if value.op is Ops.CONST: value = UOp(Ops.INS, value.dtype, (value,), GFX803Ops.V_MOV_B32)
  op = GFX803Ops.V_LSHLREV_B32 if x.op is Ops.SHL else \
       GFX803Ops.V_ASHRREV_I32 if x.dtype in dtypes.sints else GFX803Ops.V_LSHRREV_B32
  return _normalize_int(UOp(Ops.INS, x.dtype, (shift, value), op), x.dtype)

def _isel_cast(dtype:DType, value:UOp) -> UOp:
  ret = _cast(UOp(Ops.CAST, dtype, (value,)), value)
  if ret is None: raise NotImplementedError(f"gfx803 internal cast from {value.dtype} to {dtype} is not lowered")
  return ret

def _uint_add(a:UOp, b:UOp) -> UOp:
  return _int_add(UOp(Ops.ADD, dtypes.uint32, (a, b)), a, b)

def _uint_mul(a:UOp, b:UOp) -> UOp:
  return _int_mul(UOp(Ops.MUL, dtypes.uint32, (a, b)), a, b)

def _uint_neg(value:UOp) -> UOp:
  return _uint_mul(value, UOp.const(dtypes.uint32, 0xffffffff))

def _uint_sub(a:UOp, b:UOp) -> UOp:
  return _uint_add(a, _uint_neg(b))

def _uint_divmod(dividend:UOp, divisor:UOp) -> tuple[UOp, UOp]:
  # This is the reciprocal estimate and two-correction sequence emitted by
  # LLVM for GCN udiv/urem. The multiply-high steps retain all 32 bits, unlike
  # converting the dividend itself to float.
  divisor_f = _isel_cast(dtypes.float32, divisor)
  reciprocal_f = UOp(Ops.INS, dtypes.float32, (divisor_f,), GFX803Ops.V_RCP_IFLAG_F32)
  scaled_f = _float_bin(UOp(Ops.MUL, dtypes.float32, (reciprocal_f, UOp.const(dtypes.float32, 4294966784.0))),
                        reciprocal_f, UOp.const(dtypes.float32, 4294966784.0), GFX803Ops.V_MUL_F32)
  reciprocal = _isel_cast(dtypes.uint32, scaled_f)
  correction = UOp(Ops.INS, dtypes.uint32, (reciprocal, _uint_mul(_uint_neg(divisor), reciprocal)), GFX803Ops.V_MUL_HI_U32)
  reciprocal = _uint_add(reciprocal, correction)
  quotient = UOp(Ops.INS, dtypes.uint32, (dividend, reciprocal), GFX803Ops.V_MUL_HI_U32)
  remainder = _uint_sub(dividend, _uint_mul(quotient, divisor))

  for _ in range(2):
    below = _cmp(UOp(Ops.CMPLT, dtypes.bool, (remainder, divisor)), remainder, divisor)
    next_quotient = _uint_add(quotient, UOp.const(dtypes.uint32, 1))
    next_remainder = _uint_sub(remainder, divisor)
    quotient = _where(UOp(Ops.WHERE, dtypes.uint32, (below, quotient, next_quotient)), below, quotient, next_quotient)
    remainder = _where(UOp(Ops.WHERE, dtypes.uint32, (below, remainder, next_remainder)), below, remainder, next_remainder)
  return quotient, remainder

def _int_divmod(x:UOp, a:UOp, b:UOp) -> UOp:
  unsigned = x.dtype in dtypes.uints
  dividend, divisor = _isel_cast(dtypes.uint32, a), _isel_cast(dtypes.uint32, b)
  if not unsigned:
    a_negative = _cmp(UOp(Ops.CMPLT, dtypes.bool, (a, UOp.const(x.dtype, 0))), a, UOp.const(x.dtype, 0))
    b_negative = _cmp(UOp(Ops.CMPLT, dtypes.bool, (b, UOp.const(x.dtype, 0))), b, UOp.const(x.dtype, 0))
    neg_dividend, neg_divisor = _uint_neg(dividend), _uint_neg(divisor)
    dividend = _where(UOp(Ops.WHERE, dtypes.uint32, (a_negative, neg_dividend, dividend)), a_negative, neg_dividend, dividend)
    divisor = _where(UOp(Ops.WHERE, dtypes.uint32, (b_negative, neg_divisor, divisor)), b_negative, neg_divisor, divisor)

  quotient, remainder = _uint_divmod(dividend, divisor)
  if unsigned: return quotient if x.op is Ops.CDIV else remainder

  if x.op is Ops.CDIV:
    negative = _bitwise(UOp(Ops.XOR, dtypes.bool, (a_negative, b_negative)), a_negative, b_negative)
    negative_quotient = _uint_neg(quotient)
    value = _where(UOp(Ops.WHERE, dtypes.uint32, (negative, negative_quotient, quotient)), negative, negative_quotient, quotient)
  else:
    negative_remainder = _uint_neg(remainder)
    value = _where(UOp(Ops.WHERE, dtypes.uint32, (a_negative, negative_remainder, remainder)), a_negative, negative_remainder, remainder)
  return _isel_cast(x.dtype, value)

def _float_unary(x:UOp, value:UOp) -> UOp:
  if x.op is Ops.SIN:
    # V_SIN consumes turns, not radians, and has a finite input interval. Do
    # range reduction in float32 so all finite half inputs stay in range.
    wide = UOp(Ops.INS, dtypes.float32, (value,), GFX803Ops.V_CVT_F32_F16) if x.dtype is dtypes.half else value
    turns = _float_bin(UOp(Ops.MUL, dtypes.float32), wide, UOp.const(dtypes.float32, 1 / (2 * 3.141592653589793)), GFX803Ops.V_MUL_F32)
    whole = UOp(Ops.INS, dtypes.float32, (turns,), GFX803Ops.V_TRUNC)
    negative_whole = _float_bin(UOp(Ops.MUL, dtypes.float32), UOp.const(dtypes.float32, -1), whole, GFX803Ops.V_MUL_F32)
    reduced = _float_bin(UOp(Ops.ADD, dtypes.float32), turns, negative_whole, GFX803Ops.V_ADD_F32)
    result = UOp(Ops.INS, dtypes.float32, (reduced,), GFX803Ops.V_SIN)
    return UOp(Ops.INS, dtypes.half, (result,), GFX803Ops.V_CVT_F16_F32) if x.dtype is dtypes.half else result
  op = {Ops.RECIPROCAL:GFX803Ops.V_RCP_F32, Ops.SQRT:GFX803Ops.V_SQRT_F32,
        Ops.EXP2:GFX803Ops.V_EXP2_F32, Ops.LOG2:GFX803Ops.V_LOG2_F32, Ops.TRUNC:GFX803Ops.V_TRUNC}[x.op]
  # asinh/acosh lower to log2(x + sqrt(x*x +/- 1)). Keep that composite
  # argument wide so finite half inputs cannot overflow in the square.
  if x.dtype is dtypes.half and x.op is Ops.LOG2 and value.op is Ops.ADD:
    result = UOp(Ops.INS, dtypes.float32, (_half_to_float(value),), op)
    return UOp(Ops.INS, dtypes.half, (result,), GFX803Ops.V_CVT_F16_F32)
  return UOp(Ops.INS, x.dtype, (value,), op)

def _rsqrt(x:UOp, value:UOp) -> UOp:
  return UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_RSQ_F32)

def _half_to_float(value:UOp) -> UOp:
  if value.dtype is dtypes.float32: return value
  if value.op in {Ops.ADD, Ops.MUL, Ops.MAX}:
    return UOp(value.op, dtypes.float32, tuple(_half_to_float(src) for src in value.src), value.arg)
  if value.op in {Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.TRUNC}:
    return UOp(value.op, dtypes.float32, (_half_to_float(value.src[0]),), value.arg)
  if value.op is Ops.WHERE:
    return UOp(Ops.WHERE, dtypes.float32, (value.src[0], _half_to_float(value.src[1]), _half_to_float(value.src[2])), value.arg)
  if value.op is Ops.CONST: value = UOp(Ops.INS, dtypes.half, (value,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, dtypes.float32, (value,), GFX803Ops.V_CVT_F32_F16)

def _float_bin(x:UOp, a:UOp, b:UOp, ins:GFX803Ops) -> UOp:
  # a / b arrives as a * reciprocal(b). Keeping the reciprocal in half can
  # overflow even when the quotient is representable, so fuse it in float32.
  if x.dtype is dtypes.half and ins is GFX803Ops.V_MUL_F32:
    def is_inverse(v:UOp) -> bool:
      return v.op is Ops.RECIPROCAL or (v.op is Ops.INS and v.arg in {GFX803Ops.V_RCP_F32, GFX803Ops.V_RSQ_F32})
    inverse = b if is_inverse(b) else a if is_inverse(a) else None
    if inverse is not None:
      numerator = a if inverse is b else b
      inverse_op = GFX803Ops.V_RSQ_F32 if inverse.op is Ops.INS and inverse.arg is GFX803Ops.V_RSQ_F32 else GFX803Ops.V_RCP_F32
      wide_numerator, wide_denominator = _half_to_float(numerator), _half_to_float(inverse.src[0])
      wide_inverse = UOp(Ops.INS, dtypes.float32, (wide_denominator,), inverse_op)
      quotient = UOp(Ops.INS, dtypes.float32, (wide_numerator, wide_inverse), GFX803Ops.V_MUL_F32)
      return UOp(Ops.INS, dtypes.half, (quotient,), GFX803Ops.V_CVT_F16_F32)
  if b.op is Ops.CONST: a, b = b, a
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (a, b), ins)

def _int_max(x:UOp, a:UOp, b:UOp) -> UOp:
  if b.op is Ops.CONST: a, b = b, a
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (a, b), GFX803Ops.V_MAX_I32 if x.dtype in dtypes.sints else GFX803Ops.V_MAX_U32)

def _cmp(x:UOp, a:UOp, b:UOp) -> UOp:
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (a, b), {Ops.CMPLT:GFX803Ops.V_CMPLT, Ops.CMPEQ:GFX803Ops.V_CMPEQ, Ops.CMPNE:GFX803Ops.V_CMPNE}[x.op])

def _where(x:UOp, cond:UOp, true_value:UOp, false_value:UOp) -> UOp:
  if true_value.op is Ops.CONST: true_value = UOp(Ops.INS, true_value.dtype, (true_value,), GFX803Ops.V_MOV_B32)
  if false_value.op is Ops.CONST: false_value = UOp(Ops.INS, false_value.dtype, (false_value,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (cond, true_value, false_value), GFX803Ops.V_CNDMASK_B32)

def _cast(x:UOp, value:UOp) -> UOp|None:
  if x.dtype is dtypes.bool:
    zero = UOp(Ops.INS, value.dtype, (UOp.const(value.dtype, 0),), GFX803Ops.V_MOV_B32)
    return UOp(Ops.INS, x.dtype, (value, zero), GFX803Ops.V_CMPNE)
  if value.dtype is dtypes.bool and x.dtype in (*dtypes.int8s, *dtypes.int16s, *dtypes.int32s):
    return UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_MOV_B32)
  if value.dtype in (*dtypes.int8s, *dtypes.int16s, *dtypes.int32s) and x.dtype in (*dtypes.int8s, *dtypes.int16s, *dtypes.int32s):
    if value.dtype.itemsize == x.dtype.itemsize == 4: return UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_MOV_B32)
    widening = x.dtype.itemsize > value.dtype.itemsize
    if not widening: return _bfe(value, x.dtype, x.dtype.bitsize, x.dtype in dtypes.sints)
    widened = _bfe(value, x.dtype, value.dtype.bitsize, value.dtype in dtypes.sints)
    # A negative signed source widened into a still-narrow unsigned destination wraps at the destination width.
    return _bfe(widened, x.dtype, x.dtype.bitsize, False) \
      if value.dtype in dtypes.sints and x.dtype in dtypes.uints and x.dtype.itemsize < 4 else widened
  if value.dtype is dtypes.half and x.dtype is dtypes.float32:
    return UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_CVT_F32_F16)
  if value.dtype is dtypes.float32 and x.dtype is dtypes.half:
    return UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_CVT_F16_F32)
  if x.dtype is dtypes.float32 and value.dtype in (dtypes.bool, dtypes.int32, dtypes.uint32):
    op = GFX803Ops.V_CVT_F32_I32 if value.dtype is dtypes.int32 else GFX803Ops.V_CVT_F32_U32
    return UOp(Ops.INS, x.dtype, (value,), op)
  if value.dtype is dtypes.float32 and x.dtype in (*dtypes.int8s, *dtypes.int16s, *dtypes.int32s):
    wide_dtype = dtypes.int32 if x.dtype in dtypes.sints else dtypes.uint32
    op = GFX803Ops.V_CVT_I32_F32 if wide_dtype is dtypes.int32 else GFX803Ops.V_CVT_U32_F32
    wide = UOp(Ops.INS, wide_dtype, (value,), op)
    return wide if x.dtype is wide_dtype else _bfe(wide, x.dtype, x.dtype.bitsize, x.dtype in dtypes.sints)
  if value.dtype is dtypes.half and x.dtype in (*dtypes.int8s, *dtypes.int16s):
    wide_dtype = dtypes.int16 if x.dtype in dtypes.sints else dtypes.uint16
    op = GFX803Ops.V_CVT_I16_F16 if wide_dtype is dtypes.int16 else GFX803Ops.V_CVT_U16_F16
    wide = UOp(Ops.INS, wide_dtype, (value,), op)
    return _normalize_int(wide, x.dtype)
  if x.dtype is dtypes.float32 and value.dtype in (*dtypes.int8s, *dtypes.int16s):
    wide_dtype = dtypes.int32 if value.dtype in dtypes.sints else dtypes.uint32
    wide = _bfe(value, wide_dtype, value.dtype.bitsize, value.dtype in dtypes.sints)
    op = GFX803Ops.V_CVT_F32_I32 if wide_dtype is dtypes.int32 else GFX803Ops.V_CVT_F32_U32
    return UOp(Ops.INS, x.dtype, (wide,), op)
  if x.dtype is dtypes.half and value.dtype in (*dtypes.int8s, *dtypes.int16s):
    op = GFX803Ops.V_CVT_F16_I16 if value.dtype in dtypes.sints else GFX803Ops.V_CVT_F16_U16
    return UOp(Ops.INS, x.dtype, (value,), op)
  if x.dtype is dtypes.half and value.dtype in (dtypes.bool, dtypes.int32, dtypes.uint32):
    op = GFX803Ops.V_CVT_F32_I32 if value.dtype is dtypes.int32 else GFX803Ops.V_CVT_F32_U32
    widened = UOp(Ops.INS, dtypes.float32, (value,), op)
    return UOp(Ops.INS, x.dtype, (widened,), GFX803Ops.V_CVT_F16_F32)
  if value.dtype is dtypes.half and x.dtype in (dtypes.int32, dtypes.uint32):
    widened = UOp(Ops.INS, dtypes.float32, (value,), GFX803Ops.V_CVT_F32_F16)
    op = GFX803Ops.V_CVT_I32_F32 if x.dtype is dtypes.int32 else GFX803Ops.V_CVT_U32_F32
    return UOp(Ops.INS, x.dtype, (widened,), op)
  return None

def _unsupported_elementwise(x:UOp):
  raise NotImplementedError(f"gfx803 {x.op.name} {x.dtype} from {tuple(s.dtype for s in x.src)} is not lowered yet")

def _is_const(x:UOp, value:float) -> bool:
  return x.op is Ops.CONST and x.arg == value

def _relu_input(x:UOp) -> UOp|None:
  if x.op is not Ops.WHERE or len(x.src) != 3: return None
  cond, true_value, false_value = x.src
  if cond.op is not Ops.CMPLT or len(cond.src) != 2: return None
  return true_value if cond.src[1] is true_value and _is_const(cond.src[0], 0) and _is_const(false_value, 0) else None

def _add_without_const(x:UOp, value:float) -> UOp|None:
  if x.op is not Ops.ADD or len(x.src) != 2: return None
  if _is_const(x.src[0], value): return x.src[1]
  if _is_const(x.src[1], value): return x.src[0]
  return None

def _stable_hardsigmoid(x:UOp) -> UOp|None:
  # hardsigmoid is optimized into relu(y) - relu(y-1). Subtracting two
  # large half values loses the saturated 1, so recover the equivalent clamp.
  for positive, negative in (x.src, x.src[::-1]):
    y = _relu_input(positive)
    if y is None or negative.op is not Ops.MUL: continue
    if _is_const(negative.src[0], -1): negated = negative.src[1]
    elif _is_const(negative.src[1], -1): negated = negative.src[0]
    else: continue
    z = _relu_input(negated)
    if z is None: continue
    scaled_y, scaled_z = _add_without_const(y, 0.5), _add_without_const(z, -0.5)
    if scaled_y is None or scaled_y is not scaled_z: continue
    zero, one = UOp.const(x.dtype, 0), UOp.const(x.dtype, 1)
    below = UOp(Ops.CMPLT, dtypes.bool, (y, zero))
    above = UOp(Ops.CMPLT, dtypes.bool, (one, y))
    return UOp(Ops.WHERE, x.dtype, (below, zero, UOp(Ops.WHERE, x.dtype, (above, one, y))))
  return None

def _widen_half_product(a:UOp, b:UOp) -> UOp:
  # Half reductions already accumulate in float32. Widen before multiplying
  # too, matching the product precision used by GPU dot/conv contracts.
  return UOp(Ops.MUL, dtypes.float32, (_half_to_float(a), _half_to_float(b)))

def _widen_half_bias_add(a:UOp, b:UOp) -> UOp|None:
  # Conv bias arrives after a premature float32-to-half cast of the reduction.
  # Fold it into the wide accumulator and round only the final sum.
  rounded, bias = (a, b) if a.op is Ops.CAST and a.src[0].dtype is dtypes.float32 else (b, a)
  if rounded.op is not Ops.CAST or rounded.dtype is not dtypes.half or rounded.src[0].dtype is not dtypes.float32: return None
  wide_sum = UOp(Ops.ADD, dtypes.float32, (rounded.src[0], _half_to_float(bias)))
  return UOp(Ops.CAST, dtypes.half, (wide_sum,), dtypes.half)

pre_isel_matcher = PatternMatcher([
  (UPat(Ops.RECIPROCAL, (dtypes.half, dtypes.float32),
        src=(UPat(Ops.SQRT, src=(UPat.var("value"),)),), name="x"), _rsqrt),
  (UPat(Ops.CAST, dtypes.float32, src=(UPat(Ops.MUL, dtypes.half, src=(UPat.var("a"), UPat.var("b"))),)), _widen_half_product),
  (UPat(Ops.ADD, dtypes.half, src=(UPat.var("a"), UPat.var("b"))), _widen_half_bias_add),
  (UPat(Ops.ADD, dtypes.half, name="x"), _stable_hardsigmoid),
])
isel_matcher = PatternMatcher([
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), _abi),
  (UPat(Ops.BUFFER, name="x"), _buffer),
  (UPat(Ops.RANGE, name="x"), _range),
  (UPat(Ops.BARRIER, name="x"), _barrier),
  (UPat((Ops.ADD, Ops.SUB), src=(UPat.var("base"), UPat.var("size")), name="x"), _stack_adjust),
  (UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("idx")), name="x"), _global_index),
  (UPat(Ops.LOAD, src=(UPat.var("addr"), UPat.var("alt"), UPat.var("gate")), name="x"), _load),
  (UPat(Ops.LOAD, src=(UPat.var("addr"),), name="x"), _load),
  (UPat(Ops.ADD, dtypes.int8s+dtypes.int16s+dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_add),
  (UPat(Ops.MUL, dtypes.int8s+dtypes.int16s+dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_mul),
  (UPat((Ops.CDIV, Ops.CMOD), dtypes.int8s+dtypes.int16s+dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_divmod),
  (UPat((Ops.SHL, Ops.SHR), dtypes.int8s+dtypes.int16s+dtypes.int32s, src=(UPat.var("value"), UPat.var("shift")), name="x"), _shift),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), (dtypes.bool, *dtypes.int8s, *dtypes.int16s, *dtypes.int32s),
        src=(UPat.var("a"), UPat.var("b")), name="x"), _bitwise),
  (UPat(Ops.BITCAST, src=(UPat.var("value"),), name="x"), _bitcast),
  (UPat((Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.TRUNC, Ops.SIN), (dtypes.half, dtypes.float32),
        src=(UPat.var("value"),), name="x"), _float_unary),
  ((UPat.var("a", (dtypes.half, dtypes.float32)) + UPat.var("b", (dtypes.half, dtypes.float32))).named("x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_ADD_F32)),
  (UPat(Ops.MUL, (dtypes.half, dtypes.float32), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_MUL_F32)),
  (UPat(Ops.MAX, (dtypes.half, dtypes.float32), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_MAX_F32)),
  (UPat(Ops.MAX, dtypes.int8s+dtypes.int16s+dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_max),
  (UPat((Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE), dtypes.bool, src=(UPat.var("a"), UPat.var("b")), name="x"), _cmp),
  (UPat(Ops.WHERE, src=(UPat.var("cond"), UPat.var("true_value"), UPat.var("false_value")), name="x"), _where),
  (UPat(Ops.CAST, src=(UPat.var("value"),), name="x"), _cast),
  (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("value"), UPat.var("gate")), name="x"), _store),
  (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("value")), name="x"), _store),
  (UPat(GroupOp.Elementwise, name="x"), _unsupported_elementwise),
  (UPat(Ops.INS, name="x"), _alloc_vreg),
])

def _finish(x:UOp) -> tuple[UOp, list[UOp]]:
  return x, [UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_ENDPGM)]

def _lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  label = ".LOOP_" + "_".join(str(v) for v in x.arg[:-1])
  acc = UOp(Ops.INS, x.dtype, (UOp.const(x.dtype, 0),), GFX803Ops.V_MOV_B32, x.tag)
  ctx.loop_label[acc], ctx.locals[acc] = label, x.src[0]
  return acc, [acc, UOp(Ops.INS, dtypes.void, arg=GFX803Ops.LABEL, tag=label)]

def _lower_end(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  acc = x.src[-1]
  inc = UOp(Ops.INS, acc.dtype, (UOp.const(acc.dtype, 1), acc), GFX803Ops.V_ADD_U32, acc.tag)
  cmp = UOp(Ops.INS, dtypes.void, (ctx.locals[acc], inc), GFX803Ops.V_CMP_GT_U32)
  branch = UOp(Ops.INS, dtypes.void, (cmp,), GFX803Ops.S_CBRANCH_VCCNZ, ctx.loop_label[acc])
  return inc, [inc, cmp, branch]

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), _finish),
  (UPat(Ops.RANGE, name="x"), _lower_range),
  (UPat(Ops.END, name="x"), _lower_end),
  (UPat(GroupOp.All - {Ops.INS}, name="x"), lambda x: (x, [])),
])

def _physical_reg(x:UOp) -> Register:
  reg = greg(x)
  if not isinstance(reg, Register): raise RuntimeError(f"expected physical register for {x.op}, got {reg!r}")
  return reg

def _register_counts(x:UOp) -> tuple[int, int, int]:
  vgprs = sgprs = 0
  for operand in (x, *x.src):
    reg = greg(operand)
    if not isinstance(reg, Register): continue
    count = reg.index + max(1, reg.size // 4)
    if reg.name.startswith("v"): vgprs = max(vgprs, count)
    elif reg.name.startswith("s"): sgprs = max(sgprs, count)
  return vgprs, sgprs, 0

def _word(value:int) -> bytes: return struct.pack("<I", value & 0xffffffff)

def _vop1(op:int, dst:int, src0:int) -> int:
  return 0x7e000000 | ((dst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)

def _vop2(op:int, dst:int, src0:int, src1:int) -> int:
  return ((op & 0x3f) << 25) | ((dst & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)

def _vop3(op:int, dst:int, src0:int, src1:int) -> tuple[int, int]:
  return 0xd0000000 | ((op & 0x3ff) << 16) | (dst & 0xff), (src0 & 0x1ff) | ((src1 & 0x1ff) << 9)

def _vop3_3(op:int, dst:int, src0:int, src1:int, src2:int) -> tuple[int, int]:
  return _vop3(op, dst, src0, src1)[0], (src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((src2 & 0x1ff) << 18)

def _vopc(op:int, src0:int, src1:int) -> int:
  return 0x7c000000 | ((op & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)

def _mubuf(op:int, offset:int, vdata:int, srsrc:int=SCRATCH_RSRC_INDEX) -> bytes:
  if not 0 <= offset <= 0xfff: raise OverflowError(f"gfx803 scratch byte offset out of range: {offset}")
  if srsrc % 4: raise ValueError(f"gfx803 buffer resource must start on four SGPRs: s{srsrc}")
  return _word(0xe0000000 | ((op & 0x7f) << 18) | offset) + _word(0x80000000 | ((srsrc // 4) << 16) | (vdata << 8))

def _half_bits(value:float) -> int:
  try: return struct.unpack("<H", struct.pack("<e", value))[0]
  except OverflowError: return 0x7c00 if value > 0 else 0xfc00

def _src0(x:UOp) -> tuple[int, tuple[int, ...]]:
  if x.op is Ops.CONST:
    if dtypes.is_float(x.dtype):
      if x.dtype is dtypes.half:
        value = _half_bits(float(x.arg))
        inline = {0x0000:128, 0x3800:240, 0xb800:241, 0x3c00:242, 0xbc00:243,
                  0x4000:244, 0xc000:245, 0x4400:246, 0xc400:247, 0x3118:248}
        return (inline[value], ()) if value in inline else (255, (value,))
      value = struct.unpack("<I", struct.pack("<f", float(x.arg)))[0]
      inline = {0x00000000:128, 0x3f000000:240, 0xbf000000:241, 0x3f800000:242, 0xbf800000:243,
                0x40000000:244, 0xc0000000:245, 0x40800000:246, 0xc0800000:247, 0x3e22f983:248}
      return (inline[value], ()) if value in inline else (255, (value,))
    value = int(x.arg)
    if 0 <= value <= 64: return 128 + value, ()
    if -16 <= value < 0: return 192 - value, ()
    # Unsigned UOps keep their normalized 32-bit value, but GCN's inline
    # negative constants are bit-identical and avoid an extra literal dword.
    if 0xfffffff0 <= value <= 0xffffffff: return 192 - (value - (1 << 32)), ()
    return 255, (value & 0xffffffff,)
  reg = _physical_reg(x)
  return (256 + reg.index if reg.name.startswith("v") else reg.index), ()

def _raw_src0(x:UOp) -> tuple[int, tuple[int, ...], str]:
  if x.op is Ops.CONST and x.dtype is dtypes.half:
    value = _half_bits(float(x.arg))
    src, literal = _src0(UOp.const(dtypes.uint32, value))
    return src, literal, str(value) if not literal else f"{value:#x}"
  src, literal = _src0(x)
  return src, literal, str(x.arg) if x.op is Ops.CONST else _physical_reg(x).name

_VOP2_ENCODINGS:dict[tuple[GFX803Ops, DType|None], tuple[int, str]] = {
  (GFX803Ops.V_ADD_U32, None):(0x19, "v_add_u32_e32"), (GFX803Ops.V_LSHLREV_B32, None):(0x12, "v_lshlrev_b32_e32"),
  (GFX803Ops.V_LSHRREV_B32, None):(0x10, "v_lshrrev_b32_e32"), (GFX803Ops.V_ASHRREV_I32, None):(0x11, "v_ashrrev_i32_e32"),
  (GFX803Ops.V_AND_B32, None):(0x13, "v_and_b32_e32"), (GFX803Ops.V_OR_B32, None):(0x14, "v_or_b32_e32"),
  (GFX803Ops.V_XOR_B32, None):(0x15, "v_xor_b32_e32"), (GFX803Ops.V_MAX_I32, None):(0x0d, "v_max_i32_e32"),
  (GFX803Ops.V_MAX_U32, None):(0x0f, "v_max_u32_e32"),
  (GFX803Ops.V_ADD_F32, dtypes.half):(0x1f, "v_add_f16_e32"), (GFX803Ops.V_ADD_F32, dtypes.float32):(0x01, "v_add_f32_e32"),
  (GFX803Ops.V_MUL_F32, dtypes.half):(0x22, "v_mul_f16_e32"), (GFX803Ops.V_MUL_F32, dtypes.float32):(0x05, "v_mul_f32_e32"),
  (GFX803Ops.V_MAX_F32, dtypes.half):(0x2d, "v_max_f16_e32"), (GFX803Ops.V_MAX_F32, dtypes.float32):(0x0b, "v_max_f32_e32"),
}
_VOP2_OPS = {op for op, _ in _VOP2_ENCODINGS}

_VOP1_ENCODINGS:dict[tuple[GFX803Ops, DType|None], tuple[int, str]] = {
  (GFX803Ops.V_CVT_F32_F16, None):(0x0b, "v_cvt_f32_f16_e32"), (GFX803Ops.V_CVT_F16_F32, None):(0x0a, "v_cvt_f16_f32_e32"),
  (GFX803Ops.V_CVT_F32_I32, None):(0x05, "v_cvt_f32_i32_e32"), (GFX803Ops.V_CVT_F32_U32, None):(0x06, "v_cvt_f32_u32_e32"),
  (GFX803Ops.V_CVT_I32_F32, None):(0x08, "v_cvt_i32_f32_e32"), (GFX803Ops.V_CVT_U32_F32, None):(0x07, "v_cvt_u32_f32_e32"),
  (GFX803Ops.V_CVT_I16_F16, None):(0x3c, "v_cvt_i16_f16_e32"), (GFX803Ops.V_CVT_U16_F16, None):(0x3b, "v_cvt_u16_f16_e32"),
  (GFX803Ops.V_CVT_F16_I16, None):(0x3a, "v_cvt_f16_i16_e32"), (GFX803Ops.V_CVT_F16_U16, None):(0x39, "v_cvt_f16_u16_e32"),
  (GFX803Ops.V_TRUNC, dtypes.half):(0x46, "v_trunc_f16_e32"), (GFX803Ops.V_TRUNC, dtypes.float32):(0x1c, "v_trunc_f32_e32"),
  (GFX803Ops.V_RCP_F32, dtypes.half):(0x3d, "v_rcp_f16_e32"), (GFX803Ops.V_RCP_F32, dtypes.float32):(0x22, "v_rcp_f32_e32"),
  (GFX803Ops.V_RCP_IFLAG_F32, None):(0x23, "v_rcp_iflag_f32_e32"),
  (GFX803Ops.V_SQRT_F32, dtypes.half):(0x3e, "v_sqrt_f16_e32"), (GFX803Ops.V_SQRT_F32, dtypes.float32):(0x27, "v_sqrt_f32_e32"),
  (GFX803Ops.V_RSQ_F32, dtypes.half):(0x3f, "v_rsq_f16_e32"), (GFX803Ops.V_RSQ_F32, dtypes.float32):(0x24, "v_rsq_f32_e32"),
  (GFX803Ops.V_EXP2_F32, dtypes.half):(0x41, "v_exp_f16_e32"), (GFX803Ops.V_EXP2_F32, dtypes.float32):(0x20, "v_exp_f32_e32"),
  (GFX803Ops.V_LOG2_F32, dtypes.half):(0x40, "v_log_f16_e32"), (GFX803Ops.V_LOG2_F32, dtypes.float32):(0x21, "v_log_f32_e32"),
  (GFX803Ops.V_SIN, dtypes.half):(0x49, "v_sin_f16_e32"), (GFX803Ops.V_SIN, dtypes.float32):(0x29, "v_sin_f32_e32"),
}
_VOP1_OPS = {op for op, _ in _VOP1_ENCODINGS}

_SCRATCH_ENCODINGS:dict[GFX803Ops, tuple[int, str]] = {
  GFX803Ops.SCRATCH_LOAD_U8:(0x10, "buffer_load_ubyte"), GFX803Ops.SCRATCH_LOAD_S8:(0x11, "buffer_load_sbyte"),
  GFX803Ops.SCRATCH_LOAD_U16:(0x12, "buffer_load_ushort"), GFX803Ops.SCRATCH_LOAD_S16:(0x13, "buffer_load_sshort"),
  GFX803Ops.SCRATCH_LOAD_B32:(0x14, "buffer_load_dword"), GFX803Ops.SCRATCH_LOAD_B64:(0x14, "buffer_load_dword"),
  GFX803Ops.SCRATCH_STORE_B8:(0x18, "buffer_store_byte"), GFX803Ops.SCRATCH_STORE_B16:(0x1a, "buffer_store_short"),
  GFX803Ops.SCRATCH_STORE_B32:(0x1c, "buffer_store_dword"), GFX803Ops.SCRATCH_STORE_B64:(0x1c, "buffer_store_dword"),
}
_SCRATCH_LOADS = {op for op in _SCRATCH_ENCODINGS if "LOAD" in op.name}
_SCRATCH_STORES = set(_SCRATCH_ENCODINGS) - _SCRATCH_LOADS

_MEMORY_ENCODINGS:dict[GFX803Ops, tuple[int, str]] = {
  GFX803Ops.FLAT_LOAD_U8:(0xdc400000, "flat_load_ubyte"), GFX803Ops.FLAT_LOAD_S8:(0xdc440000, "flat_load_sbyte"),
  GFX803Ops.FLAT_LOAD_U16:(0xdc480000, "flat_load_ushort"), GFX803Ops.FLAT_LOAD_S16:(0xdc4c0000, "flat_load_sshort"),
  GFX803Ops.FLAT_LOAD_B32:(0xdc500000, "flat_load_dword"), GFX803Ops.FLAT_STORE_B8:(0xdc600000, "flat_store_byte"),
  GFX803Ops.FLAT_STORE_B16:(0xdc680000, "flat_store_short"), GFX803Ops.FLAT_STORE_B32:(0xdc700000, "flat_store_dword"),
  GFX803Ops.DS_LOAD_U8:(0xd8740000, "ds_read_u8"), GFX803Ops.DS_LOAD_S8:(0xd8720000, "ds_read_i8"),
  GFX803Ops.DS_LOAD_U16:(0xd8780000, "ds_read_u16"), GFX803Ops.DS_LOAD_S16:(0xd8760000, "ds_read_i16"),
  GFX803Ops.DS_LOAD_B32:(0xd86c0000, "ds_read_b32"), GFX803Ops.DS_STORE_B8:(0xd83c0000, "ds_write_b8"),
  GFX803Ops.DS_STORE_B16:(0xd83e0000, "ds_write_b16"), GFX803Ops.DS_STORE_B32:(0xd81a0000, "ds_write_b32"),
}
_GATED_MEMORY_OPS = {op for op in GFX803Ops if op.name.startswith("GATED_FLAT_")}
_ASYNC_MEMORY_OPS = {GFX803Ops.S_LOAD_B64, *_SCRATCH_ENCODINGS, *_MEMORY_ENCODINGS, *_GATED_MEMORY_OPS}

def _typed_encoding(encodings:dict[tuple[GFX803Ops, DType|None], tuple[int, str]], x:UOp) -> tuple[int, str]:
  return encodings.get((x.arg, x.dtype), encodings[(x.arg, None)]) if (x.arg, None) in encodings else encodings[(x.arg, x.dtype)]

def _encode_scratch(x:UOp, dst:Register|None) -> tuple[bytes, str]:
  loading = x.arg in _SCRATCH_LOADS
  reg, offset = (dst, x.src[0]) if loading else (_physical_reg(x.src[0]), x.src[1])
  assert reg is not None
  action = "fill" if loading else "spill"
  if not reg.name.startswith("v"): raise NotImplementedError(f"gfx803 scalar-register {action}s are not implemented")
  if offset.op is not Ops.CONST: raise RuntimeError(f"gfx803 scratch {action} offset must be constant")
  scratch_offset, (opcode, mnemonic) = int(offset.arg), _SCRATCH_ENCODINGS[x.arg]
  words = 2 if x.arg.name.endswith("_B64") else 1
  data = b"".join(_mubuf(opcode, scratch_offset + i*4, reg.index + i) for i in range(words))
  text = "\n".join(f"{mnemonic} v{reg.index+i}, off, s[{SCRATCH_RSRC_INDEX}:{SCRATCH_RSRC_INDEX+3}], 0" +
                   (f" offset:{scratch_offset+i*4}" if scratch_offset+i*4 else "") for i in range(words))
  return data, text

def _encode_memory(x:UOp, dst:Register|None) -> tuple[bytes, str]:
  opcode, mnemonic = _MEMORY_ENCODINGS[x.arg]
  addr, loading, flat = _physical_reg(x.src[0]), "LOAD" in x.arg.name, x.arg.name.startswith("FLAT_")
  if loading:
    assert dst is not None
    reg, shift = dst, 24
  else: reg, shift = _physical_reg(x.src[1]), 8
  address = f"v[{addr.index}:{addr.index+1}]" if flat else f"v{addr.index}"
  operands = f"v{reg.index}, {address}" if loading else f"{address}, v{reg.index}"
  return _word(opcode) + _word((reg.index << shift) | addr.index), f"{mnemonic} {operands}"

def _encode_gated_memory(x:UOp, dst:Register|None) -> tuple[bytes, str]:
  loading = "LOAD" in x.arg.name
  addr, gate = _physical_reg(x.src[0]), _physical_reg(x.src[-1])
  if not gate.name.startswith("v"): raise RuntimeError(f"gfx803 {'load' if loading else 'store'} gate must be a VGPR, got {gate}")
  opcode, mnemonic = _MEMORY_ENCODINGS[GFX803Ops[x.arg.name.removeprefix("GATED_")]]
  init_data, init_text = b"", []
  if loading:
    assert dst is not None
    reg, shift = dst, 24
    src, literal, src_text = _raw_src0(x.src[1])
    # Compare before initializing the destination: the allocator may reuse the dying gate VGPR for dst.
    init_data = _word(_vop1(0x01, dst.index, src)) + b"".join(_word(v) for v in literal)
    init_text = [f"v_mov_b32_e32 v{dst.index}, {src_text}"]
  else: reg, shift = _physical_reg(x.src[1]), 8
  data = (_word(_vopc(0xcd, 128, gate.index)) + init_data +
          b"".join(_word(w) for w in (0xbea0206a, opcode, (reg.index << shift) | addr.index, 0xbefe0120)))
  operands = f"v{reg.index}, v[{addr.index}:{addr.index+1}]" if loading else f"v[{addr.index}:{addr.index+1}], v{reg.index}"
  text = "\n".join([f"v_cmp_ne_u32_e32 vcc, 0, v{gate.index}", *init_text, "s_and_saveexec_b64 s[32:33], vcc",
                    f"{mnemonic} {operands}", "s_mov_b64 exec, s[32:33]"])
  return data, text

def _encode(x:UOp, branch_offset:int=0) -> GFX803Instruction:
  counts = _register_counts(x)
  dst = _physical_reg(x) if x.dtype is not dtypes.void and x.arg not in \
    {GFX803Ops.REG_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_ADDR, GFX803Ops.LDS_BUFFER} else None

  lds_size = scratch_size = 0
  if x.arg in {GFX803Ops.REG_BUFFER_META, GFX803Ops.REG_BUFFER, GFX803Ops.SCRATCH_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_ADDR}:
    data, text = b"", ""
  elif x.arg is GFX803Ops.LDS_BUFFER:
    lds_size = int(x.src[1].arg) * x.dtype.itemsize
    # GFX8 uses M0 as the LDS aperture bound. Its launch value is undefined,
    # so every kernel touching LDS must initialize it before the first DS op.
    data, text = _word(0xbefc00c1), "s_mov_b32 m0, -1"
  elif x.arg is GFX803Ops.SCRATCH_SETUP:
    if x.src[0].op is not Ops.CONST or (scratch_size:=int(x.src[0].arg)) <= 0:
      raise RuntimeError("gfx803 scratch frame size must be a positive constant")
    # Enabling scratch shifts the incoming ABI to s[0:3]=resource, s[4:5]=kernarg,
    # s[6:8]=WGIDs and s9=private-segment wave byte offset. Preserve the
    # renderer's normal body ABI and keep the resource above its scalar pool
    # and the s[32:33] EXEC-save pair used by gated memory operations.
    words = [0x80000900, 0x82018001, 0xbea40100, 0xbea60102, 0xbe800104, 0xbe820006, 0xbe830007, 0xbe840008]
    lines = ["s_add_u32 s0, s0, s9", "s_addc_u32 s1, s1, 0",
             "s_mov_b64 s[36:37], s[0:1]", "s_mov_b64 s[38:39], s[2:3]",
             "s_mov_b64 s[0:1], s[4:5]", "s_mov_b32 s2, s6", "s_mov_b32 s3, s7", "s_mov_b32 s4, s8"]
    data, text = b"".join(_word(w) for w in words), "\n".join(lines)
    counts = (counts[0], max(counts[1], SCRATCH_RSRC_INDEX + 4), counts[2])
  elif x.arg is GFX803Ops.SCRATCH_TEARDOWN:
    data, text = b"", ""
  elif x.arg in _SCRATCH_ENCODINGS:
    data, text = _encode_scratch(x, dst)
    counts = (counts[0], max(counts[1], SCRATCH_RSRC_INDEX + 4), counts[2])
  elif x.arg is GFX803Ops.S_LOAD_B64:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    if offset.op is not Ops.CONST: raise RuntimeError("s_load_dwordx2 offset must be constant")
    data = _word(0xc0060000 | (dst.index << 6) | (base.index >> 1)) + _word(int(offset.arg))
    text = f"s_load_dwordx2 s[{dst.index}:{dst.index+1}], s[{base.index}:{base.index+1}], {int(offset.arg):#x}"
  elif x.arg is GFX803Ops.V_MOV_B32:
    assert dst is not None
    src, literal, src_text = _raw_src0(x.src[0])
    data = _word(_vop1(0x01, dst.index, src)) + b"".join(_word(v) for v in literal)
    text = f"v_mov_b32_e32 v{dst.index}, {src_text}"
  elif x.arg is GFX803Ops.FLAT_ADDR:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    words = [0x7e000200 | (dst.index << 17) | base.index,
             0x7e000200 | ((dst.index + 1) << 17) | (base.index + 1)]
    lines = [f"v_mov_b32_e32 v{dst.index}, s{base.index}", f"v_mov_b32_e32 v{dst.index+1}, s{base.index+1}"]
    if offset.op is Ops.CONST:
      off = int(offset.arg)
      if off:
        src0, literal = _src0(offset)
        words.append(_vop2(0x19, dst.index, src0, dst.index))
        words.extend(literal)
        words.append(_vop2(0x1c, dst.index + 1, 128, dst.index + 1))
        lines += [f"v_add_u32_e32 v{dst.index}, vcc, {off:#x}, v{dst.index}",
                  f"v_addc_u32_e32 v{dst.index+1}, vcc, 0, v{dst.index+1}, vcc"]
    else:
      src0, literal = _src0(offset)
      words.append(_vop2(0x19, dst.index, src0, dst.index))
      words.extend(literal)
      words.append(_vop2(0x1c, dst.index + 1, 128, dst.index + 1))
      lines += [f"v_add_u32_e32 v{dst.index}, vcc, {_physical_reg(offset).name}, v{dst.index}",
                f"v_addc_u32_e32 v{dst.index+1}, vcc, 0, v{dst.index+1}, vcc"]
    data, text = b"".join(_word(w) for w in words), "\n".join(lines)
  elif x.arg is GFX803Ops.LDS_ADDR:
    assert dst is not None
    idx, shift = x.src[:2]
    if idx.op is Ops.CONST:
      byte_offset = int(idx.arg) << int(shift.arg)
      src, literal = _src0(UOp.const(dtypes.uint32, byte_offset))
      data = _word(0x7e000200 | (dst.index << 17) | src) + b"".join(_word(v) for v in literal)
      text = f"v_mov_b32_e32 v{dst.index}, {byte_offset}"
    else:
      idx_reg = _physical_reg(idx)
      if not idx_reg.name.startswith("v"): raise RuntimeError(f"LDS index must be a VGPR, got {idx_reg}")
      if int(shift.arg):
        data = _word(_vop2(0x12, dst.index, 128 + int(shift.arg), idx_reg.index))
        text = f"v_lshlrev_b32_e32 v{dst.index}, {int(shift.arg)}, v{idx_reg.index}"
      else:
        data = _word(0x7e000200 | (dst.index << 17) | 256 | idx_reg.index)
        text = f"v_mov_b32_e32 v{dst.index}, v{idx_reg.index}"
  elif x.arg in _MEMORY_ENCODINGS:
    data, text = _encode_memory(x, dst)
  elif x.arg in _GATED_MEMORY_OPS:
    data, text = _encode_gated_memory(x, dst)
    counts = (counts[0], max(counts[1], 34), counts[2])
  elif x.arg is GFX803Ops.REG_STORE_B32:
    addr = _physical_reg(x.src[0])
    src, literal, src_text = _raw_src0(x.src[1])
    data = _word(_vop1(0x01, addr.index, src)) + b"".join(_word(v) for v in literal)
    text = f"v_mov_b32_e32 v{addr.index}, {src_text}"
  elif x.arg in _VOP2_OPS:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    op, mnemonic = _typed_encoding(_VOP2_ENCODINGS, x)
    data = _word(_vop2(op, dst.index, src0, src1.index)) + b"".join(_word(v) for v in literal)
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    text = f"{mnemonic} v{dst.index}, {'vcc, ' if x.arg is GFX803Ops.V_ADD_U32 else ''}{src0_text}, v{src1.index}"
  elif x.arg in {GFX803Ops.V_MUL_LO_U32, GFX803Ops.V_MUL_HI_U32}:
    assert dst is not None
    src0_code, literal0 = _src0(x.src[0])
    src1_code, literal1 = _src0(x.src[1])
    if literal0 or literal1: raise RuntimeError(f"{x.arg} does not support literal constants on gfx803")
    opcode = 0x285 if x.arg is GFX803Ops.V_MUL_LO_U32 else 0x286
    mnemonic = "v_mul_lo_u32" if x.arg is GFX803Ops.V_MUL_LO_U32 else "v_mul_hi_u32"
    data = b"".join(_word(w) for w in _vop3(opcode, dst.index, src0_code, src1_code))
    operands = [str(s.arg) if s.op is Ops.CONST else _physical_reg(s).name for s in x.src]
    text = f"{mnemonic} v{dst.index}, {operands[0]}, {operands[1]}"
  elif x.arg in {GFX803Ops.V_BFE_I32, GFX803Ops.V_BFE_U32}:
    assert dst is not None
    srcs = [_src0(source) for source in x.src]
    if any(literal for _, literal in srcs): raise RuntimeError(f"{x.arg} does not support literal constants on gfx803")
    opcode, mnemonic = (0x1c9, "v_bfe_i32") if x.arg is GFX803Ops.V_BFE_I32 else (0x1c8, "v_bfe_u32")
    data = b"".join(_word(w) for w in _vop3_3(opcode, dst.index, *(source for source, _ in srcs)))
    operands = [str(source.arg) if source.op is Ops.CONST else _physical_reg(source).name for source in x.src]
    text = f"{mnemonic} v{dst.index}, {', '.join(operands)}"
  elif x.arg in _VOP1_OPS:
    assert dst is not None
    src_reg = _physical_reg(x.src[0])
    op, mnemonic = _typed_encoding(_VOP1_ENCODINGS, x)
    data, text = _word(_vop1(op, dst.index, 256 + src_reg.index)), f"{mnemonic} v{dst.index}, v{src_reg.index}"
  elif x.arg in {GFX803Ops.V_CMPLT, GFX803Ops.V_CMPEQ, GFX803Ops.V_CMPNE}:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    src_dtype = x.src[0].dtype
    kind = "f16" if src_dtype is dtypes.half else "f32" if src_dtype is dtypes.float32 else "i32" if src_dtype in dtypes.sints else "u32"
    cmp_code = {GFX803Ops.V_CMPLT:{"f16":0x21, "f32":0x41, "i32":0xc1, "u32":0xc9},
                GFX803Ops.V_CMPEQ:{"f16":0x22, "f32":0x42, "i32":0xc2, "u32":0xca},
                GFX803Ops.V_CMPNE:{"f16":0x2d, "f32":0x4d, "i32":0xc5, "u32":0xcd}}[x.arg][kind]
    suffix = kind
    # Materializing true into dst before the compare is not alias-safe: the
    # allocator may legally reuse a dying compare input for dst. VOP3 accepts
    # inline constants in both data positions, so only write dst after VCC is set.
    words = [_vopc(cmp_code, src0, src1.index), *literal, *_vop3_3(0x100, dst.index, 128, 129, 106)]
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    cmp_name = ({GFX803Ops.V_CMPLT:"lt", GFX803Ops.V_CMPEQ:"eq"}.get(x.arg) or
                ("neq" if src_dtype in (dtypes.half, dtypes.float32) else "ne"))
    text = (f"v_cmp_{cmp_name}_{suffix}_e32 vcc, {src0_text}, v{src1.index}\n"
            f"v_cndmask_b32_e64 v{dst.index}, 0, 1, vcc")
    data = b"".join(_word(w) for w in words)
  elif x.arg is GFX803Ops.V_CNDMASK_B32:
    assert dst is not None
    cond, true_value, false_value = map(_physical_reg, x.src)
    if not all(value.name.startswith("v") for value in (cond, true_value, false_value)):
      raise RuntimeError(f"v_cndmask inputs must be VGPRs, got {cond}, {true_value} and {false_value}")
    words = [_vopc(0xcd, 128, cond.index), _vop2(0x00, dst.index, 256 + false_value.index, true_value.index)]
    data = b"".join(_word(w) for w in words)
    text = (f"v_cmp_ne_u32_e32 vcc, 0, v{cond.index}\n"
            f"v_cndmask_b32_e32 v{dst.index}, v{false_value.index}, v{true_value.index}, vcc")
  elif x.arg is GFX803Ops.V_CMP_GT_U32:
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    data = _word(_vopc(0xcc, src0, src1.index)) + b"".join(_word(v) for v in literal)
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    text = f"v_cmp_gt_u32_e32 vcc, {src0_text}, v{src1.index}"
  elif x.arg is GFX803Ops.S_WAITCNT:
    data, text = _word(0xbf8c0000), "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"
  elif x.arg is GFX803Ops.S_BARRIER:
    data, text = _word(0xbf8a0000), "s_barrier"
  elif x.arg is GFX803Ops.LABEL:
    data, text = b"", f"{x.tag}:"
  elif x.arg is GFX803Ops.S_CBRANCH_VCCNZ:
    if not -0x8000 <= branch_offset <= 0x7fff: raise OverflowError(f"gfx803 branch is out of range: {branch_offset}")
    data, text = _word(0xbf870000 | (branch_offset & 0xffff)), f"s_cbranch_vccnz {x.tag}"
  elif x.arg is GFX803Ops.S_ENDPGM:
    data, text = _word(0xbf810000), "s_endpgm"
  else: raise RuntimeError(f"cannot encode gfx803 instruction {x.arg}")
  return GFX803Instruction(data, text, counts, lds_size, scratch_size)

class AMDASMRenderer(ISARenderer):
  device = "AMD"
  supports_float4 = False
  has_local, has_shared = True, False
  shared_max, global_max, global_prod_max = HIPRenderer.shared_max, HIPRenderer.global_max, HIPRenderer.global_prod_max
  pre_isel_matcher, isel_matcher = pre_isel_matcher, isel_matcher
  pre_regalloc_matcher = None
  post_regalloc_matcher = post_regalloc_matcher
  # Advertising the native bit operations also enables tinygrad's constant
  # integer divide/modulo strength reduction before instruction selection.
  code_for_op = {op:lambda: None for op in
                 (Ops.AND, Ops.SHL, Ops.SHR, Ops.SQRT, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.TRUNC, Ops.SIN)}

  def __init__(self, target:Target):
    if target.arch != "gfx803": raise RuntimeError(f"AMDASMRenderer only supports gfx803, got {target.arch}")
    super().__init__(target)

  def supported_dtypes(self):
    return {dtypes.bool, *dtypes.int8s, *dtypes.int16s, *dtypes.int32s, dtypes.half, dtypes.float32}
  def stack_pointer(self) -> UOp: return _fixed_reg(dtypes.uint32, SCRATCH_PTR)
  def spill(self, disp:UOp, x:UOp) -> UOp:
    reg = _physical_reg(x)
    if not reg.name.startswith("v"): raise NotImplementedError("gfx803 scalar-register spills are not implemented")
    op = _scratch_store_op(dtypes.uint64 if reg.size == 8 else x.dtype)
    return UOp(Ops.INS, dtypes.void, (x, disp), op)
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    if not reg.name.startswith("v"): raise NotImplementedError("gfx803 scalar-register fills are not implemented")
    op = _scratch_load_op(dtypes.uint64 if reg.size == 8 else x.dtype)
    return UOp(Ops.INS, x.dtype, (disp,), op, (reg,))
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), GFX803Ops.V_MOV_B32, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    return f".{function_name}:\n" + "\n".join(str(_encode(u)) for u in uops if u.arg is not GFX803Ops.DEFINE)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # Scalar and vector memory operations are asynchronous on GCN3. A full
    # wait after each access is conservative but gives the first backend path
    # unambiguous dependency ordering; later scheduling can coalesce waits.
    spill_size = max((int(u.src[0].arg) for u in lin.src if u.arg is GFX803Ops.SCRATCH_SETUP), default=0)
    buffer_size = max((int(u.src[2].arg) for u in lin.src if u.arg is GFX803Ops.SCRATCH_BUFFER_META), default=0)
    buffer_base = (spill_size + 7) // 8 * 8
    total_scratch = buffer_base + buffer_size
    normalized:list[UOp] = []
    has_setup = False
    for u in lin.src:
      if u.arg is GFX803Ops.SCRATCH_SETUP:
        u, has_setup = u.replace(src=(UOp.const(u.src[0].dtype, total_scratch),)), True
      elif u.arg in _SCRATCH_LOADS and len(u.src) > 1 and _scratch_buffer_addr(u.src[1]) is not None:
        u = u.replace(src=(UOp.const(dtypes.int32, buffer_base + int(u.src[0].arg)), *u.src[1:]))
      elif u.arg in _SCRATCH_STORES and len(u.src) > 2 and _scratch_buffer_addr(u.src[2]) is not None:
        u = u.replace(src=(u.src[0], UOp.const(dtypes.int32, buffer_base + int(u.src[1].arg)), *u.src[2:]))
      normalized.append(u)
    if total_scratch and not has_setup:
      normalized.insert(0, UOp(Ops.INS, dtypes.void, (UOp.const(dtypes.uint32, total_scratch),), GFX803Ops.SCRATCH_SETUP))

    ordered:list[UOp] = []
    for u in normalized:
      if u.arg is GFX803Ops.DEFINE: continue
      ordered.append(u)
      if u.arg in _ASYNC_MEMORY_OPS:
        ordered.append(UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_WAITCNT))
    labels:dict[str, int] = {}
    pc = 0
    for u in ordered:
      if u.arg is GFX803Ops.LABEL: labels[str(u.tag)] = pc
      else: pc += len(_encode(u).data)
    encoded:list[UOp] = []
    pc = 0
    for u in ordered:
      if u.arg is GFX803Ops.LABEL: continue
      branch_offset = 0
      if u.arg is GFX803Ops.S_CBRANCH_VCCNZ:
        target = labels[str(u.tag)]
        if (target - pc - 4) % 4: raise RuntimeError("gfx803 branch target is not dword aligned")
        branch_offset = (target - pc - 4) // 4
      inst = _encode(u, branch_offset)
      encoded.append(u.replace(arg=inst))
      pc += len(inst.data)
    return assemble_linear(prg, lin.replace(src=tuple(encoded)), self.target.arch)
