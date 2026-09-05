# flake8: noqa: E702
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import auto
from typing import cast

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, greg
from tinygrad.uop import FastEnum
from tinygrad.uop.ops import GroupOp, Ops, PatternMatcher, UOp, UPat

class GFX803Ops(FastEnum):
  # DEFINE is a fixed-register placeholder and FLAT_ADDR expands to multiple real GCN3 instructions. The remaining values map one-to-one to
  # hardware.
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
  register_counts: tuple[int, int, int]
  lds_size: int = 0
  scratch_size: int = 0

  def to_bytes(self) -> bytes: return self.data

# The old linear allocator in this branch does not model overlapping wide registers. Keep 64-bit addresses and scalar values in disjoint VGPR
# banks.
KERNARG_PTR = Register("s[0:1]", 0, size=8)
WGID = tuple(Register(f"s{i}", i, size=4) for i in range(2, 5))
WIID = tuple(Register(f"v{i}", i, size=4) for i in range(3))
SCRATCH_PTR = Register("v3", 3, size=4)
SGPR64 = tuple(Register(f"s[{i}:{i+1}]", i, size=8) for i in range(6, 32, 2))
# Broadcasts and masked indexing keep many 64-bit flat addresses live at once. Keep 40 address pairs disjoint from scalar values because this
# allocator does not model pair overlap.
VGPR64 = tuple(Register(f"v[{i}:{i+1}]", i, size=8) for i in range(4, 84, 2))
VGPR32 = tuple(Register(f"v{i}", i, size=4) for i in range(84, 224))
REG_BUFFER_BASE = 224
REG_BUFFER_COUNT = 256 - REG_BUFFER_BASE
SCRATCH_RSRC_INDEX = 36
SUPPORTED_DTYPES = {dtypes.bool, *dtypes.int8s, *dtypes.int16s, *dtypes.int32s, dtypes.half, dtypes.float32}

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
  # Keep the PARAM reachable for ELF kernarg metadata and preserve its place in the schedule. The True tag prevents instruction selection from
  # revisiting it. Scalar kernargs arrive in SGPRs, while the current integer selector expects varying ALU values in VGPRs. Broadcast the low
  # dword before using one in index arithmetic; pointer arguments keep the full loaded pair.
  value = UOp(Ops.INS, x.dtype, (load,), GFX803Ops.V_MOV_B32) if x.addrspace is AddrSpace.ALU else load
  return value.after(x.rtag())

def _unafter(x:UOp) -> UOp: return _unafter(x.src[0]) if x.op is Ops.AFTER else x

def _global_index(ctx:IselContext, x:UOp, base:UOp, idx:UOp) -> UOp:
  deps:tuple[UOp, ...] = ()
  root = base
  while root.op is Ops.AFTER:
    deps, root = deps + root.src[1:], root.src[0]
  # Index selection can precede BUFFER selection. Normalize metadata in either order.
  if root.op is Ops.BUFFER: root = _buffer(ctx, root)
  if root.arg in {GFX803Ops.REG_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_META}:
    if idx.op is not Ops.CONST: raise NotImplementedError("gfx803 indirect private-buffer indexing is not implemented")
    offset, size, elem = int(root.src[0].arg), int(root.src[1].arg), int(idx.arg)
    if not 0 <= elem < size: raise RuntimeError(f"unsupported gfx803 private buffer shape/index: {size=}, {elem=}")
    if root.arg is GFX803Ops.SCRATCH_BUFFER_META:
      ref = UOp(Ops.INS, x.dtype, (root, _const_u32(offset + elem * x.dtype.itemsize)), GFX803Ops.SCRATCH_BUFFER_ADDR)
    else:
      reg_idx = REG_BUFFER_BASE + offset + elem
      if reg_idx >= 256: raise RuntimeError(f"gfx803 register buffers exceed the reserved {REG_BUFFER_COUNT} VGPRs")
      ref = UOp(Ops.INS, x.dtype, (root,), GFX803Ops.REG_BUFFER, (Register(f"v{reg_idx}", reg_idx, size=4),))
    return ref.after(*deps)
  shift = x.dtype.itemsize.bit_length() - 1
  if root.arg is GFX803Ops.LDS_BUFFER:
    return UOp(Ops.INS, dtypes.uint32, (idx, _const_u32(shift), root, *deps), GFX803Ops.LDS_ADDR)
  if idx.op is Ops.CONST:
    byte_offset = int(idx.arg) * x.dtype.itemsize
    if not 0 <= byte_offset <= 0xffffffff: raise OverflowError(f"gfx803 global byte offset out of range: {byte_offset}")
    offset_uop = _const_u32(byte_offset)
  else: offset_uop = idx << shift if shift else idx
  return UOp(Ops.INS, dtypes.uint64, (base, offset_uop), GFX803Ops.FLAT_ADDR)

def _memory_op(prefix:str, dtype:DType, store:bool=False) -> GFX803Ops:
  kind = "B" if store or dtype.itemsize >= 4 else "S" if dtype in dtypes.sints else "U"
  return GFX803Ops[f"{prefix}_{'STORE' if store else 'LOAD'}_{kind}{dtype.itemsize * 8}"]

def _memory(ctx:IselContext, x:UOp) -> UOp:
  store, addr = x.op is Ops.STORE, x.src[0]
  dtype = x.src[1].dtype if store else x.dtype
  if dtype not in SUPPORTED_DTYPES: raise NotImplementedError(f"gfx803 memory dtype {dtype} is not lowered")
  gate = x.src[-1] if len(x.src) == 3 else None
  root = _unafter(addr)
  if root.op is Ops.INDEX: root = _unafter(_global_index(ctx, root, *root.src))
  prefix = {GFX803Ops.REG_BUFFER:"REG", GFX803Ops.SCRATCH_BUFFER_ADDR:"SCRATCH", GFX803Ops.LDS_ADDR:"DS"}.get(root.arg, "FLAT")
  if gate is not None and prefix != "FLAT": raise NotImplementedError(f"gfx803 gated {prefix} access is not implemented")
  value = _to_vgpr(x.src[1]) if store else x
  src:tuple[UOp, ...]
  if prefix == "SCRATCH":
    src = (value, root.src[1], addr) if store else (root.src[1], addr)
  else: src = (addr, value) if store else (addr,)
  if gate is not None: src += (gate,) if store else (x.src[1], gate)
  op = (GFX803Ops.REG_STORE_B32 if store else GFX803Ops.V_MOV_B32) if prefix == "REG" else \
       _memory_op("GATED_FLAT" if gate is not None else prefix, dtype, store)
  return UOp(Ops.INS, x.dtype, src, op)

def _to_vgpr(x:UOp) -> UOp: return UOp(Ops.INS, x.dtype, (x,), GFX803Ops.V_MOV_B32) if x.op is Ops.CONST else x

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

def _range(ctx:IselContext, x:UOp) -> UOp|None: return x.replace(tag=(ctx.vreg(VGPR32),)) if not isinstance(x.tag, tuple) else None

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
  if b.op is Ops.CONST: a, b = b, a
  return _normalize_int(UOp(Ops.INS, x.dtype, (a, _to_vgpr(b)), GFX803Ops.V_ADD_U32), x.dtype)

def _int_mul(x:UOp, a:UOp, b:UOp) -> UOp:
  if a.op is Ops.CONST: a, b = b, a
  if b.op is Ops.CONST and (c:=int(b.arg)) > 0 and c & (c - 1) == 0:
    result = UOp(Ops.INS, x.dtype, (b.const_like(c.bit_length() - 1), _to_vgpr(a)), GFX803Ops.V_LSHLREV_B32)
  else:
    b = _to_vgpr(b) if b.op is Ops.CONST and not -16 <= int(b.arg) <= 64 else b
    result = UOp(Ops.INS, x.dtype, (_to_vgpr(a), b), GFX803Ops.V_MUL_LO_U32)
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

def _int_divmod(x:UOp, a:UOp, b:UOp) -> UOp:
  # LLVM's GCN reciprocal estimate with two corrections retains all 32 dividend bits.
  unsigned = x.dtype in dtypes.uints
  dividend, divisor = a.cast(dtypes.uint32), b.cast(dtypes.uint32)
  if not unsigned:
    dividend, divisor = (a < 0).where(-dividend, dividend), (b < 0).where(-divisor, divisor)
  estimate = UOp(Ops.INS, dtypes.float32, (divisor.cast(dtypes.float32),), GFX803Ops.V_RCP_IFLAG_F32)
  reciprocal = (estimate * 4294966784.0).cast(dtypes.uint32)
  reciprocal = reciprocal + UOp(Ops.INS, dtypes.uint32, (reciprocal, -divisor * reciprocal), GFX803Ops.V_MUL_HI_U32)
  quotient = UOp(Ops.INS, dtypes.uint32, (dividend, reciprocal), GFX803Ops.V_MUL_HI_U32)
  remainder = dividend - quotient * divisor
  for _ in range(2):
    below = remainder < divisor
    quotient, remainder = below.where(quotient, quotient + 1), below.where(remainder, remainder - divisor)
  result = quotient if x.op is Ops.CDIV else remainder
  if not unsigned:
    negative = (a < 0) ^ (b < 0) if x.op is Ops.CDIV else a < 0
    result = negative.where(-result, result)
  return result.cast(x.dtype)

def _float_unary(x:UOp, value:UOp) -> UOp:
  if x.op is Ops.SIN:
    # V_SIN consumes turns in a finite interval; reduce in float32 to cover all finite half inputs.
    turns = value.cast(dtypes.float32) * (1 / (2 * 3.141592653589793))
    return UOp(Ops.INS, dtypes.float32, (turns - turns.trunc(),), GFX803Ops.V_SIN).cast(x.dtype)
  op = {Ops.RECIPROCAL:GFX803Ops.V_RCP_F32, Ops.SQRT:GFX803Ops.V_SQRT_F32,
        Ops.EXP2:GFX803Ops.V_EXP2_F32, Ops.LOG2:GFX803Ops.V_LOG2_F32, Ops.TRUNC:GFX803Ops.V_TRUNC}[x.op]
  # Widen the composite asinh/acosh argument so its square cannot overflow half.
  if x.dtype is dtypes.half and x.op is Ops.LOG2 and value.op is Ops.ADD:
    return UOp(Ops.INS, dtypes.float32, (_half_to_float(value),), op).cast(dtypes.half)
  return UOp(Ops.INS, x.dtype, (value,), op)

def _half_to_float(value:UOp) -> UOp:
  if value.dtype is dtypes.float32: return value
  if value.op in {Ops.ADD, Ops.MUL, Ops.MAX, Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.TRUNC}:
    return UOp(value.op, dtypes.float32, tuple(_half_to_float(src) for src in value.src), value.arg)
  if value.op is Ops.WHERE:
    return UOp(Ops.WHERE, dtypes.float32, (value.src[0], _half_to_float(value.src[1]), _half_to_float(value.src[2])), value.arg)
  return UOp(Ops.INS, dtypes.float32, (_to_vgpr(value),), GFX803Ops.V_CVT_F32_F16)

def _float_bin(x:UOp, a:UOp, b:UOp, ins:GFX803Ops) -> UOp:
  # a/b arrives as a*reciprocal(b); widen the reciprocal to avoid overflow for representable quotients.
  if x.dtype is dtypes.half and ins is GFX803Ops.V_MUL_F32:
    def is_inverse(v:UOp) -> bool: return v.op is Ops.RECIPROCAL or (v.op is Ops.INS and v.arg in {GFX803Ops.V_RCP_F32, GFX803Ops.V_RSQ_F32})
    inverse = b if is_inverse(b) else a if is_inverse(a) else None
    if inverse is not None:
      numerator = a if inverse is b else b
      inverse_op = GFX803Ops.V_RSQ_F32 if inverse.op is Ops.INS and inverse.arg is GFX803Ops.V_RSQ_F32 else GFX803Ops.V_RCP_F32
      wide_inverse = UOp(Ops.INS, dtypes.float32, (_half_to_float(inverse.src[0]),), inverse_op)
      return (_half_to_float(numerator) * wide_inverse).cast(dtypes.half)
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
  src, dst = value.dtype, x.dtype
  if src not in SUPPORTED_DTYPES or dst not in SUPPORTED_DTYPES: return None
  if dst is dtypes.bool: return UOp(Ops.INS, dst, (value, _to_vgpr(value.const_like(0))), GFX803Ops.V_CMPNE)
  if not dtypes.is_float(src) and not dtypes.is_float(dst):
    if src is dtypes.bool or src.itemsize == dst.itemsize == 4: return UOp(Ops.INS, dst, (value,), GFX803Ops.V_MOV_B32)
    if dst.itemsize <= src.itemsize: return _bfe(value, dst, dst.bitsize, dst in dtypes.sints)
    widened = _bfe(value, dst, src.bitsize, src in dtypes.sints)
    # Signed-to-unsigned widening still wraps when the destination is narrower than a VGPR.
    return _bfe(widened, dst, dst.bitsize, False) if src in dtypes.sints and dst in dtypes.uints and dst.itemsize < 4 else widened
  if src is dtypes.half and dst in dtypes.int32s or dst is dtypes.half and src in (dtypes.bool, *dtypes.int32s):
    return value.cast(dtypes.float32).cast(dst)
  if not dtypes.is_float(src) and dst is dtypes.float32 and src is not dtypes.bool and src.itemsize < 4:
    value = _bfe(value, dtypes.int32 if src in dtypes.sints else dtypes.uint32, src.bitsize, src in dtypes.sints)
  def code(dt:DType, other:DType) -> str: return f"F{dt.bitsize}" if dtypes.is_float(dt) else f"{'I' if dt in dtypes.sints else 'U'}{other.bitsize}"
  op = GFX803Ops[f"V_CVT_{code(dst, src)}_{code(src, dst)}"]
  result = UOp(Ops.INS, dst, (_to_vgpr(value),), op)
  return _normalize_int(result, dst) if not dtypes.is_float(dst) else result

def _unsupported_elementwise(x:UOp):
  raise NotImplementedError(f"gfx803 {x.op.name} {x.dtype} from {tuple(s.dtype for s in x.src)} is not lowered yet")

def _stable_hardsigmoid(scaled:UOp) -> UOp:
  # Recover the clamp: subtracting two large half ReLUs loses the saturated 1.
  y = scaled + 0.5
  return (y < 0).where(0, (y > 1).where(1, y))

def _widen_half_product(a:UOp, b:UOp) -> UOp:
  # Half reductions already accumulate in float32. Widen before multiplying too, matching the product precision used by GPU dot/conv contracts.
  return UOp(Ops.MUL, dtypes.float32, (_half_to_float(a), _half_to_float(b)))

def _widen_half_bias_add(a:UOp, b:UOp) -> UOp|None:
  # Conv bias arrives after a premature float32-to-half cast of the reduction. Fold it into the wide accumulator and round only the final sum.
  rounded, bias = (a, b) if a.op is Ops.CAST and a.src[0].dtype is dtypes.float32 else (b, a)
  if rounded.op is not Ops.CAST or rounded.dtype is not dtypes.half or rounded.src[0].dtype is not dtypes.float32: return None
  wide_sum = UOp(Ops.ADD, dtypes.float32, (rounded.src[0], _half_to_float(bias)))
  return UOp(Ops.CAST, dtypes.half, (wide_sum,), dtypes.half)

hs_y, hs_z = UPat.var("scaled", dtypes.half) + 0.5, UPat.var("scaled", dtypes.half) + -0.5
pre_isel_matcher = PatternMatcher([
  (UPat(Ops.RECIPROCAL, (dtypes.half, dtypes.float32),
        src=(UPat(Ops.SQRT, src=(UPat.var("value"),)),), name="x"), lambda x,value: UOp(Ops.INS, x.dtype, (value,), GFX803Ops.V_RSQ_F32)),
  (UPat(Ops.CAST, dtypes.float32, src=(UPat(Ops.MUL, dtypes.half, src=(UPat.var("a"), UPat.var("b"))),)), _widen_half_product),
  (UPat(Ops.ADD, dtypes.half, src=(UPat.var("a"), UPat.var("b"))), _widen_half_bias_add),
  ((hs_y > 0).where(hs_y, 0) + (hs_z > 0).where(hs_z, 0) * -1, _stable_hardsigmoid),
])
isel_matcher = PatternMatcher([
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), _abi),
  (UPat(Ops.BUFFER, name="x"), _buffer),
  (UPat(Ops.RANGE, name="x"), _range),
  (UPat(Ops.BARRIER, name="x"), lambda x: UOp(Ops.INS, dtypes.void, x.src, GFX803Ops.S_BARRIER)),
  (UPat((Ops.ADD, Ops.SUB), src=(UPat.var("base"), UPat.var("size")), name="x"), _stack_adjust),
  (UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("idx")), name="x"), _global_index),
  (UPat((Ops.LOAD, Ops.STORE), name="x"), _memory),
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
  (UPat(GroupOp.Elementwise, name="x"), _unsupported_elementwise),
  (UPat(Ops.INS, name="x"), _alloc_vreg),
])

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
  (UPat(Ops.SINK, name="x"), lambda x: (x, [UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_ENDPGM)])),
  (UPat(Ops.RANGE, name="x"), _lower_range),
  (UPat(Ops.END, name="x"), _lower_end),
  (UPat(GroupOp.All - {Ops.INS}, name="x"), lambda x: (x, [])),
])

def _physical_reg(x:UOp) -> Register:
  reg = greg(x)
  if not isinstance(reg, Register): raise RuntimeError(f"expected physical register for {x.op}, got {reg!r}")
  return reg

def _register_counts(x:UOp) -> tuple[int, int, int]:
  regs = [reg for operand in (x, *x.src) if isinstance(reg:=greg(operand), Register)]
  def count(bank:str): return max((r.index + max(1, r.size // 4) for r in regs if r.name.startswith(bank)), default=0)
  return count("v"), count("s"), 0

def _word(value:int) -> bytes: return struct.pack("<I", value & 0xffffffff)

def _vop1(op:int, dst:int, src0:int) -> int: return 0x7e000000 | ((dst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)

def _vop2(op:int, dst:int, src0:int, src1:int) -> int: return ((op & 0x3f) << 25) | ((dst & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)

def _vop3(op:int, dst:int, *srcs:int) -> tuple[int, int]:
  return 0xd0000000 | ((op & 0x3ff) << 16) | (dst & 0xff), sum((src & 0x1ff) << (i*9) for i, src in enumerate(srcs))

def _vopc(op:int, src0:int, src1:int) -> int: return 0x7c000000 | ((op & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)

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
    # Unsigned UOps keep their normalized 32-bit value, but GCN's inline negative constants are bit-identical and avoid an extra literal dword.
    if 0xfffffff0 <= value <= 0xffffffff: return 192 - (value - (1 << 32)), ()
    return 255, (value & 0xffffffff,)
  reg = _physical_reg(x)
  return (256 + reg.index if reg.name.startswith("v") else reg.index), ()

def _raw_src0(x:UOp) -> tuple[int, tuple[int, ...]]:
  return _src0(_const_u32(_half_bits(float(x.arg)))) if x.op is Ops.CONST and x.dtype is dtypes.half else _src0(x)

_VOP2_ENCODINGS:dict[GFX803Ops, int|tuple[int, int]] = {GFX803Ops[name]:cast(int|tuple[int, int], code) for name, code in {
  "V_ADD_U32":0x19, "V_LSHLREV_B32":0x12, "V_LSHRREV_B32":0x10, "V_ASHRREV_I32":0x11, "V_AND_B32":0x13, "V_OR_B32":0x14, "V_XOR_B32":0x15,
  "V_MAX_I32":0xd, "V_MAX_U32":0xf, "V_ADD_F32":(0x1, 0x1f), "V_MUL_F32":(0x5, 0x22), "V_MAX_F32":(0xb, 0x2d),
}.items()}

_VOP1_ENCODINGS:dict[GFX803Ops, int|tuple[int, int]] = {GFX803Ops[name]:cast(int|tuple[int, int], code) for name, code in {
  "V_CVT_F32_F16":0xb, "V_CVT_F16_F32":0xa, "V_CVT_F32_I32":0x5, "V_CVT_F32_U32":0x6, "V_CVT_I32_F32":0x8, "V_CVT_U32_F32":0x7,
  "V_CVT_I16_F16":0x3c, "V_CVT_U16_F16":0x3b, "V_CVT_F16_I16":0x3a, "V_CVT_F16_U16":0x39, "V_TRUNC":(0x1c, 0x46), "V_RCP_F32":(0x22, 0x3d),
  "V_RCP_IFLAG_F32":0x23, "V_SQRT_F32":(0x27, 0x3e), "V_RSQ_F32":(0x24, 0x3f), "V_EXP2_F32":(0x20, 0x41), "V_LOG2_F32":(0x21, 0x40),
  "V_SIN":(0x29, 0x49),
}.items()}
_VOP3_ENCODINGS = {GFX803Ops.V_MUL_LO_U32:0x285, GFX803Ops.V_MUL_HI_U32:0x286, GFX803Ops.V_BFE_I32:0x1c9, GFX803Ops.V_BFE_U32:0x1c8}

_SCRATCH_ENCODINGS:dict[GFX803Ops, int] = {GFX803Ops[name]:code for name, code in {
  "SCRATCH_LOAD_U8":0x10, "SCRATCH_LOAD_S8":0x11, "SCRATCH_LOAD_U16":0x12, "SCRATCH_LOAD_S16":0x13, "SCRATCH_LOAD_B32":0x14,
  "SCRATCH_LOAD_B64":0x14, "SCRATCH_STORE_B8":0x18, "SCRATCH_STORE_B16":0x1a, "SCRATCH_STORE_B32":0x1c, "SCRATCH_STORE_B64":0x1c,
}.items()}
_SCRATCH_LOADS = {op for op in _SCRATCH_ENCODINGS if "LOAD" in op.name}
_SCRATCH_STORES = set(_SCRATCH_ENCODINGS) - _SCRATCH_LOADS

_MEMORY_ENCODINGS:dict[GFX803Ops, int] = {GFX803Ops[name]:code for name, code in {
  "FLAT_LOAD_U8":0xdc400000, "FLAT_LOAD_S8":0xdc440000, "FLAT_LOAD_U16":0xdc480000, "FLAT_LOAD_S16":0xdc4c0000, "FLAT_LOAD_B32":0xdc500000,
  "FLAT_STORE_B8":0xdc600000, "FLAT_STORE_B16":0xdc680000, "FLAT_STORE_B32":0xdc700000, "DS_LOAD_U8":0xd8740000, "DS_LOAD_S8":0xd8720000,
  "DS_LOAD_U16":0xd8780000, "DS_LOAD_S16":0xd8760000, "DS_LOAD_B32":0xd86c0000, "DS_STORE_B8":0xd83c0000, "DS_STORE_B16":0xd83e0000,
  "DS_STORE_B32":0xd81a0000,
}.items()}
_GATED_MEMORY_OPS = {op for op in GFX803Ops if op.name.startswith("GATED_FLAT_")}
_ASYNC_MEMORY_OPS = {GFX803Ops.S_LOAD_B64, *_SCRATCH_ENCODINGS, *_MEMORY_ENCODINGS, *_GATED_MEMORY_OPS}
_SIMPLE_ENCODINGS = {GFX803Ops.S_WAITCNT:0xbf8c0000, GFX803Ops.S_BARRIER:0xbf8a0000, GFX803Ops.S_ENDPGM:0xbf810000}

def _typed_encoding(encodings:dict[GFX803Ops, int|tuple[int, int]], x:UOp) -> int:
  code = encodings[x.arg]
  return code[x.dtype is dtypes.half] if isinstance(code, tuple) else code

def _encode_scratch(x:UOp, dst:Register|None) -> bytes:
  reg, offset = (dst, x.src[0]) if x.arg in _SCRATCH_LOADS else (_physical_reg(x.src[0]), x.src[1])
  assert reg is not None
  if not reg.name.startswith("v"): raise NotImplementedError("gfx803 scalar-register spills are not implemented")
  if offset.op is not Ops.CONST: raise RuntimeError("gfx803 scratch offset must be constant")
  return b"".join(_mubuf(_SCRATCH_ENCODINGS[x.arg], int(offset.arg) + i*4, reg.index + i)
                  for i in range(2 if x.arg.name.endswith("_B64") else 1))

def _encode_memory(x:UOp, dst:Register|None) -> bytes:
  reg, shift = (dst, 24) if "LOAD" in x.arg.name else (_physical_reg(x.src[1]), 8)
  assert reg is not None
  return _word(_MEMORY_ENCODINGS[x.arg]) + _word((reg.index << shift) | _physical_reg(x.src[0]).index)

def _encode_gated_memory(x:UOp, dst:Register|None) -> bytes:
  gate = _physical_reg(x.src[-1])
  if not gate.name.startswith("v"): raise RuntimeError(f"gfx803 memory gate must be a VGPR, got {gate}")
  # Capture the predicate before initializing dst, which can alias the dying gate.
  data = _word(_vopc(0xcd, 128, gate.index))
  if "LOAD" in x.arg.name: data += _encode(x.replace(arg=GFX803Ops.V_MOV_B32, src=(x.src[1],))).data
  memory = _encode_memory(x.replace(arg=GFX803Ops[x.arg.name.removeprefix("GATED_")]), dst)
  return data + _word(0xbea0206a) + memory + _word(0xbefe0120)

def _encode(x:UOp, branch_offset:int=0) -> GFX803Instruction:
  counts = _register_counts(x)
  dst = _physical_reg(x) if x.dtype is not dtypes.void and x.arg not in \
    {GFX803Ops.REG_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_META, GFX803Ops.SCRATCH_BUFFER_ADDR, GFX803Ops.LDS_BUFFER} else None

  lds_size = scratch_size = 0
  if x.arg in {GFX803Ops.REG_BUFFER_META, GFX803Ops.REG_BUFFER, GFX803Ops.SCRATCH_BUFFER_META,
               GFX803Ops.SCRATCH_BUFFER_ADDR, GFX803Ops.SCRATCH_TEARDOWN, GFX803Ops.LABEL}: data = b""
  elif x.arg is GFX803Ops.LDS_BUFFER:
    lds_size = int(x.src[1].arg) * x.dtype.itemsize
    # GFX8 requires M0 initialized as the LDS aperture bound before the first DS instruction.
    data = _word(0xbefc00c1)
  elif x.arg is GFX803Ops.SCRATCH_SETUP:
    if x.src[0].op is not Ops.CONST or (scratch_size:=int(x.src[0].arg)) <= 0:
      raise RuntimeError("gfx803 scratch frame size must be a positive constant")
    # Enabling scratch shifts the incoming ABI to s[0:3]=resource, s[4:5]=kernarg, s[6:8]=WGIDs and s9=private-segment wave byte offset. Preserve
    # the renderer's normal body ABI and keep the resource above its scalar pool and the s[32:33] EXEC-save pair used by gated memory operations.
    words = [0x80000900, 0x82018001, 0xbea40100, 0xbea60102, 0xbe800104, 0xbe820006, 0xbe830007, 0xbe840008]
    data = b"".join(_word(w) for w in words)
    counts = (counts[0], max(counts[1], SCRATCH_RSRC_INDEX + 4), counts[2])
  elif x.arg in _SCRATCH_ENCODINGS:
    data = _encode_scratch(x, dst)
    counts = (counts[0], max(counts[1], SCRATCH_RSRC_INDEX + 4), counts[2])
  elif x.arg is GFX803Ops.S_LOAD_B64:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    if offset.op is not Ops.CONST: raise RuntimeError("s_load_dwordx2 offset must be constant")
    data = _word(0xc0060000 | (dst.index << 6) | (base.index >> 1)) + _word(int(offset.arg))
  elif x.arg in {GFX803Ops.V_MOV_B32, GFX803Ops.REG_STORE_B32}:
    value = x.src[0]
    if x.arg is GFX803Ops.REG_STORE_B32: dst, value = _physical_reg(value), x.src[1]
    assert dst is not None
    src, literal = _raw_src0(value)
    data = _word(_vop1(0x01, dst.index, src)) + b"".join(_word(v) for v in literal)
  elif x.arg is GFX803Ops.FLAT_ADDR:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    words = [0x7e000200 | (dst.index << 17) | base.index,
             0x7e000200 | ((dst.index + 1) << 17) | (base.index + 1)]
    if offset.op is not Ops.CONST or int(offset.arg):
      src0, literal = _src0(offset)
      words += [_vop2(0x19, dst.index, src0, dst.index), *literal, _vop2(0x1c, dst.index + 1, 128, dst.index + 1)]
    data = b"".join(_word(w) for w in words)
  elif x.arg is GFX803Ops.LDS_ADDR:
    idx, shift = x.src[:2]
    if idx.op is Ops.CONST: idx = _const_u32(int(idx.arg) << int(shift.arg))
    shifting = idx.op is not Ops.CONST and int(shift.arg) != 0
    inst = _encode(x.replace(arg=GFX803Ops.V_LSHLREV_B32 if shifting else GFX803Ops.V_MOV_B32, src=(shift, idx) if shifting else (idx,)))
    data = inst.data
  elif x.arg in _MEMORY_ENCODINGS:
    data = _encode_memory(x, dst)
  elif x.arg in _GATED_MEMORY_OPS:
    data = _encode_gated_memory(x, dst)
    counts = (counts[0], max(counts[1], 34), counts[2])
  elif x.arg in _VOP2_ENCODINGS:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    op = _typed_encoding(_VOP2_ENCODINGS, x)
    data = _word(_vop2(op, dst.index, src0, src1.index)) + b"".join(_word(v) for v in literal)
  elif x.arg in _VOP3_ENCODINGS:
    assert dst is not None
    srcs = [_src0(source) for source in x.src]
    if any(literal for _, literal in srcs): raise RuntimeError(f"{x.arg} does not support literal constants on gfx803")
    data = b"".join(_word(w) for w in _vop3(_VOP3_ENCODINGS[x.arg], dst.index, *(source for source, _ in srcs)))
  elif x.arg in _VOP1_ENCODINGS:
    assert dst is not None
    src_reg = _physical_reg(x.src[0])
    op = _typed_encoding(_VOP1_ENCODINGS, x)
    data = _word(_vop1(op, dst.index, 256 + src_reg.index))
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
    # Materializing true into dst before the compare is not alias-safe: the allocator may legally reuse a dying compare input for dst. VOP3
    # accepts inline constants in both data positions, so only write dst after VCC is set.
    words = [_vopc(cmp_code, src0, src1.index), *literal, *_vop3(0x100, dst.index, 128, 129, 106)]
    data = b"".join(_word(w) for w in words)
  elif x.arg is GFX803Ops.V_CNDMASK_B32:
    assert dst is not None
    cond, true_value, false_value = map(_physical_reg, x.src)
    if not all(value.name.startswith("v") for value in (cond, true_value, false_value)):
      raise RuntimeError(f"v_cndmask inputs must be VGPRs, got {cond}, {true_value} and {false_value}")
    words = [_vopc(0xcd, 128, cond.index), _vop2(0x00, dst.index, 256 + false_value.index, true_value.index)]
    data = b"".join(_word(w) for w in words)
  elif x.arg is GFX803Ops.V_CMP_GT_U32:
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    data = _word(_vopc(0xcc, src0, src1.index)) + b"".join(_word(v) for v in literal)
  elif x.arg in _SIMPLE_ENCODINGS: data = _word(_SIMPLE_ENCODINGS[x.arg])
  elif x.arg is GFX803Ops.S_CBRANCH_VCCNZ:
    if not -0x8000 <= branch_offset <= 0x7fff: raise OverflowError(f"gfx803 branch is out of range: {branch_offset}")
    data = _word(0xbf870000 | (branch_offset & 0xffff))
  else: raise RuntimeError(f"cannot encode gfx803 instruction {x.arg}")
  return GFX803Instruction(data, counts, lds_size, scratch_size)

class AMDASMRenderer(ISARenderer):
  device = "AMD"
  supports_float4 = False
  has_local, has_shared = True, False
  shared_max, global_max, global_prod_max = HIPRenderer.shared_max, HIPRenderer.global_max, HIPRenderer.global_prod_max
  pre_isel_matcher, isel_matcher = pre_isel_matcher, isel_matcher
  pre_regalloc_matcher = None
  post_regalloc_matcher = post_regalloc_matcher
  # Advertising the native bit operations also enables tinygrad's constant integer divide/modulo strength reduction before instruction selection.
  code_for_op = {op:lambda: None for op in
                 (Ops.AND, Ops.SHL, Ops.SHR, Ops.SQRT, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.TRUNC, Ops.SIN)}

  def __init__(self, target:Target):
    if target.arch != "gfx803": raise RuntimeError(f"AMDASMRenderer only supports gfx803, got {target.arch}")
    super().__init__(target)

  def supported_dtypes(self): return SUPPORTED_DTYPES
  def stack_pointer(self) -> UOp: return _fixed_reg(dtypes.uint32, SCRATCH_PTR)
  def spill(self, disp:UOp, x:UOp) -> UOp:
    reg = _physical_reg(x)
    if not reg.name.startswith("v"): raise NotImplementedError("gfx803 scalar-register spills are not implemented")
    op = _memory_op("SCRATCH", dtypes.uint64 if reg.size == 8 else x.dtype, True)
    return UOp(Ops.INS, dtypes.void, (x, disp), op)
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp:
    if not reg.name.startswith("v"): raise NotImplementedError("gfx803 scalar-register fills are not implemented")
    op = _memory_op("SCRATCH", dtypes.uint64 if reg.size == 8 else x.dtype)
    return UOp(Ops.INS, x.dtype, (disp,), op, (reg,))
  def copy(self, x:UOp, reg:Register) -> UOp: return UOp(Ops.INS, x.dtype, (x,), GFX803Ops.V_MOV_B32, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    # Show selected operations without maintaining a second native assembly emitter.
    return f".{function_name}:\n" + "\n".join(
      f"{u.arg.name.lower()} {u.dtype} {greg(u)} <- " + ", ".join(str(s.arg) if s.op is Ops.CONST else str(greg(s)) for s in u.src)
      for u in uops if u.arg is not GFX803Ops.DEFINE)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # Scalar and vector memory operations are asynchronous on GCN3. A full wait after each access is conservative but gives the first backend
    # path unambiguous dependency ordering; later scheduling can coalesce waits.
    spill_size = max((int(u.src[0].arg) for u in lin.src if u.arg is GFX803Ops.SCRATCH_SETUP), default=0)
    buffer_size = max((int(u.src[2].arg) for u in lin.src if u.arg is GFX803Ops.SCRATCH_BUFFER_META), default=0)
    buffer_base = (spill_size + 7) // 8 * 8
    total_scratch = buffer_base + buffer_size
    ordered:list[UOp] = []
    has_setup = False
    for u in lin.src:
      if u.arg is GFX803Ops.DEFINE: continue
      if u.arg is GFX803Ops.SCRATCH_SETUP:
        u, has_setup = u.replace(src=(UOp.const(u.src[0].dtype, total_scratch),)), True
      elif u.arg in _SCRATCH_LOADS and len(u.src) > 1 and _unafter(u.src[1]).arg is GFX803Ops.SCRATCH_BUFFER_ADDR:
        u = u.replace(src=(UOp.const(dtypes.int32, buffer_base + int(u.src[0].arg)), *u.src[1:]))
      elif u.arg in _SCRATCH_STORES and len(u.src) > 2 and _unafter(u.src[2]).arg is GFX803Ops.SCRATCH_BUFFER_ADDR:
        u = u.replace(src=(u.src[0], UOp.const(dtypes.int32, buffer_base + int(u.src[1].arg)), *u.src[2:]))
      ordered.append(u)
      if u.arg in _ASYNC_MEMORY_OPS: ordered.append(UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_WAITCNT))
    if total_scratch and not has_setup:
      ordered.insert(0, UOp(Ops.INS, dtypes.void, (UOp.const(dtypes.uint32, total_scratch),), GFX803Ops.SCRATCH_SETUP))
    instructions = [(u, _encode(u)) for u in ordered]
    labels:dict[str, int] = {}
    pc = 0
    for u, inst in instructions:
      if u.arg is GFX803Ops.LABEL: labels[str(u.tag)] = pc
      pc += len(inst.data)
    encoded:list[UOp] = []
    pc = 0
    for u, inst in instructions:
      if u.arg is GFX803Ops.LABEL: continue
      if u.arg is GFX803Ops.S_CBRANCH_VCCNZ:
        inst = _encode(u, (labels[str(u.tag)] - pc - 4) // 4)
      encoded.append(u.replace(arg=inst))
      pc += len(inst.data)
    return assemble_linear(prg, lin.replace(src=tuple(encoded)), self.target.arch)
