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
  DEFINE = auto()
  S_LOAD_B64 = auto()
  V_MOV_B32 = auto()
  FLAT_ADDR = auto()
  FLAT_LOAD_B32 = auto()
  FLAT_STORE_B32 = auto()
  V_ADD_F32 = auto()
  S_WAITCNT = auto()
  S_ENDPGM = auto()


@dataclass(frozen=True)
class GFX803Instruction:
  data: bytes
  text: str
  register_counts: tuple[int, int, int]

  def to_bytes(self) -> bytes: return self.data
  def __str__(self) -> str: return self.text


# The old linear allocator in this branch does not model overlapping wide
# registers. Keep 64-bit addresses and scalar values in disjoint VGPR banks.
KERNARG_PTR = Register("s[0:1]", 0, size=8)
SGPR64 = tuple(Register(f"s[{i}:{i+1}]", i, size=8) for i in range(2, 32, 2))
VGPR64 = tuple(Register(f"v[{i}:{i+1}]", i, size=8) for i in range(2, 32, 2))
VGPR32 = tuple(Register(f"v{i}", i, size=4) for i in range(32, 96))


def _fixed_reg(dtype:DType, reg:Register) -> UOp: return UOp(Ops.INS, dtype, arg=GFX803Ops.DEFINE, tag=(reg,))
def _const_u32(value:int) -> UOp: return UOp.const(dtypes.uint32, value)


def _abi(ctx:IselContext, x:UOp) -> UOp|None:
  if x.tag is True: return None
  if x.op is Ops.SPECIAL: raise NotImplementedError(f"gfx803 SPECIAL {x.arg!r} is not lowered yet")

  offset = sum(8 if u.addrspace is not AddrSpace.ALU else u.dtype.itemsize for u in ctx.func_args[:ctx.func_args.index(x)])
  kernarg = _fixed_reg(dtypes.uint64, KERNARG_PTR)
  load = UOp(Ops.INS, dtypes.uint64, (kernarg, _const_u32(offset)), GFX803Ops.S_LOAD_B64)
  # Keep the PARAM reachable for ELF kernarg metadata and preserve its place in
  # the schedule. The True tag prevents instruction selection from revisiting it.
  return load.after(x.rtag())


def _global_index(x:UOp, base:UOp, idx:UOp) -> UOp|None:
  if idx.op is not Ops.CONST: raise NotImplementedError("gfx803 dynamic global addresses are not lowered yet")
  byte_offset = int(idx.arg) * x.dtype.itemsize
  if not 0 <= byte_offset <= 0xffffffff: raise OverflowError(f"gfx803 global byte offset out of range: {byte_offset}")
  return UOp(Ops.INS, dtypes.uint64, (base, _const_u32(byte_offset)), GFX803Ops.FLAT_ADDR)


def _load(x:UOp, addr:UOp) -> UOp|None:
  if x.dtype is not dtypes.float32: raise NotImplementedError(f"gfx803 load dtype {x.dtype} is not lowered yet")
  return UOp(Ops.INS, x.dtype, (addr,), GFX803Ops.FLAT_LOAD_B32)


def _store(x:UOp, addr:UOp, value:UOp) -> UOp|None:
  if value.dtype is not dtypes.float32: raise NotImplementedError(f"gfx803 store dtype {value.dtype} is not lowered yet")
  return UOp(Ops.INS, dtypes.void, (addr, value), GFX803Ops.FLAT_STORE_B32)


def _alloc_vreg(ctx:IselContext, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void or (isinstance(x.tag, tuple) and isinstance(x.tag[0], Register)): return None
  regs = SGPR64 if x.arg is GFX803Ops.S_LOAD_B64 else VGPR64 if x.arg is GFX803Ops.FLAT_ADDR else VGPR32
  return x.replace(tag=(ctx.vreg(regs),))


pre_isel_matcher = PatternMatcher([])
isel_matcher = PatternMatcher([
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), _abi),
  (UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("idx")), name="x"), _global_index),
  (UPat(Ops.LOAD, src=(UPat.var("addr"),), name="x"), _load),
  ((UPat.var("a", dtypes.float32) + UPat.var("b", dtypes.float32)).named("x"),
   lambda x,a,b: UOp(Ops.INS, x.dtype, (a, b), GFX803Ops.V_ADD_F32)),
  (UPat(Ops.STORE, src=(UPat.var("addr"), UPat.var("value")), name="x"), _store),
  (UPat(Ops.INS, name="x"), _alloc_vreg),
])


def _finish(x:UOp) -> tuple[UOp, list[UOp]]:
  return x, [UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_ENDPGM)]


post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), _finish),
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


def _vop2(op:int, dst:int, src0:int, src1:int) -> int:
  return ((op & 0x3f) << 25) | ((dst & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)


def _encode(x:UOp) -> GFX803Instruction:
  counts = _register_counts(x)
  dst = _physical_reg(x) if x.dtype is not dtypes.void else None

  if x.arg is GFX803Ops.S_LOAD_B64:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    if offset.op is not Ops.CONST: raise RuntimeError("s_load_dwordx2 offset must be constant")
    data = _word(0xc0060000 | (dst.index << 6) | (base.index >> 1)) + _word(int(offset.arg))
    text = f"s_load_dwordx2 s[{dst.index}:{dst.index+1}], s[{base.index}:{base.index+1}], {int(offset.arg):#x}"
  elif x.arg is GFX803Ops.V_MOV_B32:
    assert dst is not None
    src = _physical_reg(x.src[0])
    data = _word(0x7e000200 | (dst.index << 17) | src.index)
    text = f"v_mov_b32_e32 v{dst.index}, {src.name}"
  elif x.arg is GFX803Ops.FLAT_ADDR:
    assert dst is not None
    base, offset = _physical_reg(x.src[0]), x.src[1]
    if offset.op is not Ops.CONST: raise RuntimeError("flat address offset must be constant")
    off = int(offset.arg)
    words = [0x7e000200 | (dst.index << 17) | base.index,
             0x7e000200 | ((dst.index + 1) << 17) | (base.index + 1)]
    lines = [f"v_mov_b32_e32 v{dst.index}, s{base.index}", f"v_mov_b32_e32 v{dst.index+1}, s{base.index+1}"]
    if off:
      if off <= 64: words.append(_vop2(0x19, dst.index, 128 + off, dst.index))
      else: words.extend((_vop2(0x19, dst.index, 255, dst.index), off))
      words.append(_vop2(0x1c, dst.index + 1, 128, dst.index + 1))
      lines += [f"v_add_u32_e32 v{dst.index}, vcc, {off:#x}, v{dst.index}",
                f"v_addc_u32_e32 v{dst.index+1}, vcc, 0, v{dst.index+1}, vcc"]
    data, text = b"".join(_word(w) for w in words), "\n".join(lines)
  elif x.arg is GFX803Ops.FLAT_LOAD_B32:
    assert dst is not None
    addr = _physical_reg(x.src[0])
    data = _word(0xdc500000) + _word((dst.index << 24) | addr.index)
    text = f"flat_load_dword v{dst.index}, v[{addr.index}:{addr.index+1}]"
  elif x.arg is GFX803Ops.FLAT_STORE_B32:
    addr, value = _physical_reg(x.src[0]), _physical_reg(x.src[1])
    data = _word(0xdc700000) + _word((value.index << 8) | addr.index)
    text = f"flat_store_dword v[{addr.index}:{addr.index+1}], v{value.index}"
  elif x.arg is GFX803Ops.V_ADD_F32:
    assert dst is not None
    a, b = _physical_reg(x.src[0]), _physical_reg(x.src[1])
    data = _word(_vop2(0x01, dst.index, 256 + a.index, b.index))
    text = f"v_add_f32_e32 v{dst.index}, v{a.index}, v{b.index}"
  elif x.arg is GFX803Ops.S_WAITCNT:
    data, text = _word(0xbf8c0000), "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"
  elif x.arg is GFX803Ops.S_ENDPGM:
    data, text = _word(0xbf810000), "s_endpgm"
  else: raise RuntimeError(f"cannot encode gfx803 instruction {x.arg}")
  return GFX803Instruction(data, text, counts)


class AMDASMRenderer(ISARenderer):
  device = "AMD"
  supports_float4 = False
  has_local, has_shared = True, False
  shared_max, global_max, global_prod_max = HIPRenderer.shared_max, HIPRenderer.global_max, HIPRenderer.global_prod_max
  pre_isel_matcher, isel_matcher = pre_isel_matcher, isel_matcher
  pre_regalloc_matcher = None
  post_regalloc_matcher = post_regalloc_matcher

  def __init__(self, target:Target):
    if target.arch != "gfx803": raise RuntimeError(f"AMDASMRenderer only supports gfx803, got {target.arch}")
    super().__init__(target)

  def supported_dtypes(self): return {dtypes.float32, dtypes.int32, dtypes.uint32, dtypes.bool}
  def stack_pointer(self) -> UOp: raise NotImplementedError("gfx803 spills are not implemented yet")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("gfx803 spills are not implemented yet")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("gfx803 spills are not implemented yet")
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), GFX803Ops.V_MOV_B32, (reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    return f".{function_name}:\n" + "\n".join(str(_encode(u)) for u in uops if u.arg is not GFX803Ops.DEFINE)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # Scalar and vector memory operations are asynchronous on GCN3. A full
    # wait after each access is conservative but gives the first backend path
    # unambiguous dependency ordering; later scheduling can coalesce waits.
    ordered:list[UOp] = []
    for u in lin.src:
      if u.arg is GFX803Ops.DEFINE: continue
      ordered.append(u)
      if u.arg in {GFX803Ops.S_LOAD_B64, GFX803Ops.FLAT_LOAD_B32, GFX803Ops.FLAT_STORE_B32}:
        ordered.append(UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_WAITCNT))
    encoded = tuple(u.replace(arg=_encode(u)) for u in ordered)
    return assemble_linear(prg, lin.replace(src=encoded), self.target.arch)
