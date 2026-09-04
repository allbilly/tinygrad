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
  V_ADD_U32 = auto()
  V_LSHLREV_B32 = auto()
  V_MUL_LO_U32 = auto()
  V_ADD_F32 = auto()
  V_MUL_F32 = auto()
  V_MAX_F32 = auto()
  V_CMPLT = auto()
  V_CMPEQ = auto()
  V_CMPNE = auto()
  V_CNDMASK_B32 = auto()
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
WGID = tuple(Register(f"s{i}", i, size=4) for i in range(2, 5))
WIID = tuple(Register(f"v{i}", i, size=4) for i in range(3))
SGPR64 = tuple(Register(f"s[{i}:{i+1}]", i, size=8) for i in range(6, 32, 2))
VGPR64 = tuple(Register(f"v[{i}:{i+1}]", i, size=8) for i in range(4, 68, 2))
VGPR32 = tuple(Register(f"v{i}", i, size=4) for i in range(68, 256))


def _fixed_reg(dtype:DType, reg:Register) -> UOp: return UOp(Ops.INS, dtype, arg=GFX803Ops.DEFINE, tag=(reg,))
def _const_u32(value:int) -> UOp: return UOp.const(dtypes.uint32, value)


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
  return load.after(x.rtag())


def _global_index(x:UOp, base:UOp, idx:UOp) -> UOp|None:
  if idx.op is Ops.CONST:
    byte_offset = int(idx.arg) * x.dtype.itemsize
    if not 0 <= byte_offset <= 0xffffffff: raise OverflowError(f"gfx803 global byte offset out of range: {byte_offset}")
    offset = _const_u32(byte_offset)
  else:
    shift = x.dtype.itemsize.bit_length() - 1
    offset = UOp(Ops.SHL, idx.dtype, (idx, UOp.const(idx.dtype, shift))) if shift else idx
  return UOp(Ops.INS, dtypes.uint64, (base, offset), GFX803Ops.FLAT_ADDR)


def _load(x:UOp, addr:UOp) -> UOp|None:
  if x.dtype is not dtypes.float32: raise NotImplementedError(f"gfx803 load dtype {x.dtype} is not lowered yet")
  return UOp(Ops.INS, x.dtype, (addr,), GFX803Ops.FLAT_LOAD_B32)


def _store(x:UOp, addr:UOp, value:UOp) -> UOp|None:
  if value.dtype is not dtypes.float32: raise NotImplementedError(f"gfx803 store dtype {value.dtype} is not lowered yet")
  if value.op is Ops.CONST: value = UOp(Ops.INS, value.dtype, (value,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, dtypes.void, (addr, value), GFX803Ops.FLAT_STORE_B32)


def _alloc_vreg(ctx:IselContext, x:UOp) -> UOp|None:
  if x.dtype is dtypes.void or (isinstance(x.tag, tuple) and isinstance(x.tag[0], Register)): return None
  regs = SGPR64 if x.arg is GFX803Ops.S_LOAD_B64 else VGPR64 if x.arg is GFX803Ops.FLAT_ADDR else VGPR32
  return x.replace(tag=(ctx.vreg(regs),))


def _int_add(x:UOp, a:UOp, b:UOp) -> UOp:
  return UOp(Ops.INS, x.dtype, (b, a) if b.op is Ops.CONST else (a, b), GFX803Ops.V_ADD_U32)


def _int_mul(x:UOp, a:UOp, b:UOp) -> UOp:
  const:UOp
  value:UOp
  if a.op is Ops.CONST: const, value = a, b
  elif b.op is Ops.CONST: const, value = b, a
  else: return UOp(Ops.INS, x.dtype, (a, b), GFX803Ops.V_MUL_LO_U32)
  c = int(const.arg)
  if c > 0 and c & (c - 1) == 0:
    return UOp(Ops.INS, x.dtype, (UOp.const(x.dtype, c.bit_length() - 1), value), GFX803Ops.V_LSHLREV_B32)
  if not -16 <= c <= 64: const = UOp(Ops.INS, x.dtype, (const,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (value, const), GFX803Ops.V_MUL_LO_U32)


def _float_bin(x:UOp, a:UOp, b:UOp, ins:GFX803Ops) -> UOp:
  if b.op is Ops.CONST: a, b = b, a
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (a, b), ins)


def _cmp(x:UOp, a:UOp, b:UOp) -> UOp:
  if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, (b,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (a, b), {Ops.CMPLT:GFX803Ops.V_CMPLT, Ops.CMPEQ:GFX803Ops.V_CMPEQ, Ops.CMPNE:GFX803Ops.V_CMPNE}[x.op])


def _where(x:UOp, cond:UOp, true_value:UOp, false_value:UOp) -> UOp:
  if true_value.op is Ops.CONST: true_value = UOp(Ops.INS, true_value.dtype, (true_value,), GFX803Ops.V_MOV_B32)
  return UOp(Ops.INS, x.dtype, (cond, true_value, false_value), GFX803Ops.V_CNDMASK_B32)


pre_isel_matcher = PatternMatcher([])
isel_matcher = PatternMatcher([
  (UPat((Ops.PARAM, Ops.SPECIAL), name="x"), _abi),
  (UPat(Ops.INDEX, src=(UPat.var("base"), UPat.var("idx")), name="x"), _global_index),
  (UPat(Ops.LOAD, src=(UPat.var("addr"),), name="x"), _load),
  (UPat(Ops.ADD, dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_add),
  (UPat(Ops.MUL, dtypes.int32s, src=(UPat.var("a"), UPat.var("b")), name="x"), _int_mul),
  (UPat(Ops.SHL, dtypes.int32s, src=(UPat.var("value"), UPat.var("shift")), name="x"),
   lambda x,value,shift: UOp(Ops.INS, x.dtype, (shift, value), GFX803Ops.V_LSHLREV_B32)),
  ((UPat.var("a", dtypes.float32) + UPat.var("b", dtypes.float32)).named("x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_ADD_F32)),
  (UPat(Ops.MUL, dtypes.float32, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_MUL_F32)),
  (UPat(Ops.MAX, dtypes.float32, src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda x,a,b: _float_bin(x, a, b, GFX803Ops.V_MAX_F32)),
  (UPat((Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE), dtypes.bool, src=(UPat.var("a"), UPat.var("b")), name="x"), _cmp),
  (UPat(Ops.WHERE, src=(UPat.var("cond"), UPat.var("true_value"), UPat.var("false_value")), name="x"), _where),
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


def _vop3(op:int, dst:int, src0:int, src1:int) -> tuple[int, int]:
  return 0xd0000000 | ((op & 0x3ff) << 16) | (dst & 0xff), (src0 & 0x1ff) | ((src1 & 0x1ff) << 9)


def _vopc(op:int, src0:int, src1:int) -> int:
  return 0x7c000000 | ((op & 0xff) << 17) | ((src1 & 0xff) << 9) | (src0 & 0x1ff)


def _src0(x:UOp) -> tuple[int, tuple[int, ...]]:
  if x.op is Ops.CONST:
    if dtypes.is_float(x.dtype):
      value = struct.unpack("<I", struct.pack("<f", float(x.arg)))[0]
      inline = {0x00000000:128, 0x3f000000:240, 0xbf000000:241, 0x3f800000:242, 0xbf800000:243,
                0x40000000:244, 0xc0000000:245, 0x40800000:246, 0xc0800000:247}
      return (inline[value], ()) if value in inline else (255, (value,))
    value = int(x.arg)
    if 0 <= value <= 64: return 128 + value, ()
    if -16 <= value < 0: return 192 - value, ()
    return 255, (value & 0xffffffff,)
  reg = _physical_reg(x)
  return (256 + reg.index if reg.name.startswith("v") else reg.index), ()


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
    src, literal = _src0(x.src[0])
    data = _word(0x7e000200 | (dst.index << 17) | src) + b"".join(_word(v) for v in literal)
    text = f"v_mov_b32_e32 v{dst.index}, {x.src[0].arg if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name}"
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
  elif x.arg is GFX803Ops.FLAT_LOAD_B32:
    assert dst is not None
    addr = _physical_reg(x.src[0])
    data = _word(0xdc500000) + _word((dst.index << 24) | addr.index)
    text = f"flat_load_dword v{dst.index}, v[{addr.index}:{addr.index+1}]"
  elif x.arg is GFX803Ops.FLAT_STORE_B32:
    addr, value = _physical_reg(x.src[0]), _physical_reg(x.src[1])
    data = _word(0xdc700000) + _word((value.index << 8) | addr.index)
    text = f"flat_store_dword v[{addr.index}:{addr.index+1}], v{value.index}"
  elif x.arg in {GFX803Ops.V_ADD_U32, GFX803Ops.V_LSHLREV_B32}:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    op, mnemonic = (0x19, "v_add_u32_e32") if x.arg is GFX803Ops.V_ADD_U32 else (0x12, "v_lshlrev_b32_e32")
    data = _word(_vop2(op, dst.index, src0, src1.index)) + b"".join(_word(v) for v in literal)
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    text = f"{mnemonic} v{dst.index}, {'vcc, ' if x.arg is GFX803Ops.V_ADD_U32 else ''}{src0_text}, v{src1.index}"
  elif x.arg is GFX803Ops.V_MUL_LO_U32:
    assert dst is not None
    src0_code, literal0 = _src0(x.src[0])
    src1_code, literal1 = _src0(x.src[1])
    if literal0 or literal1: raise RuntimeError("v_mul_lo_u32 does not support literal constants on gfx803")
    data = b"".join(_word(w) for w in _vop3(0x285, dst.index, src0_code, src1_code))
    operands = [str(s.arg) if s.op is Ops.CONST else _physical_reg(s).name for s in x.src]
    text = f"v_mul_lo_u32 v{dst.index}, {operands[0]}, {operands[1]}"
  elif x.arg in {GFX803Ops.V_ADD_F32, GFX803Ops.V_MUL_F32, GFX803Ops.V_MAX_F32}:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    op, mnemonic = {GFX803Ops.V_ADD_F32:(0x01, "v_add_f32_e32"), GFX803Ops.V_MUL_F32:(0x05, "v_mul_f32_e32"),
                    GFX803Ops.V_MAX_F32:(0x0b, "v_max_f32_e32")}[x.arg]
    data = _word(_vop2(op, dst.index, src0, src1.index)) + b"".join(_word(v) for v in literal)
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    text = f"{mnemonic} v{dst.index}, {src0_text}, v{src1.index}"
  elif x.arg in {GFX803Ops.V_CMPLT, GFX803Ops.V_CMPEQ, GFX803Ops.V_CMPNE}:
    assert dst is not None
    src0, literal = _src0(x.src[0])
    src1 = _physical_reg(x.src[1])
    if not src1.name.startswith("v"): raise RuntimeError(f"{x.arg} second source must be a VGPR, got {src1}")
    cmp_code = {
      GFX803Ops.V_CMPLT:{dtypes.float32:0x41, dtypes.int32:0xc1, dtypes.uint32:0xc9, dtypes.bool:0xc9},
      GFX803Ops.V_CMPEQ:{dtypes.float32:0x42, dtypes.int32:0xc2, dtypes.uint32:0xca, dtypes.bool:0xca},
      GFX803Ops.V_CMPNE:{dtypes.float32:0x4d, dtypes.int32:0xc5, dtypes.uint32:0xcd, dtypes.bool:0xcd},
    }[x.arg][x.src[0].dtype]
    suffix = "f32" if x.src[0].dtype is dtypes.float32 else "i32" if x.src[0].dtype is dtypes.int32 else "u32"
    words = [0x7e000200 | (dst.index << 17) | 129, _vopc(cmp_code, src0, src1.index), *literal,
             _vop2(0x00, dst.index, 128, dst.index)]
    src0_text = str(x.src[0].arg) if x.src[0].op is Ops.CONST else _physical_reg(x.src[0]).name
    cmp_name = ({GFX803Ops.V_CMPLT:"lt", GFX803Ops.V_CMPEQ:"eq"}.get(x.arg) or
                ("neq" if x.src[0].dtype is dtypes.float32 else "ne"))
    text = (f"v_mov_b32_e32 v{dst.index}, 1\n"
            f"v_cmp_{cmp_name}_{suffix}_e32 vcc, {src0_text}, v{src1.index}\n"
            f"v_cndmask_b32_e32 v{dst.index}, 0, v{dst.index}, vcc")
    data = b"".join(_word(w) for w in words)
  elif x.arg is GFX803Ops.V_CNDMASK_B32:
    assert dst is not None
    cond, true_value, false_value = _physical_reg(x.src[0]), _physical_reg(x.src[1]), x.src[2]
    if not cond.name.startswith("v") or not true_value.name.startswith("v"):
      raise RuntimeError(f"v_cndmask inputs must be VGPRs, got {cond} and {true_value}")
    false_src, literal = _src0(false_value)
    words = [_vopc(0xcd, 128, cond.index), _vop2(0x00, dst.index, false_src, true_value.index), *literal]
    false_text = str(false_value.arg) if false_value.op is Ops.CONST else _physical_reg(false_value).name
    data = b"".join(_word(w) for w in words)
    text = (f"v_cmp_ne_u32_e32 vcc, 0, v{cond.index}\n"
            f"v_cndmask_b32_e32 v{dst.index}, {false_text}, v{true_value.index}, vcc")
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
