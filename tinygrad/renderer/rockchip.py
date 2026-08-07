from __future__ import annotations
import base64, os, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 11, 1
_HEADER, _STAGE = struct.Struct("<4sHHHHHHIII"), struct.Struct("<BBHIIII")
_RELOC, _SCRATCH = struct.Struct("<HHBBIqIH"), struct.Struct("<II")
_GATHER = struct.Struct("<BHHI")  # itemsize, dst_scratch, src_index, n_offsets
_FILL = struct.Struct("<BBHI")  # dst_kind, pad, dst_index, count
_PACK = struct.Struct("<BHHI")  # itemsize, dst_scratch, src_index, count
_HALFOUT = struct.Struct("<HBBHI")  # src_scratch, dst_kind, pad, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, pad, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<III")  # count, ew_cfg, dst_scratch_for_cvt (-1u32 = none)

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum): DPU = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1; CONSTANT = 2
# DPU DATA_FORMAT OUT_PRECISION: 2=fp16 out, 5=fp32 out. Env ROCKCHIP_EW_OUT=fp16|fp32 (default fp32).
class RKEWOut(IntEnum): FP16 = 2; FP32 = 5

def ew_out_precision() -> RKEWOut:
  v = os.getenv("ROCKCHIP_EW_OUT", "fp32").strip().lower()
  if v in ("fp16", "half", "2"): return RKEWOut.FP16
  if v in ("fp32", "float", "5"): return RKEWOut.FP32
  raise ValueError(f"ROCKCHIP_EW_OUT={v!r}; expected fp16|fp32")

@dataclass(frozen=True)
class RKReloc:
  stage: int; word: int; kind: RKBufferKind; index: int
  addend: int = 0; shift: int = 0; mask: int = 0xffffffff; field_shift: int = 0

@dataclass(frozen=True)
class RKScratch: size: int; alignment: int = 4096

@dataclass(frozen=True)
class RKStage:
  engine: RKEngine; commands: tuple[int, ...]; relocs: tuple[RKReloc, ...] = (); flags: int = 0

@dataclass(frozen=True)
class RKGather:
  src_index: int; dst_scratch: int; offsets: tuple[int, ...]
  itemsize: int = 2  # 2=half ARG, 4=float ARG (cast to half at use)

@dataclass(frozen=True)
class RKPack:
  """Copy contiguous half ARG[src] into EW scratch (slot-packed OUT=5 / contig OUT=2)."""
  src_index: int; dst_scratch: int; count: int
  itemsize: int = 2

@dataclass(frozen=True)
class RKFill:
  dst: RKArg; count: int

@dataclass(frozen=True)
class RKHalfOut:
  """Copy EW result scratch → contiguous half ARG."""
  src_scratch: int; dst: RKArg; count: int

@dataclass(frozen=True)
class RKEWOp:
  """One logical EW; runtime expands per image.out_precision."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int
  # OUT=5 only: after chunks, convert this scratch f32→half in-slot.
  cvt_scratch: int | None = None

@dataclass(frozen=True)
class RKImage:
  target: RKTarget; stages: tuple[RKStage, ...] = ()
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); packs: tuple[RKPack, ...] = (); fill: RKFill|None = None
  ew_ops: tuple[RKEWOp, ...] = (); half_out: RKHalfOut|None = None
  out_precision: RKEWOut = RKEWOut.FP32

  # legacy alias used by runtime/encode
  @property
  def fills(self) -> RKFill|None: return object.__getattribute__(self, "fill")

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0; itemsize: int = 2

def encode_image(image:RKImage) -> bytes:
  # flags: bit0=fill, bit1=half_out, bits8-15=out_precision (2 or 5)
  flags = (1 if image.fill is not None else 0) | (2 if image.half_out is not None else 0) | (int(image.out_precision) << 8)
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), 0, 0,
                               len(image.scratch), len(image.gathers), 0, len(image.constants), flags))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in image.gathers:
    out += _GATHER.pack(g.itemsize, g.dst_scratch, g.src_index, len(g.offsets))
    out += struct.pack(f"<{len(g.offsets)}i", *g.offsets)
  out += struct.pack("<H", len(image.packs))
  for p in image.packs: out += _PACK.pack(p.itemsize, p.dst_scratch, p.src_index, p.count)
  out += struct.pack("<H", len(image.ew_ops))
  for op in image.ew_ops:
    out += _EWOP.pack(int(op.dst.kind), 0, op.dst.index, int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index)
    out += _EWOP2.pack(op.count, op.ew_cfg, 0xffffffff if op.cvt_scratch is None else op.cvt_scratch)
    out += struct.pack("<iii", op.dst.addend, op.lhs.addend, op.rhs.addend)
  if image.fill is not None:
    f = image.fill
    out += _FILL.pack(int(f.dst.kind), 0, f.dst.index, f.count)
  if image.half_out is not None:
    h = image.half_out
    out += _HALFOUT.pack(h.src_scratch, int(h.dst.kind), 0, h.dst.index, h.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, ver, target, nstage, nreloc, nscratch, ngather, ncmd, nconst, flags = _HEADER.unpack_from(blob)
  out_prec_raw = (flags >> 8) & 0xff
  out_prec = RKEWOut(out_prec_raw) if out_prec_raw else RKEWOut.FP32
  if magic != RKIMAGE_MAGIC or flags & ~0xff03 or nstage or nreloc or ncmd: raise ValueError("invalid RKImage header")
  if out_prec not in (RKEWOut.FP16, RKEWOut.FP32): raise ValueError(f"invalid out_precision {out_prec_raw}")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    itemsize, dst_scratch, src_index, n_off = _GATHER.unpack_from(blob, off); off += _GATHER.size
    if itemsize in (0, int(RKBufferKind.ARG)): itemsize = 2
    if itemsize not in (2, 4): raise ValueError("invalid RKGather itemsize")
    offs = struct.unpack_from(f"<{n_off}i", blob, off); off += 4 * n_off
    gathers.append(RKGather(src_index, dst_scratch, offs, itemsize))
  npack, = struct.unpack_from("<H", blob, off); off += 2
  packs:list[RKPack] = []
  for _ in range(npack):
    itemsize, dst_scratch, src_index, count = _PACK.unpack_from(blob, off); off += _PACK.size
    if itemsize in (0, int(RKBufferKind.ARG)): itemsize = 2
    if itemsize not in (2, 4): raise ValueError("invalid RKPack itemsize")
    packs.append(RKPack(src_index, dst_scratch, count, itemsize))
  nop, = struct.unpack_from("<H", blob, off); off += 2
  ew_ops:list[RKEWOp] = []
  for _ in range(nop):
    dk, _, di, lk, li, rk_, ri = _EWOP.unpack_from(blob, off); off += _EWOP.size
    count, ew_cfg, cvt = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg,
                         None if cvt == 0xffffffff else cvt))
  fill = None
  if flags & 1:
    dst_kind, _, dst_index, count = _FILL.unpack_from(blob, off); off += _FILL.size
    fill = RKFill(RKArg(RKBufferKind(dst_kind), dst_index), count)
  half_out = None
  if flags & 2:
    src_scratch, dst_kind, _, dst_index, count = _HALFOUT.unpack_from(blob, off); off += _HALFOUT.size
    half_out = RKHalfOut(src_scratch, RKArg(RKBufferKind(dst_kind), dst_index), count)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  return RKImage(RKTarget(target), (), scratch, blob[off:], ver, tuple(gathers), tuple(packs), fill,
                 tuple(ew_ops), half_out, out_prec)

def patch_image(image:RKImage, address:Callable[[RKBufferKind,int],int]) -> tuple[tuple[int,...],...]:
  patched = [list(s.commands) for s in image.stages]
  for s in image.stages:
    for r in s.relocs:
      w, v = patched[r.stage][r.word], (patched[r.stage][r.word]>>16)&0xffffffff
      field = ((address(r.kind, r.index)+r.addend)>>r.shift)&r.mask
      fm = (r.mask<<r.field_shift)&0xffffffff
      patched[r.stage][r.word] = (w & ~0xffffffff0000) | (((v & ~fm) | ((field<<r.field_shift)&fm)) << 16)
  return tuple(map(tuple, patched))

_DPU, _RDMA = 0x1001, 0x2001
# OUT=5: mtx512 NC1HWC2 C2=4, ≤8 elems/64B slot. OUT=2: contiguous half (elementwise.py).
_EW_CHUNK, _EW_SLOT_BYTES = 8, 64
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_EW_CFG = {Ops.ADD: 0x108002c0 | (2 << 16), Ops.MUL: 0x108002c0 | (1 << 2) | (1 << 8)}
_FP32_TILE, _FP32_TILE_BYTES = _EW_CHUNK, _EW_SLOT_BYTES
def _cmd(t,r,v): return ((t&0xffff)<<48)|((v&0xffffffff)<<16)|(r&0xffff)
def _scratch_bytes(count:int, out_precision:RKEWOut=RKEWOut.FP32) -> int:
  if out_precision is RKEWOut.FP16: return max(count * 2, 64)
  return ((count + _EW_CHUNK - 1) // _EW_CHUNK) * _EW_SLOT_BYTES

def emit_ew_stage(stage:int, dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int,
                  out_precision:RKEWOut=RKEWOut.FP32) -> RKStage:
  """DPU EW body (no PC tail). OUT=5: mtx512 ≤8. OUT=2: contiguous half WIDTH cube."""
  if out_precision is RKEWOut.FP16:
    if not (0 < count <= _MAX_EW_ELEMS_FP16): raise ValueError(f"EW fp16 count {count} out of range")
    w = (count + 7) // 8 - 1
    regs = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
      (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
      (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),
      (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,w),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
      (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7),
      (_DPU,rk.REG_DPU_EW_CFG,ew_cfg),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,(1<<16)|1),
      (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,w),
      (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
      (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
    cmds = [_cmd(*x) for x in regs]; relocs = []
    for t,r,a in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                 (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
      relocs.append(RKReloc(stage, len(cmds), a.kind, a.index, a.addend)); cmds.append(_cmd(t, r, 0))
    cmds.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, (2<<15)|(15<<11)|(2<<5)|(1<<3)|1))
    return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET)

  if not (0 < count <= _EW_CHUNK): raise ValueError(f"EW chunk count {count} out of range")
  ch = count - 1
  ow = (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1)  # size_e=1
  regs = ((_DPU,rk.REG_DPU_S_POINTER,0xe),(_DPU,rk.REG_DPU_FEATURE_MODE_CFG,0x1e5),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(5<<29)|(2<<26)|2),
    (_DPU,rk.REG_DPU_DST_SURF_STRIDE,1<<4),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,0),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(ch<<16)|ch),
    (_DPU,rk.REG_DPU_BS_CFG,0x53),(_DPU,rk.REG_DPU_BN_CFG,0x53),(_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,ow),(_DPU,rk.REG_DPU_WDMA_SIZE_0,ch),(_DPU,rk.REG_DPU_WDMA_SIZE_1,0),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,0),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0),(_DPU,rk.REG_DPU_EW_CFG,ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,0),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,4<<4),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,0),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,ch),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x40000008))
  cmds = [_cmd(*x) for x in regs]; relocs = []
  for t,r,a in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
               (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(stage, len(cmds), a.kind, a.index, a.addend)); cmds.append(_cmd(t, r, 0))
  cmds.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849))
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET)

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM:
    if not u.src: return None
    u = u.src[0]
  return u

def _eval_int(u:UOp, env:dict[UOp, int]) -> int:
  if u.op is Ops.CONST: return int(u.arg)
  if u.op is Ops.RANGE: return env[u]
  if u.op is Ops.ADD: return _eval_int(u.src[0], env) + _eval_int(u.src[1], env)
  if u.op is Ops.MUL: return _eval_int(u.src[0], env) * _eval_int(u.src[1], env)
  if u.op is Ops.SUB: return _eval_int(u.src[0], env) - _eval_int(u.src[1], env)
  raise RuntimeError(f"RKPLAN_REJECT:unsupported_index {u.op.name}")

def _index_ranges(ix:UOp) -> list[UOp]:
  return [u for u in ix.toposort() if u.op is Ops.RANGE]

def _iter_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  if not ranges: return [{}]
  order:list[UOp] = []
  seen:set[UOp] = set()
  def add(r:UOp) -> None:
    if r in seen: return
    for s in r.src[1:]:
      if s.op is Ops.RANGE: add(s)
    seen.add(r); order.append(r)
  for r in ranges: add(r)
  envs:list[dict[UOp, int]] = [{}]
  for r in order:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    lim = int(r.src[0].arg)
    envs = [{**e, r: i} for e in envs for i in range(lim)]
  return envs

def _gather_offsets(out_index:UOp, load_ix:UOp, count:int) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  for r in _index_ranges(load_ix):
    if r not in ranges: raise RuntimeError("RKPLAN_REJECT:gather_index")
  offsets = [-1] * count
  for env in _iter_range_env(ranges):
    dst, src = _eval_int(out_index, env), _eval_int(load_ix, env)
    if not (0 <= dst < count) or src < 0: raise RuntimeError("RKPLAN_REJECT:gather_index")
    offsets[dst] = src
  if any(o < 0 for o in offsets): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(offsets)

def _ew_leaf(u:UOp, out_index:UOp, count:int, oslot:int) -> RKArg|float|tuple[UOp, UOp]|None:
  if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half: return _ew_leaf(u.src[0], out_index, count, oslot)
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.half: return float(u.arg)
  if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
    p, ix = u.src[0].src[0], u.src[0].src[1]
    if p.dtype.scalar() not in (dtypes.half, dtypes.float, dtypes.float32) or p.arg.slot == oslot or p.src[0].op is not Ops.CONST:
      return None
    itemsize = 4 if p.dtype.scalar() in (dtypes.float, dtypes.float32) else 2
    if ix.key == out_index.key and int(p.src[0].arg) == count: return RKArg(RKBufferKind.ARG, p.arg.slot, itemsize=itemsize)
    return (p, ix)
  return None

def _unsupported_ew_ops(uops:list[UOp], out_index:UOp, count:int, oslot:int, supported:dict) -> list[str]:
  bad:list[str] = []
  for i, u in enumerate(uops):
    if u.op in (Ops.CONST, Ops.PARAM, Ops.RANGE, Ops.END, Ops.SINK, Ops.STORE, Ops.INDEX): continue
    if u.op in (Ops.ADD, Ops.MUL) and u.dtype.scalar() is dtypes.int: continue
    if u.op in supported and u.dtype.scalar() is dtypes.half: continue
    if u.op is Ops.LOAD or (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half):
      if _ew_leaf(u, out_index, count, oslot) is None: bad.append(f"{i}:{u.op.name}")
      continue
    if u.op is Ops.CAST or u.op is Ops.REDUCE or u.op in GroupOp.ALU:
      bad.append(f"{i}:{u.op.name}")
  return bad

def _mul_reduction_terms(u:UOp) -> list[UOp]|None:
  if u.op is Ops.MUL and u.dtype.scalar() is dtypes.half: return [u]
  if u.op is not Ops.ADD or u.dtype.scalar() is not dtypes.half: return None
  lhs, rhs = _mul_reduction_terms(u.src[0]), _mul_reduction_terms(u.src[1])
  return None if lhs is None or rhs is None else lhs + rhs

def _compensated_mul_sum(terms:list[UOp]) -> UOp:
  """Kahan sum built after symbolic simplification so compensation remains as DPU EW ops."""
  zero, neg_one = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half)
  total, correction = terms[0], zero
  for term in terms[1:]:
    adjusted = term.alu(Ops.ADD, correction.alu(Ops.MUL, neg_one))
    updated = total.alu(Ops.ADD, adjusted)
    correction = updated.alu(Ops.ADD, total.alu(Ops.MUL, neg_one)).alu(Ops.ADD, adjusted.alu(Ops.MUL, neg_one))
    total = updated
  return total

def _sub_half(lhs:UOp, rhs:UOp, neg_one:UOp) -> UOp: return lhs.alu(Ops.ADD, rhs.alu(Ops.MUL, neg_one))

def _split_half(x:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  scaled = x.alu(Ops.MUL, splitter)
  big = _sub_half(scaled, x, neg_one)
  high = _sub_half(scaled, big, neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one)
  error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low))
  error = error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  rhs_error = _sub_half(rhs, rhs_virtual, neg_one)
  return total, lhs_error.alu(Ops.ADD, rhs_error)

def _precise_mul_sum(terms:list[UOp]) -> UOp:
  """Recover FP16 product residuals and accumulate a two-half expansion using only DPU EW ops."""
  zero, neg_one, splitter = UOp.const(0.0, dtypes.half), UOp.const(-1.0, dtypes.half), UOp.const(65.0, dtypes.half)
  products, errors = zip(*(_two_product(term, neg_one, splitter) for term in terms))
  high, low = products[0], zero
  for part in products[1:] + errors:
    high, error = _two_sum(high, part, neg_one)
    low = low.alu(Ops.ADD, error)
  return high.alu(Ops.ADD, low)

def lower_ew(uops:list[UOp]) -> RKImage:
  out_prec = ew_out_precision()
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out_p = outs[0]; assert out_p is not None
  count, oslot = int(out_p.src[0].arg), out_p.arg.slot
  store = stores[0]
  if store.src[0].op is not Ops.INDEX: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  if count <= 0: return RKImage(RKTarget.RK3588, out_precision=out_prec)
  out_index, out, val = store.src[0].src[1], RKArg(RKBufferKind.ARG, oslot), store.src[1]
  if val.op is Ops.CONST and val.dtype.scalar() is dtypes.half:
    return RKImage(RKTarget.RK3588, (), (), struct.pack("<e", float(val.arg)), fill=RKFill(out, count),
                   out_precision=out_prec)
  supported = RockchipRenderer.code_for_op
  if any(u.op is Ops.REDUCE or
         (u.op is Ops.CAST and u.dtype.scalar() is not dtypes.half) or
         (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and _ew_leaf(u, out_index, count, oslot) is None) or
         (u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.float, dtypes.float32, dtypes.float64)) for u in uops):
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  if any(u.op in GroupOp.ALU and u.op not in supported and u.dtype.scalar() is dtypes.half for u in uops):
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  if (terms:=_mul_reduction_terms(val)) is not None:
    if (reduce_mode:=os.getenv("ROCKCHIP_EW_REDUCE", "sequential").strip().lower()) == "kahan": val = _compensated_mul_sum(terms)
    elif reduce_mode == "twoproduct": val = _precise_mul_sum(terms)
  order:list[UOp] = []
  visited:dict[UOp, bool] = {}
  def visit(u:UOp) -> bool:
    if u in visited: return visited[u]
    if u.op in supported and u.dtype.scalar() is dtypes.half:
      if not all(visit(s) for s in u.src):
        visited[u] = False
        return False
      if u not in order: order.append(u)
      visited[u] = True
      return visited[u]
    visited[u] = _ew_leaf(u, out_index, count, oslot) is not None
    return visited[u]
  if not visit(val) or not order or order[-1] is not val:
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  uses = {u: sum(s is u for n in order for s in n.src) for u in order}
  values:dict[UOp, RKArg] = {}
  free:list[int] = []
  const_scratch:dict[bytes, int] = {}
  gather_scratch:dict[tuple[int, tuple[int, ...]], int] = {}
  arg_pack:dict[int, int] = {}
  gathers:list[RKGather] = []
  packs:list[RKPack] = []
  for expr in order:
    for s in expr.src:
      leaf = _ew_leaf(s, out_index, count, oslot)
      if isinstance(leaf, float):
        k = struct.pack("<e", leaf)
        if k not in const_scratch: const_scratch[k] = len(const_scratch)
  scratch_count = len(const_scratch)
  ew_ops:list[RKEWOp] = []
  def operand(s:UOp) -> RKArg:
    nonlocal scratch_count
    if s in values: return values[s]
    leaf = _ew_leaf(s, out_index, count, oslot)
    assert leaf is not None
    if isinstance(leaf, float): return RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", leaf)])
    if isinstance(leaf, RKArg):
      if leaf.index not in arg_pack:
        arg_pack[leaf.index] = scratch_count
        packs.append(RKPack(leaf.index, scratch_count, count, leaf.itemsize))
        scratch_count += 1
      return RKArg(RKBufferKind.SCRATCH, arg_pack[leaf.index])
    p, ix = leaf
    offsets = _gather_offsets(out_index, ix, count)
    if max(offsets) >= int(p.src[0].arg): raise RuntimeError("RKPLAN_REJECT:gather_index")
    key = (p.arg.slot, offsets)
    if key not in gather_scratch:
      gather_scratch[key] = scratch_count
      itemsize = 4 if p.dtype.scalar() in (dtypes.float, dtypes.float32) else 2
      gathers.append(RKGather(p.arg.slot, scratch_count, offsets, itemsize))
      scratch_count += 1
    return RKArg(RKBufferKind.SCRATCH, gather_scratch[key])
  for expr in order:
    lhs, rhs = operand(expr.src[0]), operand(expr.src[1])
    if (reuse:=next((values[x] for x in expr.src if x in values and uses[x] == 1 and
                     values[x].kind is RKBufferKind.SCRATCH), None)) is not None and expr is not val:
      dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    cvt = dst.index if out_prec is RKEWOut.FP32 else None
    ew_ops.append(RKEWOp(dst, lhs, rhs, count, _EW_CFG[expr.op], cvt_scratch=cvt))
    values[expr] = dst
    for dep in expr.src:
      if dep in uses:
        uses[dep] -= 1
        arg = values[dep]
        if uses[dep] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  half_out = RKHalfOut(values[val].index, out, count)
  nbytes = _scratch_bytes(count, out_prec)
  constants = b""
  if const_scratch:
    by = {slot: bits for bits, slot in const_scratch.items()}
    constants = b"".join(by[i] if i in by else struct.pack("<e", 0.0)
                         for i in range(max(const_scratch.values()) + 1))
  return RKImage(RKTarget.RK3588, (), tuple(RKScratch(nbytes) for _ in range(scratch_count)),
                 constants, gathers=tuple(gathers), packs=tuple(packs), ew_ops=tuple(ew_ops),
                 half_out=half_out, out_precision=out_prec)

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

_pm_fp32_to_fp16 = PatternMatcher([
  (UPat(Ops.ADD, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
  (UPat(Ops.MAX, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.MAX, x.src[1].cast(dtypes.half))),
  (UPat(Ops.NEG, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.NEG)),
  (UPat(Ops.EXP2, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.EXP2)),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)),
   lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
])
def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  code_for_op = {Ops.ADD: lambda: None, Ops.MUL: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_ew(_fp16_rewrite(uops)))).decode()
