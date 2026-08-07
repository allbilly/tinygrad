from __future__ import annotations
# ruff: noqa: E702
import base64, os, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, cdiv, cmod, floordiv, floormod
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION = b"RKIM", 13
_HEADER = struct.Struct("<4sHHHHIII")  # magic, version, target, scratch, gathers, ops, constants, flags
_SCRATCH, _GATHER, _GATHER_AXIS = struct.Struct("<II"), struct.Struct("<HHIBBi"), struct.Struct("<IIi")
_FILL = struct.Struct("<BBHI")  # dst_kind, pad, dst_index, count
_EWOP = struct.Struct("<BBHIIII")  # dst_kind, pad, dst_index, lhs_kind, lhs_index, rhs_kind, rhs_index
_EWOP2 = struct.Struct("<II")  # count, ew_cfg

class RKTarget(IntEnum): RK3588 = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

@dataclass(frozen=True)
class RKScratch: size: int; alignment: int = 4096

@dataclass(frozen=True)
class RKGather:
  """Materialize an affine or fallback index map into contiguous FP16 scratch."""
  src_index: int; dst_scratch: int; count: int; base: int = 0
  axes: tuple[tuple[int, int, int], ...] = ()  # dst divisor, range limit, source stride
  offsets: tuple[int, ...] = ()

@dataclass(frozen=True)
class RKFill: dst: RKArg; count: int

@dataclass(frozen=True)
class RKEWOp:
  """One contiguous FP16 DPU elementwise operation."""
  dst: RKArg; lhs: RKArg; rhs: RKArg; count: int; ew_cfg: int

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); fill: RKFill|None = None; ew_ops: tuple[RKEWOp, ...] = ()
  chain_limit: int = 512

@dataclass(frozen=True)
class RKReloc: word: int; arg: RKArg

@dataclass(frozen=True)
class RKStage: commands: tuple[int, ...]; relocs: tuple[RKReloc, ...]

def encode_image(image:RKImage) -> bytes:
  flags = (1 if image.fill is not None else 0) | (image.chain_limit << 8)
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.scratch), len(image.gathers),
                               len(image.ew_ops), len(image.constants), flags))
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in image.gathers:
    fallback = bool(g.offsets)
    out += _GATHER.pack(g.dst_scratch, g.src_index, g.count, int(fallback), len(g.axes), g.base)
    if fallback: out += struct.pack(f"<{g.count}i", *g.offsets)
    else:
      for axis in g.axes: out += _GATHER_AXIS.pack(*axis)
  for op in image.ew_ops:
    out += _EWOP.pack(int(op.dst.kind), 0, op.dst.index, int(op.lhs.kind), op.lhs.index, int(op.rhs.kind), op.rhs.index)
    out += _EWOP2.pack(op.count, op.ew_cfg) + struct.pack("<iii", op.dst.addend, op.lhs.addend, op.rhs.addend)
  if image.fill is not None: out += _FILL.pack(int(image.fill.dst.kind), 0, image.fill.dst.index, image.fill.count)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, version, target, nscratch, ngather, nop, nconst, flags = _HEADER.unpack_from(blob)
  chain_limit = flags >> 8
  if magic != RKIMAGE_MAGIC or version != RKIMAGE_VERSION or flags & 0xfe or chain_limit not in (256, 512):
    raise ValueError("invalid RKImage header")
  off = _HEADER.size
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    dst_scratch, src_index, count, fallback, naxes, base = _GATHER.unpack_from(blob, off); off += _GATHER.size
    if fallback not in (0, 1) or (fallback and naxes): raise ValueError("invalid RKGather")
    if fallback:
      offsets = struct.unpack_from(f"<{count}i", blob, off); off += 4*count
      gathers.append(RKGather(src_index, dst_scratch, count, offsets=offsets))
    else:
      axes = tuple(_GATHER_AXIS.unpack_from(blob, off+i*_GATHER_AXIS.size) for i in range(naxes)); off += naxes*_GATHER_AXIS.size
      gathers.append(RKGather(src_index, dst_scratch, count, base, axes))
  ew_ops:list[RKEWOp] = []
  for _ in range(nop):
    dk, _, di, lk, li, rk_, ri = _EWOP.unpack_from(blob, off); off += _EWOP.size
    count, ew_cfg = _EWOP2.unpack_from(blob, off); off += _EWOP2.size
    da, la, ra = struct.unpack_from("<iii", blob, off); off += 12
    ew_ops.append(RKEWOp(RKArg(RKBufferKind(dk), di, da), RKArg(RKBufferKind(lk), li, la),
                         RKArg(RKBufferKind(rk_), ri, ra), count, ew_cfg))
  fill = None
  if flags & 1:
    dst_kind, _, dst_index, count = _FILL.unpack_from(blob, off); off += _FILL.size
    fill = RKFill(RKArg(RKBufferKind(dst_kind), dst_index), count)
  if off + nconst != len(blob): raise ValueError("invalid RKImage size")
  return RKImage(RKTarget(target), scratch, blob[off:], version, tuple(gathers), fill, tuple(ew_ops), chain_limit)

def patch_stage(stage:RKStage, address:Callable[[RKBufferKind, int], int]) -> tuple[int, ...]:
  commands = list(stage.commands)
  for reloc in stage.relocs:
    word = commands[reloc.word]
    value = (address(reloc.arg.kind, reloc.arg.index) + reloc.arg.addend) & 0xffffffff
    commands[reloc.word] = (word & ~0xffffffff0000) | (value << 16)
  return tuple(commands)

_DPU, _RDMA = 0x1001, 0x2001
_MAX_EW_ELEMS_FP16 = 64000  # elementwise.py tile cap
_EW_CFG = {Ops.ADD: 0x108002c0 | (2 << 16), Ops.MUL: 0x108002c0 | (1 << 2) | (1 << 8)}
def _cmd(target:int, reg:int, value:int) -> int: return ((target&0xffff)<<48)|((value&0xffffffff)<<16)|(reg&0xffff)
def _scratch_bytes(count:int) -> int: return max(count * 2, 64)

def emit_ew_stage(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int) -> RKStage:
  """Build one contiguous FP16 DPU EW command body without its PC-chain tail."""
  if not (0 < count <= _MAX_EW_ELEMS_FP16): raise ValueError(f"EW fp16 count {count} out of range")
  width = (count + 7) // 8 - 1
  regs:tuple[tuple[int, int, int], ...] = ((_DPU,rk.REG_DPU_S_POINTER,0xe),
    (_DPU,rk.REG_DPU_FEATURE_MODE_CFG,(15<<5)|(2<<1)|1),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),
    (_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,width),(_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),
    (_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,(7<<16)|7),
    (_DPU,rk.REG_DPU_EW_CFG,ew_cfg),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,(1<<16)|1),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,width),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,(1<<30)|(2<<2)))
  commands = [_cmd(*x) for x in regs]
  relocs:list[RKReloc] = []
  for target, reg, arg in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
                           (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(len(commands), arg)); commands.append(_cmd(target, reg, 0))
  commands.append(_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, (2<<15)|(15<<11)|(2<<5)|(1<<3)|1))
  return RKStage(tuple(commands), tuple(relocs))

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
  if u.op is Ops.CDIV: return cdiv(_eval_int(u.src[0], env), _eval_int(u.src[1], env))
  if u.op is Ops.CMOD: return cmod(_eval_int(u.src[0], env), _eval_int(u.src[1], env))
  if u.op is Ops.FLOORDIV: return floordiv(_eval_int(u.src[0], env), _eval_int(u.src[1], env))
  if u.op is Ops.FLOORMOD: return floormod(_eval_int(u.src[0], env), _eval_int(u.src[1], env))
  if u.op is Ops.CMPLT: return int(_eval_int(u.src[0], env) < _eval_int(u.src[1], env))
  if u.op is Ops.CMPNE: return int(_eval_int(u.src[0], env) != _eval_int(u.src[1], env))
  if u.op is Ops.AND: return _eval_int(u.src[0], env) & _eval_int(u.src[1], env)
  if u.op is Ops.OR: return _eval_int(u.src[0], env) | _eval_int(u.src[1], env)
  raise RuntimeError(f"RKPLAN_REJECT:unsupported_index {u.op.name}")

def _index_ranges(index:UOp) -> list[UOp]: return [u for u in index.toposort() if u.op is Ops.RANGE]

def _iter_range_env(ranges:list[UOp]) -> list[dict[UOp, int]]:
  if not ranges: return [{}]
  order:list[UOp] = []
  seen:set[UOp] = set()
  def add(r:UOp) -> None:
    if r in seen: return
    for src in r.src[1:]:
      if src.op is Ops.RANGE: add(src)
    seen.add(r); order.append(r)
  for r in ranges: add(r)
  envs:list[dict[UOp, int]] = [{}]
  for r in order:
    if r.src[0].op is not Ops.CONST: raise RuntimeError("RKPLAN_REJECT:unsupported_index")
    envs = [{**env, r: i} for env in envs for i in range(int(r.src[0].arg))]
  return envs

def _affine_index(u:UOp) -> tuple[int, dict[UOp, int]]|None:
  if u.op is Ops.CONST: return int(u.arg), {}
  if u.op is Ops.RANGE: return 0, {u: 1}
  if u.op not in (Ops.ADD, Ops.SUB, Ops.MUL): return None
  lhs, rhs = _affine_index(u.src[0]), _affine_index(u.src[1])
  if lhs is None or rhs is None: return None
  if u.op is Ops.MUL:
    if lhs[1] and rhs[1]: return None
    scale, affine = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
    return affine[0]*scale, {r: coeff*scale for r, coeff in affine[1].items()}
  sign = -1 if u.op is Ops.SUB else 1
  coeffs = lhs[1].copy()
  for r, coeff in rhs[1].items():
    if (merged:=coeffs.get(r, 0) + sign*coeff): coeffs[r] = merged
    elif r in coeffs: del coeffs[r]
  return lhs[0] + sign*rhs[0], coeffs

def _gather_offsets(out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> tuple[int, ...]:
  ranges = _index_ranges(out_index)
  if any(r not in ranges for r in _index_ranges(load_index) + ([] if gate is None else _index_ranges(gate))):
    raise RuntimeError("RKPLAN_REJECT:gather_index")
  offsets = [-2] * count
  for env in _iter_range_env(ranges):
    dst, src = _eval_int(out_index, env), _eval_int(load_index, env)
    if not (0 <= dst < count): raise RuntimeError("RKPLAN_REJECT:gather_index")
    offsets[dst] = src if gate is None or _eval_int(gate, env) else -1
    if offsets[dst] < -1: raise RuntimeError("RKPLAN_REJECT:gather_index")
  if any(offset == -2 for offset in offsets): raise RuntimeError("RKPLAN_REJECT:gather_index")
  return tuple(offsets)

def _gather_plan(src_index:int, dst_scratch:int, out_index:UOp, load_index:UOp, gate:UOp|None, count:int) -> RKGather:
  out_affine, load_affine = _affine_index(out_index), _affine_index(load_index)
  if gate is None and out_affine is not None and load_affine is not None and out_affine[0] == 0:
    expected = 1
    axes:list[tuple[int, int, int]] = []
    for r, dst_stride in sorted(out_affine[1].items(), key=lambda item: item[1]):
      if dst_stride != expected or r.src[0].op is not Ops.CONST: break
      limit = int(r.src[0].arg)
      if limit <= 0: break
      if (src_stride:=load_affine[1].get(r, 0)): axes.append((dst_stride, limit, src_stride))
      expected *= limit
    else:
      if expected == count and all(r in out_affine[1] for r in load_affine[1]):
        return RKGather(src_index, dst_scratch, count, load_affine[0], tuple(axes))
  return RKGather(src_index, dst_scratch, count, offsets=_gather_offsets(out_index, load_index, gate, count))

def _ew_leaf(u:UOp, out_index:UOp, count:int, oslot:int) -> RKArg|float|tuple[UOp, UOp, UOp|None]|None:
  if u.op is Ops.CAST and u.dtype.scalar() is dtypes.half: return _ew_leaf(u.src[0], out_index, count, oslot)
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.half: return float(u.arg)
  if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
    param, index, gate = u.src[0].src[0], u.src[0].src[1], u.src[2] if len(u.src) > 2 else None
    if len(u.src) > 1 and (u.src[1].op is not Ops.CONST or float(u.src[1].arg) != 0.0): return None
    if param.dtype.scalar() is not dtypes.half or param.arg.slot == oslot or param.src[0].op is not Ops.CONST: return None
    if gate is None and index.key == out_index.key and int(param.src[0].arg) == count: return RKArg(RKBufferKind.ARG, param.arg.slot)
    return param, index, gate
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
    if u.op is Ops.CAST or u.op is Ops.REDUCE or u.op in GroupOp.ALU: bad.append(f"{i}:{u.op.name}")
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
  high = _sub_half(scaled, _sub_half(scaled, x, neg_one), neg_one)
  return high, _sub_half(x, high, neg_one)

def _two_product(term:UOp, neg_one:UOp, splitter:UOp) -> tuple[UOp, UOp]:
  lhs_high, lhs_low = _split_half(term.src[0], neg_one, splitter)
  rhs_high, rhs_low = _split_half(term.src[1], neg_one, splitter)
  error = _sub_half(lhs_high.alu(Ops.MUL, rhs_high), term, neg_one)
  error = error.alu(Ops.ADD, lhs_high.alu(Ops.MUL, rhs_low)).alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_high))
  return term, error.alu(Ops.ADD, lhs_low.alu(Ops.MUL, rhs_low))

def _two_sum(lhs:UOp, rhs:UOp, neg_one:UOp) -> tuple[UOp, UOp]:
  total = lhs.alu(Ops.ADD, rhs)
  rhs_virtual = _sub_half(total, lhs, neg_one)
  lhs_error = _sub_half(lhs, _sub_half(total, rhs_virtual, neg_one), neg_one)
  return total, lhs_error.alu(Ops.ADD, _sub_half(rhs, rhs_virtual, neg_one))

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
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out_param = outs[0]; assert out_param is not None
  count, oslot, store = int(out_param.src[0].arg), out_param.arg.slot, stores[0]
  if store.src[0].op is not Ops.INDEX: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  if count <= 0: return RKImage(RKTarget.RK3588)
  out_index, out, val = store.src[0].src[1], RKArg(RKBufferKind.ARG, oslot), store.src[1]
  if val.op is Ops.CONST and val.dtype.scalar() is dtypes.half:
    return RKImage(RKTarget.RK3588, constants=struct.pack("<e", float(val.arg)), fill=RKFill(out, count))
  supported = RockchipRenderer.code_for_op
  if any(u.op is Ops.REDUCE or (u.op is Ops.CAST and u.dtype.scalar() is not dtypes.half) or
         (u.op is Ops.CAST and u.dtype.scalar() is dtypes.half and _ew_leaf(u, out_index, count, oslot) is None) or
         (u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.float, dtypes.float32, dtypes.float64)) for u in uops):
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  if any(u.op in GroupOp.ALU and u.op not in supported and u.dtype.scalar() is dtypes.half for u in uops):
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  chain_limit = 512
  if (terms:=_mul_reduction_terms(val)) is not None:
    reduce_mode = os.getenv("ROCKCHIP_EW_REDUCE", "sequential").strip().lower()
    if reduce_mode == "kahan": val = _compensated_mul_sum(terms)
    elif reduce_mode == "twoproduct": val, chain_limit = _precise_mul_sum(terms), 256
    elif reduce_mode != "sequential": raise ValueError(f"invalid ROCKCHIP_EW_REDUCE={reduce_mode!r}")
  order:list[UOp] = []
  visited:dict[UOp, bool] = {}
  def visit(u:UOp) -> bool:
    if u in visited: return visited[u]
    if u.op in supported and u.dtype.scalar() is dtypes.half:
      visited[u] = all(visit(src) for src in u.src)
      if visited[u]: order.append(u)
    else: visited[u] = _ew_leaf(u, out_index, count, oslot) is not None
    return visited[u]
  if not visit(val) or not order or order[-1] is not val:
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {_unsupported_ew_ops(uops, out_index, count, oslot, supported)}")
  uses = {u: 0 for u in order}
  for node in order:
    for src in node.src:
      if src in uses: uses[src] += 1
  values:dict[UOp, RKArg] = {}
  leaves:dict[UOp, RKArg] = {}
  free:list[int] = []
  const_scratch:dict[bytes, int] = {}
  gather_scratch:dict[tuple[int, int, int, tuple[tuple[int, int, int], ...], tuple[int, ...]], int] = {}
  gathers:list[RKGather] = []
  for expr in order:
    for src in expr.src:
      if isinstance((leaf:=_ew_leaf(src, out_index, count, oslot)), float):
        bits = struct.pack("<e", leaf)
        if bits not in const_scratch: const_scratch[bits] = len(const_scratch)
  scratch_count = len(const_scratch)
  ew_ops:list[RKEWOp] = []
  def operand(u:UOp) -> RKArg:
    nonlocal scratch_count
    if u in values: return values[u]
    if u in leaves: return leaves[u]
    leaf = _ew_leaf(u, out_index, count, oslot)
    assert leaf is not None
    if isinstance(leaf, float): ret = RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", leaf)])
    elif isinstance(leaf, RKArg): ret = leaf
    else:
      param, index, gate = leaf
      plan = _gather_plan(param.arg.slot, 0, out_index, index, gate, count)
      if plan.offsets: low, high = min(plan.offsets, default=0), max(plan.offsets, default=-1)
      else:
        low = high = plan.base
        for _, limit, stride in plan.axes:
          if stride < 0: low += (limit-1)*stride
          else: high += (limit-1)*stride
      if low < -1 or high >= int(param.src[0].arg): raise RuntimeError("RKPLAN_REJECT:gather_index")
      key = param.arg.slot, plan.count, plan.base, plan.axes, plan.offsets
      if key not in gather_scratch:
        gather_scratch[key] = scratch_count
        gathers.append(RKGather(param.arg.slot, scratch_count, plan.count, plan.base, plan.axes, plan.offsets))
        scratch_count += 1
      ret = RKArg(RKBufferKind.SCRATCH, gather_scratch[key])
    leaves[u] = ret
    return ret
  for expr in order:
    lhs, rhs = operand(expr.src[0]), operand(expr.src[1])
    if expr is val: dst = out
    elif (reuse:=next((values[src] for src in expr.src if src in values and uses[src] == 1 and
                       values[src].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    ew_ops.append(RKEWOp(dst, lhs, rhs, count, _EW_CFG[expr.op]))
    values[expr] = dst
    for dep in expr.src:
      if dep in uses:
        uses[dep] -= 1
        arg = values[dep]
        if uses[dep] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  constants = b""
  if const_scratch:
    by_slot = {slot: bits for bits, slot in const_scratch.items()}
    constants = b"".join(by_slot.get(i, b"\0\0") for i in range(max(by_slot) + 1))
  return RKImage(RKTarget.RK3588, tuple(RKScratch(_scratch_bytes(count)) for _ in range(scratch_count)), constants,
                 gathers=tuple(gathers), ew_ops=tuple(ew_ops), chain_limit=chain_limit)

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

_pm_fp32_to_fp16 = PatternMatcher([
  (UPat(Ops.ADD, dtypes.float, name="x"), lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)), lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
  # Fold zero padding into the gather mask. This changes only host integer layout; the selected values still feed DPU EW.
  (UPat(Ops.WHERE, dtypes.half, src=(UPat.var("gate"), UPat(Ops.LOAD, dtypes.half, name="load"), UPat.cvar("zero"))),
   lambda gate,load,zero: load.replace(src=(load.src[0], load.src[1], gate.alu(Ops.AND, load.src[2])))
   if len(load.src) > 2 and float(load.src[1].arg) == 0.0 and float(zero.arg) == 0.0 else None),
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
