from __future__ import annotations
import base64, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from tinygrad.device import Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import GroupOp, Ops, UOp, UPat, PatternMatcher, graph_rewrite

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 5, 1
_HEADER, _STAGE = struct.Struct("<4sHHHHHHIII"), struct.Struct("<BBHIIII")
_RELOC, _SCRATCH = struct.Struct("<HHBBIqIH"), struct.Struct("<II")
_GATHER = struct.Struct("<BBHI")  # src_kind, dst_scratch, src_index, n_offsets
_HOSTSUM = struct.Struct("<BBHI")  # dst_kind, pad, dst_index, n_srcs; followed by n_srcs u16 scratch idxs + u32 count

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum): DPU = 1
class RKBufferKind(IntEnum): ARG = 0; SCRATCH = 1; CONSTANT = 2

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

@dataclass(frozen=True)
class RKHostSum:
  """Sum half scratch vectors in float32 into dst (gemm ADD reduce; half EW ADD loses too much)."""
  dst: RKArg; srcs: tuple[int, ...]; count: int

@dataclass(frozen=True)
class RKImage:
  target: RKTarget; stages: tuple[RKStage, ...]
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION
  gathers: tuple[RKGather, ...] = (); host_sum: RKHostSum|None = None

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

def encode_image(image:RKImage) -> bytes:
  cmds:list[int] = []; relocs:list[RKReloc] = []; rows:list[tuple] = []
  for s in image.stages:
    c0, r0 = len(cmds), len(relocs); cmds.extend(s.commands); relocs.extend(s.relocs)
    rows.append((int(s.engine), s.flags, 0, c0, len(s.commands), r0, len(s.relocs)))
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(rows), len(relocs),
                               len(image.scratch), len(image.gathers), len(cmds), len(image.constants),
                               1 if image.host_sum is not None else 0))
  for row in rows: out += _STAGE.pack(*row)
  for r in relocs: out += _RELOC.pack(r.stage, r.word, int(r.kind), r.shift, r.index, r.addend, r.mask, r.field_shift)
  for sc in image.scratch: out += _SCRATCH.pack(sc.size, sc.alignment)
  for g in image.gathers:
    out += _GATHER.pack(int(RKBufferKind.ARG), g.dst_scratch, g.src_index, len(g.offsets))
    out += struct.pack(f"<{len(g.offsets)}i", *g.offsets)
  if image.host_sum is not None:
    hs = image.host_sum
    out += _HOSTSUM.pack(int(hs.dst.kind), 0, hs.dst.index, len(hs.srcs))
    out += struct.pack(f"<{len(hs.srcs)}H", *hs.srcs) + struct.pack("<I", hs.count)
  return bytes(out) + (struct.pack(f"<{len(cmds)}Q", *cmds) if cmds else b"") + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, ver, target, nstage, nreloc, nscratch, ngather, ncmd, nconst, has_hostsum = _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or has_hostsum not in (0, 1): raise ValueError("invalid RKImage header")
  off = _HEADER.size
  rows = [_STAGE.unpack_from(blob, off+i*_STAGE.size) for i in range(nstage)]; off += nstage*_STAGE.size
  relocs = []
  for _ in range(nreloc):
    st, word, kind, shift, index, addend, mask, fshift = _RELOC.unpack_from(blob, off); off += _RELOC.size
    relocs.append(RKReloc(st, word, RKBufferKind(kind), index, addend, shift, mask, fshift))
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  gathers:list[RKGather] = []
  for _ in range(ngather):
    src_kind, dst_scratch, src_index, n_off = _GATHER.unpack_from(blob, off); off += _GATHER.size
    if src_kind != int(RKBufferKind.ARG): raise ValueError("invalid RKGather src kind")
    offs = struct.unpack_from(f"<{n_off}i", blob, off); off += 4 * n_off
    gathers.append(RKGather(src_index, dst_scratch, offs))
  host_sum = None
  if has_hostsum:
    dst_kind, _, dst_index, n_srcs = _HOSTSUM.unpack_from(blob, off); off += _HOSTSUM.size
    srcs = struct.unpack_from(f"<{n_srcs}H", blob, off); off += 2 * n_srcs
    count, = struct.unpack_from("<I", blob, off); off += 4
    host_sum = RKHostSum(RKArg(RKBufferKind(dst_kind), dst_index), srcs, count)
  if off + ncmd * 8 + nconst != len(blob): raise ValueError("invalid RKImage size")
  cmds = struct.unpack_from(f"<{ncmd}Q", blob, off) if ncmd else (); off += ncmd * 8
  stages = []
  for i,(eng,flags,r0,c0,clen,rstart,rlen) in enumerate(rows):
    if r0 or c0+clen > ncmd or rstart+rlen > nreloc: raise ValueError("invalid RKImage stage")
    stages.append(RKStage(RKEngine(eng), cmds[c0:c0+clen], tuple(relocs[rstart:rstart+rlen]), flags))
  return RKImage(RKTarget(target), tuple(stages), scratch, blob[off:], ver, tuple(gathers), host_sum)
def patch_image(image:RKImage, address:Callable[[RKBufferKind,int],int]) -> tuple[tuple[int,...],...]:
  patched = [list(s.commands) for s in image.stages]
  for s in image.stages:
    for r in s.relocs:
      w, v = patched[r.stage][r.word], (patched[r.stage][r.word]>>16)&0xffffffff
      field = ((address(r.kind, r.index)+r.addend)>>r.shift)&r.mask
      fm = (r.mask<<r.field_shift)&0xffffffff
      patched[r.stage][r.word] = (w & ~0xffffffff0000) | (((v & ~fm) | ((field<<r.field_shift)&fm)) << 16)
  return tuple(map(tuple, patched))

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x81
# DPU EW_CFG for ops listed in RockchipRenderer.code_for_op (allbilly/rk3588 elementwise.py).
_EW_CFG = {Ops.ADD: 0x108002c0 | (2 << 16), Ops.MUL: 0x108002c0 | (1 << 2) | (1 << 8)}
def _cmd(t,r,v): return ((t&0xffff)<<48)|((v&0xffffffff)<<16)|(r&0xffff)

def emit_ew_stage(stage:int, dst:RKArg, lhs:RKArg, rhs:RKArg, count:int, ew_cfg:int) -> RKStage:
  """DPU elementwise binary stage (ADD/MUL); same body as ADD with EW_CFG from elementwise.py."""
  w = (count+7)//8-1
  regs = ((_DPU,rk.REG_DPU_S_POINTER,0xe),(_DPU,rk.REG_DPU_FEATURE_MODE_CFG,0x1e5),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),(_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,w),
    (_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),(_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0x70007),
    (_DPU,rk.REG_DPU_BS_CFG,0x53),(_DPU,rk.REG_DPU_BN_CFG,0x53),(_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,2),(_DPU,rk.REG_DPU_WDMA_SIZE_0,7),(_DPU,rk.REG_DPU_WDMA_SIZE_1,w),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,0),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0),(_DPU,rk.REG_DPU_EW_CFG,ew_cfg),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,0x10001),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,0x40),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,w),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x40000008))
  cmds = [_cmd(*x) for x in regs]; relocs = []
  for t,r,a in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
               (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(stage, len(cmds), a.kind, a.index, a.addend)); cmds.append(_cmd(t, r, 0))
  cmds += [_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), _cmd(_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET)

def emit_ew_chain(out:RKArg, inputs:tuple[RKArg, ...], count:int, ew_cfg:int) -> RKImage:
  """Reduce N same-shape half buffers with N-1 DPU EW stages; intermediates use scratch."""
  if len(inputs) < 2: raise ValueError("EW chain needs at least two inputs")
  nbytes = ((count + 7) // 8 * 8) * 2  # atom-pad FP16 surface
  scratch = tuple(RKScratch(nbytes) for _ in range(max(0, len(inputs)-2)))
  stages:list[RKStage] = []
  acc = inputs[0]
  for i, rhs in enumerate(inputs[1:]):
    last = i == len(inputs)-2
    dst = out if last else RKArg(RKBufferKind.SCRATCH, i)
    stages.append(emit_ew_stage(i, dst, acc, rhs, count, ew_cfg))
    acc = dst
  return RKImage(RKTarget.RK3588, tuple(stages), scratch)

def emit_ew_const(out:RKArg, vec:RKArg, count:int, ew_cfg:int, const_val:float) -> RKImage:
  """vector ⊗ scalar: splat const into scratch[0] (value in image.constants as one fp16)."""
  nbytes = ((count + 7) // 8 * 8) * 2
  return RKImage(RKTarget.RK3588, (emit_ew_stage(0, out, vec, RKArg(RKBufferKind.SCRATCH, 0), count, ew_cfg),),
                 (RKScratch(nbytes),), struct.pack("<e", const_val))

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
  """Cartesian product of RANGE domains (src[0]=limit CONST; optional parent RANGEs in src[1:])."""
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
  if u.op is Ops.CONST and u.dtype.scalar() is dtypes.half: return float(u.arg)
  if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0].op is Ops.PARAM:
    p, ix = u.src[0].src[0], u.src[0].src[1]
    if p.dtype.scalar() is not dtypes.half or p.arg.slot == oslot or p.src[0].op is not Ops.CONST: return None
    if ix.key == out_index.key and int(p.src[0].arg) == count: return RKArg(RKBufferKind.ARG, p.arg.slot)
    return (p, ix)
  return None

def _unsupported_ew_ops(uops:list[UOp], out_index:UOp, count:int, oslot:int, supported:dict) -> list[str]:
  bad:list[str] = []
  for i, u in enumerate(uops):
    if u.op in (Ops.CONST, Ops.PARAM, Ops.RANGE, Ops.END, Ops.SINK, Ops.STORE, Ops.INDEX): continue
    if u.op in (Ops.ADD, Ops.MUL) and u.dtype.scalar() is dtypes.int: continue  # address math
    if u.op in supported and u.dtype.scalar() is dtypes.half: continue
    if u.op is Ops.LOAD:
      if _ew_leaf(u, out_index, count, oslot) is None: bad.append(f"{i}:{u.op.name}")
      continue
    if u.op is Ops.CAST or u.op is Ops.REDUCE or u.op in GroupOp.ALU:
      bad.append(f"{i}:{u.op.name}")
  return bad

def _mul_reduction_muls(u:UOp) -> list[UOp]|None:
  """If u is an ADD-tree of half MULs, return those MUL nodes (gemm-style); else None."""
  if u.op is Ops.MUL and u.dtype.scalar() is dtypes.half: return [u]
  if u.op is Ops.ADD and u.dtype.scalar() is dtypes.half:
    left, right = _mul_reduction_muls(u.src[0]), _mul_reduction_muls(u.src[1])
    if left is None or right is None: return None
    return left + right
  return None

def lower_ew(uops:list[UOp]) -> RKImage:
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out_p = outs[0]; assert out_p is not None
  count, oslot = int(out_p.src[0].arg), out_p.arg.slot
  store = stores[0]
  if store.src[0].op is not Ops.INDEX or count <= 0: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  out_index, out, val = store.src[0].src[1], RKArg(RKBufferKind.ARG, oslot), store.src[1]
  # Pure half EW only: reject CAST / float ALU / REDUCE; half ALU must be in code_for_op.
  supported = RockchipRenderer.code_for_op
  if any(u.op is Ops.CAST or u.op is Ops.REDUCE or
         (u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.float, dtypes.float32, dtypes.float64)) for u in uops):
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  if any(u.op in GroupOp.ALU and u.op not in supported and u.dtype.scalar() is dtypes.half for u in uops):
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  # Schedule half code_for_op tree; leaves = same-index LOAD, gather LOAD, or scalar CONST.
  order:list[UOp] = []
  def visit(u:UOp) -> bool:
    if u.op in supported and u.dtype.scalar() is dtypes.half:
      if not all(visit(s) for s in u.src): return False
      if u not in order: order.append(u)
      return True
    return _ew_leaf(u, out_index, count, oslot) is not None
  if not visit(val) or not order or order[-1] is not val:
    bad = _unsupported_ew_ops(uops, out_index, count, oslot, supported)
    raise RuntimeError(f"RKPLAN_REJECT:unsupported_graph {bad}")
  # Gemm-style Σ MUL: DPU EW for MULs, float32 host sum for the ADD reduce (half ADD misses fp16 tol).
  mul_red = _mul_reduction_muls(val)
  host_sum:RKHostSum|None = None
  if mul_red is not None and len(mul_red) >= 2:
    order = list(dict.fromkeys(mul_red))  # MULs only, stable unique
  uses = {u: sum(s is u for n in order for s in n.src) for u in order}
  values:dict[UOp, RKArg] = {}
  free:list[int] = []
  const_scratch:dict[bytes, int] = {}  # fp16 bits key (nan-safe)
  gather_scratch:dict[tuple[int, tuple[int, ...]], int] = {}
  gathers:list[RKGather] = []
  # Const scratches first (runtime splats constants[i] → scratch[i]); gathers after.
  for expr in order:
    for s in expr.src:
      leaf = _ew_leaf(s, out_index, count, oslot)
      if isinstance(leaf, float):
        k = struct.pack("<e", leaf)
        if k not in const_scratch: const_scratch[k] = len(const_scratch)
  scratch_count = len(const_scratch)
  stages:list[RKStage] = []
  def operand(s:UOp) -> RKArg:
    nonlocal scratch_count
    if s in values: return values[s]
    leaf = _ew_leaf(s, out_index, count, oslot)
    assert leaf is not None
    if isinstance(leaf, float): return RKArg(RKBufferKind.SCRATCH, const_scratch[struct.pack("<e", leaf)])
    if isinstance(leaf, RKArg): return leaf
    p, ix = leaf
    offsets = _gather_offsets(out_index, ix, count)
    if max(offsets) >= int(p.src[0].arg): raise RuntimeError("RKPLAN_REJECT:gather_index")
    key = (p.arg.slot, offsets)
    if key not in gather_scratch:
      gather_scratch[key] = scratch_count
      gathers.append(RKGather(p.arg.slot, scratch_count, offsets))
      scratch_count += 1
    return RKArg(RKBufferKind.SCRATCH, gather_scratch[key])
  prod_scratches:list[int] = []
  for expr in order:
    lhs, rhs = operand(expr.src[0]), operand(expr.src[1])
    if mul_red is not None and len(mul_red) >= 2:
      # MUL into scratch; host floats-sum writes `out`
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
      prod_scratches.append(slot)
    elif expr is val: dst = out
    elif (reuse:=next((values[x] for x in expr.src if x in values and uses[x] == 1 and
                       values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    stages.append(emit_ew_stage(len(stages), dst, lhs, rhs, count, _EW_CFG[expr.op]))
    values[expr] = dst
    for dep in expr.src:
      if dep in uses:
        uses[dep] -= 1
        arg = values[dep]
        if uses[dep] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  if prod_scratches: host_sum = RKHostSum(out, tuple(prod_scratches), count)
  nbytes = ((count + 7) // 8 * 8) * 2
  constants = b""
  if const_scratch:
    by = {slot: bits for bits, slot in const_scratch.items()}
    constants = b"".join(by[i] if i in by else struct.pack("<e", 0.0)
                         for i in range(max(const_scratch.values()) + 1))
  return RKImage(RKTarget.RK3588, tuple(stages), tuple(RKScratch(nbytes) for _ in range(scratch_count)),
                 constants, gathers=tuple(gathers), host_sum=host_sum)
class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

# Other branch demotes float ALU→half via PatternMatcher. Apply here (post-SPEC) so WHERE/float graphs stay valid through verify.
_pm_fp32_to_fp16 = PatternMatcher([
  (UPat(Ops.ADD, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
  (UPat(Ops.MAX, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.MAX, x.src[1].cast(dtypes.half))),
  (UPat(Ops.NEG, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.NEG)),
  (UPat(Ops.EXP2, dtypes.float, name="x"),
   lambda x: x.src[0].cast(dtypes.half).alu(Ops.EXP2)),
  # Fold ACC cast noise: half ← float ← half MUL/ADD; half ← half.
  (UPat(Ops.CAST, dtypes.half, src=(UPat(Ops.CAST, dtypes.float, src=(UPat(dtype=dtypes.half, name="x"),)),)),
   lambda x: x),
  (UPat(Ops.CAST, dtypes.half, src=(UPat(dtype=dtypes.half, name="x"),)), lambda x: x),
])

def _fp16_rewrite(uops:list[UOp]) -> list[UOp]:
  sink = next(u for u in uops if u.op is Ops.SINK)
  return list(graph_rewrite(sink, _pm_fp32_to_fp16, name="rockchip float→half").toposort())

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  # Keys feed codegen supported_ops (decomp); must match DPU EW ops in _EW_CFG.
  code_for_op = {Ops.ADD: lambda: None, Ops.MUL: lambda: None}
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_ew(_fp16_rewrite(uops)))).decode()
