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

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 3, 1
_HEADER, _STAGE = struct.Struct("<4sHHHHHHIII"), struct.Struct("<BBHIIII")
_RELOC, _SCRATCH = struct.Struct("<HHBBIqIH"), struct.Struct("<II")

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
class RKImage:
  target: RKTarget; stages: tuple[RKStage, ...]
  scratch: tuple[RKScratch, ...] = (); constants: bytes = b""; version: int = RKIMAGE_VERSION

@dataclass(frozen=True)
class RKArg: kind: RKBufferKind; index: int; addend: int = 0

def encode_image(image:RKImage) -> bytes:
  cmds:list[int] = []; relocs:list[RKReloc] = []; rows:list[tuple] = []
  for s in image.stages:
    c0, r0 = len(cmds), len(relocs); cmds.extend(s.commands); relocs.extend(s.relocs)
    rows.append((int(s.engine), s.flags, 0, c0, len(s.commands), r0, len(s.relocs)))
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(rows), len(relocs),
                               len(image.scratch), 0, len(cmds), len(image.constants), 0))
  for row in rows: out += _STAGE.pack(*row)
  for r in relocs: out += _RELOC.pack(r.stage, r.word, int(r.kind), r.shift, r.index, r.addend, r.mask, r.field_shift)
  for s in image.scratch: out += _SCRATCH.pack(s.size, s.alignment)
  return bytes(out) + (struct.pack(f"<{len(cmds)}Q", *cmds) if cmds else b"") + image.constants

def decode_image(blob:bytes) -> RKImage:
  magic, ver, target, nstage, nreloc, nscratch, res, ncmd, nconst, res2 = _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or res or res2: raise ValueError("invalid RKImage header")
  if _HEADER.size + nstage*_STAGE.size + nreloc*_RELOC.size + nscratch*_SCRATCH.size + ncmd*8 + nconst != len(blob):
    raise ValueError("invalid RKImage size")
  off = _HEADER.size
  rows = [_STAGE.unpack_from(blob, off+i*_STAGE.size) for i in range(nstage)]; off += nstage*_STAGE.size
  relocs = []
  for _ in range(nreloc):
    st, word, kind, shift, index, addend, mask, fshift = _RELOC.unpack_from(blob, off); off += _RELOC.size
    relocs.append(RKReloc(st, word, RKBufferKind(kind), index, addend, shift, mask, fshift))
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(nscratch)); off += nscratch*_SCRATCH.size
  cmds = struct.unpack_from(f"<{ncmd}Q", blob, off) if ncmd else (); off += ncmd*8
  stages = []
  for i,(eng,flags,r0,c0,clen,rstart,rlen) in enumerate(rows):
    if r0 or c0+clen > ncmd or rstart+rlen > nreloc: raise ValueError("invalid RKImage stage")
    stages.append(RKStage(RKEngine(eng), cmds[c0:c0+clen], tuple(relocs[rstart:rstart+rlen]), flags))
  return RKImage(RKTarget(target), tuple(stages), scratch, blob[off:], ver)

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
# EW_CFG from allbilly/rk3588 examples/elementwise.py
_EW = {Ops.ADD: 0x108002c0 | (2 << 16), Ops.MUL: 0x108002c0 | (1 << 2) | (1 << 8)}
# Non-EW half ALU / mask ops → reject (broadcast/pad/logaddexp must not false-accept).
_FORBIDDEN = {Ops.SUB, Ops.FDIV, Ops.MAX, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
              Ops.AND, Ops.OR, Ops.XOR, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT}
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

def lower_ew(uops:list[UOp]) -> RKImage:
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out = outs[0]; assert out is not None
  count, oslot = int(out.src[0].arg), out.arg.slot
  # Pure half EW only: reject CAST / float ALU / REDUCE (stops matmul ACC-loop false-accept as EW MUL).
  if any(u.op is Ops.CAST or u.op is Ops.REDUCE or
         (u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.float, dtypes.float32, dtypes.float64)) for u in uops):
    raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  if any(u.op in _FORBIDDEN and u.dtype.scalar() is dtypes.half for u in uops):
    raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  ew_ops = {u.op for u in uops if u.op in _EW and u.dtype.scalar() is dtypes.half}
  # After float→half demote, unrolled matmul is half MUL+ADD MAC tree (not a single EW op).
  if len(ew_ops) != 1: raise RuntimeError(f"RKPLAN_REJECT:mixed_ew_ops:{sorted(o.name for o in ew_ops)}")
  op = next(iter(ew_ops))
  # Contiguous same-shape vector LOADs; optional ConstFloat scalar operand (test_scalar_mul / mul_naninf).
  ins:list[int] = []
  const_vals:list[float] = []
  for u in uops:
    if u.op is op and u.dtype.scalar() is dtypes.half:
      for s in u.src:
        if s.op is Ops.CONST: const_vals.append(float(s.arg))
    if u.op is not Ops.LOAD: continue
    p = _root_param(u)
    if p is None or p.dtype.scalar() is not dtypes.half or p.arg.slot == oslot: continue
    if p.src[0].op is not Ops.CONST or int(p.src[0].arg) != count: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
    if p.arg.slot not in ins: ins.append(p.arg.slot)
  if count <= 0: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  out_arg = RKArg(RKBufferKind.ARG, oslot)
  if len(ins) >= 2: return emit_ew_chain(out_arg, tuple(RKArg(RKBufferKind.ARG, s) for s in ins), count, _EW[op])
  if len(ins) == 1 and const_vals:
    return emit_ew_const(out_arg, RKArg(RKBufferKind.ARG, ins[0]), count, _EW[op], const_vals[0])
  raise RuntimeError("RKPLAN_REJECT:unsupported_graph")

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
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_ew(_fp16_rewrite(uops)))).decode()
