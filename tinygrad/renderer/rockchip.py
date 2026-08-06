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
from tinygrad.uop.ops import Ops, UOp

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

_DPU, _RDMA, _PC, _EW_ADD = 0x1001, 0x2001, 0x81, 0x108002c0|(2<<16)
def _cmd(t,r,v): return ((t&0xffff)<<48)|((v&0xffffffff)<<16)|(r&0xffff)

def emit_add(dst:RKArg, lhs:RKArg, rhs:RKArg, count:int) -> RKImage:
  w = (count+7)//8-1
  regs = ((_DPU,rk.REG_DPU_S_POINTER,0xe),(_DPU,rk.REG_DPU_FEATURE_MODE_CFG,0x1e5),
    (_DPU,rk.REG_DPU_DATA_FORMAT,(2<<29)|(2<<26)|2),(_DPU,rk.REG_DPU_DATA_CUBE_WIDTH,w),
    (_DPU,rk.REG_DPU_DATA_CUBE_HEIGHT,0),(_DPU,rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,0),(_DPU,rk.REG_DPU_DATA_CUBE_CHANNEL,0x70007),
    (_DPU,rk.REG_DPU_BS_CFG,0x53),(_DPU,rk.REG_DPU_BN_CFG,0x53),(_DPU,rk.REG_DPU_BS_ALU_CFG,0),(_DPU,rk.REG_DPU_BS_MUL_CFG,0),
    (_DPU,rk.REG_DPU_BS_OW_CFG,2),(_DPU,rk.REG_DPU_WDMA_SIZE_0,7),(_DPU,rk.REG_DPU_WDMA_SIZE_1,w),
    (_DPU,rk.REG_DPU_BN_MUL_CFG,0),(_DPU,rk.REG_DPU_BN_RELUX_CMP_VALUE,0),(_DPU,rk.REG_DPU_EW_CFG,_EW_ADD),
    (_DPU,rk.REG_DPU_EW_CVT_SCALE_VALUE,1),(_DPU,rk.REG_DPU_OUT_CVT_OFFSET,0),(_DPU,rk.REG_DPU_OUT_CVT_SCALE,0x10001),
    (_DPU,rk.REG_DPU_OUT_CVT_SHIFT,0),(_DPU,rk.REG_DPU_SURFACE_ADD,0x40),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_S_POINTER,0xe),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,w),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,0),(_RDMA,rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,7),
    (_RDMA,rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,0x40000008))
  cmds = [_cmd(*x) for x in regs]; relocs = []
  for t,r,a in ((_DPU,rk.REG_DPU_DST_BASE_ADDR,dst),(_RDMA,rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,lhs),
               (_RDMA,rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,rhs)):
    relocs.append(RKReloc(0, len(cmds), a.kind, a.index, a.addend)); cmds.append(_cmd(t, r, 0))
  cmds += [_cmd(_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), _cmd(_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  return RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET),))

def _root_param(u:UOp) -> UOp|None:
  while u.op is not Ops.PARAM:
    if not u.src: return None
    u = u.src[0]
  return u

def lower_add(uops:list[UOp]) -> RKImage:
  stores = [u for u in uops if u.op is Ops.STORE]
  outs = [_root_param(u.src[0]) for u in stores]
  if (not stores or any(p is None or p.dtype.scalar() is not dtypes.half or p.src[0].op is not Ops.CONST for p in outs) or
      len({p.arg.slot for p in outs}) != 1): raise RuntimeError("RKPLAN_REJECT:unsupported_graph")  # type: ignore[union-attr]
  out = outs[0]; assert out is not None
  count, oslot = int(out.src[0].arg), out.arg.slot
  ins = tuple(dict.fromkeys(p.arg.slot for u in uops if u.op is Ops.LOAD and (p:=_root_param(u)) is not None
                            and p.dtype.scalar() is dtypes.half and p.arg.slot != oslot))
  if count <= 0 or len(ins) != 2 or not any(u.op is Ops.ADD and u.dtype.scalar() is dtypes.half for u in uops):
    raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
  return emit_add(RKArg(RKBufferKind.ARG, oslot), RKArg(RKBufferKind.ARG, ins[0]), RKArg(RKBufferKind.ARG, ins[1]), count)

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  compiler = RockchipCompiler("rockchip")
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def render(self, uops:list[UOp]) -> str: return base64.b64encode(encode_image(lower_add(uops))).decode()
