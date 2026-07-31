from __future__ import annotations
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 1, 1
_HEADER = struct.Struct("<4sHHHHHHIII")
_STAGE = struct.Struct("<BBHIIIIIQQ")
_RELOC = struct.Struct("<HHBBHqII")
_SCRATCH = struct.Struct("<II")

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum):
  DPU = 1
  CMAC = 2
  CONV = 3
  PPU = 4
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
class RKDPUOp(IntEnum):
  COPY = 0
  ADD = 1
  MUL = 2
  MAX = 3
  SUB = 4

@dataclass(frozen=True)
class RKReloc:
  stage: int
  word: int
  kind: RKBufferKind
  index: int
  addend: int = 0
  shift: int = 0
  mask: int = 0xffffffff
  field_shift: int = 0

@dataclass(frozen=True)
class RKScratch:
  size: int
  alignment: int = 4096

@dataclass(frozen=True)
class RKStage:
  engine: RKEngine
  commands: tuple[int, ...]
  relocs: tuple[RKReloc, ...] = ()
  reads: tuple[int, ...] = ()
  writes: tuple[int, ...] = ()
  dependencies: int = 0
  flags: int = 0

@dataclass(frozen=True)
class RKImage:
  target: RKTarget
  stages: tuple[RKStage, ...]
  scratch: tuple[RKScratch, ...] = ()
  constants: bytes = b""
  version: int = RKIMAGE_VERSION

@dataclass(frozen=True)
class RKArg:
  kind: RKBufferKind
  index: int

@dataclass(frozen=True)
class RKDPUStage:
  op: RKDPUOp
  dst: RKArg
  lhs: RKArg|float
  rhs: RKArg|float|None
  count: int

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...]

@dataclass(frozen=True)
class _DPUExpr:
  op: RKDPUOp
  src: tuple[_DPUExpr|RKArg|float, ...]

def _slot_mask(slots:tuple[int, ...]) -> int:
  if any(x < 0 or x >= 64 for x in slots): raise ValueError("RKImage supports argument slots 0..63")
  return sum(1 << x for x in slots)

def validate_image(image:RKImage) -> None:
  if image.version != RKIMAGE_VERSION: raise ValueError(f"unsupported RKImage version {image.version}")
  if len(image.stages) > 64: raise ValueError("too many RKImage stages")
  for stage_idx, stage in enumerate(image.stages):
    if stage.dependencies >> stage_idx: raise ValueError("stage dependency must refer to an earlier stage")
    _slot_mask(stage.reads), _slot_mask(stage.writes)
    for reloc in stage.relocs:
      if reloc.stage != stage_idx or not 0 <= reloc.word < len(stage.commands): raise ValueError("invalid relocation location")
      if reloc.index < 0 or reloc.shift < 0 or reloc.field_shift < 0 or reloc.mask >> 32: raise ValueError("invalid relocation field")
  for scratch in image.scratch:
    if scratch.size < 0 or scratch.alignment <= 0 or scratch.alignment & (scratch.alignment-1): raise ValueError("invalid scratch declaration")

def encode_image(image:RKImage) -> bytes:
  validate_image(image)
  commands:list[int] = []
  relocs:list[RKReloc] = []
  stage_rows:list[tuple[int, ...]] = []
  for stage in image.stages:
    command_start, reloc_start = len(commands), len(relocs)
    commands.extend(stage.commands)
    relocs.extend(stage.relocs)
    stage_rows.append((int(stage.engine), stage.flags, 0, stage.dependencies, command_start, len(stage.commands), reloc_start, len(stage.relocs),
                       _slot_mask(stage.reads), _slot_mask(stage.writes)))
  out = bytearray(_HEADER.pack(RKIMAGE_MAGIC, image.version, int(image.target), len(image.stages), len(relocs), len(image.scratch), 0,
                               len(commands), len(image.constants), 0))
  for engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes in stage_rows:
    out += _STAGE.pack(engine, flags, reserved, deps, command_start, command_count, reloc_start, reloc_count, reads, writes)
  for reloc in relocs:
    out += _RELOC.pack(reloc.stage, reloc.word, int(reloc.kind), reloc.shift, reloc.index, reloc.addend, reloc.mask, reloc.field_shift)
  for scratch in image.scratch: out += _SCRATCH.pack(scratch.size, scratch.alignment)
  if commands: out += struct.pack(f"<{len(commands)}Q", *commands)
  return bytes(out) + image.constants

def decode_image(blob:bytes) -> RKImage:
  if len(blob) < _HEADER.size: raise ValueError("truncated RKImage header")
  magic, version, target, stage_count, reloc_count, scratch_count, reserved, command_count, constant_size, reserved2 = _HEADER.unpack_from(blob)
  if magic != RKIMAGE_MAGIC or reserved or reserved2: raise ValueError("invalid RKImage header")
  expected = _HEADER.size + stage_count*_STAGE.size + reloc_count*_RELOC.size + scratch_count*_SCRATCH.size + command_count*8 + constant_size
  if expected != len(blob): raise ValueError("invalid RKImage size")
  off, rows = _HEADER.size, []
  for _ in range(stage_count):
    rows.append(_STAGE.unpack_from(blob, off))
    off += _STAGE.size
  relocs = []
  for _ in range(reloc_count):
    stage, word, kind, shift, index, addend, mask, field_shift = _RELOC.unpack_from(blob, off)
    off += _RELOC.size
    relocs.append(RKReloc(stage, word, RKBufferKind(kind), index, addend, shift, mask, field_shift))
  scratch = tuple(RKScratch(*_SCRATCH.unpack_from(blob, off+i*_SCRATCH.size)) for i in range(scratch_count))
  off += scratch_count*_SCRATCH.size
  commands = struct.unpack_from(f"<{command_count}Q", blob, off) if command_count else ()
  off += command_count*8
  stages = []
  def slots(mask:int) -> tuple[int, ...]: return tuple(x for x in range(64) if mask & (1 << x))
  for idx, (engine, flags, row_reserved, deps, command_start, command_len, reloc_start, reloc_len, reads, writes) in enumerate(rows):
    if row_reserved or command_start+command_len > command_count or reloc_start+reloc_len > reloc_count: raise ValueError("invalid RKImage stage")
    stages.append(RKStage(RKEngine(engine), tuple(commands[command_start:command_start+command_len]),
                          tuple(relocs[reloc_start:reloc_start+reloc_len]), slots(reads), slots(writes), deps, flags))
    if any(r.stage != idx for r in stages[-1].relocs): raise ValueError("relocation belongs to wrong stage")
  image = RKImage(RKTarget(target), tuple(stages), scratch, blob[off:], version)
  validate_image(image)
  return image

def patch_image(image:RKImage, address:Callable[[RKBufferKind, int], int]) -> tuple[tuple[int, ...], ...]:
  validate_image(image)
  patched = [list(stage.commands) for stage in image.stages]
  for stage in image.stages:
    for reloc in stage.relocs:
      word = patched[reloc.stage][reloc.word]
      value = (word >> 16) & 0xffffffff
      field = (((address(reloc.kind, reloc.index) + reloc.addend) >> reloc.shift) << reloc.field_shift) & reloc.mask
      patched[reloc.stage][reloc.word] = (word & ~0xffffffff0000) | (((value & ~reloc.mask) | field) << 16)
  return tuple(tuple(stage) for stage in patched)

def _unwrap_same_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype is u.src[0].dtype: u = u.src[0]
  return u

def _parse_dpu_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float]) -> _DPUExpr|RKArg|float|None:
  u = _unwrap_same_cast(u)
  if u in memo: return memo[u]
  if u.op is Ops.INDEX and u.dtype is dtypes.half and u.src[0].op is Ops.PARAM and u.src[1].key == output_index.key:
    ret:RKArg|float|_DPUExpr = RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  elif u.op is Ops.CONST and isinstance(u.arg, (int, float)): ret = float(u.arg)
  elif u.op in (Ops.ADD, Ops.MUL, Ops.MAX):
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in src): return None
    ret = _DPUExpr({Ops.ADD:RKDPUOp.ADD, Ops.MUL:RKDPUOp.MUL, Ops.MAX:RKDPUOp.MAX}[u.op], src)  # type: ignore[arg-type]
  else: return None
  memo[u] = ret
  return ret

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Lower one contiguous fp16 store to a UOp-free primitive DPU plan."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1 or (store:=stores[0]).src[0].op is not Ops.INDEX or store.src[0].dtype is not dtypes.half: return None
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op is not Ops.RANGE or out_param.src[0].op is not Ops.CONST: return None
  count = int(out_param.src[0].arg)
  if count <= 0 or int(out_index.src[0].arg) != count: return None
  root = _parse_dpu_expr(store.src[1], out_index, {})
  if root is None: return None
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if not isinstance(root, _DPUExpr):
    stage = RKDPUStage(RKDPUOp.ADD, output, 0.0, root, count) if isinstance(root, float) else RKDPUStage(RKDPUOp.COPY, output, root, None, count)
    return RKDPUProgram((stage,), ())
  order:list[_DPUExpr] = []
  def visit(expr:_DPUExpr) -> None:
    for src in expr.src:
      if isinstance(src, _DPUExpr) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  visit(root)
  uses = {expr:sum(src is expr for node in order for src in node.src) for expr in order}
  values:dict[_DPUExpr, RKArg] = {}
  free:list[int] = []
  scratch_count, stages = 0, []
  for expr in order:
    src = tuple(values[x] if isinstance(x, _DPUExpr) else x for x in expr.src)
    if expr is root: dst = output
    elif (reuse:=next((values[x] for x in expr.src if isinstance(x, _DPUExpr) and uses[x] == 1 and
                       values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else scratch_count
      if slot == scratch_count: scratch_count += 1
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    stages.append(RKDPUStage(expr.op, dst, src[0], src[1], count))
    values[expr] = dst
    for source in expr.src:
      if isinstance(source, _DPUExpr):
        uses[source] -= 1
        arg = values[source]
        if uses[source] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  size = ((count+7)//8)*16
  return RKDPUProgram(tuple(stages), tuple(RKScratch(size) for _ in range(scratch_count)))

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_EW_BASE = 0x108002c0
_EW_CFG = {RKDPUOp.ADD:_EW_BASE | (2 << 16), RKDPUOp.MUL:_EW_BASE | (1 << 2) | (1 << 8),
           RKDPUOp.MAX:_EW_BASE, RKDPUOp.SUB:_EW_BASE | (4 << 16)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def emit_dpu(program:RKDPUProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  constants, constant_offsets, stages = bytearray(), {}, []
  def materialize(value:RKArg|float, count:int) -> RKArg:
    if isinstance(value, RKArg): return value
    bits = struct.pack("<e", value)
    key = bits, count
    if key not in constant_offsets:
      constant_offsets[key] = len(constants)
      constants.extend(bits * (((count+7)//8)*8))
    return RKArg(RKBufferKind.CONSTANT, constant_offsets[key])
  for stage_idx, plan in enumerate(program.stages):
    lhs, rhs = materialize(plan.lhs, plan.count), materialize(plan.rhs, plan.count) if plan.rhs is not None else None
    width = (plan.count+7)//8-1
    cmds = [_command(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x0e), _command(_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
            _command(_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002), _command(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width),
            _command(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0), _command(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
            _command(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0 if plan.op is RKDPUOp.COPY else _EW_CFG[plan.op]),
            _command(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x0e),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008),
            _command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
            _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0)]
    relocs = [RKReloc(stage_idx, 13, plan.dst.kind, plan.dst.index), RKReloc(stage_idx, 14, lhs.kind, lhs.index)]
    if rhs is not None:
      cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0))
      relocs.append(RKReloc(stage_idx, 15, rhs.kind, rhs.index))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    operands = (lhs,) if rhs is None else (lhs, rhs)
    reads = tuple(sorted({x.index for x in operands if x.kind is RKBufferKind.ARG}))
    writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half}
  def native_program(self, ast:UOp) -> UOp|None:
    if (plan:=lower_dpu(ast)) is None: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
    return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""),
      UOp(Ops.BINARY, arg=encode_image(emit_dpu(plan)))), arg=ProgramInfo.from_sink(ast, self.target))
