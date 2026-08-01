from __future__ import annotations
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, cast
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk, rockchip_lut as rklut
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKIMAGE_MAGIC, RKIMAGE_VERSION, RK_STAGE_RESET = b"RKIM", 3, 1
_HEADER = struct.Struct("<4sHHHHHHIII")
_STAGE = struct.Struct("<BBHQIIIIQQ")
_RELOC = struct.Struct("<HHBBIqIH")
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
  EXP2 = 5
  MASK = 6
  DIV = 7

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
  addend: int = 0

@dataclass(frozen=True)
class RKDPUStage:
  op: RKDPUOp
  dst: RKArg
  lhs: RKArg|float
  rhs: RKArg|float|None
  count: int
  out_dtype: DType = dtypes.half

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...]

@dataclass(frozen=True)
class RKView:
  slot: int
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  offset: int = 0
  kind: RKBufferKind = RKBufferKind.ARG

@dataclass(frozen=True)
class RKContract:
  out: RKView
  lhs: RKView
  rhs: RKView
  reduce_axes: tuple[int, ...]

@dataclass(frozen=True)
class RKPool:
  out: RKView
  inp: RKView
  kernel: tuple[int, int]

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
      if reloc.index < 0 or reloc.index >> 32 or not 0 <= reloc.shift < 64 or not 0 <= reloc.field_shift < 32 or reloc.mask >> 32:
        raise ValueError("invalid relocation field")
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
      field = ((address(reloc.kind, reloc.index) + reloc.addend) >> reloc.shift) & reloc.mask
      field_mask = (reloc.mask << reloc.field_shift) & 0xffffffff
      patched[reloc.stage][reloc.word] = (word & ~0xffffffff0000) | (((value & ~field_mask) | ((field << reloc.field_shift) & field_mask)) << 16)
  return tuple(tuple(stage) for stage in patched)

def _unwrap_same_cast(u:UOp) -> UOp:
  while u.op is Ops.CAST and u.dtype is u.src[0].dtype: u = u.src[0]
  return u

def _canonical_abs(u:UOp) -> UOp|None:
  u = _unwrap_same_cast(u)
  if u.op is not Ops.MUL: return None
  for data, sign in (u.src, u.src[::-1]):
    data, sign = _unwrap_same_cast(data), _unwrap_same_cast(sign)
    if data.op is not Ops.INDEX or sign.op is not Ops.WHERE: continue
    cond, nonzero, zero = _unwrap_same_cast(sign.src[0]), _unwrap_same_cast(sign.src[1]), _unwrap_same_cast(sign.src[2])
    if cond.op is not Ops.CMPNE or zero.op is not Ops.CONST or float(zero.arg) != 0 or nonzero.op is not Ops.WHERE: continue
    less, negative, positive = (_unwrap_same_cast(x) for x in nonzero.src)
    compared = tuple(_unwrap_same_cast(x) for x in cond.src)
    if (less.op is Ops.CMPLT and compared[0].key == data.key and compared[1].op is Ops.CONST and float(compared[1].arg) == 0 and
        _unwrap_same_cast(less.src[0]).key == data.key and _unwrap_same_cast(less.src[1]).op is Ops.CONST and
        float(_unwrap_same_cast(less.src[1]).arg) == 0 and negative.op is Ops.CONST and float(negative.arg) == -1 and
        positive.op is Ops.CONST and float(positive.arg) == 1): return data
  return None

def _parse_dpu_expr(u:UOp, output_index:UOp, memo:dict[UOp, _DPUExpr|RKArg|float]) -> _DPUExpr|RKArg|float|None:
  u = _unwrap_same_cast(u)
  if u in memo: return memo[u]
  if u.op is Ops.INDEX and u.dtype is dtypes.half and u.src[0].op is Ops.PARAM and u.src[1].key == output_index.key:
    ret:RKArg|float|_DPUExpr = RKArg(RKBufferKind.ARG, u.src[0].arg.slot)
  elif u.op is Ops.CONST and isinstance(u.arg, (int, float)): ret = float(u.arg)
  elif (abs_input:=_canonical_abs(u)) is not None:
    operand = _parse_dpu_expr(abs_input, output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.MAX, (operand, _DPUExpr(RKDPUOp.MUL, (operand, -1.0))))
  elif u.op is Ops.MUL and any(_unwrap_same_cast(x).op is Ops.RECIPROCAL for x in u.src):
    reciprocal = next(i for i,x in enumerate(u.src) if _unwrap_same_cast(x).op is Ops.RECIPROCAL)
    numerator, denominator = u.src[1-reciprocal], _unwrap_same_cast(u.src[reciprocal]).src[0]
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in (numerator, denominator))
    if any(x is None for x in src): return None
    ret = _DPUExpr(RKDPUOp.DIV, cast(tuple[_DPUExpr|RKArg|float, ...], src))
  elif u.op in (Ops.ADD, Ops.MUL, Ops.MAX):
    src = tuple(_parse_dpu_expr(x, output_index, memo) for x in u.src)
    if any(x is None for x in src): return None
    ret = _DPUExpr({Ops.ADD:RKDPUOp.ADD, Ops.MUL:RKDPUOp.MUL, Ops.MAX:RKDPUOp.MAX}[u.op], src)  # type: ignore[arg-type]
  elif u.op is Ops.EXP2:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.EXP2, (operand,))
  elif u.op is Ops.RECIPROCAL:
    operand = _parse_dpu_expr(u.src[0], output_index, memo)
    if operand is None: return None
    ret = _DPUExpr(RKDPUOp.DIV, (1.0, operand))
  elif u.op is Ops.WHERE and (cond:=_unwrap_same_cast(u.src[0])).op is Ops.CMPLT:
    lhs_u, rhs_u = (_unwrap_same_cast(x) for x in cond.src)
    true_u, false_u = (_unwrap_same_cast(x) for x in u.src[1:])
    ordered_max = true_u.key == rhs_u.key and false_u.key == lhs_u.key
    ordered_min = true_u.key == lhs_u.key and false_u.key == rhs_u.key
    parsed = tuple(_parse_dpu_expr(x, output_index, memo) for x in (lhs_u, rhs_u))
    if any(x is None for x in parsed): return None
    operands = cast(tuple[_DPUExpr|RKArg|float, ...], parsed)
    if ordered_max: ret = _DPUExpr(RKDPUOp.MAX, operands)
    elif ordered_min:
      negative = tuple(_DPUExpr(RKDPUOp.MUL, (x, -1.0)) for x in operands)
      ret = _DPUExpr(RKDPUOp.MUL, (_DPUExpr(RKDPUOp.MAX, negative), -1.0))
    else:
      arms = tuple(_parse_dpu_expr(x, output_index, memo) for x in (true_u, false_u))
      if any(x is None for x in arms): return None
      true, false = cast(tuple[_DPUExpr|RKArg|float, _DPUExpr|RKArg|float], arms)
      mask = _DPUExpr(RKDPUOp.MASK, (_DPUExpr(RKDPUOp.SUB, (operands[1], operands[0])),))
      ret = _DPUExpr(RKDPUOp.ADD, (_DPUExpr(RKDPUOp.MUL, (true, mask)),
        _DPUExpr(RKDPUOp.MUL, (false, _DPUExpr(RKDPUOp.SUB, (1.0, mask))))))
  else: return None
  memo[u] = ret
  return ret

def lower_dpu(sink:UOp) -> RKDPUProgram|None:
  """Lower one contiguous fp16/int32 store to a UOp-free primitive DPU plan."""
  stores = [u for u in sink.toposort() if u.op is Ops.STORE]
  if len(stores) != 1 or (store:=stores[0]).src[0].op is not Ops.INDEX or store.src[0].dtype not in (dtypes.half, dtypes.int): return None
  out_index, out_param = store.src[0].src[1], store.src[0].src[0]
  if out_param.op is not Ops.PARAM or out_index.op not in (Ops.RANGE, Ops.CONST) or out_param.src[0].op is not Ops.CONST: return None
  count = int(out_param.src[0].arg)
  if count <= 0 or (out_index.op is Ops.RANGE and int(out_index.src[0].arg) != count) or \
     (out_index.op is Ops.CONST and (count != 1 or int(out_index.arg) != 0)): return None
  root = _parse_dpu_expr(store.src[1], out_index, {})
  if root is None: return None
  # Native int32 WDMA emits four values per eight-lane fp16 atom. Constant
  # fills can safely double their source lanes; dynamic conversion needs an
  # explicit packed layout and is deliberately not inferred here.
  if store.src[0].dtype is dtypes.int and not isinstance(root, float): return None
  output = RKArg(RKBufferKind.ARG, out_param.arg.slot)
  if not isinstance(root, _DPUExpr):
    if store.src[0].dtype is dtypes.int:
      fill_stages = tuple(RKDPUStage(RKDPUOp.ADD, RKArg(output.kind, output.index, start*4), 0.0, root,
                                    min(64, count-start), dtypes.int) for start in range(0, count, 64))
      return RKDPUProgram(fill_stages, ()) if len(fill_stages) <= 64 else None
    stage = RKDPUStage(RKDPUOp.ADD, output, 0.0, root, count, store.src[0].dtype)
    return RKDPUProgram((stage,), ())
  order:list[_DPUExpr] = []
  def visit(expr:_DPUExpr) -> None:
    for src in expr.src:
      if isinstance(src, _DPUExpr) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  visit(root)
  if any(expr.op is RKDPUOp.EXP2 for expr in order) and count != 128: return None
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
    stages.append(RKDPUStage(expr.op, dst, src[0], src[1] if len(src) > 1 else None, count,
                             store.src[0].dtype if expr is root else dtypes.half))
    values[expr] = dst
    for source in expr.src:
      if isinstance(source, _DPUExpr):
        uses[source] -= 1
        arg = values[source]
        if uses[source] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  size = ((count+7)//8)*16
  return RKDPUProgram(tuple(stages), tuple(RKScratch(size) for _ in range(scratch_count)))

def _strip_casts(u:UOp) -> UOp:
  while u.op is Ops.CAST: u = u.src[0]
  return u

def _affine(u:UOp) -> tuple[dict[int, int], int]|None:
  if u.op is Ops.RANGE: return ({u.arg[0]:1}, 0)
  if u.op is Ops.CONST: return ({}, int(u.arg))
  if u.op is Ops.ADD:
    a, b = _affine(u.src[0]), _affine(u.src[1])
    if a is None or b is None: return None
    return ({k:a[0].get(k, 0)+b[0].get(k, 0) for k in a[0].keys()|b[0].keys()}, a[1]+b[1])
  if u.op is Ops.MUL:
    const, value = (u.src[0], u.src[1]) if u.src[0].op is Ops.CONST else (u.src[1], u.src[0])
    if const.op is not Ops.CONST or (aff:=_affine(value)) is None: return None
    return ({k:v*int(const.arg) for k,v in aff[0].items()}, aff[1]*int(const.arg))
  return None

def lower_contract(sink:UOp) -> RKContract|None:
  """Recognize directly legal M=1, K=32 affine contractions and row sums."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return None
  store, reduce = stores[0], reductions[0]
  if store.src[0].op is not Ops.INDEX or store.src[0].dtype is not dtypes.half or reduce.arg[0] is not Ops.ADD or len(reduce.src) != 2: return None
  body, red = _strip_casts(reduce.src[0]), reduce.src[1]
  out_param, out_aff = store.src[0].src[0], _affine(store.src[0].src[1])
  if out_param.op is not Ops.PARAM or out_aff is None or out_aff[1] or len(out_aff[0]) != 1: return None
  if body.op is Ops.INDEX:
    inp_aff = _affine(body.src[1])
    out_axis, red_axis, n = next(iter(out_aff[0])), red.arg[0], int(out_param.src[0].arg)
    if (body.dtype is not dtypes.half or body.src[0].op is not Ops.PARAM or red.op is not Ops.RANGE or int(red.src[0].arg) != 32 or
        not 4 <= n <= 16 or out_aff[0] != {out_axis:1} or inp_aff != ({out_axis:32, red_axis:1}, 0) or
        int(body.src[0].src[0].arg) != n*32): return None
    ones = RKView(0, (1,32), (32,1), kind=RKBufferKind.CONSTANT)
    return RKContract(RKView(out_param.arg.slot, (1,n), (n,1)), ones,
                      RKView(body.src[0].arg.slot, (n,32), (32,1)), (red_axis,))
  if body.op is not Ops.MUL or red.op is not Ops.RANGE or int(red.src[0].arg) != 32: return None
  lhs, rhs = (_strip_casts(x) for x in body.src)
  if any(x.op is not Ops.INDEX or x.dtype is not dtypes.half or x.src[0].op is not Ops.PARAM for x in (lhs, rhs)): return None
  lhs_aff, rhs_aff = _affine(lhs.src[1]), _affine(rhs.src[1])
  if lhs_aff is None or rhs_aff is None or any(x[1] for x in (lhs_aff, rhs_aff)): return None
  red_axis = red.arg[0]
  out_axes = tuple(out_aff[0])
  if len(out_axes) != 1 or out_aff[0][out_axes[0]] != 1 or lhs_aff[0] != {red_axis:1}: return None
  n_axis, n = out_axes[0], next(int(u.src[0].arg) for u in sink.toposort() if u.op is Ops.RANGE and u.arg[0] == out_axes[0])
  if not 1 <= n <= 16 or rhs_aff[0] != {n_axis:32, red_axis:1}: return None
  if out_param.op is not Ops.PARAM or int(out_param.src[0].arg) != n or int(lhs.src[0].src[0].arg) != 32 or int(rhs.src[0].src[0].arg) != n*32:
    return None
  return RKContract(RKView(out_param.arg.slot, (1,n), (n,1)), RKView(lhs.src[0].arg.slot, (1,32), (32,1)),
                    RKView(rhs.src[0].arg.slot, (n,32), (32,1)), (red_axis,))

def lower_pool(sink:UOp) -> RKPool|None:
  """Recognize global MAX over an explicitly legal (K, 8) HWC surface."""
  stores, reductions = [u for u in sink.toposort() if u.op is Ops.STORE], [u for u in sink.toposort() if u.op is Ops.REDUCE]
  if len(stores) != 1 or len(reductions) != 1: return None
  store, reduce = stores[0], reductions[0]
  value = _strip_casts(reduce.src[0])
  if reduce.arg[0] is not Ops.MAX or len(reduce.src) != 2 or value.op is not Ops.INDEX or value.dtype is not dtypes.half: return None
  red, out_param, inp_param = reduce.src[1], store.src[0].src[0], value.src[0]
  if red.op is not Ops.RANGE or store.src[0].op is not Ops.INDEX or out_param.op is not Ops.PARAM or inp_param.op is not Ops.PARAM: return None
  out_aff, inp_aff = _affine(store.src[0].src[1]), _affine(value.src[1])
  if out_aff is None or inp_aff is None or out_aff[1] or inp_aff[1] or len(out_aff[0]) != 1: return None
  channel_axis, red_axis = next(iter(out_aff[0])), red.arg[0]
  k, channels = int(red.src[0].arg), int(out_param.src[0].arg)
  if channels != 8 or out_aff[0] != {channel_axis:1} or inp_aff[0] != {red_axis:channels, channel_axis:1}: return None
  split = next(((h, k//h) for h in range(2, min(k,16)+1) if k%h == 0 and 2 <= k//h <= 16 and h != 9 and k//h != 9 and
                (h,k//h) not in ((3,6),(6,3),(12,12))), (1,k) if 4 <= k <= 16 else None)
  if split is None or int(inp_param.src[0].arg) != k*channels: return None
  return RKPool(RKView(out_param.arg.slot, (channels,), (1,)), RKView(inp_param.arg.slot, (k,channels), (channels,1)), split)

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_TARGET_CNA, _TARGET_CORE = 0x201, 0x801
_TARGET_PPU, _TARGET_PPU_RDMA = 0x4001, 0x8001
_EW_BASE = 0x108002c0
_EW_CFG = {RKDPUOp.ADD:_EW_BASE | (2 << 16), RKDPUOp.MUL:_EW_BASE | (1 << 2) | (1 << 8),
           RKDPUOp.MAX:_EW_BASE, RKDPUOp.SUB:_EW_BASE | (4 << 16), RKDPUOp.DIV:_EW_BASE | (3 << 16) | (1 << 8)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _emit_lut(stage_idx:int, plan:RKDPUStage, src:RKArg) -> RKStage:
  """Emit one generated 128-element EXP2 LUT task; fitting stays in extra/rockchip/gen_lut.py."""
  width, surf_stride = 15, 256
  cmds = []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    for value in rklut.RK_LUT_EXP2[table_id*513:(table_id+1)*513]:
      cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value & 0xffffffff if value < 0 else value))
  fixed = (
    (_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x30), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30),
    (_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width),
    (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    (_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 2), (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, 7),
    (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, width), (_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x20040),
    (_TARGET_DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000), (_TARGET_DPU, rk.REG_DPU_BN_MUL_CFG, rklut.RK_LUT_EXP2_BN_MUL << 16),
    (_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x302), (_TARGET_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001), (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SHIFT, rklut.RK_LUT_EXP2_MINUS_EXP << 12),
    (_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (_TARGET_DPU, 0x40c4, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_CFG, 0x68), (_TARGET_DPU, rk.REG_DPU_LUT_INFO, 0x50500),
    (_TARGET_DPU, rk.REG_DPU_LUT_LE_START, 0xffffc000), (_TARGET_DPU, rk.REG_DPU_LUT_LE_END, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_START, 0), (_TARGET_DPU, rk.REG_DPU_LUT_LO_END, 0x4000),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 16434 << 16), (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 13 << 5),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), (_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
  cmds.extend(_command(*x) for x in fixed[:4])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[4:30])
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[30:])
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, src.kind, src.index, src.addend))
  reads = (src.index,) if src.kind is RKBufferKind.ARG else ()
  writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

def _emit_mask(stage_idx:int, plan:RKDPUStage, src:RKArg) -> RKStage:
  width = (plan.count+7)//8-1
  regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53),
    (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width), (rk.REG_DPU_BN_MUL_CFG, 0),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_BS_CFG, 0x40040), (rk.REG_DPU_BS_ALU_CFG, 0x33800000),
    (rk.REG_DPU_BS_MUL_CFG, 0x40000000), (rk.REG_DPU_BN_CFG, 0x40082), (rk.REG_DPU_BN_MUL_CFG, 0x7c000000),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000), (rk.REG_DPU_EW_CFG, _EW_BASE|1),
    (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1), (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
  cmds = [_command(_TARGET_DPU, reg, value) for reg,value in regs]
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))
  cmds.extend(_command(_TARGET_DPU_RDMA, reg, value) for reg,value in rdma)
  relocs = []
  for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, src),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, src)):
    cmds.append(_command(target_id, reg, 0))
    relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
           _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  reads = (src.index,) if src.kind is RKBufferKind.ARG else ()
  writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET)

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
    physical_count = plan.count*2 if plan.out_dtype is dtypes.int else plan.count
    lhs, rhs = materialize(plan.lhs, physical_count), materialize(plan.rhs, physical_count) if plan.rhs is not None else None
    if plan.op is RKDPUOp.EXP2:
      stages.append(_emit_lut(stage_idx, plan, lhs))
      continue
    if plan.op is RKDPUOp.MASK:
      stages.append(_emit_mask(stage_idx, plan, lhs))
      continue
    width = (physical_count+7)//8-1
    native_int = plan.out_dtype is dtypes.int
    dpu_regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
      (rk.REG_DPU_DATA_FORMAT, ((4 if native_int else 2)<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0), (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
      (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0),
      (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 3 if native_int else 7), (rk.REG_DPU_WDMA_SIZE_1, width),
      (rk.REG_DPU_BN_MUL_CFG, 0), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0),
      (rk.REG_DPU_EW_CFG, 0 if plan.op is RKDPUOp.COPY else _EW_CFG[plan.op]), (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
      (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 1 if plan.op is RKDPUOp.DIV or native_int else 0x10001),
      (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
    rdma_regs = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008))
    cmds = [_command(_TARGET_DPU, *x) for x in dpu_regs] + [_command(_TARGET_DPU_RDMA, *x) for x in rdma_regs]
    relocs = []
    for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, lhs)):
      cmds.append(_command(target_id, reg, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
    if rhs is not None:
      cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, rhs.kind, rhs.index, rhs.addend))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if plan.op is RKDPUOp.DIV else 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    operands = (lhs,) if rhs is None else (lhs, rhs)
    reads = tuple(sorted({x.index for x in operands if x.kind is RKBufferKind.ARG}))
    writes = (plan.dst.index,) if plan.dst.kind is RKBufferKind.ARG else ()
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), reads, writes, (1 << (stage_idx-1)) if stage_idx else 0, RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

def emit_contract(plan:RKContract, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit a direct FP16 CMAC surface; inputs and output are already in hardware-legal layouts."""
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  m, align = 1, 32
  def e(target_id:int, reg:int, value:int) -> int: return _command(target_id, reg, value)
  commands = (
    e(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x0e), e(_TARGET_CNA, rk.REG_CNA_CONV_CON1, 0x20000120),
    e(_TARGET_CNA, rk.REG_CNA_CONV_CON2, 0x4000), e(_TARGET_CNA, rk.REG_CNA_CONV_CON3, 9),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE1, 0x1f0020),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE2, 1), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE3, 1),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE0, 0x800), e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE1, 0x40),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE2, 0x1010020), e(_TARGET_CNA, rk.REG_CNA_CBUF_CON0, 0xb1),
    e(_TARGET_CNA, rk.REG_CNA_CBUF_CON1, 1), e(_TARGET_CNA, rk.REG_CNA_CVT_CON0, 0xb),
    *(e(_TARGET_CNA, reg, 0x10000) for reg in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4)),
    e(_TARGET_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0), e(_TARGET_CNA, rk.REG_CNA_DMA_CON0, 0xf000f),
    e(_TARGET_CNA, rk.REG_CNA_DMA_CON1, 4), e(_TARGET_CNA, rk.REG_CNA_DMA_CON2, 0),
    e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE0, 0x10001), e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE1, align),
    e(_TARGET_CNA, rk.REG_CNA_DCOMP_ADDR0, 0), e(_TARGET_CORE, rk.REG_CORE_MISC_CFG, 0x201),
    e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_0, 0), e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align-1),
    e(_TARGET_CORE, rk.REG_CORE_RESERVED_3030, 0), e(_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e4),
    e(_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002), e(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    e(_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, 0x10), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, m-1), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0x70007),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x1f001f), e(_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 0x126), e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, align-1),
    e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, 0), e(_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x383), e(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    e(_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 0x40), e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
  relocs = (RKReloc(0, 18, plan.lhs.kind, plan.lhs.slot), RKReloc(0, 24, plan.rhs.kind, plan.rhs.slot),
            RKReloc(0, 31, plan.out.kind, plan.out.slot))
  reads = tuple(x.slot for x in (plan.lhs, plan.rhs) if x.kind is RKBufferKind.ARG)
  constants = struct.pack("<e", 1.0)*32 if any(x.kind is RKBufferKind.CONSTANT for x in (plan.lhs, plan.rhs)) else b""
  return RKImage(target, (RKStage(RKEngine.CMAC, commands, relocs, reads, (plan.out.slot,), flags=RK_STAGE_RESET),), constants=constants)

def emit_pool(plan:RKPool, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  height, width = plan.kernel
  h, w, c, stride = height-1, width-1, plan.out.shape[0]-1, width*plan.out.shape[0]*2
  e = _command
  commands = (e(_TARGET_PPU, rk.REG_PPU_S_POINTER, 0x0e), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_S_POINTER, 0x0e),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, w), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, h),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, c), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_WIDTH, 0),
    e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_HEIGHT, 0), e(_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, c),
    e(_TARGET_PPU, rk.REG_PPU_OPERATION_MODE_CFG, 0x11), e(_TARGET_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (h<<20)|(w<<16)|(h<<8)|w),
    e(_TARGET_PPU, rk.REG_PPU_DST_BASE_ADDR, 0), e(_TARGET_PPU, rk.REG_PPU_DST_SURF_STRIDE, 1),
    e(_TARGET_PPU, rk.REG_PPU_DATA_FORMAT, 0x10002), e(_TARGET_PPU, rk.REG_PPU_MISC_CTRL, 3),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH, w), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, h),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, c), e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, 0),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, stride),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, stride*height),
    e(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_RDMA_DATA_FORMAT, 2), e(_TARGET_PPU_RDMA, 0x7038, 1),
    e(_TARGET_PPU, rk.REG_PPU_RECIP_KERNEL_WIDTH, 0), e(_TARGET_PPU, rk.REG_PPU_RECIP_KERNEL_HEIGHT, 0),
    e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x60))
  relocs = (RKReloc(0, 10, RKBufferKind.ARG, plan.out.slot, shift=4, mask=0x0fffffff, field_shift=4),
            RKReloc(0, 17, RKBufferKind.ARG, plan.inp.slot))
  return RKImage(target, (RKStage(RKEngine.PPU, commands, relocs, (plan.inp.slot,), (plan.out.slot,), flags=RK_STAGE_RESET),))

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int}
  def native_program(self, ast:UOp) -> UOp|None:
    if (dpu:=lower_dpu(ast)) is not None: image = emit_dpu(dpu)
    elif (contract:=lower_contract(ast)) is not None: image = emit_contract(contract)
    elif (pool:=lower_pool(ast)) is not None: image = emit_pool(pool)
    else: raise RuntimeError("RKPLAN_REJECT:unsupported_graph")
    return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""),
      UOp(Ops.BINARY, arg=encode_image(image))), arg=ProgramInfo.from_sink(ast, self.target))
