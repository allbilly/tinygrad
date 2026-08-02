import hashlib, math

from tinygrad.uop.ops import Ops, UOp

def affine(u:UOp) -> tuple[dict[int, int], int]|None:
  if u.op is Ops.RANGE: return ({u.arg[0]:1}, 0)
  if u.op is Ops.CONST: return ({}, int(u.arg))
  if u.op is Ops.ADD:
    a, b = affine(u.src[0]), affine(u.src[1])
    if a is None or b is None: return None
    return ({k:a[0].get(k, 0)+b[0].get(k, 0) for k in a[0].keys()|b[0].keys()}, a[1]+b[1])
  if u.op is Ops.MUL:
    const, value = (u.src[0], u.src[1]) if u.src[0].op is Ops.CONST else (u.src[1], u.src[0])
    if const.op is not Ops.CONST or (result:=affine(value)) is None: return None
    return ({k:v*int(const.arg) for k,v in result[0].items()}, result[1]*int(const.arg))
  return None

def _const_category(value) -> str:
  if isinstance(value, float) and math.isnan(value): return "NAN"
  if isinstance(value, float) and math.isinf(value): return "POS_INF" if value > 0 else "NEG_INF"
  if value == 0: return "ZERO"
  if value == 1: return "ONE"
  if value == -1: return "NEG_ONE"
  if isinstance(value, int) or isinstance(value, float) and value.is_integer(): return "POS_INT" if value > 0 else "NEG_INT"
  if isinstance(value, (int, float)): return "POS_FRAC" if value > 0 else "NEG_FRAC"
  return type(value).__name__.upper()

def rk_fingerprint(sink:UOp) -> tuple:
  """Stable graph-family identity that omits buffer slots and exact constant values."""
  nodes = sink.toposort()
  axis_ids = {axis:i for i,axis in enumerate(sorted({u.arg[0] for u in nodes if u.op is Ops.RANGE}))}
  digest:dict[UOp, str] = {}
  indexes:list[tuple] = []
  reductions:list[tuple] = []
  for u in nodes:
    shape = tuple(x if isinstance(x, int) else str(x) for x in u._shape) if u._shape is not None else ()
    arg:tuple|None = None
    if u.op is Ops.PARAM: arg = (u.addrspace.name,)
    elif u.op is Ops.CONST: arg = (_const_category(u.arg),)
    elif u.op is Ops.RANGE: arg = (axis_ids[u.arg[0]], u.arg[-1].name, int(u.src[0].arg) if u.src[0].op is Ops.CONST else "dynamic")
    elif u.op is Ops.REDUCE:
      arg = (u.arg[0].name, u.arg[1])
      reductions.append(arg)
    elif u.op is Ops.INDEX:
      result = affine(u.src[1])
      index:tuple = ("nonaffine",) if result is None else (tuple(sorted((axis_ids.get(k, -1), v) for k,v in result[0].items())), result[1])
      indexes.append((u.dtype.name, index))
      arg = index
    payload = (u.op.name, u.dtype.name, shape, arg, tuple(digest[x] for x in u.src))
    digest[u] = hashlib.sha256(repr(payload).encode()).hexdigest()[:16]
  op_counts = tuple((op.name, sum(u.op is op for u in nodes)) for op in sorted({u.op for u in nodes}, key=lambda x:x.name))
  params = tuple(sorted(((u.dtype.name, tuple(x if isinstance(x, int) else str(x) for x in u.shape), u.addrspace.name)
                         for u in nodes if u.op is Ops.PARAM), key=repr))
  constants = tuple(sorted(_const_category(u.arg) for u in nodes if u.op is Ops.CONST))
  return (("graph", digest[sink]), ("ops", op_counts), ("params", params), ("constants", constants),
          ("indexes", tuple(sorted(indexes, key=repr))), ("reductions", tuple(sorted(reductions, key=repr))))
