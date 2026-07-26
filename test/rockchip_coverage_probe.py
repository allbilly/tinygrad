import os, collections
os.environ["DEV"]="NULL"
from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, GroupOp
from tinygrad.uop.symbolic import sym
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_mops
from tinygrad.dtype import AddrSpace

def pre(ast):
  s = graph_rewrite(ast, pm_mops, bottom_up=True)
  s = graph_rewrite(s, pm_load_collapse)
  s = graph_rewrite(s, pm_split_ranges+pm_flatten_range, ctx={})
  s = graph_rewrite(s, sym+pm_flatten_range)
  return graph_rewrite(s, pm_flatten_range+pm_simplify_ranges, ctx={})

def is_affine(idx):
  # affine in RANGEs: only ADD/MUL-by-const/RANGE/CONST, plus a WHERE(valid,...,Invalid) wrapper
  for u in idx.toposort():
    if u.op in {Ops.RANGE, Ops.CONST}: continue
    if u.op in {Ops.ADD, Ops.MUL, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.CAST}: continue
    if u.op in {Ops.CDIV, Ops.CMOD, Ops.SHL, Ops.SHR}: continue
    if u.op in {Ops.LOAD, Ops.INDEX}: return False   # data-dependent index = gather
    return False
  return True

def classify(ast):
  s = pre(ast)
  topo = list(s.toposort())
  reduces = [u for u in topo if u.op is Ops.REDUCE]
  params = [u for u in topo if u.op is Ops.PARAM]
  # dtypes of tensor data touched
  dts = {u.dtype.scalar() for u in params}
  alu = [u for u in topo if u.op in GroupOp.ALU and u.dtype.scalar() in (dtypes.half, dtypes.float, dtypes.double)]
  intalu = [u for u in topo if u.op in GroupOp.ALU and not dtypes.is_float(u.dtype.scalar()) and u.dtype.scalar() is not dtypes.bool
            and any(sr.op is Ops.LOAD or (sr.op is Ops.INDEX and sr.src[0].op is Ops.PARAM) for sr in u.backward_slice)]
  idxs = [u.src[1] for u in topo if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if not all(is_affine(i) for i in idxs): return "GATHER/non-affine"
  if any(d not in (dtypes.half, dtypes.float) for d in dts): return f"dtype:{sorted(d.name for d in dts if d not in (dtypes.half,dtypes.float))}"
  if len(reduces) > 1: return "multi-reduce"
  if len(reduces) == 1:
    r = reduces[0]; op = r.arg[0]
    body = r.src[0] if r.src[0].op is not Ops.CAST else r.src[0].src[0]
    if op is Ops.ADD and body.op is Ops.MUL: return "CNA(CMAC) contraction"
    if op is Ops.ADD: return "CNA(CMAC) sum-as-gemm(ones)"
    if op is Ops.MAX: return "PPU/reduce-max"
    return f"reduce:{op.name}"
  return "DPU elementwise"

CASES = {}
def add(name, fn): CASES[name] = fn
a = Tensor.rand(64,64,dtype=dtypes.half).realize(); b = Tensor.rand(64,64,dtype=dtypes.half).realize()
v = Tensor.rand(64,dtype=dtypes.half).realize(); x4 = Tensor.rand(1,16,32,32,dtype=dtypes.half).realize()
w4 = Tensor.rand(32,16,3,3,dtype=dtypes.half).realize(); ai = Tensor.ones(64,64,dtype=dtypes.int32).contiguous().realize()
idx = Tensor([1,3,5],dtype=dtypes.int32).realize()
add("add",          lambda: a+b);              add("mul",        lambda: a*b)
add("sub",          lambda: a-b);              add("div",        lambda: a/b)
add("neg",          lambda: -a);               add("relu",       lambda: a.relu())
add("maximum",      lambda: a.maximum(b));     add("where",      lambda: (a>b).where(a,b))
add("exp",          lambda: a.exp());          add("log",        lambda: a.log())
add("sqrt",         lambda: a.sqrt());         add("sigmoid",    lambda: a.sigmoid())
add("sin",          lambda: a.sin());          add("recip",      lambda: a.reciprocal())
add("cmplt->bool",  lambda: (a<b).cast(dtypes.half))
add("sum(all)",     lambda: a.sum());          add("sum(axis)",  lambda: a.sum(1))
add("mean",         lambda: a.mean(1));        add("max(axis)",  lambda: a.max(1))
add("softmax",      lambda: a.softmax());      add("matmul",     lambda: a@b)
add("matvec",       lambda: a@v);              add("dot",        lambda: v@v)
add("conv2d",       lambda: x4.conv2d(w4,padding=1))
add("conv2d+bias",  lambda: x4.conv2d(w4,Tensor.rand(32,dtype=dtypes.half).realize(),padding=1))
add("maxpool",      lambda: x4.max_pool2d(2)); add("avgpool",    lambda: x4.avg_pool2d(2))
add("batchnorm",    lambda: nn.BatchNorm(16)(x4.float()).half())
add("int_add",      lambda: (ai+ai).cast(dtypes.half))
add("gather",       lambda: a[idx])
add("cat",          lambda: Tensor.cat(a,b));  add("pad",        lambda: a.pad(((1,1),(1,1))))
add("transpose",    lambda: a.T.contiguous()); add("cumsum",     lambda: a.cumsum(1))
add("argmax",       lambda: a.argmax(1).cast(dtypes.half))
add("layernorm",    lambda: a.layernorm())

tally = collections.Counter(); rows=[]
for name, fn in CASES.items():
  try:
    t = fn()
    t = t.contiguous() if isinstance(t, Tensor) else t
    lin = t.schedule_linear()
    ks = [c.src[0] for c in lin.src if c.src[0].op is Ops.SINK]
    cls = [classify(k) for k in ks]
    rows.append((name, len(ks), cls))
    for c in cls: tally[c]+=1
  except Exception as e:
    rows.append((name, -1, [f"ERR {type(e).__name__}: {str(e)[:60]}"])); tally["ERROR"]+=1

print(f"{'op':16s} {'#k':>3s}  kernel classes")
print("-"*100)
for n,k,c in rows: print(f"{n:16s} {k:3d}  {', '.join(c)}")
print("\n=== kernel tally ===")
for k,n in tally.most_common(): print(f"{n:4d}  {k}")
