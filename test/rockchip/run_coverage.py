#!/usr/bin/env python3
"""Run the full test_ops.py suite with DEV=NULL, capture every scheduled kernel,
and classify it into R1/R2/R3/R4/reject per the rockchip_plan §4.2 interception point.

Usage:  python test/rockchip/run_coverage.py
"""
import os, sys, collections, atexit
os.environ["DEV"] = "NULL"
os.environ["SKIP_OPS_CHECK"] = "1"

from tinygrad.dtype import dtypes
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, GroupOp
from tinygrad.uop.symbolic import sym
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse
from tinygrad.schedule.rangeify import pm_mops

CAPTURED = collections.Counter()
PER_TEST = {}
_current_test = ["unknown"]

def pre(ast):
  s = graph_rewrite(ast, pm_mops, bottom_up=True)
  s = graph_rewrite(s, pm_load_collapse)
  s = graph_rewrite(s, pm_split_ranges+pm_flatten_range, ctx={})
  s = graph_rewrite(s, sym+pm_flatten_range)
  return graph_rewrite(s, pm_flatten_range+pm_simplify_ranges, ctx={})

def is_affine(idx):
  for u in idx.toposort():
    if u.op in {Ops.RANGE, Ops.CONST}: continue
    if u.op in {Ops.ADD, Ops.MUL, Ops.WHERE, Ops.CMPLT, Ops.CMPNE, Ops.AND, Ops.OR, Ops.CAST}: continue
    if u.op in {Ops.CDIV, Ops.CMOD, Ops.SHL, Ops.SHR}: continue
    if u.op in {Ops.LOAD, Ops.INDEX}: return False
    return False
  return True

def classify(ast):
  try:
    s = pre(ast)
  except Exception:
    return "PRE_ERROR"
  topo = list(s.toposort())
  reduces = [u for u in topo if u.op is Ops.REDUCE]
  params = [u for u in topo if u.op is Ops.PARAM]
  dts = {u.dtype.scalar() for u in params}
  idxs = [u.src[1] for u in topo if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM]
  if not all(is_affine(i) for i in idxs): return "REJECT:GATHER/non-affine"
  if any(d not in (dtypes.half, dtypes.float) for d in dts):
    bad = sorted(d.name for d in dts if d not in (dtypes.half, dtypes.float))
    return f"REJECT:dtype:{bad}"
  if len(reduces) > 1: return "REJECT:multi-reduce"
  if len(reduces) == 1:
    r = reduces[0]; op = r.arg[0]
    body = r.src[0] if r.src[0].op is not Ops.CAST else r.src[0].src[0]
    if op is Ops.ADD and body.op is Ops.MUL: return "R1:CNA(CMAC) contraction"
    if op is Ops.ADD: return "R2:CNA(CMAC) sum-as-gemm"
    if op is Ops.MAX: return "R4:PPU/reduce-max"
    return f"REJECT:reduce:{op.name}"
  return "R3:DPU elementwise"

_orig_run_linear = _tensor_mod.run_linear
def _capture_run_linear(linear, *args, **kwargs):
  try:
    for c in linear.src:
      sink = c.src[0] if hasattr(c, 'src') and c.src and len(c.src) > 0 and c.src[0].op is Ops.SINK else None
      if sink is not None:
        cls = classify(sink)
        CAPTURED[cls] += 1
        PER_TEST.setdefault(_current_test[0], []).append(cls)
  except Exception:
    CAPTURED["CAPTURE_ERROR"] += 1
  return _orig_run_linear(linear, *args, **kwargs)
_tensor_mod.run_linear = _capture_run_linear

# pytest hooks
def pytest_runtest_setup(item):
  _current_test[0] = item.nodeid

@atexit.register
def _print_tally():
  if not CAPTURED: return
  total = sum(CAPTURED.values())
  lines = []
  lines.append(f"\n{'='*70}")
  lines.append(f"ROCKCHIP COVERAGE PROBE: {total} kernels from test/backend/test_ops.py")
  lines.append(f"{'='*70}")
  for k, n in CAPTURED.most_common():
    pct = 100.0*n/total
    lines.append(f"  {n:5d}  ({pct:5.1f}%)  {k}")
  cna = sum(n for k,n in CAPTURED.items() if "CNA" in k)
  dpu = CAPTURED.get("R3:DPU elementwise", 0)
  ppu = CAPTURED.get("R4:PPU/reduce-max", 0)
  reject = sum(n for k,n in CAPTURED.items() if k.startswith("REJECT:") or k == "CAPTURE_ERROR")
  lines.append(f"\n  --- unit summary ---")
  lines.append(f"  CNA+CMAC (R1+R2): {cna:5d}  ({100.0*cna/total:5.1f}%)")
  lines.append(f"  DPU (R3):         {dpu:5d}  ({100.0*dpu/total:5.1f}%)")
  lines.append(f"  PPU (R4):         {ppu:5d}  ({100.0*ppu/total:5.1f}%)")
  lines.append(f"  reject:           {reject:5d}  ({100.0*reject/total:5.1f}%)")
  lines.append(f"  structural coverage (CNA+DPU+PPU): {100.0*(cna+dpu+ppu)/total:.1f}%")
  lines.append(f"{'='*70}")
  reject_tests = [(t, sum(1 for c in cs if c.startswith("REJECT"))) for t, cs in PER_TEST.items()]
  reject_tests = [(t, n) for t, n in reject_tests if n > 0]
  if reject_tests:
    reject_tests.sort(key=lambda x: -x[1])
    lines.append(f"\nTop 30 tests by reject count:")
    for t, n in reject_tests[:30]:
      lines.append(f"  {n:3d} rejects  {t}")
  output = "\n".join(lines)
  print(output)
  sys.stdout.flush()
  with open("/tmp/rockchip_coverage_tally.txt", "w") as f:
    f.write(output + "\n")

if __name__ == "__main__":
  import pytest
  # no -n (xdist forks workers that lose the monkeypatch); run in-process
  pytest.main([
    "test/backend/test_ops.py",
    "-q", "--tb=line", "-s", "--capture=no",
    "-p", "no:randomly",
    "--no-header",
    "-p", __name__,
  ], plugins=[sys.modules[__name__]])
