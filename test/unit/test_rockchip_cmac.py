import hashlib
import pytest
from tinygrad.renderer.rockchip import RKArg, RKBufferKind
from tinygrad.renderer.rockchip_cmac import (CMACAxes, CMACFamily, CMACFallback, CMACPlan, CMACReject, StaticAxis,
  StaticIndex, emit_cmac_stage, plan_cmac, serialize_cmac_stage, validate_cmac_plan)

def _args(): return tuple(RKArg(RKBufferKind.ARG, i) for i in range(3))

def _dense_axes(m=2, n=20, k=65):
  return CMACAxes((StaticAxis("m", m), StaticAxis("n", n), StaticAxis("k", k)), ("m",), ("n",), ("k",))

def _dense_maps(n=20, k=65):
  return StaticIndex(("m", "n"), (n, 1)), StaticIndex(("m", "k"), (k, 1)), StaticIndex(("n", "k"), (k, 1))

def test_matrixizer_exhaustively_tiles_m_n_k_and_reuses_static_weights():
  lhs, weights, out = _args()
  output, left, right = _dense_maps()
  plan = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(), output_map=output, lhs_map=left, rhs_map=right,
                   lhs=lhs, rhs=weights, out=out, lhs_count=2*65, rhs_count=20*65)
  assert isinstance(plan, CMACPlan)
  validate_cmac_plan(plan, 2*65, 20*65)
  assert plan.shape.m == 2 and plan.shape.n == 20 and plan.shape.k == 65
  assert len(plan.tiles) == 2*2*3 and plan.scratch_slots == 1
  assert plan.barriers == (1, 2, 4, 5, 7, 8, 10, 11)
  first_rhs, reused_rhs = plan.tiles[0].rhs, plan.tiles[6].rhs
  assert first_rhs is not None and reused_rhs is not None and first_rhs.reuse == reused_rhs.reuse
  assert plan.tiles[-1].lhs.indices[0] == 129 and plan.tiles[-1].lhs.indices[1:] == (-1,)*31

def test_static_table_indices_and_local_add_have_the_same_oracle():
  lhs, weights, out = _args()
  axes = CMACAxes((StaticAxis("m", 1), StaticAxis("n", 4), StaticAxis("k", 32)), ("m",), ("n",), ("k",))
  output = StaticIndex(("m", "n"), (4, 1))
  left = StaticIndex(("m", "k"), (32, 1))
  table = StaticIndex(("n", "k"), values=tuple(n*32+(31-k) for n in range(4) for k in range(32)))
  plan = plan_cmac(family=CMACFamily.AFFINE_MADD, axes=axes, output_map=output, lhs_map=left, rhs_map=table,
                   lhs=lhs, rhs=weights, out=out, lhs_count=32, rhs_count=128)
  assert isinstance(plan, CMACPlan) and plan.tiles[0].rhs is not None
  assert plan.tiles[0].rhs.indices[:4] == (31, 30, 29, 28)
  local = plan_cmac(family=CMACFamily.LOCAL_ADD, local=True, axes=axes, output_map=output, lhs_map=left,
                    lhs=lhs, out=out, lhs_count=32)
  assert isinstance(local, CMACPlan) and local.rhs is None

@pytest.mark.parametrize("operation,reason", (("max", CMACReject.EXTREMA), ("int", CMACReject.INTEGER),
  ("bool", CMACReject.BOOLEAN)))
def test_non_additive_and_non_fp16_inputs_are_explicit_fallbacks(operation, reason):
  lhs, weights, out = _args()
  output, left, right = _dense_maps()
  result = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(), output_map=output, lhs_map=left, rhs_map=right,
                     lhs=lhs, rhs=weights, out=out, lhs_count=130, rhs_count=1300, operation=operation)
  assert isinstance(result, CMACFallback) and result.reason is reason
  dtype = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(), output_map=output, lhs_map=left, rhs_map=right,
                    lhs=lhs, rhs=weights, out=out, lhs_count=130, rhs_count=1300, input_dtype="fp32")
  assert isinstance(dtype, CMACFallback) and dtype.reason is CMACReject.DTYPE

def test_dynamic_and_bad_index_maps_do_not_escape_the_static_oracle():
  lhs, weights, out = _args()
  output, left, right = _dense_maps()
  dynamic = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(), output_map=output, lhs_map=left, rhs_map=right,
                      lhs=lhs, rhs=weights, out=out, lhs_count=130, rhs_count=1300, dynamic=True)
  assert isinstance(dynamic, CMACFallback) and dynamic.reason is CMACReject.DYNAMIC
  bad = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(), output_map=output, lhs_map=left,
                  rhs_map=StaticIndex(("n", "k"), (1, 9999)), lhs=lhs, rhs=weights, out=out, lhs_count=130, rhs_count=1300)
  assert isinstance(bad, CMACFallback) and bad.reason in {CMACReject.AXES, CMACReject.BOUNDS}

def test_frozen_donor_stage_is_only_a_host_serialization_oracle():
  lhs, weights, out = _args()
  output, left, right = _dense_maps(n=8, k=32)
  plan = plan_cmac(family=CMACFamily.MADD, axes=_dense_axes(m=1, n=8, k=32), output_map=output, lhs_map=left, rhs_map=right,
                   lhs=lhs, rhs=weights, out=out, lhs_count=32, rhs_count=256)
  assert isinstance(plan, CMACPlan)
  stage = emit_cmac_stage(plan.tiles[0])
  assert hashlib.sha256(serialize_cmac_stage(stage)).hexdigest() == "e1a4fb0194156e87680375eab9594f22f9ae545b4be50776819fb8e83c5e4af1"
