import os
from unittest.mock import patch
from conftest import _failure_events, _failure_kind, _first_reject
from tinygrad.runtime.support.rockchip_telemetry import clear, drain, record

def test_disabled_telemetry_is_empty():
  clear()
  with patch.dict(os.environ, {}, clear=True): record("kernel", lane="RK_DPU")
  assert drain() == []

def test_telemetry_records_and_drains_in_order():
  clear()
  with patch.dict(os.environ, {"ROCKCHIP_TELEMETRY":"coverage.json"}, clear=True):
    record("reject", reject_kind="unsupported_dtype")
    record("kernel", lane="RK_DPU", stage_count=2)
  events = drain()
  assert [(x["kind"], x.get("lane")) for x in events] == [("reject", None), ("kernel", "RK_DPU")]
  assert events[0]["sequence"] < events[1]["sequence"]
  assert drain() == []

def test_coverage_first_reject_uses_event_order():
  rejects = [{"sequence": 9, "reject_kind": "unsupported_layout"}, {"sequence": 4, "reject_kind": "plan_stage_limit"}]
  assert _first_reject(rejects) == rejects[1]
  assert _first_reject([]) is None

def test_coverage_failure_kind_distinguishes_non_rejects():
  reject = [{"sequence": 1, "reject_kind": "unsupported_layout"}]
  assert _failure_kind(True, reject, []) == "NATIVE_REJECT"
  assert _failure_kind(True, [], [{"outcome": "FAIL"}]) == "DEVICE_FAILURE"
  assert _failure_kind(True, [], [{"outcome": "PASS"}]) == "POST_EXECUTION_FAILURE"
  assert _failure_kind(True, [], []) == "NON_DEVICE_FAILURE"
  assert _failure_kind(False, reject, []) is None

def test_coverage_promotes_only_failed_subcase_events():
  failed_reject = {"sequence": 3, "reject_kind": "plan_stage_limit"}
  method = {"kernels": [], "rejects": [], "subcases": [
    {"raw_outcome": "PASS", "kernels": [{"outcome": "PASS"}], "rejects": [{"sequence": 1}]},
    {"raw_outcome": "FAIL", "kernels": [], "rejects": [failed_reject]}]}
  kernels, rejects = _failure_events(method, False)
  assert kernels == [] and rejects == [failed_reject]
  assert _first_reject(rejects) == failed_reject
