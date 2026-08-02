import os
from unittest.mock import patch
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
