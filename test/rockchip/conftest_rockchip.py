"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's only supported dtype) and torch default to fp16
for fair comparison. Unsupported ops raise RuntimeError(RKPLAN_REJECT:...) and
fail the test honestly — no skip conversion, no FORWARD_ONLY, no monkeypatch.

Usage:
  DEV=ROCKCHIP python -m pytest test/backend/test_ops.py -p test.rockchip.conftest_rockchip
"""
import os
os.environ.setdefault("DEFAULT_FLOAT", "half")

import torch
torch.set_default_dtype(torch.float16)
