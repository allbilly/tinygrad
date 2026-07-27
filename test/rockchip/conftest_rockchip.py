"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's only supported dtype) and torch default to fp16
for fair comparison. Unsupported ops raise RuntimeError(RKPLAN_REJECT:...) and
fail the test honestly — no skip conversion, no monkeypatch.

PR1 is an inference/forward-only backend. Gradients are explicitly deferred
(PR8). FORWARD_ONLY=1 must be passed on the command line, not hidden here, so
that the result is never mistaken for normal full-suite behavior.

Usage:
  FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest test/backend/test_ops.py -p test.rockchip.conftest_rockchip
"""
import os
os.environ.setdefault("DEFAULT_FLOAT", "half")

import torch
torch.set_default_dtype(torch.float16)
