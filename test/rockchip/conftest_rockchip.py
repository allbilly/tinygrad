"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's primary dtype) and torch default to fp16 for
fair comparison. The one 3x3 bitcast reference is temporarily run as fp32:
PyTorch cannot reinterpret an odd-width fp16 row as int32. PyTorch's
architecture-specific fused SDPA kernel is disabled so attention is compared
with its portable MATH implementation. Unsupported ops raise
RuntimeError(RKPLAN_REJECT:...) and fail the test honestly.

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
torch.backends.cuda.enable_flash_sdp(False)

import pytest
from tinygrad import dtypes

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
  """Keep the canonical equal-width bitcast test constructible under the HALF suite."""
  is_bitcast = item.name == "test_bitcast" and item.path.name == "test_ops.py"
  if not is_bitcast:
    yield
    return
  torch_dtype, tinygrad_dtype = torch.get_default_dtype(), dtypes.default_float
  torch.set_default_dtype(torch.float32)
  dtypes.default_float = dtypes.float32
  try:
    yield
  finally:
    torch.set_default_dtype(torch_dtype)
    dtypes.default_float = tinygrad_dtype
