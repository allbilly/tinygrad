"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's primary dtype) and torch default to fp16 for
fair comparison. The 3x3 bitcast, CPU avg_pool3d, and cosine reference are
temporarily run as fp32: PyTorch cannot reinterpret an odd-width fp16 row,
does not implement CPU fp16 avg_pool3d, and the cosine test compares a
default-float Torch constant with an integer-promoted tinygrad constant.
PyTorch's architecture-specific fused SDPA kernel is disabled so attention
is compared with its portable MATH implementation. Unsupported ops raise
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

def pytest_collection_modifyitems(items):
  """Exclude the two manual backward-only methods from the forward contract."""
  if os.environ.get("FORWARD_ONLY") != "1": return
  backward_only = {"test_cmp_ne_backwards", "test_cmp_lt_backwards"}
  for item in items:
    if item.path.name == "test_ops.py" and item.name in backward_only:
      item.add_marker(pytest.mark.skip(reason="ROCKCHIP forward-only contract excludes manual gradient tests"))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
  """Keep CPU reference gaps constructible while preserving HALF everywhere else."""
  needs_fp32_reference = item.path.name == "test_ops.py" and item.name in ("test_arange", "test_bitcast", "test_avg_pool3d", "test_cos")
  if not needs_fp32_reference:
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
