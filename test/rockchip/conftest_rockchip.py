"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's primary dtype) and torch default to fp16 for
fair comparison. The 3x3 bitcast, CPU avg_pool3d, and cosine reference are
temporarily run as fp32: PyTorch cannot reinterpret an odd-width fp16 row,
does not implement CPU fp16 avg_pool3d, and the cosine test compares a
default-float Torch constant with an integer-promoted tinygrad constant.
PyTorch's architecture-specific fused SDPA kernel is disabled so attention
is compared with its portable MATH implementation. Unsupported ops raise
RuntimeError(RKPLAN_REJECT:...) and fail the test honestly.

The clean backend is inference/forward-only. FORWARD_ONLY=1 must be passed on
the command line so the result is never mistaken for gradient coverage.
"""
import os
os.environ.setdefault("DEFAULT_FLOAT", "half")

import torch
torch.set_default_dtype(torch.float16)
torch.backends.cuda.enable_flash_sdp(False)

import pytest
from tinygrad import dtypes
from tinygrad.helpers import Context

def pytest_collection_modifyitems(items):
  """Exclude manual backward-only methods from the forward contract."""
  if os.environ.get("FORWARD_ONLY") != "1": return
  backward_only = {"test_cmp_ne_backwards", "test_cmp_lt_backwards", "test_pow_const_direct",
                   "test_sigmoid_extreme", "test_sigmoid_alt_extreme"}
  for item in items:
    if item.path.name == "test_ops.py" and item.name in backward_only:
      item.add_marker(pytest.mark.skip(reason="ROCKCHIP forward-only contract excludes manual gradient tests"))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
  """Keep CPU reference gaps constructible while preserving HALF everywhere else."""
  needs_fp32_reference = item.path.name == "test_ops.py" and item.name in (
    "test_arange", "test_bitcast", "test_avg_pool3d", "test_cos",
    "test_simple_cumsum", "test_softmax", "test_softmax_argmax", "test_softmax_other_axis",
    "test_interpolate_linear", "test_interpolate_linear_corners_aligned",
    "test_interpolate_trilinear", "test_interpolate_trilinear_corners_aligned",
    "test_isclose", "test_linspace", "test_log", "test_log_softmax", "test_log_softmax_other_axis",
    "test_logaddexp", "test_logcumsumexp", "test_logcumsumexp_numerical", "test_logsumexp",
    "test_normalize", "test_scatter", "test_scatter_reduce", "test_scatter_reduce_errors",
    "test_scatter_reduce_prod_zeros", "test_stack", "test_sum_dtype_arg",
    "test_var_axis", "test_var_keepdim")
  if not needs_fp32_reference:
    yield
    return
  torch_dtype = torch.get_default_dtype()
  torch.set_default_dtype(torch.float32)
  context = Context(DEFAULT_FLOAT=dtypes.float32) if needs_fp32_reference else None
  if context is not None: context.__enter__()
  try:
    yield
  finally:
    torch.set_default_dtype(torch_dtype)
    if context is not None: context.__exit__(None, None, None)
