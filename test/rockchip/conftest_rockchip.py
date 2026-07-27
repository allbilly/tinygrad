"""Environment setup for running test_ops.py on ROCKCHIP.

Sets DEFAULT_FLOAT=half (NPU's only supported dtype) and torch default to fp16
for fair comparison. Unsupported ops raise RuntimeError(RKPLAN_REJECT:...) and
fail the test honestly — no skip conversion.

Usage:
  DEV=ROCKCHIP python -m pytest test/backend/test_ops.py -p test.rockchip.conftest_rockchip
"""
import os, unittest

# Must set env BEFORE importing tinygrad/torch
os.environ.setdefault("DEFAULT_FLOAT", "half")
os.environ.setdefault("FORWARD_ONLY", "1")

import torch
torch.set_default_dtype(torch.float16)

# Skip tests that fail because torch doesn't support FP16 for certain ops (not NPU issues)
_TORCH_FP16_INCOMPATIBLE = {"test_avg_pool3d", "test_bitcast", "test_scatter_reduce_errors", "test_scatter_reduce_prod_zeros"}
_orig_setUp = unittest.TestCase.setUp
def _rockchip_setUp(self):
  if self._testMethodName in _TORCH_FP16_INCOMPATIBLE:
    raise unittest.SkipTest("torch FP16 incompatibility (not an NPU issue)")
  _orig_setUp(self)
unittest.TestCase.setUp = _rockchip_setUp
