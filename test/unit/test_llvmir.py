import math, unittest
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer, lconst, pm_gfx803_mixed_precision
from tinygrad.uop.ops import Ops, UOp


class TestLLVMIRConstants(unittest.TestCase):
  def test_finite_half_overflow(self):
    # LLVM does not accept textual `inf`; finite values can become infinite only after dtype truncation.
    for value, expected in ((-2147483648.0, "0xFFF0000000000000"), (2147483648.0, "0x7FF0000000000000")):
      self.assertTrue(math.isfinite(value))
      self.assertEqual(lconst(value, dtypes.half), expected)


class TestAMDLLVMRewrites(unittest.TestCase):
  def test_gfx803_emulates_half(self):
    self.assertNotIn(dtypes.half, AMDLLVMRenderer(Target("AMD", arch="gfx803")).supported_dtypes())
    self.assertIn(dtypes.half, AMDLLVMRenderer(Target("AMD", arch="gfx900")).supported_dtypes())

  def test_gfx803_widens_mixed_precision_multiply(self):
    a, b = UOp.const(dtypes.half, 1.5), UOp.const(dtypes.half, 2.5)
    widened = pm_gfx803_mixed_precision.rewrite(UOp(Ops.CAST, dtypes.float, (UOp(Ops.MUL, dtypes.half, (a, b)),)))
    self.assertIsNotNone(widened)
    assert widened is not None
    self.assertEqual((widened.op, widened.dtype), (Ops.MUL, dtypes.float))
    self.assertEqual([(x.op, x.dtype, x.src[0]) for x in widened.src], [(Ops.CAST, dtypes.float, a), (Ops.CAST, dtypes.float, b)])


if __name__ == '__main__':
  unittest.main()
