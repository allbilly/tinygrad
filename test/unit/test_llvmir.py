import math, unittest
from tinygrad.dtype import dtypes
from tinygrad.renderer.llvmir import lconst


class TestLLVMIRConstants(unittest.TestCase):
  def test_finite_half_overflow(self):
    # LLVM does not accept textual `inf`; finite values can become infinite only after dtype truncation.
    for value, expected in ((-2147483648.0, "0xFFF0000000000000"), (2147483648.0, "0x7FF0000000000000")):
      self.assertTrue(math.isfinite(value))
      self.assertEqual(lconst(value, dtypes.half), expected)


if __name__ == '__main__':
  unittest.main()
