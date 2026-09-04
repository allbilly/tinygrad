import struct, unittest
import numpy as np

from tinygrad import Tensor
from tinygrad.codegen.decomp.dtype import f2f, pm_float_decomp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.uop.spec import spec_program, type_verify


class TestFloatDecomp(unittest.TestCase):
  def test_half_subnormal_widens_exactly(self):
    for bits, expected in ((0x0001, 2.0**-24), (0x0091, 145 * 2.0**-24), (0x8091, -145 * 2.0**-24)):
      widened = f2f(UOp.const(dtypes.ushort, bits), dtypes.half, dtypes.float).simplify()
      self.assertEqual((widened.op, widened.dtype, widened.arg), (Ops.CONST, dtypes.float, expected))

  def test_float_to_half_subnormal_rounds_even(self):
    cases = ((2.0**-25, 0x0000), (struct.unpack('f', struct.pack('I', 0x33000001))[0], 0x0001),
             (2.0**-24, 0x0001), (3 * 2.0**-25, 0x0002), (5 * 2.0**-25, 0x0002),
             (1023 * 2.0**-24, 0x03ff), (1023.5 * 2.0**-24, 0x0400), (-2.0**-24, 0x8001))
    for value, expected in cases:
      bits = struct.unpack('I', struct.pack('f', value))[0]
      narrowed = f2f(UOp.const(dtypes.uint, bits), dtypes.float, dtypes.half).simplify()
      self.assertEqual((narrowed.op, narrowed.dtype, narrowed.arg), (Ops.CONST, dtypes.ushort, expected))

  def test_float_to_half_overflow_rounding_boundary(self):
    for value, expected in ((65504.0, 0x7bff), (65508.0, 0x7bff), (65519.0, 0x7bff), (65520.0, 0x7c00),
                            (-65519.0, 0xfbff), (-65520.0, 0xfc00)):
      bits = struct.unpack('I', struct.pack('f', value))[0]
      narrowed = f2f(UOp.const(dtypes.uint, bits), dtypes.float, dtypes.half).simplify()
      self.assertEqual((narrowed.op, narrowed.dtype, narrowed.arg), (Ops.CONST, dtypes.ushort, expected))

  def test_after_preserves_emulated_storage_dtype(self):
    rng = UOp.range(4, 0, dtype=dtypes.int)
    local = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG).replace(src=(UOp.const(dtypes.int, 1),))
    sink = local.after(rng).index(UOp.const(dtypes.int, 0)).load().sink()

    rewritten = graph_rewrite(sink, pm_float_decomp, ctx=(dtypes.half, dtypes.float), bottom_up=True)
    after = next(u for u in rewritten.toposort() if u.op is Ops.AFTER)
    self.assertEqual((after.dtype, after.src[0].dtype, after.tag), (dtypes.ushort, dtypes.ushort, dtypes.half))
    type_verify(rewritten, spec_program)

  def test_nonfloat_consumer_rounds_emulated_half(self):
    # float32(half(1.1)) * 10 is just below 11, while half multiplication rounds to 11.
    with Context(EMULATED_DTYPES="half"):
      x = Tensor(np.array([0x3c66], dtype=np.uint16).view(np.float16))
      self.assertEqual((x * 10).cast(dtypes.uint8).item(), 11)
      self.assertTrue((x * 10 == 11).item())


if __name__ == "__main__":
  unittest.main()
