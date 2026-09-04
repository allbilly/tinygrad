import unittest

from tinygrad.codegen.decomp.dtype import pm_float_decomp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.uop.spec import spec_program, type_verify


class TestFloatDecomp(unittest.TestCase):
  def test_after_preserves_emulated_storage_dtype(self):
    rng = UOp.range(4, 0, dtype=dtypes.int)
    local = UOp.placeholder((1,), dtypes.half, 0, addrspace=AddrSpace.REG).replace(src=(UOp.const(dtypes.int, 1),))
    sink = local.after(rng).index(UOp.const(dtypes.int, 0)).load().sink()

    rewritten = graph_rewrite(sink, pm_float_decomp, ctx=(dtypes.half, dtypes.float), bottom_up=True)
    after = next(u for u in rewritten.toposort() if u.op is Ops.AFTER)
    self.assertEqual((after.dtype, after.src[0].dtype, after.tag), (dtypes.ushort, dtypes.ushort, dtypes.half))
    type_verify(rewritten, spec_program)


if __name__ == "__main__":
  unittest.main()
