import unittest
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

class NativeRenderer(Renderer):
  def native_program(self, ast:UOp) -> UOp|None:
    return UOp(Ops.PROGRAM, src=(ast, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=b"native")),
               arg=ProgramInfo.from_sink(ast, self.target))

class TestNativeProgram(unittest.TestCase):
  def test_renderer_can_intercept_early_sink(self):
    a, b = Tensor.empty(4), Tensor.empty(4)
    sink = next(x.src[0] for x in (a+b).schedule_linear().src if x.src[0].op is Ops.SINK).replace(arg=KernelInfo())
    program = to_program(sink, NativeRenderer(Target("NATIVE")))
    self.assertEqual(program.src[3].op, Ops.BINARY)
    self.assertEqual(program.src[3].arg, b"native")

if __name__ == "__main__": unittest.main()
