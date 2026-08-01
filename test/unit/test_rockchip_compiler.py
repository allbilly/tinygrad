import unittest
from dataclasses import fields, is_dataclass
from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.rockchip import RKBufferKind, RKDPUProgram, RockchipRenderer, decode_image, emit_dpu, lower_dpu
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

class CaptureRenderer(Renderer):
  ast:UOp|None = None
  def native_program(self, ast:UOp) -> UOp|None:
    self.ast = ast
    return UOp(Ops.PROGRAM, src=(ast,), arg=ProgramInfo.from_sink(ast, self.target))

def sink(expr:Tensor) -> UOp:
  raw = next(x.src[0] for x in expr.schedule_linear().src if x.src[0].op is Ops.SINK).replace(arg=KernelInfo())
  renderer = CaptureRenderer(Target("CAPTURE"))
  full_rewrite_to_sink(raw, renderer)
  assert renderer.ast is not None
  return renderer.ast

def contains_uop(obj) -> bool:
  if isinstance(obj, UOp): return True
  if is_dataclass(obj): return any(contains_uop(getattr(obj, field.name)) for field in fields(obj))
  if isinstance(obj, (tuple, list, dict)): return any(contains_uop(x) for x in (obj.values() if isinstance(obj, dict) else obj))
  return False

class TestDPUCompiler(unittest.TestCase):
  def test_add_matches_frozen_oracle(self):
    a, b = Tensor.empty(4,4,dtype=dtypes.half), Tensor.empty(4,4,dtype=dtypes.half)
    plan = lower_dpu(sink(a+b))
    self.assertIsInstance(plan, RKDPUProgram)
    image = emit_dpu(plan)
    self.assertEqual(image.stages[0].commands, (
      0x10010000000e4004, 0x1001000001e5400c, 0x1001480000024010, 0x1001000000014030,
      0x1001000000004034, 0x100100070007403c, 0x1001108202c04070, 0x1001000100014084,
      0x20010000000e5004, 0x200100000001500c, 0x2001000000005010, 0x2001000000075014,
      0x2001400000085034, 0x1001000000004020, 0x2001000000005018, 0x2001000000005038,
      0x2001000178495044, 0x0081000000180008))
    self.assertEqual(tuple((r.word, r.kind, r.index) for r in image.stages[0].relocs),
                     ((13, RKBufferKind.ARG, 0), (14, RKBufferKind.ARG, 1), (15, RKBufferKind.ARG, 2)))

  def test_plan_is_uop_free_and_reuses_scratch(self):
    a, b, c, d = (Tensor.empty(16,dtype=dtypes.half) for _ in range(4))
    plan = lower_dpu(sink(((a+b)*c)+d))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))
    self.assertEqual((len(plan.stages), len(plan.scratch), len(emit_dpu(plan).stages)), (3, 1, 3))

  def test_renderer_produces_decodable_machine_image(self):
    a, b = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    program = RockchipRenderer(Target("ROCKCHIP")).native_program(sink(a.maximum(b)))
    self.assertIsNotNone(program)
    self.assertEqual(len(decode_image(program.src[3].arg).stages), 1)

  def test_rejects_noncontiguous_and_nonhalf(self):
    a, b = Tensor.empty(4,4,dtype=dtypes.half), Tensor.empty(4,4,dtype=dtypes.half)
    self.assertIsNone(lower_dpu(sink(a.T+b)))
    x, y = Tensor.empty(16,dtype=dtypes.float), Tensor.empty(16,dtype=dtypes.float)
    self.assertIsNone(lower_dpu(sink(x+y)))

if __name__ == "__main__": unittest.main()
