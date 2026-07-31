import unittest
from dataclasses import fields, is_dataclass
from tinygrad import Tensor, dtypes
from tinygrad.codegen import early_simplify
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import (RKBufferKind, RKDPUProgram, RockchipRenderer, decode_image, emit_dpu, lower_dpu)
from tinygrad.uop.ops import Ops, UOp

def sink(expr:Tensor) -> UOp:
  return early_simplify(next(x.src[0] for x in expr.schedule_linear().src if x.src[0].op is Ops.SINK))

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

  def test_typed_plan_retains_no_uops(self):
    a, b, c = (Tensor.empty(16,dtype=dtypes.half) for _ in range(3))
    plan = lower_dpu(sink((a+b)*c))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))
    self.assertEqual(len(plan.stages), 2)
    self.assertEqual(len(plan.scratch), 1)
    self.assertEqual(len(emit_dpu(plan).stages), 2)

  def test_mul_matches_frozen_oracle(self):
    a, b = Tensor.empty(4,4,dtype=dtypes.half), Tensor.empty(4,4,dtype=dtypes.half)
    image = emit_dpu(lower_dpu(sink(a*b)))
    self.assertEqual(len(image.stages[0].commands), 18)
    self.assertEqual(image.stages[0].commands[6], 0x1001108003c44070)
    self.assertEqual(image.stages[0].commands[:6], (
      0x10010000000e4004, 0x1001000001e5400c, 0x1001480000024010,
      0x1001000000014030, 0x1001000000004034, 0x100100070007403c))

  def test_fill_is_dpu_add_not_constant_copy(self):
    plan = lower_dpu(sink(Tensor.full((16,), 3.5, dtype=dtypes.half)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op.name, "ADD")
    self.assertEqual(len(emit_dpu(plan).constants), 64)

  def test_liveness_reuses_dead_scratch(self):
    a, b, c, d = (Tensor.empty(16,dtype=dtypes.half) for _ in range(4))
    plan = lower_dpu(sink(((a+b)*c)+d))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(len(plan.stages), 3)
    self.assertEqual(len(plan.scratch), 1)

  def test_renderer_produces_decodable_machine_image(self):
    a, b = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    program = RockchipRenderer(Target("ROCKCHIP")).native_program(sink(a.maximum(b)))
    self.assertIsNotNone(program)
    image = decode_image(program.src[3].arg)
    self.assertEqual(len(image.stages), 1)

  def test_rejects_noncontiguous_and_nonhalf(self):
    a, b = Tensor.empty(4,4,dtype=dtypes.half), Tensor.empty(4,4,dtype=dtypes.half)
    self.assertIsNone(lower_dpu(sink(a.T+b)))
    x, y = Tensor.empty(16,dtype=dtypes.float), Tensor.empty(16,dtype=dtypes.float)
    self.assertIsNone(lower_dpu(sink(x+y)))

if __name__ == "__main__": unittest.main()
