import hashlib, struct, unittest
from dataclasses import fields, is_dataclass
from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.rockchip import (RKBufferKind, RKContract, RKDPUProgram, RKLUTStage, RKMaskStage, RockchipRenderer, decode_image,
  emit_contract, emit_dpu, lower_contract, lower_dpu)
from tinygrad.runtime.autogen import rockchip_lut as rklut
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

  def test_division_canonicalizes_to_generic_fdiv(self):
    a, b = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink(a/b))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op, Ops.FDIV)

  def test_where_uses_generic_mask_stage(self):
    a, b = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink((a<0).where(a, b)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertTrue(any(isinstance(stage, RKMaskStage) for stage in plan.stages))
    self.assertFalse(contains_uop(plan))

  def test_exp2_uses_generated_generic_lut(self):
    payload = struct.pack(f"<{len(rklut.RK_LUT_EXP2)}h", *rklut.RK_LUT_EXP2)
    self.assertEqual(hashlib.sha256(payload).hexdigest(), rklut.RK_LUT_EXP2_SHA256)
    errors = []
    for bits in range(1 << 16):
      x = struct.unpack("<e", struct.pack("<H", bits))[0]
      if not -2 <= x <= 2: continue
      position, base = ((x+2)*256, 0) if x < 0 else (x*256, rklut.RK_LUT_EXP2_ENTRIES)
      index = min(511, max(0, int(position//1)))
      got = struct.unpack("<e", struct.pack("<e", ((1-(position-index))*rklut.RK_LUT_EXP2[base+index] +
        (position-index)*rklut.RK_LUT_EXP2[base+index+1]) / 8192))[0]
      reference = 2**x
      errors.append((abs(got-reference), abs(got-reference)/reference))
    self.assertEqual((len(errors), max(x[0] for x in errors), max(x[1] for x in errors)),
      (rklut.RK_LUT_EXP2_VERIFIED_INPUTS, rklut.RK_LUT_EXP2_SIM_MAX_ABS_ERROR, rklut.RK_LUT_EXP2_SIM_MAX_REL_ERROR))
    plan = lower_dpu(sink(Tensor.empty(128, dtype=dtypes.half).exp2()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertIsInstance(plan.stages[0], RKLUTStage)
    self.assertFalse(contains_uop(plan))
    image = emit_dpu(plan)
    self.assertEqual((len(image.stages[0].commands), tuple(r.word for r in image.stages[0].relocs)), (1064, (1032, 1059)))

  def test_direct_affine_contract_is_typed(self):
    a, packed_b = Tensor.empty(1,32,dtype=dtypes.half), Tensor.empty(8,32,dtype=dtypes.half)
    plan = lower_contract(sink(a@packed_b.T))
    self.assertIsInstance(plan, RKContract)
    self.assertFalse(contains_uop(plan))
    image = emit_contract(plan)
    self.assertEqual((len(image.stages[0].commands), tuple(r.word for r in image.stages[0].relocs)), (46, (18,24,31)))
    self.assertIsNone(lower_contract(sink(a@Tensor.empty(32,8,dtype=dtypes.half))))
    self.assertIsNone(lower_contract(sink((a@packed_b.T).sigmoid())))

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
