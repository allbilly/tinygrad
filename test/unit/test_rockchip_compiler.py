import hashlib, struct, unittest
from dataclasses import fields, is_dataclass
from tinygrad import Tensor, dtypes
from tinygrad.codegen import early_simplify
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import (RKBufferKind, RKContract, RKDPUOp, RKDPUProgram, RKPool, RockchipRenderer, decode_image,
                                        emit_contract, emit_dpu, emit_pool, encode_image, lower_contract, lower_dpu, lower_pool)
from tinygrad.runtime.autogen import rockchip_lut as rklut
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
      0x1001000000004034, 0x1001000000004038, 0x100100070007403c, 0x1001000000534040,
      0x1001000000534060, 0x1001000000004044, 0x1001000000004048, 0x1001000000024050,
      0x1001000000074058, 0x100100000001405c, 0x1001000000004068, 0x100100000000406c,
      0x1001108202c04070, 0x1001000000014078, 0x1001000000004080, 0x1001000100014084,
      0x1001000000004088, 0x10010000004040c0, 0x20010000000e5004, 0x200100000001500c,
      0x2001000000005010, 0x2001000000075014, 0x2001400000085034, 0x1001000000004020,
      0x2001000000005018, 0x2001000000005038, 0x2001000178495044, 0x0081000000180008))
    self.assertEqual(tuple((r.word, r.kind, r.index) for r in image.stages[0].relocs),
                     ((27, RKBufferKind.ARG, 0), (28, RKBufferKind.ARG, 1), (29, RKBufferKind.ARG, 2)))

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
    self.assertEqual(len(image.stages[0].commands), 32)
    self.assertEqual(image.stages[0].commands[16], 0x1001108003c44070)
    self.assertEqual(image.stages[0].commands[:7], (
      0x10010000000e4004, 0x1001000001e5400c, 0x1001480000024010,
      0x1001000000014030, 0x1001000000004034, 0x1001000000004038, 0x100100070007403c))

  def test_wide_fills_tile_native_wdma_limits(self):
    for dtype, count, tile in ((dtypes.int, 2925, 64), (dtypes.float, 6, 4)):
      plan = lower_dpu(sink(Tensor.full((count,), 4, dtype=dtype)))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual(len(plan.stages), (count+tile-1)//tile)
      self.assertTrue(all(stage.count <= tile and stage.out_dtype is dtype for stage in plan.stages))
      self.assertEqual(tuple(stage.dst.addend for stage in plan.stages), tuple(range(0, count*4, tile*4)))
      image = emit_dpu(plan)
      self.assertEqual(decode_image(encode_image(image)), image)
    self.assertIsNone(lower_dpu(sink(Tensor.full((257,), 4, dtype=dtypes.float))))

  def test_fill_is_dpu_add_not_constant_copy(self):
    plan = lower_dpu(sink(Tensor.full((16,), 3.5, dtype=dtypes.half)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op.name, "ADD")
    self.assertEqual(len(emit_dpu(plan).constants), 64)

  def test_copy_is_canonical_dpu_add_zero(self):
    x = Tensor.empty(45,65,dtype=dtypes.half)
    plan = lower_dpu(sink(x/1))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op.name, "ADD")
    self.assertEqual(plan.stages[0].lhs, 0.0)

  def test_scalar_fill_uses_const_zero_index(self):
    plan = lower_dpu(sink(Tensor.ones((), dtype=dtypes.half)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].count, 1)

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

  def test_exp2_uses_generated_lut(self):
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
    self.assertEqual(len(errors), rklut.RK_LUT_EXP2_VERIFIED_INPUTS)
    self.assertEqual(max(x[0] for x in errors), rklut.RK_LUT_EXP2_SIM_MAX_ABS_ERROR)
    self.assertEqual(max(x[1] for x in errors), rklut.RK_LUT_EXP2_SIM_MAX_REL_ERROR)
    plan = lower_dpu(sink(Tensor.empty(128, dtype=dtypes.half).exp2()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))
    self.assertEqual(plan.stages[0].op.name, "EXP2")
    image = emit_dpu(plan)
    self.assertEqual(len(image.stages[0].commands), 1064)
    self.assertEqual(image.stages[0].commands[:3],
      (0x1001000200004100, 0x1001000008004104, 0x1001000008064104))
    self.assertEqual(tuple(r.word for r in image.stages[0].relocs), (1032, 1059))
    small = lower_dpu(sink(Tensor.empty(16, dtype=dtypes.half).exp2()))
    large = lower_dpu(sink(Tensor.empty(2925, dtype=dtypes.half).exp2()))
    self.assertEqual((len(small.stages), small.stages[0].count, len(small.scratch)), (13, 16, 4))
    self.assertEqual((len(large.stages), large.stages[0].count, len(large.scratch)), (13, 2925, 4))
    self.assertEqual(tuple(stage.op for stage in large.stages).count(RKDPUOp.EXP2), 1)

  def test_rejects_noncontiguous_and_nonhalf(self):
    a, b = Tensor.empty(4,4,dtype=dtypes.half), Tensor.empty(4,4,dtype=dtypes.half)
    self.assertIsNone(lower_dpu(sink(a.T+b)))
    x, y = Tensor.empty(16,dtype=dtypes.float), Tensor.empty(16,dtype=dtypes.float)
    self.assertIsNone(lower_dpu(sink(x+y)))

  def test_ordered_where_normalizes_to_dpu_extrema(self):
    x = Tensor.empty(16,dtype=dtypes.half)
    for expression in (x.relu(), x.clip(-1, 1)):
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertFalse(contains_uop(plan))
    self.assertEqual(lower_dpu(sink(x.relu())).stages[0].op.name, "MAX")

  def test_generic_where_uses_typed_mask_graph(self):
    x, y = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink((x<0).where(x, y)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))
    self.assertIn("MASK", tuple(stage.op.name for stage in plan.stages))

  def test_composed_predicates_preserve_shared_mask_liveness(self):
    sign = lower_dpu(sink(Tensor.empty(3,dtype=dtypes.half).sign()))
    clipped = lower_dpu(sink(Tensor.empty(3,dtype=dtypes.half).clip(0, 0)))
    self.assertIsInstance(sign, RKDPUProgram)
    self.assertIsInstance(clipped, RKDPUProgram)
    self.assertEqual(len(sign.scratch), 3)
    self.assertEqual(len(clipped.scratch), 2)
    self.assertTrue(all(stage.lhs != stage.rhs for stage in sign.stages if stage.op.name == "MAX"))

  def test_relu_difference_canonicalizes_to_ordered_clamp(self):
    plan = lower_dpu(sink(Tensor.empty(8,dtype=dtypes.half).hardsigmoid()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(tuple(stage.op.name for stage in plan.stages), ("MUL", "ADD", "MAX", "MUL", "MAX", "MUL"))
    self.assertEqual(len(plan.scratch), 1)

  def test_hardswish_uses_two_generated_luts(self):
    plan = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).hardswish()))
    self.assertEqual((len(plan.stages), len(plan.scratch)), (36, 5))
    self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.HARDSWISH), 1)
    self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.HARDSWISH_LOCAL), 1)
    self.assertFalse(contains_uop(plan))

  def test_tanh_uses_two_generated_luts(self):
    plan = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).tanh()))
    self.assertEqual((len(plan.stages), len(plan.scratch)), (35, 5))
    self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.TANH), 1)
    self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.TANH_LOCAL), 1)
    self.assertFalse(contains_uop(plan))

  def test_sigmoid_and_silu_reuse_two_generated_luts(self):
    sigmoid = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).sigmoid()))
    silu = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).silu()))
    self.assertEqual((len(sigmoid.stages), len(silu.stages)), (24, 25))
    for plan in (sigmoid, silu):
      self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.SIGMOID), 1)
      self.assertEqual(tuple(stage.op for stage in plan.stages).count(RKDPUOp.SIGMOID_LOCAL), 1)
      self.assertFalse(contains_uop(plan))

  def test_reciprocal_lowers_to_typed_division(self):
    x, y = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink(x.reciprocal()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op.name, "DIV")
    self.assertFalse(contains_uop(plan))
    self.assertEqual(tuple(stage.op.name for stage in lower_dpu(sink(x/y)).stages), ("DIV",))

  def test_abs_canonicalizes_to_mul_max(self):
    plan = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).abs()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(tuple(stage.op.name for stage in plan.stages), ("MUL", "MAX"))
    self.assertFalse(contains_uop(plan))

  def test_direct_affine_contract_is_typed(self):
    a, packed_b = Tensor.empty(1,32,dtype=dtypes.half), Tensor.empty(8,32,dtype=dtypes.half)
    plan = lower_contract(sink(a@packed_b.T))
    self.assertIsInstance(plan, RKContract)
    self.assertFalse(contains_uop(plan))
    image = emit_contract(plan)
    self.assertEqual(len(image.stages[0].commands), 46)
    self.assertEqual(tuple(r.word for r in image.stages[0].relocs), (18,24,31))
    self.assertEqual(image.stages[0].commands[:30], (
      0x10010000000e4004, 0x020120000120100c, 0x0201000040001010, 0x0201000000091014,
      0x0201000100011020, 0x0201001f00201024, 0x0201000000011028, 0x020100000001102c,
      0x0201000008001030, 0x0201000000401034, 0x0201010100201038, 0x0201000000b11040,
      0x0201000000011044, 0x02010000000b104c, 0x0201000100001050, 0x0201000100001054,
      0x0201000100001058, 0x020100010000105c, 0x0201000000001070, 0x0201000f000f1078,
      0x020100000004107c, 0x0201000000001080, 0x0201000100011084, 0x0201000000201088,
      0x0201000000001110, 0x0801000002013010, 0x0801000000003014, 0x08010000001f3018,
      0x0801000000003030, 0x1001000001e4400c))

  def test_contract_rejects_unpacked_rhs(self):
    a, b = Tensor.empty(1,32,dtype=dtypes.half), Tensor.empty(32,8,dtype=dtypes.half)
    self.assertIsNone(lower_contract(sink(a@b)))

  def test_spatial_conv_rejects_without_device_layout_stage(self):
    inp, weight = Tensor.empty(1,8,5,5,dtype=dtypes.half), Tensor.empty(8,8,3,3,dtype=dtypes.half)
    with self.assertRaisesRegex(RuntimeError, "RKPLAN_REJECT:unsupported_graph"):
      RockchipRenderer(Target("ROCKCHIP")).native_program(sink(inp.conv2d(weight)))

  def test_row_sum_is_constant_backed_contract(self):
    plan = lower_contract(sink(Tensor.empty(8,32,dtype=dtypes.half).sum(axis=1)))
    self.assertIsInstance(plan, RKContract)
    self.assertFalse(contains_uop(plan))
    self.assertIs(plan.lhs.kind, RKBufferKind.CONSTANT)
    image = emit_contract(plan)
    self.assertEqual(len(image.constants), 64)
    self.assertEqual(image.stages[0].reads, (plan.rhs.slot,))
    self.assertEqual(image.stages[0].relocs[0].kind, RKBufferKind.CONSTANT)

  def test_global_max_requires_explicit_hwc_layout(self):
    source = Tensor.empty(8,8,dtype=dtypes.half)
    plan = lower_pool(sink(source.max(axis=0)))
    self.assertIsInstance(plan, RKPool)
    self.assertFalse(contains_uop(plan))
    image = emit_pool(plan)
    self.assertEqual(len(image.stages[0].commands), 25)
    self.assertEqual(tuple(r.word for r in image.stages[0].relocs), (10,17))
    self.assertIsNone(lower_pool(sink(Tensor.empty(8,7,dtype=dtypes.half).max(axis=0))))

if __name__ == "__main__": unittest.main()
