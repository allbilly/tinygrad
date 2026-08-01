import hashlib, struct, unittest
from dataclasses import fields, is_dataclass
from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.rockchip import (RKALUStage, RKBufferKind, RKContract, RKDPUProgram, RKLUTStage, RKMaskStage, RockchipRenderer,
  decode_image, emit_contract, emit_dpu, encode_image, lower_contract, lower_dpu)
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
      0x1001000000004034, 0x1001000000004038, 0x100100070007403c, 0x1001000000534040,
      0x1001000000534060, 0x1001000000004044, 0x1001000000004048, 0x1001000000024050,
      0x1001000000074058, 0x100100000001405c, 0x1001000000004068, 0x100100000000406c,
      0x1001108202c04070, 0x1001000000014078, 0x1001000000004080, 0x1001000100014084,
      0x1001000000004088, 0x10010000004040c0, 0x20010000000e5004, 0x200100000001500c,
      0x2001000000005010, 0x2001000000075014, 0x2001400000085034, 0x1001000000004020,
      0x2001000000005018, 0x2001000000005038, 0x2001000178495044, 0x0081000000180008))
    self.assertEqual(tuple((r.word, r.kind, r.index) for r in image.stages[0].relocs),
                     ((27, RKBufferKind.ARG, 0), (28, RKBufferKind.ARG, 1), (29, RKBufferKind.ARG, 2)))

  def test_wide_fills_tile_native_wdma_limits(self):
    for dtype, count, tile in ((dtypes.int, 2925, 64), (dtypes.float, 6, 4)):
      plan = lower_dpu(sink(Tensor.full((count,), 4, dtype=dtype)))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual(len(plan.stages), (count+tile-1)//tile)
      self.assertTrue(all(isinstance(stage, RKALUStage) and stage.count <= tile and stage.out_dtype is dtype for stage in plan.stages))
      self.assertEqual(tuple(stage.dst.addend for stage in plan.stages if isinstance(stage, RKALUStage)),
                       tuple(range(0, count*4, tile*4)))
      image = emit_dpu(plan)
      self.assertEqual(decode_image(encode_image(image)), image)
    self.assertIsNone(lower_dpu(sink(Tensor.full((257,), 4, dtype=dtypes.float))))

  def test_plan_is_uop_free_and_reuses_scratch(self):
    a, b, c, d = (Tensor.empty(16,dtype=dtypes.half) for _ in range(4))
    plan = lower_dpu(sink(((a+b)*c)+d))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))
    self.assertEqual((len(plan.stages), len(plan.scratch), len(emit_dpu(plan).stages)), (3, 1, 3))

  def test_scalar_fill_uses_const_zero_index(self):
    plan = lower_dpu(sink(Tensor.ones((), dtype=dtypes.half)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].count, 1)

  def test_fp16_abs_and_ordered_extrema_use_generic_alu(self):
    x = Tensor.empty(16,dtype=dtypes.half)
    absolute = lower_dpu(sink(x.abs()))
    self.assertIsInstance(absolute, RKDPUProgram)
    self.assertEqual(tuple(stage.op for stage in absolute.stages if isinstance(stage, RKALUStage)), (Ops.MUL, Ops.MAX))
    for expression in (x.relu(), x.clip(-1, 1)):
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertFalse(contains_uop(plan))
    self.assertEqual(lower_dpu(sink(x.relu())).stages[0].op, Ops.MAX)

  def test_relu_difference_uses_stable_generic_clip(self):
    plan = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).hardsigmoid()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertLessEqual(len(plan.stages), 6)
    self.assertTrue(any(isinstance(stage, RKALUStage) and stage.op is Ops.MAX for stage in plan.stages))
    self.assertFalse(contains_uop(plan))

  def test_composed_fp16_predicates_preserve_mask_liveness(self):
    x = Tensor.empty(3,dtype=dtypes.half)
    sign, clipped = lower_dpu(sink(x.sign())), lower_dpu(sink(x.clip(0, 0)))
    self.assertIsInstance(sign, RKDPUProgram)
    self.assertIsInstance(clipped, RKDPUProgram)
    self.assertTrue(any(isinstance(stage, RKMaskStage) for stage in sign.stages))
    self.assertFalse(contains_uop(sign))

  def test_threshold_where_avoids_multiply_blend(self):
    x = Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink((x < 0).where(x, 1)))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertFalse(contains_uop(plan))

  def test_division_canonicalizes_to_generic_fdiv(self):
    a, b = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    plan = lower_dpu(sink(a/b))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(plan.stages[0].op, Ops.FDIV)
    signed_inf = lower_dpu(sink(float("inf")/a))
    self.assertIsInstance(signed_inf, RKDPUProgram)
    self.assertTrue(any(isinstance(stage, RKMaskStage) for stage in signed_inf.stages))
    self.assertFalse(contains_uop(signed_inf))

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

  def test_exp_composes_generated_generic_luts(self):
    for name in ("EXP", "EXP_LOCAL"):
      table = getattr(rklut, f"RK_LUT_{name}")
      payload = struct.pack(f"<{len(table)}h", *table)
      self.assertEqual(hashlib.sha256(payload).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_EXP_SIM_MAX_REL_ERROR, 1e-3)
    plan = lower_dpu(sink(Tensor.empty(128, dtype=dtypes.half).exp()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)},
                     {rklut.RKLUTId.EXP, rklut.RKLUTId.EXP_LOCAL})
    self.assertFalse(contains_uop(plan))
    self.assertEqual(len(emit_dpu(plan).stages), len(plan.stages))

  def test_expm1_composes_generated_generic_luts(self):
    for name in ("EXPM1", "EXPM1_LOCAL"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_EXPM1_SIM_MAX_ABS_ERROR, 3e-3)
    x = Tensor.empty(128, dtype=dtypes.half)
    for expression in (x.exp()-1, 1-x.exp()):
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)},
                       {rklut.RKLUTId.EXPM1, rklut.RKLUTId.EXPM1_LOCAL})
      self.assertFalse(contains_uop(plan))

  def test_sigmoid_family_uses_generated_generic_luts(self):
    for name in ("SIGMOID", "SIGMOID_LOCAL"):
      table = getattr(rklut, f"RK_LUT_{name}")
      payload = struct.pack(f"<{len(table)}h", *table)
      self.assertEqual(hashlib.sha256(payload).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_SIGMOID_SIM_MAX_ABS_ERROR, 3e-4)
    for expression in (Tensor.empty(128, dtype=dtypes.half).sigmoid(), Tensor.empty(128, dtype=dtypes.half).silu(),
                       Tensor.empty(128, dtype=dtypes.half).quick_gelu()):
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)},
                       {rklut.RKLUTId.SIGMOID, rklut.RKLUTId.SIGMOID_LOCAL})
      self.assertFalse(contains_uop(plan))

  def test_tanh_uses_generated_tables_and_local_polynomial(self):
    for name in ("TANH", "TANH_MID", "TANH_LOCAL"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_TANH_SIM_MAX_REL_ERROR, 1e-3)
    plan = lower_dpu(sink(Tensor.empty(128, dtype=dtypes.half).tanh()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)},
                     {rklut.RKLUTId.TANH, rklut.RKLUTId.TANH_MID})
    self.assertFalse(contains_uop(plan))

  def test_inverse_trig_uses_generated_math_assets(self):
    for name in ("ASIN", "ASIN_LOCAL", "ASIN_EDGE", "ACOS", "ATAN"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_ASIN_SIM_MAX_ABS_ERROR, 3e-3)
    self.assertLess(rklut.RK_LUT_ACOS_SIM_MAX_ABS_ERROR, 3e-3)
    self.assertLess(rklut.RK_LUT_ATAN_SIM_MAX_REL_ERROR, 1e-3)
    expected = ((Tensor.empty(128, dtype=dtypes.half).asin(), {rklut.RKLUTId.ASIN, rklut.RKLUTId.ASIN_EDGE}),
                (Tensor.empty(128, dtype=dtypes.half).acos(), {rklut.RKLUTId.ACOS, rklut.RKLUTId.ASIN_EDGE}),
                (Tensor.empty(128, dtype=dtypes.half).atan(), {rklut.RKLUTId.ATAN}))
    for expression, luts in expected:
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)}, luts)
      self.assertLessEqual(len(plan.stages), 64)
      self.assertFalse(contains_uop(plan))

  def test_atanh_uses_generated_broad_and_edge_assets(self):
    for name in ("ATANH", "ATANH_EDGE"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_ATANH_SIM_MAX_REL_ERROR, 1e-3)
    plan = lower_dpu(sink(Tensor.empty(128,dtype=dtypes.half).atanh()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)},
                     {rklut.RKLUTId.ATANH, rklut.RKLUTId.ATANH_EDGE})
    self.assertLessEqual(len(plan.stages), 64)
    self.assertFalse(contains_uop(plan))

  def test_inverse_hyperbolic_uses_generated_multirange_assets(self):
    for name in ("ASINH", "ASINH_MID", "ASINH_NEAR", "ACOSH", "ACOSH_MID", "ACOSH_EDGE"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    self.assertLess(rklut.RK_LUT_ASINH_SIM_MAX_REL_ERROR, 1e-3)
    self.assertLess(rklut.RK_LUT_ACOSH_SIM_MAX_REL_ERROR, 1e-3)
    expected = ((Tensor.empty(128,dtype=dtypes.half).asinh(), {rklut.RKLUTId.ASINH, rklut.RKLUTId.ASINH_MID, rklut.RKLUTId.ASINH_NEAR}),
                (Tensor.empty(128,dtype=dtypes.half).acosh(), {rklut.RKLUTId.ACOSH, rklut.RKLUTId.ACOSH_MID, rklut.RKLUTId.ACOSH_EDGE}))
    for expression, luts in expected:
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)}, luts)
      self.assertLessEqual(len(plan.stages), 64)
      self.assertFalse(contains_uop(plan))

  def test_sqrt_uses_generated_seed_and_generic_refinement(self):
    payload = struct.pack(f"<{len(rklut.RK_LUT_SQRT)}h", *rklut.RK_LUT_SQRT)
    self.assertEqual(hashlib.sha256(payload).hexdigest(), rklut.RK_LUT_SQRT_SHA256)
    self.assertLess(rklut.RK_LUT_SQRT_SIM_MAX_REL_ERROR, 1e-3)
    plan = lower_dpu(sink(Tensor.empty(16, dtype=dtypes.half).sqrt()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(sum(isinstance(stage, RKLUTStage) and stage.lut is rklut.RKLUTId.SQRT for stage in plan.stages), 1)
    self.assertLessEqual(len(plan.stages), 32)
    self.assertFalse(contains_uop(plan))

  def test_rsqrt_uses_generated_seed_and_generic_refinement(self):
    payload = struct.pack(f"<{len(rklut.RK_LUT_RSQRT)}h", *rklut.RK_LUT_RSQRT)
    self.assertEqual(hashlib.sha256(payload).hexdigest(), rklut.RK_LUT_RSQRT_SHA256)
    self.assertLess(rklut.RK_LUT_RSQRT_SIM_MAX_REL_ERROR, 1e-3)
    plan = lower_dpu(sink(Tensor.empty(16, dtype=dtypes.half).rsqrt()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(sum(isinstance(stage, RKLUTStage) and stage.lut is rklut.RKLUTId.RSQRT for stage in plan.stages), 1)
    self.assertLessEqual(len(plan.stages), 48)
    self.assertFalse(contains_uop(plan))

  def test_logarithms_use_generated_tables_and_native_subtraction(self):
    for name in ("LOG2", "LOG2_LOCAL", "LOG10", "LOG10_LOCAL"):
      table = getattr(rklut, f"RK_LUT_{name}")
      self.assertEqual(hashlib.sha256(struct.pack(f"<{len(table)}h", *table)).hexdigest(), getattr(rklut, f"RK_LUT_{name}_SHA256"))
    for expression, ids in ((Tensor.empty(16,dtype=dtypes.half).log2(), {rklut.RKLUTId.LOG2,rklut.RKLUTId.LOG2_LOCAL}),
                            (Tensor.empty(16,dtype=dtypes.half).log10(), {rklut.RKLUTId.LOG10,rklut.RKLUTId.LOG10_LOCAL})):
      plan = lower_dpu(sink(expression))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual({stage.lut for stage in plan.stages if isinstance(stage, RKLUTStage)}, ids)
      self.assertTrue(any(isinstance(stage, RKALUStage) and stage.op is Ops.SUB for stage in plan.stages))
      self.assertLessEqual(len(plan.stages), 58)
      self.assertFalse(contains_uop(plan))
    x, y = Tensor.empty(16,dtype=dtypes.half), Tensor.empty(16,dtype=dtypes.half)
    composed = lower_dpu(sink((x.log2()*y).exp2()))
    self.assertIsInstance(composed, RKDPUProgram)
    self.assertEqual(len(composed.stages), 64)
    self.assertEqual({stage.lut for stage in composed.stages if isinstance(stage, RKLUTStage)},
                     {rklut.RKLUTId.LOG2, rklut.RKLUTId.LOG2_LOCAL, rklut.RKLUTId.EXP2})
    self.assertFalse(contains_uop(composed))

  def test_round_uses_generated_algorithm23_lut(self):
    payload = struct.pack(f"<{len(rklut.RK_LUT_ROUNDOFF)}h", *rklut.RK_LUT_ROUNDOFF)
    self.assertEqual(hashlib.sha256(payload).hexdigest(), rklut.RK_LUT_ROUNDOFF_SHA256)
    plan = lower_dpu(sink(Tensor.empty(16,dtype=dtypes.half).round()))
    self.assertIsInstance(plan, RKDPUProgram)
    self.assertEqual(sum(isinstance(stage, RKLUTStage) and stage.lut is rklut.RKLUTId.ROUNDOFF for stage in plan.stages), 1)
    self.assertFalse(contains_uop(plan))

  def test_trunc_floor_ceil_compose_roundoff_lut(self):
    for function in (lambda x:x.trunc(), lambda x:x.floor(), lambda x:x.ceil()):
      plan = lower_dpu(sink(function(Tensor.empty(16,dtype=dtypes.half))))
      self.assertIsInstance(plan, RKDPUProgram)
      self.assertEqual(sum(isinstance(stage, RKLUTStage) and stage.lut is rklut.RKLUTId.ROUNDOFF for stage in plan.stages), 1)
      self.assertFalse(contains_uop(plan))

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
