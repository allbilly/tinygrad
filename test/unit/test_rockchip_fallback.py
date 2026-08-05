import os, unittest
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import RockchipRenderer
from tinygrad.runtime.rockchip_fallback import (RKHC_MAGIC, RKPY_MAGIC, build_rkhc_program, build_rkpy_program,
                                               decode_rkhc, decode_rkpy, encode_rkhc, encode_rkpy)
from tinygrad.uop.ops import KernelInfo, Ops

def sink(tensor:Tensor):
  return next(x.src[0] for x in tensor.schedule_linear().src if x.src[0].op is Ops.SINK).replace(arg=KernelInfo())

class TestRockchipFallback(unittest.TestCase):
  def test_rkpy_roundtrip_and_validation(self):
    payload = b"generic linear uops"
    self.assertEqual(decode_rkpy(encode_rkpy(payload)), payload)
    for malformed in (b"", b"RKPY", encode_rkpy(payload)[:-1], b"NOPE"+encode_rkpy(payload)[4:]):
      with self.assertRaises(ValueError): decode_rkpy(malformed)

  def test_rkhc_roundtrip_and_validation(self):
    payload = b"generic compiled uops"
    self.assertEqual(decode_rkhc(encode_rkhc(payload)), payload)
    for malformed in (b"", b"RKHC", encode_rkhc(payload)[:-1], b"NOPE"+encode_rkhc(payload)[4:]):
      with self.assertRaises(ValueError): decode_rkhc(malformed)

  def test_builds_explicit_python_program_envelope(self):
    program = build_rkpy_program(sink(Tensor.empty(7, dtype=dtypes.half).sin()), Target("ROCKCHIP"))
    self.assertEqual(program.src[3].arg[:4], RKPY_MAGIC)
    self.assertGreater(len(decode_rkpy(program.src[3].arg)), 0)
    self.assertEqual(program.arg.target, Target("ROCKCHIP"))
    self.assertTrue(any(u.op is Ops.SIN for u in program.src[1].src))

  def test_builds_explicit_host_program_envelope(self):
    program = build_rkhc_program(sink(Tensor.empty(7, dtype=dtypes.half).sin()), Target("ROCKCHIP"))
    self.assertEqual(program.src[3].arg[:4], RKHC_MAGIC)
    self.assertGreater(len(decode_rkhc(program.src[3].arg)), 0)
    self.assertEqual(program.arg.target, Target("ROCKCHIP"))

  def test_strict_mode_still_rejects(self):
    with self.assertRaisesRegex(RuntimeError, "RKPLAN_REJECT"):
      x, y = Tensor.empty(7,dtype=dtypes.float), Tensor.empty(7,dtype=dtypes.float)
      RockchipRenderer(Target("ROCKCHIP")).native_program(sink(x+y))

  def test_host_mode_bypasses_native_lowering(self):
    with patch.dict(os.environ, {"ROCKCHIP_FALLBACK":"HOST"}):
      program = RockchipRenderer(Target("ROCKCHIP")).native_program(sink(Tensor.empty(7, dtype=dtypes.half)+1))
    assert program is not None
    self.assertEqual(program.src[3].arg[:4], RKHC_MAGIC)

if __name__ == "__main__": unittest.main()
