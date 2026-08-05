import os, unittest
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import RockchipRenderer
from tinygrad.runtime.ops_rockchip import RKHostMemory, RockchipAllocator
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

  def test_mixed_mode_can_fall_back_to_non_dma_host_allocation(self):
    class Device:
      def _gpu_alloc(self, size): raise OSError("no contiguous DMA surface")
      def _gpu_free(self, buf): raise AssertionError("host memory must not use DRM free")
    allocator = RockchipAllocator(Device())  # type: ignore[arg-type]
    for mode in ("CLANG","COST"):
      with self.subTest(mode=mode), patch.dict(os.environ, {"ROCKCHIP_FALLBACK":mode}):
        buf = allocator._alloc(12345, BufferSpec())
        self.assertIsInstance(buf.meta, RKHostMemory)
        allocator._as_buffer(buf)[:4] = b"RKHC"
        self.assertEqual(bytes(allocator._as_buffer(buf)[:4]), b"RKHC")
        allocator._free(buf, BufferSpec())

  def test_cost_mode_routes_pathological_native_plan_to_compiled_host(self):
    x, y, z = (Tensor.empty(1575,dtype=dtypes.half) for _ in range(3))
    expression = sink(x.lerp(y,z))
    with patch.dict(os.environ, {"ROCKCHIP_FALLBACK":"COST"}):
      program = RockchipRenderer(Target("ROCKCHIP")).native_program(expression)
    assert program is not None
    self.assertEqual(program.src[3].arg[:4],RKHC_MAGIC)
    with patch.dict(os.environ, {"ROCKCHIP_FALLBACK":"CLANG"}):
      native = RockchipRenderer(Target("ROCKCHIP")).native_program(expression)
    assert native is not None
    self.assertEqual(native.src[3].arg[:4],b"RKIM")

  def test_invalid_fallback_mode_rejects_even_when_graph_is_native(self):
    with patch.dict(os.environ, {"ROCKCHIP_FALLBACK":"TYPO"}), self.assertRaisesRegex(RuntimeError,"invalid ROCKCHIP_FALLBACK"):
      RockchipRenderer(Target("ROCKCHIP")).native_program(sink(Tensor.empty(8,dtype=dtypes.half)+1))

  def test_strict_mode_never_falls_back_to_host_allocation(self):
    class Device:
      def _gpu_alloc(self, size): raise OSError("no contiguous DMA surface")
    allocator = RockchipAllocator(Device())  # type: ignore[arg-type]
    with patch.dict(os.environ, {"ROCKCHIP_FALLBACK":"0"}), self.assertRaises(OSError):
      allocator._alloc(12345, BufferSpec())

if __name__ == "__main__": unittest.main()
