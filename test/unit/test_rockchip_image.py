import unittest
from tinygrad.renderer.rockchip import (RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKEngine, RKImage, RKLUTId, RKLUTStage,
  RKMaskStage, RKReloc, RKScratch, RKStage, RKTarget, decode_image, encode_image, patch_image)
from tinygrad.uop.ops import Ops, UOp

class TestRKImage(unittest.TestCase):
  def test_typed_stage_union_has_no_uops(self):
    arg0, arg1 = RKArg(RKBufferKind.ARG, 0), RKArg(RKBufferKind.ARG, 1)
    plan = RKDPUProgram((RKALUStage(Ops.ADD, arg0, arg1, 1.0, 8), RKMaskStage(arg0, arg1, 8),
                         RKLUTStage(RKLUTId.EXP2, arg0, arg1, 8)))
    self.assertFalse(any(isinstance(x, UOp) for stage in plan.stages for x in stage.__dict__.values()))
    with self.assertRaises(ValueError): RKALUStage(Ops.SIN, arg0, arg1, 1.0, 8)

  def test_roundtrip_is_deterministic(self):
    image = RKImage(RKTarget.RK3588, (
      RKStage(RKEngine.DPU, (0x1001000012340040, 0x0081000000180008),
              (RKReloc(0, 0, RKBufferKind.ARG, 1, addend=64, shift=4, mask=0xfffffff0),), (1,), (0,)),),
      (RKScratch(8192, 4096),), b"constant payload")
    blob = encode_image(image)
    self.assertEqual(decode_image(blob), image)
    self.assertEqual(encode_image(decode_image(blob)), blob)

  def test_relocation_patches_only_value_field(self):
    command = 0x1001a5a5a5a50040
    image = RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, (command,),
      (RKReloc(0, 0, RKBufferKind.ARG, 3, addend=0x40, shift=4, mask=0x00fffff0),)),))
    patched = patch_image(image, lambda kind, index: 0x12345000)[0][0]
    self.assertEqual(patched & 0xffff000000000000, command & 0xffff000000000000)
    self.assertEqual(patched & 0xffff, command & 0xffff)
    self.assertEqual((patched >> 16) & 0x00fffff0, ((((0x12345000+0x40) >> 4) << 0) & 0x00fffff0))

  def test_rejects_malformed_images(self):
    with self.assertRaises(ValueError): encode_image(RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, (1,), dependencies=1),)))
    valid = encode_image(RKImage(RKTarget.RK3588, ()))
    for blob in (valid[:-1], valid+b"x", b"NOPE"+valid[4:]):
      with self.assertRaises(ValueError): decode_image(blob)

if __name__ == "__main__": unittest.main()
