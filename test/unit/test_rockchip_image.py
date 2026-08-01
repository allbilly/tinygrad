import unittest
from tinygrad.renderer.rockchip import (RKBufferKind, RKEngine, RKImage, RKReloc, RKScratch, RKStage, RKTarget,
                                        decode_image, encode_image, patch_image)

class TestRKImage(unittest.TestCase):
  def test_roundtrip_is_deterministic(self):
    image = RKImage(RKTarget.RK3588, (
      RKStage(RKEngine.DPU, (0x1001000012340040, 0x0081000000180008),
              (RKReloc(0, 0, RKBufferKind.ARG, 1, addend=64, shift=4, mask=0xfffffff0),), (1,), (0,)),
      RKStage(RKEngine.CMAC, (7, 8), dependencies=1, reads=(0,), writes=(2,))),
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
    expected = ((0x12345000 + 0x40) >> 4) & 0x00fffff0
    self.assertEqual((patched >> 16) & 0x00fffff0, expected)

  def test_relocation_field_shift_applies_after_source_mask(self):
    image = RKImage(RKTarget.RK3588, (RKStage(RKEngine.PPU, (0x4001000000006070,),
      (RKReloc(0, 0, RKBufferKind.ARG, 0, shift=4, mask=0x0fffffff, field_shift=4),)),))
    patched = patch_image(image, lambda kind, index: 0x12345670)[0][0]
    self.assertEqual((patched >> 16) & 0xffffffff, 0x12345670)

  def test_large_constant_relocation_roundtrip(self):
    image = RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, (0,),
      (RKReloc(0, 0, RKBufferKind.CONSTANT, 70000),)),), constants=b"\0"*70002)
    self.assertEqual(decode_image(encode_image(image)), image)

  def test_rejects_malformed_images(self):
    image = RKImage(RKTarget.RK3588, (RKStage(RKEngine.DPU, (1,), dependencies=1),))
    with self.assertRaises(ValueError): encode_image(image)
    valid = encode_image(RKImage(RKTarget.RK3588, ()))
    for blob in (valid[:-1], valid+b"x", b"NOPE"+valid[4:]):
      with self.assertRaises(ValueError): decode_image(blob)

if __name__ == "__main__": unittest.main()
