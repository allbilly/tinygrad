import ctypes
import unittest

from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa.gfx803 import AMDASMRenderer, GFX803Ops, _encode
from tinygrad.runtime.autogen import amdgpu_kd
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import Ops, UOp
from test.amd.helpers import llvm_assemble, llvm_disasm


def _reg(dtype, name:str, index:int, size:int=4) -> UOp:
  return UOp(Ops.INS, dtype, arg=GFX803Ops.DEFINE, tag=(Register(name, index, size=size),))


def _assembled(*lines:str) -> bytes:
  return b"".join(llvm_assemble(list(lines), "gfx803", "+wavefrontsize64"))


class TestGFX803Encoder(unittest.TestCase):
  def test_scalar_memory_add(self):
    kernarg, ptr = _reg(dtypes.uint64, "s[0:1]", 0, 8), _reg(dtypes.uint64, "s[2:3]", 2, 8)
    addr = _reg(dtypes.uint64, "v[2:3]", 2, 8)
    a, b, out = (_reg(dtypes.float32, f"v{i}", i) for i in (32, 33, 34))

    cases = [
      (UOp(Ops.INS, dtypes.uint64, (kernarg, UOp.const(dtypes.uint32, 8)), GFX803Ops.S_LOAD_B64, ptr.tag),
       ("s_load_dwordx2 s[2:3], s[0:1], 0x8",)),
      (UOp(Ops.INS, dtypes.uint64, (ptr, UOp.const(dtypes.uint32, 4)), GFX803Ops.FLAT_ADDR, addr.tag),
       ("v_mov_b32_e32 v2, s2", "v_mov_b32_e32 v3, s3", "v_add_u32 v2, vcc, 4, v2", "v_addc_u32 v3, vcc, 0, v3, vcc")),
      (UOp(Ops.INS, dtypes.float32, (addr,), GFX803Ops.FLAT_LOAD_B32, a.tag), ("flat_load_dword v32, v[2:3]",)),
      (UOp(Ops.INS, dtypes.float32, (a, b), GFX803Ops.V_ADD_F32, out.tag), ("v_add_f32_e32 v34, v32, v33",)),
      (UOp(Ops.INS, dtypes.void, (addr, out), GFX803Ops.FLAT_STORE_B32), ("flat_store_dword v[2:3], v34",)),
      (UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_WAITCNT), ("s_waitcnt 0",)),
      (UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_ENDPGM), ("s_endpgm",)),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))


class TestGFX803Program(unittest.TestCase):
  def test_tinygrad_float4_add_elf(self):
    a = Tensor.empty(4, dtype=dtypes.float32, device="NULL").contiguous().realize()
    b = Tensor.empty(4, dtype=dtypes.float32, device="NULL").contiguous().realize()
    ast = (a+b).schedule_linear().src[-1].src[0]
    program = to_program(ast, AMDASMRenderer(Target("AMD", "ASM", "gfx803")))
    self.assertEqual([u.op for u in program.src], [Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY])

    image, sections, relocs = elf_loader(program.src[-1].arg)
    self.assertEqual(relocs, [])
    text = next(section for section in sections if section.name == ".text")
    rodata = next(section for section in sections if section.name == ".rodata")
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(
      bytes(image[rodata.header.sh_addr:rodata.header.sh_addr+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))

    self.assertEqual(desc.kernarg_size, 24)
    self.assertEqual(desc.kernel_code_properties, 1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT)
    self.assertEqual(rodata.header.sh_addr + desc.kernel_code_entry_byte_offset, text.header.sh_addr)
    self.assertEqual(len(text.content) % 256, 0)

    lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
    code_lines = lines[:lines.index("s_endpgm")+1]
    self.assertEqual(_assembled(*lines), text.content)
    self.assertEqual(sum(line.startswith("v_add_f32") for line in code_lines), 4)
    self.assertEqual(sum(line.startswith("flat_load_dword") for line in code_lines), 8)
    self.assertEqual(sum(line.startswith("flat_store_dword") for line in code_lines), 4)


if __name__ == "__main__": unittest.main()
