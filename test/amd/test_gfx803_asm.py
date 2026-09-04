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

  def test_dynamic_indexing(self):
    s2, v0 = _reg(dtypes.int32, "s2", 2), _reg(dtypes.int32, "v0", 0)
    idx, tmp = _reg(dtypes.int32, "v36", 36), _reg(dtypes.int32, "v37", 37)
    ptr, addr = _reg(dtypes.uint64, "s[6:7]", 6, 8), _reg(dtypes.uint64, "v[4:5]", 4, 8)
    cases = [
      (UOp(Ops.INS, dtypes.int32, (s2,), GFX803Ops.V_MOV_B32, idx.tag), ("v_mov_b32_e32 v36, s2",)),
      (UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.int32, 2), v0), GFX803Ops.V_LSHLREV_B32, idx.tag),
       ("v_lshlrev_b32_e32 v36, 2, v0",)),
      (UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.int32, 1), v0), GFX803Ops.V_ADD_U32, idx.tag),
       ("v_add_u32_e32 v36, vcc, 1, v0",)),
      (UOp(Ops.INS, dtypes.int32, (v0, UOp.const(dtypes.int32, 4)), GFX803Ops.V_MUL_LO_U32, tmp.tag),
       ("v_mul_lo_u32 v37, v0, 4",)),
      (UOp(Ops.INS, dtypes.uint64, (ptr, idx), GFX803Ops.FLAT_ADDR, addr.tag),
       ("v_mov_b32_e32 v4, s6", "v_mov_b32_e32 v5, s7", "v_add_u32_e32 v4, vcc, v36, v4",
        "v_addc_u32_e32 v5, vcc, 0, v5, vcc")),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))

  def test_float_alu_and_select(self):
    a, b, cond, out = (_reg(dtype, f"v{i}", i) for dtype, i in
                       ((dtypes.float32, 36), (dtypes.float32, 37), (dtypes.bool, 38), (dtypes.float32, 39)))
    cases = [
      (UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.float32, 2.5),), GFX803Ops.V_MOV_B32, out.tag),
       ("v_mov_b32_e32 v39, 2.5",)),
      (UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.float32, 2.5), b), GFX803Ops.V_ADD_F32, out.tag),
       ("v_add_f32_e32 v39, 2.5, v37",)),
      (UOp(Ops.INS, dtypes.float32, (a, b), GFX803Ops.V_MUL_F32, out.tag), ("v_mul_f32_e32 v39, v36, v37",)),
      (UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.float32, 0), b), GFX803Ops.V_MAX_F32, out.tag),
       ("v_max_f32_e32 v39, 0, v37",)),
      (UOp(Ops.INS, dtypes.bool, (a, b), GFX803Ops.V_CMPLT, cond.tag),
       ("v_mov_b32_e32 v38, 1", "v_cmp_lt_f32_e32 vcc, v36, v37", "v_cndmask_b32_e32 v38, 0, v38, vcc")),
      (UOp(Ops.INS, dtypes.bool, (a, b), GFX803Ops.V_CMPNE, cond.tag),
       ("v_mov_b32_e32 v38, 1", "v_cmp_neq_f32_e32 vcc, v36, v37", "v_cndmask_b32_e32 v38, 0, v38, vcc")),
      (UOp(Ops.INS, dtypes.float32, (cond, a, b), GFX803Ops.V_CNDMASK_B32, out.tag),
       ("v_cmp_ne_u32_e32 vcc, 0, v38", "v_cndmask_b32_e32 v39, v37, v36, vcc")),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))


class TestGFX803Program(unittest.TestCase):
  @staticmethod
  def _add_program(n:int):
    a = Tensor.empty(n, dtype=dtypes.float32, device="NULL").contiguous().realize()
    b = Tensor.empty(n, dtype=dtypes.float32, device="NULL").contiguous().realize()
    return to_program((a+b).schedule_linear().src[-1].src[0], AMDASMRenderer(Target("AMD", "ASM", "gfx803")))

  @staticmethod
  def _elf(program):
    image, sections, relocs = elf_loader(program.src[-1].arg)
    text = next(section for section in sections if section.name == ".text")
    rodata = next(section for section in sections if section.name == ".rodata")
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(
      bytes(image[rodata.header.sh_addr:rodata.header.sh_addr+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
    return text, rodata, desc, relocs

  def test_tinygrad_float4_add_elf(self):
    program = self._add_program(4)
    self.assertEqual([u.op for u in program.src], [Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY])

    text, rodata, desc, relocs = self._elf(program)

    self.assertEqual(relocs, [])
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

  def test_dynamic_add_elf(self):
    wgid_x = 1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT
    for n, grid, local, uses_wgid, uses_lidx in [(5, (5, 1, 1), (1, 1, 1), True, False),
                                                  (16, (1, 1, 1), (4, 1, 1), False, True),
                                                  (1024, (8, 1, 1), (32, 1, 1), True, True)]:
      with self.subTest(n=n):
        program = self._add_program(n)
        text, rodata, desc, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        self.assertEqual(relocs, [])
        self.assertEqual((program.arg.global_size, program.arg.local_size), (grid, local))
        self.assertEqual(bool(desc.compute_pgm_rsrc2 & wgid_x), uses_wgid)
        self.assertEqual(rodata.header.sh_addr + desc.kernel_code_entry_byte_offset, text.header.sh_addr)
        self.assertEqual(_assembled(*lines), text.content)
        self.assertEqual(any("s2" in line for line in lines), uses_wgid)
        self.assertEqual(any(line.startswith(("v_add_u32", "v_lshlrev")) and "v0" in line for line in lines), uses_lidx)

  def test_elementwise_alu_elf(self):
    a = Tensor.empty(16, dtype=dtypes.float32, device="NULL").contiguous().realize()
    b = Tensor.empty(16, dtype=dtypes.float32, device="NULL").contiguous().realize()
    cases = {
      "add_const": (a+2.5, ("v_add_f32",)), "sub": (a-b, ("v_mul_f32", "v_add_f32")),
      "mul": (a*b, ("v_mul_f32",)), "max": (a.maximum(b), ("v_max_f32",)),
      "relu": (a.relu(), ("v_cmp_lt_f32", "v_cndmask_b32")), "where": ((a<b).where(a, b), ("v_cmp_lt_f32", "v_cndmask_b32")),
      "full": (Tensor.full((16,), 2.5, dtype=dtypes.float32, device="NULL").contiguous(), ("v_mov_b32",)),
    }
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    for name, (result, expected) in cases.items():
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, _, _ = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        self.assertEqual(_assembled(*lines), text.content)
        for mnemonic in expected: self.assertTrue(any(line.startswith(mnemonic) for line in lines), mnemonic)


if __name__ == "__main__": unittest.main()
