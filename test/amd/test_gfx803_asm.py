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
       ("v_cmp_lt_f32_e32 vcc, v36, v37", "v_cndmask_b32_e64 v38, 0, 1, vcc")),
      (UOp(Ops.INS, dtypes.bool, (a, b), GFX803Ops.V_CMPNE, cond.tag),
       ("v_cmp_neq_f32_e32 vcc, v36, v37", "v_cndmask_b32_e64 v38, 0, 1, vcc")),
      (UOp(Ops.INS, dtypes.float32, (cond, a, b), GFX803Ops.V_CNDMASK_B32, out.tag),
       ("v_cmp_ne_u32_e32 vcc, 0, v38", "v_cndmask_b32_e32 v39, v37, v36, vcc")),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))

    # The destination may alias a dying input. The compare must read v36
    # before materializing its boolean result back into v36.
    aliased = UOp(Ops.INS, dtypes.bool, (a, b), GFX803Ops.V_CMPNE, a.tag)
    self.assertEqual(_encode(aliased).to_bytes(), _assembled(
      "v_cmp_neq_f32_e32 vcc, v36, v37", "v_cndmask_b32_e64 v36, 0, 1, vcc"))

  def test_lds_loop_and_gated_store(self):
    idx, lds_addr = _reg(dtypes.int32, "v36", 36), _reg(dtypes.uint32, "v37", 37)
    value, loaded, gate = _reg(dtypes.float32, "v38", 38), _reg(dtypes.float32, "v39", 39), _reg(dtypes.bool, "v40", 40)
    global_addr = _reg(dtypes.uint64, "v[4:5]", 4, 8)
    lds = UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 16)), GFX803Ops.LDS_BUFFER, True)

    lds_inst = _encode(lds)
    self.assertEqual(lds_inst.lds_size, 64)
    self.assertEqual(lds_inst.to_bytes(), _assembled("s_mov_b32 m0, -1"))
    cases = [
      (UOp(Ops.INS, dtypes.uint32, (idx, UOp.const(dtypes.int32, 2), lds), GFX803Ops.LDS_ADDR, lds_addr.tag),
       ("v_lshlrev_b32_e32 v37, 2, v36",), 0),
      (UOp(Ops.INS, dtypes.float32, (lds_addr,), GFX803Ops.DS_LOAD_B32, loaded.tag),
       ("ds_read_b32 v39, v37",), 0),
      (UOp(Ops.INS, dtypes.void, (lds_addr, value), GFX803Ops.DS_STORE_B32),
       ("ds_write_b32 v37, v38",), 0),
      (UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_BARRIER), ("s_barrier",), 0),
      (UOp(Ops.INS, dtypes.void, (UOp.const(dtypes.int32, 16), idx), GFX803Ops.V_CMP_GT_U32),
       ("v_cmp_gt_u32_e32 vcc, 16, v36",), 0),
      (UOp(Ops.INS, dtypes.void, arg=GFX803Ops.S_CBRANCH_VCCNZ, tag=".LOOP"),
       ("s_cbranch_vccnz -10",), -10),
      (UOp(Ops.INS, dtypes.void, (global_addr, value, gate), GFX803Ops.GATED_FLAT_STORE_B32),
       ("v_cmp_ne_u32_e32 vcc, 0, v40", "s_and_saveexec_b64 s[32:33], vcc",
        "flat_store_dword v[4:5], v38", "s_mov_b64 exec, s[32:33]"), 0),
    ]
    for uop, lines, branch_offset in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop, branch_offset).to_bytes(), _assembled(*lines))

  def test_half_memory_alu_and_casts(self):
    a, b, out = (_reg(dtypes.half, f"v{i}", i) for i in (40, 41, 42))
    out_float, gate = _reg(dtypes.float32, "v43", 43), _reg(dtypes.bool, "v44", 44)
    addr, lds_addr = _reg(dtypes.uint64, "v[4:5]", 4, 8), _reg(dtypes.uint32, "v45", 45)
    cases = [
      (UOp(Ops.INS, dtypes.half, (UOp.const(dtypes.half, 2.5),), GFX803Ops.V_MOV_B32, out.tag),
       ("v_mov_b32_e32 v42, 0x4100",)),
      (UOp(Ops.INS, dtypes.half, (addr,), GFX803Ops.FLAT_LOAD_U16, out.tag),
       ("flat_load_ushort v42, v[4:5]",)),
      (UOp(Ops.INS, dtypes.void, (addr, out), GFX803Ops.FLAT_STORE_B16),
       ("flat_store_short v[4:5], v42",)),
      (UOp(Ops.INS, dtypes.half, (lds_addr,), GFX803Ops.DS_LOAD_U16, out.tag), ("ds_read_u16 v42, v45",)),
      (UOp(Ops.INS, dtypes.void, (lds_addr, out), GFX803Ops.DS_STORE_B16), ("ds_write_b16 v45, v42",)),
      (UOp(Ops.INS, dtypes.half, (a, b), GFX803Ops.V_ADD_F32, out.tag), ("v_add_f16_e32 v42, v40, v41",)),
      (UOp(Ops.INS, dtypes.half, (a, b), GFX803Ops.V_MUL_F32, out.tag), ("v_mul_f16_e32 v42, v40, v41",)),
      (UOp(Ops.INS, dtypes.half, (a, b), GFX803Ops.V_MAX_F32, out.tag), ("v_max_f16_e32 v42, v40, v41",)),
      (UOp(Ops.INS, dtypes.half, (UOp.const(dtypes.half, 2.5), b), GFX803Ops.V_ADD_F32, out.tag),
       ("v_add_f16_e32 v42, 2.5, v41",)),
      (UOp(Ops.INS, dtypes.float32, (a,), GFX803Ops.V_CVT_F32_F16, out_float.tag), ("v_cvt_f32_f16_e32 v43, v40",)),
      (UOp(Ops.INS, dtypes.half, (out_float,), GFX803Ops.V_CVT_F16_F32, out.tag), ("v_cvt_f16_f32_e32 v42, v43",)),
      (UOp(Ops.INS, dtypes.bool, (a, b), GFX803Ops.V_CMPLT, gate.tag),
       ("v_cmp_lt_f16_e32 vcc, v40, v41", "v_cndmask_b32_e64 v44, 0, 1, vcc")),
      (UOp(Ops.INS, dtypes.void, (addr, out, gate), GFX803Ops.GATED_FLAT_STORE_B16),
       ("v_cmp_ne_u32_e32 vcc, 0, v44", "s_and_saveexec_b64 s[32:33], vcc",
        "flat_store_short v[4:5], v42", "s_mov_b64 exec, s[32:33]")),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))

  def test_integer_bool_memory_and_casts(self):
    i32, u32 = _reg(dtypes.int32, "v40", 40), _reg(dtypes.uint32, "v41", 41)
    f32, half = _reg(dtypes.float32, "v42", 42), _reg(dtypes.half, "v43", 43)
    boolean, gate = _reg(dtypes.bool, "v44", 44), _reg(dtypes.bool, "v46", 46)
    addr, lds_addr = _reg(dtypes.uint64, "v[4:5]", 4, 8), _reg(dtypes.uint32, "v45", 45)
    cases = [
      (UOp(Ops.INS, dtypes.bool, (addr,), GFX803Ops.FLAT_LOAD_U8, boolean.tag), ("flat_load_ubyte v44, v[4:5]",)),
      (UOp(Ops.INS, dtypes.half, (addr, UOp.const(dtypes.half, 0), gate), GFX803Ops.GATED_FLAT_LOAD_U16, half.tag),
       ("v_mov_b32_e32 v43, 0", "v_cmp_ne_u32_e32 vcc, 0, v46", "s_and_saveexec_b64 s[32:33], vcc",
        "flat_load_ushort v43, v[4:5]", "s_mov_b64 exec, s[32:33]")),
      (UOp(Ops.INS, dtypes.void, (addr, boolean), GFX803Ops.FLAT_STORE_B8), ("flat_store_byte v[4:5], v44",)),
      (UOp(Ops.INS, dtypes.bool, (lds_addr,), GFX803Ops.DS_LOAD_U8, boolean.tag), ("ds_read_u8 v44, v45",)),
      (UOp(Ops.INS, dtypes.void, (lds_addr, boolean), GFX803Ops.DS_STORE_B8), ("ds_write_b8 v45, v44",)),
      (UOp(Ops.INS, dtypes.int32, (addr,), GFX803Ops.FLAT_LOAD_B32, i32.tag), ("flat_load_dword v40, v[4:5]",)),
      (UOp(Ops.INS, dtypes.void, (addr, u32), GFX803Ops.FLAT_STORE_B32), ("flat_store_dword v[4:5], v41",)),
      (UOp(Ops.INS, dtypes.float32, (i32,), GFX803Ops.V_CVT_F32_I32, f32.tag), ("v_cvt_f32_i32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.float32, (u32,), GFX803Ops.V_CVT_F32_U32, f32.tag), ("v_cvt_f32_u32_e32 v42, v41",)),
      (UOp(Ops.INS, dtypes.int32, (f32,), GFX803Ops.V_CVT_I32_F32, i32.tag), ("v_cvt_i32_f32_e32 v40, v42",)),
      (UOp(Ops.INS, dtypes.uint32, (f32,), GFX803Ops.V_CVT_U32_F32, u32.tag), ("v_cvt_u32_f32_e32 v41, v42",)),
      (UOp(Ops.INS, dtypes.float32, (half,), GFX803Ops.V_CVT_F32_F16, f32.tag), ("v_cvt_f32_f16_e32 v42, v43",)),
      (UOp(Ops.INS, dtypes.void, (addr, boolean, gate), GFX803Ops.GATED_FLAT_STORE_B8),
       ("v_cmp_ne_u32_e32 vcc, 0, v46", "s_and_saveexec_b64 s[32:33], vcc",
        "flat_store_byte v[4:5], v44", "s_mov_b64 exec, s[32:33]")),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))

  def test_bitwise_and_shifts(self):
    a, b, out = (_reg(dtypes.uint32, f"v{i}", i) for i in (40, 41, 42))
    signed = _reg(dtypes.int32, "v43", 43)
    cases = [
      (UOp(Ops.INS, dtypes.uint32, (a, b), GFX803Ops.V_AND_B32, out.tag), ("v_and_b32_e32 v42, v40, v41",)),
      (UOp(Ops.INS, dtypes.uint32, (UOp.const(dtypes.uint32, 3), b), GFX803Ops.V_OR_B32, out.tag),
       ("v_or_b32_e32 v42, 3, v41",)),
      (UOp(Ops.INS, dtypes.uint32, (a, b), GFX803Ops.V_XOR_B32, out.tag), ("v_xor_b32_e32 v42, v40, v41",)),
      (UOp(Ops.INS, dtypes.uint32, (UOp.const(dtypes.uint32, 8), b), GFX803Ops.V_LSHRREV_B32, out.tag),
       ("v_lshrrev_b32_e32 v42, 8, v41",)),
      (UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.uint32, 8), signed), GFX803Ops.V_ASHRREV_I32, signed.tag),
       ("v_ashrrev_i32_e32 v43, 8, v43",)),
      (UOp(Ops.INS, dtypes.int32, (signed, a), GFX803Ops.V_MAX_I32, signed.tag), ("v_max_i32_e32 v43, v43, v40",)),
      (UOp(Ops.INS, dtypes.uint32, (a, b), GFX803Ops.V_MAX_U32, out.tag), ("v_max_u32_e32 v42, v40, v41",)),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))

  def test_native_reciprocal_sqrt(self):
    f32, out_f32 = _reg(dtypes.float32, "v40", 40), _reg(dtypes.float32, "v42", 42)
    half, out_half = _reg(dtypes.half, "v41", 41), _reg(dtypes.half, "v43", 43)
    cases = [
      (UOp(Ops.INS, dtypes.float32, (f32,), GFX803Ops.V_RCP_F32, out_f32.tag), ("v_rcp_f32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.float32, (f32,), GFX803Ops.V_SQRT_F32, out_f32.tag), ("v_sqrt_f32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.float32, (f32,), GFX803Ops.V_RSQ_F32, out_f32.tag), ("v_rsq_f32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.half, (half,), GFX803Ops.V_RCP_F32, out_half.tag), ("v_rcp_f16_e32 v43, v41",)),
      (UOp(Ops.INS, dtypes.half, (half,), GFX803Ops.V_SQRT_F32, out_half.tag), ("v_sqrt_f16_e32 v43, v41",)),
      (UOp(Ops.INS, dtypes.half, (half,), GFX803Ops.V_RSQ_F32, out_half.tag), ("v_rsq_f16_e32 v43, v41",)),
      (UOp(Ops.INS, dtypes.float32, (f32,), GFX803Ops.V_EXP2_F32, out_f32.tag), ("v_exp_f32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.float32, (f32,), GFX803Ops.V_LOG2_F32, out_f32.tag), ("v_log_f32_e32 v42, v40",)),
      (UOp(Ops.INS, dtypes.half, (half,), GFX803Ops.V_EXP2_F32, out_half.tag), ("v_exp_f16_e32 v43, v41",)),
      (UOp(Ops.INS, dtypes.half, (half,), GFX803Ops.V_LOG2_F32, out_half.tag), ("v_log_f16_e32 v43, v41",)),
    ]
    for uop, lines in cases:
      with self.subTest(op=uop.arg, dtype=uop.dtype): self.assertEqual(_encode(uop).to_bytes(), _assembled(*lines))


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

  def test_small_matmul_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    for n in (2, 4, 8):
      with self.subTest(n=n):
        a = Tensor.empty(n, n, dtype=dtypes.float32, device="NULL").contiguous().realize()
        b = Tensor.empty(n, n, dtype=dtypes.float32, device="NULL").contiguous().realize()
        program = to_program((a@b).schedule_linear().src[-1].src[0], renderer)
        text, _, _, _ = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        self.assertEqual((program.arg.global_size, program.arg.local_size), ((1, 1, 1), (n, n, 1)))
        self.assertEqual(_assembled(*lines), text.content)
        self.assertEqual(sum(line.startswith("v_mul_f32") for line in lines), n)
        self.assertEqual(sum(line.startswith("v_add_f32") for line in lines), n-1)

  def test_grouped_reduction_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    x = Tensor.empty(2, 16, dtype=dtypes.float32, device="NULL").contiguous().realize()
    a = Tensor.empty(16, 16, dtype=dtypes.float32, device="NULL").contiguous().realize()
    b = Tensor.empty(16, 16, dtype=dtypes.float32, device="NULL").contiguous().realize()
    for name, result, grid in (("sum", x.sum(axis=1), (2, 1, 1)), ("matmul", a@b, (16, 16, 1))):
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, desc, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        branch = next(line for line in code_lines if line.startswith("s_cbranch_vccnz"))
        branch_offset = int(branch.rsplit(" ", 1)[1])
        branch_offset -= 0x10000 if branch_offset & 0x8000 else 0

        self.assertEqual(relocs, [])
        self.assertEqual((program.arg.global_size, program.arg.local_size), (grid, (16, 1, 1)))
        self.assertEqual(desc.group_segment_fixed_size, 64)
        self.assertEqual((desc.compute_pgm_rsrc1 >> 6) & 0xf, 4)  # s[32:33] makes 40 allocated SGPRs.
        self.assertEqual(_assembled(*lines), text.content)
        self.assertEqual(branch_offset, -10)
        self.assertEqual(sum(line == "s_mov_b32 m0, -1" for line in code_lines), 1)
        self.assertEqual(sum(line.startswith("ds_write_b32") for line in code_lines), 1)
        self.assertEqual(sum(line.startswith("ds_read_b32") for line in code_lines), 1)
        self.assertEqual(sum(line == "s_barrier" for line in code_lines), 1)
        self.assertEqual(sum(line.startswith("s_and_saveexec_b64") for line in code_lines), 1)
        self.assertEqual(sum(line.startswith("flat_store_dword") for line in code_lines), 1)

  def test_half_programs_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    a = Tensor.empty(16, dtype=dtypes.half, device="NULL").contiguous().realize()
    b = Tensor.empty(16, dtype=dtypes.half, device="NULL").contiguous().realize()
    ma = Tensor.empty(16, 16, dtype=dtypes.half, device="NULL").contiguous().realize()
    mb = Tensor.empty(16, 16, dtype=dtypes.half, device="NULL").contiguous().realize()
    cases = {
      "add": (a+b, ("flat_load_ushort", "v_add_f16", "flat_store_short")),
      "sum": (a.sum(), ("flat_load_ushort", "v_cvt_f32_f16", "ds_write_b32", "v_add_f32", "v_cvt_f16_f32", "flat_store_short")),
      "matmul": (ma@mb, ("flat_load_ushort", "v_mul_f16", "ds_write_b32", "v_add_f32", "flat_store_short")),
    }
    for name, (result, expected) in cases.items():
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, _, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        self.assertEqual(relocs, [])
        self.assertEqual(_assembled(*lines), text.content)
        for mnemonic in expected: self.assertTrue(any(line.startswith(mnemonic) for line in code_lines), mnemonic)

  def test_integer_bool_programs_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    i32 = Tensor.empty(16, dtype=dtypes.int32, device="NULL").contiguous().realize()
    half = Tensor.empty(16, dtype=dtypes.half, device="NULL").contiguous().realize()
    boolean = Tensor.empty(16, dtype=dtypes.bool, device="NULL").contiguous().realize()
    cases = {
      "int_add": (i32+2, ("flat_load_dword", "v_add_u32", "flat_store_dword")),
      "half_compare": (half<0, ("flat_load_ushort", "v_cmp_lt_f16", "flat_store_byte")),
      "bool_to_uint": (boolean.cast(dtypes.uint32), ("flat_load_ubyte", "v_mov_b32", "flat_store_dword")),
      "int_to_float": (i32.float(), ("flat_load_dword", "v_cvt_f32_i32", "flat_store_dword")),
    }
    for name, (result, expected) in cases.items():
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, _, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        self.assertEqual(relocs, [])
        self.assertEqual(_assembled(*lines), text.content)
        for mnemonic in expected: self.assertTrue(any(line.startswith(mnemonic) for line in code_lines), mnemonic)

  def test_bitwise_programs_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    i32 = Tensor.empty(16, dtype=dtypes.int32, device="NULL").contiguous().realize()
    u32 = Tensor.empty(16, dtype=dtypes.uint32, device="NULL").contiguous().realize()
    boolean = Tensor.empty(16, dtype=dtypes.bool, device="NULL").contiguous().realize()
    cases = {
      "int_and": (i32 & 255, ("v_and_b32",)),
      "uint_or_xor": ((u32 | 3) ^ 1, ("v_or_b32", "v_xor_b32")),
      "logical_shift": (u32 >> 3, ("v_lshrrev_b32",)),
      "arithmetic_shift": (i32 >> 3, ("v_ashrrev_i32",)),
      "bool_and": (boolean & (boolean != True), ("v_cmp_ne_u32", "v_and_b32")),  # noqa: E712
      "gated_load": (i32.pad((1, 1)).contiguous(), ("s_and_saveexec_b64", "flat_load_dword")),
    }
    for name, (result, expected) in cases.items():
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, _, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        self.assertEqual(relocs, [])
        self.assertEqual(_assembled(*lines), text.content)
        for mnemonic in expected: self.assertTrue(any(line.startswith(mnemonic) for line in code_lines), mnemonic)

  def test_native_reciprocal_sqrt_programs_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    cases = {
      "rcp_f32": (dtypes.float32, "reciprocal", ("v_rcp_f32",)), "sqrt_f32": (dtypes.float32, "sqrt", ("v_sqrt_f32",)),
      "rsqrt_f32": (dtypes.float32, "rsqrt", ("v_rsq_f32",)), "rcp_f16": (dtypes.half, "reciprocal", ("v_rcp_f16",)),
      "sqrt_f16": (dtypes.half, "sqrt", ("v_sqrt_f16",)), "rsqrt_f16": (dtypes.half, "rsqrt", ("v_rsq_f16",)),
      "exp2_f32": (dtypes.float32, "exp2", ("v_exp_f32",)), "log2_f32": (dtypes.float32, "log2", ("v_log_f32",)),
      "exp2_f16": (dtypes.half, "exp2", ("v_exp_f16",)), "log2_f16": (dtypes.half, "log2", ("v_log_f16",)),
    }
    for name, (dtype, op, expected) in cases.items():
      with self.subTest(name=name):
        source = Tensor.empty(16, dtype=dtype, device="NULL").contiguous().realize()
        result = getattr(source, op)()
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, _, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        self.assertEqual(relocs, [])
        self.assertEqual(_assembled(*lines), text.content)
        for mnemonic in expected: self.assertTrue(any(line.startswith(mnemonic) for line in code_lines), mnemonic)

  def test_integer_max_reductions_elf(self):
    renderer = AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
    cases = {
      "signed": (Tensor.empty(2, 16, dtype=dtypes.int32, device="NULL").contiguous().realize().max(axis=1), "v_max_i32"),
      "unsigned": (Tensor.empty(2, 16, dtype=dtypes.uint32, device="NULL").contiguous().realize().max(axis=1), "v_max_u32"),
      "argmax": (Tensor.empty(32, dtype=dtypes.half, device="NULL").contiguous().realize().argmax(), "v_max_i32"),
    }
    for name, (result, mnemonic) in cases.items():
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], renderer)
        text, _, desc, relocs = self._elf(program)
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        code_lines = lines[:lines.index("s_endpgm")+1]
        self.assertEqual(relocs, [])
        self.assertGreater(desc.group_segment_fixed_size, 0)
        self.assertEqual(_assembled(*lines), text.content)
        self.assertTrue(any(line.startswith(mnemonic) for line in code_lines))


if __name__ == "__main__": unittest.main()
