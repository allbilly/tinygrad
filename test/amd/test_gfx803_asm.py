import ctypes, unittest

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

def _renderer(): return AMDASMRenderer(Target("AMD", "ASM", "gfx803"))
def _tensor(dtype, *shape): return Tensor.empty(*shape, dtype=dtype, device="NULL").contiguous().realize()

def _assembled(*lines:str) -> bytes: return b"".join(llvm_assemble(list(lines), "gfx803", "+wavefrontsize64"))

# Golden bytes were checked against LLVM assembly for each case. Operand prefixes preserve mixed dtypes.
ENCODINGS = [
  ("V_AND_B32 uint v42 v40 v41", "28535426"),
  ("V_OR_B32 uint v42 3 v41", "83525428"),
  ("V_XOR_B32 uint v42 v40 v41", "2853542a"),
  ("V_MUL_HI_U32 uint v42 v40 v41", "2a0086d228530200"),
  ("V_BFE_I32 int v45 short:v44 uint:0 uint:16", "2d00c9d12c014102"),
  ("V_BFE_U32 uint v42 v40 0 8", "2a00c8d128012102"),
  ("V_LSHRREV_B32 uint v42 8 v41", "88525420"),
  ("V_ASHRREV_I32 int v43 uint:8 v43", "88565622"),
  ("V_MAX_I32 int v43 v43 uint:v40", "2b51561a"),
  ("V_MAX_U32 uint v42 v40 v41", "2853541e"),
  ("V_MOV_B32 int v36 s2", "0202487e"),
  ("V_LSHLREV_B32 int v36 2 v0", "82004824"),
  ("V_ADD_U32 int v36 1 v0", "81004832"),
  ("V_MUL_LO_U32 int v37 v0 4", "250085d200090100"),
  ("FLAT_ADDR ulong v[4:5] s[6:7] int:v36", "0602087e07020a7e24090832800a0a38"),
  ("V_MOV_B32 float v39 2.5", "ff024e7e00002040"),
  ("V_ADD_F32 float v39 2.5 v37", "ff4a4e0200002040"),
  ("V_MUL_F32 float v39 v36 v37", "244b4e0a"),
  ("V_MAX_F32 float v39 0.0 v37", "804a4e16"),
  ("V_CMPLT bool v38 float:v36 float:v37", "244b827c260000d18002a901"),
  ("V_CMPNE bool v38 float:v36 float:v37", "244b9a7c260000d18002a901"),
  ("V_CNDMASK_B32 float v39 bool:v38 v36 v37", "804c9a7d25494e00"),
  ("V_CMPNE bool v36 float:v36 float:v37", "244b9a7c240000d18002a901"),
  ("V_MOV_B32 half v42 2.5", "ff02547e00410000"),
  ("V_MOV_B32 half v42 -2147483648.0", "ff02547e00fc0000"),
  ("FLAT_LOAD_U16 half v42 ulong:v[4:5]", "000048dc0400002a"),
  ("FLAT_STORE_B16 void - ulong:v[4:5] half:v42", "000068dc042a0000"),
  ("DS_LOAD_U16 half v42 uint:v45", "000078d82d00002a"),
  ("DS_STORE_B16 void - uint:v45 half:v42", "00003ed82d2a0000"),
  ("V_ADD_F32 half v42 v40 v41", "2853543e"),
  ("V_MUL_F32 half v42 v40 v41", "28535444"),
  ("V_MAX_F32 half v42 v40 v41", "2853545a"),
  ("V_ADD_F32 half v42 2.5 v41", "ff52543e00410000"),
  ("V_CVT_F32_F16 float v43 half:v40", "2817567e"),
  ("V_CVT_F16_F32 half v42 float:v43", "2b15547e"),
  ("V_CVT_I16_F16 short v46 half:v40", "28795c7e"),
  ("V_CVT_U16_F16 ushort v47 half:v40", "28775e7e"),
  ("V_CVT_F16_I16 half v42 short:v46", "2e75547e"),
  ("V_CVT_F16_U16 half v42 ushort:v47", "2f73547e"),
  ("V_TRUNC half v42 v40", "288d547e"),
  ("V_TRUNC float v43 v43", "2b39567e"),
  ("V_CMPLT bool v44 half:v40 half:v41", "2853427c2c0000d18002a901"),
  ("GATED_FLAT_STORE_B16 void - ulong:v[4:5] half:v42 bool:v44", "80589a7d6a20a0be000068dc042a00002001febe"),
  ("FLAT_LOAD_U8 bool v44 ulong:v[4:5]", "000040dc0400002c"),
  ("FLAT_LOAD_U8 uchar v47 ulong:v[4:5]", "000040dc0400002f"),
  ("FLAT_LOAD_S8 char v48 ulong:v[4:5]", "000044dc04000030"),
  ("FLAT_LOAD_U16 ushort v50 ulong:v[4:5]", "000048dc04000032"),
  ("FLAT_LOAD_S16 short v51 ulong:v[4:5]", "00004cdc04000033"),
  ("GATED_FLAT_LOAD_U16 half v43 ulong:v[4:5] 0.0 bool:v46", "805c9a7d8002567e6a20a0be000048dc0400002b2001febe"),
  ("FLAT_STORE_B8 void - ulong:v[4:5] bool:v44", "000060dc042c0000"),
  ("DS_LOAD_U8 bool v44 uint:v45", "000074d82d00002c"),
  ("DS_LOAD_S8 char v48 uint:v45", "000072d82d000030"),
  ("DS_LOAD_S16 short v51 uint:v45", "000076d82d000033"),
  ("DS_STORE_B8 void - uint:v45 bool:v44", "00003cd82d2c0000"),
  ("FLAT_LOAD_B32 int v40 ulong:v[4:5]", "000050dc04000028"),
  ("FLAT_STORE_B32 void - ulong:v[4:5] uint:v41", "000070dc04290000"),
  ("V_CVT_F32_I32 float v42 int:v40", "280b547e"),
  ("V_CVT_F32_U32 float v42 uint:v41", "290d547e"),
  ("V_CVT_I32_F32 int v40 float:v42", "2a11507e"),
  ("V_CVT_U32_F32 uint v41 float:v42", "2a0f527e"),
  ("V_CVT_F32_F16 float v42 half:v43", "2b17547e"),
  ("V_CMPLT bool v44 char:v48 char:v49", "3063827d2c0000d18002a901"),
  ("GATED_FLAT_STORE_B8 void - ulong:v[4:5] bool:v44 bool:v46", "805c9a7d6a20a0be000060dc042c00002001febe"),
  ("GATED_FLAT_LOAD_U16 half v46 ulong:v[4:5] 0.0 bool:v46", "805c9a7d80025c7e6a20a0be000048dc0400002e2001febe"),
  ("LDS_ADDR uint v37 int:v36 int:2 lds", "82484a24"),
  ("DS_LOAD_B32 float v39 uint:v37", "00006cd825000027"),
  ("DS_STORE_B32 void - uint:v37 float:v38", "00001ad825260000"),
  ("S_BARRIER void -", "00008abf"),
  ("V_CMP_GT_U32 void - int:16 int:v36", "9048987d"),
  ("S_CBRANCH_VCCNZ void -10", "f6ff87bf"),
  ("GATED_FLAT_STORE_B32 void - ulong:v[4:5] float:v38 bool:v40", "80509a7d6a20a0be000070dc042600002001febe"),
  ("V_RCP_F32 float v42 v40", "2845547e"),
  ("V_RCP_IFLAG_F32 float v42 v40", "2847547e"),
  ("V_SQRT_F32 float v42 v40", "284f547e"),
  ("V_RSQ_F32 float v42 v40", "2849547e"),
  ("V_RCP_F32 half v43 v41", "297b567e"),
  ("V_SQRT_F32 half v43 v41", "297d567e"),
  ("V_RSQ_F32 half v43 v41", "297f567e"),
  ("V_EXP2_F32 float v42 v40", "2841547e"),
  ("V_LOG2_F32 float v42 v40", "2843547e"),
  ("V_EXP2_F32 half v43 v41", "2983567e"),
  ("V_LOG2_F32 half v43 v41", "2981567e"),
  ("V_SIN float v42 v40", "2853547e"),
  ("V_SIN half v43 v41", "2993567e"),
  ("SCRATCH_STORE_B64 void - ulong:v[4:5] int:8", "080070e0000409800c0070e000050980"),
  ("SCRATCH_LOAD_B64 ulong v[4:5] int:8", "080050e0000409800c0050e000050980"),
  ("SCRATCH_STORE_B32 void - float:v84 int:4", "040070e000540980"),
  ("SCRATCH_LOAD_B32 float v84 int:4", "040050e000540980"),
  ("SCRATCH_STORE_B16 void - half:v85 int:2", "020068e000550980"),
  ("SCRATCH_LOAD_U16 half v85 int:2", "020048e000550980"),
  ("SCRATCH_STORE_B16 void - short:v86 int:6", "060068e000560980"),
  ("SCRATCH_LOAD_S16 short v86 int:6", "06004ce000560980"),
  ("SCRATCH_STORE_B8 void - uchar:v87 int:1", "010060e000570980"),
  ("SCRATCH_LOAD_U8 uchar v87 int:1", "010040e000570980"),
  ("SCRATCH_STORE_B8 void - char:v88 int:3", "030060e000580980"),
  ("SCRATCH_LOAD_S8 char v88 int:3", "030044e000580980"),
  ("S_LOAD_B64 ulong s[2:3] s[0:1] uint:8", "800006c008000000"),
  ("FLAT_ADDR ulong v[2:3] s[2:3] uint:4", "0202047e0302067e8404043280060638"),
  ("FLAT_LOAD_B32 float v32 ulong:v[2:3]", "000050dc02000020"),
  ("V_ADD_F32 float v34 v32 v33", "20434402"),
  ("FLAT_STORE_B32 void - ulong:v[2:3] float:v34", "000070dc02220000"),
  ("S_WAITCNT void -", "00008cbf"),
  ("S_ENDPGM void -", "000081bf"),
]

def _operand(text, dtype):
  if ":" in text and not text.startswith(("v[", "s[")):
    name, text = text.split(":", 1)
    dtype = getattr(dtypes, name)
  if text == "lds": return UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 16)), GFX803Ops.LDS_BUFFER)
  if text.startswith(("v", "s")):
    index = int(text[2:].split(":")[0]) if "[" in text else int(text[1:])
    return UOp(Ops.INS, dtype, arg=GFX803Ops.DEFINE, tag=(Register(text, index, size=8 if "[" in text else 4),))
  return UOp.const(dtype, float(text) if dtypes.is_float(dtype) else int(text))

class TestGFX803Encoder(unittest.TestCase):
  def test_encodings(self):
    for case, expected in ENCODINGS:
      with self.subTest(case=case):
        op, dtype, dst, *src = case.split()
        dtype, op = getattr(dtypes, dtype), GFX803Ops[op]
        branch = op is GFX803Ops.S_CBRANCH_VCCNZ
        tag = dst if branch else None if dst == "-" else _operand(dst, dtype).tag
        inst = _encode(UOp(Ops.INS, dtype, tuple(_operand(s, dtype) for s in src), op, tag), int(dst) if branch else 0)
        self.assertEqual(inst.to_bytes(), bytes.fromhex(expected))
        self.assertEqual(inst.to_bytes(), _assembled(*llvm_disasm(inst.to_bytes(), "gfx803", "+wavefrontsize64")))

  def test_private_scratch(self):
    renderer = _renderer()
    setup = _encode(UOp(Ops.INS, dtypes.void, (UOp.const(dtypes.uint32, 144),), GFX803Ops.SCRATCH_SETUP))
    self.assertEqual(setup.scratch_size, 144)
    self.assertEqual(setup.to_bytes(), _assembled("s_add_u32 s0, s0, s9", "s_addc_u32 s1, s1, 0",
      "s_mov_b64 s[36:37], s[0:1]", "s_mov_b64 s[38:39], s[2:3]", "s_mov_b64 s[0:1], s[4:5]",
      "s_mov_b32 s2, s6", "s_mov_b32 s3, s7", "s_mov_b32 s4, s8"))
    for dtype in (dtypes.uint64, dtypes.float32, dtypes.half, dtypes.int16, dtypes.uint8, dtypes.int8):
      with self.subTest(dtype=dtype):
        value, offset = _operand("v[4:5]" if dtype is dtypes.uint64 else "v84", dtype), UOp.const(dtypes.int32, 8)
        for inst in (renderer.spill(offset, value), renderer.fill(offset, value, value.tag[0])):
          binary = _encode(inst).to_bytes()
          self.assertEqual(binary, _assembled(*llvm_disasm(binary, "gfx803", "+wavefrontsize64")))
    lds = _encode(_operand("lds", dtypes.float32))
    self.assertEqual(lds.lds_size, 64)
    self.assertEqual(lds.to_bytes(), _assembled("s_mov_b32 m0, -1"))

class TestGFX803Program(unittest.TestCase):
  def test_elf_contracts(self):
    # Numerical/operator coverage lives in test/backend/test_ops.py. Check the ISA-specific ABI here.
    inputs = [_tensor(dtypes.float32, 64) for _ in range(4)]
    cases = [("five_buffers", sum(inputs[1:], start=inputs[0]), None, None)]
    cases += [(f"add_{n}", _tensor(dtypes.float32, n) + _tensor(dtypes.float32, n), grid, local)
              for n, grid, local in ((4, (1,1,1), (1,1,1)), (5, (5,1,1), (1,1,1)), (16, (1,1,1), (4,1,1)), (1024, (8,1,1), (32,1,1)))]
    cases += [(f"matmul_{n}", _tensor(dtypes.float32, n, n) @ _tensor(dtypes.float32, n, n), (1,1,1), (n,n,1)) for n in (2,4,8)]
    cases += [("sum", _tensor(dtypes.float32, 2, 16).sum(axis=1), (2,1,1), (16,1,1)),
              ("matmul", _tensor(dtypes.float32, 16, 16) @ _tensor(dtypes.float32, 16, 16), (16,16,1), (16,1,1)),
              ("spill", _tensor(dtypes.half, 1,1,5,5,5).pad((1,2,3,4,1,2), mode="replicate").contiguous(), None, None),
              ("private", _tensor(dtypes.half, 1,1,5,5).max_pool2d(kernel_size=(3,3), padding=1, return_indices=True)[1], None, None)]
    for name, result, grid, local in cases:
      with self.subTest(name=name):
        program = to_program(result.schedule_linear().src[-1].src[0], _renderer())
        self.assertEqual([u.op for u in program.src], [Ops.SINK, Ops.LINEAR, Ops.SOURCE, Ops.BINARY])
        image, sections, relocs = elf_loader(program.src[-1].arg)
        text, rodata = (next(s for s in sections if s.name == name) for name in (".text", ".rodata"))
        desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(
          bytes(image[rodata.header.sh_addr:rodata.header.sh_addr+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
        lines = llvm_disasm(text.content, "gfx803", "+wavefrontsize64")
        self.assertEqual(relocs, [])
        self.assertEqual(_assembled(*lines), text.content)
        self.assertEqual(rodata.header.sh_addr + desc.kernel_code_entry_byte_offset, text.header.sh_addr)
        self.assertEqual(len(text.content) % 256, 0)
        if grid is not None: self.assertEqual((program.arg.global_size, program.arg.local_size), (grid, local))
        if name.startswith("add_") or name == "five_buffers":
          self.assertEqual(desc.kernarg_size, 40 if name == "five_buffers" else 24)
          self.assertEqual(desc.kernel_code_properties, 1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT)
        if name == "five_buffers":
          self.assertEqual((desc.compute_pgm_rsrc1 >> 6) & 0xf, 2)
          self.assertTrue(any(line.startswith("s_load_dwordx2 s[14:15]") for line in lines))
        if name in ("sum", "matmul"):
          self.assertEqual(desc.group_segment_fixed_size, 64)
          self.assertEqual((desc.compute_pgm_rsrc1 >> 6) & 0xf, 4)
          branch = next(line for line in lines if line.startswith("s_cbranch_vccnz"))
          self.assertEqual(int(branch.rsplit(" ", 1)[1]) & 0xffff, 0xfff6)
          for op in ("s_mov_b32 m0, -1", "ds_write_b32", "ds_read_b32", "s_barrier", "s_and_saveexec_b64", "flat_store_dword"):
            self.assertEqual(sum(line.startswith(op) for line in lines), 1)
        if name in ("spill", "private"):
          self.assertGreater(desc.private_segment_fixed_size, 0)
          self.assertEqual((desc.compute_pgm_rsrc2 >> amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT) & 0x1f, 6)
          self.assertTrue(desc.compute_pgm_rsrc2 & (1 << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT))
          for op in ("buffer_store_dword", "buffer_load_dword"): self.assertTrue(any(line.startswith(op) for line in lines))
          self.assertFalse(any(line.startswith(("buffer_store_dwordx2", "buffer_load_dwordx2")) for line in lines))
        if name == "private":
          self.assertTrue(any(u.arg is GFX803Ops.SCRATCH_BUFFER_META for u in program.src[1].src))
          self.assertEqual(desc.private_segment_fixed_size, 144)
          self.assertIn("buffer_store_dword v84, off, s[36:39], 0 offset:140", lines)

if __name__ == "__main__": unittest.main()
