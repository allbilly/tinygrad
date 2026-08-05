from __future__ import annotations
import platform, struct
from dataclasses import replace
from typing import Any, cast
from tinygrad.codegen import to_program
from tinygrad.device import Device, TinyELF
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKPY_MAGIC, RKPY_VERSION = b"RKPY", 1
RKHC_MAGIC, RKHC_VERSION = b"RKHC", 1
_RKPY_HEADER = struct.Struct("<4sHI")

def _encode(magic:bytes, version:int, payload:bytes) -> bytes: return _RKPY_HEADER.pack(magic, version, len(payload)) + payload
def _decode(blob:bytes, magic:bytes, version:int) -> bytes:
  if len(blob) < _RKPY_HEADER.size: raise ValueError(f"truncated {magic.decode()} header")
  actual_magic, actual_version, payload_size = _RKPY_HEADER.unpack_from(blob)
  if actual_magic != magic: raise ValueError(f"invalid {magic.decode()} magic")
  if actual_version != version: raise ValueError(f"unsupported {magic.decode()} version {actual_version}")
  if len(blob) != _RKPY_HEADER.size + payload_size: raise ValueError(f"invalid {magic.decode()} payload size")
  return blob[_RKPY_HEADER.size:]

def encode_rkpy(payload:bytes) -> bytes: return _encode(RKPY_MAGIC, RKPY_VERSION, payload)
def encode_rkhc(payload:bytes) -> bytes: return _encode(RKHC_MAGIC, RKHC_VERSION, payload)

def decode_rkpy(blob:bytes) -> bytes: return _decode(blob, RKPY_MAGIC, RKPY_VERSION)
def decode_rkhc(blob:bytes) -> bytes: return _decode(blob, RKHC_MAGIC, RKHC_VERSION)

def build_rkpy_program(ast:UOp, target:Target) -> UOp:
  """Compile one rejected early-simplified sink with tinygrad's generic Python UOps renderer."""
  program = to_program(ast, PythonRenderer(target))
  assert program.op is Ops.PROGRAM and len(program.src) == 4 and program.src[3].op is Ops.BINARY
  assert isinstance(program.arg, ProgramInfo)
  return program.replace(src=program.src[:3] + (program.src[3].replace(arg=encode_rkpy(program.src[3].arg)),),
                         arg=replace(program.arg, target=target))

def build_rkhc_program(ast:UOp, target:Target) -> UOp:
  """Compile one rejected sink as generic host UOps while retaining Rockchip program dispatch."""
  machine = platform.machine().lower()
  host_target = Target("CPU", "CLANG", {'amd64':'x86_64', 'aarch64':'arm64'}.get(machine, machine)+",native")
  program = to_program(ast, ClangRenderer(host_target))
  assert program.op is Ops.PROGRAM and len(program.src) == 4 and program.src[3].op is Ops.BINARY
  assert isinstance(program.arg, ProgramInfo)
  return program.replace(src=program.src[:3] + (program.src[3].replace(arg=encode_rkhc(program.src[3].arg)),),
                         arg=replace(program.arg, target=target))

class RKPythonProgram:
  """Run generic linear UOps directly over CPU-mapped Rockchip GEM buffers."""
  def __init__(self, dev:Any, obj:TinyELF, payload:bytes):
    self.dev = dev
    self.program = PythonProgram(cast(Any, dev), replace(obj, lib=payload))
    self.uop_count = len(self.program.uops)

  def __call__(self, *bufs:Any, **kwargs):
    mapped = tuple(self.dev.allocator._as_buffer(buf) for buf in bufs)
    return self.program(*mapped, **kwargs)

class RKHostProgram:
  """Run one generic compiled UOps program over CPU-mapped Rockchip GEM buffers."""
  def __init__(self, obj:TinyELF, payload:bytes):
    cpu = Device["CPU"]
    self.program = cpu.runtime(replace(obj, lib=payload, target=cpu.renderer.target))

  def __call__(self, *bufs:Any, **kwargs): return self.program(*bufs, **kwargs)
