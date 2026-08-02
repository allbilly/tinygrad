from __future__ import annotations
import struct
from dataclasses import replace
from typing import Any, cast
from tinygrad.codegen import to_program
from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

RKPY_MAGIC, RKPY_VERSION = b"RKPY", 1
_RKPY_HEADER = struct.Struct("<4sHI")

def encode_rkpy(payload:bytes) -> bytes: return _RKPY_HEADER.pack(RKPY_MAGIC, RKPY_VERSION, len(payload)) + payload

def decode_rkpy(blob:bytes) -> bytes:
  if len(blob) < _RKPY_HEADER.size: raise ValueError("truncated RKPY header")
  magic, version, payload_size = _RKPY_HEADER.unpack_from(blob)
  if magic != RKPY_MAGIC: raise ValueError("invalid RKPY magic")
  if version != RKPY_VERSION: raise ValueError(f"unsupported RKPY version {version}")
  if len(blob) != _RKPY_HEADER.size + payload_size: raise ValueError("invalid RKPY payload size")
  return blob[_RKPY_HEADER.size:]

def build_rkpy_program(ast:UOp, target:Target) -> UOp:
  """Compile one rejected early-simplified sink with tinygrad's generic Python UOps renderer."""
  program = to_program(ast, PythonRenderer(target))
  assert program.op is Ops.PROGRAM and len(program.src) == 4 and program.src[3].op is Ops.BINARY
  assert isinstance(program.arg, ProgramInfo)
  return program.replace(src=program.src[:3] + (program.src[3].replace(arg=encode_rkpy(program.src[3].arg)),),
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
