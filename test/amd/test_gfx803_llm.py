import os
import pathlib
import subprocess
import sys
import unittest


class TestGFX803LLM(unittest.TestCase):
  def test_tiny_half_gpt2_compile(self):
    source = """
from tinygrad import Device, Tensor, Variable, dtypes
from tinygrad.renderer.isa.gfx803 import AMDASMRenderer

if AMDASMRenderer not in Device['NULL'].renderers: Device['NULL'].renderers.append(AMDASMRenderer)
from examples.gpt2 import Transformer
from tinygrad.nn.state import get_state_dict

model = Transformer(dim=16, n_heads=4, n_layers=1, norm_eps=1e-5, vocab_size=32, max_seq_len=16)
params = get_state_dict(model)
for parameter in params.values():
  parameter.replace(Tensor.ones(parameter.shape, dtype=dtypes.half, device=parameter.device).contiguous())
Tensor.realize(*params.values())
for pos, token_id in enumerate((1, 2)):
  token = Variable('tokens', 0, 31).bind(token_id)
  start_pos = Variable('start_pos', 0 if pos == 0 else 1, 15).bind(pos)
  output = model(token, start_pos, temperature=0.0)
  assert output.shape == (1,) and output.dtype is dtypes.int32
print('tiny half autoregressive GPT-2 compiled')
"""
    root = pathlib.Path(__file__).parents[2]
    env = {**os.environ, "DEV":"NULL:AMDASM:gfx803", "NULL_ALLOW_COPYOUT":"1", "FORWARD_ONLY":"1", "DEFAULT_FLOAT":"HALF",
           "HALF":"1", "MAX_CONTEXT":"16", "JIT":"0", "DEBUG":"0"}
    result = subprocess.run([sys.executable, "-c", source], cwd=root, env=env, capture_output=True, text=True, timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tiny half autoregressive GPT-2 compiled", result.stdout)

  def test_llama3_1b_compile(self):
    source = """
from tinygrad import Device, Tensor, Variable, dtypes
from tinygrad.renderer.isa.gfx803 import AMDASMRenderer

if AMDASMRenderer not in Device['NULL'].renderers: Device['NULL'].renderers.append(AMDASMRenderer)
from examples.llama3 import MODEL_PARAMS
from extra.models.llama import Transformer
from tinygrad.nn.state import get_state_dict

params = MODEL_PARAMS['1B']['args']
model = Transformer(**params, max_context=8, jit=False)
model.freqs_cis = Tensor.empty(1, 16, 1, params['dim']//params['n_heads']//2, 2, dtype=dtypes.half).contiguous().realize()
state = get_state_dict(model)
for parameter in state.values():
  parameter.replace(Tensor.empty(parameter.shape, dtype=dtypes.half, device=parameter.device).contiguous().realize())
for pos, token_id in enumerate((128000, 9906)):
  start_pos = Variable('start_pos', 0 if pos == 0 else 1, 7).bind(pos)
  output = model.forward(Tensor([[token_id]], dtype=dtypes.int32), start_pos, 0.0, 0, 0.0, 0.0, 0.0)
  assert output.shape == () and output.dtype is dtypes.int32
print('Llama 3.2 1B autoregressive graph compiled')
"""
    root = pathlib.Path(__file__).parents[2]
    env = {**os.environ, "DEV":"NULL:AMDASM:gfx803", "NULL_ALLOW_COPYOUT":"1", "FORWARD_ONLY":"1", "DEFAULT_FLOAT":"HALF",
           "JIT":"0", "DEBUG":"0"}
    result = subprocess.run([sys.executable, "-c", source], cwd=root, env=env, capture_output=True, text=True, timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("Llama 3.2 1B autoregressive graph compiled", result.stdout)


if __name__ == "__main__": unittest.main()
