"""Divided-axis reset regressions, including the actual production compiler."""
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.codegen import to_program,to_program_cache
from tinygrad.helpers import Context,Target,ceildiv
from tinygrad.renderer import rockchip as rk
from tinygrad.uop.ops import Ops,UOp
from test.unit.test_rockchip_uops import _execute_raw_dynamic_image

@pytest.mark.parametrize("rows",(1,2,3))
@pytest.mark.parametrize("width",(7,8,9,31,32,33))
@pytest.mark.parametrize("divisor",(4,16))
def test_divided_axis_resets_at_every_outer_coordinate(rows,width,divisor):
  row,col=UOp.range(rows,0,dtype=rk.dtypes.int),UOp.range(width,1,dtype=rk.dtypes.int)
  gather=rk._gather_plan(1,0,row*width+col,row*ceildiv(width,divisor)+col.alu(Ops.CDIV,col.const_like(divisor)),None,rows*width)
  actual=gather.offsets or tuple(gather.base+sum(lane//step%limit*stride for step,limit,stride in gather.axes) for lane in range(gather.count))
  expected=tuple(r*ceildiv(width,divisor)+c//divisor for r in range(rows) for c in range(width))
  assert actual==expected

@pytest.mark.parametrize("rows",(1,2,3))
@pytest.mark.parametrize("width",(7,8,9,31,32,33))
@pytest.mark.parametrize("divisor",(4,16))
def test_production_repeated_view_resets_divided_axis(rows,width,divisor):
  groups=ceildiv(width,divisor)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",rows*groups,rk.dtypes.half,num=19700)).reshape(rows,groups,1)
    view=source.expand(rows,groups,divisor).reshape(rows,groups*divisor)[:,:width]
    calls=(view.bitcast(rk.dtypes.int16)^128).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    program=to_program(calls[0].src[0],rk.RockchipRenderer(Target(device="ROCKCHIP")))
    image=rk.decode_image(next(node.arg for node in program.src if node.op is Ops.BINARY))
  words=(np.arange(rows*groups,dtype=np.uint16)*313+41).reshape(rows,groups)
  expected=(np.repeat(words,divisor,axis=1)[:,:width]^128).astype("<u2").tobytes()
  assert _execute_raw_dynamic_image(image,rows*width*2,words.astype("<u2").tobytes())==expected

@pytest.mark.parametrize("rows",(1,2,3))
@pytest.mark.parametrize("width",(7,8,9,31,32,33))
@pytest.mark.parametrize("divisor",(4,16))
@pytest.mark.parametrize("operation",(Ops.CMOD,Ops.FLOORMOD))
def test_modulo_axis_resets_at_every_outer_coordinate(rows,width,divisor,operation):
  row,col=UOp.range(rows,0,dtype=rk.dtypes.int),UOp.range(width,1,dtype=rk.dtypes.int)
  gather=rk._gather_plan(1,0,row*width+col,col.alu(operation,col.const_like(divisor)),None,rows*width)
  actual=gather.offsets or tuple(gather.base+sum(lane//step%limit*stride for step,limit,stride in gather.axes) for lane in range(gather.count))
  assert actual==tuple(c%divisor for r in range(rows) for c in range(width))

@pytest.mark.parametrize("rows",(1,2,3))
@pytest.mark.parametrize("width",(7,8,9,31,32,33))
@pytest.mark.parametrize("divisor",(4,16))
def test_production_tiled_view_resets_modulo_axis(rows,width,divisor):
  groups=ceildiv(width,divisor)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",divisor,rk.dtypes.half,num=19800)).reshape(1,1,divisor)
    view=source.expand(rows,groups,divisor).reshape(rows,groups*divisor)[:,:width]
    calls=(view.bitcast(rk.dtypes.int16)^128).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    program=to_program(calls[0].src[0],rk.RockchipRenderer(Target(device="ROCKCHIP")))
    image=rk.decode_image(next(node.arg for node in program.src if node.op is Ops.BINARY))
  words=np.arange(divisor,dtype=np.uint16)*313+41
  expected=(np.tile(words,groups)[:width][None,:].repeat(rows,axis=0)^128).astype("<u2").tobytes()
  assert _execute_raw_dynamic_image(image,rows*width*2,words.astype("<u2").tobytes())==expected
