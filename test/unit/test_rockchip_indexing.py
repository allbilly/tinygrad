"""Divided-axis reset regressions, including the actual production compiler."""
import hashlib,math,struct
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

@pytest.mark.parametrize("rows",(1,3))
@pytest.mark.parametrize("width",(7,8,9,12,31,32,33))
@pytest.mark.parametrize("divisor",(2,4))
@pytest.mark.parametrize("operations",((Ops.CDIV,Ops.CMOD),(Ops.FLOORDIV,Ops.FLOORMOD)))
def test_periodic_divided_axes_compose_without_crossing_row_resets(rows,width,divisor,operations):
  row,col=UOp.range(rows,0,dtype=rk.dtypes.int),UOp.range(width,1,dtype=rk.dtypes.int)
  quotient=col.alu(operations[0],col.const_like(divisor))
  address=row*32+17-quotient.alu(operations[1],col.const_like(3))*7+col.alu(operations[1],col.const_like(3))
  gather=rk._gather_plan(1,0,row*width+col,address,None,rows*width)
  actual=gather.offsets or tuple(gather.base+sum(lane//step%limit*stride for step,limit,stride in gather.axes) for lane in range(gather.count))
  assert actual==tuple(r*32+17-(c//divisor%3)*7+c%3 for r in range(rows) for c in range(width))
  if rows==1 or width%(divisor*3)==0: assert gather.axes and not gather.offsets

@pytest.mark.parametrize("rows",(1,3))
@pytest.mark.parametrize("width",(7,8,9,12,31,32,33))
@pytest.mark.parametrize("divisor",(2,4))
def test_production_tiled_repeated_view_composes_periodic_division(rows,width,divisor):
  period=3
  groups=ceildiv(width,period*divisor)
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",period,rk.dtypes.half,num=19900)).reshape(1,1,period,1)
    view=source.expand(rows,groups,period,divisor).reshape(rows,groups*period*divisor)[:,:width]
    calls=(view.bitcast(rk.dtypes.int16)^128).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    program=to_program(calls[0].src[0],rk.RockchipRenderer(Target(device="ROCKCHIP")))
    image=rk.decode_image(next(node.arg for node in program.src if node.op is Ops.BINARY))
  words=np.asarray((0x8000,0x7c00,0x7e55),dtype="<u2")
  expected=(np.tile(np.repeat(words,divisor),groups)[:width][None,:].repeat(rows,axis=0)^128).astype("<u2").tobytes()
  assert _execute_raw_dynamic_image(image,rows*width*2,words.tobytes())==expected

@pytest.mark.parametrize("dtype",(rk.dtypes.int16,rk.dtypes.int32,rk.dtypes.uint32,rk.dtypes.int64))
def test_lossless_weak_integer_address_cast_keeps_affine_gather(dtype):
  lane=UOp.range(17,0,dtype=dtype)
  address=(lane*3+1).cast(rk.dtypes.weakint)
  gather=rk._gather_plan(1,0,lane,address,None,17)
  assert gather.axes and not gather.offsets
  assert tuple(gather.base+sum(i//step%limit*stride for step,limit,stride in gather.axes) for i in range(17))==tuple(i*3+1 for i in range(17))

def test_weak_integer_address_cast_preserves_float_rounding():
  lane=UOp.range(7,0,dtype=rk.dtypes.int32)
  address=(lane.cast(rk.dtypes.float)*0.75+0.5).cast(rk.dtypes.weakint)
  gather=rk._gather_plan(1,0,lane,address,None,7)
  assert gather.offsets==tuple(int(i*0.75+0.5) for i in range(7))


@pytest.mark.parametrize("count",(4095,4096,4097))
def test_masked_gather_cache_keeps_size_cutoff(count):
  lane=UOp.range(count,23000,dtype=rk.dtypes.int32)
  cache=rk._small_gather_offsets
  cache.cache_clear()
  try:
    first=rk._gather_plan(1,0,lane,lane-1,lane>0,count)
    second=rk._gather_plan(1,0,lane,lane-1,lane>0,count)
    assert first==second and first.offsets==(-1,*range(count-1))
    assert cache.cache_info()==((1,1,2048,1) if count<=4096 else (0,0,2048,0))
  finally: cache.cache_clear()


@pytest.mark.parametrize("count",(17,4095,4096,4097))
def test_production_masked_gather_at_cache_boundary(count,record_property):
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count-1,rk.dtypes.int16,num=88000))
    calls=(source.pad((1,0))^128).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    program=to_program(calls[0].src[0],rk.RockchipRenderer(Target(device="ROCKCHIP")))
    blob=next(node.arg for node in program.src if node.op is Ops.BINARY)
    image=rk.decode_image(blob)
  assert any(isinstance(op,rk.RKGather) and len(op.offsets)==count for op in image.program)
  assert rk._small_gather_offsets.cache_info().currsize==0
  words=(np.arange(count-1,dtype=np.uint16)*313+41).astype("<u2")
  expected=(np.pad(words,(1,0))^128).astype("<u2").tobytes()
  assert _execute_raw_dynamic_image(image,count*2,words.tobytes())==expected
  record_property("image_sha256",hashlib.sha256(blob).hexdigest())
  record_property("physical_ops",len(image.program))
  record_property("scratch_bytes",sum(image.scratch))


@pytest.mark.parametrize("value",(0.0,-0.0,2**-24,-2**-24,1.1,65504.0,65519.0,math.nextafter(65520.0,0.0),
                                65520.0,-65520.0,math.inf,-math.inf,math.nan))
def test_static_half_encoding_owns_rounding_and_overflow(value):
  lane=UOp.range(4,95100)
  expression=UOp.const(value,rk.dtypes.double)
  # Keep the wider semantic constant: pre-rounding to HALF would hide a packing overflow.
  try: expected=int.from_bytes(struct.pack("<e",value),"little")
  except OverflowError:
    with pytest.raises(OverflowError,match="float too large to pack with e format"):
      rk._static_values(lane,expression,4,rk._storage_bits)
  else: assert rk._static_values(lane,expression,4,rk._storage_bits)==(expected,)*4


@pytest.mark.parametrize("unique",(False,True))
@pytest.mark.parametrize("value",(65520.0,-65520.0))
def test_static_half_overwrite_cannot_hide_unencodable_candidate(unique,value):
  lane=UOp.range(2,95101)
  expression=(lane<1).where(UOp.const(value,rk.dtypes.double),UOp.const(1.0,rk.dtypes.double))
  with pytest.raises(OverflowError,match="float too large to pack with e format"):
    rk._static_values(lane//2,expression,1,rk._storage_bits,unique=unique)


def test_static_half_placement_validates_destination_before_encoding():
  lane=UOp.range(2,95102)
  with pytest.raises(rk._RKGenericReject,match="static_index"):
    rk._static_values(lane-1,UOp.const(65520.0,rk.dtypes.double),2,rk._storage_bits)


@pytest.mark.parametrize("unique",(False,True))
def test_static_half_duplicate_destination_distinguishes_signed_zero(unique):
  lane=UOp.range(2,95103)
  expression=(lane<1).where(UOp.const(-0.0,rk.dtypes.double),UOp.const(0.0,rk.dtypes.double))
  if unique:
    with pytest.raises(rk._RKGenericReject,match="static_index"):
      rk._static_values(lane//2,expression,1,rk._storage_bits)
  else: assert rk._static_values(lane//2,expression,1,rk._storage_bits,unique=False)==(0,)


@pytest.mark.parametrize("count",(7,8,257))
@pytest.mark.parametrize("pattern",("ramp","selection","large"))
def test_production_static_half_encoding_composes(count,pattern,record_property):
  with Context(DEV="ROCKCHIP",DEFAULT_FLOAT="HALF",NOOPT=0):
    source=Tensor(UOp.new_buffer("ROCKCHIP",count,rk.dtypes.half,num=95104))
    lane=Tensor.arange(count,dtype=rk.dtypes.int)
    if pattern=="ramp": mapped=(lane%7).cast(rk.dtypes.half)*0.25-1
    elif pattern=="selection": mapped=(lane%3==0).where(0.5,-0.25)
    else: mapped=(lane%3==0).where(65504.0,-65504.0)
    calls=(source+mapped).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    program=to_program(calls[0].src[0],rk.RockchipRenderer(Target(device="ROCKCHIP")))
    blob=next(node.arg for node in program.src if node.op is Ops.BINARY)
    image=rk.decode_image(blob)
  values=(np.arange(count)%7*0.25).astype("<f2")
  lanes=np.arange(count)
  mapped=(lanes%7*0.25-1) if pattern=="ramp" else np.where(lanes%3==0,0.5,-0.25) if pattern=="selection" else np.where(lanes%3==0,65504.0,-65504.0)
  expected=(values+mapped.astype("<f2")).astype("<f2").tobytes()
  assert any(isinstance(op,rk.RKEWOp) for op in image.program)
  assert _execute_raw_dynamic_image(image,count*2,values.tobytes())==expected
  record_property("image_sha256",hashlib.sha256(blob).hexdigest())
  record_property("physical_ops",len(image.program))
  record_property("scratch_bytes",sum(image.scratch))
