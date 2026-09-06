"""Reduction movement and physical ownership contracts; hardware acceptance lives in the full backend census."""
import functools, math
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer import rockchip as rk
from tinygrad.renderer.rockchip import RKGather, _lower_uop_program
from tinygrad.uop.ops import Ops, UOp
from test.unit.test_rockchip_uops import _execute_raw_dynamic_image


@pytest.mark.parametrize('special', (False, True))
@pytest.mark.parametrize('depth', (1, 2))
@pytest.mark.parametrize('wide', (False, True))
@pytest.mark.parametrize('bits', (0x8000, 0x7e01, 0x7c00, 0xfc00))
def test_range_selected_load_preserves_axis_cast_and_raw_payload(special:bool, depth:int, wide:bool, bits:int):
  count,extents=6,(3,5)[:depth]
  output,data,indices=(UOp.param(0,dtypes.half,(count,)),UOp.param(1,dtypes.half,(math.prod(extents),)),
                       UOp.param(2,dtypes.int,(count*depth,)))
  lane=UOp.range(count,0,dtype=dtypes.int)
  axes=tuple(UOp.special(size,f'reduce{i}',dtypes.int) if special else UOp.range(size,i+1,dtype=dtypes.int)
             for i,size in enumerate(extents))
  choices=tuple(indices.index(lane+i*count).load() for i in range(depth))
  equal=functools.reduce(lambda a,b:a&b,((axis!=choice)!=UOp.const(True,dtypes.bool) for axis,choice in zip(axes,choices)))
  address=functools.reduce(lambda left,pair:left*pair[1]+pair[0],zip(axes,extents),UOp.const(0,dtypes.int))
  body=equal.where(data.index(address).load(),UOp.const(0.0,dtypes.half))
  reduction=(body.cast(dtypes.float) if wide else body).reduce(*axes,arg=Ops.ADD)
  image=_lower_uop_program(list(output.index(lane).store(reduction.cast(dtypes.half)).sink().toposort()))
  assert image is not None and sum(isinstance(op,RKGather) and op.index is not None for op in image.program)==1
  payload=np.full(math.prod(extents),bits,dtype='<u2')
  coordinates=np.asarray([(0,size-1,-1,size,-(1<<31),(1<<31)-1) for size in extents],dtype='<i4')
  expected=np.asarray((bits,bits,0,0,0,0),dtype='<u2').tobytes()
  assert _execute_raw_dynamic_image(image,count*2,payload.tobytes(),coordinates.tobytes())==expected


@pytest.mark.parametrize('kind',('dense','padded','coefficient_table','raw_pair'))
@pytest.mark.parametrize('failure',('false','reject','bug'))
def test_production_owned_emission_rejection_preserves_the_following_program(kind:str, failure:str, monkeypatch):
  with Context(DEV='ROCKCHIP',DEFAULT_FLOAT='HALF',NOOPT=0):
    def source(shape, number):
      return Tensor(UOp.new_buffer('ROCKCHIP',math.prod(shape),dtypes.half,num=number)).reshape(shape)
    if kind=='coefficient_table': result=source((2,3,12,20),29000).interpolate(size=(9,31),mode='linear')
    elif kind=='raw_pair': result=source((2,3,4),29001).permute(1,0,2).bitcast(dtypes.int)
    else:
      m,k,n=(2,32,32) if kind=='dense' else (2,7,3)
      result=source((m,k),29002)@source((k,n),29003)
    calls=[node for node in result.schedule_linear().toposort() if node.op is Ops.CALL and node.src[0].op is Ops.SINK]
    renderer=rk.RockchipRenderer(Target(device='ROCKCHIP'))
    def compile_images():
      to_program_cache.clear()
      return tuple(rk.decode_image(next(node.arg for node in to_program(call.src[0],renderer).src if node.op is Ops.BINARY))
                   for call in calls)
    expected=compile_images()
    name='_lower_raw_fp16_bitcast' if kind=='raw_pair' else '_lower_cmac_reduce'
    emit=getattr(rk,name)
    accepted=[]
    def with_rejected_predecessor(*args):
      plan=args[-1]
      def state(): return tuple(plan.scratch),tuple(plan.program),dict(plan.bindings),plan.slot
      before=state()
      emitted=False
      def reject():
        nonlocal emitted
        emitted=emit(*args)
        plan.parameter(dtypes.half,3)
        if failure!='false': raise (ValueError if failure=='bug' else rk._RKGenericReject)('rejected owned emission')
        return False
      if failure=='bug':
        with pytest.raises(ValueError,match='rejected owned emission'): plan.lower(reject)
      else: assert not plan.lower(reject)
      assert state()==before
      result=emit(*args)
      assert result is emitted
      if result: accepted.append(len(before[0]))
      return result
    monkeypatch.setattr(rk,name,with_rejected_predecessor)
    assert compile_images()==expected
  assert accepted
  if kind=='coefficient_table': assert any(accepted), 'table-backed contraction must exercise a nonzero scratch base'


@pytest.mark.parametrize('rows,width',((9,4001),(16,4095),(16,4096)))
@pytest.mark.parametrize('suffix',('plain','bias','alias'))
def test_production_mapped_cacc_preserves_half_boundary_and_aliased_consumers(rows:int, width:int, suffix:str):
  # This idealized executor checks layout/rounding; real CACC NaN propagation is a separate, known hardware defect.
  with Context(DEV='ROCKCHIP',DEFAULT_FLOAT='HALF',NOOPT=0):
    tensors=tuple(Tensor(UOp.new_buffer('ROCKCHIP',rows*width,dtypes.half,num=31000+i)).reshape(rows,width) for i in range(3))
    selector,lhs,rhs=tensors
    target=Tensor(UOp.new_buffer('ROCKCHIP',rows,dtypes.half,num=31003))
    reduced=(selector<0).where(lhs,rhs).sum(1,dtype=dtypes.half)
    output=reduced if suffix=='plain' else reduced+2**-11 if suffix=='bias' else target.assign(reduced+target)
    calls=[node for node in output.schedule_linear().toposort() if node.op is Ops.CALL and node.src[0].op is Ops.SINK]
    # Assignment retains the scheduler's separate update kernel and must preserve its old output input.
    assert len(calls)==(2 if suffix=='alias' else 1)
    to_program_cache.clear()
    programs=[to_program(call.src[0],rk.RockchipRenderer(Target(device='ROCKCHIP'))) for call in calls]
    images=tuple(rk.decode_image(next(node.arg for node in program.src if node.op is Ops.BINARY)) for program in programs)
  assert tuple((op.m,op.n,op.k,op.out_fp16) for image in images for op in image.program if isinstance(op,rk.RKCMAC))==((1,rows,width,True),)
  values=np.zeros((rows,width),dtype='<f2')
  for row in range(rows):
    if row%7==0: values[row,:2]=(1,2**-11)
    elif row%7==1: values[row,0]=512
    elif row%7==2: values[row,:4]=(65504,65504,-65504,-65504)
    elif row%7==3: values[row,0]=np.inf
    elif row%7==4: values[row,0]=-np.inf
    elif row%7==5: values[row,0]=np.nan
    else: values[row,:]=-0.0
  selector_values=np.where(np.indices((rows,width))[1]%2,-1,1).astype('<f2')
  left=np.where(selector_values<0,values,np.nan).astype('<f2')
  right=np.where(selector_values<0,np.nan,values).astype('<f2')
  bias=np.full(rows,2**-11,dtype='<f2')
  memory={tensor.uop.buf_uop:array.tobytes() for tensor,array in zip((*tensors,target),(selector_values,left,right,bias))}
  with np.errstate(all='ignore'):
    for image,call in zip(images,calls):
      args=tuple(arg.buf_uop for arg in call.src[1:])
      # bytearray in the test executor accepts either a byte count or the existing aliased output bytes.
      seed=memory.get(args[0],math.prod(args[0].shape)*args[0].dtype.itemsize)
      memory[args[0]]=_execute_raw_dynamic_image(image,seed,*(memory[arg] for arg in args[1:]))
    actual=np.frombuffer(memory[calls[-1].src[1].buf_uop],dtype='<f2')
    expected=values.astype(np.float64).sum(1).astype('<f2')
    if suffix!='plain': expected=(expected+bias).astype('<f2')
  np.testing.assert_array_equal(actual,expected)
