"""Reduction movement and physical ownership contracts; hardware acceptance lives in the full backend census."""
import functools, math
import hashlib
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


def _owned_row_expression(a:Tensor,b:Tensor,kind:str) -> Tensor:
  if kind=='all': return (a<0).all(1)
  if kind=='count': return (a<0).cast(dtypes.int).sum(1)
  selected=(a<0).where(a,b)
  if kind=='sum': return selected.sum(1,dtype=dtypes.half)
  if kind=='prod': return selected.prod(1)
  if kind=='max': return selected.max(1)
  raise AssertionError(kind)


def _owned_row_values(rows:int,width:int,kind:str):
  left=np.full((rows,width),-1,dtype='<f2')
  left[1::2,-1]=1
  right=np.full((rows,width),-2 if kind=='max' else 1,dtype='<f2')
  selected=np.where(left<0,left,right)
  expected=np.all(left<0,axis=1) if kind=='all' else np.sum(left<0,axis=1,dtype='<i4') if kind=='count' else (
    np.sum(selected,axis=1,dtype='<f2') if kind=='sum' else np.prod(selected,axis=1,dtype='<f2') if kind=='prod' else
    np.max(selected,axis=1))
  return left,right,expected


@pytest.mark.parametrize('rows',(3,9))
@pytest.mark.parametrize('width',(33,65))
@pytest.mark.parametrize('kind',('sum','max','prod','all','count'))
def test_production_owned_row_tree_preserves_ragged_neutrals(rows:int,width:int,kind:str,monkeypatch,record_property):
  emitted=[]
  original=rk._reduce_mapped_rows
  def observe(*args,**kwargs):
    result=original(*args,**kwargs)
    emitted.append(result)
    return result
  monkeypatch.setattr(rk,'_reduce_mapped_rows',observe)
  with Context(DEV='ROCKCHIP',DEFAULT_FLOAT='HALF',NOOPT=0):
    a,b=(Tensor(UOp.new_buffer('ROCKCHIP',rows*width,dtypes.half,num=67500+slot)).reshape(rows,width) for slot in range(2))
    calls=_owned_row_expression(a,b,kind).schedule_linear().src
    assert len(calls)==1
    to_program_cache.clear()
    blob=next(node.arg for node in to_program(calls[0].src[0],rk.RockchipRenderer(Target(device='ROCKCHIP'))).src if node.op is Ops.BINARY)
    image=rk.decode_image(blob)
  assert emitted and not any(isinstance(op,rk.RKCMAC) for op in image.program)
  left,right,expected=_owned_row_values(rows,width,kind)
  bindings={a.uop.buf_uop:left.tobytes(),b.uop.buf_uop:right.tobytes()}
  actual=np.frombuffer(_execute_raw_dynamic_image(image,expected.nbytes,*(bindings[arg.buf_uop] for arg in calls[0].src[2:])),dtype=expected.dtype)
  np.testing.assert_array_equal(actual,expected)
  record_property('image_sha256',hashlib.sha256(blob).hexdigest())
  record_property('scratch_bytes',sum(image.scratch))
  record_property('physical_ops',len(image.program))


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


@pytest.mark.parametrize('special',(False,True))
@pytest.mark.parametrize('wide',(False,True))
@pytest.mark.parametrize('bits',(0x8000,0x7e31,0x7c00,0xfc00))
@pytest.mark.parametrize('swap',(False,True))
def test_composed_selected_reductions_preserve_raw_load_payloads(special:bool,wide:bool,bits:int,swap:bool):
  """This compiler-level fixture retains two REDUCE nodes; ordinary Tensor indexing can collapse them earlier."""
  count=6
  out,left,right,indices=(UOp.param(slot,dtype,(size,)) for slot,dtype,size in
                          ((0,dtypes.half,count),(1,dtypes.half,3),(2,dtypes.half,5),(3,dtypes.int,count*2)))
  lane=UOp.range(count,0,dtype=dtypes.int)
  reductions=[]
  for i,(source,size) in enumerate(((left,3),(right,5))):
    axis=UOp.special(size,f'selected{i}',dtypes.int) if special else UOp.range(size,i+1,dtype=dtypes.int)
    index=indices.index(lane+i*count).load()
    body=((axis!=index)!=UOp.const(True,dtypes.bool)).where(source.index(axis).load(),UOp.const(0.0,dtypes.half))
    reductions.append((body.cast(dtypes.float) if wide else body).reduce(axis,arg=Ops.ADD).cast(dtypes.half))
  value=(lane%2).eq(int(swap)).where(*reductions)
  image=_lower_uop_program(list(out.index(lane).store(value).sink().toposort()))
  assert image is not None and sum(isinstance(op,RKGather) and op.index is not None for op in image.program)==2
  payloads=(np.full(3,bits,dtype='<u2'),np.full(5,0xbc00,dtype='<u2'))
  choices=np.asarray(((0,2,-1,3,-(1<<31),(1<<31)-1),(4,0,5,-1,(1<<31)-1,-(1<<31))),dtype='<i4')
  selected=[np.asarray([payload[index] if 0<=index<len(payload) else 0 for index in choice],dtype='<u2')
            for payload,choice in zip(payloads,choices)]
  expected=np.where(np.arange(count)%2==int(swap),*selected).astype('<u2').tobytes()
  assert _execute_raw_dynamic_image(image,count*2,*(payload.tobytes() for payload in payloads),choices.tobytes())==expected


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
