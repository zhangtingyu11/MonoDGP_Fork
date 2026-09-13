from copy import deepcopy
import pytest
import torch
from torch import nn
from lib.losses.geometry_depth_gate import (
    matched_geometry_depth_weights, depth_gradient_proxy, validate_gate_config)
from lib.models.monodgp.monodgp import SetCriterion


def fixture(device='cpu', scale=1.):
    target = {
        'labels': torch.tensor([1, 1, 1], device=device),
        'depth': torch.full((3, 1), 20.*scale, device=device),
        'depth_unit_scale': torch.full((3, 1), scale, device=device),
        'src_size_3d': torch.tensor([[1.5,1.6,4.]]*3, device=device),
        'projective_rotation_y': torch.zeros(3,1,device=device),
        'boxes_3d': torch.tensor([[.5,.5,.1,.1,.1,.1]]*3,device=device),
        'projective_input_size': torch.tensor([100.,100.],device=device),
        'projective_image_effective_calib': torch.tensor(
            [[100.,0.,50.,0.],[0.,100.,50.,0.],[0.,0.,1.,0.]],device=device),
        'physical_ray_heading': torch.tensor(True,device=device),
    }
    out = {'pred_depth': torch.tensor([[[20.1*scale,.3],[20.25*scale,.4],[22.*scale,.5]]],device=device,requires_grad=True),
           'pred_boxes': target['boxes_3d'][None].clone(),
           'pred_3d_dim': target['src_size_3d'][None].clone(),
           'pred_angle': torch.zeros(1,3,24,device=device),
           'pred_logits': torch.tensor([[[0.,5.,0.]]*3],device=device)}
    indices=[(torch.arange(3),torch.arange(3))]
    return out,[target],indices


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_geometry_and_native_gradient_contract(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('Host CUDA required')
    out,targets,idx=fixture(device)
    q,r=matched_geometry_depth_weights(out,targets,idx,{})
    # Independent closed-form IoU: identical axis-aligned boxes displaced in Z.
    delta=torch.tensor([.1,.25,2.],device=device)
    intersection=(1.6-delta).clamp_min(0)
    expected=intersection/(3.2-intersection)
    torch.testing.assert_close(r['current_iou'],expected,atol=3e-6,rtol=0)
    assert q[0]==.5 and .5<q[1]<1 and q[2]==1
    assert not q.requires_grad
    cache={'source_index':(torch.zeros(3,dtype=torch.long,device=device),torch.arange(3,device=device)),
           'matched_targets':{'depth':targets[0]['depth']}}
    old=SetCriterion.loss_depths(None,out,targets,idx,3,matched_cache=cache)
    new=SetCriterion.loss_depths(None,out,targets,idx,3,matched_cache=cache,depth_gradient_weights=q)
    assert old.keys()==new.keys()
    for key in old:assert torch.equal(old[key],new[key]),key
    g0=torch.autograd.grad(old['loss_depth'],out['pred_depth'],retain_graph=True)[0]
    g1=torch.autograd.grad(new['loss_depth'],out['pred_depth'])[0]
    assert torch.equal(g0[:,:,1],g1[:,:,1])
    torch.testing.assert_close(g1[0,:,0],g0[0,:,0]*q,rtol=0,atol=0)


def test_physical_units_and_invalid_geometry_fallback():
    o,t,i=fixture(); q,r=matched_geometry_depth_weights(o,t,i,{})
    oo,tt,ii=fixture(scale=1.7);qq,rr=matched_geometry_depth_weights(oo,tt,ii,{})
    torch.testing.assert_close(q,qq,atol=2e-5,rtol=0)
    o['pred_3d_dim'][0,0,0]=-1
    t[0]['depth'][1]=70
    o['pred_boxes'][0,2,0]=.8
    q,_=matched_geometry_depth_weights(o,t,i,{})
    assert torch.equal(q,torch.ones_like(q))


def test_missing_metadata_and_nonfinite_depth_fail_explicitly():
    o,t,i=fixture();del t[0]['physical_ray_heading']
    with pytest.raises(ValueError):matched_geometry_depth_weights(o,t,i,{})
    o,t,i=fixture()
    with torch.no_grad():o['pred_depth'][0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):matched_geometry_depth_weights(o,t,i,{})


@pytest.mark.parametrize('cfg',[{'minimum_gradient':0},{'iou_full':.7},{'iou_start':float('nan')}])
def test_invalid_configuration_rejected(cfg):
    with pytest.raises(ValueError):validate_gate_config(cfg)


def test_empty_matching_and_proxy_detach():
    o,t,_=fixture();i=[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]
    q,_=matched_geometry_depth_weights(o,t,i,{})
    assert q.numel()==0
    z=torch.tensor([2.,3.],requires_grad=True);q=torch.tensor([.5,1.],requires_grad=True)
    result=depth_gradient_proxy(z,q);assert torch.equal(result,z)
    result.sum().backward();assert torch.equal(z.grad,q.detach()) and q.grad is None


class FixedMatcher(nn.Module):
    def forward(self,outputs,targets,**kwargs):
        return [(torch.arange(3),torch.arange(3))]


def criterion(enabled):
    return SetCriterion(3,FixedMatcher(),{'loss_depth':1},.25,['depths'],[],group_num=1,
                        geometry_depth_gate={'enabled':enabled},query_monitoring={'enabled':False},
                        use_post_match_cache=True)


def test_real_criterion_gate_final_training_only(monkeypatch):
    import lib.models.monodgp.monodgp as module
    monkeypatch.setattr(module, 'DDNLoss', lambda **kwargs: nn.Identity())
    o,t,_=fixture();o['aux_outputs']=[{k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in o.items()}]
    o['inter_outputs']=[]
    base=criterion(False);new=criterion(True)
    base.train();new.train()
    l0=base(o,t,None);l1=new(o,t,None)
    for key in l0:assert torch.equal(l0[key],l1[key]),key
    final_q=new.geometry_depth_gate_receipt['weights']
    terms=[o['pred_depth'],o['aux_outputs'][0]['pred_depth']]
    g0=torch.autograd.grad(l0['loss_depth']+l0['loss_depth_0'],terms,retain_graph=True)
    g1=torch.autograd.grad(l1['loss_depth']+l1['loss_depth_0'],terms)
    torch.testing.assert_close(g1[0][0,:,0],g0[0][0,:,0]*final_q,rtol=0,atol=0)
    assert torch.equal(g0[0][:,:,1],g1[0][:,:,1]) and torch.equal(g0[1],g1[1])
    new.eval();base.eval();a=new(o,t,None);b=base(o,t,None)
    assert a.keys()==b.keys() and new.geometry_depth_gate_receipt is None
    for key in a:assert torch.equal(a[key],b[key]),key


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_uniform_control_matches_average_not_geometric_allocation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('Host CUDA required')
    out,targets,idx=fixture(device)
    q,_=matched_geometry_depth_weights(out,targets,idx,{})
    uniform,receipt=matched_geometry_depth_weights(out,targets,idx,{'mode':'uniform_mean'})
    assert torch.equal(uniform,q.mean().expand_as(q))
    assert torch.equal(receipt['geometric_weights'],q)
    assert q[-1]==1 and uniform[-1]<1  # The control deliberately also attenuates hard matches.
    with pytest.raises(ValueError):validate_gate_config({'mode':'unknown'})
