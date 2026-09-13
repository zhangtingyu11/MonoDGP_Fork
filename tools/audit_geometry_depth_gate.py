"""Bounded development contracts. No optimizer, checkpoint writes, or AP run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, time, argparse, subprocess, importlib.metadata, importlib.util, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.helpers.config_helper import load_config
from lib.helpers.trainer_helper import Trainer
from lib.helpers.utils_helper import set_random_seed
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.losses.geometry_depth_gate import matched_geometry_depth_weights
from lib.losses.asymmetric_interval_depth_loss import paired_iou3d


def area(p):
    if len(p)<3:return 0.
    p=np.asarray(p);p=p-p[0]
    return abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))*.5


def polygon(c,d,yaw):
    h,w,l=d
    a=np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]])
    rot=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    return a@rot.T+c[[0,2]]


def independent_iou(c,d,y,c2,d2,y2):
    p=list(polygon(c,d,y));q=polygon(c2,d2,y2)
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        new=[]
        for start,end in zip(p,p[1:]+p[:1]):
            cross=lambda v:(b[0]-a[0])*(v[1]-a[1])-(b[1]-a[1])*(v[0]-a[0])
            f0,f1=cross(start),cross(end)
            if (f0>=0)!=(f1>=0):new.append(start+(end-start)*f0/(f0-f1))
            if f1>=0:new.append(end)
        p=new
        if not p:break
    height=max(0.,min(c[1]+d[0]/2,c2[1]+d2[0]/2)-max(c[1]-d[0]/2,c2[1]-d2[0]/2))
    intersection=area(p)*height
    return intersection/(np.prod(d)+np.prod(d2)-intersection)


def independent_contract():
    rng=np.random.default_rng(444)
    c=rng.uniform([-20,-2,3],[20,2,65],(256,3));c2=c+rng.normal(0,.7,(256,3))
    d=rng.uniform([1.2,1.2,2.],[2.,2.2,5.],(256,3));d2=d*rng.uniform(.8,1.2,(256,3))
    y=rng.uniform(-np.pi,np.pi,256);y2=y+rng.normal(0,.3,256)
    expected=np.array([independent_iou(*r) for r in zip(c,d,y,c2,d2,y2)])
    # Match production's translated coordinate calculation.
    args=[torch.tensor(v,dtype=torch.float32) for v in [c-c2,d,y,np.zeros_like(c2),d2,y2]]
    actual=paired_iou3d(*args).numpy();err=float(np.max(abs(actual-expected)))
    assert err<2e-5,err
    return {'pairs':256,'max_abs_iou_error':err,'reference':'independent NumPy polygon clipping + height intersection'}


def perfect_prediction(t):
    n=len(t['labels']);angle=torch.zeros(1,n,24)
    bins=t['heading_bin'].long().flatten();angle[0,torch.arange(n),bins]=10
    angle[0,torch.arange(n),12+bins]=t['heading_res'].flatten()
    return {'pred_boxes':t['boxes_3d'][None], 'pred_logits':torch.nn.functional.one_hot(t['labels'].long(),3)[None].float()*10,
            'pred_angle':angle,'pred_depth':torch.cat([t['depth'],torch.zeros(n,1)],1)[None],
            'pred_3d_dim':t['size_3d'][None]}


def augmented_contract(cfg):
    rows=[]
    for flip,crop,mix in [(0,0,0),(1,0,0),(1,1,0),(0,1,1),(1,1,1)]:
        dc=dict(cfg['dataset']);dc.update(random_flip=flip,random_crop=crop,random_mixup3d=mix,aug_pd=False)
        ds=KITTI_Dataset('train',dc);count=donors=eligible=0;worst=1.
        for i in range(16):
            np.random.seed(444+i)
            _,_,raw,info=ds[i]
            batched={k:torch.as_tensor(v)[None] for k,v in raw.items()}
            t=Trainer.prepare_targets(None,batched,1)[0];n=len(t['labels'])
            if not n:continue
            o=perfect_prediction(t);idx=[(torch.arange(n),torch.arange(n))]
            q,r=matched_geometry_depth_weights(o,[t],idx,cfg['model']['geometry_depth_gate'])
            active=r['eligible'];count+=n;donors+=int(t['mixup_is_donor'].sum());eligible+=int(active.sum())
            if active.any():
                worst=min(worst,float(r['current_iou'][active].min()))
                assert torch.all(r['current_iou'][active]>.999), (flip,crop,mix,i,r['current_iou'])
                assert torch.all(q[active]==.5)
        rows.append(dict(flip=flip,crop=crop,mixup=mix,targets=count,donor_targets=donors,eligible=eligible,min_iou=worst))
    assert sum(x['donor_targets'] for x in rows)>0
    return rows


def norm_cos(a,b):
    aa=sum(float(x.double().square().sum()) for x in a if x is not None)
    bb=sum(float(x.double().square().sum()) for x in b if x is not None)
    dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return {'native_norm':aa**.5,'candidate_norm':bb**.5,'cosine':dot/(aa*bb)**.5 if aa*bb else None}


def live_contract(cfg):
    import lib.models.monodgp.backbone as backbone
    from lib.helpers.model_helper import build_model
    from lib.models.monodgp.ops.functions.ms_deform_attn_func import set_force_deterministic_msda
    from lib.models.monodgp.ops.functions.deterministic_bilinear import set_force_deterministic_bilinear_backward
    set_random_seed(444);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    set_force_deterministic_msda(True);set_force_deterministic_bilinear_backward(True)
    backbone.is_main_process=lambda:False
    model,criterion=build_model(cfg['model'])
    checkpoint=ROOT/'outputs/V2-0059_实验59_MixUp概率0.3排除自身/checkpoint_best.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state'],strict=True);del state
    model.cuda().train();criterion.cuda().train()
    print('LIVE_SETTINGS',json.dumps({'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'strict_determinism':torch.are_deterministic_algorithms_enabled(),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}),flush=True)
    ds=KITTI_Dataset('train',cfg['dataset']);loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    rows=[]
    for k,(images,calibs,raw,info) in enumerate(loader):
        if k==2:break
        raw={n:v.cuda() for n,v in raw.items()};t=Trainer.prepare_targets(None,raw,len(images))
        out=model(images.cuda(),calibs.cuda(),t,raw.get('model_image_size',raw['img_size']),dn_args=None)
        criterion.geometry_depth_gate_enabled=False;before=criterion(out,t,None)
        criterion.geometry_depth_gate_enabled=True;after=criterion(out,t,None);receipt=criterion.geometry_depth_gate_receipt
        for key in before:assert torch.equal(before[key],after[key]),key
        a=sum(v*criterion.weight_dict[key] for key,v in before.items() if key in criterion.weight_dict)
        b=sum(v*criterion.weight_dict[key] for key,v in after.items() if key in criterion.weight_dict)
        g0=torch.autograd.grad(a,out['pred_depth'],retain_graph=True)[0]
        g1=torch.autograd.grad(b,out['pred_depth'],retain_graph=True)[0]
        assert torch.equal(g0[:,:,1],g1[:,:,1])
        expected=g0.clone()
        expected[receipt['source_batch'],receipt['source_query'],0] *= receipt['weights']
        torch.testing.assert_close(g1,expected,atol=0,rtol=0)
        # Also cover actual shared-parameter gradients; no optimizer or .grad writes.
        native=torch.autograd.grad(a,params,retain_graph=True,allow_unused=True)
        candidate=torch.autograd.grad(b,params,allow_unused=True)
        assert all(torch.isfinite(g).all() for g in candidate if g is not None)
        module_stats={}
        for prefix in ['backbone','bbox_embed','dim_embed_3d','depth_embed','det3d_transformer']:
            module_ids={id(p) for p in getattr(model,prefix).parameters()}
            ids=[j for j,(_,p) in enumerate(named) if id(p) in module_ids]
            module_stats[prefix]=norm_cos([native[j] for j in ids],[candidate[j] for j in ids])
        q=receipt['weights'];j=receipt['current_iou'];sup=receipt['supported']
        benchmark={}
        reference_path=ROOT/'diagnostics/geometry_depth_gate_dev_20260911/gate_before_batching.py'
        if reference_path.exists():
            spec=importlib.util.spec_from_file_location('gate_reference',reference_path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            matches=criterion.matcher(out,t,group_num=criterion.group_num)
            calls={'before_batching':module.matched_geometry_depth_weights,'batched':matched_geometry_depth_weights}
            timings={key:[] for key in calls}
            reference_q,_=calls['before_batching'](out,t,matches,cfg['model']['geometry_depth_gate'])
            optimized_q,_=calls['batched'](out,t,matches,cfg['model']['geometry_depth_gate'])
            assert torch.equal(reference_q,optimized_q)
            for iteration in range(14):
                order=list(calls) if iteration%2==0 else list(reversed(calls))
                for name in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    calls[name](out,t,matches,cfg['model']['geometry_depth_gate'])
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    if iteration>=4:timings[name].append(elapsed)
            benchmark={'weights_bitwise_equal':True,'median_wall_ms':{key:statistics.median(v) for key,v in timings.items()},'measured_calls_each':10}
        rows.append({'batch':k,'isolated_gate_benchmark':benchmark,'image_ids':info['img_id'].tolist(),'matched':len(q),'supported':int(sup.sum()),
                     'active':int((q<1).sum()),'mean_weight':float(q.mean()),'loss':float(a.detach()),
                     'forward_losses_bitwise_equal':True,'uncertainty_output_grad_bitwise_equal':True,
                     'all_parameter_grads_finite':True,'mean_output_gradient_exact_gate':True,'parameter_gradient':norm_cos(native,candidate),'modules':module_stats})
        print('LIVE_BATCH',json.dumps(rows[-1]),flush=True)
        assert all(p.grad is None for p in params)
        del out,before,after,a,b,g0,g1,native,candidate
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/'configs/monodgp_geometry_depth_gate_dev.yaml');torch.set_num_threads(4)
    manifest={'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':sys.argv,
              'optimizer_steps':0,'checkpoint_writes':0,'ap_evaluation':False,
              'python_version':sys.version,'torchvision':importlib.metadata.version('torchvision'),
              'numba':importlib.metadata.version('numba'),'numba_cuda':importlib.metadata.version('numba-cuda'),
              'source_sha256':{str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [Path('lib/losses/geometry_depth_gate.py'),Path('lib/models/monodgp/monodgp.py'),Path('tools/audit_geometry_depth_gate.py'),Path('configs/monodgp_geometry_depth_gate_dev.yaml')]}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result={'independent_iou':independent_contract()};print('INDEPENDENT',result,flush=True)
    result['augmented_targets']=augmented_contract(cfg);print('AUGMENTED',json.dumps(result['augmented_targets']),flush=True)
    result['live_batches']=live_contract(cfg);result['status']='DEVELOPMENT_CONTRACT_PASS_NOT_AP_EVIDENCE'
    (output/'result.json').write_text(json.dumps(result,indent=2));print('DONE',flush=True)

if __name__=='__main__':main()
