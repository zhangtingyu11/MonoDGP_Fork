"""Bounded same-state, three-arm optimizer probe; no checkpoint or AP outputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,hashlib,copy,argparse,subprocess,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.helpers.config_helper import load_config
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.trainer_helper import Trainer
from lib.helpers.utils_helper import set_random_seed
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.losses.geometry_depth_gate import matched_geometry_depth_weights


def digest_state(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_gpu(batch):
    images,calibs,raw,info=batch
    raw={k:v.cuda() for k,v in raw.items()}
    targets=Trainer.prepare_targets(None,raw,len(images))
    return images.cuda(),calibs.cuda(),targets,raw.get('model_image_size',raw['img_size']),info


def forward(model,batch):return model(*batch[:4],dn_args=None)


@torch.no_grad()
def evaluation(model,criterion,batch,config,fixed_indices=None):
    model.eval();criterion.eval()
    outputs=forward(model,batch)
    indices=criterion.matcher(outputs,batch[2],group_num=1) if fixed_indices is None else fixed_indices
    _,r=matched_geometry_depth_weights(outputs,batch[2],indices,config)
    b,q=r['source_batch'],r['source_query']
    labels=torch.cat([t['labels'][j.to(t['labels'].device)] for t,(_,j) in zip(batch[2],indices)])
    score=outputs['pred_logits'][b,q].sigmoid()[torch.arange(len(b),device=b.device),labels.long()]*torch.exp(-outputs['pred_depth'][b,q,1])
    pop=labels.eq(1)&(r['gt_physical_depth']>2)&(r['gt_physical_depth']<65)
    return {'iou':r['current_iou'][pop].cpu().tolist(),
            'absolute_error_m':(r['predicted_physical_depth']-r['gt_physical_depth']).abs()[pop].cpu().tolist(),
            'score':score[pop].cpu().tolist()},indices


def compare(before,after):
    old=np.array(before['iou']);new=np.array(after['iou'])
    assert len(old)==len(new)
    return {'count':len(old),'qualified_before':int((old>=.7).sum()),'qualified_after':int((new>=.7).sum()),
            'rescued':int(((old<.7)&(new>=.7)).sum()),'damaged':int(((old>=.7)&(new<.7)).sum()),
            'mean_iou_delta':float(np.mean(new-old)),
            'mae_delta_m':float(np.mean(after['absolute_error_m'])-np.mean(before['absolute_error_m'])),
            'mean_abs_score_change':float(np.mean(np.abs(np.array(after['score'])-np.array(before['score']))))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/monodgp_geometry_depth_gate_dev.yaml');gate=cfg['model']['geometry_depth_gate']
    import lib.models.monodgp.backbone as backbone
    from lib.models.monodgp.ops.functions.ms_deform_attn_func import set_force_deterministic_msda
    from lib.models.monodgp.ops.functions.deterministic_bilinear import set_force_deterministic_bilinear_backward
    set_random_seed(444);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    set_force_deterministic_msda(True);set_force_deterministic_bilinear_backward(True)
    ckpt=ROOT/'outputs/V2-0059_实验59_MixUp概率0.3排除自身/checkpoint_best.pth'
    ckpt_hash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert state['epoch']==178 and state['optimizer_state'] is not None
    manifest={'python':sys.executable,'python_version':sys.version.split()[0],
              'pytorch':torch.__version__,'torchvision':importlib.metadata.version('torchvision'),
              'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'numba':importlib.metadata.version('numba'),'numba_cuda':importlib.metadata.version('numba-cuda'),
              'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
              'strict_determinism':torch.are_deterministic_algorithms_enabled(),
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'command':[sys.executable,*sys.argv],'checkpoint':str(ckpt),'checkpoint_sha256':ckpt_hash,
              'learning_rates':[g['lr'] for g in state['optimizer_state']['param_groups']],
              'arms':['native','geometric_gate','uniform_mean_q'],
              'update_batches':2,'batch_size':16,'steps_per_arm_per_batch':1,'validation_batches':1,
              'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/probe_geometry_depth_gate_update.py','lib/losses/geometry_depth_gate.py','lib/models/monodgp/monodgp.py','configs/monodgp_geometry_depth_gate_dev.yaml']},
              'ap_evaluation':False,'checkpoint_writes':False,'normalizer':'existing full matched num_boxes',
              'uniform_control':'same arithmetic mean q, NOT same total gradient energy'}
    expected={'python_version':'3.10.20','pytorch':'2.8.0+cu129','torchvision':'0.23.0+cu129','cuda':'12.9','cudnn':91002,'numba':'0.66.0','numba_cuda':'0.30.4'}
    assert all(manifest[k]==v for k,v in expected.items()),manifest
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2));print('MANIFEST_OK',flush=True)
    backbone.is_main_process=lambda:False
    model,criterion=build_model(cfg['model']);model.cuda();criterion.cuda()
    train=iter(DataLoader(KITTI_Dataset('train',cfg['dataset']),batch_size=16,shuffle=False,num_workers=0))
    set_random_seed(444)
    training_batches=[batch_gpu(next(train)) for _ in range(2)]
    validation=batch_gpu(next(iter(DataLoader(KITTI_Dataset('val',cfg['dataset']),batch_size=16,shuffle=False,num_workers=0))))
    result={'manifest':manifest,'probes':[],'scope':'Local E178 optimizer response, two fixed train batches and first val batch; not AP or generalization proof.'}
    for batch_id,batch in enumerate(training_batches):
        model.load_state_dict(state['model_state'],strict=True)
        base_train,train_idx=evaluation(model,criterion,batch,gate)
        base_val,val_idx=evaluation(model,criterion,validation,gate)
        baseline_hash=digest_state(model)
        entry={'batch':batch_id,'image_ids':batch[4]['img_id'].tolist(),'validation_image_ids':validation[4]['img_id'].tolist(),'baseline_train':base_train,'baseline_val':base_val,'arms':{}}
        reference_prediction=reference_loss=reference_q=None
        for arm in manifest['arms']:
            model.load_state_dict(state['model_state'],strict=True)
            assert digest_state(model)==baseline_hash
            optimizer=build_optimizer(cfg['optimizer'],model)
            optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
            set_random_seed(1000+batch_id);model.train();criterion.train();optimizer.zero_grad(set_to_none=True)
            outputs=forward(model,batch)
            if reference_prediction is None:reference_prediction=outputs['pred_depth'].detach().clone()
            else:assert torch.equal(reference_prediction,outputs['pred_depth'])
            criterion.geometry_depth_gate_enabled=True
            losses=criterion(outputs,batch[2],None);r=criterion.geometry_depth_gate_receipt
            indices=criterion.matcher(outputs,batch[2],group_num=criterion.group_num)
            q=r['weights'];num_boxes=max(sum(len(t['labels']) for t in batch[2])*criterion.group_num,1)
            if reference_q is None:reference_q=q.clone()
            else:assert torch.equal(q,reference_q)
            used_q=q if arm=='geometric_gate' else (torch.ones_like(q) if arm=='native' else torch.full_like(q,q.mean()))
            cache=criterion._post_match_cache(outputs,batch[2],indices,['depths'])
            losses['loss_depth']=criterion.loss_depths(outputs,batch[2],indices,num_boxes,matched_cache=cache,depth_gradient_weights=used_q)['loss_depth']
            total=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
            if reference_loss is None:reference_loss=total.detach().clone()
            else:assert torch.equal(total,reference_loss)
            total.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            record={'q_mean':float(used_q.mean()),'loss_before':float(total.detach()),'optimizer_step':1,'finite_parameters':True,'starting_model_hash':baseline_hash}
            del outputs,losses,total,optimizer,cache
            model.zero_grad(set_to_none=True)
            post_train,_=evaluation(model,criterion,batch,gate,train_idx)
            post_val,_=evaluation(model,criterion,validation,gate,val_idx)
            rematched_val,_=evaluation(model,criterion,validation,gate)
            record.update({'train_fixed':compare(base_train,post_train),'val_fixed':compare(base_val,post_val),
                           'train_after':post_train,'val_after':post_val,
                           'val_rematched_qualified':sum(v>=.7 for v in rematched_val['iou']),
                           'val_rematched_count':len(rematched_val['iou'])})
            entry['arms'][arm]=record
            print('ARM',batch_id,arm,json.dumps({k:record[k] for k in ['q_mean','train_fixed','val_fixed','val_rematched_qualified']}),flush=True)
        result['probes'].append(entry)
        (outdir/'result.partial.json').write_text(json.dumps(result,indent=2))
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==ckpt_hash
    result['checkpoint_file_unchanged']=True;result['status']='COMPLETED_LOCAL_PROBE_NOT_AP_EVIDENCE'
    (outdir/'result.json').write_text(json.dumps(result,indent=2));print('DONE',flush=True)

if __name__=='__main__':main()
