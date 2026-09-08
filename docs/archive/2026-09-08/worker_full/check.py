import os, sys, json, hashlib, logging, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torchvision
import numba
import importlib.metadata

ROOT=Path('/home/zhangtingyu/Project/Mono3D/MonoDGP')
OUT=Path('/tmp/monodgp-worker-full-TVvCQh')
sys.path.insert(0,str(ROOT))
from tools.write_run_manifest import EXPECTED
from lib.helpers.config_helper import load_config
from lib.helpers.utils_helper import set_random_seed
import lib.helpers.dataloader_helper as dh
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.models.monodgp.ops.functions.ms_deform_attn_func import ensure_deterministic_msda_available, set_force_deterministic_msda
from lib.models.monodgp.ops.functions.deterministic_bilinear import set_force_deterministic_bilinear_backward

def digest(x):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype,a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.dtype,v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v: add(item)
        else: h.update(repr(v).encode())
    add(x)
    return h.hexdigest()

def inspect_worker(worker_id):
    # Read-only observer: unlike the old callback, never reseed or draw RNG.
    info=torch.utils.data.get_worker_info()
    state=dict(epoch=info.dataset.augmentation_epoch,worker=worker_id,
               torch_seed=info.seed,numpy_state=digest(np.random.get_state()))
    (OUT/f'{sys.argv[1]}-epoch{state["epoch"]}-worker{worker_id}.json').write_text(json.dumps(state))

class ShortLoader:
    def __init__(self,loader,rows):
        self.loader=loader; self.dataset=loader.dataset; self.rows=rows
    def __len__(self): return len(self.loader)
    def __iter__(self):
        iterator=iter(self.loader)
        try:
            for i in range(len(self.loader)):
                batch=next(iterator)
                self.rows.append(dict(epoch=self.dataset.augmentation_epoch,batch=i,
                                      ids=batch[3]['img_id'].tolist(),
                                      images=digest(batch[0]),raw_targets=digest(batch[2]),
                                      info=digest(batch[3])))
                yield batch
        finally:
            iterator._shutdown_workers()

def trial(name):
    started=time.time()
    actual=dict(executable=sys.executable,python='.'.join(map(str,sys.version_info[:3])),
        torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),numba=numba.__version__,
        numba_cuda=importlib.metadata.version('numba-cuda'))
    assert actual==EXPECTED,(actual,EXPECTED)
    assert torch.cuda.is_available()
    cfg=load_config(ROOT/'configs/monodgp_exp59.yaml')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    set_force_deterministic_msda(True); ensure_deterministic_msda_available()
    set_force_deterministic_bilinear_backward(True)
    set_random_seed(444)
    cfg['trainer']['swanlab']['enabled']=False
    cfg['trainer']['save_path']=str(OUT)
    cfg['trainer']['max_epoch']=1
    # Consume every batch of the original full training DataLoader.
    dh.my_worker_init_fn=inspect_worker
    data=[]; steps=[]
    train_loader,test_loader=dh.build_dataloader(cfg['dataset'])
    short=ShortLoader(train_loader,data)
    model,criterion=build_model(cfg['model']); model=model.cuda()
    optimizer=build_optimizer(cfg['optimizer'],model)
    scheduler,warmup=build_lr_scheduler(cfg['lr_scheduler'],optimizer,last_epoch=-1)
    logger=logging.getLogger(name); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.WARNING)
    trainer=Trainer(cfg=cfg['trainer'],model=model,optimizer=optimizer,
                    train_loader=short,test_loader=test_loader,lr_scheduler=scheduler,
                    warmup_lr_scheduler=warmup,logger=logger,loss=criterion,model_name=name)
    current={}
    def before_model(module,args):
        current['effective_inputs']=digest(args)
    def after_loss(module,args,losses):
        assert all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
        current['losses']=digest(losses)
        total=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
        current['total_loss']=float(total.detach())
    model.register_forward_pre_hook(before_model)
    criterion.register_forward_hook(after_loss)
    original_step=optimizer.step
    def step(*a,**kw):
        grads={n:p.grad for n,p in model.named_parameters()}
        assert all(torch.isfinite(v).all() for v in grads.values() if v is not None)
        current['gradients']=digest(grads)
        result=original_step(*a,**kw)
        current['model_state']=digest(model.state_dict())
        current['optimizer_state']=digest(optimizer.state_dict())
        steps.append(current.copy()); current.clear()
        print(name,'step',len(steps),'loss',steps[-1]['total_loss'],flush=True)
        return result
    optimizer.step=step
    manifest=dict(runtime=actual,git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  command=' '.join([sys.executable,*sys.argv]),config=cfg,initial_state=digest(model.state_dict()))
    (OUT/f'{name}-manifest.json').write_text(json.dumps(manifest,indent=2))
    for epoch in range(1):
        np.random.seed(np.random.get_state()[1][0]+epoch)
        short.dataset.set_epoch(epoch)
        if hasattr(criterion,'set_epoch'): criterion.set_epoch(epoch+1)
        trainer.train_one_epoch(epoch)
        trainer.epoch+=1
        scheduler.step()
    report=dict(initial_state=manifest['initial_state'],data=data,steps=steps,seconds=time.time()-started)
    assert len(data)==len(steps)==232
    ids=[i for row in data for i in row['ids']]
    assert len(ids)==len(set(ids))==3712
    assert set(ids)==set(map(int,short.dataset.idx_list))
    (OUT/f'{name}.json').write_text(json.dumps(report,indent=2))
    print(name,'COMPLETE',report['seconds'],flush=True)

def compare():
    a=json.loads((OUT/'A.json').read_text()); b=json.loads((OUT/'B.json').read_text())
    checks={k:a[k]==b[k] for k in ['initial_state','data','steps']}
    workers=[json.loads((OUT/f'A-epoch0-worker{w}.json').read_text())==
             json.loads((OUT/f'B-epoch0-worker{w}.json').read_text()) for w in range(4)]
    result=dict(checks=checks,worker_states_equal=workers,batches_per_run=232,
                samples_per_run=3712,unique_samples_per_run=3712,
                mean_loss=[sum(x['total_loss'] for x in run['steps'])/232 for run in [a,b]],
                seconds=[a['seconds'],b['seconds']])
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert all(checks.values()) and all(workers)

if __name__=='__main__':
    if sys.argv[1]=='compare': compare()
    else: trial(sys.argv[1])
