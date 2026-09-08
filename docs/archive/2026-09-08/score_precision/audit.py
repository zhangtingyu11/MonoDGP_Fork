import sys, json, time, logging, hashlib, subprocess, platform, importlib.metadata
from pathlib import Path
ROOT=Path('/home/zhangtingyu/Project/Mono3D/MonoDGP')
sys.path.insert(0,str(ROOT))
import torch, torchvision, numba
import numpy as np
from torch.utils.data import DataLoader
from lib.helpers.config_helper import load_config
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.helpers.bev_nms_helper import classwise_bev_nms_variants
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_eval_python import kitti_common
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from tools.write_run_manifest import EXPECTED

OUT=Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
logger=logging.getLogger('precision-audit')
checkpoint=ROOT/'outputs/V2-0047_实验47_去除三维IoU质量头/checkpoint_best.pth'
config=checkpoint.parent/'resolved_config.yaml'
cfg=load_config(config)
runtime=dict(executable=sys.executable,python=platform.python_version(),torch=torch.__version__,
             torchvision=torchvision.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
             numba=numba.__version__,numba_cuda=importlib.metadata.version('numba-cuda'))
assert runtime==EXPECTED,(runtime,EXPECTED)
assert torch.cuda.is_available()
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory=False
set_random_seed(cfg.get('random_seed',444))
manifest=dict(runtime=runtime,command=' '.join([sys.executable,*sys.argv]),
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    config=str(config),batch_size=cfg['dataset']['batch_size'],
    protocol='one forward, historical top50 c*d threshold0; all geometry unchanged at two-decimal precision; only evaluator score changes; NMS0.8 survivors computed ONCE with original full scores')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
logger.info('MANIFEST %s',json.dumps(manifest,ensure_ascii=False))
ds=KITTI_Dataset('val',cfg['dataset'])
loader=DataLoader(ds,batch_size=cfg['dataset']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
model,criterion=build_model(cfg['model']); del criterion
epoch,recorded_ap,_=load_checkpoint(model,None,str(checkpoint),'cpu',logger)
assert epoch==247
model.cuda().eval()
rows={}
start=time.time()
with torch.no_grad():
    for step,(inputs,calibs,targets,info) in enumerate(loader):
        sizes=info.get('model_image_size',info['img_size']).cuda(non_blocking=True)
        outputs=model(inputs.cuda(non_blocking=True),calibs.cuda(non_blocking=True),targets,sizes,dn_args=0)
        dets=extract_dets_from_outputs(outputs,topk=50).cpu().numpy()
        host_info={k:v.numpy() for k,v in info.items()}
        host_calibs=[ds.get_calib(int(i)) for i in host_info['img_id']]
        decoded=decode_detections(dets,host_info,host_calibs,ds.cls_mean_size,0.0)
        rows.update({int(k):np.asarray(v,dtype=np.float32) for k,v in decoded.items()})
        if step%30==0 or step+1==len(loader):logger.info('FORWARD %d/%d images=%d',step+1,len(loader),len(rows))
assert len(rows)==len(ds)==3769
assert all(len(v)==50 and np.all(v[:,0]==1) for v in rows.values())
np.savez_compressed(OUT/'raw_predictions.npz',image_ids=np.asarray(list(rows)),predictions=np.stack(list(rows.values())))
report=dict(manifest=manifest,epoch=epoch,recorded_ap=recorded_ap,images=len(ds),forward_seconds=time.time()-start,evaluations={},stats={})
logger.info('NMS compute once')
nms=classwise_bev_nms_variants(rows,[.8])[.8]
gt=kitti_common.get_label_annos(ds.label_dir,[int(i) for i in ds.idx_list])
for mode,predictions in [('no_nms',rows),('nms080',nms)]:
    rounded=ds._decoded_predictions_to_annos(predictions)
    original=[]
    for image_id,anno in zip(ds.idx_list,rounded):
        original.append({**anno,'score':np.asarray([p[-1] for p in predictions[int(image_id)]],dtype=np.float64)})
    assert all(np.array_equal(a[k],b[k]) for a,b in zip(rounded,original) for k in a if k!='score')
    full_scores=np.concatenate([v['score'] for v in original])
    rounded_scores=np.concatenate([v['score'] for v in rounded])
    _,tie_counts=np.unique(rounded_scores,return_counts=True)
    report['stats'][mode]=dict(predictions=int(len(full_scores)),geometry_and_order_identical=True,
        raw_unique_scores=int(np.unique(full_scores).size),rounded_unique_scores=int(len(tie_counts)),
        rounded_zero_count=int(np.count_nonzero(rounded_scores==0)),raw_zero_count=int(np.count_nonzero(full_scores==0)),
        max_tie_group=int(tie_counts.max()),raw_score_quantiles=np.quantile(full_scores,[0,.1,.5,.9,.99,1]).tolist())
    report['evaluations'][mode]={}
    for precision,annos in [('two_decimals',rounded),('original',original)]:
        logger.info('EVALUATE %s %s',mode,precision)
        text,metrics,ap=get_official_eval_result(gt,annos,0)
        logger.info('%s',text)
        report['evaluations'][mode][precision]=dict(selection_score=float(ap),metrics={k:float(v) for k,v in metrics.items()})
        if mode=='no_nms' and precision=='two_decimals':
            assert abs(float(ap)-recorded_ap)<1e-10,('Baseline not reproduced',ap,recorded_ap)
        (OUT/'partial.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
report['seconds']=time.time()-start
(OUT/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
logger.info('COMPLETE seconds=%.1f',report['seconds'])
