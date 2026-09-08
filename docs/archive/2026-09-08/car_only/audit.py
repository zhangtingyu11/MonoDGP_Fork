import os, sys, json, time, logging, hashlib, subprocess, platform
from pathlib import Path
ROOT = Path('/home/zhangtingyu/Project/Mono3D/MonoDGP')
sys.path.insert(0, str(ROOT))
import torch, torchvision, numba, importlib.metadata
import numpy as np
from torch.utils.data import DataLoader
from lib.helpers.config_helper import load_config
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.helpers.bev_nms_helper import classwise_bev_nms_variants
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from tools.write_run_manifest import EXPECTED

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('car-only-audit')
checkpoint = ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/checkpoint_best.pth'
config = checkpoint.parent / 'resolved_config.yaml'
cfg = load_config(config)
runtime = dict(executable=sys.executable, python=platform.python_version(),
               torch=torch.__version__, torchvision=torchvision.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               numba=numba.__version__, numba_cuda=importlib.metadata.version('numba-cuda'))
assert runtime == EXPECTED, (runtime, EXPECTED)
assert torch.cuda.is_available(), 'Host CUDA required'
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = False
set_random_seed(cfg.get('random_seed', 444))
manifest = dict(runtime=runtime, command=' '.join([sys.executable, *sys.argv]),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                checkpoint=str(checkpoint), config=str(config), batch_size=cfg['dataset']['batch_size'],
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                protocol='one forward; historical class-flat top50 vs Car-channel top50; unchanged c*d and geometry; threshold0; original two-decimal evaluator; no NMS and BEV NMS0.8')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
logger.info('MANIFEST %s', json.dumps(manifest,ensure_ascii=False))
assert not cfg['model']['iou_quality_head']['enabled']
ds = KITTI_Dataset('val', cfg['dataset'])
loader = DataLoader(ds,batch_size=cfg['dataset']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
model, criterion = build_model(cfg['model'])
del criterion
epoch, recorded_ap, _ = load_checkpoint(model,None,str(checkpoint),'cpu',logger)
assert epoch == 247, epoch
model.cuda().eval()
results = {'historical':{},'car_only':{}}
class_counts = np.zeros(3,dtype=np.int64)
affected = []
extra_scores = []
common_identical = True
start = time.time()
with torch.no_grad():
    for step,(inputs,calibs,targets,info) in enumerate(loader):
        sizes=info.get('model_image_size',info['img_size']).cuda(non_blocking=True)
        outputs=model(inputs.cuda(non_blocking=True),calibs.cuda(non_blocking=True),targets,sizes,dn_args=0)
        assert outputs['pred_logits'].shape[1:] == (50,3)
        selected=torch.topk(outputs['pred_logits'].sigmoid().flatten(1),50,dim=1).indices.cpu().numpy()
        host_info={k:v.numpy() for k,v in info.items()}
        host_calibs=[ds.get_calib(int(i)) for i in host_info['img_id']]
        for variant in results:
            source=outputs
            if variant=='car_only':
                logits=outputs['pred_logits'].clone()
                logits[:,:,0]=-float('inf'); logits[:,:,2]=-float('inf')
                source={**outputs,'pred_logits':logits}
                car_selected=torch.topk(logits.sigmoid().flatten(1),50,dim=1).indices.cpu().numpy()//3
            dets=extract_dets_from_outputs(source,topk=50).cpu().numpy()
            decoded=decode_detections(dets,host_info,host_calibs,ds.cls_mean_size,0.0)
            results[variant].update({int(k):np.asarray(v,dtype=np.float32) for k,v in decoded.items()})
        for row,i in enumerate(host_info['img_id']):
            i=int(i)
            labels=selected[row]%3
            class_counts+=np.bincount(labels,minlength=3)
            old=results['historical'][i]
            new=results['car_only'][i]
            old_car=old[old[:,0]==1]
            old_query=selected[row][labels==1]//3
            new_order={int(q):j for j,q in enumerate(car_selected[row])}
            new_common=new[[new_order[int(q)] for q in old_query]]
            common_identical &= np.array_equal(old_car,new_common)
            if len(old_car)<50:
                added=new[[j for j,q in enumerate(car_selected[row]) if q not in set(old_query)]]
                extra_scores.extend(added[:,-1].tolist())
                affected.append(dict(image_id=i,missing_car=50-len(old_car),historical_car=len(old_car)))
        if step%30==0 or step+1==len(loader):
            logger.info('FORWARD %d/%d images=%d affected=%d',step+1,len(loader),len(results['historical']),len(affected))
forward_seconds=time.time()-start
assert len(results['historical'])==len(ds)==3769
assert common_identical, 'Common Car predictions changed: not a clean control'
report=dict(manifest=manifest,epoch=epoch,recorded_ap=recorded_ap,images=len(ds),
            historical_class_counts=class_counts.tolist(),affected_images=len(affected),
            added_car_count=len(extra_scores),affected_details=affected,
            common_car_predictions_bitwise_identical=common_identical,forward_seconds=forward_seconds,
            extra_score_quantiles=np.quantile(extra_scores,[0,.5,.9,1]).tolist() if extra_scores else [],evaluations={})
for variant,rows in results.items():
    report['evaluations'][variant]={}
    logger.info('EVALUATE %s no_nms',variant)
    report['evaluations'][variant]['no_nms']=ds.eval(rows,logger,return_metrics=True)
    logger.info('EVALUATE %s nms080',variant)
    filtered=classwise_bev_nms_variants(rows,[.8])[.8]
    report['evaluations'][variant]['nms080']=ds.eval(filtered,logger,return_metrics=True)
    report['evaluations'][variant]['nms_prediction_count']=sum(len(v) for v in filtered.values())
    (OUT/'partial.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
report['seconds']=time.time()-start
(OUT/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
logger.info('COMPLETE affected=%d added=%d seconds=%.1f',len(affected),len(extra_scores),report['seconds'])
