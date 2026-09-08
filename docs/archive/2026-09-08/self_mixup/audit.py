import sys, types, json, time, random, subprocess
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import yaml
from PIL import Image
import torch

ROOT = Path('/home/zhangtingyu/Project/Mono3D/MonoDGP')
OUT = Path('/tmp/monodgp-self-mixup-6tqGHJ')
sys.path.insert(0, str(ROOT))
# Dataset-only diagnostic: do not initialize the CUDA AP evaluator.
stub = types.ModuleType('lib.datasets.kitti.kitti_eval_python.eval')
stub.get_official_eval_result = stub.get_distance_eval_result = lambda *a, **k: None
sys.modules[stub.__name__] = stub
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

torch.set_num_threads(2)
cfg = yaml.safe_load((ROOT / 'outputs/V2-0047_实验47_去除三维IoU质量头/resolved_config.yaml').read_text())
ds = KITTI_Dataset('train', cfg['dataset'])
assert ds.full_p2_projection and not ds.cross_focal_mixup
assert ds.cross_focal_mixup_policy == 'legacy' and not ds.mixup_virtual_focal
manifest = dict(executable=sys.executable, torch=torch.__version__,
                git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                command=' '.join(sys.argv), dataset=cfg['dataset'], uses_cuda=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
t0=time.time()
groups=defaultdict(list)
rows=[]
for pos,s in enumerate(ds.idx_list):
    i=int(s)
    cal=ds.get_calib(i)
    with Image.open(Path(ds.image_dir)/f'{i:06d}.png') as img:
        size=img.size
    objs=ds.get_label(i)
    row=dict(pos=pos,id=i,n=len(objs),
             eligible_cars=int(sum(o.cls_type=='Car' and o.level_str!='UnKnown'
                 and 2<=o.pos[-1]<=65 and o.trucation<=0.5 and o.occlusion<=2 for o in objs)))
    rows.append(row)
    groups[(tuple(cal.P2.flatten()),size)].append(row)
N=len(rows)
for group in groups.values():
    counts=np.array([r['n'] for r in group])
    for r in group:
        m=int(np.count_nonzero(counts+r['n'] < ds.max_objs))
        success=1-(1-m/N)**ds.mixup_max_attempts
        r.update(accepted_donors=m,success_given_trigger=success,
                 self_given_trigger=success/m if m and 2*r['n']<ds.max_objs else 0.)
print('METADATA',N,'groups',len(groups),flush=True)
summary={}
for p in [0.5,0.3,0.2,0.1,0.0]:
    total=p*sum(r['self_given_trigger'] for r in rows)
    successful=p*sum(r['success_given_trigger'] for r in rows)
    summary[str(p)]=dict(expected_self_images_per_epoch=total,
         expected_self_images_250_epochs=250*total,
         percent_all_images=100*total/N,
         percent_successful_mixups=100*total/successful if successful else 0.,
         expected_self_with_eligible_source_car=p*sum(r['self_given_trigger'] for r in rows if r['eligible_cars']))
# Force ONLY donor identity and MixUp trigger. Keep actual photometric,
# flip/crop and all target construction/filtering from the recorded config.
ds.random_mixup3d=1.0
candidates=[r for r in rows if r['self_given_trigger'] and r['eligible_cars']]
rng=np.random.default_rng(444)
chosen=[candidates[int(i)] for i in rng.choice(len(candidates),size=min(16,len(candidates)),replace=False)]
tests=[]
keys=['labels','boxes','boxes_3d','depth','size_3d','src_size_3d',
      'heading_bin','heading_res','calibs','depth_unit_scale','projective_rotation_y']
for r in chosen:
    for seed in [444,445]:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        with patch('numpy.random.choice', return_value=str(r['id'])) as mocked:
            _,_,targets,_=ds[r['pos']]
        valid=targets['mask_2d'].astype(bool)
        donor=targets['mixup_is_donor'].astype(bool)
        primary_idx=np.flatnonzero(valid & ~donor)
        donor_idx=np.flatnonzero(valid & donor)
        pairs=[(int(a),int(b)) for a in primary_idx for b in donor_idx
               if all(np.array_equal(targets[k][a],targets[k][b]) for k in keys)]
        result=dict(id=r['id'],seed=seed,donor_draws=mocked.call_count,
                    valid_primary=len(primary_idx),valid_donor=len(donor_idx),
                    exact_duplicate_pairs=len(pairs),pairs=pairs)
        tests.append(result)
        print('FORCED',json.dumps(result),flush=True)
report=dict(manifest=manifest,images=N,calibration_size_groups=len(groups),
            group_sizes=sorted(len(g) for g in groups.values()),
            probability_method='Exact independent uniform donor draws, first accepted among at most 50; actual P2/size/capacity constraints. Expected counts, not historical observations.',
            frequency=summary,forced_tests=tests,
            forced_tests_with_duplicates=sum(bool(x['exact_duplicate_pairs']) for x in tests),
            total_duplicate_pairs=sum(x['exact_duplicate_pairs'] for x in tests),
            seconds=time.time()-t0)
(OUT/'result.json').write_text(json.dumps(report,indent=2))
(OUT/'metadata.json').write_text(json.dumps(rows))
print('COMPLETE',json.dumps(summary),flush=True)
