"""Original nuScenes metadata, reusing Exp63 image augmentation/target encoding.

No KITTI text labels or fake difficulty labels are consumed. The model's camera
yaw-only and in-image-centre representation remains unchanged by user choice.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_utils import Calibration
from .front_geometry import NEAR_PLANE
from .front_eval import ground_truth, predictions, evaluate_boxes


class NuScenesFrontDataset(KITTI_Dataset):
    def __init__(self, split, cfg):
        root = Path(cfg['root_dir'])
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['protocol'] != 'nuscenes_front_car_v1':
            raise ValueError('Missing/completed native nuScenes index required')
        with (root/f'{split}.jsonl').open() as handle:
            self.records = [json.loads(line) for line in handle]
        self.by_id = {r['id']:r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError('Duplicate native sample IDs')
        if any(len(r['annotations'])>50 for r in self.records):
            raise ValueError('Native targets exceed Exp63 capacity; do not silently truncate')
        super().__init__(split, copy.deepcopy(cfg))
        if not self.full_p2_projection or self.class_merging or self.use_dontcare:
            raise ValueError('Native front protocol requires full projection and Car-only labels')
        if self.writelist != ['Car'] or self.cross_focal_mixup:
            raise ValueError('This native protocol supports Car and accepted same-calibration MixUp only')
        self._eval_gt = None

    def _load_split_ids(self):
        return [str(record['id']) for record in self.records]

    def get_image(self, idx):
        with Image.open(self.by_id[int(idx)]['image']) as image:
            return image.convert('RGB')

    def get_calib(self, idx):
        p = np.zeros((3, 4), dtype=np.float64)
        p[:, :3] = self.by_id[int(idx)]['intrinsic']
        return Calibration(dict(P2=p, R0=np.eye(3), Tr_velo2cam=np.eye(4)[:3]), use_full_p2=True)

    def get_label(self, idx):
        objects = []
        for ann in self.by_id[int(idx)]['annotations']:
            w, length, h = ann['size']
            centre = np.asarray(ann['camera_center'], dtype=np.float64)
            bottom = centre.copy()
            bottom[1] += h/2
            yaw = ann['camera_yaw']
            # Compatibility fields are not used as visibility/difficulty facts:
            # native eligibility overrides below, and evaluation uses raw GT.
            objects.append(SimpleNamespace(cls_type='Car', h=h, w=w, l=length,
                box2d=np.asarray(ann['bbox'], dtype=np.float64), pos=bottom,
                ry=yaw, alpha=float(yaw-np.arctan2(centre[0],centre[2])),
                trucation=0., occlusion=0., level_str='Native', score=-1.))
        return objects

    def _passes_primary_training_filter(self, obj):
        return obj.pos[2] > NEAR_PLANE

    def _passes_donor_training_filter(self, obj):
        return self._passes_primary_training_filter(obj)

    def eval(self, results, logger, return_metrics=False):
        if self._eval_gt is None:
            self._eval_gt = ground_truth(self.records)
        pred, counts = predictions(results, self.records)
        report = evaluate_boxes(self._eval_gt, pred)
        report['prediction_counts'] = counts
        logger.info('nuScenes CAM_FRONT Car subset (NOT official 10-class NDS): %s',
                    json.dumps(dict(metrics=report['metrics'], prediction_counts=counts)))
        return report if return_metrics else report['selection_score']
