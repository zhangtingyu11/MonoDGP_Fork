import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile, ImageEnhance
import random
from skimage import io
import skimage.transform
import torch.nn.functional as F
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tqdm
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.pd import PhotometricDistort

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import paint_clipped_box
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.mixup_geometry import (
    camera_normalized_mixup_geometry,
    classify_box_canvas_visibility,
    classify_mixup_object_visibility,
    merge_mixup_object_regions,
    mixup_box_crosses_valid_boundary,
    mixup_box_valid_ratio,
    mixup_object_requires_boundary_protection,
    projective_point,
    transform_projective_box,
    warp_and_blend_mixup,
    warp_mixup_support,
)
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy
#from .pd import PhotometricDistort


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('root_dir')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        #self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.istrain = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)
        # Opt-in so all historical experiments retain their exact projection
        # and virtual-flip behavior.  The corrected path couples full-P2
        # projection with a physical camera-coordinate horizontal reflection.
        self.full_p2_projection = bool(
            cfg.get('full_p2_projection', False))
        
        self.random_mixup3d = cfg.get('random_mixup3d', 0.5)
        self.cross_focal_mixup = bool(cfg.get('cross_focal_mixup', False))
        self.cross_focal_mixup_policy = cfg.get(
            'cross_focal_mixup_policy', 'legacy')
        self.mixup_geometry_monitoring = bool(
            cfg.get('mixup_geometry_monitoring', False))
        self.mixup_valid_mask_threshold = float(
            cfg.get('mixup_valid_mask_threshold', 0.999))
        self.mixup_min_object_valid_ratio = float(
            cfg.get('mixup_min_object_valid_ratio', 0.999))
        self.mixup_max_attempts = int(cfg.get('mixup_max_attempts', 50))
        self.mixup_virtual_focal = bool(
            cfg.get('mixup_virtual_focal', False))
        self.mixup_virtual_focal_multipliers = tuple(float(value) for value in
            cfg.get('mixup_virtual_focal_multipliers', (0.9, 1.0, 1.1)))
        self.augmentation_epoch = 0
        if self.cross_focal_mixup and not self.full_p2_projection:
            raise ValueError(
                'cross_focal_mixup requires full_p2_projection=true')
        if self.cross_focal_mixup_policy not in (
                'legacy', 'unified_v2', 'protected_legacy_v3'):
            raise ValueError(
                'cross_focal_mixup_policy must be legacy, unified_v2, '
                'or protected_legacy_v3')
        if (self.cross_focal_mixup_policy == 'unified_v2'
                and not self.cross_focal_mixup):
            raise ValueError(
                'unified_v2 MixUp policy requires cross_focal_mixup=true')
        if (self.cross_focal_mixup_policy == 'protected_legacy_v3'
                and not self.cross_focal_mixup):
            raise ValueError(
                'protected_legacy_v3 MixUp policy requires '
                'cross_focal_mixup=true')
        if not 0.0 <= self.mixup_valid_mask_threshold <= 1.0:
            raise ValueError('mixup_valid_mask_threshold must be in [0, 1]')
        if not 0.0 <= self.mixup_min_object_valid_ratio <= 1.0:
            raise ValueError(
                'mixup_min_object_valid_ratio must be in [0, 1]')
        if self.mixup_max_attempts <= 0:
            raise ValueError('mixup_max_attempts must be positive')
        if self.mixup_virtual_focal:
            if not self.cross_focal_mixup:
                raise ValueError(
                    'mixup_virtual_focal requires cross_focal_mixup=true')
            if not self.mixup_virtual_focal_multipliers:
                raise ValueError(
                    'mixup_virtual_focal_multipliers must not be empty')
            if any(value <= 0.0 for value in
                   self.mixup_virtual_focal_multipliers):
                raise ValueError(
                    'mixup_virtual_focal_multipliers must be positive')
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.depth_downsample_factor = 16
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx [str]: Index of the sample
        Returns:
            depth [np.ndarray(H, W)]: Depth map
        """
        depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
        assert os.path.exists(depth_file)
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        #depth = Image.open(depth_file)
        return depth
    
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(
            calib_file, use_full_p2=self.full_p2_projection)

    def _mixup_calibrations_match(self, first, second):
        if self.full_p2_projection:
            return np.array_equal(first.P2, second.P2)
        return (second.cu == first.cu and second.cv == first.cv
                and second.fu == first.fu and second.fv == first.fv)

    def _mixup_object_is_trainable(self, object_3d):
        """Use one object-level eligibility rule for Experiment 34 donors."""
        return (
            object_3d.cls_type in self.writelist
            and object_3d.level_str != 'UnKnown'
            and 2.0 <= float(object_3d.pos[-1]) <= 65.0
            and object_3d.trucation <= 0.5
            and object_3d.occlusion <= 2)

    def _mixup_object_requires_boundary_protection(self, object_3d):
        """Protect every labelled task-class object from an opacity seam.

        Detection/Region eligibility is intentionally irrelevant here.  A Car
        remains visible in RGB even when its difficulty or depth excludes it
        from a particular loss, so the MixUp support edge must not cut through
        its annotated 2-D box.
        """
        return mixup_object_requires_boundary_protection(
            object_3d.cls_type, self.writelist)

    def _virtual_focal_introduces_canvas_cut(
            self, primary_objects, donor_objects, donor_homography,
            image_flip_h, baseline_affine_h, virtual_affine_h):
        """Whether virtual focal newly changes a labelled Car to partial."""
        for objects, source_h in (
                (primary_objects, np.eye(3, dtype=np.float64)),
                (donor_objects, donor_homography)):
            baseline_to_input = (
                baseline_affine_h.astype(np.float64)
                @ image_flip_h @ source_h)
            virtual_to_input = (
                virtual_affine_h.astype(np.float64)
                @ image_flip_h @ source_h)
            for object_3d in objects:
                if not self._mixup_object_requires_boundary_protection(
                        object_3d):
                    continue
                baseline_box = transform_projective_box(
                    object_3d.box2d, baseline_to_input)
                virtual_box = transform_projective_box(
                    object_3d.box2d, virtual_to_input)
                baseline_state = classify_box_canvas_visibility(
                    baseline_box, self.resolution)
                virtual_state = classify_box_canvas_visibility(
                    virtual_box, self.resolution)
                if baseline_state != 'partial' and virtual_state == 'partial':
                    return True
        return False

    

    def _decoded_predictions_to_annos(self, results):
        """Convert decoded predictions to KITTI annotations without disk I/O.

        Values are rounded to the same two decimals used by the historical
        text export, so switching to in-memory evaluation does not silently
        change AP merely by retaining extra floating-point precision.
        """
        annos = []
        for image_id in self.idx_list:
            predictions = results.get(int(image_id), [])
            count = len(predictions)
            names = []
            alpha = np.empty(count, dtype=np.float64)
            bbox = np.empty((count, 4), dtype=np.float64)
            dimensions_hwl = np.empty((count, 3), dtype=np.float64)
            location = np.empty((count, 3), dtype=np.float64)
            rotation_y = np.empty(count, dtype=np.float64)
            score = np.empty(count, dtype=np.float64)

            for index, prediction in enumerate(predictions):
                # This exactly mirrors Tester.save_results(..., "{:.2f}")
                # followed by kitti_common.get_label_anno(...).
                quantized = np.array(
                    [float(f"{value:.2f}") for value in prediction[1:]],
                    dtype=np.float64,
                )
                names.append(self.class_name[int(prediction[0])])
                alpha[index] = quantized[0]
                bbox[index] = quantized[1:5]
                dimensions_hwl[index] = quantized[5:8]
                location[index] = quantized[8:11]
                rotation_y[index] = quantized[11]
                score[index] = quantized[12]

            annos.append({
                'name': np.asarray(names),
                'truncated': np.zeros(count, dtype=np.float64),
                'occluded': np.zeros(count, dtype=np.int64),
                'alpha': alpha,
                'bbox': bbox,
                # The evaluator stores dimensions as length, height, width.
                'dimensions': dimensions_hwl[:, [2, 0, 1]],
                'location': location,
                'rotation_y': rotation_y,
                'score': score,
            })
        return annos

    def eval(self, results, logger, return_metrics=False):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = self._decoded_predictions_to_annos(results)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        metrics = {}
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            metrics.update({
                key: float(value) for key, value in results_dict.items()
            })
            logger.info(results_str)
        if return_metrics:
            return {
                'selection_score': float(car_moderate),
                'metrics': metrics,
            }
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def set_epoch(self, epoch):
        """Set the deterministic augmentation epoch before worker creation."""
        self.augmentation_epoch = int(epoch)

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H
        
        
        if self.split!='test':
            dst_W, dst_H = img_size
            
        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        random_mix_flag = False
        mixup_requested = 0.0
        mixup_applied = 0.0
        mixup_cross_focal = 0.0
        mixup_valid_ratio = 0.0
        mixup_attempts = 0.0
        mixup_reject_capacity = 0.0
        mixup_reject_geometry = 0.0
        mixup_reject_no_overlap = 0.0
        mixup_reject_partial_object = 0.0
        mixup_reject_primary_mask_boundary = 0.0
        mixup_reject_donor_mask_boundary = 0.0
        mixup_reject_center_outside = 0.0
        mixup_reject_no_valid_target = 0.0
        mixup_focal_scale_x = 0.0
        mixup_focal_scale_y = 0.0
        mixup_virtual_focal_multiplier = 1.0
        mixup_virtual_focal_requested_multiplier = 1.0
        mixup_virtual_focal_cancelled = 0.0
        mixup_donor_index = -1
        mixup_donor_target_count = 0
        mixup_donor_source_indices = np.full(
            self.max_objs, -1, dtype=np.int64)
        mixup_retained_support_min = 0.0
        mixup_retained_support_observed = 0.0
        mixup_projection_residual_sum = 0.0
        mixup_projection_residual_max = 0.0
        mixup_depth_shift_abs_sum = 0.0
        mixup_depth_shift_abs_max = 0.0
        mixup_primary_donor_overlap_ratio = 0.0
        mixup_donor_image = None
        mixup_homography = None
        mixup_objects = None
        mixup_valid_mask = None
        calib = self.get_calib(index)

        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True
                mixup_requested = 1.0
                if self.mixup_virtual_focal:
                    multiplier_index = (
                        index + self.augmentation_epoch
                    ) % len(self.mixup_virtual_focal_multipliers)
                    mixup_virtual_focal_multiplier = (
                        self.mixup_virtual_focal_multipliers[
                            multiplier_index])
                    mixup_virtual_focal_requested_multiplier = (
                        mixup_virtual_focal_multiplier)
                      
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # A focal multiplier s is exactly a common crop/resize scale when the
        # full transformed P2 is used: crop_size / s multiplies both focal
        # lengths by s.  It is composed into the existing affine so each RGB
        # source is resampled only once.  If donor selection later fails, the
        # baseline affine is restored below.
        baseline_crop_size = np.asarray(crop_size, dtype=np.float64).copy()
        baseline_trans, baseline_trans_inv = get_affine_transform(
            center, baseline_crop_size, 0, self.resolution, inv=1)
        baseline_affine_h = np.eye(3, dtype=np.float32)
        baseline_affine_h[:2] = baseline_trans.astype(
            np.float32, copy=False)
        baseline_affine_inv_h = np.eye(3, dtype=np.float32)
        baseline_affine_inv_h[:2] = baseline_trans_inv.astype(
            np.float32, copy=False)
        if random_mix_flag and self.mixup_virtual_focal:
            crop_size = baseline_crop_size / mixup_virtual_focal_multiplier

        # The affine is sampled once for both RGB and labels.  Computing its
        # matrix here lets the protected cross-P2 policy validate the exact
        # final-input support without changing the historical same-P2 path.
        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1)
        affine_h = np.eye(3, dtype=np.float32)
        affine_h[:2] = trans.astype(np.float32, copy=False)
        affine_inv_h = np.eye(3, dtype=np.float32)
        affine_inv_h[:2] = trans_inv.astype(np.float32, copy=False)

        if (random_mix_flag == True
                and self.cross_focal_mixup_policy != 'unified_v2'):
            count_num = 0
            random_mix_flag = False
            primary_objects = self.get_label(index)
            protected_policy = (
                self.cross_focal_mixup_policy == 'protected_legacy_v3')
            while count_num < self.mixup_max_attempts:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(random_index)
                if (not self.cross_focal_mixup
                        and not self._mixup_calibrations_match(
                            calib, calib_temp)):
                    continue
                is_cross_focal = (
                    self.cross_focal_mixup
                    and not np.array_equal(calib.P2, calib_temp.P2))
                img_temp = self.get_image(random_index)
                img_size_temp = np.array(img_temp.size)
                if (not is_cross_focal
                        and not np.array_equal(img_size_temp, img_size)):
                    continue
                objects_2 = self.get_label(random_index)
                if len(primary_objects) + len(objects_2) >= self.max_objs:
                    mixup_reject_capacity += 1.0
                    continue

                if is_cross_focal:
                    try:
                        homography, translation = (
                            camera_normalized_mixup_geometry(
                                calib.P2, calib_temp.P2))
                        transformed_objects = copy.deepcopy(objects_2)
                        for object_2 in transformed_objects:
                            object_2.box2d = transform_projective_box(
                                object_2.box2d, homography)
                            object_2.pos = (
                                object_2.pos.astype(np.float64)
                                + translation).astype(np.float32)
                            object_2.dis_to_cam = np.linalg.norm(object_2.pos)
                    except ValueError:
                        mixup_reject_geometry += 1.0
                        continue

                    image_flip_h = np.eye(3, dtype=np.float64)
                    if random_flip_flag:
                        image_flip_h[0, 0] = -1.0
                        image_flip_h[0, 2] = float(img_size[0])
                    if (mixup_virtual_focal_multiplier != 1.0
                            and self._virtual_focal_introduces_canvas_cut(
                                primary_objects, objects_2, homography,
                                image_flip_h, baseline_affine_h, affine_h)):
                        mixup_virtual_focal_multiplier = 1.0
                        mixup_virtual_focal_cancelled = 1.0
                        crop_size = baseline_crop_size
                        trans, trans_inv = baseline_trans, baseline_trans_inv
                        affine_h = baseline_affine_h.copy()
                        affine_inv_h = baseline_affine_inv_h.copy()

                    if protected_policy:
                        recipient_to_input = (
                            affine_h.astype(np.float64) @ image_flip_h)
                        donor_to_input = (
                            recipient_to_input @ homography)
                        candidate_valid_mask, _ = warp_mixup_support(
                            np.asarray(img_temp).shape, donor_to_input,
                            self.resolution,
                            self.mixup_valid_mask_threshold)
                        if not np.any(candidate_valid_mask):
                            mixup_reject_no_overlap += 1.0
                            continue

                        primary_boundary_crossed = False
                        for primary_object in primary_objects:
                            if not self._mixup_object_requires_boundary_protection(
                                    primary_object):
                                continue
                            try:
                                primary_final_box = transform_projective_box(
                                    primary_object.box2d,
                                    recipient_to_input)
                            except ValueError:
                                mixup_reject_geometry += 1.0
                                primary_boundary_crossed = True
                                break
                            crosses_boundary, _ = (
                                mixup_box_crosses_valid_boundary(
                                    primary_final_box,
                                    candidate_valid_mask,
                                    self.mixup_min_object_valid_ratio))
                            if crosses_boundary:
                                mixup_reject_primary_mask_boundary += 1.0
                                primary_boundary_crossed = True
                                break
                        if primary_boundary_crossed:
                            continue

                        donor_boundary_crossed = False
                        donor_supported_ratios = []
                        for donor_object, transformed_object in zip(
                                objects_2, transformed_objects):
                            if not self._mixup_object_requires_boundary_protection(
                                    transformed_object):
                                continue
                            try:
                                donor_final_box = transform_projective_box(
                                    donor_object.box2d, donor_to_input)
                            except ValueError:
                                mixup_reject_geometry += 1.0
                                donor_boundary_crossed = True
                                break
                            crosses_boundary, donor_ratio = (
                                mixup_box_crosses_valid_boundary(
                                    donor_final_box,
                                    candidate_valid_mask,
                                    self.mixup_min_object_valid_ratio))
                            if donor_ratio > 0.0:
                                donor_supported_ratios.append(donor_ratio)
                            if crosses_boundary:
                                mixup_reject_donor_mask_boundary += 1.0
                                donor_boundary_crossed = True
                                break
                        if donor_boundary_crossed:
                            continue
                        mixup_valid_mask = candidate_valid_mask
                        mixup_retained_support_min = float(
                            min(donor_supported_ratios)
                            if donor_supported_ratios else 0.0)
                        mixup_retained_support_observed = float(
                            bool(donor_supported_ratios))

                    mixup_donor_image = img_temp
                    mixup_homography = homography
                    mixup_objects = transformed_objects
                    mixup_cross_focal = 1.0
                    mixup_focal_scale_x = float(
                        abs(calib.fu / calib_temp.fu))
                    mixup_focal_scale_y = float(
                        abs(calib.fv / calib_temp.fv))
                else:
                    image_flip_h = np.eye(3, dtype=np.float64)
                    if random_flip_flag:
                        image_flip_h[0, 0] = -1.0
                        image_flip_h[0, 2] = float(img_size[0])
                    if (mixup_virtual_focal_multiplier != 1.0
                            and self._virtual_focal_introduces_canvas_cut(
                                primary_objects, objects_2,
                                np.eye(3, dtype=np.float64), image_flip_h,
                                baseline_affine_h, affine_h)):
                        mixup_virtual_focal_multiplier = 1.0
                        mixup_virtual_focal_cancelled = 1.0
                        crop_size = baseline_crop_size
                        trans, trans_inv = baseline_trans, baseline_trans_inv
                        affine_h = baseline_affine_h.copy()
                        affine_inv_h = baseline_affine_inv_h.copy()
                    mixup_objects = objects_2
                    if random_flip_flag == True:
                        img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                    img = Image.blend(img, img_temp, alpha=0.5)
                    mixup_applied = 1.0
                    mixup_valid_ratio = 1.0
                    mixup_retained_support_min = 1.0
                    mixup_retained_support_observed = float(any(
                        object_2.cls_type in self.writelist
                        for object_2 in objects_2))
                    mixup_focal_scale_x = 1.0
                    mixup_focal_scale_y = 1.0
                random_mix_flag = True
                mixup_donor_index = random_index
                break
            mixup_attempts = float(count_num)

        if (mixup_requested > 0.0 and not random_mix_flag
                and self.mixup_virtual_focal):
            mixup_virtual_focal_multiplier = 1.0
            crop_size = baseline_crop_size
            trans, trans_inv = baseline_trans, baseline_trans_inv
            affine_h = baseline_affine_h.copy()
            affine_inv_h = baseline_affine_inv_h.copy()

        # add affine transformation for 2d images.
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        if (random_mix_flag
                and self.cross_focal_mixup_policy == 'unified_v2'):
            # Experiment 34 uses a fresh, object-consistent donor policy.  A
            # candidate is accepted only when every trainable donor target is
            # either fully outside the actual RGB warp or fully supported with
            # an encodable projected 3-D center.  Rejected candidates consume
            # an attempt and sampling continues.
            random_mix_flag = False
            primary_input = np.asarray(img)
            primary_objects = self.get_label(index)
            for count_num in range(1, self.mixup_max_attempts + 1):
                mixup_attempts = float(count_num)
                random_index = int(np.random.choice(self.idx_list))
                if random_index == index:
                    continue
                donor_calib = self.get_calib(random_index)
                donor_image = self.get_image(random_index)
                donor_objects = self.get_label(random_index)
                try:
                    homography, translation = (
                        camera_normalized_mixup_geometry(
                            calib.P2, donor_calib.P2))
                    image_flip_h = np.eye(3, dtype=np.float64)
                    if random_flip_flag:
                        image_flip_h[0, 0] = -1.0
                        image_flip_h[0, 2] = float(img_size[0])
                    donor_to_input = (
                        affine_h.astype(np.float64)
                        @ image_flip_h @ homography)
                    blended, donor_valid_mask, _ = warp_and_blend_mixup(
                        primary_input, np.asarray(donor_image),
                        donor_to_input, self.resolution,
                        valid_threshold=self.mixup_valid_mask_threshold)
                except ValueError:
                    mixup_reject_geometry += 1.0
                    continue
                if not np.any(donor_valid_mask):
                    mixup_reject_no_overlap += 1.0
                    continue

                transformed_objects = copy.deepcopy(donor_objects)
                kept_objects = []
                kept_source_indices = []
                kept_support_ratios = []
                kept_projection_residuals = []
                kept_depth_shifts = []
                rejected_state = None
                for source_index, (original_object, transformed_object) in enumerate(
                        zip(donor_objects, transformed_objects)):
                    try:
                        transformed_object.box2d = transform_projective_box(
                            original_object.box2d, homography)
                    except ValueError:
                        rejected_state = 'geometry'
                        break
                    transformed_object.pos = (
                        transformed_object.pos.astype(np.float64)
                        + translation).astype(np.float32)
                    transformed_object.dis_to_cam = np.linalg.norm(
                        transformed_object.pos)
                    if not self._mixup_object_is_trainable(
                            transformed_object):
                        continue

                    try:
                        final_box = transform_projective_box(
                            original_object.box2d, donor_to_input)
                    except ValueError:
                        rejected_state = 'geometry'
                        break
                    donor_center_3d = (
                        original_object.pos
                        + np.array(
                            [0.0, -original_object.h / 2.0, 0.0],
                            dtype=np.float32))
                    donor_center_2d, _ = donor_calib.rect_to_img(
                        donor_center_3d.reshape(1, 3))
                    try:
                        final_center = projective_point(
                            donor_center_2d[0], donor_to_input)
                    except ValueError:
                        rejected_state = 'geometry'
                        break
                    state, support_ratio = classify_mixup_object_visibility(
                        final_box, final_center, donor_valid_mask,
                        self.mixup_min_object_valid_ratio)
                    if state == 'outside':
                        continue
                    if state != 'complete':
                        rejected_state = state
                        break
                    kept_objects.append(transformed_object)
                    kept_source_indices.append(source_index)
                    kept_support_ratios.append(support_ratio)
                    if self.mixup_geometry_monitoring:
                        recipient_center_3d = (
                            transformed_object.pos
                            + np.array(
                                [0.0, -transformed_object.h / 2.0, 0.0],
                                dtype=np.float32))
                        recipient_center_2d, _ = calib.rect_to_img(
                            recipient_center_3d.reshape(1, 3))
                        recipient_to_input = (
                            affine_h.astype(np.float64) @ image_flip_h)
                        encoded_center = projective_point(
                            recipient_center_2d[0], recipient_to_input)
                        kept_projection_residuals.append(float(
                            np.linalg.norm(final_center - encoded_center)))
                        kept_depth_shifts.append(abs(float(
                            transformed_object.pos[2]
                            - original_object.pos[2])))
                    else:
                        kept_projection_residuals.append(0.0)
                        kept_depth_shifts.append(0.0)

                if rejected_state == 'geometry':
                    mixup_reject_geometry += 1.0
                    continue
                if rejected_state == 'partial':
                    mixup_reject_partial_object += 1.0
                    continue
                if rejected_state == 'center_outside':
                    mixup_reject_center_outside += 1.0
                    continue
                if not kept_objects:
                    mixup_reject_no_valid_target += 1.0
                    continue
                if len(primary_objects) + len(kept_objects) > self.max_objs:
                    mixup_reject_capacity += 1.0
                    continue

                mixup_objects = kept_objects
                mixup_donor_index = random_index
                mixup_donor_target_count = len(kept_objects)
                mixup_donor_source_indices[:len(kept_source_indices)] = (
                    kept_source_indices)
                mixup_retained_support_min = float(
                    min(kept_support_ratios))
                mixup_retained_support_observed = 1.0
                mixup_projection_residual_sum = float(
                    sum(kept_projection_residuals))
                mixup_projection_residual_max = float(
                    max(kept_projection_residuals))
                mixup_depth_shift_abs_sum = float(sum(kept_depth_shifts))
                mixup_depth_shift_abs_max = float(max(kept_depth_shifts))
                mixup_donor_image = donor_image
                mixup_homography = homography
                mixup_valid_mask = donor_valid_mask
                mixup_cross_focal = float(
                    not np.array_equal(calib.P2, donor_calib.P2))
                mixup_focal_scale_x = float(
                    abs(calib.fu / donor_calib.fu))
                mixup_focal_scale_y = float(
                    abs(calib.fv / donor_calib.fv))
                mixup_valid_ratio = float(donor_valid_mask.mean())
                mixup_applied = 1.0
                random_mix_flag = True
                img = Image.fromarray(blended)
                break

        if (random_mix_flag and mixup_homography is not None
                and self.cross_focal_mixup_policy != 'unified_v2'):
            image_flip_h = np.eye(3, dtype=np.float64)
            if random_flip_flag:
                image_flip_h[0, 0] = -1.0
                image_flip_h[0, 2] = float(img_size[0])
            donor_to_input = (
                affine_h.astype(np.float64)
                @ image_flip_h @ mixup_homography)
            blended, mixup_valid_mask, _ = warp_and_blend_mixup(
                np.asarray(img), np.asarray(mixup_donor_image),
                donor_to_input, self.resolution,
                valid_threshold=self.mixup_valid_mask_threshold)
            if np.any(mixup_valid_mask):
                has_partial_object = False
                if self.cross_focal_mixup_policy == 'legacy':
                    # Preserve the historical one-shot cross-P2 behavior for
                    # old configs.  The protected policy already checked both
                    # primary and donor boxes inside its retry loop.
                    for original_object, transformed_object in zip(
                            objects_2, mixup_objects):
                        if transformed_object.cls_type not in self.writelist:
                            continue
                        if (transformed_object.level_str == 'UnKnown'
                                or transformed_object.pos[-1] < 2):
                            continue
                        final_box = transform_projective_box(
                            original_object.box2d, donor_to_input)
                        valid_ratio = mixup_box_valid_ratio(
                            final_box, mixup_valid_mask)
                        if (0.0 < valid_ratio
                                < self.mixup_min_object_valid_ratio):
                            has_partial_object = True
                            break
                if has_partial_object:
                    random_mix_flag = False
                    mixup_objects = None
                    mixup_reject_partial_object += 1.0
                    mixup_cross_focal = 0.0
                    mixup_focal_scale_x = 0.0
                    mixup_focal_scale_y = 0.0
                else:
                    img = Image.fromarray(blended)
                    mixup_applied = 1.0
                    mixup_valid_ratio = float(mixup_valid_mask.mean())
            else:
                random_mix_flag = False
                mixup_objects = None
                mixup_reject_no_overlap += 1.0
                mixup_cross_focal = 0.0
                mixup_focal_scale_x = 0.0
                mixup_focal_scale_y = 0.0

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        if self.full_p2_projection:
            info.update({
                'model_image_size': self.resolution.astype(
                    np.float32, copy=True),
                'projective_input_size': self.resolution.astype(
                    np.float32, copy=True),
                'image_affine_inverse': affine_inv_h,
            })

        if self.split == 'test':
            calib = self.get_calib(index)
            model_calib = (
                affine_h @ calib.P2
                if self.full_p2_projection else calib.P2)
            return img, model_calib.astype(np.float32), img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        # data augmentation for labels
        if random_flip_flag:
            physical_flip = self.aug_calib or self.full_p2_projection
            if physical_flip:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if physical_flip:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # In the corrected path the model sees pixels after the image affine,
        # so its camera matrix must describe that same coordinate system.
        # Horizontal reflection is already baked into calib.P2 above.
        model_calib = (
            affine_h @ calib.P2
            if self.full_p2_projection else calib.P2)
        model_calib = model_calib.astype(np.float32)

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        depth_unit_scale = np.ones((self.max_objs, 1), dtype=np.float32)
        projective_rotation_y = np.zeros(
            (self.max_objs, 1), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        mixup_is_donor = np.zeros((self.max_objs), dtype=bool)

        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool) # (H, W)
        mixup_obj_region = (
            np.zeros_like(obj_region)
            if (random_mix_flag
                and (mixup_cross_focal
                     or self.cross_focal_mixup_policy == 'unified_v2'))
            else None)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()
            
            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            
            # create object region
            paint_clipped_box(obj_region, bbox_2d)

            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)

            center_3d, rect_depth = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment

            if (random_flip_flag
                    and not (self.aug_calib or self.full_p2_projection)):
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue		

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.full_p2_projection:
                # Augmented P2 and input-pixel box height already account for
                # crop/resize.  Keep every downstream depth in physical Z.
                depth[i] = objects[i].pos[-1]
                depth_unit_scale[i] = 1.0
            elif self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale
                depth_unit_scale[i] = crop_scale
            
            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale
                depth_unit_scale[i] = 1.0 / crop_scale
            
            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]
                depth_unit_scale[i] = 1.0

            # encoding heading angle
            if self.full_p2_projection:
                heading_angle = (
                    objects[i].ry
                    - np.arctan2(objects[i].pos[0], objects[i].pos[2]))
            else:
                heading_angle = calib.ry2alpha(
                    objects[i].ry,
                    (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)
            projective_rotation_y[i] = objects[i].ry

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = model_calib
            
        if random_mix_flag == True:
            # if False:
                objects = mixup_objects
                # data augmentation for labels
                if random_flip_flag:
                    physical_flip = (
                        self.aug_calib or self.full_p2_projection)
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.ry = np.pi - object.ry
                        if physical_flip:
                            object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if (objects[i].level_str == 'UnKnown'
                            or objects[i].pos[-1] < 2):
                        continue
                    if (self.cross_focal_mixup_policy == 'unified_v2'
                            and (objects[i].pos[-1] > 65
                                 or objects[i].trucation > 0.5
                                 or objects[i].occlusion > 2)):
                        continue
                    if (self.cross_focal_mixup_policy
                            == 'protected_legacy_v3'
                            and objects[i].pos[-1] > 65):
                        continue
                    # process 2d bbox & get 2d center
                    bbox_2d = objects[i].box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                    
                    # process 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                    
                    # Keep cross-focal donor regions separate until they can
                    # be intersected with the exact RGB warp support.  The
                    # same-camera path retains the historical full-frame
                    # binary-union target.
                    paint_clipped_box(
                        mixup_obj_region
                        if mixup_obj_region is not None else obj_region,
                        bbox_2d)

                    corner_2d = bbox_2d.copy()

                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    if (random_flip_flag
                            and not (self.aug_calib
                                     or self.full_p2_projection)):
                        center_3d[0] = img_size[0] - center_3d[0]
                    center_3d = affine_transform(center_3d.reshape(-1), trans)

                    # filter 3d center out of img
                    proj_inside_img = True

                    if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                        proj_inside_img = False
                    if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                        proj_inside_img = False

                    if proj_inside_img == False:
                            continue
                    if (self.cross_focal_mixup
                            and mixup_valid_mask is not None):
                        center_x = int(np.floor(center_3d[0]))
                        center_y = int(np.floor(center_3d[1]))
                        if not mixup_valid_mask[center_y, center_x]:
                            continue

                    # class
                    cls_id = self.cls2id[objects[i].cls_type]
                    labels[i + object_num] = cls_id

        
                    # encoding 2d/3d boxes
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    size_2d[i + object_num] = 1. * w, 1. * h

                    center_2d_norm = center_2d / self.resolution
                    size_2d_norm = size_2d[i + object_num] / self.resolution

                    corner_2d_norm = corner_2d
                    corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
                    corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
                    center_3d_norm = center_3d / self.resolution

                    l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
                    t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

                    if l < 0 or r < 0 or t < 0 or b < 0:
                        if self.clip_2d:
                            l = np.clip(l, 0, 1)
                            r = np.clip(r, 0, 1)
                            t = np.clip(t, 0, 1)
                            b = np.clip(b, 0, 1)
                        else:
                            continue		

                    boxes[i + object_num] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
                    boxes_3d[i + object_num] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
        
                    # encoding depth
                    if self.full_p2_projection:
                        depth[i + object_num] = objects[i].pos[-1]
                        depth_unit_scale[i + object_num] = 1.0
                    elif self.depth_scale == 'normal':
                        depth[i + object_num] = objects[i].pos[-1] * crop_scale
                        depth_unit_scale[i + object_num] = crop_scale
                    
                    elif self.depth_scale == 'inverse':
                        depth[i + object_num] = objects[i].pos[-1] / crop_scale
                        depth_unit_scale[i + object_num] = 1.0 / crop_scale
                    
                    elif self.depth_scale == 'none':
                        depth[i + object_num] = objects[i].pos[-1]
                        depth_unit_scale[i + object_num] = 1.0
        
                    # encoding heading angle
                    #heading_angle = objects[i].alpha
                    if self.full_p2_projection:
                        heading_angle = (
                            objects[i].ry
                            - np.arctan2(
                                objects[i].pos[0], objects[i].pos[2]))
                    else:
                        heading_angle = calib.ry2alpha(
                            objects[i].ry,
                            (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)
                    projective_rotation_y[i + object_num] = objects[i].ry

                    #offset_3d[i + object_num] = center_3d - center_heatmap
                    src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                    mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                    size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                    if (self.cross_focal_mixup_policy == 'unified_v2'
                            or (objects[i].trucation <= 0.5
                                and objects[i].occlusion <= 2)):
                        mask_2d[i + object_num] = 1
                        mixup_is_donor[i + object_num] = True
                    
                    calibs[i + object_num] = model_calib

        if (self.cross_focal_mixup_policy == 'unified_v2'
                and random_mix_flag):
            donor_target_slice = mask_2d[
                object_num:object_num + len(mixup_objects)]
            if not donor_target_slice.all():
                raise RuntimeError(
                    'unified_v2 accepted a donor that did not produce one '
                    'complete target for every retained object')

        if (self.cross_focal_mixup_policy == 'protected_legacy_v3'
                and random_mix_flag):
            mixup_donor_target_count = int(mixup_is_donor.sum())

        if mixup_obj_region is not None:
            if self.mixup_geometry_monitoring:
                donor_region = mixup_obj_region & mixup_valid_mask
                region_union = obj_region | donor_region
                union_count = int(region_union.sum())
                mixup_primary_donor_overlap_ratio = (
                    float((obj_region & donor_region).sum()) / union_count
                    if union_count else 0.0)
            obj_region = merge_mixup_object_regions(
                obj_region, mixup_obj_region, mixup_valid_mask)

        # collect return data
        inputs = img
        
        projection_h = np.eye(3, dtype=np.float32)
        if (random_flip_flag
                and not (self.aug_calib or self.full_p2_projection)):
            projection_h[0, 0] = -1.0
            projection_h[0, 2] = float(img_size[0])
        projective_image_effective_calib = (
            affine_h @ projection_h @ calib.P2).astype(np.float32)

        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size': img_size,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'depth_unit_scale': depth_unit_scale,
                   'projective_rotation_y': projective_rotation_y,
                   'projective_image_effective_calib': (
                       projective_image_effective_calib),
                   'projective_input_size': self.resolution.astype(
                       np.float32, copy=True),
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'mixup_is_donor': mixup_is_donor,
                   'obj_region': obj_region}
        if self.full_p2_projection:
            targets['physical_ray_heading'] = np.bool_(True)
            targets['model_image_size'] = self.resolution.astype(
                np.float32, copy=True)
        if self.cross_focal_mixup:
            targets.update({
                'mixup_requested': np.float32(mixup_requested),
                'mixup_applied': np.float32(mixup_applied),
                'mixup_cross_focal': np.float32(mixup_cross_focal),
                'mixup_valid_ratio': np.float32(mixup_valid_ratio),
                'mixup_attempts': np.float32(mixup_attempts),
                'mixup_reject_capacity': np.float32(
                    mixup_reject_capacity),
                'mixup_reject_geometry': np.float32(
                    mixup_reject_geometry),
                'mixup_reject_no_overlap': np.float32(
                    mixup_reject_no_overlap),
                'mixup_reject_partial_object': np.float32(
                    mixup_reject_partial_object),
                'mixup_reject_primary_mask_boundary': np.float32(
                    mixup_reject_primary_mask_boundary),
                'mixup_reject_donor_mask_boundary': np.float32(
                    mixup_reject_donor_mask_boundary),
                'mixup_reject_center_outside': np.float32(
                    mixup_reject_center_outside),
                'mixup_reject_no_valid_target': np.float32(
                    mixup_reject_no_valid_target),
                'mixup_focal_scale_x': np.float32(mixup_focal_scale_x),
                'mixup_focal_scale_y': np.float32(mixup_focal_scale_y),
                'mixup_virtual_focal_multiplier': np.float32(
                    mixup_virtual_focal_multiplier),
                'mixup_virtual_focal_requested_multiplier': np.float32(
                    mixup_virtual_focal_requested_multiplier),
                'mixup_virtual_focal_cancelled': np.float32(
                    mixup_virtual_focal_cancelled),
                'mixup_donor_index': np.int64(mixup_donor_index),
                'mixup_donor_target_count': np.int64(
                    mixup_donor_target_count),
                'mixup_primary_slot_count': np.int64(object_num),
                'mixup_donor_source_indices': mixup_donor_source_indices,
                'mixup_retained_support_min': np.float32(
                    mixup_retained_support_min),
                'mixup_retained_support_observed': np.float32(
                    mixup_retained_support_observed),
                'mixup_projection_residual_sum': np.float32(
                    mixup_projection_residual_sum),
                'mixup_projection_residual_max': np.float32(
                    mixup_projection_residual_max),
                'mixup_depth_shift_abs_sum': np.float32(
                    mixup_depth_shift_abs_sum),
                'mixup_depth_shift_abs_max': np.float32(
                    mixup_depth_shift_abs_max),
                'mixup_primary_donor_overlap_ratio': np.float32(
                    mixup_primary_donor_overlap_ratio),
            })

        return inputs, model_calib, targets, info


if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    
    
    cfg = {'root_dir': '/hy-tmp/data/kitti',
           'random_mixup3d': 0.0, 'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 
           'use_dontcare': False, 'class_merging': False, 'writelist':['Car'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    #print(dataset.writelist)
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='load')
    for batch_idx, (inputs, calibs, targets, info) in enumerate(dataloader):
        boxes_3d = targets['boxes_3d'][0]
        img_size = targets['img_size'][0]
        size_3d = targets['size_3d'][0]
        calibs = targets['calibs'][0]
        depth = targets['depth'][0]
        
        for i in range(len(depth)):
            if depth[i] == 0:
                break
            height_norm = boxes_3d[i][4] + boxes_3d[i][5]
            box2d_height = height_norm * img_size[1: 2] #np.clip(height_norm * img_size[1: 2], a_min=1.0, a_max=None)
            depth_geo = size_3d[i][0] * calibs[i][1, 1] / box2d_height
            depth_err = depth[i] - depth_geo
            size_3d_geo = depth[i] * box2d_height / calibs[i][1, 1]
            height_err = size_3d_geo - size_3d[i][0]
            print(float(height_err))
            #size_3d_ = box2d_height * depth[0]  /  calib.P2[0, 0]
        progress_bar.update()
    progress_bar.close()
        # print(targets['size_3d'][0][0])
