import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile, ImageEnhance
import random
from skimage import io
import skimage.transform
import cv2
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
        calib = self.get_calib(index)

        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True
                      
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

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(random_index)
                
                if self._mixup_calibrations_match(calib, calib_temp):
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img_temp.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs: 
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break
                            
        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        affine_h = np.eye(3, dtype=np.float32)
        affine_h[:2] = trans.astype(np.float32, copy=False)
        affine_inv_h = np.eye(3, dtype=np.float32)
        affine_inv_h[:2] = trans_inv.astype(np.float32, copy=False)

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

        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool) # (H, W)

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
                objects = self.get_label(random_index)
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

                    if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
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

                    if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                        mask_2d[i + object_num] = 1
                    
                    calibs[i + object_num] = model_calib

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
                   'obj_region': obj_region}
        if self.full_p2_projection:
            targets['physical_ray_heading'] = np.bool_(True)
            targets['model_image_size'] = self.resolution.astype(
                np.float32, copy=True)

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
