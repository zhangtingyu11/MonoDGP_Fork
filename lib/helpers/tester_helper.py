import os
import shutil

import numpy as np
import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import fused_quality_score
from lib.helpers.decode_helper import decode_detections
import time
from lib.helpers.swanlab_helper import ScalarMeanAccumulator
from lib.helpers.swanlab_helper import GeometryIntervalAccumulator
from lib.helpers.bev_nms_helper import classwise_bev_nms_variants
from lib.helpers.quality_ranking_monitor import QualityRankingAccumulator
from lib.helpers.nms_best_query_monitor import NMSBestQueryAccumulator


class CudaEvalBatchPrefetcher:
    """Keep one validation batch ready while the current batch is evaluated."""

    def __init__(self, iterator, device, copy_stream):
        self.iterator = iterator
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CudaEvalBatchPrefetcher requires a CUDA device')
        self.copy_stream = copy_stream
        expected_device = (self.device.index if self.device.index is not None
                           else torch.cuda.current_device())
        if self.copy_stream.device.index != expected_device:
            raise ValueError('CUDA evaluation prefetch stream is on the wrong device')
        self._next_batch = None
        self._next_host_sources = None
        self._next_ready = None
        self._retained_host_sources = []
        self._preload()

    def _preload(self):
        try:
            inputs, calibs, targets, info = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            self._next_host_sources = None
            self._next_ready = None
            return

        img_sizes = info.get('model_image_size', info['img_size'])
        host_sources = (inputs, calibs, img_sizes)
        if not all(tensor.is_pinned() for tensor in host_sources):
            raise RuntimeError(
                'CUDA evaluation prefetch requires pinned source tensors')
        with torch.cuda.stream(self.copy_stream):
            moved = tuple(
                tensor.to(self.device, non_blocking=True)
                for tensor in host_sources)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.copy_stream)
        self._next_batch = (
            moved[0], moved[1], targets, info, moved[2])
        self._next_host_sources = host_sources
        self._next_ready = ready

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._next_ready)
        batch = self._next_batch
        host_sources = self._next_host_sources
        ready = self._next_ready
        for tensor in (batch[0], batch[1], batch[4]):
            tensor.record_stream(current_stream)

        # Pinned CPU storage must remain alive until its asynchronous copy is
        # complete. Two retained batches cover the one-batch look-ahead.
        self._retained_host_sources.append((host_sources, ready))
        if len(self._retained_host_sources) > 2:
            _, old_ready = self._retained_host_sources.pop(0)
            old_ready.synchronize()

        self._preload()
        return batch

    def close(self):
        """Finish outstanding copies and release validation-local references."""
        if self._next_ready is not None:
            self._next_ready.synchronize()
        for _, ready in self._retained_host_sources:
            ready.synchronize()
        self._retained_host_sources.clear()
        self._next_batch = None
        self._next_host_sources = None
        self._next_ready = None
        self.iterator = None


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None,
                 model_name='monodgp', criterion=None, tracker=None):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.criterion = criterion
        self.tracker = tracker
        self.last_loss_summary = None
        self.last_geometry_interval_summary = None
        self.last_inference_stats = None
        self.last_score_variant_results = {}
        self.last_quality_ranking_summary = None
        self.last_nms_best_query_summary = None
        nms_monitor_cfg = cfg.get('nms_best_query_monitoring', {})
        self.nms_best_query_monitoring_enabled = bool(
            nms_monitor_cfg.get('enabled', False))
        self.nms_best_query_monitoring_threshold = float(
            nms_monitor_cfg.get('bev_iou_threshold', 0.8))
        self.nms_best_query_monitoring_min_iou_delta = float(
            nms_monitor_cfg.get('min_iou_delta', 1e-6))
        if (self.nms_best_query_monitoring_enabled
                and self.criterion is None):
            raise ValueError(
                'NMS best-query monitoring requires the criterion')
        self.quality_score_specs = tuple(
            {
                'name': str(spec['name']),
                'alpha': float(spec.get('alpha', 1.0)),
                'beta': float(spec.get('beta', 1.0)),
                'gamma': float(spec.get('gamma', 1.0)),
                'historical_topk': bool(
                    spec.get('historical_topk', False)),
                'classification_only': bool(
                    spec.get('classification_only', False)),
            }
            for spec in cfg.get('quality_score_fusions', ()))
        score_names = [spec['name'] for spec in self.quality_score_specs]
        if len(score_names) != len(set(score_names)):
            raise ValueError('quality score fusion names must be unique')
        self.primary_quality_score = cfg.get('primary_quality_score')
        if self.quality_score_specs and self.primary_quality_score is None:
            raise ValueError(
                'quality score fusions require primary_quality_score')
        if (self.primary_quality_score is not None
                and self.primary_quality_score not in score_names):
            raise ValueError('primary quality score is not registered')
        self.use_cuda_eval_prefetch = bool(
            train_cfg.get('use_cuda_eval_prefetch', False))
        if self.use_cuda_eval_prefetch and self.device.type != 'cuda':
            raise RuntimeError('CUDA evaluation prefetch is enabled without CUDA')
        self.cuda_eval_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.use_cuda_eval_prefetch else None)

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']
        primary_only = bool(
            self.train_cfg.get('primary_ap_only_validation', False))

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            results = self.inference(
                collect_diagnostics=not primary_only,
                primary_only=primary_only)
            self.evaluate(results)

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                results = self.inference(
                    collect_diagnostics=not primary_only,
                    primary_only=primary_only)
                self.evaluate(results)

    def inference(self, collect_diagnostics=True, primary_only=False):
        """Run validation inference.

        ``collect_diagnostics=False`` is the lightweight path used by early
        AP checks: it skips validation losses and ranking monitors.
        ``primary_only=True`` additionally decodes only the configured primary
        score, so pre-registered score grids do not add work.
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        if collect_diagnostics and self.criterion is not None:
            self.criterion.eval()

        results = {}
        active_quality_score_specs = self.quality_score_specs
        if primary_only and active_quality_score_specs:
            active_quality_score_specs = tuple(
                spec for spec in active_quality_score_specs
                if spec['name'] == self.primary_quality_score)
            if len(active_quality_score_specs) != 1:
                raise RuntimeError(
                    'lightweight validation could not resolve the primary '
                    'quality score')
        variant_results = {
            spec['name']: {} for spec in active_quality_score_specs}
        ranking_monitor = (
            QualityRankingAccumulator(
                self.quality_score_specs,
                car_class_id=int(self.cfg.get(
                    'quality_ranking_car_class_id', 1)))
            if (collect_diagnostics
                and self.cfg.get('quality_ranking_monitoring', False))
            else None)
        nms_best_query_monitor = (
            NMSBestQueryAccumulator(
                self.criterion.matcher.iou3d_decode_mean_sizes,
                bev_iou_threshold=(
                    self.nms_best_query_monitoring_threshold),
                min_iou_delta=(
                    self.nms_best_query_monitoring_min_iou_delta))
            if (collect_diagnostics
                and self.nms_best_query_monitoring_enabled)
            else None)
        loss_accumulator = ScalarMeanAccumulator()
        geometry_interval_accumulator = GeometryIntervalAccumulator()
        validation_start = time.time()
        self.logger.info(
            'Validation inference started: batches=%d, images=%d',
            len(self.dataloader), len(self.dataloader.dataset))
        batch_source = self.dataloader
        prefetched = self.use_cuda_eval_prefetch
        if prefetched:
            batch_source = CudaEvalBatchPrefetcher(
                iter(self.dataloader), self.device,
                copy_stream=self.cuda_eval_copy_stream)
        try:
            for batch_idx, batch in enumerate(batch_source):
                if prefetched:
                    inputs, calibs, targets, info, img_sizes = batch
                else:
                    inputs, calibs, targets, info = batch
                    # load evaluation data and move data to GPU.
                    inputs = inputs.to(self.device)
                    calibs = calibs.to(self.device)
                    img_sizes = info.get(
                        'model_image_size', info['img_size']).to(self.device)

                ###dn
                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
                ###

                if collect_diagnostics and self.criterion is not None:
                    device_targets = {
                        key: (value.to(self.device, non_blocking=True)
                              if torch.is_tensor(value) else value)
                        for key, value in targets.items()
                    }
                    mask = device_targets['mask_2d']
                    prepared_targets = []
                    key_list = {
                        'labels', 'boxes', 'calibs', 'depth', 'size_3d',
                        'heading_bin', 'heading_res', 'boxes_3d',
                        'src_size_3d', 'depth_unit_scale',
                        'projective_rotation_y'}
                    for batch_index in range(inputs.shape[0]):
                        item = {}
                        for key, value in device_targets.items():
                            if key in key_list:
                                item[key] = value[batch_index][mask[batch_index]]
                            elif key in ('depth_map', 'obj_region'):
                                item[key] = value[batch_index]
                            elif key in (
                                    'img_size',
                                    'projective_input_size',
                                    'projective_image_effective_calib',
                                    'physical_ray_heading'):
                                item[key] = value[batch_index]
                        prepared_targets.append(item)
                    loss_dict = self.criterion(
                        outputs, prepared_targets, mask_dict=None)
                    loss_accumulator.add(loss_dict)
                    geometry_interval_accumulator.add(getattr(
                        self.criterion,
                        'geometry_conditioned_interval_depth_receipts', {}
                    ).get('final'))

                    if ranking_monitor is not None:
                        ranking_monitor.add(
                            outputs,
                            self.criterion.last_final_iou3d_matrix,
                            prepared_targets,
                            info,
                            self.dataloader.dataset,
                            targets['mask_2d'])
                    if nms_best_query_monitor is not None:
                        nms_best_query_monitor.add(
                            outputs,
                            self.criterion.last_final_iou3d_matrix,
                            prepared_targets)

                # get corresponding calibs & transform tensor to numpy
                calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.dataloader.dataset.cls_mean_size
                if active_quality_score_specs:
                    for spec in active_quality_score_specs:
                        ranking_scores = (
                            None if spec['historical_topk'] else
                            outputs['pred_logits'].sigmoid()
                            if spec['classification_only'] else
                            fused_quality_score(
                                outputs, alpha=spec['alpha'],
                                beta=spec['beta'], gamma=spec['gamma']))
                        dets = extract_dets_from_outputs(
                            outputs=outputs, K=self.max_objs,
                            topk=self.cfg['topk'],
                            ranking_scores=ranking_scores)
                        decoded = decode_detections(
                            dets=dets.detach().cpu().numpy(),
                            info=info, calibs=calibs,
                            cls_mean_size=cls_mean_size,
                            threshold=self.cfg.get('threshold', 0.2))
                        decoded = {
                            image_id: np.asarray(
                                predictions, dtype=np.float32)
                            for image_id, predictions in decoded.items()
                        }
                        variant_results[spec['name']].update(decoded)
                    results = variant_results[self.primary_quality_score]
                else:
                    dets = extract_dets_from_outputs(
                        outputs=outputs, K=self.max_objs,
                        topk=self.cfg['topk'])
                    dets = decode_detections(
                        dets=dets.detach().cpu().numpy(),
                        info=info,
                        calibs=calibs,
                        cls_mean_size=cls_mean_size,
                        threshold=self.cfg.get('threshold', 0.2))
                    results.update(dets)
        finally:
            if prefetched:
                batch_source.close()

        validation_seconds = time.time() - validation_start
        image_count = len(self.dataloader.dataset)
        self.last_loss_summary = loss_accumulator.finalize()
        self.last_geometry_interval_summary = (
            geometry_interval_accumulator.finalize())
        self.last_inference_stats = {
            'seconds': validation_seconds,
            'images_per_second': image_count / validation_seconds,
            'images': image_count,
            'batches': len(self.dataloader),
        }
        self.last_score_variant_results = variant_results
        self.last_quality_ranking_summary = (
            ranking_monitor.finalize() if ranking_monitor is not None else None)
        self.last_nms_best_query_summary = (
            nms_best_query_monitor.finalize()
            if nms_best_query_monitor is not None else None)
        self.logger.info(
            'Validation inference completed: batches=%d, images=%d, '
            'seconds=%.3f, images_per_second=%.3f',
            len(self.dataloader), image_count, validation_seconds,
            image_count / validation_seconds)

        # Validation is evaluated directly from memory.  Writing thousands of
        # KITTI text files is reserved for explicitly requested exports.
        if self.cfg.get('export_predictions', False):
            self.logger.info('==> Exporting KITTI prediction files ...')
            self.save_results(results)
        return results

    def evaluate_quality_score_variants(self, primary_evaluation=None):
        """Evaluate every pre-registered score from the same forward pass."""
        report = {}
        for name, results in self.last_score_variant_results.items():
            if (name == self.primary_quality_score
                    and primary_evaluation is not None):
                report[name] = primary_evaluation
                continue
            self.logger.info('Quality score evaluation: %s', name)
            report[name] = self.evaluate(results, return_metrics=True)
        return report

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self, results, return_metrics=False):
        result = self.dataloader.dataset.eval(
            results=results, logger=self.logger,
            return_metrics=return_metrics)
        return result

    def evaluate_bev_nms(self, results, thresholds):
        """Evaluate requested BEV-NMS thresholds without another forward."""
        variants = classwise_bev_nms_variants(results, thresholds)
        report = {}
        baseline_count = sum(len(items) for items in results.values())
        for threshold, filtered in variants.items():
            self.logger.info(
                'Best-refresh class-wise BEV NMS evaluation: threshold=%.2f',
                threshold)
            evaluation = self.evaluate(filtered, return_metrics=True)
            remaining = sum(len(items) for items in filtered.values())
            report[f'{threshold:.2f}'] = {
                'selection_score': float(evaluation['selection_score']),
                'prediction_count': int(remaining),
                'removed_prediction_count': int(baseline_count - remaining),
                'metrics': evaluation['metrics'],
            }
        return report

    def evaluate_best_refresh_bev_nms(self, results):
        """Evaluate the pre-registered NMS grid without another forward pass."""
        return self.evaluate_bev_nms(
            results,
            self.cfg.get('best_refresh_bev_nms_thresholds', ()))
