import json
import os

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.swanlab_helper import ScalarMeanAccumulator
from lib.helpers.swanlab_helper import GeometryIntervalAccumulator
from lib.helpers.swanlab_helper import chinese_geometry_metrics
from lib.helpers.swanlab_helper import chinese_grouped_monitoring
from lib.helpers.swanlab_helper import scalar_values_to_floats
from lib.helpers.gradient_monitor import GradientMonitor
from lib.helpers.gradient_monitor import chinese_gradient_metrics

from utils import misc


_AP_DIFFICULTIES = (
    ('easy', '简单'),
    ('moderate', '中等'),
    ('hard', '困难'),
)

_CORE_GRADIENT_NAMES = {
    '全模型梯度L2范数中位数',
    '全模型梯度L2范数P95',
    '全模型梯度L2范数最大值',
}

_DEPTH_MEAN_CLIP_LABELS = {
    'depth_mean_clip_applied_fraction': '裁剪预测比例',
    'depth_mean_pre_clip_max_absolute_gradient': (
        '裁剪前最大深度均值梯度绝对值'),
    'depth_mean_pre_clip_max_absolute_gradient_max': (
        '裁剪前最大深度均值梯度绝对值'),
    'depth_mean_clip_minimum_retained_fraction': (
        '最严重裁剪预测的梯度保留比例'),
    'depth_mean_clip_minimum_retained_fraction_min': (
        '最严重裁剪预测的梯度保留比例'),
    'depth_mean_clip_retained_energy_fraction': (
        '整体深度均值梯度能量保留比例'),
}

def grouped_gradient_payload(metrics, scope, epoch_summary=False):
    payload = {}
    metrics = {
        key: value for key, value in metrics.items()
        if not key.startswith('depth_mean_')
    }
    for name, value in chinese_gradient_metrics(metrics).items():
        if '梯度与参数范数比' in name:
            continue
        group = (
            f'{scope}核心概览'
            if epoch_summary and name in _CORE_GRADIENT_NAMES
            else f'{scope}梯度诊断')
        payload[f'{group}/{name}'] = value
    return payload


def depth_mean_clipping_payload(metrics, scope):
    return {
        f'{scope}深度均值梯度裁剪/{label}': float(metrics[key])
        for key, label in _DEPTH_MEAN_CLIP_LABELS.items()
        if key in metrics
    }


def update_best_ap_snapshots(snapshots, metrics, epoch):
    """Keep all three AP values from each difficulty's best epoch."""
    current = {}
    for difficulty, _ in _AP_DIFFICULTIES:
        key = f'Car_3d_{difficulty}_R40'
        if key not in metrics:
            return snapshots
        current[difficulty] = float(metrics[key])
    for selected, _ in _AP_DIFFICULTIES:
        previous = snapshots.get(selected)
        if previous is None or current[selected] > previous[selected]:
            snapshots[selected] = {'epoch': int(epoch), **current}
    return snapshots


def historical_best_ap_payload(snapshots):
    payload = {}
    for selected, selected_chinese in _AP_DIFFICULTIES:
        snapshot = snapshots.get(selected)
        if snapshot is None:
            continue
        prefix = f'历史最佳结果/以{selected_chinese}难度为准'
        payload[f'{prefix}/对应轮次'] = snapshot['epoch']
        for difficulty, chinese in _AP_DIFFICULTIES:
            qualifier = '最高' if difficulty == selected else '同轮'
            payload[
                f'{prefix}/{qualifier}{chinese}难度三维AP_R40'
            ] = snapshot[difficulty]
    return payload


def best_refresh_nms_payload(report):
    payload = {}
    for threshold, values in report.items():
        prefix = f'best刷新时BEV NMS诊断/阈值{threshold}'
        metrics = values['metrics']
        for difficulty, chinese in _AP_DIFFICULTIES:
            key = f'Car_3d_{difficulty}_R40'
            if key in metrics:
                payload[f'{prefix}/{chinese}难度三维AP_R40'] = metrics[key]
        payload[f'{prefix}/保留预测数量'] = values['prediction_count']
        payload[f'{prefix}/删除预测数量'] = values[
            'removed_prediction_count']
    return payload


def quality_score_payload(current_report, best_report):
    payload = {}
    for name, evaluation in current_report.items():
        prefix = f'质量排序分数/{name}'
        payload[f'{prefix}/当前中等难度三维AP_R40'] = float(
            evaluation['selection_score'])
        for selected, selected_chinese in _AP_DIFFICULTIES:
            best = best_report.get(name, {}).get(selected)
            if best is None:
                continue
            best_prefix = f'{prefix}/以{selected_chinese}难度为准'
            payload[f'{best_prefix}/历史最优轮次'] = int(best['epoch'])
            for difficulty, chinese in _AP_DIFFICULTIES:
                key = f'Car_3d_{difficulty}_R40'
                if key in best['metrics']:
                    qualifier = '最高' if difficulty == selected else '同轮'
                    payload[f'{best_prefix}/{qualifier}{chinese}AP_R40'] = (
                        best['metrics'][key])
    return payload


def quality_ranking_payload(summary):
    payload = {}
    if not summary:
        return payload
    for score_name, values in summary.get(
            'query_correlation', {}).items():
        prefix = f'质量排序相关度/{score_name}'
        payload[f'{prefix}/Pearson'] = values['pearson']
        payload[f'{prefix}/Spearman'] = values['spearman']
    for score_name, values in summary.get(
            'one_to_one_oracle', {}).get('all', {}).items():
        prefix = f'质量排序最优query/{score_name}'
        labels = {
            'best_iou_mean': '每个GT最高三维IoU均值',
            'best_query_rank_median': '每GT最高IoU query排名中位数',
            'best_query_rank_p90': '每GT最高IoU query排名P90',
            'one_to_one_query_rank_median': (
                '全局一对一IoU分配query排名中位数'),
            'one_to_one_query_rank_p90': (
                '全局一对一IoU分配query排名P90'),
            'top1_identity_fraction': 'Top1与最优query一致率',
            'top1_iou_regret': 'Top1三维IoU遗憾',
            'high_quality_count': '存在IoU至少0.7好query的GT数',
            'high_quality_best_top1_recall': (
                'IoU至少0.7最优query进入Top1比例'),
            'high_quality_best_top3_recall': (
                'IoU至少0.7最优query进入Top3比例'),
            'high_quality_best_top5_recall': (
                'IoU至少0.7最优query进入Top5比例'),
        }
        for key, label in labels.items():
            payload[f'{prefix}/{label}'] = values[key]
    return payload


def _write_json_atomically(path, payload):
    temporary_path = path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def _tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def _to_device_nonblocking(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: _to_device_nonblocking(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_to_device_nonblocking(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device_nonblocking(item, device) for item in value]
    return value


class CudaBatchPrefetcher:
    """Keep one training batch ready while the current batch is computing."""

    def __init__(self, iterator, device, copy_stream):
        self.iterator = iterator
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CudaBatchPrefetcher requires a CUDA device')
        self.copy_stream = copy_stream
        expected_device = (self.device.index if self.device.index is not None
                           else torch.cuda.current_device())
        if self.copy_stream.device.index != expected_device:
            raise ValueError('CUDA prefetch stream is on the wrong device')
        self._next_batch = None
        self._next_host_batch = None
        self._next_ready = None
        self._retained_host_batches = []
        self._preload()

    def _preload(self):
        try:
            host_batch = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            self._next_host_batch = None
            self._next_ready = None
            return

        if not isinstance(host_batch, (tuple, list)) or len(host_batch) != 4:
            raise ValueError('training batch must contain four fields')
        transferable = host_batch[:3]
        leaves = list(_tensor_leaves(transferable))
        if not leaves or not all(tensor.is_pinned() for tensor in leaves):
            raise RuntimeError(
                'CUDA prefetch requires every transferred CPU tensor to be pinned')

        with torch.cuda.stream(self.copy_stream):
            moved = _to_device_nonblocking(transferable, self.device)
            ready = torch.cuda.Event(blocking=False)
            ready.record(self.copy_stream)
        self._next_batch = (*moved, host_batch[3])
        self._next_host_batch = host_batch
        self._next_ready = ready

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._next_ready)
        batch = self._next_batch
        host_batch = self._next_host_batch
        ready = self._next_ready
        for tensor in _tensor_leaves(batch[:3]):
            tensor.record_stream(current_stream)

        # The CPU source must outlive its asynchronous host-to-device copy.
        self._retained_host_batches.append((host_batch, ready))
        if len(self._retained_host_batches) > 2:
            _, old_ready = self._retained_host_batches.pop(0)
            old_ready.synchronize()

        self._preload()
        return batch

    def close(self):
        """Finish outstanding copies and release epoch-local references."""
        if self._next_ready is not None:
            self._next_ready.synchronize()
        for _, ready in self._retained_host_batches:
            ready.synchronize()
        self._retained_host_batches.clear()
        self._next_batch = None
        self._next_host_batch = None
        self._next_ready = None
        self.iterator = None


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name,
                 tracker=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.tracker = tracker
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.use_cuda_batch_prefetch = bool(
            cfg.get('use_cuda_batch_prefetch', False))
        if self.use_cuda_batch_prefetch and self.device.type != 'cuda':
            raise RuntimeError('CUDA batch prefetch is enabled without CUDA')
        # One stream is reused for the entire Trainer lifetime. Creating one
        # stream per epoch previously caused allocator growth across epochs.
        self.cuda_batch_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.use_cuda_batch_prefetch else None)

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        best_result = self.best_result
        best_epoch = self.best_epoch
        best_ap_snapshots = {}
        quality_score_best = {}
        self.logger.info(
            "Training started: epochs=%d, start_epoch=%d",
            self.cfg['max_epoch'], start_epoch)
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            train_summary = self.train_one_epoch(epoch)
            self.epoch += 1
            if self.tracker is not None:
                payload = {
                    '训练轮次': self.epoch,
                }
                payload.update(chinese_grouped_monitoring(
                        train_summary['mean_raw_losses'],
                        self.detr_loss.weight_dict,
                        scope='每轮训练',
                        final_query_label='全部11组'))
                payload.update({
                    f'每轮训练可行区间诊断/{key}': value
                    for key, value in chinese_geometry_metrics(
                        train_summary['geometry_interval']).items()
                })
                payload.update(grouped_gradient_payload(
                    train_summary['gradient_monitoring'],
                    scope='每轮训练', epoch_summary=True))
                payload.update(depth_mean_clipping_payload(
                    train_summary['gradient_monitoring'], scope='每轮训练'))
                self.tracker.log(
                    payload, step=self.epoch * len(self.train_loader))

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                validation_start_epoch = max(
                    1, int(self.cfg.get('validation_start_epoch', 1)))
                if (self.tester is not None
                        and self.epoch >= validation_start_epoch):
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    results = self.tester.inference()
                    evaluation = self.tester.evaluate(
                        results, return_metrics=True)
                    quality_score_report = (
                        self.tester.evaluate_quality_score_variants(
                            primary_evaluation=evaluation))
                    for name, score_evaluation in (
                            quality_score_report.items()):
                        score_best = quality_score_best.setdefault(name, {})
                        for difficulty, _ in _AP_DIFFICULTIES:
                            metric = f'Car_3d_{difficulty}_R40'
                            current_value = score_evaluation['metrics'].get(
                                metric)
                            previous = score_best.get(difficulty)
                            if (current_value is not None
                                    and (previous is None
                                         or current_value
                                         > previous['selected_value'])):
                                score_best[difficulty] = {
                                    'epoch': int(self.epoch),
                                    'selected_value': float(current_value),
                                    'metrics': score_evaluation['metrics'],
                                }
                    if quality_score_report:
                        diagnostics_dir = os.path.join(
                            self.output_dir, 'diagnostics')
                        os.makedirs(diagnostics_dir, exist_ok=True)
                        _write_json_atomically(os.path.join(
                            diagnostics_dir, 'quality_score_grid.json'), {
                                'epoch': int(self.epoch),
                                'primary_score': (
                                    self.tester.primary_quality_score),
                                'current': quality_score_report,
                                'best': quality_score_best,
                            })
                    if self.tester.last_quality_ranking_summary:
                        diagnostics_dir = os.path.join(
                            self.output_dir, 'diagnostics')
                        os.makedirs(diagnostics_dir, exist_ok=True)
                        _write_json_atomically(os.path.join(
                            diagnostics_dir,
                            'quality_ranking_monitor.json'), {
                                'epoch': int(self.epoch),
                                'summary': self.tester.last_quality_ranking_summary,
                            })
                    update_best_ap_snapshots(
                        best_ap_snapshots, evaluation['metrics'], self.epoch)
                    cur_result = evaluation['selection_score']
                    nms_report = {}
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                        nms_report = self.tester.evaluate_best_refresh_bev_nms(
                            results)
                        if nms_report:
                            diagnostics_dir = os.path.join(
                                self.output_dir, 'diagnostics')
                            os.makedirs(diagnostics_dir, exist_ok=True)
                            report_path = os.path.join(
                                diagnostics_dir,
                                'best_refresh_bev_nms.json')
                            temporary_path = report_path + '.tmp'
                            with open(temporary_path, 'w', encoding='utf-8') as handle:
                                json.dump({
                                    'checkpoint_selection': {
                                        'metric': 'Car_3d_moderate_R40',
                                        'nms': 'none',
                                        'epoch': int(self.epoch),
                                        'score': float(cur_result),
                                    },
                                    'bev_nms': nms_report,
                                }, handle, indent=2, ensure_ascii=False)
                            os.replace(temporary_path, report_path)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

                    if self.tracker is not None:
                        validation_loss_summary = (
                            self.tester.last_loss_summary or {})
                        payload = {
                            '训练轮次': self.epoch,
                        }
                        payload.update(historical_best_ap_payload(
                            best_ap_snapshots))
                        payload.update(best_refresh_nms_payload(nms_report))
                        payload.update(quality_score_payload(
                            quality_score_report, quality_score_best))
                        payload.update(quality_ranking_payload(
                            self.tester.last_quality_ranking_summary))
                        payload.update(chinese_grouped_monitoring(
                                validation_loss_summary,
                                self.detr_loss.weight_dict,
                                scope='每轮验证',
                                final_query_label='验证使用查询组'))
                        payload.update({
                            f'每轮验证可行区间诊断/{key}': value
                            for key, value in chinese_geometry_metrics(
                                self.tester.last_geometry_interval_summary
                                or {}).items()
                        })
                        for difficulty, chinese in _AP_DIFFICULTIES:
                            metric = f'Car_3d_{difficulty}_R40'
                            if metric in evaluation['metrics']:
                                payload[
                                    f'每轮三维检测精度/{chinese}难度AP_R40'
                                ] = evaluation['metrics'][metric]
                        self.tracker.log(
                            payload, step=self.epoch * len(self.train_loader))

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        self.detr_loss.train()
        batch_count = len(self.train_loader)
        log_frequency = max(1, int(self.cfg.get('log_frequency', 30)))
        swanlab_interval = max(
            1, int(self.cfg.get('swanlab_batch_interval', 5)))
        self.logger.info(
            "Train epoch started: epoch=%d/%d, batches=%d",
            epoch + 1, self.cfg['max_epoch'], batch_count)
        batch_source = self.train_loader
        prefetched = self.use_cuda_batch_prefetch
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        raw_loss_accumulator = ScalarMeanAccumulator()
        geometry_interval_accumulator = GeometryIntervalAccumulator()
        gradient_cfg = self.cfg.get('gradient_monitoring', {})
        gradient_monitor = (
            GradientMonitor(
                self.model,
                module_interval=gradient_cfg.get('module_interval', 30))
            if gradient_cfg.get('enabled', False) else None)
        depth_mean_clip_receipts = []
        if prefetched:
            batch_source = CudaBatchPrefetcher(
                iter(self.train_loader), self.device,
                copy_stream=self.cuda_batch_copy_stream)
        try:
            for batch_idx, (inputs, calibs, targets, info) in enumerate(batch_source):
                if not prefetched:
                    inputs = inputs.to(self.device)
                    calibs = calibs.to(self.device)
                    for key in targets.keys():
                        targets[key] = targets[key].to(self.device)
                img_sizes = targets['img_size']
                targets = self.prepare_targets(targets, inputs.shape[0])
                ##dn
                dn_args = None
                if self.cfg["use_dn"]:
                    dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
                ###
                # train one batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
                mask_dict=None
                #ipdb.set_trace()
                self.detr_loss.collect_iou3d_matching_comparison = (
                    batch_idx % swanlab_interval == 0
                    or batch_idx + 1 == batch_count)
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

                detr_losses_dict = misc.reduce_dict(detr_losses_dict)
                raw_loss_accumulator.add(detr_losses_dict)
                geometry_receipt = getattr(
                    self.detr_loss,
                    'geometry_conditioned_interval_depth_receipts', {}
                ).get('final')
                geometry_interval_accumulator.add(geometry_receipt)
                detr_losses_log = sum(
                    detr_losses_dict[key] * weight_dict[key]
                    for key in detr_losses_dict
                    if key in weight_dict).detach().item()
                epoch_loss_sum += detr_losses_log
                epoch_batch_count += 1

                should_log = (
                    batch_idx % log_frequency == 0
                    or batch_idx + 1 == batch_count)
                if should_log:
                    loss_keys = sorted(detr_losses_dict)
                    loss_values = torch.stack([
                        detr_losses_dict[key].detach().reshape(())
                        for key in loss_keys
                    ]).cpu().tolist()
                    loss_text = ", ".join(
                        f"{key}={value:.6f}"
                        for key, value in zip(loss_keys, loss_values))
                    learning_rates = ",".join(
                        f"{group['lr']:.8g}"
                        for group in self.optimizer.param_groups)
                    self.logger.info(
                        "Train metrics: epoch=%d/%d, step=%d/%d, "
                        "lr=[%s], loss_detr=%.6f, losses={%s}",
                        epoch + 1, self.cfg['max_epoch'],
                        batch_idx + 1, batch_count, learning_rates,
                        detr_losses_log, loss_text)

                swanlab_payload = None
                swanlab_step = None
                if (self.tracker is not None
                        and (batch_idx % swanlab_interval == 0
                             or batch_idx + 1 == batch_count)):
                    global_step = epoch * batch_count + batch_idx + 1
                    raw_values = scalar_values_to_floats(detr_losses_dict)
                    swanlab_payload = {
                        '训练轮次': epoch + 1,
                        '训练进度/当前批次': batch_idx,
                    }
                    swanlab_payload.update(chinese_grouped_monitoring(
                        raw_values, weight_dict,
                        scope='训练中每5批',
                        final_query_label='全部11组'))
                    if geometry_receipt is not None:
                        count_keys = (
                            'matched_count', 'unique_matched_car_count',
                            'eligible_car_count',
                            'supported_at_gt_count',
                            'valid_interval_count', 'inside_count',
                            'outside_count', 'fallback_native_count')
                        counts = scalar_values_to_floats({
                            key: geometry_receipt[key] for key in count_keys
                        })
                        valid = counts['valid_interval_count']
                        eligible = counts['eligible_car_count']
                        counts['valid_interval_fraction'] = (
                            valid / eligible if eligible else 0.0)
                        counts['outside_fraction'] = (
                            counts['outside_count'] / valid if valid else 0.0)
                        swanlab_payload.update({
                            f'训练中每5批可行区间诊断/{key}': value
                            for key, value in chinese_geometry_metrics(
                                counts).items()
                        })
                    swanlab_step = global_step

                detr_losses.backward()
                unwrapped_model = (
                    self.model.module
                    if hasattr(self.model, 'module') else self.model)
                depth_mean_clip_snapshot = (
                    unwrapped_model.depth_mean_gradient_clipping_metrics()
                    if hasattr(
                        unwrapped_model,
                        'depth_mean_gradient_clipping_metrics') else {})
                if depth_mean_clip_snapshot:
                    depth_mean_clip_receipts.append(
                        depth_mean_clip_snapshot)
                gradient_snapshot = (
                    gradient_monitor.observe(
                        batch_idx, is_last_batch=(
                            batch_idx + 1 == batch_count))
                    if gradient_monitor is not None else {})
                gradient_snapshot.update(depth_mean_clip_snapshot)
                if should_log and gradient_snapshot:
                    gradient_keys = sorted(gradient_snapshot)
                    gradient_values = torch.stack(tuple(
                        gradient_snapshot[key].detach().float().reshape(())
                        for key in gradient_keys)).cpu().tolist()
                    gradient_text = ", ".join(
                        f"{key}={value:.6g}"
                        for key, value in zip(
                            gradient_keys, gradient_values))
                    self.logger.info(
                        "Gradient metrics: epoch=%d/%d, step=%d/%d, {%s}",
                        epoch + 1, self.cfg['max_epoch'],
                        batch_idx + 1, batch_count, gradient_text)
                if swanlab_payload is not None:
                    swanlab_payload.update(grouped_gradient_payload(
                        scalar_values_to_floats(gradient_snapshot),
                        scope='训练中每5批'))
                    swanlab_payload.update(depth_mean_clipping_payload(
                        scalar_values_to_floats(gradient_snapshot),
                        scope='训练中每5批'))
                    self.tracker.log(
                        swanlab_payload, step=swanlab_step)
                self.optimizer.step()

        finally:
            if prefetched:
                batch_source.close()
        gradient_summary = (
            gradient_monitor.finalize()
            if gradient_monitor is not None else {})
        if depth_mean_clip_receipts:
            prediction_count = torch.stack(tuple(
                receipt['depth_mean_clip_prediction_count']
                for receipt in depth_mean_clip_receipts)).sum()
            clipped_count = torch.stack(tuple(
                receipt['depth_mean_clip_applied_count']
                for receipt in depth_mean_clip_receipts)).sum()
            pre_clip_max = torch.stack(tuple(
                receipt['depth_mean_pre_clip_max_absolute_gradient']
                for receipt in depth_mean_clip_receipts)).max()
            minimum_scale = torch.stack(tuple(
                receipt['depth_mean_clip_minimum_retained_fraction']
                for receipt in depth_mean_clip_receipts)).min()
            pre_clip_energy = torch.stack(tuple(
                receipt['depth_mean_pre_clip_energy']
                for receipt in depth_mean_clip_receipts)).sum()
            post_clip_energy = torch.stack(tuple(
                receipt['depth_mean_post_clip_energy']
                for receipt in depth_mean_clip_receipts)).sum()
            clipped_batches = torch.stack(tuple(
                receipt['depth_mean_clip_applied_count'] > 0
                for receipt in depth_mean_clip_receipts)).sum()
            gradient_summary.update({
                'depth_mean_clip_applied_count': int(
                    clipped_count.cpu()),
                'depth_mean_clip_prediction_count': int(
                    prediction_count.cpu()),
                'depth_mean_clip_applied_fraction': float(
                    (clipped_count.float()
                     / prediction_count.clamp_min(1).float()).cpu()),
                'depth_mean_clip_applied_batch_fraction': float(
                    (clipped_batches.float()
                     / len(depth_mean_clip_receipts)).cpu()),
                'depth_mean_pre_clip_max_absolute_gradient_max': float(
                    pre_clip_max.cpu()),
                'depth_mean_clip_minimum_retained_fraction_min': float(
                    minimum_scale.cpu()),
                'depth_mean_clip_retained_energy_fraction': float(
                    torch.where(
                        pre_clip_energy > 0,
                        post_clip_energy
                        / pre_clip_energy.clamp_min(1e-30),
                        torch.ones_like(pre_clip_energy)).cpu()),
            })
        summary = {
            'batch_count': epoch_batch_count,
            'mean_loss': (epoch_loss_sum / epoch_batch_count
                          if epoch_batch_count else float('nan')),
            'mean_raw_losses': raw_loss_accumulator.finalize(),
            'geometry_interval': geometry_interval_accumulator.finalize(),
            'gradient_monitoring': gradient_summary,
        }
        self.logger.info(
            "Train epoch completed: epoch=%d/%d, batches=%d, "
            "mean_loss=%.6f",
            epoch + 1, self.cfg['max_epoch'],
            summary['batch_count'], summary['mean_loss'])
        return summary

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = [
            'labels', 'boxes', 'calibs', 'depth', 'size_3d',
            'heading_bin', 'heading_res', 'boxes_3d', 'src_size_3d',
            'depth_unit_scale', 'projective_rotation_y']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == 'depth_map':
                    target_dict[key] = val[bz]
                if key == 'obj_region':
                    target_dict[key] = val[bz]
                if key in (
                        'img_size',
                        'projective_input_size',
                        'projective_image_effective_calib'):
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list
