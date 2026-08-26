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

_MIXUP_TARGET_KEYS = (
    'mixup_requested', 'mixup_applied', 'mixup_cross_focal',
    'mixup_valid_ratio', 'mixup_attempts', 'mixup_reject_capacity',
    'mixup_reject_geometry', 'mixup_reject_no_overlap',
    'mixup_reject_partial_object',
    'mixup_reject_primary_mask_boundary',
    'mixup_reject_donor_mask_boundary',
    'mixup_reject_center_outside', 'mixup_reject_no_valid_target',
    'mixup_focal_scale_x', 'mixup_focal_scale_y',
    'mixup_virtual_focal_multiplier',
    'mixup_virtual_focal_requested_multiplier',
    'mixup_virtual_focal_cancelled',
    'mixup_donor_target_count', 'mixup_retained_support_min',
    'mixup_retained_support_observed',
    'mixup_projection_residual_sum', 'mixup_projection_residual_max',
    'mixup_depth_shift_abs_sum', 'mixup_depth_shift_abs_max',
    'mixup_primary_donor_overlap_ratio',
)


def collect_mixup_counts(targets):
    if not all(key in targets for key in _MIXUP_TARGET_KEYS):
        return {}
    requested = targets['mixup_requested'].detach().float().reshape(-1)
    applied = targets['mixup_applied'].detach().float().reshape(-1)
    cross_focal = targets['mixup_cross_focal'].detach().float().reshape(-1)
    valid_ratio = targets['mixup_valid_ratio'].detach().float().reshape(-1)
    focal_scale_x = targets['mixup_focal_scale_x'].detach().float().reshape(-1)
    focal_scale_y = targets['mixup_focal_scale_y'].detach().float().reshape(-1)
    virtual_focal = (
        targets['mixup_virtual_focal_multiplier']
        .detach().float().reshape(-1))
    virtual_focal_requested = (
        targets['mixup_virtual_focal_requested_multiplier']
        .detach().float().reshape(-1))
    virtual_focal_cancelled = (
        targets['mixup_virtual_focal_cancelled']
        .detach().float().reshape(-1))
    donor_target_count = (
        targets['mixup_donor_target_count'].detach().float().reshape(-1))
    retained_support = (
        targets['mixup_retained_support_min'].detach().float().reshape(-1))
    retained_support_observed = (
        targets['mixup_retained_support_observed']
        .detach().float().reshape(-1))
    applied_mask = applied > 0
    return {
        'sample_count': requested.new_tensor(float(requested.numel())),
        'requested_count': requested.sum(),
        'applied_count': applied.sum(),
        'cross_focal_count': cross_focal.sum(),
        'valid_ratio_sum': valid_ratio.sum(),
        'cross_valid_ratio_sum': (valid_ratio * cross_focal).sum(),
        'attempt_sum': targets['mixup_attempts'].detach().float().sum(),
        'reject_capacity_count': (
            targets['mixup_reject_capacity'].detach().float().sum()),
        'reject_geometry_count': (
            targets['mixup_reject_geometry'].detach().float().sum()),
        'reject_no_overlap_count': (
            targets['mixup_reject_no_overlap'].detach().float().sum()),
        'reject_partial_object_count': (
            targets['mixup_reject_partial_object'].detach().float().sum()),
        'reject_primary_mask_boundary_count': (
            targets['mixup_reject_primary_mask_boundary']
            .detach().float().sum()),
        'reject_donor_mask_boundary_count': (
            targets['mixup_reject_donor_mask_boundary']
            .detach().float().sum()),
        'reject_center_outside_count': (
            targets['mixup_reject_center_outside'].detach().float().sum()),
        'reject_no_valid_target_count': (
            targets['mixup_reject_no_valid_target'].detach().float().sum()),
        'focal_scale_x_sum': focal_scale_x.sum(),
        'focal_scale_y_sum': focal_scale_y.sum(),
        'cross_focal_scale_x_sum': (focal_scale_x * cross_focal).sum(),
        'cross_focal_scale_y_sum': (focal_scale_y * cross_focal).sum(),
        'virtual_focal_sum': (virtual_focal * applied).sum(),
        'virtual_focal_cancelled_count': (
            virtual_focal_cancelled * applied).sum(),
        'virtual_focal_requested_0_9_count': (
            applied * torch.isclose(
                virtual_focal_requested,
                virtual_focal_requested.new_tensor(0.9))).sum(),
        'virtual_focal_requested_1_1_count': (
            applied * torch.isclose(
                virtual_focal_requested,
                virtual_focal_requested.new_tensor(1.1))).sum(),
        'virtual_focal_cancelled_0_9_count': (
            applied * virtual_focal_cancelled * torch.isclose(
                virtual_focal_requested,
                virtual_focal_requested.new_tensor(0.9))).sum(),
        'virtual_focal_cancelled_1_1_count': (
            applied * virtual_focal_cancelled * torch.isclose(
                virtual_focal_requested,
                virtual_focal_requested.new_tensor(1.1))).sum(),
        'virtual_focal_0_9_count': (
            applied * torch.isclose(
                virtual_focal, virtual_focal.new_tensor(0.9))).sum(),
        'virtual_focal_1_0_count': (
            applied * torch.isclose(
                virtual_focal, virtual_focal.new_tensor(1.0))).sum(),
        'virtual_focal_1_1_count': (
            applied * torch.isclose(
                virtual_focal, virtual_focal.new_tensor(1.1))).sum(),
        'donor_target_count_sum': donor_target_count.sum(),
        'total_target_count_sum': (
            targets['mask_2d'].detach().float().sum()),
        'retained_support_minimum': torch.where(
            applied_mask & (retained_support_observed > 0), retained_support,
            torch.ones_like(retained_support)).min(),
        'retained_support_observed_count': (
            (applied_mask & (retained_support_observed > 0)).float().sum()),
        'projection_residual_sum': (
            targets['mixup_projection_residual_sum'].detach().float().sum()),
        'projection_residual_maximum': (
            targets['mixup_projection_residual_max'].detach().float().max()),
        'depth_shift_abs_sum': (
            targets['mixup_depth_shift_abs_sum'].detach().float().sum()),
        'depth_shift_abs_maximum': (
            targets['mixup_depth_shift_abs_max'].detach().float().max()),
        'overlap_ratio_sum': (
            targets['mixup_primary_donor_overlap_ratio']
            .detach().float().sum()),
        'overlap_positive_count': (
            (targets['mixup_primary_donor_overlap_ratio'] > 0)
            .detach().float().sum()),
    }


def add_mixup_counts(total, current):
    if not current:
        return total
    if not total:
        return {key: value.clone() for key, value in current.items()}
    for key, value in current.items():
        if key in (
                'retained_support_minimum',
                'projection_residual_maximum',
                'depth_shift_abs_maximum'):
            operation = torch.minimum if key == 'retained_support_minimum' else torch.maximum
            total[key] = operation(total[key], value)
        else:
            total[key] = total[key] + value
    return total


def mixup_monitor_payload(counts, scope):
    if not counts:
        return {}
    values = scalar_values_to_floats(counts)
    sample_count = values['sample_count']
    requested = values['requested_count']
    applied = values['applied_count']
    cross_focal = values['cross_focal_count']
    safe_sample = sample_count if sample_count else 1.0
    safe_requested = requested if requested else 1.0
    safe_applied = applied if applied else 1.0
    safe_cross_focal = cross_focal if cross_focal else 1.0
    prefix = f'{scope}跨焦距MixUp'
    payload = {
        f'{prefix}/请求样本比例': requested / safe_sample,
        f'{prefix}/实际启用样本比例': applied / safe_sample,
        f'{prefix}/请求后成功率': applied / safe_requested,
        f'{prefix}/成功样本中跨P2比例': (
            values['cross_focal_count'] / safe_applied),
        f'{prefix}/成功样本平均有效像素覆盖率': (
            values['valid_ratio_sum'] / safe_applied),
        f'{prefix}/跨P2样本平均有效像素覆盖率': (
            values['cross_valid_ratio_sum'] / safe_cross_focal),
        f'{prefix}/请求样本平均候选尝试次数': (
            values['attempt_sum'] / safe_requested),
        f'{prefix}/成功样本平均水平焦距倍率': (
            values['focal_scale_x_sum'] / safe_applied),
        f'{prefix}/成功样本平均垂直焦距倍率': (
            values['focal_scale_y_sum'] / safe_applied),
        f'{prefix}/跨P2样本平均水平焦距倍率': (
            values['cross_focal_scale_x_sum'] / safe_cross_focal),
        f'{prefix}/跨P2样本平均垂直焦距倍率': (
            values['cross_focal_scale_y_sum'] / safe_cross_focal),
        f'{prefix}/成功样本平均虚拟焦距倍率': (
            values['virtual_focal_sum'] / safe_applied),
        f'{prefix}/成功样本虚拟焦距因新增裁车取消比例': (
            values['virtual_focal_cancelled_count'] / safe_applied),
        f'{prefix}/请求0.9虚拟焦距样本取消比例': (
            values['virtual_focal_cancelled_0_9_count']
            / max(values['virtual_focal_requested_0_9_count'], 1.0)),
        f'{prefix}/请求1.1虚拟焦距样本取消比例': (
            values['virtual_focal_cancelled_1_1_count']
            / max(values['virtual_focal_requested_1_1_count'], 1.0)),
        f'{prefix}/成功样本虚拟焦距0.9比例': (
            values['virtual_focal_0_9_count'] / safe_applied),
        f'{prefix}/成功样本虚拟焦距1.0比例': (
            values['virtual_focal_1_0_count'] / safe_applied),
        f'{prefix}/成功样本虚拟焦距1.1比例': (
            values['virtual_focal_1_1_count'] / safe_applied),
        f'{prefix}/成功样本平均供体GT数': (
            values['donor_target_count_sum'] / safe_applied),
        f'{prefix}/全部训练GT中供体GT比例': (
            values['donor_target_count_sum']
            / max(values['total_target_count_sum'], 1.0)),
        f'{prefix}/具有可统计供体车辆的成功样本比例': (
            values['retained_support_observed_count'] / safe_applied),
        f'{prefix}/供体三维中心投影一致性平均误差像素': (
            values['projection_residual_sum']
            / max(values['donor_target_count_sum'], 1.0)),
        f'{prefix}/供体三维中心投影一致性最大误差像素': (
            values['projection_residual_maximum']),
        f'{prefix}/供体转换前后深度绝对变化均值米': (
            values['depth_shift_abs_sum']
            / max(values['donor_target_count_sum'], 1.0)),
        f'{prefix}/供体转换前后深度绝对变化最大值米': (
            values['depth_shift_abs_maximum']),
        f'{prefix}/主图供体Region平均重叠比例': (
            values['overlap_ratio_sum'] / safe_applied),
        f'{prefix}/成功样本中主图供体Region发生重叠比例': (
            values['overlap_positive_count'] / safe_applied),
        f'{prefix}/每个请求因目标数上限拒绝次数': (
            values['reject_capacity_count'] / safe_requested),
        f'{prefix}/每个请求因投影几何拒绝次数': (
            values['reject_geometry_count'] / safe_requested),
        f'{prefix}/每个请求因无有效覆盖拒绝次数': (
            values['reject_no_overlap_count'] / safe_requested),
        f'{prefix}/请求样本因供体目标部分可见取消比例': (
            values['reject_partial_object_count'] / safe_requested),
        f'{prefix}/每个请求因有效区边界切过主图车辆拒绝次数': (
            values['reject_primary_mask_boundary_count'] / safe_requested),
        f'{prefix}/每个请求因有效区边界切过供体车辆拒绝次数': (
            values['reject_donor_mask_boundary_count'] / safe_requested),
        f'{prefix}/每个请求因三维中心不可编码拒绝次数': (
            values['reject_center_outside_count'] / safe_requested),
        f'{prefix}/每个请求因无可训练供体目标拒绝次数': (
            values['reject_no_valid_target_count'] / safe_requested),
    }
    if values['retained_support_observed_count'] > 0:
        payload[f'{prefix}/保留供体框最小RGB有效覆盖率'] = (
            values['retained_support_minimum'])
    return payload


_MIXUP_MATCHED_METRIC_LABELS = {
    'matched_count': '匹配数量',
    'matched_class_probability': '匹配query真实类别概率',
    'bbox_component_mae': '二维框四分量平均绝对误差',
    'giou_error': '二维框GIoU误差',
    'center_error_pixels': '三维投影中心像素误差',
    'depth_mae_m': '深度绝对误差米',
    'dimension_component_mae': '三维尺寸分量平均绝对误差',
    'angle_class_accuracy': '航向角分箱正确率',
    'angle_residual_mae': '航向角残差绝对误差',
}


def mixup_matched_target_payload(raw_values, scope):
    payload = {}
    for source, source_label in (
            ('primary', '主图GT'), ('donor', '供体GT')):
        prefix = f'monitor_mixup_{source}_'
        for metric, label in _MIXUP_MATCHED_METRIC_LABELS.items():
            key = prefix + metric
            if key in raw_values:
                payload[
                    f'{scope}MixUp主供体匹配对照/{source_label}/{label}'
                ] = raw_values[key]
    return payload

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


def nms_best_selection_payload(state, current):
    """Expose the every-validation NMS arm and its independently saved best."""
    if not current:
        return {}
    threshold = str(state['threshold'])
    prefix = f'每轮BEV NMS独立选优/阈值{threshold}'
    payload = {
        f'{prefix}/当前轮次': int(current['epoch']),
        f'{prefix}/当前中等难度三维AP_R40': float(
            current['selection_score']),
        f'{prefix}/当前保留预测数量': int(current['prediction_count']),
        f'{prefix}/当前删除预测数量': int(
            current['removed_prediction_count']),
        f'{prefix}/历史最优轮次': int(state['epoch']),
        f'{prefix}/历史最高中等难度三维AP_R40': float(state['score']),
    }
    for difficulty, chinese in _AP_DIFFICULTIES:
        metric = f'Car_3d_{difficulty}_R40'
        if metric in current['metrics']:
            payload[f'{prefix}/当前{chinese}难度三维AP_R40'] = float(
                current['metrics'][metric])
        if metric in state.get('metrics', {}):
            payload[f'{prefix}/最优轮同轮{chinese}难度三维AP_R40'] = float(
                state['metrics'][metric])
    for control_threshold, control in current.get(
            'control_reports', {}).items():
        control_prefix = f'每轮BEV NMS对照/阈值{control_threshold}'
        payload[f'{control_prefix}/当前轮次'] = int(current['epoch'])
        payload[f'{control_prefix}/当前保留预测数量'] = int(
            control['prediction_count'])
        payload[f'{control_prefix}/当前删除预测数量'] = int(
            control['removed_prediction_count'])
        for difficulty, chinese in _AP_DIFFICULTIES:
            metric = f'Car_3d_{difficulty}_R40'
            if metric in control['metrics']:
                payload[f'{control_prefix}/{chinese}难度三维AP_R40'] = (
                    float(control['metrics'][metric]))
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
            'top3_identity_fraction': '最优query进入Top3比例',
            'top1_iou_regret': 'Top1三维IoU遗憾',
            'pairwise_order_accuracy_gap_ge_0_1': (
                'IoU差至少0.1候选对排序正确率'),
            'pairwise_order_pair_count_gap_ge_0_1': (
                'IoU差至少0.1候选对数'),
            'high_quality_count': '存在IoU至少0.7好query的GT数',
            'high_quality_top1_iou_ge_0_7_fraction': (
                '存在好query时Top1仍达IoU0.7比例'),
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


def nms_best_query_payload(summary):
    if not summary:
        return {}
    labels = {
        'gt_count': '参与统计GT总数',
        'best_query_retained_count': '最高IoU query被NMS保留数量',
        'best_query_retained_fraction': '最高IoU query被NMS保留率',
        'best_query_suppressed_count': '最高IoU query被NMS压掉数量',
        'best_query_suppressed_fraction': '最高IoU query被NMS压掉率',
        'best_query_suppressed_by_worse_count': '被更差query压掉数量',
        'best_query_suppressed_by_worse_fraction': '被更差query压掉率',
        'best_iou_mean': 'NMS前每GT最高三维IoU均值',
        'kept_best_iou_mean': 'NMS后每GT最高三维IoU均值',
        'nms_iou_regret_mean': 'NMS三维IoU遗憾均值',
        'suppressed_iou_gap_mean': '压掉最优query的IoU差均值',
    }
    return {
        f'NMS最优query诊断/{label}': summary[key]
        for key, label in labels.items()
        if key in summary
    }


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
        nms_best_cfg = cfg.get('nms_best_selection', {})
        self.nms_best_selection_enabled = bool(
            nms_best_cfg.get('enabled', False))
        self.nms_best_selection_threshold = float(
            nms_best_cfg.get('bev_iou_threshold', 0.8))
        if not 0.0 <= self.nms_best_selection_threshold <= 1.0:
            raise ValueError('NMS best-selection threshold must be in [0, 1]')
        self.nms_report_thresholds = tuple(dict.fromkeys(
            float(value) for value in nms_best_cfg.get(
                'report_bev_iou_thresholds', ())))
        if any(not 0.0 <= value <= 1.0
               for value in self.nms_report_thresholds):
            raise ValueError('NMS report thresholds must be in [0, 1]')
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

    def _initial_nms_best_state(self):
        threshold = f'{self.nms_best_selection_threshold:.2f}'
        default = {
            'threshold': threshold,
            'score': 0.0,
            'epoch': 0,
            'metrics': {},
        }
        if not self.nms_best_selection_enabled:
            return default
        path = os.path.join(
            self.output_dir, 'diagnostics', 'nms_best_selection.json')
        if not os.path.isfile(path):
            return default
        with open(path, 'r', encoding='utf-8') as handle:
            receipt = json.load(handle)
        saved = receipt.get('best', {})
        if str(saved.get('threshold')) != threshold:
            raise ValueError('saved NMS-best threshold does not match config')
        return {
            'threshold': threshold,
            'score': float(saved.get('selection_score', 0.0)),
            'epoch': int(saved.get('epoch', 0)),
            'metrics': saved.get('metrics', {}),
        }

    def _evaluate_nms_best_selection(self, results, state):
        if not self.nms_best_selection_enabled:
            return state, {}
        if self.tester is None:
            raise RuntimeError('NMS best selection requires a tester')
        threshold = self.nms_best_selection_threshold
        key = f'{threshold:.2f}'
        thresholds = tuple(dict.fromkeys((
            threshold, *self.nms_report_thresholds)))
        report = self.tester.evaluate_bev_nms(results, thresholds)
        current_report = report[key]
        current = {
            'threshold': key,
            'epoch': int(self.epoch),
            **current_report,
            'control_reports': {
                report_key: value
                for report_key, value in report.items()
                if report_key != key
            },
        }
        if float(current_report['selection_score']) > float(state['score']):
            state = {
                'threshold': key,
                'score': float(current_report['selection_score']),
                'epoch': int(self.epoch),
                'metrics': current_report['metrics'],
            }
            checkpoint_name = os.path.join(
                self.output_dir,
                f'checkpoint_best_bev_nms_{key.replace(".", "_")}')
            save_checkpoint(
                get_checkpoint_state(
                    self.model, self.optimizer, self.epoch,
                    state['score'], state['epoch']),
                checkpoint_name)
        diagnostics_dir = os.path.join(self.output_dir, 'diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)
        _write_json_atomically(os.path.join(
            diagnostics_dir, 'nms_best_selection.json'), {
                'selection_metric': 'Car_3d_moderate_R40',
                'current': current,
                'best': {
                    'threshold': state['threshold'],
                    'selection_score': state['score'],
                    'epoch': state['epoch'],
                    'metrics': state['metrics'],
                },
            })
        self.logger.info(
            'BEV NMS %.2f current=%.6f epoch=%d; best=%.6f epoch=%d',
            threshold, float(current_report['selection_score']), self.epoch,
            float(state['score']), int(state['epoch']))
        for control_key, control in current['control_reports'].items():
            self.logger.info(
                'BEV NMS %s control current=%.6f epoch=%d',
                control_key, float(control['selection_score']), self.epoch)
        return state, current
        
    def train(self):
        start_epoch = self.epoch

        best_result = self.best_result
        best_epoch = self.best_epoch
        best_ap_snapshots = {}
        quality_score_best = {}
        nms_best_state = self._initial_nms_best_state()
        self.logger.info(
            "Training started: epochs=%d, start_epoch=%d",
            self.cfg['max_epoch'], start_epoch)
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
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
                payload.update(mixup_monitor_payload(
                    train_summary['mixup_counts'], scope='每轮训练'))
                payload.update(mixup_matched_target_payload(
                    train_summary['mean_raw_losses'], scope='每轮训练'))
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
                early_validation_interval = max(
                    0, int(self.cfg.get('early_validation_interval', 0)))
                formal_validation = self.epoch >= validation_start_epoch
                early_validation = (
                    self.epoch < validation_start_epoch
                    and early_validation_interval > 0
                    and self.epoch % early_validation_interval == 0)
                early_validation_updates_best = bool(
                    self.cfg.get('early_validation_updates_best', False))
                if (self.tester is not None
                        and (formal_validation or early_validation)):
                    validation_kind = (
                        'formal' if formal_validation else 'early-diagnostic')
                    self.logger.info(
                        "Test Epoch %d (%s)", self.epoch, validation_kind)
                    results = self.tester.inference(
                        collect_diagnostics=formal_validation,
                        primary_only=early_validation)
                    evaluation = self.tester.evaluate(
                        results, return_metrics=True)
                    nms_best_state, current_nms_selection = (
                        self._evaluate_nms_best_selection(
                            results, nms_best_state))
                    if early_validation:
                        early_suffix = (
                            'best selection enabled'
                            if early_validation_updates_best
                            else 'best selection unchanged')
                        self.logger.info(
                            'Early diagnostic AP: epoch=%d, '
                            'Car_3d_easy_R40=%.6f, '
                            'Car_3d_moderate_R40=%.6f, '
                            'Car_3d_hard_R40=%.6f; %s',
                            self.epoch,
                            evaluation['metrics'].get(
                                'Car_3d_easy_R40', float('nan')),
                            evaluation['metrics'].get(
                                'Car_3d_moderate_R40', float('nan')),
                            evaluation['metrics'].get(
                                'Car_3d_hard_R40', float('nan')),
                            early_suffix)
                        payload = {'训练轮次': self.epoch}
                        for difficulty, chinese in _AP_DIFFICULTIES:
                            metric = f'Car_3d_{difficulty}_R40'
                            if metric in evaluation['metrics']:
                                payload[
                                    '前120轮诊断三维检测精度/'
                                    f'{chinese}难度AP_R40'
                                ] = evaluation['metrics'][metric]

                        if early_validation_updates_best:
                            update_best_ap_snapshots(
                                best_ap_snapshots,
                                evaluation['metrics'], self.epoch)
                            cur_result = evaluation['selection_score']
                            nms_report = {}
                            if cur_result > best_result:
                                best_result = cur_result
                                best_epoch = self.epoch
                                ckpt_name = os.path.join(
                                    self.output_dir, 'checkpoint_best')
                                save_checkpoint(
                                    get_checkpoint_state(
                                        self.model, self.optimizer,
                                        self.epoch, best_result, best_epoch),
                                    ckpt_name)
                                nms_report = (
                                    self.tester
                                    .evaluate_best_refresh_bev_nms(results))
                                if nms_report:
                                    diagnostics_dir = os.path.join(
                                        self.output_dir, 'diagnostics')
                                    os.makedirs(
                                        diagnostics_dir, exist_ok=True)
                                    report_path = os.path.join(
                                        diagnostics_dir,
                                        'best_refresh_bev_nms.json')
                                    _write_json_atomically(report_path, {
                                        'checkpoint_selection': {
                                            'metric': (
                                                'Car_3d_moderate_R40'),
                                            'nms': 'none',
                                            'epoch': int(self.epoch),
                                            'score': float(cur_result),
                                        },
                                        'bev_nms': nms_report,
                                    })
                            self.logger.info(
                                "Best Result:{}, epoch:{}".format(
                                    best_result, best_epoch))
                            payload.update(historical_best_ap_payload(
                                best_ap_snapshots))
                            payload.update(best_refresh_nms_payload(
                                nms_report))
                        if self.tracker is not None:
                            payload.update(nms_best_selection_payload(
                                nms_best_state, current_nms_selection))
                            self.tracker.log(
                                payload,
                                step=self.epoch * len(self.train_loader))
                        continue
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
                    if self.tester.last_nms_best_query_summary:
                        diagnostics_dir = os.path.join(
                            self.output_dir, 'diagnostics')
                        os.makedirs(diagnostics_dir, exist_ok=True)
                        _write_json_atomically(os.path.join(
                            diagnostics_dir,
                            'nms_best_query_monitor.json'), {
                                'epoch': int(self.epoch),
                                'bev_iou_threshold': (
                                    self.tester
                                    .nms_best_query_monitoring_threshold),
                                'summary': (
                                    self.tester
                                    .last_nms_best_query_summary),
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
                        payload.update(nms_best_selection_payload(
                            nms_best_state, current_nms_selection))
                        payload.update(quality_score_payload(
                            quality_score_report, quality_score_best))
                        payload.update(quality_ranking_payload(
                            self.tester.last_quality_ranking_summary))
                        payload.update(nms_best_query_payload(
                            self.tester.last_nms_best_query_summary))
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
        batch_monitor_scope = f'训练中每{swanlab_interval}批'
        mixup_target_cfg = self.cfg.get('mixup_target_monitoring', {})
        mixup_target_monitoring_enabled = bool(
            mixup_target_cfg.get('enabled', False))
        mixup_target_interval = max(
            1, int(mixup_target_cfg.get('interval', 30)))
        self.logger.info(
            "Train epoch started: epoch=%d/%d, batches=%d",
            epoch + 1, self.cfg['max_epoch'], batch_count)
        batch_source = self.train_loader
        prefetched = self.use_cuda_batch_prefetch
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        raw_loss_accumulator = ScalarMeanAccumulator()
        geometry_interval_accumulator = GeometryIntervalAccumulator()
        mixup_epoch_counts = {}
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
                mixup_batch_counts = collect_mixup_counts(targets)
                mixup_epoch_counts = add_mixup_counts(
                    mixup_epoch_counts, mixup_batch_counts)
                img_sizes = targets.get(
                    'model_image_size', targets['img_size'])
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
                self.detr_loss.collect_mixup_target_monitoring = (
                    mixup_target_monitoring_enabled
                    and (batch_idx % mixup_target_interval == 0
                         or batch_idx + 1 == batch_count))
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
                        scope=batch_monitor_scope,
                        final_query_label='全部11组'))
                    swanlab_payload.update(mixup_monitor_payload(
                        mixup_batch_counts, scope=batch_monitor_scope))
                    swanlab_payload.update(mixup_matched_target_payload(
                        raw_values, scope=batch_monitor_scope))
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
                            f'{batch_monitor_scope}可行区间诊断/{key}': value
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
                        scope=batch_monitor_scope))
                    swanlab_payload.update(depth_mean_clipping_payload(
                        scalar_values_to_floats(gradient_snapshot),
                        scope=batch_monitor_scope))
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
            'mixup_counts': mixup_epoch_counts,
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
            'depth_unit_scale', 'projective_rotation_y',
            'mixup_is_donor']
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
                        'projective_image_effective_calib',
                        'physical_ray_heading'):
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list
