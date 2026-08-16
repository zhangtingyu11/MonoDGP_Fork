"""Small, optional SwanLab adapter for train/validation observability."""

from __future__ import annotations

import math
import os
import re

import torch


GEOMETRY_INTERVAL_CHINESE_NAMES = {
    'matched_count': 'Hungarian匹配预测总数（全部类别）',
    'unique_matched_car_count': '去重后参与匹配的真实车辆数',
    'eligible_car_count': '参与区间统计的车辆匹配预测总数（训练含11组）',
    'supported_at_gt_count': '真值深度处三维IoU达到阈值的车辆匹配预测数',
    'valid_interval_count': '可计算有效深度区间的车辆匹配预测数',
    'fallback_native_count': '无法形成有效区间并回退原始深度损失的匹配预测数',
    'inside_count': '预测深度位于有效区间内的车辆匹配预测数',
    'outside_count': '预测深度位于有效区间外的车辆匹配预测数',
    'valid_interval_fraction': '可形成有效区间的车辆匹配预测占比',
    'outside_fraction': '有效区间预测中深度落在区间外的占比',
    'mean_iou_at_gt': '车辆匹配预测在真值深度处的平均三维IoU',
    'depth_mae_all_car': '全部车辆匹配预测的深度MAE',
    'depth_mae_supported': '具有有效区间的车辆匹配预测深度MAE',
    'depth_mae_unsupported': '无法形成有效区间的车辆匹配预测深度MAE',
    'depth_mae_inside': '预测深度位于有效区间内的匹配预测MAE',
    'depth_mae_outside': '预测深度位于有效区间外的匹配预测MAE',
    'mean_outside_boundary_distance': '区间外匹配预测距最近区间边界的平均距离',
    'left_width_virtual_p10': '左侧区间宽度P10',
    'left_width_virtual_median': '左侧区间宽度中位数',
    'left_width_virtual_p90': '左侧区间宽度P90',
    'right_width_virtual_p10': '右侧区间宽度P10',
    'right_width_virtual_median': '右侧区间宽度中位数',
    'right_width_virtual_p90': '右侧区间宽度P90',
    'total_width_virtual_p10': '总区间宽度P10',
    'total_width_virtual_median': '总区间宽度中位数',
    'total_width_virtual_p90': '总区间宽度P90',
}

LOSS_CHINESE_NAMES = {
    'loss_ce': '分类损失',
    'loss_bbox': '二维框位置损失',
    'loss_giou': '二维框重叠损失',
    'loss_dim': '三维尺寸损失',
    'loss_angle': '航向角损失',
    'loss_depth': '深度均值损失',
    'loss_center': '投影三维中心损失',
    'loss_depth_map': '密集深度图损失',
    'loss_region': '物体区域损失',
    'loss_quality': '三维IoU质量回归损失',
}

LOSS_DIAGNOSTIC_CHINESE_NAMES = {
    'monitor_angle_classification': '航向角分类交叉熵',
    'monitor_angle_residual': '航向角残差绝对误差',
    'monitor_depth_mae': '深度绝对误差',
    'monitor_depth_weighted_absolute': '不确定性加权深度绝对误差',
    'monitor_depth_log_scale_mean': '深度对数尺度均值',
    'monitor_depth_log_scale_p10': '深度对数尺度P10',
    'monitor_depth_log_scale_p50': '深度对数尺度P50',
    'monitor_depth_log_scale_p90': '深度对数尺度P90',
    'monitor_depth_precision_mean': '深度置信精度均值（exp负对数尺度）',
    'monitor_depth_precision_p90': '深度置信精度P90（exp负对数尺度）',
    'monitor_depth_mae_precision_correlation': '深度误差与置信精度相关系数',
    'monitor_depth_precision_gt_2_fraction': '深度置信精度大于2的匹配预测占比',
    'monitor_depth_precision_gt_4_fraction': '深度置信精度大于4的匹配预测占比',
    'monitor_depth_precision_gt_8_fraction': '深度置信精度大于8的匹配预测占比',
    'monitor_depth_target_fraction_lt_0_1m': '深度误差小于0.1米的匹配预测占比',
    'monitor_depth_target_fraction_0_1_to_0_5m': '深度误差0.1至0.5米的匹配预测占比',
    'monitor_depth_target_fraction_0_5_to_1m': '深度误差0.5至1米的匹配预测占比',
    'monitor_depth_target_fraction_ge_1m': '深度误差大于等于1米的匹配预测占比',
    'monitor_depth_local_gradient_energy_fraction_lt_0_1m': '深度误差小于0.1米的输出局部梯度能量占比',
    'monitor_depth_local_gradient_energy_fraction_0_1_to_0_5m': '深度误差0.1至0.5米的输出局部梯度能量占比',
    'monitor_depth_local_gradient_energy_fraction_0_5_to_1m': '深度误差0.5至1米的输出局部梯度能量占比',
    'monitor_depth_local_gradient_energy_fraction_ge_1m': '深度误差大于等于1米的输出局部梯度能量占比',
    'monitor_iou3d_matching_identity_change_fraction': (
        '三个三维Decoder层中匹配身份改变比例'),
    'monitor_iou3d_matching_mean_iou3d_gain': (
        '三个三维Decoder层中匹配后三维IoU平均增量'),
    'monitor_iou3d_matching_mean_giou2d_delta': (
        '三个三维Decoder层中匹配后二维GIoU平均变化'),
    'monitor_iou3d_matching_mean_gt_class_score_delta': (
        '三个三维Decoder层中匹配GT类别分数平均变化'),
    'monitor_iou3d_matching_current_mean_iou3d': (
        '三个三维Decoder层中当前匹配预测的平均三维IoU'),
    'monitor_iou3d_matching_current_mean_giou2d': (
        '三个三维Decoder层中当前匹配预测的平均二维GIoU'),
    'monitor_iou3d_matching_current_mean_gt_class_score': (
        '三个三维Decoder层中当前匹配预测的GT类别平均分数'),
    'monitor_iou3d_matching_best_iou3d_query_mean_gt_class_score': (
        '三个三维Decoder层中每个GT三维IoU最高预测的GT类别平均分数'),
    'monitor_high_iou_negative_downweighted_fraction': (
        '未匹配预测中负分类权重被降低的比例'),
    'monitor_high_iou_negative_ignored_fraction': (
        '未匹配预测中负分类权重降为零的比例'),
    'monitor_high_iou_negative_mean_weight': (
        '未匹配预测的负分类权重均值'),
    'monitor_high_iou_negative_mean_gt_class_score': (
        '三维IoU大于等于0.5未匹配预测的GT类别平均分数'),
    'monitor_high_iou_negative_ignored_mean_gt_class_score': (
        '三维IoU大于等于0.7未匹配预测的GT类别平均分数'),
    'monitor_quality_iou_mae': '质量头预测三维IoU绝对误差',
    'monitor_quality_target_iou_mean': '质量头监督目标三维IoU均值',
    'monitor_quality_predicted_iou_mean': '质量头预测三维IoU均值',
}

_IOU3D_MATCHING_DIAGNOSTICS = {
    'monitor_iou3d_matching_identity_change_fraction',
    'monitor_iou3d_matching_mean_iou3d_gain',
    'monitor_iou3d_matching_mean_giou2d_delta',
    'monitor_iou3d_matching_mean_gt_class_score_delta',
    'monitor_iou3d_matching_current_mean_iou3d',
    'monitor_iou3d_matching_current_mean_giou2d',
    'monitor_iou3d_matching_current_mean_gt_class_score',
    'monitor_iou3d_matching_best_iou3d_query_mean_gt_class_score',
}

_ONLINE_FINAL_DIAGNOSTICS = {
    'monitor_angle_classification',
    'monitor_angle_residual',
    'monitor_depth_mae',
    'monitor_depth_log_scale_mean',
    'monitor_depth_precision_mean',
    'monitor_depth_precision_p90',
    'monitor_depth_mae_precision_correlation',
    'monitor_depth_precision_gt_2_fraction',
    'monitor_depth_precision_gt_4_fraction',
    'monitor_depth_precision_gt_8_fraction',
    'monitor_depth_target_fraction_lt_0_1m',
    'monitor_depth_target_fraction_0_1_to_0_5m',
    'monitor_depth_target_fraction_0_5_to_1m',
    'monitor_depth_target_fraction_ge_1m',
    'monitor_depth_local_gradient_energy_fraction_lt_0_1m',
    'monitor_depth_local_gradient_energy_fraction_0_1_to_0_5m',
    'monitor_depth_local_gradient_energy_fraction_0_5_to_1m',
    'monitor_depth_local_gradient_energy_fraction_ge_1m',
    'monitor_high_iou_negative_downweighted_fraction',
    'monitor_high_iou_negative_ignored_fraction',
    'monitor_high_iou_negative_mean_weight',
    'monitor_high_iou_negative_mean_gt_class_score',
    'monitor_high_iou_negative_ignored_mean_gt_class_score',
    'monitor_quality_iou_mae',
    'monitor_quality_target_iou_mean',
    'monitor_quality_predicted_iou_mean',
}

_ONLINE_GROUP0_DIAGNOSTICS = {
    'monitor_angle_classification',
    'monitor_angle_residual',
    'monitor_depth_mae',
    'monitor_depth_log_scale_mean',
    'monitor_depth_precision_mean',
}

_ONLINE_CARDINALITY_METRICS = {
    'monitor_cardinality_gt_car_count',
    'monitor_cardinality_all_groups_predicted_car_count',
    'monitor_cardinality_all_groups_car_absolute_error',
    'monitor_cardinality_group0_predicted_car_count',
    'monitor_cardinality_group0_car_absolute_error',
}

CARDINALITY_CHINESE_NAMES = {
    'monitor_cardinality_gt_count': '每张图真实目标数',
    'monitor_cardinality_gt_car_count': '每张图真实车辆数',
    'monitor_cardinality_all_groups_predicted_count': (
        '全部查询组每图每组预测目标数'),
    'monitor_cardinality_all_groups_predicted_car_count': (
        '全部查询组每图每组预测车辆数'),
    'monitor_cardinality_all_groups_absolute_error': (
        '全部查询组预测目标数绝对误差'),
    'monitor_cardinality_all_groups_car_absolute_error': (
        '全部查询组预测车辆数绝对误差'),
    'monitor_cardinality_all_groups_signed_error': (
        '全部查询组预测目标数有符号误差'),
    'monitor_cardinality_all_groups_car_signed_error': (
        '全部查询组预测车辆数有符号误差'),
    'monitor_cardinality_group0_predicted_count': '第0查询组每图预测目标数',
    'monitor_cardinality_group0_predicted_car_count': '第0查询组每图预测车辆数',
    'monitor_cardinality_group0_absolute_error': '第0查询组预测目标数绝对误差',
    'monitor_cardinality_group0_car_absolute_error': (
        '第0查询组预测车辆数绝对误差'),
    'monitor_cardinality_group0_signed_error': '第0查询组预测目标数有符号误差',
    'monitor_cardinality_group0_car_signed_error': (
        '第0查询组预测车辆数有符号误差'),
}


def _split_layer_suffix(key):
    inter_match = re.search(r'_inter_(\d+)$', key)
    if inter_match:
        layer_index = int(inter_match.group(1))
        return key[:inter_match.start()], f'二维中间层第{layer_index + 1}层'
    auxiliary_match = re.search(r'_(\d+)$', key)
    if auxiliary_match:
        layer_index = int(auxiliary_match.group(1))
        return key[:auxiliary_match.start()], f'辅助Decoder第{layer_index + 1}层'
    return key, '最终Decoder层'


def chinese_grouped_monitoring(raw_losses, weight_dict, scope,
                               final_query_label):
    """Create non-overlapping, readable SwanLab metric groups."""
    result = {}
    final_query_keys = {
        'loss_ce', 'loss_bbox', 'loss_giou', 'loss_dim', 'loss_angle',
        'loss_depth', 'loss_center', 'loss_quality'}
    shared_keys = {'loss_depth_map', 'loss_region'}
    final_query_total = 0.0
    full_total = 0.0
    auxiliary_totals = {}
    intermediate_totals = {}

    for full_key, value in raw_losses.items():
        if full_key not in weight_dict:
            continue
        base_key, layer_name = _split_layer_suffix(full_key)
        if base_key not in LOSS_CHINESE_NAMES:
            continue
        weighted = value * float(weight_dict[full_key])
        full_total += weighted
        metric_name = f'{LOSS_CHINESE_NAMES[base_key]}（加权）'
        if layer_name == '最终Decoder层' and base_key in final_query_keys:
            result[f'{scope}最终查询损失/{metric_name}'] = weighted
            final_query_total += weighted
        elif layer_name == '最终Decoder层' and base_key in shared_keys:
            result[f'{scope}共享损失/{metric_name}'] = weighted
        elif layer_name.startswith('辅助Decoder'):
            auxiliary_totals[layer_name] = (
                auxiliary_totals.get(layer_name, 0.0) + weighted)
        elif layer_name.startswith('二维中间层'):
            intermediate_totals[layer_name] = (
                intermediate_totals.get(layer_name, 0.0) + weighted)

    result[f'{scope}核心概览/完整训练目标总损失'] = full_total
    result[
        f'{scope}核心概览/{final_query_label}最终查询损失合计'
    ] = final_query_total
    for layer_name, total in auxiliary_totals.items():
        result[
            f'{scope}辅助Decoder损失/{layer_name}/该层加权损失合计'
        ] = total
    for layer_name, total in intermediate_totals.items():
        result[
            f'{scope}二维中间层损失/{layer_name}/该层加权损失合计'
        ] = total

    for full_key, value in raw_losses.items():
        base_key, layer_name = _split_layer_suffix(full_key)
        if (base_key in _ONLINE_FINAL_DIAGNOSTICS
                and layer_name == '最终Decoder层'):
            if base_key.startswith('monitor_angle_'):
                weight_key = 'loss_angle'
            elif base_key.startswith('monitor_depth_'):
                weight_key = 'loss_depth'
            else:
                weight_key = None
            suffix = full_key[len(base_key):]
            coefficient = float(weight_dict.get(
                f'{weight_key}{suffix}', 1.0)) if weight_key else 1.0
            if base_key.startswith('monitor_angle_'):
                diagnostic_group = '航向角诊断'
            elif base_key.startswith('monitor_depth_'):
                diagnostic_group = '深度诊断'
            elif base_key.startswith('monitor_quality_'):
                diagnostic_group = '三维IoU质量头诊断'
            else:
                diagnostic_group = '高IoU未匹配预测诊断'
            result[f'{scope}{diagnostic_group}/{layer_name}/'
                   f'{LOSS_DIAGNOSTIC_CHINESE_NAMES[base_key]}'] = (
                       value * coefficient)
        elif base_key in _ONLINE_CARDINALITY_METRICS:
            result[
                f'{scope}预测数量诊断/{CARDINALITY_CHINESE_NAMES[base_key]}'
            ] = value
        elif base_key in _IOU3D_MATCHING_DIAGNOSTICS:
            result[
                f'{scope}三维IoU匹配诊断/'
                f'{LOSS_DIAGNOSTIC_CHINESE_NAMES[base_key]}'
            ] = value

    group0_prefix = 'monitor_group0_'
    group0_total = 0.0
    for full_key, value in raw_losses.items():
        if not full_key.startswith(group0_prefix):
            continue
        base_key = full_key[len(group0_prefix):]
        if base_key in LOSS_CHINESE_NAMES:
            coefficient = float(weight_dict.get(base_key, 1.0))
            name = f'{LOSS_CHINESE_NAMES[base_key]}（加权）'
            group0_total += value * coefficient
        elif base_key in _ONLINE_GROUP0_DIAGNOSTICS:
            coefficient = float(weight_dict.get(
                'loss_angle' if base_key.startswith('monitor_angle_')
                else 'loss_depth', 1.0))
            name = LOSS_DIAGNOSTIC_CHINESE_NAMES[base_key]
        else:
            continue
        result[f'{scope}第0查询组对照/{name}'] = value * coefficient
    if any(key.startswith(group0_prefix) for key in raw_losses):
        result[f'{scope}核心概览/第0查询组最终查询损失合计'] = (
            group0_total)
    return result


def scalar_values_to_floats(values):
    """Transfer a dictionary of scalar tensors to the host in one sync."""
    keys = [
        key for key, value in values.items()
        if torch.is_tensor(value) and value.numel() == 1
    ]
    if not keys:
        return {}
    host_values = torch.stack([
        values[key].detach().reshape(()) for key in keys
    ]).cpu().tolist()
    return dict(zip(keys, host_values))


def chinese_geometry_metrics(summary):
    return {
        GEOMETRY_INTERVAL_CHINESE_NAMES[key]: value
        for key, value in summary.items()
        if key in GEOMETRY_INTERVAL_CHINESE_NAMES
    }


class ScalarMeanAccumulator:
    """Accumulate scalar tensors on their device and synchronize at finalize."""

    def __init__(self):
        self._sums = {}
        self._counts = {}

    def add(self, values):
        for key, value in values.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            detached = value.detach().reshape(()).to(dtype=torch.float64)
            if key not in self._sums:
                self._sums[key] = torch.zeros_like(detached)
                self._counts[key] = 0
            self._sums[key].add_(detached)
            self._counts[key] += 1

    def finalize(self):
        return scalar_values_to_floats({
            key: total / self._counts[key]
            for key, total in self._sums.items()
        })


class GeometryIntervalAccumulator:
    """Aggregate the historical feasible-depth interval receipt."""

    _COUNT_KEYS = (
        'matched_count', 'unique_matched_car_count', 'eligible_car_count',
        'supported_at_gt_count', 'valid_interval_count',
        'fallback_native_count', 'inside_count', 'outside_count')
    _VECTOR_METRICS = {
        'depth_mae_all_car': 'absolute_error_virtual',
        'depth_mae_supported': 'supported_absolute_error_virtual',
        'depth_mae_unsupported': 'unsupported_absolute_error_virtual',
        'depth_mae_inside': 'inside_absolute_error_virtual',
        'depth_mae_outside': 'outside_absolute_error_virtual',
        'mean_outside_boundary_distance': (
            'outside_boundary_distance_virtual'),
    }

    def __init__(self):
        # Keep the small detached receipts on their producing device and
        # synchronize only once at epoch finalization.  Per-batch .cpu() calls
        # otherwise serialize the training stream hundreds of times per epoch.
        self.counts = {key: None for key in self._COUNT_KEYS}
        self.iou_sums = []
        self.iou_count = 0
        self.metric_sums = {
            key: [] for key in self._VECTOR_METRICS}
        self.metric_counts = {key: 0 for key in self._VECTOR_METRICS}
        self.widths = {'left_width_virtual': [], 'right_width_virtual': []}

    def add(self, receipt):
        if not receipt:
            return
        for key in self._COUNT_KEYS:
            value = receipt[key].detach()
            self.counts[key] = (
                value if self.counts[key] is None
                else self.counts[key] + value)
        iou = receipt['iou_at_gt'].detach().float().reshape(-1)
        if iou.numel():
            self.iou_sums.append(iou.sum())
            self.iou_count += iou.numel()
        for metric, field in self._VECTOR_METRICS.items():
            values = receipt[field].detach().float().reshape(-1)
            if values.numel():
                self.metric_sums[metric].append(values.sum())
                self.metric_counts[metric] += values.numel()
        for field in self.widths:
            values = receipt[field].detach().float().reshape(-1)
            if values.numel():
                self.widths[field].append(values)

    def finalize(self):
        counts = {
            key: (float(value.cpu()) if value is not None else 0.0)
            for key, value in self.counts.items()
        }
        eligible = counts['eligible_car_count']
        valid = counts['valid_interval_count']
        result = dict(counts)
        if self.iou_sums:
            iou_sums = torch.stack(tuple(self.iou_sums)).cpu().tolist()
            iou_sum = sum(iou_sums)
        else:
            iou_sum = 0.0
        result.update({
            'valid_interval_fraction': valid / eligible if eligible else 0.0,
            'outside_fraction': (
                counts['outside_count'] / valid if valid else 0.0),
            'mean_iou_at_gt': (
                iou_sum / self.iou_count if self.iou_count else 0.0),
        })
        for key, chunk_sums in self.metric_sums.items():
            if chunk_sums:
                host_sums = torch.stack(tuple(chunk_sums)).cpu().tolist()
                result[key] = (
                    sum(host_sums) / self.metric_counts[key])
            else:
                result[key] = 0.0
        left = (torch.cat(self.widths['left_width_virtual']).cpu()
                if self.widths['left_width_virtual'] else torch.empty(0))
        right = (torch.cat(self.widths['right_width_virtual']).cpu()
                 if self.widths['right_width_virtual'] else torch.empty(0))
        for name, values in (
                ('left_width_virtual', left),
                ('right_width_virtual', right),
                ('total_width_virtual', left + right)):
            quantiles = (torch.quantile(
                values, values.new_tensor((.1, .5, .9))).tolist()
                if values.numel() else (0.0, 0.0, 0.0))
            result[f'{name}_p10'] = float(quantiles[0])
            result[f'{name}_median'] = float(quantiles[1])
            result[f'{name}_p90'] = float(quantiles[2])
        return result


class SwanLabTracker:
    """Initialize and write SwanLab metrics without making it a hard dependency."""

    def __init__(self, cfg, run_config, output_dir, logger, model_name):
        self.logger = logger
        self.enabled = bool(cfg.get('enabled', False))
        self.required = bool(cfg.get('required', False))
        self.run = None
        self._swanlab = None
        if not self.enabled:
            return

        try:
            import swanlab

            self._swanlab = swanlab
            init_kwargs = {
                'project': str(cfg.get('project', 'MonoDGP')),
                'experiment_name': str(
                    cfg.get('experiment_name', model_name)),
                'description': cfg.get('description'),
                'group': cfg.get('group'),
                'tags': list(cfg.get('tags', ())),
                'config': run_config,
                'logdir': os.path.join(output_dir, 'swanlog'),
                'mode': str(cfg.get('mode', 'online')),
                'id': cfg.get('id'),
                'resume': cfg.get('resume'),
            }
            workspace = cfg.get('workspace')
            if workspace:
                init_kwargs['workspace'] = str(workspace)
            init_kwargs = {
                key: value for key, value in init_kwargs.items()
                if value is not None
            }
            self.run = swanlab.init(**init_kwargs)
            self.logger.info(
                'SWANLAB_INITIALIZED project=%s experiment=%s mode=%s id=%s',
                init_kwargs['project'], init_kwargs['experiment_name'],
                init_kwargs['mode'], getattr(self.run, 'id', None))
        except Exception as error:
            self.enabled = False
            if self.run is not None and self._swanlab is not None:
                try:
                    self._swanlab.finish()
                except Exception:
                    pass
                self.run = None
            if self.required:
                raise RuntimeError(
                    'SwanLab is required but initialization failed') from error
            self.logger.exception(
                'SwanLab initialization failed; continuing without tracking: %s',
                error)

    @property
    def active(self):
        return self.run is not None

    def log(self, values, step=None):
        if not self.active:
            return
        payload = {}
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().reshape(()).cpu().item()
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[str(key)] = value
        if not payload:
            return
        self.run.log(payload, step=step)

    def finish(self):
        if self.run is None:
            return
        self._swanlab.finish()
        self.logger.info('SWANLAB_FINISHED id=%s', getattr(self.run, 'id', None))
        self.run = None
