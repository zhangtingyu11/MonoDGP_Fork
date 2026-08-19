"""Read-only gradient diagnostics for training observability."""

from __future__ import annotations

import torch


MODULE_CHINESE_NAMES = {
    'iou_quality_head': '三维IoU质量头',
    'prediction_heads': '预测头',
    'backbone': '骨干网络',
    'depth_predictor': '深度预测器',
    'det2d_transformer': '二维Transformer',
    'det3d_transformer': '三维Transformer',
    'feature_and_region_heads': '特征投影与区域头',
    'other': '其他参数',
}


def _unique_trainable_parameters(modules, claimed):
    parameters = []
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            identity = id(parameter)
            if parameter.requires_grad and identity not in claimed:
                claimed.add(identity)
                parameters.append(parameter)
    return parameters


def _l2_norm(tensors, fallback):
    values = [tensor.detach() for tensor in tensors]
    if not values:
        return fallback.new_zeros(())
    per_tensor = torch._foreach_norm(values, 2.0)
    return torch.linalg.vector_norm(torch.stack(tuple(
        value.float() for value in per_tensor)), ord=2)


class GradientMonitor:
    """Collect global norms every batch and module snapshots periodically."""

    def __init__(self, model, module_interval=30):
        model = model.module if hasattr(model, 'module') else model
        self.module_interval = max(1, int(module_interval))
        claimed = set()
        prediction_modules = [getattr(model, name, None) for name in (
            'class_embed', 'bbox_embed', 'dim_embed_3d', 'angle_embed',
            'depth_embed', 'query_embed')]
        specifications = (
            ('iou_quality_head', [
                getattr(model, 'iou_quality_embed', None)]),
            ('prediction_heads', prediction_modules),
            ('backbone', [getattr(model, 'backbone', None)]),
            ('depth_predictor', [getattr(model, 'depth_predictor', None)]),
            ('det2d_transformer', [
                getattr(model, 'det2d_transformer', None)]),
            ('det3d_transformer', [
                getattr(model, 'det3d_transformer', None)]),
            ('feature_and_region_heads', [
                getattr(model, 'input_proj', None),
                getattr(model, 'region_head', None)]),
        )
        self.module_parameters = {
            key: _unique_trainable_parameters(modules, claimed)
            for key, modules in specifications}
        other_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in claimed]
        if other_parameters:
            self.module_parameters['other'] = other_parameters
        self.module_parameters = {
            key: parameters
            for key, parameters in self.module_parameters.items()
            if parameters}
        self.parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad]
        self.fallback = next(model.parameters()).detach()
        self.global_norms = []
        self.module_grad_norms = {
            key: [] for key in self.module_parameters}
        self.module_ratios = {
            key: [] for key in self.module_parameters}
        self.missing_gradient_tensor_counts = []

    @staticmethod
    def _gradients(parameters):
        return [
            parameter.grad for parameter in parameters
            if parameter.grad is not None]

    def observe(self, batch_index, is_last_batch=False):
        global_norm = _l2_norm(
            self._gradients(self.parameters), self.fallback)
        self.global_norms.append(global_norm.detach())
        current = {
            'global_l2_norm': global_norm.detach(),
            'global_gradient_is_finite': torch.isfinite(
                global_norm).to(dtype=torch.int64),
        }
        take_module_snapshot = (
            batch_index % self.module_interval == 0 or is_last_batch)
        if not take_module_snapshot:
            return current

        missing = sum(
            parameter.grad is None for parameter in self.parameters)
        missing_tensor = torch.as_tensor(
            missing, device=global_norm.device, dtype=torch.int64)
        self.missing_gradient_tensor_counts.append(missing_tensor)
        current['missing_gradient_tensor_count'] = missing_tensor
        for key, parameters in self.module_parameters.items():
            gradients = self._gradients(parameters)
            grad_norm = _l2_norm(gradients, self.fallback)
            parameter_norm = _l2_norm(parameters, self.fallback)
            ratio = grad_norm / parameter_norm.clamp_min(
                torch.finfo(parameter_norm.dtype).eps)
            self.module_grad_norms[key].append(grad_norm.detach())
            self.module_ratios[key].append(ratio.detach())
            current[f'{key}_grad_l2_norm'] = grad_norm.detach()
            current[f'{key}_grad_to_parameter_ratio'] = ratio.detach()
        return current

    def finalize(self):
        norms = (torch.stack(tuple(self.global_norms)).float()
                 if self.global_norms
                 else self.fallback.new_empty((0,), dtype=torch.float32))
        finite = torch.isfinite(norms)
        finite_norms = norms[finite]
        result = {
            'nonfinite_batch_count': int((~finite).sum().cpu()),
            'observed_batch_count': int(norms.numel()),
        }
        if finite_norms.numel():
            quantiles = torch.quantile(
                finite_norms, finite_norms.new_tensor((.5, .95)))
            scalars = torch.stack((
                finite_norms.min(), quantiles[0], quantiles[1],
                finite_norms.max())).cpu().tolist()
            result.update({
                'global_l2_norm_min': scalars[0],
                'global_l2_norm_median': scalars[1],
                'global_l2_norm_p95': scalars[2],
                'global_l2_norm_max': scalars[3],
            })
        else:
            result.update({
                'global_l2_norm_min': float('nan'),
                'global_l2_norm_median': float('nan'),
                'global_l2_norm_p95': float('nan'),
                'global_l2_norm_max': float('nan'),
            })
        result['missing_gradient_tensor_count_max'] = (
            int(torch.stack(tuple(
                self.missing_gradient_tensor_counts)).max().cpu())
            if self.missing_gradient_tensor_counts else 0)
        for key in self.module_parameters:
            grad_norms = self.module_grad_norms[key]
            ratios = self.module_ratios[key]
            result[f'{key}_grad_l2_norm_mean'] = (
                float(torch.stack(tuple(grad_norms)).mean().cpu())
                if grad_norms else 0.0)
            result[f'{key}_grad_to_parameter_ratio_mean'] = (
                float(torch.stack(tuple(ratios)).mean().cpu())
                if ratios else 0.0)
        return result


def chinese_gradient_metrics(metrics):
    names = {
        'global_l2_norm': '全模型梯度L2范数',
        'global_gradient_is_finite': '当前批全模型梯度是否有限（1为正常）',
        'global_l2_norm_min': '全模型梯度L2范数最小值',
        'global_l2_norm_median': '全模型梯度L2范数中位数',
        'global_l2_norm_p95': '全模型梯度L2范数P95',
        'global_l2_norm_max': '全模型梯度L2范数最大值',
        'nonfinite_batch_count': '出现NaN或Inf梯度的批次数',
        'observed_batch_count': '已检查梯度的批次数',
        'missing_gradient_tensor_count': '当前批无梯度参数张量数',
        'missing_gradient_tensor_count_max': '单批无梯度参数张量数最大值',
    }
    for key, chinese in MODULE_CHINESE_NAMES.items():
        names[f'{key}_grad_l2_norm'] = f'{chinese}梯度L2范数'
        names[f'{key}_grad_to_parameter_ratio'] = (
            f'{chinese}梯度与参数范数比')
        names[f'{key}_grad_l2_norm_mean'] = (
            f'{chinese}梯度L2范数均值')
        names[f'{key}_grad_to_parameter_ratio_mean'] = (
            f'{chinese}梯度与参数范数比均值')
    return {
        names.get(key, key): value for key, value in metrics.items()
    }
