import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

def _correlations(score, target):
    finite = np.isfinite(score) & np.isfinite(target)
    score = score[finite]
    target = target[finite]
    if score.size < 2 or np.std(score) == 0 or np.std(target) == 0:
        return {'pearson': 0.0, 'spearman': 0.0}
    pearson = float(np.corrcoef(score, target)[0, 1])
    score_rank = rankdata(score, method='average')
    target_rank = rankdata(target, method='average')
    spearman = float(np.corrcoef(score_rank, target_rank)[0, 1])
    return {'pearson': pearson, 'spearman': spearman}


class QualityRankingAccumulator:
    """Measure whether each score ranks the one-to-one IoU oracle highly."""

    def __init__(self, score_specs, car_class_id=1):
        self.score_specs = tuple(score_specs)
        self.car_class_id = int(car_class_id)
        self.query_iou = []
        self.query_scores = {
            name: [] for name in ('classification', 'depth')
        }
        self.query_scores.update({spec['name']: [] for spec in score_specs})
        self.oracle_rows = []

    def add(self, outputs, iou3d_matrix, prepared_targets, info, dataset,
            raw_target_mask):
        if iou3d_matrix is None:
            raise RuntimeError('quality ranking monitor requires exact 3D IoU')
        classification = outputs['pred_logits'].sigmoid()[
            ..., self.car_class_id].detach().cpu().numpy()
        depth = np.exp(-outputs['pred_depth'][
            ..., 1].detach().cpu().numpy())
        quality = None
        if 'pred_quality' in outputs:
            quality = (((outputs['pred_quality'] + 1.0) * 0.5)
                       .clamp(0, 1)[..., 0].detach().cpu().numpy())
            self.query_scores.setdefault('quality', [])
        iou3d = iou3d_matrix.detach().cpu().numpy()
        fused = {}
        for spec in self.score_specs:
            if spec.get('historical_topk', False):
                fused[spec['name']] = classification * depth
            elif spec.get('classification_only', False):
                fused[spec['name']] = classification
            else:
                if quality is None:
                    raise RuntimeError(
                        'quality score fusion requires pred_quality')
                fused[spec['name']] = (
                    np.maximum(classification, 1e-12) ** spec['alpha']
                    * np.maximum(quality, 1e-12) ** spec['beta']
                    * np.maximum(depth, 1e-12) ** spec['gamma'])

        image_ids = info['img_id'].detach().cpu().numpy().tolist()
        masks = raw_target_mask.detach().cpu().numpy().astype(bool)
        for batch_index, (target, image_id) in enumerate(
                zip(prepared_targets, image_ids)):
            target_count = int(target['labels'].numel())
            if target_count == 0:
                self.query_iou.append(np.zeros_like(
                    classification[batch_index]))
                self.query_scores['classification'].append(
                    classification[batch_index])
                if quality is not None:
                    self.query_scores['quality'].append(
                        quality[batch_index])
                self.query_scores['depth'].append(depth[batch_index])
                for name, values in fused.items():
                    self.query_scores[name].append(values[batch_index])
                continue
            image_iou = iou3d[batch_index, :, :target_count]
            max_iou = image_iou.max(axis=1)
            self.query_iou.append(max_iou)
            self.query_scores['classification'].append(
                classification[batch_index])
            if quality is not None:
                self.query_scores['quality'].append(
                    quality[batch_index])
            self.query_scores['depth'].append(depth[batch_index])
            for name, values in fused.items():
                self.query_scores[name].append(values[batch_index])

            labels = target['labels'].detach().cpu().numpy()
            car_targets = np.flatnonzero(labels == self.car_class_id)
            if car_targets.size == 0:
                continue
            car_iou = image_iou[:, car_targets]
            query_index, local_target_index = linear_sum_assignment(-car_iou)
            oracle_for_target = {
                int(target_index): int(query)
                for query, target_index in zip(
                    query_index, local_target_index)
            }
            objects = dataset.get_label(int(image_id))
            selected_objects = [
                objects[index] for index in np.flatnonzero(masks[batch_index])
                if index < len(objects)]
            if len(selected_objects) != target_count:
                raise ValueError(
                    'KITTI metadata does not match prepared targets')
            for local_index, target_index in enumerate(car_targets):
                if local_index not in oracle_for_target:
                    continue
                one_to_one_query = oracle_for_target[local_index]
                best_query = int(np.argmax(car_iou[:, local_index]))
                best_iou = float(car_iou[best_query, local_index])
                metadata = selected_objects[target_index]
                row = {
                    'difficulty': metadata.level_str.lower(),
                    'distance': float(metadata.pos[-1]),
                    'occlusion': int(metadata.occlusion),
                    'one_to_one_query': one_to_one_query,
                    'one_to_one_iou': float(
                        car_iou[one_to_one_query, local_index]),
                    'best_query': best_query,
                    'best_iou': best_iou,
                    'target_iou': car_iou[:, local_index],
                    'scores': {
                        name: values[batch_index]
                        for name, values in {
                            'classification': classification,
                            'depth': depth,
                            **({'quality': quality}
                               if quality is not None else {}),
                            **fused,
                        }.items()
                    },
                }
                self.oracle_rows.append(row)

    @staticmethod
    def _groups(row):
        distance = row['distance']
        distance_group = ('distance_lt20' if distance < 20 else
                          'distance_20_40' if distance < 40 else
                          'distance_ge40')
        return ('all', f"difficulty_{row['difficulty']}", distance_group,
                f"occlusion_{row['occlusion']}")

    def finalize(self):
        if not self.query_iou:
            return {}
        target = np.concatenate(self.query_iou)
        summary = {'query_correlation': {}}
        for name, chunks in self.query_scores.items():
            summary['query_correlation'][name] = _correlations(
                np.concatenate(chunks), target)

        grouped = {}
        for row in self.oracle_rows:
            for group in self._groups(row):
                grouped.setdefault(group, []).append(row)
        summary['one_to_one_oracle'] = {}
        for group, rows in grouped.items():
            group_summary = {}
            for score_name in self.query_scores:
                best_ranks = []
                one_to_one_ranks = []
                identities = []
                top3_identities = []
                regrets = []
                best_ious = []
                selected_top1_ious = []
                ordered_pair_count = 0
                ordered_pair_correct = 0
                for row in rows:
                    scores = row['scores'][score_name]
                    order = np.argsort(-scores, kind='stable')
                    best_query = row['best_query']
                    one_to_one_query = row['one_to_one_query']
                    best_ranks.append(
                        int(np.flatnonzero(order == best_query)[0]) + 1)
                    one_to_one_ranks.append(
                        int(np.flatnonzero(
                            order == one_to_one_query)[0]) + 1)
                    identities.append(int(order[0] == best_query))
                    top3_identities.append(int(best_query in order[:3]))
                    target_iou = row['target_iou']
                    selected_top1_ious.append(float(target_iou[order[0]]))
                    regrets.append(float(
                        row['best_iou'] - target_iou[order[0]]))
                    best_ious.append(row['best_iou'])
                    iou_difference = (
                        target_iou[:, None] - target_iou[None, :])
                    valid_pair = (
                        np.triu(np.ones_like(
                            iou_difference, dtype=bool), k=1)
                        & (np.abs(iou_difference) >= 0.1))
                    if np.any(valid_pair):
                        score_difference = (
                            scores[:, None] - scores[None, :])
                        signed_difference = (
                            np.sign(iou_difference[valid_pair])
                            * score_difference[valid_pair])
                        ordered_pair_count += int(valid_pair.sum())
                        ordered_pair_correct += int(
                            np.count_nonzero(signed_difference > 0))
                best_ranks = np.asarray(best_ranks)
                one_to_one_ranks = np.asarray(one_to_one_ranks)
                best_ious = np.asarray(best_ious)
                selected_top1_ious = np.asarray(selected_top1_ious)
                high_quality = best_ious >= 0.7
                high_quality_count = int(high_quality.sum())
                high_quality_ranks = best_ranks[high_quality]
                group_summary[score_name] = {
                    'count': int(best_ranks.size),
                    'best_iou_mean': float(best_ious.mean()),
                    'best_query_rank_median': float(
                        np.median(best_ranks)),
                    'best_query_rank_p90': float(
                        np.quantile(best_ranks, 0.9)),
                    'one_to_one_query_rank_median': float(
                        np.median(one_to_one_ranks)),
                    'one_to_one_query_rank_p90': float(
                        np.quantile(one_to_one_ranks, 0.9)),
                    'top1_identity_fraction': float(np.mean(identities)),
                    'top3_identity_fraction': float(
                        np.mean(top3_identities)),
                    'top1_iou_regret': float(np.mean(regrets)),
                    'pairwise_order_accuracy_gap_ge_0_1': (
                        float(ordered_pair_correct / ordered_pair_count)
                        if ordered_pair_count else 0.0),
                    'pairwise_order_pair_count_gap_ge_0_1': (
                        ordered_pair_count),
                    'high_quality_count': high_quality_count,
                    'high_quality_top1_iou_ge_0_7_fraction': (
                        float(np.mean(
                            selected_top1_ious[high_quality] >= 0.7))
                        if high_quality_count else 0.0),
                    'high_quality_best_top1_recall': (
                        float(np.mean(high_quality_ranks <= 1))
                        if high_quality_count else 0.0),
                    'high_quality_best_top3_recall': (
                        float(np.mean(high_quality_ranks <= 3))
                        if high_quality_count else 0.0),
                    'high_quality_best_top5_recall': (
                        float(np.mean(high_quality_ranks <= 5))
                        if high_quality_count else 0.0),
                }
            summary['one_to_one_oracle'][group] = group_summary
        return summary
