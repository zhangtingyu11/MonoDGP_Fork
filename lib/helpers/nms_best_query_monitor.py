"""Validation-set monitor for the exact greedy NMS fate of each GT best query."""

from __future__ import annotations

import numpy as np
import torch

from lib.losses.nms_aware_iou_ranking_loss import (
    _assign_queries_to_targets,
    _triggered_nms_pairs_batched,
)


class NMSBestQueryAccumulator:
    """Aggregate counts over the full validation set, never over batch means."""

    def __init__(self, decode_mean_sizes, bev_iou_threshold=0.8,
                 min_iou_delta=1e-6):
        self.decode_mean_sizes = decode_mean_sizes
        self.bev_iou_threshold = float(bev_iou_threshold)
        self.min_iou_delta = float(min_iou_delta)
        if not 0.0 <= self.bev_iou_threshold <= 1.0:
            raise ValueError('NMS monitor threshold must be in [0, 1]')
        self.gt_count = 0
        self.best_query_retained_count = 0
        self.best_query_suppressed_count = 0
        self.best_query_suppressed_by_worse_count = 0
        self.best_iou_sum = 0.0
        self.kept_best_iou_sum = 0.0
        self.iou_regret_sum = 0.0
        self.suppression_iou_gap_sum = 0.0

    @staticmethod
    def _greedy_keep(scores, overlap, candidates):
        order = sorted(
            (int(index) for index in candidates),
            key=lambda index: float(scores[index]), reverse=True)
        kept = []
        suppressed_by = {}
        while order:
            current = order.pop(0)
            kept.append(current)
            remaining = []
            for candidate in order:
                if overlap[current, candidate]:
                    suppressed_by[candidate] = current
                else:
                    remaining.append(candidate)
            order = remaining
        return kept, suppressed_by

    @torch.no_grad()
    def add(self, outputs, iou3d_matrix, targets):
        if iou3d_matrix is None:
            raise RuntimeError('NMS best-query monitor requires exact 3D IoU')
        target_counts, _, assigned_target, assigned_labels = (
            _assign_queries_to_targets(iou3d_matrix, targets))
        batch, first, second, _ = _triggered_nms_pairs_batched(
            outputs, targets, assigned_target, assigned_labels,
            target_counts,
            decode_mean_sizes=self.decode_mean_sizes,
            group_num=1,
            bev_iou_threshold=self.bev_iou_threshold,
            require_same_target=False,
            strict_threshold=True)

        logits = outputs['pred_logits'].detach().cpu().numpy()
        iou = iou3d_matrix.detach().clamp(0, 1).cpu().numpy()
        labels = assigned_labels.detach().cpu().numpy()
        edge_batch = batch.detach().cpu().numpy()
        edge_first = first.detach().cpu().numpy()
        edge_second = second.detach().cpu().numpy()
        query_count = logits.shape[1]
        overlaps = np.zeros(
            (logits.shape[0], query_count, query_count), dtype=bool)
        overlaps[edge_batch, edge_first, edge_second] = True
        overlaps[edge_batch, edge_second, edge_first] = True

        for batch_index, target in enumerate(targets):
            target_labels = target['labels'].detach().cpu().numpy()
            for target_index, label in enumerate(target_labels):
                candidates = np.flatnonzero(labels[batch_index] == label)
                if candidates.size == 0:
                    continue
                target_iou = iou[batch_index, :, target_index]
                best_query = int(candidates[
                    np.argmax(target_iou[candidates])])
                kept, suppressed_by = self._greedy_keep(
                    logits[batch_index, :, int(label)],
                    overlaps[batch_index], candidates)
                kept = np.asarray(kept, dtype=np.int64)
                best_iou = float(target_iou[best_query])
                kept_best_iou = float(target_iou[kept].max())
                regret = max(0.0, best_iou - kept_best_iou)

                self.gt_count += 1
                self.best_iou_sum += best_iou
                self.kept_best_iou_sum += kept_best_iou
                self.iou_regret_sum += regret
                if best_query in suppressed_by:
                    self.best_query_suppressed_count += 1
                    suppressor = suppressed_by[best_query]
                    suppression_gap = max(
                        0.0, best_iou - float(target_iou[suppressor]))
                    self.suppression_iou_gap_sum += suppression_gap
                    if suppression_gap > self.min_iou_delta:
                        self.best_query_suppressed_by_worse_count += 1
                else:
                    self.best_query_retained_count += 1

    def finalize(self):
        count = self.gt_count
        suppressed = self.best_query_suppressed_count
        return {
            'gt_count': count,
            'best_query_retained_count': self.best_query_retained_count,
            'best_query_retained_fraction': (
                self.best_query_retained_count / count if count else 0.0),
            'best_query_suppressed_count': suppressed,
            'best_query_suppressed_fraction': (
                suppressed / count if count else 0.0),
            'best_query_suppressed_by_worse_count': (
                self.best_query_suppressed_by_worse_count),
            'best_query_suppressed_by_worse_fraction': (
                self.best_query_suppressed_by_worse_count / count
                if count else 0.0),
            'best_iou_mean': (
                self.best_iou_sum / count if count else 0.0),
            'kept_best_iou_mean': (
                self.kept_best_iou_sum / count if count else 0.0),
            'nms_iou_regret_mean': (
                self.iou_regret_sum / count if count else 0.0),
            'suppressed_iou_gap_mean': (
                self.suppression_iou_gap_sum / suppressed
                if suppressed else 0.0),
        }
