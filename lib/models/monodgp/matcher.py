# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy


class QuerySelection(nn.Module):
    def __init__(self, group_num_all=11, queries_all=50, group_num_select=6, queries_select=50):
        """
        Initializes the QuerySelection module.
        
        Parameters:
            group_num_all: Total number of groups in the input queries (e.g., 11).
            queries_all: Number of queries in each group (e.g., 50).
            group_num_select: Number of groups to select (e.g., 6).
            queries_select: Number of queries to select in each selected group (e.g., 50).
        """
        super(QuerySelection, self).__init__()
        self.group_num_all = group_num_all
        self.queries_all = queries_all
        self.group_num_select = group_num_select
        self.queries_select = queries_select

    def forward(self, outputs, targets, C):
        """
        Performs the query selection based on the cost matrix C.
        
        Parameters:
            C: Cost matrix of shape [batch_size, total_queries, total_targets], 
               where total_queries = group_num_all * queries_all.
        
        Returns:
            selected_queries: Tensor containing indices of the selected queries.
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        total_queries = num_queries
        assert total_queries == self.group_num_all * self.queries_all, "Mismatch in total number of queries"

        # Step 1: Reshape or split the cost matrix into groups
        C_grouped = C.view(bs, self.group_num_all, self.queries_all, -1)  # [batch_size, group_num_all, queries_all, total_targets]

        # Step 2: Select the groups with the lowest costs
        # Compute some form of group-wise cost (e.g., sum or mean over the queries and targets)
        group_costs = C_grouped.mean(dim=(2, 3))  # [batch_size, group_num_all]

        # Step 3: Identify the indices of the top `group_num_select` groups with the lowest costs
        _, top_group_indices = torch.topk(group_costs, self.group_num_select, dim=1, largest=False)

        # Step 4: Gather the corresponding queries from these groups
        selected_queries = []
        for batch_idx in range(bs):
            top_groups = top_group_indices[batch_idx]
            queries_from_selected_groups = [C_grouped[batch_idx, g] for g in top_groups]
            selected_queries_batch = torch.cat(queries_from_selected_groups, dim=0)  # [group_num_select * queries_select, total_targets]
            selected_queries.append(selected_queries_batch)

        selected_queries = torch.stack(selected_queries)  # [batch_size, group_num_select * queries_select, total_targets]

        return selected_queries
    
    
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1,
                 cost_bbox: float = 1, cost_giou: float = 1,
                 use_batched_same_image_cost=False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_batched_same_image_cost = bool(
            use_batched_same_image_cost)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def prepare_targets(self, targets):
        """Pack unchanged targets once for all matcher layers."""
        if not targets:
            raise ValueError("matcher requires a non-empty batch")
        sizes = tuple(int(len(target["boxes"])) for target in targets)
        for batch_index, (target, size) in enumerate(zip(targets, sizes)):
            if (len(target["labels"]) != size
                    or len(target["boxes_3d"]) != size):
                raise ValueError(
                    f"target {batch_index} matcher fields disagree in length")
        labels = torch.nn.utils.rnn.pad_sequence(
            [target["labels"].long() for target in targets],
            batch_first=True, padding_value=0)
        boxes_3d = torch.nn.utils.rnn.pad_sequence(
            [target["boxes_3d"] for target in targets],
            batch_first=True, padding_value=0.0)
        return {
            "sizes": sizes,
            "labels": labels,
            "boxes_3d": boxes_3d,
            "boxes_xyxy": box_cxcylrtb_to_xyxy(boxes_3d),
        }

    @staticmethod
    def _batched_generalized_box_iou(boxes1, boxes2):
        """Pairwise GIoU within each image: [B,N,4] x [B,M,4]."""
        assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
        assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
        area1 = ((boxes1[..., 2] - boxes1[..., 0])
                 * (boxes1[..., 3] - boxes1[..., 1]))
        area2 = ((boxes2[..., 2] - boxes2[..., 0])
                 * (boxes2[..., 3] - boxes2[..., 1]))
        lt = torch.maximum(
            boxes1[:, :, None, :2], boxes2[:, None, :, :2])
        rb = torch.minimum(
            boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[..., 0] * wh[..., 1]
        union = area1[:, :, None] + area2[:, None, :] - intersection
        iou = intersection / union
        enclosing_lt = torch.minimum(
            boxes1[:, :, None, :2], boxes2[:, None, :, :2])
        enclosing_rb = torch.maximum(
            boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
        enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
        enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
        return iou - (enclosing_area - union) / enclosing_area

    def _forward_batched(self, outputs, targets, group_num,
                         prepared_targets=None):
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        if prepared_targets is None:
            prepared_targets = self.prepare_targets(targets)
        if len(prepared_targets["sizes"]) != bs:
            raise ValueError("prepared matcher target batch size changed")
        labels = prepared_targets["labels"]
        target_boxes = prepared_targets["boxes_3d"]
        target_xyxy = prepared_targets["boxes_xyxy"]
        if (labels.device != outputs["pred_logits"].device
                or target_boxes.device != outputs["pred_boxes"].device):
            raise ValueError("prepared matcher targets are on the wrong device")

        out_prob = outputs["pred_logits"].sigmoid()
        alpha = 0.25
        gamma = 2.0
        neg_cost = ((1 - alpha) * out_prob.pow(gamma)
                    * (-(1 - out_prob + 1e-8).log()))
        pos_cost = (alpha * (1 - out_prob).pow(gamma)
                    * (-(out_prob + 1e-8).log()))
        gather_index = labels[:, None, :].expand(
            bs, num_queries, labels.shape[1])
        cost_class = torch.gather(pos_cost - neg_cost, 2, gather_index)
        cost_3dcenter = torch.cdist(
            outputs["pred_boxes"][..., :2], target_boxes[..., :2], p=1)
        cost_bbox = torch.cdist(
            outputs["pred_boxes"][..., 2:6], target_boxes[..., 2:6], p=1)
        prediction_xyxy = box_cxcylrtb_to_xyxy(outputs["pred_boxes"])
        cost_giou = -self._batched_generalized_box_iou(
            prediction_xyxy, target_xyxy)
        cost = (self.cost_bbox * cost_bbox
                + self.cost_3dcenter * cost_3dcenter
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou)
        cost = torch.nan_to_num(
            cost, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()

        queries_per_group = num_queries // group_num
        if queries_per_group * group_num != num_queries:
            raise ValueError("query count is not divisible by group count")
        indices = []
        for batch_index, target_count in enumerate(
                prepared_targets["sizes"]):
            source_parts = []
            target_parts = []
            for group_index in range(group_num):
                begin = group_index * queries_per_group
                end = begin + queries_per_group
                source, target = linear_sum_assignment(
                    cost[batch_index, begin:end, :target_count])
                source_parts.append(source + begin)
                target_parts.append(target)
            source = np.concatenate(source_parts)
            target = np.concatenate(target_parts)
            indices.append((torch.as_tensor(source, dtype=torch.int64),
                            torch.as_tensor(target, dtype=torch.int64)))
        return indices

    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11,
                prepared_targets=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.use_batched_same_image_cost:
            return self._forward_batched(
                outputs, targets, group_num, prepared_targets)

        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        sizes = [len(v["boxes"]) for v in targets]
        #indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(
        cost_class=cfg['set_cost_class'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_giou=cfg['set_cost_giou'],
        use_batched_same_image_cost=cfg.get(
            'use_batched_same_image_matcher_cost', False))
