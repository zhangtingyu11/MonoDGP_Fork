import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .det2d_transformer import build_det2d_transformer
from .det3d_transformer import build_det3d_transformer

from .region_seg_head import RegionSegHead
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import quality_focal_loss, sigmoid_focal_loss
from lib.losses.asymmetric_interval_depth_loss import (
    asymmetric_interval_and_uncertainty_loss)
from lib.losses.query_quality_ranking_loss import (
    all_query_quality_ranking_loss)
from lib.losses.nms_aware_iou_ranking_loss import (
    nms_aware_iou_ranking_loss)
from .position_encoding import PositionEmbeddingCamRay


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class _ClipDepthMeanGradients(torch.autograd.Function):
    """Identity forward that clips only the predicted mean-depth gradient."""

    @staticmethod
    def forward(ctx, values, max_norm, receipt_sink):
        ctx.max_norm = float(max_norm)
        ctx.receipt_sink = receipt_sink
        return values

    @staticmethod
    def backward(ctx, gradients):
        float_gradients = gradients.float()
        mean_gradients = float_gradients[..., :1]
        absolute_mean_gradients = mean_gradients.abs()
        finite_pairs = torch.isfinite(float_gradients).all(
            dim=-1, keepdim=True)
        supervised = finite_pairs & (float_gradients != 0).any(
            dim=-1, keepdim=True)
        finite_mean = torch.isfinite(absolute_mean_gradients)
        mean_scales = torch.where(
            finite_mean,
            (ctx.max_norm / absolute_mean_gradients.clamp_min(1e-12))
            .clamp_max(1.0),
            torch.ones_like(absolute_mean_gradients))
        clipped = finite_mean & (absolute_mean_gradients > ctx.max_norm)
        pre_clip_energy = torch.where(
            finite_mean, mean_gradients.square(),
            torch.zeros_like(mean_gradients)).sum()
        post_clip_energy = torch.where(
            finite_mean, (mean_gradients * mean_scales).square(),
            torch.zeros_like(mean_gradients)).sum()
        ctx.receipt_sink.append({
            'prediction_count': supervised.sum().detach(),
            'clipped_count': clipped.sum().detach(),
            'pre_clip_max_absolute_gradient': torch.where(
                finite_mean, absolute_mean_gradients,
                torch.zeros_like(absolute_mean_gradients)).max().detach(),
            'minimum_scale': torch.where(
                clipped, mean_scales,
                torch.ones_like(mean_scales)).min().detach(),
            'pre_clip_energy': pre_clip_energy.detach(),
            'post_clip_energy': post_clip_energy.detach(),
        })
        channel_scales = torch.cat((
            mean_scales, torch.ones_like(mean_scales)), dim=-1)
        return (gradients * channel_scales.to(dtype=gradients.dtype),
                None, None)


def clip_depth_mean_gradients(values, max_norm, receipt_sink=None):
    """Clip ``depth_mean`` gradients while leaving ``log_scale`` untouched."""
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError('depth mean gradient max_norm must be positive')
    if values.shape[-1] != 2:
        raise ValueError('depth output must have exactly two channels')
    if receipt_sink is None:
        receipt_sink = []
    return _ClipDepthMeanGradients.apply(values, max_norm, receipt_sink)


_POST_MATCH_TARGET_FIELDS = {
    'labels': ('labels',),
    'boxes': ('boxes_3d',),
    'center': ('boxes_3d',),
    'depths': ('depth',),
    'dims': ('size_3d',),
    'angles': ('heading_bin', 'heading_res'),
}


def build_post_match_cache(outputs, targets, indices, active_losses):
    """Gather one layer's matched indices and target fields exactly once."""
    if 'pred_boxes' not in outputs:
        raise KeyError("post-match cache requires pred_boxes")
    batch_size, query_count = outputs['pred_boxes'].shape[:2]
    if len(targets) != batch_size or len(indices) != batch_size:
        raise ValueError("post-match cache batch size mismatch")

    requested_fields = []
    for loss_name in active_losses:
        for field in _POST_MATCH_TARGET_FIELDS.get(loss_name, ()):
            if field not in requested_fields:
                requested_fields.append(field)

    source_parts = []
    target_parts = []
    batch_parts = []
    target_offset = 0
    for batch_index, (target, pair) in enumerate(zip(targets, indices)):
        if len(pair) != 2:
            raise ValueError(
                "matcher entry must contain source and target indices")
        source_index = torch.as_tensor(
            pair[0], dtype=torch.int64, device='cpu')
        target_index = torch.as_tensor(
            pair[1], dtype=torch.int64, device='cpu')
        if source_index.ndim != 1 or target_index.ndim != 1:
            raise ValueError("matcher indices must be one-dimensional")
        if source_index.numel() != target_index.numel():
            raise ValueError("matcher source/target index length mismatch")

        target_count = None
        for field in requested_fields:
            value = target[field]
            if value.device != outputs['pred_boxes'].device:
                raise ValueError(f"target field {field} is on the wrong device")
            if value.ndim == 0:
                raise ValueError(
                    f"target field {field} must have a leading dimension")
            if target_count is None:
                target_count = int(value.shape[0])
            elif int(value.shape[0]) != target_count:
                raise ValueError(
                    "target fields have inconsistent leading dimensions")
        if target_count is None:
            target_count = int(target['labels'].shape[0])

        if source_index.numel():
            if (int(source_index.min()) < 0
                    or int(source_index.max()) >= query_count):
                raise IndexError("matcher source index is out of range")
            if (int(target_index.min()) < 0
                    or int(target_index.max()) >= target_count):
                raise IndexError("matcher target index is out of range")
        source_parts.append(source_index)
        target_parts.append(target_index + target_offset)
        batch_parts.append(torch.full_like(source_index, batch_index))
        target_offset += target_count

    batch_index = torch.cat(batch_parts, dim=0)
    source_index = torch.cat(source_parts, dim=0)
    global_target_index = torch.cat(target_parts, dim=0)
    packed_indices = torch.stack(
        (batch_index, source_index, global_target_index), dim=0).to(
            device=outputs['pred_boxes'].device)

    matched_targets = {}
    for field in requested_fields:
        flat_field = torch.cat([target[field] for target in targets], dim=0)
        matched_targets[field] = flat_field.index_select(
            0, packed_indices[2])
    return {
        'source_index': (packed_indices[0], packed_indices[1]),
        'matched_targets': matched_targets,
        'matched_count': int(packed_indices.shape[1]),
        'fields': tuple(requested_fields),
    }


class MonoDGP(nn.Module):
    """ This is the MonoDGP module that performs monocualr 3D object detection """
    def __init__(self, backbone, depth_predictor, det2d_transformer, det3d_transformer,
                  num_classes, num_queries, num_feature_levels, 
                  aux_loss=True, with_box_refine=False, init_box=False,
                  group_num=11, depth_mean_gradient_clipping=None,
                  iou_quality_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            det2d_transformer: transformer architecture. See det2d_transformer.py
            det3d_transformer: transformer architecture. See det3d_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.det2d_transformer = det2d_transformer
        self.det3d_transformer = det3d_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = det2d_transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.region_head = RegionSegHead(d_model=hidden_dim)

        self.num_feature_levels = num_feature_levels
        
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        quality_cfg = iou_quality_head or {}
        self.iou_quality_head_enabled = bool(
            quality_cfg.get('enabled', False))
        if self.iou_quality_head_enabled:
            quality_init_seed = int(
                quality_cfg.get('init_seed', 290029))
            # Adding this head must not shift the RNG stream used by existing
            # MonoDGP parameters.  This keeps every parameter shared with the
            # Experiment-26 control bitwise identical at initialization.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    quality_init_seed)
                self.iou_quality_embed = MLP(
                    hidden_dim, hidden_dim, 1, 2)

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes
        depth_clip_cfg = depth_mean_gradient_clipping or {}
        self.depth_mean_gradient_clipping_enabled = bool(
            depth_clip_cfg.get('enabled', False))
        self.depth_mean_gradient_clipping_max_norm = float(
            depth_clip_cfg.get('max_absolute_gradient', 0.03))
        if (self.depth_mean_gradient_clipping_enabled
                and self.depth_mean_gradient_clipping_max_norm <= 0):
            raise ValueError(
                'depth_mean_gradient_clipping.max_absolute_gradient '
                'must be positive')
        self.depth_mean_gradient_clipping_receipts = []

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # One prediction head is required per decoder output.  The historical
        # ``+ 1`` registered a fourth head that neither the 2D nor the 3D
        # three-layer decoder ever consumed.
        num_pred = det3d_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.det2d_transformer.decoder.bbox_embed = self.bbox_embed
            self.det3d_transformer.decoder.bbox_embed = self.bbox_embed
            
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
            if self.iou_quality_head_enabled:
                self.iou_quality_embed = _get_clones(
                    self.iou_quality_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            if self.iou_quality_head_enabled:
                self.iou_quality_embed = nn.ModuleList([
                    self.iou_quality_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None


    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        features, pos = self.backbone(images)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # region enhancement
        enhanced_srcs, region_probs, seg_embed = self.region_head(srcs)

        if self.training:
            query_embeds = self.query_embed.weight
        else:
            # only use one group in inference
            query_embeds = self.query_embed.weight[:self.num_queries]

        srcs = enhanced_srcs
        pred_depth_map_logits, depth_pos_embed = self.depth_predictor(
            srcs, masks[1], seg_embed[1] + pos[1])
        
        #pos_3d = []
        # for l, feat in enumerate(features):
        #     depth_pos_3d = self.position_embed(feat, calibs=None, depth_map = pred_depth_map_logits)
        #     pos[l] = depth_pos_3d
        #pos = pos_3d

        intermediate_output = self.det2d_transformer(srcs, masks, pos, query_embeds)
        
        hs_2d = intermediate_output['hs']
        init_reference_2d = intermediate_output['init_reference_out']
        inter_references_2d = intermediate_output['inter_references_out']
        
        inter_coords = []
        inter_classes = []

        for lvl in range(hs_2d.shape[0]):
            if lvl == 0:
                reference = init_reference_2d
            else:
                reference = inter_references_2d[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs_2d[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            inter_coord = tmp.sigmoid()
            inter_coords.append(inter_coord)

            # classes
            inter_class = self.class_embed[lvl](hs_2d[lvl])
            inter_classes.append(inter_class)

        inter_coord = torch.stack(inter_coords)
        inter_class = torch.stack(inter_classes)

        query_embeds = hs_2d[-1]
        hs, init_reference, inter_references = self.det3d_transformer(intermediate_output, query_embeds, depth_pos_embed)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []       
        outputs_depths = []
        outputs_angles = []
        outputs_qualities = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = self.dim_embed_3d[lvl](hs[lvl])
            outputs_3d_dims.append(size3d)

            # depth_geo_err
            depth_geo_err = self.depth_embed[lvl](hs[lvl])
            
            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            # Object height is measured vertically, so the corresponding
            # focal term is fy.  In KITTI raw P2 has fx == fy; this also stays
            # correct after the same image affine is folded into P2.
            depth_geo = (size3d[:, :, 0] / box2d_height
                         * calibs[:, 1, 1].unsqueeze(1))
            
            # depth_map
            # outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2)   #.detach()
            # depth_map = F.grid_sample(
            #     weighted_depth.unsqueeze(1),
            #     outputs_center3d,
            #     mode='bilinear',
            #     align_corners=True).squeeze(1)    
            
            # depth average + sigma
            # depth_ave = torch.cat([( (1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
            
            depth_ave = torch.cat([depth_geo.unsqueeze(-1) + depth_geo_err[:, :, 0: 1],          
                                    depth_geo_err[:, :, 1: 2]], -1)

            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)
            if self.iou_quality_head_enabled:
                # Deliberately do not detach ``hs``: the quality loss is
                # allowed to improve the shared 3D decoder representation.
                outputs_qualities.append(
                    self.iou_quality_embed[lvl](hs[lvl]))

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        self.depth_mean_gradient_clipping_receipts = []
        if self.training and self.depth_mean_gradient_clipping_enabled:
            outputs_depth = clip_depth_mean_gradients(
                outputs_depth,
                self.depth_mean_gradient_clipping_max_norm,
                self.depth_mean_gradient_clipping_receipts)
        outputs_angle = torch.stack(outputs_angles)
  
        out = dict()
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        if self.iou_quality_head_enabled:
            outputs_quality = torch.stack(outputs_qualities)
            out['pred_quality'] = outputs_quality[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['pred_region_prob'] = region_probs

        out['inter_outputs'] = self._set_inter_loss(inter_class, inter_coord)
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,
                outputs_depth,
                outputs_quality if self.iou_quality_head_enabled else None)
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim,
                      outputs_angle, outputs_depth, outputs_quality=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        values = [{'pred_logits': a, 'pred_boxes': b,
                   'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                  for a, b, c, d, e in zip(
                      outputs_class[:-1], outputs_coord[:-1],
                      outputs_3d_dim[:-1], outputs_angle[:-1],
                      outputs_depth[:-1])]
        if outputs_quality is not None:
            for value, quality in zip(values, outputs_quality[:-1]):
                value['pred_quality'] = quality
        return values
    
    @torch.jit.unused
    def _set_inter_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

    def depth_mean_gradient_clipping_metrics(self):
        """Return the most recent backward's mean-depth clipping receipt."""
        if not self.depth_mean_gradient_clipping_receipts:
            return {}
        receipts = self.depth_mean_gradient_clipping_receipts
        prediction_count = torch.stack(tuple(
            receipt['prediction_count'] for receipt in receipts)).sum()
        clipped_count = torch.stack(tuple(
            receipt['clipped_count'] for receipt in receipts)).sum()
        pre_clip_max = torch.stack(tuple(
            receipt['pre_clip_max_absolute_gradient']
            for receipt in receipts)).max()
        minimum_scale = torch.stack(tuple(
            receipt['minimum_scale'] for receipt in receipts)).min()
        pre_clip_energy = torch.stack(tuple(
            receipt['pre_clip_energy'] for receipt in receipts)).sum()
        post_clip_energy = torch.stack(tuple(
            receipt['post_clip_energy'] for receipt in receipts)).sum()
        fraction = (clipped_count.float()
                    / prediction_count.clamp_min(1).float())
        retained_energy_fraction = torch.where(
            pre_clip_energy > 0,
            post_clip_energy / pre_clip_energy.clamp_min(1e-30),
            torch.ones_like(pre_clip_energy))
        return {
            'depth_mean_clip_prediction_count': prediction_count,
            'depth_mean_clip_applied_count': clipped_count,
            'depth_mean_clip_applied_fraction': fraction,
            'depth_mean_pre_clip_max_absolute_gradient': pre_clip_max,
            'depth_mean_clip_minimum_retained_fraction': minimum_scale,
            'depth_mean_pre_clip_energy': pre_clip_energy,
            'depth_mean_post_clip_energy': post_clip_energy,
            'depth_mean_clip_retained_energy_fraction': (
                retained_energy_fraction),
        }


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDGP.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 inter_losses, group_num=11,
                 geometry_interval_monitoring=None,
                 query_monitoring=None,
                 iou3d_matching_monitoring=None,
                 high_iou_unmatched_negative_weighting=None,
                 iou_quality_head=None,
                 iou_classification=None,
                 use_vectorized_ddn_rasterization=False,
                 use_aligned_giou_loss=False,
                 use_post_match_cache=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.inter_losses = inter_losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss(
            use_vectorized_rasterization=use_vectorized_ddn_rasterization)
        self.bce = nn.BCELoss()
        self.bce_noReduce = nn.BCELoss(reduction='none')

        self.group_num = group_num
        self.use_aligned_giou_loss = bool(use_aligned_giou_loss)
        self.use_post_match_cache = bool(use_post_match_cache)
        monitor_cfg = geometry_interval_monitoring or {}
        self.geometry_interval_monitoring_enabled = bool(
            monitor_cfg.get('enabled', False))
        self.geometry_interval_monitoring_car_class_id = int(
            monitor_cfg.get('car_class_id', 1))
        self.geometry_interval_monitoring_iou_threshold = float(
            monitor_cfg.get('iou_threshold', 0.7))
        self.geometry_interval_monitoring_decode_means = tuple(
            tuple(float(component) for component in row)
            for row in monitor_cfg.get(
                'decode_mean_sizes', ((0.0, 0.0, 0.0),) * 3))
        self.geometry_conditioned_interval_depth_receipts = {}
        query_monitor_cfg = query_monitoring or {}
        self.query_monitoring_enabled = bool(
            query_monitor_cfg.get('enabled', True))
        self.query_monitoring_topk = int(
            query_monitor_cfg.get('inference_topk', 50))
        self.query_monitoring_score_threshold = float(
            query_monitor_cfg.get('score_threshold', 0.2))
        self.query_monitoring_car_class_id = int(
            query_monitor_cfg.get('car_class_id', 1))
        iou3d_monitor_cfg = iou3d_matching_monitoring or {}
        self.iou3d_matching_monitoring_enabled = bool(
            iou3d_monitor_cfg.get('enabled', False))
        self.collect_iou3d_matching_comparison = False
        negative_cfg = high_iou_unmatched_negative_weighting or {}
        self.high_iou_unmatched_negative_weighting_enabled = bool(
            negative_cfg.get('enabled', False))
        self.high_iou_unmatched_negative_lower = float(
            negative_cfg.get('full_weight_below_iou', 0.5))
        self.high_iou_unmatched_negative_upper = float(
            negative_cfg.get('zero_weight_at_iou', 0.7))
        if not (0 <= self.high_iou_unmatched_negative_lower
                < self.high_iou_unmatched_negative_upper <= 1):
            raise ValueError(
                'high-IoU negative thresholds must satisfy '
                '0 <= lower < upper <= 1')
        if (self.high_iou_unmatched_negative_weighting_enabled
                and self.matcher.cost_iou3d == 0):
            raise ValueError(
                'high-IoU negative weighting requires 3D-IoU matching cost')
        quality_cfg = iou_quality_head or {}
        self.iou_quality_head_enabled = bool(
            quality_cfg.get('enabled', False))
        iou_classification_cfg = iou_classification or {}
        self.iou_classification_enabled = bool(
            iou_classification_cfg.get('enabled', False))
        self.iou_classification_beta = float(
            iou_classification_cfg.get('beta', 2.0))
        nms_ranking_cfg = iou_classification_cfg.get('nms_ranking', {})
        self.iou_classification_nms_ranking_enabled = bool(
            nms_ranking_cfg.get('enabled', False))
        self.iou_classification_nms_bev_threshold = float(
            nms_ranking_cfg.get('bev_iou_threshold', 0.8))
        self.iou_classification_nms_min_iou_delta = float(
            nms_ranking_cfg.get('min_iou_delta', 1e-6))
        if self.iou_classification_beta < 0:
            raise ValueError('3D-IoU classification beta must be non-negative')
        if (self.iou_classification_nms_ranking_enabled
                and not self.iou_classification_enabled):
            raise ValueError(
                'NMS-aware ranking requires 3D-IoU classification')
        if not 0.0 <= self.iou_classification_nms_bev_threshold <= 1.0:
            raise ValueError('NMS ranking BEV-IoU threshold must be in [0, 1]')
        if not 0.0 <= self.iou_classification_nms_min_iou_delta < 1.0:
            raise ValueError('NMS ranking minimum IoU delta must be in [0, 1)')
        if (self.iou_classification_enabled
                and self.matcher.cost_iou3d == 0):
            raise ValueError(
                '3D-IoU classification requires exact 3D-IoU matching')
        if (self.iou_classification_enabled
                and self.high_iou_unmatched_negative_weighting_enabled):
            raise ValueError(
                '3D-IoU classification replaces unmatched-negative weighting')
        self.iou_quality_target_encoding = quality_cfg.get(
            'target_encoding', 'cia_ssd')
        self.iou_quality_supervision = quality_cfg.get(
            'supervision', 'hungarian_positive')
        supported_quality_supervision = {
            'hungarian_positive', 'all_query_same_gt_ranking'}
        if (self.iou_quality_head_enabled
                and self.iou_quality_supervision
                not in supported_quality_supervision):
            raise ValueError(
                'unsupported IoU quality supervision: '
                f'{self.iou_quality_supervision}')
        self.iou_quality_ranking_iou_gap = float(
            quality_cfg.get('ranking_iou_gap', 0.1))
        self.iou_quality_low_iou_threshold = float(
            quality_cfg.get('low_iou_threshold', 0.1))
        self.iou_quality_low_iou_weight = float(
            quality_cfg.get('low_iou_weight', 0.1))
        self.iou_quality_full_weight_iou = float(
            quality_cfg.get('full_weight_iou', 0.5))
        if (self.iou_quality_head_enabled
                and self.iou_quality_target_encoding != 'cia_ssd'):
            raise ValueError(
                'only the cia_ssd IoU quality target is supported')
        if (self.iou_quality_head_enabled
                and self.matcher.cost_iou3d == 0):
            raise ValueError(
                'the IoU quality head requires exact 3D-IoU matching')
        self.last_final_iou3d_matrix = None
        self.collect_mixup_target_monitoring = False

    @torch.no_grad()
    def _classification_query_weights(self, src_logits, indices,
                                      targets=None):
        """Downweight only unmatched queries that already have high 3D IoU."""
        weights = src_logits.new_ones(src_logits.shape[:2])
        if not self.high_iou_unmatched_negative_weighting_enabled:
            return None, {}
        iou3d = getattr(self.matcher, 'last_iou3d_matrix', None)
        if iou3d is None:
            return None, {}
        if iou3d.shape[:2] != src_logits.shape[:2]:
            raise ValueError('matcher 3D-IoU matrix does not match logits')

        max_iou = (iou3d.max(dim=-1).values if iou3d.shape[-1]
                   else weights.new_zeros(weights.shape))
        matched = torch.zeros_like(weights, dtype=torch.bool)
        for batch_index, (source, _) in enumerate(indices):
            matched[batch_index, source.to(device=matched.device)] = True
        lower = self.high_iou_unmatched_negative_lower
        upper = self.high_iou_unmatched_negative_upper
        weights = ((upper - max_iou) / (upper - lower)).clamp(0, 1)
        weights.masked_fill_(matched, 1)

        unmatched = ~matched
        unmatched_count = unmatched.sum().clamp_min(1)
        downweighted = unmatched & (weights < 1)
        ignored = unmatched & (weights == 0)
        metrics = {
            'monitor_high_iou_negative_downweighted_fraction': (
                downweighted.sum() / unmatched_count),
            'monitor_high_iou_negative_ignored_fraction': (
                ignored.sum() / unmatched_count),
            'monitor_high_iou_negative_mean_weight': (
                weights[unmatched].mean() if unmatched.any()
                else weights.new_ones(())),
        }
        high_iou_unmatched = unmatched & (max_iou >= lower)
        high_iou_scores = []
        ignored_high_iou_scores = []
        if targets is not None:
            probabilities = src_logits.sigmoid()
            for batch_index, target in enumerate(targets):
                target_labels = target['labels'].reshape(-1).long()
                if (target_labels.numel() == 0
                        or not high_iou_unmatched[batch_index].any()):
                    continue
                closest_target = iou3d[
                    batch_index, :, :target_labels.numel()].argmax(dim=-1)
                selected = high_iou_unmatched[batch_index]
                high_iou_scores.append(probabilities[batch_index][
                    selected, target_labels[closest_target[selected]]])
                ignored_selected = ignored[batch_index]
                if ignored_selected.any():
                    ignored_high_iou_scores.append(probabilities[batch_index][
                        ignored_selected,
                        target_labels[closest_target[ignored_selected]]])
        metrics['monitor_high_iou_negative_mean_gt_class_score'] = (
            torch.cat(high_iou_scores).mean()
            if high_iou_scores else weights.new_zeros(()))
        metrics['monitor_high_iou_negative_ignored_mean_gt_class_score'] = (
            torch.cat(ignored_high_iou_scores).mean()
            if ignored_high_iou_scores else weights.new_zeros(()))
        return weights, metrics

    @staticmethod
    def _append_iou3d_matching_receipt(matcher, receipts):
        receipt = getattr(matcher, 'last_iou3d_receipt', {})
        if int(receipt.get('comparison_count', 0)) > 0:
            receipts.append(receipt.copy())

    @staticmethod
    def _iou3d_matching_metrics(receipts, device):
        comparison_count = sum(
            int(receipt['comparison_count']) for receipt in receipts)
        if comparison_count == 0:
            return {}
        denominator = float(comparison_count)
        values = {
            'monitor_iou3d_matching_identity_change_fraction': sum(
                int(receipt['changed_count']) for receipt in receipts)
                / denominator,
            'monitor_iou3d_matching_mean_iou3d_gain': sum(
                float(receipt['iou3d_gain_sum']) for receipt in receipts)
                / denominator,
            'monitor_iou3d_matching_mean_giou2d_delta': sum(
                float(receipt['giou2d_delta_sum']) for receipt in receipts)
                / denominator,
            'monitor_iou3d_matching_mean_gt_class_score_delta': sum(
                float(receipt['class_score_delta_sum'])
                for receipt in receipts) / denominator,
            'monitor_iou3d_matching_current_mean_iou3d': sum(
                float(receipt['current_iou3d_sum'])
                for receipt in receipts) / denominator,
            'monitor_iou3d_matching_current_mean_giou2d': sum(
                float(receipt['current_giou2d_sum'])
                for receipt in receipts) / denominator,
            'monitor_iou3d_matching_current_mean_gt_class_score': sum(
                float(receipt['current_class_score_sum'])
                for receipt in receipts) / denominator,
            'monitor_iou3d_matching_best_iou3d_query_mean_gt_class_score': sum(
                float(receipt['best_iou3d_query_class_score_sum'])
                for receipt in receipts) / denominator,
        }
        return {
            key: torch.tensor(value, dtype=torch.float32, device=device)
            for key, value in values.items()
        }

    @staticmethod
    def _pearson(values, targets):
        values = values.float()
        targets = targets.float()
        values = values - values.mean()
        targets = targets - targets.mean()
        denominator = (
            values.square().sum().sqrt()
            * targets.square().sum().sqrt()).clamp_min(1e-12)
        return (values * targets).sum() / denominator

    @torch.no_grad()
    def _iou_classification_targets(self, src_logits, targets):
        iou3d = getattr(self.matcher, 'last_iou3d_matrix', None)
        if iou3d is None:
            raise RuntimeError(
                'matcher did not expose exact 3D-IoU classification targets')
        if iou3d.shape[:2] != src_logits.shape[:2]:
            raise ValueError(
                'matcher 3D-IoU matrix does not match classification logits')
        soft_targets = torch.zeros_like(src_logits)
        selected_scores = []
        selected_ious = []
        for batch_index, target in enumerate(targets):
            target_count = int(target['labels'].numel())
            if target_count == 0:
                continue
            image_iou = iou3d[batch_index, :, :target_count].detach()
            max_iou, closest_target = image_iou.max(dim=-1)
            closest_labels = target['labels'].reshape(-1).long().index_select(
                0, closest_target)
            soft_targets[batch_index].scatter_(
                1, closest_labels[:, None], max_iou[:, None])
            selected_scores.append(
                src_logits[batch_index].sigmoid().gather(
                    1, closest_labels[:, None]).squeeze(1).detach())
            selected_ious.append(max_iou)

        metrics = {}
        if selected_ious:
            scores = torch.cat(selected_scores)
            ious = torch.cat(selected_ious)
            metrics = {
                'monitor_iou_classification_target_mean': ious.mean(),
                'monitor_iou_classification_target_nonzero_fraction': (
                    (ious > 0).float().mean()),
                'monitor_iou_classification_target_ge_0_1_fraction': (
                    (ious >= 0.1).float().mean()),
                'monitor_iou_classification_target_ge_0_5_fraction': (
                    (ious >= 0.5).float().mean()),
                'monitor_iou_classification_target_ge_0_7_fraction': (
                    (ious >= 0.7).float().mean()),
                'monitor_iou_classification_score_mae': (
                    scores - ious).abs().mean(),
                'monitor_iou_classification_score_iou_pearson': (
                    self._pearson(scores, ious)),
            }
        return soft_targets, metrics

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True,
                    matched_cache=None, apply_high_iou_weighting=True,
                    apply_iou_classification=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
        target_classes_o = (
            matched_cache['matched_targets']['labels']
            if matched_cache is not None
            else torch.cat([t["labels"][J]
                            for t, (_, J) in zip(targets, indices)]))
        use_iou_classification = bool(
            self.iou_classification_enabled
            and apply_iou_classification
            and {'pred_depth', 'pred_3d_dim', 'pred_angle'}.issubset(outputs))
        if use_iou_classification:
            target_classes_onehot, weighting_metrics = (
                self._iou_classification_targets(src_logits, targets))
            loss_ce = quality_focal_loss(
                src_logits, target_classes_onehot, num_boxes,
                beta=self.iou_classification_beta) * src_logits.shape[1]
            if self.iou_classification_nms_ranking_enabled:
                weighting_metrics.update(nms_aware_iou_ranking_loss(
                    src_logits, outputs, targets,
                    self.matcher.last_iou3d_matrix,
                    decode_mean_sizes=(
                        self.matcher.iou3d_decode_mean_sizes),
                    group_num=(self.group_num if self.training else 1),
                    bev_iou_threshold=(
                        self.iou_classification_nms_bev_threshold),
                    min_iou_delta=(
                        self.iou_classification_nms_min_iou_delta)))
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes,
                dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o.reshape(-1).long()
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1],
                 src_logits.shape[2] + 1], dtype=src_logits.dtype,
                layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(
                2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            if apply_high_iou_weighting:
                query_weights, weighting_metrics = (
                    self._classification_query_weights(
                        src_logits, indices, targets=targets))
            else:
                query_weights, weighting_metrics = None, {}
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes,
                alpha=self.focal_alpha, gamma=2,
                query_weights=query_weights) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce, **weighting_metrics}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Count detections with the same top-k/threshold rule as inference."""
        pred_logits = outputs['pred_logits']
        batch_size, query_count, class_count = pred_logits.shape
        group_num = self.group_num if self.training else 1
        if query_count % group_num:
            raise ValueError(
                f'{query_count} queries cannot be split into {group_num} groups')
        queries_per_group = query_count // group_num
        grouped_scores = pred_logits.sigmoid().view(
            batch_size, group_num, queries_per_group * class_count)
        topk = min(self.query_monitoring_topk, grouped_scores.shape[-1])
        scores, indexes = torch.topk(grouped_scores, topk, dim=-1)
        kept = scores >= self.query_monitoring_score_threshold
        labels = indexes.remainder(class_count)
        predicted = kept.sum(dim=-1).float()
        predicted_car = (
            kept & (labels == self.query_monitoring_car_class_id)
        ).sum(dim=-1).float()

        gt = pred_logits.new_tensor([len(target['labels']) for target in targets])
        gt_car = torch.stack([
            (target['labels'] == self.query_monitoring_car_class_id).sum()
            for target in targets
        ]).to(dtype=pred_logits.dtype)

        def summarize(prefix, values, car_values):
            signed = values - gt[:, None]
            car_signed = car_values - gt_car[:, None]
            return {
                f'{prefix}_predicted_count': values.mean(),
                f'{prefix}_predicted_car_count': car_values.mean(),
                f'{prefix}_absolute_error': signed.abs().mean(),
                f'{prefix}_car_absolute_error': car_signed.abs().mean(),
                f'{prefix}_signed_error': signed.mean(),
                f'{prefix}_car_signed_error': car_signed.mean(),
            }

        losses = {
            'monitor_cardinality_gt_count': gt.mean(),
            'monitor_cardinality_gt_car_count': gt_car.mean(),
        }
        losses.update(summarize(
            'monitor_cardinality_all_groups', predicted, predicted_car))
        losses.update(summarize(
            'monitor_cardinality_group0',
            predicted[:, :1], predicted_car[:, :1]))
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes,
                      matched_cache=None):
        
        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_boxes = (
            matched_cache['matched_targets']['boxes_3d']
            if matched_cache is not None
            else torch.cat([t['boxes_3d'][i]
                            for t, (_, i) in zip(targets, indices)], dim=0))
        target_3dcenter = target_boxes[:, 0:2]

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes,
                   matched_cache=None):
        
        assert 'pred_boxes' in outputs
        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_boxes = (
            matched_cache['matched_targets']['boxes_3d']
            if matched_cache is not None
            else torch.cat([t['boxes_3d'][i]
                            for t, (_, i) in zip(targets, indices)], dim=0))
        target_2dboxes = target_boxes[:, 2:6]

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        src_boxes_xyxy = box_ops.box_cxcylrtb_to_xyxy(src_boxes)
        target_boxes_xyxy = box_ops.box_cxcylrtb_to_xyxy(target_boxes)
        if self.use_aligned_giou_loss:
            matched_giou = box_ops.generalized_box_iou_aligned(
                src_boxes_xyxy, target_boxes_xyxy)
        else:
            matched_giou = torch.diag(box_ops.generalized_box_iou(
                src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = 1 - matched_giou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes,
                    matched_cache=None):

        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = (
            matched_cache['matched_targets']['depth']
            if matched_cache is not None
            else torch.cat([t['depth'][i]
                            for t, (_, i) in zip(targets, indices)], dim=0)
            ).squeeze()
         
        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        absolute_error = torch.abs(depth_input - target_depths)
        weighted_absolute = (
            1.4142 * torch.exp(-depth_log_variance) * absolute_error)
        depth_loss = weighted_absolute + depth_log_variance
        
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        with torch.no_grad():
            normalizer = float(num_boxes)
            precision = torch.exp(-depth_log_variance)
            finite_count = max(int(absolute_error.numel()), 1)
            error_mean = absolute_error.mean()
            precision_mean = precision.mean()
            if absolute_error.numel() > 1:
                centered_error = absolute_error - error_mean
                centered_precision = precision - precision_mean
                correlation_denominator = torch.sqrt(
                    centered_error.square().sum()
                    * centered_precision.square().sum())
                error_precision_correlation = (
                    (centered_error * centered_precision).sum()
                    / correlation_denominator.clamp_min(
                        torch.finfo(absolute_error.dtype).eps))
            else:
                error_precision_correlation = absolute_error.new_zeros(())

            # For the depth mean output, |dL/d(depth)| is sqrt(2) * precision
            # whenever the residual is non-zero. The common normalizer and
            # sqrt(2) cancel when reporting each MAE bin's share of local
            # gradient energy. This is intentionally an output-gradient
            # diagnostic, not a claim about full parameter-gradient share.
            local_gradient_energy = precision.square() * (
                absolute_error != 0).to(dtype=precision.dtype)
            total_gradient_energy = local_gradient_energy.sum().clamp_min(
                torch.finfo(local_gradient_energy.dtype).eps)
            error_bins = (
                ('lt_0_1m', absolute_error < 0.1),
                ('0_1_to_0_5m',
                 (absolute_error >= 0.1) & (absolute_error < 0.5)),
                ('0_5_to_1m',
                 (absolute_error >= 0.5) & (absolute_error < 1.0)),
                ('ge_1m', absolute_error >= 1.0),
            )
            losses.update({
                'monitor_depth_mae': absolute_error.sum() / normalizer,
                'monitor_depth_weighted_absolute': (
                    weighted_absolute.sum() / normalizer),
                'monitor_depth_log_scale_mean': (
                    depth_log_variance.sum() / normalizer),
                'monitor_depth_precision_mean': precision.mean(),
                'monitor_depth_mae_precision_correlation': (
                    error_precision_correlation),
                'monitor_depth_precision_gt_2_fraction': (
                    (precision > 2).sum() / finite_count),
                'monitor_depth_precision_gt_4_fraction': (
                    (precision > 4).sum() / finite_count),
                'monitor_depth_precision_gt_8_fraction': (
                    (precision > 8).sum() / finite_count),
            })
            for bin_name, bin_mask in error_bins:
                losses[
                    f'monitor_depth_target_fraction_{bin_name}'
                ] = bin_mask.sum() / finite_count
                losses[
                    f'monitor_depth_local_gradient_energy_fraction_{bin_name}'
                ] = (
                    local_gradient_energy[bin_mask].sum()
                    / total_gradient_energy)
            if depth_log_variance.numel():
                quantiles = depth_log_variance.float().quantile(
                    depth_log_variance.new_tensor((0.1, 0.5, 0.9)))
                precision_p90 = precision.float().quantile(0.9)
                losses.update({
                    'monitor_depth_log_scale_p10': quantiles[0],
                    'monitor_depth_log_scale_p50': quantiles[1],
                    'monitor_depth_log_scale_p90': quantiles[2],
                    'monitor_depth_precision_p90': precision_p90,
                })
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes,
                  matched_cache=None):

        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = (
            matched_cache['matched_targets']['size_3d']
            if matched_cache is not None
            else torch.cat([t['size_3d'][i]
                            for t, (_, i) in zip(targets, indices)], dim=0))

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes,
                    matched_cache=None):

        idx = (matched_cache['source_index'] if matched_cache is not None
               else self._get_src_permutation_idx(indices))
        heading_input = outputs['pred_angle'][idx]
        if matched_cache is not None:
            target_heading_cls = matched_cache['matched_targets']['heading_bin']
            target_heading_res = matched_cache['matched_targets']['heading_res']
        else:
            target_heading_cls = torch.cat(
                [t['heading_bin'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            target_heading_res = torch.cat(
                [t['heading_res'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(
            heading_target_cls.shape[0], 12,
            device=heading_target_cls.device).scatter_(
                dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        with torch.no_grad():
            losses['monitor_angle_classification'] = (
                cls_loss.sum() / float(num_boxes))
            losses['monitor_angle_residual'] = (
                reg_loss.sum() / float(num_boxes))
        return losses

    def loss_quality(self, outputs, targets, indices, num_boxes):
        """Supervise the configured exact-3D-IoU quality objective.

        The exact IoU values are numerical supervision supplied by the
        matcher.  Predictions retain their complete gradient path through
        the quality head and the shared 3D decoder features.
        """
        if 'pred_quality' not in outputs:
            raise KeyError('quality loss requires pred_quality')
        iou3d = getattr(self.matcher, 'last_iou3d_matrix', None)
        if iou3d is None:
            raise RuntimeError('matcher did not expose exact 3D-IoU targets')
        if iou3d.shape[:2] != outputs['pred_quality'].shape[:2]:
            raise ValueError('matcher 3D-IoU matrix does not match quality')

        if self.iou_quality_supervision == 'all_query_same_gt_ranking':
            return all_query_quality_ranking_loss(
                outputs['pred_quality'], iou3d,
                tuple(len(target['labels']) for target in targets),
                group_num=(self.group_num if self.training else 1),
                ranking_iou_gap=self.iou_quality_ranking_iou_gap,
                low_iou_threshold=self.iou_quality_low_iou_threshold,
                low_iou_weight=self.iou_quality_low_iou_weight,
                full_weight_iou=self.iou_quality_full_weight_iou)
        predictions = []
        targets_iou = []
        for batch_index, (source_index, target_index) in enumerate(indices):
            if source_index.numel() == 0:
                continue
            source_index = source_index.to(iou3d.device)
            target_index = target_index.to(iou3d.device)
            predictions.append(outputs['pred_quality'][
                batch_index, source_index, 0])
            targets_iou.append(iou3d[
                batch_index, source_index, target_index])
        if predictions:
            prediction = torch.cat(predictions)
            target_iou = torch.cat(targets_iou).detach().clamp(0, 1)
        else:
            prediction = outputs['pred_quality'].reshape(-1)[:0]
            target_iou = prediction.detach()

        # CIA-SSD maps IoU=0.5 to 0 and IoU=1.0 to 1.  Values below
        # 0.5 remain valid negative regression targets down to -1.
        encoded_target = 2.0 * (target_iou - 0.5)
        loss = F.smooth_l1_loss(
            prediction, encoded_target, reduction='sum') / num_boxes
        result = {'loss_quality': loss}
        with torch.no_grad():
            decoded_prediction = ((prediction + 1.0) * 0.5).clamp(0, 1)
            result['monitor_quality_iou_mae'] = (
                (decoded_prediction - target_iou).abs().sum() / num_boxes)
            result['monitor_quality_target_iou_mean'] = (
                target_iou.sum() / num_boxes)
            result['monitor_quality_predicted_iou_mean'] = (
                decoded_prediction.sum() / num_boxes)
        return result

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_probs = outputs['pred_region_prob']
        gt_region = torch.cat([t['obj_region'].unsqueeze(0) for t in targets], dim=0)

        loss = 0
        losses = dict()
        for region_prob in region_probs:
            gt_region_resized = F.interpolate(gt_region.unsqueeze(1).float(), size=region_prob.shape[2:], mode='bilinear', align_corners=True)
            # Compute intersection and union
            intersection = (region_prob * gt_region_resized).sum()
            total = region_prob.sum() + gt_region_resized.sum()
            # Compute Dice Coefficient
            dice_coef = (2. * intersection + 1) / (total + 1)
            # Compute Dice Loss
            dice_loss = 1 - dice_coef
            loss += dice_loss

        losses['loss_region'] = loss

        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'quality': self.loss_quality,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'region': self.loss_region,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        matched_cache = kwargs.pop('matched_cache', None)
        if matched_cache is not None and loss in _POST_MATCH_TARGET_FIELDS:
            kwargs['matched_cache'] = matched_cache
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _post_match_cache(self, outputs, targets, indices, active_losses):
        if not self.use_post_match_cache:
            return None
        return build_post_match_cache(
            outputs, targets, indices, active_losses)

    @torch.no_grad()
    def _group0_loss_metrics(self, outputs, targets, indices, num_boxes):
        """Re-evaluate final-layer matched losses for inference query group 0."""
        query_count = outputs['pred_logits'].shape[1]
        if query_count % self.group_num:
            raise ValueError(
                f'{query_count} queries cannot be split into '
                f'{self.group_num} groups')
        queries_per_group = query_count // self.group_num
        group0_indices = []
        for source_index, target_index in indices:
            keep = source_index < queries_per_group
            group0_indices.append((source_index[keep], target_index[keep]))

        query_fields = (
            'pred_logits', 'pred_boxes', 'pred_depth', 'pred_3d_dim',
            'pred_angle')
        group0_outputs = {
            key: outputs[key][:, :queries_per_group]
            for key in query_fields
        }
        active_losses = (
            'labels', 'boxes', 'depths', 'dims', 'angles', 'center')
        matched_cache = self._post_match_cache(
            group0_outputs, targets, group0_indices, active_losses)
        metrics = {}
        for loss_name in active_losses:
            kwargs = {'matched_cache': matched_cache}
            if loss_name == 'labels':
                kwargs['log'] = False
                # This is a read-only diagnostic over a 50-query slice.  Its
                # logits do not correspond to the matcher's cached 550-query
                # IoU matrix and it must not affect the optimized loss.
                kwargs['apply_high_iou_weighting'] = False
                kwargs['apply_iou_classification'] = False
            values = self.get_loss(
                loss_name, group0_outputs, targets, group0_indices,
                num_boxes, **kwargs)
            metrics.update({
                f'monitor_group0_{key}': value.detach()
                for key, value in values.items()
            })
        return metrics

    @torch.no_grad()
    def _mixup_target_metrics(self, outputs, targets, indices,
                              matched_cache):
        """Compare final-layer matched primary and MixUp donor targets."""
        if matched_cache is None or not all(
                'mixup_is_donor' in target for target in targets):
            return {}
        source_index = matched_cache['source_index']
        batch_index = source_index[0]
        matched = matched_cache['matched_targets']
        donor_mask = torch.cat([
            target['mixup_is_donor'][target_index]
            for target, (_, target_index) in zip(targets, indices)
        ]).bool()
        if donor_mask.numel() == 0:
            return {}

        logits = outputs['pred_logits'][source_index]
        labels = matched['labels'].reshape(-1).long()
        class_probability = logits.sigmoid().gather(
            1, labels[:, None]).squeeze(1)

        predicted_boxes = outputs['pred_boxes'][source_index]
        target_boxes = matched['boxes_3d']
        bbox_component_mae = (
            predicted_boxes[:, 2:6] - target_boxes[:, 2:6]
        ).abs().mean(dim=1)
        giou_error = 1 - box_ops.generalized_box_iou_aligned(
            box_ops.box_cxcylrtb_to_xyxy(predicted_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes))

        input_sizes = torch.stack([
            target['projective_input_size'] for target in targets
        ]).to(device=predicted_boxes.device, dtype=predicted_boxes.dtype)
        center_error_pixels = (
            (predicted_boxes[:, :2] - target_boxes[:, :2])
            * input_sizes.index_select(0, batch_index)
        ).norm(dim=1)

        predicted_depth = outputs['pred_depth'][source_index][:, 0]
        target_depth = matched['depth'].reshape(-1)
        depth_mae = (predicted_depth - target_depth).abs()

        predicted_dims = outputs['pred_3d_dim'][source_index]
        target_dims = matched['size_3d']
        dimension_component_mae = (
            predicted_dims - target_dims).abs().mean(dim=1)

        predicted_angle = outputs['pred_angle'][source_index].view(-1, 24)
        target_angle_class = matched['heading_bin'].reshape(-1).long()
        target_angle_residual = matched['heading_res'].reshape(-1)
        angle_class_correct = (
            predicted_angle[:, :12].argmax(dim=1) == target_angle_class
        ).to(dtype=predicted_angle.dtype)
        predicted_residual = predicted_angle[:, 12:24].gather(
            1, target_angle_class[:, None]).squeeze(1)
        angle_residual_mae = (
            predicted_residual - target_angle_residual).abs()

        values = {
            'matched_class_probability': class_probability,
            'bbox_component_mae': bbox_component_mae,
            'giou_error': giou_error,
            'center_error_pixels': center_error_pixels,
            'depth_mae_m': depth_mae,
            'dimension_component_mae': dimension_component_mae,
            'angle_class_accuracy': angle_class_correct,
            'angle_residual_mae': angle_residual_mae,
        }
        metrics = {}
        for name, subset in (
                ('primary', ~donor_mask), ('donor', donor_mask)):
            count = subset.sum()
            metrics[f'monitor_mixup_{name}_matched_count'] = count.float()
            denominator = count.clamp_min(1)
            for metric_name, metric_values in values.items():
                metrics[f'monitor_mixup_{name}_{metric_name}'] = (
                    metric_values[subset].sum() / denominator)
        return metrics

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.geometry_conditioned_interval_depth_receipts = {}
        collect_iou3d_comparison = bool(
            self.training
            and self.iou3d_matching_monitoring_enabled
            and self.collect_iou3d_matching_comparison)
        self.matcher.collect_iou3d_comparison = collect_iou3d_comparison
        iou3d_matching_receipts = []
        group_num = self.group_num if self.training else 1
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        prepared_matcher_targets = (
            self.matcher.prepare_targets(targets)
            if getattr(self.matcher, 'use_batched_same_image_cost', False)
            else None)

        # Compute Det 2D loss
        for i, inter_outputs in enumerate(outputs['inter_outputs']):
            indices = self.matcher(
                inter_outputs, targets, group_num=group_num,
                prepared_targets=prepared_matcher_targets)
            matched_cache = self._post_match_cache(
                inter_outputs, targets, indices, self.inter_losses)
            for loss in self.inter_losses:
                l_dict = self.get_loss(
                    loss, inter_outputs, targets, indices, num_boxes,
                    matched_cache=matched_cache)
                l_dict = {k + f'_inter_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # Compute Det 2D and 3D loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'inter_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(
            outputs_without_aux, targets, group_num=group_num,
            prepared_targets=prepared_matcher_targets)
        final_iou3d = getattr(self.matcher, 'last_iou3d_matrix', None)
        self.last_final_iou3d_matrix = (
            final_iou3d.detach()
            if ((self.iou_quality_head_enabled
                 or self.iou_classification_enabled)
                and final_iou3d is not None) else None)
        if collect_iou3d_comparison:
            self._append_iou3d_matching_receipt(
                self.matcher, iou3d_matching_receipts)
        matched_cache = self._post_match_cache(
            outputs, targets, indices, self.losses)
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes,
                matched_cache=matched_cache))
        if self.training and self.collect_mixup_target_monitoring:
            losses.update(self._mixup_target_metrics(
                outputs_without_aux, targets, indices, matched_cache))
        if (self.query_monitoring_enabled and self.training
                and group_num > 1):
            group0_num_boxes = sum(len(t['labels']) for t in targets)
            group0_num_boxes = torch.as_tensor(
                [group0_num_boxes], dtype=torch.float,
                device=next(iter(outputs.values())).device)
            group0_num_boxes = torch.clamp(
                group0_num_boxes / get_world_size(), min=1).item()
            losses.update(self._group0_loss_metrics(
                outputs_without_aux, targets, indices, group0_num_boxes))
        if self.geometry_interval_monitoring_enabled:
            with torch.no_grad():
                _, _, receipt = asymmetric_interval_and_uncertainty_loss(
                    outputs, targets, indices, num_boxes,
                    car_class_id=self.geometry_interval_monitoring_car_class_id,
                    iou_threshold=(
                        self.geometry_interval_monitoring_iou_threshold),
                    decode_mean_sizes=(
                        self.geometry_interval_monitoring_decode_means))
            self.geometry_conditioned_interval_depth_receipts['final'] = receipt

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(
                    aux_outputs, targets, group_num=group_num,
                    prepared_targets=prepared_matcher_targets)
                if collect_iou3d_comparison:
                    self._append_iou3d_matching_receipt(
                        self.matcher, iou3d_matching_receipts)
                active_losses = [
                    loss for loss in self.losses
                    if loss not in ('depth_map', 'region', 'cardinality')]
                matched_cache = self._post_match_cache(
                    aux_outputs, targets, indices, active_losses)
                for loss in self.losses:
                    if loss in ('depth_map', 'region', 'cardinality'):
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    kwargs['matched_cache'] = matched_cache
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self.matcher.collect_iou3d_comparison = False
        losses.update(self._iou3d_matching_metrics(
            iou3d_matching_receipts,
            outputs['pred_logits'].device))
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    det2d_transformer = build_det2d_transformer(cfg)
    det3d_transformer = build_det3d_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDGP(
        backbone = backbone,
        depth_predictor = depth_predictor,
        det2d_transformer = det2d_transformer,
        det3d_transformer = det3d_transformer,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        init_box=cfg['init_box'],
        group_num=cfg['group_num'],
        depth_mean_gradient_clipping=cfg.get(
            'depth_mean_gradient_clipping'),
        iou_quality_head=cfg.get('iou_quality_head'))

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_region'] = cfg['region_loss_coef']
    quality_cfg = cfg.get('iou_quality_head') or {}
    quality_enabled = bool(quality_cfg.get('enabled', False))
    if quality_enabled:
        quality_loss_coef = float(quality_cfg.get('loss_coef', 1.0))
        quality_supervision = quality_cfg.get(
            'supervision', 'hungarian_positive')
        if quality_supervision == 'all_query_same_gt_ranking':
            weight_dict['loss_quality_point'] = (
                quality_loss_coef
                * float(quality_cfg.get('point_loss_coef', 1.0)))
            weight_dict['loss_quality_rank'] = (
                quality_loss_coef
                * float(quality_cfg.get('rank_loss_coef', 0.2)))
        else:
            weight_dict['loss_quality'] = quality_loss_coef
    iou_classification_cfg = cfg.get('iou_classification') or {}
    nms_ranking_cfg = iou_classification_cfg.get('nms_ranking', {})
    if bool(nms_ranking_cfg.get('enabled', False)):
        weight_dict['loss_iou_classification_nms_rank'] = float(
            nms_ranking_cfg.get('loss_coef', 0.1))

    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    inter_weight_dict = {}
    inter_keys = ['loss_ce', 'loss_bbox', 'loss_center', 'loss_giou']
    layers = cfg['dec_layers']
    for i in range(layers):
        inter_weight_dict.update({k + f'_inter_{i}': v for k, v in weight_dict.items() if k in inter_keys})
    weight_dict.update(inter_weight_dict)

    inter_losses = ['labels', 'boxes', 'center']
    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'region']
    if quality_enabled:
        losses.append('quality')

    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        inter_losses=inter_losses,
        group_num=cfg['group_num'],
        geometry_interval_monitoring=cfg.get(
            'geometry_interval_monitoring'),
        query_monitoring=cfg.get('query_monitoring'),
        iou3d_matching_monitoring=cfg.get(
            'iou3d_matching_monitoring'),
        high_iou_unmatched_negative_weighting=cfg.get(
            'high_iou_unmatched_negative_weighting'),
        iou_quality_head=cfg.get('iou_quality_head'),
        iou_classification=cfg.get('iou_classification'),
        use_vectorized_ddn_rasterization=cfg.get(
            'use_vectorized_ddn_rasterization', False),
        use_aligned_giou_loss=cfg.get('use_aligned_giou_loss', False),
        use_post_match_cache=cfg.get('use_post_match_cache', False)
        )

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
