# roi_head.py

from models.rpn import delta_decoder
from modules.config import DetectorConfigs
from torch import nn
import torchvision
import torch

cfg = DetectorConfigs()

def assign_levels(proposals, canonical_scale, canonical_level, min_level, max_level):
    x1, y1, x2, y2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    w = x2 - x1
    h = y2 - y1

    area_sqrt = torch.sqrt(w * h)
    level_float = canonical_level + torch.log2(area_sqrt / canonical_scale)

    level = torch.floor(level_float)
    level = level.clamp(min=min_level, max=max_level)
    return level.long()


def ciou_loss(pred_boxes, gt_boxes):
    # center‑size conversion
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
    pred_w  = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h  = pred_boxes[:, 3] - pred_boxes[:, 1]

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_w  = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h  = gt_boxes[:, 3] - gt_boxes[:, 1]

    center_dist = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    # Intersection over Union 
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = pred_w * pred_h
    gt_area   = gt_w * gt_h
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # enclosing box diagonal
    encl_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    encl_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    encl_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    encl_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    encl_diag = (encl_x2 - encl_x1) ** 2 + (encl_y2 - encl_y1) ** 2 + 1e-7

    # aspect‑ratio penalty
    pred_angle = torch.atan(pred_w / (pred_h + 1e-7))
    gt_angle   = torch.atan(gt_w   / (gt_h + 1e-7))
    v = (4 / (torch.pi ** 2)) * ((gt_angle - pred_angle) ** 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    #  final CIoU and loss 
    ciou = iou - (center_dist / encl_diag) - (alpha * v)
    loss = 1 - ciou
    return loss.mean()


class RoIHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_size = cfg.roi_pool_size
        hidden_dim = cfg.roi_hidden_dim
        in_dim = cfg.fpn_channels * self.pool_size * self.pool_size

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, 1)    # face / background logit
        self.bbox_pred = nn.Linear(hidden_dim, 4)    # (dx, dy, dw, dh)

    def forward(self, pyramid_features, proposals, gt_boxes=None):
        device = proposals.device

        # Assign each proposal to a pyramid level
        levels = assign_levels(
            proposals,
            cfg.roi_canonical_scale,
            cfg.roi_canonical_level,
            min_level=2,
            max_level=5 if not cfg.use_p6 else 6
        )

        # RoIAlign per level, keeping original indices for reordering
        pooled_list = []
        indices_list = []

        for lvl in levels.unique(sorted=True):
            mask = (levels == lvl)
            lvl_proposals = proposals[mask]
            idx = mask.nonzero(as_tuple=True)[0]          # original positions

            feat = pyramid_features[f"P{lvl.item()}"]
            stride = cfg.pyramid_strides[f"P{lvl.item()}"]
            spatial_scale = 1.0 / stride

            pooled = torchvision.ops.roi_align(
                feat,
                [lvl_proposals],
                output_size=self.pool_size,
                spatial_scale=spatial_scale
            )
            pooled_list.append(pooled)
            indices_list.append(idx)

        if len(pooled_list) == 0:
            return None

        # Concatenate and restore original order
        all_pooled = torch.cat(pooled_list, dim=0)          # (total, C, pool, pool)
        all_indices = torch.cat(indices_list, dim=0)        # (total,)
        _, sorted_order = torch.sort(all_indices)
        pooled = all_pooled[sorted_order]                   # align with proposals

        # Fully‑connected layers
        x = pooled.flatten(1)               # (N, C*pool*pool)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        cls_logits = self.cls_score(x).squeeze(1)   # (N,)
        bbox_deltas = self.bbox_pred(x)             # (N, 4)

        # Inference
        if gt_boxes is None:
            scores = torch.sigmoid(cls_logits)
            final_boxes = delta_decoder(proposals, bbox_deltas)

            keep = scores > cfg.roi_score_threshold 
            final_boxes = final_boxes[keep]
            scores = scores[keep]

            keep_idx = torchvision.ops.nms(final_boxes, scores, cfg.nms_thresh)
            return final_boxes[keep_idx], scores[keep_idx]

        # Training
        # Matching
        ious = torchvision.ops.box_iou(proposals, gt_boxes)
        max_iou, matched_gt = ious.max(dim=1)

        labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=device)
        labels[max_iou >= cfg.positive_iou_thresh] = 1
        labels[max_iou < cfg.negative_iou_thresh] = 0

        # Balanced sampling
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        num_pos_avail = pos_mask.sum().item()
        num_neg_avail = neg_mask.sum().item()

        num_pos_wanted = int(cfg.roi_batch_size * cfg.positive_fraction)
        num_pos_wanted = min(num_pos_wanted, num_pos_avail)
        num_neg_wanted = cfg.roi_batch_size - num_pos_wanted
        num_neg_wanted = min(num_neg_wanted, num_neg_avail)

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        pos_perm = torch.randperm(num_pos_avail)[:num_pos_wanted]
        neg_perm = torch.randperm(num_neg_avail)[:num_neg_wanted]
        sampled_pos = pos_indices[pos_perm]
        sampled_neg = neg_indices[neg_perm]

        sampled_indices = torch.cat([sampled_pos, sampled_neg])
        sample_mask = torch.zeros(proposals.shape[0], dtype=torch.bool, device=device)
        sample_mask[sampled_indices] = True

        # Classification loss (focal loss + label smoothing)
        sampled_logits = cls_logits[sample_mask]
        sampled_labels = labels[sample_mask].float()

        pos_target = 1.0 - cfg.label_smoothing
        neg_target = cfg.label_smoothing
        smoothed_targets = torch.where(sampled_labels == 1, pos_target, neg_target)

        cls_loss = torchvision.ops.sigmoid_focal_loss(
            sampled_logits,
            smoothed_targets,
            alpha=cfg.focal_loss_alpha,
            gamma=cfg.focal_loss_gamma,
            reduction='mean'
        )

        # Regression loss (CIoU on positives only)
        pos_sample_mask = (sampled_labels == 1)
        if pos_sample_mask.sum() > 0:
            pos_proposals = proposals[sample_mask][pos_sample_mask]
            pos_gt_idx = matched_gt[sample_mask][pos_sample_mask]
            pos_gt_boxes = gt_boxes[pos_gt_idx]
            pos_pred_deltas = bbox_deltas[sample_mask][pos_sample_mask]

            pos_pred_boxes = delta_decoder(pos_proposals, pos_pred_deltas)
            reg_loss = ciou_loss(pos_pred_boxes, pos_gt_boxes) * cfg.reg_loss_weight
        else:
            reg_loss = torch.tensor(0.0, device=device)

        return {"cls_loss": cls_loss, "reg_loss": reg_loss}