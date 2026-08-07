from modules.config import DetectorConfigs
from torch import nn
import torchvision
import torch

cfg = DetectorConfigs()

class RPN_Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnv1 = nn.Conv2d(in_channels=cfg.fpn_channels, out_channels=cfg.rpn_conv_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.cnv2 = nn.Conv2d(in_channels=cfg.rpn_conv_channels, out_channels=cfg.num_anchors, kernel_size=1, stride=1, padding=0) 
        self.cnv3 = nn.Conv2d(in_channels=cfg.rpn_conv_channels, out_channels=cfg.num_anchors*4, kernel_size=1, stride=1, padding=0) # (dx, dy, dw, dh per anchor)

    def forward(self, x):
        x = self.cnv1(x)
        x = self.relu(x)

        objectness_logits = self.cnv2(x)
        regression_deltas = self.cnv3(x)

        return (objectness_logits, regression_deltas)

def generate_anchors(level_name:str, featurs:tuple, stride:int):
    y_centers, x_centers, anchors = [], [], []
    base_scale = cfg.base_scales[level_name]
    # Calculating centers
    for i in range(featurs[0]):  y_centers.append((i + 0.5)*stride)
    for j in range(featurs[1]):  x_centers.append((j + 0.5)*stride)
    # Calculating width/height for each anchor
    for y_center in y_centers:
        for x_center in x_centers:
            for scale in cfg.scales:
                for aspect_ratio in cfg.aspect_ratios:
                    anchor_width  = base_scale * scale * (aspect_ratio**0.5)
                    anchor_height = base_scale * scale / (aspect_ratio**0.5)
                    # Calculating the points
                    x1 = x_center - anchor_width  / 2
                    y1 = y_center - anchor_height / 2
                    x2 = x_center + anchor_width  / 2
                    y2 = y_center + anchor_height / 2
                    # Adding output positions
                    anchors.append((x1, y1, x2, y2))
    # Returning all anchors
    return torch.tensor(anchors, dtype=torch.float32)

def id_anchors(pyramid_features):
    all_anchors = []
    level_ids = []
    for level_name, feat_map in pyramid_features.items():
        _, _, H, W = feat_map.shape
        stride = cfg.pyramid_strides[level_name]
        anchors = generate_anchors(level_name, (H, W), stride) 
        all_anchors.append(anchors)
        level_ids.append(torch.full((anchors.shape[0],), cfg.level_to_index[level_name], dtype=torch.long))
    return torch.cat(all_anchors, dim=0), torch.cat(level_ids, dim=0)

def match_anchors_to_gt(anchors, gt_boxes):
    IoUs = torchvision.ops.box_iou(anchors, gt_boxes) # shape (N, M)
    max_iou_per_anchor, matched_gt_idx = IoUs.max(dim=1)   # both shape (N,)
    # Creating the labels
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long)
    # Applying the threshold for labels
    labels[max_iou_per_anchor > cfg.rpn_positive_iou_thresh] = 1
    labels[max_iou_per_anchor < cfg.rpn_negative_iou_thresh] = 0
    # Find the best anchors with the highest IoU
    best_anchor_iou, best_anchor_idx = IoUs.max(dim=0)   # shape (M,)
    # Update matched_gt_idx 
    # positive labels
    labels[best_anchor_idx] = 1
    matched_gt_idx[best_anchor_idx] = torch.arange(len(gt_boxes), device=anchors.device)
    # ignored labels
    matched_gt_idx[labels != 1] = -1

    return (labels, matched_gt_idx)
