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

def sample_anchors(labels, num_samples, positive_fraction):
    # Separating 1, 0 labels
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    # Calculating the number of positive and negatives
    num_pos_avail = pos_mask.sum().item()
    num_neg_avail = neg_mask.sum().item()
    # Calculating available positive and negatives for loss function
    num_pos_wanted = int(num_samples * positive_fraction)
    num_pos_wanted = min(num_pos_wanted, num_pos_avail)   # cannot exceed available
    num_neg_wanted = num_samples - num_pos_wanted
    num_neg_wanted = min(num_neg_wanted, num_neg_avail)   # cannot exceed available
    # Get the indices of all positive and negative anchors
    pos_indices = torch.where(pos_mask)[0] 
    neg_indices = torch.where(neg_mask)[0]
    # Selecting positive anchors randomly 
    pos_perm = torch.randperm(num_pos_avail)[:num_pos_wanted]
    selected_pos = pos_indices[pos_perm] 
    # Selecting negative anchors randomly 
    neg_perm = torch.randperm(num_neg_avail)[:num_neg_wanted]
    selected_neg = neg_indices[neg_perm]
    # Combining selected negatives and positive anchors
    selected = torch.cat([selected_pos, selected_neg])
    # Creates a tensor with all anchors set to False
    sample_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    # Changes selected anchors to True
    sample_mask[selected] = True

    return sample_mask

def delta_decoder(anchors, deltas):
    # Calculating anchors real centers
    anc_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anc_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    # Calculating anchor height/width
    anc_w = anchors[:, 2] - anchors[:, 0]
    anc_h = anchors[:, 3] - anchors[:, 1]
    
    pred_cx = anc_cx + deltas[:, 0] * anc_w
    pred_cy = anc_cy + deltas[:, 1] * anc_h
    pred_w = anc_w * torch.exp(deltas[:, 2])
    pred_h = anc_h * torch.exp(deltas[:, 3])
    x1 = pred_cx - pred_w / 2.0
    y1 = pred_cy - pred_h / 2.0
    x2 = pred_cx + pred_w / 2.0
    y2 = pred_cy + pred_h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)

def delta_encoder(anchors, gt_s):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_s[:, 0], gt_s[:, 1], gt_s[:, 2], gt_s[:, 3]
    # Calculating anchor centers
    anc_cx = (a_x1 + a_x2) / 2
    anc_cy = (a_y1 + a_y2) / 2
    # Calculating anchor height/width
    anc_w = a_x2 - a_x1
    anc_h = a_y2 - a_y1

    # Calculating ground truth centers
    gt_cx = (gt_x1 + gt_x2) / 2
    gt_cy = (gt_y1 + gt_y2) / 2
    # Calculating ground truth height/width
    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1

    # Computing delta for the center shift
    dx = (gt_cx - anc_cx) / anc_w
    dy = (gt_cy - anc_cy) / anc_h
    # Computing delta width and height
    dw = torch.log(gt_w / anc_w)
    dh = torch.log(gt_h / anc_h)

    return torch.stack([dx, dy, dw, dh], dim=1)

def generate_proposal(all_anchors, objectness_scores, deltas):
    # Decoding all the boxes
    decoded_boxes = delta_decoder(all_anchors, deltas) 
    # Creating a mask with the boxes where score is over the threshhold
    keep_mask = objectness_scores > cfg.score_thresh
    # Applying mask to just keep boxes over the threshhold
    decoded_boxes = decoded_boxes[keep_mask]
    scores = objectness_scores[keep_mask]
    # Sorting boxes by score
    sorted_indices = scores.argsort(descending=True)
    decoded_boxes = decoded_boxes[sorted_indices]
    scores = scores[sorted_indices]
    # Keeping pre_nms_top_n 
    if decoded_boxes.shape[0] > cfg.pre_nms_top_n:
        decoded_boxes = decoded_boxes[:cfg.pre_nms_top_n]
        scores = scores[:cfg.pre_nms_top_n]
    # Applying NMS
    keep_indices = torchvision.ops.nms(decoded_boxes, scores, cfg.nms_thresh)
    # Keeping only post_nms_top_n 
    if keep_indices.shape[0] > cfg.post_nms_top_n:
        keep_indices = keep_indices[:cfg.post_nms_top_n]
    proposals = decoded_boxes[keep_indices]
    proposal_scores = scores[keep_indices]
    return (proposals, proposal_scores)
