from models.rpn import delta_decoder
from torch import nn
import torchvision
import torch 

def assign_levels(proposals, canonical_scale, canonical_level, min_level, max_level):
    x1, y1, x2, y2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    w = x2 - x1
    h = y2 - y1

    area_sqrt = torch.sqrt(w*h)
    level_float = canonical_level + torch.log2(area_sqrt / canonical_scale)

    level = torch.floor(level_float)
    level = level.clamp(min=min_level, max=max_level)

    return level.long() 

def ciou_loss(pred_boxes, gt_boxes):
    # center‑size conversion for predicted boxes
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
    pred_w  = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h  = pred_boxes[:, 3] - pred_boxes[:, 1]
    # center‑size conversion for ground-truth boxes
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_w  = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h  = gt_boxes[:, 3] - gt_boxes[:, 1] 
    center_dist = (pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2
    # Finding coordinates of intersection rectangle 
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    # Computing area of predicted and ground truth 
    pred_area = pred_w * pred_h
    gt_area   = gt_w * gt_h
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-7)
    # Coordinates of the smallest rectangle that completely covers both boxes
    encl_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    encl_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    encl_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    encl_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    encl_diag = (encl_x2 - encl_x1)**2 + (encl_y2 - encl_y1)**2 + 1e-7
    # Converting aspect ration to angle 
    pred_angle = torch.atan(pred_w / (pred_h + 1e-7))
    gt_angle   = torch.atan(gt_w   / (gt_h + 1e-7))
    v = (4 / (torch.pi**2)) * ((gt_angle - pred_angle)**2)
    # Computing a balancing weight (alpha) that depends on IoU and v
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    # Complete CIoU score
    ciou = iou - (center_dist / encl_diag) - (alpha * v)
    loss = 1 - ciou
    return loss.mean()
