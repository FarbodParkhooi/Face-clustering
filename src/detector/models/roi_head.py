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
