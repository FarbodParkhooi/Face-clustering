from modules.config import DetectorConfigs
import numpy as np
import random
import cv2

cfg = DetectorConfigs()

def random_scale_jitter(image, boxes):
    # Get current height and width
    h, w = image.shape[:2]
    # Current shorter side
    short_side = min(h, w)

    # Randomly choose a new shorter side length
    target_short_side = random.uniform(cfg.aug_scale_min, cfg.aug_scale_max)

    # Compute scaling factor
    scale = target_short_side / short_side

    # New width and height after scaling
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize image
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Scale the bounding boxes
    if len(boxes) > 0:
        boxes = boxes * scale

    return image, boxes