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

def random_crop(image, boxes):
    # Get image dimensions
    h, w = image.shape[:2]
    original_area = h * w

    # Randomly choose crop scale and aspect ratio
    crop_scale = random.uniform(cfg.aug_crop_scale_min, cfg.aug_crop_scale_max)
    crop_aspect = random.uniform(cfg.aug_crop_aspect_min, cfg.aug_crop_aspect_max)

    # Compute crop width and height from area and aspect ratio
    crop_area = original_area * crop_scale
    crop_w = int(round(np.sqrt(crop_area * crop_aspect)))
    crop_h = int(round(np.sqrt(crop_area / crop_aspect)))

    # Ensure crop fits inside the image
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    # Randomly choose top-left corner
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Perform the crop
    cropped_image = image[y1:y2, x1:x2]

    # Adjust bounding boxes
    if len(boxes) == 0:
        return cropped_image, boxes

    # Copy boxes so we can modify them
    new_boxes = []
    crop_box = np.array([x1, y1, x2, y2])  # for IoU calculation

    for box in boxes:
        bx1, by1, bx2, by2 = box

        # Compute original box area
        box_area = (bx2 - bx1) * (by2 - by1)
        if box_area <= 0:
            continue

        # Compute intersection with crop
        inter_x1 = max(bx1, x1)
        inter_y1 = max(by1, y1)
        inter_x2 = min(bx2, x2)
        inter_y2 = min(by2, y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Keep only if a large enough portion of the face remains
        if inter_area / box_area < cfg.aug_crop_keep_iou:
            continue

        # Clip box to crop boundaries and shift coordinates
        new_x1 = max(bx1, x1) - x1
        new_y1 = max(by1, y1) - y1
        new_x2 = min(bx2, x2) - x1
        new_y2 = min(by2, y2) - y1

        new_boxes.append([new_x1, new_y1, new_x2, new_y2])

    return cropped_image, np.array(new_boxes)
