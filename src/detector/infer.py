# infer.py

import os
import json
import cv2
import torch
import torchvision
import numpy as np

from modules.config import DetectorConfigs
from models.detector import Detector

def main():
    cfg = DetectorConfigs()

    # Pick device with fallback
    if cfg.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model = Detector()
    model.load_state_dict(torch.load(cfg.final_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare output directories
    os.makedirs(cfg.infer_output_dir, exist_ok=True)
    crops_dir = os.path.join(cfg.infer_output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    metadata = []

    # Iterate over input images
    for fname in os.listdir(cfg.infer_image_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(cfg.infer_image_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = image.astype(np.float32) / 255.0
        img = (img - mean) / std
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Forward
        with torch.no_grad():
            boxes, scores = model(img_tensor)

        if boxes is None or len(boxes) == 0:
            continue

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        # Score filter
        keep = scores >= cfg.infer_score_threshold
        boxes, scores = boxes[keep], scores[keep]

        # Minimum size filter
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= cfg.infer_min_face_size) & (heights >= cfg.infer_min_face_size)
        boxes, scores = boxes[keep], scores[keep]

        # Optional second NMS pass
        if len(boxes) > 0:
            keep_idx = torchvision.ops.nms(
                torch.from_numpy(boxes).float(),
                torch.from_numpy(scores).float(),
                cfg.infer_nms_threshold
            ).numpy()
            boxes, scores = boxes[keep_idx], scores[keep_idx]

        # Save crops and metadata
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(orig_w, x2); y2 = min(orig_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (cfg.infer_crop_size, cfg.infer_crop_size))
            crop_name = f"{os.path.splitext(fname)[0]}_face{i}.jpg"
            crop_path = os.path.join(crops_dir, crop_name)
            cv2.imwrite(crop_path, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

            metadata.append({
                "image": fname,
                "crop": crop_name,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score)
            })

    # Save metadata
    with open(cfg.infer_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Inference complete. {len(metadata)} faces saved to {crops_dir}")


if __name__ == "__main__":
    main()  