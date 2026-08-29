import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from modules.config import DetectorConfigs
from modules.augment import apply_augmentations

cfg = DetectorConfigs()

class FaceDataset(Dataset):
    def __init__(self, json_path, config=cfg):
        """
        json_path: path to a JSON file containing annotations.
        Each entry must have:
            - "image": path to the image file
            - "boxes": list of [x1, y1, x2, y2]
        """
        super().__init__()
        self.config = config
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        boxes = item["boxes"]   # list of [x1,y1,x2,y2]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert boxes to numpy array of shape (N,4)
        boxes_np = np.array(boxes, dtype=np.float32).reshape(-1, 4)

        # Apply augmentations in order, and keep the returned results
        image, boxes_np = apply_augmentations(image, boxes_np)

        # Convert image to tensor and normalize
        # Normalize using ImageNet mean and std (common for pretrained backbones)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)

        # Convert boxes and labels to tensors
        boxes_tensor = torch.from_numpy(boxes_np)   # (N, 4)
        labels_tensor = torch.ones(boxes_tensor.shape[0], dtype=torch.long)   # all faces

        return image, boxes_tensor, labels_tensor
