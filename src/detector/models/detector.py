from torch import nn
from models.backbone import backbone
from models.panet import PANet
from models.rpn import RPN
from models.roi_head import RoIHead
from modules.config import DetectorConfigs

cfg = DetectorConfigs()

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone()   # ResNeXt feature extractor
        self.panet = PANet()         # Feature pyramid neck
        self.rpn = RPN()             # Region proposal network
        self.roi_head = RoIHead()    # Second‑stage classification/regression

    def forward(self, images, gt_boxes=None):
        # Extract multi‑scale features from backbone
        c_features = self.backbone(images)   # dict with C2–C5

        # Build enriched pyramid features via PANet
        pyramid = self.panet(c_features)     # dict with P2–P5 (and P6 if enabled)

        # RPN forward
        if gt_boxes is not None:
            # Training mode: RPN returns proposals and its losses
            # Assumption: batch size = 1 for simplicity (can be extended later)
            proposals, rpn_losses = self.rpn(pyramid, gt_boxes[0])
        else:
            # Inference mode: RPN returns proposals only
            proposals = self.rpn(pyramid)

        # RoI head forward
        if gt_boxes is not None:
            # Training: pass proposals + GT boxes for the single image
            roi_losses = self.roi_head(pyramid, proposals, gt_boxes[0])
            # Combine losses
            total_loss = {
                "rpn_cls": rpn_losses["cls_loss"],
                "rpn_reg": rpn_losses["reg_loss"],
                "roi_cls": roi_losses["cls_loss"],
                "roi_reg": roi_losses["reg_loss"],
            }
            return total_loss
        else:
            # Inference: get final detections
            boxes, scores = self.roi_head(pyramid, proposals)
            return boxes, scores