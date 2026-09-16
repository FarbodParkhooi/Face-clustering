# backbone.py

from modules.config import DetectorConfigs
from torchvision import models
from torch import nn

cfg = DetectorConfigs()

class backbone(nn.Module):
    def __init__(self):
        super().__init__()

        if cfg.backbone_arch == "resnext101_32x8d":
            # Loading the model
            if cfg.backbone_pretrained:
                self.model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
            else:
                self.model = models.resnext101_32x8d(weights=None)
            # Removing classification heads
            self.model.avgpool = nn.Identity()
            self.model.fc = nn.Identity()

        self.cnv1 = self.model.conv1 
        self.btn1 = self.model.bn1
        self.relu = self.model.relu
        self.mxpl = self.model.maxpool

        self.lyr1 = self.model.layer1
        self.lyr2 = self.model.layer2
        self.lyr3 = self.model.layer3
        self.lyr4 = self.model.layer4

    def forward(self, x):
        x = self.cnv1(x)
        x = self.btn1(x)
        x = self.relu(x)
        x = self.mxpl(x)

        layer1_feat = self.lyr1(x)
        layer2_feat = self.lyr2(x)
        layer3_feat = self.lyr3(x)
        layer4_feat = self.lyr4(x)        

        return {
            "C2" : layer1_feat,  # Stride  4
            "C3" : layer2_feat,  # Stride  8 
            "C4" : layer3_feat,  # Stride 16
            "C5" : layer4_feat   # Stride 32
        }
