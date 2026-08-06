from dataclasses import dataclass, field

@dataclass(frozen=True)
class DetectorConfigs():
    # Backbone
    backbone_arch:str = "resnext101_32x8d"
    backbone_pretrained:bool = True

    # PANet
    fpn_channels:int = 512
    use_p6:bool = True 
