from dataclasses import dataclass, field

@dataclass(frozen=True)
class DetectorConfigs():
    backbone_arch:str = "resnext101_32x8d"
    backbone_pretrained:bool = True
