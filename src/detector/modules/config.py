from dataclasses import dataclass, field

@dataclass(frozen=True)
class DetectorConfigs():
    # Backbone
    backbone_arch:str = "resnext101_32x8d"
    backbone_pretrained:bool = True

    # PANet
    fpn_channels:int = 512
    use_p6:bool = True 

    # Anchor
    scales:list[int] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    aspect_ratios:list[int] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    num_anchors:int = len(scales) * len(aspect_ratios)  # 5 * 4 = 20
    rpn_conv_channels:int = 128
    base_scales:dict = {
        "P2" :  4*4,  # Stride * CONSTANT
        "P3" :  8*4,  # Stride * CONSTANT
        "P4" : 16*8,  # Stride * CONSTANT
        "P5" : 32*8,  # Stride * CONSTANT
        "P6" : 64*4   # Stride * CONSTANT
    }
    pyramid_strides:dict = {
        "P2" :  4,
        "P3" :  8,
        "P4" : 16,
        "P5" : 32,
        "P6" : 64
    }
    level_to_index:dict = {
        "P2" : 0,
        "P3" : 1,
        "P4" : 2,
        "P5" : 3,
        "P6" : 4
    }
    anchors_per_image:int = 1024
    rpn_positive_fraction:int = 0.5
    rpn_pre_nms_top_n:int = 10000
    rpn_post_nms_top_n:int = 512

    # CIoU
    rpn_positive_iou_thresh = 0.65
    rpn_negative_iou_thresh = 0.45
