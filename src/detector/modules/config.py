from dataclasses import dataclass, field

@dataclass(frozen=True)
class DetectorConfigs():
    # Backbone
    backbone_arch: str = "resnext101_32x8d"
    backbone_pretrained: bool = True

    # PANet
    fpn_channels: int = 512
    use_p6: bool = True

    # Anchor
    scales: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    aspect_ratios: list = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    num_anchors: int = 20  # matches len(scales)*len(aspect_ratios)
    rpn_conv_channels: int = 128

    base_scales: dict = field(default_factory=lambda: {
        "P2": 4 * 4,
        "P3": 8 * 4,
        "P4": 16 * 8,
        "P5": 32 * 8,
        "P6": 64 * 4
    })

    pyramid_strides: dict = field(default_factory=lambda: {
        "P2": 4,
        "P3": 8,
        "P4": 16,
        "P5": 32,
        "P6": 64
    })

    level_to_index: dict = field(default_factory=lambda: {
        "P2": 0,
        "P3": 1,
        "P4": 2,
        "P5": 3,
        "P6": 4
    })

    anchors_per_image: int = 1024
    rpn_positive_fraction: float = 0.5          # fixed type: float
    rpn_pre_nms_top_n: int = 10000
    rpn_post_nms_top_n: int = 512

    # IoU thresholds for matching
    rpn_positive_iou_thresh: float = 0.65       # added type annotation
    rpn_negative_iou_thresh: float = 0.45       # added type annotation