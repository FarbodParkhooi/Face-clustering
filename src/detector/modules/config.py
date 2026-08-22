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

    # RPN
    anchors_per_image: int = 1024
    label_smoothing: float = 0.1

    # IoU thresholds for matching
    positive_iou_thresh: float = 0.65
    negative_iou_thresh: float = 0.45
    positive_fraction: float = 0.5

    # Pre NMS and NMS
    score_thresh: float = 0.5  # Minimum objectness to keep a box
    pre_nms_top_n: int = 10000
    nms_thresh: float = 0.7
    post_nms_top_n:int = 512

    # Focal Loss
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

    # RoI Head
    roi_canonical_scale: float = 224.0
    roi_canonical_level: int   = 4
    roi_pool_size: int = 7
    roi_hidden_dim: int = 256
    roi_batch_size: int = 256
    roi_score_threshold: float = 0.05

    # Regression
    reg_loss_weight: float = 1.0

    # Augmentation
    # Random scale jitter
    aug_scale_min: float = 420.0
    aug_scale_max: float = 2048.0
    # Random crop
    aug_crop_scale_min: float = 0.3
    aug_crop_scale_max: float = 1.0
    aug_crop_aspect_min: float = 0.4
    aug_crop_aspect_max: float = 2.3
    aug_crop_keep_iou: float = 0.6
    # Random horizontal flip
    aug_flip_prob: float = 0.5
    # Random color jitter
    aug_color_jitter_prob: float = 0.3
    aug_brightness: float = 0.2 
    aug_contrast: float = 0.2 
    aug_saturation: float = 0.2 
    aug_hue: float = 0.1 
    # Random Blur
    aug_blur_prob: float = 0.3
