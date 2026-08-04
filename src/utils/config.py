from dataclasses import dataclass, field
import torchvision.models as models

@dataclass(frozen=True)
class DetectorConfig():
    pass

@dataclass(frozen=True)
class EmbeddingConfig():
    pass

@dataclass(frozen=True)
class ClusteringConfig():
    pass
