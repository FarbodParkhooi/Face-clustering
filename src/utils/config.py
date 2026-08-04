from dataclasses import dataclass, field

# Creating frozen class for constant configs
@dataclass(frozen=True)
class Detector():
    pass

@dataclass(frozen=True)
class Embedding():
    pass

@dataclass(frozen=True)
class Clustering():
    pass
