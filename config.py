from dataclasses import dataclass


@dataclass
class TrainParams:
    n_classes: int
    n_epoch: int
    batch_size: int
    
@dataclass
class OptimizerParams:
    lr: float
    momentum: float

@dataclass
class ImageParams:
    size_h: int
    size_w: int
    
@dataclass
class Path:
    data_path: str

@dataclass
class ModelParams:
    train_params: TrainParams
    image_params: ImageParams
    optim_params: OptimizerParams
    path: Path