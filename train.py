from pathlib import Path

import hydra
import mlflow
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from mlflow.utils.git_utils import get_git_commit
from torch import nn

from binary_classifier import BasicBlockNet, Dataset, Trainer
from config import ModelParams

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelParams)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: ModelParams):
    data = Dataset(
        cfg.image_params.size_h, cfg.image_params.size_w, cfg.path.data_path, "train"
    )
    data.load_train_dataset(cfg.train_params.batch_size)
    model = BasicBlockNet(
        cfg.train_params.n_classes, cfg.image_params.size_h, cfg.image_params.size_w
    )
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.optim_params.lr, momentum=cfg.optim_params.momentum
    )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, cfg.train_params.n_epoch)

    mlflow.set_tracking_uri(uri=cfg.mlflow_params.url)
    mlflow.set_experiment(cfg.mlflow_params.name)

    with mlflow.start_run():
        mlflow.log_params(cfg.train_params)
        mlflow.log_param("git_commit", get_git_commit(Path.cwd()))
        trainer.train(data)

    trainer.save_model(cfg.path.save_model_dirname, cfg.path.save_model_filename)


if __name__ == "__main__":
    main()
