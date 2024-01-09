from binary_classifier import Dataset, BasicBlockNet, Trainer
import torch
from torch import nn
import torch.optim as optim
import hydra
from hydra.core.config_store import ConfigStore
from config import ModelParams
import mlflow


cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelParams)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: ModelParams):
    data = Dataset(cfg.image_params.size_h, cfg.image_params.size_w,
                   cfg.path.data_path, 'train')
    data.load_train_dataset(cfg.train_params.batch_size)
    model = BasicBlockNet(cfg.train_params.n_classes,
                          cfg.image_params.size_h,
                          cfg.image_params.size_w)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.optim_params.lr,
                          momentum=cfg.optim_params.momentum)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, cfg.train_params.n_epoch)
    print(trainer.train(data))


if __name__ == '__main__':
    main()
