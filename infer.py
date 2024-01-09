from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from binary_classifier import Dataset
from config import ModelParams

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelParams)


def predict(model, loader):
    predictions = []
    model.eval()
    for data, _ in tqdm(loader):
        preds = model(data)
        predictions += preds.argmax(dim=1).tolist()
    return predictions


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: ModelParams):
    data = Dataset(
        cfg.image_params.size_h, cfg.image_params.size_w, cfg.path.data_path, "test"
    )
    data.load_test_dataset(cfg.train_params.batch_size)
    path = Path(cfg.path.save_model_dirname) / cfg.path.save_model_filename
    model = torch.load(path)
    predictions = predict(model, data.test_loader)
    df = pd.DataFrame(predictions, columns=["preds"])
    df.to_csv("predictions.csv")


if __name__ == "__main__":
    main()
