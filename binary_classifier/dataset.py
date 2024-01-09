import zipfile
from pathlib import Path

import torch
from dvc.repo import Repo
from torchvision import datasets, transforms


class Dataset:
    def __init__(
        self, size_h: int, size_w: int, data_path_name: str, stage: str
    ) -> None:
        self.stage = stage
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.size_h = size_h
        self.size_w = size_w
        self.transformer = transforms.Compose(
            [
                transforms.Resize((self.size_h, self.size_w)),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )
        self.data_path = Path(data_path_name)
        if stage == "train":
            repo = Repo(".")
            repo.pull()
            with zipfile.ZipFile(f"{data_path_name}.zip", "r") as zip_ref:
                zip_ref.extractall(".")

    def load_train_dataset(self, batch_size):
        self.train_dataset = datasets.ImageFolder(
            self.data_path / "train", transform=self.transformer
        )
        self.val_dataset = datasets.ImageFolder(
            self.data_path / "val", transform=self.transformer
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size
        )

    def load_test_dataset(self, batch_size):
        self.test_dataset = datasets.ImageFolder(
            self.data_path / "test", transform=self.transformer
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
