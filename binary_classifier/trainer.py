from pathlib import Path

import mlflow
import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, n_epochs, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = n_epochs

    def test(self, loader):
        loss_log = []
        acc_log = []
        self.model.eval()

        for data, target in loader:
            predictions = self.model(data)
            loss = self.criterion(predictions, target)
            loss_log.append(loss.item())

            acc = torch.sum(predictions.argmax(dim=1) == target) / predictions.shape[0]
            acc_log.append(acc.item())

        return np.mean(loss_log), np.mean(acc_log)

    def train_epoch(self, train_loader):
        loss_log = []
        acc_log = []
        self.model.train()

        for data, target in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.criterion(predictions, target)
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())

            acc = torch.sum(predictions.argmax(dim=1) == target) / predictions.shape[0]
            acc_log.append(acc.item())

        return loss_log, acc_log

    def train(self, data):
        train_loader = data.train_loader
        val_loader = data.val_loader
        train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

        for epoch in tqdm(range(self.n_epochs)):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.test(val_loader)

            train_loss_log.extend(train_loss)
            train_acc_log.extend(train_acc)

            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            mlflow.log_metric("train loss", np.mean(train_loss), step=epoch)
            mlflow.log_metric("val loss", val_loss, step=epoch)
            mlflow.log_metric("train acc", np.mean(train_acc), step=epoch)
            mlflow.log_metric("val acc", val_acc, step=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

        return train_loss_log, train_acc_log, val_loss_log, val_acc_log

    def save_model(self, dir_name, file_name):
        path = Path(dir_name) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, path)
