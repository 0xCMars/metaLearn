from os import cpu_count

import numpy as np
from omegaconf import DictConfig
from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from src.data.OrderBookBatch import OrderBookSample, OrderBookBatch
from src.data.datasets import OrderBookDateset


class OrderbookDataModule(LightningDataModule):
    def __init__(self, config, data, look_back=2):
        super().__init__()
        self.__config = config
        self.__data = data
        self.__look_back = look_back
        self.__n_workers = cpu_count() if self.__config.num_workers == -1 else self.__config.num_workers

    @staticmethod
    def collate_wrapper(batch: List[OrderBookSample]) -> OrderBookBatch:
        return OrderBookBatch(batch)

    def __create_dataset(self, dataset: np.ndarray, look_back, val=False) -> Dataset:
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i: (i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back][0])
        train_size = int(len(dataX) * 0.7)
        if not val:
            dataX = torch.tensor(dataX[:train_size]).float()
            dataY = torch.tensor(dataY[:train_size]).float()
        else:
            dataX = torch.tensor(dataX[train_size:]).float()
            dataY = torch.tensor(dataY[train_size:]).float()
        return OrderBookDateset(dataX, dataY)

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.__create_dataset(self.__data, self.__look_back)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameter.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.__create_dataset(self.__data, self.__look_back, val=True)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.__create_dataset(self.__data, self.__look_back, val=True)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
            self,
            batch: OrderBookBatch,
            device: Optional[torch.device] = None,
            dataloader_idx = None
    ) -> OrderBookBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch