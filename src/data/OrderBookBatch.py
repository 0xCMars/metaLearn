from dataclasses import dataclass
from typing import List
import torch

from torch import Tensor

@dataclass
class OrderBookSample:
    data: Tensor
    label: Tensor

class OrderBookBatch:
    def __init__(self, orderbooks: List[OrderBookSample]):
        self.labels = torch.tensor([orderbook.label for orderbook in orderbooks])
        self.data = []
        for orderbook in orderbooks:
            self.data.append(orderbook.data)
        self.data = torch.stack(self.data)
        self.sz = len(orderbooks)

    def __len__(self):
        return self.sz

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.data.to(device)
