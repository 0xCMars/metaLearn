from torch.utils.data import Dataset

from src.data.OrderBookBatch import OrderBookSample
from torch import Tensor


class OrderBookDateset(Dataset):
    def __init__(self, data: Tensor, label: Tensor) -> None:
        self.__data = data  # tensor
        self.__label = label # price
        self.__n_samples = len(data)

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> OrderBookSample:
        return OrderBookSample(data=self.__data[index], label=self.__label[index])
