from typing import List
from torch import float64
from dataclasses import dataclass

@dataclass
class Order:
    price: float64
    size: float64

@dataclass
class OrderBook:
    bids: List
    asks: List