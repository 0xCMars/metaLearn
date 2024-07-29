from omegaconf import OmegaConf, DictConfig

from argparse import ArgumentParser
from typing import cast
from pytorch_lightning import seed_everything
import pandas as pd

from src.data.datamodule import OrderbookDataModule
from src.models.lstm import lstm_reg
from src.train import train

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser

def getData(dataFile: str):
    data_csv = pd.read_csv(dataFile)
    data_csv = data_csv.drop(data_csv.columns[[0, 1, -1]], axis=1)
    return data_csv.values

def run(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    seed_everything(config.seed, workers=True)

    data = getData(config.dataset.csvData)
    data_module = OrderbookDataModule(config, data)

    # feature = config.model.input_size  # 20喂数据
    # seq_len = config.model.seq_len  # 利用过去两次的bid/ask px 和 size作为输入序列
    # hidden_size = config.model.hidden_size  # 隐藏层
    # num_layers = config.model.num_layers
    # output_size = config.model.output_size  # 输出维度 输出输出的预测的price

    model = lstm_reg(config)

    train(model, data_module, config)



if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    run(__args.config)