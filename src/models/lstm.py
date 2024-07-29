import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, SGD, Adamax, RMSprop
from typing import Dict

from src.data.OrderBookBatch import OrderBookBatch
import torch.nn.functional as F
from src.utils.metrics import Statistic


class lstm_reg(LightningModule):
    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.__config = config
        self.input_size = config.model.input_size
        self.hidden_size = config.model.hidden_size
        self.num_layers = config.model.num_layers
        self.output_size = config.model.output_size
        self.num_directions = 1  # 单向LSTM
        #         self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))  # output
        #         print("output is",output.shape)
        #         print(output[0])

        pred = self.linear(output)
        #         print("pred is",pred.shape)
        #         print(pred[0])

        pred = pred[:, -1, :]
        #         print("pred2 is",pred.shape)
        #         print(pred[0])

        return pred

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        parameters = [self.parameters()]
        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in parameters],
            self.__config.hyper_parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)


    def training_step(self, batch: OrderBookBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: OrderBookBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: OrderBookBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)

        return {"loss": loss, "statistic": statistic}

        # ========== EPOCH END ==========
        def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                                   step: str) -> Dict[str, torch.Tensor]:
            with torch.no_grad():
                losses = [
                    so if isinstance(so, torch.Tensor) else so["loss"]
                    for so in step_outputs
                ]
                mean_loss = torch.stack(losses).mean()
            return {f"{step}_loss": mean_loss}

        def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
            log = self._prepare_epoch_end_log(step_outputs, group)
            statistic = Statistic.union_statistics(
                [out["statistic"] for out in step_outputs])
            log.update(statistic.calculate_metrics(group))
            self.log_dict(log, on_step=False, on_epoch=True)

        def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
            self._shared_epoch_end(training_step_output, "train")

        def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
            self._shared_epoch_end(validation_step_output, "val")

        def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
            self._shared_epoch_end(test_step_output, "test")
