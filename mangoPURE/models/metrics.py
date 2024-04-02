from .base import TimedLoss
import torch
from typing import Dict


class SigmoidTimedLoss(TimedLoss):
    def __call__(self, batch: Dict) -> torch.Tensor:
        """
        :param batch: dictionary with params:
            "attention_mask" - the mask of audio embeddings
            "labels" - true labels of the model
            "head_output" - last output of the model
        :return: binary cross entropy loss
        """
        mask = batch["attention_mask"]
        loss_sum = mask * torch.nn.BCEWithLogitsLoss()(batch["head_output"], batch["labels"])
        loss = loss_sum.sum() / mask.sum()
        return loss
