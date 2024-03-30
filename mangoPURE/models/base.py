import torch
from ..data.base import TimedAudioBatch, MixedExample
from typing import Any
from dataclasses import dataclass


class Embedder(torch.nn.Module):
    def forward(self, batch: TimedAudioBatch) -> TimedAudioBatch:
        """
        The model (transformer usually) should get the input of shape (B x T_in x F_in)
            and return the tensor of (B x T_out x F_out)
        Also the function handles the metadata in the batch
        """
        pass


class TimedHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, batch: TimedAudioBatch) -> TimedAudioBatch:
        """
        The model should get output from embedder (usually transformer) of the shape (B x T x F_in)
            and transform it to (B x T x NUM_CLASSES) where the last dimension is something like sigmoid or softmax
        """
        pass


class TimedLoss:
    def __call__(self, batch: TimedAudioBatch) -> torch.Tensor:
        pass


class MixedToTimedCollator:
    def __call__(self, batch_list: list[MixedExample]) -> TimedAudioBatch:
        pass


@dataclass
class ModelOutput:
    loss: torch.Tensor
    metadata: dict[str, Any] = None


class TimedModelWrapper(torch.nn.Module):
    def __init__(
            self,
            embedder: Embedder,
            head: TimedHead,
            loss_fn: TimedLoss,
    ):
        super().__init__()
        self.embedder = embedder
        self.head = head
        self.loss_fn = loss_fn

    def forward(self, batch: TimedAudioBatch) -> ModelOutput:
        """
        This function should get binput batch and output loss for training
            and other metadata for computing metrics
        """
