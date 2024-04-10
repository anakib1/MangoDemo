import torch
from ..data.base import MixedExample
from typing import Dict


class Embedder(torch.nn.Module):
    def forward(self, batch: Dict) -> Dict:
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

    def forward(self, batch: Dict) -> Dict:
        """
        The model should get output from embedder (usually transformer) of the shape (B x T x F_in)
            and transform it to (B x T x NUM_CLASSES) where the last dimension is something like sigmoid or softmax
        """
        pass


class SoloHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, batch: Dict) -> Dict:
        """
        The model transforms (B X T X F_in) into (B x NUM_CLASSES).
        :param batch:
        :return:
        """

    pass


class Loss:
    def __call__(self, batch: Dict) -> torch.Tensor:
        pass


class MixedToTimedCollator:
    def __call__(self, batch_list: list[MixedExample]) -> Dict:
        pass


class TimedModel(torch.nn.Module):
    def __init__(
            self,
            embedder: Embedder,
            head: TimedHead,
            loss_fn: Loss,
    ):
        super().__init__()
        self.embedder = embedder
        self.head = head
        self.loss_fn = loss_fn


class SoloModel(torch.nn.Module):
    def __init__(self, embedder: Embedder, head: TimedHead, loss_fn: Loss):
        super().__init__()
        self.embedder = embedder
        self.head = head
        self.loss_fn = loss_fn
