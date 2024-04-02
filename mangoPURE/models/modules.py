from .base import Embedder, TimedHead, TimedLoss, TimedModel
from transformers import WhisperModel
import torch
from typing import Dict


class WhisperEmbedder(Embedder):
    def __init__(self, hf_name: str):
        super().__init__()
        self.model = WhisperModel.from_pretrained(hf_name).encoder

    def forward(self, batch: Dict) -> Dict:
        """
        :param batch: dictionary that should have key "input_features"
            that whisper encoder takes
        :return: batch with whisper embedder output ("embeddings" ket in batch)
        """
        batch["embeddings"] = self.model(input_features=batch["input_features"]).last_hidden_state
        del batch["input_features"]
        return batch


class LinearTimedHead(TimedHead):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        self.model = torch.nn.Linear(input_dim, num_classes)

    def forward(self, batch: Dict) -> Dict:
        """
        :param batch: dictionary that should have key "embeddings",
            embedder module usually provide this data
        :return: batch with head output ("head_output" key in batch)
        """
        batch["head_output"] = self.model(batch["embeddings"])
        del batch["embeddings"]
        return batch


class WhisperTimedModel(TimedModel):
    def __init__(
            self,
            embedder: WhisperEmbedder,
            head: TimedHead,
            loss_fn: TimedLoss,
    ):
        super().__init__(embedder, head, loss_fn)

    def forward(self, batch: Dict) -> Dict:
        batch = self.embedder(batch)
        batch = self.head(batch)
        loss = self.loss_fn(batch)
        batch["loss"] = loss
        return batch
