from .base import Embedder, TimedHead, Loss, TimedModel, SoloHead
from transformers import WhisperModel
import torch
from typing import Dict, Union


class WhisperEmbedder(Embedder):
    def __init__(self, hf_name: str):
        super().__init__()
        self.model = WhisperModel.from_pretrained(hf_name).encoder

    def forward(self, batch) -> Dict:
        """
        :param batch: dictionary that should have key "input_features"
            that whisper encoder takes
        :return: batch with whisper embedder output ("embeddings" ket in batch)
        """
        batch["embeddings"] = self.model(input_features=batch["input_features"]).last_hidden_state
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
        return batch


class LinearSoloHead(SoloHead):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        self.model = torch.nn.Linear(input_dim, num_classes)

    def forward(self, batch: Dict) -> Dict:
        batch['head_output'] = self.model(batch['embeddings'].mean(dim=1))
        return batch


class WhisperTimedModel(TimedModel):
    def __init__(
            self,
            embedder: WhisperEmbedder,
            head: Union[TimedHead, SoloHead],
            loss_fn: Loss,
    ):
        super().__init__(embedder, head, loss_fn)

    def forward(self, batch) -> Dict:
        batch = self.embedder(batch)
        batch = self.head(batch)
        loss = self.loss_fn(batch)
        batch["loss"] = loss
        return {'loss': loss, 'head_output': batch['head_output'], 'labels': batch['labels']}
