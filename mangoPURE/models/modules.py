from .base import Embedder, TimedHead, Loss, TimedModel, SoloHead
from transformers import WhisperModel
import torch
from typing import Dict, Union
from .external.soundnet import SoundNetRaw


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

    def forward(self, batch: Dict) -> Dict:
        batch = self.embedder(batch)
        batch = self.head(batch)
        loss = self.loss_fn(batch)
        batch["loss"] = loss
        return {'loss': loss, 'head_output': batch['head_output'], 'labels': batch['labels']}


class SoundNet(torch.nn.Module):
    def __init__(
            self,
            num_classes: int,
            audio_len: int,
            hardcode_len: int,
    ):
        """
        :param num_classes: number of classes to predict
        :param hardcode_len: hardcoded length of transformer input dimension
        """
        self.audio_len = audio_len
        self.net = SoundNetRaw(clip_length=hardcode_len, n_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_audios: torch.Tensor, labels: torch.Tensor) -> Dict:
        assert input_audios.shape[1] == self.audio_len, f"The input to the model should be fixed: {self.audio_len}"
        x = torch.unsqueeze(input_audios, dim=1)
        y = self.net.forward(x)
        loss = self.loss(y, labels)
        return {"loss": loss, "logits": y}
