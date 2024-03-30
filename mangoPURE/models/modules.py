from .base import Embedder, TimedHead
from transformers import WhisperModel
from ..data.base import TimedAudioBatch
from ..data.utils import Resize
import torch


class WhisperEmbedder(Embedder):
    def __init__(self, hf_name: str):
        super().__init__()
        self.model = WhisperModel.from_pretrained(hf_name).encoder

    def forward(self, batch: TimedAudioBatch) -> TimedAudioBatch:
        start_size = batch.audio.shape[2]
        batch.audio = self.model(batch.audio).last_hidden_state
        new_size = batch.audio.shape[1]
        batch = Resize.resize_timed_audio_batch(batch, start_size, new_size)
        return batch


class LinearTimedHead(TimedHead):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        self.model = torch.nn.Linear(input_dim, num_classes)

    def forward(self, batch: TimedAudioBatch) -> TimedAudioBatch:
        batch.audio = self.model(batch.audio)
        return batch

