from .base import TimedLoss
import torch
from ..data.utils import TimedAudioBatch


class SigmoidTimedLoss(TimedLoss):
    def __init__(self, info_type: str = "noise"):
        """
        :param info_type: ["noise", "speaker"] selects which metadata from batch to use
        """
        self.info_type = info_type

    def __call__(self, batch: TimedAudioBatch) -> torch.Tensor:
        mask = torch.zeros(batch.audio.shape)
        for i, length in enumerate(batch.lengths):
            mask[i, :length, :] = 1.0
        labels = torch.zeros(batch.audio.shape)
        labels[:, :, 0] = 1.0
        batch_segments = None
        if self.info_type == "noise":
            batch_segments = batch.noises_info
        elif self.info_type == "speaker":
            batch_segments = batch.speakers_info
        assert batch_segments is not None, "segments is None for some reason"
        for i, segments in enumerate(batch_segments):
            for segment in segments:
                labels[i, segment.start:segment.start + segment.length, 0] = 0.0
                labels[i, segment.start:segment.start + segment.length, segment.class_id] = 1.0
        loss = mask * torch.nn.BCEWithLogitsLoss()(batch.audio, labels)
        return loss.sum() / mask.sum()
