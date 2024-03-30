from .base import TimedModelWrapper, TimedHead, TimedLoss, ModelOutput
from .modules import WhisperEmbedder
from ..data.base import TimedAudioBatch


class WhisperTimedWrapper(TimedModelWrapper):
    def __init__(
            self,
            embedder: WhisperEmbedder,
            head: TimedHead,
            loss_fn: TimedLoss,
    ):
        super().__init__(embedder, head, loss_fn)

    def forward(self, batch: TimedAudioBatch) -> ModelOutput:
        batch = self.embedder(batch)
        batch = self.head(batch)
        loss = self.loss_fn(batch)
        return ModelOutput(
            loss=loss,
        )
