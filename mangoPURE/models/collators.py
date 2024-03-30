from ..data.base import MixedExample, TimedAudioBatch
from ..data.utils import Resize
from .base import MixedToTimedCollator
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor
from ..globals import RATE


@dataclass
class WhisperToTimedBatch(MixedToTimedCollator):
    feature_extractor: WhisperFeatureExtractor

    def __call__(self, batch_list: list[MixedExample]) -> TimedAudioBatch:
        audio_list = list(map(lambda x: x.audio, batch_list))
        lengths = [x.shape[0] for x in audio_list]
        start_size = lengths[0]
        noises_info = None if batch_list[0].noises_info is None \
            else list(map(lambda x: x.noises_info, batch_list))
        speakers_info = None if batch_list[0].speakers_info is None \
            else list(map(lambda x: x.speakers_info, batch_list))
        audio_features = self.feature_extractor(list(map(lambda x: x.numpy(), audio_list)),
                                                return_tensors="pt", return_attention_mask=True, sampling_rate=RATE)
        audio_features_batch = self.feature_extractor.pad(audio_features)
        end_size = audio_features_batch["attention_mask"][0].sum().tolist()

        batch = TimedAudioBatch(
            audio=audio_features_batch["input_features"],
            lengths=lengths,
            noises_info=noises_info,
            speakers_info=speakers_info
        )
        batch = Resize.resize_timed_audio_batch(batch, start_size, end_size)
        return batch
