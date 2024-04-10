import torch

from ..data.base import MixedExample
from .base import MixedToTimedCollator
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor
from typing import Dict, List
from .utils import Whisper, Labels


@dataclass
class WhisperToTimedBatchConfig:
    create_labels: bool = False
    output_timestamps: int = 1500
    output_features: int = 384
    noise_classes: int = 10
    speaker_classes: int = 10
    diar_type: str = "noise"  # can be "speaker" or "noise"


@dataclass
class WhisperToTimedBatch(MixedToTimedCollator):
    feature_extractor: WhisperFeatureExtractor
    config: WhisperToTimedBatchConfig

    def __call__(self, batch_list: List[MixedExample]) -> Dict:
        """
        Creates dict with "input_features" and "attention_mask"
        if create_labels=True also provides labels for the loss, see utils.Labels.create_labels
        """
        for x in batch_list:
            Whisper.update_mixed_example(x)
        batch = Whisper.extract_features(self.feature_extractor, [x.audio for x in batch_list])

        if self.config.diar_type not in ["noise", "speaker"]:
            raise Exception("diar_type should be either 'noise' or 'speaker'")
        num_classes = self.config.noise_classes if self.config.diar_type == "noise" else self.config.speaker_classes

        batch["attention_mask"] = Labels.resize_attention_mask(
            batch["attention_mask"],
            num_classes,
            self.config.output_timestamps
        )

        if self.config.create_labels:
            if self.config.diar_type == "speaker":
                batch["labels"] = Labels.create_labels([
                    x.speakers_info for x in batch_list],
                    self.config.speaker_classes,
                    self.config.output_timestamps
                )
            if batch_list[0].noises_info is not None:
                batch["labels"] = Labels.create_labels([
                    x.noises_info for x in batch_list],
                    self.config.noise_classes,
                    self.config.output_timestamps
                )
        return batch


@dataclass
class OneNoiseCollator:
    feature_extractor: WhisperFeatureExtractor

    def __call__(self, batch_list: List[MixedExample]) -> Dict:
        """
        Creates dict with "input_features" and "attention_mask"
        if create_labels=True also provides labels for the loss, see utils.Labels.create_labels
        """

        batch = Whisper.extract_features(self.feature_extractor, [x.audio for x in batch_list])

        batch['labels'] = torch.tensor([x.noises_info[0].class_id for x in batch_list])
        return {'batch': batch}


@dataclass
class SoundNetCollator:
    audio_len: int

    def __call__(self, batch_list: List[MixedExample]) -> Dict:
        audio_list = [x.audio for x in batch_list]
        labels = torch.tensor([x.noises_info[0].class_id for x in batch_list])
        audio_tensor = torch.stack(audio_list, dim=0)
        return {"input_audios": audio_tensor, "labels": labels}
