from dataclasses import dataclass

from transformers import WhisperProcessor, Wav2Vec2Processor
import torch
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
from .SpeakerAttributedMixer import SpeakerAttributeExample
from .DatasetMixer import MixedExample
from typing import List


@dataclass
class DiarizationOnlyCollator:
    processor: WhisperProcessor

    def __call__(self, batch: List[MixedExample]):
        features = [self.processor(sample.audio, sampling_rate=16_000, return_tensors='pt').input_features[0] for sample
                    in batch]
        ret = self.processor.feature_extractor.pad([{"input_features": feature} for feature in features],
                                                   return_tensors='pt')

        labels = torch.stack([sample.diarization for sample in batch]).float()

        ret['labels'] = labels

        return ret


@dataclass
class TandemCollator:
    processor: Wav2Vec2Processor

    def __call__(self, batch: List[MixedExample]):
        max_length = 360_000
        text_max_length = 350
        max_diarization_length = max_length // 320
        features = [
            {'input_values':
                 self.processor(sample.audio, sampling_rate=16_000, return_tensors='pt', max_length=max_length + 200,
                                padding='max_length', truncation=True).input_values[0]}
            for sample in batch]
        label_features = [{"input_ids": self.processor.tokenizer(text=sample.transcription, padding='max_length',
                                                                 max_length=text_max_length, truncation=True).input_ids}
                          for sample in batch]
        diar_labels = torch.stack([sample.diarization[:max_diarization_length] for sample in batch]).float()
        clf_labels = torch.tensor([sample.noise_id for sample in batch], dtype=torch.long)

        batch = self.processor.feature_extractor.pad(features, return_tensors="pt")
        label_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch['diarization_labels'] = diar_labels
        batch['classification_labels'] = clf_labels

        return batch


@dataclass
class SpeakerAttributionCollator:
    tokenizer: PreTrainedTokenizerFast
    feature_extractor: WhisperFeatureExtractor
    inventory: torch.Tensor

    def __call__(self, batch: List[SpeakerAttributeExample]):
        ret = {}

        tokenized = self.tokenizer([sample.transcription for sample in batch], return_tensors='pt',
                                   padding='max_length', max_length=256, truncation=True)

        ret['target_asr_ids'] = tokenized['input_ids'].masked_fill(tokenized['attention_mask'].ne(1), -100)

        ret['target_diar_ids'] = torch.stack(
            [torch.cat([sample.speaker_attributions,
                        torch.empty(256 - len(sample.speaker_attributions), dtype=torch.long).fill_(-100)])
             for sample in batch])

        features = [self.feature_extractor(sample.audio, sampling_rate=16_000, return_tensors='pt').input_features[0]
                    for sample in batch]
        features = self.feature_extractor.pad([{"input_features": feature} for feature in features],
                                              return_tensors='pt').input_features

        ret['input_features'] = features
        ret['speaker_inventory'] = self.inventory.unsqueeze(0).repeat(len(batch), 1, 1)

        return ret


@dataclass
class ClassificationCollator:
    processor: WhisperProcessor

    def __call__(self, batch: List[MixedExample]):
        features = [self.processor(sample.audio, sampling_rate=16_000, return_tensors='pt').input_features[0] for sample
                    in batch]
        ret = self.processor.feature_extractor.pad([{"input_features": feature} for feature in features],
                                                   return_tensors='pt')

        labels = torch.tensor([sample.noise_id for sample in batch], dtype=torch.long)
        ret['labels'] = labels
        return ret
