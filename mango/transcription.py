import numpy as np
from dataclasses import dataclass

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast


class BaseTranscriptor(object):
    def transcribe(self, waveform: np.ndarray) -> str:
        pass


class DummyTranscriptor(BaseTranscriptor):
    def transcribe(self, waveform: np.ndarray, sr: int = 16_000) -> str:
        return "This would be the default output of the transcription."


@dataclass
class WhisperTranscriptionConfig:
    whisper_checkpoint: str = None
    processor_checkpoint: str = None  # todo: remove
    language: str = None


class WhisperTranscriptor(BaseTranscriptor):
    def __init__(self, config: WhisperTranscriptionConfig):
        self.config = config
        self.whisper = WhisperForConditionalGeneration.from_pretrained(config.whisper_checkpoint, device_map='auto')
        self.whisper.eval()
        self.processor = WhisperProcessor.from_pretrained(config.processor_checkpoint)
        self.processor.tokenizer = WhisperTokenizerFast.from_pretrained(config.processor_checkpoint)
        self.processor.tokenizer.language = self.config.language

    def transcribe(self, waveform: np.ndarray, sr: int = 16_000) -> str:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(torch.tensor(waveform, dtype=torch.float))
        features = self.processor(waveform, return_tensors='pt', sampling_rate=16_000)
        features = features.to(self.whisper.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="uk", task="transcribe")

        generated_ids = self.whisper.generate(**features, forced_decoder_ids=forced_decoder_ids)
        decoded_sequence = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded_sequence
