from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torchaudio
from transformers import WhisperForAudioClassification, WhisperProcessor
from .models.classification import EncoderClassifier, InternalClassifierConfig
from huggingface_hub import hf_hub_download


@dataclass
class BaseClassifierConfig:
    """
    Contains id2label mapping for label names of a classifier.
    """
    id2label: Dict[int, str] = None


class BaseClassifier(object):
    """
    Base class for any classification model.
    """

    def __init__(self, config: BaseClassifierConfig):
        self.config = config

    """
    Performs classification of raw waveform and returns scores as defined in config.
    """

    def classify(self, waveform: np.ndarray, sr: int = 16_000) -> dict:
        pass


class DummyClassifier(BaseClassifier):

    def classify(self, waveform: np.array, sr: int = 16_000) -> dict:
        ret = np.abs(np.random.randn(len(self.config.id2label)))
        ret = ret / ret.sum()
        return {self.config.id2label.get(x, 'unknown'): ret[x] for x in range(len(ret))}


@dataclass
class WhisperClassifierConfig(BaseClassifierConfig):
    whisper_checkpoint: str = None
    processor_checkpoint: str = None


class WhisperClassifier(BaseClassifier):
    def __init__(self, config: WhisperClassifierConfig):
        super().__init__(config)
        self.whisper = WhisperForAudioClassification.from_pretrained(config.whisper_checkpoint)
        self.whisper.eval()
        self.processor = WhisperProcessor.from_pretrained(config.processor_checkpoint)
        if self.config.id2label is None and self.whisper.config.id2label is not None:
            self.config.id2label = self.whisper.config.id2label

    def classify(self, waveform: np.ndarray, sr: int = 16_000) -> dict:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(torch.tensor(waveform, dtype=torch.float))
        features = self.processor(waveform, return_tensors='pt', sampling_rate=16_000)
        features = features.to(self.whisper.device)
        logits = self.whisper(**features).logits
        scores = torch.softmax(logits, dim=-1).detach().cpu().numpy().flatten()

        return {self.config.id2label.get(x, 'unknown'): scores[x] for x in range(len(scores))}


@dataclass
class WhisperClassifierExConfig(WhisperClassifierConfig):
    hf_api_model_path = 'anakib1/mango-clf-small'
    hf_api_processor_path = 'openai/whisper-small'
    run_id = 'run-24-02-09.23-55'


class WhisperClassifierEx(BaseClassifier):
    def __init__(self, config: WhisperClassifierExConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eend = EncoderClassifier(InternalClassifierConfig(num_labels=10, d_model=384, n_layers=4))
        subfolder = f'runs/{self.config.run_id}' if self.config.run_id is not None else None
        model_weights = hf_hub_download(repo_id=config.hf_api_model_path,
                                        subfolder=subfolder,
                                        filename='model.pt')
        self.eend.load_state_dict(torch.load(model_weights, self.device))
        self.eend.eval()
        self.processor = WhisperProcessor.from_pretrained(config.hf_api_processor_path)

    def classify(self, waveform: np.ndarray, sr: int = 16_000) -> dict:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(torch.tensor(waveform, dtype=torch.float))
        features = self.processor(waveform, return_tensors='pt', sampling_rate=16_000)
        features = features.to(self.device)
        logits = self.eend(**features)['logits']
        scores = torch.softmax(logits, dim=-1).detach().cpu().numpy().flatten()

        return {self.config.id2label.get(x, 'unknown'): scores[x] for x in range(len(scores))}
