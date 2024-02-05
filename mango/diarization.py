from dataclasses import dataclass

import torch
import torchaudio
from transformers import WhisperProcessor

from .whisper_ex.diarization import WhisperConfigForSpeakerDiarization, WhisperModelForSpeakerDiarization
from .models import InternalEENDConfig, MangoEEND
import numpy as np
from huggingface_hub import hf_hub_download


@dataclass
class DiarizationConfig:
    frame_length_ms: int = 20
    max_num_speakers: int = None


class BaseDiarizer:
    def __init__(self, config: DiarizationConfig):
        self.config = config

    def diarize(self, waveform: np.array, sr: int = 16_000) -> np.ndarray:
        pass


class DummyDiarizer(BaseDiarizer):

    def diarize(self, waveform: np.array, sr: int = 16_000) -> np.ndarray:
        num_bins = int(len(waveform) / sr * 1_000 // self.config.frame_length_ms)
        return np.random.rand(num_bins, self.config.max_num_speakers)


@dataclass
class WhisperBasedDiarizationConfig(DiarizationConfig):
    whisper_checkpoint: str = None
    processor_checkpoint: str = None  # todo remove


class WhisperDiarizer(BaseDiarizer):
    config: WhisperBasedDiarizationConfig

    def __init__(self, config: WhisperBasedDiarizationConfig):
        super().__init__(config)
        self.whisper_config = WhisperConfigForSpeakerDiarization.from_pretrained(self.config.whisper_checkpoint)
        self.whisper = WhisperModelForSpeakerDiarization.from_pretrained(self.config.whisper_checkpoint)
        self.whisper.eval()
        self.processor = WhisperProcessor.from_pretrained(self.config.processor_checkpoint)

    def diarize(self, waveform: np.array, sr: int = 16_000) -> np.ndarray:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(torch.tensor(waveform, dtype=torch.float))
        features = self.processor(waveform, return_tensors='pt', sampling_rate=16_000)
        features = features.to(self.whisper.device)
        with torch.no_grad():
            model_outputs = self.whisper(**features)
        predictions = model_outputs['logits']
        return torch.sigmoid(predictions).detach().cpu().numpy().squeeze(0)


@dataclass
class EENDConfig(DiarizationConfig):
    hf_api_model_path: str = 'anakib1/eend-sa'
    hf_api_model_name: str = 'model.pt'
    hf_api_processor_path: str = 'openai/whisper-small'  # todo remove


class EENDDiarizer(BaseDiarizer):
    config: EENDConfig

    def __init__(self, config: EENDConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eend = MangoEEND(InternalEENDConfig(num_speakers=config.max_num_speakers, d_model=384, n_layers=4))
        model_weights = hf_hub_download(repo_id=config.hf_api_model_path,
                                        filename=config.hf_api_model_name)
        self.eend.load_state_dict(torch.load(model_weights, self.device))
        self.eend.eval()
        self.processor = WhisperProcessor.from_pretrained(config.hf_api_processor_path)

    def diarize(self, waveform: np.array, sr: int = 16_000) -> np.ndarray:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(torch.tensor(waveform, dtype=torch.float))
        features = self.processor(waveform, return_tensors='pt', sampling_rate=16_000)
        features = features.to(self.device)
        with torch.no_grad():
            model_outputs = self.eend(**features)
        predictions = model_outputs['logits']
        return torch.sigmoid(predictions).squeeze(0).cpu().numpy()
