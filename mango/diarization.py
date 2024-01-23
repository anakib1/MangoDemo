import logging
from dataclasses import dataclass

import torch
import torchaudio
from transformers import WhisperProcessor

from .whisper_ex.diarization import WhisperConfigForSpeakerDiarization, WhisperModelForSpeakerDiarization

import numpy as np
from pyannote.core import Timeline, Segment, notebook
import matplotlib.pyplot as plt

import io


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
        model_outputs = self.whisper(**features)
        predictions = model_outputs['logits']
        return torch.sigmoid(predictions).detach().cpu().numpy().squeeze(0)


def draw_diarization(data: np.ndarray) -> np.ndarray:
    """
    Returns picture of speaker diarization.
    :param data: torch.tensor: model output of shape [1, T, config.n_speakers]
    :return: np.ndarray of shape [400, 800, 4] - picture of diarization.
    """

    return _repr_timeline(_create_timeline(data))


def _create_timeline(data: np.ndarray):
    if len(data.shape) == 3:
        if data.shape[0] != 1:
            logging.warning('Batched input passed, incorrect behaviour is about to happen.')
        data = data.squeeze(0)
    len_in_seconds = len(data) / (1_500 / 30)
    num_bins = 50

    data = data[:len(data) // num_bins * num_bins, :]

    bin_in_seconds = len_in_seconds / num_bins
    binned_data = np.median(data.reshape(-1, len(data) // num_bins, data.shape[1]), axis=1)
    timeline = Timeline()

    for speaker in range(3):
        start = None
        for i in range(len(binned_data)):
            if binned_data[i, speaker] > 0.5 and start is None:
                start = i
            elif binned_data[i, speaker] < 0.5 and start is not None:
                timeline.add(Segment(start * bin_in_seconds, i * bin_in_seconds))
                start = None

        if start is not None:
            timeline.add(Segment(start * bin_in_seconds, len(binned_data) * bin_in_seconds))

    return timeline


def _repr_timeline(timeline: Timeline):
    plt.rcParams["figure.figsize"] = (8, 4)
    fig, ax = plt.subplots()
    notebook.plot_timeline(timeline, ax=ax)
    with io.BytesIO() as buff:
        fig.savefig(buff, format='png')
        buff.seek(0)
        im = plt.imread(buff)

    return im
