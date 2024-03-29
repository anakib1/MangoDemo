from .base import AudioProvider, AudioData
from datasets import Dataset as HfDataset
import random
import librosa
from .utils import Name2IdMapper
from ..globals import RATE
import torch


class CV13Random(AudioProvider):
    def __init__(self, dataset: HfDataset):
        self.dataset = dataset
        self.mapper = Name2IdMapper()

    def provide(self) -> AudioData:
        """
        Provides random example form cv13 dataset
        """
        pos = random.randint(0, len(self.dataset) - 1)
        rate = self.dataset[pos]["audio"]["sampling_rate"]
        audio = self.dataset[pos]["audio"]["array"]
        class_name = self.dataset[pos]["client_id"]
        self.mapper.add_name(class_name)
        audio_16 = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
        return AudioData(
            audio=torch.tensor(audio_16),
            class_id=self.mapper.get_id(class_name)
        )


class UrbanRandom(AudioProvider):
    def __init__(self, dataset: HfDataset):
        self.dataset = dataset
        self.mapper = Name2IdMapper()

    def provide(self) -> AudioData:
        """
        Provides random example form urban dataset
        """
        pos = random.randint(0, len(self.dataset) - 1)
        rate = self.dataset[pos]["audio"]["sampling_rate"]
        audio = self.dataset[pos]["audio"]["array"]
        class_name = self.dataset[pos]["class"]
        self.mapper.add_name(class_name)
        audio_16 = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
        return AudioData(
            audio=torch.tensor(audio_16),
            class_id=self.mapper.get_id(class_name)
        )