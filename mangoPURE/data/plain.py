from datasets import Dataset as HfDataset
from torchaudio.transforms import Resample
from .mixer import MixedExample
from .base import SegmentInfo
from .utils import Name2IdMapper
from ..globals import RATE


class AudioDataset:
    def __init__(self, dataset: HfDataset):
        self.dataset = dataset
        self.idx = 0
        self.len = len(dataset)
        self.mapper = Name2IdMapper(start_id=0)

    def generate(self) -> MixedExample:
        rate = self.dataset[self.idx]["audio"]["sampling_rate"]
        audio = self.dataset[self.idx]["audio"]["array"]
        noise = self.mapper.add_name(self.dataset[self.idx]["class"])
        audio = Resample(rate, RATE)(audio)
        self.idx += 1
        if self.idx >= self.len:
            self.idx = 0
        return MixedExample(audio=audio, noises_info=[SegmentInfo(class_id=noise, start=0.0, end=1.0)])
