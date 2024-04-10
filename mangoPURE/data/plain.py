from datasets import Dataset as HfDataset
from torchaudio.transforms import Resample
from .mixer import MixedExample
from .base import SegmentInfo, MixerTransform
from .utils import Name2IdMapper
from ..globals import RATE
from typing import List


class AudioDataset:
    def __init__(self, dataset: HfDataset, transforms: List[MixerTransform] = None):
        self.dataset = dataset
        self.idx = 0
        self.len = len(dataset)
        self.transforms = transforms if transforms is not None else []
        self.mapper = Name2IdMapper(start_id=0)

    def generate(self) -> MixedExample:
        rate = self.dataset[self.idx]["audio"]["sampling_rate"]
        audio = self.dataset[self.idx]["audio"]["array"]
        noise = self.mapper.add_name(self.dataset[self.idx]["class"])
        audio = Resample(rate, RATE)(audio)
        self.idx += 1
        if self.idx >= self.len:
            self.idx = 0
        example = MixedExample(audio=audio, noises_info=[SegmentInfo(class_id=noise, start=0.0, end=1.0)])
        for transform in self.transforms:
            example = transform(example)
        return example
