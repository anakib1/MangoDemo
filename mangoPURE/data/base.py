import torch
from dataclasses import dataclass
from datasets import Dataset as HfDataset


@dataclass
class SegmentInfo:
    start: int
    length: int
    class_id: int


@dataclass
class MixedExample:
    audio: torch.Tensor = None
    speakers_info: list[SegmentInfo] = None
    noises_info: list[SegmentInfo] = None


class MixerTransform:
    def __call__(
        self,
        example: MixedExample,
    ) -> MixedExample:
        """
        The abstract function for transforming MixedExample
        It can also create example
        """
        pass
