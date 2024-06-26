from dataclasses import dataclass
from typing import List
import torch


@dataclass
class SegmentInfo:
    start: float
    end: float
    class_id: int


@dataclass
class SegmentData:
    audio: torch.Tensor
    type: str
    start: int
    metadata: dict = None


@dataclass
class MixedExample:
    """
    Audio - 1D torch tensor in default (16_000) sampling rate
    """
    audio: torch.Tensor = None
    speakers_info: List[SegmentInfo] = None
    noises_info: List[SegmentInfo] = None
    _segments: List[SegmentData] = None
    _base: SegmentData = None


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


@dataclass
class AudioData:
    audio: torch.Tensor
    class_id: int


class AudioProvider:
    def provide(self) -> AudioData:
        """
        This function should provide the data audio example in a given format
        """
        pass
