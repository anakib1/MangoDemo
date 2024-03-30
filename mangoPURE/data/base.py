from dataclasses import dataclass

import torch


@dataclass
class SegmentInfo:
    start: int
    length: int
    class_id: int


@dataclass
class SegmentData:
    audio: torch.Tensor
    type: str
    start: int
    metadata: dict = None


@dataclass
class TimedAudioBatch:
    audio: torch.Tensor
    lengths: list[int]
    noises_info: list[list[SegmentInfo]] = None
    speakers_info: list[list[SegmentInfo]] = None


@dataclass
class MixedExample:
    """
    Audio - 1D torch tensor in default (16_000) sampling rate
    """
    audio: torch.Tensor = None
    speakers_info: list[SegmentInfo] = None
    noises_info: list[SegmentInfo] = None
    _segments: list[SegmentData] = None
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
