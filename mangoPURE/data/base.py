from dataclasses import dataclass

import torch


@dataclass
class SegmentInfo:
    """
    Segments information are given in sampling resolution.
    """
    start: int
    length: int
    class_id: int


@dataclass
class MixedExample:
    """
    Audio - 1D torch tensor in default (16_000) sampling rate
    """
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
