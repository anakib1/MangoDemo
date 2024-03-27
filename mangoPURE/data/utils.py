import torch
from typing import Union
import random
from .base import MixedExample, SegmentInfo


def calc_audio_adjustment_coef(
        original_input: torch.Tensor,
        noise: torch.Tensor,
        snr_dB: Union[float, list[float], tuple[float]],
        no_noise_coef: bool = False
    ) -> float:
    """
    Calculates adjustment coefficient in order to add noise appropriately
    :param original_input: input audio 1d tensor
    :param noise: input noise 1d tensor
    :param snr_dB: aimed signal-to-noise ratio
    :param no_noise_coef: returns 1.0 if this parameter is true
    :return: ratio to successfully add the noise
    """
    if no_noise_coef:
        return 1.0
    if isinstance(snr_dB, Union[list, tuple]):
        snr_dB = random.choice(snr_dB)
    power_y = torch.mean(torch.square(original_input))
    power_noise = torch.mean(torch.square(noise))
    scaling_factor = torch.sqrt(power_y / (10 ** (snr_dB / 10) * power_noise))
    return scaling_factor.tolist()


def expand_tensor(arr: torch.Tensor, length: int) -> torch.Tensor:
    """
    expands given torch tensor to the given length
    """
    num = length // arr.shape[0]
    ans = arr.tile(num)
    ans = torch.cat([ans, arr[:length - num * arr.shape[0]]])
    return ans


class Resize:
    @staticmethod
    def resize_segment(segment_info: SegmentInfo, start_size: int, end_size: int) -> SegmentInfo:
        """
        This function resizes the info when audio (embedding) length changes
        Be careful it works INPLACE!!!
        """
        segment_info.start = int(segment_info.start / start_size * end_size)
        segment_info.length = int(segment_info.length / start_size * end_size)
        return segment_info

    @staticmethod
    def resize_mixed_example_info(mixed_example: MixedExample, new_size: int) -> MixedExample:
        """
        This function resizes the info in the example when audio (embedding) length changes
        Be careful it works INPLACE!!!
        """
        size = mixed_example.audio.shape[0]
        if mixed_example.speakers_info is not None:
            for seg in mixed_example.speakers_info:
                Resize.resize_segment(seg, size, new_size)
        if mixed_example.noises_info is not None:
            for seg in mixed_example.noises_info:
                Resize.resize_segment(seg, size, new_size)
        return mixed_example