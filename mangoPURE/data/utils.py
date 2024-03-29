import torch
from typing import Union
import random
from .base import MixedExample, SegmentInfo


def calc_audio_adjustment_coef(
        original_input: torch.Tensor,
        noise: torch.Tensor,
        snr_dB: float,
    ) -> float:
    """
    Calculates adjustment coefficient in order to add noise appropriately
    :param original_input: input audio 1d tensor
    :param noise: input noise 1d tensor
    :param snr_dB: aimed signal-to-noise ratio
    :return: ratio to successfully add the noise
    """
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


class Generate:

    @staticmethod
    def position_segment_in_audio(audio: torch.tensor, segment: torch.Tensor) -> (torch.Tensor, int):
        """
        :param audio: the base audio where we should insert the segment
        :param segment: the segment to insert
        :return: our segment (it may be truncated due to positioning at the end or at the start)
            and start position of the segment
        """
        audio_len = audio.shape[0]
        segment_len = segment.shape[0]

        if segment_len > audio_len:
            segment = segment[:audio_len]
            segment_len = audio_len

        pos_start = random.randint(0, audio_len - 1)
        if pos_start + segment_len <= audio_len:
            return segment, pos_start
        else:
            end_len = audio_len - pos_start
            start_len = segment_len - end_len
            if start_len > end_len:
                return segment[-start_len:], 0
            else:
                return segment[:end_len], pos_start


class Name2IdMapper:
    """
    This class handles conversion from names to integer ids
    """
    def __init__(self, start_id=1):
        self.name2id = dict()
        self.id2name = dict()
        self.current_new_id = start_id

    def get_name(self, id):
        return self.id2name[id]

    def get_id(self, name):
        return self.name2id[name]

    def add_name(self, name):
        """
        Adds name to mapper if the name does not exist
        """
        if name not in self.name2id:
            self.name2id[name] = self.current_new_id
            self.id2name[self.current_new_id] = name
            self.current_new_id += 1
