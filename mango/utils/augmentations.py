from dataclasses import dataclass
import random
import numpy as np
from typing import Union


@dataclass
class SampleSegmentTransform:
    min_len: float = 10.0
    max_len: float = 20.0

    def call(self, sample: np.ndarray, sample_rate: int):
        length_seconds = sample.shape[0] / sample_rate
        if length_seconds < self.min_len:
            raise Exception("length of audio is lower than expected")
        desired_length_seconds = random.uniform(self.min_len, min(self.max_len, length_seconds))
        desired_length = int(desired_length_seconds * sample_rate)
        start = random.randint(0, sample.shape[0] - desired_length - 1)
        sample = sample[start:start + desired_length]
        return sample

    def __call__(self, samples: Union[tuple[np.ndarray], np.ndarray], sample_rate: int):
        if isinstance(samples, tuple):
            samples = tuple(self.call(sample, sample_rate) for sample in samples)
        else:
            samples = self.call(samples, sample_rate)
        return samples
