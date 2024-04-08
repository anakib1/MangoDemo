import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from .base import MixerTransform, MixedExample
from ..globals import RATE
import numpy as np


class AugmentationTransform(MixerTransform):
    def __init__(self, use_gaussian: bool = True, use_stretch: bool = True, use_pitchshift: bool = True,
                 use_shift: bool = True):
        augments = []
        if use_gaussian:
            augments.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
        if use_stretch:
            augments.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5))
        if use_pitchshift:
            augments.append(PitchShift(min_semitones=-4, max_semitones=4, p=0.5))
        if use_shift:
            augments.append(Shift(p=0.5))
        self.augment = Compose(augments)

    def __call__(self, sample: MixedExample):
        sample.audio = self.augment(samples=np.asarray(sample.audio), sample_rate=RATE)
        sample.audio = torch.as_tensor(sample.audio)
        return sample
