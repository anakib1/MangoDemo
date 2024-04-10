from torch_audiomentations import Compose, AddColoredNoise, PitchShift, Shift
from .base import MixerTransform, MixedExample
from ..globals import RATE


class AugmentationTransform(MixerTransform):
    def __init__(self, use_gaussian: bool = True, use_pitchshift: bool = True,
                 use_shift: bool = True):
        augments = []
        if use_gaussian:
            augments.append(AddColoredNoise(p=0.5))
        if use_pitchshift:
            augments.append(PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, p=0.5, sample_rate=RATE))
        if use_shift:
            augments.append(Shift(p=0.5))
        self.augment = Compose(augments)

    def __call__(self, sample: MixedExample):
        sample.audio = self.augment(samples=sample.audio.reshape(1, 1, -1), sample_rate=RATE).flatten()
        return sample
