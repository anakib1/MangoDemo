from .base import MixedExample, SegmentInfo, MixerTransform, AudioProvider, SegmentData
import random
import torch
from .utils import expand_tensor, calc_audio_adjustment_coef
from .utils import Generate
import logging
from ..globals import RATE
from typing import List


logger = logging.getLogger(__name__)


class CreateRandomBlankAudio(MixerTransform):
    def __init__(
            self,
            min_secs: float = 10.0,
            max_secs: float = 20.0,
    ):
        """
        parameters specify the target length of audio
        """
        self.min_len = int(RATE * min_secs)
        self.max_len = int(RATE * max_secs)

    def __call__(
            self,
            example: MixedExample,
    ) -> MixedExample:
        """
        It creates MixedExample with random one speaker audio
        """
        random_len = random.randint(self.min_len, self.max_len)
        audio = torch.zeros(random_len)
        mixed_example = MixedExample(
            _base=SegmentData(
                audio=audio,
                type="blank",
                start=0
            )
        )
        return mixed_example


class CreateRandomSpeaker(MixerTransform):
    def __init__(
            self,
            audio_provider: AudioProvider
    ):
        self.audio_provider = audio_provider

    def __call__(
            self,
            example: MixedExample,
    ) -> MixedExample:
        """
        It creates MixedExample with random one speaker audio
        """
        speaker_example = self.audio_provider.provide()

        mixed_example = MixedExample(
            speakers_info=[SegmentInfo(
                start=0,
                length=speaker_example.audio.shape[0],
                class_id=speaker_example.class_id
            )],
            _base=SegmentData(
                audio=speaker_example.audio,
                type="speaker",
                start=0
            )
        )
        return mixed_example


class AddRandomSpeakerSegment(MixerTransform):
    def __init__(
            self,
            audio_provider: AudioProvider
    ):
        self.audio_provider = audio_provider

    def __call__(
            self,
            example: MixedExample,
    ) -> MixedExample:
        """
        Adds a random speaker voice segment to the audio
        """
        speaker_example = self.audio_provider.provide()

        speaker_audio, start_pos = Generate.position_segment_in_audio(example._base.audio, speaker_example.audio)

        if example.speakers_info is None:
            example.speakers_info = []
        example.speakers_info.append(SegmentInfo(
            start=start_pos,
            length=speaker_audio.shape[0],
            class_id=speaker_example.class_id
        ))

        if example._segments is None:
            example._segments = []
        example._segments.append(SegmentData(
            audio=speaker_audio,
            type="speaker",
            start=start_pos
        ))

        return example


class AddSeveralRandomSpeakerSegments(AddRandomSpeakerSegment):
    def __init__(
            self,
            audio_provider: AudioProvider,
            min_speakers: int = 2,
            max_speakers: int = 4,
    ):
        """
        parameters specify the range of speakers to generate
        """
        super().__init__(audio_provider)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def __call__(
            self,
            example: MixedExample
    ) -> MixedExample:
        """
        The same as AddRandomSpeakerSegment, but it generates
        several (random number) speakers to the audio
        """
        speakers_num = random.randint(self.min_speakers, self.max_speakers)
        for i in range(speakers_num):
            example = super().__call__(example)
        return example


class MergeAll(MixerTransform):
    def __call__(
            self,
            example: MixedExample
    ) -> MixedExample:
        """
        Merges all data created by previous transforms into single audio
        """
        if example._base is None:
            raise Exception("the base audio was not created")
        example.audio = example._base.audio
        has_speaker = example._base.type == "speaker"
        if example._segments is not None:
            for segment in example._segments:
                if segment.type == "speaker":
                    example.audio[segment.start:segment.start + segment.audio.shape[0]] += segment.audio
                    has_speaker = True
            noise_coefs = dict()
            for i, segment in enumerate(example._segments):
                if segment.type == "noise":
                    noise_coefs[i] = calc_audio_adjustment_coef(example.audio, segment.audio, segment.metadata["snr"])
            for i, segment in enumerate(example._segments):
                if segment.type == "noise":
                    noise_coef = noise_coefs[i] if has_speaker else 1.0
                    example.audio[segment.start:segment.start + segment.audio.shape[0]] += segment.audio * noise_coef
        return example


class AddRandomFilledNoise(MixerTransform):
    def __init__(
            self,
            audio_provider: AudioProvider,
            snrs: List[float] = (10.0, 15.0, 20.0),
    ):
        """
        :param snrs: specifies range of snrs to apply to noise
            !!! if speaker audio is added beforehand !!!
        """
        self.audio_provider = audio_provider
        self.snrs = snrs

    def __call__(
            self,
            example: MixedExample,
    ) -> MixedExample:
        """
        It fills the audio with random noise from dataset
        """
        noise_example = self.audio_provider.provide()

        if example._base is None:
            raise Exception("the base audio was not created")
        noise_len = example._base.audio.shape[0]
        noise_audio = expand_tensor(noise_example.audio, noise_len)

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.append(SegmentInfo(
            start=0,
            length=noise_len,
            class_id=noise_example.class_id
        ))

        if example._segments is None:
            example._segments = []
        example._segments.append(SegmentData(
            audio=noise_audio,
            type="noise",
            start=0,
            metadata={"snr": random.choice(self.snrs)}
        ))
        return example


class AddRandomNoiseSegment(MixerTransform):
    def __init__(
            self,
            audio_provider: AudioProvider,
            snrs: List[float] = (10.0, 15.0, 20.0),
    ):
        """
        See AddRandomFilledNoise
        """
        self.audio_provider = audio_provider
        self.snrs = snrs

    def __call__(
            self,
            example: MixedExample,
    ) -> MixedExample:
        """
        Adds to the audio random noise from dataset at a random place
        """
        noise_example = self.audio_provider.provide()

        noise_audio, start_pos = Generate.position_segment_in_audio(example._base.audio, noise_example.audio)

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.append(SegmentInfo(
            start=start_pos,
            length=noise_audio.shape[0],
            class_id=noise_example.class_id
        ))

        if example._segments is None:
            example._segments = []
        example._segments.append(SegmentData(
            audio=noise_audio,
            type="noise",
            start=start_pos,
            metadata={"snr": random.choice(self.snrs)}
        ))
        return example


class AddSeveralRandomNoiseSegments(AddRandomNoiseSegment):
    def __init__(
            self,
            audio_provider: AudioProvider,
            snrs: List[float] = (10.0, 15.0, 20.0),
            min_noises: int = 2,
            max_noises: int = 4,
    ):
        """
        additional parameters specify range of noises to generate
        see also AddRandomNoiseSegment
        """
        super().__init__(audio_provider, snrs)
        self.min_noises = min_noises
        self.max_noises = max_noises

    def __call__(
            self,
            example: MixedExample
    ) -> MixedExample:
        """
        The same as AddRandomNoiseSegment, but it generates
        several (random number) noises in the audio
        """
        noises_num = random.randint(self.min_noises, self.max_noises)
        for i in range(noises_num):
            example = super().__call__(example)
        return example
