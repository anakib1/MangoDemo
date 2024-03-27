from datasets import Dataset as HfDataset
from .base import MixedExample, SegmentInfo, MixerTransform
import random
import torch
from .utils import expand_tensor, calc_audio_adjustment_coef


class CreateRandomBlankAudio(MixerTransform):
    def __init__(
        self,
        min_secs: float = 10.0,
        max_secs: float = 20.0,
        rate: int = 16000,
    ):
        self.min_len = int(rate * min_secs)
        self.max_len = int(rate * max_secs)

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
            audio=audio
        )
        return mixed_example


class CreateRandomSpeaker(MixerTransform):
    def __init__(self, speaker_dataset: HfDataset):
        self.speaker_dataset = speaker_dataset

    def __call__(
        self,
        example: MixedExample,
    ) -> MixedExample:
        """
        It creates MixedExample with random one speaker audio
        """
        random_pos = random.randint(0, len(self.speaker_dataset) - 1)
        speaker_example = self.speaker_dataset[random_pos]
        audio = speaker_example["audio"]["array"]

        mixed_example = MixedExample(
            audio=audio,
            speakers_info=[SegmentInfo(
                start=0,
                length=audio.shape[0],
                class_id=speaker_example["speaker_id"]
            )]
        )
        return mixed_example


class AddRandomSpeakerSegment(MixerTransform):
    def __init__(
        self,
        speaker_dataset: HfDataset,
    ):
        """
        :param speaker_dataset: the dataset of speaker audio
            It has to havc "audio":"array" and "speaker_id" columns
        """
        self.speaker_dataset = speaker_dataset

    def __call__(
        self,
        example: MixedExample,
    ) -> MixedExample:
        """
        Adds a random speaker voice segment to the audio
        """
        random_pos = random.randint(0, len(self.speaker_dataset) - 1)
        speaker_example = self.speaker_dataset[random_pos]
        speaker_audio = speaker_example["audio"]["array"]
        speaker_len = speaker_audio.shape[0]
        audio_len = example.audio.shape[0]

        if speaker_len > audio_len:
            speaker_audio = speaker_audio[:audio_len]
            speaker_len = audio_len

        pos_start = random.randint(0, audio_len - 1)
        if pos_start + speaker_len <= audio_len:
            example.audio[pos_start:pos_start + speaker_len] += speaker_audio
            start, length = pos_start, speaker_len
        else:
            end_len = audio_len - pos_start
            start_len = speaker_len - end_len
            if start_len > end_len:
                example.audio[:start_len] += speaker_audio[-start_len:]
                start, length = 0, start_len
            else:
                example.audio[-end_len:] += speaker_audio[:end_len]
                start, length = pos_start, end_len

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.append(SegmentInfo(
            start=start,
            length=length,
            class_id=speaker_example["speaker_id"]
        ))

        return example


class AddSeveralRandomSpeakerSegments(AddRandomSpeakerSegment):
    def __init__(
        self,
        speaker_dataset: HfDataset,
        min_speakers: int = 2,
        max_speakers: int = 4,
    ):
        super().__init__(speaker_dataset)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def __call__(
        self,
        example: MixedExample
    ) -> MixedExample:
        """
        The same as AddRandomSpeakerSegment, but it generates
        several (random number) speakers in the audio
        """
        speakers_num = random.randint(self.min_speakers, self.max_speakers)
        for i in range(speakers_num):
            example = super().__call__(example)
        return example


class AddRandomFilledNoise(MixerTransform):
    def __init__(
        self,
        noise_dataset: HfDataset,
        snrs: tuple[float] = (10.0, 15.0, 20.0),
        no_noise_coef: bool = True
    ):
        """
        :param noise_dataset: the dataset of noises audio
            It has to havc "audio":"array" and "noise_id" columns
        :param snrs: __call__ generates random noise coefficient from this list
        :param no_noise_coef: if False does not use noise coefficient at all
        """
        self.noises_dataset = noise_dataset
        self.snrs = snrs
        self.no_noise_coef = no_noise_coef

    def __call__(
        self,
        example: MixedExample,
    ) -> MixedExample:
        """
        It fills the audio with random noise from dataset
        """
        random_pos = random.randint(0, len(self.noises_dataset) - 1)
        noise_example = self.noises_dataset[random_pos]
        noise_len = example.audio.shape[0]
        noise_audio = expand_tensor(noise_example["audio"]["array"], noise_len)

        noise_coef = calc_audio_adjustment_coef(example.audio, noise_audio, self.snrs, self.no_noise_coef)
        example.audio += noise_audio * noise_coef

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.append(SegmentInfo(
            start=0,
            length=noise_len,
            class_id=noise_example["noise_id"]
        ))
        return example


class AddRandomNoiseSegment(MixerTransform):
    def __init__(
        self,
        noises_dataset: HfDataset,
        snrs: tuple[float] = (10.0, 15.0, 20.0),
        no_noise_coef: bool = True
    ):
        """
        :param noises_dataset: the dataset of noises audio
            It has to havc "audio":"array" and "noise_id" columns
        :param snrs: __call__ generates random noise coefficient from this list
        :param no_noise_coef: if False does not use noise coefficient at all
        """
        self.noises_dataset = noises_dataset
        self.snrs = snrs
        self.no_noise_coef = no_noise_coef

    def __call__(
        self,
        example: MixedExample,
    ) -> MixedExample:
        """
        Adds to the audio random noise from dataset at a random place
        """
        random_pos = random.randint(0, len(self.noises_dataset) - 1)
        noise_example = self.noises_dataset[random_pos]
        noise_audio = noise_example["audio"]["array"]
        noise_len = noise_audio.shape[0]
        audio_len = example.audio.shape[0]

        if noise_len > audio_len:
            noise_audio = noise_audio[:audio_len]
            noise_len = audio_len

        noise_coef = calc_audio_adjustment_coef(example.audio, noise_audio, self.snrs, self.no_noise_coef)

        pos_start = random.randint(0, audio_len - 1)
        if pos_start + noise_len <= audio_len:
            example.audio[pos_start:pos_start + noise_len] += noise_coef * noise_audio
            start, length = pos_start, noise_len
        else:
            end_len = audio_len - pos_start
            start_len = noise_len - end_len
            if start_len > end_len:
                example.audio[:start_len] += noise_coef * noise_audio[-start_len:]
                start, length = 0, start_len
            else:
                example.audio[-end_len:] += noise_coef * noise_audio[:end_len]
                start, length = pos_start, end_len

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.append(SegmentInfo(
            start=start,
            length=length,
            class_id=noise_example["noise_id"]
        ))

        return example


class AddSeveralRandomNoiseSegments(AddRandomNoiseSegment):
    def __init__(
        self,
        noises_dataset: HfDataset,
        snrs: tuple[float] = (10.0, 15.0, 20.0),
        no_noise_coef: bool = True,
        min_noises: int = 2,
        max_noises: int = 4,
    ):
        """
        :param min_noises: min noises segments in audio
        :param max_noises: max noises segments in audio
        see also AddRandomNoiseSegment
        """
        super().__init__(noises_dataset, snrs, False)
        self.no_noise_coef = no_noise_coef
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
        example_helper = MixedExample(
            audio=torch.zeros(example.audio.shape[0])
        )
        noises_num = random.randint(self.min_noises, self.max_noises)
        for i in range(noises_num):
            example_helper = super().__call__(example_helper)

        if example.noises_info is None:
            example.noises_info = []
        example.noises_info.extend(example_helper.noises_info)
        noise_coef = calc_audio_adjustment_coef(example.audio, example.audio, self.snrs, self.no_noise_coef)
        example.audio += noise_coef * example_helper.audio

        return example
