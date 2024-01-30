import logging
from dataclasses import dataclass

import numpy as np
import torch
import datasets
from datasets import Audio


@dataclass
class DatasetMixerConfig:
    max_speakers: int = 2
    max_utterance_length: float = 30
    no_overlap: bool = False
    utterances_count: int = None
    output_sampling_rate: int = 16_000
    frame_resolution_ms: int = 20
    min_repetitions: int = 1
    max_repetitions: int = 3
    beta: float = 1


@dataclass
class MixedExample:
    audio: torch.Tensor
    noise_id: int
    diarization: torch.Tensor
    transcription: str
    # todo maybe add some mappings?


def calculate_adjustment_coef(original_input: torch.Tensor, noise: torch.Tensor, snr_dB: float) -> float:
    """
    Calclulates adjustement coefficient for noise adding
    :param original_input: tensor of shape (S,) - audio
    :param noise: tensor of shape (S,) - noise
    :param snr_dB: aimed signal-to-noise ratio
    :return: ratio for the noise to be successfully added.
    """
    power_y = torch.mean(torch.square(original_input))
    power_noise = torch.mean(torch.square(noise))
    scaling_factor = torch.sqrt(power_y / (10 ** (snr_dB / 10) * power_noise))
    return scaling_factor


class DatasetMixer:
    """
    Class that mixes speakers and noises into synthetic dataset.
    """
    def __init__(self, config: DatasetMixerConfig, utterances: datasets.Dataset, noises: datasets.Dataset, rirs=None):
        """
        Instantiates a new DatasetMixer.
        :param config: configuration of this Dataset. Read DatasetMixerConfig docs.
        :param utterances: one-speaker utterances for synthesizing data. Should contain column 'audio', column 'speaker_id' and column 'transcription'
        :param noises: noises for synthesizing data. Should contain column 'audio' and column 'label'
        :param rirs: not currently used.
        """
        self.num_speakers = None
        self.num_noises = None
        self.noise2id = None
        self.noise2audio = None
        self.label2id = None
        self.speaker2id = None
        self.speakers2utterance = None
        if rirs is not None:
            logging.warning('RIRs are not yet supported.')

        self.utterances = utterances
        self.noises = noises
        self.config = config

        self.snrs = [10, 15, 20]

        self.preprocess()

    def preprocess(self) -> None:
        """
        Instantiates noise and speakers map
        :return: None
        """
        self.speakers2utterance = {}
        self.speaker2id = {}
        for i, speaker_id in enumerate(self.utterances['speaker_id']):
            self.speakers2utterance.setdefault(speaker_id, []).append(i)
            self.speaker2id.setdefault(speaker_id, len(self.speaker2id))
        self.num_speakers = len(self.speaker2id)

        self.noise2id = {}
        self.noise2audio = {}
        for i, noise_label in enumerate(self.noises['label']):
            self.noise2audio.setdefault(noise_label, []).append(i)
            self.noise2id.setdefault(noise_label, len(self.noise2id))
        self.num_noises = len(self.noise2id)

        self.utterances = self.utterances.cast_column('audio', Audio(sampling_rate=self.config.output_sampling_rate))
        self.noises = self.noises.cast_column('audio', Audio(sampling_rate=self.config.output_sampling_rate))

    def generate(self) -> MixedExample:
        """
        Generates new synthetic example.
        :return: MixedExample
        """
        num_speakers = np.random.randint(2, self.config.max_speakers + 1)
        speakers = np.random.choice(list(self.speakers2utterance.keys()), self.config.max_speakers)

        total_length = int(self.config.max_utterance_length * self.config.output_sampling_rate)
        filled_length = 0

        audio = torch.zeros(total_length)
        frame_resolution = int(self.config.output_sampling_rate / 1_000 * self.config.frame_resolution_ms)
        diarization = torch.zeros(total_length // frame_resolution, num_speakers, dtype=torch.long)

        transcriptions = []

        while filled_length < total_length:
            speaker_id = np.random.randint(num_speakers)
            speaker = speakers[speaker_id]

            n_u = np.random.randint(self.config.min_repetitions, self.config.max_repetitions)
            for u in range(n_u):
                if filled_length >= total_length:
                    break

                utterance_id = int(np.random.choice(self.speakers2utterance[speaker]))
                utterance_audio: torch.Tensor = self.utterances[utterance_id]['audio']['array']

                max_size = total_length - filled_length
                loaded_audio = utterance_audio[:max_size]

                diarization[filled_length // frame_resolution:filled_length // frame_resolution + len(
                    loaded_audio) // frame_resolution, speaker_id] = 1
                audio[filled_length: filled_length + len(loaded_audio)] = loaded_audio

                filled_length += len(utterance_audio)

                transcriptions.append(self.utterances[utterance_id]['transcription'])

                # add noise.

                d = int(np.random.exponential(scale=self.config.beta) * self.config.output_sampling_rate)
                d = min(d, max_size)

                filled_length += d

        background_noise_cls = np.random.choice(list(self.noise2id.keys()))
        noise_row_id = int(np.random.choice(self.noise2audio[background_noise_cls]))
        background_noise_audio = self.noises[noise_row_id]['audio']['array']
        snr = np.random.choice(self.snrs)

        repeats = total_length // len(background_noise_audio)
        background_noise_audio = background_noise_audio.repeat(repeats)
        background_noise_audio = torch.concatenate(
            (background_noise_audio, torch.zeros(total_length - len(background_noise_audio))),
            dim=0)

        audio += background_noise_audio * calculate_adjustment_coef(audio, background_noise_audio, snr)
        return MixedExample(
            audio=audio,
            noise_id=self.noise2id[background_noise_cls],
            diarization=diarization,
            transcription=' '.join(transcriptions)
        )


class DatasetMixerWrapped(DatasetMixer, torch.utils.data.Dataset):
    """
    Wrapper for torch.utils.dataDataset.
    """

    def __init__(self, config: DatasetMixerConfig, utterances, noises, rirs=None):
        super(DatasetMixerWrapped, self).__init__(config, utterances, noises, rirs)

    def __getitem__(self, idx):
        return self.generate()

    def __len__(self):
        return self.config.utterances_count
