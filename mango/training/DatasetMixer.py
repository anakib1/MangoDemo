import logging
from dataclasses import dataclass

import numpy as np
import torch
import datasets
from datasets import Audio
from typing import List, Tuple, Dict, Union


@dataclass
class DatasetMixerConfig:
    min_speakers: int = 1
    max_speakers: int = 2
    min_noises: int = 1
    max_noises: int = 5
    max_utterance_length: float = 30
    no_overlap: bool = False
    utterances_count: int = None
    output_sampling_rate: int = 16_000
    frame_resolution_ms: int = 20
    min_repetitions: int = 1
    max_repetitions: int = 3
    beta: float = 1
    noise_sample_type: str = "whole"   # whole means place noise for the whole video, "random" - for a random segment


@dataclass
class MixedExample:
    audio: torch.Tensor
    noise_id: Union[int, list]
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


def expand_array(arr: np.array, length: int) -> np.array:
    """
    :param arr: given array
    :param length: the length to expand given array
    :return: the expanded array
    """
    num = length // arr.shape[0]
    ans = np.tile(arr, num)
    ans = np.concatenate([ans, arr[:length - num * arr.shape[0]]])
    return ans

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
        self.label2id: Dict[str, int] = None
        self.speaker2id: Dict[str, int] = None
        self.speakers2utterance = None
        if rirs is not None:
            logging.warning('RIRs are not yet supported.')

        self.utterances = utterances
        self.noises = noises
        self.config = config
        self.frame_resolution = int(self.config.output_sampling_rate / 1_000 * self.config.frame_resolution_ms)

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
            noise_label = str(noise_label)
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
        num_speakers = np.random.randint(self.config.min_speakers, self.config.max_speakers + 1)
        speakers = np.random.choice(list(self.speakers2utterance.keys()), num_speakers, replace=False)

        total_length = int(self.config.max_utterance_length * self.config.output_sampling_rate)

        audio = torch.zeros(total_length)
        diarization = torch.zeros(total_length // self.frame_resolution, self.config.max_speakers, dtype=torch.long)

        transcriptions = []
        if self.config.no_overlap:
            self.generate_no_overlap(audio, diarization, transcriptions, speakers)
        else:
            self.generate_overlap(audio, diarization, transcriptions, speakers)
        transcriptions = self.process_transcriptions(transcriptions)

        background_noise_cls = self.add_noise_to_audio(audio)

        if len(transcriptions) == 0:
            transcriptions.append('<pad>')  # crutch for wer metric.

        return MixedExample(
            audio=audio,
            noise_id=self.noise2id[background_noise_cls],
            diarization=diarization,
            transcription=' '.join(transcriptions)
        )

    def add_noise_to_audio(self, audio: torch.Tensor, background_noise_cls=None) -> str:
        """
        :param audio: the audio where to add the noise
        :param background_noise_cls: (optional) the noise class to add
        :return: background_noise_cls (created if None)
        """
        total_length = len(audio)
        if background_noise_cls is None:
            background_noise_cls = np.random.choice(list(self.noise2id.keys()))
        noise_row_id = int(np.random.choice(self.noise2audio[background_noise_cls]))
        background_noise_audio = self.noises[noise_row_id]['audio']['array']
        snr = np.random.choice(self.snrs)

        repeats = total_length // len(background_noise_audio)
        background_noise_audio = np.tile(background_noise_audio, repeats)
        background_noise_audio = torch.concatenate(
            (background_noise_audio, torch.zeros(total_length - len(background_noise_audio))),
            dim=0)

        audio += background_noise_audio * calculate_adjustment_coef(audio, background_noise_audio, snr)

        return background_noise_cls

    def add_noise_random_place(self, audio: torch.Tensor, background_noise_cls=None) -> (int, (int, int)):
        """
        Adds noise to random place in the audio and returns the data about the noise that has been added
        :param audio: audio
        :param noise_mask: mask of shape (timestamps x num_classes + 1)
        :param background_noise_cls: noise class to add
        :return: (noise_id, (start_pos, length)) of the noise audio
        """
        total_length = len(audio)
        if background_noise_cls is None:
            background_noise_cls = np.random.choice(list(self.noise2id.keys()))
        noise_row_id = int(np.random.choice(self.noise2audio[background_noise_cls]))
        background_noise_audio = self.noises[noise_row_id]['audio']['array']
        snr = np.random.choice(self.snrs)
        noise_id = self.noise2id[background_noise_cls]

        noise_length = int(total_length * 0.1 + np.random.beta(2, 5) * total_length * 0.9)
        start_pos = np.random.randint(0, total_length - 1)
        if start_pos + noise_length > total_length:
            if start_pos + noise_length - total_length > total_length - start_pos:
                noise_length = start_pos + noise_length - total_length
                start_pos = 0
            else:
                noise_length = total_length - start_pos
        background_noise_audio = torch.tensor(expand_array(background_noise_audio, noise_length))
        background_noise_audio = torch.concatenate([torch.zeros(start_pos), background_noise_audio, torch.zeros(total_length - (start_pos + noise_length))])
        audio += background_noise_audio * calculate_adjustment_coef(audio, background_noise_audio, snr)

        return noise_id, (start_pos, noise_length)

    def generate_no_overlap(self, audio: torch.Tensor, diarization: torch.Tensor, transcriptions: List,
                            speakers: np.array) -> None:
        """
        Generates example without overlapped speech.
        :param audio: place for audio to be added
        :param diarization: place for diarization to be added
        :param transcriptions: place for transcriptions in the correct order
        :param speakers: speakers to use for generation
        :return: None
        """
        filled_length = 0
        total_length = len(audio)
        num_speakers = len(speakers)

        while filled_length < total_length:
            speaker_index = np.random.randint(num_speakers)
            speaker = speakers[speaker_index]
            filled_length = self.add_one_speaker(audio, diarization, transcriptions, speaker, filled_length)

    def generate_overlap(self, audio: torch.Tensor, diarization: torch.Tensor, transcriptions: List,
                         speakers: np.array) -> None:
        """
        Generates example with overlapped speach
        :param audio: place for audio to be added
        :param diarization: place for diarization to be added
        :param transcriptions: place for transcriptions in the correct order
        :param speakers: speakers to use for generation
        :return: None
        """
        for speaker_index, speaker in enumerate(speakers):
            self.add_one_speaker(audio, diarization, transcriptions, speaker_index, speaker, 0)

    def add_one_speaker(self, audio: torch.Tensor, diarization: torch.Tensor, transcriptions: List, speaker_index: int,
                        speaker, filled_length: int) -> int:
        """
        Appends one speaker's random utterances to the filled_length part of audio.
        :param audio: audio to append
        :param diarization: diarization array to fill
        :param transcriptions: transcription array to fill
        :param speaker_index: index of speaker in terms of the current example
        :param speaker: speaker name in terms of the whole mixer
        :param filled_length: start time to begin appending
        :return: filled_length
        """
        n_u = np.random.randint(self.config.min_repetitions, self.config.max_repetitions + 1)
        for u in range(n_u):
            if filled_length >= audio.shape[0]:
                break

            utterance_id = int(np.random.choice(self.speakers2utterance[speaker]))
            utterance_audio: torch.Tensor = self.utterances[utterance_id]['audio']['array']

            max_size = audio.shape[0] - filled_length

            d = int(np.random.exponential(scale=self.config.beta) * self.config.output_sampling_rate)
            d = min(d, max_size)

            filled_length += d
            max_size -= d

            loaded_audio = utterance_audio
            if len(loaded_audio) * 0.8 > max_size:
                break
            loaded_audio = loaded_audio[:max_size]

            diarization[filled_length // self.frame_resolution:filled_length // self.frame_resolution + len(
                loaded_audio) // self.frame_resolution, speaker_index] = 1
            audio[filled_length: filled_length + len(loaded_audio)] += loaded_audio

            transcriptions.append((filled_length, self.utterances[utterance_id]['transcription'], speaker))
            filled_length += len(utterance_audio)

            # add blanks.

        return filled_length

    def process_transcriptions(self, transcriptions: List[Tuple[int, str]]) -> List[str]:
        """
        Place for implementing more sophisticated methods of transcription processing, like adding <ch> tokens etc.
        :param transcriptions: list of transcriptions with timestamps.
        :return: List[str] - processed transcriptions.
        """
        return [x[1] for x in sorted(transcriptions, key=lambda x: x[0])]


class DatasetMixerWrapper(torch.utils.data.Dataset):
    """
    Wrapper for torch.utils.dataDataset.
    """
    def __init__(self, mixer: DatasetMixer):
        super().__init__()
        self.mixer = mixer

    def __getitem__(self, idx):
        return self.mixer.generate()

    def __len__(self):
        return self.mixer.config.utterances_count
