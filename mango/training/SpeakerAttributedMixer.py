import logging
from dataclasses import dataclass

import numpy as np
import torch
import datasets
from datasets import Audio
from typing import List, Tuple

from .DatasetMixer import DatasetMixerConfig, DatasetMixer, MixedExample


@dataclass
class SpeakerAttributeExample(MixedExample):
    speaker_attributions: torch.Tensor = None


class SpeakerAttributedMixer(DatasetMixer):

    def __init__(self, config: DatasetMixerConfig, utterances: datasets.Dataset, noises: datasets.Dataset, rirs=None):
        super().__init__(config, utterances, noises, rirs)
        self.speaker_change_token = '<sc>'

    def generate(self) -> SpeakerAttributeExample:
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
        transcriptions, speaker_attributions = self.process_transcriptions(transcriptions)

        sample_num_noises = np.random.randint(self.config.min_noises, self.config.max_noises + 1)
        noise_ids = []
        arr = list(self.noise2id.keys())
        for i in range(sample_num_noises):
            noise_label = np.random.choice(arr)
            noise_ids.append(self.noise2id[noise_label])
            arr.remove(noise_label)
            assert self.config.noise_sample_type in ["whole", "random"]
            if self.config.noise_sample_type == "whole":
                self.add_noise_to_audio(audio, background_noise_cls=noise_label)
            else:
                self.add_noise_random_place(audio, background_noise_cls=noise_label)

        if len(transcriptions) == 0:
            transcriptions.append('<pad>')  # crutch for wer metric.

        return SpeakerAttributeExample(
            audio=audio,
            noise_id=noise_ids,
            diarization=diarization,
            transcription=' '.join(transcriptions),
            speaker_attributions=torch.tensor(speaker_attributions, dtype=torch.long)
        )

    def process_transcriptions(self, transcriptions: List[Tuple[int, str, str]]) -> Tuple[List[str], List[int]]:
        """
        Place for implementing more sophisticated methods of transcription processing, like adding <ch> tokens etc.
        :param transcriptions: list of transcriptions with timestamps.
        :return: List[str] - processed transcriptions.
        """
        start_ordered_transcriptions = sorted(transcriptions, key=lambda x: x[0])
        words = []
        speaker_attributions = []
        for start_time, transcription, speaker_id in start_ordered_transcriptions:
            current_words = transcription.split()

            words.extend(current_words)
            words.append(self.speaker_change_token)

            speaker_attributions.extend([self.speaker2id[speaker_id]] * (len(current_words) + 1))

        return words, speaker_attributions

