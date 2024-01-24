import json
import logging

import torchaudio
from typing import Tuple, Dict
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import WhisperProcessor
from typing import Optional
import os


class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = len(data_dict[self.keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = {key: self.data_dict[key][index] for key in self.keys}
        return sample


class FileDataset(Dataset):
    def __init__(self, files, generate_func):
        self.files = files
        self.generate_func = generate_func

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.generate_func(self.files[index])


@dataclass
class SynthConfig:
    dataset_prefix: str = '/kaggle/input/synthnooverlap-2/SynthCommonVoice0.3'
    max_num_speakers: int = 3
    noises2id: Dict[str, int] = None


class SynthDataset:
    def __init__(self, config: SynthConfig, processor: Optional[WhisperProcessor] = None):
        self.lazy_dataset = None
        self.dataset = None
        self.config = config
        if processor is None:
            if hasattr(config, 'base_checkpoint'):
                self.processor = WhisperProcessor.from_pretrained(config.base_checkpoint)
            else:
                raise Exception('You need to pass processor argument, or base_checkpoint field in config')
        else:
            self.processor = processor

    def parse_annotation(self, split: str, annotation_id: str) -> Tuple[np.ndarray, Dict, str, str]:
        box_size = 20

        annotation_path = f'{self.config.dataset_prefix}/Annotations/{split}/{annotation_id}.rttm'
        meta_path = f'{self.config.dataset_prefix}/Annotated/{split}/{annotation_id}.uem'

        with open(meta_path, 'r') as f:
            lines = f.readlines()
        if len(lines) != 1:
            print('Warn, found invalid .uem: ', meta_path)
            return None

        words = lines[0].split()
        if len(words) != 4:
            print('Warn, found invalid .uem: ', meta_path)
            return None

        total_length = float(words[-1])

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        arr_len = int((total_length * 1_000) // box_size)  # number of 20-ms boxes.

        labels = np.zeros((arr_len, self.config.max_num_speakers))

        speakers_map = {}

        for line in lines:
            words = line.split()
            if len(words) != 10:
                print('Warn, found invalid line: ', line)
                continue
            speaker_name = words[7]
            start_turn = float(words[3])
            turn_duration = float(words[4])

            if not speaker_name in speakers_map:
                if len(speakers_map) == self.config.max_num_speakers:
                    return None
                speakers_map[speaker_name] = len(speakers_map)

            current_speaker_id = speakers_map[speaker_name]
            start_index = int(start_turn * 1_000 // box_size)
            index_len = int((turn_duration * 1000) // box_size)

            labels[start_index:start_index + index_len, current_speaker_id] = 1

        transcriptions_json = json.load(
            open(f'{self.config.dataset_prefix}/Lists/{split}.json', 'r', encoding='UTF-8'))

        return labels, speakers_map, transcriptions_json[annotation_id]['transcription'], \
            transcriptions_json[annotation_id]['noise_class']

    def generate_example(self, split: str, example_id: str) -> Dict:
        """
        Generates one example from split name and example id.
        :param split: split name
        :param example_id: example uuid as atr
        :return: Tuple of audio sampled in 16000 freq, diarization labels per 20-ms frame, transcription string,
        noise class string, speakers map dictionary - speaker name from diarization labels.
        """
        labels, speaker_map, transcription, noise_cls = self.parse_annotation(split, example_id)
        auido_arr, sr = torchaudio.load(self.config.dataset_prefix + f'/Audio/{split}/{example_id}.wav')
        audio_arr = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16_000)(auido_arr)[0]

        input_features = self.processor(audio_arr, sampling_rate=16_000).input_features[0]

        return {
            'input_features': input_features,
            'labels': self.processor(text=transcription).input_ids,
            'diarization_labels': labels,
            'classification_labels': self.config.noises2id.get(noise_cls, 'UNDEFINED'),
            'speaker_map': speaker_map
        }

    def generate_split(self, split_name: str, split_limit: int) -> DictDataset:
        data_dict = dict()
        for example in tqdm(os.listdir(f'{self.config.dataset_prefix}/Audio/{split_name}')[:split_limit]):
            example_id = example.split('.')[0]
            example = self.generate_example(split_name, example_id)
            for k, v in example.items():
                if not k in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)

        return DictDataset(data_dict)

    def generate_synth(self, limit: Optional[int] = 10_000) -> Dict[str, DictDataset]:
        self.dataset = {
            'train': self.generate_split('train', limit),
            'dev': self.generate_split('dev', limit),
            'test': self.generate_split('test', limit)
        }

        return self.dataset

    def lazy_generate_split(self, split_name, split_limit: int):
        files = list(map(lambda x: x.split('.')[0],
                         os.listdir(f'{self.config.dataset_prefix}/Audio/{split_name}')[:split_limit]))
        logging.info(f"Added {len(files)} lazy filenames for {split_name} split.")
        return FileDataset(files, lambda filename: self.generate_example(split_name, filename))

    def lazy_generate(self, limit: Optional[int] = 10_000):
        self.lazy_dataset = {
            'train': self.lazy_generate_split('train', limit),
            'dev': self.lazy_generate_split('dev', limit),
            'test': self.lazy_generate_split('test', limit)
        }

    @property
    def train(self):
        if self.dataset is not None:
            return self.dataset['train']
        return self.lazy_dataset['train']

    @property
    def test(self):
        if self.dataset is not None:
            return self.dataset['test']
        return self.lazy_dataset['test']

    @property
    def dev(self):
        if self.dataset is not None:
            return self.dataset['dev']
        return self.lazy_dataset['dev']
