from transformers import WhisperFeatureExtractor
import torch
from typing import List, Dict
from ..data.base import SegmentInfo, MixedExample
from ..globals import RATE


class Whisper:

    @staticmethod
    def update_segment_infos(audio: torch.Tensor, infos: List[SegmentInfo], max_len_sec: float = 30.0):
        """
        Transforms segment infos according to whisper feature extractor output
        !!!updates info in-place
        :param max_len_sec: maximum audio length in seconds
        """
        num_secs = audio.shape[0] / RATE
        if num_secs > max_len_sec:
            raise Exception("audio can not be longer than 30 seconds")
        norm_coef = num_secs / 30.0
        for info in infos:
            info.start *= norm_coef
            info.end *= norm_coef

    @staticmethod
    def update_mixed_example(example: MixedExample):
        """
        Transforms MixedExample segment infos according to whisper feature extractor output
        !!!updates info in-place
        """
        if example.noises_info is not None:
            Whisper.update_segment_infos(example.audio, example.noises_info)
        if example.speakers_info is not None:
            Whisper.update_segment_infos(example.audio, example.speakers_info)

    @staticmethod
    def extract_features(extractor: WhisperFeatureExtractor, audio: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extracts features using Whisper feature extractor
        """
        list_feats = [extractor(x, return_tensors="pt", return_attention_mask=True, sampling_rate=RATE) for x in audio]
        dict_feats = dict(
            input_features=torch.cat([x["input_features"] for x in list_feats], dim=0),
            attention_mask=torch.cat([x["attention_mask"] for x in list_feats], dim=0)
        )
        return dict_feats


class Labels:

    @staticmethod
    def create_labels(infos: List[List[SegmentInfo]], num_classes: int, timestamps: int) -> torch.Tensor:
        """
        The function creates the labels for diarization
        :param infos: list of segment infos from MixedExample
        :param num_classes: number of classes to predict in audio
            (number of classes in dataset + void class)
        :param features: num of features in model output embedding
        :param timestamps: the number of timestamps in model output embedding
        :return: the labels with shape (batch x timestamps x features x num_classes)
            where 1 - means present and 0 - not present
        """
        ans = torch.zeros((len(infos), timestamps, num_classes))
        ans[:, :, 0] = 1.0
        for batch_i in range(len(infos)):
            for info in infos[batch_i]:
                start = int(info.start * timestamps)
                end = int(info.end * timestamps)
                ans[batch_i, start:end, info.class_id] = 1.0
                ans[batch_i, start:end, 0] = 0.0
        return ans

    @staticmethod
    def resize_attention_mask(mask: torch.Tensor, num_classes: int, new_size: int = None) -> torch.Tensor:
        """
        :param mask: mask of shape (batch x timestamps)
        :param num_classes: number of classes in the dataset (+ void class)
        :param new_size: new number of timestamps
        :return: the new mask that loss function usually needs
            the shape is (batch x timestamps x features x num_classes)
        """
        batch_size = mask.shape[0]
        if new_size is None:
            new_size = batch_size

        ans = torch.zeros((batch_size, new_size, num_classes))
        for i in range(batch_size):
            length = int(mask[i].sum().numpy() / mask.shape[1] * new_size)
            ans[i][:length] = 1.0
        return ans













