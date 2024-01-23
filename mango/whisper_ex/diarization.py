import torch
from torch import nn
from itertools import permutations
from torch.nn import functional as F
import numpy as np

from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperModel


def calculate_diarization_accuracy(labels: torch.Tensor, preds: torch.Tensor):
    if type(preds) != torch.Tensor:
        preds = torch.tensor(preds)
    if type(labels) != torch.Tensor:
        labels = torch.tensor(labels)

    decisions = torch.sigmoid(preds) > 0.5
    n_ref = labels.sum(dim=-1)
    n_pred = decisions.sum(dim=-1)

    res = {}
    res['speech_scored'] = int(((n_ref > 0) & (n_pred > 0)).sum())  # frames with speech in both
    res['speech_miss'] = int(
        ((n_ref > 0) & (n_pred == 0)).sum())  # frames with speech in ref and w/o speech in pred
    res['speech_falarm'] = int(
        ((n_ref == 0) & (n_pred > 0)).sum())  # frames with speech in pred and w/o speech in label
    res['speaker_miss'] = int(
        torch.max((n_ref - n_pred), torch.zeros_like(n_ref)).sum())  # number of speaker-wide frame missed by model
    res['speaker_falarm'] = int(torch.max((n_pred - n_ref), torch.zeros_like(
        n_ref)).sum())  # number of speaker-wide frame false alarmed by model
    n_map = ((labels == 1) & (decisions == 1)).sum(dim=-1)
    res['speaker_error'] = int(
        (torch.min(n_ref, n_pred) - n_map).sum())  # number of speaker-wide frames confused by model
    res['speaker_correct'] = float(
        (labels == decisions).sum() / labels.shape[1])  # total number of correct guesses speakerwide
    res['diarization_error'] = int(
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])  # diarization error

    res['frames'] = len(labels)  # total number of frames
    res['speaker_wide_frames'] = int(n_ref.sum())  # total number of spears in speaker-wide frames

    for key in ['speech_scored', 'speech_miss', 'speech_falarm', 'speaker_correct']:
        other_key = key + '_ratio'
        res[other_key] = res[key] / res['frames']

    for key in ['speaker_miss', 'speaker_falarm', 'speaker_error', 'diarization_error']:
        other_key = key + '_ratio'
        res[other_key] = res[key] / res['speaker_wide_frames']

    return res


class WhisperConfigForSpeakerDiarization(WhisperConfig):
    def __init__(self, max_num_speakers: int = 2, *args, **kwargs):
        self.max_num_speakers = max_num_speakers
        super().__init__(*args, **kwargs)


class WhisperModelForSpeakerDiarization(WhisperModel):
    def pit_loss(self, labels, preds):
        label_perms = [labels[..., list(p)] for p
                       in permutations(range(self.config.max_num_speakers))]

        losses = torch.stack([F.binary_cross_entropy_with_logits(preds.float(), l.float()) for l in label_perms])

        min_loss = losses.min() * len(labels)
        min_index = losses.argmin().detach()

        return min_loss, label_perms[min_index]

    def batch_pit_loss(self, labels, preds):
        loss_w_labels = [self.pit_loss(label, pred)
                         for (label, pred) in zip(labels, preds)]
        losses, perms = zip(*loss_w_labels)
        loss = torch.stack(losses).sum()
        n_frames = np.sum([label.shape[0] for label in labels])
        loss = loss / n_frames
        return loss, torch.stack(perms)

    def __init__(self, config: WhisperConfigForSpeakerDiarization):
        super().__init__(config)
        self.linear_layer = nn.Linear(config.d_model, self.config.max_num_speakers)
        self.post_init()


    def forward(self, input_features, labels=None, attention_mask=None, head_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        """
        Performs speaker diarization on the given features extracted by processor. Returns

        Namings: T - initial seq len, T1 - extracted sequence len

        :param input_features: torch.Tensor of [B, T, D] shape
        :param labels: torch.Tensor of [B, T1, config.n_speakers] shape
        :param attention_mask: torch.Tensor of [B, T] shape
        :param head_mask: todo
        :param output_attentions: todo
        :param output_hidden_states: todo
        :param return_dict: todo
        :return: dict of three elements:
        loss - torch.scalar the loss of the diarization los on the best permutation,
        preds - torch.Tensor of [B, T1, config.n_speakers] of unnormalized logits - the predictions of the diarization (according to the best permutation),
        perm - torch.LongTensor of [B, T1, config.n_speakers] the best permutation that was performed (some permutation of labels).
        """
        input_features = self._mask_input_features(input_features, attention_mask)
        encoder_output = self.encoder(input_features, attention_mask, head_mask, output_attentions,
                                      output_hidden_states, return_dict)

        pred = self.linear_layer(encoder_output.last_hidden_state)

        loss, perms = None, None
        if labels is not None:
            loss, perms = self.batch_pit_loss(labels, pred)

        return {'loss': loss, 'logits': pred, 'perm': perms}
