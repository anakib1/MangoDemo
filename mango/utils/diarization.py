import torch
from typing import Dict, Tuple
import numpy as np
from itertools import permutations
from torch.nn import functional as F
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import io


def calculate_diarization_accuracy(labels: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
    """
    Calculates the diarization metrics for passed labels and preds in ONE batch
    :param labels: tensor of shape (N, M) of 0 and 1 - diarization labels
    :param preds: tensor of shape (N, M) of floats - diarization logits using the best permutation.
    :return: dictionary with various diarization metrics.
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(labels, torch.Tensor):
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


def pit_loss(labels, preds, num_speakers):
    label_perms = [labels[..., list(p)] for p
                   in permutations(range(num_speakers))]

    losses = torch.stack([F.binary_cross_entropy_with_logits(preds.float(), label.float()) for label in label_perms])

    min_loss = losses.min() * len(labels)
    min_index = losses.argmin().detach()

    return min_loss, label_perms[min_index]


def batch_pit_loss(labels, preds, num_speakers: int) -> Tuple[torch.FloatType, torch.Tensor]:
    loss_w_labels = [pit_loss(label, pred, num_speakers)
                     for (label, pred) in zip(labels, preds)]
    losses, perms = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([label.shape[0] for label in labels])
    loss = loss / n_frames
    return loss, torch.stack(perms)


def draw_diarization(data: np.ndarray) -> np.ndarray:
    """
    Returns picture of speaker diarization.
    :param data: torch.tensor: model output of shape [1, T, config.n_speakers]
    :return: np.ndarray of shape [400, 800, 4] - picture of diarization.
    """
    if len(data.shape) == 3:
        if data.shape[0] != 1:
            logging.warning('Batched input passed, incorrect behaviour is about to happen.')
        data = data.squeeze(0)

    # assumes frame resolution 20 ms
    res = np.zeros((int(np.ceil(data.shape[0] / 50)), data.shape[1]))
    for j in range(0, data.shape[0], 50):
        res[j//50, :] = np.median(data[j:j+50, :], axis=0) >= 0.5

    fig, ax = plt.subplots()
    ax.imshow(res.T, cmap=ListedColormap(['white', 'black']))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_yticklabels([f'Speaker {i}' for i in range(data.shape[1])])
    with io.BytesIO() as buff:
        fig.savefig(buff, format='png')
        buff.seek(0)
        im = plt.imread(buff)
    return im
