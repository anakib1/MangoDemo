from torch import nn
import torch
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from typing import Dict
from ..utils.diarization import batch_pit_loss


@dataclass
class InternalWav2AllConfig:
    wav2vec_checkpoint: str = 'facebook/wav2vec2-base-960h'
    diarization_layer_id: int = 4
    classification_layer_id: int = 4
    ffn_dim: int = 512
    num_speakers: int = 2
    num_noises: int = 2
    asr_loss_weight = 1.0
    diar_loss_weight = 1.0
    clf_loss_weight = 1.0


class Wav2All(nn.Module):

    def __init__(self, config: InternalWav2AllConfig):
        """
        Instantiates a MangoDiarization model with a given config
        :param config: model configuration
        """
        super().__init__()
        self.config = config
        self.wav2vec2_config = Wav2Vec2Config.from_pretrained(config.wav2vec_checkpoint)
        self.wav2vec2: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(config.wav2vec_checkpoint)
        self.diarization_ffn = nn.Sequential(nn.Linear(self.wav2vec2_config.output_hidden_size, self.config.ffn_dim),
                                             nn.Linear(self.config.ffn_dim, self.config.num_speakers))
        self.classification_ffn = nn.Sequential(nn.Linear(self.wav2vec2_config.output_hidden_size, self.config.ffn_dim),
                                                nn.Linear(self.config.ffn_dim, self.config.num_noises))

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor = None, diarization_labels: torch.Tensor = None,
                classification_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        wav2vec2output = self.wav2vec2(input_values, labels = labels, output_hidden_states=True)
        hidden_states = wav2vec2output.hidden_states

        total_loss = torch.tensor(0.0, device = hidden_states[0].device)

        ret = {}
        if self.config.asr_loss_weight > 0:
            ret['asr_logits'] = wav2vec2output.logits
            ret['asr_loss'] = wav2vec2output.loss
            ret['asr_labels'] = labels  # todo remove
            total_loss += wav2vec2output.loss * self.config.asr_loss_weight

        if self.config.diar_loss_weight > 0:
            diarization_logits = self.diarization_ffn(hidden_states[self.config.diarization_layer_id])
            diarization_loss, best_alignments = batch_pit_loss(diarization_labels, diarization_logits, self.config.num_speakers)
            ret['diarization_loss'] = diarization_loss
            ret['diarization_logits'] = diarization_logits
            ret['diarization_alignment'] = best_alignments
            total_loss += diarization_loss * self.config.diar_loss_weight

        if self.config.clf_loss_weight > 0:
            classification_logits = self.classification_ffn(hidden_states[self.config.classification_layer_id]).mean(
                dim=1)
            classification_loss = nn.CrossEntropyLoss()(classification_logits, classification_labels)
            ret['classification_loss'] = classification_loss
            ret['classification_logits'] = classification_logits
            ret['classification_labels'] = classification_labels
            total_loss += classification_loss * self.config.clf_loss_weight

        ret['loss'] = total_loss
        return ret
