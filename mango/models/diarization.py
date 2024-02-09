import torch
from torch import nn
from ..utils.diarization import batch_pit_loss
from dataclasses import dataclass
from typing import Dict
from transformers.models.whisper.modeling_whisper import WhisperEncoder


@dataclass
class InternalEENDConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 1024
    num_speakers: int = 2


class MangoEEND(nn.Module):
    """
    Model implements kind of EENND-SA.
    """

    def __init__(self, config: InternalEENDConfig):
        """
        Instantiates a MangoDiarization model with a given config
        :param config: model configuration
        """
        super().__init__()
        self.config = config
        self.features = nn.Sequential(nn.Conv1d(80, config.d_model, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                                   dim_feedforward=config.dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.num_speakers))

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """

        :param input_features: mel-banks of shape (B, S, 80)
        :param labels: onehot diarization labels of shape (B, S`, M)
        :return: dictionary of model logits, PIT-loss and best permutations (alignments).
        """
        features = self.features(input_features).permute(2, 0, 1)

        features = self.transformer(features).permute(1, 0, 2)
        logits = self.classifier(features)

        loss, alignments = None, None
        if labels is not None:
            loss, alignments = batch_pit_loss(labels, logits, self.config.num_speakers)

        return {'logits': logits, 'loss': loss, 'alignments': alignments}


@dataclass
class WhisperBasedEENDConfig:
    num_speakers: int = 2
    whisper_checkpoint: str = 'openai/whisper-small'
    classifier_dim = 512


class WhisperBasedEEND(nn.Module):

    def __init__(self, config: WhisperBasedEENDConfig):
        """
        Instantiates a WhusperBasedEEND model with a given config
        :param config:
        """
        super().__init__()
        self.config = config
        self.whisper = WhisperEncoder.from_pretrained(config.whisper_checkpoint)
        self.classifier = nn.Sequential(nn.Linear(self.whisper.config.d_model, self.config.classifier_dim), nn.ReLU(),
                                        nn.Linear(config.classifier_dim, self.config.num_speakers))

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.whisper(input_features)
        logits = self.classifier(features)

        loss, alignments = None, None
        if labels is not None:
            loss, alignments = batch_pit_loss(labels, logits, self.config.num_speakers)

        return {'logits': logits, 'loss': loss, 'alignments': alignments}

