from dataclasses import dataclass
from torch import nn
import torch
from typing import Dict
from transformers.models.whisper.modeling_whisper import WhisperEncoder


@dataclass
class InternalClassifierConfig:
    num_labels: int = 10
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 1024


class EncoderClassifier(nn.Module):
    """
    Model implements kind of EENND-SA.
    """

    def __init__(self, config: InternalClassifierConfig):
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
        self.classifier = nn.Sequential(nn.LayerNorm(config.d_model),
                                        nn.Linear(config.d_model, config.num_labels))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        features = self.features(input_features).permute(2, 0, 1)

        features = self.transformer(features).permute(1, 0, 2).mean(dim=1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return {'logits': logits, 'loss': loss, 'labels': labels}


@dataclass
class WhisperBasedCLFConfig:
    num_labels: int = 10
    whisper_checkpoint: str = 'openai/whisper-tiny'
    classifier_dim = 512


class WhisperBasedClassifier(nn.Module):

    def __init__(self, config: WhisperBasedCLFConfig):
        """
        Instantiates a WhusperBasedClassifier model with a given config
        :param config:
        """
        super().__init__()
        self.config = config
        self.whisper = WhisperEncoder.from_pretrained(config.whisper_checkpoint)
        self.classifier = nn.Sequential(nn.Linear(self.whisper.config.d_model, self.config.classifier_dim), nn.ReLU(),
                                        nn.Linear(config.classifier_dim, self.config.num_labels))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.whisper(input_features).last_hidden_state.mean(dim=1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return {'logits': logits, 'loss': loss, 'labels': labels}
