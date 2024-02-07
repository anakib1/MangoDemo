import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class SA_ASRConfig:
    d_model: int = 384
    n_heads: int = 4
    dim_feedforward: int = 1024
    n_layers_speech: int = 2
    n_layers_speaker: int = 2


class SA_ASR(nn.Module):
    def __init__(self, config: SA_ASRConfig):
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                                   dim_feedforward=config.dim_feedforward)
        self.speechEncoder = nn.TransformerEncoder(encoder_layer, config.n_layers_speech, enable_nested_tensor=False)
        self.speakerEncoder = nn.TransformerEncoder(encoder_layer, config.n_layers_speaker, enable_nested_tensor=False)
        self.speechAttention = nn.MultiheadAttention(self.d_model, 1)

        self.asrRnn = nn.RNN(config.d_model, config.d_model, batch_first=True)

        super().__init__()

    def forward(self, input_features: torch.Tensor, speaker_inventory:torch.Tensor):
        asr_features = self.speechEncoder(input_features)
        speaker_features = self.speakerEncoder(input_features)
        speech_rnn_features = self.asrRnn(asr_features)
        speech_encoder_




