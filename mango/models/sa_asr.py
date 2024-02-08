import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict
from torch.nn import functional as F


@dataclass
class SAASRConfig:
    d_model: int = 384
    n_heads: int = 4
    dim_feedforward: int = 1024
    n_layers_speech: int = 2
    n_layers_speaker: int = 2
    input_dim: int = 80
    max_text_sequence_length: int = 256
    audio_sequence_length: int = 1500
    vocab_size: int = 60
    gama: float = 1
    pad_token_id: int = 0


SOS_token = 0


class SAASR(nn.Module):
    def __init__(self, config: SAASRConfig):
        super().__init__()
        self.config = config

        self.feature_extractor = nn.Sequential(nn.Conv1d(config.input_dim, config.d_model, 3, padding=1),
                                               nn.ReLU(),
                                               nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1),
                                               nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                                   dim_feedforward=config.dim_feedforward, batch_first=True)
        self.asrEncoder = nn.TransformerEncoder(encoder_layer, config.n_layers_speech, enable_nested_tensor=False)
        self.diarEncoder = nn.TransformerEncoder(encoder_layer, config.n_layers_speaker, enable_nested_tensor=False)
        self.attention = nn.MultiheadAttention(num_heads=1, batch_first=True, embed_dim=1, kdim=1, vdim=384)
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.asr_rnn = nn.GRUCell(config.d_model * 2, config.audio_sequence_length)
        self.diar_rnn = nn.GRUCell(config.d_model * 2, config.d_model)

        self.asr_out = nn.Linear(config.d_model, config.vocab_size)

        self.diar_to_asr = nn.Linear(config.d_model, config.d_model)
        self.hidden_to_asr = nn.Linear(config.audio_sequence_length, config.d_model)

    def forward(self, input_features: torch.Tensor, speaker_inventory: torch.Tensor, target_asr_ids: torch.Tensor,
                target_diar_ids: torch.Tensor) -> Dict[str, torch.Tensor]:

        input_features = self.feature_extractor(input_features).permute(0, 2, 1)

        batch_size = input_features.size(0)
        encoder_asr_features = self.asrEncoder(input_features)
        encoder_diar_features = self.diarEncoder(input_features)

        current_token = torch.empty(batch_size, 1, dtype=torch.long, device=input_features.device).fill_(SOS_token)

        asr_context = torch.zeros(batch_size, self.config.d_model, device=input_features.device)
        asr_decoder_hidden_state = torch.zeros(batch_size, self.config.audio_sequence_length,
                                               device=input_features.device)
        diar_decoder_hidden_state = torch.zeros(batch_size, self.config.d_model, device=input_features.device)
        attentions = torch.zeros(batch_size, self.config.audio_sequence_length, 1, device=input_features.device)

        asr_outputs = []
        diar_outputs = []

        for i in range(self.config.max_text_sequence_length):
            current_token_embedding = self.embeddings(current_token).squeeze(1)
            asr_decoder_hidden_state = self.asr_rnn(
                torch.concatenate([asr_context, current_token_embedding], dim=1),
                asr_decoder_hidden_state)

            attentions, _ = self.attention(asr_decoder_hidden_state.unsqueeze(2), attentions, encoder_asr_features)
            asr_context = torch.bmm(encoder_asr_features.transpose(2, 1), attentions).squeeze(2)
            speaker_context = torch.bmm(encoder_diar_features.transpose(2, 1), attentions).squeeze(2)

            diar_decoder_hidden_state = self.diar_rnn(
                torch.concatenate([speaker_context, current_token_embedding], dim=1), diar_decoder_hidden_state)

            scores = torch.bmm(speaker_inventory, diar_decoder_hidden_state.unsqueeze(2)).squeeze(2)
            norm = torch.norm(diar_decoder_hidden_state.unsqueeze(2), p=2, dim=1) * torch.norm(speaker_inventory, p=2,
                                                                                               dim=2).reshape(
                batch_size, speaker_inventory.shape[1])
            beta_attentions = F.softmax(scores / norm, dim=1)

            speaker_state = torch.bmm(speaker_inventory.transpose(2, 1), beta_attentions.unsqueeze(2)).squeeze(2)

            asr_logits = self.asr_out(
                asr_context + self.hidden_to_asr(asr_decoder_hidden_state) + self.diar_to_asr(speaker_state))

            asr_outputs.append(asr_logits)
            diar_outputs.append(beta_attentions)

            if target_asr_ids is not None:
                current_token = target_asr_ids[:, i].unsqueeze(1)
                current_token = current_token.masked_fill(current_token == -100, self.config.pad_token_id)
            else:
                _, topi = asr_logits.topk(1)
                current_token = topi.squeeze(-1).detach()

        asr_outputs = torch.stack(asr_outputs, dim=1)
        diar_outputs = torch.stack(diar_outputs, dim=1)

        loss = None
        if target_diar_ids is not None and target_asr_ids is not None:
            loss = (F.nll_loss(asr_outputs.transpose(2, 1), target_asr_ids) + self.config.gama *
                    F.nll_loss(diar_outputs.transpose(2, 1), target_diar_ids))

        return {
            'asr_outputs': asr_outputs,
            'diar_outputs': diar_outputs,
            'asr_labels': target_asr_ids,
            'diar_labels': target_diar_ids,
            'loss': loss
        }
