from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperModel
from ..utils.diarization import batch_pit_loss

class WhisperConfigForSpeakerDiarization(WhisperConfig):
    def __init__(self, max_num_speakers: int = 2, *args, **kwargs):
        self.max_num_speakers = max_num_speakers
        super().__init__(*args, **kwargs)


class WhisperModelForSpeakerDiarization(WhisperModel):

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
            loss, perms = batch_pit_loss(labels, pred, self.config.max_num_speakers)

        return {'loss': loss, 'logits': pred, 'perm': perms}
