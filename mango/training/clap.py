import torch

from .MangoTrainer import MangoTrainer, TrainingOutput, TrainerConfig, EvalOutput
from typing import Dict, Union

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClapTrainerConfig(TrainerConfig):
    num_repeats: int = 1


class MultiGpuClap(torch.nn.Module):
    def __init__(self, audio_encoder: torch.nn.Module, text_encoder: torch.nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, audio: Union[torch.Tensor, Dict], text: Union[torch.Tensor, Dict],
                calculate_loss: bool = False) -> Dict:
        if calculate_loss:
            logits = text @ torch.transpose(audio, 0, 1)
            N = logits.shape[0]
            ans = {"logits": logits}
            target = torch.arange(0, N, dtype=torch.long).to(logits.device)
            loss = self.loss_fn(logits, target) + self.loss_fn(torch.transpose(logits, 0, 1), target)
            ans["loss"] = loss
            return ans
        else:
            return {'audio_embeddings': self.audio_encoder(**audio),
                    'text_embeddings': self.text_encoder(**text)}


class ClapTrainer(MangoTrainer):
    config: ClapTrainerConfig
    model: MultiGpuClap

    def train_iteration(self, epoch_index: int) -> TrainingOutput:
        self.model.train()

        train_outputs = {}
        losses = []

        logger.debug(f'Starting epoch {epoch_index}')

        accumulated_text_inputs = []
        accumulated_audio_inputs = []
        accumulated_text_embeddings = []
        accumulated_audio_embeddings = []

        for i, batch in enumerate(self.train_loader):

            with torch.no_grad():
                accumulated_text_inputs.append(batch['text_model_input'])
                accumulated_audio_inputs.append(batch['audio_model_input'])
                embed = self.model(batch['audio_model_input'], batch['text_model_input'])
                accumulated_text_embeddings.append(embed['audio_embeddings'])
                accumulated_audio_embeddings.append(embed['text_embeddings'])
                del embed

            if (i + 1) % self.config.num_repeats > 0:
                continue

            self.optimizer.zero_grad()
            for j in range(self.config.num_repeats):
                embed = self.model(batch['audio_model_input'], batch['text_model_input'])
                gradient_text_embeddings = embed['audio_embeddings']
                gradient_audio_embeddings = embed['text_embeddings']

                output = self.model(torch.concatenate(
                    accumulated_audio_embeddings[:j] + [gradient_audio_embeddings] + accumulated_audio_embeddings[
                                                                                     j + 1:], dim=0),
                    torch.concatenate(accumulated_text_embeddings[:j] + [
                        gradient_text_embeddings] + accumulated_text_embeddings[j + 1:], dim=0),
                    calculate_loss=True)

                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                self.accelerator.backward(loss)
                if self.config.grad_clip:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                del embed

            self.optimizer.step()

            with torch.no_grad():
                output = self.model(torch.concatenate(accumulated_audio_embeddings, dim=0),
                                    torch.concatenate(accumulated_text_embeddings, dim=0),
                                    calculate_loss=True)

            for k, v in output.items():
                if k not in train_outputs:
                    train_outputs[k] = []
                train_outputs[k].append(self.accelerator.gather_for_metrics(v))
            loss = output['loss']
            loss = self.accelerator.gather_for_metrics(loss).cpu().numpy()
            try:
                losses.extend(loss)
            except Exception:
                losses.append(loss)

            accumulated_text_inputs = []
            accumulated_audio_inputs = []
            accumulated_text_embeddings = []
            accumulated_audio_embeddings = []

            if self.accelerator.is_main_process:
                self.train_bar.update(1)
                self.global_train_step += 1
                if self.global_train_step % self.config.logs_frequency_batches == 0:
                    logger.info(f'Global train step {self.global_train_step}')

        return TrainingOutput(epoch_id=epoch_index, losses=losses,
                              model_outputs=self.group_predictions(train_outputs))

    def eval_iteration(self, epoch_index) -> EvalOutput:
        """
        Evaluation iteration of training
        :param epoch_index: id of the current epoch
        :return: result of evaluation iteration
        """
        self.model.eval()

        model_outputs = {}
        losses = []

        accumulated_text_embeddings = []
        accumulated_audio_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                embed = self.model(batch['audio_model_input'], batch['text_model_input'])
                accumulated_text_embeddings.append(embed['audio_embeddings'])
                accumulated_audio_embeddings.append(embed['text_embeddings'])

                if (i + 1) % self.config.num_repeats > 0:
                    continue

                output = self.model(torch.concatenate(accumulated_audio_embeddings, dim=0),
                                         torch.concatenate(accumulated_text_embeddings, dim=0),
                                         calculate_loss=True)
                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                for k, v in output.items():
                    if k not in model_outputs:
                        model_outputs[k] = []
                    model_outputs[k].append(self.accelerator.gather_for_metrics(v))
                loss = self.accelerator.gather_for_metrics(loss).cpu().numpy()
                try:
                    losses.extend(loss)
                except Exception:
                    losses.append(loss)

                if self.accelerator.is_main_process:
                    self.eval_bar.update(1)
                    self.global_eval_step += 1
                    if self.global_eval_step % self.config.logs_frequency_batches == 0:
                        logger.info(f'Global eval step {self.global_eval_step}')

                accumulated_text_embeddings = []
                accumulated_audio_embeddings = []

        return EvalOutput(epoch_id=epoch_index, losses=losses, model_outputs=self.group_predictions(model_outputs))
