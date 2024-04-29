import torch

from .MangoTrainer import MangoTrainer, TrainingOutput, TrainerConfig, EvalOutput
from typing import Dict

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

    def loss(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Dict:
        logits = text_embeddings @ torch.transpose(audio_embeddings, 0, 1)
        N = logits.shape[0]
        ans = {"logits": logits}
        target = torch.arange(0, N, dtype=torch.long).to(logits.device)
        loss = self.loss_fn(logits, target) + self.loss_fn(torch.transpose(logits, 0, 1), target)
        ans["loss"] = loss
        return ans


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
                accumulated_text_embeddings.append(self.model.audio_encoder(**batch['audio_model_input']))
                accumulated_audio_embeddings.append(self.model.text_encoder(**batch['text_model_input']))

            if (i + 1) % self.config.num_repeats > 0:
                continue

            self.optimizer.zero_grad()
            for j in range(self.config.num_repeats):
                gradient_text_embeddings = self.model.text_encoder(**accumulated_text_inputs[j])
                gradient_audio_embeddings = self.model.audio_encoder(**accumulated_audio_inputs[j])

                output = self.model.loss(torch.concatenate(
                    accumulated_audio_embeddings[:j] + [gradient_audio_embeddings] + accumulated_audio_embeddings[
                                                                                     j + 1:], dim=0),
                    torch.concatenate(accumulated_text_embeddings[:j] + [
                        gradient_text_embeddings] + accumulated_text_embeddings[j + 1:], dim=0))

                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                self.accelerator.backward(loss)
                if self.config.grad_clip:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            with torch.no_grad():
                output = self.model.loss(torch.concatenate(accumulated_audio_embeddings, dim=0),
                                         torch.concatenate(accumulated_text_embeddings, dim=0))

            for k, v in output.items():
                if k not in train_outputs:
                    train_outputs[k] = []
                train_outputs[k].append(self.accelerator.gather_for_metrics(v))
            loss = output['loss']
            losses.extend(self.accelerator.gather_for_metrics(torch.as_tensor(loss).sum()).cpu().numpy())

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
                accumulated_text_embeddings.append(self.model.audio_encoder(**batch['audio_model_input']))
                accumulated_audio_embeddings.append(self.model.text_encoder(**batch['text_model_input']))

                if (i + 1) % self.config.num_repeats > 0:
                    continue

                output = self.model.loss(torch.concatenate(accumulated_audio_embeddings, dim=0),
                                         torch.concatenate(accumulated_text_embeddings, dim=0))
                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                for k, v in output.items():
                    if k not in model_outputs:
                        model_outputs[k] = []
                    model_outputs[k].append(self.accelerator.gather_for_metrics(v))
                losses.append(self.accelerator.gather_for_metrics(loss).cpu().numpy())

                if self.accelerator.is_main_process:
                    self.eval_bar.update(1)
                    self.global_eval_step += 1
                    if self.global_eval_step % self.config.logs_frequency_batches == 0:
                        logger.info(f'Global eval step {self.global_eval_step}')

                accumulated_text_embeddings = []
                accumulated_audio_embeddings = []

        return EvalOutput(epoch_id=epoch_index, losses=losses, model_outputs=self.group_predictions(model_outputs))
