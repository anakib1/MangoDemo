import pathlib

import numpy as np
import torch
import accelerate
from dataclasses import dataclass
from typing import List, Any, Dict, Union
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainingOutput:
    epoch_id: int = None


@dataclass
class TrainingOutput(BaseTrainingOutput):
    losses: List[float] = None
    model_outputs: Dict[str, Any] = None


@dataclass
class EvalOutput(TrainingOutput):
    predictions: List[Dict[str, Any]] = None


@dataclass
class TrainerConfig:
    model_name: str = None
    concatenate_batches: bool = True
    use_tensorboard: bool = True
    logs_frequency_batches: int = 1
    save_strategy: str = 'end'
    push_to_hub: bool = True


class MangoTrainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 eval_loader: DataLoader, config: TrainerConfig, accelerator=None, optimizer=None):
        """
        Instantiates a MangoTrainer instance
        :param model: model to train. could any nn.Module that returns dict with 'loss' key.
        :param train_loader: loader to sample training batches
        :param eval_loader: loader to sample evaluation batches
        :param config: configuration of the trainer
        :param accelerator: accelerator instance for custom config
        :param optimizer: optimizer instance for custom gradient descend
        """
        self.eval_bar = None
        self.train_bar = None
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.project_dir = f'./output/run-{datetime.now().strftime("%y-%m-%d|%H-%M")}'
        if accelerator is None:
            accelerator = accelerate.Accelerator(log_with='tensorboard',
                                                 project_dir=self.project_dir)
        self.accelerator = accelerator
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        self.optimizer = optimizer

        self.model = accelerator.prepare_model(self.model)
        self.optimizer = accelerator.prepare_optimizer(self.optimizer)
        self.train_loader = accelerator.prepare_data_loader(self.train_loader)
        self.eval_loader = accelerator.prepare_data_loader(self.eval_loader)
        self.api = HfApi()
        self.global_train_step = 0
        self.global_eval_step = 0

    def train(self, num_epochs=1, compute_metrics=None) -> None:
        """
        Starts training of the given trainer instance.
        :param num_epochs: number of epoch to train
        :param compute_metrics: callback that will be called after each training and evaluation iteration
        :return: None
        """

        if self.accelerator.is_main_process:
            logger.info("Should create progress bars from the main process.")
            self.train_bar = tqdm(desc='train', total=num_epochs * len(self.train_loader))
            self.eval_bar = tqdm(desc='eval', total=num_epochs * len(self.eval_loader))

        if compute_metrics is None:
            compute_metrics = lambda output: {}

        if self.config.use_tensorboard:
            self.accelerator.init_trackers(self.config.model_name)

        for epoch in range(num_epochs):
            train_outputs = self.train_iteration(epoch)
            self.accelerator.wait_for_everyone()

            metrics = compute_metrics(train_outputs)
            self.log_results(metrics, train_outputs)

            val_outputs = self.eval_iteration(epoch)
            self.accelerator.wait_for_everyone()

            metrics = compute_metrics(val_outputs)
            self.log_results(metrics, val_outputs)

            if self.config.save_strategy == 'epoch':
                self.save_model()

            logger.info(f'Epoch {epoch} passed')

        if self.config.save_strategy == 'end':
            self.save_model()

        self.accelerator.end_training()

    def save_model(self):
        if not self.accelerator.is_main_process:
            return
        try:
            pathlib.Path(self.project_dir).mkdir(parents=True, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), f'{self.project_dir}/model.pt')
            repo_id = f'anakib1/{self.config.model_name}'
            if not self.api.repo_exists(repo_id):
                self.api.create_repo(repo_id, repo_type='model')
            self.api.upload_folder(
                folder_path=f'{self.config.model_name}',
                repo_id=repo_id,
                repo_type="model"
            )
        except Exception as ex:
            logger.error(f'Pushing model failed. Exception: {ex}')

    def log_results(self, results: Dict[str, float], outputs: Union[TrainingOutput, EvalOutput]) -> None:
        iteration_class = 'eval' if isinstance(outputs, EvalOutput) else 'train'
        results.update({'loss': np.mean(outputs.losses)})
        try:
            if self.config.use_tensorboard:
                for k, v in results.items():
                    self.accelerator.log({f'{k}/{iteration_class}': v}, outputs.epoch_id)
        except Exception as ex:
            logger.error(f"Logging failed. Exception: {ex}")

    def group_predictions(self, predictions: Dict[str, List[torch.Tensor]]) -> Union[
        Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Groups prediction to the tensors and not lists of tensor.
        :param predictions: dictionary of model outputs.
        :return: dictionary of grouped model predictions
        """
        if not self.config.concatenate_batches:
            return predictions
        ret = {}
        for k, v in predictions.items():
            if not isinstance(v, list) or len(v) == 0:
                logger.warning(f'Unexpected model output at key {k}. Skipping it.')
                continue
            if len(v[0].shape) == 0:
                ret[k] = torch.tensor(v)
            else:
                ret[k] = torch.concatenate(v)
        return ret

    def train_iteration(self, epoch_index: int) -> TrainingOutput:
        """
        Runs train iteration.
        :param epoch_index: id of the current epoch
        :return: The result of the iteration
        """
        self.model.train()

        train_outputs = {}
        losses = []

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(**batch)
            if 'loss' not in output:
                raise Exception("Model 'forward' function did not return 'loss' as expected. ")
            loss = output['loss']

            self.accelerator.backward(loss)

            self.optimizer.step()

            for k, v in output.items():
                if k not in train_outputs:
                    train_outputs[k] = []
                train_outputs[k].append(v.detach())
            losses.append(float(loss))

            if self.accelerator.is_main_process:
                self.train_bar.update(1)
                self.global_train_step += 1
                if self.global_train_step % self.config.logs_frequency_batches == 0:
                    logger.debug(f'Global train step {self.global_train_step}')

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

        with torch.no_grad():
            for batch in self.eval_loader:
                output = self.model(**batch)
                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                for k, v in output.items():
                    if k not in model_outputs:
                        model_outputs[k] = []
                    model_outputs[k].append(v)
                losses.append(float(loss))

                if self.accelerator.is_main_process:
                    self.eval_bar.update(1)
                    self.global_eval_step += 1
                    if self.global_eval_step % self.config.logs_frequency_batches == 0:
                        logger.debug(f'Global eval step {self.global_eval_step}')

        return EvalOutput(epoch_id=epoch_index, losses=losses, model_outputs=self.group_predictions(model_outputs))
