import pathlib

import numpy as np
import torch
import accelerate
from dataclasses import dataclass, asdict
from typing import List, Any, Dict, Union, Literal
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from .trackers import BaseTracker

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
    logs_frequency_batches: int = 1
    save_strategy: str = 'end'
    push_to_hub: bool = True
    mixed_precision: Union[None, Literal['fp16']] = None
    grad_clip: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-3
    scheduler_strategy: str = 'batch'
    early_stopping_patience: int = None
    hf_user: str = 'anakib1'
    gradient_accumulation_steps: int = 1


class MangoTrainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 eval_loader: DataLoader, config: TrainerConfig, accelerator=None, optimizer=None, scheduler=None,
                 trackers: List[BaseTracker] | None = None):
        """
        Instantiates a MangoTrainer instance
        :param model: model to train. could any nn.Module that returns dict with 'loss' key.
        :param train_loader: loader to sample training batches
        :param eval_loader: loader to sample evaluation batches
        :param config: configuration of the trainer
        :param accelerator: accelerator instance for custom config
        :param optimizer: optimizer instance for custom gradient descend
        """
        self.run_dir = None
        self.epoch = None
        self.eval_bar = None
        self.train_bar = None
        self.config = config
        self.model = model
        self.hparams = {}
        if "config" in dir(self.model):
            self.hparams.update(asdict(self.model.config))
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.project_dir = pathlib.Path('output').joinpath(self.config.model_name)
        if trackers is None:
            trackers = []
        self.trackers = trackers
        if accelerator is None:
            plugin = accelerate.utils.GradientAccumulationPlugin(num_steps=self.config.gradient_accumulation_steps,
                                                                 sync_with_dataloader=False)
            accelerator = accelerate.Accelerator(log_with='tensorboard',
                                                 project_dir=self.project_dir,
                                                 mixed_precision=self.config.mixed_precision,
                                                 gradient_accumulation_plugin=plugin)
        self.accelerator = accelerator
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, last_epoch=-1)
        self.scheduler = scheduler

        self.model, self.optimizer, self.scheduler, self.train_loader, self.eval_loader = accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader)
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

        if self.accelerator.is_main_process:
            run_name = f'runs/run-{datetime.now().strftime("%y-%m-%d.%H-%M")}'
            hps = {"num_steps": num_epochs * len(self.train_loader),
                   "learning_rate": self.config.lr,
                   "weight_decay": self.config.weight_decay}
            hps.update(self.hparams)
            for logger_callback in self.trackers:
                logger_callback.init_run(run_name, hps)
            self.run_dir = self.project_dir.joinpath(run_name)

        train_losses = []

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_outputs = self.train_iteration(epoch)

            train_losses.append(np.mean(train_outputs.losses))
            if self.config.early_stopping_patience is not None:
                if len(train_losses) > self.config.early_stopping_patience and np.min(
                        train_losses[:-self.config.early_stopping_patience]) < \
                        np.min(train_losses[-self.config.early_stopping_patience:]):
                    logger.info(f"Stopping training at epoch {epoch}. Early stopping criterion reached")
                    self.accelerator.set_trigger()

            self.accelerator.wait_for_everyone()
            if self.config.scheduler_strategy == 'epoch':
                self.scheduler.step()

            metrics = compute_metrics(train_outputs)
            self.log_results(metrics, train_outputs)

            val_outputs = self.eval_iteration(epoch)
            self.accelerator.wait_for_everyone()

            metrics = compute_metrics(val_outputs)
            self.log_results(metrics, val_outputs)

            if self.config.save_strategy == 'epoch':
                self.save_model(epoch)

            logger.info(f'Epoch {epoch} passed')

            if self.accelerator.check_trigger():
                break

        if self.config.save_strategy == 'end':
            self.save_model(num_epochs - 1)

        self.accelerator.end_training()

    def save_model(self, epoch_id: int):
        if not self.accelerator.is_main_process:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for tracker in self.trackers:
            try:
                tracker.save_model(unwrapped_model, epoch_id)
            except Exception as ex:
                logger.error(f'Tracker {tracker} failed to save model. Exception: {ex}')

    def log_results(self, results: Dict[str, float], outputs: Union[TrainingOutput, EvalOutput]) -> None:
        iteration_class = 'eval' if isinstance(outputs, EvalOutput) else 'train'
        results.update({'loss': np.mean(outputs.losses)})
        results.update({'lr': self.scheduler.get_last_lr()[0]})
        results = {f'{k}/{iteration_class}': v for k, v in results.items()}

        for tracker in self.trackers:
            try:
                tracker.log_epoch(results, outputs.epoch_id)
            except Exception as ex:
                logger.error(f"Logging failed. Exception: {ex}")

    def group_predictions(self, predictions: Dict[str, List[torch.Tensor]]) -> \
            Union[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
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

        logger.debug(f'Starting epoch {epoch_index}')
        for i, batch in enumerate(self.train_loader):
            logger.debug(f'Starting batch {i}')
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                output = self.model(**batch)
                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                self.accelerator.backward(loss)
                if self.config.grad_clip:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.config.scheduler_strategy == 'batch':
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        self.scheduler.step(self.epoch + i / len(self.train_loader))
                    else:
                        self.scheduler.step()

            for k, v in output.items():
                if k not in train_outputs:
                    train_outputs[k] = []
                train_outputs[k].append(v.detach())
            losses.append(float(loss))
            logger.debug(f'Losses aggregated successfully.')

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
