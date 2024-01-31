import pathlib

import numpy as np
import torch
import accelerate
from dataclasses import dataclass
from typing import List, Any, Dict, Union
import logging
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    report_predictions: bool = False
    concatenate_batches: bool = True
    use_tensorboard: bool = True
    push_to_hub_strategy: str = 'end'


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
        self.writer = None
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        if accelerator is None:
            accelerator = accelerate.Accelerator()
        self.accelerator = accelerator
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())
        self.optimizer = optimizer

        self.model = accelerator.prepare_model(self.model)
        self.optimizer = accelerator.prepare_optimizer(self.optimizer)
        self.train_loader = accelerator.prepare_data_loader(self.train_loader)
        self.eval_loader = accelerator.prepare_data_loader(self.eval_loader)
        self.api = HfApi()

    def train(self, num_epochs=1, compute_metrics=None) -> None:
        """
        Starts training of the given trainer instance.
        :param num_epochs: number of epoch to train
        :param compute_metrics: callback that will be called after each training and evaluation iteration
        :return: None
        """

        if compute_metrics is None:
            compute_metrics = lambda output: np.mean(output.loss)

        if self.config.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=str(pathlib.Path(self.config.model_name).joinpath('runs').joinpath(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                )))
        try:
            for epoch in tqdm(range(num_epochs)):
                train_outputs = self.train_iteration(epoch, self.config.report_predictions)
                metrics = compute_metrics(train_outputs)
                self.log_results(metrics, train_outputs)

                val_outputs = self.eval_iteration(epoch)
                metrics = compute_metrics(val_outputs)
                self.log_results(metrics, val_outputs)

                if self.config.push_to_hub_strategy == 'epoch':
                    self.push_to_hub()

            if self.config.push_to_hub_strategy == 'end':
                self.push_to_hub()
        finally:
            self.writer.close()

    def push_to_hub(self):
        torch.save(self.model.state_dict(), f'{self.config.model_name}/model.pt')
        repo_id = f'anakib1/{self.config.model_name}'
        if not self.api.repo_exists(repo_id):
            self.api.create_repo(repo_id, repo_type='model')
        self.api.upload_folder(
            folder_path=f'{self.config.model_name}',
            repo_id=repo_id,
            repo_type="model"
        )

    def log_results(self, results: Dict[str, float], outputs: Union[TrainingOutput, EvalOutput]) -> None:
        iteration_class = 'eval' if isinstance(outputs, EvalOutput) else 'train'
        results.update({'loss': np.mean(outputs.losses)})
        if self.config.use_tensorboard:
            for k, v in results.items():
                self.writer.add_scalar(f'{k}/{iteration_class}', v, outputs.epoch_id)
        else:
            raise NotImplemented('Only tensorboard is supported for now')

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

    def train_iteration(self, epoch_index: int, report_predictions=False) -> TrainingOutput:
        """
        Runs train iteration.
        :param epoch_index: id of the current epoch
        :param report_predictions: whether to include all model outputs in result
        :return: The result of the iteration
        """
        self.model.train()

        train_outputs = {} if report_predictions else None
        losses = []

        for batch in self.train_loader:
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            output = self.model(**batch)
            if 'loss' not in output:
                raise Exception("Model 'forward' function did not return 'loss' as expected. ")
            loss = output['loss']

            self.accelerator.backward(loss)

            self.optimizer.step()

            if report_predictions:
                for k, v in output.items():
                    if k not in train_outputs:
                        train_outputs[k] = []
                    train_outputs[k].append(v.detach())
            losses.append(float(loss))

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
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                output = self.model(**batch)
                if 'loss' not in output:
                    raise Exception("Model 'forward' function did not return 'loss' as expected. ")
                loss = output['loss']

                for k, v in output.items():
                    if k not in model_outputs:
                        model_outputs[k] = []
                    model_outputs[k].append(v)
                losses.append(float(loss))

        return EvalOutput(epoch_id=epoch_index, losses=losses, model_outputs=self.group_predictions(model_outputs))
