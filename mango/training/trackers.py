from typing import Dict, Any
import tensorflow as tf
import torch.nn
from pathlib import Path
from huggingface_hub import HfApi


class BaseTrackerCallback:
    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        pass

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        pass

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        pass


class LocalCallback(BaseTrackerCallback):

    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        self.run_dir = run_name

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        super().log_epoch(metrics, epoch_id)

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f'{self.run_dir}/model.pt')


class HuggingfaceCallback(BaseTrackerCallback):
    def __init__(self, project_dir: str, hf_user: str, hf_model_name: str):
        self.hf_user = hf_user
        self.hf_model_name = hf_model_name
        self.project_dir = project_dir
        self.api = HfApi()

    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        super().init_run(run_name, hparams)

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        super().log_epoch(metrics, epoch_id)

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        repo_id = f'{self.hf_user}/{self.hf_model_name}'
        if not self.api.repo_exists(repo_id):
            self.api.create_repo(repo_id, repo_type='model')
        self.api.upload_folder(
            folder_path=f'{self.project_dir}',
            repo_id=repo_id,
            repo_type="model"
        )


class TensorboardTrackerCallback(BaseTrackerCallback):
    def __init__(self, runs_prefix: str = 'tb_logs'):
        self.runs_prefix = runs_prefix
        self.writer = None

    def init_run(self, run_name: str, hparams: Dict[str]):
        self.writer = tf.summary.create_file_writer(self.runs_prefix + '/' + run_name)

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        with self.writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, epoch_id)
