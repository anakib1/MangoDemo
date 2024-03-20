from typing import Dict, Any, Union

import numpy as np
import tensorflow as tf
import torch.nn
from pathlib import Path
from huggingface_hub import HfApi
from tensorboard.plugins.hparams import api as hp


def try_float(obj) -> Union[float, None]:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return None


def try_arr(obj) -> Union[np.array, None]:
    try:
        return np.array(obj)
    except Exception:
        return None


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
        self.metrics_writer = None

    def init_run(self, run_name: str, hparams: Dict[str, Any] = None):
        self.metrics_writer = tf.summary.create_file_writer(self.runs_prefix + '/' + run_name)
        if hparams is None:
            hparams = {}
        with self.metrics_writer.as_default():
            hp.hparams(hparams)

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        with self.metrics_writer.as_default():
            for k, v in metrics.items():
                v_float = try_float(v)
                if v_float is not None:
                    tf.summary.scalar(k, v_float, epoch_id)
                else:
                    arr = try_arr(v)
                    if arr is not None:
                        if len(arr.shape) == 4:  # image maybe
                            tf.summary.image(k, arr, step=epoch_id, max_outputs=10)
                        else:
                            tf.summary.text(k, str(arr), step=epoch_id)
                    else:
                        tf.summary.text(k, str(v), step=epoch_id)
