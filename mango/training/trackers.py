import os
from typing import Dict, Any, Union

import PIL.Image
import neptune
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


def try_image(obj) -> Union[np.array, None]:
    arr = try_arr(obj)
    if arr is None or len(arr.shape) != 3:
        return None
    return arr


def try_images(obj) -> Union[np.array, None]:
    arr = try_arr(obj)
    if arr is None or len(arr.shape) != 4:
        return None
    return arr


class BaseTracker:
    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        pass

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        pass

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        pass


class LocalTracker(BaseTracker):

    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        self.run_dir = run_name

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        super().log_epoch(metrics, epoch_id)

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f'{self.run_dir}/model.pt')


class HuggingfaceTracker(BaseTracker):
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


class TensorboardTracker(BaseTracker):
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
                if try_float(v) is not None:
                    tf.summary.scalar(k, try_float(v), epoch_id)
                elif try_images(v) is not None:  # image maybe
                    tf.summary.image(k, try_images(v), step=epoch_id, max_outputs=10)
                else:
                    tf.summary.text(k, str(v), step=epoch_id)


class NeptuneTracker(BaseTracker):
    def __init__(self, project_name: str, token: str, model_name: str):
        self.project_name = project_name
        self.token = token
        self.model_name = model_name
        self.run = neptune.init_run(project=project_name, api_token=token, dependencies='infer')

    def init_run(self, run_name: str, hparams: Dict[str, Any]):
        self.run['name'] = run_name
        for k, v in hparams.items():
            self.run[f'hparams/{k}'] = v

    def log_epoch(self, metrics: Dict[str, Any], epoch_id: int):
        for k, v in metrics.items():
            if try_float(v) is not None:
                self.run[f'metrics/{k}'].append(try_float(v))
            elif try_image(v) is not None:  # image maybe
                self.run[f'images/{k}'].append(PIL.Image.fromarray(try_image(v)))
            elif try_images(v) is not None:
                for i, img in enumerate(try_images(v)):
                    self.run[f'images/{k}-{i}'].append(PIL.Image.fromarray(img))
            else:
                self.run[f'misc/{k}'].append(str(v))

    def save_model(self, model: torch.nn.Module, epoch_id: int):
        model_file = f'{self.model_name}-{epoch_id}.pt'
        torch.save(model.state_dict(), model_file)
        self.run[f'model_checkpoints/{epoch_id}'].upload(model_file)
        self.run.sync()
        os.remove(model_file)
