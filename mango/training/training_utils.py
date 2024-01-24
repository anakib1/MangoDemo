from dataclasses import dataclass
from .dataset_utils import SynthConfig


@dataclass
class TrainingConfig(SynthConfig):
    base_checkpoint: str = 'openai/whisper-small'
    model_checkpoint_name: str = 'whisper-small-diarization-0.3'
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 16
    num_noise_labels: int = 18
