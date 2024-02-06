import unittest
from mango.training.MangoTrainer import TrainerConfig, MangoTrainer
import torch
from torch import nn
from typing import Dict
from dataclasses import dataclass


class MockData(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {'input_values': torch.randn(1000), 'label': torch.randn(1)}


@dataclass
class MockConfig:
    model_name: str = 'mock'


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.layers = nn.Sequential(nn.Linear(1000, 256), nn.Linear(256, 1))

    def forward(self, input_values: torch.Tensor, label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        ret = {'pred': self.layers(input_values)}
        if label is not None:
            ret.update({'loss': nn.functional.mse_loss(ret['pred'], label)})
        return ret


class TrainerTest(unittest.TestCase):
    def test_training(self):
        data = MockData()
        loader = torch.utils.data.DataLoader(data, batch_size=16)
        model = MockModel()
        trainer_config = TrainerConfig(push_to_hub=False, model_name='mock')
        trainer = MangoTrainer(model, loader, loader, trainer_config)

        trainer.train(4)

    def test_scheduler(self):
        data = MockData()
        loader = torch.utils.data.DataLoader(data, batch_size=16)
        model = MockModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=4, steps_per_epoch=len(loader))

        trainer_config = TrainerConfig(push_to_hub=False, model_name='mock', scheduler_strategy='batch')
        trainer = MangoTrainer(model, loader, loader, trainer_config, optimizer=optimizer, scheduler=scheduler)

        trainer.train(4)


if __name__ == '__main__':
    unittest.main()
