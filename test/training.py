import logging
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
        s = torch.randn(1000)
        return {'input_values': s, 'label': torch.tensor(1.0) if torch.sum(s) > 0 else torch.tensor(0.0)}


@dataclass
class MockConfig:
    model_name: str = 'mock'


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.layers = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(), nn.Linear(512, 512),  nn.ReLU(), nn.Linear(512, 1))

    def forward(self, input_values: torch.Tensor, label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        ret = {'pred': self.layers(input_values)}
        if label is not None:
            ret.update({'loss': nn.functional.binary_cross_entropy_with_logits(ret['pred'].flatten(), label)})
        return ret


class TrainerTest(unittest.TestCase):
    def test_training(self):
        data = MockData()
        loader = torch.utils.data.DataLoader(data, batch_size=16)
        model = MockModel()
        trainer_config = TrainerConfig(push_to_hub=False, model_name='mock')
        trainer = MangoTrainer(model, loader, loader, trainer_config)

        trainer.train(100)

        test = data[0]
        pred = 1 if model(test['input_values'])['pred'] > 0 else 0
        self.assertEqual(pred, test['label'])

    def test_scheduler(self):
        data = MockData()
        loader = torch.utils.data.DataLoader(data, batch_size=16)
        model = MockModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, eta_min=1e-4)

        trainer_config = TrainerConfig(push_to_hub=False, model_name='mock', scheduler_strategy='batch')
        trainer = MangoTrainer(model, loader, loader, trainer_config, optimizer=optimizer, scheduler=scheduler)

        trainer.train(16)

    def test_early_stopping(self):
        logging.getLogger('mango').setLevel(logging.INFO)
        data = MockData()
        loader = torch.utils.data.DataLoader(data, batch_size=16)
        model = MockModel()

        trainer_config = TrainerConfig(push_to_hub=False, model_name='mock', scheduler_strategy='batch',
                                       early_stopping_patience=2)
        trainer = MangoTrainer(model, loader, loader, trainer_config)

        trainer.train(100)

        self.assertNotEquals(trainer.epoch, 99)


if __name__ == '__main__':
    unittest.main()
