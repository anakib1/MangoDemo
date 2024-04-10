from .base import MixerTransform, MixedExample
from torch.utils.data import Dataset


class DatasetMixer:
    def __init__(
            self,
            transforms: list[MixerTransform],
    ):
        """
        :param transforms: list of transforms to perform by each generate action
        """
        self.transforms = transforms

    def generate(self) -> MixedExample:
        """
        Generates one example. It uses all the transforms
        """
        mixed_example = MixedExample()
        for transform in self.transforms:
            mixed_example = transform(
                mixed_example,
            )
        return mixed_example


class TorchDatasetWrapper(Dataset):
    def __init__(self, mixer: DatasetMixer, num_examples: int = 1500):
        self.mixer = mixer
        self.num_examples = num_examples

    def __getitem__(self, idx):
        return self.mixer.generate()

    def __len__(self):
        return self.num_examples
