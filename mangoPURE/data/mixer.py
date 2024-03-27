from .base import MixerTransform, MixedExample


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
