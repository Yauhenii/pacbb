from typing import Tuple
from torch.utils import data
from torchvision import datasets, transforms

from core.dataset.loader import AbstractLoader


class MNISTLoader(AbstractLoader):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def load(self) -> Tuple[data.Dataset, data.Dataset]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(
            self._dataset_path, train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            self._dataset_path, train=False, download=True, transform=transform
        )
        return train, test
