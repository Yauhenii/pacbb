from abc import ABC, abstractmethod
from typing import Dict
import torch

from core.pipeline import AbstractPipeline
from core.model import AbstractModel
from core.dataset import DatasetHandler


class AbstractTrainingPipeline(AbstractPipeline, ABC):
    def __init__(self):
        self._dataset_handler = None
        self._prior_model = None
        self._posterior_model = None

    @property
    def dataset_handler(self) -> DatasetHandler:
        return self._dataset_handler

    @property
    def prior_model(self) -> AbstractModel:
        return self._prior_model

    @property
    def posterior_model(self) -> AbstractModel:
        return self._posterior_model

    @abstractmethod
    def train(
        self, model_config: Dict, dataset_config: Dict, split_strategy_config: Dict, device: torch.device
    ) -> None:
        pass
