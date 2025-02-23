import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class McAllesterObjective(AbstractObjective):
    """
    McAllester bound objective (based on McAllester, 1999 and related works),
    combining empirical loss with a square-root term involving KL and delta,
    plus additional constants (e.g., 5/2 ln(n)) in the bounding expression.
    """

    def __init__(self, kl_penalty: float, delta: float):
        """
        Args:
            kl_penalty (float): Multiplier for the KL divergence term.
            delta (float): Confidence parameter in the PAC-Bayes bound.
        """
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Compute the McAllester objective.

        Derived from a PAC-Bayes bound that includes terms like ln(num_samples)
        and -ln(delta).

        Args:
            loss (Tensor): Empirical loss or risk.
            kl (Tensor): KL divergence.
            num_samples (float): Number of samples/training size.

        Returns:
            Tensor: A scalar objective = loss + sqrt( [kl_penalty*KL + 5/2 ln(n) - ln(delta) + 8 ] / [2n - 1] ).
        """
        kl = kl * self._kl_penalty
        kl_ratio = torch.div(
            kl + 5 / 2 * np.log(num_samples) - np.log(self._delta) + 8,
            2 * num_samples - 1,
        )
        return loss + torch.sqrt(kl_ratio)
