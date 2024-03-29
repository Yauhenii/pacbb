from torch import nn, Tensor
import torch.nn.functional as F

from core.layer import AbstractProbLayer


class ProbConv2d(nn.Linear, AbstractProbLayer):
    def forward(self, input: Tensor) -> Tensor:
        if self.probabilistic_mode:
            sampled_weight = self._weight_dist.sample()
            sampled_bias = self._bias_dist.sample()
        else:
            if not self.training:
                sampled_weight = self._weight_dist.mu
                sampled_bias = self._bias_dist.mu
            else:
                raise ValueError('Only training with probabilistic mode is allowed')
        return F.conv2d(input, sampled_weight, sampled_bias, self.stride, self.padding, self.dilation, self.groups)

