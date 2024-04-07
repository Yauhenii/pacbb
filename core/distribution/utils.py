import math
from typing import List, Union, Type, Callable, Dict

import torch
from torch import nn, Tensor
from torch.nn import Parameter

from core.layer.utils import get_torch_layers
from core.distribution import AbstractVariable


def from_flat(model: nn.Module,
              attribute_mapping: dict[str, str],
              rho: Union[Tensor, List[float]],
              distribution: Type[AbstractVariable],
              get_layers_func: Callable[[nn.Module], List[nn.Module]] = get_torch_layers,
              ) -> dict[int, dict[str, AbstractVariable]]:

    distributions = {}
    shift = 0
    for i, layer in enumerate(get_layers_func(model)):
        weight_cutoff = shift + layer.out_features * layer.in_features
        bias_cutoff = weight_cutoff + layer.out_features
        weight_distribution = distribution(mu=layer.__getattr__(attribute_mapping['weight_mu']),
                                           rho=rho[shift: weight_cutoff].reshape(layer.out_features, layer.in_features))
        bias_distribution = distribution(mu=layer.__getattr__(attribute_mapping['bias_mu']),
                                         rho=rho[weight_cutoff: bias_cutoff])
        distributions[i] = {'weight': weight_distribution, 'bias': bias_distribution}
        shift = bias_cutoff
    return distributions


def _from_any(model: nn.Module,
              distribution: Type[AbstractVariable],
              requires_grad: bool,
              get_layers_func: Callable[[nn.Module], List[nn.Module]],
              weight_mu_fill_func: Callable[[nn.Module], Tensor],
              weight_rho_fill_func: Callable[[nn.Module], Tensor],
              bias_mu_fill_func: Callable[[nn.Module], Tensor],
              bias_rho_fill_func: Callable[[nn.Module], Tensor],
            ) -> Dict[int, dict[str, AbstractVariable]]:
    distributions = {}
    for i, layer in enumerate(get_layers_func(model)):
        weight_distribution = distribution(mu=weight_mu_fill_func(layer),
                                           rho=weight_rho_fill_func(layer),
                                           mu_requires_grad=requires_grad,
                                           rho_requires_grad=requires_grad)
        if layer.bias is not None:
            bias_distribution = distribution(mu=bias_mu_fill_func(layer),
                                             rho=bias_rho_fill_func(layer),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
        else:
            bias_distribution = None
        distributions[i] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def from_random(model: nn.Module,
                rho: Tensor,
                distribution: Type[AbstractVariable],
                requires_grad: bool = True,
                get_layers_func: Callable[[nn.Module], List[nn.Module]] = get_torch_layers,
                ) -> Dict[int, dict[str, AbstractVariable]]:
    def get_truncated_normal_fill_tensor(layer: nn.Module) -> Tensor:
        t = torch.Tensor(*layer.weight.shape)
        if hasattr(layer, 'in_features'):
            in_features = layer.in_features
        elif hasattr(layer, 'in_channels') and hasattr(layer, 'kernel_size'):
            in_features = layer.in_channels * math.prod(layer.kernel_size)
        else:
            raise ValueError(f'Unsupported layer of type: {type(layer)}')
        w = 1 / math.sqrt(in_features)
        return truncated_normal_fill_tensor(t, 0, w, -2 * w, 2 * w)
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_mu_fill_func=get_truncated_normal_fill_tensor,
                     weight_rho_fill_func=lambda layer: torch.ones(*layer.weight.shape) * rho,
                     bias_mu_fill_func=lambda layer: torch.zeros(*layer.bias.shape),
                     bias_rho_fill_func=lambda layer: torch.ones(*layer.bias.shape) * rho)


def from_zeros(model: nn.Module,
               rho: Tensor,
               distribution: Type[AbstractVariable],
               requires_grad: bool = True,
               get_layers_func: Callable[[nn.Module], List[nn.Module]] = get_torch_layers,
               ) -> Dict[int, dict[str, AbstractVariable]]:
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_mu_fill_func=lambda layer: torch.zeros(*layer.weight.shape),
                     weight_rho_fill_func=lambda layer: torch.ones(*layer.weight.shape) * rho,
                     bias_mu_fill_func=lambda layer: torch.zeros(*layer.bias.shape),
                     bias_rho_fill_func=lambda layer: torch.ones(*layer.bias.shape) * rho)


def from_layered(model: torch.nn.Module,
                 attribute_mapping: dict[str, str],
                 distribution: Type[AbstractVariable],
                 requires_grad: bool = True,
                 get_layers_func: Callable[[nn.Module], List[nn.Module]] = get_torch_layers,
                 ) -> Dict[int, dict[str, AbstractVariable]]:
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_mu_fill_func=lambda layer: layer.__getattr__(attribute_mapping['weight_mu']).detach().clone(),
                     weight_rho_fill_func=lambda layer: layer.__getattr__(attribute_mapping['weight_rho']).detach().clone(),
                     bias_mu_fill_func=lambda layer: layer.__getattr__(attribute_mapping['bias_mu']).detach().clone(),
                     bias_rho_fill_func=lambda layer: layer.__getattr__(attribute_mapping['bias_rho']).detach().clone())


def from_copy(dist: Dict[int, dict[str, AbstractVariable]],
              distribution: Type[AbstractVariable],
              requires_grad: bool = True) -> Dict[int, dict[str, AbstractVariable]]:
    distributions = {}
    for i, layer in dist.items():
        weight_distribution = distribution(mu=layer['weight'].mu.detach().clone(),
                                           rho=layer['weight'].rho.detach().clone(),
                                           mu_requires_grad=requires_grad,
                                           rho_requires_grad=requires_grad)
        if layer['bias'] is not None:
            bias_distribution = distribution(mu=layer['bias'].mu.detach().clone(),
                                             rho=layer['bias'].rho.detach().clone(),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
        else:
            bias_distribution = None
        distributions[i] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def compute_kl(dist1: Dict[int, Dict[str, AbstractVariable]], dist2: Dict[int, Dict[str, AbstractVariable]]) -> Tensor:
    kl_list = []
    for idx in dist1:
        for key in dist1[idx]:
            if dist1[idx][key] is not None and dist2[idx][key] is not None:
                kl = dist1[idx][key].compute_kl(dist2[idx][key])
                kl_list.append(kl)
    return torch.stack(kl_list).sum()


def get_params(dist: Dict[int, Dict[str, AbstractVariable]]) -> List[Parameter]:
    params = []
    for i, layer in dist.items():
        for key, value in layer.items():
            params.append(value.mu)
            params.append(value.rho)
    return params


def compute_standard_normal_cdf(x: float) -> float:
    """
    Compute the standard normal cumulative distribution function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The cumulative distribution function value at x.
    """
    # TODO: replace with numpy
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def truncated_normal_fill_tensor(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    # TODO: refactor
    with torch.no_grad():
        # Get upper and lower cdf values
        l = compute_standard_normal_cdf((a - mean) / std)
        u = compute_standard_normal_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
