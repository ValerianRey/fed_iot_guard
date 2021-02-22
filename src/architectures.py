from typing import List

import torch
import torch.nn as nn


def get_smooth_layers(dim_encoding: int, n_layers_encoder: int, n_layers_decoder: int, n_neurons_in: int = 115) -> List[int]:
    reduction_factor = (dim_encoding / n_neurons_in) ** (1 / n_layers_encoder)
    augmentation_factor = (n_neurons_in / dim_encoding) ** (1 / n_layers_decoder)

    hidden_layers = [int(n_neurons_in * (reduction_factor ** (i + 1))) for i in range(n_layers_encoder - 1)] + \
                    [dim_encoding] + \
                    [int(dim_encoding * (augmentation_factor ** (i + 1))) for i in range(n_layers_decoder - 1)]

    return hidden_layers


class SimpleAutoencoder(nn.Module):
    def __init__(self, activation_function: nn.Module, hidden_layers: List[int], verbose: bool = False) -> None:
        super(SimpleAutoencoder, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, n_neurons_in, bias=True))

        if verbose:
            print(self.seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Threshold(nn.Module):
    # This class is only a wrapper around the threshold that allows to use directly the federated aggregation on it.
    def __init__(self, threshold: torch.Tensor):
        super(Threshold, self).__init__()
        self.threshold = nn.Parameter(threshold, requires_grad=False)


class BinaryClassifier(nn.Module):
    def __init__(self, activation_function: nn.Module, hidden_layers: List[int], verbose: bool = False) -> None:
        super(BinaryClassifier, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, 1, bias=True))
        self.seq.add_module('sigmoid', nn.Sigmoid())

        if verbose:
            print(self.seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NormalizingModel(nn.Module):
    def __init__(self, model: torch.nn.Module, sub: torch.Tensor, div: torch.Tensor) -> None:
        super(NormalizingModel, self).__init__()
        self.sub = nn.Parameter(sub, requires_grad=False)
        self.div = nn.Parameter(div, requires_grad=False)
        self.model = model

    # Manually change normalization values
    def set_sub_div(self, sub: torch.Tensor, div: torch.Tensor) -> None:
        self.sub = nn.Parameter(sub, requires_grad=False)
        self.div = nn.Parameter(div, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.sub) / self.div
