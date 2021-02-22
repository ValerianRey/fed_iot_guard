from typing import List

import torch


def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    state_dict_mean = global_model.state_dict()

    for key in state_dict_mean:
        state_dict_mean[key] = torch.stack([model.state_dict()[key] for model in models], dim=-1).mean(dim=-1)

    global_model.load_state_dict(state_dict_mean)


def federated_median(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    state_dict_median = global_model.state_dict()

    for key in state_dict_median:
        state_dict_median[key], _ = torch.stack([model.state_dict()[key] for model in models], dim=-1).median(dim=-1)

    global_model.load_state_dict(state_dict_median)


# Shortcut for __federated_trimmed_mean(global_model, models, 1) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_1(global_model: torch.nn.Module, models: List[torch.nn.Module]):
    __federated_trimmed_mean(global_model, models, 1)


# Shortcut for __federated_trimmed_mean(global_model, models, 2) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_2(global_model: torch.nn.Module, models: List[torch.nn.Module]):
    __federated_trimmed_mean(global_model, models, 2)


def __federated_trimmed_mean(global_model: torch.nn.Module, models: List[torch.nn.Module], trim_num_up: int) -> None:
    n = len(models)
    n_remaining = n - 2 * trim_num_up
    state_dict_result = global_model.state_dict()

    for key in state_dict_result:
        sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
        trimmed_tensor = torch.narrow(sorted_tensor, -1, trim_num_up, n_remaining)

        state_dict_result[key] = trimmed_tensor.mean(dim=-1)

    global_model.load_state_dict(state_dict_result)
