from typing import List, Set, Tuple

import torch
from context_printer import ContextPrinter as Ctp
import numpy as np
from copy import deepcopy


def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
        state_dict_mean = global_model.state_dict()
        for key in state_dict_mean:
            state_dict_mean[key] = torch.stack([model.state_dict()[key] for model in models], dim=-1).mean(dim=-1)
        global_model.load_state_dict(state_dict_mean)


def federated_median(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
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

    with torch.no_grad():
        state_dict_result = global_model.state_dict()
        for key in state_dict_result:
            sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
            trimmed_tensor = torch.narrow(sorted_tensor, -1, trim_num_up, n_remaining)
            state_dict_result[key] = trimmed_tensor.mean(dim=-1)
        global_model.load_state_dict(state_dict_result)


def s_resampling(models: List[torch.nn.Module], s: int) -> Tuple[List[torch.nn.Module], List[List[int]]]:
    T = len(models)
    c = [0 for _ in range(T)]
    output_models = []
    output_indexes = []
    for t in range(T):
        j = [-1 for _ in range(s)]
        for i in range(s):
            while True:
                j[i] = np.random.randint(T)
                if c[j[i]] < s:
                    c[j[i]] += 1
                    break
        output_indexes.append(j)
        with torch.no_grad():
            g_t_bar = deepcopy(models[0])
            sampled_models = [models[j[i]] for i in range(s)]
            federated_averaging(g_t_bar, sampled_models)
            output_models.append(g_t_bar)

    return output_models, output_indexes


def model_update_scaling(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], factor: float) -> None:
    with torch.no_grad():
        for model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.state_dict().items():
                param_delta = model.state_dict()[key] - original_param
                param_delta = param_delta * factor
                new_state_dict.update({key: original_param + param_delta})
            model.load_state_dict(new_state_dict)


def model_canceling_attack(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], n_honest: int) -> None:
    factor = - n_honest / len(malicious_clients_models)
    with torch.no_grad():
        for model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.state_dict().items():
                new_state_dict.update({key: original_param * factor})
            model.load_state_dict(new_state_dict)


# Attack in which all malicious clients mimic the model of a single good client. The mimicked client should always be the same throughout
# the federation rounds.
def mimic_attack(models: List[torch.nn.Module], malicious_clients: Set[int], mimicked_client: int) -> None:
    with torch.no_grad():
        for i in malicious_clients:
            models[i] = deepcopy(models[mimicked_client])
