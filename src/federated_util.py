from typing import List

import torch


def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    state_dict_mean = global_model.state_dict()

    for key in state_dict_mean:
        state_dict_mean[key] = torch.stack([model.state_dict()[key] for model in models], dim=-1).mean(dim=-1)

    global_model.load_state_dict(state_dict_mean)
