from typing import Tuple, List

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from torch.utils.data import DataLoader


def get_sub_div(data: torch.Tensor, normalization: str) -> Tuple[torch.tensor, torch.tensor]:
    if normalization == '0-mean 1-var':
        sub = data.mean(dim=0)
        div = data.std(dim=0)
    elif normalization == 'min-max':
        sub = data.min(dim=0)[0]
        div = data.max(dim=0)[0] - sub
    elif normalization == 'none':
        sub = torch.zeros(data.shape[1])
        div = torch.ones(data.shape[1])
    else:
        raise NotImplementedError

    return sub, div


def set_model_sub_div(normalization: str, model: torch.nn.Module, train_dl: DataLoader) -> None:
    data = train_dl.dataset[:][0]
    Ctp.print('Computing normalization with {} train samples'.format(len(data)))
    sub, div = get_sub_div(data, normalization)
    model.set_sub_div(sub, div)


def set_model_add_mult(normalization: str, model: torch.nn.Module, train_dl: DataLoader) -> None:
    data = train_dl.dataset[:][0]
    Ctp.print('Computing normalization with {} train samples'.format(len(data)))
    sub, div = get_sub_div(data, normalization)
    model.set_add_mult(add=sub, mult=div)


def set_models_sub_divs(normalization: str, models: List[torch.nn.Module], clients_dl_train: List[DataLoader], color: Color = Color.NONE) -> None:
    Ctp.enter_section('Computing the normalization values for each client', color)
    for i, (model, train_dl) in enumerate(zip(models, clients_dl_train)):
        set_model_sub_div(normalization, model, train_dl)
    Ctp.exit_section()


def set_models_add_mults(normalization: str, models: List[torch.nn.Module], clients_dl_train: List[DataLoader], color: Color = Color.NONE) -> None:
    Ctp.enter_section('Computing the normalization values for each client', color)
    for i, (model, train_dl) in enumerate(zip(models, clients_dl_train)):
        set_model_add_mult(normalization, model, train_dl)
    Ctp.exit_section()
