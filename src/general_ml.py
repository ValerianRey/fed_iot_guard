from types import SimpleNamespace
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


def set_models_sub_divs(args: SimpleNamespace, models: List[torch.nn.Module], clients_dl_train: List[DataLoader], color: Color = Color.NONE) -> None:
    Ctp.enter_section('Computing the normalization values for each client', color)
    n_clients = len(clients_dl_train)
    for i, (model, dl_train) in enumerate(zip(models, clients_dl_train)):
        data = dl_train.dataset[:][0]
        Ctp.print('[{}/{}] computing normalization with {} train samples'.format(i + 1, n_clients, len(data)))
        sub, div = get_sub_div(data, normalization=args.normalization)
        model.set_sub_div(sub, div)
    Ctp.exit_section()
