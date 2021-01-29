from data import get_sub_div
from context_printer import Color
from context_printer import ContextPrinter as Ctp


def set_models_sub_divs(args, models, clients_dl_train, color=Color.NONE):
    Ctp.enter_section('Computing the normalization values for each client', color)
    n_clients = len(clients_dl_train)
    for i, (model, dl_train) in enumerate(zip(models, clients_dl_train)):
        Ctp.print('[{}/{}] computing normalization with {} train samples'.format(i + 1, n_clients, len(dl_train.dataset)))
        sub, div = get_sub_div(dl_train.dataset[:][0], normalization=args.normalization)
        model.set_sub_div(sub, div)
    Ctp.exit_section()
