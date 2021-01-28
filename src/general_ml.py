from data import get_sub_div
from print_util import ContextPrinter, Color


def set_models_sub_divs(args, models, clients_dl_train, ctp: ContextPrinter, color=Color.NONE):
    ctp.print('Computing the normalization values for each client', color=color, bold=True)
    ctp.add_bar(color)
    n_clients = len(clients_dl_train)
    for i, (model, dl_train) in enumerate(zip(models, clients_dl_train)):
        ctp.print('[{}/{}] computing normalization with {} train samples'.format(i + 1, n_clients, len(dl_train.dataset)))
        sub, div = get_sub_div(dl_train.dataset[:][0], normalization=args.normalization)
        model.set_sub_div(sub, div)
    ctp.remove_header()