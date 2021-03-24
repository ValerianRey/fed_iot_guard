import itertools
from copy import deepcopy
from time import time
from types import SimpleNamespace
from typing import List, Dict, Callable, Union

from context_printer import ContextPrinter as Ctp, Color

from data import ClientData, split_client_data_current_fold, split_client_data, DeviceData, device_names, get_client_data
from metrics import BinaryClassificationResult
from saving import create_new_numbered_dir, save_results_gs
from supervised_experiments import local_classifier_train_val
from unsupervised_experiments import local_autoencoder_train_val


# Returns the list of unique clients as a set of tuples. Each tuple represents a client, and each tuple's element represents a device.
def get_all_clients_devices(configurations: List[Dict[str, list]]) -> List[tuple]:
    all_clients_devices_dict = {}  # We use a dict to keep the insertion order. The values of the dict are dummy values, only the keys are important
    # We read the configurations in reverse order so that the non collaborative results end up properly ordered
    for configuration in reversed(configurations):
        clients_devices = configuration['clients_devices']
        for client_devices in clients_devices:
            all_clients_devices_dict.update({tuple(client_devices): 0})

    return list(all_clients_devices_dict)


# Compute the result of the experiment summed over the splits of the cross validation
def compute_cv_result(train_val_data: ClientData, experiment: str, params: SimpleNamespace,
                      n_splits: int) -> Union[BinaryClassificationResult, float]:
    result = BinaryClassificationResult() if experiment == 'classifier' else 0.
    for fold in range(n_splits):
        Ctp.enter_section('Fold [{}/{}]'.format(fold + 1, n_splits), Color.GRAY)
        train_data, val_data = split_client_data_current_fold(train_val_data, n_splits, fold)
        if experiment == 'classifier':
            result += local_classifier_train_val(train_data, val_data, params=params)
        elif experiment == 'autoencoder':
            result += local_autoencoder_train_val(train_data, val_data, params=params)
        else:
            raise ValueError()
        Ctp.exit_section()

    return result


# Compute the result of the experiment on a specified proportion of validation data
def compute_single_split_result(train_val_data: ClientData, experiment: str, params: SimpleNamespace,
                                val_part: float) -> Union[BinaryClassificationResult, float]:
    train_data, val_data = split_client_data(train_val_data, p_second_split=val_part, p_unused=0.0)
    if experiment == 'classifier':
        result = local_classifier_train_val(train_data, val_data, params=params)
    elif experiment == 'autoencoder':
        result = local_autoencoder_train_val(train_data, val_data, params=params)
    else:
        raise ValueError()

    return result


def run_grid_search(all_data: List[DeviceData], setup: str, experiment: str,
                    splitting_function: Callable, constant_params: dict, varying_params: dict, configurations: List[Dict[str, list]],
                    collaborative: bool = False) -> None:
    # Create the path in which we store the results
    base_path = 'grid_search_results/' + setup + '_' + experiment + '/run_'

    # Compute the different sets of hyper-parameters to test in the grid search
    params_product = list(itertools.product(*varying_params.values()))

    params_dict = deepcopy(constant_params)

    if params_dict['n_splits'] == 1 and params_dict['val_part'] is None:
        raise ValueError('val_part should be specified when not using cross-validation')

    # First we compute the list of unique clients in the configurations, and we compute the grid search results for each client.
    # This way we do not make extra computations if the same client appears in several configurations
    all_clients_devices = get_all_clients_devices(configurations)
    Ctp.print(all_clients_devices)
    clients_results = {}
    for i, client_devices_tuple in enumerate(all_clients_devices):
        client_devices = list(client_devices_tuple)
        Ctp.enter_section('Client {} with devices: '.format(i) + device_names(client_devices), Color.WHITE)
        client_data = get_client_data(all_data, client_devices)
        train_val_data, _ = splitting_function(client_data, p_test=params_dict['p_test'], p_unused=params_dict['p_unused'])
        clients_results[repr(client_devices)] = {}

        for j, experiment_params_tuple in enumerate(params_product):  # Grid search: we iterate over the sets of parameters to be tested
            start_time = time()
            experiment_params = {key: arg for (key, arg) in zip(varying_params.keys(), experiment_params_tuple)}
            params_dict.update(experiment_params)
            Ctp.enter_section('Experiment [{}/{}] with params: '.format(j + 1, len(params_product)) + str(experiment_params), Color.NONE)
            params = SimpleNamespace(**params_dict)
            if params_dict['n_splits'] == 1:  # We do not use cross-validation
                result = compute_single_split_result(train_val_data, experiment, params, params_dict['val_part'])
            else:  # Cross validation: we sum the results over the folds
                result = compute_cv_result(train_val_data, experiment, params, params_dict['n_splits'])
            clients_results[repr(client_devices)][repr(experiment_params)] = result
            Ctp.print("Elapsed time: {:.1f} seconds".format(time() - start_time))
            Ctp.exit_section()
        Ctp.exit_section()

    if collaborative:
        # Now that we have the results for each client we can recombine them into the original configurations by summing the results
        configurations_results = {}
        for i, configuration in enumerate(configurations):
            configurations_results[repr(configuration['clients_devices'])] = {}
            for j, experiment_params_tuple in enumerate(params_product):
                experiment_params = {key: arg for (key, arg) in zip(varying_params.keys(), experiment_params_tuple)}
                configurations_results[repr(configuration['clients_devices'])][repr(experiment_params)] = BinaryClassificationResult() \
                    if experiment == 'classifier' else 0.
                for client_devices in configuration['clients_devices']:  # We sum the results of each client in the configuration
                    result = clients_results[repr(client_devices)][repr(experiment_params)]
                    configurations_results[repr(configuration['clients_devices'])][repr(experiment_params)] += result

        # We save the results in a json file
        results_path = create_new_numbered_dir(base_path)
        save_results_gs(results_path, configurations_results, constant_params)

    else:
        # We save the results in a json file
        results_path = create_new_numbered_dir(base_path)
        save_results_gs(results_path, clients_results, constant_params)
