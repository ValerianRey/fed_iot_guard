from copy import deepcopy
from time import time
from types import SimpleNamespace
from typing import Callable, Tuple, List, Dict

import numpy as np
from context_printer import ContextPrinter as Ctp, Color

from data import FederationData, ClientData, DeviceData, get_configuration_data, get_initial_splitting
from metrics import BinaryClassificationResult
from saving import create_new_numbered_dir, save_results_test

from supervised_experiments import local_classifiers_train_test, federated_classifiers_train_test
from unsupervised_experiments import local_autoencoders_train_test, federated_autoencoders_train_test


# Computes the results of multiple random reruns of the same experiment
def compute_rerun_results(clients_train_val: FederationData, clients_test: FederationData, test_devices_data: ClientData,
                          experiment: str, federated: bool, params: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    local_results = []
    new_devices_results = []
    for run_id in range(params.n_random_reruns):  # Multiple reruns: we run the same experiment multiple times to get better confidence in the results
        Ctp.enter_section('Run [{}/{}]'.format(run_id + 1, params.n_random_reruns), Color.GRAY)
        start_time = time()
        if experiment == 'classifier':
            if federated:
                malicious_clients = set(np.random.choice(len(clients_train_val), params.n_malicious, replace=False))
                params.malicious_clients = malicious_clients
                Ctp.print('Malicious clients (indexes from 1 to 8): ' + repr([mc+1 for mc in malicious_clients]))
                local_result, new_devices_result = federated_classifiers_train_test(clients_train_val, clients_test, test_devices_data, params=params)
            else:
                local_result, new_devices_result = local_classifiers_train_test(clients_train_val, clients_test, test_devices_data, params=params)
        elif experiment == 'autoencoder':
            if federated:
                local_result, new_devices_result = federated_autoencoders_train_test(clients_train_val, clients_test, test_devices_data, params=params)
            else:
                local_result, new_devices_result = local_autoencoders_train_test(clients_train_val, clients_test, test_devices_data, params=params)
        else:
            raise ValueError()

        local_results.append(local_result)
        new_devices_results.append(new_devices_result)
        Ctp.print("Elapsed time: {:.1f} seconds".format(time() - start_time))
        Ctp.exit_section()
    return local_results, new_devices_results


# This function is used to test the performance of a model with a given set of hyper-parameters on the test set
def test_hyperparameters(all_data: List[DeviceData], setup: str, experiment: str, federated: bool, splitting_function: Callable,
                         constant_params: dict, configurations_params: List[dict], configurations: List[Dict[str, list]]) -> None:
    # Create the path in which we store the results
    base_path = 'test_results/' + setup + '_' + experiment + ('_federated' if federated else '') + '/run_'
    results_path = create_new_numbered_dir(base_path)

    params_dict = deepcopy(constant_params)
    local_results, new_devices_results = {}, {}

    for j, (configuration, configuration_params) in enumerate(zip(configurations, configurations_params)):
        # Multiple configurations: we iterate over the possible configurations of the clients. Each configuration has its hyper-parameters
        clients_devices_data, test_devices_data = get_configuration_data(all_data, configuration['clients_devices'], configuration['test_devices'])
        clients_train_val, clients_test = get_initial_splitting(splitting_function, clients_devices_data,
                                                                p_test=params_dict['p_test'], p_unused=params_dict['p_unused'])
        params_dict.update(configuration)  # Update the constant hyper-parameters with the dict containing the configuration setup
        params_dict.update(configuration_params)  # Update the hyper-parameters with the configuration-specific hyper-parameters
        params = SimpleNamespace(**params_dict)
        Ctp.enter_section('Configuration [{}/{}]: '.format(j + 1, len(configurations)) + str(configuration), Color.NONE)
        local_result, new_result = compute_rerun_results(clients_train_val, clients_test, test_devices_data, experiment, federated, params)
        local_results[repr(configuration)], new_devices_results[repr(configuration)] = local_result, new_result
        Ctp.exit_section()

    # We save the results in a json file
    save_results_test(results_path, local_results, new_devices_results, constant_params, configurations_params)
