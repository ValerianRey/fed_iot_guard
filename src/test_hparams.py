from copy import deepcopy
from types import SimpleNamespace
from typing import Callable, Tuple, List, Dict

from context_printer import ContextPrinter as Ctp, Color

from data import FederationData, ClientData, DeviceData, get_configuration_data, get_initial_splitting
from metrics import BinaryClassificationResult
from saving import create_new_numbered_dir, save_results_test


# Computes the results of multiple random reruns of the same experiment
def compute_rerun_results(clients_train_val: FederationData, clients_test: FederationData, test_devices_data: ClientData,
                          experiment_function: Callable, params: SimpleNamespace, n_random_reruns: int) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    local_results = []
    new_devices_results = []
    for run_id in range(n_random_reruns):  # Multiple reruns: we run the same experiment multiple times to get better confidence in the results
        Ctp.enter_section('Run [{}/{}]'.format(run_id + 1, n_random_reruns), Color.GRAY)
        local_result, new_devices_result = experiment_function(clients_train_val, clients_test, test_devices_data, params=params)
        local_results.append(local_result)
        new_devices_results.append(new_devices_result)
        Ctp.exit_section()
    return local_results, new_devices_results


# This function is used to test the performance of a model with a given set of hyper-parameters on the test set
def test_hyperparameters(all_data: List[DeviceData], experiment: str, experiment_function: Callable, splitting_function: Callable,
                         configurations_params: List[dict], configurations: List[Dict[str, list]],
                         p_test: float, p_unused: float, n_random_reruns: int = 5) -> None:
    # Create the path in which we store the results
    base_path = 'test_results/' + experiment + '/run_'
    results_path = create_new_numbered_dir(base_path)

    params_dicts = deepcopy(configurations_params)
    local_results, new_devices_results = {}, {}

    for j, (configuration, configuration_params) in enumerate(zip(configurations, params_dicts)):
        # Multiple configurations: we iterate over the possible configurations of the clients. Each configuration has its hyper-parameters
        clients_devices_data, test_devices_data = get_configuration_data(all_data, configuration['clients_devices'], configuration['test_devices'])
        clients_train_val, clients_test = get_initial_splitting(splitting_function, clients_devices_data, p_test=p_test, p_unused=p_unused)
        configuration_params.update(configuration)  # Update the config-specific hyper-parameters with the dict containing the configuration setup
        params = SimpleNamespace(**configuration_params)
        Ctp.enter_section('Configuration [{}/{}]: '.format(j + 1, len(configurations)) + str(configuration), Color.NONE)
        local_results[repr(configuration)], new_devices_results[repr(configuration)] = compute_rerun_results(clients_train_val, clients_test,
                                                                                                             test_devices_data, experiment_function,
                                                                                                             params, n_random_reruns)
        Ctp.exit_section()

    # We save the results in a json file
    save_results_test(results_path, local_results, new_devices_results, configurations_params)
