import json
import os
from typing import Optional, Any, Union, List
from types import FunctionType

from context_printer import ContextPrinter as Ctp


def dumper(obj: Any) -> Union[dict, str]:
    try:
        return obj.to_json()
    except AttributeError:
        if isinstance(obj, type) or isinstance(obj, FunctionType):
            return obj.__name__
        else:
            return obj.__dict__


def save_results_test(path: str, local_results: dict, new_devices_results: dict, thresholds: Optional[dict],
                      constant_params, configurations_params: List[dict]) -> None:
    # Save the results to a new unique file (file name based on current time)
    with open(path + 'local_results.json', 'w') as outfile:
        json.dump(local_results, outfile, default=dumper, indent=2)
    with open(path + 'new_devices_results.json', 'w') as outfile:
        json.dump(new_devices_results, outfile, default=dumper, indent=2)
    with open(path + 'constant_params.json', 'w') as outfile:
        json.dump(constant_params, outfile, default=dumper, indent=2)
    with open(path + 'configurations_params.json', 'w') as outfile:
        json.dump(configurations_params, outfile, default=dumper, indent=2)

    if thresholds is not None:
        with open(path + 'thresholds.json', 'w') as outfile:
            json.dump(thresholds, outfile, default=dumper, indent=2)


def save_results_gs(path: str, local_results: dict, constant_params: dict) -> None:
    # Save the results to a new unique file (file name based on current time)
    with open(path + 'local_results.json', 'w') as outfile:
        json.dump(local_results, outfile, default=dumper, indent=2)
    with open(path + 'constant_params.json', 'w') as outfile:
        json.dump(constant_params, outfile, default=dumper, indent=2)


def create_new_numbered_dir(base_path: str) -> Optional[str]:
    for run_id in range(1000):
        path = base_path + repr(run_id) + '/'
        if not os.path.exists(path):
            Ctp.print('Creating folder ' + path)
            os.makedirs(path)
            return path
    return None
