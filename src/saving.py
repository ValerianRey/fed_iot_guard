from time import time
import json
import os
from metrics import BinaryClassificationResults


def dumper(obj):
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__


def save_results(path, local_results, new_devices_results):
    # Save the results to a new unique file (file name based on current time)
    current_time = time()
    with open(path + 'local_results_{}.json'.format(current_time), 'w') as outfile:
        json.dump(local_results, outfile, default=dumper, indent=2)
    with open(path + 'new_devices_results_{}.json'.format(current_time), 'w') as outfile:
        json.dump(new_devices_results, outfile, default=dumper, indent=2)


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_dummy_results(path):
    dummy_results = {'a': [BinaryClassificationResults(tp=1, fp=5)]}
    with open(path + 'dummy.json', 'w') as outfile:
        json.dump(dummy_results, outfile, default=dumper)
