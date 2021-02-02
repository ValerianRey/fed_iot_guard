from metrics import BinaryClassificationResults, dumper
import json
import os

results = BinaryClassificationResults(tp=10)
results.add_tp(10)
results.add_fp(9)
results.add_tn(8)
results.add_fn(7)

path = '../results/'

if not os.path.exists(path):
    os.makedirs(path)

results = {"a": [results]}

with open(path + 'test.json', 'w') as outfile:
    json.dump(results, outfile, default=dumper)
