#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json

from .program import run
from libs.qiskit_device import set_debug_circuit_writer

# make sure the file is in the input folder
# use train.json for training data
# use test.json for testing data
input_file = "test.json"
with open(f"./input/{input_file}") as file:
    inp = json.load(file)

# set_debug_circuit_writer(True)

data = inp["data"]
params = inp["params"]

response = run(data, params)

print()
print(response.to_json())
