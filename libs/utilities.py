#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def reformat_for_json(data):
    """Outputs data to json file """
    # create a json compatible dictionary
    return json.loads(json.dumps({**data}, cls=NpEncoder))


def export_to_json(data, fp=None):
    if not fp:
        fp = "model/data.json"
    with open(fp, "w", encoding='utf-8') as file:
        json.dump(data, file, cls=NpEncoder)
