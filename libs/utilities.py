#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json
import numpy as np

def export_to_json(data, fp = ""):
    """Outputs data to json file """
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    if fp:
        foo = open(fp, "w")
    else:
        # set default file path as backfall case
        foo = open(f"model/data.json", "w")
    json.dump({**data}, foo, cls=NpEncoder)
    foo.close()
    return None