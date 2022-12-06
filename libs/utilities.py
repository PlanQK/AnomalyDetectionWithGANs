#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
This file provides utility functions and classes for json files.
"""
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """Class that extends the JSONEncoder to work with numpy objects
    by converting them to their standard counterparts.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


def reformat_for_json(data):
    """Transform data into a json compatible format."""
    # create a json compatible dictionary
    return json.loads(json.dumps({**data}, cls=NpEncoder))


def export_to_json(data, fp=None):
    """Writes json data to file.

    Args:
        data: The data to write
        fp (optional): The file path. If none specified, writes to model/data.json.
    """
    if not fp:
        fp = "model/data.json"
    with open(fp, "w", encoding="utf-8") as file:
        json.dump(data, file, cls=NpEncoder)
