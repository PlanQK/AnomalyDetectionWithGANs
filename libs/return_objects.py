#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
This file provides objects meant as return values.
"""
import json


class Response:
    def to_json(self):
        return json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True)


class ErrorResponse(Response):
    """
    Represents an error to be passed back to the caller.

    Args:
        code (str): application-specific code representing the type of problem encountered
        detail (str): application-specific error message describing the detail of the problem encountered
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail


class ResultResponse(Response):
    """
    Represents the result to be passed back to the caller.

    Args:
        result (dict): application-specific result to be passed back to the caller; must be JSON serializable
    """

    def __init__(self, result: dict, metadata: dict = None):
        self.result = result
        self.metadata = metadata