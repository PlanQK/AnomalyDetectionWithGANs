#!/usr/bin/env python3

from qiskit import IBMQ


IBMQ.save_account("mysecrettoken",overwrite=True)
