#
# This contains a cirq sampler class, that uses IBMQ as a backend.
# It allows the execution of circuits using IBM's quantum stack.
#
from .qiskit_device import get_qiskit_sampler, set_debug_circuit_writer
