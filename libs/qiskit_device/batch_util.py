# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A module for running Cirq objects."""
import numpy as np
import cirq


class SampledExpectationBatch:
    """
    This class allows the batched execution of a set of circuits and returns
    the expectation values for the measurements.

    Usually TFQ allows noncommuting PauliStrings as measurements. This class
    currently only allows commuting single qubit measurements. With this
    restriction we were able to bundle all the observables into a single circuit
    execution.
    """

    def __init__(self, circuits, observables, repetitions):
        assert len(observables) == len(
            circuits
        ), "circuits & observables need to have the same length"

        # TODO assert observables are only single pauli measurements
        # this makes everything easier for now

        self._circuits = circuits
        self._observables = observables
        self._repetitions = int(
            max(np.array(repetitions).flatten())
        )  # different number of shots is not supported by backend
        self._all_exp_vals = np.full(
            (len(circuits), len(observables[0])), -2, dtype=np.float32
        )

    def collect(self, sampler):
        """
        Runs all circuits using the given sampler and collects the measured results.
        """
        circuit_list = []
        for i, value in enumerate(self._circuits):
            circuit_list.append(
                self.circuit_with_measurements(value, self._observables[i])
            )
        results = sampler.run_batch(circuit_list, repetitions=self._repetitions)
        for i in range(len(self._circuits)):
            for j in range(len(self._observables[i])):
                individual_result = self._convert_to_expectation(results[i], f"out_{j}")
                self._all_exp_vals[i][j] = individual_result
        return self._all_exp_vals

    @staticmethod
    def _convert_to_expectation(result, key):
        parities = result.histogram(key=key, fold_func=lambda bits: np.sum(bits) % 2)
        return (parities[0] - parities[1]) / (parities[0] + parities[1])

    @staticmethod
    def circuit_with_measurements(circuit, measurements):
        """This assumes that the measurements commute"""
        c = circuit.copy()
        c.append(
            cirq.Moment(
                [
                    cirq.measure(*sorted(measurement), key=f"out_{i}")
                    for i, measurement in enumerate(measurements)
                ]
            )
        )
        return c


def _fixed_circuit_plus_pauli_string_measurements(circuit, pauli_string):
    """A circuit measuring the given observable at the end of the given circuit."""
    assert pauli_string
    circuit = circuit.copy()
    # Uses cirq.SingleQubitCliffordGates which aren't serializable by engine in
    # cirq 0.6. This is a workaround until fixed.
    # circuit.append(cirq.Moment(pauli_string.to_z_basis_ops()))
    circuit.append(cirq.Moment(cirq.decompose(pauli_string.to_z_basis_ops())))
    circuit.append(cirq.Moment([cirq.measure(*sorted(pauli_string.keys()), key="out")]))
    return circuit


def _validate_inputs(circuits, param_resolvers, simulator, sim_type):
    """Type check and sanity check inputs."""
    if not isinstance(circuits, (list, tuple, np.ndarray)):
        raise TypeError("circuits must be a list or array." f" Given: {type(circuits)}")

    if any(not isinstance(x, cirq.Circuit) for x in circuits):
        raise TypeError("circuits must contain cirq.Circuit objects")

    if not isinstance(param_resolvers, (list, tuple, np.ndarray)):
        raise TypeError(
            "param_resolvers must be a list or array."
            f" Given: {type(param_resolvers)}"
        )

    if any(not isinstance(x, cirq.ParamResolver) for x in param_resolvers):
        raise TypeError("param_resolvers must contain cirq.ParamResolvers.")

    if not len(circuits) == len(param_resolvers):
        raise ValueError("Circuit batch size does not match resolve batch size.")

    if sim_type == "analytic":
        if not isinstance(simulator, cirq.SimulatesFinalState):
            raise TypeError(
                "For analytic operations only"
                " cirq.SimulatesFinalState"
                f" is required. Given: {type(simulator)}"
            )

    elif sim_type == "expectation":
        if not isinstance(
            simulator,
            (
                cirq.sim.simulator.SimulatesExpectationValues,
                cirq.DensityMatrixSimulator,
            ),
        ):
            # TODO(zaqqwerty): remove DM sim check once cirq #3964 is resolved.
            raise TypeError(
                "For expectation operations a "
                "cirq.sim.simulator.SimulatesExpectationValues "
                "or cirq.DensityMatrixSimulator"
                f"is required.  Given: {type(simulator)}"
            )

    elif sim_type == "sample":
        if not isinstance(simulator, cirq.Sampler):
            raise TypeError(
                "For sample based operations a cirq.Sampler is "
                f"required. Given: {type(simulator)}"
            )
    else:
        raise ValueError("Invalid simulator type specified.")


def _check_empty(circuits):
    """Returns true if circuits is the empty tensor."""
    return len(circuits) == 0


def batch_calculate_state(circuits, param_resolvers, simulator):
    """Compute states from a batch of circuits.

    Returns a NumPy array containing the final circuit state for each
    `cirq.Circuit` in `circuits`, given that the corresponding
    `cirq.ParamResolver` in `param_resolvers` was used to resolve any symbols
    in it. If simulator is a `cirq.DensityMatrixSimulator` this final state will
    be a density matrix, else this final state will be a state vector. More
    specifically, for a given `i`, `batch_calculate_state` will use
    `param_resolvers[i]` to resolve the symbols in `circuits[i]` and then place
    the final state in the return list at index `i`.

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        simulator: Simulator object.  Can be any `cirq.SimulatesFinalState`;
            if `simulator` is not a `cirq.DensityMatrixSimulator`, this function
            assumes all final states are dense state vectors.

    Returns:
        `np.ndarray` containing the resulting state information. In the case of
        `cirq.DensityMatrixSimulator` the shape is
        [len(circuits), <size of biggest state>, <size of biggest state>], else
        the shape is [len(circuits), <size of biggest state>].
    """
    _validate_inputs(circuits, param_resolvers, simulator, "analytic")
    if _check_empty(circuits):
        empty_ret = np.zeros((0, 0), dtype=np.complex64)
        if isinstance(simulator, cirq.DensityMatrixSimulator):
            empty_ret = np.zeros((0, 0, 0), dtype=np.complex64)
        return empty_ret

    biggest_circuit = max(len(circuit.all_qubits()) for circuit in circuits)

    # Default to state vector unless we see densitymatrix.
    return_mem_shape = (len(circuits), 1 << biggest_circuit)
    post_process = lambda x: x.final_state_vector
    if isinstance(simulator, cirq.DensityMatrixSimulator):
        return_mem_shape = (
            len(circuits),
            1 << biggest_circuit,
            1 << biggest_circuit,
        )
        post_process = lambda x: x.final_density_matrix

    batch_states = np.ones(return_mem_shape, dtype=np.complex64) * -2
    for index, (program, param) in enumerate(zip(circuits, param_resolvers)):
        result = simulator.simulate(program, param)
        state_size = 1 << len(program.all_qubits())
        state = post_process(result).astype(np.complex64)
        sub_index = (slice(None, state_size, 1),) * (batch_states.ndim - 1)
        batch_states[index][sub_index] = state

    return batch_states


def batch_calculate_expectation(circuits, param_resolvers, ops, simulator):
    """Compute expectations from a batch of circuits.

    Returns a `np.ndarray` containing the expectation values of `ops`
    applied to a specific circuit in `circuits`, given that the
    corresponding `cirq.ParamResolver` in `param_resolvers` was used to resolve
    any symbols in the circuit. Specifically the returned array at index `i,j`
    will be equal to the expectation value of `ops[i][j]` on `circuits[i]` with
    `param_resolvers[i]` used to resolve any symbols in `circuits[i]`.
    Expectation calculations will be carried out using the simulator object.

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        ops: 2d Python `list` of `cirq.PauliSum` objects where `ops[i][j]` will
            be used to calculate the expectation on `circuits[i]` for all `j`,
            after `param_resolver[i]` is used to resolve any parameters
            in the circuit.
        simulator: Simulator object. Must inherit
            `cirq.sim.simulator.SimulatesExpectationValues` or
            `cirq.DensityMatrixSimulator`.

    Returns:
        `np.ndarray` containing the expectation values. Shape is:
            [len(circuits), len(ops[0])]
    """
    _validate_inputs(circuits, param_resolvers, simulator, "expectation")

    if _check_empty(circuits):
        return np.zeros((0, 0), dtype=np.float32)

    if not isinstance(ops, (list, tuple, np.ndarray)):
        raise TypeError("ops must be a list or array." f" Given: {type(ops)}")

    if len(ops) != len(circuits):
        raise ValueError("Shape of ops and circuits do not match.")

    for sub_list in ops:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError("elements of ops must be type list.")
        for x in sub_list:
            if not isinstance(x, cirq.PauliSum):
                raise TypeError(
                    "ops must contain only cirq.PauliSum objects." f" Given: {type(x)}"
                )

    all_exp_vals = np.ones(shape=(len(circuits), len(ops[0])), dtype=np.float32) * -2
    for i, (c, p, op_row) in enumerate(zip(circuits, param_resolvers, ops)):
        # Convention in TFQ is to set expectations of empty circuits to -2.
        if len(c) == 0:
            continue
        # TODO(zaqqwerty): remove DM sim check once cirq #3964 is resolved.
        if isinstance(simulator, cirq.DensityMatrixSimulator):
            qubits = c.all_qubits()
            pairs = zip(sorted(qubits), list(range(len(qubits))))
            qubit_order = dict(pairs)
            sim_result = simulator.simulate(c, p)
            for j, op in enumerate(op_row):
                dm = sim_result.final_density_matrix
                all_exp_vals[i][j] = op.expectation_from_density_matrix(
                    dm, qubit_order, check_preconditions=False
                )
        else:
            # Valid observables always have real expectation values.
            all_exp_vals[i] = np.real(
                np.asarray(simulator.simulate_expectation_values(c, op_row, p))
            )

    return all_exp_vals


def batch_calculate_sampled_expectation(
    circuits, param_resolvers, ops, n_samples, sampler
):
    """Compute expectations from sampling a batch of circuits.

    Returns a `np.ndarray` containing the expectation values of `ops`
    applied to a specific circuit in `circuits`, given that the
    corresponding `cirq.ParamResolver` in `param_resolvers` was used to resolve
    any symbols in the circuit. Specifically the returned array at index `i,j`
    will be equal to the expectation value of `ops[i][j]` on `circuits[i]` with
    `param_resolvers[i]` used to resolve any symbols in `circuits[i]`.
    Expectation estimations will be carried out using the sampler object.
    Expectations for ops[i][j] are estimated by drawing n_samples[i][j]
    samples.

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        ops: 2d Python `list` of `cirq.PauliSum` objects where `ops[i][j]` will
            be used to calculate the expectation on `circuits[i]` for all `j`,
            after `param_resolver[i]` is used to resolve any parameters
            in the circuit.
        n_samples: 2d Python `list` of `int`s where `n_samples[i][j]` is
            equal to the number of samples to draw in each term of `ops[i][j]`
            when estimating the expectation.
        sampler: Anything inheriting `cirq.Sampler`.

    Returns:
        `np.ndarray` containing the expectation values. Shape is:
            [len(circuits), len(ops[0])]
    """
    _validate_inputs(circuits, param_resolvers, sampler, "sample")
    if _check_empty(circuits):
        return np.zeros((0, 0), dtype=np.float32)

    if not isinstance(ops, (list, tuple, np.ndarray)):
        raise TypeError("ops must be a list or array." f" Given: {type(ops)}")

    if len(ops) != len(circuits):
        raise ValueError("Shape of ops and circuits do not match.")

    if len(n_samples) != len(circuits):
        raise ValueError("Shape of n_samples does not match circuits.")

    for sub_list in n_samples:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError("Elements of n_elements must be lists of ints.")
        for x in sub_list:
            if not isinstance(x, int):
                raise TypeError("Non-integer value found in n_samples.")
            if x <= 0:
                raise ValueError("n_samples contains sample value <= 0.")

    for sub_list in ops:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError("elements of ops must be type list.")
        for x in sub_list:
            if not isinstance(x, cirq.PauliSum):
                raise TypeError(
                    "ops must contain only cirq.PauliSum objects." f" Given: {type(x)}"
                )

    circuit_list = []
    for (c, params) in zip(circuits, param_resolvers):
        # (#679) Just ignore empty programs.
        if len(c.all_qubits()) == 0:
            continue
        circuit_list.append(cirq.resolve_parameters(c, params))

    collector = SampledExpectationBatch(
        circuit_list, ops, n_samples
    )  # test with fixed value, what are samples_per_term?
    all_exp_vals = collector.collect(sampler)

    # return an np.array of shape (len(circuits),len(ops[0])) with values [-1,1]

    return all_exp_vals


def batch_sample(circuits, param_resolvers, n_samples, simulator):
    """Sample from circuits.

    Returns a `np.ndarray` containing n_samples samples from all the circuits in
    circuits given that the corresponding `cirq.ParamResolver` in
    `param_resolvers` was used to resolve any symbols. Specifically the
    returned array at index `i,j` will correspond to a `np.ndarray` of
    booleans representing bitstring `j` that was sampled from `circuits[i]`.
    Samples are drawn using the provided simulator object (Currently supported
    are `cirq.DensityMatrixSimulator` and `cirq.Simulator`).

    Note: In order to keep numpy shape consistent, smaller circuits will
        have sample bitstrings padded with -2 on "qubits that don't exist
        in the circuit".

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        n_samples: `int` describing number of samples to draw from each
            circuit.
        simulator: Simulator object. Currently
            supported are `cirq.DensityMatrixSimulator` and `cirq.Simulator`.

    Returns:
        `np.ndarray` containing the samples with invalid qubits blanked out.
        It's shape is
        [len(circuits), n_samples, <# qubits in largest circuit>].
        circuits that are smaller than #qubits in largest circuit have null
        qubits in bitstrings mapped to -2.
    """
    _validate_inputs(circuits, param_resolvers, simulator, "sample")
    if _check_empty(circuits):
        return np.zeros((0, 0, 0), dtype=np.int8)

    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an int." f"Given: {type(n_samples)}")

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")

    biggest_circuit = max(len(circuit.all_qubits()) for circuit in circuits)
    return_mem_shape = (len(circuits), n_samples, biggest_circuit)
    return_array = np.ones(return_mem_shape, dtype=np.int8) * -2

    for batch, (c, resolver) in enumerate(zip(circuits, param_resolvers)):
        if len(c.all_qubits()) == 0:
            continue

        qb_keys = [(q, str(i)) for i, q in enumerate(sorted(c.all_qubits()))]
        c_m = c + cirq.Circuit(cirq.measure(q, key=i) for q, i in qb_keys)
        run_c = cirq.resolve_parameters(c_m, resolver)
        bits = simulator.sample(run_c, repetitions=n_samples)
        flat_m = bits[[x[1] for x in qb_keys]].to_numpy().astype(np.int8)
        return_array[batch, :, biggest_circuit - len(qb_keys) :] = flat_m

    return return_array
