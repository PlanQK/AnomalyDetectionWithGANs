"""
This file contains classes providing the quantum circuits used for the GANomaly model.
"""
import math
import numpy as np
import sympy
import cirq


class RandomCircuitBase:
    """Base class for different Parametrized circuits based on a random initialization."""

    def __init__(self, qubits, total_num_cycles=3):
        self.qubits = qubits
        self.circuit = cirq.Circuit()
        self.total_num_cycles = int(total_num_cycles)
        self.num_qubits = len(self.qubits)

        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz],
            int(total_num_cycles) * int(self.num_qubits),
        )
        self.bases = self.bases.reshape(int(total_num_cycles), int(self.num_qubits))
        num_variables = int(self.num_qubits) * int(total_num_cycles)
        self.control_params = np.array(
            [sympy.Symbol(f"w{i}") for i in range(num_variables)]
        )

        self.control_params = self.control_params.reshape(
            int(total_num_cycles), int(self.num_qubits)
        )

    def build_circuit(self):
        """Constructs the circuit by generating all its cycles."""
        for i in range(int(self.total_num_cycles)):
            self.generate_cycle(i)
        return self.circuit

    def generate_initial_parameters(self):
        return (
            (np.random.random([int(self.total_num_cycles), len(self.qubits)]) * 2 - 1)
            * np.pi
        ).flatten()

    def get_readout(self):
        """Returns a list containing a Pauli Z-gates on every qubit."""
        return [cirq.Z(q) for q in self.qubits]

    def generate_cycle(self, cycle_pos):
        """Generates the cycle of the circuit with the specified position."""
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )


class IdentityCircuitBase:
    """Base class for the different Parametrized circuits using identity blocks (arXiv:1903.05076).
    Derive this class to write your own specialized circuit.
    """

    def __init__(self, qubits, total_num_cycles=3):
        self.qubits = qubits
        self.circuit = cirq.Circuit()
        self.total_num_cycles = int(total_num_cycles)
        self.num_qubits = len(self.qubits)
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz],
            int(math.ceil(int(total_num_cycles) / 2) * int(self.num_qubits)),
        )
        self.bases = self.bases.reshape(
            int(math.ceil(int(total_num_cycles) / 2)), int(self.num_qubits)
        )

        num_variables = int(self.num_qubits) * int(total_num_cycles)

        self.control_params = np.array(
            [sympy.Symbol(f"w{i}") for i in range(num_variables)]
        )
        self.control_params = self.control_params.reshape(
            int(total_num_cycles), int(self.num_qubits)
        )

    def generate_cycle(self, cycle_pos):
        """Generates the cycle of the circuit with the specified position."""
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def generate_inv_cycle(self, cycle_pos):
        """Generates the inverse cycle for the specified position."""
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def get_readout(self):
        """Returns a list containing a Pauli Z-gates on every qubit."""
        return [cirq.Z(q) for q in self.qubits]

    def set_bases(self, bases):
        """Sets the bases of the circuit from a provided string."""
        assert len(bases) == len(self.bases.flatten())
        self.bases = []
        for element in bases:
            if element == "X":
                basis = cirq.rx
            elif element == "Y":
                basis = cirq.ry
            elif element == "Z":
                basis = cirq.rz
            else:
                raise ValueError(
                    'Basis needs to be one of the three strings: "X" "Y" "Z"'
                )
            self.bases.append(basis)
        self.bases = np.array(self.bases)
        self.bases = self.bases.reshape(
            int(math.ceil(int(self.total_num_cycles) / 2)), int(self.num_qubits)
        )

    def get_bases(self):
        """Returns the bases of the circuit encoded in a string."""
        bases = []
        for element in self.bases.flatten():
            if element is cirq.rx:
                basis = "X"
            elif element is cirq.ry:
                basis = "Y"
            elif element is cirq.rz:
                basis = "Z"
            else:
                raise ValueError(
                    "The bases are only allowed to consist of the "
                    "cirq Rotations: rx, ry, rz"
                )
            bases.append(basis)
        return bases

    def generate_initial_parameters(self):
        starting_parameters = (
            np.random.random([int(int(self.total_num_cycles) / 2), len(self.qubits)])
            * 2
            - 1
        ) * np.pi
        if int(self.total_num_cycles) == 1:
            return np.zeros([1, len(self.qubits)]).flatten()
        if int(self.total_num_cycles) % 2:
            return np.concatenate(
                (
                    np.zeros((1, len(self.qubits))),
                    starting_parameters,
                    np.flip(-1 * starting_parameters, axis=0),
                )
            ).flatten()
        else:
            return np.concatenate(
                (starting_parameters, np.flip(-1 * starting_parameters, axis=0))
            ).flatten()

    def build_circuit(self):
        """Generate the circuit by generating its cycles and inverse cycles."""
        # if odd then number of cycles then add an empty layer at the front
        if int(self.total_num_cycles) % 2:
            self.generate_cycle(-1)

        for i in range(int(int(self.total_num_cycles) / 2)):
            self.generate_cycle(i)

        for i in range(int(int(self.total_num_cycles) / 2)):
            self.generate_inv_cycle(i)
        return self.circuit


class CompleteRotationCircuitIdentity(IdentityCircuitBase):
    """Derived class of `IdentityCircuitBase` involving rotations along all three axes for each set of entanglement gates."""

    def __init__(self, qubits, totalNumCycles=3):
        super().__init__(qubits, total_num_cycles=totalNumCycles)
        self.control_params = np.array(
            [
                sympy.Symbol(f"w{i}")
                for i in range(3 * int(self.num_qubits) * int(totalNumCycles))
            ]
        )
        self.control_params = self.control_params.reshape(
            3 * int(totalNumCycles), int(self.num_qubits)
        )

    def generate_cycle(self, cycle_pos):
        add = 0
        if int(self.total_num_cycles) % 2:
            add = 1
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                [
                    cirq.rx(self.control_params[3 * (cycle_pos + add)][qubit_id]).on(
                        qubit
                    ),
                    cirq.ry(
                        self.control_params[3 * (cycle_pos + add) + 1][qubit_id]
                    ).on(qubit),
                    cirq.rz(
                        self.control_params[3 * (cycle_pos + add) + 2][qubit_id]
                    ).on(qubit),
                ]
            )
        # layer of entangling gates
        if cycle_pos == int(int(self.total_num_cycles) / 2) - 1:
            return
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )

    def generate_inv_cycle(self, cycle_pos):
        # layer of entangling gates
        if cycle_pos != 0:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )

        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                [
                    cirq.rz(
                        self.control_params[
                            3 * (int(self.total_num_cycles) - cycle_pos) - 1
                        ][qubit_id]
                    ).on(qubit),
                    cirq.ry(
                        self.control_params[
                            3 * (int(self.total_num_cycles) - cycle_pos) - 2
                        ][qubit_id]
                    ).on(qubit),
                    cirq.rx(
                        self.control_params[
                            3 * (int(self.total_num_cycles) - cycle_pos) - 3
                        ][qubit_id]
                    ).on(qubit),
                ]
            )

    def generate_initial_parameters(self):
        starting_parameters = (
            np.random.random(
                [3 * int(int(self.total_num_cycles) / 2), len(self.qubits)]
            )
            * 2
            - 1
        ) * np.pi
        if int(self.total_num_cycles) == 1:
            return np.zeros([3, len(self.qubits)]).flatten()
        if int(self.total_num_cycles) % 2:
            return np.concatenate(
                (
                    np.zeros([3, len(self.qubits)]),
                    starting_parameters,
                    np.flip(-1 * starting_parameters, axis=0),
                )
            ).flatten()
        else:
            return np.concatenate(
                (starting_parameters, np.flip(-1 * starting_parameters, axis=0))
            ).flatten()


class CompleteRotationCircuitRandom(RandomCircuitBase):
    """Derived class of `RandomCircuitBase` involving rotations along all three axes for each set of entanglement gates."""

    def __init__(self, qubits, totalNumCycles=3):
        super().__init__(qubits, total_num_cycles=totalNumCycles)
        self.control_params = np.array(
            [
                sympy.Symbol(f"w{i}")
                for i in range(3 * int(self.num_qubits) * int(totalNumCycles))
            ]
        )
        self.control_params = self.control_params.reshape(
            3 * int(totalNumCycles), int(self.num_qubits)
        )

    def generate_cycle(self, cycle_pos):
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                [
                    cirq.rx(self.control_params[3 * (cycle_pos)][qubit_id]).on(qubit),
                    cirq.ry(self.control_params[3 * (cycle_pos) + 1][qubit_id]).on(
                        qubit
                    ),
                    cirq.rz(self.control_params[3 * (cycle_pos) + 2][qubit_id]).on(
                        qubit
                    ),
                ]
            )
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )

    def generate_initial_parameters(self):
        return (
            (
                np.random.random([int(self.total_num_cycles), 3 * len(self.qubits)]) * 2
                - 1
            )
            * np.pi
        ).flatten()


class StrongEntanglementIdentity(IdentityCircuitBase):
    """Derived class of `IdentityCircuitBase` involving rotations along a specified axis
    before each layer of entanglement gates."""

    def generate_cycle(self, cycle_pos):
        add = 0
        if int(self.total_num_cycles) % 2:
            add = 1
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos + add][qubit_id](
                    self.control_params[cycle_pos + add][qubit_id]
                ).on(qubit)
            )  # layer of entangling gates
        if cycle_pos == int(int(self.total_num_cycles) / 2) - 1:
            return
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )

    def generate_inv_cycle(self, cycle_pos):
        # layer of entangling gates
        if cycle_pos != 0:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )

        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[len(self.bases) - cycle_pos - 1][qubit_id](
                    self.control_params[int(self.total_num_cycles) - cycle_pos - 1][
                        qubit_id
                    ]
                ).on(qubit)
            )


class StrongEntanglementRandom(RandomCircuitBase):
    """Derived class of `RandomCircuitBase` involving rotations along a specified axis
    before each layer of entanglement gates."""

    def generate_cycle(self, cycle_pos):
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos][qubit_id](
                    self.control_params[cycle_pos][qubit_id]
                ).on(qubit)
            )  # layer of entangling gates
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )
        for qubit_id, qubit in enumerate(self.qubits):
            if qubit_id % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        qubit,
                        self.qubits[(qubit_id + 1) % len(self.qubits)],
                    )
                )


class LittleEntanglementIdentity(IdentityCircuitBase):
    """Derived class of `IdentityCircuitBase` involving rotations along a specified axis
    with the layers of entanglement gates split in half."""

    def generate_cycle(self, cycle_pos):
        add = 0
        if int(self.total_num_cycles) % 2:
            add = 1
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos + add][qubit_id](
                    self.control_params[cycle_pos + add][qubit_id]
                ).on(qubit)
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
        if cycle_pos == int(int(self.total_num_cycles) / 2) - 1:
            return
        if cycle_pos % 2:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )
        else:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )

    def generate_inv_cycle(self, cycle_pos):
        # layer of entangling gates
        if cycle_pos != 0:
            if cycle_pos % 2:
                for qubit_id, qubit in enumerate(self.qubits):
                    if qubit_id % 2 == 0:
                        self.circuit.append(
                            cirq.CZ(
                                qubit,
                                self.qubits[(qubit_id + 1) % len(self.qubits)],
                            )
                        )
            else:
                for qubit_id, qubit in enumerate(self.qubits):
                    if qubit_id % 2 == 1:
                        self.circuit.append(
                            cirq.CZ(
                                qubit,
                                self.qubits[(qubit_id + 1) % len(self.qubits)],
                            )
                        )

        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[len(self.bases) - cycle_pos - 1][qubit_id](
                    self.control_params[int(self.total_num_cycles) - cycle_pos - 1][
                        qubit_id
                    ]
                ).on(qubit)
            )


class LittleEntanglementRandom(RandomCircuitBase):
    """Derived class of `RandomCircuitBase` involving rotations along a specified axis
    with the layers of entanglement gates split in half."""

    def generate_cycle(self, cycle_pos):
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos][qubit_id](
                    self.control_params[cycle_pos][qubit_id]
                ).on(qubit)
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
        if cycle_pos % 2:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )
        else:
            for qubit_id, qubit in enumerate(self.qubits):
                if qubit_id % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            qubit,
                            self.qubits[(qubit_id + 1) % len(self.qubits)],
                        )
                    )


class SemiClassicalIdentity(IdentityCircuitBase):
    """Derived class of `IdentityCircuitBase` involving rotations along a specified axis
    without entanglement layers."""

    def generate_cycle(self, cycle_pos):
        add = 0
        if int(self.total_num_cycles) % 2:
            add = 1
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos + add][qubit_id](
                    self.control_params[cycle_pos + add][qubit_id]
                ).on(qubit)
            )

    def generate_inv_cycle(self, cycle_pos):
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[len(self.bases) - cycle_pos - 1][qubit_id](
                    self.control_params[int(self.total_num_cycles) - cycle_pos - 1][
                        qubit_id
                    ]
                ).on(qubit)
            )


class SemiClassicalRandom(RandomCircuitBase):
    """Derived class of `RandomCircuitBase` involving rotations along a specified axis
    without entanglement layers."""

    def generate_cycle(self, cycle_pos):
        for qubit_id, qubit in enumerate(self.qubits):
            self.circuit.append(
                self.bases[cycle_pos][qubit_id](
                    self.control_params[cycle_pos][qubit_id]
                ).on(qubit)
            )
