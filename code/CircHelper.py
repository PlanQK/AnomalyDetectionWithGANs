import numpy as np
import math
import sympy
import cirq


class StatePrepCircuit:
    def __init__(self):
        self.randomBases = np.random.choice([cirq.rx, cirq.ry], 500)

    def createSimpleStatePrepCircuit(self, parameters):
        """Generate a simple state where the parameters corespond to the
    coefficient of the |1> state for each qubit individually.
    
    Parameters need to be in [-1,1].
    """
        # transform that the mapping is linear
        parameters = np.arcsin(paremeters)
        circuit = cirq.Circuit()
        qubits = cirq.GridQubit(1, len(parameters))
        for i in range(len(parameters)):
            circuit.append(cirq.rx(parameters[i]).on(qubit[i]))
        return circuit

    def createEntangledStatePrepCircuit(self, parameters, steps):
        """Generate a circuit that uses several steps of rotations and entangling layers.

    The number of parameters need to be divisible by the number of steps.
    """
        assert len(parameters) % steps == 0
        circuit = cirq.Circuit()
        qubits = cirq.GridQubit.rect(1, int(len(parameters) / steps))
        for step in range(steps):
            for qubitID in range(int(len(parameters) / steps)):
                # rotations
                paramId = qubitID + int(len(parameters) / steps) * step
                circuit.append(
                    self.randomBases[paramId](parameters[paramId]).on(qubits[qubitID])
                )
                # entangling
                if qubitID % 2 == 0:
                    circuit.append(
                        cirq.CZ(qubits[qubitID], qubits[(qubitID + 1) % len(qubits)])
                    )
            for qubitID in range(int(len(parameters) / steps)):
                if qubitID % 2 == 1:
                    circuit.append(
                        cirq.CZ(qubits[qubitID], qubits[(qubitID + 1) % len(qubits)])
                    )
        return circuit


# @title
class ParametrizedCircuitBase:
    """Base class for the different Parametrized circuits. Derive this class to
  write your own specialized circuit.
  """

    def __init__(self, x=3, y=3, totalNumCycles=3):
        self.qubits = cirq.GridQubit.rect(x, y)
        self.circuit = cirq.Circuit()
        self.totalNumCycles = totalNumCycles
        self.controlParams = []

    def generateCycle(self):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def startConfig(self):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def getReadOut(self):
        return [cirq.Z(q) for q in self.qubits]

    def buildCircuit(self):
        self.circuit = cirq.Circuit()
        for i in range(self.totalNumCycles):
            self.generateCycle()
        return self.circuit

    def _startConfigZero(self):
        return

    def _startConfigSuperposition(self):
        for qubit in self.qubits:
            self.circuit.append(cirq.H(qubit))
        return


class SimpleParametrizedCircuit(ParametrizedCircuitBase):
    """
  Creates a circuit with a single layer of parametrized X rotations.
  This is to test if the optimization will work as intended.
  """

    def __init__(self):
        super().__init__()
        self.totalNumCycles = 1

    def startConfig(self):
        self._startConfigZero()

    def generateCycle(self):
        for qubit in self.qubits:
            symb = sympy.Symbol(f"w{len(self.controlParams)}")
            self.circuit.append(cirq.rx(symb).on(qubit))
            self.controlParams.append(symb)


class RandomParametrizedCircuit(ParametrizedCircuitBase):
    """Almost random circuit.
  
  Following: https://arxiv.org/abs/1903.05076
  The circuit starts of as identity: U(theta1)*U^dagger(theta2). We try to
  circumvent the barren plateau problem by initializing the weights theta1=theta2.
  This will start with non vanishing gradients but does not guarantee that
  we avoid barren plateaus.
  """

    def __init__(self, x=3, y=3, totalNumCycles=3):
        super().__init__(x=x, y=y, totalNumCycles=totalNumCycles)
        self.inputParams = np.array([sympy.Symbol(f"input{i}") for i in range(x * y)])
        numVariables = x * y * totalNumCycles
        # needs to be reversed so we generate the bases and parameters beforehand
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(numVariables)]
        )
        self.controlParams = self.controlParams.reshape(totalNumCycles, x * y)
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz], int(math.ceil(totalNumCycles / 2) * x * y)
        )
        self.bases = self.bases.reshape(int(math.ceil(totalNumCycles / 2)), x * y)
        self.circuit = cirq.Circuit()

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random([int(self.totalNumCycles / 2), len(self.qubits)]) * 2 - 1
        ) * np.pi
        if self.totalNumCycles == 1:
            return np.zeros([1, len(self.qubits)]).flatten()
        if self.totalNumCycles % 2:
            return np.concatenate(
                (
                    np.zeros((1, len(self.qubits))),
                    startingParameters,
                    np.flip(-1 * startingParameters, axis=0),
                )
            ).flatten()
        else:
            return np.concatenate(
                (startingParameters, np.flip(-1 * startingParameters, axis=0))
            ).flatten()

    def purelyRandomInitParameters(self):
        return (
            (np.random.random([self.totalNumCycles, len(self.qubits)]) * 2 - 1) * np.pi
        ).flatten()

    def buildCircuit(self):
        # state input
        for i in range(len(self.inputParams)):
            self.circuit.append(cirq.rx(self.inputParams[i] * np.pi).on(self.qubits[i]))

        # if odd then number of cycles then add an empty layer at the front
        if self.totalNumCycles % 2:
            self.generateCycle(-1)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateCycle(i)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateInvCycle(i)
        return self.circuit

    def generateInvCycle(self, cyclePos):
        # layer of entangling gates
        if cyclePos != 0:
            for qubitID in range(len(self.qubits)):
                if qubitID % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )
            for qubitID in range(len(self.qubits)):
                if qubitID % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )

        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[len(self.bases) - cyclePos - 1][qubitID](
                    self.controlParams[self.totalNumCycles - cyclePos - 1][qubitID]
                ).on(self.qubits[qubitID])
            )

    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )  # layer of entangling gates
        if cyclePos == int(self.totalNumCycles / 2) - 1:
            return
        for qubitID in range(len(self.qubits)):
            # TODO: moment
            if qubitID % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        self.qubits[qubitID],
                        self.qubits[(qubitID + 1) % len(self.qubits)],
                    )
                )
        for qubitID in range(len(self.qubits)):
            if qubitID % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        self.qubits[qubitID],
                        self.qubits[(qubitID + 1) % len(self.qubits)],
                    )
                )


class completeRotationCircuit(RandomParametrizedCircuit):
    def __init__(self, x=3, y=3, totalNumCycles=3):
        super().__init__(x=x, y=y, totalNumCycles=totalNumCycles)
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(3 * x * y * totalNumCycles)]
        )
        self.controlParams = self.controlParams.reshape(3 * totalNumCycles, x * y)

    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                [
                    cirq.rx(self.controlParams[3 * (cyclePos + add)][qubitID]).on(
                        self.qubits[qubitID]
                    ),
                    cirq.ry(self.controlParams[3 * (cyclePos + add) + 1][qubitID]).on(
                        self.qubits[qubitID]
                    ),
                    cirq.rz(self.controlParams[3 * (cyclePos + add) + 2][qubitID]).on(
                        self.qubits[qubitID]
                    ),
                ]
            )
        # layer of entangling gates
        if cyclePos == int(self.totalNumCycles / 2) - 1:
            return
        for qubitID in range(len(self.qubits)):
            # TODO: moment
            if qubitID % 2 == 0:
                self.circuit.append(
                    cirq.CZ(
                        self.qubits[qubitID],
                        self.qubits[(qubitID + 1) % len(self.qubits)],
                    )
                )
        for qubitID in range(len(self.qubits)):
            if qubitID % 2 == 1:
                self.circuit.append(
                    cirq.CZ(
                        self.qubits[qubitID],
                        self.qubits[(qubitID + 1) % len(self.qubits)],
                    )
                )

    def generateInvCycle(self, cyclePos):
        # layer of entangling gates
        if cyclePos != 0:
            for qubitID in range(len(self.qubits)):
                if qubitID % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )
            for qubitID in range(len(self.qubits)):
                if qubitID % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )

        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                [
                    cirq.rz(
                        self.controlParams[3 * (self.totalNumCycles - cyclePos) - 1][
                            qubitID
                        ]
                    ).on(self.qubits[qubitID]),
                    cirq.ry(
                        self.controlParams[3 * (self.totalNumCycles - cyclePos) - 2][
                            qubitID
                        ]
                    ).on(self.qubits[qubitID]),
                    cirq.rx(
                        self.controlParams[3 * (self.totalNumCycles - cyclePos) - 3][
                            qubitID
                        ]
                    ).on(self.qubits[qubitID]),
                ]
            )

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random([3 * int(self.totalNumCycles / 2), len(self.qubits)]) * 2
            - 1
        ) * np.pi
        if self.totalNumCycles == 1:
            return np.zeros([3, len(self.qubits)]).flatten()
        if self.totalNumCycles % 2:
            return np.concatenate(
                (
                    np.zeros([3, len(self.qubits)]),
                    startingParameters,
                    np.flip(-1 * startingParameters, axis=0),
                )
            ).flatten()
        else:
            return np.concatenate(
                (startingParameters, np.flip(-1 * startingParameters, axis=0))
            ).flatten()

    def purelyRandomInitParameters(self):
        return (
            (np.random.random([self.totalNumCycles, 3 * len(self.qubits)]) * 2 - 1)
            * np.pi
        ).flatten()


class littleEntanglement(RandomParametrizedCircuit):
    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )  # layer of entangling gates
        if cyclePos == int(self.totalNumCycles / 2) - 1:
            return
        if cyclePos % 2:
            for qubitID in range(len(self.qubits)):
                # TODO: moment
                if qubitID % 2 == 0:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )
        else:
            for qubitID in range(len(self.qubits)):
                if qubitID % 2 == 1:
                    self.circuit.append(
                        cirq.CZ(
                            self.qubits[qubitID],
                            self.qubits[(qubitID + 1) % len(self.qubits)],
                        )
                    )

    def generateInvCycle(self, cyclePos):
        # layer of entangling gates
        if cyclePos != 0:
            if cyclePos % 2:
                for qubitID in range(len(self.qubits)):
                    if qubitID % 2 == 0:
                        self.circuit.append(
                            cirq.CZ(
                                self.qubits[qubitID],
                                self.qubits[(qubitID + 1) % len(self.qubits)],
                            )
                        )
            else:
                for qubitID in range(len(self.qubits)):
                    if qubitID % 2 == 1:
                        self.circuit.append(
                            cirq.CZ(
                                self.qubits[qubitID],
                                self.qubits[(qubitID + 1) % len(self.qubits)],
                            )
                        )

        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[len(self.bases) - cyclePos - 1][qubitID](
                    self.controlParams[self.totalNumCycles - cyclePos - 1][qubitID]
                ).on(self.qubits[qubitID])
            )


class semiClassical(RandomParametrizedCircuit):
    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )  # layer of entangling gates

    def generateInvCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[len(self.bases) - cyclePos - 1][qubitID](
                    self.controlParams[self.totalNumCycles - cyclePos - 1][qubitID]
                ).on(self.qubits[qubitID])
            )
