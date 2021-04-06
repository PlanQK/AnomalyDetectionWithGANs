import numpy as np
import math
import sympy
import cirq


class IdentityCircuitBase:
    """Base class for the different Parametrized circuits. Derive this class to
  write your own specialized circuit.
  """

    def __init__(self, x=3, y=3, totalNumCycles=3):
        self.x = x
        self.y = y
        self.qubits = cirq.GridQubit.rect(x, y)
        self.circuit = cirq.Circuit()
        self.totalNumCycles = totalNumCycles
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz],
            int(math.ceil(totalNumCycles / 2) * x * y),
        )
        self.bases = self.bases.reshape(
            int(math.ceil(totalNumCycles / 2)), x * y
        )

        self.inputParams = np.array(
            [sympy.Symbol(f"input{i}") for i in range(x * y)]
        )
        numVariables = x * y * totalNumCycles

        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(numVariables)]
        )
        self.controlParams = self.controlParams.reshape(totalNumCycles, x * y)
        self.circuit = cirq.Circuit()

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

    def setBases(self, bases):
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
            int(math.ceil(self.totalNumCycles / 2)), self.x * self.y
        )

    def getBases(self):
        bases = []
        for element in self.bases.flatten():
            if element == cirq.rx:
                basis = "X"
            elif element == cirq.ry:
                basis = "Y"
            elif element == cirq.rz:
                basis = "Z"
            else:
                raise ValueError(
                    "The bases are only allowed to consist of the "
                    "cirq Rotations: rx, ry, rz"
                )
            bases.append(basis)
        return bases

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random([int(self.totalNumCycles / 2), len(self.qubits)])
            * 2
            - 1
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

    def buildCircuit(self):
        # state input
        for i in range(len(self.inputParams)):
            self.circuit.append(
                cirq.rx(self.inputParams[i] * np.pi).on(self.qubits[i])
            )

        # if odd then number of cycles then add an empty layer at the front
        if self.totalNumCycles % 2:
            self.generateCycle(-1)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateCycle(i)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateInvCycle(i)
        return self.circuit


class RandomCircuitBase(IdentityCircuitBase):
    def __init__(self, x=3, y=3, totalNumCycles=3):
        super().__init__(x=x, y=y, totalNumCycles=totalNumCycles)

        self.controlParams = self.controlParams.reshape(totalNumCycles, x * y)
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz], totalNumCycles * x * y
        )
        self.bases = self.bases.reshape(totalNumCycles, x * y)

    def buildCircuit(self):
        # state input
        for i in range(len(self.inputParams)):
            self.circuit.append(
                cirq.rx(self.inputParams[i] * np.pi).on(self.qubits[i])
            )

        for i in range(int(self.totalNumCycles)):
            self.generateCycle(i)
        return self.circuit

    def generateInitialParameters(self):
        return (
            (np.random.random([self.totalNumCycles, len(self.qubits)]) * 2 - 1)
            * np.pi
        ).flatten()


class CompleteRotationCircuitIdentity(IdentityCircuitBase):
    def __init__(self, x=3, y=3, totalNumCycles=3):
        super().__init__(x=x, y=y, totalNumCycles=totalNumCycles)
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(3 * x * y * totalNumCycles)]
        )
        self.controlParams = self.controlParams.reshape(
            3 * totalNumCycles, x * y
        )

    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                [
                    cirq.rx(
                        self.controlParams[3 * (cyclePos + add)][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.ry(
                        self.controlParams[3 * (cyclePos + add) + 1][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.rz(
                        self.controlParams[3 * (cyclePos + add) + 2][qubitID]
                    ).on(self.qubits[qubitID]),
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
                        self.controlParams[
                            3 * (self.totalNumCycles - cyclePos) - 1
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.ry(
                        self.controlParams[
                            3 * (self.totalNumCycles - cyclePos) - 2
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.rx(
                        self.controlParams[
                            3 * (self.totalNumCycles - cyclePos) - 3
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                ]
            )

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random(
                [3 * int(self.totalNumCycles / 2), len(self.qubits)]
            )
            * 2
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


class CompleteRotationCircuitRandom(RandomCircuitBase):
    def __init__(self, x=3, y=3, totalNumCycles=3):
        super().__init__(x=x, y=y, totalNumCycles=totalNumCycles)
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(3 * x * y * totalNumCycles)]
        )
        self.controlParams = self.controlParams.reshape(
            3 * totalNumCycles, x * y
        )

    def generateCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                [
                    cirq.rx(self.controlParams[3 * (cyclePos)][qubitID]).on(
                        self.qubits[qubitID]
                    ),
                    cirq.ry(
                        self.controlParams[3 * (cyclePos) + 1][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.rz(
                        self.controlParams[3 * (cyclePos) + 2][qubitID]
                    ).on(self.qubits[qubitID]),
                ]
            )
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

    def generateInitialParameters(self):
        return (
            (
                np.random.random([self.totalNumCycles, 3 * len(self.qubits)])
                * 2
                - 1
            )
            * np.pi
        ).flatten()


class StrongEntanglementIdentity(IdentityCircuitBase):
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
                    self.controlParams[self.totalNumCycles - cyclePos - 1][
                        qubitID
                    ]
                ).on(self.qubits[qubitID])
            )


class StrongEntanglementRandom(RandomCircuitBase):
    def generateCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos][qubitID](
                    self.controlParams[cyclePos][qubitID]
                ).on(self.qubits[qubitID])
            )  # layer of entangling gates
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


class LittleEntanglementIdentity(IdentityCircuitBase):
    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
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
                    self.controlParams[self.totalNumCycles - cyclePos - 1][
                        qubitID
                    ]
                ).on(self.qubits[qubitID])
            )


class LittleEntanglementRandom(RandomCircuitBase):
    def generateCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos][qubitID](
                    self.controlParams[cyclePos][qubitID]
                ).on(self.qubits[qubitID])
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
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


class SemiClassicalIdentity(IdentityCircuitBase):
    def generateCycle(self, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )

    def generateInvCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[len(self.bases) - cyclePos - 1][qubitID](
                    self.controlParams[self.totalNumCycles - cyclePos - 1][
                        qubitID
                    ]
                ).on(self.qubits[qubitID])
            )


class SemiClassicalRandom(RandomCircuitBase):
    def generateCycle(self, cyclePos):
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos][qubitID](
                    self.controlParams[cyclePos][qubitID]
                ).on(self.qubits[qubitID])
            )
