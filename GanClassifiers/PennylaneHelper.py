import math
import numpy as np
import pennylane as qml


class IdentityCircuitBase:
    def __init__(self, numQubits, totalNumCycles):
        self.numQubits = numQubits
        self.totalNumCycles = totalNumCycles
        bases = np.random.choice(
            [qml.RX, qml.RY, qml.RZ], int(math.ceil(totalNumCycles / 2) * numQubits)
        )
        self.bases = bases.reshape(int(math.ceil(totalNumCycles / 2)), numQubits)
        self.numVariables = numQubits * totalNumCycles
        return

    def generateCycle(self, params, cyclePos):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def generateInvCycle(self, params, cyclePos):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def startConfig(self):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )

    def buildCircuit(self, params):
        # if odd then number of cycles then add an empty layer at the front
        if self.totalNumCycles % 2:
            self.generateCycle(-1)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateCycle(params, i)

        for i in range(int(self.totalNumCycles / 2)):
            self.generateInvCycle(params, i)
        return

    def initializeQubits(self, inputs):
        for i in range(self.numQubits):
            qml.RX(inputs[i], wires=i)

    def setBases(self, bases):
        assert len(bases) == int(math.ceil(self.totalNumCycles / 2) * self.numQubits)
        self.bases = []
        for element in bases:
            if element == "X":
                basis = qml.RX
            elif element == "Y":
                basis = qml.RY
            elif element == "Z":
                basis = qml.RZ
            else:
                raise ValueError(
                    'Basis needs to be one of the three strings: "X" "Y" "Z"'
                )
            self.bases.append(basis)
        self.bases = np.array(self.bases)
        self.bases = self.bases.reshape(
            int(math.ceil(self.totalNumCycles / 2)), self.numQubits
        )

    def getBases(self):
        bases = []
        for element in self.bases.flatten():
            if element == qml.RX:
                basis = "X"
            elif element == qml.RY:
                basis = "Y"
            elif element == qml.RZ:
                basis = "Z"
            else:
                raise ValueError(
                    "The bases are only allowed to consist of the cirq Rotations: rx, ry, rz"
                )
            bases.append(basis)
        return bases

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random([int(self.totalNumCycles / 2), self.numQubits]) * 2 - 1
        ) * np.pi
        if self.totalNumCycles == 1:
            return np.zeros([1, self.numQubits]).flatten()
        # for odd number of layers
        if self.totalNumCycles % 2:
            return np.concatenate(
                (
                    np.zeros((1, self.numQubits)),
                    startingParameters,
                    np.flip(-1 * startingParameters, axis=0),
                )
            ).flatten()
        # for even number of layers
        else:
            return np.concatenate(
                (startingParameters, np.flip(-1 * startingParameters, axis=0))
            ).flatten()

    def measureZ(self):
        return [qml.expval(qml.PauliZ(qubitID)) for qubitID in range(self.numQubits)]


class LittleEntanglementIdentity(IdentityCircuitBase):
    def generateCycle(self, params, cyclePos):
        add = 0
        if self.totalNumCycles % 2:
            add = 1
        for qubitID in range(self.numQubits):
            self.bases[cyclePos + add][qubitID](
                params[(cyclePos + add) * self.numQubits + qubitID], wires=qubitID
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
        if cyclePos == int(self.totalNumCycles / 2) - 1:
            return
        flipFlop = cyclePos % 2
        print("  ")
        for qubitID in range(int(self.numQubits / 2)):
            qml.CNOT(
                wires=[
                    (flipFlop + 2 * qubitID) % self.numQubits,
                    (flipFlop + 2 * qubitID + 1) % self.numQubits,
                ]
            )
        return

    def generateInvCycle(self, params, cyclePos):
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
        if cyclePos != 0:
            flipFlop = cyclePos % 2
            for qubitID in range(int((self.numQubits) / 2)):
                pos1 = (flipFlop + 2 * qubitID) % self.numQubits
                pos2 = (flipFlop + 2 * qubitID + 1) % self.numQubits
                qml.CNOT(wires=[pos1, pos2])
        for qubitID in range(self.numQubits):
            self.bases[len(self.bases) - cyclePos - 1][qubitID](
                params[-(cyclePos * self.numQubits + qubitID)], wires=qubitID
            )
        return
