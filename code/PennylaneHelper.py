import numpy as np
import pennylane as qml


class QmlParametrizedCircuit:
    def __init__(self, numQubits, layers):
        self.numQubits = numQubits
        self.layers = layers
        tmp = list(np.random.choice([qml.RX, qml.RY, qml.RZ], size=layers * numQubits))
        self.bases = tmp + list(reversed(tmp))
        return

    def parametrizedCircuit(self, params):
        flipFlop = 0
        for i in range(self.layers):
            for q in range(self.numQubits):
                self.bases[i * self.numQubits + q](
                    params[i * self.numQubits + q], wires=q
                )
            for q in range(self.numQubits):
                qml.CNOT(
                    wires=[
                        (flipFlop + q) % self.numQubits,
                        (flipFlop + q + 1) % self.numQubits,
                    ]
                )
            flipFlop = (flipFlop + 1) % 2
        return

    def invertedParametrizedCircuit(self, params):
        flipFlop = 0
        if self.layers % 2:
            flipFlop = 1
        for i in range(self.layers):
            for q in range(self.numQubits):
                qml.CNOT(
                    wires=[
                        (flipFlop + q) % self.numQubits,
                        (flipFlop + q + 1) % self.numQubits,
                    ]
                )
            for q in range(self.numQubits):
                self.bases[(2 * self.layers - i - 1) * self.numQubits + q](
                    -params[(2 * self.layers - i - 1) * self.numQubits + q], wires=q
                )
            flipFlop = (flipFlop + 1) % 2
        return

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random([self.layers, self.numQubits]) * 2 - 1
        ) * np.pi
        return np.concatenate(
            (startingParameters, np.flip(-1 * startingParameters, axis=0))
        ).flatten()
