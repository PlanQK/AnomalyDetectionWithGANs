import numpy as np
import math
import sympy
import cirq

import tensorflow as tf
import tensorflow_quantum as tfq
from typing import Generator

class StandardCircuit:
    """
    Very easy PoC circuit.
    """
    def __init__(self, qubits):
        self.qubits = qubits
        self.num_qubits = len(self.qubits)
        self.circuit = cirq.Circuit()
        self.inputParams = np.array(
            [sympy.Symbol(f"input{i}") for i in range(int(self.num_qubits))]
        )
        self.controlParams = np.array([])

    def buildCircuit(self):
        for i in range(int(self.num_qubits)):
            # create circuit
            self.circuit.append(cirq.rx(self.inputParams[i]).on(self.qubits[i]))

        # 1st entanglement set
        for i in range(0, int(self.num_qubits) - 1):
            for j in range(i + 1, int(self.num_qubits)):
                self.circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))

        return self.circuit

    def getReadOut(self):
        return [cirq.Z(q) for q in self.qubits]

class IdentityCircuitBase:
    """Base class for the different Parametrized circuits. Derive this class to
  write your own specialized circuit.
  """

    def __init__(self, qubits, totalNumCycles=3):
        self.qubits = qubits
        self.circuit = cirq.Circuit()
        self.totalNumCycles = int(totalNumCycles)
        self.num_qubits = len(self.qubits)
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz],
            int(math.ceil(int(totalNumCycles) / 2) * int(self.num_qubits)),
        )
        self.bases = self.bases.reshape(
            int(math.ceil(int(totalNumCycles) / 2)), int(self.num_qubits)
        )

        self.inputParams = np.array(
            [sympy.Symbol(f"input{i}") for i in range(int(self.num_qubits))]
        )
        numVariables = int(self.num_qubits) * int(totalNumCycles)

        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(numVariables)]
        )
        self.controlParams = self.controlParams.reshape(int(totalNumCycles), int(self.num_qubits))

    def generateCycle(self):
        raise NotImplementedError(
            "This is the base class. You need to specialize this function"
        )
        
    def generateInvCycle(self):
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
            int(math.ceil(int(self.totalNumCycles) / 2)), int(self.num_qubits)
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
            np.random.random([int(int(self.totalNumCycles) / 2), len(self.qubits)])
            * 2
            - 1
        ) * np.pi
        if int(self.totalNumCycles) == 1:
            return np.zeros([1, len(self.qubits)]).flatten()
        if int(self.totalNumCycles) % 2:
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
        if int(self.totalNumCycles) % 2:
            self.generateCycle(-1)

        for i in range(int(int(self.totalNumCycles) / 2)):
            self.generateCycle(i)

        for i in range(int(int(self.totalNumCycles) / 2)):
            self.generateInvCycle(i)
        return self.circuit


class RandomCircuitBase(IdentityCircuitBase):
    def __init__(self, qubits, totalNumCycles=3):
        super().__init__(qubits, totalNumCycles=totalNumCycles)

        self.controlParams = self.controlParams.reshape(int(totalNumCycles), int(self.num_qubits))
        self.bases = np.random.choice(
            [cirq.rx, cirq.ry, cirq.rz], int(totalNumCycles) * int(self.num_qubits)
        )
        self.bases = self.bases.reshape(int(totalNumCycles), int(self.num_qubits))

    def buildCircuit(self):
        # state input
        for i in range(len(self.inputParams)):
            self.circuit.append(
                cirq.rx(self.inputParams[i] * np.pi).on(self.qubits[i])
            )

        for i in range(int(int(self.totalNumCycles))):
            self.generateCycle(i)
        return self.circuit

    def generateInitialParameters(self):
        return (
            (np.random.random([int(self.totalNumCycles), len(self.qubits)]) * 2 - 1)
            * np.pi
        ).flatten()


class CompleteRotationCircuitIdentity(IdentityCircuitBase):
    def __init__(self, qubits, totalNumCycles=3):
        super().__init__(qubits, totalNumCycles=totalNumCycles)
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(3 * int(self.num_qubits) * int(totalNumCycles))]
        )
        self.controlParams = self.controlParams.reshape(
            3 * int(totalNumCycles), int(self.num_qubits)
        )

    def generateCycle(self, cyclePos):
        add = 0
        if int(self.totalNumCycles) % 2:
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
        if cyclePos == int(int(self.totalNumCycles) / 2) - 1:
            return
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
                            3 * (int(self.totalNumCycles) - cyclePos) - 1
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.ry(
                        self.controlParams[
                            3 * (int(self.totalNumCycles) - cyclePos) - 2
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                    cirq.rx(
                        self.controlParams[
                            3 * (int(self.totalNumCycles) - cyclePos) - 3
                        ][qubitID]
                    ).on(self.qubits[qubitID]),
                ]
            )

    def generateInitialParameters(self):
        startingParameters = (
            np.random.random(
                [3 * int(int(self.totalNumCycles) / 2), len(self.qubits)]
            )
            * 2
            - 1
        ) * np.pi
        if int(self.totalNumCycles) == 1:
            return np.zeros([3, len(self.qubits)]).flatten()
        if int(self.totalNumCycles) % 2:
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
    def __init__(self, qubits, totalNumCycles=3):
        super().__init__(qubits, totalNumCycles=totalNumCycles)
        self.controlParams = np.array(
            [sympy.Symbol(f"w{i}") for i in range(3 * int(self.num_qubits) * int(totalNumCycles))]
        )
        self.controlParams = self.controlParams.reshape(
            3 * int(totalNumCycles), int(self.num_qubits)
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
                np.random.random([int(self.totalNumCycles), 3 * len(self.qubits)])
                * 2
                - 1
            )
            * np.pi
        ).flatten()


class StrongEntanglementIdentity(IdentityCircuitBase):
    def generateCycle(self, cyclePos):
        add = 0
        if int(self.totalNumCycles) % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )  # layer of entangling gates
        if cyclePos == int(int(self.totalNumCycles) / 2) - 1:
            return
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
                    self.controlParams[int(self.totalNumCycles) - cyclePos - 1][
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
        if int(self.totalNumCycles) % 2:
            add = 1
        for qubitID in range(len(self.qubits)):
            self.circuit.append(
                self.bases[cyclePos + add][qubitID](
                    self.controlParams[cyclePos + add][qubitID]
                ).on(self.qubits[qubitID])
            )
        # layer of entangling gates
        # last one does not need entangling gates as they cancel the inverted
        if cyclePos == int(int(self.totalNumCycles) / 2) - 1:
            return
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
                    self.controlParams[int(self.totalNumCycles) - cyclePos - 1][
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
        if int(self.totalNumCycles) % 2:
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
                    self.controlParams[int(self.totalNumCycles) - cyclePos - 1][
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

class ReUploadingPrescribedPQC(tf.keras.layers.Layer):
    """
    An object representing a variation upon tfq.ControlledPQC layers. Here individual parameters can be encoded
    multiple times and at different positions without having to do that manually.

    Slightly different functionality than the other classes in here. This is not only a circuit but a whole tfq layer

    Parameters
    ----------
    

    Attributes
    ----------
    theta_val : tf.Variable
    phis : List[sy.symbol]
    thetas : List[sy.symbol]
    symbols: List[str]
    indices : List[str]
    observable : List[cq.Gate]
    varCircuit : cq.Circuit
    computationLayer : tfq.layers.ControlledPQC
    numParameters : int
    numFeatures : int
    numQubits : int

    Code from the xAI Use Case
    c Patrick Steinmueller
    """
    def __init__(self,
                 name="re-uploading_prescribed_PQC"):
        super(ReUploadingPrescribedPQC, self).__init__(name=name)
        self.generateCircuitForGAN()

    def initialize(self, generationFunction: Generator[cirq.Operation, None, None]) -> None:
        """
        Initialized the object.

        Parameters
        ----------
        generationFunction : Generator[cq.Operation, None, None]
            Generator yielding the gates in the sequence that they are applied in the quantum circuit.
        """
        self.theta_val = tf.Variable(
            initial_value=tf.random_uniform_initializer(minval=np.pi * 0, maxval=np.pi * 2) \
                (shape=(1, self.numParameters), dtype=tf.dtypes.float32),
            trainable=True,
            name="thetas")
        # print("start: ", self.theta_val)

        self.qubits = cirq.GridQubit.rect(1, self.numQubits)
        self.phis = [sympy.symbols('phi_' + str(i)) for i in range(self.numFeatures)]
        self.thetas = [sympy.symbols('theta_' + str(i)) for i in range(self.numParameters)]
        self.symbols = [str(s) for s in self.phis + self.thetas]
        self.indices = [self.symbols.index(s) for s in sorted(self.symbols)]
        self.observable = [cirq.Z(q) for q in self.qubits]  # measure on all qubits
        self.varCircuit = cirq.Circuit()
        self.varCircuit.append(generationFunction)

        if (False):
            executor = cirq_rigetti.circuit_sweep_executors.with_quilc_parametric_compilation
            qc = get_qc(
                f'{self.numQubits}q-qvm',
                as_qvm=True,
                noisy=True,
                compiler_timeout=100000)
            sampler = cirq_rigetti.RigettiQCSSampler(
                quantum_computer=qc,
                executor=executor)

        #        self.computationLayer = tfq.layers.ControlledPQC( self.varCircuit,
        #                                                self.observable,
        #                                                backend=sampler,
        #                                                repetitions=1000)

        self.computationLayer = tfq.layers.ControlledPQC(self.varCircuit,
                                                         self.observable)
        self.x = tfq.layers.Expectation()

        # print(self.varCircuit)

        # output = self.computationLayer([inputcircuit_den_wir_nicht_brauche_also_leer, duplizierte_numerische_wert])
        # => self.abstractLayer(nicht_dupliziert_numerische_werte)

    def generateCircuitForGAN(self):
        self.numParameters = 40   # trainable parameters
        self.numFeatures = 20     # input data
        self.numQubits = 10

        def makeCircuit():
            # 1st input
            for i in range(int(self.numQubits)):
                yield cirq.rx(self.phis[i]).on(self.qubits[i])

            # 1st trainable parameters
            for i in range(int(self.numQubits)):
                yield cirq.rx(self.thetas[i]).on(self.qubits[i])

            # 1st entanglement set
            for j in range(0, int(self.numQubits) - 1):
                for l in range(j + 1, int(self.numQubits)):
                    yield cirq.CNOT(self.qubits[j], self.qubits[l])

            # 2nd input
            j = int(self.numQubits)
            for i in range(int(self.numQubits)):
                yield cirq.rx(self.phis[i+j]).on(self.qubits[i])

            # 2nd trainable parameters
            i = int(self.numQubits)
            for k in range(int(self.numQubits)):
                yield cirq.rx(self.thetas[i+k]).on(self.qubits[k])
                yield cirq.rz(self.thetas[(i*2)+k]).on(self.qubits[k])
                yield cirq.ry(self.thetas[(i*3)+k]).on(self.qubits[k])

            # 2nd entanglement set
            for l in range(0, int(self.numQubits) - 1):
                for m in range(l + 1, int(self.numQubits)):
                    yield cirq.CNOT(self.qubits[l], self.qubits[m])

        self.initialize(makeCircuit())

    def call(self, inputs: tf.Variable, **kwargs) -> np.ndarray:
        """
        Every tensorflow layer is equipped with a call method, to make the layer itself a callable function.

        Parameters
        ----------
        inputs: np.ndarray shape of List[0] = [batch_dim, data_length]
        """
        # Get length of first input
        # structure of the input list of [batches of inputs]
        # the list length corresponds to the number of "actions" and is assumed to
        # be one here
        # the batch dimension is the number of different inputs that we are
        # considering here
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        # print(batch_dim)

        # we always need to specify an circuit object, that preprocesses the |0>
        # state. If we don't want that, we input an empty circuit
        tiled_up_circuits = tf.repeat(tfq.convert_to_tensor([cirq.Circuit()]),
                                      repeats=batch_dim)
        tiled_up_theta_vals = tf.tile(self.theta_val, multiples=[batch_dim, 1])
        # print(tiled_up_theta_vals)
        tiled_up_input_vals = tf.tile(inputs[0], multiples=[1, 1])
        # print(tiled_up_input_vals)

        joined_vars = tf.concat([tf.cast(tiled_up_input_vals, dtype=tf.float32), tf.cast(tiled_up_theta_vals, dtype=tf.float32)], axis=1)
        # print(joined_vars)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        funk = self.computationLayer([tiled_up_circuits, joined_vars]) #plyint-disable-non-callable
        return funk

    def getTFWeights(self):
        return self.theta_val

    def getWeights(self):
        return self.theta_val.numpy()

    def setWeights(self, weights):
        super().set_weights([weights.reshape(1, self.numParameters)])