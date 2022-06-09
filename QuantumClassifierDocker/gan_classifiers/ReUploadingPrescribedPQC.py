import tensorflow as tf
import tensorflow_quantum as tfq

import numpy as np
import sympy as sy
import cirq as cq

from typing import Generator


class ReUploadingPrescribedPQC(tf.keras.layers.Layer):
    """
    The Object Representing a variation upon tfq.ControlledPQC layers. Here individual parameters can be encoded
    multiple times and at different positions without having to do that manually.

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

    def initialize(self, generationFunction: Generator[cq.Operation, None, None]) -> None:
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
        print("start: ", self.theta_val)

        self.qubits = cq.GridQubit.rect(1, self.numQubits)
        self.phis = [sy.symbols('phi_' + str(i)) for i in range(self.numFeatures)]
        self.thetas = [sy.symbols('theta_' + str(i)) for i in range(self.numParameters)]
        self.symbols = [str(s) for s in self.phis + self.thetas]
        self.indices = [self.symbols.index(s) for s in sorted(self.symbols)]
        self.observable = [cq.Z(q) for q in self.qubits]  # measure on all qubits
        self.varCircuit = cq.Circuit()
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
                yield cq.rx(self.phis[i]).on(self.qubits[i])

            # 1st trainable parameters
            for i in range(int(self.numQubits)):
                yield cq.rx(self.thetas[i]).on(self.qubits[i])

            # 1st entanglement set
            for j in range(0, int(self.numQubits) - 1):
                for l in range(j + 1, int(self.numQubits)):
                    yield cq.CNOT(self.qubits[j], self.qubits[l])

            # 2nd input
            j = int(self.numQubits)
            for i in range(int(self.numQubits)):
                yield cq.rx(self.phis[i+j]).on(self.qubits[i])

            # 2nd trainable parameters
            i = int(self.numQubits)
            for k in range(int(self.numQubits)):
                yield cq.rx(self.thetas[i+k]).on(self.qubits[k])
                yield cq.rz(self.thetas[(i*2)+k]).on(self.qubits[k])
                yield cq.ry(self.thetas[(i*3)+k]).on(self.qubits[k])

            # 2nd entanglement set
            for l in range(0, int(self.numQubits) - 1):
                for m in range(l + 1, int(self.numQubits)):
                    yield cq.CNOT(self.qubits[l], self.qubits[m])

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
        tiled_up_circuits = tf.repeat(tfq.convert_to_tensor([cq.Circuit()]),
                                      repeats=batch_dim)
        tiled_up_theta_vals = tf.tile(self.theta_val, multiples=[batch_dim, 1])
        # print(tiled_up_theta_vals)
        tiled_up_input_vals = tf.tile(inputs[0], multiples=[1, 1])
        # print(tiled_up_input_vals)

        joined_vars = tf.concat([tiled_up_input_vals, tiled_up_theta_vals], axis=1)
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

if __name__ == "__main__":
    tmp = ReUploadingPrescribedPQC()
    input_shape = tf.keras.Input(shape=tmp.numFeatures,
                                dtype=tf.dtypes.float32,
                                name="input")
    expec = tmp([input_shape])

    model = tf.keras.Model(inputs=[input_shape],
                                outputs=expec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)

    result = model(tf.constant([[i for i in range(20)]], dtype=tf.float32))

    print(result)

    print("end: ", tmp.getTFWeights())