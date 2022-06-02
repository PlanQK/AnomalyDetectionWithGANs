
import numpy as np
import tensorflow as tf

import tensorflow_quantum as tfq
import sympy
import cirq


class StandardCircuit:
    """
    Very easy PoC circuit.
    """
    def __init__(self, qubits):
        self.qubits = qubits
        self.num_qubits = len(self.qubits)
        self.circuit = cirq.Circuit()
        self.inputParams = np.array(
            [sympy.Symbol(f"thetaX({i})") for i in range(int(self.num_qubits))]
        )
        self.controlParams = np.array([])

    def buildCircuit(self, input_data):
        # params = sympy.symbols(' '.join([f'theta({q})' for q in range(self.num_qubits)]))
        # params = np.asarray(params).reshape((1, self.num_qubits, 3))

        # print(params)

        # 1st trainable parameter
        for i in range(int(self.num_qubits)):
            # create circuit
            self.circuit.append(cirq.rx(self.inputParams[i]).on(self.qubits[i]))
            # self.circuit.append(cirq.rx(sympy.symbols(f"theta({i})")).on(self.qubits[i]))

        # 1st entanglement set
        for i in range(0, int(self.num_qubits) - 1):
            for j in range(i + 1, int(self.num_qubits)):
                self.circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
        
        # 2nd input data
        for i in range(len(input_data)):
            self.circuit.append(cirq.rx(input_data[len(input_data) - i - 1]).on(self.qubits[i]))

        print(self.circuit)
        return self.circuit

    def getReadOut(self):
        return [cirq.Z(q) for q in self.qubits]


def transform(input_data, qubits, amount_qubits=10):
    circuit = cirq.Circuit()
    transformed_inputs = 2 * np.arcsin(input_data)
    for j in range(amount_qubits):
        circuit.append(cirq.rx(transformed_inputs[j]).on(qubits[j%amount_qubits]))
    return tfq.convert_to_tensor([circuit])


if __name__ == "__main__":
    qubits = cirq.GridQubit.rect(1, 10)
    repetitions = 100
    input_data = np.array([.1, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    tf_dummy_input = tf.keras.Input(shape=(), dtype=tf.string, name="circuit_input")

    qc_instance = StandardCircuit(qubits)

    quantum_weights = qc_instance.inputParams.tolist() + qc_instance.controlParams.tolist()
    circuit = qc_instance.buildCircuit(input_data)

    # readout
    readout = qc_instance.getReadOut()
    quantum_circuit = circuit + readout

    #print(quantum_circuit)

    # build main quantum circuit
    tf_main_circuit = tfq.layers.PQC(circuit, readout, repetitions=int(repetitions),
                                    differentiator=tfq.differentiators.ForwardDifference())

    model = tf.keras.Model(tf_dummy_input, tf_main_circuit(tf_dummy_input), name="Decoder")

    converted = transform(input_data, qubits)
    res = model(converted)
    print(res)

    print(tf_main_circuit.trainable_variables)
    print(tf_main_circuit.symbol_values())