"""
This file holds classes which contain the neural network classifiers of the GANomaly model.
"""
import logging
import json

import cirq
import numpy
import tensorflow as tf
import tensorflow_quantum as tfq
import libs.gan_classifiers.QuantumCircuits as quantumCircuits

# qiskit backend
from libs.qiskit_device import get_qiskit_sampler, set_debug_circuit_writer
from qiskit import *


logger = logging.getLogger(__name__ + ".py")


class Discriminator(tf.keras.Model):
    def __init__(self, num_features, parameters):
        latent_dim = int(parameters["latent_dimensions"])

        discInput = tf.keras.layers.Input(
            shape=(num_features), name="DiscInput"
        )
        model = tf.keras.layers.Dense(num_features)(discInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(num_features / 2)))(model)
        model = tf.keras.layers.Dense(1)(model)
        super().__init__(discInput, model, name="Discriminator")


class Encoder(tf.keras.Model):
    def __init__(self, num_features, parameters):
        latent_dim = int(parameters["latent_dimensions"])
        encInput = tf.keras.layers.Input(shape=(num_features), name="EncInput")
        model = tf.keras.layers.Dense(num_features)(encInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(latent_dim, int(num_features / 2)))(
            model
        )
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(latent_dim, int(num_features / 4)))(
            model
        )
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(latent_dim)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        super().__init__(encInput, model, name="Encoder")
        # only after super can we set member variables
        self.latent_dim = latent_dim


class ClassicalDecoder(tf.keras.Model):
    def __init__(self, num_features, parameters):
        latent_dim = int(parameters["latent_dimensions"])
        decInput = tf.keras.layers.Input(shape=(latent_dim), name="DecInput")
        model = tf.keras.layers.Dense(latent_dim)(decInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(num_features, int(latent_dim * 2)))(
            model
        )
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(num_features, int(latent_dim * 4)))(
            model
        )
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(num_features)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        super().__init__(decInput, model, name="Decoder")
        # only after super can we set member variables
        self.latent_dim = latent_dim


class QuantumDecoder(tf.keras.Model):
    """
    backend_mapping = {
        "noiseless": "noiseless",
        "IBM - Aer": get_qiskit_sampler(
            Aer.get_backend("statevector_simulator")
        ),
        #
        # "IBM - Hardware": get_qiskit_sampler(
        #    backend("Q4", "API KEY")
        # )
    }
    """

    def __init__(self, num_features, parameters):
        # settings from parameters
        latent_dim = int(parameters["latent_dimensions"])
        repetitions = parameters["shots"]
        quantum_circuit_type = parameters["quantum_circuit_type"]
        totalNumCycles = parameters["quantum_depth"]

        qubits = cirq.GridQubit.rect(1, latent_dim)

        tf_dummy_input = tf.keras.Input(
            shape=(), dtype=tf.string, name="circuit_input"
        )

        qc_instance = getattr(quantumCircuits, quantum_circuit_type)(
            qubits, totalNumCycles
        )
        circuit = qc_instance.buildCircuit()

        # readout
        readout = qc_instance.getReadOut()

        if parameters["quantum_backend"] == "noiseless":
            backend = "noiseless"
        elif parameters["quantum_backend"] == "IBM - Aer":
            backend = get_qiskit_sampler(
                Aer.get_backend("statevector_simulator")
            )
        elif parameters["quantum_backend"] == "IBM - Hardware":
            provider = IBMQ.enable_account(parameters["IBMQ_token"])
            backend = get_qiskit_sampler(
                provider.get_backend(parameters["IBMQ_backend"])
            )
        else:
            raise ValueError(
                "'quantum_backend' has to be either 'noiseless', 'IBM - Aer', or 'IBM - Hardware'."
            )

        # build main quantum circuit
        tf_main_circuit = tfq.layers.PQC(
            circuit,
            readout,
            repetitions=int(repetitions),
            backend=backend,
            differentiator=tfq.differentiators.ParameterShift(),
        )(tf_dummy_input)

        # upscaling layer
        upscaling_layer = tf.keras.layers.Dense(
            min(num_features, int(latent_dim * 2))
        )(tf_main_circuit)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(
            upscaling_layer
        )
        upscaling_layer = tf.keras.layers.Dense(
            min(num_features, int(latent_dim * 4))
        )(upscaling_layer)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(
            upscaling_layer
        )
        upscaling_layer = tf.keras.layers.Dense(num_features)(upscaling_layer)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(
            upscaling_layer
        )
        super().__init__(tf_dummy_input, upscaling_layer, name="Decoder")
        # only after super can we set member variables
        self.qubits = qubits
        self.latent_dim = latent_dim
        self.quantum_circuit = circuit + readout
        self.totalNumCycles = totalNumCycles

    def transform_z_to_z_quantum(self, z):
        z_np = z.numpy()
        result = []
        for i in range(len(z_np)):
            circuit = cirq.Circuit()
            transformed_inputs = 2 * numpy.arcsin(z_np[i])
            for j in range(int(self.latent_dim)):
                circuit.append(
                    cirq.rx(transformed_inputs[j]).on(self.qubits[j])
                )
            result.append(circuit)
        result = tfq.convert_to_tensor(result)
        return result


class Classifier:
    """Base class for the different Gan Classifiers."""

    def __init__(self, data, parameters):
        """Instantiate all required models for the GANomalyNetwork."""
        tf.keras.backend.set_floatx("float64")
        self.num_features = data.feature_length

    def print_model_summaries(self):
        """
        Print a model of all models in std_out. Keep in mind that the same model for the encoder is used for both
        of its occurrences.
        """
        self.auto_encoder.summary()
        self.auto_decoder.summary()
        self.discriminator.summary()

    def save(self):
        return {
            "auto_encoder_weights": self.auto_encoder.get_weights(),
            "auto_decoder_weights": self.auto_decoder.get_weights(),
            "encoder_weights": self.encoder.get_weights(),
            "discriminator_weights": self.discriminator.get_weights(),
            "latent_dim": self.encoder.latent_dim,
        }

    def load(self, data):
        """Load a previously trained classifier"""
        weights_auto_encoder = [
            numpy.array(w) for w in data["auto_encoder_weights"]
        ]
        self.auto_encoder.set_weights(weights_auto_encoder)

        weights_auto_decoder = [
            numpy.array(w) for w in data["auto_decoder_weights"]
        ]
        self.auto_decoder.set_weights(weights_auto_decoder)

        weights_encoder = [numpy.array(w) for w in data["encoder_weights"]]
        self.encoder.set_weights(weights_encoder)

        weights_discriminator = [
            numpy.array(w) for w in data["discriminator_weights"]
        ]
        self.discriminator.set_weights(weights_discriminator)

    def transform_z_to_z_quantum(self, z):
        # only needed for quantum network
        return z

    def predict(self, x):
        mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        z = self.auto_encoder(x, training=False)
        z_quantum = self.transform_z_to_z_quantum(z)
        x_hat_normal = self.auto_decoder(z_quantum, training=False)
        z_hat_normal = self.encoder(x_hat_normal, training=False)
        return mae(z, z_hat_normal)

    def generate(self, x):
        z = self.auto_encoder(x, training=False)
        z_quantum = self.transform_z_to_z_quantum(z)
        return self.auto_decoder(z_quantum, training=False)


class ClassicalDenseClassifier(Classifier):
    """
    Class containing all required network structures for the GANomaly method as classical dense networks.
    """

    def __init__(self, data, parameters):
        """Instantiate all required models for the GANomalyNetwork."""
        super().__init__(data=data, parameters=parameters)
        self.auto_encoder = Encoder(self.num_features, parameters)
        self.auto_decoder = ClassicalDecoder(self.num_features, parameters)
        self.encoder = Encoder(self.num_features, parameters)
        self.discriminator = Discriminator(self.num_features, parameters)


class QuantumDecoderClassifier(Classifier):
    """
    Class containing all required network structures for the GANomaly method as classical dense networks.
    """

    def __init__(self, data, parameters):
        """Instantiate all required models for the GANomalyNetwork."""
        super().__init__(data=data, parameters=parameters)
        self.auto_encoder = Encoder(self.num_features, parameters)
        self.auto_decoder = QuantumDecoder(self.num_features, parameters)
        self.encoder = Encoder(self.num_features, parameters)
        self.discriminator = Discriminator(self.num_features, parameters)

    def transform_z_to_z_quantum(self, z):
        return self.auto_decoder.transform_z_to_z_quantum(z)

    def print_model_summaries(self):
        """
        Print a model of all models in std_out. Keep in mind that the same model for the encoder is used for both
        of its occurrences.
        """
        self.auto_encoder.summary()
        self.auto_decoder.summary()
        print("Quantum-Layer in decoder:\n")
        print(self.quantum_circuit)
        self.discriminator.summary()
