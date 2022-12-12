"""
This file holds classes which contain the neural network classifiers of the GANomaly model.
"""
import logging

import cirq
import numpy
import tensorflow as tf
import tensorflow_quantum as tfq
from qiskit import Aer, IBMQ

import libs.gan_classifiers.QuantumCircuits as quantumCircuits

# qiskit backend
from libs.qiskit_device import get_qiskit_sampler


logger = logging.getLogger(__name__ + ".py")


class Discriminator(tf.keras.Model):
    def __init__(self, num_features, parameters):
        disc_input = tf.keras.layers.Input(shape=(num_features), name="DiscInput")
        model = tf.keras.layers.Dense(num_features)(disc_input)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(num_features / 2)))(model)
        model = tf.keras.layers.Dense(1)(model)
        super().__init__(disc_input, model, name="Discriminator")


class Encoder(tf.keras.Model):
    def __init__(self, num_features, parameters):
        latent_dim = int(parameters["latent_dimensions"])
        enc_input = tf.keras.layers.Input(shape=num_features, name="EncInput")
        model = tf.keras.layers.Dense(num_features)(enc_input)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(latent_dim, int(num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(latent_dim, int(num_features / 4)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(latent_dim)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        super().__init__(enc_input, model, name="Encoder")
        # only after super can we set member variables
        self.latent_dim = latent_dim


class ClassicalDecoder(tf.keras.Model):
    def __init__(self, num_features, parameters):
        latent_dim = int(parameters["latent_dimensions"])
        dec_input = tf.keras.layers.Input(shape=latent_dim, name="DecInput")
        model = tf.keras.layers.Dense(latent_dim)(dec_input)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(num_features, int(latent_dim * 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(num_features, int(latent_dim * 4)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(num_features)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        super().__init__(dec_input, model, name="Decoder")
        # only after super can we set member variables
        self.latent_dim = latent_dim
    def transform_z_to_z_quantum(self, z):
        return z


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
        total_num_cycles = parameters["quantum_depth"]

        qubits = cirq.GridQubit.rect(1, latent_dim)

        tf_dummy_input = tf.keras.Input(shape=(), dtype=tf.string, name="circuit_input")

        qc_instance = getattr(quantumCircuits, quantum_circuit_type)(
            qubits, total_num_cycles
        )
        circuit = qc_instance.build_circuit()

        # readout
        readout = qc_instance.get_readout()

        if parameters["quantum_backend"] == "noiseless":
            backend = "noiseless"
        elif parameters["quantum_backend"] == "IBM - Aer":
            backend = get_qiskit_sampler(Aer.get_backend("statevector_simulator"))
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
        upscaling_layer = tf.keras.layers.Dense(min(num_features, int(latent_dim * 2)))(
            tf_main_circuit
        )
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)
        upscaling_layer = tf.keras.layers.Dense(min(num_features, int(latent_dim * 4)))(
            upscaling_layer
        )
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)
        upscaling_layer = tf.keras.layers.Dense(num_features)(upscaling_layer)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)
        super().__init__(tf_dummy_input, upscaling_layer, name="Decoder")
        # only after super can we set member variables
        self.qubits = qubits
        self.latent_dim = latent_dim
        self.quantum_circuit = circuit + readout
        self.total_num_cycles = total_num_cycles

    def transform_z_to_z_quantum(self, z):
        z_np = z.numpy()
        result = []
        for pair in enumerate(z_np):
            circuit = cirq.Circuit()
            transformed_inputs = 2 * numpy.arcsin(pair[1])
            for j in range(int(self.latent_dim)):
                circuit.append(cirq.rx(transformed_inputs[j]).on(self.qubits[j]))
            result.append(circuit)
        result = tfq.convert_to_tensor(result)
        return result


class Classifier:
    """GAN classifier class collecting the different models. Can be classical or quantum depending on parameter."""

    def __init__(self, data, parameters):
        """Instantiate all required models for the GANomalyNetwork."""
        tf.keras.backend.set_floatx("float64")
        self.num_features = data.feature_length
        self.auto_encoder = Encoder(self.num_features, parameters)
        self.encoder = Encoder(self.num_features, parameters)
        self.discriminator = Discriminator(self.num_features, parameters)
        self.auto_decoder = self.make_auto_decoder(parameters)

    def make_auto_decoder(self, parameters):
        """Returns either a classical or quantum decoder depending on the given parameter."""
        if parameters['method'] == "classical":
            return ClassicalDecoder(self.num_features, parameters)
        if parameters['method'] == "quantum":
            return QuantumDecoder(self.num_features, parameters)


    def print_model_summaries(self):
        """
        Print a model of all models via the logger. Keep in mind that the same model for the encoder is used for both
        of its occurrences.
        """
        self.auto_encoder.summary(print_fn=logger.info)
        self.auto_decoder.summary(print_fn=logger.info)
        if isinstance(self.auto_decoder, QuantumDecoder):
            logger.info("Quantum-Layer in decoder:\n")
            logger.info(self.auto_encoder.quantum_circuit)
        self.discriminator.summary(print_fn=logger.info)

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
        weights_auto_encoder = [numpy.array(w) for w in data["auto_encoder_weights"]]
        self.auto_encoder.set_weights(weights_auto_encoder)

        weights_auto_decoder = [numpy.array(w) for w in data["auto_decoder_weights"]]
        self.auto_decoder.set_weights(weights_auto_decoder)

        weights_encoder = [numpy.array(w) for w in data["encoder_weights"]]
        self.encoder.set_weights(weights_encoder)

        weights_discriminator = [numpy.array(w) for w in data["discriminator_weights"]]
        self.discriminator.set_weights(weights_discriminator)

    def transform_z_to_z_quantum(self, z):
        # only needed for quantum network
        return self.auto_decoder.transform_z_to_z_quantum(z)

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
