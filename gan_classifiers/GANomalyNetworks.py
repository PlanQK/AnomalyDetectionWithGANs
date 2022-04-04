"""
This file holds classes which contain the neural network classifiers of the GANomaly model.
"""
import logging
import json

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from gan_classifiers.EnvironmentVariableManager import EnvironmentVariableManager
from gan_classifiers.QuantumCircuits import CompleteRotationCircuitIdentity, CompleteRotationCircuitRandom, \
    StandardCircuit, StrongEntanglementIdentity, StrongEntanglementRandom, LittleEntanglementIdentity, \
    LittleEntanglementRandom, SemiClassicalIdentity, SemiClassicalRandom

logger = logging.getLogger(__name__ + ".py") 

class Classifier:
    """Base class for the different Gan Classifiers.

    Ideally, only the getEncoder, getDecoder, getDiscriminator and getGANomaly methods need to be
    specified to obtain a new classifier.
    """
    def __init__(self, data):
        """Instantiate all required models for the GANomalyNetwork.
        """
        tf.keras.backend.set_floatx("float64")
        self.num_features = data.feature_length
        self.envMgr = EnvironmentVariableManager()
        self.latent_dim = int(self.envMgr["latent_dimensions"])
        self.threshold = 0

    def getEncoder(self):
        """Return the Tensorflow model for the Encoder.

        Depending on the classifier type this includes the circuit
        specification for the quantum generator
        """
        raise NotImplementedError("Need to use the derived class")

    def getDecoder(self):
        """Return the Tensorflow model for the Encoder.

        Depending on the classifier type this includes the circuit
        specification for the quantum generator
        """
        raise NotImplementedError("Need to use the derived class")

    def getDiscriminator(self):
        """Return the Tensorflow model for the Encoder.

        Depending on the classifier type this includes the circuit
        specification for the quantum generator
        """
        raise NotImplementedError("Need to use the derived class")

    def getGANomaly(self, auto_encoder, auto_decoder, encoder, discriminator):
        """Generate the Tensorflow model for the GANomaly method.

        This combines all sub-networks into a
        complete classifier.

        Args:
            auto_encoder ([type]): Tensorflow model for the auto_encoder
            auto_decoder ([type]): Tensorflow model for the auto_decoder
            encoder ([type]): Tensorflow model for the encoder
            discriminator ([type]): Tensorflow model for the discriminator
        """
        raise NotImplementedError("Need to use the derived class")

    def print_model_summaries(self):
        """
        Print a model of all models in std_out. Keep in mind that the same model for the encoder is used for both
        of its occurrences.
        """
        self.auto_encoder.summary()
        self.auto_decoder.summary()
        self.discriminator.summary()

    def save(self, step, MCC, threshold, overwrite_best=False):
        """Store the trained weights and parameters."""
        if overwrite_best:
            data = {
                "latent_dim": self.latent_dim,
                "threshold": threshold,
            }
            self.threshold = threshold
            self.auto_encoder.save_weights(
                f"model/checkpoint/{self.__class__.__name__}_auto_encoder_weights"
            )
            self.auto_decoder.save_weights(
                f"model/checkpoint/{self.__class__.__name__}_auto_decoder_weights"
            )
            self.encoder.save_weights(
                f"model/checkpoint/{self.__class__.__name__}_encoder_weights"
            )
            self.discriminator.save_weights(
                f"model/checkpoint/{self.__class__.__name__}_discriminator_weights"
            )
            with open(
                f"model/checkpoint/{self.__class__.__name__}_other_parameters", "w"
            ) as json_file:
                json.dump(data, json_file)

        self.auto_encoder.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_auto_encoder_weights_step_{step}_MCC_{MCC:.2f}"
        )
        self.auto_decoder.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_auto_decoder_weights_{step}_MCC_{MCC:.2f}"
        )
        self.encoder.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_encoder_weights_{step}_MCC_{MCC:.2f}"
        )
        self.discriminator.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_discriminator_weights_{step}_MCC_{MCC:.2f}"
        )

        return None

    def loadClassifier(self):
        """Load a previously trained classifier from files."""
        data = {}
        with open(f"model/checkpoint/{self.__class__.__name__}_other_parameters") as json_file:
            data = json.load(json_file)
        self.latent_dim = data["latent_dim"]
        self.threshold = data["threshold"]
        self.auto_encoder.load_weights(
            f"model/checkpoint/{self.__class__.__name__}_auto_encoder_weights"
        )
        self.auto_decoder.load_weights(
            f"model/checkpoint/{self.__class__.__name__}_auto_decoder_weights"
        )
        self.encoder.load_weights(
            f"model/checkpoint/{self.__class__.__name__}_encoder_weights"
        )
        self.discriminator.load_weights(
            f"model/checkpoint/{self.__class__.__name__}_discriminator_weights"
        )
        return None

class ClassicalDenseNetworks(Classifier):
    """
    Class containing all required network structures for the GANomaly method as classical dense networks.
    """

    def __init__(self, data):
        """Instantiate all required models for the GANomalyNetwork.
        """
        super().__init__(data=data)
        self.auto_encoder = self.getEncoder()
        self.auto_decoder = self.getDecoder()
        self.encoder = self.getEncoder()
        self.discriminator = self.getDiscriminator()

    def getDiscriminator(self):
        """
        Return the tensorflow model of the Discriminator.
        """
        discInput = tf.keras.layers.Input(
            shape=(self.num_features), name="DiscInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(discInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(self.num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(self.num_features / 2)))(model)
        model = tf.keras.layers.Dense(1, activation="sigmoid")(model)
        return tf.keras.Model(discInput, model, name="Discriminator")

    def getEncoder(self):
        """
        Return the tensorflow model of the Encoder.
        """
        encInput = tf.keras.layers.Input(
            shape=(self.num_features), name="EncInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(encInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(self.latent_dim, int(self.num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(self.latent_dim, int(self.num_features / 4)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(self.latent_dim)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        return tf.keras.Model(encInput, model, name="Encoder")

    def getDecoder(self):
        """
        Return the tensorflow model of the Decoder.
        """
        decInput = tf.keras.layers.Input(
            shape=(self.latent_dim), name="DecInput"
        )
        model = tf.keras.layers.Dense(self.latent_dim)(decInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(self.num_features, int(self.latent_dim * 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(min(self.num_features, int(self.latent_dim * 4)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(self.num_features)(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        return tf.keras.Model(decInput, model, name="Decoder")

class QuantumDecoderNetworks(Classifier):
    """
    Class containing all required network structures for the GANomaly method as classical dense networks.
    """

    def __init__(self, data):
        """Instantiate all required models for the GANomalyNetwork.
        """
        super().__init__(data=data)
        self.repetitions = self.envMgr["shots"]
        self.qubits = cirq.GridQubit.rect(1, self.latent_dim)
        self.quantum_weights = None
        self.quantum_circuit = None
        self.quantum_circuit_type = self.envMgr["quantum_circuit_type"]
        self.totalNumCycles = self.envMgr["quantum_depth"]
        self.auto_encoder = self.getEncoder()
        self.auto_decoder = self.getDecoder()
        self.encoder = self.getEncoder()
        self.discriminator = self.getDiscriminator()


    def getDiscriminator(self):
        """
        Return the tensorflow model of the Discriminator.
        """
        discInput = tf.keras.layers.Input(
            shape=(self.num_features), name="DiscInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(discInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(self.num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(1, int(self.num_features / 2)))(model)
        model = tf.keras.layers.Dense(1, activation="sigmoid")(model)
        return tf.keras.Model(discInput, model, name="Discriminator")

    def getEncoder(self):
        """
        Return the tensorflow model of the Encoder.
        """
        encInput = tf.keras.layers.Input(
            shape=(self.num_features), name="EncInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(encInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(self.latent_dim, int(self.num_features / 2)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        model = tf.keras.layers.Dense(max(self.latent_dim, int(self.num_features / 4)))(model)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        # sigmoid necessary to map model output to [0, 1] (as quantum-decoder maps that to [0, \pi])
        model = tf.keras.layers.Dense(self.latent_dim, activation="sigmoid")(model)
        return tf.keras.Model(encInput, model, name="Encoder")

    def getDecoder(self):
        """
        Return the tensorflow model of the Decoder.
        """
        
        tf_dummy_input = tf.keras.Input(shape=(), dtype=tf.string, name="circuit_input")

        if self.quantum_circuit_type == "standard":
            qc_instance = StandardCircuit(self.qubits)
        elif self.quantum_circuit_type == "CompleteRotationCircuitIdentity":
            qc_instance = CompleteRotationCircuitIdentity(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "CompleteRotationCircuitRandom":
            qc_instance = CompleteRotationCircuitRandom(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "StrongEntanglementIdentity":
            qc_instance = StrongEntanglementIdentity(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "StrongEntanglementRandom":
            qc_instance = StrongEntanglementRandom(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "LittleEntanglementIdentity":
            qc_instance = LittleEntanglementIdentity(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "LittleEntanglementRandom":
            qc_instance = LittleEntanglementRandom(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "SemiClassicalIdentity":
            qc_instance = SemiClassicalIdentity(self.qubits, self.totalNumCycles)
        elif self.quantum_circuit_type == "SemiClassicalRandom":
            qc_instance = SemiClassicalRandom(self.qubits, self.totalNumCycles)

        self.quantum_weights = qc_instance.inputParams.tolist() + qc_instance.controlParams.tolist()
        circuit = qc_instance.buildCircuit()

        # readout
        readout = qc_instance.getReadOut()
        self.quantum_circuit = circuit + readout

        # build main quantum circuit
        tf_main_circuit = tfq.layers.PQC(circuit, readout, repetitions=int(self.repetitions),
                                         differentiator=tfq.differentiators.ForwardDifference())(tf_dummy_input)

        # upscaling layer
        upscaling_layer = tf.keras.layers.Dense(min(self.num_features, int(self.latent_dim * 2)))(tf_main_circuit)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)
        upscaling_layer = tf.keras.layers.Dense(min(self.num_features, int(self.latent_dim * 4)))(upscaling_layer)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)
        upscaling_layer = tf.keras.layers.Dense(self.num_features)(upscaling_layer)
        upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)

        model = tf.keras.Model(tf_dummy_input, upscaling_layer, name="Decoder")

        return model

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
