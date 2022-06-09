"""
This file holds classes which contain the neural network classifiers of the GANomaly model.
"""
from base64 import encode
import logging
import json
import numpy as np

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from gan_classifiers.EnvironmentVariableManager import EnvironmentVariableManager
from gan_classifiers.QuantumCircuits import CompleteRotationCircuitIdentity, CompleteRotationCircuitRandom, \
    StandardCircuit, StrongEntanglementIdentity, StrongEntanglementRandom, LittleEntanglementIdentity, \
    LittleEntanglementRandom, SemiClassicalIdentity, SemiClassicalRandom, ReUploadingPrescribedPQC

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
    
    def decoder_training_step(self):
        """_summary_

        Raises:
            NotImplementedError: meant to only be executed from the derived class
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
            for i in range(len(self.auto_decoder)):
                self.auto_decoder[i].save_weights(
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
        for i in range(len(self.auto_decoder)):
            self.auto_decoder[i].save_weights(
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
        for i in range(len(self.auto_decoder)):
            self.auto_decoder[i].load_weights(
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

    def __init__(self, data, sim_amount_layers=10):
        """
        Instantiate all required models for the GANomalyNetwork.
        """
        super().__init__(data=data)
        self.auto_encoder = self.getEncoder(sim_amount_layers)
        self.auto_decoder = [self.getDecoder(sim_amount_layers)]
        self.encoder = self.getEncoder(sim_amount_layers)
        self.discriminator = self.getDiscriminator()

    def getDiscriminator(self, sim_amount_layers=8):
        """
        Build a tensorflow model of the Discriminator with the given depth (sim_amount_layers).

        Return the model.
        """
        discInput = tf.keras.layers.Input(
            shape=(self.num_features), name="DiscInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(discInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        for size in [int(s) for s in np.linspace(self.num_features, 1, num=sim_amount_layers)]:
            if size == self.num_features: # to ensure that the input layer is not occuring twice
                continue
            model = tf.keras.layers.Dense(size)(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        return tf.keras.Model(discInput, model, name="Discriminator")

    def getEncoder(self, sim_amount_layers=10):
        """
        Build the tensorflow model of the Encoder with the given depth (sim_amount_layers).

        Return the model.
        """
        encInput = tf.keras.layers.Input(
            shape=(self.num_features), name="EncInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(encInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        for size in [int(s) for s in np.linspace(self.num_features, self.latent_dim, num=sim_amount_layers)]:
            if size == self.num_features:
                continue
            model = tf.keras.layers.Dense(size)(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        return tf.keras.Model(encInput, model, name="Encoder")

    def getDecoder(self, sim_amount_layers=10):
        """Build the tensorflow model of the Decoder with the given depth (sim_amount_layers).

        Return the model.
        """
        decInput = tf.keras.layers.Input(
            shape=(self.latent_dim), name="DecInput"
        )
        model = tf.keras.layers.Dense(self.latent_dim)(decInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        for size in [int(s) for s in np.linspace(self.latent_dim, self.num_features, num=sim_amount_layers)]:
            if size == self.latent_dim:
                continue
            model = tf.keras.layers.Dense(size)(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        # FK: currently not working. Seems like an error with compatibility of keras and tensorflow. Not sure, though, cause it works for quantum
        # tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

        return tf.keras.Model(decInput, model, name="Decoder")

    def decoder_training_step(self, encoded_data, training=True):
        return self.auto_decoder[0](encoded_data, training=training)

class QuantumDecoderNetworks(Classifier):
    """
    Class containing all required network structures for the GANomaly method as classical dense networks.
    """

    def __init__(self, data):
        """Instantiate all required models for the GANomalyNetwork.
        """
        super().__init__(data=data)
        self.repetitions = self.envMgr["shots"]
        # self.qubits = cirq.GridQubit.rect(1, self.latent_dim)
        self.amount_qubits = 10
        self.qubits = cirq.GridQubit.rect(1, self.amount_qubits)
        self.amount_circuits = int(self.num_features / self.amount_qubits) # TODO this has to be more refined when not 150 - 50

        self.quantum_weights = None
        self.quantum_circuit = None
        self.quantum_circuit_type = self.envMgr["quantum_circuit_type"]
        self.totalNumCycles = self.envMgr["quantum_depth"]
        self.auto_encoder = self.getEncoder()
        self.auto_decoder = self.getDecoder()
        self.encoder = self.getEncoder()
        self.discriminator = self.getDiscriminator()


    def getDiscriminator(self, sim_amount_layers=8):
        """
        Return the tensorflow model of the Discriminator with the given depth (sim_amount_layers).
        """
        discInput = tf.keras.layers.Input(
            shape=(self.num_features), name="DiscInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(discInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        for size in [int(s) for s in np.linspace(self.num_features, 1, num=sim_amount_layers)]:
            if size == self.num_features: # to ensure that the input layer is not occuring twice
                continue
            model = tf.keras.layers.Dense(size)(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)
        return tf.keras.Model(discInput, model, name="Discriminator")

    def getEncoder(self, sim_amount_layers=10):
        """
        Return the tensorflow model of the Encoder with the given depth (sim_amount_layers).
        """
        encInput = tf.keras.layers.Input(
            shape=(self.num_features), name="EncInput"
        )
        model = tf.keras.layers.Dense(self.num_features)(encInput)
        model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        for size in [int(s) for s in np.linspace(self.num_features, self.latent_dim, num=sim_amount_layers)]:
            if size == self.num_features:
                continue
            model = tf.keras.layers.Dense(size)(model)
            model = tf.keras.layers.LeakyReLU(alpha=0.05)(model)

        return tf.keras.Model(encInput, model, name="Encoder")
    
    def decoder_training_step(self, encoded_data, training=True):
        """Do one training step for the quantum decoder.
        The encoded data (specifically, a batch from it) is needed

        Args:
            encoded_data (tensorflow.Tensor): the input data which got encoded by the Encoder
            training (bool, optional): training or validation phase. Defaults to True.

        Returns:
            tensorflow.Tensor: decoded data = the encoded data which got processed in the quantum decoder
        """
        all_results = []

        indices = []
        enc_dim = encoded_data.shape[1]
        step_size = int(self.num_features / enc_dim)
        counter = -1
        for feat in range(self.num_features):
            if feat % step_size == 0:
                counter += 1
            
            if counter >= enc_dim:
                counter = 0
            indices.append(counter)

        cnt = 1
        for feat in range(self.num_features):
            idx = (cnt + (feat % step_size)) % enc_dim
            indices.append(idx)

            if feat % step_size == (step_size-1):
                cnt += 1

        enc = encoded_data.numpy()
        for i in range(encoded_data.shape[0]): # for every sample in the batch
            one_sample_all = []
            for idx in indices:
                one_sample_all.append(enc[i][idx])
            
            for circ in range(self.amount_circuits):
                one_sample_result = self.auto_decoder[circ](one_sample_all[(circ*self.amount_qubits*2):((circ+1)*self.amount_qubits*2)],
                                                            training=training)
                all_results.append(one_sample_result)

        return tf.cast(tf.concat(all_results, 1), dtype=tf.dtypes.float64)

    def get_part_of_input(self, enc_data, step):
        """_summary_

        Args:
            enc_data (_type_): _description_
            step (_type_): _description_

        Returns:
            tf.Variable: tensor for input to the auto decoder. shape has to be (1, x) with x = numQubits*2
        """
        # return z
        z_np = enc_data.numpy()
        result = []
        for i in range(len(z_np)): # iterate over every encoded input
            circuit = cirq.Circuit()
            transformed_inputs = 2 * np.arcsin(z_np[i])
            counter = 1
            while(counter <= 3):
                circuit.append(cirq.rx())
                counter += 1
            for j in range(step*self.amount_qubits, (step+1)*self.amount_qubits):
                circuit.append(cirq.rx(transformed_inputs[j]).on(self.qubits[j%self.amount_qubits]))
            result.append(circuit)
        result = tfq.convert_to_tensor(result)
        stop = "stop"
        return result

    def getDecoder(self, sim_amount_layers=10, plot_model=False):
        """
        Return the tensorflow model of the Decoder with the given depth (sim_amount_layers).

        You can plot the architecture of the model into a model.png file, if you set the argument plot_model to True
        """

        all_models = []
        for i in range(self.amount_circuits):
        
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
            elif self.quantum_circuit_type == "FakeNewsCustom":
                tfq_layer = ReUploadingPrescribedPQC()
                input_shape = tf.keras.Input(shape=tfq_layer.numFeatures, dtype=tf.dtypes.float32, name="input")
                expec = tfq_layer([input_shape])

                model = tf.keras.Model(inputs=[input_shape], outputs=expec, name="Decoder")
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.02),
                              loss=tf.keras.losses.MeanSquaredError())

                all_models.append(model)
                if plot_model:
                    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

                continue

            self.quantum_weights = qc_instance.inputParams.tolist() + qc_instance.controlParams.tolist()
            circuit = qc_instance.buildCircuit()

            # print(circuit)

            # readout
            readout = qc_instance.getReadOut()
            self.quantum_circuit = circuit + readout

            # print(self.quantum_circuit)

            # build main quantum circuit
            tf_main_circuit = tfq.layers.PQC(circuit, readout, repetitions=int(self.repetitions),
                                                differentiator=tfq.differentiators.ForwardDifference())(tf_dummy_input)

            # upscaling layer
            # size_firstDenseLayer = min(self.num_features, int(self.latent_dim * 2))
            # upscaling_layer = tf.keras.layers.Dense(size_firstDenseLayer)(tf_main_circuit)
            # upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)

            # for size in [int(s) for s in np.linspace(size_firstDenseLayer, self.num_features, num=sim_amount_layers)]:
            #     if size == size_firstDenseLayer:
            #         continue
            #     upscaling_layer = tf.keras.layers.Dense(size)(upscaling_layer)
            #     upscaling_layer = tf.keras.layers.LeakyReLU(alpha=0.05)(upscaling_layer)

            model = tf.keras.Model(tf_dummy_input, tf_main_circuit, name="Decoder")
            all_models.append(model)

            if plot_model:
                tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

        return all_models

    def print_model_summaries(self):
        """
        Print a model of all models in std_out. Keep in mind that the same model for the encoder is used for both
        of its occurrences.
        """
        self.auto_encoder.summary()
        self.auto_decoder[0].summary()
        print("Quantum-Layer in decoder:\n")
        print(self.quantum_circuit)
        self.discriminator.summary()