import json
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import pennylane as qml
from .DataIO import (
    get_feature_length,
    NoLabelSampler,
    LabelSampler,
    load_prediction_set_no_labels,
)
from .slimtasq import (
    AnoGanAnsatz,
    WGanOptimization,
    AnoGanCost,
    AnoWGan,
    ThresholdWrapper,
)
from . import CircHelper

from . import PennylaneHelper


class generatorInputSampler:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def __call__(self, batchSize):
        return np.array(
            [np.random.uniform(0, 1, self.latent_dim) for i in range(batchSize)]
        )


class generatorInputSamplerTFQ(generatorInputSampler):
    def __call__(self, batchSize):
        return [
            tfq.convert_to_tensor([cirq.Circuit()] * batchSize),
            super().__call__(batchSize),
            np.array([[1]] * batchSize),
        ]


class Classifier:
    def __init__(self, n_steps=1200, totalNumCycles=4):
        self.anoGan = None
        self.num_features = get_feature_length()

        # only allow the number of qubits to be between 1-9
        self.latent_dim = max(1, int(self.num_features / 3))
        self.latent_dim = min(9, self.latent_dim)
        self.totalNumCycles = totalNumCycles
        self.n_steps = n_steps

        self.cost = AnoGanCost(self.ansatz)
        self.opt = WGanOptimization(
            tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
            "WGAN",
            n_steps=n_steps,
            updateInterval=n_steps + 10,
        )

        self.ansatz = AnoGanAnsatz(self.__class__.__name__)
        self.ansatz.generator = self.getGenerator(bases)
        self.ansatz.discriminator = self.getDiscriminator()
        self.ansatz.anoGanModel = self.getAnoGan(
            self.ansatz.generator, self.ansatz.discriminator
        )

    def getDiscriminator(self):
        discSeq = tf.keras.Sequential()
        discSeq.add(tf.keras.layers.Dense(self.num_features))
        discSeq.add(tf.keras.layers.Dense(max(1, int(self.num_features / 2))))
        discSeq.add(tf.keras.layers.Dense(max(1, int(self.num_features / 2))))

        discInput = tf.keras.layers.Input(shape=(self.num_features))
        features = discSeq(discInput)
        valid = tf.keras.layers.Dense(1)(features)
        return tf.keras.Model(discInput, valid)

    def getGenerator(self, bases):
        raise NotImplementedError("Need to use the derived class")

    def getAnoGan(self, generator, discriminator):
        raise NotImplementedError("Need to use the derived class")

    def train(self):
        self.opt.run(self.cost)
        # now optimize threshold
        self.anoGan = self.cost.buildAnoGan(self.opt.opt)

    def predict(self):
        X = load_prediction_set_no_labels()
        return self.anoGan.predict(X)

    @classmethod
    def loadClassifier(cls):
        data = {}
        with open(f"model/{cls.__name__}_other_parameters") as json_file:
            data = json.load(json_file)
            threshold = data["threshold"]
            bases = data["bases"]
        qc = cls(bases=bases)
        qc.ansatz.generator.load_weights(f"model/{cls.__name__}_generator_weights")
        qc.ansatz.discriminator.load_weights(
            f"model/{cls.__name__}_discriminator_weights"
        )
        qc.ansatz.anoGanModel.load_weights(f"model/{cls.__name__}_anoGan_weights")

        qc.anoGan = AnoWGan(qc.ansatz, qc.opt.opt)
        qc.anoGan = ThresholdWrapper(qc.anoGan)
        qc.anoGan._threshold = threshold
        return qc

    def save(self):
        data = {
            "threshold": self.anoGan._threshold,
            "bases": self.circuitObject.getBases(),
        }
        self.ansatz.generator.save_weights(
            f"model/{self.__class__.__name__}_generator_weights"
        )
        self.ansatz.discriminator.save_weights(
            f"model/{self.__class__.__name__}_discriminator_weights"
        )
        self.ansatz.anoGanModel.save_weights(
            f"model/{self.__class__.__name__}_anoGan_weights"
        )
        with open(
            f"model/{self.__class__.__name__}_other_parameters", "w"
        ) as json_file:
            json.dump(data, json_file)


class ClassicalClassifier(Classifier):
    def __init__(self, n_steps, bases):
        super().__init__(n_steps, bases)

        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSampler(self.latent_dim)
        self.ansatz.getTestSample = LabelSampler()

        self.ansatz.trainingDataSampler = LabelSampler()
        self.ansatz.anoGanInputs = [np.array([[1]])]
        self.ansatz.checkAnsatz()


class TfqSimulator(Classifier):
    def __init__(self, n_steps=1200, bases=None):
        super().__init__(n_steps)
        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSamplerTFQ(self.latent_dim)
        self.ansatz.getTestSample = LabelSampler()
        self.ansatz.anoGanInputs = [
            tfq.convert_to_tensor([cirq.Circuit()]),
            np.array([[1]]),
        ]
        self.ansatz.trainingDataSampler = LabelSampler()
        self.ansatz.checkAnsatz()

    def getGenerator(self, bases):
        self.circuitObject = CircHelper.LittleEntanglementIdentity(
            self.latent_dim, 1, self.totalNumCycles
        )
        if bases:
            self.circuitObject.setBases(bases)
        circuit = self.circuitObject.buildCircuit()
        # build generator
        circuitInput = tf.keras.Input(shape=(), dtype=tf.string, name="circuitInput")
        circuitInputParam = tf.keras.Input(
            shape=len(self.circuitObject.inputParams), name="circuitInputParam"
        )
        paramsInput = tf.keras.Input(shape=1, name="paramsInput")
        paramsInput2_layer = tf.keras.layers.Dense(
            np.prod(self.circuitObject.controlParams.shape),
            input_shape=(1,),
            activation="sigmoid",
            name="paramsInputDense",
        )

        paramsInput2 = paramsInput2_layer(paramsInput)

        sampler = tfq.layers.ControlledPQC(circuit, self.circuitObject.getReadOut())
        concat = tf.concat([circuitInputParam, paramsInput2], axis=1)
        expectation = sampler([circuitInput, concat])

        generatedSample = tf.keras.layers.Dense(
            self.num_features, activation="sigmoid", name="postProcessing2"
        )(expectation)

        generator = tf.keras.Model(
            inputs=[circuitInput, circuitInputParam, paramsInput],
            outputs=generatedSample,
        )

        # set the weights of the quantum circuit according to arxiv:1903.05076
        paramsInput2_layer.set_weights(
            [
                np.array([self.circuitObject.generateInitialParameters()]),
                np.zeros((np.prod(self.circuitObject.controlParams.shape),)),
            ]
        )
        return generator

    def getAnoGan(self, generator, discriminator, anoGan_disc_weight=1):
        # anoGan
        oneInput = tf.keras.layers.Input(shape=1, name="oneInput")
        adjustInput = tf.keras.layers.Dense(
            self.latent_dim, activation="sigmoid", name="adjustInput"
        )(oneInput)
        circuitInput2 = tf.keras.Input(shape=(), dtype=tf.string, name="circuitInput2")
        genOutput = generator(inputs=[circuitInput2, adjustInput, oneInput])
        discOutput = discriminator(genOutput)

        inverse_weight = 1.0 / anoGan_disc_weight
        return tf.keras.Model(
            inputs=[circuitInput2, oneInput],
            outputs=[inverse_weight * genOutput, anoGan_disc_weight * discOutput],
        )


class PennylaneSimulator(Classifier):
    def __init__(self, n_steps=1200, bases=None):
        super().__init__(n_steps)

        # Pennylane specifics
        self.device = qml.device("default.qubit", wires=self.latent_dim)

        self.circuitObject = PennylaneHelper.LittleEntanglementIdentity(
            self.latent_dim, self.totalNumCycles
        )

        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSampler(self.latent_dim)
        self.ansatz.getTestSample = LabelSampler()

        self.ansatz.trainingDataSampler = LabelSampler()
        self.ansatz.anoGanInputs = [np.array([[1]])]
        self.ansatz.checkAnsatz()

    def getGenerator(self, bases):
        # pennylane expects a qnode for the quantum neural network layer
        # the qnode is defined as a decorated function as below
        @qml.qnode(self.device, interface="tf")
        def circuit(inputs, weights):
            self.circuitObject.initializeQubits(inputs)
            self.circuitObject.buildCircuit(weights)
            return self.circuitObject.measureZ()

        circuitInputParam = tf.keras.Input(
            shape=self.circuitObject.numQubits, name="circuitInputParam"
        )

        sampler = qml.qnn.KerasLayer(
            circuit,
            {"weights": self.circuitObject.numVariables},
            self.circuitObject.numQubits,
            dtype=tf.dtypes.float32,
        )
        expectation = sampler(circuitInputParam)

        generatedSample = tf.keras.layers.Dense(
            self.num_features, activation="sigmoid", name="postProcessing2"
        )(expectation)

        generator = tf.keras.Model(inputs=[circuitInputParam], outputs=generatedSample)
        return generator

    def getAnoGan(self, generator, discriminator, anoGan_disc_weight=1):
        # anoGan
        oneInput = tf.keras.layers.Input(shape=1, name="oneInput")
        adjustInput = tf.keras.layers.Dense(
            self.latent_dim, activation="sigmoid", name="adjustInput", use_bias=False
        )(oneInput)
        genOutput = generator(adjustInput)
        discOutput = discriminator(genOutput)

        inverse_weight = 1.0 / anoGan_disc_weight
        return tf.keras.Model(
            inputs=[oneInput],
            outputs=[inverse_weight * genOutput, anoGan_disc_weight * discOutput],
        )


class PennylaneIbmQ(PennylaneSimulator):
    """Run the Pennylane QWGan with the IBM Quantum backend
    """

    def __init__(self, n_steps=1200, bases=None):
        super().__init__(n_steps=n_steps, bases=bases)
        # TODO
        self.device = qml.device("default.qubit", wires=self.latent_dim)
