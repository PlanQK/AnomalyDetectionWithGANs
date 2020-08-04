import json
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
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
from .CircHelper import littleEntanglement


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
    def __init__(self, n_steps=1200):
        self.anoGan = None
        self.num_features = get_feature_length()

        # only allow the number of qubits to be between 1-9
        self.latent_dim = max(1, int(self.num_features / 3))
        self.latent_dim = min(9, self.latent_dim)
        self.n_steps = n_steps

    def getDiscriminator(self):
        discSeq = tf.keras.Sequential()
        discSeq.add(tf.keras.layers.Dense(self.num_features))
        discSeq.add(tf.keras.layers.Dropout(0.25))
        discSeq.add(tf.keras.layers.Dense(max(1, int(self.num_features / 2))))
        discSeq.add(tf.keras.layers.Dense(self.latent_dim))

        discInput = tf.keras.layers.Input(shape=(self.num_features))
        features = discSeq(discInput)
        valid = tf.keras.layers.Dense(1)(features)
        return tf.keras.Model(discInput, valid)

    def train(self):
        self.opt.run(self.cost)
        # now optimize threshold
        self.anoGan = self.cost.buildAnoGan(self.opt.opt)

    def predict(self):
        X = load_prediction_set_no_labels()
        return self.anoGan.predict(X)

    @staticmethod
    def loadClassifier():
        data = {}
        qc = QuantumClassifier()
        qc.ansatz.generator.load_weights("model/generator_weights")
        qc.ansatz.discriminator.load_weights("model/discriminator_weights")
        qc.ansatz.anoGanModel.load_weights("model/anoGan_weights")
        with open("model/other_parameters") as json_file:
            data = json.load(json_file)
            threshold = data["threshold"]

        qc.anoGan = AnoWGan(qc.ansatz, qc.opt.opt)
        qc.anoGan = ThresholdWrapper(qc.anoGan)
        qc.anoGan._threshold = threshold
        return qc

    def save(self):
        data = {"threshold": self.anoGan._threshold}
        self.ansatz.generator.save_weights("model/generator_weights")
        self.ansatz.discriminator.save_weights("model/discriminator_weights")
        self.ansatz.anoGanModel.save_weights("model/anoGan_weights")
        with open("model/other_parameters", "w") as json_file:
            json.dump(data, json_file)


class QuantumClassifier(Classifier):
    def __init__(self, n_steps=1200):
        super().__init__(n_steps)
        self.ansatz = AnoGanAnsatz("Quantum classifier")
        self.ansatz.generator = self.getGenerator()
        self.ansatz.discriminator = self.getDiscriminator()
        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSamplerTFQ(self.latent_dim)
        self.ansatz.getTestSample = LabelSampler()
        self.ansatz.anoGanModel = self.getAnoGan(
            self.ansatz.generator, self.ansatz.discriminator
        )
        self.ansatz.anoGanInputs = [
            tfq.convert_to_tensor([cirq.Circuit()]),
            np.array([[1]]),
        ]
        self.ansatz.trainingDataSampler = LabelSampler()
        self.ansatz.checkAnsatz()

        self.cost = AnoGanCost(self.ansatz)
        self.opt = WGanOptimization(
            tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
            "WGAN",
            n_steps=n_steps,
            updateInterval=n_steps + 10,
        )

    def getGenerator(self):
        rpc = littleEntanglement(self.latent_dim, 1, 1)
        circuit = rpc.buildCircuit()
        # build generator
        circuitInput = tf.keras.Input(shape=(), dtype=tf.string, name="circuitInput")
        circuitInputParam = tf.keras.Input(
            shape=len(rpc.inputParams), name="circuitInputParam"
        )
        paramsInput = tf.keras.Input(shape=1, name="paramsInput")
        paramsInput2_layer = tf.keras.layers.Dense(
            np.prod(rpc.controlParams.shape),
            input_shape=(1,),
            activation="sigmoid",
            name="paramsInputDense",
        )

        paramsInput2 = paramsInput2_layer(paramsInput)

        sampler = tfq.layers.ControlledPQC(circuit, rpc.getReadOut())
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
                np.array([rpc.generateInitialParameters()]),
                np.zeros((np.prod(rpc.controlParams.shape),)),
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
