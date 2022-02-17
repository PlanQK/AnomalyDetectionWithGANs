"""This file contains different classes for the classification.

One for each configuration. The main classes currently are:
    - ClassicalClassifier: The purely classical AnoGan
    - TfqSimulator: a simulation reliant on the tensorflow quantum framework
    - PennylaneSimulator: a simulation with the pennylane framework
    - PennylaneIbmQ: calculation on the IBM Quantum computer with the
                        Pennylane framework

Internally, each classifier contains several instances from various
classes. Their design comes from the tasq framework, which divides the
problem into Optimization, Cost, and Ansatz. It was initially developed
for optimization problems, but now we use it for AI as well.

Optimization object: Performs Training of the GAN model
Cost object: Calculation of metrics, e.g. F1-Score
Ansatz: Combination of all ingredients: optimization, cost, training data etc.
"""
import json

import cirq
import numpy as np
import pennylane as qml
import tensorflow as tf
import tensorflow_quantum as tfq
from cirq_rigetti import circuit_sweep_executors, RigettiQCSSampler, ExecutionCounter
from pyquil import get_qc

from . import CircHelper
from . import PennylaneHelper
from .DataIO import (
    get_feature_length,
    NoLabelSampler,
    load_prediction_set_no_labels,
)
from .slimtasq import (
    AnoGanAnsatz,
    WGanOptimization,
    AnoGanCost,
    AnoWGan,
    EnvironmentVariableManager,
)


class generatorInputSampler:
    """Class to generate uniformly distributed latent variables.

    These are used by the generator to create new samples.
    """

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def __call__(self, batchSize):
        return np.array(
            [
                np.random.uniform(0, 1, self.latent_dim).astype(np.float64)
                for i in range(batchSize)
            ]
        )


class generatorInputSamplerTFQ(generatorInputSampler):
    """Generate the input to the TFQ Generator.

    This consists of the same uniformly distributed latent variables as in
    the generatorInputSampler. TFQ additionally requires the input
    state (empty circuit). The helper input (1) is another workaround
    to be able to optimize the weights of the TFQ layer (variational
    parameters).
    """

    def __call__(self, batchSize):
        return [
            tfq.convert_to_tensor([cirq.Circuit()] * batchSize),
            super().__call__(batchSize),
            np.array([[1]] * batchSize).astype(np.float64),
        ]


class Classifier:
    """Base class for the different Gan Classifiers.

    Ideally, only the getGenerator and getAnoGan methods need to be
    specified to obtain a new classifier.
    """

    def __init__(self, bases=None):
        """Instantiate all the dependent classes for the AnoGan method.
        """
        tf.keras.backend.set_floatx("float64")
        self.anoGan = None
        self.num_features = get_feature_length()
        self.envMgr = EnvironmentVariableManager()
        self.latent_dim = int(self.envMgr["latentDim"])
        self.totalNumCycles = int(self.envMgr["totalDepth"])

        self.opt = WGanOptimization(
            tf.keras.optimizers.Adam(float(self.envMgr["adamTrainingRate"]), beta_1=0.5),
            "WGAN",
            n_steps=int(self.envMgr["trainingSteps"]),
            updateInterval=int(self.envMgr["trainingSteps"]) + 10,
            batchSize=int(self.envMgr["batchSize"]),
            discriminatorIterations=int(self.envMgr["discriminatorIterations"]),
            gpWeight=float(self.envMgr["gpWeight"]),
        )

        self.ansatz = AnoGanAnsatz(self.__class__.__name__)
        self.execution_count_rigetti = ExecutionCounter()
        self.ansatz.generator = self.getGenerator(bases)
        self.ansatz.discriminator = self.getDiscriminator()
        self.ansatz.anoGanModel = self.getAnoGan(
            self.ansatz.generator, self.ansatz.discriminator
        )

    def getDiscriminator(self):
        """Return the Tensorflow model for the Discriminator.
        For comparability this should be the same for the different approaches.
        """
        discSeq = tf.keras.Sequential()
        discSeq.add(tf.keras.layers.Dense(self.num_features))
        discSeq.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        discSeq.add(tf.keras.layers.Dense(max(1, int(self.num_features / 2))))
        discSeq.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        discSeq.add(tf.keras.layers.Dense(max(1, int(self.num_features / 2))))

        discInput = tf.keras.layers.Input(
            shape=(self.num_features), name="DiscInput"
        )
        features = discSeq(discInput)
        valid = tf.keras.layers.Dense(1)(features)
        return tf.keras.Model(discInput, valid)

    def getGenerator(self, bases):
        """Return the Tensorflow model for the generator.

        Depending on the classifier type this includes the circuit
        specification for the quantum generator
        Args:
            bases ([str]): The bases of the rotational gates. If
                    these are not specified random bases are chosen.
        """
        raise NotImplementedError("Need to use the derived class")

    def getAnoGan(self, generator, discriminator):
        """Generate the Tensorflow model for the anogan method.

        This combines both the generator and discriminator into a
        complete classifier.

        Args:
            generator ([type]): Tensorflow model for the generator
            discriminator ([type]): Tensorflow model for the discriminator
        """
        raise NotImplementedError("Need to use the derived class")

    def train(self):
        """Perform the training on the defined gan model."""
        self.opt.run(self.cost)
        self.anoGan = self.cost.buildAnoGan(self.opt.opt)

    def predict(self):
        """Perform the prediction on the defined gan model."""
        X = load_prediction_set_no_labels()
        predictions = self.anoGan.predict(
            X.to_numpy(),
            int(
                EnvironmentVariableManager()[
                    "latentVariableOptimizationIterations"
                ]
            )
        )
        return predictions

    @classmethod
    def loadClassifier(cls):
        """Load a previously trained classifier from files."""
        data = {}
        with open(f"model/checkpoint/{cls.__name__}_other_parameters") as json_file:
            data = json.load(json_file)
        qc = cls(bases=data["bases"])
        qc.latent_dim = data["latent_dim"]
        qc.ansatz.generator.load_weights(
            f"model/checkpoint/{cls.__name__}_generator_weights"
        )
        qc.ansatz.discriminator.load_weights(
            f"model/checkpoint/{cls.__name__}_discriminator_weights"
        )
        qc.ansatz.anoGanModel.load_weights(
            f"model/checkpoint/{cls.__name__}_anoGan_weights"
        ).expect_partial()

        qc.anoGan = AnoWGan(qc.ansatz, qc.opt.opt)
        return qc

    def save(self):
        """Store the trained weights and parameters."""
        data = {
            "latent_dim": self.latent_dim,
            "bases": self.circuitObject.getBases(),
        }
        self.ansatz.generator.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_generator_weights"
        )
        self.ansatz.discriminator.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_discriminator_weights"
        )
        self.ansatz.anoGanModel.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_anoGan_weights"
        )
        with open(
            f"model/checkpoint/{self.__class__.__name__}_other_parameters", "w"
        ) as json_file:
            json.dump(data, json_file)


class ClassicalClassifier(Classifier):
    """Use a purely classical GAN.

    This class generates a classical generator that contains
    dense layers with batch renormalization.
    """

    def __init__(self, bases=None):
        super().__init__(bases=bases)

        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSampler(
            self.latent_dim
        )
        self.ansatz.anoGanInputs = [np.array([[1]])]
        self.cost = AnoGanCost(self.ansatz)

    def getGenerator(self, _):
        """The classical generator consists of several dense layers,
        and
        """
        genSeq = tf.keras.Sequential()
        noise = tf.keras.layers.Input(shape=(self.latent_dim))
        for layer in range(self.totalNumCycles):
            genSeq.add(tf.keras.layers.Dense(9, input_dim=self.latent_dim))
            genSeq.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            genSeq.add(tf.keras.layers.BatchNormalization(momentum=0.8))

        genSeq.add(
            tf.keras.layers.Dense(self.num_features, activation="sigmoid")
        )
        return tf.keras.Model(noise, genSeq(noise))

    def getAnoGan(self, generator, discriminator, anoGan_disc_weight=1):
        # anoGan
        oneInput = tf.keras.layers.Input(shape=1, name="oneInput")
        adjustInput = tf.keras.layers.Dense(
            self.latent_dim,
            activation="sigmoid",
            name="adjustInput",
            use_bias=False,
        )(oneInput)
        genOutput = generator(adjustInput)
        discOutput = discriminator(genOutput)

        inverse_weight = 1.0 / anoGan_disc_weight
        return tf.keras.Model(
            inputs=[oneInput],
            outputs=[
                inverse_weight * genOutput,
                anoGan_disc_weight * discOutput,
            ],
        )

    @classmethod
    def loadClassifier(cls):
        """Load a previously trained classifier from files."""
        data = {}
        with open(f"model/checkpoint/{cls.__name__}_other_parameters") as json_file:
            data = json.load(json_file)
        qc = cls(bases=None)
        qc.latent_dim = data["latent_dim"]
        qc.ansatz.generator.load_weights(
            f"model/checkpoint/{cls.__name__}_generator_weights"
        )
        qc.ansatz.discriminator.load_weights(
            f"model/checkpoint/{cls.__name__}_discriminator_weights"
        )
        qc.ansatz.anoGanModel.load_weights(
            f"model/checkpoint/{cls.__name__}_anoGan_weights"
        ).expect_partial()

        qc.anoGan = AnoWGan(qc.ansatz, qc.opt.opt)
        return qc

    def save(self):
        data = {"latent_dim": self.latent_dim}

        self.ansatz.generator.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_generator_weights"
        )
        self.ansatz.discriminator.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_discriminator_weights"
        )
        self.ansatz.anoGanModel.save_weights(
            f"model/checkpoint/{self.__class__.__name__}_anoGan_weights"
        )
        with open(
            f"model/checkpoint/{self.__class__.__name__}_other_parameters", "w"
        ) as json_file:
            json.dump(data, json_file)


class TfqSimulator(Classifier):
    def __init__(self, bases=None):
        super().__init__(bases=bases)
        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSamplerTFQ(
            self.latent_dim
        )
        self.ansatz.anoGanInputs = [
            tfq.convert_to_tensor([cirq.Circuit()]),
            np.array([[1]]),
        ]
        self.cost = AnoGanCost(self.ansatz)

    def getGenerator(self, bases):
        self.circuitObject = CircHelper.LittleEntanglementIdentity(
            self.latent_dim, 1, self.totalNumCycles
        )
        if bases:
            self.circuitObject.setBases(bases)
        circuit = self.circuitObject.buildCircuit()
        # build generator
        circuitInput = tf.keras.Input(
            shape=(), dtype=tf.string, name="circuitInput"
        )
        circuitInputParam = tf.keras.Input(
            shape=len(self.circuitObject.inputParams),
            name="circuitInputParam",
            dtype="float64",
        )
        paramsInput = tf.keras.Input(
            shape=1, name="paramsInput", dtype="float64"
        )
        paramsInput2_layer = tf.keras.layers.Dense(
            np.prod(self.circuitObject.controlParams.shape),
            input_shape=(1,),
            activation="sigmoid",
            name="paramsInputDense",
            dtype="float64",
        )

        paramsInput2 = paramsInput2_layer(paramsInput)

        if self.envMgr["backend"] == "rigetti":
            executor = circuit_sweep_executors.with_quilc_parametric_compilation
            qc = get_qc(
                'Aspen-11',
                as_qvm=True,
                noisy=False,
                compiler_timeout=100000
            )

            Rigetti_Sampler = RigettiQCSSampler(
                quantum_computer=qc,
                executor=executor,
                execution_counter=self.execution_count_rigetti
            )

            sampler = tfq.layers.ControlledPQC(
                circuit, self.circuitObject.getReadOut(), dtype="float32", backend=Rigetti_Sampler, repetitions=1000
            )
        else:
            sampler = tfq.layers.ControlledPQC(
                circuit, self.circuitObject.getReadOut(), dtype="float32"
            )

        concat = tf.concat([circuitInputParam, paramsInput2], axis=1)
        expectation = sampler([circuitInput, concat])

        generatedSample = tf.keras.layers.Dense(
            self.num_features,
            activation="sigmoid",
            name="postProcessing2",
            dtype="float64",
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
        oneInput = tf.keras.layers.Input(
            shape=1, name="oneInput", dtype="float64"
        )
        adjustInput = tf.keras.layers.Dense(
            self.latent_dim,
            activation="sigmoid",
            name="adjustInput",
            use_bias=False,
            dtype="float64",
        )(oneInput)
        circuitInput2 = tf.keras.Input(
            shape=(), dtype=tf.string, name="circuitInput2"
        )
        genOutput = generator(inputs=[circuitInput2, adjustInput, oneInput])
        discOutput = discriminator(genOutput)

        inverse_weight = 1.0 / anoGan_disc_weight
        return tf.keras.Model(
            inputs=[circuitInput2, oneInput],
            outputs=[
                inverse_weight * genOutput,
                anoGan_disc_weight * discOutput,
            ],
        )


class PennylaneSimulator(Classifier):
    def __init__(self, bases=None):
        # copied code from the base init because we need it earlier
        self.num_features = get_feature_length()
        self.envMgr = EnvironmentVariableManager()
        self.latent_dim = int(self.envMgr["latentDim"])
        self.totalNumCycles = int(self.envMgr["totalDepth"])
        # Pennylane specifics
        self.device = qml.device("default.qubit", wires=self.latent_dim)
        #self.device = qml.device(
        #   'qiskit.ibmq',
        #   wires=self.latent_dim,
        #   backend='ibmq_16_melbourne',
        #   hub='MYHUB',  # optional
        #   group='MYGROUP',  # optional
        #   project='anoGan'  # optional
        # )

        self.circuitObject = PennylaneHelper.LittleEntanglementIdentity(
            self.latent_dim, self.totalNumCycles
        )

        super().__init__(bases=bases)

        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSampler(
            self.latent_dim
        )
        self.ansatz.anoGanInputs = [np.array([[1]])]
        self.cost = AnoGanCost(self.ansatz)

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
        )
        expectation = sampler(circuitInputParam)

        generatedSample = tf.keras.layers.Dense(
            self.num_features, activation="sigmoid", name="postProcessing2"
        )(expectation)

        generator = tf.keras.Model(
            inputs=[circuitInputParam], outputs=generatedSample
        )
        return generator

    def getAnoGan(self, generator, discriminator, anoGan_disc_weight=1):
        # anoGan
        oneInput = tf.keras.layers.Input(shape=1, name="oneInput")
        adjustInput = tf.keras.layers.Dense(
            self.latent_dim,
            activation="sigmoid",
            name="adjustInput",
            use_bias=False,
        )(oneInput)
        genOutput = generator(adjustInput)
        discOutput = discriminator(genOutput)

        inverse_weight = 1.0 / anoGan_disc_weight
        return tf.keras.Model(
            inputs=[oneInput],
            outputs=[
                inverse_weight * genOutput,
                anoGan_disc_weight * discOutput,
            ],
        )

from qiskit import IBMQ

class PennylaneIbmQ(PennylaneSimulator):
    """Run the Pennylane QWGan with the IBM Quantum backend
    """

    def __init__(self, bases=None):
        # copied code from the base init because we need it earlier
        self.num_features = get_feature_length()
        self.envMgr = EnvironmentVariableManager()
        # only allow the number of qubits to be between 1-9
        self.latent_dim = int(self.envMgr["latentDim"])
        self.totalNumCycles = int(self.envMgr["totalDepth"])
        # Pennylane specifics
        
        IBMQ.load_account()

        providers=IBMQ.providers()

        print(providers)

        provider=providers[-1]
        
        self.device = qml.device(
            "qiskit.ibmq",
            wires=self.latent_dim,
#            backend="ibmq_qasm_simulator",
            backend=self.envMgr["backend"],#"ibmq_16_melbourne",
            provider=provider,
#            ibmqx_token=self.envMgr["ibmqx_token"],
        )

        self.circuitObject = PennylaneHelper.LittleEntanglementIdentity(
            self.latent_dim, self.totalNumCycles
        )
        Classifier.__init__(self, bases=bases)

        self.ansatz.trueInputSampler = NoLabelSampler()
        self.ansatz.latentVariableSampler = generatorInputSampler(
            self.latent_dim
        )
        self.ansatz.anoGanInputs = [np.array([[1]])]
        self.cost = AnoGanCost(self.ansatz)
