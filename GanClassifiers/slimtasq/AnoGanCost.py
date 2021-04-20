import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from .EnvironmentVariableManager import EnvironmentVariableManager

class AnoGanCost:
    """Calculate the costs associated with anomaly detection for the AnoGan architecture
    Specifically these are:
        - false positive rate
        - false negative rate
        - precision
        - accuracy
        - f1 score
    """

    def __init__(self, ansatz, **kwargs):
        """Create the cost object.

        Args:
            ansatz (AnoGanAnsatz): This object contains the required structure for the AnoGan ansatz.
        """
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []

    def buildAnoGan(self, opt):
        """Create the AnoGan classifier (wrapped in a threshold object) and optimize the threshold for the outlier score.

        Args:
            opt (keras.optimizer): A optimizer such as tf.keras.Adam that is used for the creation of the AnoGan class.

        Returns:
            ThresholdWrapper: Optimized AnoGan object, ready for anomaly detection.
        """
        return AnoWGan(self.ansatz, opt)


class AnoWGan:
    """
    Helper class that turns a trained GAN into an AnoGAN.
    It relies on the correct definition of the ansatz.
    """

    def __init__(self, ansatz, opt):
        """Create the AnoGan object. This object contains the complete Anomaly detection algorithm.
        Given that all elements in ansatz are properly defined and trained beforehand.

        Args:
            ansatz (AnoGanAnsatz): Ansatz containing the structure of the Generator and Discriminator. The inputs are also specified.
            opt (tf.keras.optimizer): Optimizer for finding the latent vector that results in the closest generated sample.
        """
        self.curInput = None
        # since input layer can't be trained: create a new network that has an additional layer directly after the input
        # only this layer gets trained
        self.ansatz = ansatz
        self.network = ansatz.anoGanModel
        self.ansatz.discriminator.trainable = False
        self.ansatz.generator.trainable = False
        self.network.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=self.loss)
        self.anoganWeights = self.ansatz.anoGanModel.get_weights()
        self.initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
        self.numberRandomGuesses = int(EnvironmentVariableManager()["latentVarRandomGuesses"])

    def initializeAnomalyRun(self, inputSample):
        """We reinitialize the network and perform several random guesses
        to find the random starting values that result in the smallest
        distance to the input sample.
        """
        #first reset the network
        self.ansatz.anoGanModel.set_weights(self.anoganWeights)
        # pick the best set of latent variables from a set of uniformly
        # distributed random variables
        distances = []
        proposedWeights = []
        for i in range(self.numberRandomGuesses):
            proposedWeights.append(self.initializer(self.network.layers[1].get_weights()[0].shape))
            self.network.layers[1].set_weights([proposedWeights[-1],])
            distances.append(np.sum(np.abs(inputSample - self.network.predict(self.ansatz.anoGanInputs,)[0])))
        self.network.layers[1].set_weights([proposedWeights[np.argmin(distances)],])
        

    def predict(self, inputSamples, iterations=20):
        """Calculate the outlier score for a list of input samples.

        Args:
            inputSamples (list): a list of input samples, which need to be investigated for outliers
            iterations (int, optional): Optimization steps for finding the best latent variable. Defaults to 20.

        Returns:
            list: a list of results with one outlier score for each input sample
        """
        result = []
        for singleInputSample in inputSamples:
            self.initializeAnomalyRun(singleInputSample)
            singleInputSample = np.array([singleInputSample])
            discriminatorOutput = self.ansatz.discriminator.predict(singleInputSample)
            lossValue = self.network.fit(
                self.ansatz.anoGanInputs,
                [
                    singleInputSample / self.ansatz.discriminatorWeight,
                    discriminatorOutput * self.ansatz.discriminatorWeight,
                ],
                batch_size=1,
                epochs=iterations,
                initial_epoch=0,
                verbose=0,
            )
            result.append(lossValue.history["loss"][-1])
        return result

    def loss(self, yTrue, yPred):
        """Calculate the distance between generated and real sample.

        Args:
            yTrue (tf.vector): vector resulting from real sample
            yPred (tf.vector): vector resulting from generated sample

        Returns:
            float: distance between generated and real sample
        """
        # cost for too high and low weights we want to keep them in the interval [0, 1]
        weights = self.network.layers[1].weights[0]
        offset = K.sum(K.exp(tf.ones(weights.shape, dtype=tf.dtypes.float64)))
        weights = K.square(weights*2 - 1)
        weightCost = K.sum(K.exp(tf.clip_by_value(weights, 1, 10000)))
        return K.sum(K.abs(yTrue - yPred)) + weightCost - offset

    def continueTraining(self):
        """If the network is trained afterwards again, set the trainability flag back to true.
        """
        self.ansatz.discriminator.trainable = True
        self.ansatz.generator.trainable = True

    def __del__(self):
        """When the object is deleted make sure that the elements in the ansatz can be trained.
        """
        self.continueTraining()
