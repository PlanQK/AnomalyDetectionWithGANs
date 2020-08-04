import numpy as np
import sklearn
import skopt
from .ThresholdWrapper import ThresholdWrapper
import tensorflow as tf
import tensorflow.keras.backend as K


class AnoGanCost:
    def __init__(self, ansatz, **kwargs):
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []

    def buildAnoGan(self, opt):
        print("Starting threshold optimization")
        anoWGan = AnoWGan(self.ansatz, opt)
        anoWGan = ThresholdWrapper(anoWGan)

        X, Y = self.ansatz.trainingDataSampler()

        SPACE = [skopt.space.Real(0.5, 3.2, name="thr")]

        @skopt.utils.use_named_args(SPACE)
        def objective(thr):
            anoWGan.threshold = thr
            predicted = anoWGan.predict(X)
            return -sklearn.metrics.f1_score(Y.to_numpy(), predicted)

        results = skopt.forest_minimize(
            objective, SPACE, n_calls=10, n_random_starts=10
        )
        anoWGan._threshold = results.x[0]
        print(f"Threshold {anoWGan.threshold}")
        print("finished threshold optimization")
        return anoWGan

    def calculateMetrics(self, opt):
        X, Y = self.ansatz.getTestSample()
        model = self.buildAnoGan(opt)
        prediction = model.predict(X)

        return {
            "false positive": AnoGanCost.getFalsePositiveRate(Y, prediction),
            "false negative": AnoGanCost.getFalseNegativeRate(Y, prediction),
            "precision": sklearn.metrics.precision_score(Y, prediction),
            "recall": sklearn.metrics.recall_score(Y, prediction),
            "average precision": sklearn.metrics.average_precision_score(Y, prediction),
            "f1 score": sklearn.metrics.f1_score(Y, prediction),
        }

    @staticmethod
    def getFalsePositiveRate(Y, prediction):
        testingsetNegatives = len(Y == 0)
        unique, counts = np.unique(Y.values - prediction, return_counts=True)
        classificationInfo = dict(zip(unique, counts))
        try:
            return float(classificationInfo.get(-1, 0)) / testingsetNegatives
        except:
            return float("nan")

    @staticmethod
    def getFalseNegativeRate(Y, prediction):
        testingsetPositives = len(Y == 1)
        unique, counts = np.unique(Y.values - prediction, return_counts=True)
        classificationInfo = dict(zip(unique, counts))
        try:
            return float(classificationInfo.get(1, 0)) / testingsetPositives
        except:
            return float("nan")


class AnoWGan:
    """
    Helper class that makes out of a trained GAN an AnoGAN.
    It relies on the correct definition of the ansatz.
    """

    def __init__(self, ansatz, opt):
        self.curInput = None
        # since input layer can't be trained: create a new network that has an additional layer directly after the input
        # only this layer gets trained
        self.ansatz = ansatz
        self.network = ansatz.anoGanModel
        self.ansatz.discriminator.trainable = False
        self.ansatz.generator.trainable = False
        self.network.compile(optimizer=opt, loss=AnoWGan.loss)

    def predict(self, inputSamples, iterations=20):
        result = []
        for singleInputSample in inputSamples:
            singleInputSample = np.array([singleInputSample])
            discriminatorOutput = self.ansatz.discriminator.predict(singleInputSample)
            lossValue = self.network.fit(
                self.ansatz.anoGanInputs,
                [singleInputSample, discriminatorOutput],
                batch_size=1,
                epochs=iterations,
                verbose=0,
            )
            result.append(lossValue.history["loss"][-1])
        return result

    @staticmethod
    def loss(yTrue, yPred):
        return K.sum(K.abs(yTrue - yPred))

    def continueTraining(self):
        self.ansatz.discriminator.trainable = True
        self.ansatz.generator.trainable = True

    def __del__(self):
        self.continueTraining()
