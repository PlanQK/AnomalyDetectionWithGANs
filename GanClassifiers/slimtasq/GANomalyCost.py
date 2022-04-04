import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tqdm import tqdm
import skopt
from .EnvironmentVariableManager import EnvironmentVariableManager

class GANomalyCost:
    """Calculate the costs associated with anomaly detection for the GANomaly architecture
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
            ansatz (GANomalyAnsatz): This object contains the required structure for the AnoGan ansatz.
        """
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []
        self.max_validation_samples = 100 # todo: add this parameter to the config file/standard params

    def buildAnoGan(self, opt):
        """Create the AnoGan classifier (wrapped in a threshold object) and optimize the threshold for the outlier score.

        Args:
            opt (keras.optimizer): A optimizer such as tf.keras.Adam that is used for the creation of the AnoGan class.

        Returns:
            ThresholdWrapper: Optimized AnoGan object, ready for anomaly detection.
        """
        pass
        #return AnoWGan(self.ansatz, opt)

    def calculateMetrics(self, step=-1, validation_or_test="validation"):
        """Calculate the metrics on the validation dataset.

        Args:
            opt (tf.keras.optimizer): [unused] Optimizer required for
                the AnoGan architecture e.g. Adam optimizer.

        Returns:
            dict: dict containing the results for the different metrics.
        """
        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        if validation_or_test == "validation":
            x_normal, x_unnormal = self.ansatz.validationSampler(batchSize=self.max_validation_samples)
        elif validation_or_test == "test":
            x_normal, x_unnormal = self.ansatz.testSampler()

        # normal error
        z_normal = self.ansatz.auto_encoder(
            x_normal, training=False
        )
        x_hat_normal = self.ansatz.auto_decoder(
            z_normal, training=False
        )
        z_hat_normal = self.ansatz.encoder(
            x_hat_normal, training=False
        )
        enc_loss_normal = mae(z_normal, z_hat_normal)

        # unnormal error
        z_unnormal = self.ansatz.auto_encoder(
            x_unnormal, training=False
        )
        x_hat_unnormal = self.ansatz.auto_decoder(
            z_unnormal, training=False
        )
        z_hat_unnormal = self.ansatz.encoder(
            x_hat_unnormal, training=False
        )
        enc_loss_unnormal = mae(z_unnormal, z_hat_unnormal)

        sample_num = 4 if len(x_normal) >= 4 and len(x_unnormal) >= 4 else min(len(x_normal), len(x_unnormal))
        x_normal_samples = x_normal[:sample_num]
        x_hat_normal_samples = x_hat_normal.numpy()[:sample_num]
        x_unnormal_samples = x_unnormal[:sample_num]
        x_hat_unnormal_samples = x_hat_unnormal.numpy()[:sample_num]

        # rescaling
        maximum = max(np.max(enc_loss_normal), np.max(enc_loss_unnormal))
        minimum = min(np.min(enc_loss_normal), np.min(enc_loss_unnormal))
        enc_loss_normal = (enc_loss_normal-minimum)/(maximum-minimum)
        enc_loss_unnormal = (enc_loss_unnormal-minimum)/(maximum-minimum)

        res = self.optimize_scaled_anomaly_score(enc_loss_normal.numpy(), enc_loss_unnormal.numpy())

        if validation_or_test == "validation":
            res["x_normal_samples"] = x_normal_samples
            res["x_hat_normal_samples"] = x_hat_normal_samples
            res["x_unnormal_samples"] = x_unnormal_samples
            res["x_hat_unnormal_samples"] = x_hat_unnormal_samples
            if self.ansatz.best_mcc < res["MCC"]:
                self.ansatz.best_mcc = res["MCC"]
                self.ansatz.save(step=step, MCC=res["MCC"])
                self.ansatz.save()
                print("\nModel with new highscore saved!")
        elif validation_or_test == "test":
            res["TP"] = [res["TP"]]
            res["FP"] = [res["FP"]]
            res["TN"] = [res["TN"]]
            res["FN"] = [res["FN"]]

        return res

    @staticmethod
    def optimize_scaled_anomaly_score(enc_loss_normal, enc_loss_unnormal):
        """
        Args:
            enc_loss_normal: np.array of shape (len(enc_loss_normal)) holding the scaled anomaly scores for each sample
            enc_loss_unnormal: np.array of shape (len(enc_loss_unnormal)) holding the scaled anomaly scores for each sample
        """
        prepare_normal = np.dstack((enc_loss_normal, -np.ones_like(enc_loss_normal)))[0]
        prepare_unnormal = np.dstack((enc_loss_unnormal, np.ones_like(enc_loss_unnormal)))[0]
        complete_set = np.vstack((prepare_normal, prepare_unnormal)).tolist()
        sorted_complete_set = sorted(complete_set, key=lambda x:x[0])

        epsilon = 10**(-7)
        best_threshold = 0.
        best_index = 0
        optimizer_score = -len(sorted_complete_set)
        sorted_true_labels = [sorted_complete_set[0][1]]
        sorted_complete_set[0][1] = int(sorted_complete_set[0][1])
        for i in range(1, len(sorted_complete_set)):
            sorted_complete_set[i][1] = int(sorted_complete_set[i][1])
            sorted_true_labels.append(sorted_complete_set[i][1])
            new_score = np.sum(sorted_complete_set[i:], axis=0)[1] - np.sum(sorted_complete_set[:i+1], axis=0)[1]
            if optimizer_score < new_score:
                best_threshold = sorted_complete_set[i][0] - epsilon
                best_index = i
                optimizer_score = new_score

        if best_index == 0:
            labels_below_threshold = sorted_true_labels
            labels_above_threshold = []
        elif best_index == len(sorted_true_labels) - 1:
            labels_below_threshold = []
            labels_above_threshold = sorted_true_labels
        else:
            labels_below_threshold = sorted_true_labels[:best_index+1]
            labels_above_threshold = sorted_true_labels[best_index+1:]

        unique, counts = np.unique(labels_below_threshold, return_counts=True)
        distribution_below_threshold = dict(zip(unique, counts))
        unique, counts = np.unique(labels_above_threshold, return_counts=True)
        distribution_above_threshold = dict(zip(unique, counts))

        distribution_below_threshold.setdefault(-1, 0)
        distribution_below_threshold.setdefault(1, 0)
        distribution_above_threshold.setdefault(-1, 0)
        distribution_above_threshold.setdefault(1, 0)

        result = {}
        result["TP"] = distribution_above_threshold[1]
        result["FP"] = distribution_above_threshold[-1]
        result["TN"] = distribution_below_threshold[-1]
        result["FN"] = distribution_below_threshold[1]
        result["best_threshold"] = best_threshold
        result["MCC"] = (result["TP"] * result["TN"] - result["FP"] * result["FN"]) / (
                    (result["TP"] + result["FP"]) * (result["TP"] + result["FN"]) * (result["TN"] + result["FP"]) * (
                        result["TN"] + result["FN"])) ** (1 / 2)

        return result

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
        envMgr = EnvironmentVariableManager()
        self.ansatz = ansatz
        self.network = ansatz.anoGanModel
        self.ansatz.discriminator.trainable = False
        self.ansatz.generator.trainable = False
        self.network.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=self.loss)
        self.anoganWeights = self.ansatz.anoGanModel.get_weights()
        self.initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
        self.optimizer = envMgr["latentVariableOptimizer"]

    def initializeAnomalyRun(self, inputSample):
        """We reinitialize the network and perform several random guesses
        to find the random starting values that result in the smallest
        distance to the input sample.
        """
        #first reset the network
        self.ansatz.anoGanModel.set_weights(self.anoganWeights)

    def optimizeWithTF(self, discriminatorOutput, singleInputSample, iterations):
        # pick the best set of latent variables from a set of uniformly
        # distributed random variables
        distances = []
        proposedWeights = []
        for i in range(10):
            proposedWeights.append(self.initializer(self.network.get_layer(name="adjustInput").get_weights()[0].shape))
            self.network.get_layer(name="adjustInput").set_weights([proposedWeights[-1],])
            distances.append(np.sum(np.abs(singleInputSample - self.network.predict(self.ansatz.anoGanInputs,)[0])))
        self.network.get_layer(name="adjustInput").set_weights([proposedWeights[np.argmin(distances)],])

        # now start the optimization
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

        return lossValue.history["loss"][-1]

    def optimizeWithForrestMinimize(self, discriminatorOutput, singleInputSample, iterations):
        def loss(latentVars):
            latentVars = np.array(latentVars).reshape(
                self.network.get_layer(name="adjustInput").get_weights()[0].shape
            )
            self.network.get_layer(name="adjustInput").set_weights([latentVars, ])
            yPred = self.network.predict(self.ansatz.anoGanInputs,)
            nonlocal singleInputSample
            nonlocal discriminatorOutput
            residualCost = K.sum(K.abs(singleInputSample - yPred[0]))
            discriminatorCost = K.sum(K.abs(discriminatorOutput - yPred[1]))
            return float(
                residualCost / self.ansatz.discriminatorWeight +
                discriminatorCost * self.ansatz.discriminatorWeight
            )
        weights = self.network.get_layer(name="adjustInput").get_weights()[0]
        dimensions = [(0.0,1.0)] * weights.shape[1]
        result = skopt.forest_minimize(loss, dimensions, n_points=200, n_calls=iterations)
        return loss(result.x)

    def optimizedCost(self, discriminatorOutput, singleInputSample, iterations):
        if self.optimizer == "TF":
            return self.optimizeWithTF(discriminatorOutput, singleInputSample, iterations)
        if self.optimizer == "forest_minimize":
            return self.optimizeWithForrestMinimize(discriminatorOutput, singleInputSample, iterations)
        raise NotImplementedError("The optimizer you chose is not specified")

    def predict(self, inputSamples, iterations=20):
        """Calculate the outlier score for a list of input samples.

        Args:
            inputSamples (list): a list of input samples, which need to be investigated for outliers
            iterations (int, optional): Optimization steps for finding the best latent variable. Defaults to 20.

        Returns:
            list: a list of results with one outlier score for each input sample
        """
        result = []
        for singleInputSample in tqdm.tqdm(inputSamples):
            self.initializeAnomalyRun(singleInputSample)
            singleInputSample = np.array([singleInputSample])
            discriminatorOutput = self.ansatz.discriminator.predict(singleInputSample)
            result.append(self.optimizedCost(discriminatorOutput, singleInputSample, iterations))
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
        weights = self.network.get_layer(name="adjustInput").weights[0]
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
