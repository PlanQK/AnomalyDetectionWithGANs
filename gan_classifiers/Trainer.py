import time
import numpy as np

import cirq
import tensorflow as tf
from tqdm import tqdm
import tensorflow_quantum as tfq

from gan_classifiers.EnvironmentVariableManager import EnvironmentVariableManager

class Trainer:
    """
    Class holding the required methods and attributes to conduct the training or testing procedure.
    """

    def __init__(
            self,
            Data,
            Classifier
    ):
        """Initialize necessary parameters to train the model via a Wasserstein-Loss based GANomaly ansatz.
        
        This implementation is supposed to remedy some of the training difficulties of GANs (e.g.
        vanishing gradients).
        Args:
            Classifier: sub-classed Classifier-object defined in GANomalyNetworks.py.
                    opt_disc and opt_gen (tf.keras.optimizer, optional): Optimizer to perform gradient descent.
                    Defaults to Adam.
            n_steps (int, optional): Optimization steps. Defaults to 1000.
            batch_size (int, optional): Number of training samples for each
                    optimization step. Defaults to 32.
            discriminator_iterations (int, optional): How often the
                    discriminator is trained for each generator optimization.
                    Defaults to 5.
            updateInterval (int, optional): Determines how often model performance is checked against the validation
                    dataset. If the performance reaches a new highscore, the model is saved to the checkpoint-folder.
                    Defaults to 50.
            gradient_penalty_weight (float, optional): Weight of the Wasserstein-GAN penalty added to the discriminator loss.
            adv-, con-, enc_loss_weight (float/int, optional): Weights of the generator loss components adversarial,
                    contextual and encoding loss respectively.
        """
        tf.keras.backend.set_floatx("float64")
        self.envMgr = EnvironmentVariableManager()

        self.Data = Data
        self.Classifier = Classifier

        self.validation = True if self.envMgr["train_or_predict"] == "train" else False
        self.opt_disc = tf.keras.optimizers.Adam(beta_1=0.5, lr=float(self.envMgr["discriminator_training_rate"]))
        self.opt_gen = tf.keras.optimizers.Adam(beta_1=0.5, lr=float(self.envMgr["generator_training_rate"]))
        self.latent_dim = self.envMgr["latent_dimensions"]
        self.n_steps = self.envMgr["training_steps"]
        self.step_counter = 0
        self.batch_size = self.envMgr["batch_size"]
        self.discriminator_iterations = self.envMgr["discriminator_iterations"]
        self.updateInterval = self.envMgr["validation_interval"]
        self.validation_samples = min(int(self.envMgr["validation_samples"]), len(self.Data.test_data_normal),
                                     len(self.Data.validation_data_unnormal))
        self.gradient_penalty_weight = self.envMgr["gradient_penalty_weight"]
        self.adv_loss_weight = self.envMgr["adv_loss_weight"]
        self.con_loss_weight = self.envMgr["con_loss_weight"]
        self.enc_loss_weight = self.envMgr["enc_loss_weight"]
        self.g_loss = 0
        self.adv_loss = 0
        self.con_loss = 0
        self.enc_loss = 0
        self.d_loss = 0
        self.best_mcc = -1.
        self.train_hist = {"step_number": [], "total_runtime": 0, "runtime_per_step": 0, "generator_loss": [],
                           "adversarial_loss": [], "contextual_loss": [], "encoder_loss": [],
                           "discriminator_loss": [], "TP": [], "FP": [], "TN": [], "FN": [],
                           "threshold": [], "MCC": [], "x_normal_samples": [], "x_hat_normal_samples": [],
                           "x_unnormal_samples": [], "x_hat_unnormal_samples": []}
        self.quantum = False 

    def train(self):
        """Run the training procedure. The first validation and logging of losses happens after the validation interval
        steps. If it would start at step 0, the first major changes in losses and metrics would be protocolled, leading
        to significant overstretching in the plots created at the end of the training procedure.

        Returns:
            tuple(list, list): A tuple containing the costs snapshots and
                    parameter snapshots. The parameters are empty in this
                    implementation.
        """

        self.status = "running"

        tic = time.perf_counter()

        for _ in tqdm(range(int(self.n_steps)), desc=self.__str__()):
            self._step()
            self.step_counter += 1
            if (int(self.step_counter) % int(self.updateInterval)) == 0:
                metrics = self.calculateMetrics(step=self.step_counter)
                metrics["generator_loss"] = self.g_loss
                metrics["discriminator_loss"] = self.d_loss
                metrics["adversarial_loss"] = self.adv_loss
                metrics["contextual_loss"] = self.con_loss
                metrics["encoder_loss"] = self.enc_loss
                for key, value in metrics.items():
                    self.train_hist[key].append(value)
                self.train_hist["step_number"].append(self.step_counter)
                # FK: plotting L_adv, L_con and L_enc as well just for information. Remove if not wanted
                print(f"\nMCC: {self.train_hist['MCC'][-1]}, Generator loss: {self.g_loss}, Adversarial loss: {self.adv_loss}, Contextual loss: {self.con_loss}, Encoder loss: {self.enc_loss}, Discriminator loss: {self.d_loss}")
        toc = time.perf_counter()

        self.train_hist["total_runtime"] = toc - tic
        if self.n_steps != 0:
            self.train_hist["runtime_per_step"] = (toc - tic) / int(self.n_steps)
        self.status = "done"
        return self.train_hist

    def _step(self):
        """Perform a single step of the optimization.
        This step consists of several updates for the discriminator parameters
        and a single update for the generator parameters.
        """
        x = self.Data.get_train_data(self.batch_size)
        for i in range(int(self.discriminator_iterations)):
            with tf.GradientTape(persistent=True) as tape:
                z = self.Classifier.auto_encoder(
                    x, training=True
                )
                z_quantum = self.transform_z_to_z_quantum(z)
                x_hat = self.Classifier.auto_decoder(
                    z_quantum, training=True
                )
                z_hat = self.Classifier.encoder(
                    x_hat, training=True
                )
                d = self.Classifier.discriminator(
                    x, training=True
                )
                d_hat = self.Classifier.discriminator(
                    x_hat, training=True
                )

                # discriminator losses
                d_cost = self.discriminatorLoss(d, d_hat)
                gp = self.gradient_penalty(x, x_hat)
                d_loss = d_cost + gp * float(self.gradient_penalty_weight)

                # generator losses
                if i == (int(self.discriminator_iterations) - 1):
                    self.adv_loss, self.con_loss, self.enc_loss = self.generatorLoss(x, x_hat, z, z_hat, d, d_hat)
                    g_loss = self.adv_loss + self.con_loss + self.enc_loss
                    self.g_loss = float(g_loss)
                    self.adv_loss = float(self.adv_loss)
                    self.con_loss = float(self.con_loss)
                    self.enc_loss = float(self.enc_loss)
                    self.d_loss = float(d_loss)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, self.Classifier.discriminator.trainable_variables
            )
            # Update the weights of the discriminator
            self.opt_disc.apply_gradients(
                zip(d_gradient, self.Classifier.discriminator.trainable_variables)
            )

            # Update the weights of the generator
            if i == (int(self.discriminator_iterations) - 1):
                gen_gradient = tape.gradient(
                    g_loss,
                    self.Classifier.auto_encoder.trainable_variables + self.Classifier.auto_decoder.trainable_variables + self.Classifier.encoder.trainable_variables
                )
                self.opt_gen.apply_gradients(
                    zip(gen_gradient,
                        self.Classifier.auto_encoder.trainable_variables + self.Classifier.auto_decoder.trainable_variables + self.Classifier.encoder.trainable_variables)
                )

        return None

    def transform_z_to_z_quantum(self, z):
        # No transformation is done
        return z
        
    @staticmethod
    def discriminatorLoss(d, d_hat):
        """Calculate the loss for the discriminator optimization steps.

        Args:
            d (tf.tensor):
                     discriminator output for x as input
            d_hat (tf.tensor):
                    discriminator output for x_hat as input

        Returns:
            tf.tensor: value for the loss
        """
        bce = tf.losses.BinaryCrossentropy(from_logits=False)
        x_loss = bce(tf.zeros_like(d), d)
        x_hat_loss = bce(tf.ones_like(d_hat), d_hat)
        return (x_loss + x_hat_loss) / 2

    def gradient_penalty(self, x, x_hat):
        """Calculate the gradient penalty. This is an addition to the
        loss function to ensure lipschitz continuity. This loss contribution
        is calculated from an interpolated image between generated
        and real sample.

        Args:
            cost (GanCost): required for the GanAnsatz
            x (list): batch of normal input samples
            x_hat (list): batch of samples generated by the auto_encoder

        Returns:
            tf.tensor: gradient penalty contribution to the loss
        """
        # get the interpolated image
        alpha = tf.random.uniform(
            [int(self.batch_size), 1], 0.0, 1.0, dtype=tf.dtypes.float64
        )
        diff = x_hat - x
        interpolated = x + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.Classifier.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def generatorLoss(self, x, x_hat, z, z_hat, d, d_hat):
        """Calculate the loss for the generator optimization step.

        Args:
            x (tf.tensor):
                     normal input samples
            x_hat (tf.tensor):
                    autoencoder output of normal input samples
            z (tf.tensor):
                     encoding of x in latent space by the encoder of the autoencoder
            z_hat (tf.tensor):
                     encoding of x_hat in latent space by the encoder after the autoencoder
            d (tf.tensor):
                     discriminator output for x as input
            d_hat (tf.tensor):
                    discriminator output for x_hat as input

        Returns:
            tf.tensor: value for the loss
        """
        mae = tf.keras.losses.MeanAbsoluteError()
        mse = tf.keras.losses.MeanSquaredError()
        adv_loss = mse(d, d_hat)
        con_loss = mae(x, x_hat)
        enc_loss = mse(z, z_hat)

        return adv_loss*float(self.adv_loss_weight) , con_loss*float(self.con_loss_weight), enc_loss*float(self.enc_loss_weight)

    def calculateMetrics(self, step=0, validation_or_test="validation"):
        """Calculate the metrics on the validation dataset.

        Args:
            opt (tf.keras.optimizer): [unused] Optimizer required for
                the AnoGan architecture e.g. Adam optimizer.

        Returns:
            dict: dict containing the results for the different metrics.
        """

        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        if validation_or_test == "validation":
            x_normal, x_unnormal = self.Data.get_validation_data(batch_size=int(self.validation_samples))
        elif validation_or_test == "test":
            x_normal, x_unnormal = self.Data.get_test_data()

        # normal error
        z_normal = self.Classifier.auto_encoder(
            x_normal, training=False
        )
        z_quantum_normal = self.transform_z_to_z_quantum(z_normal)
        x_hat_normal = self.Classifier.auto_decoder(
            z_quantum_normal, training=False
        )
        z_hat_normal = self.Classifier.encoder(
            x_hat_normal, training=False
        )
        enc_loss_normal = mae(z_normal, z_hat_normal)

        # unnormal error
        z_unnormal = self.Classifier.auto_encoder(
            x_unnormal, training=False
        )
        z_quantum_unnormal = self.transform_z_to_z_quantum(z_unnormal)            
        x_hat_unnormal = self.Classifier.auto_decoder(
            z_quantum_unnormal, training=False
        )
        z_hat_unnormal = self.Classifier.encoder(
            x_hat_unnormal, training=False
        )
        enc_loss_unnormal = mae(z_unnormal, z_hat_unnormal)

        # get x and x_hat normal and unnormal samples
        sample_num = 4 if len(x_normal) >= 4 and len(x_unnormal) >= 4 else min(len(x_normal), len(x_unnormal))
        x_normal_samples = x_normal[:sample_num]
        x_hat_normal_samples = x_hat_normal.numpy()[:sample_num]
        x_unnormal_samples = x_unnormal[:sample_num]
        x_hat_unnormal_samples = x_hat_unnormal.numpy()[:sample_num]

        res = self.optimize_anomaly_threshold_and_get_metrics(enc_loss_normal.numpy(), enc_loss_unnormal.numpy(),
                                                              validation=self.validation)

        if validation_or_test == "validation":
            res["x_normal_samples"] = x_normal_samples
            res["x_hat_normal_samples"] = x_hat_normal_samples
            res["x_unnormal_samples"] = x_unnormal_samples
            res["x_hat_unnormal_samples"] = x_hat_unnormal_samples
            if self.quantum:
                # Save weights of quantum layer ([1] as [0] is the input_layer)
                self.train_hist["quantum_weights"].append(self.Classifier.auto_decoder.layers[1].get_weights()[0])
            tmp_res = float(str(res["MCC"]).replace(" (nan error)", '')) # FK: needed for nan error, which happens in true_divide by a division with 0
            if self.best_mcc < tmp_res:
                self.best_mcc = tmp_res
                self.Classifier.save(step=step, MCC=tmp_res, threshold=res["threshold"], overwrite_best=True)
                print("\nModel with new highscore saved!")
        elif validation_or_test == "test":
            res["TP"] = [res["TP"]]
            res["FP"] = [res["FP"]]
            res["TN"] = [res["TN"]]
            res["FN"] = [res["FN"]]

        return res

    def optimize_anomaly_threshold_and_get_metrics(self, enc_loss_normal, enc_loss_unnormal, validation=True):
        """
        Args:
            enc_loss_normal: np.array of shape (len(enc_loss_normal)) holding the scaled anomaly scores for each sample
            enc_loss_unnormal: np.array of shape (len(enc_loss_unnormal)) holding the scaled anomaly scores for each sample
        """

        # enrich scaled anomaly scores with their true labels for each sample
        prepare_normal = np.dstack((enc_loss_normal, -np.ones_like(enc_loss_normal)))[0]
        prepare_unnormal = np.dstack((enc_loss_unnormal, np.ones_like(enc_loss_unnormal)))[0]
        complete_set = np.vstack((prepare_normal, prepare_unnormal)).tolist()
        sorted_complete_set = sorted(complete_set, key=lambda x:x[0])
        sorted_true_labels = np.asarray(sorted_complete_set)[:, 1].astype(int).tolist()

        # determine optimal threshold and according index in sorted set of samples
        if validation:
            epsilon = 10**(-7)
            threshold = 0.
            best_index = 0
            optimizer_score = -len(sorted_complete_set)
            sorted_complete_set[0][1] = int(sorted_complete_set[0][1])
            for i in range(1, len(sorted_complete_set)):
                sorted_complete_set[i][1] = int(sorted_complete_set[i][1])
                new_score = np.sum(sorted_complete_set[i:], axis=0)[1] - np.sum(sorted_complete_set[:i+1], axis=0)[1]
                if optimizer_score < new_score:
                    threshold = sorted_complete_set[i][0] - epsilon
                    best_index = i
                    optimizer_score = new_score

        # during testing, only look for the index of the last sample, whose anomaly score falls below the threshold
        else:
            threshold = self.Classifier.threshold
            best_index = 0
            for i in range(1, len(sorted_complete_set)):
                if sorted_complete_set[i][0] > threshold:
                    continue
                else:
                    best_index = i

        # partition samples according to threshold
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

        # compute result metrics
        result = {}
        result["TP"] = distribution_above_threshold[1]
        result["FP"] = distribution_above_threshold[-1]
        result["TN"] = distribution_below_threshold[-1]
        result["FN"] = distribution_below_threshold[1]
        result["threshold"] = threshold
        try:
            result["MCC"] = (result["TP"] * result["TN"] - result["FP"] * result["FN"]) / (
                    (result["TP"] + result["FP"]) * (result["TP"] + result["FN"]) * (result["TN"] + result["FP"]) * (
                        result["TN"] + result["FN"])) ** (1 / 2)
        except RuntimeWarning or RuntimeError as w:
            result["MCC"] = "0.0 (nan error)"

        return result

class QuantumDecoderTrainer(Trainer):
    """
    Class holding the required methods and attributes to conduct the training or testing procedure for the quantum method.
    """

    def __init__(
            self,
            Data,
            Classifier
    ):
        super().__init__(Data=Data, Classifier=Classifier)
        self.train_hist["quantum_weights"] = []
        self.quantum = True

    def transform_z_to_z_quantum(self, z):
        z_np = z.numpy()
        result = []
        for i in range(len(z_np)):
            circuit = cirq.Circuit()
            transformed_inputs = 2 * np.arcsin(z_np[i])
            for j in range(int(self.latent_dim)):
                circuit.append(cirq.rx(transformed_inputs[j]).on(self.Classifier.qubits[j]))
            result.append(circuit)
        result = tfq.convert_to_tensor(result)
        stop = "stop"
        return result