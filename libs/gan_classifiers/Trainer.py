"""
This file provides the Trainer class which conducts the training process.
"""
import logging
import time

import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger()


class Trainer:
    """
    Class holding the required methods and attributes to conduct the training or testing procedure.
    """

    def __init__(self, data, classifier, metrics_object, parameters):
        """Initialize necessary parameters to train the model via a Wasserstein-Loss based GANomaly ansatz.

        This implementation is supposed to remedy some of the training difficulties of GANs (e.g.
        vanishing gradients).
        Args:
            classifier: sub-classed Classifier-object defined in GANomalyNetworks.py.
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

        self.data = data
        self.classifier = classifier

        self.validation = (
            True if parameters["train_or_predict"] == "train" else False
        )
        self.opt_disc = tf.keras.optimizers.Adam(
            beta_1=0.5,
            learning_rate=float(parameters["discriminator_training_rate"]),
        )
        self.opt_gen = tf.keras.optimizers.Adam(
            beta_1=0.5,
            learning_rate=float(parameters["generator_training_rate"]),
        )

        self.metrics_object = metrics_object
        self.latent_dim = parameters["latent_dimensions"]
        self.n_steps = parameters["training_steps"]
        self.step_counter = 0
        self.batch_size = parameters["batch_size"]
        self.discriminator_iterations = parameters["discriminator_iterations"]
        self.update_interval = parameters["validation_interval"]
        self.gradient_penalty_weight = parameters["gradient_penalty_weight"]
        self.adv_loss_weight = parameters["adv_loss_weight"]
        self.con_loss_weight = parameters["con_loss_weight"]
        self.enc_loss_weight = parameters["enc_loss_weight"]
        self.g_loss = 0
        self.adv_loss = 0
        self.con_loss = 0
        self.enc_loss = 0
        self.d_loss = 0
        self.best_mcc = -1.0

    def train(self):
        """Run the training procedure. The first validation and logging of losses happens after the validation interval
        steps. If it would start at step 0, the first major changes in losses and metrics would be protocolled, leading
        to significant overstretching in the plots created at the end of the training procedure.

        Returns:
            tuple(list, list): A tuple containing the costs snapshots and
                    parameter snapshots. The parameters are empty in this
                    implementation.
        """
        tic = time.perf_counter()

        for _ in tqdm(range(int(self.n_steps)), desc=self.__str__()):
            self._step()
            self.step_counter += 1
            if (int(self.step_counter) % int(self.update_interval)) == 0:
                # generic Metric info
                self.metrics_object.update_key(
                    "step_number", self.step_counter
                )
                toc = time.perf_counter()
                self.metrics_object.update_key("total_runtime", toc - tic)
                self.metrics_object.update_key(
                    "runtime_per_step", (toc - tic) / self.step_counter
                )
                self.metrics_object.update_key("generator_loss", self.g_loss)
                self.metrics_object.update_key(
                    "adversarial_loss", self.adv_loss
                )
                self.metrics_object.update_key(
                    "contextual_loss", self.con_loss
                )
                self.metrics_object.update_key("encoder_loss", self.enc_loss)
                self.metrics_object.update_key(
                    "discriminator_loss", self.d_loss
                )
                self.metrics_object.metric_during_training(
                    self.classifier.predict, self.classifier.generate
                )

                if self.metrics_object.is_best():
                    self.best_weights = self.classifier.save()
                    logger.info("\nModel with new highscore saved!")
                self.metrics_object.finalize()
        return self.best_weights

    def _step(self):
        """Perform a single step of the optimization.
        This step consists of several updates for the discriminator parameters
        and a single update for the generator parameters.
        """
        x = self.data.get_train_data(self.batch_size)
        for i in range(int(self.discriminator_iterations)):
            with tf.GradientTape(persistent=True) as tape:
                z = self.classifier.auto_encoder(x, training=True)
                z_quantum = self.classifier.transform_z_to_z_quantum(z)
                x_hat = self.classifier.auto_decoder(z_quantum, training=True)
                z_hat = self.classifier.encoder(x_hat, training=True)
                d = self.classifier.discriminator(x, training=True)
                d_hat = self.classifier.discriminator(x_hat, training=True)

                # discriminator losses
                d_cost = self.discriminator_loss(d, d_hat)
                gp = self.gradient_penalty(x, x_hat)
                d_loss = d_cost + gp * float(self.gradient_penalty_weight)

                # generator losses
                if i == (int(self.discriminator_iterations) - 1):
                    (
                        adv_loss,
                        con_loss,
                        enc_loss,
                    ) = self.generator_loss(x, x_hat, z, z_hat, d, d_hat)
                    g_loss = adv_loss + con_loss + enc_loss
                    self.g_loss = float(g_loss)
                    self.adv_loss = float(adv_loss)
                    self.con_loss = float(con_loss)
                    self.enc_loss = float(enc_loss)
                    self.d_loss = float(d_loss)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, self.classifier.discriminator.trainable_variables
            )
            # Update the weights of the discriminator
            self.opt_disc.apply_gradients(
                zip(
                    d_gradient,
                    self.classifier.discriminator.trainable_variables,
                )
            )

            # Update the weights of the generator
            if i == (int(self.discriminator_iterations) - 1):
                gen_gradient = tape.gradient(
                    g_loss,
                    self.classifier.auto_encoder.trainable_variables
                    + self.classifier.auto_decoder.trainable_variables
                    + self.classifier.encoder.trainable_variables,
                )
                self.opt_gen.apply_gradients(
                    zip(
                        gen_gradient,
                        self.classifier.auto_encoder.trainable_variables
                        + self.classifier.auto_decoder.trainable_variables
                        + self.classifier.encoder.trainable_variables,
                    )
                )

    @staticmethod
    def discriminator_loss(d, d_hat):
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
            pred = self.classifier.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def generator_loss(self, x, x_hat, z, z_hat, d, d_hat):
        """Calculaete the loss for the generator optimization step.

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

        return (
            adv_loss * float(self.adv_loss_weight),
            con_loss * float(self.con_loss_weight),
            enc_loss * float(self.enc_loss_weight),
        )
