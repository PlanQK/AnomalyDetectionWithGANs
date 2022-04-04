import tensorflow as tf
from tqdm import tqdm
import time


class WGanOptimization:
    def __init__(
            self,
            opt_disc,
            opt_gen,
            name,
            n_steps=600,
            batchSize=64,
            discriminatorIterations=5,
            updateInterval=50,
            gpWeight=10,
            adv_loss_weight=1,
            con_loss_weight=50,
            enc_loss_weight=1,
            opt_args={},
            step_args={},
            interface="autograd",
            **kwargs,
    ):
        """Generate a Wasserstein Gan optimization object. This implementation
        is supposed to remedy some of the training difficulties of Gans (e.g.
        vanishing gradients).

        Args:
            opt (tf.keras.optimizer): Optimizer to perform gradient descent
            name (str): A name to identify this object among others.
            n_steps (int, optional): Optimization steps. Defaults to 600.
            batchSize (int, optional): Number of training samples for each
                    optimization step. Defaults to 64.
            discriminatorIterations (int, optional): How often the
                    discriminator is trained for each generator optimization.
                    Defaults to 5.
            updateInterval (int, optional): How often are debug costs and
                    metrics (in optimization steps). Defaults to 50.
            opt_args (dict, optional): Additional arguments for the optimizer.
                    Defaults to {}.
            step_args (dict, optional): Additional arguments for the step
                    function. Defaults to {}.
            interface (str, optional): Unused, remnant from pennylane.
                    Defaults to "autograd".
        """
        self.opt_disc = opt_disc
        self.opt_gen = opt_gen
        self.name = name
        self.n_steps = n_steps
        self.runtime = {}
        self.step_counter = 0
        self.batchSize = batchSize
        self.discriminatorIterations = discriminatorIterations
        self.updateInterval = updateInterval
        self.gpWeight = gpWeight
        self.adv_loss_weight = adv_loss_weight
        self.con_loss_weight = con_loss_weight
        self.enc_loss_weight = enc_loss_weight
        self.interface = ""
        self.g_loss = 0
        self.adv_loss = 0
        self.con_loss = 0
        self.enc_loss = 0
        self.d_loss = 0
        if "gpWeight" in opt_args:
            self.gpWeight = opt_args["gpWeight"]
        self.customParams = {}
        if "customParams" in kwargs:
            self.customParams = kwargs.pop("customParams")

    def reset(self):
        """
        Reset the internal optimizer state in order to
        start a new independent optimization run.
        """
        self.step_counter = 0
        super().reset()

    __axstep = 0

    def _step(self, cost):
        """Perform a single step of the optimization.
        This step consists of several updates for the discriminator parameters
        and a single update for the generator parameters.

        Args:
            cost (GanCost): Metrics for the performance of the gan. Main
                    requirement is the GanAnsatz stored in the cost object.
        """
        self.__axstep = self.__axstep + 1
        #print("WGAN optim step ", self.__axstep)
        x = cost.ansatz.trainSampler(self.batchSize)
        for i in range(self.discriminatorIterations):
            with tf.GradientTape(persistent=True) as tape:
                z = cost.ansatz.auto_encoder(
                    x, training=True
                )
                x_hat = cost.ansatz.auto_decoder(
                    z, training=True
                )
                z_hat = cost.ansatz.encoder(
                    x_hat, training=True
                )
                d = cost.ansatz.discriminator(
                    x, training=True
                )
                d_hat = cost.ansatz.discriminator(
                    x_hat, training=True
                )

                # discriminator losses
                d_cost = self.discriminatorLoss(d, d_hat)
                gp = self.gradient_penalty(cost, x, x_hat)
                d_loss = d_cost + gp * self.gpWeight
                self.d_loss = float(d_loss)

                # generator losses
                if i == (self.discriminatorIterations - 1):
                    self.adv_loss, self.con_loss, self.enc_loss = self.generatorLoss(x, x_hat, z, z_hat, d, d_hat)
                    g_loss = self.adv_loss + self.con_loss + self.enc_loss
                    self.g_loss = float(g_loss)
                    self.adv_loss = float(self.adv_loss)
                    self.con_loss =float(self.con_loss)
                    self.enc_loss = float(self.enc_loss)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, cost.ansatz.discriminator.trainable_variables
            )
            # Update the weights of the discriminator
            self.opt_disc.apply_gradients(
                zip(d_gradient, cost.ansatz.discriminator.trainable_variables)
            )
            if i == (self.discriminatorIterations - 1):
                gen_gradient = tape.gradient(
                    g_loss,
                    cost.ansatz.auto_encoder.trainable_variables + cost.ansatz.auto_decoder.trainable_variables + cost.ansatz.encoder.trainable_variables
                )
                self.opt_gen.apply_gradients(
                    zip(gen_gradient, cost.ansatz.auto_encoder.trainable_variables + cost.ansatz.auto_decoder.trainable_variables + cost.ansatz.encoder.trainable_variables)
                )

    def run(self, cost, additional_step_args=None):
        """Run the optimization over many steps.

        Args:
            cost (GanCost): Metrics for the performance of the gan.
            additional_step_args (dict, optional): unused here.
                    Defaults to None.

        Returns:
            tuple(list, list): A tuple containing the costs snapshots and
                    parameter snapshots. The parameters are empty in this
                    implementation.
        """
        self.status = "running"
        # Override costs and params of previous run (if any)
        self.train_hist = {"generator loss": [], "adv loss": [], "con loss": [], "enc loss": [],
                           "discriminator loss": [], "TP": [], "FP": [], "TN": [], "FN": [],
                           "best_threshold": [], "MCC": [], "x_normal_samples": [], "x_hat_normal_samples": [],
                           "x_unnormal_samples": [], "x_hat_unnormal_samples": []}
        self.params_hist = {"stepNumber": []}

        tic = time.perf_counter()

        for _ in tqdm(range(self.n_steps), desc=self.__str__()):
            self.step_counter += 1
            self._step(cost)
            if (self.step_counter % self.updateInterval) == 0:
                cost_curr = cost.calculateMetrics(step=self.step_counter)
                cost_curr["generator loss"] = self.g_loss
                cost_curr["discriminator loss"] = self.d_loss
                cost_curr["adv loss"] = self.adv_loss
                cost_curr["con loss"] = self.con_loss
                cost_curr["enc loss"] = self.enc_loss
                for key, value in cost_curr.items():
                    self.train_hist[key].append(value)
                self.customParams.update({"stepNumber": self.step_counter})
                self.params_hist["stepNumber"].append(self.step_counter)
                print(f"\nMCC: {self.train_hist['MCC'][-1]}, Gen loss: {self.g_loss}, Disc loss: {self.d_loss}")
        toc = time.perf_counter()

        self.runtime["total"] = toc - tic
        if self.n_steps != 0:
            self.runtime["per_step"] = (toc - tic) / self.n_steps
        self.status = "done"
        return self.train_hist, self.params_hist

    @staticmethod
    def discriminatorLoss(
            d, d_hat
    ):
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

    def generatorLoss(self, x, x_hat, z, z_hat, d, d_hat):
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

        return adv_loss*self.adv_loss_weight , con_loss*self.con_loss_weight, enc_loss*self.enc_loss_weight

    def gradient_penalty(self, cost, x, x_hat):
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
            [self.batchSize, 1], 0.0, 1.0, dtype=tf.dtypes.float64
        )
        diff = x_hat - x
        interpolated = x + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = cost.ansatz.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def _to_dict(self):
        """Return the class as a serialized dictionary.

        Returns:
            [dict]: Dictionary representation of the object
        """
        repr_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
            "opt": self.opt,
            "name": self.name,
            "n_steps": self.n_steps,
            "step_counter": self.step_counter,
            "opt_args": self.opt_args,
            "batchSize": self.batchSize,
            "discriminatorIterations": self.discriminatorIterations,
            "updateInterval": self.updateInterval,
            "gpWeight": self.gpWeight,
            "step_args": self.step_args,
            "runtime": self.runtime,
            "status": self.status,
            "params_opt": self.params_opt,
            "costs_opt": self.costs_opt,
        }
        return repr_dict

    @classmethod
    def _from_dict(cls, dct):
        """Create a new object from a serialized dict.

        Args:
            dct (dict): serialized form of a previous object

        Raises:
            ValueError: dct is missing elements or has invalid parameters.

        Returns:
            WGanOptimization: The new de-serialized object.
        """
        opt = dct.pop("opt")
        name = dct.pop("name")
        n_steps = dct.pop("n_steps")
        step_counter = dct.pop("step_counter")
        opt_args = dct.pop("opt_args")
        batchSize = dct.pop("batchSize")
        discriminatorIterations = dct.pop("discriminatorIterations")
        updateInterval = dct.pop("updateInterval")
        gpWeight = dct.pop("gpWeight")
        step_args = dct.pop("step_args", None)
        runtime = dct.pop("runtime", dict())
        status = dct.pop("status", "initialised")
        params_opt = dct.pop("params_opt", [])
        costs_opt = dct.pop("costs_opt", [])

        obj = cls(
            opt,
            name,
            n_steps=n_steps,
            batchSize=batchSize,
            discriminatorIterations=discriminatorIterations,
            updateInterval=updateInterval,
            gpWeight=gpWeight,
            opt_args=opt_args,
            step_args=step_args,
            interface=None,
        )
        obj.step_counter = step_counter
        obj.runtime = runtime
        obj.status = status
        obj.params_opt = params_opt
        obj.costs_opt = costs_opt
        return obj
