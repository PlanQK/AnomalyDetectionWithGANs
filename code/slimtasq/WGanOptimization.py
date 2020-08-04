import tensorflow as tf
from tqdm import tqdm
import time


class WGanOptimization:
    def __init__(
        self,
        opt,
        name,
        n_steps=600,
        batchSize=64,
        discriminatorIterations=5,
        updateInterval=50,
        gpWeight=10,
        opt_args={},
        step_args={},
        interface="autograd",
        **kwargs,
    ):
        self.opt = opt
        self.name = name
        self.n_steps = n_steps
        self.runtime = {}
        self.step_counter = 0
        self.batchSize = batchSize
        self.discriminatorIterations = discriminatorIterations
        self.updateInterval = updateInterval
        self.gpWeight = gpWeight
        self.interface = ""
        if "gpWeight" in opt_args:
            self.gpWeight = opt_args["gpWeight"]
        self.customParams = {}
        if "customParams" in kwargs:
            self.customParams = kwargs.pop("customParams")

    def reset(self):
        self.step_counter = 0
        super().reset()

    def _step(self, cost):
        real_images = cost.ansatz.trueInputSampler(self.batchSize)
        for i in range(self.discriminatorIterations):
            # Get the latent vector
            random_latent_vectors = cost.ansatz.latentVariableSampler(self.batchSize)
            with tf.GradientTape() as tape:
                fake_images = cost.ansatz.generator(
                    random_latent_vectors, training=True
                )
                fake_logits = cost.ansatz.discriminator(fake_images, training=True)
                real_logits = cost.ansatz.discriminator(real_images, training=True)

                d_cost = self.discriminatorLoss(real_logits, fake_logits)
                gp = self.gradient_penalty(cost, real_images, fake_images)
                d_loss = d_cost + gp * self.gpWeight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, cost.ansatz.discriminator.trainable_variables
            )
            # Update the weights of the discriminator using the discriminator optimizer
            self.opt.apply_gradients(
                zip(d_gradient, cost.ansatz.discriminator.trainable_variables)
            )

        # Train the generator now.
        random_latent_vectors = cost.ansatz.latentVariableSampler(self.batchSize)
        with tf.GradientTape() as tape:
            generated_images = cost.ansatz.generator(
                random_latent_vectors, training=True
            )
            gen_img_logits = cost.ansatz.discriminator(generated_images, training=True)
            g_loss = self.generatorLoss(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, cost.ansatz.generator.trainable_variables)
        self.opt.apply_gradients(
            zip(gen_gradient, cost.ansatz.generator.trainable_variables)
        )

    def run(self, cost, additional_step_args=None):
        self.status = "running"
        # Override costs and params of previous run (if any)
        self.costs_opt = []
        self.params_opt = []

        tic = time.perf_counter()
        for _ in tqdm(range(self.n_steps), desc=self.__str__()):
            self.step_counter += 1
            self._step(cost)
            if (self.step_counter % self.updateInterval) == 0:
                cost_curr = cost.calculateMetrics(self.opt)
                self.costs_opt.append(cost_curr)
                self.customParams.update({"stepNumber": self.step_counter})
                self.params_opt.append(self.customParams.copy())
        toc = time.perf_counter()

        self.runtime["total"] = toc - tic
        if self.n_steps != 0:
            self.runtime["per_step"] = (toc - tic) / self.n_steps
        self.status = "done"
        return self.costs_opt, self.params_opt

    @staticmethod
    def discriminatorLoss(realSampleDiscriminatorOutput, fakeSampleDiscriminatorOutput):
        real_loss = tf.reduce_mean(realSampleDiscriminatorOutput)
        fake_loss = tf.reduce_mean(fakeSampleDiscriminatorOutput)
        return fake_loss - real_loss

    @staticmethod
    def generatorLoss(fakeSampleDiscriminatorOutput):
        return -tf.reduce_mean(fakeSampleDiscriminatorOutput)

    def gradient_penalty(self, cost, realSample, fakeSample):
        """ Calculates the gradient penalty (a method to maintain lipschitz continuity).

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.uniform(
            [self.batchSize, 1], 0.0, 1.0, dtype=tf.dtypes.float32
        )
        diff = fakeSample - realSample
        interpolated = realSample + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = cost.ansatz.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def _to_dict(self):
        # TODO: add ansatz to repr_dict
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
        """[summary]

        Args:
            dct ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
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
