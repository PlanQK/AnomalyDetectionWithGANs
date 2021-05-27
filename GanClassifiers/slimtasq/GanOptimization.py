import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time


class GanOptimization:
    def __init__(
        self,
        opt,
        name,
        n_steps=600,
        batchSize=64,
        discriminatorIterations=5,
        updateInterval=50,
        opt_args={},
        step_args={},
        interface="autograd",
        **kwargs,
    ):
        """Gnerate a standard Gan optimization object. This class might be
            plagued by vanishing gradients.

        Args:
            opt (tf.keras.optimizer): Optimizer to perform gradient descent
            name (str): A name to identify this object among others
            n_steps (int, optional): Optimization steps. Defaults to 600
            batchSize (int, optional): Number of training samples for
                            each optimization step. Defaults to 64.
            discriminatorIterations (int, optional): How often the
                            discriminator is trained for each generator
                            optimization. Defaults to 5.
            updateInterval (int, optional): How often are debug costs and
                            metrics (in optimization steps). Defaults to 50.
            opt_args (dict, optional): Additional arguments for
                            the optimizer. Defaults to {}.
            step_args (dict, optional): Additional arguments for the step
                            function. Defaults to {}.
            interface (str, optional): Unused, remnant from pennylane.
                            Defaults to "autograd".
        """
        super().__init__(
            opt, name, n_steps, opt_args=opt_args, step_args=step_args
        )
        self.opt = opt
        self.name = name
        self.n_steps = n_steps
        self.step_counter = 0
        self.batchSize = batchSize
        self.discriminatorIterations = discriminatorIterations
        self.updateInterval = updateInterval
        self.interface = ""
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

    __axstep=0
    def _step(self, cost):
        """Perform a single step of the optimization.
        This step consists of several updates for the discriminator parameters
        and a single update for the generator parameters.

        Args:
            cost (GanCost): Metrics for the performance of the gan. Main
                    requirement is the GanAnsatz stored in the cost object.
        """
        self.__axstep=self.__axstep+1
        print("GAN step ",self.__axstep);
        real_images = cost.ansatz.trueInputSampler(self.batchSize)
        for i in range(self.discriminatorIterations):
            # Get the latent vector
            random_latent_vectors = cost.ansatz.latentVariableSampler(
                self.batchSize
            )
            with tf.GradientTape() as tape:
                fake_images = cost.ansatz.generator(
                    random_latent_vectors, training=True
                )
                fake_logits = cost.ansatz.discriminator(
                    fake_images, training=True
                )
                real_logits = cost.ansatz.discriminator(
                    real_images, training=True
                )

                d_loss = self.discriminatorLoss(real_logits, fake_logits)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, cost.ansatz.discriminator.trainable_variables
            )
            # Update the weights of the discriminator
            self.opt.apply_gradients(
                zip(d_gradient, cost.ansatz.discriminator.trainable_variables)
            )

        # Train the generator now.
        random_latent_vectors = cost.ansatz.latentVariableSampler(
            self.batchSize
        )
        with tf.GradientTape() as tape:
            generated_images = cost.ansatz.generator(
                random_latent_vectors, training=True
            )
            gen_img_logits = cost.ansatz.discriminator(
                generated_images, training=True
            )
            g_loss = self.generatorLoss(gen_img_logits)

        gen_gradient = tape.gradient(
            g_loss, cost.ansatz.generator.trainable_variables
        )
        self.opt.apply_gradients(
            zip(gen_gradient, cost.ansatz.generator.trainable_variables)
        )

    def run(self, cost, additional_step_args=None):
        """Run the optimization over many steps.

        Args:
            cost (GanCost): Metrics for the performance of the gan.
            additional_step_args (dict, optional): unused here. Defaults to
            None.

        Returns:
            tuple(list, list): A touple containing the costs snapshots and
                        parameter snapshots. The parameters are empty in this
                        implementation.
        """
        self.status = "running"
        # Override costs and params of previous run (if any)
        self.costs_opt = []
        self.params_opt = []
        print("before optimiztation")
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
        print("after optimization")

        self.runtime["total"] = toc - tic
        if self.n_steps != 0:
            self.runtime["per_step"] = (toc - tic) / self.n_steps
        self.status = "done"
        return self.costs_opt, self.params_opt

    def discriminatorLoss(
        self, realSampleDiscriminatorOutput, fakeSampleDiscriminatorOutput
    ):
        """Calculate the loss for the discriminator optimization steps.

        Args:
            realSampleDiscriminatorOutput (tf.tensor):
                    output from the real sample
            fakeSampleDiscriminatorOutput (tf.tensor):
                    output from the generated samples

        Returns:
            tf.tensor: value for the loss
        """
        valid = tf.convert_to_tensor(
            np.ones((self.batchSize, 1)), dtype=tf.float32
        )
        real_loss = tf.reduce_mean(valid - realSampleDiscriminatorOutput)
        fake_loss = tf.reduce_mean(fakeSampleDiscriminatorOutput)
        return fake_loss + real_loss

    def generatorLoss(self, fakeSampleDiscriminatorOutput):
        """Calculaete the loss for the generator optimization step.

        Args:
            fakeSampleDiscriminatorOutput (tf.tensor):
                    discriminator output from a generated sample

        Returns:
            tf.tensor: value for the loss
        """
        return -tf.reduce_mean(fakeSampleDiscriminatorOutput)

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
            GanOptimization: The new de-serialized object.
        """
        opt = dct.pop("opt")
        name = dct.pop("name")
        n_steps = dct.pop("n_steps")
        step_counter = dct.pop("step_counter")
        opt_args = dct.pop("opt_args")
        batchSize = dct.pop("batchSize")
        discriminatorIterations = dct.pop("discriminatorIterations")
        updateInterval = dct.pop("updateInterval")
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
