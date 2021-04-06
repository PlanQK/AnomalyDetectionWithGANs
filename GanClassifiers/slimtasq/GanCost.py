class GanCost:
    def __init__(self, ansatz, **kwargs):
        """Create the cost object.

        Args:
            ansatz (AnoGanAnsatz): This object contains the required
                structure for the AnoGan ansatz.
        """
        ansatz.checkAnsatz()
        self.ansatz = ansatz
        self.init_params = []

    def calculateMetrics(self, opt):
        """Calculate the metrics. For GANs there are no clear metrics.
        Therefore, this method returns 5 samples from the generator.

        Args:
            opt (tf.keras.optimizer): [unused] Optimizer required for
                the AnoGan architecture e.g. Adam optimizer.

        Returns:
            dict: dict containing the results for the different metrics.
        """
        inputData = self.ansatz.latentVariableSampler(5)
        snapshots = self.ansatz.generator.predict(inputData)
        return {"samples": snapshots}

    def get_config(self):
        """Return the parameters needed to create a copy of this object.
        Overridden method from JSONEncoder.

        Returns:
            dict: parameters
        """
        config = {
            "ansatz": self.ansatz._to_dict(),
            "init_params": self.init_params,
        }
        return config

    def __str__(self):
        return "GanCost " + str(self.ansatz)

    def _to_dict(self):
        """Return the class as a serialized dictionary.

        Returns:
            dict: Dictionary representation of the object
        """
        repr_dict = {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
        }
        repr_dict.update(self.get_config())
        return repr_dict
