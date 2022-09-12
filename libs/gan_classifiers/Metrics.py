import numpy as np


class Metric:
    def __init__(self, data, parameters):
        self.data = data
        self.validation_samples = int(parameters["validation_samples"])
        self.metrics = self._metrics_template()
        self.metricHistory = []

    def update_key(self, key, value):
        assert (
            key in self.metrics
        ), "Error: trying to update nonexistent metric"
        self.metrics[key] = value

    @staticmethod
    def _metrics_template():
        """Creation function for an empty set of metrics"""
        return {
            "step_number": None,
            "total_runtime": None,
            "runtime_per_step": None,
            "generator_loss": None,
            "adversarial_loss": None,
            "contextual_loss": None,
            "encoder_loss": None,
            "discriminator_loss": None,
        }

    def finalize(self):
        """All metrics are collected and are stored in the history.
        A new set of empty metric data is generated.
        """
        self.metricHistory.append(self.metrics)
        self.metrics = self._metrics_template()

    def getLastMetrics(self):
        return self.metricHistory[-1].copy()

    def history_from_key(self, key):
        """Get a list of all historic entries for a given key."""
        return [m[key] for m in self.metricHistory]

    def is_best(self):
        """Returns if the current set of metrics is the best in the
        training history.

        This function is used to determine if the previous model
        weights should be overwritten."""
        return True

    def get(self, key):
        return self.metrics[key]


class SupervisedMetric(Metric):
    def __init__(self, data, parameters):
        super().__init__(data, parameters)
        self.threshold = float(parameters.get("threshold", 0.0))

    def _metrics_template(self):
        template = super()._metrics_template()
        template.update(
            {
                "TP": None,
                "FP": None,
                "TN": None,
                "FN": None,
                "threshold": None,
                "MCC": None,
                "normalScores": None,
                "anomalyScores": None,
            }
        )
        return template

    @staticmethod
    def optimize_anomaly_threshold(enc_loss_normal, enc_loss_unnormal):
        """
        Args:
            enc_loss_normal: np.array of shape (len(enc_loss_normal)) holding the scaled anomaly scores for each sample
            enc_loss_unnormal: np.array of shape (len(enc_loss_unnormal)) holding the scaled anomaly scores for each sample

        Return: The optimal threshold value using the validation data
        """

        # enrich scaled anomaly scores with their true labels for each sample
        prepare_normal = np.dstack(
            (enc_loss_normal, -np.ones_like(enc_loss_normal))
        )[0]
        prepare_unnormal = np.dstack(
            (enc_loss_unnormal, np.ones_like(enc_loss_unnormal))
        )[0]
        complete_set = np.vstack((prepare_normal, prepare_unnormal)).tolist()
        sorted_complete_set = sorted(complete_set, key=lambda x: x[0])

        # determine optimal threshold and according index in sorted set of samples
        epsilon = 10 ** (-7)
        threshold = 0.0
        optimizer_score = -len(sorted_complete_set)
        sorted_complete_set[0][1] = int(sorted_complete_set[0][1])
        for i in range(1, len(sorted_complete_set)):
            sorted_complete_set[i][1] = int(sorted_complete_set[i][1])
            new_score = (
                np.sum(sorted_complete_set[i:], axis=0)[1]
                - np.sum(sorted_complete_set[: i + 1], axis=0)[1]
            )
            if optimizer_score < new_score:
                threshold = sorted_complete_set[i][0] - epsilon
                optimizer_score = new_score

        return threshold

    def metric_during_training(self, prediction_func, _):
        """Calculate the metrics during training."""
        return self.calculate_metrics(
            self.data.get_validation_data(
                batch_size=int(self.validation_samples)
            ),
            prediction_func,
            None,
            during_training=True,
        )

    def calculate_metrics(
        self, dataset, prediction_func, _, during_training=False
    ):
        """Calculate the metrics on the validation dataset."""
        x_normal, x_annomaly = dataset
        enc_loss_normal = prediction_func(x_normal).numpy()
        enc_loss_unnormal = prediction_func(x_annomaly).numpy()

        if during_training:
            self.threshold = self.optimize_anomaly_threshold(
                enc_loss_normal.numpy(), enc_loss_unnormal.numpy()
            )

        # compute result metrics
        self.update_key(
            "TP", np.count_nonzero(enc_loss_unnormal > self.threshold)
        )
        self.update_key(
            "FP", np.count_nonzero(enc_loss_normal > self.threshold)
        )
        self.update_key(
            "TN", np.count_nonzero(enc_loss_normal <= self.threshold)
        )
        self.update_key(
            "FN", np.count_nonzero(enc_loss_unnormal <= self.threshold)
        )
        self.update_key("threshold", self.threshold)
        self.update_key("normalScores", enc_loss_normal)
        self.update_key("anomalyScores", enc_loss_unnormal)
        try:
            self.update_key(
                "MCC",
                (
                    self.metrics["TP"] * self.metrics["TN"]
                    - self.metrics["FP"] * self.metrics["FN"]
                )
                / (
                    (self.metrics["TP"] + self.metrics["FP"])
                    * (self.metrics["TP"] + self.metrics["FN"])
                    * (self.metrics["TN"] + self.metrics["FP"])
                    * (self.metrics["TN"] + self.metrics["FN"])
                )
                ** (1 / 2),
            )
        except:
            # division by 0
            self.update_key("MCC", 0.0)

        return self.metrics

    def is_best(self):
        mcc = (
            self.metrics["MCC"] if isinstance(self.metrics["MCC"], int) else 0
        )
        return not np.any(np.array(self.history_from_key("MCC")) > mcc)


class UnsupervisedMetric(Metric):
    def _metrics_template(self):
        template = super()._metrics_template()
        template.update(
            {
                "anomaly score": [],
                "original_samples": [],
                "generated_samples": [],
            }
        )
        return template

    def metric_during_training(self, prediction_func, generation_func):
        x = self.data.get_validation_data(self.validation_samples)
        self.calculate_metrics(x, prediction_func, generation_func)

    def calculate_metrics(self, dataset, prediction_func, generation_func):
        self.update_key("original_samples", dataset[: self.validation_samples])
        self.update_key(
            "generated_samples",
            generation_func(dataset[: self.validation_samples]).numpy(),
        )
        self.update_key("anomaly score", prediction_func(dataset).numpy())
        return self.metrics
