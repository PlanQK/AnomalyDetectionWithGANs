"""
This file handles the input data.
"""
import numpy as np


class Data:
    """
    Interface class for supplying training and evaluation data.

    As the anomaly detection can be used in both a supervised
    and unsupervised setting we have two derived Classes:

    SupervisedData: Implements logic for all the methods.

    UnsupervisedData: This will raise an error if methods are
    accessed that rely on labeled data.
    """

    def __init__(self, data):
        self.data = data
        # Initialize train_data as None to avoid errors,
        # derived classes need to set this to a useful value.
        self.train_data = None

    def get_validation_data(self, batch_size):
        """If supervised, returns samples from both normal and anomalous validation data.
        If unsupervised returns samples from the training data.
        Args:
            batch_size : number of entries to sample from (each part of) the data
        Returns:
            The sampled data
        """
        raise NotImplementedError(
            """Get validation data is not implemented. This is usually
            caused by calling metrics evaluations in an unsupervised setting"""
        )

    def get_test_data(self):
        """Returns all the data, split into normal and anomalous in the supervised setting"""
        raise NotImplementedError(
            """Get test data is not implemented. This is usually
            caused by calling metrics evaluations in an unsupervised setting"""
        )

    def get_train_data(self, batch_size):
        """Returns a random sample of the training data.
        Args:
            batch_size: the number of entries sampled

        Returns:
            The sampled part of the training data
        """
        return self.train_data.sample(int(batch_size)).to_numpy().astype(np.float64)


class SupervisedData(Data):
    """
    Class holding the training, validation and test data.

    Args:
        data : array with data points to be used for the training i.e. training set
        or the classification i.e. test set.
    """

    def __init__(self, data):
        super().__init__(data)

        self.feature_length = len(self.data.columns) - 1

        # Drop the class
        self.normal_samples = self.data[self.data[self.feature_length] == 0].drop(
            self.feature_length, axis=1
        )
        self.unnormal_samples = self.data[self.data[self.feature_length] == 1].drop(
            self.feature_length, axis=1
        )

        # create training data set
        partition_normal_indices = [
            int(len(self.normal_samples) * 0.8),
            int(len(self.normal_samples)),
        ]
        partition_unnormal_indices = int(len(self.unnormal_samples))

        self.train_data = self.normal_samples[: partition_normal_indices[0]]

        validation_data_normal = self.normal_samples[
            partition_normal_indices[0] : partition_normal_indices[1]
        ]
        validation_data_unnormal = self.unnormal_samples[:partition_unnormal_indices]
        # Constrain the validation data to the following min value to reach a 50:50 balanced distribution
        # of normal and unnormal samples
        minimum = min(len(validation_data_normal), len(validation_data_unnormal))
        self.validation_data_normal = validation_data_normal[:minimum]
        self.validation_data_unnormal = validation_data_unnormal[:minimum]

    def get_validation_data(self, batch_size):
        """Returns samples from both the normal and anomalous validation data.

        Args:
            batch_size : number of entries to sample from each of them

        Returns:
            The combined samples from both parts of the validation data.
        """
        batch_size = min(
            batch_size,
            len(self.validation_data_unnormal),
            len(self.validation_data_normal),
        )
        normal = (
            self.validation_data_normal.sample(int(batch_size))
            .to_numpy()
            .astype(np.float64)
        )
        unnormal = (
            self.validation_data_unnormal.sample(int(batch_size))
            .to_numpy()
            .astype(np.float64)
        )
        return (normal, unnormal)

    def get_test_data(self):
        """Returns all the data, both normal and anomalous.

        Returns:
            The data, starting with the normal entries.
        """
        normal = self.normal_samples.to_numpy().astype(np.float64)
        unnormal = self.unnormal_samples.to_numpy().astype(np.float64)
        return (normal, unnormal)


class UnsupervisedData(Data):
    """
    Class holding the training data for unsupervised learning.

    Args:
        data : array with data points to be used for the training.
    """

    def __init__(self, data):
        super().__init__(data)
        self.feature_length = len(self.data.columns)
        self.train_data = data

    def get_validation_data(self, batch_size):
        """Returns a random sample of the training data (just like `get_train_data` since this is the unsupervised case)
        Args:
            batch_size: the number of entries sampled

        Returns:
            The sampled part of the training data
        """
        return self.train_data.sample(int(batch_size)).to_numpy().astype(np.float64)

    def get_test_data(self):
        """Returns all the (training) data"""
        return self.train_data.to_numpy().astype(np.float64)
