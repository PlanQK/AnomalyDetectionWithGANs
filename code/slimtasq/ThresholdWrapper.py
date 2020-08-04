import numpy as np


class Base:
    def __init__(self, model):
        pass


class Model(Base):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def predict(self, X):
        raise NotImplementedError("You need to override this function")


class ThresholdWrapper(Model):
    """
    Converts continuous (outlier) scores into a binary classification.
    """

    def __init__(self, model):
        super().__init__(model)
        self._threshold = 0.5

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, thr):
        self._threshold = thr

    def predict(self, X):
        # decorates the predict function to return the correct format
        # output is normally in {-1, 1} and not {0,1}
        result = np.array(
            [1 if i >= self._threshold else 0 for i in self.model.predict(X.to_numpy())]
        )
        return result
