import numpy as np

class LossFunction:
    def __init__(self):
        self.dL_dX = None
        self.labels = None

    def calculate_loss(self, pred, truth):
        pass

    def backward_propagate(self, grads, truth):
        pass


class BinaryCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, predictions, labels):
        return np.mean(
            -labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)
        )

    def backward_propagate(self, preds, labels):
        samples = len(preds)
        outputs = len(preds[0])
        clipped_gradients = np.clip(preds, 1e-7, 1 - 1e-7)
        self.dL_dX = (
            -(labels / clipped_gradients - (1 - labels) / (1 - clipped_gradients))
            / outputs
        )
        self.dL_dX = self.dL_dX / samples


class CategoricalCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, pred, truth):
        return -np.sum(truth * np.log(pred + 10**-100))

    def backward_propagate(self, grads, truth):
        pass
