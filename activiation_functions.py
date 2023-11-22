import numpy as np

class ActivationLayer:
    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.outputs = None

    def forward_propagate(self, inputs):
        pass

    def backward_propagate(self, gradients):
        pass


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward_propagate(self, gradients):
        self.dL_dX = gradients.copy()
        self.dL_dX[self.inputs <= 0] = 0


class SoftMax(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward_propagate(self, inputs):
        self.inputs = inputs
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = inputs / np.sum(inputs, axis=1, keepdims=True)
        return self.outputs

class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.dL_dX = None

    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward_propagate(self, gradients):
        self.dL_dX = gradients * (1 - self.outputs) * self.outputs
