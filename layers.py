import numpy as np

class Layer:
    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.outputs = None

    def forward_propagate(self, inputs):
        self.inputs = inputs

    def backward_propagate(self, gradients):
        pass


class DenseLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.dL_dX = None
        self.dL_dB = None
        self.dL_dW = None

        self.b = np.zeros((1, n_neurons))
        self.w = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.w) + self.b

    def backward_propagate(self, gradients):
        self.dL_dW = np.dot(self.inputs.T, gradients)
        self.dL_dB = np.sum(gradients, axis=0, keepdims=True)
        self.dL_dX = np.dot(gradients, self.w.T)
