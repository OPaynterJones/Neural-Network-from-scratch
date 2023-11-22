import numpy as np
from .layers import Layer, DenseLayer
from .loss_functions import LossFunction
from .optimisers import Optimiser

class NeuralNetwork:
    def __init__(self, layers, loss_function, optimiser, network_structure: list[tuple | None]):
        self.layers: list[Layer] = []

        for layer, layer_structure in zip(layers, network_structure):
            if layer_structure:
                self.layers.append(layer(*layer_structure))
            else:
                self.layers.append(layer())

        self.loss_function: LossFunction = loss_function()
        self.optimiser: Optimiser = optimiser()

    def train(self, training_samples: np.array, labels: np.array, batch_size=None, epochs=800):
        # training samples should have inputs on columns, and samples on rows for each batch 

        for e in range(epochs):
            # loop through the batches
            pred = self.__forward_pass(training_samples)

            loss = self.loss_function.calculate_loss(pred, labels)

            self.loss_function.backward_propagate(pred, labels)

            self.__backward_pass(self.loss_function.dL_dX)
            self.__optimise_wb()

            acc = np.mean(np.around(pred) == labels) * 100
        print(f"Loss {loss}\nLast known accuracy: {acc}")

    def predict(self, test_samples):
        return self.__forward_pass(test_samples)

    def test(self, test_samples, labels):
        pred = self.predict(test_samples)
        loss = self.loss_function.calculate_loss(pred, labels)
        print("---------TEST RESULTS---------")
        acc = np.mean(np.around(pred) == labels) * 100
        print(f"Loss: {loss}\nAccuracy: {acc}")

    def __forward_pass(self, batch):
        self.layers[0].forward_propagate(batch)
        X = self.layers[0].outputs
        for layer in self.layers[1:]:
            layer.forward_propagate(X)
            X = layer.outputs

        return X

    def __backward_pass(self, dL_dPred):
        dL_dX = dL_dPred
        for layer in self.layers[::-1]:
            layer.backward_propagate(dL_dX)
            dL_dX = layer.dL_dX

    def __optimise_wb(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.optimiser.update_params(layer)