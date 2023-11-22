from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layers import DenseLayer


class Optimiser:
    def __init__(self):
        self.alpha = 0.1

    def update_params(self, layer):
        layer.w += -self.alpha * layer.dL_dW
        layer.b += -self.alpha * layer.dL_dBf
