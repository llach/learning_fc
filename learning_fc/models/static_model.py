import numpy as np

class StaticModel:

    def __init__(self, q) -> None:
        self.q = q

    def predict(self, *args, **kwargs):
        return np.array(2*[self.q]), {}