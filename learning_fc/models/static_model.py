from learning_fc import safe_rescale

class StaticModel:

    def __init__(self, q) -> None:
        self.q = q

    def predict(self, *args, **kwargs):
        return safe_rescale(2*[self.q], [0.0, 0.045]), {}