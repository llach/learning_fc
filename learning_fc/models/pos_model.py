from learning_fc import safe_rescale
from learning_fc.models import BaseModel

class PosModel(BaseModel):

    def predict(self, obs, **kwargs):
        q = safe_rescale(obs[:2], [-1, 1], [0.0, 0.045])
        deltaq = safe_rescale(obs[2:4], [-1, 1], [-0.045, 0.045])

        return self._deltaq_to_qdes(q, deltaq), {}