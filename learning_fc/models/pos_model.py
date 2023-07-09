from learning_fc import safe_rescale
from learning_fc.models import BaseModel

class PosModel(BaseModel):

    def predict(self, obs, **kwargs):
        q = self.env.q
        deltaq = self.env.q_deltas

        return self._deltaq_to_qdes(q, deltaq), {}