from learning_fc import safe_rescale
from learning_fc.envs import ControlMode

class BaseModel:

    def __init__(self, env) -> None:
        self.env = env
        self.control_mode = env.control_mode

    def _deltaq_to_qdes(self, q, deltaq):
        if self.control_mode == ControlMode.Position:
            qdes = q+deltaq
            return safe_rescale(qdes, [0.0, 0.045], [-1, 1])
        elif self.control_mode == ControlMode.PositionDelta:
            return safe_rescale(deltaq, [-self.env.dq_max, self.env.dq_max], [-1, 1])
        else:
                assert False, f"unknown control mode {self.control_mode}"

    def predict(self, obs, **kwargs): raise NotImplementedError
        