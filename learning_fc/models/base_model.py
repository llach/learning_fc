from learning_fc import safe_rescale
from learning_fc.envs import ControlMode

class BaseModel:

    def __init__(self, control_mode=ControlMode.Position) -> None:
        self.control_mode = control_mode

    def _deltaq_to_qdes(self, q, deltaq):
        if self.control_mode == ControlMode.Position:
            qdes = q+deltaq
            return safe_rescale(qdes, [0.0, 0.045], [-1, 1])
        elif self.control_mode == ControlMode.PositionDelta:
            return safe_rescale(deltaq, [-0.045, 0.045], [-1, 1])
        else:
                assert False, f"unknown control mode {self.control_mode}"

    def predict(self, obs, **kwargs): raise NotImplementedError
        