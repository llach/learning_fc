import numpy as np
from enum import Enum

from learning_fc import safe_rescale
from learning_fc.models import BaseModel

class ControllerPhase(int, Enum):
    POSITION_CTRL=0
    FORCE_CLOSURE=1
    FORCE_CTRL=2

class ForcePI(BaseModel):

    def __init__(self, env, Kp=.5, Ki=3.1, k=160, closing_vel=0.02, q_limits=[0.0, 0.045], verbose=False, **kwargs):
        self.verbose = verbose
        self.q_limits = q_limits

        # controller parameters
        self.k = k      # object stiffness
        self.Kp = Kp    # factor for p-part
        self.Ki = Ki    # factor for i-part
        self.closing_vel = closing_vel  # closing velocity during position control

        # reset (or initialize) controller state
        self.reset()

        BaseModel.__init__(self, env=env)

    def reset(self):
        self.joint_transition = [False, False]
        self.error_integral = 0.0

        self.phase = ControllerPhase.POSITION_CTRL

    def get_q(self, q, f_t, fgoal):
        delta_qs = np.zeros_like(q)
        delta_q_ = 0

        for i, f in enumerate(f_t):

            # phase I: closing
            if np.abs(f) < self.env.ftheta and not self.joint_transition[i]:
                delta_qs[i] = -self.closing_vel*self.env.dt*5
            # phase II: contact acquisition → waiting for force-closure
            elif not self.joint_transition[i]:
                self.phase = ControllerPhase.FORCE_CLOSURE
                self.joint_transition[i] = True
                if self.verbose: print(f"joint {i} transition @ {q[i]}")

            # phase III: force control
            if np.all(self.joint_transition):
                if self.phase != ControllerPhase.FORCE_CTRL:
                    self.phase = ControllerPhase.FORCE_CTRL
                    if self.verbose: print("transition to force control!")

                ''' from: https://github.com/llach/force_controller_core/blob/master/src/force_controller.cpp
                  // calculate new desired position
                  delta_F_ = target_force_ - *force_;
                  double delta_q_force = (delta_F_ / k_);

                  error_integral_ += delta_q_force * dt;
                  delta_q_ = K_p_ * delta_q_force + K_i_ * error_integral_;

                  // calculate new position and velocity
                  q_des_ = q - delta_q_;
                '''

                # force delta → position delta
                delta_f = fgoal - f
                delta_q = delta_f / self.k

                # integrate error TODO clip error integral?
                self.error_integral += delta_q * self.env.dt
                delta_q_ += self.Kp * delta_q + self.Ki * self.error_integral

                delta_qs[i] = -delta_q_

        # equally distribute position delta
        if self.phase == ControllerPhase.FORCE_CTRL: delta_qs = np.array(2*[-delta_q_/2])

        # if self.phase == ControllerPhase.POSITION_CTRL:
        #     print("fc", delta_qs)
        # if self.phase == ControllerPhase.FORCE_CTRL:
        #     print("fc", delta_qs)

        return delta_qs
    
    def predict(self, q, force, fgoal, *args, **kwargs):
        """
        interface to be compatible with stable baselines' API
        """
        deltaq = self.get_q(q, force, fgoal)
        return self._deltaq_to_qdes(q, deltaq), {}
