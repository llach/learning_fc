import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from .gripper_env import GripperEnv
from learning_fc import safe_rescale
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperPosEnv(GripperEnv):

    def __init__(self, 
                 obs_config=ObsConfig.Q_DQ, 
                 max_steps=50, 
                 ra_scale=0.0,
                 rv_scale=0.0,
                 rp_scale=1.0,
                 control_mode=ControlMode.Position, 
                 **kwargs
        ):
        self.rp_scale = rp_scale
        self.ra_scale = ra_scale
        self.rv_scale = rv_scale

        self.qgoal_range    = [0.0, 0.045]

        GripperEnv.__init__(
            self,
            obs_config=obs_config,
            control_mode=control_mode,
            **kwargs,
        )

    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        super()._update_state()
        self.q_deltas = self.qgoal - self.q

    def _get_obs(self):
        """ concatenate internal state as observation
        """ 
        _obs = super()._get_obs()
        return np.concatenate([
            _obs,
            safe_rescale(self.q_deltas, [-0.045, 0.045]) if Observation.PosDelta in self.obs_config else [],
        ])
    
    def _get_reward(self):
        delta  = max(self.qgoal_range[1]-self.qgoal, self.qgoal-self.qgoal_range[0])
        deltaq = np.abs(self.qgoal - self.q)
        
        self.r_pos  =   self.rp_scale * np.sum(1-np.round((deltaq/delta), 3))
        self.r_act  = - self.ra_scale * self._action_penalty()
        self.r_vel  = - self.rv_scale * self._qdot_penalty()

        return self.r_pos + self.r_act + self.r_vel
    
    def _is_done(self): return False

    def _reset_model(self, root):
        """ reset data, set joints to initial positions and randomize
        """

        # remove object from scene
        wb = root.findall(".//worldbody")[0]
        obj = root.findall(".//body[@name='object']")[0]
        wb.remove(obj)

        # sample goal position
        self.qgoal = round(np.random.uniform(*self.qgoal_range), 4)
    
    def reset_model(self):
        super().reset_model()

        self.data.joint("finger_joint_l").qpos = round(np.random.uniform(*self.qgoal_range), 4)
        self.data.joint("finger_joint_r").qpos = round(np.random.uniform(*self.qgoal_range), 4)

        self._update_state()
        return self._get_obs() 
    
    def get_goal(self): return self.qgoal
    def set_goal(self, g): self.qgoal = g