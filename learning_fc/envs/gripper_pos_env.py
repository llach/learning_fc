import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from .gripper_env import GripperEnv
from learning_fc import safe_rescale
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperPosEnv(GripperEnv):

    def __init__(self, obs_config=ObsConfig.Q_DQ, max_steps=50, eps=0.0005, control_mode=ControlMode.Position, **kwargs):
        self.eps = eps              # radius ε for rewards with fixed ε
        self.max_steps = max_steps  # #steps to terminate after  

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
    
    # IV.2) no ε-env, linear velocity penalty
    def _get_reward(self):
        delta  = max(self.qgoal_range[1]-self.qgoal, self.qgoal-self.qgoal_range[0])
        deltaq = np.abs(self.qgoal - self.q)
        posreward = np.sum(1-(deltaq/delta))

        return posreward - self._qdot_penalty()
    
    def _is_done(self): return False

    def _reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """
        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        # remove object from scene
        wb = root.findall(".//worldbody")[0]
        obj = root.findall(".//body[@name='object']")[0]
        wb.remove(obj)

        # sample goal position
        self.qgoal = round(np.random.uniform(*self.qgoal_range), 4)

        # create model from modified XML
        return mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))
    
    def reset_model(self):
        obs = super().reset_model()

        self.data.qpos[self._name_2_qpos_id("gripper_left_finger_joint")]  = round(np.random.uniform(*self.qgoal_range), 4)
        self.data.qpos[self._name_2_qpos_id("gripper_right_finger_joint")] = round(np.random.uniform(*self.qgoal_range), 4)

        return obs
    
    def get_goal(self): return self.qgoal
    def set_goal(self, g): self.qgoal = g