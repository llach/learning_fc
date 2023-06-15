import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium.spaces import Box
from .gripper_env import GripperEnv
from learning_fc import safe_rescale


class GripperPosEnv(GripperEnv):

    def __init__(self, max_steps=50, eps=0.0005, **kwargs):
        self.eps = eps              # radius ε for rewards with fixed ε
        self.max_steps = max_steps  # #steps to terminate after  

        self.qgoal_range    = [0.0, 0.045]

        observation_space = Box(low=-1, high=1, shape=(6,), dtype=np.float64)

        GripperEnv.__init__(
            self,
            model_path="/Users/llach/repos/tiago_mj/force_gripper.xml",
            observation_space=observation_space,
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
        return np.concatenate([
            safe_rescale(self.q, [0.0, 0.045]),
            safe_rescale(self.q_deltas, [-0.045, 0.045]),
            safe_rescale(self.qdot, [-self.vmax, self.vmax]),
        ])
    
    # # variant I) continuous reward - no restrictions
    # def _get_reward(self):
    #     return -np.sum(np.abs(self.qgoal - self.q))
    
    # # variant II.1) continuous reward inside ε-environment 
    # # DON'T USE, THIS ONE HAS A LOGIC ERROR
    # def _get_reward(self):
    #     deltaq = np.abs(self.qgoal - self.q)
    #     if not np.all(deltaq<self.eps): return 0

    #     # reward is super small since ε is, make it bigger with 1e5
    #     return np.sum(np.abs(self.qgoal - self.q))*1e5

    # # variant II.2) continuous reward inside ε-environment, normalized
    # def _get_reward(self):
        # deltaq = np.abs(self.qgoal - self.q)
        # if not np.all(deltaq<self.eps): return 0

        # return np.sum(1-(deltaq/self.eps))

    # variant III) continuous reward normalized inside ε-environment & velocity penalty
    # def _get_reward(self):
    #     vnorm    = np.clip(np.abs(self.qdot), 0, self.vmax)/self.vmax
    #     vpenalty = np.e**(self.valpha*((vnorm-1)))
    #     vpenalty = np.sum(vpenalty)

    #     deltaq = np.abs(self.qgoal - self.q)
    #     if not np.all(deltaq<self.eps): return -0.1*vpenalty

    #     posreward = np.sum(1-(deltaq/self.eps))

    #     return posreward - 2*vpenalty
    
    # IV.1) no ε-env, exponential velocity penalty
    # def _get_reward(self):
    #     vnorm    = np.clip(np.abs(self.qdot), 0, self.vmax)/self.vmax
    #     vpenalty = np.e**(self.valpha*((vnorm-1)))
    #     vpenalty = np.sum(vpenalty)

    #     delta  = max(self.qgoal_range[1]-self.qgoal, self.qgoal-self.qgoal_range[0])
    #     deltaq = np.abs(self.qgoal - self.q)
    #     posreward = np.sum(1-(deltaq/delta))

    #     return posreward - vpenalty
    
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
    
    def set_goal(self, g): self.qgoal = g