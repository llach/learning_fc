import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

import learning_fc
from learning_fc import safe_rescale, get_pad_forces
from learning_fc.enums import ControlMode, Observation

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "azimuth": 153,
    "distance": 0.33,
    "elevation": -49,
    "lookat": [-0.00099796, -0.00137387, 0.04537828]
}

class GripperEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, obs_config, model_path=learning_fc.__path__[0]+"/assets/franka_force.xml", rqdot_scale=0.0, vmax=0.02, amax=1.0, qinit_range=[0.045, 0.045], fmax=1.0, ftheta=0.001, control_mode=ControlMode.Position, **kwargs):
        self.amax = amax        # maximum acceleration 
        self.vmax = vmax        # maximum joint velocity
        self.fmax = fmax        # maximum contact force
        self.ftheta = ftheta    # contact force noise threshold
        self.obs_config = obs_config    # contents of observation space    
        self.rqdot_scale = rqdot_scale  # scaling factor for qdot penalty
        self.qinit_range = qinit_range
        self.control_mode = control_mode

        observation_space = Box(low=-1, high=1, shape=(2*len(obs_config),), dtype=np.float64)

        utils.EzPickle.__init__(self, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=10,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
    
    def _name_2_qpos_id(self, name):
        """ given a joint name, return their `qpos`-array address
        """
        jid =self.data.joint(name).id
        return self.model.jnt_qposadr[jid]
    
    def _set_action_space(self):
        """ torso joint is ignored, this env is for gripper behavior only
        """
        self.action_space = Box(
            low  = np.array([-1, -1]), 
            high = np.array([1, 1]), 
            dtype=np.float32
        )
        return self.action_space
    
    def _make_action(self, ain):
        """ 
        * limits joint to vmax by contraining the maximum position delta applied
        * creates full `data.ctrl`-compatible array even though some joints are not actuated 
        """
        # transform actions to qdes depending on control mode
        if self.control_mode == ControlMode.Position:
            ain = safe_rescale(ain, [-1, 1], [0.0, 0.045])
            self.qdes = ain
        elif self.control_mode == ControlMode.PositionDelta:
            ain = safe_rescale(ain, [-1, 1], [-0.045, 0.045])
            self.qdes = np.clip(self.q+ain, 0, 0.045)
        else:
            assert False, f"unknown control mode {self.control_mode}"

        # print(self.q, self.qdes)

        # create action array, insert gripper actions at proper indices
        aout = np.zeros_like(self.data.ctrl)
        aout[self.data.actuator("finger_left").id]  = self.qdes[0]
        aout[self.data.actuator("finger_right").id] = self.qdes[1]

        return aout
    
    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        ### update relevant robot state variables 

        # joint states
        self.q = np.array([
            self.data.joint("finger_joint_l").qpos[0],
            self.data.joint("finger_joint_r").qpos[0]
        ])
        self.qdot = np.array([
            self.data.joint("finger_joint_l").qvel[0],
            self.data.joint("finger_joint_r").qvel[0]
        ])
        self.qacc = np.array([
            self.data.joint("finger_joint_l").qacc[0],
            self.data.joint("finger_joint_r").qacc[0]
        ])

        # contact force and binary in_contact state
        self.force = get_pad_forces(self.model, self.data)
        self.in_contact = self.force > self.ftheta

    def _get_obs(self):
        """ concatenate internal state as observation
        """
        return np.concatenate([
                safe_rescale(self.q,     [0, 0.045]) if Observation.Pos in self.obs_config else [], 
                safe_rescale(self.force, [0, self.fmax]) if Observation.Force in self.obs_config else [], 
                safe_rescale(self.qdot, [-self.vmax, self.vmax]) if Observation.Vel in self.obs_config else [],
                safe_rescale(self.qacc, [-self.amax, self.amax]) if Observation.Acc in self.obs_config else [],
            ])
    
    def _qdot_penalty(self):
        vnorm = np.clip(np.abs(self.qdot), 0, self.vmax)/self.vmax
        return self.rqdot_scale*np.sum(vnorm)

    def _is_done(self): raise NotImplementedError
    def _get_reward(self): raise NotImplementedError
    def _reset_model(self): raise NotImplementedError
    
    def get_goal(self): raise NotImplementedError
    def set_goal(self, g): raise NotImplementedError

    def reset_model(self):
        """ reset data, set joints to initial positions and randomize
        """

        # initial joint positions need to be known before model reset in child env
        self.qinit_l = round(np.random.uniform(*self.qinit_range), 3)
        self.qinit_r = round(np.random.uniform(*self.qinit_range), 3)

        xmlmodel = ET.parse(self.fullpath)
        root = xmlmodel.getroot()

        root.find("compiler").attrib["meshdir"] = learning_fc.__path__[0]+"/assets/meshes/"

        # create model from modified XML
        self._reset_model(root)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # load data, set starting joint values (open gripper)
        self.data  = mujoco.MjData(self.model)
        self.data.qpos[self._name_2_qpos_id("finger_joint_l")] = self.qinit_l
        self.data.qpos[self._name_2_qpos_id("finger_joint_r")] = self.qinit_r

        # update renderer's pointers, otherwise scene will be empty
        self.mujoco_renderer.model = self.model
        self.mujoco_renderer.data  = self.data
        
        # viewers' models also need to be updated 
        if len(self.mujoco_renderer._viewers)>0:
            for _, v in self.mujoco_renderer._viewers.items():
                v.model = self.model
                v.data = self.data

        self._update_state()
        return self._get_obs()

    def step(self, a):
        """
        action: [q_left, q_right]

        returns:
            observations
            reward
            terminated
            truncated
            info
        """
        
        # `self.do_simulation` invovled an action space shape check that this environment won't pass due to underactuation
        self._step_mujoco_simulation(self._make_action(a), self.frame_skip)
        if self.render_mode == "human":
            self.render()

        # update internal state variables
        self._update_state()
        
        return (
            self._get_obs(),
            self._get_reward(),
            self._is_done(),  # terminated
            False,  # truncated
            {},     # info
        )
    