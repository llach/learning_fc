import mujoco
import numpy as np

from enum import Enum
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

from learning_fc import safe_rescale, total_contact_force

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 0.8,
    "azimuth": -160,
    "elevation": -45,
    "lookat": [0.006, 0.0, 0.518]
}

class ControlMode(str, Enum):
    Position="position"
    PositionDelta="position_delta"

class GripperEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, model_path, observation_space, rqdot_scale=0.0, vmax=0.02, amax=1.0, qinit_range=[0.045, 0.045], fmax=0.85, ftheta=0.05, control_mode=ControlMode.Position, **kwargs):
        self.amax = amax        # maximum acceleration 
        self.vmax = vmax        # maximum joint velocity
        self.fmax = fmax        # maximum contact force
        self.ftheta = ftheta    # contact force noise threshold
        self.rqdot_scale = rqdot_scale # scaling factor for qdot penalty
        self.qinit_range = qinit_range
        self.control_mode = control_mode

        utils.EzPickle.__init__(self, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=10,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # maximum position delta per timestep to not surpass max velocity
        self.dq_max = self.vmax*self.dt*5 # the 5 is needed to actually get to vmax

        # reload the model with environment randomization
        self.reset_model()
    
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

        # create action array, insert gripper actions at proper indices
        aout = np.zeros_like(self.data.ctrl)
        aout[self.data.actuator("gripper_left_finger_joint").id]  = self.qdes[0]
        aout[self.data.actuator("gripper_right_finger_joint").id] = self.qdes[1]

        return aout
    
    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        ### update relevant robot state variables 

        # joint states
        self.q = np.array([
            self.data.joint("gripper_left_finger_joint").qpos[0],
            self.data.joint("gripper_right_finger_joint").qpos[0]
        ])
        self.qdot = np.array([
            self.data.joint("gripper_left_finger_joint").qvel[0],
            self.data.joint("gripper_right_finger_joint").qvel[0]
        ])
        self.qacc = np.array([
            self.data.joint("gripper_left_finger_joint").qacc[0],
            self.data.joint("gripper_right_finger_joint").qacc[0]
        ])

        # contact force and binary in_contact state
        self.force = np.array([
            np.sum(np.abs(total_contact_force(self.model, self.data, "object", "left_finger_bb")[0])),
            np.sum(np.abs(total_contact_force(self.model, self.data, "object", "right_finger_bb")[0]))
        ])
        self.in_contact = self.force > self.ftheta

    def _get_obs(self):
        """ concatenate internal state as observation
        """
        return np.concatenate([
                safe_rescale(self.q,    [0.0,  0.045]), 
                safe_rescale(self.qdot, [-self.vmax, self.vmax]),
                safe_rescale(self.qacc, [-self.amax, self.amax])
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

        self.model = self._reset_model()

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # load data, set starting joint values (open gripper)
        self.data  = mujoco.MjData(self.model)
        self.data.qpos[self._name_2_qpos_id("gripper_left_finger_joint")]  = self.qinit_l
        self.data.qpos[self._name_2_qpos_id("gripper_right_finger_joint")] = self.qinit_r

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
    