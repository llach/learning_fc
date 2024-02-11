import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

import learning_fc
from learning_fc import safe_rescale, get_pad_forces
from learning_fc.enums import ControlMode, Observation, ObsConfig

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "azimuth": 153,
    "distance": 0.33,
    "elevation": -49,
    "lookat": [-0.00099796, -0.00137387, 0.04537828]
}

VIDEO_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "azimuth": 0,
    "distance": 0.30,
    "elevation": -76,
    "lookat": [-0.00099796, -0.00137387, 0.04537828]
}

def obs_space_shape(obs_conf):
    def _obs2len(on):
        if on == Observation.ObjVel: return 1
        return 2
    return (np.sum([_obs2len(o) for o in obs_conf]),)

class GripperEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    FMAX = 0.315
    def __init__(
            self, 
            amax=1.0, 
            vmax=0.02, 
            dq_max=0.003,
            dq_min=0.0003, 
            f_scale=1.0,
            fth=0.05, 
            noise_q=0.000027,
            noise_f=0.013,
            with_bias=False,
            qinit_range=[0.045, 0.045], 
            obs_config=ObsConfig.Q_F_DF, 
            control_mode=ControlMode.Position, 
            model_path="assets/franka_force.xml", 
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        ):
        self.amax = amax        # maximum acceleration 
        self.vmax = vmax        # maximum joint velocity
        self.dq_min = dq_min    # limits of action space for position delta control mode
        self.dq_max = dq_max    # limits of action space for position delta control mode
        self.fth = fth    # contact force noise threshold
        self.f_scale = f_scale  # force scaling factor
        self.noise_q = noise_q  # std of normally distributed noise added to joint positions
        self.noise_f = noise_f  # std of normally distributed noise added to contact forces
        self.obs_config = obs_config    # contents of observation space    
        self.with_bias = with_bias
        self.qinit_range = qinit_range
        self.control_mode = control_mode

        self.ain = np.array([0, 0])

        observation_space = Box(
            low=np.float32(-1.), 
            high=np.float32(1.), 
            shape=obs_space_shape(obs_conf=obs_config), 
            dtype=np.float32
        )

        utils.EzPickle.__init__(self, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=learning_fc.__path__[0]+f"/{model_path}",
            frame_skip=20,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
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
            low   = np.array([-1, -1], dtype=np.float32), 
            high  = np.array([ 1,  1], dtype=np.float32), 
            dtype = np.float32
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
            ain = safe_rescale(ain, [-1, 1], [-self.dq_max, self.dq_max])
            self.qdes = np.clip(self.q+ain, 0, 0.045)
        else:
            assert False, f"unknown control mode {self.control_mode}"
        
        # motors can't realize arbitrarily small position deltas, so we emulate it here
        # self.qdes = np.where(np.abs(self.q-self.qdes)>self.dq_min, self.qdes, self.q) 

        # create action array, insert gripper actions at proper indices
        aout = np.zeros_like(self.data.ctrl)
        aout[self.data.actuator("finger_left").id]  = self.qdes[0]
        aout[self.data.actuator("finger_right").id] = self.qdes[1]

        return aout
    
    def _get_f(self, f):
        return self.f_m * f
    
    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """
        ### update relevant robot state variables 

        # joint states
        self.q = np.array([
            self.data.joint("finger_joint_l").qpos[0],
            self.data.joint("finger_joint_r").qpos[0]
        ]) + np.random.normal(0.0, self.noise_q, (2,))
        self.qacc = np.array([
            self.data.joint("finger_joint_l").qacc[0],
            self.data.joint("finger_joint_r").qacc[0]
        ]) 
        self.qdot = (self.last_q - self.q)/self.dt
        
        # contact force and force change
        self.force = self._get_f(get_pad_forces(self.model, self.data)) + np.random.normal(0.0, self.noise_f, (2,))
        self.fdot = (self.last_f - self.force)/self.dt
        
        # binary contact states
        self.in_contact  = self.force > self.fth
        self.had_contact = self.in_contact | self.had_contact

    def _enum2obs(self, on):
        if on == Observation.Pos: return safe_rescale(self.q, [0.0, 0.045])
        if on == Observation.Des: return safe_rescale(self.qdes, [0.0, 0.045])
        if on == Observation.Vel: return safe_rescale(self.qdot, [-self.vmax, self.vmax])
        if on == Observation.Force: return self.force#safe_rescale(self.force, [0, self.max_fmax])
        if on == Observation.FDot: return self.fdot 
        if on == Observation.Action: return self.ain
        if on == Observation.PosDelta: return safe_rescale(self.q_deltas, [-0.045, 0.045])
        if on == Observation.ForceDelta: return self.force_deltas#safe_rescale(self.force_deltas, [-self.fgoal, self.fgoal])
        if on == Observation.InCon: return self.in_contact
        if on == Observation.HadCon: return self.had_contact

        assert False, f"unknown Observation {on}"

    def _get_obs(self):
        """ concatenate internal state as observation
        """

        obs = []
        for on in self.obs_config: obs.append(self._enum2obs(on))
        return np.concatenate(obs).astype(np.float32)
    
    def _qdot_penalty(self):
        vnorm = np.clip(np.abs(self.qdot), 0, self.vmax)/self.vmax
        return np.sum(vnorm)
    
    def _action_penalty(self):
        return np.sum(np.abs(self.last_a - self.ain))
    
    def action_bias(self, h, force, fgoal):
        if h[0] == 0 and h[1] == 0: return np.array([1, 1]) # no contact → policy has full control
        if h[0] == 1 and h[1] == 1:                         # full contact → scale actions for safety
            dfs = 1-np.abs((fgoal-force)/fgoal)
            return np.clip(dfs, 0.7, 1.0)
        return np.array([0 if hi else 1 for hi in h])

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

        self.qdes = np.array([self.qinit_l, self.qinit_r])
        self.ain = np.array([0, 0])
        self.had_contact = np.array([0, 0], dtype=bool)

        self.last_a = np.array([0, 0])
        self.last_q = np.array([self.qinit_l, self.qinit_r])
        self.last_f = np.array([0, 0])
        
        self.t = 0
        self.t_since_force_closure = 0

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
        if type(a)==list: a=np.array(a)
        
        if self.t == 0: 
            self.last_a = a.copy() # avoid penalty on first timestep

        self.ain = a.copy()
        if self.with_bias:
            self.ain = self.action_bias(self.had_contact, self.force, self.fgoal) * self.ain
        
        # `self.do_simulation` invovled an action space shape check that this environment won't pass due to underactuation
        self._step_mujoco_simulation(self._make_action(self.ain), self.frame_skip)
        if self.render_mode == "human":
            self.render()

        if np.all(self.had_contact): self.t_since_force_closure += 1

        # update internal state variables
        self._update_state()

        obs = self._get_obs()
        rew = self._get_reward()
        don = self._is_done()  # terminated

        self.t += 1

        self.last_a = a.copy()
        self.last_q = self.q.copy()
        self.last_f = self.force.copy()

        return (
            obs,
            rew,
            don,
            False,  # truncated
            {},     # info
        )
    
if __name__ == "__main__": env = GripperEnv()