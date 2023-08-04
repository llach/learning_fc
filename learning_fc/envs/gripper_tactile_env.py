import mujoco
import numpy as np

from learning_fc import safe_rescale
from learning_fc.envs import GripperEnv
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperTactileEnv(GripperEnv):
    
    # constants
    QY_SGN_l = -1
    QY_SGN_r =  1
    INITIAL_OBJECT_POS  = np.array([0, 0, 0.05])
    INITIAL_OBJECT_SIZE = np.array([0.02, 0.05])

    OBJ_V_MAX = 0.0025
    
    SOLREF = [0.02, 1] # default: [0.02, 1]
    SOLIMP = [0.5, 0.95, 0.0066, 0.1, 2] # default: [0.9, 0.95, 0.001, 0.5, 2] [0, 0.95, 0.01, 0.5, 2] 

    SOLREF_RANGE = (
        [0.008, 0.8],   # minimum parameter values
        [0.05, 1.1]     # maximum parameter values
    ) # sampling range for solref parameters

    SOLIMP_RANGE = (
        [0.4, 0.9,  0.0,   0.0,  2],
        [0.9, 0.95, 0.015, 0.55, 2] 
    )# sampling range for solimp parameters

    def __init__(
            self,      
            fgoal_range=[0.05, 0.9], 
            wo_range=[0.01, 0.035], 
            oy_init=None, 
            oy_range=None,
            xi_max=0.005,
            rf_scale=1.0, 
            ro_scale=1.0, 
            ra_scale=0.0,
            rv_scale=0.0, 
            rp_scale=0.0, 
            co_scale=0.0,
            ov_max=0.0001,
            sample_solref = False,
            sample_solimp = False,
            control_mode=ControlMode.Position, 
            obs_config=ObsConfig.F_DF, 
            max_contact_steps=-1,
            **kwargs
        ):
        self.ov_max   = ov_max
        self.xi_max   = xi_max          # maximum position error to reach fmax
        self.rf_scale = rf_scale        # scaling factor for force reward
        self.ro_scale = ro_scale        # scaling factor for object movement penalty
        self.ra_scale = ra_scale        # scaling factor for action difference penalty
        self.rv_scale = rv_scale        # scaling factor for joint velocity penalty
        self.rp_scale = rp_scale        # scaling factor for object proximity
        self.co_scale = co_scale        # scaling factor for in-contact reward
        self.oy_range = oy_range        # sampling range for object width
        self.wo_range = wo_range        # sampling range for object width
        self.fgoal_range = fgoal_range  # sampling range for fgoal
        self.sample_solref = sample_solref  # toggle solref sampling
        self.sample_solimp = sample_solimp  # toggle solimp sampling
        self.max_contact_steps = max_contact_steps

        if oy_init is not None:
            self.oy_range = [oy_init, oy_init]

        # solver parameters that control object deformation and contact force behavior
        self.solref = self.SOLREF
        self.solimp = self.SOLIMP

        assert np.min(wo_range) >= 0.01 and np.max(wo_range)<=0.35, "wo_range has to be in [0.01, 0.035]"

        GripperEnv.__init__(
            self,
            obs_config=obs_config,
            control_mode=control_mode,
            **kwargs,
        )

        self.set_goal(0)

    def _update_state(self):
        """ updates internal state variables that may be used as observations
        """

        # object state
        obj_pos_t = self.data.joint("object_joint").qpos[:3]
        self.obj_v = obj_pos_t - self.obj_pos
        self.obj_pos = obj_pos_t.copy()
        self.oy_t = obj_pos_t[1]

        super()._update_state()

        # force deltas to goal
        self.force_deltas = self.fgoal - self.force
    
    def _object_pos_penalty(self):
        return 1 if np.abs(self.obj_v[1]) > self.ov_max else 0
    
    def _force_reward(self):
        total_deltaf = np.sum(np.abs(self.fgoal - self.force))
        return 1 - np.tanh(total_deltaf)
    
    def _contact_reward(self):
        return np.sum(self.in_contact)

    def _object_proximity_reward(self):
        """ fingers don't move towards the object sometimes → encourage them with small, positive rewards
        """

        def _doi(q, sgn):
            if np.sign(self.oy_t) == sgn:
                d = np.abs(self.oy_t) + self.wo
            else:
                d = self.wo - np.abs(self.oy_t)
            return 1-np.clip(q-d, 0.0, self.d_o)/self.d_o

        return np.sum([_doi(self.q[0], self.QY_SGN_l), _doi(self.q[1], self.QY_SGN_r)])

    def _get_reward(self):
        self.r_force    =   self.rf_scale * self._force_reward()
        self.r_con      =   self.co_scale * self._contact_reward()
        self.r_obj_prox =   self.rp_scale * self._object_proximity_reward()
        self.r_obj_pos  = - self.ro_scale * self._object_pos_penalty()
        self.r_act      = - self.ra_scale * self._action_penalty()
        self.r_qvel     = 0# - self.rv_scale * self._qdot_penalty()

        return self.r_force + self.r_obj_pos + self.r_con + self.r_obj_prox + self.r_qvel + self.r_act
    
    def _is_done(self): 
        if self.max_contact_steps != -1 and self.t_since_force_closure >= self.max_contact_steps: return True

    def _enum2obs(self, on):
        if on == Observation.ObjVel: return safe_rescale([self.obj_v[1]], [0.0, self.OBJ_V_MAX])

        return super()._enum2obs(on)

    def _reset_model(self, root):
        """ reset data, set joints to initial positions and randomize
        """

        #-----------------------------
        # object parameter variation
        self.wo = round(np.random.uniform(*self.wo_range), 3)

        # oy sampling constraints
        oy_q_const  = (0.97*0.045)-self.wo      # qmax-wo
        oy_xi_const = self.wo - 2*self.xi_max
        oy_max = min(oy_q_const, oy_xi_const)

        if self.oy_range is not None:
            oy_range = np.clip(self.oy_range, -oy_max, oy_max)
        else:
            oy_range = [-oy_max, oy_max]

        self.oy = round(np.random.uniform(*oy_range), 3)
            
        self.obj_pos    = self.INITIAL_OBJECT_POS.copy()
        self.obj_pos[1] = self.oy

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos'] = ' '.join(map(str, self.obj_pos))

        if self.sample_solref: self.solref = np.random.uniform(*self.SOLREF_RANGE)
        if self.sample_solimp: self.solimp = np.random.uniform(*self.SOLIMP_RANGE)

        objgeom = obj.findall(".//geom")[0]
        objgeom.attrib['solimp'] = ' '.join(map(str, self.solimp))
        objgeom.attrib['solref'] = ' '.join(map(str, self.solref))

        object_dims    = self.INITIAL_OBJECT_SIZE.copy()
        object_dims[0] = self.wo
        objgeom.attrib["size"] = ' '.join(map(str, object_dims))
        
        assert np.abs(self.wo) > np.abs(self.oy), "|wo| > |oy|"
        self.total_object_movement = 0

        # sample goal force
        self.set_goal(round(np.random.uniform(*self.fgoal_range), 3))

        self.d_o = 0.045-(self.wo-np.abs(self.oy))

    def set_goal(self, x): 
        # set goal force and calculate interval sizes above and below goal force
        self.fgoal = x
        self.frange_upper = self.fmax - self.fgoal
        self.frange_lower = self.fgoal # fmin is 0
        self.fram = min([self.frange_lower, self.frange_upper])

    def set_solver_parameters(self, solimp=None, solref=None):
        """ see https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        """
        if solimp is not None: self.solimp = solimp
        if solref is not None: self.solref = solref

    def get_goal(self): return self.fgoal