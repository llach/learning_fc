import mujoco
import numpy as np

from learning_fc import safe_rescale, interp
from learning_fc.envs import GripperEnv
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperTactileEnv(GripperEnv):
    
    # constants
    QY_SGN_l = -1
    QY_SGN_r =  1
    INITIAL_OBJECT_POS  = np.array([0, 0, 0.05])
    INITIAL_OBJECT_SIZE = np.array([0.02, 0.05])

    OBJ_V_MAX = 0.0025
    
    SOLREF = [0.02, 1.0]
    SOLIMP = [0.0, 0.99, 0.01, 0.5, 2]

    # SOLREF_HARD = [0.008, 0.9]
    # SOLREF_SOFT = [0.025, 1.1]

    # SOLREF_RANGE = (
    #     [0.008, 0.9],   # minimum parameter values
    #     [0.025, 1.1]     # maximum parameter values
    # ) # sampling range for solref parameters

    SOLIMP_HARD = [0.00, 0.99, 0.002, 0.5, 2]
    SOLIMP_SOFT = [0.00, 0.99, 0.01,  0.5, 2]

    WIDTH_RANGE = [0.003, 0.01]

    BIASPRM = [0, -100, -9]

    BIASPRM_RANGE = (
        [0, -100, -13],
        [0, -100, -6]
    )

    M_RANGE = [0.5, 5]

    FGOAL_MIN_RANGE = [0.01, 0.16]
    FGOAL_MAX_RANGE = [0.47, 0.92]

    def __init__(
            self,      
            fgoal_range=[0.05, 1.0], 
            wo_range=[0.01, 0.035], 
            oy_init=None, 
            oy_range=None,
            xi_max=0.006,
            rf_scale=1.0, 
            ro_scale=1.0, 
            ra_scale=0.0,
            rp_scale=0.0, 
            ov_max=0.0001,
            sample_biasprm = False,
            randomize_stiffness = False,
            control_mode=ControlMode.Position, 
            obs_config=ObsConfig.F_DF,
            **kwargs
        ):
        self.ov_max   = ov_max
        self.xi_max   = xi_max          # maximum position error to reach fmax
        self.rf_scale = rf_scale        # scaling factor for force reward
        self.ro_scale = ro_scale        # scaling factor for object movement penalty
        self.ra_scale = ra_scale        # scaling factor for action difference penalty
        self.rp_scale = rp_scale        # scaling factor for object proximity
        self.oy_range = oy_range        # sampling range for object width
        self.wo_range = wo_range        # sampling range for object width
        self.fgoal_range = fgoal_range  # sampling range for fgoal
        self.randomize_stiffness = randomize_stiffness
        self.sample_biasprm = sample_biasprm

        if oy_init is not None:
            self.oy_range = [oy_init, oy_init]

        # solver parameters that control object deformation and contact force behavior
        self.solref = self.SOLREF
        self.solimp = self.SOLIMP

        # actuator parameters
        self.biasprm = self.BIASPRM

        assert np.min(wo_range) >= 0.01 and np.max(wo_range)<=0.35, "wo_range has to be in [0.01, 0.035]"

        GripperEnv.__init__(
            self,
            obs_config=obs_config,
            control_mode=control_mode,
            **kwargs,
        )
        
        self.set_goal(0)
        self.change_stiffness(0.5)
        self.max_fmax = self.FMAX*self.M_RANGE[1]
        self.fgoal_range_max = [self.FGOAL_MIN_RANGE[0], self.FGOAL_MAX_RANGE[1]]

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
        return self.rforce(self.fgoal, self.force)
    
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

        return np.sum([_doi(self.q[0], self.QY_SGN_l), _doi(self.q[1], self.QY_SGN_r)])/2

    def _get_reward(self):
        self.r_force    =   self.rf_scale * self._force_reward()
        self.r_obj_prox =   self.rp_scale * self._object_proximity_reward()
        self.r_obj_pos  = - self.ro_scale * self._object_pos_penalty()
        self.r_act      = - self.ra_scale * self._action_penalty()

        return self.r_force + self.r_obj_pos + self.r_obj_prox  + self.r_act
    
    def _is_done(self): return False

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

        if self.randomize_stiffness: self.change_stiffness(np.random.uniform(0,1))

        objgeom = obj.findall(".//geom")[0]
        objgeom.attrib['solimp'] = ' '.join(map(str, self.solimp))
        objgeom.attrib['solref'] = ' '.join(map(str, self.solref))

        object_dims    = self.INITIAL_OBJECT_SIZE.copy()
        object_dims[0] = self.wo
        objgeom.attrib["size"] = ' '.join(map(str, object_dims))
        
        assert np.abs(self.wo) > np.abs(self.oy), "|wo| > |oy|"
        self.total_object_movement = 0

        act_default = root.findall(".//general[@dyntype='none']")[0]
        act_default.attrib["biasprm"] = " ".join(map(str, self.biasprm))
        
        self.set_goal(round(np.random.uniform(*self.fgoal_range), 3))

        self.d_o = 0.045-(self.wo-np.abs(self.oy))

    def rforce(self, fgoal, forces):
        total_deltaf = np.sum(np.abs(fgoal - forces))
        return 1 - np.tanh(total_deltaf)

    def set_goal(self, x): self.fgoal = x

    def set_solver_parameters(self, solimp=None, solref=None):
        """ see https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        """
        if solimp is not None: self.solimp = solimp
        if solref is not None: self.solref = solref

    def change_stiffness(self, kappa):
        # assert 0 <= kappa and kappa <= 1, "0 <= kappa and kappa <= 1"
        self.kappa = kappa # 0 is hard, 1 is soft

        self.f_m = interp(1-kappa, self.M_RANGE)
        self.fmax = self._get_f(self.FMAX)

        self.sol_width = interp(kappa, self.WIDTH_RANGE)
        self.solimp[2] = self.sol_width

        self.fgoal_min = self.FGOAL_MIN_RANGE[0]
        self.fgoal_max = 0.97*self.fmax

        self.fgoal_range = [self.FGOAL_MIN_RANGE[0], 0.97*self.fmax]

    def get_goal(self): return self.fgoal