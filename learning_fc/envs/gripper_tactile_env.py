import mujoco
import numpy as np

from learning_fc import safe_rescale
from learning_fc.envs import GripperEnv
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperTactileEnv(GripperEnv):
    # constants
    INITIAL_OBJECT_POS  = np.array([0, 0, 0.05])
    INITIAL_OBJECT_SIZE = np.array([0.02, 0.05])
    
    SOLREF = [0.02, 1] # default: [0.02, 1]
    SOLIMP = [0.5, 0.95, 0.0066, 0.1, 2] # default: [0.9, 0.95, 0.001, 0.5, 2] [0, 0.95, 0.01, 0.5, 2] 

    def __init__(
            self,      
            fgoal_range=[0.05, 0.9], 
            wo_range=[0.01, 0.035], 
            oy_init=None, 
            xi_max=0.005,
            rf_scale=1.0, 
            ro_scale=100.0, 
            control_mode=ControlMode.Position, 
            obs_config=ObsConfig.F_DF, 
            **kwargs
        ):
        self.oy_init = oy_init          # object position. None → sampling
        self.xi_max   = xi_max          # maximum position error to reach fmax
        self.rf_scale = rf_scale        # scaling factor for force reward
        self.ro_scale = ro_scale        # scaling factor for objet movement penalty
        self.wo_range = wo_range        # sampling range for object width
        self.fgoal_range = fgoal_range  # sampling range for fgoal

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

        super()._update_state()

        # force deltas to goal
        self.force_deltas = self.fgoal - self.force

    def _get_obs(self):
        """ concatenate internal state as observation
        """
        _obs = super()._get_obs()
        return np.concatenate([
            _obs,
            safe_rescale(self.force_deltas, [-self.fgoal, self.fgoal]) if Observation.ForceDelta in self.obs_config else [],
        ])
    
    def _object_pos_penalty(self):
        self.total_object_movement += np.abs(self.obj_v[1])
        return self.total_object_movement
    
    def _force_reward(self):
        deltaf = self.fgoal - self.force
        
        rforce = 0
        for df in deltaf:
            if df <= 0: # overshooting
                rforce += 1-(np.clip(np.abs(df), 0.0, self.frange_upper)/self.frange_upper)
            elif df > 0:
                rforce += 1-(np.clip(df, 0.0, self.frange_lower)/self.frange_lower)
        return self.rf_scale*rforce

    def _get_reward(self):
        self.r_force   = self.rf_scale * self._force_reward()
        self.r_obj_pos = self.ro_scale * self._object_pos_penalty()

        return self.r_force - self.r_obj_pos
    
    def _is_done(self): return False

    def _reset_model(self, root):
        """ reset data, set joints to initial positions and randomize
        """

        #-----------------------------
        # object parameter variation
        self.wo = round(np.random.uniform(*self.wo_range), 3)

        if self.oy_init is None:
            # sampling constraints
            oy_q_const  = (0.97*0.045)-self.wo      # qmax-wo
            oy_xi_const = self.wo - self.xi_max
            oy_abs = min(oy_q_const, oy_xi_const)

            self.oy = round(np.random.uniform(-oy_abs, oy_abs), 3)
        else: self.oy = self.oy_init
            
        self.obj_pos    = self.INITIAL_OBJECT_POS.copy()
        self.obj_pos[1] = self.oy

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos'] = ' '.join(map(str, self.obj_pos))

        objgeom = obj.findall(".//geom")[0]
        objgeom.attrib['solimp'] = ' '.join(map(str, self.solimp))

        object_dims    = self.INITIAL_OBJECT_SIZE.copy()
        object_dims[0] = self.wo
        objgeom.attrib["size"] = ' '.join(map(str, object_dims))
        
        assert np.abs(self.wo) > np.abs(self.oy), "|wo| > |oy|"
        self.total_object_movement = 0

        # sample goal force
        self.set_goal(round(np.random.uniform(*self.fgoal_range), 3))

    def set_goal(self, x): 
        # set goal force and calculate interval sizes above and below goal force
        self.fgoal = x
        self.frange_upper = self.fmax - self.fgoal
        self.frange_lower = self.fgoal # fmin is 0

    def set_solver_parameters(self, solimp=None, solref=None):
        """ see https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        """
        self.solimp = solimp
        self.solref = solref

    def get_goal(self): return self.fgoal