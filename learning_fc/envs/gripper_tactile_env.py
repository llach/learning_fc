import mujoco
import numpy as np

from learning_fc import safe_rescale
from learning_fc.envs import GripperEnv
from learning_fc.enums import ControlMode, ObsConfig, Observation


class GripperTactileEnv(GripperEnv):

    # constants
    QY_SGN_l =  1
    QY_SGN_r = -1
    
    INITIAL_OBJECT_POS  = np.array([0, 0, 0.05])
    INITIAL_OBJECT_SIZE = np.array([0.02, 0.05])
    
    SOLREF = [0.02, 1] # default: [0.02, 1]
    SOLIMP = [0, 0.95, 0.005, 0.5, 2] # default: [0.9, 0.95, 0.001, 0.5, 2] [0, 0.95, 0.01, 0.5, 2] 

    def __init__(
            self,      
            fgoal_range=[0.3, 1.5], 
            oy_range=[0, 0], 
            wo_range=[0.01, 0.035], 
            rf_scale=1.0, 
            ro_scale=100.0, 
            control_mode=ControlMode.Position, 
            obs_config=ObsConfig.F_DF, 
            **kwargs
        ):
        self.rf_scale = rf_scale        # scaling factor for force reward
        self.ro_scale = ro_scale        # scaling factor for objet movement penalty
        self.wo_range = wo_range        # sampling range for object width
        self.oy_range = oy_range        # sampling range for object position
        self.fgoal_range = fgoal_range  # sampling range for fgoal

        # solver parameters that control object deformation and contact force behavior
        self.solref = self.SOLREF
        self.solimp = self.SOLIMP

        assert min(wo_range) >= 0.01 and max(wo_range)<=0.35, "wo_range has to be in [0.01, 0.035]"

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
        super()._update_state()

        # force deltas to goal
        self.force_deltas = self.fgoal - self.force

        # object state
        obj_pos_t = self.data.joint("object_joint").qpos[:3]
        self.obj_v = obj_pos_t - self.obj_pos
        self.obj_pos = obj_pos_t.copy()

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
        # random object start 
        self.oy = round(np.random.uniform(*self.oy_range), 3) # object y position
        self.obj_pos    = self.INITIAL_OBJECT_POS.copy()
        self.obj_pos[1] = self.oy

        obj = root.findall(".//body[@name='object']")[0]
        obj.attrib['pos'] = ' '.join(map(str, self.obj_pos))

        objgeom = obj.findall(".//geom")[0]
        objgeom.attrib['solimp'] = ' '.join(map(str, self.solimp))
        
        # store object half-width (radius for cylinders)
        self.wo = round(np.random.uniform(*self.wo_range), 3)

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