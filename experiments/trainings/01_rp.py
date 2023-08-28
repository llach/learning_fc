import learning_fc
import numpy as np

from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train

ALG  = "ppo"
OBS  = [
    Observation.Pos,
    Observation.Des,
    Observation.Force,
    Observation.ForceDelta,
    Observation.Action,
]

CTRL = ControlMode.PositionDelta
TIME = int(2e6)

if __name__ == "__main__":
    for rp in np.linspace(0.0, 0.4, 10):
        ekw = dict(
            control_mode=CTRL, 
            obs_config=OBS, 
            ro_scale=0,
            ra_scale=0,
            rp_scale=rp,
            rf_scale=1-rp,
            sample_solimp=True,
            sample_fscale=True,
            sample_biasprm=True,
            noise_f=0.002,
            ftheta=0.0075,
            model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        )
        train(
            env_name="gripper_tactile", 
            model_name=ALG,
            nenv=10,
            max_steps=200,
            train_kw=dict(
                timesteps=TIME,
            ),
            env_kw=ekw, 
            model_kw=dict(
                policy_kwargs=dict(
                    net_arch=[80,80]
                )
            ),
            frame_stack=3,
        )