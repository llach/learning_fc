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

ekw = dict(
    control_mode=CTRL, 
    obs_config=OBS, 
    ro_scale=0,
    ra_scale=0,
    rp_scale=0.33,
    rf_scale=0.67,
    sample_solimp=True,
    sample_fscale=True,
    sample_biasprm=True,
    noise_f=0.002,
    ftheta=0.0075,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
)

if __name__ == "__main__":
    for k in np.arange(5):
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
            frame_stack=int(k+1), # not doing int() here fucks up json smh
        )