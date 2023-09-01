import learning_fc

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
TIME = int(2.5e6)

ekw = dict(
    control_mode=CTRL, 
    obs_config=OBS, 
    ro_scale=0,
    ra_scale=0,
    rp_scale=0.1,
    rf_scale=0.9,
    sample_biasprm=True,
    randomize_stiffness=True,
    noise_f=0.002,
    fth=0.0075,
    model_path="assets/pal_force.xml",
)

if __name__ == "__main__":
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
            learning_rate=6e-4,
            policy_kwargs=dict(
                net_arch=[30,30]
            )
        ),
        frame_stack=1,
    )