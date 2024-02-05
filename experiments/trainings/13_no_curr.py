from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos,
    Observation.Force, 
    Observation.ForceDelta,
    Observation.HadCon,
    Observation.Action
]
CTRL = ControlMode.PositionDelta
TIME = int(3.5e6)

ekw = dict(
    control_mode=CTRL, 
    obs_config=OBS, 
    ov_max=0.00005,
    ro_scale=2.0,
    ra_scale=0.2,
    rp_scale=0.0,
    rf_scale=1.0,
    wo_range=[0.015, 0.035],
    oy_range=[-0.040, 0.040],
    sample_biasprm=False,
    randomize_stiffness=False,
    noise_f=0.002,
    fth=0.02,
    model_path="assets/pal_force.xml",
)

if __name__ == "__main__":
    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        nenv=10,
        max_steps=150,
        train_kw=dict(
            timesteps=TIME,
        ),
        env_kw=ekw, 
        model_kw=dict(
            learning_rate=6e-4,
            policy_kwargs=dict(
                net_arch=[50,50]
            )
        ),
        frame_stack=3,
    )