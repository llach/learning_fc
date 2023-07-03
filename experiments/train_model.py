from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

ALG  = "ppo"
OBS  = ObsConfig.Q_F_DF
CTRL = ControlMode.Position

train(
    env_name="gripper_tactile", 
    model_name=ALG,
    nenv=1,
    train_kw=dict(timesteps=7e5),
    env_kw=dict(control_mode=CTRL, obs_config=OBS, ro_scale=0), frame_stack=1,     
    # train_kw=dict(timesteps=2e3) # quick training for debugging
    )