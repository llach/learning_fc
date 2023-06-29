from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

ALG  = "ppo"
OBS  = ObsConfig.DF
CTRL = ControlMode.PositionDelta


train(
    env_name="gripper_tactile", 
    model_name=ALG,
    env_kw=dict(control_mode=CTRL, obs_config=OBS, obj_pos_range=[0, 0], ro_scale=0), frame_stack=1,     
    # train_kw=dict(timesteps=2e3) # quick training for debugging
    )