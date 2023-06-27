from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

ALG  = "ppo"
OBS  = ObsConfig.F_DF
CTRL = ControlMode.PositionDelta

for k in range(3):
    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        env_kw=dict(control_mode=CTRL, obs_config=OBS, obj_pos_range=[-0.018, 0.018], ro_scale=10), frame_stack=k+1,     
        # train_kw=dict(timesteps=2e3) # quick training for debugging
        )