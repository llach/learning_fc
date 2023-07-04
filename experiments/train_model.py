from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

ALG  = "ppo"
OBS  = ObsConfig.Q_F_DF
CTRL = ControlMode.Position

if __name__ == "__main__": 
    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        nenv=6,
        train_kw=dict(timesteps=1e6),
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS, 
            ro_scale=0, 
            oy_init=0,
            wo_range=[0.025, 0.025]   
        ), 
        frame_stack=1,     
        # train_kw=dict(timesteps=2e3) # quick training for debugging
        )