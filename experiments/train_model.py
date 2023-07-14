from learning_fc.enums import ControlMode, ObsConfig, Observation as Obs
from learning_fc.training import train

ALG  = "ppo"
OBS  = ObsConfig.Q_VEL_F_DF_IN_HAD
CTRL = ControlMode.PositionDelta

if __name__ == "__main__": 
    for rfs in [3]:
        train(
            env_name="gripper_tactile", 
            model_name=ALG,
            nenv=6,
            train_kw=dict(timesteps=15e5),
            max_steps=200,
            env_kw=dict(
                control_mode=CTRL, 
                obs_config=OBS, 
                co_scale=1, 
                # ro_scale=4,
                rp_scale=1,
                rf_scale=rfs,
                # oy_init=-0.015,
                # wo_range=[0.025, 0.025]   
            ), 
            frame_stack=3,
            # train_kw=dict(timesteps=2e3) # quick training for debugging
            )