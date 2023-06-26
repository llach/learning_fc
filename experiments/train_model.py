from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

for alg in ["ppo"]:
    for ctrl in [ControlMode.PositionDelta]:
        for obs in [ObsConfig.F_DF]:
            train(
                env_name="gripper_tactile", 
                model_name=alg,
                env_kw=dict(control_mode=ctrl, obs_config=obs, obj_pos_range=[-0.018, 0.018]),     
                # train_kw=dict(timesteps=2e3) # quick training for debugging
                )