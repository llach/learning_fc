from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.training import train

for alg in ["sac"]:#, "ppo"]:
    for ctrl in [ControlMode.Position, ControlMode.PositionDelta]:
        for obs in [ObsConfig.F_DF, ObsConfig.Q_F_DF]:

            train(
                env_name="gripper_tactile", 
                model_name=alg,
                env_kw=dict(control_mode=ctrl, obs_config=obs),     
                model_kw=dict(learning_starts=2e3)
                # train_kw=dict(timesteps=2e3) # quick training for debugging
            )