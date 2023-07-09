from learning_fc.enums import ControlMode, ObsConfig, Observation as Obs
from learning_fc.training import train

ALG  = "ppo"
OBS  = [Obs.Pos, Obs.Vel, Obs.PosDelta]
CTRL = ControlMode.PositionDelta

if __name__ == "__main__": 
    for rv in [
        2.0
    ]:
        train(
            env_name="gripper_pos", 
            model_name=ALG,
            nenv=6,
            train_kw=dict(timesteps=2e5),
            max_steps=50,
            env_kw=dict(
                control_mode=CTRL, 
                obs_config=OBS, 
                # co_scale=1, 
                # ro_scale=100,
                # rf_scale=0,
                rp_scale=1,
                # rv_scale=rv,'
                # oy_init=-0.015,
                # wo_range=[0.025, 0.025]   
            ), 
            model_kw=dict(
                # gamma=ga,
            ),
            frame_stack=1,
            # train_kw=dict(timesteps=2e3) # quick training for debugging
            )