from learning_fc import model_path
from learning_fc.enums import ControlMode, ObsConfig, Observation as Obs
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = ObsConfig.Q_VEL_F_DF_IN_HAD_ACT
CTRL = ControlMode.PositionDelta
TIME = int(2e6)


# ro_schedule = ParamSchedule(
#     var_name="ro_scale",
#     start=0.0, 
#     stop=0.6,
#     first_value=1.0,
#     final_value=10.0,
#     total_timesteps=TIME
# )

ra_schedule = ParamSchedule(
    var_name="ra_scale",
    start=0.0, 
    stop=0.6,
    first_value=1.0,
    final_value=5.0,
    total_timesteps=TIME
)


if __name__ == "__main__": 
    for rfs in [3]:
        train(
            env_name="gripper_tactile", 
            model_name=ALG,
            nenv=6,
            train_kw=dict(timesteps=TIME),
            max_steps=200,
            env_kw=dict(
                control_mode=CTRL, 
                obs_config=OBS, 
                co_scale=1,
                rp_scale=1,
                ro_scale=10,
                rf_scale=rfs,
                oy_init=0.02,
                wo_range=[0.025, 0.025]   
            ), 
            frame_stack=3,
            weights=f"{model_path}/_archive/2023-07-15_11-53-31__gripper_tactile__ppo__no_move__nenv-6__k-3/weights/_best_model.zip",
            schedules=[
                ra_schedule
            ]
        )