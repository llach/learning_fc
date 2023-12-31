from learning_fc import model_path
from learning_fc.enums import ControlMode, ObsConfig, Observation as Obs
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = ObsConfig.Q_VEL_F_DF_IN_HAD_ACT
CTRL = ControlMode.PositionDelta
TIME = int(2e6)

if __name__ == "__main__": 

    ro_schedule = ParamSchedule(
        var_name="ro_scale",
        start=0.0, 
        stop=0.5,
        first_value=0.0,
        final_value=40.0,
        total_timesteps=TIME
    )

    ra_schedule = ParamSchedule(
        var_name="ra_scale",
        start=0.0, 
        stop=0.5,
        first_value=0.0,
        final_value=15.0,
        total_timesteps=TIME
    )

    rf_schedule = ParamSchedule(
        var_name="rf_scale",
        start=0.0, 
        stop=0.5,
        first_value=3.0,
        final_value=7.0,
        total_timesteps=TIME
    )

    wo_schedule = ParamSchedule(
        var_name="wo_range",
        start=0.0, 
        stop=0.5,
        first_value=[0.020, 0.025],
        final_value=[0.015, 0.035],
        total_timesteps=TIME
    )

    oy_schedule = ParamSchedule(
        var_name="oy_range",
        start=0.0, 
        stop=0.5,
        first_value=[0.0, 0.0],
        final_value=[-0.040, 0.040],
        total_timesteps=TIME
    )

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
            ro_scale=0,
            ra_scale=0,
            rf_scale=3,
        ), 
        frame_stack=3,
        schedules=[
            ro_schedule,
            ra_schedule,
            rf_schedule,
            wo_schedule,
            oy_schedule 
        ]
    )

    
    # train(
    #     env_name="gripper_tactile", 
    #     model_name=ALG,
    #     nenv=6,
    #     train_kw=dict(timesteps=TIME),
    #     max_steps=200,
    #     env_kw=dict(
    #         control_mode=CTRL, 
    #         obs_config=OBS, 
    #         co_scale=1,
    #         rp_scale=1,
    #         ro_scale=ro_schedule.final_value,
    #         ra_scale=0,
    #         rf_scale=rf_schedule.final_value,
    #         wo_range=wo_schedule.final_value,
    #         oy_range=oy_schedule.final_value    
    #     ), 
    #     frame_stack=3,
    #     schedules=[
    #         ra_schedule,
    #     ],
    #     weights=f"{model_path}/2023-07-17_13-03-20__gripper_tactile__ppo__pos_delta__obs_q-qdot-f-df-inC-hadC-act__nenv-6__k-3/weights/_best_model.zip"
    # )