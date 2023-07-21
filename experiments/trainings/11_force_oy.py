"""
based on experiments/trainings/10_force_centered.py

goal: add object pose variation with minimal additions possible 
"""
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.Force, 
    Observation.ForceDelta,
    Observation.Action,
    Observation.InCon,
    Observation.HadCon
]
CTRL = ControlMode.PositionDelta
TIME = int(2e6)

if __name__ == "__main__": 

    ro_schedule = ParamSchedule(
        var_name="ro_scale",
        start=0.0, 
        stop=0.8,
        first_value=0.0,
        final_value=30.0,
        total_timesteps=TIME
    )

    ra_schedule = ParamSchedule(
        var_name="ra_scale",
        start=0.0, 
        stop=0.5,
        first_value=0.0,
        final_value=5.0,
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
            ov_max=0.00005,
            ro_scale=0,
            ra_scale=0,
            rf_scale=3,
        ), 
        frame_stack=1,
        schedules=[
            ra_schedule,
            ro_schedule,
            rf_schedule,
            wo_schedule,
            oy_schedule,
        ]
    )