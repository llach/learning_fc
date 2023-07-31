"""
first script for force control without object movements.
O contains basically all we have (minus acceleration)
r also has components for object proximity and contact, which are likely to be superfluous  .

next steps:
    * reduce to MVP in terms of reward and observations
    * still no variation about object stiffness
"""
from learning_fc import model_path
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.Vel,
    Observation.Force, 
    Observation.ForceDelta,
    Observation.Action,
    Observation.HadCon,
    Observation.ObjVel,
]
CTRL = ControlMode.PositionDelta
TIME = int(2e6)

if __name__ == "__main__": 

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

    ro_schedule = ParamSchedule(
        var_name="ro_scale",
        start=0, 
        stop=0.5,
        first_value=0.0,
        final_value=3.0,
        total_timesteps=TIME
    )

    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        save_on_best=ro_schedule.stop,
        nenv=6,
        train_kw=dict(timesteps=TIME),
        max_steps=250,
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS, 
            ov_max=0.00005,
            rp_scale=0.2,
            rf_scale=0.8,
            co_scale=0,
            ro_scale=0,
            ra_scale=0,
        ), 
        weights=f"{model_path}/30_base/weights/_best_model.zip",
        frame_stack=5,
        schedules=[
            ro_schedule,
            wo_schedule,
            oy_schedule
        ]
    )