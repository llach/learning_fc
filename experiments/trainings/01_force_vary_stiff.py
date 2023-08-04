from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.Des, 
    Observation.Force, 
    Observation.ForceDelta,
    Observation.HadCon,
    Observation.Action
]
CTRL = ControlMode.PositionDelta
TIME = int(25e5)
STOP = int(15e5)

if __name__ == "__main__": 

    vo_schedule = ParamSchedule(
        var_name="ov_max",
        start=0.0, 
        stop=STOP,
        first_value=0.0002,
        final_value=0.00005,
        total_timesteps=TIME
    )

    ro_schedule = ParamSchedule(
        var_name="ro_scale",
        start=0.0, 
        stop=STOP,
        first_value=0.0,
        final_value=1.0,
        total_timesteps=TIME
    )

    ra_schedule = ParamSchedule(
        var_name="ra_scale",
        start=0.0, 
        stop=STOP,
        first_value=0.0,
        final_value=0.1,
        total_timesteps=TIME
    )

    wo_schedule = ParamSchedule(
        var_name="wo_range",
        start=0.0, 
        stop=STOP,
        first_value=[0.020, 0.025],
        final_value=[0.015, 0.035],
        total_timesteps=TIME
    )

    oy_schedule = ParamSchedule(
        var_name="oy_range",
        start=0.0, 
        stop=STOP,
        first_value=[0.0, 0.0],
        final_value=[-0.040, 0.040],
        total_timesteps=TIME
    )

    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        nenv=10,
        max_steps=200,
        train_kw=dict(
            timesteps=TIME,
        ),
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS, 
            ov_max=0.0002,
            sample_solref=True,
            sample_solimp=True,
            co_scale=0,
            rp_scale=0.2,
            ro_scale=0,
            ra_scale=0,
            rf_scale=0.8,
        ), 
        model_kw=dict(
            learning_rate=6e-4,
            policy_kwargs=dict(
                net_arch=[60,60]
            )
        ),
        frame_stack=1,
        schedules=[
            ro_schedule,
            ra_schedule,
            wo_schedule,
            oy_schedule,
            vo_schedule,
        ]
    )