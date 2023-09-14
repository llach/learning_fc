from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos,
    Observation.Force, 
    Observation.ForceDelta,
    Observation.HadCon,
    Observation.Action
]
CTRL = ControlMode.PositionDelta
TIME = int(4.5e6)
STOP = int(1.5e6)

def _get_schedules():
    ah_scale_schedule = ParamSchedule(
        var_name="ah_scale",
        start=0.0, 
        stop=STOP,
        first_value=0.1,
        final_value=5.0,
        total_timesteps=TIME
    )

    ah_max_schedule = ParamSchedule(
        var_name="ah_max",
        start=0.0, 
        stop=STOP,
        first_value=0.3,
        final_value=0.03,
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

    return [
        ah_scale_schedule,
        ah_max_schedule,
        wo_schedule,
        oy_schedule,
    ]


ekw = dict(
    control_mode=CTRL, 
    obs_config=OBS, 
    ov_max=0.0002,
    ro_scale=0,
    ra_scale=0.2,
    rp_scale=0.0,
    rf_scale=1.0,
    sample_biasprm=True,
    randomize_stiffness=True,
    noise_f=0.01,
    fth=0.0075,
    model_path="assets/pal_force.xml",
)

if __name__ == "__main__":
    train(
        env_name="gripper_tactile", 
        model_name=ALG,
        nenv=10,
        max_steps=150,
        train_kw=dict(
            timesteps=TIME,
        ),
        env_kw=ekw, 
        model_kw=dict(
            learning_rate=6e-4,
            policy_kwargs=dict(
                net_arch=[50,50]
            )
        ),
        frame_stack=3,
        schedules=_get_schedules(),
    )