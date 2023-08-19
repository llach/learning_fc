import learning_fc
import numpy as np

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
TIME = int(45e5)
STOP = int(15e5)

def _get_schedules():
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

    return [
        ro_schedule,
        ra_schedule,
        wo_schedule,
        oy_schedule,
        vo_schedule,
    ]

ekw = dict(
    control_mode=CTRL, 
    obs_config=OBS, 
    ov_max=0.0002,
    ro_scale=0,
    ra_scale=0,
    rp_scale=0.2,
    rf_scale=0.8,
    sample_solref=True,
    sample_solimp=True,
    sample_fscale=True,
    sample_biasprm=True,
    noise_f=0.002,
    ftheta=0.0075,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
)

if __name__ == "__main__": 
    for lr in np.linspace(1e-5, 1e-3, 6):
        for ent in np.linspace(0.0, 0.01, 3):
            train(
                env_name="gripper_tactile", 
                model_name=ALG,
                nenv=10,
                max_steps=200,
                train_kw=dict(
                    timesteps=TIME,
                ),
                env_kw=ekw, 
                model_kw=dict(
                    learning_rate=lr,
                    ent_coef=ent,
                    policy_kwargs=dict(
                        net_arch=[60,60]
                    )
                ),
                frame_stack=3,
                schedules=_get_schedules()
            )
