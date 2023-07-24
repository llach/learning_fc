"""
Simple position control agent to real-robot software infra

now with action in observation space and penalty during training
having ra_schedule can lead to low-frequency oscillations around the goal
since we need to stop abruptly (which is what r_act penalizes), this makes sense. tuning gamma might have an effect here.
"""
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train
from learning_fc.callbacks import ParamSchedule

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.PosDelta, 
    Observation.Action
]
CTRL = ControlMode.PositionDelta
TIME = int(1e6)

if __name__ == "__main__": 

    rv_schedule = ParamSchedule(
        var_name="rv_scale",
        start=0.0, 
        stop=0.5,
        first_value=0.0,
        final_value=3.0,
        total_timesteps=TIME
    )

    train(
        env_name="gripper_pos", 
        model_name=ALG,
        nenv=6,
        train_kw=dict(timesteps=TIME),
        max_steps=200,
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS,
            ra_scale=0,
            rp_scale=3,
        ), 
        model_kw=dict(
            policy_kwargs=dict(
                net_arch=[10,10]
            )
        ),
        frame_stack=1,
        schedules=[
            rv_schedule
        ]
    )