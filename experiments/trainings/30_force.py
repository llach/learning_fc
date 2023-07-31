"""
first script for force control without object movements.
O is already minimal
r is normalized (max(r(t)) = 1), and only force and proximity rewards are used

next steps:
    * object movement minimization
    * still no variation about object stiffness
"""
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train

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
TIME = int(3e6)

if __name__ == "__main__": 

    train(
        env_name="gripper_tactile", 
        model_name=ALG,
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
        frame_stack=5,
    )