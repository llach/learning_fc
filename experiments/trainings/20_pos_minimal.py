"""
Simple position control agent to real-robot software infra
"""
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.PosDelta, 
    Observation.Action
]
CTRL = ControlMode.PositionDelta
TIME = int(1e6)

if __name__ == "__main__": 

    train(
        env_name="gripper_pos", 
        model_name=ALG,
        nenv=6,
        train_kw=dict(timesteps=TIME),
        max_steps=300,
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS,
        ), 
        model_kw=dict(
            policy_kwargs=dict(
                net_arch=[10,10]
            ),
        ),
        frame_stack=1,
    )