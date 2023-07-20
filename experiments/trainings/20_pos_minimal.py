"""
Simple position control agent to real-robot software infra
"""
from learning_fc.enums import ControlMode, Observation
from learning_fc.training import train

ALG  = "ppo"
OBS  = [
    Observation.Pos, 
    Observation.PosDelta, 
]
CTRL = ControlMode.PositionDelta
TIME = int(5e5)

if __name__ == "__main__": 

    train(
        env_name="gripper_pos", 
        model_name=ALG,
        nenv=6,
        train_kw=dict(timesteps=TIME),
        max_steps=100,
        env_kw=dict(
            control_mode=CTRL, 
            obs_config=OBS,
        ), 
        frame_stack=1,
    )