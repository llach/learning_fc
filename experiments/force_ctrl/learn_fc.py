from learning_fc.training import train

train(
    env_name="gripper_tactile", 
    model_name="ppo",
    logdir="/tmp/tactile/"
)
