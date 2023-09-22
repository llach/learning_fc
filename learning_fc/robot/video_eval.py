import os
import time
import rospy
import numpy as np

from datetime import datetime

from learning_fc import model_path, datefmt
from learning_fc.robot import RobotInterface
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model


N_TRIALS = 30
N_SECS = 6.0


# load policy and env
# policy_trial, indb = "2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb", True
policy_trial, indb = "2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb", False
# policy_trial, indb = "2023-09-15_08-22-36__gripper_tactile__ppo__k-3__lr-0.0006_M2_nor", False

env, model, _, params = make_eval_env_model(f"{model_path}/{policy_trial}" , with_vis=False, checkpoint="best")
k = 1 if "frame_stack" not in params["make_env"] else params["make_env"]["frame_stack"]
env.set_attr("fth", 0.02)
# load Force Controller (even though we don't use the policy model, we need the env)
# model, indb = ForcePI(env), False

ri = RobotInterface(
    model, 
    env, 
    fth=env.fth,
    k=k, 
    goal=0.0, 
    freq=25, 
    with_indb=True
)
ri.reset()
r = rospy.Rate(51)

# open gripper
ri.reset()
ri.actuate([0.045, 0.045])
time.sleep(0.5)

for _ in range(N_TRIALS):

    # time for object rearrangement / decision to stop evaluation
    inp = input("goal?\n")
    if inp == "q": break
    else:
            try:
                goal = float(inp)
                assert goal >= 0, "goal >= 0"
                ri.set_goal(goal)
                print(f"new goal: {goal}")
            except Exception as e:
                print(f"can't convert {goal} to a number:\n{e}")
                continue


    # grasp object  
    if isinstance(model, ForcePI): model.reset()
    ri.reset()
    ri.set_goal(goal)
    ri.run()

    start = time.time()
    while time.time() - start < N_SECS: r.sleep()
    ri.stop()
    ri.reset()
    ri.actuate([0.045, 0.045])
    time.sleep(0.5)

ri.reset()
ri.actuate([0.045, 0.045])
ri.shutdown()
exit()