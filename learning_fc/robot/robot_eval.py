import os
import time
import rospy
import numpy as np

from datetime import datetime

from learning_fc import model_path, datefmt
from learning_fc.robot import RobotInterface
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model

objects = {   #   wo,   dp,  fmin, fmax
     "tape":  [0.045, 0.009, 0.07, 0.33],
     "spy":   [0.015, 0.004, 0.10, 0.50],
     "spb":   [0.042, 0.006, 0.10, 0.70],
     "wind":  [0.063, 0.003, 0.12, 0.82],
     "pring": [0.074, 0.000, 0.12, 0.85],
     "wood":  [0.059, 0.000, 0.13, 0.90],
     "mug":   [0.081, 0.000, 0.15, 1.00],
}

N_TRIALS = 20
N_SECS = 6.0

# make sure eval data dir exists
eval_data_dir = f"{model_path}/robot_eval"
os.makedirs(eval_data_dir, exist_ok=True)

# load policy and env
policy_trial = "2023-09-12_09-38-10__gripper_tactile__ppo__k-3__lr-0.0006_M2"
env, model, _, params = make_eval_env_model(f"{model_path}/{policy_trial}" , with_vis=False, checkpoint="best")
k = 1 if "frame_stack" not in params["make_env"] else params["make_env"]["frame_stack"]
env.set_attr("fth", 0.05)

# load Force Controller (even though we don't use the policy model, we need the env)
model = ForcePI(env)

# get model and object name for saving
model_name = input("model name: ")
assert model_name in ["pol", "nodr", "fc"], f'{model_name} not in ["pol", "nodr", "fc"]'
assert model_name != "fc" or isinstance(model, ForcePI), 'model_name != "fc" or isinstance(model, ForcePI)'

obj_name = input("object name: ")
assert obj_name in list(objects.keys()), f"{obj_name} in {list(objects.keys())}"

wo, dp, fmin, fmax = objects[obj_name]
wo /= 2 # sim uses radius, we measured width
dp /= 2

trial_dir = f"{eval_data_dir}/{model_name}/"
resfile = f"{trial_dir}/results.csv"
os.makedirs(trial_dir, exist_ok=True)

model_info = policy_trial if not isinstance(model, ForcePI) else str(model)
with open(f"{trial_dir}/model_info", "w") as f:
    f.write(model_info)

ri = RobotInterface(model, env, k=k, goal=0.0, freq=25, datadir=trial_dir)
ri.reset()
r = rospy.Rate(51)

input("start?")
for _ in range(N_TRIALS):
    # open gripper
    ri.reset()
    ri.actuate([0.045, 0.045])
    time.sleep(0.5)
    
    # sample and print oy
    oymax = round(wo-dp, 3)
    print(f"oq={np.random.uniform(-oymax, oymax)}")

    # time for object rearrangement / decision to stop evaluation
    inp = input("next?")
    if inp == "q": break

    sample_name = f"{obj_name}__{datetime.utcnow().strftime(datefmt)}"
    fgoal = round(np.random.uniform(fmin, fmax), 3)
    print(f"collecting {model_name} sample {sample_name} - fgoal {fgoal}")

    # grasp object  
    if isinstance(model, ForcePI): model.reset()
    ri.set_goal(fgoal)
    ri.reset()
    ri.run()

    start = time.time()
    while time.time() - start < N_SECS: r.sleep()
    ri.stop()

    # store all important info
    cumr = round(ri.cumr, 2)
    ri.save_hist(sample_name)

    if not os.path.isfile(resfile):
        with open(resfile, "w") as f:
            f.write("sample,fgoal,r\n")
    
    with open(resfile, "a") as f:
            f.write(f"{sample_name},{fgoal},{cumr}\n")

    # print for easy copying
    print(f"{sample_name}     {fgoal}      {cumr}")

ri.reset()
ri.actuate([0.045, 0.045])
ri.shutdown()
exit()