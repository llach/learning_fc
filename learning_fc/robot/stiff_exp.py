import os
import time
import rospy
import numpy as np

from datetime import datetime

from learning_fc import model_path, datefmt
from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import safe_rescale
from learning_fc.enums import ControlMode
from learning_fc.robot import RobotInterface
from learning_fc.models import StaticModel

objects = {   #   wo,   dp,  fmin, fmax
     "tape":  [0.045, 0.009, 0.07, 0.33],
     "spy":   [0.015, 0.004, 0.10, 0.50],
     "spb":   [0.042, 0.006, 0.10, 0.70],
     "wind":  [0.063, 0.003, 0.12, 0.82],
     "pring": [0.074, 0.000, 0.12, 0.85],
     "wood":  [0.059, 0.000, 0.13, 0.90],
     "mug":   [0.081, 0.000, 0.15, 1.00],
}

dqmax = 0.003
dqmin = 0.0003
step_size = 0.0003
dqs = np.arange(dqmin, dqmax, step_size)

# make sure eval data dir exists
stiff_data_dir = f"{model_path}/stiff_exp/"
resfile = f"{stiff_data_dir}/results.csv"
os.makedirs(stiff_data_dir, exist_ok=True)

model = StaticModel(dqmin)
env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[0.025, 0.025],
    model_path="assets/pal_force.xml",
    noise_f=0.002,
    f_scale=3.1,
)

# get object name for saving
obj_name = input("object name: ")
assert obj_name in list(objects.keys()), f"{obj_name} in {list(objects.keys())}"

ri = RobotInterface(model, env, goal=0.0, freq=25, datadir=stiff_data_dir)
ri.reset()
r = rospy.Rate(51)

input("start?")
for dq in dqs:
    dq = round(dq, 4)

    # open gripper
    ri.reset()
    ri.actuate([0.045, 0.045])
    time.sleep(0.5)

    # time for object rearrangement / decision to stop evaluation
    inp = input("next?")
    if inp == "q": break

    sample_name = f"{obj_name}__{dq}__{datetime.utcnow().strftime(datefmt)}"
    print(f"collecting sample {sample_name}")

    # grasp object  
    model.q = safe_rescale(-dq, [-env.dq_max, env.dq_max], [-1, 1])
    ri.reset()
    ri.run()

    input("done?")
    ri.stop()

    # store all important info
    forces = ri.hist["force"]
    ffinal = np.mean(forces[-10:])
    ri.save_hist(sample_name)

    if not os.path.isfile(resfile):
        with open(resfile, "w") as f:
            f.write("sample,obj,dq,ffinal\n")
    
    with open(resfile, "a") as f:
            f.write(f"{sample_name},{obj_name},{dq},{ffinal}\n")

    # print for easy copying
    print(f"{sample_name}     {dq}      {ffinal}")

ri.reset()
ri.actuate([0.045, 0.045])
ri.shutdown()
exit()