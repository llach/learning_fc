import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.utils import get_q_f
from learning_fc.envs import GripperTactileEnv


def load_q_f(obj_name):
    qs, fs = [], []
    files_dir = f"{model_path}/data/dq/"
    for fi in os.listdir(files_dir):
        if obj_name not in fi: continue 
        with open(f"{files_dir}{fi}", "rb") as f:
            data = pickle.load(f)
        qs.append(np.array(data["obs"]["q"]))
        fs.append(np.array(data["obs"]["f"]))
    return qs, fs

nsteps = 150

_, r_wood   = load_q_f("wood_mid")
_, r_sponge = load_q_f("sponge_mid")

r_wood = r_wood[:nsteps]
r_sponge = r_sponge[:nsteps]

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    noise_f=0.002,
    f_scale=2.6,
)

env.solimp = env.SOLIMP_HARD
env.wo_range = 2*[0.016]
_, e_wood = get_q_f(env, nsteps)

env.solimp = env.SOLIMP_SOFT
env.wo_range = 2*[0.0265]
_, e_sponge = get_q_f(env, nsteps)


fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw={'height_ratios': [3, 2]}, figsize=(13,8))
# r1, r2 = axes[1,1].plot(np.arange(n_steps-1), np.diff(f_rob, axis=0)/dts[-1])
xs = np.arange(nsteps)

axes[0,0].plot(xs, r_wood)
axes[0,0].plot(xs, e_wood)

axes[0,1].plot(xs, r_sponge)
axes[0,1].plot(xs, e_sponge)

plt.show()