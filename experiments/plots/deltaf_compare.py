import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.utils import get_q_f
from learning_fc.envs import GripperTactileEnv

from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE


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

def diff_traj(traj, dt): 
    tdiff = np.diff(traj, axis=0)/dt
    return np.concatenate([[tdiff[0]], tdiff])


nsteps = 130

_, r_wood   = load_q_f("wood_mid")
_, r_sponge = load_q_f("sponge_mid")

r_wood = r_wood[1][:nsteps]
r_sponge = r_sponge[1][:nsteps]

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    noise_f=0.002,
    f_scale=2.6,
)

env.solimp = [0.00, 0.99, 0.0016, 0.2, 1]
env.wo_range = 2*[0.0155]
env.f_scale = 2.9
_, e_wood = get_q_f(env, nsteps)

env.solimp = [0.00, 0.99, 0.0065, 0.5, 3]
env.wo_range = 2*[0.0255]
env.f_scale = 2.6
_, e_sponge = get_q_f(env, nsteps)

mode = PLOTMODE.debug
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw={'height_ratios': [3, 2]}, figsize=(13,8))
xs = np.arange(nsteps)

rw0, = axes[0,0].plot(xs, r_wood[:,0])
rw1, = axes[0,0].plot(xs, r_wood[:,1])
ew0, = axes[0,0].plot(xs, e_wood[:,0])
ew1, = axes[0,0].plot(xs, e_wood[:,1])

rwend = np.max(r_wood[-1])
ewend = np.max(e_wood[-1])

rs0, = axes[0,1].plot(xs, r_sponge[:,0])
rs1, = axes[0,1].plot(xs, r_sponge[:,1])
es0, = axes[0,1].plot(xs, e_sponge[:,0])
es1, = axes[0,1].plot(xs, e_sponge[:,1])

rsend = np.max(r_sponge[-1])
esend = np.max(e_sponge[-1])

setup_axis(
    axes[0,0], 
    ylabel=r"$f_i$" if tex else "f", 
    xlim=[0, 130], 
    ylim=[-0.005, 0.7],
    yticks=np.linspace(0,7,8)*0.1,
    yticklabels=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
    remove_xticks=True,
    legend_items=[
        [(rw0, rw1), (ew0, ew1)],
        [
            f"real  | f(T)={rwend:.2f}",
            f"sim  | f(T)={ewend:.2f}",
        ]
    ]
)

setup_axis(
    axes[0,1], 
    xlim=[0, 130], 
    ylim=[-0.005, 0.7],
    yticks=np.linspace(0,7,8)*0.1,
    yticklabels=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
    remove_xticks=True,
    remove_yticks=True,
    legend_items=[
        [(rs0, rs1), (es0, es1)],
        [
            f"real  | f(T)={rsend:.2f}",
            f"sim  | f(T)={esend:.2f}",
        ]
    ]
)

dr_wood = diff_traj(r_wood, env.dt)
dr_sponge = diff_traj(r_sponge, env.dt)

de_wood = diff_traj(e_wood, env.dt)
de_sponge = diff_traj(e_sponge, env.dt)

drw0, = axes[1,0].plot(xs, dr_wood[:,0])
drw1, = axes[1,0].plot(xs, dr_wood[:,1])
dew0, = axes[1,0].plot(xs, de_wood[:,0])
dew1, = axes[1,0].plot(xs, de_wood[:,1])

drwmax = np.max(dr_wood)
dewmax = np.max(de_wood)

drs0, = axes[1,1].plot(xs, dr_sponge[:,0])
drs1, = axes[1,1].plot(xs, dr_sponge[:,1])
des0, = axes[1,1].plot(xs, de_sponge[:,0])
des1, = axes[1,1].plot(xs, de_sponge[:,1])

drsmax = np.max(dr_sponge)
desmax = np.max(de_sponge)

setup_axis(
    axes[1,0], 
    xlabel=r"$t$" if tex else "t",
    ylabel=r"$\partial f_i / \partial t$" if tex else "df/dt", 
    xlim=[0, 130], 
    ylim=[-1,10],
    legend_items=[
        [(drw0, drw1), (dew0, dew1)],
        [
            f"real | max(df/dt)={drwmax:.1f}",
            f"sim | max(df/dt)={dewmax:.1f}",
        ]
    ]
)

setup_axis(
    axes[1,1], 
    xlabel=r"$t$" if tex else "t",
    xlim=[0, 130], 
    ylim=[-1,10],
    remove_yticks=True,
    legend_items=[
        [(drs0, drs1), (des0, des1)],
        [
            f"real | max(df/dt)={drsmax:.1f}",
            f"sim | max(df/dt)={desmax:.1f}",
        ]
    ]
)

plt.show()