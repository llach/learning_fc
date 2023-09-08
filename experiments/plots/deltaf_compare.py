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
    files_dir = f"{model_path}/data/"
    for fi in os.listdir(files_dir):
        print(fi)
        if obj_name not in fi: continue 
        with open(f"{files_dir}{fi}", "rb") as f:
            data = pickle.load(f)
        qs.append(np.array(data["obs"]["q"]))
        fs.append(np.array(data["obs"]["f"]))
    return qs, fs

def diff_traj(traj, dt): 
    tdiff = np.diff(traj, axis=0)/dt
    return np.concatenate([[tdiff[0]], tdiff])

def dfdq(fs, qs):
    dd = np.diff(qs, axis=0)/(np.diff(fs, axis=0)+1e-4)
    return np.concatenate([[dd[0]], dd])


nsteps = 70

qrwood, r_wood   = load_q_f("wood3")
qrsponge, r_sponge = load_q_f("sponge3")

qrwood = qrwood[0][:nsteps]
qrsponge = qrsponge[0][:nsteps]
r_wood = r_wood[0][:nsteps]
r_sponge = r_sponge[0][:nsteps]

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    model_path="assets/pal_force.xml",
    noise_f=0.002,
    f_scale=2.6,
)

env.solimp = [0.00, 0.99, 0.009, 0.5, 2]
env.wo_range = 2*[0.0295/2]
env.f_scale = 3.15
qewood, e_wood = get_q_f(env, nsteps)

env.solimp = [0.00, 0.99, 0.01, 0.5, 2]
env.wo_range = 2*[0.0255]
env.f_scale = 2.4
qesponge, e_sponge = get_q_f(env, nsteps)

mode = PLOTMODE.debug
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw={'height_ratios': [3, 2]}, figsize=(13,8))
xs = np.arange(nsteps)

rw0, = axes[0,0].plot(xs, r_wood[:,0], c=Colors.tab10_0)
rw1, = axes[0,0].plot(xs, r_wood[:,1], c=Colors.tab10_1)
ew0, = axes[0,0].plot(xs, e_wood[:,0], c=Colors.tab10_2)
ew1, = axes[0,0].plot(xs, e_wood[:,1], c=Colors.tab10_3)

rwend = np.max(r_wood[-1])
ewend = np.max(e_wood[-1])

rs0, = axes[0,1].plot(xs, r_sponge[:,0], c=Colors.tab10_0)
rs1, = axes[0,1].plot(xs, r_sponge[:,1], c=Colors.tab10_1)
es0, = axes[0,1].plot(xs, e_sponge[:,0], c=Colors.tab10_2)
es1, = axes[0,1].plot(xs, e_sponge[:,1], c=Colors.tab10_3)

rsend = np.max(r_sponge[-1])
esend = np.max(e_sponge[-1])

setup_axis(
    axes[0,0], 
    ylabel=r"$f_i$" if tex else "f", 
    xlim=[0, nsteps], 
    ylim=[-0.005, 1.0],
    yticks=np.linspace(0,7,8)*0.1,
    yticklabels=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
    remove_xticks=True,
    legend_items=[
        [(rw0, rw1), (ew0, ew1)],
        [
            f"real  | $\\bar f(T)={rwend:.2f}$",
            f"sim  | $\\bar f(T)={ewend:.2f}$",
        ]
    ]
)

setup_axis(
    axes[0,1], 
    xlim=[0, nsteps], 
    ylim=[-0.005, 1.0],
    yticks=np.linspace(0,7,8)*0.1,
    yticklabels=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
    remove_xticks=True,
    remove_yticks=True,
    legend_items=[
        [(rs0, rs1), (es0, es1)],
        [
            f"real  | $\\bar f(T)={rsend:.2f}$",
            f"sim  | $\\bar f(T)={esend:.2f}$",
        ]
    ]
)

dr_wood = diff_traj(r_wood, env.dt)
dr_sponge = diff_traj(r_sponge, env.dt)

de_wood = diff_traj(e_wood, env.dt)
de_sponge = diff_traj(e_sponge, env.dt)

dr_wood = dfdq(r_wood, qrwood)
dr_sponge = dfdq(r_sponge, qrsponge)

de_wood = dfdq(e_wood, qewood)
# de_sponge = dfdq(e_sponge, qesponge)


drw0, = axes[1,0].plot(xs, dr_wood[:,0], c=Colors.tab10_0)
drw1, = axes[1,0].plot(xs, dr_wood[:,1], c=Colors.tab10_1)
dew0, = axes[1,0].plot(xs, de_wood[:,0], c=Colors.tab10_2)
dew1, = axes[1,0].plot(xs, de_wood[:,1], c=Colors.tab10_3)

drwmax = np.max(dr_wood)
dewmax = np.max(de_wood)

drs0, = axes[1,1].plot(xs, dr_sponge[:,0], c=Colors.tab10_0)
drs1, = axes[1,1].plot(xs, dr_sponge[:,1], c=Colors.tab10_1)
des0, = axes[1,1].plot(xs, de_sponge[:,0], c=Colors.tab10_2)
des1, = axes[1,1].plot(xs, de_sponge[:,1], c=Colors.tab10_3)

drsmax = np.max(dr_sponge)
desmax = np.max(de_sponge)

setup_axis(
    axes[1,0], 
    xlabel=r"$t$" if tex else "t",
    ylabel=r"$\frac{\partial f_i}{\partial t}$" if tex else "df/dt", 
    xlim=[0, nsteps], 
    # ylim=[-1,10],
    legend_items=[
        [(drw0, drw1), (dew0, dew1)],
        [
            f"real | $\\max(\\frac{'{'}\\partial f_i{'}{'}\\partial t{'}'})={drwmax:.1f}$" if tex else f"real | max(df/dt)={drwmax:.1f}",
            f"sim | $\\max(\\frac{'{'}\\partial f_i{'}{'}\\partial t{'}'})={dewmax:.1f}$" if tex else f"sim | max(df/dt)={dewmax:.1f}",
        ]
    ]
)

setup_axis(
    axes[1,1], 
    xlabel=r"$t$" if tex else "t",
    xlim=[0, nsteps], 
    # ylim=[-1,10],
    remove_yticks=True,
    legend_items=[
        [(drs0, drs1), (des0, des1)],
        [
            f"real | $\\max(\\frac{'{'}\\partial f_i{'}{'}\\partial t{'}'})={drsmax:.1f}$",
            f"sim | $\\max(\\frac{'{'}\\partial f_i{'}{'}\\partial t{'}'})={desmax:.1f}$",
        ]
    ]
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig("dfdt_compare")