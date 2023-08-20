import pickle
import learning_fc 

import numpy as np
import matplotlib.pyplot as plt

from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE, EPS_SEP_LINE_KW

with open(f"{learning_fc.__path__[0]}/../traj.pkl", "rb") as f:
    data = pickle.load(f)

mode = PLOTMODE.debug
tex = set_rcParams(mode=mode, ftype=FIGTYPE.multirow, nrows=2)
fig, axes = plt.subplots(nrows=2)

ntrials = 4
nsteps  = 200
total_steps = ntrials*nsteps
xs = np.arange(total_steps).reshape((ntrials, nsteps))

forces = data["force"].reshape((ntrials, nsteps, -1))
for x, y in zip(xs, forces): # avoids discontinuity on episode end
    lline, = axes[0].plot(x, y[:,0], c=Colors.tab10_0)
    rline, = axes[0].plot(x, y[:,1], c=Colors.tab10_1)

goals = data["goals"].reshape((ntrials, nsteps))
for x, y in zip(xs, goals):
    gline, = axes[0].plot(x, y, c=Colors.fgoal, lw=1.0)


actions = data["actions"].reshape((ntrials, nsteps, -1))
for x, y in zip(xs, actions):
    axes[1].plot(x, y[:,0], c=Colors.tab10_0)
    axes[1].plot(x, y[:,1], c=Colors.tab10_1)

reset_lines = ((np.arange(ntrials)+1)*nsteps)[:-1]
for ax in axes.flatten(): 
    for rl in reset_lines: epline = ax.axvline(rl, **EPS_SEP_LINE_KW)

setup_axis(
    axes[0], 
    xlabel=None,
    ylabel=r"$f_i$" if tex else "force", 
    xlim=[0, total_steps], 
    ylim=[-0.015, 0.7],
    remove_xticks=True,
    legend_items=[[(gline,), (epline,)], [r"$f^\text{goal}$" if tex else "fgoal", r"$T$" if tex else "T"]],
    legend_loc="upper left"
)

setup_axis(
    axes[1], 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$a_i$" if tex else "action", 
    xlim=[0, total_steps], 
    ylim=[-1.02, 1.0],
    yticks=np.arange(-1,1.1,0.5),
    legend_items=[[(lline,), (rline,)], ["left", "right"]],
    legend_loc="upper left"
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig(f"mr_test")