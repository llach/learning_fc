import pickle
import learning_fc 

import numpy as np
import matplotlib.pyplot as plt

from learning_fc.plotting import grey, purple, green, set_rcParams, setup_axis, finish_fig, PLOTMODE, FIGTYPE

with open(f"{learning_fc.__path__[0]}/../forces.pkl", "rb") as f:
    forces = pickle.load(f)

mode = PLOTMODE.camera_ready
set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

lines = []
stable_tresh = 7
for i, f in enumerate(forces):
    c = grey if i<stable_tresh else purple
    lw = 1.8 if i<stable_tresh else 2.0

    # peaks are scaled down to make things clearer
    if i == 9: f = np.where(f>0.25, 0.6*f, f)

    line, = ax.plot(f, lw=lw, color=c)
    lines.append(line)

legend_items =[
    [(lines[0],), (lines[-1],)],
    ["stable", "instable"]
]

setup_axis(
    ax, 
    xlabel=r"$t$" if mode == PLOTMODE.camera_ready else "t", 
    ylabel=r"$f^\text{left}$" if mode == PLOTMODE.camera_ready else "force", 
    xlim=[0, 200], 
    ylim=[0, 0.3],
    legend_items=legend_items,
    legend_loc="upper left",
)

finish_fig(fig)
plt.savefig(f"test")
if mode == PLOTMODE.debug: plt.show()