import pickle
import learning_fc 

import numpy as np
import matplotlib.pyplot as plt

from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE

with open(f"{learning_fc.__path__[0]}/../forces.pkl", "rb") as f:
    forces = pickle.load(f)

mode = PLOTMODE.camera_ready
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

lines = []
stable_tresh = 7
for i, f in enumerate(forces):
    c = Colors.grey if i<stable_tresh else Colors.purple
    lw = 1.4 if i<stable_tresh else 1.8

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
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$f^\text{left}$" if tex else "force", 
    xlim=[0, 200], 
    ylim=[0, 0.3],
    legend_items=legend_items,
    legend_loc="upper left",
    remove_first_ytick=True,
)

plt.savefig(f"test")
if mode == PLOTMODE.debug: plt.show()