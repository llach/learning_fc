import json

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE, SEND_LINE_KW
from stable_baselines3.common.results_plotter import load_results

trial = find_latest_model_in_path(model_path, filters=["ppo"])

with open(f"{trial}/parameters.json", "r") as f:
    params = json.load(f)
total_steps = params["train"]["timesteps"]
max_send = np.max([s["dur"] for s in params["make_model"]["schedules"]])

df = load_results(trial)
x = np.cumsum(df.l.values)
y = df.r.values
y_mean = df.r.rolling(150).mean()
y_std = df.r.rolling(150).std()

mode = PLOTMODE.camera_ready
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

mline, = ax.plot(x, y_mean, c=Colors.mean_r, lw=2)
fill   = ax.fill_between(x, y_mean+y_std, y_mean-y_std, color=Colors.sigma_r, alpha=0.45, lw=0)

# TODO set ymin/max dynamically
sline = ax.axvline(max_send, ymin=0.58, ymax=0.89, **SEND_LINE_KW)

setup_axis(
    ax, 
    xlabel="Training Steps", 
    ylabel="Episode Rewards", 
    xlim=[0,total_steps],
    ylim=[0,225],
    remove_first_ytick=False,
    xticks=[0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 3.5e6, 4e6, 4.5e6],
    xticklabels=["", "", "1M", "", "2M", "", "3M", "", "4M", ""],
    legend_items=[
        [(mline,), (fill,), (sline,)],
        [
            r"mean $\int r(t)$" if tex else "mean r", 
            r"$\pm$ std. dev." if tex else "± sigma",
            r"$\max_i(s_i^\text{end})$" if tex else "max(s_i^end)",
        ]
    ],
    legend_loc="center right",
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig("curve")