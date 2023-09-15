import os
import pandas
import pathlib
import numpy as np
from numpy.polynomial.polynomial import polyfit
pa = pathlib.Path(__file__).parent.absolute()

import matplotlib.pyplot as plt
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE


sp_dat = pandas.read_csv(f"{pa}/sponge.csv")[1:]
wo_dat = pandas.read_csv(f"{pa}/wood.csv")[1:]
dqs = wo_dat.dq.values


mode = PLOTMODE.paper
tex = set_rcParams(mode=mode)
fig, ax = plt.subplots(figsize=(7.8, 4.5))

for df, lbl, c in zip([sp_dat, wo_dat], ["Sponge", "Wood"], [Colors.tab10_0, Colors.tab10_1]):
    b, m = polyfit(df.dq, df.f, 1)
    ax.scatter(df.dq, df.f, label=lbl, c=c)
    print(b, m)
    ax.plot(df.dq, b + m*df.dq, linestyle='-', alpha=0.3)

setup_axis(
    ax, 
    xlabel=r"$\Delta q^\text{des}$" if tex else "delta q", 
    ylabel=r"$f(T)$" if tex else "f(T)", 
    xlim=[0.0005, 0.0031], 
)
ax.legend()

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig(f"{os.environ['HOME']}/stiff_comp.pdf")