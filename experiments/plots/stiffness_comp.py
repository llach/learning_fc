import os
import pandas
import pathlib
import numpy as np
from numpy.polynomial.polynomial import polyfit
pa = pathlib.Path(__file__).parent.absolute()

import matplotlib.pyplot as plt

from learning_fc import cm_uni, cm_itten
import plot_config


sp_dat = pandas.read_csv(f"{pa}/data/sponge.csv")[1:]
wo_dat = pandas.read_csv(f"{pa}/data/wood.csv")[1:]
dqs = wo_dat.dq.values


fig, ax = plt.subplots(figsize=(6, 4))

for df, lbl, c in zip([sp_dat, wo_dat], ["Sponge", "Wood"], [cm_itten["deeporange"], cm_itten["purple"]]):
    b, m = polyfit(df.dq, df.f, 1)
    ax.scatter(df.dq, df.f, label=lbl, c=c)
    print(b, m);
    ax.plot(df.dq, b + m*df.dq, linestyle='-', alpha=0.3,c=c)


ax.set_xlabel(r"$\Delta q^\text{des}$")
ax.set_ylabel(r"$f(T)$")
ax.set_xlim(*[0.0005, 0.00305])

leg = ax.legend()
leg.get_frame().set_linewidth(0.3)

plt.savefig(f"{os.environ['HOME']}/repos/diss/images/rl_ctrl/stiff_comp.pdf")
plt.show()