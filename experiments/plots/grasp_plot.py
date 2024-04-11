import os
import pickle
import pathlib
import numpy as np
import matplotlib as mpl
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

from learning_fc import cm_uni, cm_itten
import plot_config
cmap = cm["tab10"]


pklpath = f"{pathlib.Path(__file__).parent.absolute()}/data/external_forces_FC_28.02.2022_12:09:45.pkl"
with open(pklpath, "rb") as f:
    try:
        data = pickle.load(f, encoding="latin1") # Python3
    except TypeError:
        data = pickle.load(f) # Python2

fro = 200
cutoff = 1550
deltaT = np.array(data['deltaT'][fro:cutoff])
f = data['f'][fro:cutoff]
fDiff = data['fDiff'][fro:cutoff]
fAdd = data['fAdd'][fro:cutoff]

mgzn = data['mgzn'][fro:cutoff]
cosGrav = data['cosGrav'][fro:cutoff]

qDes = data['qDes'][fro:cutoff]

noiseT = data['noiseT']
driftT = data['driftT']
targetF = data['targetF']

deltaT -= deltaT[0]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,7.5))

# f1, f2
# multi-color y-axis label
c1 = "#FF6600"
c2 = "#004E78"

noise_t_color = color='#202020'

axes[0].plot(deltaT, f[:,0], label=r'$f_1$', color=cm_itten["orange"]) 
axes[0].plot(deltaT, f[:,1],  label=r'$f_2$', color=cm_uni["blue"])

axes[0].text(0.2, noiseT+0.3, r"$f_\theta$", fontsize=12)

axes[0].axhline(y=noiseT, color=noise_t_color, linestyle=(0, (5, 5)), linewidth=0.5)
axes[0].axhline(y=-noiseT, color=noise_t_color, linestyle=(0, (5, 5)), linewidth=0.5)

axes[0].set_ylabel(r"Forces $f_i$ \, [$N$]")
leg = axes[0].legend()
leg.get_frame().set_linewidth(0.3)


# f1 + f2
axes[1].set_ylabel(r"Internal object force $f^\text{int}$ \, [$N$]")
# axes[1].text(0.88, targetF+0.15, r"$f^\text{goal}$" , fontsize=12)

axes[1].plot(deltaT, fAdd, c=cm_uni["darkgreen"], label=r"$f_1 + f_2$")
axes[1].axhline(y=targetF, color=cmap(7), linestyle='-', linewidth=1.0, label=r"$f^\text{goal}$")
leg = axes[1].legend()
leg.get_frame().set_linewidth(0.3)

axes[1].set_ylim(-0.2, 4.5)

# f1 - f2 + mgn
axes[2].set_ylabel(r"External force $f^\text{ext}$ \, [$N$]")
fdiffplot, = axes[2].plot(deltaT, fDiff, c=cm_itten["pink"], label=r"$f_1 - f_2 + m\,\mathbf{g}\cdot\mathbf{n}$")

frac=2
axes[2].text(0.2, (frac*driftT)+0.8, r"$f_\phi$", fontsize=12)
axes[2].set_ylim(-3.7, 3.7)

axes[2].axhline(y=frac*driftT, color=noise_t_color, linestyle='--', linewidth=0.5)
axes[2].axhline(y=-frac*driftT, color=noise_t_color, linestyle='--', linewidth=0.5)

ax2t = axes[2].twinx()
color_ot = "#EE0000"

# ot can only be calculated after both fingers touch -> correct for that offset (before force closure, the following calculation is biased by the non-touching finger's trajectory)
ot = (((qDes[:, 0]-qDes[:, 1])/2)*1000) - 8.2
ot[0:245] = ot[240:485]
otplot, = ax2t.plot(deltaT, ot, c=cm_uni["blue"], label=r"$O(t)$")
ax2t.set_ylabel(r"Object position $O(t)$")

leg = axes[2].legend([fdiffplot, otplot], [r"$f_1 - f_2 + m\,\mathbf{g}\cdot\mathbf{n}$", r"$O(t)$"], loc=3)
leg.get_frame().set_linewidth(0.3)

for ax in axes[:-1].flatten():
    ax.get_xaxis().set_ticklabels([]) # hide x axis tick labels for all but last rows
axes[-1].spines[['right']].set_visible(True)

for ax in axes:
    ax.spines[['right']].set_visible(True)
    ax.set_xlim(0,13.5)

events = [(0.69, r"1$^\text{st}$ touch"), (1.82, r"2$^\text{nd}$ touch"), (5.43, ""), (6.94, ""), (9.42, ""), (10.72, "")]
for i, a in enumerate(axes):
    for j, (x, text) in enumerate(events):
        if i == 0:
            offset = 0.22 if j == 0 else 0.18
            yoff = 0 if j == 0 else 0.1
            a.text(x+offset, 0.9-yoff, text, transform=a.get_xaxis_transform(), ha='center', va='center')
        a.axvline(x=x, linestyle=(0,(5,5)), color='gray', linewidth=0.8)

axes[-1].set_xlabel(r"Time [$s$]")

fig.align_ylabels()
# fig.text(0.5, 0.007, 'time [$s$]', fontsize=18, ha='center')
# fig.tight_layout(rect=[0, 0.01, 1, 1])
# fig.subplots_adjust(hspace=0.1)
# fig.subplots_adjust(wspace=0.005)
plt.savefig(f"{os.environ['HOME']}/repos/diss/images/ctrl/grasp_new.pdf")
plt.show()


# phases = [(0, 1.82, "Closing"),
#           (1.82, 5.43, "Holding"),
#           (5.43, 6.94, "Right"),
#           (6.94, 9.42, "Relaxing"),
#           (9.42, 10.72, "Left"),
#           (10.72, deltaT[-1], "Relaxing")]