import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
import matplotlib as mpl

import numpy as np

pklpath = "data/external_forces_FC_28.02.2022_12:09:45.pkl"
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

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['mathtext.fontset'] = 'cm'

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

# f1, f2
# multi-color y-axis label
c1 = "#FF6600"
c2 = "#004E78"

label_fs = 17
legend_fs = 16
noise_t_color = color='#202020'

axes[0].plot(deltaT, f[:,0], label='$f_1$', color=c1)
axes[0].plot(deltaT, f[:,1],  label='$f_2$', color=c2)

axes[0].text(0, noiseT+0.3, "$f_{\\theta}}$", fontsize="x-large")

axes[0].axhline(y=noiseT, color=noise_t_color, linestyle='--', linewidth=1.0)
axes[0].axhline(y=-noiseT, color=noise_t_color, linestyle='--', linewidth=1.0)

axes[0].set_ylabel("Force measurements [$N$]", fontsize=label_fs-2.3)
axes[0].legend(prop={'size': legend_fs})

# f1 + f2
axes[1].set_ylabel("Internal object force [$N$]", fontsize=label_fs)
axes[1].text(-0.05, targetF+0.15, "$f^{\\,\\mathrm{goal}}$", fontsize="x-large")

axes[1].plot(deltaT, fAdd, c="#FFB14E", label="$f_1 + f_2$")
axes[1].axhline(y=targetF, color='#585858', linestyle='-', linewidth=1.0)
axes[1].legend(prop={'size': legend_fs})

axes[1].set_ylim(-0.2, 4.5)

# f1 - f2 + mgn
axes[2].set_ylabel("External force [$N$]", fontsize=label_fs)
fdiffplot, = axes[2].plot(deltaT, fDiff, c="#EA5F94", label="$f_1 - f_2 + m\\,\\mathbf{g}\\cdot\\mathbf{n}$")

frac=2
axes[2].text(0, (frac*driftT)+0.8, "$f_{\\phi}}$", fontsize="x-large")
axes[2].set_ylim(-3.7, 3.7)

axes[2].axhline(y=frac*driftT, color=noise_t_color, linestyle='--', linewidth=1.0)
axes[2].axhline(y=-frac*driftT, color=noise_t_color, linestyle='--', linewidth=1.0)

ax2t = axes[2].twinx()
color_ot = "#EE0000"

# ot can only be calculated after both fingers touch -> correct for that offset (before force closure, the following calculation is biased by the non-touching finger's trajectory)
ot = (((qDes[:, 0]-qDes[:, 1])/2)*1000) - 8.2
ot[0:245] = ot[240:485]
otplot, = ax2t.plot(deltaT, ot, c="blue", label="$O(t)$")
ax2t.set_ylabel("Object position $O(t)$", fontsize=label_fs)
# ax2t.text(-0.1, 0.002, "$O(t = \\mathrm{III})$", fontsize="x-large")
# ax2t.tick_params(axis="y",direction="in", pad=-25)

axes[2].legend([fdiffplot, otplot], ["$f_1 - f_2 + m\\,\\mathbf{g}\\cdot\\mathbf{n}$", "$O(t)$"], loc=3, prop={'size': legend_fs})

for ax in axes[:-1].flatten():
    ax.get_xaxis().set_visible(False) # hide xlabels for all but last rows
    
for ax in axes:
    ax.tick_params(axis='both', which='both', labelsize=13)
ax2t.tick_params(axis='both', which='both', labelsize=13)

phases = [(0, 1.82, "Closing"),
          (1.82, 5.43, "Holding"),
          (5.43, 6.94, "Right"),
          (6.94, 9.42, "Relaxing"),
          (9.42, 10.72, "Left"),
          (10.72, deltaT[-1], "Relaxing")]

rhe = 1.05
pad=0.01
for start, end, text in phases:
    a = axes[0]
    a.add_patch(mpl.patches.FancyBboxPatch(
        (start+pad, rhe), width=end-start-2*pad, height=0.13, color="silver", transform=a.get_xaxis_transform(),
        clip_on=False, boxstyle="round,pad=0,rounding_size=0.04", mutation_aspect=1))
    a.text((start+end)/2, rhe+0.13/2, text, transform=a.get_xaxis_transform(), ha='center', va='center', fontsize=13)

events = [(0.69, "1st touch"), (1.82, "2nd touch"), (5.43, ""), (6.94, ""), (9.42, ""), (10.72, "")]
for i, a in enumerate(axes):
    for j, (x, text) in enumerate(events):
        if i == 0:
            offset = 0.17 if j == 0 else 0.13
            yoff = 0 if j == 0 else 0.1
            a.text(x+offset, 0.9-yoff, text, transform=a.get_xaxis_transform(), ha='center', va='center', fontsize="11")
        a.axvline(x=x, linestyle='--', color='gray', linewidth=1.)

fig.align_ylabels()
fig.text(0.5, 0.007, 'time [$s$]', fontsize=18, ha='center')
fig.tight_layout(rect=[0, 0.01, 1, 1])
fig.subplots_adjust(hspace=0.1)
fig.subplots_adjust(wspace=0.005)
plt.savefig("grasp_new.pdf", dpi=300)
plt.show()
