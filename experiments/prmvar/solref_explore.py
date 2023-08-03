import mujoco
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from learning_fc import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.live_vis import TactileVis

with_vis = 0
trials   = 5
steps    = 25

STIFF_RANGE = [0.008,0.05]
DAMP_RANGE  = [0.8,1.1]

env = GripperTactileEnv(
    oy_range=[0,0],
    wo_range=[0.02, 0.02],
    dq_max=0.01,
    **{"render_mode": "human"} if with_vis else {}
)
vis = TactileVis(env) if with_vis else None

q = np.zeros((trials**2, steps, 2))
qdes = np.zeros((trials**2, steps, 2))
forces = np.zeros((trials**2, steps, 2))

params = np.zeros((trials**2, 2))

i=0
for stiff in np.linspace(*STIFF_RANGE, trials):
    for damp in np.linspace(*DAMP_RANGE,  trials):
        params[i] = [stiff, damp]
        env.set_solver_parameters(solref=params[i])

        env.reset()
        if vis: vis.reset()

        for j in range(steps):
            action = [-1,-1]
            obs, r, _, _, _ = env.step(action)
            if vis: vis.update_plot(action=action, reward=r)

            q[i,j]=env.q
            qdes[i,j]=env.qdes
            forces[i,j]=env.force
        i += 1

env.close()

fig, axes = plt.subplots(nrows=trials, ncols=trials, figsize=(9,6))

xs = np.arange(steps)
for i, ax in enumerate(axes.flatten()):
    ax.plot(xs, forces[i])

for i, stiff in enumerate(np.linspace(*STIFF_RANGE, trials)):
    for j, damp in enumerate(np.linspace(*DAMP_RANGE,  trials)):
        if i == 0: 
            axes[i,j].xaxis.set_label_position('top') 
            axes[i,j].set_xlabel(f"d={damp:.2f}")
        if j == 0: axes[i,j].set_ylabel(f"s={stiff:.4f}")
fig.tight_layout()
plt.show()
exit()

# forces /= 100*env.wo

labels = []
curves = [tuple() for _ in range(trials)]
for i, (v, ftraj) in enumerate(zip(values, forces)):
    c, = plt.plot(ftraj, alpha=0.8)
    curves[i] += tuple([c])
    labels.append(f"{pname}={v:.4f}")
plt.ylabel("f(t) [left]")
plt.xlabel("t")
l = plt.axhline(1.0, ls="dashed", c="cyan", lw=1)
curves.append(tuple([l]))
labels.append("f final")

# ax2 = plt.twinx()
# for i, qs in enumerate(q):
#     c = ax2.plot(qs[:,0])
#     curves[i] += tuple(c)

# l = ax2.axhline(env.wo, ls="dashed", c="grey", lw=1)
# curves.append(tuple([l]))
# labels.append("obj. radius")

# ax2.set_ylim(0.005, 0.033)
# ax2.set_ylabel("joint position")

plt.legend(curves, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower right", ncol=trials//3, shadow=True)

si = env.SOLIMP
si[sidx] = pname
fmax=np.max(forces)
qmin=np.min(q)
pdep=env.wo-qmin
plt.title(f"SOLIMP={si} | fmax={fmax:.3f} | pdepth={pdep:.5f}")

plt.tight_layout()
plt.show()