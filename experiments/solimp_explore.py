import mujoco
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from learning_fc import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.live_vis import TactileVis

with_vis = 0
trials   = 10
steps    = 50

# pname, sidx, values = "dmin", 0, np.linspace(0, 1, trials)
# pname, sidx, values = "width", 2, np.linspace(0.0001, 0.017, trials)
# pname, sidx, values = "midpoint", 3, np.linspace(0, 1, trials)
pname, sidx, values = "power", 4, np.arange(10)+1

env = GripperTactileEnv(
    obj_pos_range=[0,0],
    **{"render_mode": "human"} if with_vis else {}
    )
vis = TactileVis(env) if with_vis else None

# dertermine q delta for a certain velocity
vdes = 0.15 # m/s
qdelta = vdes*0.1
qd = safe_rescale(qdelta, [0,0.045], [-1,1])

q = np.zeros((trials, steps, 2))
qdes = np.zeros((trials, steps, 2))
forces = np.zeros((trials, steps))

for i in range(trials):
    solimp = env.SOLIMP
    solimp[sidx] = values[i]
    env.set_solver_parameters(solimp=solimp)

    env.reset()
    if vis: vis.reset()

    for j in range(steps):
        action = safe_rescale(
            np.clip(env.q - qdelta, 0, 0.045),
            [0, 0.045],
            [-1,1]
        )
        action=[-1,-1]
        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        forces[i,j]=env.force[0]
        qdes[i,j]=env.qdes
        q[i,j]=env.q
env.close()

plt.figure(figsize=(9,6))

labels = []
curves = [tuple() for _ in range(trials)]
for i, (v, ftraj) in enumerate(zip(values, forces)):
    c, = plt.plot(ftraj, alpha=0.8)
    curves[i] += tuple([c])
    labels.append(f"{pname}={v:.4f}")
plt.ylabel("f(t) [left]")
plt.xlabel("t")

ax2 = plt.twinx()
for i, qs in enumerate(q):
    c = ax2.plot(qs[:,0])
    curves[i] += tuple(c)

l = ax2.axhline(env.ow, ls="dashed", c="grey", lw=1)
curves.append(tuple([l]))
labels.append("obj. radius")

ax2.set_ylim(0.005, 0.033)
ax2.set_ylabel("joint position")
plt.legend(curves, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower right", ncol=trials//3, shadow=True)

si = env.SOLIMP
si[sidx] = pname
fmax=np.max(forces)
qmin=np.min(q)
pdep=env.ow-qmin
plt.title(f"SOLIMP={si} | fmax={fmax:.3f} | pdepth={pdep:.5f}")

plt.tight_layout()
plt.show()