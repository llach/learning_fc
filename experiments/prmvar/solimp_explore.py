import learning_fc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from learning_fc import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.enums import ControlMode
from learning_fc.live_vis import TactileVis

with_vis = 0
trials   = 10
steps    = 200

# pname, sidx, values = "dmin", 0, np.linspace(0.0, 0.9, trials)
# pname, sidx, values = "width", 2, np.linspace(0.0007, 0.03, trials)
pname, sidx, values = "midpoint", 3, np.linspace(0, 0.5, trials)
# pname, sidx, values = "power", 4, np.arange(10)+1

# pname, sidx, values = "timeconst", 0, np.linspace(0.008, 0.05, trials)
# pname, sidx, values = "dampratio", 1, np.linspace(0.9, 1.1, trials)


env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.03, 0.03],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    **{"render_mode": "human"} if with_vis else {}
)
vis = TactileVis(env) if with_vis else None

q = np.zeros((trials, steps, 2))
forces = np.zeros((trials, steps))

SOLREF = [0.02, 1.0]
SOLIMP = [0.0, 0.99, 0.026, 0.5, 2]

for i in range(trials):
    solimp = SOLIMP
    solimp[sidx] = values[i]
    env.set_solver_parameters(solimp=solimp)

    # solref = SOLREF
    # solref[sidx] = values[i]
    # env.set_solver_parameters(solref=solref, solimp=SOLIMP)

    env.reset()
    if vis: vis.reset()

    for j in range(steps):
        action = [-1,-1] 
        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        forces[i,j]=env.force[0]
        q[i,j]=env.q
env.close()

""" PLOTTING
"""

fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(14 ,7), layout='constrained')

labels = []
curves = [tuple() for _ in range(trials)]
qmins = np.min(q[:,40:,:], axis=(1,2))
dps   = env.wo-qmins 
for i, (v, ftraj, dp) in enumerate(zip(values, forces, dps)):
    c, = ax1.plot(ftraj, alpha=0.8)
    curves[i] += tuple([c])
    labels.append(f"{v:.4f}|{dp:.4f}")
ax1.set_ylabel("f(t) [left]")
ax1.set_xlabel("t")
# l = ax1.axhline(1.0, ls="dashed", c="cyan", lw=1)
# curves.append(tuple([l]))
# labels.append("f final")

l = ax2.axhline(env.wo-env.xi_max, ls="dashed", c="green", lw=1)
curves.append(tuple([l]))
labels.append("xi max")

for i, qs in enumerate(q):
    c = ax2.plot(qs[:,0])
    curves[i] += tuple(c)

l = ax2.axhline(env.wo,  ls="dashed", c="grey", lw=1)
# ldes = ax2.axhline(qdes, ls="dashed", c="blue", lw=1)
curves.append(tuple([l]))
labels.append("obj. radius")
# curves.append(tuple([ldes]))
# labels.append("qdes")

ax2.set_ylim(0.005, 0.033)
ax2.set_ylabel("joint position")

fig.legend(curves, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="outside lower center", ncol=5)

si = SOLIMP
si[sidx] = pname
fmax=np.max(forces)
qmin=np.min(q)
pdep=env.wo-qmin
plt.title(f"SOLIMP={si} | fmax={fmax:.3f} | pdepth={pdep:.5f}")

# plt.tight_layout()
plt.show()