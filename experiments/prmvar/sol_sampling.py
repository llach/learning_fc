import mujoco
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from learning_fc import safe_rescale
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv
from learning_fc.live_vis import TactileVis

with_vis = 0
trials   = 5
steps    = 25

env = GripperTactileEnv(
    oy_range=[0,0],
    wo_range=[0.02, 0.02],
    sample_solref=True,
    sample_solimp=True,
    **{"render_mode": "human"} if with_vis else {}
)
vis = TactileVis(env) if with_vis else None

q = np.zeros((trials**2, steps, 2))
qdes = np.zeros((trials**2, steps, 2))
forces = np.zeros((trials**2, steps, 2))

params = np.zeros((trials**2, 2))

i = 0
for i in range(trials):
    for j in range(trials):
        env.reset()
        if vis: vis.reset()

        for k in range(steps):
            action = [-1,-1]
            obs, r, _, _, _ = env.step(action)
            if vis: vis.update_plot(action=action, reward=r)

            q[i*trials+j,k]=env.q
            qdes[i*trials+j,k]=env.qdes
            forces[i*trials+j,k]=env.force
env.close()

fig, axes = plt.subplots(nrows=trials, ncols=trials, figsize=(9,6))

xs = np.arange(steps)
for i, ax in enumerate(axes.flatten()):
    ax.plot(xs, forces[i])

fig.tight_layout()
plt.show()