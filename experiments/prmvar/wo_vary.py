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

env = GripperTactileEnv(
    oy_range=[0,0], # keep object centered
    **{"render_mode": "human"} if with_vis else {}
    )
vis = TactileVis(env) if with_vis else None

# dertermine q delta for a certain velocity
vdes = 0.15 # m/s
qdelta = vdes*0.1
qd = safe_rescale(qdelta, [0,0.045], [-1,1])

dp = np.zeros((trials,))
ffinal = np.zeros((trials,))
q = np.zeros((trials, steps, 2))
qdes = np.zeros((trials, steps, 2))
forces = np.zeros((trials, steps))

WO_RANGE = env.wo_range.copy()
wos = np.linspace(*WO_RANGE, trials)
for i, woi in enumerate(wos):
    env.wo_range = [woi,woi]

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

    ffinal[i] = np.median(forces[i,20:])
    dp[i] = woi-np.min(q[i,20:,0])
env.close()

plt.figure(figsize=(9,6))

plt.plot(wos, ffinal,  label="f_final")
plt.ylabel("f_final")
plt.xlabel("object width")
plt.ylim(0.0, 3.7)

ax2 = plt.twinx()
ax2.plot(wos, dp, c="orange", label="dp")
ax2.set_ylim(0.0, 0.01)

plt.legend()
plt.title(f"wo_range={WO_RANGE} | SOLIMP={env.solimp}")
plt.tight_layout()
plt.show()