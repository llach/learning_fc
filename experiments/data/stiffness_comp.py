import pandas
import pathlib
import numpy as np
from numpy.polynomial.polynomial import polyfit
pa = pathlib.Path(__file__).parent.absolute()

from learning_fc import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode


objects = [
    "sponge",
    "glue",
    "pringles",
    "wood"
]

real_data = dict()
for obj in objects:
    real_data = {
        **real_data,
        obj: pandas.read_csv(f"{pa}/{obj}.csv")
    }
dqs = real_data[objects[0]].dq.values

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

for i, (obj, df) in enumerate(real_data.items()):
    if i != 3: continue
    b, m = polyfit(df.dq, df.f, 1)
    ax.scatter(df.dq, df.f, label=f"real b={b:.5f} m={m:.2f}")
    ax.plot(df.dq, b + m*df.dq, linestyle='-', alpha=0.3)
    print("real", b, m)
    break

env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.035, 0.035],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path="assets/pal_force.xml",
)
env.f_m = 1
kappas = [
    0.65,
    0.6, # glue
    0.5, # pringles
]

env.f_m = 3.33
sim_data = np.zeros_like(dqs)
for i, dq in enumerate(dqs):
    _, fs = get_q_f(env, 200, qdes=safe_rescale(-dq, [-env.dq_max, env.dq_max]))
    sim_data[i] = np.mean(fs[-1])

b, m = polyfit(dqs, sim_data, 1)
print("sim", b, m)
ax.scatter(dqs, sim_data, label=f"sim b={b:.5f} m={m:.2f}")#3.7*sim_data**1.35)
ax.legend()
ax.set_xlabel("delta q")
ax.set_ylabel("final force")

fig.suptitle(obj)
fig.tight_layout()
plt.show()

pass