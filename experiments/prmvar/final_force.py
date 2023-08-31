import learning_fc
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress
from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import get_q_f, safe_rescale
from learning_fc.enums import ControlMode
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE

def get_line_props(x, y):
    res = linregress(x, y)
    return np.min(y), np.max(y), res[:2]

nsteps=500
env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.035, 0.035],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
)

env.solimp = env.SOLIMP_HARD

qdeses = np.arange(0.0004, 0.003, 0.0002)
data = np.zeros((2, len(qdeses), 2))

for i, soli in enumerate([1, 0]):
    env.change_stiffness(soli)
    print(env.fmin, env.fmax)
    for j, qdes in enumerate(qdeses):
        _, f = get_q_f(env, nsteps, qdes=safe_rescale(
            -qdes,
            [-env.dq_max, env.dq_max]
        ))
        data[i,j] = [qdes, np.mean(f[-10:])]

# data[0,:,1] = -0.08 + 1.2  * data[0,:,1]
# data[1,:,1] = -0.09 + 1.01 * data[1,:,1]

plt.plot(data[0,:,0], data[0,:,1], label="soft")
plt.plot(data[1,:,0], data[1,:,1], label="hard")

print(get_line_props(data[0,:,0], data[0,:,1]))
print(get_line_props(data[1,:,0], data[1,:,1]))

plt.xlabel("q_des")
plt.ylabel("f^final")
plt.title("final forces for d")

plt.legend()
plt.tight_layout()
plt.show()