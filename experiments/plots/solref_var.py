import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from learning_fc import model_path
from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode

env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.035, 0.035],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path="/assets/pal_force.xml",
    randomize_stiffness=False
)

def interp(int, k):
    return int[0]+(k*np.abs(int[1]-int[0]))

stiff_int = [0, 1] # small → big = soft → hard
damp_int = [0.1, 4] # small → big = hard → soft
actf_int = [3, 10] # small → big = hard → soft


stiff, damp, actf, kappa = "stiff", "damp", "act_p", "kappa"
smin, smax, pname = *stiff_int, stiff
# smin, smax, pname = *damp_int, damp
# smin, smax, pname = *actf_int, actf
# smin, smax, pname = 0, 1, kappa
st, da, af = 0.05, 10, 4


nsteps = 1000
ss = np.linspace(smin, smax, num=10)
xs = np.arange(nsteps)
force_trajs = []
pos_trajs = []

if pname == kappa: stiff_int.reverse() # other intervals go from hard to soft, this one is reversed.
for v in ss:
    if pname == actf: 
        af = v
    elif pname == damp: da = v 
    elif pname == stiff: 
        st = v
    elif pname == kappa:
        af = interp(actf_int, v)
        da = interp(damp_int, v)
        st = interp(stiff_int, v)

    print(da, st, af)
    env.actf = af
    env.solref = [-st, -da]
    # env.solref = [st, da]
    env.solimp = [0.001, 0.99, 0.001, 0.5, 1]
    q, f = get_q_f(env, nsteps, qdes=-1)
    force_trajs.append(np.mean(f, axis=1))
    pos_trajs.append(np.mean(q, axis=1))

title = ""
if pname == actf: title = f"damp={da} | stiff={st} | act_p"
elif pname == damp: title = f"stiff={st} | act_p={af} | damp"
elif pname == stiff: title = f"damp={da} | act_p={af} | stiff"
title += f" in [{smin}, {smax}]"

if pname == kappa:
    title = f"Kappa: stiff in [{','.join(map(str, stiff_int))}] | damp in [{','.join(map(str, damp_int))}] | act+p in [{','.join(map(str, actf_int))}]"

fig, ax = plt.subplots(ncols=3, figsize=(14,5))

cm = colormaps["viridis"]
for i, (q,f) in enumerate(zip(pos_trajs, force_trajs)):
    c = cm(ss[i]/smax)
    ax[0].plot(xs, f, c=c)

    ax1plt = ax[1].plot(xs, q, c=c)

    dp = np.abs(np.min([q-0.035, np.zeros_like(q)], axis=0))
    ax[2].plot(dp, f, c=c)

ax[1].set_ylim(0.0, 0.038)

ax[0].set_title("Force")
ax[1].set_title("Joint Position")

ax[0].set_xlabel("Steps")
ax[1].set_xlabel("Steps")

ax[2].set_xlabel("dp")
ax[2].set_ylabel("Force")

ax[1].axhline(env.wo_range[0], c="grey", lw=1, ls="--", label="object width")
ax[1].legend()

cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), ax=ax[1])
cbar.set_ticklabels([f"{t*smax:.2f}" for t in cbar.get_ticks()])

fig.suptitle(title)
fig.tight_layout()

# if mode == PLOTMODE.debug: 
# else:
plt.savefig(f"{model_path}/{pname}_var")
plt.show()