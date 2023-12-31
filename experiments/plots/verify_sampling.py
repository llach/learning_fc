import pickle 

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import learning_fc

from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE

ntrials = 200
nsteps  = 70
wo      = 0.035

mode = PLOTMODE.debug

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[wo, wo],
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    noise_f=0.0,
    f_scale=3.1,
    sample_biasprm=True,
    randomize_stiffness=True
)

forces  = []
fscales = []
fgoal_abs = []
fgoal_rel = []
kappas = []

for _ in range(ntrials):
    _, f = get_q_f(env, nsteps)
    forces.append(f)
    fscales.append(env.f_scale)
    fgoal_abs.append(env.get_goal())
    fgoal_rel.append(env.get_goal()/env.fmax)
    kappas.append(env.kappa)

# with open("traj.pkl", "wb") as f:
#     pickle.dump(dict(
#         forces=forces,
#         fscales=fscales,
#         fgoal_abs=fgoal_abs,
#         fgoal_rel=fgoal_rel,
#         f_alphas=f_alphas
#     ), f)
# exit(0)

# with open("traj.pkl", "rb") as f:
#     data = pickle.load(f)

# forces  = data["forces"]
# fscales = data["fscales"]
# fgoal_abs = data["fgoal_abs"]
# fgoal_rel = data["fgoal_rel"]
# f_alphas = data["f_alphas"]

tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

xs = np.arange(nsteps)

for i, force in enumerate(forces):
    ax.plot(xs, force, alpha=0.4, c=cm.rainbow(kappas[i]))

setup_axis(
    ax, 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$f^\text{left}$" if tex else "force", 
    xlim=[0, nsteps], 
    ylim=[-0.05, 1.2],
    remove_first_ytick=True,
)

# ax.hist(fgoal_abs, bins=20)
# setup_axis(
#     ax, 
#     xlabel=r"$f^\text{goal}$" if tex else "fgoal", 
#     ylabel=r"#occurrences", 
#     xlim=[0, env.max_fmax], 
#     remove_first_ytick=True,
# )

# ax.hist(fgoal_rel, bins=20)
# setup_axis(
#     ax, 
#     xlabel=r"$\frac{f^\text{goal}}{f^\text{max}}$" if tex else "fgoal/fmax", 
#     ylabel=r"#occurrences", 
#     xlim=[0, 1.0], 
#     remove_first_ytick=True,
# )

plt.show()