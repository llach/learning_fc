import numpy as np
import learning_fc
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import plot_rollouts
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 10
with_vis = 1;
# trial = f"{model_path}/30_base"
# trial = f"{model_path}/2023-08-03_17-09-24__gripper_tactile__ppo__pos_delta__obs_q-qdes-f-df-hadC-act__nenv-10__k-1"
trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(
    trial, 
    with_vis=with_vis, 
    checkpoint="best", 
    env_override = dict(
        model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        f_scale=3.0,
        sample_fscale=True,
        fgoal_range=[0.05, 0.6],
        sample_biasprm=True
    )
)

# model = ForcePI(env, Kp=0.1, Ki=0.5)
# model = StaticModel(-1)
# model = PosModel(env)

env.set_attr("fgoal_range", [0.05, 0.6])
env.set_attr("sample_solref", False)
# env.set_attr("sample_solimp", False)

# env.set_solver_parameters(solref=[0.05, 1.1])

cumrews = np.zeros((N_GOALS,))
for i in range(N_GOALS):
    obs, _ = env.reset()
    if isinstance(model, ForcePI): model.reset()
    if vis: vis.reset()
    print(env.f_scale)
    
    for j in range(150):
        ain, _ = model.predict(obs, deterministic=True)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        cumrews[i] += r
    print(cumrews[i])

print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
env.close()

# plot_rollouts(env, res, trial)
# plt.show()