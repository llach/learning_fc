import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import plot_rollouts
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 10
with_vis = 1;
trial = f"{model_path}/30_base"
trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(trial, with_vis=with_vis, checkpoint="best")

# model = ForcePI(env)
# model = StaticModel(-1)
# model = PosModel(env)

# env.set_attr("sample_solref", True)

env.set_solver_parameters(solref=[0.05, 1.1])

cumrews = np.zeros((N_GOALS,))
for i in range(N_GOALS):
    obs, _ = env.reset()
    if isinstance(model, ForcePI): model.reset()
    if vis: vis.reset()
    
    for j in range(250):
        ain, _ = model.predict(obs, deterministic=True)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        cumrews[i] += r
    print(cumrews[i])

print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
env.close()

# plot_rollouts(env, res, trial)
# plt.show()