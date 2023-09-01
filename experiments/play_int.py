import numpy as np

from learning_fc import model_path
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model
from learning_fc.utils import find_latest_model_in_path

trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(
    trial, 
    with_vis=True, 
    checkpoint="best", 
    env_override = dict(
        wo_range=[0.035, 0.035]
        # model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        # f_scale=3.0,
        # sample_fscale=True,
        # fgoal_range=[0.05, 0.6],
        # sample_biasprm=True
        # noise_f=0.002,
        # ftheta=0.008,
    )
)

def make_goals(low=0.1, high=0.6, step=0.1):
    a = np.arange(low, high, step)
    return np.concatenate([a, a[::-1][1:]])

obs, _ = env.reset()
if vis: vis.reset()

solis = [
    [0.0, 0.95, 0.002,  0.0, 1],
    [0.0, 0.95, 0.0275, 0.5, 2]
]

# for i, go in enumerate(make_goals()):
    
#     env.set_solver_parameters(solimp=solis[i%2])
#     for j in range(150):
#         ain, _ = model.predict(obs, deterministic=True)

#         if j%120 == 0: 
#             env.set_goal(go)
#             if vis: vis.draw_goal()

#         obs, r, _, _, _ = env.step(ain)
#         if vis: vis.update_plot(action=ain, reward=r)

# env.close()

env.change_stiffness(1)
env.set_goal(env.fgoal_range[1])
vis.draw_goal()

for kappa in np.arange(1, 0, -0.1):
    env.change_stiffness(kappa)
    for j in range(100):
        # kappa = 0.7 if j<0.3 else 1-j
        # env.change_stiffness(kappa)

        ain, _ = model.predict(obs, deterministic=True)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

env.close()