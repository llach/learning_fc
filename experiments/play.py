import numpy as np

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 10
with_vis = 1
# trial = f"{model_path}/30_base"
trial = find_latest_model_in_path(model_path, filters=["ppo"])
# trial = f"{model_path}/_archive/rp_experiments/2023-08-25_08-29-08__gripper_tactile__ppo__k-3"
print(trial)

env, model, vis, _ = make_eval_env_model(
    trial, 
    with_vis=with_vis, 
    checkpoint="best", 
    env_override = dict(
        control_mode=ControlMode.PositionDelta,
        oy_range=[0,0],
        wo_range=2*[0.035],
        # model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        # f_scale=3.0,
        # sample_fscale=True,
        # fgoal_range=[0.05, 0.6],
        # sample_biasprm=True
        # noise_f=0.002,
        # ftheta=0.008,
    )
)

# model = ForcePI(env, Kp=.1, Ki=1.1)
kappas = np.linspace(0, 1, N_GOALS)
# kappas = [0.4]
cumrews = np.zeros((N_GOALS,))
for i, kappa in enumerate(kappas):
    obs, _ = env.reset()

    env.change_stiffness(kappa)
    # env.set_solver_parameters(solimp=env.SOLIMP_SOFT)
    # env.set_attr("f_m", 1.78)
    env.set_goal(np.random.uniform(*env.fgoal_range))

    if isinstance(model, ForcePI): model.reset()
    if vis: vis.reset()
    
    for j in range(100):
        ain, _ = model.predict(obs, deterministic=True)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        cumrews[i] += r
    print(cumrews[i])

print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
env.close()

# plot_rollouts(env, res, trial)
# plt.show()