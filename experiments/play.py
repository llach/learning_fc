import numpy as np
import learning_fc
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import plot_rollouts
from learning_fc.utils import find_latest_model_in_path, safe_rescale
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb

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
        # model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        # f_scale=3.0,
        # sample_fscale=True,
        # fgoal_range=[0.05, 0.6],
        # sample_biasprm=True
        # noise_f=0.002,
        # ftheta=0.008,
        sample_solimp=False
    )
)

model = StaticModel(safe_rescale(-0.00031, [-env.dq_max, env.dq_max], [-1,1]))

obs, _ = env.reset()
if vis: vis.reset()
for j in range(300):
    ain, _ = model.predict(obs, deterministic=True)

    obs, r, _, _, _ = env.step(ain)
    if vis: vis.update_plot(action=ain, reward=r)



# goals = np.linspace(*env.fgoal_range, N_GOALS)
# res = deterministic_eval(env, model, vis, goals, reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)

# model = ForcePI(env)#, Kp=0.1, Ki=0.5)

# model = PosModel(env)

# env.set_attr("fgoal_range", [0.05, 0.6])
# env.set_attr("sample_solref", False)
# env.set_attr("sample_solimp", False)

# env.set_solver_parameters(solref=[0.05, 1.1])


# fscales = np.linspace(*env.FSCALE_RANGE, N_GOALS)


# cumrews = np.zeros((N_GOALS,))
# for i in range(N_GOALS):
#     obs, _ = env.reset()

#     env.set_attr("f_scale", fscales[i])
#     # env.set_goal((fscales[i]*env.FMAX)/2)
#     env.set_goal(np.random.uniform(0.05, fscales[i]*env.FMAX))

#     if isinstance(model, ForcePI): model.reset()
#     if vis: vis.reset()
    
#     for j in range(300):
#         ain, _ = model.predict(obs, deterministic=True)

#         if j%120 == 0: 
#             env.set_goal(np.random.uniform(0.05, fscales[i]*env.FMAX))
#             if vis: vis.draw_goal()

#         obs, r, _, _, _ = env.step(ain)
#         if vis: vis.update_plot(action=ain, reward=r)

#         cumrews[i] += r
#     print(cumrews[i])

# print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
# env.close()

# # plot_rollouts(env, res, trial)
# # plt.show()