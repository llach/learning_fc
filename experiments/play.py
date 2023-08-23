import numpy as np
import learning_fc
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import plot_rollouts
from learning_fc.utils import find_latest_model_in_path
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb

N_GOALS  = 4
with_vis = 1
# trial = f"{model_path}/30_base"
# trial = f"{model_path}/2023-08-11_17-38-07__gripper_tactile__ppo__pos_delta__obs_q-qdes-f-df-hadC-act__nenv-10__k-3"
trial = find_latest_model_in_path(model_path, filters=["ppo"])
print(trial)

env, model, vis, _ = make_eval_env_model(
    trial, 
    with_vis=with_vis, 
    checkpoint="best", 
    env_override = dict(
        # model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        # f_scale=3.0,
        # sample_fscale=True,
        # fgoal_range=[0.05, 0.6],
        # sample_biasprm=True
        # noise_f=0.002,
        # ftheta=0.008,
    )
)

# goals = np.linspace(*env.fgoal_range, N_GOALS)
# res = deterministic_eval(env, model, vis, goals, reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)

# res["q"] = np.concatenate(res["q"])
# res["qdes"] = np.concatenate(res["qdes"])
# res["qdot"] = np.concatenate(res["qdot"])
# res["qacc"] = np.concatenate(res["qacc"])
# res["force"] = np.concatenate(res["force"])

# res["r_force"] = np.concatenate(res["r_force"]).reshape((-1,))
# res["r_obj_pos"] = np.concatenate(res["r_obj_pos"]).reshape((-1,))
# res["r_obj_prox"] = np.concatenate(res["r_obj_prox"]).reshape((-1,))
# res["r_act"] = np.concatenate(res["r_act"]).reshape((-1,))
# res["actions"] = np.concatenate(res["actions"])
# res["obj_pos"] = np.concatenate(res["obj_pos"])
# res["obj_v"] = np.concatenate(res["obj_v"]).reshape((-1,))
# res["f_scale"] = np.concatenate(res["f_scale"]).reshape((-1,))
# res["cumr"] = np.concatenate(res["cumr"]).reshape((-1,))
# res["goals"] = np.repeat(np.array(res["goals"]).reshape((-1,)), 200)
# res["r"] = np.array(res["r"]).reshape((-1,))
# res["eps_rew"] = np.array(res["eps_rew"]).reshape((-1,))
# res["in_contact"] = np.array(res["in_contact"])

# model = ForcePI(env)#, Kp=0.1, Ki=0.5)
# model = StaticModel(-1)
# model = PosModel(env)

# env.set_attr("fgoal_range", [0.05, 0.6])
# env.set_attr("sample_solref", False)
# env.set_attr("sample_solimp", False)

# env.set_solver_parameters(solref=[0.05, 1.1])


fscales = np.linspace(*env.FSCALE_RANGE, N_GOALS)


cumrews = np.zeros((N_GOALS,))
for i in range(N_GOALS):
    obs, _ = env.reset()

    env.set_attr("f_scale", fscales[i])
    # env.set_goal((fscales[i]*env.FMAX)/2)
    env.set_goal(np.random.uniform(0.05, fscales[i]*env.FMAX))

    if isinstance(model, ForcePI): model.reset()
    if vis: vis.reset()
    
    for j in range(200):
        ain, _ = model.predict(obs, deterministic=True)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        cumrews[i] += r
    print(cumrews[i])

print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
env.close()

# plot_rollouts(env, res, trial)
# plt.show()