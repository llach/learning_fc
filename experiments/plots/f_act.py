import numpy as np

from learning_fc.plots import f_act_plot, PLOTMODE

if __name__ == "__main__":
    from learning_fc import model_path
    from learning_fc.utils import find_latest_model_in_path
    from learning_fc.training.evaluation import make_eval_env_model
    from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb

    trial = find_latest_model_in_path(model_path, filters=["ppo"])
    env, model, _, _ = make_eval_env_model(trial, with_vis=0, checkpoint="best")

    goals = np.linspace(*env.fgoal_range, 4)
    res = deterministic_eval(env, model, None, goals, reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)

    f_act_plot(res, mode=PLOTMODE.debug)
