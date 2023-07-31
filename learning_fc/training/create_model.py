from stable_baselines3 import PPO, TD3, DDPG, SAC

from learning_fc.utils import get_constructor_params
from learning_fc.callbacks import SaveOnBestTrainingRewardCallback, PeriodicSavingCallback, ParamScheduleCallback

modelname2cls = dict(
    ppo=PPO,
    td3=TD3,
    sac=SAC,
    ddpg=DDPG,
)

model_default_params = dict(policy="MlpPolicy", verbose=1) # default parameters for all policies, regardless of the algorithm

def make_model(env, model_name, logdir, timesteps, model_kw={}, training=True, save_on_best=1, save_periodic=-1, weights=None, schedules=[]):
    # store function params, remove non-hashable ones
    fn_params = locals()
    fn_params.pop("env")

    # get model class from name
    assert model_name in modelname2cls, f"unknown model {model_name}, available options {list(modelname2cls.keys())}"
    mcls = modelname2cls[model_name]

    # build model arguments and instantiate model
    mkw = {**model_default_params, **model_kw}
    model = mcls(env=env, **mkw)

    # either we train a model and create its callbacks, or we load weights for evaluation
    callbacks = []
    if training:
        if save_on_best>0:
            if save_on_best == 1:
                offset = int(0.05*timesteps)
            else:
                offset = save_on_best

            callbacks.append(
                SaveOnBestTrainingRewardCallback(
                    env=env,
                    check_freq=timesteps/25e1,
                    total_steps=timesteps,
                    save_path=logdir,
                    offset=offset,
                    mean_n=100
                )
            )
        if save_periodic > 0:
            callbacks.append(
                PeriodicSavingCallback(
                    save_path=logdir, 
                    save_freq=save_periodic, 
                    offset=1e3
                )
            )
        if schedules is not []:
            callbacks.append(
                ParamScheduleCallback(
                    env=env, 
                    schedules=schedules, 
                    log_dir=logdir, 
                    write_freq=int(0.01*timesteps)
                )
            )
    if weights is not None:
        model.set_parameters(f"{weights}")

    # get model creation parameters to replicate the environment config while testing
    init_params = get_constructor_params(mcls, model)
    params = {**fn_params, **dict(init_params=init_params), **dict(mkw=mkw)}

    return model, callbacks, params