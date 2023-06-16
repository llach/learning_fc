from stable_baselines3 import PPO, TD3, DDPG

from learning_fc.utils import get_constructor_params
from learning_fc.callbacks import SaveOnBestTrainingRewardCallback, PeriodicSavingCallback

modelname2cls = dict(
    ppo=PPO,
    td3=TD3,
    ddpg=DDPG,
)

model_default_params = dict(policy="MlpPolicy", verbose=1) # default parameters for all policies, regardless of the algorithm

def make_model(env, model_name, logdir, timesteps, model_kw={}, training=True, save_on_best=True, save_periodic=-1, weights=None):
    # store function params, remove non-hashable ones
    fn_params = locals()
    fn_params.pop("env")

    # get model class from name
    assert model_name in modelname2cls, f"unknown model {model_name}, available options {list(modelname2cls.keys())}"
    mcls = modelname2cls[model_name]

    # build model arguments and instantiate model
    mkw = model_default_params | model_kw
    model = mcls(env=env, **mkw)

    # either we train a model and create its callbacks, or we load weights for evaluation
    callbacks = []
    if training:
        if save_on_best:
            callbacks.append(
                SaveOnBestTrainingRewardCallback(
                    env=env,
                    check_freq=timesteps/25e1,
                    total_steps=timesteps,
                    save_path=logdir,
                    offset=1e3,
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
    elif weights is not None:
        model = model.load(f"{logdir}/weights/{weights}")
    else:
        assert False, "training != True && weights == None"

    # get model creation parameters to replicate the environment config while testing
    init_params = get_constructor_params(mcls, model)
    params = fn_params | dict(init_params=init_params) | dict(mkw=mkw)

    return model, callbacks, params