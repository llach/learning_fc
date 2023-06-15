import os
import json

from datetime import datetime

from learning_fc import model_path, datefmt
from learning_fc.training import make_env, make_model, tactile_eval

model_defaults = dict(
    ppo=dict(timesteps=5e5),
    td3=dict(timesteps=1e5),
    ddpg=dict(timesteps=1e5),
)

env_eval_fn = dict(
    gripper_tactile=tactile_eval
)

def generate_trial_name(env_name, model_name, name):
    datestr = datetime.utcnow().strftime(datefmt)

    trial_name = f"{env_name}__{model_name}"
    if name is not None: trial_name += f"__{name}"

    return trial_name+f"__{datestr}"

def train(env_name="gripper_tactile", model_name="ppo", env_kw={}, model_kw={}, train_kw={}, logdir=model_path, name=None, plot_title=None):
    # build training parameters dict, store locals
    tkw = model_defaults[model_name] | train_kw

    # build trial name and dir
    trial_name = generate_trial_name(env_name=env_name, model_name=model_name, name=name)
    trialdir = f"{logdir}/{trial_name}/"

    # easy ref to total steps as integer
    timesteps = int(tkw["timesteps"])

    fn_params = locals()

    # create log and trial dirs
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(trialdir, exist_ok=True)

    # environment setup
    env, _, eparams = make_env(env_name=env_name, logdir=trialdir, env_kw=env_kw, with_vis=False, training=True)

    # model setup
    model, callbacks, mparams = make_model(env=env, model_name=model_name, logdir=trialdir, model_kw=model_kw, timesteps=timesteps)
    
    # store parameters
    with open(f"{trialdir}/parameters.json", "w") as f:
        f.write(json.dumps(dict(
            make_env=eparams,
            make_model=mparams,
            train=fn_params
        ), indent=2, sort_keys=True))

    # train the agent
    try: model.learn(total_timesteps=timesteps, callback=callbacks)
    except (KeyboardInterrupt) as e: 
        print(f"\ngot exception while training:\n{e}\nattempting evaluation regardless")

    # create evaluation plots
    if env_name in env_eval_fn: env_eval_fn[env_name](trialdir, plot_title=plot_title, with_vis=False)