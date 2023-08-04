import time
import numpy as np

from learning_fc import model_path
from learning_fc.training import make_eval_env_model
from learning_fc.utils import find_latest_model_in_path

trial = find_latest_model_in_path(model_path, filters=["ppo"])
env, model, _, _ = make_eval_env_model(trial, with_vis=False, checkpoint="best")

obs_shape = env.observation_space.shape
for _ in range(100):
    obs = np.random.normal(0,1,obs_shape).astype(np.float32)

    start = time.time()
    model.predict(obs, deterministic=True)
    dt = time.time() - start
    print(dt)