from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class ProxyBaseCallback(BaseCallback):

    def __init__(self, env, verbose: bool = True):
        super(ProxyBaseCallback, self).__init__(verbose=verbose)

        self.env = env
        self.is_vec_env = type(env) == SubprocVecEnv or type(env) == VecMonitor

        if self.is_vec_env:
            self.nenv = len(env.remotes)
        else:
            self.nenv = 1
