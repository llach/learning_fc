import os
import gym
import time
import numpy as np

from learning_fc.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(CheckpointCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    Modified version of an example from the stable baselines 2 docs.

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param mean_n: (int) number of episodes the mean will be calculated on
    :param verbose: (int)
    """

    def __init__(self, env: gym.Env, check_freq: int, save_path: str, mean_n=100, total_steps: int = 0, model_name: str = "_best_model",
                 offset: int = 1e4, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(
            total_steps=total_steps,
            freq=check_freq,
            offset=offset,
            env=env, 
            verbose=verbose)
            
        self.save_path = save_path
        self.weights_path = self.save_path + "/weights/"
        self.best_mean_reward = -np.inf
        self.mean_n = mean_n 
        self.start_time = time.time()
        self.model_name = model_name

        print(f"SaveOnBestTrainingRewardCallback: saving model after {self.offset} steps, checking every {self.freq} steps, for a total of {len(self.checkpoints)} checks")

    def _init_callback(self) -> None:
        # Create folders if needed
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)

    def _on_checkpoint(self) -> bool:
        # load training reward
        x, y = ts2xy(load_results(self.save_path), 'timesteps')
        if len(x) > 0:
            # mean training reward over the last N episodes
            mean_reward = np.mean(y[-self.mean_n:])

            # new best model, thus we save it
            if mean_reward > self.best_mean_reward:
                print("!!! new best mean reward {:.2f} !!! before: {:.2f}".format(mean_reward, self.best_mean_reward))
                self.best_mean_reward = mean_reward
                print("Saving new best model to {}".format(self.weights_path))
                self.model.save(f'{self.weights_path}/{self.model_name}')
            else:
                print("Best mean reward was: {:.2f}, current: {:.2f}".format(self.best_mean_reward, mean_reward))

            if self.total_steps > 0:
                time_elapsed = time.time() - self.start_time
                fps = int(self.num_timesteps / (time_elapsed + 1e-8))
                eta_m = int((self.total_steps - self.num_timesteps) / fps / 60)

                print(f"ETA {int(eta_m / 60)}:{eta_m % 60:02d} ||  FPS {fps}")
        return True