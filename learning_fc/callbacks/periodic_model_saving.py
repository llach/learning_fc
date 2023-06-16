import os

from learning_fc.callbacks import ProxyBaseCallback


class PeriodicSavingCallback(ProxyBaseCallback):
    """
    Saves model all `save_freq` timesteps starting from `step_offset`.
    """

    def __init__(self, save_path: str, save_freq: int = 0, offset: int = 0, verbose=1):
        super(PeriodicSavingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.offset = offset
        self.weights_path = self.save_path + "/weights/"

        print(f"PeriodicSavingCallback: saving model every {self.save_freq} after {self.offset} steps")

    def _init_callback(self) -> None:
        # Create folders if needed
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.weights_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls >= self.offset and (self.n_calls % self.save_freq == 0 or self.n_calls == self.offset):
            print(f"PSC: saving checkpoint model at {self.n_calls}")
            self.model.save(f'{self.weights_path}/model{int(self.n_calls)}')

        return True
