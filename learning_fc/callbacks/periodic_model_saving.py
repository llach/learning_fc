import os

from learning_fc.callbacks import ProxyBaseCallback


class PeriodicSavingCallback(ProxyBaseCallback):
    """
    Saves model all `save_freq` timesteps starting from `step_offset`.
    """

    def __init__(self, save_path: str, save_freq: int = 0, step_offset: int = 0, verbose=1):
        super(PeriodicSavingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_offset = step_offset

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls >= self.step_offset and (self.n_calls % self.save_freq == 0 or self.n_calls == self.step_offset):
            print(f"Saving checkpoint model at {self.n_calls}")
            self.model.save(f'{self.save_path}/model{int(self.n_calls/1000)}')

        return True
