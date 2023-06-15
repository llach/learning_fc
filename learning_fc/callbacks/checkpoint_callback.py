import numpy as np

from learning_fc.callbacks import ProxyBaseCallback


class CheckpointCallback(ProxyBaseCallback):

    def __init__(self, env, total_steps: int, freq: int, offset: int = 0, verbose: bool = True):
        super(CheckpointCallback, self).__init__(env=env, verbose=verbose)

        self.offset = offset
        self.freq = freq
        self.total_steps = total_steps

        ckpts = np.arange(start=0, stop=self.total_steps, step=self.freq)
        self.checkpoints = []
        for c in ckpts:
            if c >= self.offset:
                self.checkpoints.append(c)
        self.checkpoints.append(self.total_steps)
        
        self.num_checkpoints = len(self.checkpoints)
        self.cidx = 0

    def _on_checkpoint(self):
        pass

    def _on_step(self) -> bool:
        # all checkpoints reached
        if self.cidx >= self.num_checkpoints:
            return
        if self.num_timesteps >= self.checkpoints[self.cidx]:
            self._on_checkpoint()
            self.cidx += 1