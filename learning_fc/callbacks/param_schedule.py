import time
import json
import numpy as np

from gymnasium import Env
from learning_fc import CsvWriter
from learning_fc.callbacks import ProxyBaseCallback


class ParamSchedule:

    def __init__(self, var_name, start, stop, first_value, final_value, total_timesteps):
        if type(start) == float:
            self.start = int(start*total_timesteps)
        elif type(start) == int:
            self.start = start
        else:
            assert False, f"type {type(start)} of start not supported!"
            
        if type(stop) == float:
            self.stop = int(stop*total_timesteps)
        elif type(stop) == int:
            self.stop = stop
        else:
            assert False, f"type {type(stop)} of stop not supported!"

        self.var_name = var_name

        if isinstance(first_value, list):
            self.first_value = np.array(first_value)
            self.final_value = np.array(final_value)
        else:
            self.first_value = first_value
            self.final_value = final_value

        assert start < stop, "start < stop"
        self.dur = self.stop-self.start

    def get_value(self, t):
        alpha = np.clip((t - self.start)/self.dur, 0.0, 1.0)
        return self.first_value + alpha * (self.final_value-self.first_value)

class ParamScheduleCallback(ProxyBaseCallback):

    def __init__(self, env: Env, schedules: list[ParamSchedule], log_dir: str, write_freq: int, verbose: bool = True):
        super(ParamScheduleCallback, self).__init__(env, verbose=verbose)

        self.schedules = schedules
        self.log_dir = log_dir
        self.write_freq = write_freq

        self.filename = "scheduled_params.csv"
        self.file = f"{self.log_dir}/{self.filename}"

        self.write_headers = []

        for s in schedules:
            if not self.is_vec_env:
                assert hasattr(env, s.var_name), f"env has no attribute \'{s.var_name}\'"
            self.write_headers.append(s.var_name)

        self.write_headers.extend([
            "timesteps",
            "walltime"
        ])

        self.writer = CsvWriter(self.file, self.write_headers)
        self.t_start = None
        self.t = 0

    def _on_training_start(self) -> None:
        self.t_start = time.time()

    def _on_step(self) -> bool:

        for s in self.schedules:
                self.env.set_attr(s.var_name, s.get_value(self.num_timesteps))

        if self.num_timesteps % self.write_freq == 0:
            vals = []

            for s in self.schedules:
                if self.is_vec_env:
                    vals.append(self.env.get_attr(s.var_name)[0])
                else:
                    vals.append(getattr(self.env, s.var_name))

            vals.extend([
                self.num_timesteps,
                time.time() - self.t_start
            ])
            self.writer.write(vals)
        return True
