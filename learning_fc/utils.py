import os
import json
import mujoco
import inspect
import numpy as np

from typing import Any, List
from datetime import datetime
from learning_fc import datefmt
from scipy.signal import butter, lfilter

""" evaluation
"""

def find_latest_model_in_path(path, filters=[]):
    # only consider top-level trial folders
    trial_names = [x[0] for x in os.walk(path) if "__" in x[0].split("/")[-1]] 
    
    # filter trial names by name
    if len(filters) > 0: trial_names = [tn for tn in trial_names if all(filt in tn for filt in filters)]

    # get the latest trial of all matching ones
    trial_name = None
    date = datetime(1999, 1, 1)
    for tn in trial_names:
        d = datetime.strptime(tn.split("/")[-1].split("__")[0], datefmt)
        if d > date:
            trial_name = tn
            date = d

    assert trial_name is not None, f"could not find a trial under {path} with filters {filters}"
    return trial_name

def get_q_f(env, n_steps, qdes=-1):
    """ quick rollouts for gripper closing
    """
    q_env = []
    f_env = []
    env.reset()
    for _ in range(n_steps):
        q_env.append(env.q)
        f_env.append(env.force)
        env.step ([qdes,qdes])
    q_env = np.array(q_env)
    f_env = np.array(f_env)
    return q_env, f_env

""" model / env creation
"""

def get_constructor_params(cls, obj=None):
    """ 
    get constructor arguments and return dict with their values of the instantiated object.
    `obj.__getattribute__(k)` only works if the class stores a reference to the variable. if not we'll get the default value specified in the constructor
    """
    init_args = list()
    objparams = inspect.signature(cls).parameters
    for k in objparams.keys():
        if k not in ["args", "kwargs", "kw"]: init_args.append(k)

    # https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
    # to weed out things we can't store
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    # this could be done elegantly with dict comprehension, but the jsonable check requires us to do it in a long loop
    cparams = dict()
    for k in init_args:
        p = obj.__getattribute__(k) if obj and hasattr(obj, k) else objparams[k].default
        if is_jsonable(p): cparams = {**cparams, k: p}
    return cparams

def safe_unwrap(e):
    """ unwraps an environment if it's wrapped
    """
    if hasattr(e, "unwrapped"): return e.unwrapped
    else: return e

""" MuJoCo
"""

def interp(v, interval):
    int_len = interval[1]-interval[0]
    return interval[0] + v*int_len

def safe_rescale(x, bounds1, bounds2=[-1,1]):
    x = np.clip(x, *bounds1) # make sure x is within its interval
    
    low1, high1 = bounds1
    low2, high2 = bounds2
    return (((x - low1) * (high2 - low2)) / (high1 - low1)) + low2

def total_contact_force(model, data, g1, g2):
    ft = np.zeros((6,))
    ncontacts = 0

    for i, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if (g1==name1 and g2==name2) or (g1==name2 and g2==name1):
            c_ft = np.zeros((6,))
            mujoco.mj_contactForce(model, data, i, c_ft)
            ft += c_ft
            ncontacts += 1
    return ft[:3], ft[3:], ncontacts

def get_pad_forces(model, data):
    fl, fr = 0, 0
    for j, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if name1 != "object":  continue # lowest ID geoms come first
        if name2[:3] != "pad": continue # geoms for force measurements need to have "pad" in their name

        c_ft = np.zeros((6,))
        mujoco.mj_contactForce(model, data, j, c_ft)
        # f = c_ft[0] # only consider normal force
        f = sum(c_ft[:3])

        if name2[-2:] == "_l":   fl += f 
        elif name2[-2:] == "_r": fr += f
        else: print(f"unknown pad {name2}")

    return np.array([fl, fr])

""" filtering
"""

def analyze_freqs(signal, dt=1.):
    nfreqs = int((len(signal)/2)+1)

    freqs = np.fft.fftfreq(len(signal), dt)[:nfreqs]
    mags  = np.abs(np.fft.fft(signal))[:nfreqs]

    return freqs, mags

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def rolling_butterworth_filter(data, window_size, cutoff_freq, fs, order=2):
    pad_width = window_size - 1
    padded_data = np.pad(data, (pad_width, 0), mode='constant')
    filtered_data = np.zeros_like(data)

    # Define the Butterworth filter parameters
    nyquist_freq = 0.5 * cutoff_freq
    b, a = butter(order, nyquist_freq, fs=fs, btype='low', analog=False, output='ba')

    # Apply the Butterworth filter on each window of the padded data
    for i in range(len(data)):
        window = padded_data[i:i+window_size]
        filtered_window = lfilter(b, a, window)
        filtered_data[i] = filtered_window[-1]

    return filtered_data

""" reading and writing CSVs 
"""

class CsvWriter:

    def __init__(self, file: str, headers: List[str]):
        self.file = file
        self.headers = headers

        self.headers_line = ','.join(self.headers)

        self.file_handler = open(self.file, "wt")
        self.file_handler.write(f'{self.headers_line}\n')
        self.file_handler.flush()

    def write(self, vals: List[Any]):
        self.file_handler.write(f"{','.join([str(v) for v in vals])}\n")
        self.file_handler.flush()

    def close(self):
        self.file_handler.close()

class CsvReader:

    def __init__(self, fi):
        self.data = {}
        self.headers = []

        with open(fi, "r") as f:
            first_line = True
            for l in f.readlines():
                if first_line:
                    for h in l.split(","):
                        h = h.replace("\n", "")
                        self.data.update({
                            h: []
                        })
                        self.headers.append(h)
                    first_line = False
                else:
                    for i, d in enumerate(l.split(",")):
                        self.data[self.headers[i]].append(self._parse_number(d))

    def _parse_number(self, n):
        try:
            if "[" in n:
                return np.fromstring(
                    n.replace("[", "").replace("]", ""), 
                    dtype=float, 
                    sep=" "
                )
            elif "." in n: return float(n)
            else: return int(n)
        except Exception as e:
            print(f"cannot convert {n} to neither int nor float:\n{e}")
            return 0.0