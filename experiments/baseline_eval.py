import numpy as np
import mujoco as mj
np.set_printoptions(suppress=True, precision=5)

from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI, StaticModel
from learning_fc.training import make_env

import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

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

N_GOALS = 5

# for st in np.linspace(0,.5,20):
env, _, _ = make_env(
    env_name="gripper_tactile", 
    training=False, 
    with_vis=False, 
    env_kw=dict(control_mode=ControlMode.Position, obs_config=ObsConfig.Q_DQ)#, j_arma=0.1, j_damp=1.0)#, obj_pos_range=[-0.03, -0.03])
)
model = ForcePI(env)

def after_cb(env, *args, **kwargs):
    data = env.data
    mdl  = env.model

    fl, fr = 0, 0
    for j, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if name1 != "object":  continue # lowest ID geoms come first
        if name2[:3] != "pad": continue # geoms for force measurements need to have "pad" in their name

        c_ft = np.zeros((6,))
        mj.mj_contactForce(mdl, data, j, c_ft)
        f = c_ft[0] # only consider normal force

        if name2[-2:] == "_l":   fl += f 
        elif name2[-2:] == "_r": fr += f
        else: print(f"unknown pad {name2}")

    if fl>0 or fr>0: print(fl, fr)
    return force_after_step_cb(env, *args, **kwargs)


res = oracle_results = deterministic_eval(env, model, None, np.linspace(0.3, 0.6, 6), reset_cb=force_reset_cb, after_step_cb=after_cb)

plot_rollouts(env, res, f"Baseline Rollouts")
plt.show()