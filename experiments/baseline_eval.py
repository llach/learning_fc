import numpy as np
import mujoco as mj

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

N_GOALS  = 5
with_vis = 0
env, vis, _ = make_env(
    env_name="gripper_tactile", 
    training=False, 
    with_vis=with_vis, 
    max_steps=250,
    env_kw=dict(control_mode=ControlMode.Position, obs_config=ObsConfig.Q_DQ, max_contact_steps=100)
)
model = ForcePI(env, Kp=.7, Ki=1.5, verbose=0)
# model = StaticModel(0.0)

def after_cb(env, *args, **kwargs): 
    # print(env.force, env.qdes)
    return force_after_step_cb(env, *args, **kwargs)

goals = np.linspace(*env.fgoal_range, N_GOALS)
# goals = 5*[0.6]
print(f"goals={goals}")
res = deterministic_eval(env, model, vis, goals, reset_cb=force_reset_cb, after_step_cb=after_cb)

cumrs = np.array([cr[-1] for cr in res["cumr"]])
print(cumrs)
exit(0)
r_obj_pos = np.array(res["r_obj_pos"][-1])
# oy_t = np.array(res["oy_t"][-1])
objv = np.array(res["obj_v"][-1])[:,1]
x = range(len(r_obj_pos))

# plt.plot(r_obj_pos, label="r_obj_pos", color="orange")

# ax2 = plt.twinx()
# # ax2.plot(oy_t, label="oy_t")
# ax2.plot(objv, label="objv")

# plt.legend()
# plt.tight_layout()
# plt.show()

plot_rollouts(env, res, f"Baseline Rollouts")
plt.show()