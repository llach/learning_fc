import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from learning_fc import model_path, safe_unwrap
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts
from learning_fc.utils import find_latest_model_in_path

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
with_vis = 1
# trial = f"{model_path}/2023-07-19_14-28-19__centered__minimal_reward__nenv-6__k-1"
trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(trial, with_vis=with_vis, checkpoint="best")

def as_cb(env, model, i, results, goal=None, **kw):
    return force_after_step_cb(env, model, i, results, goal=None, **kw)

# model = ForcePI(env)
# model = StaticModel(-1)
# res = deterministic_eval(env, model, vis, np.linspace(*env.fgoal_range, N_GOALS), reset_cb=force_reset_cb, after_step_cb=as_cb)
# print(np.array(res["cumr"])[:,-1])

# model = PosModel(env)

n_actions = 10

for i in range(N_GOALS):
    obs, _ = env.reset()
    if isinstance(model, ForcePI): model.reset()

    if vis: vis.reset()

    cumrew = 0
    actions = deque(maxlen=n_actions)
    for _ in range(n_actions): actions.append([0,0])

    for j in range(300):
        ain, _ = model.predict(obs, deterministic=True)
        actions.append(ain)

        clean = butter_lowpass_filter(np.array(actions), 10, fs=100)
        # print(clean)
        # ain = np.mean(clean, axis=0)

        # ain    = np.array([-1,-1])

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        cumrew += r
    print(cumrew)
env.close()


# plot_rollouts(env, res, trial)
# plt.show()