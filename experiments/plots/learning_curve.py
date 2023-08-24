import json

from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.plots import clean_lc, PLOTMODE


if __name__ == "__main__":
    trial = find_latest_model_in_path(model_path, filters=["ppo"])

    with open(f"{trial}/parameters.json", "r") as f:
        params = json.load(f)
    
    clean_lc(trial, params, mode=PLOTMODE.debug)