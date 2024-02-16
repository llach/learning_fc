import os
import time
import pickle
import numpy as np

from learning_fc import model_path, get_cumr_obj_delta
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model

dist = 0.1
width = 0.15
ntrials = 200
kappas = np.arange(0,1.1,0.2)

if __name__ == "__main__":

    trial = find_latest_model_in_path(model_path, filters=["ppo"])
    env, model, _, _ = make_eval_env_model(f"{model_path}/2024-02-11_16-40-14__gripper_tactile__ppo__k-3__lr-0.0006_M2_inbnew", with_vis=0, checkpoint="best")

    # use force reward only TODO should we also use the action penalty?
    env.set_attr("rp_scale", 0.0)
    env.set_attr("ro_scale", 0.0)
    env.set_attr("ra_scale", 0.0)
    env.set_attr("ah_scale", 0.0)
    env.set_attr("with_bias", False)

    """ Force Controller (Baseline)
    """

    fc = ForcePI(env)

    start = time.time()
    fc_res = []
    fc_objd = []
    for kappa in kappas:
        crs = []
        objds = []
        for _ in range(ntrials):
            fc.reset()
            cr, objd = get_cumr_obj_delta(env, fc)
            crs.append(cr)
            objds.append(objd)
        fc_objd.append(objds)
        fc_res.append(crs)
    print(f"fc took {time.time()-start}")

    print(np.mean(fc_res), np.std(fc_res))

    with open(f"{model_path}/fc_eval.pkl", "wb") as f:
        pickle.dump({
            "res": fc_res,
            "objd": fc_objd,
        }, f)
   
    """ Policy
    """
    start = time.time()
    pol_res = []
    env.set_attr("with_bias", True)
    for kappa in kappas:
        pol_res.append([
            get_cumr_obj_delta(env, model) for _ in range(ntrials)
        ])
    pr = np.array(pol_res)
    pol_res = pr[:,:,0]
    pol_objd = pr[:,:,1]
    env.set_attr("with_bias", False)
    print(f"pol took {time.time()-start}")

    print(np.mean(pol_res), np.std(pol_res))
    
    with open(f"{model_path}/pol_eval.pkl", "wb") as f:
        pickle.dump({
            "res": pol_res,
            "objd": pol_objd,
        }, f)

    """ Policy - without inductive bias
    """
    start = time.time()
    noib_res = []
    model.set_parameters(f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb/weights/_best_model.zip")
    for kappa in kappas:
        noib_res.append([
            get_cumr_obj_delta(env, model) for _ in range(ntrials)
        ])
    noib = np.array(noib_res)
    noib_res = noib[:,:,0]
    noib_objd = noib[:,:,1]
    print(f"noib took {time.time()-start}")

    with open(f"{model_path}/noib_eval.pkl", "wb") as f:
        pickle.dump({
            "res": noib_res,
            "objd": noib_objd,
        }, f)

    """ Policy - without domain randomization
    """
    start = time.time()
    nodr_res = []
    model.set_parameters(f"{model_path}/2023-09-15_08-22-36__gripper_tactile__ppo__k-3__lr-0.0006_M2_nor/weights/_best_model.zip")
    for kappa in kappas:
        nodr_res.append([
            get_cumr_obj_delta(env, model) for _ in range(ntrials)
        ])
    nodr = np.array(nodr_res)
    nodr_res = nodr[:,:,0]
    nodr_objd = nodr[:,:,1]
    print(f"nodr took {time.time()-start}")

    with open(f"{model_path}/nodr_eval.pkl", "wb") as f:
        pickle.dump({
            "res": nodr_res,
            "objd": nodr_objd,
        }, f)
        
    """ Policy - without learning curriculum
    """
    start = time.time()
    nocurr_res = []
    model.set_parameters(f"{model_path}/2024-02-05_13-51-07__gripper_tactile__ppo__k-3__lr-0.0006_M2_nocurr/weights/_best_model.zip")
    for kappa in kappas:
        nocurr_res.append([
            get_cumr_obj_delta(env, model) for _ in range(ntrials)
        ])
    nocurr = np.array(nocurr_res)
    nocurr_res = nocurr[:,:,0]
    nocurr_objd = nocurr[:,:,1]
    print(f"nocurr took {time.time()-start}")

    with open(f"{model_path}/nocurr_eval.pkl", "wb") as f:
        pickle.dump({
            "res": nocurr_res,
            "objd": nocurr_objd,
        }, f)

    """ Policy - without object movement penalty
    """
    start = time.time()
    nopen_res = []
    model.set_parameters(f"{model_path}/2024-02-11_21-26-18__gripper_tactile__ppo__k-3__lr-0.0006_M2_nopen/weights/_best_model.zip")
    for kappa in kappas:
        nopen_res.append([
            get_cumr_obj_delta(env, model) for _ in range(ntrials)
        ])
    nopen = np.array(nopen_res)
    nopen_res = nopen[:,:,0]
    nopen_objd = nopen[:,:,1]
    print(f"nopen took {time.time()-start}")
    print(f"rew {np.mean(nopen_res)}±{np.std(nopen_res)}")
    print(f"obj {np.mean(nopen_objd)}±{np.std(nopen_objd)}")

    with open(f"{model_path}/nopen_eval.pkl", "wb") as f:
        pickle.dump({
            "res": nopen_res,
            "objd": nopen_objd,
        }, f)