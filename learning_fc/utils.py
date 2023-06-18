import os
import json
import mujoco
import inspect
import numpy as np

from datetime import datetime
from learning_fc import datefmt

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
        d = datetime.strptime(tn.split("__")[0], datefmt)
        if d > date:
            trial_name = tn
            date = d

    assert trial_name, f"could not find a trial under {path} with filters {filters}"
    return trial_name

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
        if is_jsonable(p): cparams |= {k: p}
    return cparams

def safe_unwrap(e):
    """ unwraps an environment if it's wrapped
    """
    if hasattr(e, "unwrapped"): return e.unwrapped
    else: return e

""" MuJoCo
"""

def safe_rescale(x, bounds1, bounds2=[-1,1]):
    x = np.clip(x, *bounds1) # make sure x is within its interval
    
    low1, high1 = bounds1
    low2, high2 = bounds2
    return (((x - low1) * (high2 - low2)) / (high1 - low1)) + low2

def total_contact_force(model, data, g1, g2):
    force = np.zeros((3,))
    ncontacts = 0
    for i, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if (g1==name1 and g2==name2) or (g1==name2 and g2==name1):
            ft = np.zeros((6,))
            mujoco.mj_contactForce(model, data, i, ft)
            force += ft[:3]
            ncontacts += 1
    return force, ncontacts