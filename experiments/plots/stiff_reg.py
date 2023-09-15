import pickle

from os import listdir
from os.path import isfile, join

datadir = "/Users/llach/data/stiff_exp/"

dqs = []
spg_data = []
wood_data = []

onlyfiles = [f for f in listdir(datadir) if isfile(join(datadir, f))]

for of in onlyfiles:
    if "pkl" not in of: continue
    with open(join(datadir, of), "rb") as f:
        d = pickle.load(f)
    pass