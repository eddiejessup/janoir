import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import argparse
import sys
import scipy.stats

ts, als = [], []
for fname in sys.argv[1:-1]:
    t, al = np.loadtxt(fname, unpack=True)
    ts.append(t)
    als.append(al)

als = np.array(als)
ts = np.array(ts)

np.savetxt(sys.argv[-1], list(zip(np.mean(ts, axis=0), np.nanmean(als, axis=0), scipy.stats.sem(als, axis=0))))
