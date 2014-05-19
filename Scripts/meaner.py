#!/usr/bin/env python

import numpy as np
import argparse
import sys
import scipy.stats


def mean(fnames, out, median):
    xs, ys = [], []
    for fname in fnames:
        x, y = np.loadtxt(fname, unpack=True)
        xs.append(x)
        ys.append(y)

    ys = np.array(ys)
    xs = np.array(xs)

    if median:
        y_av = np.median(ys, axis=0)
    else:
        y_av = np.nanmean(ys, axis=0)

    y_err = scipy.stats.sem(ys, axis=0)

    np.savetxt(out, zip(np.mean(xs, axis=0), y_av, y_err))

parser = argparse.ArgumentParser(description='Mean fnames')
parser.add_argument('fnames', nargs='*')
parser.add_argument('-o', '--out')
parser.add_argument('--median', default=False, action='store_true')
args = parser.parse_args()

mean(args.fnames, args.out, args.median)
