#!/usr/bin/env python

import numpy as np
import argparse
import sys
import scipy.stats


def mean(fnames, out):
    ts, als = [], []
    for fname in fnames:
        t, al = np.loadtxt(fname, unpack=True)
        ts.append(t)
        als.append(al)

    als = np.array(als)
    ts = np.array(ts)

    np.savetxt(
        # out, zip(np.mean(ts, axis=0), np.nanmean(als, axis=0), scipy.stats.sem(als, axis=0)))
        out, zip(np.mean(ts, axis=0), np.median(als, axis=0), scipy.stats.sem(als, axis=0)))

parser = argparse.ArgumentParser(description='Mean fnames')
parser.add_argument('fnames', nargs='*')
parser.add_argument('-o', '--out')
args = parser.parse_args()

mean(args.fnames, args.out)
