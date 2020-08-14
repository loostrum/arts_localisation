#!/usr/bin/env python

import logging

import numpy as np
from blimpy import Waterfall
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_tools import ARTSFilterbankReader, calc_snr_matched_filter


if __name__ == '__main__':
    fname = '../examples/data/CB{cb:02d}_10.0sec_dm0_t03610_sb-1_tab{tab:02d}.fil'
    dm = 349

    fil = ARTSFilterbankReader(fname, cb=0)

    # load 1.024 s around midpoint
    mid = int(fil.nsamp // 2)
    chunksize = int(1.024 / fil.tsamp)
    startbin = mid - int(chunksize // 2)
    fil.read_tabs(startbin, chunksize)

    nsb = 71
    snrmin = 8
    snrs = np.zeros(nsb)

    for sb in tqdm(range(nsb), desc="Getting S/N in each SB"):
        spec = fil.get_sb(sb)
        spec.dedisperse(dm, padval='rotate')
        ts = spec.data.sum(axis=0)
        snr, width = calc_snr_matched_filter(ts, widths=range(1, 101))
        snrs[sb] = snr

    fig, ax = plt.subplots()
    ax.scatter(range(nsb), snrs)
    ax.axhline(snrmin, ls='--', c='r')
    ax.set_xlabel('SB')
    ax.set_ylabel('S/N')
    plt.show()
