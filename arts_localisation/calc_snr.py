#!/usr/bin/env python

import os
import logging
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arts_localisation import tools
from .config_parser import load_config
from .data_tools import ARTSFilterbankReader, calc_snr_matched_filter


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--output_folder', default='.', help='Output folder '
                                                             '(Default: current directory)')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # load config for S/N determination
    config = load_config(args, for_snr=True)

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        args.output_folder = os.getcwd()
    tools.makedirs(args.output_folder)
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))


if __name__ == '__main__':
    main()

    # example:
    exit()
    fname = '../examples/data/CB{cb:02d}_10.0sec_dm0_t03610_sb-1_tab{tab:02d}.fil'
    dm = 349

    fil = ARTSFilterbankReader(fname, cb=0)

    # load 1.024 s around midpoint
    mid = int(fil.nsamp // 2)
    chunksize = int(1.024 / fil.tsamp)
    startbin = mid - int(chunksize // 2)

    # plot central SB
    sb = fil.load_single_sb(35, startbin, chunksize)
    sb.data -= sb.data.mean(axis=1, keepdims=True)
    sb.dedisperse(dm)
    sb.subband(128)
    sb.downsample(4)
    plt.imshow(sb.data, aspect='auto')

    # load all TABs and get S/N per SB
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
    ax.plot(range(nsb), snrs)
    ax.axhline(snrmin, ls='--', c='r')
    ax.set_xlabel('SB')
    ax.set_ylabel('S/N')
    plt.show()
