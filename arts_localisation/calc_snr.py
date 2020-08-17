#!/usr/bin/env python

import os
import logging
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arts_localisation import tools
from arts_localisation.config_parser import load_config
from arts_localisation.data_tools import ARTSFilterbankReader, calc_snr_matched_filter


logger = logging.getLogger(__name__)


def get_burst_arrival_time(**config):
    """
    Load a single SB and determine the arrival time of a burst
    :param ARTSFilterbankReader fil_reader: Filterbank reader
    :param config: S/N configuration
    :return float toa: burst arrival time at top of band
    """
    # initialise the filterbank reader
    fil_reader = ARTSFilterbankReader(config['filterbank'], config['main_cb'])
    # load the entire file for one SB
    sb = fil_reader.load_single_sb(config['main_sb'], 0, fil_reader.nsamp)
    # dedisperse and create timeseries
    sb.dedisperse(config['dm'])
    ts = sb.data.sum(axis=0)
    # find peak
    ind_max = np.amax(ts)
    return ind_max * fil_reader.tsamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--output_folder', help='Output folder '
                                                '(Default: current directory)')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    group_overwrites = parser.add_argument_group('Config overwrites', 'Settings to overwrite from yaml config')
    group_overwrites.add_argument('--window_load', type=float, help='Window size (seconds) to use when loading data, '
                                                                    'should be at least twice the DM delay '
                                                                    'across the band')
    group_overwrites.add_argument('--window_zoom', type=float, help='Window size (seconds) to use when zooming in '
                                                                    'on pulse after dedispersion')
    group_overwrites.add_argument('--snrmin', type=float, help='S/N threshold')
    group_overwrites.add_argument('--cbs', help='Comma-separated list of CBs to calculate S/N for')
    group_overwrites.add_argument('--neighbours', action='store_true', help='Whether or not to include the neightbours'
                                                                            'of the given CBs as well')
    group_overwrites.add_argument('--dm', type=float, help='Dispersion measure (pc/cc)')
    group_overwrites.add_argument('--filterbank', help='Path to filterbank files, including {cb} and {tab}')
    group_overwrites.add_argument('--main_cb', help='CB of main detection')
    group_overwrites.add_argument('--main_sb', help='SB of main detection')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # set the CBs argument as a proper list
    args.cbs = [int(val) for val in args.cbs.split(',')]

    # load config for S/N determination
    config = load_config(args, for_snr=True)

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        args.output_folder = os.getcwd()
    tools.makedirs(args.output_folder)
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))

    # find the arrival time of the burst in the file (assumed the same for all CBs)
    toa = get_burst_arrival_time(**config)
    print(toa)


if __name__ == '__main__':
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
