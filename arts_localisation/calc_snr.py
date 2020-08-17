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
from arts_localisation.constants import NSB


logger = logging.getLogger(__name__)


def get_burst_window(**config):
    """
    Load a single SB and determine the arrival time of a burst within window_load / window_zoom of the centre
    :param ARTSFilterbankReader fil_reader: Filterbank reader
    :param config: S/N configuration
    :return: startbin_wide, chunksize_wide, startbin_small, chunksize_small (all int)
    """
    # initialise the filterbank reader
    fil_reader = ARTSFilterbankReader(config['filterbank'], config['main_cb'])
    # load the file
    chunksize_wide = int(config['window_load'] / fil_reader.tsamp)
    startbin_wide = int(.5 * (fil_reader.nsamp - chunksize_wide))
    sb = fil_reader.load_single_sb(config['main_sb'], startbin_wide, chunksize_wide)
    # dedisperse and create timeseries
    sb.dedisperse(config['dm'])
    ts = sb.data.sum(axis=0)
    # find peak
    ind_max = np.argmax(ts)
    # shift startbin such that ind_max is in the center of a chunk of size chunksize
    startbin_wide -= int(ind_max - .5 * chunksize_wide)
    # calculate the required parameters for the zoomed window
    chunksize_small = int(config['window_zoom'] / fil_reader.tsamp)
    # startbin_small is relative to startbin_wide
    startbin_small = int(.5 * (chunksize_wide - chunksize_small))
    return startbin_wide, chunksize_wide, startbin_small, chunksize_small


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

    # find the burst window
    logger.info('Finding burst window in file')
    startbin_wide, chunksize_wide, startbin_small, chunksize_small = get_burst_window(**config)

    # Run S/N determination loop for each CB
    logger.info('Calculating S/N in each SB of given CBs')
    for cb in tqdm(config['cbs'], desc='CB'):
        # initialise the filterbank reader
        fil_reader = ARTSFilterbankReader(config['filterbank'], cb)
        # load all TABs around the burst
        fil_reader.read_tabs(startbin_wide, chunksize_wide)
        # get the S/N of each SB
        snr_all = np.zeros(NSB)
        for sb in tqdm(range(NSB), desc='SB'):
            # get the SB data
            spec = fil_reader.get_sb(sb)
            # dedisperse
            spec.dedisperse(config['dm'], padval='rotate')
            # zoom in
            spec.data = spec.data[:, startbin_small:startbin_small + chunksize_small]
            # TODO: load frequency masking parameters in config_parser
            # bad_low = spec.freqs < config['fmin']
            # bad_high = spec.freqs > config['fmax']
            # bad_zero = spec.data.mean(axis=1) == 0
            # bad_data = np.logical_or(np.logical_or(bad_low, bad_high), bad_zero)
            # data.data[bad_data, :] = 0

            # create timeseries
            ts = spec.data.sum(axis=0)
            # get S/N
            # TODO: turns width range into parameter
            snr, width = calc_snr_matched_filter(ts, widths=range(1, 101))
            snr_all[sb] = snr

        if np.all(snr_all < config['snrmin']):
            logger.warning(f'No S/N above threshold found for CB{cb:02d}, not creating output file')
            continue

        # store S/N file
        output_file = f'{output_prefix}_SNR_CB{cb:02d}.txt'
        with open(output_file, 'w') as f:
            f.write('#sb snr\n')
            # format each line, keep only values above S/N threshold
            for sb, snr in enumerate(snr_all):
                if snr < config['snrmin']:
                    continue
                f.write(f'{sb:02d} {snr:.2f}\n')

        # plot
        fig, ax = plt.subplots()
        ax.scatter(range(NSB), snr_all, c='k')
        ax.axhline(config['snrmin'], label='Threshold')
        ax.set_xlabel('SB index')
        ax.set_ylabel('S/N')
        plt.show()

    logger.info("Done")


if __name__ == '__main__':
    main()
