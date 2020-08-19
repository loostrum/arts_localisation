#!/usr/bin/env python3
import os
import sys
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


def get_burst_window(burst, **config):
    """
    Load a single SB and determine the arrival time of a burst within window_load / window_zoom of the centre

    :param str burst: Name of the burst key in the config
    :param config: S/N configuration
    :return: startbin_wide, chunksize_wide, startbin_small, chunksize_small (all int)
    """
    # initialise the filterbank reader
    fil_reader = ARTSFilterbankReader(config[burst]['filterbank'], config[burst]['main_cb'])
    # load the file
    chunksize_wide = int(config['window_load'] / fil_reader.tsamp)
    startbin_wide = int(.5 * (fil_reader.nsamp - chunksize_wide))
    sb = fil_reader.load_single_sb(config[burst]['main_sb'], startbin_wide, chunksize_wide)
    # dedisperse and create timeseries
    sb.dedisperse(config['dm'])
    ts = sb.data.sum(axis=0)
    # find peak
    ind_max = np.argmax(ts)
    logger.debug(f"Found burst arrival time: {(startbin_wide + ind_max) * fil_reader.tsamp:.5f} s")
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
                                                '(Default: same directory as yaml config file)')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    group_overwrites = parser.add_argument_group('config overwrites', 'These arguments overwrite the values set'
                                                                      'in the .yaml config file')
    # global config:
    group_overwrites.add_argument('--snrmin', type=float, help='S/N threshold')
    group_overwrites.add_argument('--fmin_data', type=float, help='Lowest frequency of data')
    group_overwrites.add_argument('--bandwidth', type=float, help='Bandwidth of data')
    group_overwrites.add_argument('--width_max', type=int, help='Maximum width of matched filter in bins')
    # S/N config:
    group_overwrites.add_argument('--window_load', type=float, help='Window size (seconds) to use when loading data, '
                                                                    'should be at least twice the DM delay '
                                                                    'across the band')
    group_overwrites.add_argument('--window_zoom', type=float, help='Window size (seconds) to use when zooming in '
                                                                    'on pulse after dedispersion')
    group_overwrites.add_argument('--dm', type=float, help='Dispersion measure (pc/cc)')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        # default output folder is same folder as .yaml file
        args.output_folder = os.path.dirname(os.path.abspath(args.config))
    tools.makedirs(args.output_folder)
    # output prefix also contains the yaml filename without extension
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))

    # load config for S/N determination
    config = load_config(args, for_snr=True)

    # loop over bursts
    for burst in config['bursts']:
        logger.info(f'Processing burst {burst}')

        # find the burst window
        logger.info('Finding burst window in file')
        startbin_wide, chunksize_wide, startbin_small, chunksize_small = get_burst_window(burst, **config)

        # initialise SB vs S/N plot. One plot per burst, one panel per CB
        if args.show_plots or args.save_plots:
            ncb = len(config[burst]['cbs'])
            nrow = int(np.ceil(np.sqrt(ncb)))
            ncol = int(np.ceil(ncb / nrow))
            nplot_empty = nrow * ncol - ncb
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(16, 16 * int(np.round(ncol / nrow))),
                                     sharex=True, sharey=True, squeeze=False)
            axes = axes.flatten()

        # Run S/N determination loop for each CB
        logger.info('Calculating S/N in each SB of given CBs')
        for i, cb in enumerate(tqdm(config[burst]['cbs'], desc='CB')):
            output_file = f'{output_prefix}_{burst}_CB{cb:02d}_SNR.txt'
            if os.path.isfile(output_file):
                logger.warning(f"Skipping {burst} CB{cb:02d} because output file already exists: {output_file}")
                continue
            # initialise the filterbank reader
            fil_reader = ARTSFilterbankReader(config[burst]['filterbank'], cb)
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
                # set data outside of requested frequency range to zero
                remove_low = spec.freqs < config[burst]['fmin']
                remove_high = spec.freqs > config[burst]['fmax']
                remove_mask = np.logical_or(remove_low, remove_high)
                spec.data[remove_mask, :] = 0

                # create timeseries
                ts = spec.data.sum(axis=0)
                # get S/N
                snr, width = calc_snr_matched_filter(ts, widths=range(1, config['width_max'] + 1))
                snr_all[sb] = snr

            # store S/N file
            if np.all(snr_all < config['snrmin']):
                logger.warning(f'No S/N above threshold found for CB{cb:02d}, creating empty output file')

            with open(output_file, 'w') as f:
                f.write('#sb snr\n')
                # format each line, keep only values above S/N threshold
                for sb, snr in enumerate(snr_all):
                    if snr < config['snrmin']:
                        continue
                    f.write(f'{sb:02d} {snr:.2f}\n')

            # SB vs S/N plot
            if args.show_plots or args.save_plots:
                logger.info(f'Adding {burst} S/N to plot')
                ax = axes[i]
                ax.plot(range(NSB), snr_all, c='k', marker='o')
                ax.axhline(config['snrmin'], label='threshold')
                ax.set_xlim(0, NSB)
                ax.set_xlabel('SB index')
                ax.set_ylabel('S/N')
                ax.label_outer()
                ax.set_title(f'CB{cb:02d}')
                # only add legend to the first plot
                if i == 0:
                    ax.legend()

        # finalise the plot for this burst
        if args.show_plots or args.save_plots:
            fig.suptitle(burst)
            if args.save_plots:
                plt.savefig(f'{output_prefix}_{burst}_SNR.pdf', bbox_inches='tight')

    # show plot all the way at the end
    if args.show_plots:
        plt.show()

    logger.info("Done")


if __name__ == '__main__':
    main()
