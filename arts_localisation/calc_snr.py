#!/usr/bin/env python3
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


def get_burst_window(burst, **config):
    """
    Load a single SB and determine the arrival time of a burst within window_load and window_zoom of the centre,
    as well as the boxcar width corresponding to the highest S/N

    :param str burst: Name of the burst key in the config
    :param config: S/N configuration
    :return: startbin_wide, chunksize_wide, startbin_small, chunksize_small, boxcar_width (all int)
    """
    # initialise the filterbank reader
    fil_reader = ARTSFilterbankReader(config[burst]['filterbank'], config[burst]['main_cb'])
    # load the file
    chunksize_wide = int(config['window_load'] / fil_reader.tsamp)
    try:
        samp_arr = config[burst]['toa_filterbank'] / fil_reader.tsamp
    except KeyError:
        logger.debug('Could not read toa_filterbank from config, assuming burst occurs in centre of filterbank data')
        samp_arr = .5 * fil_reader.nsamp
    startbin_wide = int(samp_arr - .5 * chunksize_wide)
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
    # startbin_small is relative to chunksize_small
    startbin_small = int(.5 * (chunksize_wide - chunksize_small))
    # get the optimum boxcar width
    _, boxcar_width = calc_snr_matched_filter(ts, widths=range(1, config['width_max'] + 1))
    return startbin_wide, chunksize_wide, startbin_small, chunksize_small, boxcar_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--output_folder', help='Output folder '
                                                '(Default: <yaml file folder>/snr)')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    group_overwrites = parser.add_argument_group('config overwrites', 'These arguments overwrite the values set '
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
        # default output folder is same folder as .yaml file plus "snr"
        args.output_folder = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'snr')
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
        startbin_wide, chunksize_wide, startbin_small, chunksize_small, boxcar_width = get_burst_window(burst, **config)

        # initialise SB vs S/N plot. One plot per burst, one panel per CB
        if args.show_plots or args.save_plots:
            ncb = len(config[burst]['cbs'])
            nrow = int(np.ceil(np.sqrt(ncb)))
            ncol = int(np.ceil(ncb / nrow))
            nplot_empty = nrow * ncol - ncb
            if ncb > 1:
                size = 16
            else:
                size = 8
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(size, size * int(np.round(ncol / nrow))),
                                     sharex=True, sharey=True, squeeze=False)
            axes = axes.flatten()

        # Run S/N determination loop for each CB
        logger.info('Calculating S/N in each SB of given CBs')
        for i, cb in enumerate(tqdm(config[burst]['cbs'], desc='CB')):
            skip_processing = False
            output_file = f'{output_prefix}_{burst}_CB{cb:02d}_SNR.txt'
            if os.path.isfile(output_file):
                logger.warning(f"Checking existing results for {burst} CB{cb:02d} because output file already exists: {output_file}")
                sb_all, snr_all = np.loadtxt(output_file, unpack=True)
                if not len(sb_all) == NSB:
                    logger.warning(f"Number of sbs in {output_file} ({len(sb_all)}) does not match total number of SBs ({NSB}), "
                                   f"recreating output")
                else:
                    logger.info("Output file ok, loading existing results")
                    skip_processing = True

            if not skip_processing:
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
                    # get S/N using boxcar width as determined from optimum S/N in
                    # main SB (i.e. one with highest S/N in AMBER)
                    snr, _ = calc_snr_matched_filter(ts, widths=[boxcar_width])
                    snr_all[sb] = snr

                with open(output_file, 'w') as f:
                    f.write('#sb snr\n')
                    # format each line as sb, snr
                    for sb, snr in enumerate(snr_all):
                        f.write(f'{sb:02d} {snr:.2f}\n')

            # SB vs S/N plot
            if args.show_plots or args.save_plots:
                logger.info(f'Adding {burst} S/N to plot')
                ax = axes[i]
                ax.plot(range(NSB), snr_all, c='k', marker='o')
                # Add line S/N threshold if the value is available
                if 'snrmin' in config.keys():
                    ax.axhline(config['snrmin'], label='threshold')
                ax.set_xlim(0, NSB)
                ax.set_xlabel('SB index')
                ax.set_ylabel('S/N')
                ax.label_outer()
                title = f'CB{cb:02d}'
                if skip_processing:
                    # existing results loaded, mention in title
                    title = title + ' (loaded from disk)'
                ax.set_title(title)
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
