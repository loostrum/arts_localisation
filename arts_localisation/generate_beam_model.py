#!/usr/bin/env python3

import os
import sys
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle
import tqdm

from arts_localisation import tools
from arts_localisation.constants import CB_HPBW, REF_FREQ
from arts_localisation.beam_models.simulate_sb_pattern import SBPattern
from arts_localisation.config_parser import load_config

logger = logging.getLogger(__name__)
plt.rcParams['axes.formatter.useoffset'] = False


def blend_colors(args, weights=None):
    """
    args: array of colours, each as RGB tuple
    weight: weight to assign to each colour
    return: blended colour
    """

    if weights is None:
        weights = np.ones(len(args))

    # average
    out = np.sum(args * weights[..., None], axis=0) / weights.sum()
    # max 1
    out[out > 1] = 1.
    return out


def get_colour_model(sb_model, stretch_factor=5):
    """
    """
    nfreq, ndec, nha = sb_model.shape
    # init colours
    cmap = cm.get_cmap('rainbow')
    points = np.linspace(0, 1, nfreq, endpoint=True)
    colours = cmap(points)[::-1]  # to have red first

    # remove the alpha channel
    colours = np.array([c[:3] for c in colours])

    # colour image needs to be y,x,rgb
    colour_model = np.zeros((ndec, nha, 3))

    # loop over pixels, calculate their colour
    logger.info('Calculating colours')
    for y in tqdm.tqdm(range(ndec)):
        for x in range(nha):
            # intensity at each freq
            vals = sb_model[:, y, x]
            # blend colour
            c = blend_colors(colours, weights=vals)
            # intensity is SB intensity
            i = vals.mean()
            colour_model[y, x] = c * i
    colour_model *= stretch_factor

    # scale or clip such that max value is 1
    # if max < 1, scale by max
    if colour_model.max() < 1:
        colour_model /= colour_model.max()
    # else clip
    else:
        colour_model[colour_model > 1] = 1.
    return colour_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--burst', help='Burst name as in .yaml file  '
                        '(Default: assume burst name is the prefix of the .yaml file)')
    parser.add_argument('--cb', type=int, help='CB to generate (Default: "main_cb" value from .yaml file)')
    parser.add_argument('--sb', type=int, nargs='*', help='Space-separated list of SBs to generate '
                        '(Default: "main_sb" value from .yaml file)')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    parser.add_argument('--output', help='Output file for plot (Note: burst name and CB/SB indices will '
                        'be added to output filename. '
                        'Provide only a folder to let the script generate a filename automatically')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # nothing to do if plots are not saved and not shown
    if args.output is None and not args.show_plots:
        logger.error('Nothing to do; provide either an output file name for plots with --output, '
                     'or show plots with --show_plots')
        sys.exit(1)

    # set matplotlib backend to non-interactive if only saving plots
    if args.output is not None and not args.show_plots:
        plt.switch_backend('pdf')

    # load localisation config
    config = load_config(args, for_snr=False)
    logger.info('Note: warnings about e.g. missing S/N arrays or SEFD values are '
                'irrelevant for generating a beam model')
    # set burst to use
    if args.burst is None:
        # get from .yaml prefix
        args.burst = os.path.splitext(os.path.basename(args.config))[0]

    burst_config = config[args.burst]

    # set CB and SB
    if args.cb is None:
        args.cb = burst_config['main_cb']
        logger.info(f'Using value from config: CB{args.cb:02d}')
    beam_config = burst_config[f'CB{args.cb:02d}']

    if args.sb is None:
        args.sb = [burst_config['main_sb']]
        logger.info(f'Using value from config: SB{args.sb[0]:02d}')

    # Define global RA, Dec localisation area
    grid_size = config['size']  # in arcmin
    grid_res = config['resolution'] / 60  # converted from arcsec to arcmin
    dracosdec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin
    ddec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin

    dRACOSDEC, dDEC = np.meshgrid(dracosdec, ddec)

    DEC = config['dec'] * u.deg + dDEC
    RA = config['ra'] * u.deg + dRACOSDEC / np.cos(DEC)

    # convert localisation area and CB pointing (=phase centre) to HA, Dec
    ha_cb, dec_cb = tools.radec_to_hadec(*beam_config['pointing'], burst_config['tarr'])
    HA_loc, DEC_loc = tools.radec_to_hadec(RA, DEC, burst_config['tarr'])

    # calculate offsets from phase center
    # without cos(dec) factor for dHA
    dHACOSDEC_loc = (HA_loc - ha_cb) * np.cos(DEC_loc)
    dDEC_loc = (DEC_loc - dec_cb)

    # generate the SB model with CB as phase center
    logger.info('Generating SB model')
    sbp = SBPattern(ha_cb, dec_cb, dHACOSDEC_loc, dDEC_loc, fmin=burst_config['fmin'] * u.MHz,
                    fmax=burst_config['fmax'] * u.MHz, min_freq=config['fmin_data'] * u.MHz,
                    cb_model=config['cb_model'], cbnum=args.cb, sbs=args.sb)

    # plot each SB
    # get half-power CB width
    central_freq = int(np.round(config['fmin_data'] + config['bandwidth'] / 2)) * u.MHz
    cb_radius = CB_HPBW * REF_FREQ / central_freq / 2

    X = RA.to(u.deg).value
    Y = DEC.to(u.deg).value
    extent = [X[0, 0], X[-1, -1], Y[0, 0], Y[-1, -1]]
    for sb in args.sb:
        fig, ax = plt.subplots(figsize=(9, 9))

        # get SB model, scaled by integrated maximum value
        sb_model = sbp.beam_pattern_sb_full[sb] / sbp.beam_pattern_sb_int[sb].max()
        # shape is (nfreq, ndec, nha)

        # obtain colour model
        colour_model = get_colour_model(sb_model)

        # plot beam model
        ax.imshow(colour_model, origin='lower', aspect='auto',
                  extent=extent, norm=LogNorm())

        # Add CB
        patch = SphericalCircle((beam_config['ra'] * u.deg, beam_config['dec'] * u.deg),
                                cb_radius, ec='w', fc='none', ls='-', alpha=.5)
        ax.add_patch(patch)

        # limit fig to localisation region
        ax.set_xlim(X[0, 0], X[-1, -1])
        ax.set_ylim(Y[0, 0], Y[-1, -1])

        # labels etc
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')

        ax.set_title(f'{args.burst} CB{args.cb:02d} SB{sb:02d}')

        # save the fig if requested
        if args.output is not None:
            # get provided path
            path = os.path.dirname(args.output)
            if os.path.isdir(args.output):
                fname = 'SB_model'
                ext = '.pdf'
            else:
                full_fname = os.path.basename(args.output)
                fname, ext = os.path.splitext(full_fname)
                if not ext:
                    ext = '.pdf'
            output_file = os.path.join(path, f'{fname}_{args.burst}_CB{args.cb:02d}_SB{sb:02d}{ext}')
            fig.savefig(output_file, bbox_inches='tight')

    if args.show_plots:
        plt.show()
