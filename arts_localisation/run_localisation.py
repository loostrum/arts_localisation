#!/usr/bin/env python3
#
# Generate an SB model for several CBs near the same area on-sky
# then combine posteriors
import argparse
import os
import logging
import errno

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats

import convert
from constants import CB_HPBW, REF_FREQ, NSB, BANDWIDTH
from beam_models import SBPattern
from config_parser import parse_yaml

logger = logging.getLogger(__name__)

# Try switching to OSX native backend
# plt.switch_backend('pdf')
# try:
#     plt.switch_backend('macosx')
# except ImportError:
#     pass

plt.rcParams['axes.formatter.useoffset'] = False


def makedirs(path):
    """
    Mimic os.makedirs, but do not error when directory already exists
    :param str path: path to recursively create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def make_plot(chi2, X, Y, dof, title, conf_int, mode='radec', sigma_max=3,
              freq=1370 * u.MHz, t_arr=None, loc=None, cb_pos=None,
              src_pos=None):
    """
    Create plot of localisation area
    :param chi2: chi2 grid
    :param X: RA or Az
    :param Y: Dec or Alt
    :param dof: degrees of freedom
    :param title: plot title
    :param conf_int: confidence interval for localisation area
    :param mode: radec or altaz (default radec)
    :param sigma_max: used to determine maximum value for colors in plot (default 3)
    :param freq: central frequency (default 1370 MHz)
    :param t_arr: burst arrival time (only required if mode is altaz)
    :param loc: legend location (optional)
    :param cb_pos: cb pointing (optional, tuple or list of tuples)
    :param src_pos: source position (tuple)
    """
    if mode == 'altaz' and t_arr is None:
        print("t_arr is required in AltAz mode")
    dchi2 = chi2 - chi2.min()
    # best pos = point with lowest (delta)chi2
    ind = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    best_x = X[ind]
    best_y = Y[ind]

    # init figure
    fig, ax = plt.subplots()

    # calculate dchi2 value corresponding to conf_int for contour
    dchi2_value = stats.chi2.ppf(conf_int, dof)
    contour_values = [dchi2_value]

    # set vmax to sigma_max
    conf_int_max = stats.chi2.cdf(sigma_max**2, 1)
    vmax = stats.chi2.ppf(conf_int_max, dof)

    # plot delta chi 2
    img = ax.pcolormesh(X, Y, dchi2, vmax=vmax)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r'$\Delta \chi^2$', rotation=270)

    # add contours
    cont = ax.contour(X, Y, dchi2, contour_values, colors=['#FF0000', '#C00000', '#800000'])

    # add best position
    ax.plot(best_x.to(u.deg).value, best_y.to(u.deg).value, c='r', marker='.', ls='', ms=10,
            label='Best position')
    # add source position if available
    if src_pos is not None:
        if mode == 'altaz':
            hadec_src = convert.radec_to_hadec(*src_pos, t_arr)
            y_src, x_src = convert.hadec_to_altaz(hadec_src.ra, hadec_src.dec)
        else:
            x_src, y_src = src_pos
        ax.plot(x_src.to(u.deg).value, y_src.to(u.deg).value, c='cyan', marker='+', ls='', ms=10,
                label='Source position')
    # add CB position and circle if available
    if cb_pos is not None:
        if not isinstance(cb_pos, list):
            cb_pos = [cb_pos]
        for i, pos in enumerate(cb_pos):
            if mode == 'altaz':
                hadec_cb = convert.radec_to_hadec(pos.ra, pos.dec, t_arr)
                y_cb, x_cb = convert.hadec_to_altaz(hadec_cb.ra, hadec_cb.dec)
            else:
                x_cb = pos.ra
                y_cb = pos.dec
            if i == 0:
                label = 'CB center'
            else:
                label = ''
            ax.plot(x_cb.to(u.deg).value, y_cb.to(u.deg).value, c='k', marker='x', ls='', ms=10,
                    label=label)
            # add CB
            cb_radius = (CB_HPBW * REF_FREQ / freq / 2)
            patch = SphericalCircle((x_cb, y_cb), cb_radius, ec='k', fc='none', ls='-', alpha=.5)
            ax.add_patch(patch)

    # add labels
    if mode == 'altaz':
        ax.set_xlabel('Az (deg)')
        ax.set_ylabel('Alt (deg)')
    elif mode == 'radec':
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
    ax.legend(loc=loc)
    ax.set_title(title)

    return fig


def load_config(args):
    """
    Load yaml config file and overwrite settings that are alos given on command line

    :param argparse.Namespace args: Command line arguments
    :return: config (dict)
    """
    config = parse_yaml(args.config)
    # overwrite parameters also given on command line
    for key in ('ra', 'dec', 'resolution', 'size', 'fmin', 'fmax', 'bandwidth', 'cb_model'):
        value = getattr(args, key)
        if value is not None:
            logger.debug("Overwriting {} from settings with command line value".format(key))
            config['global'][key] = value

    return config


def main(args):
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # load config
    config = load_config(args)

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        args.output_folder = os.getcwd()
    makedirs(args.output_folder)
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))

    # Define global RA, Dec localisation area
    grid_size = config['global']['size']  # in arcmin
    grid_res = config['global']['resolution'] / 60  # converted from arcsec to arcmin
    dracosdec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin
    ddec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin

    dRACOSDEC, dDEC = np.meshgrid(dracosdec, ddec)

    DEC = config['global']['dec'] * u.deg + dDEC
    RA = config['global']['ra'] * u.deg + dRACOSDEC / np.cos(DEC)

    # store size
    numY, numX = RA.shape
    # save coordinate grid
    np.save('{}_coord'.format(output_prefix), np.array([RA, DEC]))

    # process each burst
    for burst in config['bursts']:
        logging.info("Processing {}".format(burst))
        burst_config = config[burst]

        # loop over CBs
        nCB = len(burst_config['beams'])
        numsb_det = 0
        chi2 = {}
        tarr = {}
        pointings = {}
        for CB in burst_config['beams']:
            print("Processing {}".format(CB))
            beam_config = burst_config[CB]

            # get alt, az of pointing
            # TODO: support HA
            # try:
            radec_cb = SkyCoord(*beam_config['pointing'])
            hadec_cb = convert.radec_to_hadec(*beam_config['pointing'], burst_config['tarr'])
            # except KeyError:
            #     # assume ha was specified instead of ra
            #     hadec_cb = SkyCoord(beam_config['ha']*u.deg, beam_config['dec']*u.deg)
            #     radec_cb = convert.hadec_to_radec(beam_config['ha']*u.deg, beam_config['dec']*u.deg, t)
            alt_cb, az_cb = convert.hadec_to_altaz(hadec_cb.ra, hadec_cb.dec)  # needed for SB position
            print("Parallactic angle: {:.2f}".format(convert.hadec_to_par(hadec_cb.ra, hadec_cb.dec)))
            print("AltAz SB rotation angle at center of CB: {:.2f}".format(convert.hadec_to_proj(hadec_cb.ra, hadec_cb.dec)))
            # save pointing
            pointings[CB] = radec_cb

            # convert localisation coordinate grid to hadec
            hadec_loc = convert.radec_to_hadec(RA, DEC, burst_config['tarr'])
            HA_loc = hadec_loc.ra
            DEC_loc = hadec_loc.dec
            # calculate offsets from phase center
            # without cos(dec) factor for dHA
            dHACOSDEC_loc = (HA_loc - hadec_cb.ra) * np.cos(DEC_loc)
            dDEC_loc = (DEC_loc - hadec_cb.dec)

            # generate the SB model with CB as phase center
            sbp = SBPattern(hadec_cb.ra, hadec_cb.dec, dHACOSDEC_loc, dDEC_loc, fmin=config['global']['fmin'] * u.MHz,
                            fmax=config['global']['fmax'] * u.MHz, min_freq=config['global']['fmin_data'] * u.MHz,
                            cb_model=config['global']['cb_model'], cbnum=int(CB[2:]))
            # get pattern integrated over frequency
            # TODO: spectral indices?
            sb_model = sbp.beam_pattern_sb_int

            # load SNR array
            try:
                data = np.loadtxt(beam_config['snr_array'], ndmin=2)
                sb_det, snr_det = data.T
                sb_det = sb_det.astype(int)
            except KeyError:
                print("No S/N array found for {}, assuming this is a non-detection beam".format(burst))
                sb_det = np.array([])
                snr_det = np.array([])
            numsb_det += len(sb_det)

            # non detection beams
            sb_non_det = np.array([sb for sb in range(NSB) if sb not in sb_det])

            # init chi2 array
            chi2[CB] = np.zeros((numY, numX))

            # find SB with highest S/N
            try:
                ind = snr_det.argmax()
                this_snr = snr_det[ind]
                this_sb = sb_det[ind]
                print("SB{:02d} SNR {}".format(this_sb, this_snr))
            except ValueError:
                # non-detection beam
                this_snr = None
                this_sb = None

            # SEFD
            try:
                sefd = beam_config['sefd']
            except KeyError:
                default_sefd = 100
                print("No SEFD found, setting to {}".format(default_sefd))
                sefd = default_sefd

            # if this is the reference burst, store the sb model of the reference SB
            if CB == burst_config['reference_cb']:
                ref_snr = this_snr
                reference_sb_model = sb_model[this_sb]
                ref_sefd = sefd

            # model of S/N relative to the reference beam
            snr_model = sb_model / reference_sb_model * ref_snr * ref_sefd / sefd

            # detection
            ndet = len(sb_det)
            if ndet > 0:
                print("Adding {} detections".format(ndet))
                chi2[CB] += np.sum((snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis]) ** 2, axis=0)
            # non detection
            nnondet = len(sb_non_det)
            if nnondet > 0:
                print("Adding {} non-detections".format(nnondet))
                # only select points where the modelled S/N is above the threshold
                snr_model_nondet = snr_model[sb_non_det]
                points = np.where(snr_model_nondet > config['global']['snrmin'])
                # temporarily create an array holding the chi2 values to add per SB
                chi2_to_add = np.zeros_like(snr_model_nondet)
                chi2_to_add[points] += (snr_model_nondet[points] - config['global']['snrmin']) ** 2
                # sum over SBs and add
                chi2[CB] += chi2_to_add.sum(axis=0)

            # # reference SB has highest S/N: modelled S/N should never be higher than reference
            bad_ind = np.any(sb_model > reference_sb_model, axis=0)
            # save chi2 before applying bad_ind_mask
            np.save('{}_{}_{}_chi2'.format(output_prefix, burst, CB), chi2[CB])
            # save region where S/N > ref_snr for non-ref SB
            np.save('{}_{}_{}_snr_too_high'.format(output_prefix, burst, CB), bad_ind)

        # chi2 of all CBs combined
        chi2_total = np.zeros((numY, numX))
        for value in chi2.values():
            chi2_total += value

        # degrees of freedom = number of data points minus number of parameters
        # total number of CBs * NSB - 1 (ref SB) - 2 (params)
        dof = nCB * NSB - 3

        # find size of localisation area within given confidence level
        max_dchi2 = stats.chi2.ppf(args.conf_int, dof)
        dchi2_total = chi2_total - chi2_total.min()
        npix_below_max = (dchi2_total < max_dchi2).sum()
        pix_area = ((config['global']['resolution'] * u.arcsec) ** 2)
        total_area = pix_area * npix_below_max
        print("Found {} pixels below within {}% confidence region".format(npix_below_max, args.conf_int * 100))
        print("Area of one pixel is {} ".format(pix_area))
        print("Localisation area is {:.2f} = {:.2f}".format(total_area, total_area.to(u.arcmin ** 2)))

        # find best position
        ind = np.unravel_index(np.argmin(chi2_total), chi2_total.shape)
        coord_best = SkyCoord(RA[ind], DEC[ind])

        print("Best position: {}".format(coord_best.to_string('hmsdms')))
        if config['global']['source_coord'] is not None:
            coord_src = SkyCoord(*config['global']['source_coord'])
            print("Source position: {}".format(coord_src.to_string('hmsdms')))
            print("Separation: {}".format(coord_src.separation(coord_best).to(u.arcsec)))

            # find closest ra,dec to source
            dist = ((RA - coord_src.ra) * np.cos(DEC)) ** 2 + (DEC - coord_src.dec) ** 2
            ind = np.unravel_index(np.argmin(dist), RA.shape)
            chi2_best = chi2_total.min()
            chi2_at_source = chi2_total[ind]
            # print info to stderr
            hdr = "ra_src, dec_src, ra_best dec_best chi2_best chi2_at_src dof numsb_det"
            summary = "{} {} {:.2f} {:.2f} {} {}".format(coord_src.to_string('hmsdms'), coord_best.to_string('hmsdms'),
                                                         chi2_best, chi2_at_source, dof, numsb_det)
            # store or print summary
            if args.outfile:
                if os.path.isfile(args.outfile):
                    mode = 'a'
                else:
                    mode = 'w'
                with open(args.outfile, mode) as f:
                    if mode == 'w':
                        f.write(hdr + '\n')
                    f.write(summary + '\n')
            else:
                print(hdr)
                print(summary)

        # plot
        central_freq = int(np.round((config['global']['fmin_data'] + config['global']['bandwidth'] / 2))) * u.MHz
        if not args.noplot:
            # per CB
            for CB in burst_config['beams']:
                if burst == burst_config['reference_cb']:
                    # one SB is not a free parameter; 2 params
                    dof = NSB - 1 - 2
                else:
                    # all SBs; 2 params
                    dof = NSB - 2
                title = r"$\Delta \chi^2$ {}".format(CB)
                fig = make_plot(chi2[CB], RA, DEC, dof, title, args.conf_int, t_arr=tarr,
                                cb_pos=pointings[CB], freq=central_freq)
                if args.saveplot:
                    fig.savefig('{}_{}_{}.pdf'.format(output_prefix, burst, CB))

            # total
            title = r"$\Delta \chi^2$ Total"
            fig = make_plot(chi2_total, RA, DEC, dof, title, args.conf_int, loc='lower right',
                            cb_pos=list(pointings.values()), freq=central_freq)

            if args.saveplot:
                fig.savefig('{}_{}_total.pdf'.format(output_prefix, burst))
            else:
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--conf_int', type=float, default=.9, help='Confidence interval for localisation region '
                                                                   '(Default: %(default)s)')
    parser.add_argument('--output_folder', default='.', help='Output folder '
                                                             '(Default: current directory)')
    parser.add_argument('--noplot', action='store_true', help='Disable plotting')
    parser.add_argument('--saveplot', action='store_true', help='Save plots')
    parser.add_argument('--outfile', help='Output file for summary')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")

    group_overwrites = parser.add_argument_group('Config overwrites', 'Settings to overwrite from yaml config')
    group_overwrites.add_argument('--ra', type=float, help='Central RA (deg)')
    group_overwrites.add_argument('--dec', type=float, help='Central Dec (deg)')
    group_overwrites.add_argument('--resolution', type=float, help='Resolution (arcmin)')
    group_overwrites.add_argument('--size', type=float, help='Localisation area size (arcmin)')
    group_overwrites.add_argument('--fmin', type=float, help='Ignore frequency below this value in MHz')
    group_overwrites.add_argument('--fmax', type=float, help='Ignore frequency below this value in MHz')
    group_overwrites.add_argument('--fmin_data', type=float, help='Lowest frequency of data')
    group_overwrites.add_argument('--bandwidth', type=float, help='Bandwidth of data')
    group_overwrites.add_argument('--cb_model', help='CB model type')

    args = parser.parse_args()
    main(args)
