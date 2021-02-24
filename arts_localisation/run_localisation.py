#!/usr/bin/env python3

import os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats

from arts_localisation import tools
from arts_localisation.constants import CB_HPBW, REF_FREQ, NSB
from arts_localisation.beam_models.simulate_sb_pattern import SBPattern
from arts_localisation.config_parser import load_config

logger = logging.getLogger(__name__)
plt.rcParams['axes.formatter.useoffset'] = False


def nested_dict_values(d):
    """
    Get all values from a nested dictionary

    :param dict d: dictionary
    :return: generator for values
    """

    for value in d.values():
        if isinstance(value, dict):
            yield from nested_dict_values(value)
        else:
            yield value


def make_plot(conf_ints, X, Y, title, conf_int, mode='radec', sigma_max=3,
              freq=1370 * u.MHz, t_arr=None, loc=None, cb_pos=None,
              src_pos=None):
    """
    Create plot of localisation area

    :param conf_ints: confidence interval grid
    :param X: RA or Az
    :param Y: Dec or Alt
    :param title: plot title
    :param conf_int: confidence interval to plot
    :param mode: radec or altaz (default radec)
    :param sigma_max: used to determine maximum value for colors in plot (default 3)
    :param freq: central frequency (default 1370 MHz)
    :param t_arr: burst arrival time (only required if mode is altaz)
    :param loc: legend location (optional)
    :param cb_pos: cb pointing (optional, tuple or list of tuples)
    :param src_pos: source position (tuple)
    :return: figure
    """
    if mode == 'altaz' and t_arr is None:
        logger.info("t_arr is required in AltAz mode")
    X = X.to(u.deg).value
    Y = Y.to(u.deg).value

    # best pos = point with lowest confidence interval
    ind = np.unravel_index(np.argmin(conf_ints), conf_ints.shape)
    best_x = X[ind]
    best_y = Y[ind]

    # init figure
    fig, ax = plt.subplots()

    # plot confidence interval
    img = ax.pcolormesh(X, Y, conf_ints, vmin=0, vmax=1, shading='nearest')
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Confidence interval', rotation=270, labelpad=15)

    # add contours
    ax.contour(X, Y, conf_ints, [conf_int], colors=['#FF0000', '#C00000', '#800000'])

    # add best position
    ax.plot(best_x, best_y, c='r', marker='.', ls='', ms=10,
            label='Best position')
    # add source position if available
    if src_pos is not None:
        if mode == 'altaz':
            ha_src, dec_src = tools.radec_to_hadec(*src_pos, t_arr)
            y_src, x_src = tools.hadec_to_altaz(ha_src, dec_src)
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
                ha_cb, dec_cb = tools.radec_to_hadec(pos.ra, pos.dec, t_arr)
                y_cb, x_cb = tools.hadec_to_altaz(ha_cb, dec_cb)
            else:
                x_cb = pos.ra
                y_cb = pos.dec
            if i == 0:
                label = 'CB center'
            else:
                label = ''
            ax.plot(x_cb.to(u.deg).value, y_cb.to(u.deg).value, c='w', marker='x', ls='', ms=10,
                    label=label)
            # add CB
            cb_radius = (CB_HPBW * REF_FREQ / freq / 2)
            patch = SphericalCircle((x_cb, y_cb), cb_radius, ec='k', fc='none', ls='-', alpha=.5)
            ax.add_patch(patch)

    # limit to localisation region
    ax.set_xlim(X[0, 0], X[-1, -1])
    ax.set_ylim(Y[0, 0], Y[-1, -1])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')
    parser.add_argument('--conf_int', type=float, default=.9, help='Confidence interval for localisation region plot '
                                                                   '(Default: %(default)s)')
    parser.add_argument('--output_folder', help='Output folder '
                                                '(Default: <yaml file folder>/localisation)')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('--store_intermediates', action='store_true', help='Store all intermediate data to '
                                                                           '<yaml file folder>/intermediates. '
                                                                           'Note: this may create very big output files')

    group_overwrites = parser.add_argument_group('Config overwrites', 'Settings to overwrite from yaml config')
    # global config:
    group_overwrites.add_argument('--snrmin', type=float, help='S/N threshold')
    group_overwrites.add_argument('--fmin_data', type=float, help='Lowest frequency of data')
    group_overwrites.add_argument('--bandwidth', type=float, help='Bandwidth of data')
    # localisation config:
    group_overwrites.add_argument('--ra', type=float, help='Central RA (deg)')
    group_overwrites.add_argument('--dec', type=float, help='Central Dec (deg)')
    group_overwrites.add_argument('--size', type=float, help='Localisation area size (arcmin)')
    group_overwrites.add_argument('--resolution', type=float, help='Resolution (arcmin)')
    group_overwrites.add_argument('--cb_model', help='CB model type')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # set matplotlib backend to non-interactive if only saving plots
    if args.save_plots and not args.show_plots:
        plt.switch_backend('pdf')

    # load config
    config = load_config(args)

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        # default output folder is same folder as .yaml file plus "localisation"
        args.output_folder = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'localisation')
    tools.makedirs(args.output_folder)
    # output prefix also contains the yaml filename without extension
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))

    # set intermediates output folder and prefix
    intermediates_folder = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'intermediates')
    if args.store_intermediates:
        tools.makedirs(intermediates_folder)
    intermediates_prefix = os.path.join(intermediates_folder, os.path.basename(args.config).replace('.yaml', ''))

    # Define global RA, Dec localisation area
    grid_size = config['size']  # in arcmin
    grid_res = config['resolution'] / 60  # converted from arcsec to arcmin
    dracosdec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin
    ddec = np.arange(-grid_size / 2, grid_size / 2 + grid_res, grid_res) * u.arcmin

    dRACOSDEC, dDEC = np.meshgrid(dracosdec, ddec)

    DEC = config['dec'] * u.deg + dDEC
    RA = config['ra'] * u.deg + dRACOSDEC / np.cos(DEC)

    # store size
    numY, numX = RA.shape
    # save coordinate grid
    np.save(f'{output_prefix}_coord', np.array([RA, DEC]))

    # process each burst
    chi2_all_bursts = np.zeros((numY, numX))
    dof_all_bursts = np.zeros((numY, numX))
    pointings_all = {}

    # central freq, used for plotting
    central_freq = int(np.round(config['fmin_data'] + config['bandwidth'] / 2)) * u.MHz

    for burst in config['bursts']:
        logger.info(f"Processing {burst}")
        burst_config = config[burst]

        # loop over CBs
        numsb_det = 0
        chi2 = {}
        tarr = {}
        pointings = {}
        dofs = {}
        for CB in burst_config['beams']:
            logger.info(f"Processing {CB}")
            beam_config = burst_config[CB]

            # get pointing
            radec_cb = SkyCoord(*beam_config['pointing'])
            ha_cb, dec_cb = tools.radec_to_hadec(*beam_config['pointing'], burst_config['tarr'])
            logger.info("Parallactic angle: {:.2f}".format(tools.hadec_to_par(ha_cb, dec_cb)))
            logger.info("Projection angle {:.2f}".format(tools.hadec_to_proj(ha_cb, dec_cb)))
            # save pointing
            pointings[CB] = radec_cb

            # convert localisation coordinate grid to hadec
            HA_loc, DEC_loc = tools.radec_to_hadec(RA, DEC, burst_config['tarr'])
            # calculate offsets from phase center
            # without cos(dec) factor for dHA
            dHACOSDEC_loc = (HA_loc - ha_cb) * np.cos(DEC_loc)
            dDEC_loc = (DEC_loc - dec_cb)

            # generate the SB model with CB as phase center
            sbp = SBPattern(ha_cb, dec_cb, dHACOSDEC_loc, dDEC_loc, fmin=burst_config['fmin'] * u.MHz,
                            fmax=burst_config['fmax'] * u.MHz, min_freq=config['fmin_data'] * u.MHz,
                            cb_model=config['cb_model'], cbnum=int(CB[2:]))
            # get pattern integrated over frequency
            # TODO: spectral indices?
            sb_model = sbp.beam_pattern_sb_int

            # load SNR array
            try:
                data = np.loadtxt(beam_config['snr_array'], ndmin=2)
                sb_det, snr_det = data.T
                sb_det = sb_det.astype(int)
                # only keep values above S/N threshold
                ind = snr_det >= config['snrmin']
                sb_det = sb_det[ind]
                snr_det = snr_det[ind]
            except KeyError:
                logger.info(f"No S/N array found for {burst}, assuming this is a non-detection beam")
                sb_det = np.array([])
                snr_det = np.array([])
            else:
                if len(sb_det) == 0:
                    logger.info(f"No SBs above S/N threshold found for {burst}")
                    sb_det = np.array([])
                    snr_det = np.array([])
            numsb_det += len(sb_det)

            # non detection beams
            sb_non_det = np.array([sb for sb in range(NSB) if sb not in sb_det])

            # init chi2 array and DoF array
            chi2[CB] = np.zeros((numY, numX))
            dofs[CB] = np.zeros((numY, numX))

            # find SB with highest S/N
            try:
                ind = snr_det.argmax()
                this_snr = snr_det[ind]
                this_sb = sb_det[ind]
                logger.info(f"SB{this_sb:02d} SNR {this_snr}")
            except ValueError:
                # non-detection beam
                this_snr = None
                this_sb = None

            # SEFD
            try:
                sefd = beam_config['sefd']
            except KeyError:
                default_sefd = 85
                logger.info(f"No SEFD found, setting to {default_sefd}")
                sefd = default_sefd

            # if this is the reference burst, store the sb model of the reference SB
            if CB == burst_config['reference_cb']:
                ref_snr = this_snr
                reference_sb_model = sb_model[this_sb]
                ref_sefd = sefd

            # model of S/N relative to the reference beam
            snr_model = sb_model / reference_sb_model * ref_snr * ref_sefd / sefd

            # store intermediates
            if args.store_intermediates:
                # SB model
                np.save(f'{intermediates_prefix}_{burst}_{CB}_SB_pattern', sb_model)
                # S/N model
                np.save(f'{intermediates_prefix}_{burst}_{CB}_SNR_model', snr_model)

            # detection
            ndet = len(sb_det)
            if ndet > 0:
                logger.info(f"Adding {ndet} detections")
                # chi2 per SB
                chi2_det = (snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis]) ** 2 / snr_model[sb_det]
                # add sum over SBs to total chi2
                chi2[CB] += chi2_det.sum(axis=0)
                # store intermediates
                if args.store_intermediates:
                    # chi2 per SB
                    np.save(f'{intermediates_prefix}_{burst}_{CB}_chi2_det', chi2_det)

                # add to DoF, this is the same for each grid point
                dofs[CB] += len(sb_det)
            # non detection
            nnondet = len(sb_non_det)
            if nnondet > 0:
                logger.info(f"Adding {nnondet} non-detections")
                # only select points where the modelled S/N is above the threshold
                snr_model_nondet = snr_model[sb_non_det]
                points = snr_model_nondet > config['snrmin']
                # temporarily create an array holding the chi2 values to add per SB
                chi2_to_add = np.zeros_like(snr_model_nondet)
                chi2_to_add[points] += (snr_model_nondet[points] - config['snrmin']) ** 2 / snr_model_nondet[points]
                # store intermediates
                if args.store_intermediates:
                    # chi2 per SB
                    np.save(f'{intermediates_prefix}_{burst}_{CB}_chi2_nondet', chi2_to_add)
                # sum over SBs and add
                chi2[CB] += chi2_to_add.sum(axis=0)
                # add number of non-detection beams with S/N > snrmin to DoF
                dofs[CB] += points.sum(axis=0)

            # # reference SB has highest S/N: modelled S/N should never be higher than reference
            bad_ind = np.any(sb_model > reference_sb_model, axis=0)
            # save chi2 before applying bad_ind_mask
            np.save(f'{output_prefix}_{burst}_{CB}_chi2', chi2[CB])
            # save region where S/N > ref_snr for non-ref SB
            np.save(f'{output_prefix}_{burst}_{CB}_snr_too_high', bad_ind)

        # store pointings
        pointings_all[burst] = pointings

        # chi2 of all CBs combined
        chi2_total = np.zeros((numY, numX))
        for value in chi2.values():
            chi2_total += value

        # chi2 of all bursts combined
        chi2_all_bursts += chi2_total

        # degrees of freedom = number of data points minus number of parameters
        # dofs array currently holds number of data points in each CB
        # total number of datapoints at each grid point
        dof_total = np.zeros((numY, numX))
        for value in dofs.values():
            dof_total += value

        # add to dof of all bursts combined and subtract one for reference SB
        dof_all_bursts += dof_total - 1
        # subtract number of parameters (2) and one for reference SB
        dof_total -= 3

        # convert the DoF array of each CB to actual DoF instead of nr of data points
        for cb, dof_cb in dofs.items():
            if cb == burst_config['reference_cb']:
                # this CB contains reference SB, so subtract one
                dofs[cb] -= 1
            # subtract two for the parameters
            dofs[cb] -= 2

        # convert chi2 to confidence intervals
        conf_ints = {}
        for cb in burst_config['beams']:
            this_chi2 = chi2[cb]
            this_dchi2 = this_chi2 - this_chi2.min()
            dof = dofs[cb]
            conf_ints[cb] = stats.chi2.cdf(this_dchi2, dof)
            # where dof <= 0, conf_int is nan. These locations are "perfect", so set conf_int to 0
            conf_ints[cb][np.isnan(conf_ints[cb])] = 0
            # save the map
            np.save(f'{output_prefix}_{burst}_{CB}_conf_int', conf_ints[cb])
        # repeat for total
        dchi2_total = chi2_total - chi2_total.min()
        conf_int_total = stats.chi2.cdf(dchi2_total, dof_total)
        conf_int_total[np.isnan(conf_int_total)] = 0
        np.save(f'{output_prefix}_{burst}_total_conf_int', conf_int_total)

        # find size of localisation area within given confidence level
        npix_below_max = (conf_int_total < args.conf_int).sum()
        pix_area = ((config['resolution'] * u.arcsec) ** 2)
        total_area = pix_area * npix_below_max
        logger.info("Found {} pixels below within {}% confidence region".format(npix_below_max, args.conf_int * 100))
        logger.info(f"Area of one pixel is {pix_area} ")
        logger.info("Localisation area is {:.2f} = {:.2f}".format(total_area, total_area.to(u.arcmin ** 2)))

        # find best position, which is at the point of the lowest confidence interval
        ind = np.unravel_index(np.argmin(conf_int_total), conf_int_total.shape)
        coord_best = SkyCoord(RA[ind], DEC[ind])
        logger.info("Best position: {}".format(coord_best.to_string('hmsdms')))

        if config['source_coord'] is not None:
            coord_src = SkyCoord(*config['source_coord'])
            logger.info("Source position: {}".format(coord_src.to_string('hmsdms')))
            logger.info("Separation: {}".format(coord_src.separation(coord_best).to(u.arcsec)))

            # find closest ra,dec to source
            dist = ((RA - coord_src.ra) * np.cos(DEC)) ** 2 + (DEC - coord_src.dec) ** 2
            ind = np.unravel_index(np.argmin(dist), RA.shape)
            conf_int_at_source = conf_int_total[ind]
            logger.info(f"Confidence interval at source (lower is better): {conf_int_at_source:.5f}")

        # plot
        logging.info("Generating plots")
        if args.show_plots or args.save_plots:
            # per CB, if there is more than one
            for CB in burst_config['beams']:
                title = f"{CB}"
                fig = make_plot(conf_ints[CB], RA, DEC, title, args.conf_int, t_arr=tarr,
                                cb_pos=pointings[CB], freq=central_freq,
                                src_pos=config['source_coord'])
                if args.save_plots:
                    fig.savefig(f'{output_prefix}_{burst}_{CB}.pdf')

            # total, if there is more than one CB
            if len(burst_config['beams']) > 1:
                fig = make_plot(conf_int_total, RA, DEC, burst, args.conf_int, loc='lower right',
                                cb_pos=list(pointings.values()), freq=central_freq,
                                src_pos=config['source_coord'])
                if args.save_plots:
                    fig.savefig(f'{output_prefix}_{burst}_total.pdf')

    # result of all bursts combined
    # first correct DoF for number of parameters
    dof_all_bursts -= 2

    # calculate combined confidence interval
    dchi2_all_bursts = chi2_all_bursts - chi2_all_bursts.min()
    conf_int_all_bursts = stats.chi2.cdf(dchi2_all_bursts, dof_all_bursts)
    # where dof <= 0, conf_int is nan. These locations are "perfect", so set conf_int to 0
    conf_int_all_bursts[np.isnan(conf_int_all_bursts)] = 0

    # save final localisation region
    np.save(f'{output_prefix}_localisation', [RA, DEC, conf_int_all_bursts])

    if args.show_plots or args.save_plots:
        # plot of all bursts combined, if there are multiple bursts
        if len(config['bursts']) > 1:
            fig = make_plot(conf_int_all_bursts, RA, DEC, 'Combined', args.conf_int, loc='lower right',
                            cb_pos=list(nested_dict_values(pointings_all)), freq=central_freq,
                            src_pos=config['source_coord'])
            if args.save_plots:
                fig.savefig(f'{output_prefix}_combined_bursts.pdf')
    if args.show_plots:
        plt.show()


if __name__ == '__main__':
    main()
