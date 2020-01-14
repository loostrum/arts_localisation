#!/usr/bin/env python3
#
# Generate an SB model for several CBs near the same area on-sky
# then combine posteriors

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats

import convert
from constants import CB_HPBW, REF_FREQ, NSB
from simulate_sb_pattern import SBPattern

# Try switching to OSX native backend
try:
    plt.switch_backend('macosx')
except ModuleNotFoundError:
    pass

plt.rcParams['axes.formatter.useoffset'] = False

FREQ = 1370*u.MHz  # reference frequency for CB radius in plot


def make_plot(chi2, X, Y, dof, title, conf_int, mode='radec', sigma_max=3,
              t_arr=None, loc=None, cb_pos=None):
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
    :param t_arr: burst arrival time (only required if mode is altaz)
    :param loc: legend location (optional)
    :param cb_pos: cb pointing (optional, tuple or list of tuples)
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
    cbar.set_label('$\Delta \chi^2$', rotation=270)

    # add contours
    cont = ax.contour(X, Y, dchi2, contour_values, colors=['#FF0000', '#C00000', '#800000'])

    # add best position
    ax.plot(best_x.to(u.deg).value, best_y.to(u.deg).value, c='r', marker='.', ls='', ms=10,
            label='Best position')
    # add source position if available
    if have_source:
        if mode == 'altaz':
            hadec_src = convert.radec_to_hadec(ra_src, dec_src, t_arr)
            y_src, x_src = convert.hadec_to_altaz(hadec_src.ra, hadec_src.dec)
        else:
            x_src = ra_src
            y_src = dec_src
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
            cb_radius = (CB_HPBW * REF_FREQ/FREQ/2)
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

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input yaml config')

    parser.add_argument('--ra', required=True, type=float, help='Central RA (deg)')
    parser.add_argument('--dec', required=True, type=float, help='Central Dec (deg)')

    parser.add_argument('--res', type=float, default=.1, help='Resolution (arcmin) '
                                                              '(Default: %(default)s)')
    parser.add_argument('--size_ra', type=float, default=2, help='Localisation area size in RA (arcmin) '
                                                                 '(Default: %(default)s)')
    parser.add_argument('--size_dec', type=float, default=2, help='Localisation area size in Dec (arcmin) '
                                                                  '(Default: %(default)s)')
    parser.add_argument('--fmin', type=float, default=1300, help='Ignore frequency below this value in MHz '
                                                                 '(Default: %(default)s)')
    parser.add_argument('--fmax', type=float, default=np.inf, help='Ignore frequency below this value in MHz '
                                                                   '(Default: %(default)s)')
    parser.add_argument('--min_freq', type=float, default=1220, help='Lowest frequency of data '
                                                                     '(Default: %(default)s)')
    parser.add_argument('--conf_int', type=float, default=.9, help='Confidence interval for localisation region '
                                                                   '(Default: %(default)s)')
    parser.add_argument('--noplot', action='store_true', help='Disable plotting')
    parser.add_argument('--outfile', help='Output file for summary')
    args = parser.parse_args()

    # load yamls files
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # source position
    have_source = True
    try:
        ra_src = conf['source']['ra'] * u.deg
        dec_src = conf['source']['dec'] * u.deg
        coord_src = SkyCoord(ra_src, dec_src, unit='deg')
    except KeyError:
        have_source = False

    # Define global RA, Dec localisation area
    dra = np.arange(-args.size_ra/2, args.size_ra/2+args.res, args.res) * u.arcmin
    ddec = np.arange(-args.size_dec / 2, args.size_dec / 2 + args.res, args.res) * u.arcmin

    dRA, dDEC = np.meshgrid(dra, ddec)

    DEC = args.dec*u.deg + dDEC
    RA = args.ra*u.deg + dRA / np.cos(DEC)

    # store size
    numY, numX = RA.shape
    # save coordinate grid
    name = args.config.replace('.yaml', '')
    np.save(name+'_coord', np.array([RA, DEC]))

    # ensure reference burst is processed first
    bursts = list(conf.keys())
    # get reference burst
    try:
        ref_burst = conf['ref_burst']
    except ValueError:
        print("Provide ref_burst in yaml file")
        exit()

    # remove non-burst keys and ref_burst from list
    for key in ['source', 'ref_burst', ref_burst]:
        try:
            bursts.remove(key)
        except ValueError:
            # key not in list
            continue
    # add ref_burst at start of list
    bursts.insert(0, ref_burst)

    # loop over bursts
    chi2 = {}
    tarr = {}
    pointings = {}
    nburst = 0
    for burst in bursts:
        print("Processing {}".format(burst))
        nburst += 1

        data = conf[burst]

        # arrival time
        t = Time(data['tstart'], format='isot', scale='utc') + TimeDelta(data['tarr'], format='sec')
        tarr[burst] = t
        # get alt, az of pointing
        try:
            radec_cb = SkyCoord(data['ra']*u.deg, data['dec']*u.deg)
            hadec_cb = convert.radec_to_hadec(data['ra']*u.deg, data['dec']*u.deg, t)
        except KeyError:
            # assume ha was specified instead of ra
            hadec_cb = SkyCoord(data['ha']*u.deg, data['dec']*u.deg)
            radec_cb = convert.hadec_to_radec(data['ha']*u.deg, data['dec']*u.deg, t)
        alt_cb, az_cb = convert.hadec_to_altaz(hadec_cb.ra, hadec_cb.dec)  # needed for SB position
        print("Parallactic angle: {:.2f}".format(convert.hadec_to_par(hadec_cb.ra, hadec_cb.dec)))
        print("AltAz SB rotation angle at center of CB: {:.2f}".format(convert.hadec_to_proj(hadec_cb.ra, hadec_cb.dec)))
        # save pointing
        pointings[burst] = radec_cb

        # convert localisation coordinate grid to hadec
        hadec_loc = convert.radec_to_hadec(RA, DEC, t)
        HA_loc = hadec_loc.ra
        DEC_loc = hadec_loc.dec
        # calculate offsets from phase center
        # without cos(dec) factor for dHA
        dHA_loc = (HA_loc - hadec_cb.ra) * np.cos(DEC_loc)
        dDEC_loc = (DEC_loc - hadec_cb.dec)

        # generate the SB model with CB as phase center
        model_type = 'gauss'
        sbp = SBPattern(hadec_cb.ra, hadec_cb.dec, dHA_loc, dDEC_loc, fmin=args.fmin*u.MHz,
                        fmax=args.fmax*u.MHz, min_freq=args.min_freq*u.MHz, cb_model=model_type, cbnum=data['cb'])
        # get pattern integrated over frequency
        # TODO: spectral indices?
        sb_model = sbp.beam_pattern_sb_int

        # load SNR array
        try:
            data = np.loadtxt(data['snr_array'])
            sb_det, snr_det = data.T
            sb_det = sb_det.astype(int)
        except KeyError:
            print("No S/N array found for {}, assuming this is a non-detection beam".format(burst))
            sb_det = np.array([])
            snr_det = np.array([])

        # non detection beams
        sb_non_det = np.array([sb for sb in range(NSB) if sb not in sb_det])
        if len(sb_non_det) > 0:
            have_nondet = True
        else:
            have_nondet = False

        # init chi2 array
        chi2[burst] = np.zeros((numY, numX))

        # find SB with highest S/N
        try:
            ind = snr_det.argmax()
            this_snr = snr_det[ind]
            this_sb = sb_det[ind]
            print("SB{:02d} SNR {}".format(this_sb, this_snr))
        except ValueError:
            this_snr = None
            this_sb = None

        # if this is the reference burst, store the sb model of the reference SB
        if burst == bursts[0]:
            ref_snr = this_snr
            reference_sb_model = sb_model[this_sb]

        # model of S/N relative to the reference beam
        snr_model = sb_model * ref_snr / reference_sb_model 

        # detection
        if len(sb_det) > 0:
            print("Adding detections")
            chi2[burst] += np.sum((snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis])**2,axis=0)
        # non detection, observed S/N set to 0
        if len(sb_non_det) > 0:
            print("Adding non-detections")
            chi2[burst] += np.sum((snr_model[sb_non_det] - 0)**2, axis=0)

        # # reference SB has highest S/N: modelled S/N should never be higher than reference
        bad_ind = np.any(sb_model > reference_sb_model, axis=0)
        # save chi2 before applying bad_ind_mask
        np.save('{}_chi2_{}'.format(name, burst), chi2[burst])
        print("Applying SB mask")
        chi2[burst][bad_ind] = 1E9
        # save region where S/N > ref_snr for non-ref SB
        np.save('{}_bad_{}'.format(name, burst), bad_ind)

    # chi2 of all CBs combined
    chi2_total = np.zeros((numY, numX))
    for value in chi2.values():
        chi2_total += value

    # degrees of freedom = number of data points minus number of parameters
    # total number of CBs * NSB - 1 (ref SB) - 2 (params)
    dof = nburst * NSB - 1 - 2

    # find size of localisation area within given confidence level
    max_dchi2 = stats.chi2.ppf(args.conf_int, dof)
    dchi2_total = chi2_total-chi2_total.min()
    npix_below_max = (dchi2_total < max_dchi2).sum()
    pix_area = ((args.res*u.arcmin)**2).to(u.arcsec**2)
    total_area = pix_area * npix_below_max
    print("Found {} pixels below within {}% confidence region".format(npix_below_max, args.conf_int*100))
    print("Area of one pixel is {} ".format(pix_area))
    print("Localisation area is {:.2f} = {:.2f}".format(total_area, total_area.to(u.arcmin**2)))

    # find best position
    ind = np.unravel_index(np.argmin(chi2_total), chi2_total.shape)
    coord_best = SkyCoord(RA[ind], DEC[ind])

    print("Best position: {}".format(coord_best.to_string('hmsdms')))
    if have_source:
        print("Source position: {}".format(coord_src.to_string('hmsdms')))
        print("Separation: {}".format(coord_src.separation(coord_best).to(u.arcsec)))

        # find closest ra,dec to source
        dist = ((RA-coord_src.ra)*np.cos(DEC))**2 + (DEC-coord_src.dec)**2
        ind = np.unravel_index(np.argmin(dist), RA.shape)
        dchi2_at_source = (chi2_total - chi2_total.min())[ind]
        # print info to stderr
        hdr = "ra_best dec_best ra_src dec_src dchi2_at_src dof"
        summary = "{} {} {:.2f} {}".format(coord_best.to_string('hmsdms'), coord_src.to_string('hmsdms'),
                                           dchi2_at_source, dof)
        # store or print summary
        if args.outfile:
            if os.path.isfile(args.outfile):
                mode = 'a'
            else:
                mode = 'w'
            with open(args.outfile, mode) as f:
                if mode == 'w':
                    f.write(hdr+'\n')
                f.write(summary+'\n')
        else:
            print(hdr)
            print(summary)

    # plot
    if not args.noplot:
        # per burst
        for burst in bursts:
            if burst == ref_burst:
                # one SB is not a free parameter; 2 params
                dof = NSB - 1 - 2
            else:
                # all SBs; 2 params
                dof = NSB - 2
            title = "$\Delta \chi^2$ {}".format(burst)
            make_plot(chi2[burst], RA, DEC, dof, title, args.conf_int, t_arr=tarr[burst],
                      cb_pos=pointings[burst])

        # total
        title = "$\Delta \chi^2$ Total"
        make_plot(chi2_total, RA, DEC, dof, title, args.conf_int, loc='lower right',
                  cb_pos=list(pointings.values()))

        plt.show()
