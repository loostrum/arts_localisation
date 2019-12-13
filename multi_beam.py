#!/usr/bin/env python3
#
# Generate an SB model for several CBs near the same area on-sky
# then combine posteriors

import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats

import convert
from constants import CB_HPBW, REF_FREQ
from simulate_sb_pattern import SBPattern

plt.rcParams['axes.formatter.useoffset'] = False

NSB = 71
MAXSNR = 8  # should be the same as used when calculating S/N arrays (typically 8)
FREQ = 1370*u.MHz  # reference frequency for CB radius in plot


def make_plot(chi2, X, Y, dof, title, mode='altaz', sigmas=None, sigma_max=4,
              t_arr=None, loc=None, cb_pos=None):
    # convert to delta chi squared so we can estimate confidence intervals
    if sigmas is None:
        sigmas = [3]
    if mode == 'altaz' and t_arr is None:
        print("t_arr is required in AltAz mode")
    dchi2 = chi2 - chi2.min()
    # best pos = point with lowest (delta)chi2
    ind = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    best_x = X[ind]
    best_y = Y[ind]

    # init figure
    fig, ax = plt.subplots()

    # calculate contour values in a readable way
    contour_values = []
    for sigma in sigmas:
        # from http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
        # convert sigma to confidence interval
        conf_int = stats.chi2.cdf(sigma ** 2, 1)
        # convert confidence interval to delta chi2
        dchi2_value = stats.chi2.ppf(conf_int, dof)
        contour_values.append(dchi2_value)

    # set vmax to sigma_max
    conf_int = stats.chi2.cdf(sigma_max**2, 1)
    vmax = stats.chi2.ppf(conf_int, dof)

    # plot delta chi 2
    img = ax.pcolormesh(X, Y, dchi2, vmax=vmax)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('$\Delta \chi^2$')

    # add contours
    ax.contour(X, Y, dchi2, contour_values, colors=['#FF0000', '#C00000', '#800000'])

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
            # add cross at center
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
        ax.set_xlabel('Ra (deg)')
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
    parser.add_argument('--size', type=float, default=2, help='Localisation area size (arcmin) '
                                                              '(Default: %(default)s)')
    parser.add_argument('--fmin', type=int, default=1300, help='Ignore frequency below this value in MHz '
                                                               '(Default: %(default)s)')
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
    dra = np.arange(-args.size/2, args.size/2+args.res, args.res) * u.arcmin
    ddec = np.arange(-args.size / 2, args.size / 2 + args.res, args.res) * u.arcmin

    dec = args.dec*u.deg + ddec
    ra = args.ra*u.deg + dra/np.cos(dec)
    RA, DEC = np.meshgrid(ra, dec)

    # loop over bursts
    chi2 = {}
    XX = {}
    YY = {}
    tarr = {}
    pointings = {}
    nburst = 0
    for burst in conf.keys():
        if burst == 'source':
            continue
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
        alt_cb, az_cb = convert.hadec_to_altaz(hadec_cb.ra, hadec_cb.dec)
        # save pointing
        pointings[burst] = radec_cb

        # convert localisation coordinate grid to altaz
        hadec_loc = convert.radec_to_hadec(RA, DEC, t)
        # convert hadec to altaz
        alt_loc, az_loc = convert.hadec_to_altaz(hadec_loc.ra, hadec_loc.dec)
        #### TEMP: store for plot
        XX[burst] = az_loc
        YY[burst] = alt_loc

        # convert localisation area to offset from this CB
        dtheta, dphi = convert.coord_to_offset(az_cb, alt_cb, az_loc, alt_loc)

        nphi, ntheta = dtheta.shape

        # define flips as in plot_sb (use mean alt az for now)
        if az_loc.mean() > 270 * u.deg or az_loc.mean() < 90 * u.deg:
            # north
            sign_du = 1
            sign_dv = -1
        else:
            # south
            sign_du = -1
            sign_dv = 1
        # for some reason SB with best detection matches B0531 only if pattern is mirrored???
        # so flip again ...
        sign_du *= -1
        sign_dv *= -1
        print("Sign du:", sign_du)
        print("Sign dv:", sign_dv)

        # generate the SB model
        # sbp = SBPattern(dtheta=dtheta*sign_du, dphi=dphi*sign_dv,
        #                 fmin=args.fmin*u.MHz, cb_model='real', cbnum=data['cb'])
        sbp = SBPattern(dtheta=dtheta*sign_du, dphi=dphi*sign_dv,
                        fmin=args.fmin*u.MHz, cb_model='gauss', cbnum=data['cb'])
        sb_model = sbp.beam_pattern_sb_sky

        # load SNR array
        data = np.loadtxt(data['snr_array'])
        sb_det, snr_det = data.T
        sb_det = sb_det.astype(int)

        # non detection beams
        sb_non_det = np.array([sb for sb in range(NSB) if sb not in sb_det])
        if len(sb_non_det) > 0:
            have_nondet = True
        else:
            have_nondet = False

        # init chi2 array
        chi2[burst] = np.zeros((nphi, ntheta))
        # use one reference SB
        ind = snr_det.argmax()
        ref_snr = snr_det[ind]
        ref_sb = sb_det[ind]
        print("SB{:02d} SNR {}".format(ref_sb, ref_snr))
        # model of S/N relative to this beam
        snr_model = sb_model * ref_snr / sb_model[ref_sb]
        # chi2
        # Detection SBs
        chi2[burst] += np.sum((snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis]) ** 2 / snr_model[sb_det], axis=0)
        # non detection SBs
        if have_nondet:
            snr_model_nondet = snr_model[sb_non_det]
            sb_mask = snr_model_nondet > MAXSNR
            chi2[burst] += np.sum((snr_model_nondet[sb_mask] - MAXSNR) ** 2 , axis=0)

        # reference SB has highest S/N: modelled S/N should never be higher than reference
        #bad = (snr_model[sb_det] > ref_snr).sum(axis=0)
        #chi2[burst][bad] = np.inf

    # degrees of freedom = number of data points minus number of parameters
    # data points = SBs minus one (reference SB)
    # parameters: theta, phi = 2
    dof = NSB - 3

    # chi2 of all CBs combined
    chi2_total = np.zeros((nphi, ntheta))
    for value in chi2.values():
        chi2_total += value

    # total number of CBs * (NSB-1) - 2 params
    dof_total = nburst * (NSB - 1) - 2
    # minus 2 for params
    dof_total -= 2

    # find size of localisation area within 3 sigma
    nsigma = 3
    max_dchi2 = stats.chi2.ppf(stats.chi2.cdf(nsigma ** 2, 1), dof_total)
    dchi2_total = chi2_total-chi2_total.min()
    npix_below_max = (dchi2_total < max_dchi2).sum()
    pix_area = ((args.res*u.arcmin)**2).to(u.arcsec**2)
    total_area = pix_area * npix_below_max
    print("Found {} pixels below {} sigma".format(npix_below_max, nsigma))
    print("Area of one pixel is {} ".format(pix_area))
    print("Localisation area is {:.2f} = {:.2f}".format(total_area, total_area.to(u.arcmin**2)))

    # find best position
    ind = np.unravel_index(np.argmin(chi2_total), chi2_total.shape)
    coord_best = SkyCoord(RA[ind], DEC[ind])

    print("Best position: {}".format(coord_best.to_string('hmsdms')))
    if have_source:
        print("Source position: {}".format(coord_src.to_string('hmsdms')))
        print("Separation: {}".format(coord_src.separation(coord_best).to(u.arcsec)))

    # plot
    # per burst
    for burst in conf.keys():
        if burst == 'source':
            continue

        title = "$\Delta \chi^2$ {}".format(burst)
        make_plot(chi2[burst], XX[burst], YY[burst], dof, title, t_arr=tarr[burst],
                  cb_pos=pointings[burst])

    # total
    title = "$\Delta \chi^2$ Total"
    make_plot(chi2_total, RA, DEC, dof_total, title, mode='radec', loc='lower right',
              cb_pos=list(pointings.values()))

    plt.show()
