#!/usr/bin/env python3
#
# Warning: The interpolation used in this script only works
# when https://github.com/scipy/scipy/issues/7327 is fixed in the next release
# Should be fixed in 1.4.0+

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5
from astropy.time import Time, TimeDelta

from constants import WSRT_LAT, WSRT_LON, WSRT_ALT, \
                      THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB, \
                      THETAMAX_SB, PHIMAX_SB, NTHETA_SB, NPHI_SB
from convert import ha_to_ra


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="File with input parameters")
    parser.add_argument('--posterior_cb', type=str, default='output/log_posterior_cb.npy',
                        help="Path to CB posterior (default: %(default)s)")
    parser.add_argument('--posterior_sb', type=str, default='output/log_posterior_sb.npy',
                        help="Path to SB posterior (default: %(default)s)")
    parser.add_argument('--output_file', type=str, default='output/log_posterior_total',
                        help="Output file for total posterior (default: %(default)s)")
    parser.add_argument('--plot', action='store_true',
                        help="Create and show plots")
    parser.add_argument('--ra_real', type=float, default=None,
                        help="RA of known source to add to figure")
    parser.add_argument('--dec_real', type=float, default=None,
                        help="Dec of known source to add to figure")

    args = parser.parse_args()

    # load posteriors
    print("Loading models")
    posterior_cb = np.load(args.posterior_cb, allow_pickle=True)
    posterior_sb = np.load(args.posterior_sb, allow_pickle=True)
    print("Done")

    # define coordinates
    theta_cb = np.linspace(-THETAMAX_CB, THETAMAX_CB, NTHETA_CB) * u.arcmin
    phi_cb = np.linspace(-PHIMAX_CB, PHIMAX_CB, NPHI_CB) * u.arcmin
    theta_sb = np.linspace(-THETAMAX_SB, THETAMAX_SB, NTHETA_SB) * u.arcmin
    phi_sb = np.linspace(-PHIMAX_SB, PHIMAX_SB, NPHI_SB) * u.arcmin

    # check shape
    assert len(phi_cb) == posterior_cb.shape[0]
    assert len(theta_cb) == posterior_cb.shape[1]
    assert len(phi_sb) == posterior_sb.shape[0]
    assert len(theta_sb) == posterior_sb.shape[1]

    # source parameters
    with open(args.input_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    tarr = Time(params['tstart']) + TimeDelta(params['t_in_obs'], format='sec')

    if params['drift']:
        # read hour angles, convert to J2000 RA,Dec
        CB_best_center = ha_to_ra(params['ha_best_cb']*u.deg, params['dec_best_cb']*u.deg, tarr)
        CB00_center = ha_to_ra(params['ha_cb00']*u.deg, params['dec_cb00']*u.deg, tarr)
    else:
        # read RA, Dec
        CB_best_center = SkyCoord(params['ra_best_cb'], params['dec_best_cb'], unit=u.deg, frame='icrs')
        CB00_center = SkyCoord(params['ra_cb00'], params['dec_cb00'], unit=u.deg, frame='icrs')

    # Get center of CB00 in J2000 RA,Dec
    print("Generating CB coordinates")
    # Convert CB offset coordinates to RA, Dec
    dec_cb = (phi_cb + CB00_center.dec).to(u.deg)
    # use mean Dec for cos(dec)
    ra_cb = (theta_cb / np.cos(np.mean(dec_cb)) + CB00_center.ra).to(u.deg)
    print("Done")

    #  Get center of best CB in J2000 RA,Dec
    print("Generating SB coordinates")
    # Convert Ra, Dec of CB to AltAz
    WSRT = EarthLocation(lat=WSRT_LAT, lon=WSRT_LON, height=WSRT_ALT)
    altaz_frame = AltAz(obstime=tarr, location=WSRT)
    CB_best_altaz = CB_best_center.transform_to(altaz_frame)
    # Convert SB offset coordinates to Alt,Az
    # altitude shift could be either positive or negative, but
    # the SB pattern is symmetrical along alt so it does not matter
    alt_sb = phi_sb + CB_best_altaz.alt
    # shift in Az is towards east; can be higher or lower Az; determine sign
    if (CB_best_altaz.az > 270*u.deg) or (CB_best_altaz.az < 90*u.deg):
        sgn = 1
    else:
        sgn = -1
    az_sb = CB_best_altaz.az + sgn * (theta_sb / np.cos(np.mean(alt_sb)))
    # create SkyCoord object in AltAz frame and convert to radec
    all_az, all_alt = np.meshgrid(az_sb, alt_sb)
    coord_sb = SkyCoord(all_az, all_alt, frame=altaz_frame).transform_to('icrs')
    X_sb = coord_sb.ra.to(u.deg).value
    Y_sb = coord_sb.dec.to(u.deg).value
    print("Done")

    # reduce to use only are that is both posteriors, i.e. in SB data as that is only one CB
    xmin, xmax = np.amin(X_sb)*u.deg, np.amax(X_sb)*u.deg
    ymin, ymax = np.amin(Y_sb)*u.deg, np.amax(Y_sb)*u.deg
    m_x = (ra_cb > xmin) & (ra_cb < xmax)
    m_y = (dec_cb > ymin) & (dec_cb < ymax)
    ra_cb = ra_cb[m_x]
    dec_cb = dec_cb[m_y]

    posterior_cb = posterior_cb[m_y][:, m_x]

    X_cb, Y_cb = np.meshgrid(ra_cb, dec_cb)

    # The CB model has ~2x more total pixels than the SB model
    # SB model is much higher res in E-W
    # interpolate CB posterior onto SB posterior grid to retain the E-W resolution
    print("Interpolating")

    points = (X_cb.flatten(), Y_cb.flatten())
    targets = (X_sb, Y_sb)
    posterior_cb_interp = griddata(points, posterior_cb.flatten(), 
                                   targets, method='linear', rescale=True)
    print("Done")

    print("Computing total posterior")
    posterior_total = posterior_sb + posterior_cb_interp
    # ensure max is 0
    posterior_total -= np.nanmax(posterior_total)
    X_total = X_sb
    Y_total = Y_sb

    # save total posterior
    np.save(args.output_file, [X_total, Y_total, posterior_total])

    # Find best location
    # CB
    best_dec_ind, best_ra_ind = np.unravel_index(np.nanargmax(posterior_cb, axis=None), posterior_cb.shape)
    best_ra = X_cb[best_dec_ind, best_ra_ind]
    best_dec = Y_cb[best_dec_ind, best_ra_ind]
    best_pos_cb = SkyCoord(best_ra, best_dec, unit=u.deg)
    print("Best position (CB): {}".format(best_pos_cb.to_string('hmsdms')))

    # CB interp
    best_dec_ind, best_ra_ind = np.unravel_index(np.nanargmax(posterior_cb_interp, axis=None), posterior_cb_interp.shape)
    best_ra = X_total[best_dec_ind, best_ra_ind]
    best_dec = Y_total[best_dec_ind, best_ra_ind]
    best_pos_cb_interp = SkyCoord(best_ra, best_dec, unit=u.deg)
    print("Best position (CB interpolated): {}".format(best_pos_cb_interp.to_string('hmsdms')))

    # SB
    best_dec_ind, best_ra_ind = np.unravel_index(np.nanargmax(posterior_sb, axis=None), posterior_sb.shape)
    best_ra = X_sb[best_dec_ind, best_ra_ind]
    best_dec = Y_sb[best_dec_ind, best_ra_ind]
    best_pos_sb = SkyCoord(best_ra, best_dec, unit=u.deg)
    print("Best position (SB): {}".format(best_pos_sb.to_string('hmsdms')))

    # total
    have_best_pos = True
    try:
        best_dec_ind, best_ra_ind = np.unravel_index(np.nanargmax(posterior_total, axis=None), posterior_total.shape)
        best_ra = X_total[best_dec_ind, best_ra_ind]
        best_dec = Y_total[best_dec_ind, best_ra_ind]
        best_pos_total = SkyCoord(best_ra, best_dec, unit=u.deg)
        print("Best position (total): {}".format(best_pos_total.to_string('hmsdms')))
    except ValueError as e:
        print("Exception caught while calculating best position:", e)
        have_best_pos = False

    # plot distance between measured and real pos
    if args.ra_real and args.dec_real:
        real_pos = SkyCoord(args.ra_real, args.dec_real, unit='deg')
        # SB
        dist = real_pos.separation(best_pos_sb)
        print("Distance between SB and real position: {}".format(dist.to(u.arcmin)))
        # CB
        dist = real_pos.separation(best_pos_cb)
        print("Distance between CB and real position: {}".format(dist.to(u.arcmin)))
        # Total
        if have_best_pos:
            dist = real_pos.separation(best_pos_total)
            print("Distance between total and real position: {}".format(dist.to(u.arcmin)))
    

    if args.plot:
        # plot posteriors in RA,Dec
        print("Plotting")
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        axes = axes.flatten()
        vmin_cb = -4.6
        vmin_sb = -4.6
        vmin_tot = -4.6

        # CB posterior
        ax = axes[0]
        img = ax.pcolormesh(X_cb, Y_cb, posterior_cb, vmin=vmin_cb)
        ax.scatter(best_pos_cb.ra, best_pos_cb.dec, c='r', s=10)
        if args.ra_real:
            ax.axvline(args.ra_real, c='r')
        if args.dec_real:
            ax.axhline(args.dec_real, c='r')
        fig.colorbar(img, ax=ax)
        ax.set_ylabel('Dec [deg]')
        ax.set_title('CB')

        # SB posterior
        ax = axes[1]
        img = ax.pcolormesh(X_sb, Y_sb, posterior_sb, vmin=vmin_sb)
        ax.scatter(best_pos_sb.ra, best_pos_sb.dec, c='r', s=10)
        if args.ra_real:
            ax.axvline(args.ra_real, c='r')
        if args.dec_real:
            ax.axhline(args.dec_real, c='r')
        fig.colorbar(img, ax=ax)
        ax.set_title('SB')

        # CB posterior interpolated
        ax = axes[2]
        img = ax.pcolormesh(X_sb, Y_sb, posterior_cb_interp, vmin=vmin_cb)
        ax.scatter(best_pos_cb_interp.ra, best_pos_cb_interp.dec, c='r', s=10)
        if args.ra_real:
            ax.axvline(args.ra_real, c='r')
        if args.dec_real:
            ax.axhline(args.dec_real, c='r')
        fig.colorbar(img, ax=ax)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_title('CB interpolated')

        # Sum of posteriors
        ax = axes[3]
        img = ax.pcolormesh(X_total, Y_total, posterior_total, vmin=vmin_tot)
        if have_best_pos:
            ax.scatter(best_pos_total.ra, best_pos_total.dec, c='r', s=10)
        if args.ra_real:
            ax.axvline(args.ra_real, c='r')
        if args.dec_real:
            ax.axhline(args.dec_real, c='r')
        fig.colorbar(img, ax=ax)
        ax.set_xlabel('RA [deg]')
        ax.set_title('CB + SB')

        print("Done")
        plt.show()

