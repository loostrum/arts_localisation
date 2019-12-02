#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from scipy import stats

import convert
from constants import THETAMAX_SB, PHIMAX_SB, NTHETA_SB, NPHI_SB


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input YAML config file')
    parser.add_argument('--chi2', required=True, help='Input SB delta chi squared')
    parser.add_argument('--dof', type=int, required=True, help='Degrees of freedom')
    # TODO:
    parser.add_argument('--save', default='', help='Output filename with coordinates and chi2')

    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # pointing center
    ra_cb = conf['ra_best_cb'] * u.deg
    dec_cb = conf['dec_best_cb'] * u.deg
    radec = SkyCoord(ra_cb, dec_cb)
    # source position
    have_src = True
    try:
        ra_src = conf['ra_src'] * u.deg
        dec_src = conf['dec_src'] * u.deg
        radec_src = SkyCoord(ra_src, dec_src)
    except KeyError:
        have_src = False

    t = Time(conf['tstart'], format='isot', scale='utc') + TimeDelta(conf['t_in_obs'], format='sec')
    # t -= TimeDelta(7200, format='sec')

    # Get apparent ha dec of the CB
    # apparent hadec
    hadec = convert.radec_to_hadec(ra_cb, dec_cb, t)
    # convert to alt az
    alt_cb, az_cb = convert.hadec_to_altaz(hadec.ra, hadec.dec)

    # same for source position
    if have_src:
        hadec_src = convert.radec_to_hadec(ra_src, dec_src, t)
        alt_src, az_src = convert.hadec_to_altaz(hadec_src.ra, hadec_src.dec)

    print("Obs time: {}".format(t))
    print("CB RA, Dec: {} {}".format(ra_cb, dec_cb))
    print("CB RA, Dec: {}".format(radec.to_string('hmsdms')))
    print()
    print("CB HA, Dec: {} {}".format(hadec.ra, hadec.dec))
    print("CB HA, Dec: {}".format(hadec.to_string('hmsdms')))
    print()
    print("CB Az, Alt: {} {}".format(az_cb, alt_cb))
    print()

    # generate coordinate grid
    theta = np.linspace(-THETAMAX_SB, THETAMAX_SB, NTHETA_SB) * u.arcmin
    phi = np.linspace(-PHIMAX_SB, PHIMAX_SB, NPHI_SB) * u.arcmin
    theta = theta.to(u.deg)
    phi = phi.to(u.deg)
    dtheta = (theta[1] - theta[0]).value
    dphi = (phi[1] - phi[0]).value

    du, dv = np.meshgrid(theta, phi)

    # define flips required to have model in AltAz order
    # SB model always has +RA = right, +Dec = up
    # RA:
    # if pointing south: lower RA = higher azimuth: model x-axis should be flipped
    # if pointing north: lower RA = lower azimuth: no flip
    # assume we never point north below NCP (WSRT cannot do this anyway)
    # Dec:
    # North above NCP, higher dec is lower alt: flip
    # South: higher dec is higher alt: no flip
    # So:
    # if north: flip y but not x
    # if south: flip x but not y
    if az_cb > 270*u.deg or az_cb < 90*u.deg:
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

    X, Y = convert.offset_to_coord(az_cb, alt_cb, du, dv)

    # load posterior
    chi2 = np.load(args.chi2)
    # convert to delta_chi2
    dchi2 = chi2 - chi2.min()
    # flip so left = lower az and down = lower dec
    dchi2 = dchi2[::sign_dv, ::sign_du]
    nphi, ntheta = dchi2.shape
    assert nphi == NPHI_SB
    assert ntheta == NTHETA_SB

    # # calculate confidence interval contours
    # post_linear = np.exp(post_raw)
    # evidence = post_linear.sum() * dtheta * dphi
    # post = post_linear / evidence
    # post_theta = post.sum(axis=0) * dtheta
    # post_phi = post.sum(axis=1) * dphi
    # probability = post * dtheta * dphi
    # assert np.allclose(probability.sum(), 1)
    # # remove zeros
    # probability[probability == 0] = np.nan

    # Get best position
    best_phi_ind, best_theta_ind = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    best_theta = theta[best_theta_ind]
    best_phi = phi[best_phi_ind]

    best_az, best_alt = convert.offset_to_coord(az_cb, alt_cb, best_theta, best_phi)
    # convert to HA Dec
    ha_best, dec_best = convert.altaz_to_hadec(best_alt, best_az)
    # and finally to RA Dec
    radec_best = convert.hadec_to_radec(ha_best, dec_best, t)

    print("Best position of source: {}".format(radec_best.to_string('hmsdms')))

    if have_src:
        # distance between best and real position
        sep = radec_src.separation(radec_best).to(u.arcmin)
        # estimates for now
        ddec = radec_src.dec - radec_best.dec
        dra = (radec_src.ra - radec_best.ra) * np.cos(radec_src.dec)
        print("Separation between real and estimated RA Dec: {}".format(sep))
        print("dRA: {}".format(dra.to(u.arcmin)))
        print("dDec: {}".format(ddec.to(u.arcmin)))
        print()

        dalt = alt_src - best_alt
        daz = (az_src - best_az) * np.cos(alt_src)
        print("dAz: {}".format(daz.to(u.arcsec)))
        print("dAlt: {}".format(dalt.to(u.arcmin)))

    # calculate contour values in a readable way
    contour_values = []
    for sigma in [1, 2, 3]:
        # from http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
        # convert sigma to confidence interval
        conf_int = stats.chi2.cdf(sigma ** 2, 1)
        # convert confidence interval to delta chi2
        dchisq_value = stats.chi2.ppf(conf_int, args.dof)
        contour_values.append(dchisq_value)

    # set vmax to 4 sigma
    conf_int = stats.chi2.cdf(4 ** 2, 1)
    vmax = stats.chi2.ppf(conf_int, args.dof)


    # plot
    fig, axes = plt.subplots(ncols=2)
    # Alt Az plot
    ax = axes[0]
    img = ax.pcolormesh(X, Y, dchi2, cmap='viridis', vmax=vmax)
    fig.colorbar(img, ax=ax)
    # add contours
    ax.contour(X, Y, dchi2, contour_values, colors=['#FF0000', '#C00000', '#800000'])
    # best position
    ax.plot(az_cb.to(u.deg).value, alt_cb.to(u.deg).value, c='k', marker='x', ls='', ms=10, label='CB center')
    ax.plot(best_az.to(u.deg).value, best_alt.to(u.deg).value, c='r', marker='.', ls='', ms=10, label='Best position')
    if have_src:
        ax.plot(az_src.to(u.deg).value, alt_src.to(u.deg).value, c='cyan', marker='+', ls='', ms=10, label='Source position')

    # real position
    # ax.axhline(dec_src.to(u.deg).value, c='r')
    # ax.axvline(ra_src.to(u.deg).value, c='r')
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Altitude (deg)')
    ax.legend()

    # RA Dec plot
    # Convert alt az to ha dec
    # first argument is alt = Y
    XX, YY = convert.altaz_to_hadec(Y, X)
    # convert ha dec to ra dec
    XY = convert.hadec_to_radec(XX, YY, t)
    XX = XY.ra
    YY = XY.dec

    if args.save:
        print('Saving RA Dec coordinate grid')
        np.save(args.save, [XX, YY])
        exit()

    ax = axes[1]
    img = ax.pcolormesh(XX, YY, dchi2, cmap='viridis', vmin=0, vmax=vmax)
    fig.colorbar(img, ax=ax)
    # add contours
    ax.contour(XX, YY, dchi2, contour_values, colors=['#FF0000', '#C00000', '#800000'])
    ax.plot(ra_cb.to(u.deg).value, dec_cb.to(u.deg).value, c='k', marker='x', ls='', ms=10, label='CB center')
    ax.plot(radec_best.ra.deg, radec_best.dec.deg, c='r', marker='.', ls='', ms=10, label='Best position')
    if have_src:
        ax.plot(ra_src.to(u.deg).value, dec_src.to(u.deg).value, c='cyan', marker='+', ls='', ms=10,
                label='Source position')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.legend()

    plt.show()
