#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz
from scipy.optimize import bisect

import convert
from constants import THETAMAX_SB, PHIMAX_SB, NTHETA_SB, NPHI_SB


def offset(threshold, prob, level, dx, dy):
    return prob[prob > threshold].sum()*dx*dy - level


def get_confident(n, xvals):
    i = 0
    while n[i] < 0.16:
        i += 1
    lower = xvals[i]
    i = 0
    while n[i] < 0.84:
        i += 1
    upper = xvals[i]
    return np.array([lower, upper])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Input YAML config file')
    parser.add_argument('--posterior', required=True, help='Input SB posterior')

    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # pointing center
    ra_cb = conf['ra_best_cb'] * u.deg
    dec_cb = conf['dec_best_cb'] * u.deg
    radec = SkyCoord(ra_cb, dec_cb)
    # source position
    ra_src = conf['ra_src'] * u.deg
    dec_src = conf['dec_src'] * u.deg
    radec_src = SkyCoord(ra_src, dec_src)

    t = Time(conf['tstart'], format='isot', scale='utc') + TimeDelta(conf['t_in_obs'], format='sec')

    # Get apparent ha dec
    # apparent hadec
    hadec = convert.ra_to_ha(ra_cb, dec_cb, t)
    # convert to alt az
    alt, az = convert.hadec_to_altaz(hadec.ra, hadec.dec)

    # same for source position
    hadec_src = convert.ra_to_ha(ra_src, dec_src, t)
    alt_src, az_src = convert.hadec_to_altaz(hadec_src.ra, hadec_src.dec)

    print("Obs time: {}".format(t))
    print("CB RA, Dec: {} {}".format(ra_cb, dec_cb))
    print("CB RA, Dec: {}".format(radec.to_string('hmsdms')))
    print()
    print("CB HA, Dec: {} {}".format(hadec.ra, hadec.dec))
    print("CB HA, Dec: {}".format(hadec.to_string('hmsdms')))
    print()
    print("CB Az, Alt: {} {}".format(az, alt))
    print()

    # generate coordinate grid
    theta = np.linspace(-THETAMAX_SB, THETAMAX_SB, NTHETA_SB) * u.arcmin
    phi = np.linspace(-PHIMAX_SB, PHIMAX_SB, NPHI_SB) * u.arcmin
    theta = theta.to(u.deg)
    phi = phi.to(u.deg)
    dtheta = (theta[1] - theta[0]).value
    dphi = (phi[1] - phi[0]).value

    du, dv = np.meshgrid(theta, phi)

    # left = lower RA in SB model
    # if pointing south: lower RA = lower azimuth: du is in correct order
    # if pointing north, above NCP: lower RA = higher azimuth
    # assume we never point north below NCP (WSRT cannot do this anyway)
    if az > 270*u.deg or az < 90*u.deg:
        du = -du

    # convert to Az, Alt
    X, Y = convert.offset_to_radec(az, alt, du, dv)

    # load posterior
    post_raw = np.load(args.posterior)
    nphi, ntheta = post_raw.shape
    assert nphi == NPHI_SB
    assert ntheta == NTHETA_SB
    # convert to log10
    # post_log10 = np.log10(np.exp(post_raw))

    # calculate confidence interval contours
    post_linear = np.exp(post_raw)
    evidence = post_linear.sum() * dtheta * dphi
    post = post_linear / evidence
    post_theta = post.sum(axis=0) * dtheta
    post_phi = post.sum(axis=1) * dphi
    probability = post * dtheta * dphi
    assert np.allclose(probability.sum(), 1)
    # remove zeros
    probability[probability == 0] = np.nan

    # get best position
    best_du = theta[post_theta.argmax()]
    best_dv = phi[post_phi.argmax()]
    best_az, best_alt = convert.offset_to_radec(az, alt, best_du, best_dv)
    # convert to HA Dec
    ha_best, dec_best = convert.altaz_to_hadec(best_alt, best_az)
    # and finally to RA Dec
    radec_best = convert.ha_to_ra(ha_best, dec_best, t)

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
    print("dAZ: {}".format(daz.to(u.arcsec)))
    print("dAlt: {}".format(dalt.to(u.arcmin)))

    thresh68 = bisect(offset, 0., post.max(), args=(post, 0.68, dtheta, dphi))
    thresh95 = bisect(offset, 0., post.max(), args=(post, 0.95, dtheta, dphi))
    thresh99 = bisect(offset, 0., post.max(), args=(post, 0.99, dtheta, dphi))

    # plot
    fig, axes = plt.subplots(ncols=2)
    # Alt Az plot
    ax = axes[0]
    img = ax.pcolormesh(X, Y, probability, cmap='viridis', vmin=0, vmax=np.nanmax(probability))
    fig.colorbar(img, ax=ax)
    # contours
    # ax.contour(X, Y, post, [thresh99, thresh95, thresh68], linewidths=1, colors='g')
    # best position
    ax.plot(az.to(u.deg).value, alt.to(u.deg).value, c='k', marker='x', ls='', ms=10, label='CB center')
    ax.plot(az_src.to(u.deg).value, alt_src.to(u.deg).value, c='cyan', marker='+', ls='', ms=10, label='Source position')
    ax.plot(best_az.to(u.deg).value, best_alt.to(u.deg).value, c='r', marker='.', ls='', ms=10, label='Best position')

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
    XY = convert.ha_to_ra(XX, YY, t)
    XX = XY.ra
    YY = XY.dec

    ax = axes[1]
    img = ax.pcolormesh(XX, YY, probability, cmap='viridis', vmin=0, vmax=np.nanmax(probability))
    fig.colorbar(img, ax=ax)
    # ax.contour(XX, YY, post, [thresh99, thresh95, thresh68], linewidths=1, colors='g')
    ax.plot(ra_cb.to(u.deg).value, dec_cb.to(u.deg).value, c='k', marker='x', ls='', ms=10, label='CB center')
    ax.plot(ra_src.to(u.deg).value, dec_src.to(u.deg).value, c='cyan', marker='+', ls='', ms=10, label='Source position')
    ax.plot(radec_best.ra.deg, radec_best.dec.deg, c='r', marker='.', ls='', ms=10, label='Best position')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.legend()

    plt.show()
