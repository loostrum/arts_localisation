#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u
from scipy.optimize import bisect

import convert
from constants import THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB


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
    parser.add_argument('--posterior', required=True, help='Input CB posterior')

    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    # pointing center
    ra_00 = conf['ra_cb00'] * u.deg
    dec_00 = conf['dec_cb00'] * u.deg
    # source position
    ra_src = conf['ra_src'] * u.deg
    dec_src = conf['dec_src'] * u.deg

    # generate coordinate grid
    theta = np.linspace(-THETAMAX_CB, THETAMAX_CB, NTHETA_CB) * u.arcmin
    phi = np.linspace(-PHIMAX_CB, PHIMAX_CB, NPHI_CB) * u.arcmin
    theta = theta.to(u.deg)
    phi = phi.to(u.deg)
    dtheta = (theta[1] - theta[0]).value
    dphi = (phi[1] - phi[0]).value
    du, dv = np.meshgrid(theta, phi)

    # convert to RA, Dec
    X, Y = convert.coord_to_radec(ra_00, dec_00, du, dv)

    # load posterior
    post_raw = np.load(args.posterior)
    nphi, ntheta = post_raw.shape
    assert nphi == NPHI_CB
    assert ntheta == NTHETA_CB
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

    best_du = (theta * post_theta).sum() * dtheta
    best_dv = (phi * post_phi).sum() * dphi
    best_theta, best_phi = convert.coord_to_radec(ra_00, dec_00, best_du, best_dv)

    thresh68 = bisect(offset, 0., post.max(), args=(post, 0.68, dtheta, dphi))
    thresh95 = bisect(offset, 0., post.max(), args=(post, 0.95, dtheta, dphi))
    thresh99 = bisect(offset, 0., post.max(), args=(post, 0.99, dtheta, dphi))

    # plot
    fig, ax = plt.subplots()
    # posterior
    #img = ax.pcolormesh(X, Y, post_log10, cmap='seismic', vmax=0, vmin=-3)
    img = ax.pcolormesh(X, Y, probability, cmap='viridis', vmin=0, vmax=np.nanmax(probability))
    fig.colorbar(img, ax=ax)
    # contours
    ax.contour(X, Y, post, [thresh99, thresh95, thresh68], linewidths=1, colors='g')
    # best position
    ax.plot(best_theta.to(u.deg).value, best_phi.to(u.deg).value, c='cyan', marker='+', ms=10)

    # real position
    ax.axhline(dec_src.to(u.deg).value, c='r')
    ax.axvline(ra_src.to(u.deg).value, c='r')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    plt.show()
