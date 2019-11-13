#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u

from constants import CB_HPBW


def add_cb_pattern(ax):
    # Add CB positions
    cb_offsets = np.loadtxt('square_39p1.cb_offsets', usecols=[1, 2], delimiter=',')
    cb_pos = np.zeros((ncb, 2)) 
    for cb, (dra, ddec) in enumerate(cb_offsets):
        dra *= 60
        ddec *= 60
        cb_pos[cb] = np.array([dra, ddec])

    cb_pos *= u.arcmin

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'alpha': .5, 
            'size': 10} 
    for cb, (dra, ddec) in enumerate(cb_pos):
        patch = Circle((dra.to(u.arcmin).value, ddec.to(u.arcmin).value), CB_HPBW.to(u.arcmin).value/2,
                       ec='k', fc='none', ls='-')
        ax.add_patch(patch)
        ax.text(dra.to(u.arcmin).value, ddec.to(u.arcmin).value, 'CB{:02d}'.format(cb), va='center', ha='center',
                fontdict=font)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="Input file with S/N per CB")
    parser.add_argument('--output_file', type=str, default='output/log_posterior_cb',
                        help="Output file for posterior (default: %(default)s)")
    parser.add_argument('--model', type=str, default='models/all_cb.npy',
                        help="Path to CB model (default: %(default)s)")
    parser.add_argument('--max_snr', type=float, default=8.,
                        help="Max S/N in non-detection beams (default: %(default)s)")
    # Option to exclude CBs disabled until implemented in rest of the code
    # parser.add_argument('--exclude_cb', nargs='*',
    #                     help="Space-separated list of CBs to ignore")
    parser.add_argument('--plot', action='store_true',
                        help="Create and show plots")

    args = parser.parse_args()

    # detections
    data = np.genfromtxt(args.input_file, names=True)

    cb_det = data['cb'].astype(int)
    snr_det = data['snr'].astype(float)

    best_ind = np.argmax(snr_det)
    try:
        len(cb_det)
        best_cb = cb_det[best_ind]
        best_snr = snr_det[best_ind]
        ndet = len(cb_det)
    except TypeError:
        # just one CB
        best_cb = cb_det
        best_snr = snr_det
        ndet = 1

    print("Loading model")
    cb_model = np.load(args.model)
    # scale CB sensitivity to highest S/N detection
    snr_model = cb_model * best_snr / cb_model[best_cb]
    del cb_model
    ncb, nphi, ntheta = snr_model.shape
    print("Done")

    # theta = RA
    # phi = Dec
    theta = np.linspace(-130, 130, ntheta)
    phi = np.linspace(-100, 100, nphi)

    # non detection beams
    cb_non_det = np.array([cb for cb in range(ncb) if cb not in cb_det])
    if len(cb_non_det) > 0:
        have_nondet= True
    else:
        have_nondet = False

    # -log(L) = sum of (snr_det - snr_model)\dagger C (snr_det - snr_model)
    # C = covariance matrix, assume identity
    # split into detection + non detection

    # Detections: model - measured, then square and sum over CBs
    print("Adding {} detections".format(ndet))
    # take care that the np sum is -log(L)
    log_l = -np.sum((snr_model[cb_det] - snr_det[..., np.newaxis, np.newaxis])**2, axis=0)
    print("Done")

    # Non detections: add as prior
    if have_nondet:
        print("Adding {} non-detections".format(len(cb_non_det)))

        # where S/N is higher than max_snr: Set to zero probability
        num_bad_cb = np.sum(snr_model[cb_non_det] - args.max_snr > 0, axis=0)
        mask = num_bad_cb > 0

        # prior is flat where S/N < max_snr, 0 where S/N > max_snr
        log_prior = np.ones((nphi, ntheta)) * np.log(1./args.max_snr)
        log_prior[mask] = -np.inf
    else:
        log_prior = 0

    # define posterior
    log_posterior = log_l + log_prior
    print("Done")

    # ensure max is 0
    log_posterior -= np.amax(log_posterior)

    # Save the posterior
    X, Y = np.meshgrid(theta, phi)
    np.save(args.output_file, [X, Y, log_posterior])

    # Get the best position
    best_phi_ind, best_theta_ind = np.unravel_index(np.nanargmax(log_posterior, axis=None), log_posterior.shape)
    best_theta = theta[best_theta_ind]
    best_phi = phi[best_phi_ind]
    print("Best position: theta={:.6f}', phi={:.6f}'".format(best_theta, best_phi))

    # Plots
    if args.plot:
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True, sharey=True)
        axes = axes.flatten()

        # Plot number of CBs > max_snr at each position
        if have_nondet:
            ax = axes[0]
            img = ax.pcolormesh(X, Y, num_bad_cb)
            ax.scatter(best_theta, best_phi, s=10, c='cyan')
            fig.colorbar(img, ax=ax)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel(r'$\phi$ [arcmin]')
            ax.set_title('Number of non-detection CBs with S/N > {:.1f}'.format(args.max_snr))
            add_cb_pattern(ax)

        # Plot posterior
        ax = axes[1]
        img = ax.pcolormesh(X, Y, log_posterior, vmin=-10)
        ax.scatter(best_theta, best_phi, s=10, c='cyan')
        fig.colorbar(img, ax=ax)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\theta$ [arcmin]')
        ax.set_ylabel(r'$\phi$ [arcmin]')
        ax.set_title('Log posterior')
        add_cb_pattern(ax)

        plt.show()
