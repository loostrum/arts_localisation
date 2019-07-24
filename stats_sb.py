#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u

from constants import CB_HPBW


def add_cb(ax):
    patch = Circle((0, 0), CB_HPBW.to(u.arcmin).value/2,
                ec='k', fc='none', ls='-')
    ax.add_patch(patch)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="Input file with S/N per SB")
    parser.add_argument('--output_file', type=str, default='output/log_posterior_sb',
                        help="Output file for posterior (default: %(default)s)")
    parser.add_argument('--model', type=str, default='models/synthesized_beam_pattern_single_cb_ha=0.000000.npy',
                        help="Path to SB model (default: %(default)s)")
    parser.add_argument('--max_snr', type=float, default=8.,
                        help="Max S/N in non-detection beams (default: %(default)s)")
    parser.add_argument('--plot', action='store_true',
                        help="Create and show plots")

    args = parser.parse_args()

    # detections
    data = np.genfromtxt(args.input_file, names=True)
    sb_det = data['sb'].astype(int)
    snr_det = data['snr'].astype(float)

    best_ind = np.argmax(snr_det)
    best_sb = sb_det[best_ind]
    best_snr = snr_det[best_ind]

    print("Loading model")
    sb_model = np.load(args.model)
    # scale SB sensitivity to highest S/N detection
    snr_model = sb_model * best_snr / sb_model[best_sb]
    del sb_model
    print("Done")

    nsb, ntheta, nphi = snr_model.shape

    theta = np.linspace(-50, 50, ntheta)
    phi = np.linspace(-50, 50, nphi)

    # non detection beams
    sb_non_det = np.array([sb for sb in range(nsb) if not sb in sb_det])
    if len(sb_non_det) > 0:
        have_nondet = True
    else:
        have_nondet = False

    # init log likelihood 
    log_l = np.zeros((ntheta, nphi))

    # Detections: model - measured, then square and sum over SBs
    print("Adding detections")
    # take care that the np sum is -log(L)
    log_l -= np.sum((snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis])**2, axis=0)
    print("Done")

    # Non detections
    if have_nondet:
        print("Adding non-detections")
        # Add non detections as prior
        # where positive: S/N is higher than max_snr. Set to zero probability
        num_bad_sb = np.sum(snr_model[sb_non_det] - args.max_snr > 0, axis=0)
        mask = num_bad_sb > 0
        
        # prior is flat where S/N < max_snr, 0 where S/N > max_snr
        log_prior = np.ones((ntheta, nphi)) * np.log(1./args.max_snr)
        log_prior[mask] = -np.inf
        print("Done")
    else:
        log_prior = 0

    # define posterior
    log_posterior = log_l + log_prior

    # ensure max is 0
    log_posterior -= np.amax(log_posterior)

    # Save the posterior
    np.save(args.output_file, log_posterior)

    # Get the best position
    best_theta_ind, best_phi_ind = np.unravel_index(np.nanargmax(log_posterior, axis=None), log_posterior.shape)
    best_theta = theta[best_theta_ind]
    best_phi = phi[best_phi_ind]
    print(np.diff(theta)[0])
    print("Best position: theta={:.6f}', phi={:.6f}'".format(best_theta, best_phi))

    # plot
    if args.plot:
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True, sharey=True)
        axes = axes.flatten()
        X, Y = np.meshgrid(theta, phi)

        # Plot number of SBs > max_snr at each position
        if have_nondet:
            ax = axes[0]
            img = ax.pcolormesh(X, Y, num_bad_sb.T)
            fig.colorbar(img, ax=ax)
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel(r'$\phi$ [arcmin]')
            ax.set_title('Number of non-detection SBs with S/N > {:.1f}'.format(args.max_snr))
            add_cb(ax)

        # Plot posterior
        ax = axes[1]
        img = ax.pcolormesh(X, Y, log_posterior.T)
        fig.colorbar(img, ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel(r'$\theta$ [arcmin]')
        ax.set_ylabel(r'$\phi$ [arcmin]')
        ax.set_title('Log posterior')
        add_cb(ax)

        plt.show()
