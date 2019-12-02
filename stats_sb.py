#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
from scipy import stats

from constants import REF_FREQ, CB_HPBW, THETAMAX_SB, PHIMAX_SB, NTHETA_SB, NPHI_SB


def add_cb(ax, freq=1370.*u.MHz):
    patch = Circle((0, 0), REF_FREQ/freq * CB_HPBW.to(u.arcmin).value/2,
                ec='k', fc='none', ls='-')
    ax.add_patch(patch)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="Input file with S/N per SB")
    parser.add_argument('--output_file', type=str, default='output/chisq_sb',
                        help="Output file for posterior (default: %(default)s)")
    parser.add_argument('--model', type=str, default='models/synthesized_beam_pattern_single_cb_ha=0.000000.npy',
                        help="Path to SB model (default: %(default)s)")
    parser.add_argument('--max_snr', type=float, default=10.,
                        help="Max S/N in non-detection beams (default: %(default)s)")
    parser.add_argument('--plot', action='store_true',
                        help="Create and show plots")

    args = parser.parse_args()

    # detections
    data = np.genfromtxt(args.input_file, names=True)
    sb_det = data['sb'].astype(int)
    snr_det = data['snr'].astype(float)

    try:
        len(snr_det)
    except TypeError:
        sb_det = np.array([sb_det])
        snr_det = np.array([snr_det])
    best_ind = np.argmax(snr_det)
    best_sb = sb_det[best_ind]
    best_snr = snr_det[best_ind]

    print("Loading model")
    sb_model = np.load(args.model)
    # scale SB sensitivity to highest S/N detection
    print("Done")

    theta = np.linspace(-THETAMAX_SB, THETAMAX_SB, NTHETA_SB)
    phi = np.linspace(-PHIMAX_SB, PHIMAX_SB, NPHI_SB)

    # nsb, nphi, ntheta = snr_model.shape
    nsb, nphi, ntheta = sb_model.shape
    assert nphi == NPHI_SB
    assert ntheta == NTHETA_SB

    # non detection beams
    sb_non_det = np.array([sb for sb in range(nsb) if sb not in sb_det])
    if len(sb_non_det) > 0:
        have_nondet = True
    else:
        have_nondet = False

    # init log likelihood 
    chisq = np.zeros((nphi, ntheta))

    for ind, ref_sb in enumerate(sb_det):
        ref_snr = snr_det[ind]
        # only one REF SB for now
        if not np.allclose(ref_snr, snr_det.max()):
            continue

        print("SB{:02d} SNR {}".format(ref_sb, ref_snr))
        # model of S/N relative to this beam
        snr_model = sb_model * ref_snr / sb_model[ref_sb]
        chisq += np.sum((snr_model[sb_det] - snr_det[..., np.newaxis, np.newaxis]) ** 2 / snr_model[sb_det], axis=0)

        # # non detections
        if have_nondet:
            print("Adding non detections")
            # non detections are only relevant where the expected S/N is above the threshold
            # find which non detection beams have a snr above the threshold
            # get model of only non detection beams
            snr_model_nondet = snr_model[sb_non_det]
            # find which beams have a S/N above the threshold
            sb_mask = snr_model_nondet > args.max_snr
            # store how many are bad
            num_bad_sb = sb_mask.sum(axis=0)
            # use max_snr as "measured" value, so badness is how much the model
            # S/N is above the threshold
            chisq += np.sum((snr_model_nondet[sb_mask] - args.max_snr) ** 2 / args.max_snr, axis=0)

    # convert chisq to delta chisq
    delta_chisq = chisq - chisq.min()

    # get degrees of freedom = npoint - nparam
    # one SB is used as reference, so npoint = number of SBs with detection minus one
    # params = theta, phi
    dof = len(sb_det) - 3
    # add non-detection SBs if available
    if have_nondet:
        dof += len(sb_non_det)

    # save result
    np.save(args.output_file+'_dof{}'.format(dof), chisq)

    # Get the best position
    best_phi_ind, best_theta_ind = np.unravel_index(np.argmin(delta_chisq), delta_chisq.shape)
    best_theta = theta[best_theta_ind]
    best_phi = phi[best_phi_ind]
    print("Best position: theta={:.6f}', phi={:.6f}'".format(best_theta, best_phi))

    # plot
    if args.plot:
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True, sharey=True)
        axes = axes.flatten()

        print("Creating plot")
        # fig, ax = plt.subplots(figsize=(8, 8))
        X, Y = np.meshgrid(theta, phi)

        # Plot number of SBs > max_snr at each position
        if have_nondet:
            ax = axes[0]
            img = ax.pcolormesh(X, Y, num_bad_sb)
            fig.colorbar(img, ax=ax)
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel(r'$\phi$ [arcmin]')
            ax.set_title('Number of non-detection SBs with S/N > {:.1f}'.format(args.max_snr))
            add_cb(ax)

        # Plot delta chisq
        ax = axes[1]
        # calculate contour values in a readable way
        contour_values = []
        for sigma in [1, 2, 3]:
            # from http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
            # convert sigma to confidence interval
            conf_int = stats.chi2.cdf(sigma ** 2, 1)
            # convert confidence interval to delta chi2
            dchisq_value = stats.chi2.ppf(conf_int, dof)
            contour_values.append(dchisq_value)

        # set vmax to 4 sigma
        conf_int = stats.chi2.cdf(4**2, 1)
        vmax = stats.chi2.ppf(conf_int, dof)

        # ax = axes[1]
        img = ax.pcolormesh(X, Y, delta_chisq, vmax=vmax)
        fig.colorbar(img, ax=ax)
        # add best position
        ax.plot(best_theta, best_phi, c='r', marker='.', ls='', ms=10,
                label='Best position')
        # add contours
        ax.contour(X, Y, delta_chisq, contour_values, colors=['#FF0000', '#C00000', '#800000'])

        ax.legend()
        ax.set_xlabel(r'$\theta$ [arcmin]')
        ax.set_ylabel(r'$\phi$ [arcmin]')
        ax.set_title('$\Delta \chi^2$')
        add_cb(ax)

        plt.show()
