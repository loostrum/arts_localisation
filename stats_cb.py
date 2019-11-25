#!/usr/bin/env python3

import argparse

import numpy as np
import yaml
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import Circle
import astropy.units as u

import convert
from constants import CB_HPBW, THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB


def add_cb_pattern(ax, ra_00=None, dec_00=None):
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
        if ra_00 is not None and dec_00 is not None:
            ra, dec = convert.offset_to_radec(ra_00, dec_00, dra, ddec)
            patch = SphericalCircle((ra, dec), CB_HPBW / 2,
                                    ec='k', fc='none', ls='-')
        else:
            dec = ddec
            ra = dra
            patch = Circle((ra.value, dec.value), CB_HPBW.to(ra.unit).value / 2,
                           ec='k', fc='none', ls='-')

        ax.add_patch(patch)
        ax.text(ra.value, dec.value, 'CB{:02d}'.format(cb), va='center', ha='center',
                fontdict=font)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="Input file with S/N per CB")
    parser.add_argument('--output_file', type=str, default='output/log_posterior_cb',
                        help="Output file for posterior (default: %(default)s)")
    parser.add_argument('--model', type=str, default='models/all_cb.npy',
                        help="Path to CB model (default: %(default)s)")
    parser.add_argument('--max_snr', type=str, default='8',
                        help="Max S/N in non-detection beams (default: %(default)s)")
    # Option to exclude CBs disabled until implemented in rest of the code
    # parser.add_argument('--exclude_cb', nargs='*',
    #                     help="Space-separated list of CBs to ignore")
    parser.add_argument('--plot', action='store_true',
                        help="Create and show plots")
    parser.add_argument('--config', help="YAML config file with pointing RA, Dec")

    args = parser.parse_args()

    # get max snr
    if args.max_snr == 'inf':
        args.max_snr = 1E100
    else:
        try:
            args.max_snr = float(args.max_snr)
        except ValueError:
            print("max_snr should be inf or float")
            exit(1)

    # create plots in RA, Dec if available
    convert_coords = False
    have_real_pos = False
    if args.config is not None:
        with open(args.config, 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)

        # pointing center
        ra_00 = conf['ra_cb00'] * u.deg
        dec_00 = conf['dec_cb00'] * u.deg
        convert_coords = True

        # source position
        try:
            ra_src = conf['ra_src'] * u.deg
            dec_src = conf['dec_src'] * u.deg
            have_real_pos = True
        except KeyError:
            pass

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
    # verified that it does not matter to which CB the model is scaled
    # scaling the CB model to be an S/N model, or scaling the S/N to the max S/N
    # yields the same best position, but a _different_ posterior
    # snr_model = cb_model * best_snr / cb_model[best_cb]
    # del cb_model
    # ncb, nphi, ntheta = snr_model.shape
    ncb, nphi, ntheta = cb_model.shape
    print("Done")

    # theta = RA
    # phi = Dec
    theta = np.linspace(-THETAMAX_CB, THETAMAX_CB, NTHETA_CB)
    phi = np.linspace(-PHIMAX_CB, PHIMAX_CB, NPHI_CB)
    assert nphi == NPHI_CB
    assert ntheta == NTHETA_CB

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
    # init log l
    log_l = np.zeros((nphi, ntheta))
    # loop over each detection CB as reference
    for ind, ref_cb in enumerate(cb_det):
        ref_snr = snr_det[ind]
        print("CB{:02d} SNR {}".format(ref_cb, ref_snr))
        # model of S/N relative to this beam
        snr_model = cb_model * ref_snr / cb_model[ref_cb]
        log_l -= np.sum((snr_model[cb_det] - snr_det[..., np.newaxis, np.newaxis]) ** 2 / snr_model[cb_det], axis=0)

    print("Done")

    # Non detections: add as prior
    # if have_nondet:
    #     print("Adding {} non-detections".format(len(cb_non_det)))
    #
    #     # where S/N is higher than max_snr: Set to zero probability
    #     num_bad_cb = np.sum(snr_model[cb_non_det] - args.max_snr > 0, axis=0)
    #     mask = num_bad_cb > 0
    #
    #     # prior is flat where S/N < max_snr, 0 where S/N > max_snr
    #     log_prior = np.ones((nphi, ntheta)) * np.log(1./args.max_snr)
    #     log_prior[mask] = -np.inf
    # else:
    #     log_prior = 0
    have_nondet = False

    # define posterior
    log_posterior = log_l #+ log_prior
    print("Done")

    # ensure max is 0
    log_posterior -= np.amax(log_posterior)

    # Save the posterior
    np.save(args.output_file, log_posterior)

    # Get the best position
    best_phi_ind, best_theta_ind = np.unravel_index(np.nanargmax(log_posterior, axis=None), log_posterior.shape)
    best_theta = theta[best_theta_ind]
    best_phi = phi[best_phi_ind]
    print("Best position: theta={:.6f}', phi={:.6f}'".format(best_theta, best_phi))

    # Plots
    if convert_coords:
        # calculate proper ra dec
        # offsets are in arcmin
        # dec = dec_00.to(u.deg).value + phi/60.
        # ra = ra_00.to(u.deg).value + theta/60. / np.cos(dec_00)
        tt, pp = np.meshgrid(theta, phi)
        ra, dec = convert.offset_to_radec(ra_00, dec_00, tt*u.arcmin, pp*u.arcmin)
        # print best position in radec
        best_x = ra[best_phi_ind][best_theta_ind]
        best_y = dec[best_phi_ind][best_theta_ind]
        print("Best position: RA={:.6f}, Dec={:.6f}".format(best_x, best_y))
        if have_real_pos:
            print("Real position: RA={:.6f} deg, Dec={:.6f} deg".format(ra_src.to(u.deg).value, dec_src.to(u.deg).value))
        # X, Y = np.meshgrid(ra, dec)
        X, Y = ra, dec

    else:
        # use offset coordinates
        best_x = best_theta
        best_y = best_phi
        X, Y = np.meshgrid(theta, phi)

    if args.plot:
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True, sharey=True)
        axes = axes.flatten()

        # Plot number of CBs > max_snr at each position
        if have_nondet and args.max_snr < 1E100:
            ax = axes[0]
            img = ax.pcolormesh(X, Y, num_bad_cb)
            ax.scatter(best_x, best_y, s=10, c='cyan')
            fig.colorbar(img, ax=ax)
            ax.set_aspect('equal')
            if have_real_pos:
                ax.axhline(dec_src.to(u.deg).value, c='r')
                ax.axvline(ra_src.to(u.deg).value, c='r')
            if convert_coords:
                ax.set_xlabel('RA (deg)')
                ax.set_ylabel('Dec (deg)')
            else:
                ax.set_xlabel(r'$\theta$ [arcmin]')
                ax.set_ylabel(r'$\phi$ [arcmin]')
            ax.set_title('Number of non-detection CBs with S/N > {:.1f}'.format(args.max_snr))

            if convert_coords:
                add_cb_pattern(ax, ra_00, dec_00)
            else:
                add_cb_pattern(ax)

        # Plot posterior
        ax = axes[1]
        img = ax.pcolormesh(X, Y, log_posterior, vmin=-10)
        ax.scatter(best_x, best_y, s=10, c='cyan')
        fig.colorbar(img, ax=ax)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        if have_real_pos:
            ax.axhline(dec_src.to(u.deg).value, c='r')
            ax.axvline(ra_src.to(u.deg).value, c='r')
        if convert_coords:
            ax.set_xlabel('RA (deg)')
            ax.set_ylabel('Dec (deg)')
        else:
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel(r'$\phi$ [arcmin]')
        ax.set_title('Log posterior')

        if convert_coords:
            add_cb_pattern(ax, ra_00, dec_00)
        else:
            add_cb_pattern(ax)

        plt.show()
