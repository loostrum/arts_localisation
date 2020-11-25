#!/usr/bin/env python3

import argparse

import tqdm
import numpy as np
import astropy.units as u

from arts_localisation.beam_models.compound_beam import CompoundBeam


if __name__ == '__main__':
    modes = ['real', 'gauss']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=modes, help="CB model type")
    parser.add_argument('--fmin', type=int, default=1220,
                        help="Minimum frequency in MHz (default: %(default)s)")
    parser.add_argument('--fmax', type=int, default=1520,
                        help="Maximum frequency in MHz (default: %(default)s)")
    parser.add_argument('--nfreq', type=int, default=32,
                        help="Number of frequency channels (default: %(default)s)")

    args = parser.parse_args()

    output_file = f'models/all_cb_{args.mode}_{args.fmin}-{args.fmax}.npy'

    # load CB offsets (decimal degrees in RA, Dec)
    cb_offsets = np.loadtxt('square_39p1.cb_offsets', usecols=[1, 2], delimiter=',')
    ncb = len(cb_offsets)

    cb_pos = np.zeros((ncb, 2))
    for cb, (dra, ddec) in enumerate(cb_offsets):
        dra *= 60
        ddec *= 60
        cb_pos[cb] = np.array([dra, ddec])
    cb_pos *= u.arcmin

    # generate grid of full CB pattern
    thetamax = 130
    phimax = 100
    ntheta = 2601
    nphi = 2001
    theta = np.linspace(-thetamax, thetamax, ntheta) * u.arcmin
    phi = np.linspace(-phimax, phimax, nphi) * u.arcmin

    cb_sens = np.zeros((ncb, len(phi), len(theta)))

    # Add the beams to the grid
    for cb in tqdm.tqdm(range(ncb)):
        dra, ddec = cb_pos[cb]
        # calculate CB integrated over frequencies
        freqs = np.linspace(args.fmin, args.fmax, args.nfreq) * u.MHz
        beam = CompoundBeam(freqs, theta - dra, phi - ddec)
        # create and normalise cb
        pattern = beam.beam_pattern(args.mode, cb=cb).mean(axis=0)
        pattern /= pattern.max()
        cb_sens[cb] = pattern

    # store grid
    np.save(output_file, cb_sens)
