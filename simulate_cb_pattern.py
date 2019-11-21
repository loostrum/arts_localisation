#!/usr/bin/env python3

import argparse

import tqdm
import numpy as np
import astropy.units as u

from compound_beam import CompoundBeam
from constants import THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB


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
    
    output_file = 'models/all_cb_{}_{}-{}.npy'.format(args.mode, args.fmin, args.fmax)

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
    theta = np.linspace(-THETAMAX_CB, THETAMAX_CB, NTHETA_CB) * u.arcmin
    phi = np.linspace(-PHIMAX_CB, PHIMAX_CB, NPHI_CB) * u.arcmin

    cb_sens = np.zeros((ncb, len(phi), len(theta)))

    # Add the beams to the grid
    for cb in tqdm.tqdm(range(ncb)):
        dra, ddec = cb_pos[cb]
        # calculate CB integrated over frequencies
        freqs = np.linspace(args.fmin, args.fmax, args.nfreq) * u.MHz
        beam = CompoundBeam(freqs, theta-dra, phi-ddec)
        # create and normalise cb
        pattern = beam.beam_pattern(args.mode, cb=cb).mean(axis=0)
        pattern /= pattern.max()
        cb_sens[cb] = pattern

    # store grid
    np.save(output_file, cb_sens)
