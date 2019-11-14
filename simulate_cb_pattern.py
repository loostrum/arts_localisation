#!/usr/bin/env python3

import tqdm
import numpy as np
import astropy.units as u

from compound_beam import CompoundBeam
from constants import THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB


if __name__ == '__main__':
    mode = 'real'
    output_file = 'models/all_cb_{}.npy'.format(mode)

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
        freqs = np.linspace(1220, 1520, 32) * u.MHz
        beam = CompoundBeam(freqs, theta-dra, phi-ddec)
        # create and normalise cb
        pattern = beam.beam_pattern(mode, cb=cb).mean(axis=0)
        pattern /= pattern.max()
        cb_sens[cb] = pattern

    # store grid
    np.save(output_file, cb_sens)
