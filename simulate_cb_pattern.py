#!/usr/bin/env python3

import tqdm
import numpy as np
import astropy.units as u

from compound_beam import CompoundBeam


if __name__ == '__main__':
    output_file = 'models/all_cb.npy'

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
    step = .1
    theta = np.arange(-130, 130+step, step) * u.arcmin
    phi = np.arange(-100, 100+step, step) * u.arcmin

    cb_sens = np.zeros((ncb, len(theta), len(phi)))

    # Add the beams to the grid
    for cb in tqdm.tqdm(range(ncb)):
        dra, ddec = cb_pos[cb]
        # calculate CB integrated over frequencies
        freqs = np.linspace(1220, 1520, 2) * u.MHz
        beam = CompoundBeam(freqs, theta-dra, phi-ddec)
        cb_sens[cb] = beam.beam_pattern('gauss').sum(axis=0)

    # store grid
    np.save(output_file, cb_sens)
