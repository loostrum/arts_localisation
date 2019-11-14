#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


if __name__ == '__main__':

    # generate grid of full CB pattern
    # X and Y range
    step = .1
    theta = np.arange(-130, 130+step, step)  # arcmin
    phi = np.arange(-100, 100+step, step)  # arcmin
    ntheta = len(theta)
    nphi = len(phi)
    ext = [theta[0], theta[-1], phi[0], phi[-1]]

    # load models, sum CBs
    cb_model_gauss = np.load('models/all_cb_gauss.npy').max(axis=0)
    cb_model_real = np.load('models/all_cb_real.npy').max(axis=0)

    assert cb_model_gauss.shape == (nphi, ntheta)
    assert cb_model_real.shape == (nphi, ntheta)

    # plot
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(16, 4.8))
    vmin = 0
    vmax = 1

    ax = axes[0]
    ax.imshow(cb_model_gauss, extent=ext, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_xlabel('RA offset (arcmin)')
    ax.set_ylabel('Dec offset (arcmin)')
    ax.set_title('Simple Gaussian beam pattern')

    ax = axes[1]
    ax.imshow(cb_model_real, extent=ext, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_xlabel('RA offset (arcmin)')
    ax.set_title('2D Gaussian fit to measured beam pattern')

    ax = axes[2]
    ax.imshow(cb_model_gauss - cb_model_real, extent=ext, aspect='auto', interpolation='none', vmin=-vmax, vmax=vmax,
              cmap='seismic')
    ax.set_xlabel('RA offset (arcmin)')
    ax.set_title('Simple - 2D fit')
    plt.show()
