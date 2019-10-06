#!/usr/bin/env python3
#
# Tool to generate the compound beam pattern


import numpy as np
import astropy.units as u
from scipy.interpolate import griddata

from constants import CB_HPBW, REF_FREQ, DISH_SIZE, CB_MODEL_FILE


class CompoundBeam(object):
    
    def __init__(self, freqs, theta=0, phi=0):
        """
        Generate a compound beam pattern
        :param freqs: observing frequencies
        :param theta: E-W offsets (default: 0)
        :param phi: N-S offsets (default: 0)
        """

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        if not isinstance(phi, np.ndarray):
            phi = np.array(phi)
        if not isinstance(freqs, np.ndarray):
            freqs = np.array(freqs)

        # ensure theta and phi use same units
        phi = phi.to(theta.unit)

        self.theta = theta
        self.phi = phi
        self.freqs = freqs

        # load beam measurements
        self.beams_measured = np.genfromtxt(CB_MODEL_FILE, delimiter=',', names=True)
        print(self.beams_measured.shape)
        

    def beam_pattern(self, mode, cb=None):
        """
        Return beam pattern for given mode
        :param mode: gauss, airy, real
        :param cb: beam number (real beam)
        :return: beam pattern
        """

        if mode == 'gauss':
            return self.__gaussian()
        elif mode == 'real':
            return self.__real(cb)
        elif mode == 'airy':
            raise NotImplementedError("Airy disk not yet implemented")
        else:
            raise ValueError("Mode should be gauss or airy")

    def __gaussian(self):
        """
        Gaussian beam pattern
        :return: beam pattern
        """

        output_grid = np.zeros((self.freqs.shape[0], self.theta.shape[0], self.phi.shape[0]))

        # convert beam width to gaussian sigma at each freq
        sigmas = CB_HPBW * REF_FREQ / self.freqs / (2. * np.sqrt(2 * np.log(2)))

        # calculate response at each sigma
        for i, sigma in enumerate(sigmas):
            arg = -.5 * (self.theta[..., None]**2 + self.phi**2) / sigma**2
            output_grid[i] = np.exp(arg) / 2

        return output_grid

    def __real(self, cb):
        """
        Measured beam pattern
        :param cb: beam number
        :return: beam pattern
        """
        beam = self.beams_measured[cb].reshape(40, 40)
        ref_freq = 1399.662250 * u.MHz

        dx = 0.027777777777778 * 60   # arcmin at ref freq

        output_grid = np.zeros((self.freqs.shape[0], self.theta.shape[0], self.phi.shape[0]))
        X_out, Y_out = np.meshgrid(self.theta, self.phi)
        # interpolate onto output grid 
        scalings = ref_freq / self.freqs
        for i, scaling in enumerate(scalings):
            theta_in = np.arange(-20, 20) * dx * scaling * u.arcmin
            phi_in = np.arange(-20, 20) * dx * scaling * u.arcmin
            X_in, Y_in = np.meshgrid(theta_in, phi_in)
            output_grid[i] = griddata((X_in.flatten(), Y_in.flatten()), beam, (X_out, Y_out))

        return output_grid
