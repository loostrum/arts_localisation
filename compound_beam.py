#!/usr/bin/env python3
#
# Tool to generate the compound beam pattern


import numpy as np
import astropy.units as u

from constants import CB_HPBW, REF_FREQ, DISH_SIZE


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

    def beam_pattern(self, mode):
        """
        Return beam pattern for given mode
        :param mode: gauss, airy
        :return: beam pattern
        """

        if mode == 'gauss':
            return self.__gaussian()
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
