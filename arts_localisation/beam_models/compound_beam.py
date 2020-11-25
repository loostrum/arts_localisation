#!/usr/bin/env python3
#
# Tool to generate the compound beam pattern

from copy import copy
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.optimize import curve_fit
from scipy.special import j1

from arts_localisation.constants import CB_HPBW, REF_FREQ, DISH_SIZE, CB_MODEL_FILE


def gauss_2d(xy, x_mean, x_sig, y_mean, y_sig, rho):
    """
    Construct a 2D Gaussian

    :param array xy: 2xN array of x, y values
    :param float x_mean: mean of x
    :param float x_sig: sigma of x
    :param float y_mean: mean of y
    :param float y_sig: sigma of y
    :param float rho: correlation between x and y
    :return: An NxN array with the Gaussian distribution
    """
    x, y = xy
    a = -1 / (2 * (1 - rho ** 2))
    b = ((x - x_mean) / x_sig) ** 2
    c = ((y - y_mean) / y_sig) ** 2
    d = (2 * rho * (x - y_mean) * (x - y_mean) / (x_sig * y_sig))
    return np.exp(a * (b + c - d))


class CompoundBeam:

    def __init__(self, freqs, theta=0 * u.deg, phi=0 * u.deg, rot=0 * u.deg):
        """
        Generate a compound beam pattern
        :param freqs: observing frequencies
        :param theta: E-W offsets (default: 0)
        :param phi: N-S offsets (default: 0)
        :param rot: rotation applied to model (parallactic angle, default 0)
        """
        if not isinstance(theta.value, np.ndarray):
            theta = np.array([theta.value]) * theta.unit
        if not isinstance(phi.value, np.ndarray):
            phi = np.array([phi.value]) * phi.unit
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit

        # ensure theta and phi use same units
        phi = phi.to(theta.unit)

        # check shape
        self.grid = False
        ndim_theta = len(theta.shape)
        ndim_phi = len(phi.shape)
        assert ndim_theta == ndim_phi
        if ndim_theta == 2:
            # shapes have to be exactly equal in this case
            assert np.array_equal(theta.shape, phi.shape)
            self.grid = True

        self.theta = theta
        self.phi = phi
        self.freqs = freqs
        self.rot = rot

        # load beam measurements
        try:
            fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), CB_MODEL_FILE)
            self.beams_measured = np.genfromtxt(fname, delimiter=',', names=True)
        except OSError as e:
            print(f"Failed to load beam measurement file, real beams not available ({e})")
            self.beams_measured = None

    def beam_pattern(self, mode, cb=None):
        """
        Return beam pattern for given mode

        :param str mode: gauss, airy, or real
        :param int cb: beam number (only used for real beam)
        :return: beam pattern
        """

        if mode == 'gauss':
            return self._gaussian()
        elif mode == 'real':
            return self._real(cb)
        elif mode == 'airy':
            return self._airy()
        else:
            raise ValueError("Mode should be gauss, real, or airy")

    def _gaussian(self):
        """
        Gaussian beam pattern

        :return: beam pattern
        """

        if self.grid:
            # shape of phi and theta are equal in this case
            output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[1]))
        else:
            # separate 1D phi and theta arrays
            output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[0]))

        # convert beam width to gaussian sigma at each freq
        sigmas = CB_HPBW * REF_FREQ / self.freqs / (2. * np.sqrt(2 * np.log(2)))

        # calculate response at each sigma
        for i, sigma in enumerate(sigmas):
            if self.grid:
                arg = -.5 * (self.theta ** 2 + self.phi ** 2) / sigma ** 2
            else:
                arg = -.5 * (self.theta ** 2 + self.phi[..., None] ** 2) / sigma ** 2
            output_grid[i] = np.exp(arg)

        return output_grid

    def _airy(self):
        """
        Airy disk beam pattern

        :return: beam pattern
        """
        if self.grid:
            # shape of phi and theta are equal in this case
            output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[1]))
        else:
            # separate 1D phi and theta arrays
            output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[0]))

        lambd = const.c / self.freqs
        k = 2 * np.pi / lambd
        a = DISH_SIZE / 2
        if self.grid:
            angle = np.sqrt(self.phi ** 2 + self.theta ** 2)
        else:
            angle = np.sqrt(self.phi[..., None] ** 2 + self.theta ** 2)
        arg = k[..., None, None] * a * np.sin(angle)[None, ...]
        arg = arg.to(1).value

        out = 2 * j1(arg) / arg
        out[arg == 0] = 1.

        return out

    def _real(self, cb, thresh=.5):
        """
        Measured beam pattern

        :param int cb: beam number
        :param flaot thresh: max NaN fraction, reverts to default gaussian above this
                            (Default: 0.5)
        :return: beam pattern
        """
        n = 40
        ref_freq = 1399.662250 * u.MHz
        ref_freq_str = '1399662250'
        name = f'B{cb:02d}_{ref_freq_str}'
        beam = self.beams_measured[name].reshape(n, n)

        dy = 1. / 36 * 60   # arcmin at ref freq
        dx = -dy  # horizontal axis of data is ha instead of ra

        # fit 2D gaussian
        x = (np.arange(n) - n / 2) * dx
        y = (np.arange(n) - n / 2) * dy
        X, Y = np.meshgrid(x, y)

        # apply rotation
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        theta -= self.rot.to(u.radian).value
        X = r * np.cos(theta)
        Y = r * np.sin(theta)

        # check NaNs
        nans = ~np.isfinite(beam)
        nan_frac = np.sum(nans) / float(n) ** 2
        if nan_frac > thresh:
            print("Warning: NaN frac is {} for CB{:02d}, "
                  "reverting to default gaussian beam".format(nan_frac, cb))
            return self.__gaussian()

        # remove nans
        Xshort = X[~nans]
        Yshort = Y[~nans]
        beam_short = beam[~nans]

        XYshort = np.vstack([Xshort.ravel(), Yshort.ravel()])

        # do fit
        # X, sigX, Y, sigY, rho
        guess = [0, 10, 0, 10, 0]
        popt, pcov = curve_fit(gauss_2d, XYshort, beam_short.flatten(), p0=guess, maxfev=10000)

        # apply model to given coordinates
        # todo: shift CB peak to expected position? Now part of fit
        # assume sigma scales with 1/freq
        output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[0]))
        scalings = ref_freq / self.freqs
        if self.grid:
            X = self.theta.to(u.arcmin).value
            Y = self.phi.to(u.arcmin).value
        else:
            X, Y = np.meshgrid(self.theta.to(u.arcmin).value, self.phi.to(u.arcmin).value)
        XY = np.vstack([X.ravel(), Y.ravel()])
        for i, scaling in enumerate(scalings):
            this_popt = copy(popt)
            # rescale sigma in both theta and phi
            this_popt[1] *= scaling
            this_popt[3] *= scaling

            # calculate response
            output_grid[i] = gauss_2d(XY, *this_popt).reshape(len(self.phi), len(self.theta))
        return output_grid
