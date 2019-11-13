#!/usr/bin/env python3
#
# Tool to generate the compound beam pattern

from copy import copy

import numpy as np
import astropy.units as u
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from constants import CB_HPBW, REF_FREQ, DISH_SIZE, CB_MODEL_FILE

def gauss_2d(xy, x_mean, x_sig, y_mean, y_sig, rho):
    """
    xy: 2xN array of x, y values
    x_mean: mean of x
    x_sig: sigma of x
    y_mean: mean of y
    y_sig: sigma of y
    rho: correlation between x and y
    """
    x, y = xy
    a = -1 / (2*(1-rho**2))
    b = ((x-x_mean)/x_sig)**2
    c = ((y-y_mean)/y_sig)**2
    d = (2*rho*(x-y_mean)*(x-y_mean) / (x_sig * y_sig))
    return np.exp(a * (b+c-d))

class CompoundBeam(object):
    
    def __init__(self, freqs, theta=0*u.deg, phi=0*u.deg, rot=0*u.deg):
        """
        Generate a compound beam pattern
        :param freqs: observing frequencies
        :param theta: E-W offsets (default: 0)
        :param phi: N-S offsets (default: 0)
        :param rot: rotation applied to model (parallactic angle, default 0)
        """

        if not isinstance(theta, np.ndarray):
            theta = np.array([theta.value]) * theta.unit
        if not isinstance(phi, np.ndarray):
            phi = np.array([phi.value]) * phi.unit
        if not isinstance(freqs, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit

        # ensure theta and phi use same units
        phi = phi.to(theta.unit)

        self.theta = theta
        self.phi = phi
        self.freqs = freqs
        self.rot = rot

        # load beam measurements
        self.beams_measured = np.genfromtxt(CB_MODEL_FILE, delimiter=',', names=True)
        

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
            raise ValueError("Mode should be gauss, real, or airy")

    def __gaussian(self):
        """
        Gaussian beam pattern
        print(freqs.shape)
        :return: beam pattern
        """

        output_grid = np.zeros((self.freqs.shape[0], self.phi.shape[0], self.theta.shape[0]))

        # convert beam width to gaussian sigma at each freq
        sigmas = CB_HPBW * REF_FREQ / self.freqs / (2. * np.sqrt(2 * np.log(2)))

        # calculate response at each sigma
        for i, sigma in enumerate(sigmas):
            arg = -.5 * (self.theta**2 + self.phi[..., None]**2) / sigma**2
            output_grid[i] = np.exp(arg) / 2

        return output_grid

    def __real(self, cb, thresh=.5):
        """
        Measured beam pattern
        :param cb: beam number
        :param thresh: max NaN fraction, reverts to default gaussian above this
        :return: beam pattern
        """
        n = 40
        ref_freq = 1399.662250 * u.MHz
        ref_freq_str = '1399662250'
        name = 'B{:02d}_{}'.format(cb, ref_freq_str)
        beam = self.beams_measured[name].reshape(n, n)

        dx = 1./36 * 60   # arcmin at ref freq
        dy = dx

        # fit 2D gaussian
        x = (np.arange(n) - n/2) * dx
        y = (np.arange(n) - n/2) * dy
        X, Y = np.meshgrid(x, y)

        # apply rotation
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        theta -= self.rot
        X = r*np.cos(theta)
        Y = r*np.sin(theta)

        # check NaNs
        nans =~np.isfinite(beam)
        nan_frac = np.sum(nans) / float(n)**2
        if nan_frac > thresh:
            print("Warning: NaN frac is {} for CB{:02d}, "
                  "reverting to default gaussian beam".format(nan_frac, cb))
            return self.__gaussian()

        # remove nans
        XY = np.vstack([X.ravel(), Y.ravel()])
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
