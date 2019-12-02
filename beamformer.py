#!/usr/bin/env python3
#
# beamformer tool to create TAB pattern

import itertools

import numpy as np
import astropy.units as u
import astropy.constants as const


class BeamFormer(object):

    def __init__(self, dish_pos, theta_proj=0*u.deg, freqs=1500*u.MHz, ntab=12):
        self.theta_proj = theta_proj
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs])
        self.dish_pos = dish_pos
        self.freqs = freqs
        self.ntab = ntab

    def __dphi(self, dtheta, baseline, grid=False):
        """
        Compute phase difference for given offset and baseline
        dphi = freq * dx /c
        :param dtheta: offsets
        :param baseline: baseline length
        :param grid: whether or not dheta is a 2D grid
        :return: phases
        """

        if grid:
            dphi = self.freqs[..., None, None] * baseline / const.c * \
                   (np.sin(self.theta_proj + dtheta) -
                    np.sin(self.theta_proj))
        else:
            dphi = self.freqs[..., None] * baseline / const.c * \
                   (np.sin(self.theta_proj + dtheta) -
                    np.sin(self.theta_proj))

        return dphi.to(1).value

    def beamform(self, dtheta, ref_dish=0, tab=0):
        """
        Compute total power for given offsets and dish positions
        :param dtheta: E-W offsets
        :param ref_dish: reference dish
        :param tab: TAB index
        :return: beamformed power, same shape as dtheta
        """
        # check number of dimensions in dtheta
        if len(dtheta.shape) == 2:
            grid = True
        else:
            assert len(dtheta.shape) == 1
            grid = False

        # initalize output E field
        if grid:
            e_tot = np.zeros((self.freqs.shape[0], dtheta.shape[0], dtheta.shape[1]), dtype=complex)
        else:
            e_tot = np.zeros((self.freqs.shape[0], dtheta.shape[0]), dtype=complex)

        baselines = self.dish_pos - self.dish_pos[ref_dish]
        for i, baseline in enumerate(baselines):
            # calculate tab phase offset
            tab_dphi = float(i) * tab/self.ntab
            # calculate geometric phase offset
            geometric_dphi = self.__dphi(dtheta, baseline, grid=grid)
            dphi = geometric_dphi + tab_dphi
            # store as complex value (amplitude of E field = 1)
            e_tot += np.exp(1j*2*np.pi*dphi)

        # normalize by number of signals
        e_tot /= len(baselines)

        # convert E field to intensity
        i_tot = np.abs(e_tot)**2
        return i_tot
