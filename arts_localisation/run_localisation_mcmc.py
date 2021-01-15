#!/usr/bin/env python3

import os
import sys
import numpy as np
from astropy.time import Time
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import corner
from schwimmbad import MPIPool
from mpi4py import MPI

from arts_localisation import tools
from arts_localisation.constants import NSB
from arts_localisation.beam_models import SBPatternSingle


# each process needs to access these models, make global instead of passing them around to save time
global models
# avoid using parallelization other than the MPI processes used in this script
os.environ["OMP_NUM_THREADS"] = "1"


class TestData:
    def __init__(self, snrmin):
        self.source_ra = 29.50333 * u.deg
        self.source_dec = 65.71675 * u.deg
        self.src = "R3_20200511_3610"
        self.burst = {'tstart': Time('2020-05-11T07:36:22.0'),
                      'toa': 3610.84 * u.s,
                      'ra_cb': 29.50333 * u.deg,
                      'dec_cb': 65.71675 * u.deg,
                      # 'snr_array': '/home/oostrum/localisation/mcmc/snr_R3/{self.src}_CB00_SNR.txt'
                      'snr_array': f'/data/arts/localisation/R3/snr/{self.src}_CB00_SNR.txt'
                      }
        # load S/N array
        data = np.loadtxt(self.burst['snr_array'], ndmin=2)
        sbs, snrs = data.T
        ind_det = snrs >= snrmin
        self.sb_det = sbs[ind_det].astype(int)
        self.sb_nondet = sbs[~ind_det].astype(int)
        self.snr_det = snrs[ind_det]

        # calculate ha, dec at burst ToA
        self.tarr = self.burst['tstart'] + self.burst['toa']
        self.ha_cb, self.dec_cb = tools.radec_to_hadec(self.burst['ra_cb'], self.burst['dec_cb'], self.tarr)


def set_guess_value(ndim, minval, maxval):
    """
    Generate a set of guess parameters

    :param int ndim: dimensions of guess vector
    :param float minval: minimum value
    :param float maxval: maximum value
    """
    return (maxval - minval) * np.random.random(ndim) + minval
