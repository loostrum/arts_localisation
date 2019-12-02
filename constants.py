#!/usr/bin/env python3
#
# WSRT constants

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation


CB_HPBW = 28.0835088*u.arcmin # calculated from CB offsets file
REF_FREQ = 1770*u.MHz  # for then it matches the measured width at 1420
# CB_HPBW = 35.066*u.arcmin  # fit CB00 Oct 1 2019  @ 1420 MHz
# REF_FREQ = 1420*u.MHz
DISH_SIZE = 25 * u.m

DISH_A8 = np.arange(8) * 144*u.m
DISH_A10 = np.arange(10) * 144*u.m
DISH_MAXISHORT = np.concatenate([np.arange(8) * 144, np.array([36, 90, 1332, 1404]) + 7*144]) * u.m

DISH = {'a8': DISH_A8, 'a10': DISH_A10, 'maxi-short': DISH_MAXISHORT}

WSRT_LAT = 52.915184*u.deg  # = 52:54:54.66
WSRT_LON = 6.60387*u.deg  # = 06:36:13.93
WSRT_ALT = 16*u.m
WSRT_LOC = EarthLocation.from_geodetic(WSRT_LON, WSRT_LAT, WSRT_ALT)

CB_MODEL_FILE = 'beam_models_190607.csv'

THETAMAX_CB = 130
PHIMAX_CB = 100
NTHETA_CB = 2601
NPHI_CB = 2001

THETAMAX_SB = 50
PHIMAX_SB = 50
NTHETA_SB = 10001
NPHI_SB = 201
