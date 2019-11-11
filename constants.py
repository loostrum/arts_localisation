#!/usr/bin/env python3
#
# WSRT constants

import numpy as np
import astropy.units as u


#CB_HPBW = 28.0835088*u.arcmin  # calculated from CB offsets file, assumed to be at 1420 MHz
CB_HPBW = 35.066*u.arcmin  # fit CB00 Oct 1 2019  @ 1420 MHz
REF_FREQ = 1420*u.MHz
DISH_SIZE = 25 * u.m

DISH_A8 = np.arange(8) * 144*u.m
DISH_A10 = np.arange(10) * 144*u.m
DISH_MAXISHORT = np.concatenate([np.arange(8) * 144, np.array([36, 90, 1332, 1404]) + 7*144]) * u.m

DISH = {'a8': DISH_A8, 'a10': DISH_A10, 'maxi-short': DISH_MAXISHORT}

WSRT_LAT = 52.915184*u.deg
WSRT_LON = 6.60387*u.deg
WSRT_ALT = 16*u.m

CB_MODEL_FILE = 'beam_models_190607.csv'
