#!/usr/bin/env python3
#
# WSRT constants

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation

#: File with definition of CB offsets
CB_OFFSETS = 'square_39p1.cb_offsets'
#: CB half-power width
CB_HPBW = 28.0835088 * u.arcmin  # calculated from CB offsets file
#: Reference frequency for CB half-power width
REF_FREQ = 1770 * u.MHz  # for then it matches the measured width at 1420
#: Dish diameter
DISH_SIZE = 25 * u.m
#: Bandwidth
BANDWIDTH = 300 * u.MHz

#: Dish positions in Apertif-8 setup
DISH_A8 = np.zeros((8, 3)) * u.m
DISH_A8[:, 1] = np.arange(8) * 144 * u.m
#: Dish positions in Apertif-10 setup
DISH_A10 = np.zeros((10, 3)) * u.m
DISH_A10[:, 1] = np.arange(10) * 144 * u.m
#: Dish positions in Maxi-short setup
DISH_MAXISHORT = np.zeros((12, 3)) * u.m
DISH_MAXISHORT[:, 1] = np.concatenate([np.arange(8) * 144, np.array([36, 90, 1332, 1404]) + 7 * 144]) * u.m


DISH = {'a8': DISH_A8, 'a10': DISH_A10, 'maxi-short': DISH_MAXISHORT}

WSRT_LAT = 52.915184 * u.deg  # = 52:54:54.66
WSRT_LON = 6.60387 * u.deg  # = 06:36:13.93
WSRT_ALT = 16 * u.m
#: WSRT location
WSRT_LOC = EarthLocation.from_geodetic(WSRT_LON, WSRT_LAT, WSRT_ALT)

#: Path to fitted CB model parameters
CB_MODEL_FILE = 'beam_models_190607.csv'

THETAMAX_CB = 130
PHIMAX_CB = 100
NTHETA_CB = 2601
NPHI_CB = 2001

THETAMAX_SB = 50
PHIMAX_SB = 50
NTHETA_SB = 10001
NPHI_SB = 201

# for 2D simulations
MAXDIST = 30  # arcmin
NPOINT = 300

#: Number of compound beams
NCB = 40
#: Number of tied-array beams
NTAB = 12
#: Number of synthesised beams
NSB = 71

#: ITRF positions of RT2 - RTD in maxi-short setup
DISH_POS_ITRF = np.array([[3828729.99081358872354031, 442735.17696416645776480, 5064923.00829000025987625],
                          [3828713.43109884625300765, 442878.21189340209821239, 5064923.00435999967157841],
                          [3828696.86994427768513560, 443021.24917263782117516, 5064923.00396999996155500],
                          [3828680.31391932582482696, 443164.28596862131962553, 5064923.00035000033676624],
                          [3828663.75159173039719462, 443307.32138055720133707, 5064923.00203999970108271],
                          [3828647.19342757249251008, 443450.35604637680808082, 5064923.00229999981820583],
                          [3828630.63486200943589211, 443593.39226634375518188, 5064922.99755000043660402],
                          [3828614.07606798363849521, 443736.42941620573401451, 5064923.00000000000000000],
                          [3828609.94224429363384843, 443772.19450029480503872, 5064922.99868000019341707],
                          [3828603.73202611599117517, 443825.83321168005932122, 5064922.99963000044226646],
                          [3828460.92418734729290009, 445059.52053928520763293, 5064922.99070999957621098],
                          [3828452.64716351125389338, 445131.03744105156511068, 5064922.98792999982833862]]) * u.meter

#: ITRF WSRT reference position
ARRAY_ITRF = np.array([3828630.63486200943589211, 443593.39226634375518188, 5064922.99755000043660402]) * u.meter
DISH_ITRF = {'a8': DISH_POS_ITRF[:8], 'maxi-short': DISH_POS_ITRF}
