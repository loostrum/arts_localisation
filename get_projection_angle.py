#!/usr/bin/env python3
# Calculate parallactic angle from ha, dec
# from https://github.com/brandon-rhodes/pyephem/issues/24:
#Equations from:
#    "A treatise on spherical astronomy" By Sir Robert Stawell Ball
#    (p. 91, as viewed on Google Books)
#
#    sin(eta)*sin(z) = cos(lat)*sin(HA)
#    cos(eta)*sin(z) = sin(lat)*cos(dec) - cos(lat)*sin(dec)*cos(HA)
#
#    Where eta is the parallactic angle, z is the zenith angle, lat is the
#    observer's latitude, dec is the declination, and HA is the hour angle.

import argparse
import astropy.units as u
import numpy as np

from convert import ha_to_proj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ha', required=True, type=float, help="Hour angle in degrees")
    parser.add_argument('--dec', required=True, type=float, help="Declination in degrees")

    args = parser.parse_args()

    ha = args.ha * u.deg
    dec = args.dec * u.deg

    print(ha_to_proj(ha, dec))
