#!/usr/bin/env python3

import argparse

import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from convert import ra_to_ha


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="Yaml config file")

    args = parser.parse_args()

    # source parameters
    with open(args.file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

        tarr = Time(params['tstart']) + TimeDelta(params['t_in_obs'], format='sec')

    if params['drift']:
        # read hour angles
        CB_best_center = SkyCoord(params['ha_best_cb'], params['dec_best_cb'], unit=u.deg, frame='icrs')
        CB00_center = SkyCoord(params['ha_cb00'], params['dec_cb00'], unit=u.deg, frame='icrs')
    else:
        # read RA, Dec, convert to WSRT HA Dec
        CB_best_center = ra_to_ha(params['ra_best_cb']*u.deg, params['dec_best_cb']*u.deg, tarr)
        CB00_center = ra_to_ha(params['ra_cb00']*u.deg, params['dec_cb00']*u.deg, tarr)

    print("Best CB apparent HA, Dec: ({:.5f}, {:.5f})".format(CB_best_center.ra, CB_best_center.dec))
    print("CB00 apparent HA, Dec: ({:.5f}, {:.5f})".format(CB00_center.ra, CB00_center.dec))
