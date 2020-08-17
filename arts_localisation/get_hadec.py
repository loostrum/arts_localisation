#!/usr/bin/env python3

import argparse

import yaml
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord

from arts_localisation.tools import radec_to_hadec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Yaml config file")

    args = parser.parse_args()

    # source parameters
    with open(args.file) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

        tarr = Time(params['tstart']) + TimeDelta(params['t_in_obs'], format='sec')

    if params['drift']:
        # read hour angles
        CB_best_center = SkyCoord(params['ha_best_cb'], params['dec_best_cb'], unit=u.deg, frame='icrs')
        CB00_center = SkyCoord(params['ha_cb00'], params['dec_cb00'], unit=u.deg, frame='icrs')
    else:
        # read RA, Dec, convert to WSRT HA Dec
        CB_best_center = radec_to_hadec(params['ra_best_cb'] * u.deg, params['dec_best_cb'] * u.deg, tarr)
        CB00_center = radec_to_hadec(params['ra_cb00'] * u.deg, params['dec_cb00'] * u.deg, tarr)

    ha_best = CB_best_center.ra
    if ha_best > 180 * u.deg:
        ha_best -= 360 * u.deg

    ha00 = CB00_center.ra
    if ha00 > 180 * u.deg:
        ha00 -= 360 * u.deg

    print(f"Best CB apparent HA, Dec: ({ha_best:.5f}, {CB_best_center.dec:.5f})")
    print(f"CB00 apparent HA, Dec: ({ha00:.5f}, {CB00_center.dec:.5f})")
