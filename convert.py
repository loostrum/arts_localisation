#!/usr/bin/env python
#
# Converter tools:
# RA <-> HA
# hh:mm:ss <-> deg
# dd:mm:ss <-> deg

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5
from astropy.time import Time, TimeDelta

from constants import WSRT_LON, WSRT_LAT


def ra_to_ha(ra, dec, t):
    """ 
    Convert J2000 RA, Dec to WSRT HA, Dec
    :param ra: right ascension with unit
    :param dec: declination with unit
    :param t: UT time (string or astropy.time.Time)
    :return: SkyCoord object of apparent HA, Dec coordinates
    """

    # Convert time to Time object if given as string
    if isinstance(t, str):
        t = Time(t)

    # Apparent LST at WSRT at this time
    lst = t.sidereal_time('apparent', WSRT_LON)
    # Equinox of date (because hour angle uses apparent coordinates)
    coord_system = FK5(equinox='J{}'.format(t.decimalyear))
    # convert coordinates to apparent
    coord_apparent = SkyCoord(ra, dec, frame='icrs').transform_to(coord_system)
    # HA = LST - apparent RA
    ha = lst - coord_apparent.ra
    # return SkyCoord of (Ha, Dec)
    return SkyCoord(ha, dec, frame=coord_system)

def ha_to_ra(ha, dec, t):
    """
    Convert WSRT HA, Dec to J2000 RA, Dec
    :param ha: hour angle with unit
    :param dec: declination with unit
    :param t: UT time (string or astropy.time.Time)
    :return: SkyCoord object of J2000 coordinates
    """

    # Convert time to Time object if given as string
    if isinstance(t, str):
        t = Time(t)

    # Apparent LST at WSRT at this time
    lst = t.sidereal_time('apparent', WSRT_LON)
    # Equinox of date (because hour angle uses apparent coordinates)
    coord_system = FK5(equinox='J{}'.format(t.decimalyear))
    # apparent RA = LST - HA
    ra_apparent = lst - ha
    coord_apparent = SkyCoord(ra_apparent, dec, frame=coord_system)
    return coord_apparent.transform_to('icrs')


def ha_to_proj(ha, dec):
    """
    Convert WSRT HA, Dec to parallactic angle 
    This is the SB rotation w.r.t. the RA-Dec frame
    :param ha: hour angle with unit
    :param dec: declination with unit
    """
    theta_proj = np.arctan(np.cos(WSRT_LAT)*np.sin(ha) / 
                    (np.sin(WSRT_LAT)*np.cos(dec) - 
                    np.cos(WSRT_LAT)*np.sin(dec)*np.cos(ha))).to(u.deg)
    return theta_proj.to(u.deg)
