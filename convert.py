#!/usr/bin/env python
#
# Converter tools
# HA/Dec to Alt/Az taken from:
# http://star-www.st-and.ac.uk/~fv/webnotes/chapter7.htm

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, FK5
from astropy.time import Time, TimeDelta

from constants import WSRT_LON, WSRT_LAT


def ra_to_ha(ra, dec, t, lon=WSRT_LON):
    """ 
    Convert J2000 RA, Dec to WSRT HA, Dec
    :param ra: right ascension with unit
    :param dec: declination with unit
    :param t: UT time (string or astropy.time.Time)
    :param lon: Longitude with unit (default: WSRT)
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

def ha_to_ra(ha, dec, t, lon=WSRT_LON):
    """
    Convert HA, Dec to J2000 RA, Dec
    :param ha: hour angle with unit
    :param dec: declination with unit
    :param t: UT time (string or astropy.time.Time)
    :param lon: Longitude with unit (default: WSRT)
    :return: SkyCoord object of J2000 coordinates
    """

    # Convert time to Time object if given as string
    if isinstance(t, str):
        t = Time(t)

    # Apparent LST at WSRT at this time
    lst = t.sidereal_time('apparent', lon)
    # Equinox of date (because hour angle uses apparent coordinates)
    coord_system = FK5(equinox='J{}'.format(t.decimalyear))
    # apparent RA = LST - HA
    ra_apparent = lst - ha
    coord_apparent = SkyCoord(ra_apparent, dec, frame=coord_system)
    return coord_apparent.transform_to('icrs')


def ha_to_par(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to parallactic angle 
    This is the SB rotation w.r.t. the RA-Dec frame
    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """
    theta_par = np.arctan(np.cos(lat)*np.sin(ha) /
                   (np.sin(lat)*np.cos(dec) -
                   np.cos(lat)*np.sin(dec)*np.cos(ha))).to(u.deg)
    return theta_par.to(u.deg)


def ha_to_proj(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to projection angle
    This is the E-W baseline projection angle
    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    alt, az = hadec_to_altaz(ha, dec, lat)
    # ToDo: verify if this is correct
    theta_proj = np.arccos(np.sqrt(np.sin(alt)**2 + (np.cos(alt)*np.cos(az))**2))
    return theta_proj.to(u.deg)


def hadec_to_altaz(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to Alt, Az
    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """
    
    colat = 90*u.deg - lat
    alt = np.arcsin(np.sin(dec)*np.sin(colat) + np.cos(dec)*np.cos(colat)*np.cos(ha))
    az = np.arcsin(-np.sin(ha)*np.cos(dec) / np.cos(alt))
    # if Dec > latitude, we are pointing north
    # else south: add 180 and flip pattern
    if dec < lat:
        az = 180*u.deg - az
    ## numpy arcsin returns value in range [-180, 180] deg
    ## ensure az is in [0, 360]
    if az < 0*u.deg:
        az += 360*u.deg
    if az > 360*u.deg:
        az -= 360*u.deg
    
    return alt.to(u.deg), az.to(u.deg)


def altaz_to_hadec(alt, az, lat=WSRT_LAT):
    """
    Convert Alt, Az to HA, Dec
    :param alt: altitude with unit
    :param az: azimuth with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    colat = 90*u.deg - lat
    dec = np.arcsin(np.sin(alt)*np.sin(colat) + np.cos(alt)*np.cos(colat)*np.cos(az))
    ha = np.arcsin(-np.sin(az)*np.cos(alt) / np.cos(dec))

    return ha, dec
