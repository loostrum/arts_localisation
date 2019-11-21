#!/usr/bin/env python
#
# Converter tools
# HA/Dec to Alt/Az taken from:
# http://www.stargazing.net/kepler/altaz.html

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.time import Time

from constants import WSRT_LON, WSRT_LAT


def ra_to_ha(ra, dec, t, lon=WSRT_LON):
    """ 
    Convert J2000 RA, Dec to apparent HA, Dec
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
    dec = coord_apparent.dec
    # return SkyCoord of (Ha, Dec)
    return SkyCoord(ha, dec, frame=coord_system)


def ha_to_ra(ha, dec, t, lon=WSRT_LON):
    """
    Convert apparent HA, Dec to J2000 RA, Dec
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

    sinalt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(sinalt)

    cosaz = (np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat))
    az = np.arccos(cosaz)

    # fix sign of az
    m = np.sin(ha) > 0
    az[m] = 360*u.deg - az[m]

    return alt.to(u.deg), az.to(u.deg)


def altaz_to_hadec(alt, az, lat=WSRT_LAT):
    """
    Convert Alt, Az to HA, Dec
    :param alt: altitude with unit
    :param az: azimuth with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    sindec = np.cos(az) * np.cos(alt) * np.cos(lat) + np.sin(alt) * np.sin(lat)
    dec = np.arcsin(sindec)

    cosha = (np.sin(alt) - np.sin(dec) * np.sin(lat)) / (np.cos(dec) * np.cos(lat))
    ha = np.arccos(cosha)

    # fix sign of ha
    m = az < 180*u.deg
    ha[m] = -ha[m]

    return ha.to(u.deg), dec.to(u.deg)


def offset_to_radec(ra0, dec0, theta, phi):
    # equations from
    # https://github.com/LSSTDESC/Coord/blob/master/coord/celestial.py
    # project: sky to plane
    # deproject: plane to sky
    # and reference therein:
    # http://mathworld.wolfram.com/GnomonicProjection.html

    # reimplementation of celestial.CelestialCoord.deproject
    # sky_coord = center.deproject(u, v)

    # u,v are plane offset coordinates, here defined in RA Dec (but in HA Dec in celestial.py)
    # use uu to avoid issues with astropy unit
    uu = theta.to(u.radian).value
    vv = phi.to(u.radian).value

    # r = sqrt(u**2 + v**2)
    # c = arctan(r)
    # we only need sin(c) and cos(c), which are a function of r**2 only
    # define r**2
    rsq = uu**2 + vv**2
    cosc = 1./np.sqrt(1 + rsq)
    # sinc = r * cos(c), but we only need sinc / r
    sinc_over_r = cosc

    # equations to get ra, dec from reference ra0, dec0 and radec offset u,v :
    # sin(dec) = cos(c) sin(dec0) + v (sin(c)/r) cos(dec0)
    # tan(ra-ra0) = u (sin(c)/r) / (cos(dec0) cos(c) - v sin(dec0) (sin(c)/r))

    # sin dec
    sindec0 = np.sin(dec0)
    cosdec0 = np.cos(dec0)
    sindec = cosc * sindec0 + vv * sinc_over_r * cosdec0
    # tan delta RA, split in numerator and denominator so we can use arctan2 to get the right quadrant
    tandra_num = uu * sinc_over_r
    tandra_denom = cosdec0 * cosc - vv * sindec0 * sinc_over_r

    # dec
    dec = np.arcsin(sindec)
    # ra
    ra = ra0 + np.arctan2(tandra_num, tandra_denom)

    return ra.to(u.deg), dec.to(u.deg)
