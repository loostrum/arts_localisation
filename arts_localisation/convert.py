#!/usr/bin/env python
#
# Converter tools
# HA/Dec to Alt/Az taken from:
# http://www.stargazing.net/kepler/altaz.html

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.time import Time

from .constants import WSRT_LON, WSRT_LAT


def limit(val, minval=-1, maxval=1):
    """
    Where val > maxval, replace by maxval

    Where val < minval, replace by minval

    :param val: input value
    :param minval: minimum value
    :param maxval: maximum value
    """
    # replace < minval
    m = val < minval
    val[m] = minval
    # replace > maxval
    m = val > maxval
    val[m] = maxval
    return val


def radec_to_hadec(ra, dec, t, lon=WSRT_LON):
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


def hadec_to_radec(ha, dec, t, lon=WSRT_LON, apparent=True):
    """
    Convert apparent HA, Dec to J2000 RA, Dec

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param t: UT time (string or astropy.time.Time)
    :param lon: Longitude with unit (default: WSRT)
    :param apparent: Whether or not HA, Dec are apparent coordinates (default: True)
    :return: SkyCoord object of J2000 coordinates
    """

    # Convert time to Time object if given as string
    if isinstance(t, str):
        t = Time(t)

    # Apparent LST at WSRT at this time
    lst = t.sidereal_time('apparent', lon)
    if apparent:
        # Equinox of date
        coord_system = FK5(equinox='J{}'.format(t.decimalyear))
    else:
        # J2000
        coord_system = FK5(equinox='J2000')
    # apparent RA = LST - HA
    ra_apparent = lst - ha
    coord_apparent = SkyCoord(ra_apparent, dec, frame=coord_system)
    return coord_apparent.transform_to('icrs')


def hadec_to_par(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to parallactic angle

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """
    theta_par = np.arctan(np.cos(lat) * np.sin(ha)
                          / (np.sin(lat) * np.cos(dec)
                          - np.cos(lat) * np.sin(dec) * np.cos(ha))).to(u.deg)
    return theta_par.to(u.deg)


def hadec_to_proj(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to projection angle
    This is the E-W baseline projection angle

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    alt, az = hadec_to_altaz(ha, dec, lat)
    cos_theta_proj = np.sqrt(np.sin(alt) ** 2 + (np.cos(alt) * np.cos(az)) ** 2)
    cos_theta_proj = limit(cos_theta_proj)
    theta_proj = np.arccos(cos_theta_proj)
    return theta_proj.to(u.deg)


def hadec_to_altaz(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to Alt, Az

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    sinalt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    # avoid sinalt out of range
    sinalt = limit(sinalt)
    alt = np.arcsin(sinalt)

    cosaz = (np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat))
    # avoid cosaz out of range
    cosaz = limit(cosaz)
    az = np.arccos(cosaz)

    # fix sign of az
    m = np.sin(ha) > 0
    az[m] = 360 * u.deg - az[m]

    return alt.to(u.deg), az.to(u.deg)


def altaz_to_hadec(alt, az, lat=WSRT_LAT):
    """
    Convert Alt, Az to HA, Dec

    :param alt: altitude with unit
    :param az: azimuth with unit
    :param lat: Latitude with unit (default: WSRT)
    """

    sindec = np.cos(az) * np.cos(alt) * np.cos(lat) + np.sin(alt) * np.sin(lat)
    # avoid sindec out of range
    sindec = limit(sindec)
    dec = np.arcsin(sindec)

    cosha = (np.sin(alt) - np.sin(dec) * np.sin(lat)) / (np.cos(dec) * np.cos(lat))
    # avoid cosha out of range
    cosha = limit(cosha)
    ha = np.arccos(cosha)

    # fix sign of ha
    m = az < 180 * u.deg
    ha[m] = -ha[m]

    return ha.to(u.deg), dec.to(u.deg)


def offset_to_coord(ra0, dec0, theta, phi):
    """
    Convert a projected offset (theta, phi) to
    coordinate with reference (ra0, dec0)

    :param ra0: Reference RA or Az
    :param dec0: Reference Dec or Alt
    :param theta: RA or Az offset
    :param phi: Dec or Alt offset
    :return: (Ra, Dec) or (Az, Alt) of offset point
    """
    # equations from
    # https://github.com/LSSTDESC/Coord/blob/master/coord/celestial.py
    # project: sky to plane
    # deproject: plane to sky
    # and reference therein:
    # http://mathworld.wolfram.com/GnomonicProjection.html

    # reimplementation of celestial.CelestialCoord.deproject
    # sky_coord = center.deproject(u, v)

    # u,v are plane offset coordinates
    # use uu to avoid issues with astropy unit
    uu = theta.to(u.radian).value
    vv = phi.to(u.radian).value

    # r = sqrt(u**2 + v**2)
    # c = arctan(r)
    # we only need sin(c) and cos(c), which are a function of r**2 only
    # define r**2
    rsq = uu**2 + vv**2
    cosc = 1. / np.sqrt(1 + rsq)
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


def coord_to_offset(ra0, dec0, ra1, dec1):
    """
    Convert point (ra1, dec1) to projected offset
    from reference (ra0, dec0)

    :param ra0: Reference RA or Az
    :param dec0: Reference Dec or Alt
    :param ra1: Target RA or Az
    :param dec1: Target Dec or Alt
    :return: (theta, phi) offset
              theta is offset in RA or Az
              phi is offset in Dec or Alt
    """
    # convert target radec into offset
    # reverse of offset_to_radec

    # x = k cos(dec) sin(ra0-ra)  # reversed sin arg, to have +x = +ra
    # y = k ( cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra-ra0) )
    # k = 1 / cos(c)
    # cos(c) = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra-ra0)

    # define convenience numbers
    sinra0 = np.sin(ra0)
    sinra1 = np.sin(ra1)
    cosra0 = np.cos(ra0)
    cosra1 = np.cos(ra1)

    sindec0 = np.sin(dec0)
    sindec1 = np.sin(dec1)
    cosdec0 = np.cos(dec0)
    cosdec1 = np.cos(dec1)

    # cos(dra) = cos(ra-ra0) = cos(ra0) cos(ra) + sin(ra0) sin(ra)
    # sin(dra) = sin(ra - ra0) = cos(ra0) sin(ra) -  sin(ra0) cos(ra) (reversed as we use +x = +ra)
    cosdra = cosra0 * cosra1 + sinra0 * sinra1
    sindra = cosra0 * sinra1 - sinra0 * cosra1

    cosc = sindec0 * sindec1 + cosdec0 * cosdec1 * cosdra
    k = 1. / cosc

    uu = k * cosdec1 * sindra * u.radian
    vv = k * (cosdec0 * sindec1 - sindec0 * cosdec1 * cosdra) * u.radian

    return uu.to(u.arcmin), vv.to(u.arcmin)


def rotate_coordinate_grid(X, Y, angle, origin=None):
    """
    Rotate input coordinate grid by given angle around given origin (default: center)

    :param X: input x coordinates
    :param Y: input y coordinates
    :param angle: rotation angle
    :param origin: tuple with origin for rotation (default: center of XY grid)
    """

    assert X.shape == Y.shape

    if origin is None:
        ny, nx = X.shape
        yind = int(ny / 2)
        xind = int(nx / 2)
        # find center of grid
        xmid = X[yind, xind]
        ymid = Y[yind, xind]
    else:
        xmid, ymid = origin

    # convert to polar
    r = np.sqrt((X - xmid) ** 2 + (Y - ymid) ** 2)
    theta = np.arctan2(Y - ymid, X - xmid)
    # apply rotation
    theta += angle
    # convert back to cartesian
    X = r * np.cos(theta) + xmid
    Y = r * np.sin(theta) + ymid

    return X, Y


def cb_index_to_pointing(cb, pointing_ra, pointing_dec):
    raise NotImplementedError
