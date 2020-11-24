#!/usr/bin/env python
#
# Converter tools
# HA/Dec to Alt/Az taken from:
# http://www.stargazing.net/kepler/altaz.html

import os
import errno

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, SphericalRepresentation
from astropy.time import Time

from arts_localisation.constants import WSRT_LON, WSRT_LAT, NCB, CB_OFFSETS


def limit(val, minval=-1, maxval=1):
    """
    Where val > maxval, replace by maxval

    Where val < minval, replace by minval

    :param val: input value
    :param minval: minimum value
    :param maxval: maximum value
    :return: limited value
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
    :return: HA, Dec with unit
    """

    # Convert time to Time object if given as string
    if isinstance(t, str):
        t = Time(t)

    coord = SkyCoord(ra, dec, frame='icrs', obstime=t)
    ha = lon - coord.itrs.spherical.lon
    ha.wrap_at(12 * u.hourangle, inplace=True)
    dec = coord.itrs.spherical.lat

    return ha, dec


def hadec_to_radec(ha, dec, t, lon=WSRT_LON):
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

    # create spherical representation of ITRS coordinates of given ha, dec
    itrs_spherical = SphericalRepresentation(lon - ha, dec, 1.)
    # create ITRS object, which requires cartesian input
    coord = SkyCoord(itrs_spherical.to_cartesian(), frame='itrs', obstime=t)
    # convert to J2000
    return coord.icrs


def hadec_to_par(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to parallactic angle

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    :return: parallactic angle
    """
    theta_par = np.arctan(np.cos(lat) * np.sin(ha)
                          / (np.sin(lat) * np.cos(dec)
                          - np.cos(lat) * np.sin(dec) * np.cos(ha))).to(u.deg)
    return theta_par.to(u.deg)


def hadec_to_proj(ha, dec, lat=WSRT_LAT):
    """
    Convert HA, Dec to E-W baseline projection angle

    :param ha: hour angle with unit
    :param dec: declination with unit
    :param lat: Latitude with unit (default: WSRT)
    :return: projection angle
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
    :return: altitude, azimuth
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
    :return: hour angle, declination
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
    :return: rotated X, Y
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
    """
    Get pointing of given CB based on telescope pointing. Assumes reference_beam = 0,
    i.e. the telescope pointing coincides with the phase center of CB00

    :param int/list cb: CB index
    :param Quantity pointing_ra: Pointing right ascension
    :param Quantity pointing_dec: Pointing Declination
    :return: (RA, Dec) tuple of CB pointing, with unit
    """
    assert cb >= 0, 'CB index cannot be lower than zero'
    assert cb < NCB, f'CB index cannot be higher than {NCB - 1}'

    # load the offsets file
    offsets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), CB_OFFSETS)
    raw_cb_offsets = np.loadtxt(offsets_file, usecols=[1, 2], delimiter=',')
    assert len(raw_cb_offsets) == NCB, f'CB offsets file does not contain {NCB} CBs'

    # get offset for requested CB
    dra, ddec = raw_cb_offsets[cb]
    dra = dra * u.deg
    ddec = ddec * u.deg

    # calculate position of the CB
    cb_ra, cb_dec = offset_to_coord(pointing_ra, pointing_dec, dra, ddec)
    return cb_ra.to(u.deg), cb_dec.to(u.deg)


def get_neighbours(cbs):
    """
    Get neighbouring CBs of given CB(s)

    :param int/list cbs: CB(s) to get neighbours for
    :return: flattened list of neighbours
    """
    # list of neighbours for all CBs, ordered per CB row
    # nth element of the list contants a list of neighbours of CB n
    # TODO: double check this list
    all_neighbours = [[17, 18, 23, 24],
                      [2, 8], [1, 3, 8, 9], [2, 4, 9, 10], [3, 5, 10, 11], [4, 6, 11, 12], [5, 7, 12, 13], [6, 13, 14],
                      [1, 2, 9, 15], [2, 3, 8, 10, 15, 16], [3, 4, 9, 11, 16, 17], [4, 5, 10, 12, 17, 18], [5, 6, 11, 13, 18, 19], [6, 7, 12, 14, 19, 20], [7, 13, 20],
                      [8, 9, 16, 21, 22], [9, 10, 15, 17, 22, 23], [0, 10, 11, 16, 18, 23, 24], [0, 11, 12, 17, 19, 24, 26], [12, 13, 18, 20, 25, 26], [13, 14, 19, 26],
                      [15, 22, 27], [15, 16, 21, 23, 27, 28], [0, 16, 17, 22, 24, 28, 29], [0, 17, 18, 23, 25, 29, 30], [18, 19, 24, 26, 30, 31], [19, 20, 25, 31, 32],
                      [21, 22, 28, 33, 34], [22, 23, 27, 29, 34, 35], [23, 24, 28, 30, 35, 36], [24, 25, 29, 31, 36, 37], [25, 26, 30, 32, 37, 38], [26, 31, 38, 39],
                      [27, 34], [27, 28, 33, 35], [28, 29, 34, 36], [29, 30, 35, 37], [30, 31, 36, 38], [31, 32, 37, 39], [32, 38]]

    output = []
    if isinstance(cbs, int):
        cbs = [cbs]
    assert min(cbs) >= 0, 'CB index cannot be lower than zero'
    assert max(cbs) < NCB, f'CB index cannot be higher than {NCB - 1}'
    for cb in cbs:
        output.extend(all_neighbours[cb])
    # remove duplicates and return in ascending order
    return sorted(list(set(output)))


def makedirs(path):
    """
    Mimic os.makedirs, but do not error when directory already exists

    :param str path: path to recursively create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
