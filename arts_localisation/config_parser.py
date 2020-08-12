#!/usr/bin/env python

import logging

import yaml
import astropy.units as u
from astropy.time import Time, TimeDelta

from .convert import cb_index_to_pointing


logger = logging.getLogger(__name__)


def parse_yaml(fname):
    """
    Parse a yaml file with settings for burst localisation
    :param str fname: Path to yaml config file

    :return: config (dict), excluded_dishes (list), source_coord (SkyCoord or None)
    """

    # read the file
    with open(fname, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # read global settings, if present
    source_coord = None
    excluded_dishes = []
    snrmin = None
    try:
        global_conf = config['global']
        # check for known source coordinates
        try:
            source_ra = global_conf['source_ra']
            source_dec = global_conf['source_dec']
            source_coord = (source_ra * u.deg, source_dec * u.deg)
        except KeyError:
            logger.debug('No source coordinates found')


        # check for S/N threshold
        try:
            snrmin = global_conf['snrmin']
        except KeyError:
            logger.debug('No S/N threshold found')

        # delete global config from raw config so only bursts are left
        del config['global']

    except KeyError:
        logger.debug('No global config found')
        pass

    # the remaining config keys are individual bursts
    for burst in config.keys():
        # parameters of the burst
        # try to read arrival time, either directly or as start time plus ToA
        try:
            tarr = Time(config[burst]['tarr'])
        except KeyError:
            try:
                tarr = Time(config[burst]['start'], format='isot', scale='utc') + TimeDelta(config[burst]['toa'], format='sec')
            except KeyError:
                logger.error("Could not get arrival time for burst {} from either tarr ot tsart + toa".format(burst))
                raise
        # set arrival time
        config[burst]['tarr'] = tarr

        # check for excluded dishes
        try:
            excluded_dishes = config[burst]['excluded_dishes']
            # convert to integers, RT2 = dish 0
            excluded_dishes = [value - 2 for value in excluded_dishes]
        except KeyError:
            logger.debug("No list of excluded dishes found")

        # check for telescope pointing
        pointing_coord = None
        try:
            pointing_ra = global_conf['pointing_ra']
            pointing_dec = global_conf['pointing_dec']
            pointing_coord = (pointing_ra * u.deg, pointing_dec * u.deg)
            logger.info("Telescope pointing found. Only use this option if ref_beam == 0.")
        except KeyError:
            logging.debug("No telescope pointing found")

        # Verify S/N limit is present if not globally set
        if snrmin is None:
            assert 'snrmin' in config[burst].keys()
        # else set local S/N limit to global value
        config[burst]['snrmin'] = snrmin

        # now check section for each compound beam, that start with CB
        beams = [key for key in config[burst].keys() if key.upper().startswith('CB')]
        for beam in beams:
            # read keys
            keys = config[burst][beam].keys()
            # convert all to lowercase
            keys = [k.lower() for k in keys]

            # if ra, dec are given, use these
            if 'ra' in keys and 'dec' in keys:
                cb_pointing = (config[burst][beam]['ra'] * u.deg, config[burst][beam]['dec'] * u.deg)
                # if pointing is also set, warn the user
                if pointing_coord is not None:
                    logger.warning("CB RA, Dec given, but telescope pointing is also set. Using CB RA, Dec")
            else:
                # telescope pointing must be set
                if pointing_coord is None:
                    logger.error("No telescope pointing and not RA, Dec set for this CB")
                    raise
                # calculate ra, dec from CB offsets
                cb_index = int(beam[2:])
                cb_pointing = cb_index_to_pointing(cb_index, *pointing_coord)
            # store the pointing of this CB
            config[burst][beam]['pointing'] = cb_pointing

            # SEFD is optional, but give user a warning if it is not present as it is easy to overlook
            if 'sefd' not in keys:
                logger.warning("No SEFD given for {} of burst {}".format(beam, burst))

            # check for S/N array; absent means it is assumed to be an upper limit beam
            if 'snr_array' not in keys:
                logger.warning("No S/N array found for {} of burst {}, assuming it "
                               "is an upper limit beam".format(beam, burst))

    return config, excluded_dishes, source_coord
