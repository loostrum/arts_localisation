#!/usr/bin/env python

import logging

import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from convert import cb_index_to_pointing


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

    # read global settings
    assert 'global' in config.keys(), 'Global config missing'

    # check if required keys are present
    for key in ('ra', 'dec', 'size', 'resolution', 'cb_model', 'fmin_data', 'bandwidth'):
        assert key in config['global'].keys(), 'Key missing: {}'.format(key)

    # check if frequency range to use is given, else set such that full range is used
    if 'fmin' not in config['global'].keys():
        config['global']['fmin'] = 0
    if 'fmax' not in config['global'].keys():
        config['global']['fmax'] = np.inf

    # check for known source coordinates
    try:
        source_ra = config['global']['source_ra']
        source_dec = config['global']['source_dec']
        config['global']['source_coord'] = (source_ra * u.deg, source_dec * u.deg)
    except KeyError:
        logger.debug('No source coordinates found')

    # check for S/N threshold
    snrmin = None
    try:
        snrmin = config['global']['snrmin']
    except KeyError:
        logger.debug('No S/N threshold found in global config')

    # the remaining config keys are individual bursts
    bursts = [key for key in config.keys() if not key == 'global']
    # there should be at least one burst
    assert bursts, 'No bursts found'
    for burst in bursts:
        # parameters of the burst
        # try to read arrival time, either directly or as start time plus ToA
        try:
            tarr = Time(config[burst]['tarr'], format='isot', scale='utc')
        except KeyError:
            try:
                tarr = Time(config[burst]['start'], format='isot', scale='utc') + TimeDelta(config[burst]['toa'], format='sec')
            except Exception as e:
                logger.error("Could not get arrival time for burst {} from either "
                             "tarr ot tsart + toa ({})".format(burst, e))
                raise
        # set arrival time
        config[burst]['tarr'] = tarr

        # check for excluded dishes
        try:
            # convert to integers, RT2 = dish 0
            config[burst]['excluded_dishes'] = [value - 2 for value in config[burst]['excluded_dishes']]
        except KeyError:
            logger.debug("No list of excluded dishes found")

        # check for telescope pointing
        pointing_coord = None
        try:
            pointing_ra = config['global']['pointing_ra']
            pointing_dec = config['global']['pointing_dec']
            pointing_coord = (pointing_ra * u.deg, pointing_dec * u.deg)
            logger.info("Telescope pointing found. Only use this option if ref_beam == 0.")
        except KeyError:
            logging.debug("No telescope pointing found")

        # Verify S/N limit is present if not globally set
        if snrmin is None:
            assert 'snrmin' in config[burst].keys(), 'No global nor local S/N threshold found'
        # else set local S/N limit to global value
        config[burst]['snrmin'] = snrmin

        # now check section for each compound beam
        beams = [key for key in config[burst].keys() if key.upper().startswith('CB')]
        # check reference CB is present and valid
        assert 'reference_cb' in config[burst].keys(), 'Reference CB missing'
        assert config[burst]['reference_cb'] in beams, 'Invalid reference CB: {}'.format(config[burst]['reference_cb'])
        # there should be at least one beam
        assert beams, 'No beams found for burst {}'.format(burst)
        for beam in beams:
            # read keys in lowercase
            keys = [key.lower() for key in config[burst][beam].keys()]

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

    return config


if __name__ == '__main__':
    print(parse_yaml('example.yaml'))
