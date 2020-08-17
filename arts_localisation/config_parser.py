#!/usr/bin/env python

import os
import logging

import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from arts_localisation.tools import cb_index_to_pointing, get_neighbours


logger = logging.getLogger(__name__)

REQUIRED_KEYS_SNR = ('snrmin', 'filterbank', 'cbs', 'neighbours', 'dm', 'window_load', 'window_zoom',
                     'main_cb', 'main_sb')
REQUIRED_KEYS_LOC = ('snrmin', 'ra', 'dec', 'size', 'resolution', 'cb_model', 'fmin_data', 'bandwidth')


def parse_yaml(fname, for_snr=False):
    """
    Parse a yaml file with settings for burst localisation
    :param str fname: Path to yaml config file
    :param bool for_snr: Only load settings related to S/N determination, skip everything else

    :return: config (dict), excluded_dishes (list, only if for_snr=False),
             source_coord (SkyCoord or None, only if for_snr=False)
    """

    # read the file
    with open(fname) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # store path to the yaml file
    yaml_dir = os.path.dirname(os.path.abspath(fname))

    # read global settings
    assert 'global' in config.keys(), 'Global config missing'

    # if for S/N determination, only load those settings
    if for_snr:
        for key in REQUIRED_KEYS_SNR:
            assert key in config['global'].keys(), f'Key missing: {key}'
        snr_config = {}
        for key in ('snrmin', 'dm', 'main_cb', 'main_sb'):
            snr_config[key] = config['global'][key]

        # check if cb and tab keys are present in path
        # first ensure two digits are used (which user may or may not have added)
        path = config['global']['filterbank'].replace('{cb}', '{cb:02d}').replace('{tab}', '{tab:02d}')
        assert '{cb:02d}' in path, '{cb} missing from path'
        assert '{tab:02d}' in path, '{tab} missing from path'
        # if a relative path, it is relative to the location of the yaml file
        if not os.path.isdir(path):
            path = os.path.join(yaml_dir, path)
        snr_config['filterbank'] = path

        # check CB list is not empty
        assert config['global']['cbs'], 'CB list cannot be empty'
        # if neighbours is True, add neighbouring beams
        if config['global']['neighbours']:
            snr_config['cbs'] = get_neighbours(config['global']['cbs'])
        else:
            snr_config['cbs'] = config['global']['cbs']

        return snr_config

    # localisation mode
    # check if required keys are present
    for key in REQUIRED_KEYS_LOC:
        assert key in config['global'].keys(), f'Key missing: {key}'

    # check if frequency range to use is given, else set such that full range is used
    if 'fmin' not in config['global'].keys():
        config['global']['fmin'] = 0
    if 'fmax' not in config['global'].keys():
        config['global']['fmax'] = np.inf

    # check for known source coordinates
    source_coord = None
    try:
        source_ra = config['global']['source_ra']
        source_dec = config['global']['source_dec']
        source_coord = (source_ra * u.deg, source_dec * u.deg)
    except KeyError:
        logger.debug('No source coordinates found')
    config['global']['source_coord'] = source_coord

    # the remaining config keys are individual bursts
    bursts = [key for key in config.keys() if not key == 'global']
    # there should be at least one burst
    assert bursts, 'No bursts found'
    # store list of bursts
    config['bursts'] = bursts
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

        # now check section for each compound beam
        beams = [key for key in config[burst].keys() if key.upper().startswith('CB')]
        # there should be at least one beam
        assert beams, f'No beams found for burst {burst}'
        # check reference CB is present and valid
        assert 'reference_cb' in config[burst].keys(), 'Reference CB missing'
        assert config[burst]['reference_cb'] in beams, 'Invalid reference CB: {}'.format(config[burst]['reference_cb'])
        # ensure reference CB is first in list
        beams.remove(config[burst]['reference_cb'])
        beams.insert(0, config[burst]['reference_cb'])
        # store list of beams
        config[burst]['beams'] = beams
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
                logger.warning(f"No SEFD given for {beam} of burst {burst}")

            # check for S/N array; absent means it is assumed to be an upper limit beam
            if 'snr_array' in keys:
                # if the file cannot be found, assume it is a relative path
                if not os.path.isfile(config[burst][beam]['snr_array']):
                    config[burst][beam]['snr_array'] = os.path.join(yaml_dir, config[burst][beam]['snr_array'])
            else:
                logger.info("No S/N array found for {} of burst {}, assuming it "
                            "is an upper limit beam".format(beam, burst))

    return config


def load_config(args, for_snr=False):
    """
    Load yaml config file and overwrite settings that are also given on command line

    :param argparse.Namespace args: Command line arguments
    :param bool for_snr: Only load settings related to S/N determination, skip everything else
    :return: config (dict)
    """
    config = parse_yaml(args.config, for_snr)
    # overwrite parameters also given on command line
    if for_snr:
        keys = REQUIRED_KEYS_SNR
    else:
        keys = REQUIRED_KEYS_LOC
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            logger.debug(f"Overwriting {key} from settings with command line value")
            if for_snr:
                config[key] = value
            else:
                config['global'][key] = value

    return config


if __name__ == '__main__':
    print(parse_yaml('example.yaml'))
    print(parse_yaml('example_snr.yaml', for_snr=True))
