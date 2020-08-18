#!/usr/bin/env python

import os
import sys
import logging

import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from arts_localisation.tools import cb_index_to_pointing, get_neighbours


logger = logging.getLogger(__name__)

REQUIRED_KEYS_GLOBAL = ('snrmin', 'fmin_data', 'bandwidth')

REQUIRED_KEYS_SNR = ('filterbank', 'cbs', 'neighbours', 'dm', 'window_load', 'window_zoom',
                     'main_cb', 'main_sb')
REQUIRED_KEYS_LOC = ('ra', 'dec', 'size', 'resolution', 'cb_model')


def parse_yaml(fname, for_snr=False):
    """
    Parse a yaml file with settings for burst localisation

    :param str fname: Path to yaml config file
    :param bool for_snr: Load settings for S/N determination, else load localisation settings

    :return: config (dict)
    """

    # read the config file
    with open(fname) as f:
        raw_config = yaml.load(f, Loader=yaml.SafeLoader)

    # store path to the config file
    yaml_dir = os.path.dirname(os.path.abspath(fname))

    # read global settings
    assert 'global' in raw_config.keys(), 'Global config missing'
    conf_global = raw_config['global']

    # verify all required keys are present
    for key in REQUIRED_KEYS_GLOBAL:
        assert key in conf_global.keys(), f'Global section key missing: {key}'

    # set fmin/fmax if not specified in config
    conf_global['fmin'] = conf_global.get('fmin', 0)
    conf_global['fmax'] = conf_global.get('fmax', np.inf)

    # copy the global config to the output config
    config = conf_global.copy()

    # if for S/N determination, only load those settings
    if for_snr:
        assert 'snr' in raw_config.keys(), 'S/N config missing'
        conf_snr = raw_config['snr']
        # verify all required S/N keys are present
        for key in REQUIRED_KEYS_SNR:
            assert key in conf_snr.keys(), f'S/N section key missing: {key}'

        # check if cb and tab keys are present in filterbank path
        # first ensure two digits are used (which user may or may not have added)
        path = conf_snr['filterbank'].replace('{cb}', '{cb:02d}').replace('{tab}', '{tab:02d}')
        assert '{cb:02d}' in path, '{cb} missing from path'
        assert '{tab:02d}' in path, '{tab} missing from path'
        # if a relative path, it is relative to the location of the yaml file
        if not os.path.isfile(path.format(cb=conf_snr['main_cb'], tab=0)):
            path = os.path.join(yaml_dir, path)
        conf_snr['filterbank'] = path

        # check CB list is not empty
        assert conf_snr['cbs'], 'CB list cannot be empty'
        # if neighbours is True, add neighbouring beams
        if conf_snr['neighbours']:
            conf_snr['cbs'] = get_neighbours(conf_snr['cbs'])

        # add the S/N settings to the global settings
        config.update(conf_snr)
        # Done for S/N mode
        return config

    # localisation mode
    assert 'localisation' in raw_config.keys(), 'Localisation config missing'
    conf_loc = raw_config['localisation']
    # check if required keys are present
    for key in REQUIRED_KEYS_LOC:
        assert key in conf_loc.keys(), f'Localisation section key missing: {key}'

    # check for known source coordinates
    source_coord = None
    try:
        source_ra = conf_loc['source_ra']
        source_dec = conf_loc['source_dec']
        source_coord = (source_ra * u.deg, source_dec * u.deg)
    except KeyError:
        logger.debug('No source coordinates found')
    conf_loc['source_coord'] = source_coord

    # copy the localisation settings to the global config
    config.update(conf_loc)

    # the remaining config keys are individual bursts
    bursts = [key for key in raw_config.keys() if key not in ['global', 'snr', 'localisation']]
    # there should be at least one burst
    assert bursts, 'No bursts found'
    # store list of bursts
    config['bursts'] = bursts
    for burst in bursts:
        conf_burst = raw_config[burst]
        # parameters of the burst
        # try to read arrival time, either directly or as start time plus ToA
        try:
            tarr = Time(conf_burst['tarr'], format='isot', scale='utc')
        except KeyError:
            try:
                tarr = Time(conf_burst['tstart'], format='isot', scale='utc') + \
                    TimeDelta(conf_burst['toa'], format='sec')
            except Exception as e:
                logger.error("Could not get arrival time for burst {} from either "
                             "tarr ot tsart + toa ({})".format(burst, e))
                raise
        # set arrival time
        conf_burst['tarr'] = tarr

        # check for excluded dishes
        try:
            # convert to integers, RT2 = dish 0
            conf_burst['excluded_dishes'] = [value - 2 for value in conf_burst['excluded_dishes']]
        except KeyError:
            logger.debug("No list of excluded dishes found")

        # check for telescope pointing
        pointing_coord = None
        try:
            pointing_ra = conf_burst['pointing_ra']
            pointing_dec = conf_burst['pointing_dec']
            pointing_coord = (pointing_ra * u.deg, pointing_dec * u.deg)
            logger.info("Telescope pointing found. Only use this option if ref_beam == 0.")
        except KeyError:
            logging.debug("No telescope pointing found")

        # now check section for each compound beam
        beams = [key for key in conf_burst.keys() if key.upper().startswith('CB')]
        # there should be at least one beam
        assert beams, f'No beams found for burst {burst}'
        # check reference CB is present and valid
        assert 'reference_cb' in conf_burst.keys(), 'Reference CB missing'
        assert conf_burst['reference_cb'] in beams, 'Invalid reference CB: {}'.format(conf_burst['reference_cb'])
        # ensure reference CB is first in list
        beams.remove(conf_burst['reference_cb'])
        beams.insert(0, conf_burst['reference_cb'])
        # store list of beams
        conf_burst['beams'] = beams
        for beam in beams:
            # read keys in lowercase
            keys = [key.lower() for key in conf_burst[beam].keys()]

            # if ra, dec are given, use these
            if 'ra' in keys and 'dec' in keys:
                cb_pointing = (conf_burst[beam]['ra'] * u.deg, conf_burst[beam]['dec'] * u.deg)
                # if pointing is also set, warn the user
                if pointing_coord is not None:
                    logger.warning("CB RA, Dec given, but telescope pointing is also set. Using CB RA, Dec")
            else:
                # telescope pointing must be set
                if pointing_coord is None:
                    logger.error("No telescope pointing and no RA, Dec set for this CB")
                    sys.exit(1)
                # calculate ra, dec from CB offsets
                cb_index = int(beam[2:])
                cb_pointing = cb_index_to_pointing(cb_index, *pointing_coord)
            # store the pointing of this CB
            conf_burst[beam]['pointing'] = cb_pointing

            # SEFD is optional, but give user a warning if it is not present as it is easy to overlook
            if 'sefd' not in keys:
                logger.warning(f"No SEFD given for {beam} of burst {burst}")

            # check for S/N array; absent means it is assumed to be an upper limit beam
            if 'snr_array' in keys:
                snr_array_path = conf_burst[beam]['snr_array']
                # if the file cannot be found, assume it is a relative path
                if not os.path.isfile(snr_array_path):
                    logger.debug('S/N array not found, trying to convert relative to absolute path')
                    snr_array_path = os.path.join(yaml_dir, snr_array_path)
                # if the file still cannot be found, give a warning
                if not os.path.isfile(snr_array_path):
                    logger.warning(f'Cannot find S/N array file {snr_array_path}')
                conf_burst[beam]['snr_array'] = snr_array_path
            else:
                logger.info("No S/N array found for {} of burst {}, assuming it "
                            "is an upper limit beam".format(beam, burst))
        # copy the burst config to the global config
        config[burst] = conf_burst

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
