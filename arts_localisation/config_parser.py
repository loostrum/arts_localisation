#!/usr/bin/env python

import os
import logging

import yaml
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from arts_localisation.tools import cb_index_to_pointing, get_neighbours, hadec_to_radec


logger = logging.getLogger(__name__)

REQUIRED_KEYS_GLOBAL = ('snrmin', 'fmin_data', 'bandwidth')

REQUIRED_KEYS_SNR = ('dm', 'window_load', 'window_zoom', 'width_max')
REQUIRED_KEYS_SNR_BURST = ('main_sb', 'main_cb', 'filterbank', 'cbs', 'neighbours')
REQUIRED_KEYS_LOC = ('dec', 'size', 'resolution', 'cb_model')


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
    logger.debug('Loading global config')
    assert 'global' in raw_config.keys(), 'Global config missing'
    conf_global = raw_config['global']

    # verify all required keys are present
    for key in REQUIRED_KEYS_GLOBAL:
        assert key in conf_global.keys(), f'Global section key missing: {key}'

    # copy the global config to the output config
    config = conf_global.copy()

    # S/N or localisation specific settings
    if for_snr:
        # S/N mode
        logger.debug('Loading S/N config')
        assert 'snr' in raw_config.keys(), 'S/N config missing'
        conf_snr = raw_config['snr']
        # verify all required S/N keys are present
        for key in REQUIRED_KEYS_SNR:
            assert key in conf_snr.keys(), f'S/N section key missing: {key}'
        # add the S/N settings to the global settings
        config.update(conf_snr)
    else:
        logger.debug('Loading localisation config')
        # localisation mode
        assert 'localisation' in raw_config.keys(), 'Localisation config missing'
        conf_loc = raw_config['localisation']
        # check if required keys are present
        for key in REQUIRED_KEYS_LOC:
            assert key in conf_loc.keys(), f'Localisation section key missing: {key}'
        # check if ha or ra is present
        assert 'ha' in conf_loc.keys() or 'ra' in conf_loc.keys(), f'Localisation key missing: ra or ha'
        # if ha is given, convert to J2000 ra, dec
        if 'ha' in conf_loc.keys():
            logger.info("Detected HADEC reference frame, converting to RADEC")
            # time must be set as well to convert to RA, Dec
            assert 'time' in conf_loc.keys(), 'Localisation key missing in HADEC mode: time'
            radec = hadec_to_radec(conf_loc['ha'] * u.deg, conf_loc['dec'] * u.deg, conf_loc['time'])
            # replace the HA,Dec by RA,Dec
            del conf_loc['ha']
            conf_loc['ra'] = radec.ra.deg
            conf_loc['dec'] = radec.dec.deg

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
    logger.debug(f'Found bursts: {bursts}')
    # there should be at least one burst
    assert bursts, 'No bursts found'
    # store list of bursts
    config['bursts'] = bursts
    for burst in bursts:
        # burst could be something else than string, which may give rise to unexpected behaviour
        if not isinstance(burst, str):
            logger.warning(f'{burst} is not interpreted as string, this may give rise to unexpected behaviour. '
                           f'Put the value between quotes to force it being read as string.')
        conf_burst = raw_config[burst]

        # if fmin / fmax are given here, overwrite any global setting
        # if neither global nor local are set, use the defaults (full band)
        freq_defaults = {'fmin': 0, 'fmax': np.inf}
        for key in ('fmin', 'fmax'):
            # if not present, use global value
            if key not in conf_burst.keys():
                # if global value not present use default
                conf_burst[key] = config.get(key, freq_defaults[key])
            else:
                # local values are present, notify that value will be overwritten if global value is present
                if key in config.keys():
                    logger.info(f'Overwriting global value for {key} with local value for burst {burst}')
            # delete the key from the global settings to avoid confusion
            try:
                del config[key]
            except KeyError:
                pass

        if for_snr:
            # S/N mode
            # check if the required keys are present
            for key in REQUIRED_KEYS_SNR_BURST:
                assert key in conf_burst.keys(), f'Burst section key missing for S/N: {key}'

            # check if cb and tab keys are present in filterbank path
            # first ensure two digits are used (which user may or may not have added)
            path = conf_burst['filterbank'].replace('{cb}', '{cb:02d}').replace('{tab}', '{tab:02d}')
            assert '{cb:02d}' in path, '{cb} missing from path'
            assert '{tab:02d}' in path, '{tab} missing from path'
            # if a relative path, it is relative to the location of the yaml file
            if not os.path.isfile(path.format(cb=conf_burst['main_cb'], tab=0)):
                path = os.path.join(yaml_dir, path)
            conf_burst['filterbank'] = path

            # check CB list is not empty
            assert conf_burst['cbs'], 'CB list cannot be empty'
            # if neighbours is True, add neighbouring beams
            if conf_burst['neighbours']:
                conf_burst['cbs'].extend(get_neighbours(conf_burst['cbs']))

        else:
            # localisation mode
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
                pointing_dec = conf_burst['pointing_dec']
                try:
                    pointing_ra = conf_burst['pointing_ra']
                    pointing_coord = (pointing_ra * u.deg, pointing_dec * u.deg)
                except KeyError:
                    # Assume hour angle was given instead of right ascension
                    pointing_ha = conf_burst['pointing_ha']
                    # convert to RADEC
                    pointing = hadec_to_radec(pointing_ha * u.deg, pointing_dec * u.deg, tarr)
                    pointing_coord = (pointing.ra, pointing.dec)

                logger.info("Telescope pointing found. Only use this option if ref_beam == 0.")
            except KeyError:
                logging.debug("No telescope pointing found")

            # now check section for each compound beam (only need for localisation mode)
            beams = [key for key in conf_burst.keys() if key.upper().startswith('CB') and key != 'cbs']
            # there should be at least one beam
            assert beams, f'No beams found for burst {burst}'
            logger.debug(f'Found beams: {beams}')
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
                try:
                    keys = [key.lower() for key in conf_burst[beam].keys()]
                except AttributeError:
                    # upper limit beam and all default parameters means there are no parameters left at all
                    logger.debug(f'No parameters found for CB{beam}')
                    conf_burst[beam] = {}
                    keys = []

                # if ra, dec are given, use these
                if 'dec' in keys:
                    try:
                        # Try RADEC
                        cb_pointing = (conf_burst[beam]['ra'] * u.deg, conf_burst[beam]['dec'] * u.deg)
                    except KeyError:
                        # Try HADEC
                        pointing = hadec_to_radec(conf_burst[beam]['ha'] * u.deg, conf_burst[beam]['dec'] * u.deg,
                                                  t=tarr)
                        cb_pointing = (pointing.ra, pointing.dec)
                    # if pointing is also set, warn the user
                    if pointing_coord is not None:
                        logger.warning(f"CB{beam:02d} RA, Dec given, "
                                       f"but telescope pointing is also set. Using CB RA, Dec")
                else:
                    # telescope pointing must be set
                    assert pointing_coord is not None, f'No telescope pointing and no RA, Dec set for CB{beam:02d}'
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
        keys = REQUIRED_KEYS_SNR + REQUIRED_KEYS_GLOBAL
    else:
        keys = REQUIRED_KEYS_LOC + REQUIRED_KEYS_GLOBAL
    for key in keys:
        try:
            value = getattr(args, key)
        except AttributeError:
            value = None
        if value is not None:
            logger.debug(f"Overwriting {key} from settings with command line value")
            print(f"Overwriting {key} from settings with command line value")
            config[key] = value

    return config


if __name__ == '__main__':
    print(parse_yaml('example.yaml'))
    print(parse_yaml('example_snr.yaml', for_snr=True))
