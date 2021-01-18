#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
import emcee
import matplotlib.pyplot as plt
import corner
from schwimmbad import MPIPool
from mpi4py import MPI

from arts_localisation import tools
from arts_localisation.constants import REF_FREQ, CB_HPBW, NSB
from arts_localisation.beam_models import SBPatternSingle
from arts_localisation.config_parser import load_config


# each process needs to access these models, make global instead of passing them around to save time
global models
# avoid using parallelization other than the MPI processes used in this script
os.environ["OMP_NUM_THREADS"] = "1"
plt.rcParams['axes.formatter.useoffset'] = False


class ArgumentParserExecption(Exception):
    pass


class ArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser with adapted error method so it raise an exception instead of exiting on an error
    """
    def error(self, message):
        """error(message: string)
        Prints a usage message incorporating the message to stderr and
        exits.
        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_usage(sys.stderr)
        raise ArgumentParserExecption(f'{self.prog}: error: {message}')


class TestData:
    def __init__(self, snrmin):
        self.source_ra = 29.50333 * u.deg
        self.source_dec = 65.71675 * u.deg
        self.src = "R3_20200511_3610"
        self.burst = {'tstart': Time('2020-05-11T07:36:22.0'),
                      'toa': 3610.84 * u.s,
                      'ra_cb': 29.50333 * u.deg,
                      'dec_cb': 65.71675 * u.deg,
                      # 'snr_array': '/home/oostrum/localisation/mcmc/snr_R3/{self.src}_CB00_SNR.txt'
                      'snr_array': f'/data/arts/localisation/R3/snr/{self.src}_CB00_SNR.txt'
                      }
        # load S/N array
        data = np.loadtxt(self.burst['snr_array'], ndmin=2)
        sbs, snrs = data.T
        ind_det = snrs >= snrmin
        self.sb_det = sbs[ind_det].astype(int)
        self.sb_nondet = sbs[~ind_det].astype(int)
        self.snr_det = snrs[ind_det]

        # calculate ha, dec at burst ToA
        self.tarr = self.burst['tstart'] + self.burst['toa']
        self.ha_cb, self.dec_cb = tools.radec_to_hadec(self.burst['ra_cb'], self.burst['dec_cb'], self.tarr)


def initialize_parameters(config, ndim, snr_data):
    """
    Create an array of initial guesses for each parameter:

    #. RA (rad)
    #. Dec (rad)
    #. Boresight S/N (per burst)
    #. Primary beam width RA (rad, per CB of each burst)
    #. Primary beam width Dec (rad, per CB of each burst)
    #. S/N offset (per CB of each burst)

    :param dict config: config from .yaml file
    :param int ndim: number of guesses to use for each parameter
    :param np.array snr_data: Max S/N for each burst
    :return: guesses (array)
    """
    # global parameters: ra and dec
    ra0 = (config['ra'] * u.deg).to(u.rad).value
    dec0 = (config['dec'] * u.deg).to(u.rad).value

    max_offset = (config['guess_pointing_max_offset'] * u.arcmin).to(u.rad).value
    guess_dec = dec0 + get_guess_values(ndim, -max_offset, max_offset)
    guess_ra = ra0 + get_guess_values(ndim, -max_offset, max_offset) / np.cos(guess_dec)

    # parameters per burst: boresight S/N
    nburst = len(config['bursts'])
    minval, maxval = np.array(config['guess_boresight_snr_range'])
    guess_boresight_snr = get_guess_values((ndim, nburst), minval, maxval) * snr_data[None, :]

    # parameters per CB per burst
    # find out how many parameters to generate
    num_guess = 0
    for burst in config['bursts']:
        nbeam = len(config[burst]['beams'])
        num_guess += nbeam

    # beam width in RA and Dec
    central_freq = int(np.round(config['fmin_data'] + config['bandwidth'] / 2)) * u.MHz
    beam_width0 = (CB_HPBW * REF_FREQ / central_freq).to(u.rad).value
    max_offset = (config['guess_beamwidth_max_offset'] * u.arcmin).to(u.rad).value
    guess_width_ra = beam_width0 + get_guess_values((ndim, num_guess), -max_offset, max_offset)
    guess_width_dec = beam_width0 + get_guess_values((ndim, num_guess), -max_offset, max_offset)

    # S/N offset
    guess_snr_offset = get_guess_values((ndim, num_guess), 0, config['guess_snr_offset_max_offset'])

    # output shape must be (nwalker, nparam)
    guesses = np.hstack([guess_ra[:, None], guess_dec[:, None], guess_boresight_snr,
                         guess_width_ra, guess_width_dec, guess_snr_offset])

    return guesses


def get_guess_values(ndim, minval, maxval):
    """
    Generate a set of guess parameters

    :param tuple,int ndim: dimensions of guess vector
    :param float minval: minimum value
    :param float maxval: maximum value
    """
    return (maxval - minval) * np.random.random(ndim) + minval


def load_snr_data(config):
    """
    Load S/N arrays from disk

    :param dict config: config from .yaml file
    :return: S/N array for each CB of each burst (array), number of CBs per burst (list)
    """
    numcb_per_burst = []
    data = []
    for burst in config['bursts']:
        nbeam = len(config[burst]['beams'])
        burst_data = np.zeros((nbeam, NSB), dtype=float)
        numcb_per_burst.append(nbeam)
        for ind, beam in enumerate(config[burst]['beams']):
            # load the S/N array
            fname = config[burst][beam]['snr_array']
            snr_data = np.loadtxt(fname, ndmin=2)
            sb_det, snr_det = snr_data.T
            if len(snr_det) != NSB:
                logging.error(f"Number of S/N values in {fname} does not equal number of SBs ({NSB}), "
                              f"please re-run arts_calc_snr to get a value in each SB")
            burst_data[ind] = snr_det
        data.append(burst_data)
    return data, numcb_per_burst


def main():
    # get MPI rank
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    # parse arguments
    args = None
    if mpi_rank == 0:
        parser = ArgumentParser()
        parser.add_argument('--config', required=True, help='Input yaml config')
        parser.add_argument('--nwalker', type=int, help='Number of MCMC walkers')
        parser.add_argument('--nstep', type=int, help='Number of MCMC steps')
        parser.add_argument('--load', help='Path to .h5 file from previous run. Will load this file instead of'
                                           'executing another MCMC run')
        parser.add_argument('--output_folder', help='Output folder '
                                                    '(Default: <yaml file folder>/localisation)')
        parser.add_argument('--show_plots', action='store_true', help='Show plots')
        parser.add_argument('--save_plots', action='store_true', help='Save plots')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
        parser.add_argument('--store', action='store_true', help='Store output of MCMC run')
        try:
            args = parser.parse_args()
            # print help if no arguments are given, and make the programme exit
            if len(sys.argv) == 1:
                parser.print_help()
                args = None
        except ArgumentParserExecption as e:
            # args will remain none, which makes the program exit. No need to further process exception here
            print(e)

    # broadcast arguments, exit if parsing failed (args=None)
    args = mpi_comm.bcast(args, root=0)
    if args is None:
        sys.exit()

    # set up logger
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel, stream=sys.stderr)
    # if loading a previous run, there is no need for extra MPI processes
    if args.load is not None and mpi_size > 1:
        if mpi_rank == 0:
            logging.warning("No need for MPI when loading run; disabling extra processes")
        else:
            sys.exit()
    # if doing a run, we need more than one process for the MPI pool to work
    elif args.load is None and mpi_size == 1:
        logging.error(f"Need more than one process to be able to execute MCMC run. Run this "
                      f"script with mpirun -np <nproc> {os.path.basename(__file__)}")
        sys.exit()
    elif args.load is None and (args.nstep is None or args.nwalker is None):
        if mpi_rank == 0:
            logging.error("Specify nwalker and nstep when doing an MCMC run")
            sys.exit()
        else:
            sys.exit()

    # set matplotlib backend to non-interactive if only saving plots
    if args.save_plots and not args.show_plots:
        plt.switch_backend('pdf')

    # get output prefix from yaml file name and output folder
    if args.output_folder is None:
        # default output folder is same folder as .yaml file plus "localisation"
        args.output_folder = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'localisation')
    if mpi_rank == 0:
        tools.makedirs(args.output_folder)
    # output prefix also contains the yaml filename without extension
    output_prefix = os.path.join(args.output_folder, os.path.basename(args.config).replace('.yaml', ''))

    # load config
    config = None
    if mpi_rank == 0:
        config = load_config(args, mcmc=True)
    config = mpi_comm.bcast(config, root=0)

    # load S/N arrays
    snr_data = None
    if mpi_rank == 0:
        snr_data = load_snr_data(config)
    snr_data, numcb_per_burst = mpi_comm.bcast(snr_data, root=0)

    # create initial parameter vector
    # get max S/N of each burst
    max_snr_per_burst = np.array([burst.max() for burst in snr_data])
    initial_guess = initialize_parameters(config, args.nwalker, max_snr_per_burst)

    # Initialize SB model for each CB of each burst
    models = []
    for burst in config['bursts']:
        burst_models = []
        for beam in config[burst]['beams']:
            # get pointing of CB in HA/Dec
            radec = config[burst][beam]['pointing']
            ha_cb, dec_cb = tools.radec_to_hadec(*radec, config[burst]['tarr'])
            # create model
            model = SBPatternSingle(ha_cb.to(u.rad).value, dec_cb.to(u.rad).value,
                                    fmin=config[burst]['fmin'],
                                    fmax=config[burst]['fmax'],
                                    min_freq=config['fmin_data'],
                                    nfreq=32,
                                    cb_model=config['cb_model'],
                                    cbnum=int(beam[-2:]))
            burst_models.append(model)
        models.append(burst_models)

    import IPython; IPython.embed()


if __name__ == '__main__':
    main()
