#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
from astropy.time import Time
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import corner
from schwimmbad import MPIPool
from mpi4py import MPI

from arts_localisation import tools
from arts_localisation.constants import NSB
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


def set_guess_value(ndim, minval, maxval):
    """
    Generate a set of guess parameters

    :param int ndim: dimensions of guess vector
    :param float minval: minimum value
    :param float maxval: maximum value
    """
    return (maxval - minval) * np.random.random(ndim) + minval


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

    # set matplotlib backend to non-interactive if only saving plots
    if args.save_plots and not args.show_plots:
        plt.switch_backend('pdf')

    # load config
    config = load_config(args, mcmc=True)


if __name__ == '__main__':
    main()
