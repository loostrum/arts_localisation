#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import corner
import schwimmbad
from mpi4py import MPI
import cProfile

from arts_localisation import tools
from arts_localisation.constants import REF_FREQ, CB_HPBW, NSB
from arts_localisation.beam_models import SBPatternSingle
from arts_localisation.config_parser import load_config


# avoid using parallelization other than the MPI/multiprocessing processes used in this script
os.environ["OMP_NUM_THREADS"] = "1"
# silence the matplotlib logger
logging.getLogger('matplotlib').setLevel(logging.ERROR)
plt.rcParams['axes.formatter.useoffset'] = False


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator


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


# @profile(filename='log_prior.prof')
def log_prior(params, numcb_per_burst):
    ra = params[0]
    dec = params[1]
    # ra and dec must be in valid range
    if not ((0 < ra < 2 * np.pi) and (-np.pi / 2 < dec < np.pi / 2)):
        return -np.inf

    # boresight S/N (per burst) must be positive
    nburst = len(numcb_per_burst)
    boresight_snr = params[2:2 + nburst]
    # if not np.all(boresight_snr > 0):
    if not np.all(0 < boresight_snr) and np.all(boresight_snr < 200):
        return -np.inf

    # primary beam widths must be positive (per CB per burst)
    # index of first burst-specific value
    offset = 2 + nburst
    burst_offset = 0
    ref_beam_width_rad = 33.7 / 60. * np.pi / 180.
    for burst_ind, ncb in enumerate(numcb_per_burst):
        start = offset + burst_offset
        pbeam_width_ra = params[start:start + ncb]
        pbeam_width_dec = params[start + ncb:start + 2 * ncb]
        snr_offset = params[start + 2 * ncb:start + 3 * ncb]
        burst_offset += 3 * ncb

        if (pbeam_width_ra < 0) or (pbeam_width_dec < 0):
            return -np.inf

        if not (0 < snr_offset < 10):
            return -np.inf

        if not (.9 * ref_beam_width_rad < pbeam_width_ra < 1.1 * ref_beam_width_rad):
            return -np.inf

        if not (.9 * ref_beam_width_rad < pbeam_width_dec < 1.1 * ref_beam_width_rad):
            return -np.inf

    # everything passed
    return 0


# @profile(filename='log_likelihood.prof')
def log_likelihood(params, snr_data, numcb_per_burst, tarr_per_burst):
    ra0 = params[0]
    dec0 = params[1]
    nburst = len(numcb_per_burst)
    boresight_snr = params[2:2 + nburst]

    # init likelihood
    logL = 0

    # loop over bursts
    offset = 2 + nburst
    burst_offset = 0
    for burst_ind, ncb in enumerate(numcb_per_burst):
        # get HA, Dec at the arrival time of this burst
        ha, dec = tools.radec_to_hadec(ra0 * u.rad, dec0 * u.rad, tarr_per_burst[burst_ind])
        ha = ha.to(u.rad).value
        dec = dec.to(u.rad).value

        # get boresight S/N for this burst
        boresight_snr_burst = boresight_snr[burst_ind]

        start = offset + burst_offset
        # loop over beams
        for cb_ind in range(ncb):
            pbeam_width_ra = params[start + cb_ind]
            pbeam_width_dec = params[start + ncb + cb_ind]
            snr_offset = params[start + 2 * ncb + cb_ind]

            # get the model of this CB
            model = models[burst_ind][cb_ind]

            # get ha, dec offset from CB centre
            dhacosdec = (ha - model.ha0) * np.cos(dec)
            ddec = dec - model.dec0

            # generate SBs
            sb_model = model.get_sb_model(dhacosdec, ddec, pbeam_width_ra, pbeam_width_dec)

            # get S/N array of this CB
            snrs = snr_data[burst_ind][cb_ind]

            # calculate likelihood
            logL += np.sum(((sb_model * (boresight_snr_burst - snr_offset) + snr_offset) - snrs) ** 2, axis=0)

            # check if logL is no longer finite, in which case we can just return here
            if not np.isfinite(logL):
                return -np.inf

        # bookkeeping
        burst_offset += 3 * ncb
    return logL


def log_posterior(params, snr_data, numcb_per_burst, tarr_per_burst):
    lp = log_prior(params, numcb_per_burst)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(params, snr_data, numcb_per_burst, tarr_per_burst)


def initialize_parameters(config, ndim, snr_data):
    """
    Create an array of initial guesses for each parameter:

    #. RA (rad)
    #. Dec (rad)
    #. Boresight S/N (per burst)
    #. Primary beam width HA (rad, per CB of each burst)
    #. Primary beam width Dec (rad, per CB of each burst)
    #. S/N offset (per CB of each burst)

    :param dict config: config from .yaml file
    :param int ndim: number of guesses to use for each parameter
    :param np.array snr_data: Max S/N for each burst
    :return: guesses (array)
    """

    # global parameters: RA and Dec
    # get arrival time of reference burst
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

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--ncore', type=int, default=1, help='Number of processes to use. If >1,'
                                                                'multiprocessing is used')
        group.add_argument('--mpi', action='store_true', help='Use MPI')
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

    # when MPI is enabled, there should be at least 2 processes
    if args.mpi and mpi_size == 1:
        logging.error(f"When using MPI, run at least two processes with e.g. "
                      f"'mpiexec -np 2 {os.path.basename(__file__)}'")
        sys.exit()
    # no MPI means there should be just one process at this point
    elif not args.mpi and mpi_size > 1:
        if mpi_rank == 0:
            logging.error("When using MPI, specify the --mpi flag")
        sys.exit()

    if args.load is None and (args.nstep is None or args.nwalker is None):
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
    initial_guess = None
    if mpi_rank == 0:
        initial_guess = initialize_parameters(config, args.nwalker, max_snr_per_burst)
    initial_guess = mpi_comm.bcast(initial_guess, root=0)
    ndim = initial_guess.shape[1]

    # Initialize SB model for each CB of each burst
    # also store arrival times
    # each process needs to access these models, make global instead of passing them around to save time
    global models
    models = []
    tarr_per_burst = []
    for burst in config['bursts']:
        tarr_per_burst.append(config[burst]['tarr'])
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

    if args.load is not None:
        # extra MPI processes are not needed when loading previous run
        if mpi_rank > 0:
            sys.exit()
        raise NotImplementedError("Loading of previous run not yet implemented")

    # setup the output file
    mcmc_file = output_prefix + f'_mcmc_{args.nwalker}walker_{args.nstep}step.h5'
    backend = emcee.backends.HDFBackend(mcmc_file)
    if mpi_rank == 0:
        try:
            os.remove(mcmc_file)
        except FileNotFoundError:
            pass
        backend.reset(args.nwalker, ndim)

    # ensure all processes have initialized models etc. before starting the pool
    mpi_comm.Barrier()

    # select the pool based on user arguments: serial, multiprocessing, or mpi
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.ncore)
    # run the worker pool
    with pool:
        # initialize MCMC sampler
        sampler = emcee.EnsembleSampler(args.nwalker, ndim, log_posterior,
                                        args=(snr_data, numcb_per_burst, tarr_per_burst),
                                        pool=pool, backend=backend)
        # run!
        sampler.run_mcmc(initial_guess, args.nstep, progress=True)

    # when using MPI, multiple processes are no longer needed here
    if mpi_rank > 0:
        sys.exit()

    # extract sample
    nburn = int(.5 * args.nstep)
    sample = sampler.get_chain(discard=nburn, flat=True)
    # convert HA/Dec to deg
    sample[:, 0] *= 180 / np.pi
    sample[:, 1] *= 180 / np.pi

    # create corner plot
    # truths = [config['source_ra'], config['source_dec']]
    # labels = ['RA', 'Dec']
    truths = [config['source_ra'], config['source_dec']]
    labels = ['RA', 'Dec']
    for burst_ind, burst in enumerate(config['bursts']):
        labels.append(f'Boresight S/N burst {burst}')
        truths.append(np.amax(snr_data[burst_ind]))
    for burst in config['bursts']:
        for beam in config[burst]['beams']:
            labels.append(f'CB width RA {burst} {beam}')
            labels.append(f'CB width Dec {burst} {beam}')
            labels.append(f'S/N offset {burst} {beam}')
            truths.append(33.7 / 60 * np.pi / 180.)
            truths.append(33.7 / 60 * np.pi / 180.)
            truths.append(3.5)

    fig = corner.corner(sample, labels=labels, truths=truths)

    # plot burn-in (not available in saved data)
    if not args.load:
        fig, ax = plt.subplots()
        for i in range(len(sampler.lnprobability)):
            ax.plot(sampler.lnprobability[i, :], linewidth=0.3, color='k', alpha=0.4)
        ax.set_ylabel('ln(P)')
        ax.set_xlabel('Step number')
        ax.axvline(nburn, linestyle='dotted')

    plt.show()


if __name__ == '__main__':
    main()
