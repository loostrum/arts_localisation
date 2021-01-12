#!/usr/bin/env python3
#
# Test bayesian method for localisation

import os
import numpy as np
from astropy.time import Time
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
from astroML.plotting import plot_mcmc
from multiprocessing import Pool

from arts_localisation import tools
from arts_localisation.constants import NSB
from arts_localisation.beam_models.simulate_sb_pattern import SBPattern
from arts_localisation.beam_models import SBModelBayes


global model


def log_prior(params, ha_cb, dec_cb, maxsnr):
    """
    """
    # ha and dec must be within valid range
    # dec = dec_cb + params[1] * u.arcmin
    # ha = ha_cb + params[0] * u.arcmin / np.cos(dec)
    # if (-180 * u.deg < ha < 180 * u.deg) and (-90 * u.deg < dec < 90 * u.deg):
    #     return 0
    # else:
    #     return -np.inf

    # # take lazy route: offset in both dec and hacosdec must be smaller than some value
    max_dist = 5 * 60  # arcmin
    # # also set max on boresight S/N
    if abs(params[0]) > max_dist or abs(params[1]) > max_dist: # or params[2] < 0:  # or params[2] > maxsnr
        return -np.inf
    else:
        return 0


def log_likelihood(params, sbs, nondet_sbs, snrs):
    """
    :param list params: dhacosdec, ddec, boresight snr, S/N for each non-det SB
    :param array sbs: array of detection SBs
    :param array nondet_sbs: array of non-detection SBs
    :param array snrs: S/N in each detection SB
    """
    # read parameters
    dhacosdec, ddec, boresight_snr = params[:3]
    # convert to radians
    dhacosdec *= np.pi / 180 / 60.
    ddec *= np.pi / 180 / 60.
    if len(params) > 3:
        snr_sb_nondet = params[3:]
        # construct S/N for each SB
        snr_all = np.zeros(NSB)
        snr_all[sbs] = snrs
        snr_all[nondet_sbs] = params[3:]
    else:
        snr_all = snrs
    assert len(snr_all) == NSB

    # create SB model
    sb_model = model.get_sb_model(dhacosdec, ddec)

    # construct likelihood: squared sum of modelled vs measured S/N over all SBs
    logL = np.sum(((sb_model * boresight_snr) - snr_all) ** 2, axis=0)
    return -logL


def log_posterior(params, sbs, nondet_sbs, snrs, ha_cb, dec_cb, maxsnr):
    return log_prior(params, ha_cb, dec_cb, maxsnr) + log_likelihood(params, sbs, nondet_sbs, snrs)


if __name__ == '__main__':
    # set up parameters
    source_ra = 29.50333 * u.deg
    source_dec = 65.71675 * u.deg

    snr_min = 8

    burst = {'tstart': Time('2020-05-11T07:36:22.0'),
             'toa': 3610.84 * u.s,
             'ra_cb': 29.50333 * u.deg,
             'dec_cb': 65.71675 * u.deg,
             'snr_array': '/data/arts/localisation/R3/snr/R3_20200511_3610_CB00_SNR.txt'
             }

    # load S/N array
    data = np.loadtxt(burst['snr_array'], ndmin=2)
    sb_det, snr_det = data.T
    sb_det = sb_det.astype(int)

    sb_nondet = [sb for sb in range(NSB) if sb not in sb_det]
    num_sb_nondet = len(sb_nondet)

    print(f"Found {num_sb_nondet} non-detection SBs")

    # calculate ha, dec at burst ToA
    tarr = burst['tstart'] + burst['toa']
    ha_cb, dec_cb = tools.radec_to_hadec(burst['ra_cb'], burst['dec_cb'], tarr)

    # guess parameters, order is ha, dec, boresight snr, snr for each nondet beam
    ndim = 3 + num_sb_nondet
    nwalkers = 256
    nsteps = 2048
    nburn = 256

    # boresight S/N random between 1 and 5 x measured max S/N
    minval = 1 * snr_det.max()
    maxval = 5 * snr_det.max()
    guess_snr = (maxval - minval) * np.random.random(nwalkers) + minval
    # boresight S/N is not allowed to be larger than this value (if this check is enabled in log_prior)
    maxsnr = 10 * snr_det.max()

    # random HA / Dec within this many arcmin of initial point
    max_offset = 1  # arcmin

    guess_ddec = 2 * max_offset * np.random.random(nwalkers) - max_offset
    guess_dhacosdec = 2 * max_offset * np.random.random(nwalkers) - max_offset

    # combine guesses
    guess = np.transpose([guess_dhacosdec, guess_ddec, guess_snr])

    # random S/N between zero and threshold for non-detection beams
    if num_sb_nondet > 0:
        print("Setting S/N in non-detection beams")
        snr_nondet_guess = np.random.random(size=(nwalkers, num_sb_nondet)) * snr_min
        # combine guesses
        guess = np.hstack([guess, snr_nondet_guess])

    # run MC
    print(f"Running MCMC with {nwalkers} walkers, {nsteps} steps, {nburn} burn-in points")
    model = SBModelBayes(ha_cb.to(u.rad).value, dec_cb.to(u.rad).value, fmin=1350, min_freq=1220.7,
                         nfreq=32)

    # import IPython; IPython.embed(); exit()

    with Pool(os.cpu_count() - 1) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                        args=(sb_det, sb_nondet, snr_det, ha_cb, dec_cb, maxsnr),
                                        pool=pool)
        sampler.run_mcmc(guess, nsteps, progress=True)

    sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
    # remove burn-in points and combine walkers
    sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    # extract only HA, Dec offsets
    dhacosdec_sample, ddec_sample = sample[:, :2].T
    # convert to RA, Dec
    dec_sample = ddec_sample * u.arcmin + dec_cb
    ha_sample = dhacosdec_sample * u.arcmin / np.cos(dec_sample) + ha_cb
    coords = tools.hadec_to_radec(ha_sample, dec_sample, tarr)
    ra_s = coords.ra.deg
    dec_s = coords.dec.deg
    radec = [ra_s, dec_s]

    # plot result
    fig = plt.figure()
    axes = plot_mcmc(radec, fig=fig, labels=['RA', 'DEC'], colors='c')
    ax = axes[0]
    ax.plot(ra_s, dec_s, '.k', alpha=0.1, ms=4, label='Samples')
    ax.plot(source_ra.value, source_dec.value, marker='o', ms=10, c='r', label='Source position')
    # plot radius at max distance
    if True:
        max_dist = 33.7/60  # deg
        angles = np.linspace(0, 2 * np.pi, 100)
        dxcosy = max_dist * np.cos(angles)
        dy = max_dist * np.sin(angles)
        y = source_dec.to(u.deg).value + dy
        dx = dxcosy / np.cos(y * np.pi / 180)
        x = source_ra.to(u.deg).value + dx
        ax.plot(x, y, c='green', alpha=.5)
    ax.legend()

    # plot burn-in
    fig, ax = plt.subplots()
    for i in range(len(sampler.lnprobability)):
        ax.plot(sampler.lnprobability[i, :], linewidth=0.3, color='k', alpha=0.4)
    ax.set_ylabel('ln(P)')
    ax.set_xlabel('Step number')
    ax.axvline(nburn, linestyle='dotted')

    plt.show()
