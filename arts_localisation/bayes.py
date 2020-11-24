#!/usr/bin/env python3
#
# Test bayesian method for localisation

import numpy as np
from astropy.time import Time
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
from astroML.plotting import plot_mcmc

from arts_localisation import tools
from arts_localisation.constants import NSB
from arts_localisation.beam_models.simulate_sb_pattern import SBPattern


def log_prior(params, ha_cb, dec_cb):
    """
    """
    # ha and dec must be within valid range
    dec = dec_cb + params[1] * u.arcmin
    ha = ha_cb + params[0] * u.arcmin / np.cos(dec)

    if (-180 * u.deg < ha < 180 * u.deg) and (-90 * u.deg < dec < 90 * u.deg):
        return 0
    else:
        return -np.inf


def log_likelihood(params, sbs, nondet_sbs, snrs, ha_cb, dec_cb):
    """
    :param list params: dhacosdec, ddec, boresight snr, S/N for each non-det SB
    :param array sbs: array of detection SBs
    :param array nondet_sbs: array of non-detection SBs
    :param array snrs: S/N in each detection SB
    :param Quantity ha_cb: pointing HA
    :param Quantity dec_cb: pointing Dec
    """
    # read parameters
    dhacosdec, ddec, boresight_snr = params[:3]
    # add units and convert to array
    dhacosdec = np.atleast_1d(dhacosdec) * u.arcmin
    ddec = np.atleast_1d(ddec) * u.arcmin
    if len(params) > 3:
        snr_sb_nondet = params[3:]
        # construct S/N for each SB
        snr_all = np.zeros(NSB)
        snr_all[sbs] = snrs
        snr_all[nondet_sbs] = params[3:]
    else:
        snr_all = snrs
    assert len(snr_all) == NSB

    # create SB pattern
    fmin = 1350 * u.MHz
    sbp = SBPattern(ha_cb, dec_cb, dhacosdec, ddec, fmin=fmin,
                    min_freq=1220.7 * u.MHz, cb_model='gauss', cbnum=0,
                    no_pbar=True)
    # get pattern integrated over frequency, and keep only SB axis (one RA, Dec)
    # TODO: spectral indices?
    sb_model = sbp.beam_pattern_sb_int[:, 0, 0]

    # construct likelihood
    logL = np.sum(((sb_model * boresight_snr) - snr_all) ** 2, axis=0)
    return logL


def log_posterior(params, sbs, nondet_sbs, snrs, ha_cb, dec_cb):
    return log_prior(params, ha_cb, dec_cb) + log_likelihood(params, sbs, nondet_sbs, snrs, ha_cb, dec_cb)


if __name__ == '__main__':
    # set up parameters
    source_ra = 29.50333 * u.deg
    source_dec = 65.71675 * u.deg

    ra = 29.50333 * u.deg
    dec = 65.71675 * u.deg

    snr_min = 8

    burst = {'tstart': Time('2020-05-11T07:36:22.0'),
             'toa': 3610.84 * u.s,
             'ra_cb': 29.50333 * u.deg,
             'dec_cb': 65.71675 * u.deg,
             'snr_array': '/mnt/win/f/Leon/Desktop/python/arts_localisation'
                          '/test_data/R3/snr/R3_20200511_3610_CB00_SNR.txt'
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
    nwalkers = 20
    nsteps = 30
    nburn = 5

    # random S/N between 1 and 10 x measured S/N
    minval = snr_det.max()
    maxval = 10 * minval
    guess_snr = (maxval - minval) * np.random.random(nwalkers) + minval

    # random HA / Dec within 30' of initial point
    max_offset = 30  # arcmin

    guess_ddec = 2 * max_offset * np.random.random(nwalkers) - max_offset
    guess_dhacosdec = 2 * max_offset * np.random.random(nwalkers) - max_offset

    # combine guesses
    guess = np.transpose([guess_dhacosdec, guess_ddec, guess_snr])

    # random S/N between zero and threshold for non-detection beams
    if num_sb_nondet > 0:
        snr_nondet_guess = np.random.random(size=(nwalkers, num_sb_nondet)) * snr_min
        # combine guesses
        guess = np.hstack([guess, snr_nondet_guess])

    # run MC
    tstart = Time.now()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=(sb_det, sb_nondet, snr_det, ha_cb, dec_cb))
    sampler.run_mcmc(guess, nsteps)
    tend = Time.now()

    print(f"Sampler took {(tend - tstart).sec:.2f} s")

    sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
    sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    # extract only HA, Dec offsets
    dhacosdec_sample, ddec_sample = sample[:, :2].T
    # convert to RA, Dec
    dec_sample = ddec_sample * u.arcmin + dec_cb
    ha_sample = dhacosdec_sample * u.arcmin / np.cos(dec_sample) + ha_cb
    coords = tools.hadec_to_radec(ha_sample, dec_sample, tarr)
    ra_s = coords.ra.deg
    dec_s = coords.dec.deg
    radec = np.transpose([ra_s, dec_s])

    # plot
    fig = plt.figure()
    axes = plot_mcmc(radec.T, fig=fig, labels=['RA', 'DEC'], colors='k')
    ax = axes[0]
    ax.plot(sample[:, 0], sample[:, 1], '.k', alpha=0.1, ms=4)
    ax.plot(source_ra.value, source_dec.value, marker='o', ms=10, c='r')
    plt.show()
