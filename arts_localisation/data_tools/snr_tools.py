#!/usr/bin/env python
#
# based on Liam's code

import numpy as np
from scipy import correlate


def calc_snr_amber(data, thresh=3):
    """
    Calculate peak S/N using the same method as AMBER:
    Outliers are removed four times before calculating the S/N as peak - median / sigma
    The result is scaled by 1.048 to account for the removed values

    :param array data: timeseries data
    :param float thresh: sigma threshold for outliers (Default: 3)
    :return: peak S/N
    """
    sig = np.std(data)
    dmax = data.max()
    dmed = np.median(data)

    # remove outliers 4 times until there
    # are no events above threshold * sigma
    for i in range(4):
        ind = np.abs(data - dmed) < thresh * sig
        sig = np.std(data[ind])
        dmed = np.median(data[ind])
        data = data[ind]

    return (dmax - dmed) / (1.048 * sig)


def calc_snr_matched_filter(data, widths=None):
    """
    Calculate S/N using several matched filter widths, then pick the highest S/N

    :param array data: timeseries data
    :param list widths: matched filters widhts to try
                        (Default: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500])
    :return: highest S/N (float), corresponding matched filter width (int)
    """
    if widths is None:
        # all possible widths as used by AMBER
        widths = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]

    snr_max = 0
    width_max = None

    # get S/N for each width, store only max S/N
    for w in widths:
        # apply boxcar-shaped filter
        mf = np.ones(w)
        data_mf = correlate(data, mf)

        # get S/N
        snr = calc_snr_amber(data_mf)

        # store if S/N is highest
        if snr > snr_max:
            snr_max = snr
            width_max = w

    return snr_max, width_max
