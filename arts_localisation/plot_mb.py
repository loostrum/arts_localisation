#!/usr/bin/env python3

import argparse
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.visualization.wcsaxes import SphericalCircle
from scipy import stats
import yaml

from arts_localisation import tools
from arts_localisation.constants import NSB, REF_FREQ, CB_HPBW

# Try switching to OSX native backend
try:
    plt.switch_backend('macosx')
except ImportError:
    pass

# confidence interval for localisation region
CONF_INT = 0.90
# CONF_INT = 0.68
# colour cycler for contour overview
cmap_list = cycle(['377eb8', 'ff7f00', '4daf4a', 'f781bf', 'a65628', '984ea3', '999999', 'e41a1c', 'dede00'])
colormap = next(cmap_list)


def do_plot(ax, RA, DEC, dchi2, dof, cb_ra, cb_dec,
            freq, CONF_INT=.90, sigma_max=3,
            title=None, src_ra=None, src_dec=None,
            contour_ax=None):
    # best pos = point with lowest (delta)chi2
    ind = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    best_ra = RA[ind]
    best_dec = DEC[ind]

    # convert CONF_INT to dchi2 value
    dchi2_contour_value = stats.chi2.ppf(CONF_INT, dof)

    # convert sigma_max to dchi2 value
    conf_int_max = stats.chi2.cdf(sigma_max ** 2, 1)
    dchi2_value_max = stats.chi2.ppf(conf_int_max, dof)

    # plot data with colorbar
    img = ax.pcolormesh(RA, DEC, dchi2, vmax=min(dchi2.max(), dchi2_value_max),
                        shading='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax)

    # add contour
    c = ax.contour(RA, DEC, dchi2, [dchi2_contour_value], colors='r')
    # get points of contour
    try:
        points = np.concatenate(c.allsegs[0])
    except ValueError:
        points = None
    if contour_ax:
        # plot contour is this axis too
        contour_ax.contour(RA, DEC, dchi2, [dchi2_contour_value], c=colormap, label=title)
        next(cmap_list)

    # add best position
    ax.plot(best_ra, best_dec, c='r', marker='.', ls='', ms=10, label='Best position')

    # add source position
    if src_ra is not None and src_dec is not None:
        ax.plot(src_ra, src_dec, c='cyan', marker='+', ls='', ms=10, label='Source position')

    # add cb position(s)
    cb_radius = .5 * CB_HPBW * REF_FREQ / freq
    if isinstance(cb_ra, float):
        cb_pos = np.array([[cb_ra, cb_dec]])
    else:
        cb_pos = np.transpose([cb_ra, cb_dec])

    for i, (ra, dec) in enumerate(cb_pos):
        # set label once
        if i == 0:
            label = 'CB center'
        else:
            label = ''
        ax.plot(ra, dec, c='k', marker='x', ls='', ms=10, label=label)

        # add CB size
        patch = SphericalCircle((ra * u.deg, dec * u.deg), cb_radius, ec='k', fc='none', ls='-', alpha=.5)
        ax.add_patch(patch)

    # labels
    ax.set_xlabel('Right Ascension (deg)')
    ax.set_ylabel('Declination (deg)')
    ax.set_title(title)
    # ax.legend()

    return points


def get_stats(RA, DEC, dchi2, dof):
    # get midpoint of coord grid
    ny, nx = RA.shape
    midx = int(nx / 2)
    midy = int(ny / 2)

    # get pixel size
    ddec = np.abs(DEC[midy, midx] - DEC[midy + 1, midx]) * u.deg
    dra_scaled = np.abs((RA[midy, midx] - RA[midy, midx + 1]) * np.cos(DEC[midy, midx])) * u.deg
    pix_size = dra_scaled * ddec

    # convert conf_int to dchi2 value
    dchi2_max = stats.chi2.ppf(CONF_INT, dof)
    # get number of points within this confidence interval
    npoint = np.sum(dchi2 < dchi2_max)

    # total localisation area
    area = pix_size * npoint
    print("Total localisation area: {:.2f} = {:.2f}".format(area.to(u.arcmin ** 2), area.to(u.arcsec ** 2)))


def fit_ellipse(points):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Input yaml config')
    parser.add_argument('--use_mask', action='store_true', help="Enable use of masks where S/N is"
                                                                " higher than reference beam")
    parser.add_argument('--freq', type=float, default=1370, help="Reference frequency in MHz "
                        "for CB size in plot. (Default: %(default)s)")
    args = parser.parse_args()

    # load yaml file
    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    # target name
    name_full = args.config.replace('.yaml', '')
    name = name_full.split('/')[-1]
    # reference burst
    ref_burst = conf['ref_burst']

    # source position
    try:
        src_ra = conf['source']['ra']
        src_dec = conf['source']['dec']
    except KeyError:
        src_ra = None
        src_dec = None

    # get lists of burst names
    bursts = list(conf.keys())
    # remove non-burst keys from list
    for key in ['source', 'ref_burst']:
        try:
            bursts.remove(key)
        except ValueError:
            # key not in list
            continue

    nburst = len(bursts)
    print(f"Found {nburst} bursts")

    # create one figure for individual bursts and one for total
    ncols = int(np.ceil(np.sqrt(nburst)))
    nrows = int(np.ceil(nburst / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    if nrows * ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # lood coordinate grid
    RA, DEC = np.load(f'{name_full}_coord.npy')

    # init totals
    chi2_total = np.zeros(RA.shape)
    cb_ra_all = []
    cb_dec_all = []

    # init final plot
    fig_final, axes_final = plt.subplots(ncols=2, sharex=True, sharey=True)
    combined_ax, contour_ax = axes_final

    # loop over bursts
    for i, burst in enumerate(bursts):
        print("Plotting {}/{}".format(i + 1, nburst))
        # degrees of freedom for one CB: NSB - nparam (2)
        # reference burst has one dof less
        if burst == ref_burst:
            dof = NSB - 3
        else:
            dof = NSB - 2
        # load chi2
        chi2 = np.load(f'{name_full}_chi2_{burst}.npy')
        # apply bad index mask if enabled
        if args.use_mask:
            # load mask
            mask = np.load(f'{name_full}_bad_{burst}.npy')
            # apply
            chi2[mask] = 1E9
        # add to total
        chi2_total += chi2
        # convert to delta chi2
        dchi2 = chi2 / chi2.min()
        # load CB position
        cb_dec = conf[burst]['dec']
        try:
            cb_ra = conf[burst]['ra']
        except KeyError:
            cb_ha = conf[burst]['ha']
            t = Time(conf[burst]['tstart']) + TimeDelta(conf[burst]['tarr'], format='sec')
            cb_radec = tools.hadec_to_radec(cb_ha * u.deg, cb_dec * u.deg, t)
            cb_ra = cb_radec.ra.deg
            cb_dec = cb_radec.dec.deg
        # store for total
        cb_ra_all.append(cb_ra)
        cb_dec_all.append(cb_dec)
        # select plot axis
        ax = axes[i]
        do_plot(ax, RA, DEC, dchi2, dof, cb_ra, cb_dec, args.freq * u.MHz,
                src_ra=src_ra, src_dec=src_dec, title=burst, contour_ax=contour_ax)

    # empty non-used plots
    nplot = nrows * ncols
    if nburst < nplot:
        for i in range(nburst, nplot):
            axes[i].axis('off')

    fig.tight_layout()

    # total
    # dof of total: nburst * nsb - 1 - nparam
    dof_total = nburst * NSB - 3
    dchi2_total = chi2_total - chi2_total.min()

    get_stats(RA, DEC, dchi2_total, dof_total)

    print("Plotting combined localisation region")
    # final localisation region

    contour_points = do_plot(combined_ax, RA, DEC, dchi2_total, dof_total, cb_ra_all, cb_dec_all, args.freq * u.MHz,
                             src_ra=src_ra, src_dec=src_dec, title=f"{name} all CBs combined")
    contour_ax.set_xlabel('Right Ascension (deg)')
    contour_ax.set_ylabel('Declination (deg)')
    contour_ax.set_title('Contours of all CBs')
    contour_ax.legend()

    fig_final.tight_layout()

    if contour_points is not None:
        fit_ellipse(contour_points)

    plt.show()
