#!/usr/bin/env python3

import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.widgets import Slider

from arts_localisation.constants import WSRT_LAT, DISH, CB_HPBW, REF_FREQ
from arts_localisation.beam_models.compound_beam import CompoundBeam
from arts_localisation.beam_models.beamformer import BeamFormer
import tools


def plot_hadec():
    """
    Plot parallactic and projection angle
    as function of HA, Dec
    """
    ha = np.linspace(-90, 90, 100) * u.deg
    dec = np.linspace(0, 90, 100) * u.deg

    X, Y = np.meshgrid(ha, dec)

    parang = tools.hadec_to_par(X, Y)
    proj = tools.hadec_to_proj(X, Y)

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8))

    # parallactic angle
    ax = axes[0]
    img = ax.pcolormesh(X, Y, parang, cmap='seismic', vmin=-90, vmax=90,
                        shading='nearest')
    ax.set_aspect('auto')
    ax.axhline(WSRT_LAT.to(u.deg).value, c='k')
    ax.axvline(0, c='k')
    fig.colorbar(img, ax=ax)
    ax.set_title('Parallactic angle')
    ax.set_ylabel('Dec (deg)')

    # projection angle
    ax = axes[1]
    img = ax.pcolormesh(X, Y, proj, cmap='seismic', vmin=-90, vmax=90,
                        shading='nearest')
    ax.set_aspect('auto')
    ax.axvline(0, c='k')
    fig.colorbar(img, ax=ax)
    ax.set_title('Projection angle')
    ax.set_xlabel('Hour angle (deg)')
    ax.set_ylabel('Dec (deg)')

    plt.show()


def plot_tab_pattern(tabnum):
    """
    Plot TAB pattern as function of HA, Dec
    Interactive
    tabnum: which tab to plot
    """
    # init beamformer and compound beam
    theta = np.linspace(-50, 50, 1000) * u.arcmin
    freqs = np.linspace(1220, 1520, 256) * u.MHz
    dish_pos = DISH['a8']
    proj = 0 * u.deg
    bf = BeamFormer(dish_pos, proj, freqs)
    cb = CompoundBeam(freqs, theta)
    # get primary beam, remove vertical coordinate
    primary_beam = cb.beam_pattern('gauss')[:, 0, :]
    # get TAB00 pattern to init plot with
    tab = bf.beamform(theta, tab=tabnum) * primary_beam

    # create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    # make room for sliders
    plt.subplots_adjust(bottom=.25)

    # slider settings
    ha_init = 0
    ha_min = -90
    ha_max = 90
    ha_step = 1

    dec_init = 0
    dec_min = -35
    dec_max = 90
    dec_step = 1

    fc = 'lightgoldenrodyellow'
    ax_ha = plt.axes([.1, .15, .8, .03], facecolor=fc)
    ax_dec = plt.axes([.1, .1, .8, .03], facecolor=fc)
    slider_ha = Slider(ax_ha, 'HA (deg)', ha_min, ha_max, valinit=ha_init, valstep=ha_step)
    slider_dec = Slider(ax_dec, 'Dec (deg)', dec_min, dec_max, valinit=dec_init, valstep=dec_step)

    # init plot
    X, Y = np.meshgrid(theta, freqs)
    pcm = ax.pcolormesh(X, Y, tab, shading='nearest')

    # function to update plot
    def update(val):
        # get new projection angle
        ha = slider_ha.val
        dec = slider_dec.val
        # create new tab
        bf.theta_proj = tools.hadec_to_proj(ha * u.deg, dec * u.deg)
        tab = bf.beamform(theta, tab=tabnum) * primary_beam

        pcm.set_array(tab[:-1, :-1].ravel())

    # attach slider to update function
    slider_ha.on_changed(update)
    slider_dec.on_changed(update)

    ax.set_xlim(theta.min().value, theta.max().value)
    ax.set_ylim(freqs.min().value, freqs.max().value)
    ax.set_title(f'8-dish tied-array beam for Gaussian compound beam (TAB{tabnum:02d})')
    ax.set_xlabel('E-W (arcmin)')
    ax.set_ylabel('Frequency (MHz)')
    plt.show()


def plot_sb_rotation():
    """
    Plot SB rotation as function of HA, Dec
    Interactive
    """

    # Load CB offsets in HA, Dec
    cb_offsets = np.loadtxt('square_39p1.cb_offsets', usecols=[1, 2], delimiter=',')
    ncb = len(cb_offsets)
    cb_pos = np.zeros((ncb, 2))
    for cb, (dra, ddec) in enumerate(cb_offsets):
        cb_pos[cb] = np.array([dra, ddec])
    cb_pos *= u.deg

    # CB width
    freq = 1370 * u.MHz
    cb_radius = CB_HPBW * REF_FREQ / freq / 2

    # LST to convert HA <-> RA
    lst = 180 * u.deg

    # Which SBs to plot in which cBs
    # SB for given CB
    beams = {0: [0, 35, 70],
             # 33: [0, 35, 70],
             # 39: [0, 35, 70],
             4: [63],
             5: [12],
             11: [34]}

    # SB separation = TAB separation at highest freq
    lambd = 299792458 * u.meter / u.second / (1500. * u.MHz)
    Bmax = 1296 * u.m
    sb_separation = ((.8 * lambd / Bmax).to(1) * u.radian).to(u.arcmin)

    # plot
    fig, axes = plt.subplots(figsize=(16, 9), ncols=2)
    # RA Dec axis
    ax = axes[0]
    # Alt Az axis
    ax2 = axes[1]
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'alpha': .5,
            'size': 10}
    # make room for sliders
    plt.subplots_adjust(bottom=.25)
    ax.set_aspect('equal')
    ax2.set_aspect('equal')

    # slider settings
    ha_init = 0
    ha_min = -90
    ha_max = 90
    ha_step = 1

    dec_init = 0
    dec_min = -35
    dec_max = 90
    dec_step = 1

    fc = 'lightgoldenrodyellow'
    ax_ha = plt.axes([.1, .15, .8, .03], facecolor=fc)
    ax_dec = plt.axes([.1, .1, .8, .03], facecolor=fc)
    slider_ha = Slider(ax_ha, 'HA (deg)', ha_min, ha_max, valinit=ha_init, valstep=ha_step)
    slider_dec = Slider(ax_dec, 'Dec (deg)', dec_min, dec_max, valinit=dec_init, valstep=dec_step)

    # plot in function
    def do_plot(ha0=0 * u.deg, dec0=0 * u.deg, parang=0 * u.deg):
        # store coordinates of CB00
        ra0 = lst - ha0
        alt0, az0 = tools.hadec_to_altaz(ha0, dec0)
        # plot CBs
        for cb, (dra, ddec) in enumerate(cb_pos):
            # RADec
            # pointing of this CB
            ra, dec = tools.offset_to_coord(ra0, dec0, dra, ddec)

            _ra = ra.to(u.deg).value
            _dec = dec.to(u.deg).value

            # plot
            patch = SphericalCircle((ra, dec), cb_radius,
                                    ec='k', fc='none', ls='-', alpha=.5)
            ax.add_patch(patch)
            ax.text(_ra, _dec, f'CB{cb:02d}', va='center', ha='center',
                    fontdict=font, clip_on=True)

            # AltAz
            ha = lst - ra
            alt, az = tools.hadec_to_altaz(ha, dec)
            dalt = alt - alt0
            daz = az - az0

            _alt = alt.to(u.deg).value
            _az = az.to(u.deg).value

            # plot
            patch = SphericalCircle((az, alt), cb_radius,
                                    ec='k', fc='none', ls='-', alpha=.5)
            ax2.add_patch(patch)
            ax2.text(_az, _alt, f'CB{cb:02d}', va='center', ha='center',
                     fontdict=font, clip_on=True)

        # plot SBs
        for cb in beams.keys():
            cb_dra, cb_ddec = cb_offsets[cb] * 60  # to arcmin
            for sb in beams[cb]:
                # SB increases towards higher RA
                sb_offset = (sb - 35) * sb_separation

                # draw line from (x, -y) to (x, y)
                # but the apply rotation by parallactic angle
                # in altaz:
                # x = +/-sb_offset, depending on azimuth:
                # higher SB = higher RA = East = either lower or higher Az
                # assume we are pointing above NCP if North
                if az0 > 270 * u.deg or az0 < 90 * u.deg:
                    sgn = 1
                else:
                    sgn = -1

                # alt az of this cb
                cb_shift_az = 0
                cb_shift_alt = 0

                # y = +/- length of line a sb_offset from center of CB
                dy = np.sqrt(cb_radius ** 2 - sb_offset ** 2)
                # alt start and end point
                alts = alt0 + dy + cb_shift_alt
                alte = alt0 - dy + cb_shift_alt
                # az start and end point
                azs = az0 + sgn * sb_offset / np.cos(alts) + cb_shift_az
                aze = az0 + sgn * sb_offset / np.cos(alte) + cb_shift_az

                # convert to HA, Dec
                has, decs = tools.altaz_to_hadec(alts, azs)
                hae, dece = tools.altaz_to_hadec(alte, aze)
                # convert HA to RA
                ras = lst - has
                rae = lst - hae

                # plot in RADec
                x = [ras.to(u.deg).value, rae.to(u.deg).value]
                y = [decs.to(u.deg).value, dece.to(u.deg).value]

                ax.plot(x, y, c='b')
                # add text above lines
                ax.text(np.mean(x), np.mean(y), f"SB{sb:02d}", va='center', ha='center')

                # plot in AltAz
                x = [azs.to(u.deg).value, aze.to(u.deg).value]
                y = [alts.to(u.deg).value, alte.to(u.deg).value]

                ax2.plot(x, y, c='b')
                # add text above lines
                ax2.text(np.mean(x), np.mean(y), f"SB{sb:02d}", va='center', ha='center')

                # polar
                # theta_start = np.arctan2(dy, sb_offset)
                # theta_end = np.arctan2(-dy, sb_offset)

                # apply parallactic angle rotation
                # works positive in HA space, but negative in RA space
                # theta_start -= parang.to(u.radian).value
                # theta_end -= parang.to(u.radian).value

                # start and end in cartesian coordinates
                # shift to correct CB and position
                # xstart = cb_radius * np.cos(theta_start) + cb_dra
                # ystart = cb_radius * np.sin(theta_start) + cb_ddec
                # xend = cb_radius * np.cos(theta_end) + cb_dra
                # yend = cb_radius * np.sin(theta_end) + cb_ddec

                # plot RA Dec
                # ax.plot((xstart, xend), (ystart, yend), c='b')
                # add text above lines
                # ax.text(np.mean([xstart, xend]), np.mean([ystart, yend]),  "SB{:02d}".format(sb),  va='center', ha='center')

                # continue
                # plot Alt Az
                # ystart, xstart = tools.hadec_to_altaz(ha0-xstart*u.arcmin, dec0+ystart*u.arcmin)
                # yend, xend = tools.hadec_to_altaz(ha0-xend*u.arcmin, dec0+yend*u.arcmin)
                # subtract center and remove cosine correction
                # #xstart = (xstart - az0).to(u.arcmin).value
                # #xend = (xend - az0).to(u.arcmin).value
                # #ystart = (ystart - alt0).to(u.arcmin).value
                # #yend = (yend - alt0).to(u.arcmin).value
                # xstart = xstart.to(u.arcmin).value
                # xend = xend.to(u.arcmin).value
                # ystart = ystart.to(u.arcmin).value
                # yend = yend.to(u.arcmin).value

        # set limits
        x = ra0.to(u.deg).value
        y = dec0.to(u.deg).value
        ax.set_xlim(x - 130 / 60., x + 130 / 60.)
        ax.set_ylim(y - 100 / 60., y + 100 / 60.)
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title('RA - Dec')

        x = az0.to(u.deg).value
        y = alt0.to(u.deg).value
        ax2.set_xlim(x - 130 / 60., x + 130 / 60.)
        ax2.set_ylim(y - 100 / 60., y + 100 / 60.)
        ax2.set_xlabel('Az (deg)')
        ax2.set_ylabel('Alt (deg)')
        ax2.set_title('Alt - Az')

    # plot once
    do_plot()

    # define update function
    def update(val):
        ha = slider_ha.val * u.deg
        dec = slider_dec.val * u.deg
        parang = tools.hadec_to_par(ha, dec)
        ax.cla()
        ax2.cla()
        do_plot(ha, dec, parang)

    # attach sliders to update function
    slider_ha.on_changed(update)
    slider_dec.on_changed(update)

    fig.suptitle('Apertif beam pattern for LST = {} hr'.format(lst.to(u.deg).value / 15.))
    plt.show()


if __name__ == '__main__':
    choices = ['hadec', 'tab', 'sb']
    parser = argparse.ArgumentParser(description="Several plots to verify parallactic "
                                                 "and projection angles",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('plot', type=str, choices=choices, help="Which plot to show")
    parser.add_argument('--tab', type=int, default=0, help="Which TAB to plot in tab mode "
                                                           "Default: (%(default)s)")
    args = parser.parse_args()

    if args.plot == 'hadec':
        plot_hadec()
    elif args.plot == 'tab':
        plot_tab_pattern(args.tab)
    elif args.plot == 'sb':
        plot_sb_rotation()
