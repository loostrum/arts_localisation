#!/usr/bin/env python3

import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from constants import WSRT_LAT, DISH, CB_HPBW, REF_FREQ
from compound_beam import CompoundBeam
from beamformer import BeamFormer
import convert


def plot_hadec():
    """
    Plot parallactic and projection angle
    as function of HA, Dec
    """
    ha = np.linspace(-90, 90, 100) * u.deg
    dec = np.linspace(0, 90, 100) * u.deg

    X, Y = np.meshgrid(ha, dec)
    
    parang = convert.ha_to_par(X, Y)
    proj = convert.ha_to_proj(X, Y)

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8))

    # parallactic angle
    ax = axes[0]
    img = ax.pcolormesh(X, Y, parang, cmap='seismic', vmin=-90, vmax=90)
    ax.set_aspect('auto')
    ax.axhline(WSRT_LAT.to(u.deg).value, c='k')
    ax.axvline(0, c='k')
    fig.colorbar(img, ax=ax)
    ax.set_title('Parallactic angle')
    ax.set_ylabel('Dec (deg)')

    # projection angle
    ax = axes[1]
    img = ax.pcolormesh(X, Y, proj, cmap='seismic', vmin=-90, vmax=90)
    ax.set_aspect('auto')
    ax.axvline(0, c='k')
    fig.colorbar(img, ax=ax)
    ax.set_title('Projection angle')
    ax.set_xlabel('Hour angle (deg)')
    ax.set_ylabel('Dec (deg)')

    plt.show()


def plot_tab_pattern():
    """
    Plot TAB pattern as function of HA, Dec
    Interactive
    """
    # init beamformer and compound beam
    theta = np.linspace(-50, 50, 1000) * u.arcmin
    freqs = np.linspace(1220, 1520, 256) * u.MHz
    dish_pos = DISH['a8']
    proj = 0*u.deg
    bf = BeamFormer(dish_pos, proj, freqs)
    cb = CompoundBeam(freqs, theta)
    # get primary beam, remove vertical coordinate 
    primary_beam = cb.beam_pattern('gauss')[:, 0, :]
    # get TAB00 pattern to init plot with
    tab = bf.beamform(theta) * primary_beam

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
    pcm = ax.pcolormesh(X, Y, tab)

    # function to update plot
    def update(val):
        # get new projection angle
        ha = slider_ha.val
        dec = slider_dec.val
        # create new tab
        bf.theta_proj = convert.ha_to_proj(ha*u.deg, dec*u.deg)
        tab = bf.beamform(theta) * primary_beam

        pcm.set_array(tab[:-1,:-1].ravel())

    # attach slider to update function
    slider_ha.on_changed(update)
    slider_dec.on_changed(update)
    
    ax.set_xlim(theta.min().value, theta.max().value)
    ax.set_ylim(freqs.min().value, freqs.max().value)
    ax.set_title('8-dish tied-array beam for Gaussian compound beam')
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
        dha = -dra
        cb_pos[cb] = np.array([dha*60, ddec*60])  # store in arcmin

    # CB width
    freq = 1370*u.MHz
    cb_radius = (CB_HPBW * freq/REF_FREQ / 2).to(u.arcmin).value

    # Which SBs to plot in which cBs
    # SB for given CB
    beams = {0: [0, 35, 70],
             33: [0, 35, 70],
             39: [0, 35, 70],
             4: [63],
             5: [12],
             11: [34]}

    # SB separation = TAB separation at highest freq
    lambd = 299792458 * u.meter / u.second / (1500. * u.MHz)
    Bmax = 1296 * u.m
    sb_separation = ((.8 * lambd / Bmax).to(1) * u.radian).to(u.arcmin).value

    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'alpha': .5,
            'size': 10}
    # make room for sliders
    plt.subplots_adjust(bottom=.25)
    ax.set_aspect('equal')

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
    def do_plot(parang=0*u.deg):
        # plot CBs
        for cb, (ha, dec) in enumerate(cb_pos):
            patch = Circle((ha, dec), cb_radius,
                           ec='k', fc='none', ls='-', alpha=.5)
            ax.add_patch(patch)
            ax.text(ha, dec, 'CB{:02d}'.format(cb), va='center', ha='center',
                    fontdict=font)

        # plot SBs
        for cb in beams.keys():
            cb_dra, cb_ddec = cb_offsets[cb] * 60  # to arcmin
            cb_dha = -cb_dra
            for sb in beams[cb]:
                sb_offset = (sb - 35) * sb_separation
                # SB increases towards higher RA
                # convert to HA offset
                sb_offset *= -1

                # draw line from (x, -y) to (x, y)
                # but the apply rotation by parallactic angle
                # define x and y

                dy = np.sqrt(cb_radius ** 2 - sb_offset ** 2)  # half length of line

                # polar
                theta_start = np.arctan2(dy, sb_offset)
                theta_end = np.arctan2(-dy, sb_offset)

                # apply parallactic angle rotation
                theta_start += parang.to(u.radian).value
                theta_end += parang.to(u.radian).value

                # start and end in cartesian coordinates
                # shift to correct CB
                xstart = cb_radius * np.cos(theta_start) + cb_dha
                ystart = cb_radius * np.sin(theta_start) + cb_ddec
                xend = cb_radius * np.cos(theta_end) + cb_dha
                yend = cb_radius * np.sin(theta_end) + cb_ddec

                # plot
                ax.plot((xstart, xend), (ystart, yend), c='b')
                # add text above lines
                ax.text(np.mean([xstart, xend]), np.mean([ystart, yend]),  "SB{:02d}".format(sb),  va='center', ha='center')

        # set limits
        ax.set_xlim(-130, 130)
        ax.set_ylim(-100, 100)
        ax.set_xlabel('HA offset (arcmin)')
        ax.set_ylabel('Dec offset (arcmin)')

    # plot once
    do_plot()

    # define update function
    def update(val):
        ha = slider_ha.val * u.deg
        dec = slider_dec.val * u.deg
        parang = convert.ha_to_par(ha, dec)
        ax.cla()
        do_plot(parang)

    # attach sliders to update function
    slider_ha.on_changed(update)
    slider_dec.on_changed(update)

    plt.show()


if __name__ == '__main__':
    choices = ['hadec', 'tab', 'sb']
    parser = argparse.ArgumentParser(description="Several plots to verify parallactic "
                                                 "and projection angles",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('plot', type=str, choices=choices, help="Which plot to show")
    args = parser.parse_args()

    if args.plot == 'hadec':
        plot_hadec()
    elif args.plot == 'tab':
        plot_tab_pattern()
    elif args.plot == 'sb':
        plot_sb_rotation()
