#!/usr/bin/env python3

import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from darc.sb_generator import SBGenerator

from beamformer import BeamFormer
from compound_beam import CompoundBeam
from constants import DISH
import convert


class SBPattern(object):

    def __init__(self, sbs=None, load=False, fname=None, theta_proj=0*u.deg, memmap_file=None,
                 cb_model='real'):
        """
        Generate Synthesized Beam pattern
        :param sbs: array of SBs to generate [Default: all]
        :param load: load beam pattern from disk instead of generating
        :param fname: file containing beam pattern
        :param theta_proj: projection angle used when generating TAB pattern
        :param memmap_file: file to use for memmap (Default: no memmap)
        :param cb_model: CB model type to use (defeault: real)
        """
        max_dist = 50  # arcmin
        npoint_theta = 10001  # has to be odd to ensure inclusion of 0
        npoint_phi = 201  # has to be odd to ensure inclusion of 0
        dish_mode = 'a8'
        min_freq = 1220 * u.MHz
        nfreq = 64  # should be multiple of 32
        ntab = 12
        nsb = 71

        # fname is required when load is True
        if load and fname is None:
            raise ValueError("fname cannot be None when load=True")

        if sbs is None:
            sbs = range(nsb)

        df = 300.*u.MHz / nfreq
        dish_pos = DISH[dish_mode]

        if load:
            self.beam_pattern_sb_sky = np.load(fname)
            self.beam_pattern_tab = None
            self.beam_pattern_tab_1d = None
            # overwrite lengths
            nsb, npoint_theta, npoint_phi = self.beam_pattern_sb_sky.shape

        dtheta = np.linspace(-max_dist, max_dist, npoint_theta) * u.arcmin
        dphi = np.linspace(-max_dist, max_dist, npoint_phi) * u.arcmin
        freqs = np.arange(nfreq) * df + min_freq + df / 2  # center of each channel

        # Generate beam pattern if load=False
        if not load:
            bf = BeamFormer(freqs=freqs, ntab=ntab, theta_proj=theta_proj, dish_pos=dish_pos)
            cb = CompoundBeam(freqs, dtheta, dphi)
            sb_gen = SBGenerator.from_science_case(4)

            # CB pattern
            print("Generating CB")
            primary_beam = cb.beam_pattern(cb_model)

            print("Generating TABs")
            # get TAB pattern for each tab, freq, theta, phi (image order: phi, then theta)
            beam_pattern_tab = np.zeros((ntab, nfreq, npoint_phi, npoint_theta))
            # second array for pattern without phi coordinate for faster SB generation
            beam_pattern_tab_1d = np.zeros((ntab, nfreq, npoint_theta))
            for tab in tqdm.tqdm(range(ntab)):
                # TAB beamformer
                tab_fringes = bf.beamform(dtheta, dish_pos, tab=tab)
                # Apply TAB pattern at each phi to primary beam pattern
                i_tot_2d = primary_beam[..., None] * tab_fringes
                # store to output grid
                beam_pattern_tab[tab] = i_tot_2d
                beam_pattern_tab_1d[tab] = tab_fringes

            print("Generating requested SBs")
            shape = (nsb, nfreq, npoint_phi, npoint_theta)
            if memmap_file is not None:
                beam_pattern_sb = np.memmap(memmap_file+'_full_sb.dat', dtype=float, mode='w+', shape=shape)
            else:
                beam_pattern_sb = np.zeros(shape)
            for sb in tqdm.tqdm(sbs):
                # loop over SB map
                beam = sb_gen.synthesize_beam(beam_pattern_tab_1d, sb)
                # apply 2D primary beam and store
                beam_pattern_sb[sb] = primary_beam[..., None] * beam

            self.beam_pattern_tab = beam_pattern_tab
            self.beam_pattern_tab_1d = beam_pattern_tab_1d
            # sum SB pattern over frequency
            print("Generating on-sky SB pattern")
            shape = (nsb, npoint_phi, npoint_theta)
            if memmap_file is not None:
                self.beam_pattern_sb_sky = np.memmap(memmap_file+'_sb.dat', dtype=float, mode='w+', shape=shape)
            else:
                self.beam_pattern_sb_sky = np.zeros(shape)
            self.beam_pattern_sb_sky = beam_pattern_sb.sum(axis=1)

        self.npoint_theta = npoint_theta
        self.npoint_phi = npoint_phi
        self.nsb = nsb
        self.theta_proj = theta_proj

        self.mid_theta = int(npoint_theta/2.)
        self.mid_phi = int(npoint_phi/2.)
        self.mid_freq = int(nfreq/2.)

        self.dtheta = dtheta
        self.dphi = dphi
        self.freqs = freqs

    def save(self):
        """
        Save on-sky SB and TAB maps
        :param prefix: file name prefix
        """
        fname_tab = "{}_PA{:.6f}".format("models/tied-array_beam_pattern_single_cb", self.theta_proj.to(u.deg).value)
        fname_sb = "{}_PA{:.6f}".format("models/synthesized_beam_pattern_single_cb", self.theta_proj.to(u.deg).value)
        np.save(fname_tab, self.beam_pattern_tab)
        np.save(fname_sb, self.beam_pattern_sb_sky)

    def plot(self, show=True):
        print("Plotting")
        # TAB00
        tab = 0
        # kwargs = {'vmin': .05, 'norm': LogNorm()}
        kwargs = {'vmin': .05}

        if self.beam_pattern_tab is not None:
            fig, axes = plt.subplots(ncols=2, figsize=(12.8, 6), sharex=True)
            axes = axes.flatten()

            # TAB00 at one freq
            ax = axes[0]
            X, Y = np.meshgrid(self.dtheta, self.dphi)
            img = ax.pcolormesh(X, Y, self.beam_pattern_tab[tab][self.mid_freq], **kwargs)
            # fig.colorbar(img, ax=ax)
            ax.set_aspect('equal')
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel(r'$\phi$ [arcmin]')
            ax.set_title('On-sky pattern at {}'.format(self.freqs[self.mid_freq]))

            # TAB00 theta vs freq
            ax = axes[1]
            X, Y = np.meshgrid(self.dtheta, self.freqs)
            img = ax.pcolormesh(X, Y, self.beam_pattern_tab[tab][:, self.mid_phi, :], **kwargs)
            # fig.colorbar(img, ax=ax)
            ax.set_xlabel(r'$\theta$ [arcmin]')
            ax.set_ylabel('Frequency [MHz]')
            ax.set_title('E-W slice through center of CB')

            fig.tight_layout()
            fig.suptitle('TAB{:02d}'.format(tab))
        else:
            print("TAB data not available - not plotting")

        # SB
        if self.beam_pattern_sb_sky is not None:
            sbs = [0, 35, 70]
            fig, axes = plt.subplots(ncols=len(sbs), figsize=(12.8, 6), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
            axes = axes.flatten()

            # SB on sky
            for i, sb in enumerate(sbs):
                ax = axes[i]
                X, Y = np.meshgrid(self.dtheta, self.dphi)
                img = ax.pcolormesh(X, Y, self.beam_pattern_sb_sky[sb], **kwargs)
                # fig.colorbar(img, ax=ax)
                ax.set_aspect('equal')
                ax.set_xlabel(r'$\theta$ [arcmin]')
                ax.set_ylabel(r'$\phi$ [arcmin]')
                ax.set_title('SB{:02d}'.format(sb))

            fig.tight_layout()
            fig.suptitle('On-sky pattern summed over frequency')
        else:
            print("SB data not available - not plotting")

        if show:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ha', required=True, type=float, help="Hour angle in degrees")
    parser.add_argument('--dec', required=True, type=float, help="Declination in degrees")
    parser.add_argument('--cb_model', required=False, type=str, default='real', help="CB model type to use "
                        "(Default: %(default)s)")
    parser.add_argument('--memmap_file', type=str, help="If present, use this file for numpy memmap")
    parser.add_argument('--plot', action='store_true', help="Create and show plots")

    args = parser.parse_args()

    # convert HA, Dec to projection angle
    ha = args.ha * u.deg
    dec = args.dec * u.deg
    theta_proj = convert.ha_to_proj(ha, dec)

    # convert args to dict and remove unused params
    kwargs = vars(args).copy()
    del kwargs['ha']
    del kwargs['dec']
    del kwargs['plot']
    #add projection angle
    kwargs['theta_proj'] = theta_proj

    # generate and store full beam pattern
    beam_pattern = SBPattern(**kwargs)
    beam_pattern.save()
    # or load a beam pattern from disk
    #beam_pattern = SBPattern(load=True, fname='models/synthesized_beam_pattern_single_cb_PA10.600506.npy')

    # plot
    if args.plot:
        beam_pattern.plot(show=True)
