#!/usr/bin/env python3

import argparse
import logging

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from arts_localisation.beam_models.sb_generator import SBGenerator
from arts_localisation.beam_models.beamformer import BeamFormer
from arts_localisation.beam_models.compound_beam import CompoundBeam
from arts_localisation.constants import DISH_ITRF, ARRAY_ITRF, NTAB, NSB


logger = logging.getLogger(__name__)


class SBPattern:

    def __init__(self, ha, dec, dha, ddec, sbs=None, load=False, fname=None, memmap_file=None, cb_model='gauss', cbnum=None,
                 min_freq=1220 * u.MHz, fmin=1220 * u.MHz, fmax=1520 * u.MHz, nfreq=64, dish_mode='a8'):
        """
        Generate Synthesised Beam pattern.

        :param Quantity ha: hour angle of phase center
        :param Quantity dec: declination of phase center
        :param array dha: array of hour angle offset coords (without cos(dec) factor), with unit
        :param array ddec: array of declination offset coords, with unit
        :param array sbs: array of SBs to generate [Default: all]
        :param bool load: load beam pattern from disk instead of generating
        :param str fname: file containing beam pattern
        :param str memmap_file: file to use for memmap (Default: no memmap)
        :param str cb_model: CB model type to use (default: gauss)
        :param int cbnum: which CB to use for modelling (only relevant if cb_model is 'real')
        :param Quantity min_freq: lowest frequency of data
        :param Quantity: minimum frequency to consider (Default 1220 MHz)
        :param Quantity: maximum frequency to consider (Default: 1520 MHz)
        :param int nfreq: number of frequency channels, should be multiple of nsub=32 (default:64)
        :param str dish_mode: Apertif setup (Default: a8, i.e. 8 equidistant dishes)

        """
        df = 300. * u.MHz / nfreq

        # fname is required when load is True
        if load and fname is None:
            raise ValueError("fname cannot be None when load=True")

        # cb is required when cb_model is real
        if cb_model == 'real' and cbnum is None:
            raise ValueError("cbnum cannot be None when cb_model='real'")

        if sbs is None:
            sbs = range(NSB)

        if load:
            self.beam_pattern_sb_int = np.load(fname)
            self.beam_pattern_tab = None
            self.beam_pattern_tab_1d = None

        # check shape
        ndim_dha = len(dha.shape)
        ndim_ddec = len(ddec.shape)
        assert ndim_dha == ndim_ddec
        if ndim_dha == 2:
            # already a grid, shapes have to be exactly equal
            assert np.array_equal(dha.shape, ddec.shape)
            # store grid
            dHA = dha
            dDEC = ddec
        elif ndim_dha == 1:
            # create grid
            dHA, dDEC = np.meshgrid(dha, ddec)
        else:
            raise ValueError("Input coordinates should be 1D or 2D arrays")

        # set coordinates
        numDEC, numHA = dHA.shape
        DEC = dec + dDEC
        HA = ha + dHA / np.cos(DEC)

        freqs = np.arange(nfreq) * df + min_freq + df / 2  # center of each channel
        mask = np.logical_or(freqs > fmax, freqs < fmin)

        # Generate beam pattern if load=False
        if not load:
            # init beamformer
            bf = BeamFormer(DISH_ITRF[dish_mode], freqs, ref_pos=ARRAY_ITRF)
            bf.set_coordinates_and_phases(HA, DEC, ha, dec)
            # init compound beam
            cb = CompoundBeam(freqs, dHA, dDEC)
            # init SB generator
            sb_gen = SBGenerator.from_science_case(4)

            # CB pattern
            if cbnum is not None:
                logger.info(f"Generating CB{cbnum:02d}")
            else:
                logger.info("Generating CB")
            primary_beam = cb.beam_pattern(cb_model, cb=cbnum)

            logger.info("Generating TABs")
            # get TAB pattern for each tab, freq, ha, dec (image order: dec, then ha)
            beam_pattern_tab = np.zeros((NTAB, nfreq, numDEC, numHA), dtype=np.float32)

            for tab in tqdm.tqdm(range(NTAB)):
                # run TAB beamformer
                tab_fringes = bf.beamform(tab)
                # zero out bad freqs
                tab_fringes[mask] = 0
                # apply primary beam
                intensity = tab_fringes * primary_beam
                # store to output grid
                beam_pattern_tab[tab] = intensity.astype(np.float32)

            logger.info("Generating requested SBs")
            shape = (NSB, nfreq, numDEC, numHA)
            if memmap_file is not None:
                beam_pattern_sb = np.memmap(memmap_file + '_full_sb.dat', dtype=np.float, mode='w+', shape=shape)
            else:
                beam_pattern_sb = np.zeros(shape, dtype=np.float32)
            for sb in tqdm.tqdm(sbs):
                # apply 2D primary beam and store
                beam = sb_gen.synthesize_beam(beam_pattern_tab, sb)
                beam_pattern_sb[sb] = primary_beam * beam

            self.beam_pattern_tab = beam_pattern_tab

            # integrate SB pattern over frequency
            logger.info("Integrating SB pattern over frequency")
            shape = (NSB, numDEC, numHA)
            if memmap_file is not None:
                self.beam_pattern_sb_int = np.memmap(memmap_file + '_sb.dat', dtype=float, mode='w+', shape=shape)
            else:
                self.beam_pattern_sb_int = np.zeros(shape)
            self.beam_pattern_sb_full = beam_pattern_sb
            self.beam_pattern_sb_int = beam_pattern_sb.mean(axis=1)

        self.mid_ha = int(numHA / 2)
        self.mid_dec = int(numDEC / 2)
        self.mid_freq = int(nfreq / 2)

        self.dHA = dHA
        self.dDEC = dDEC
        self.freqs = freqs

    def save(self, prefix):
        """
        Save on-sky SB and TAB maps

        :param str prefix: file name prefix
        """

        fname_tab = f"models/tied-array_beam_pattern_{prefix}"
        fname_sb = f"models/synthesized_beam_pattern_{prefix}"
        np.save(fname_tab, self.beam_pattern_tab)
        np.save(fname_sb, self.beam_pattern_sb_int)

    def plot(self, show=True, tab=0):
        """
        Plot the generated pattern

        :param bool show: Also show the plot
        :param int tab: TAB to plot
        """
        logger.info("Plotting")
        # kwargs = {'vmin': .05, 'norm': LogNorm()}
        kwargs = {'vmin': .05}

        if self.beam_pattern_tab is not None:
            fig, axes = plt.subplots(ncols=2, figsize=(12.8, 6), sharex=True)
            axes = axes.flatten()

            # TAB00 at one freq
            ax = axes[0]
            X, Y = self.dHA, self.dDEC
            img = ax.pcolormesh(X, Y, self.beam_pattern_tab[tab][self.mid_freq], **kwargs)
            # fig.colorbar(img, ax=ax)
            ax.set_aspect('equal')
            ax.set_xlabel('dHA [arcmin]')
            ax.set_ylabel('dDec [arcmin]')
            ax.set_title('On-sky pattern at {}'.format(self.freqs[self.mid_freq]))

            # TAB00 HA vs freq
            ax = axes[1]
            X, Y = np.meshgrid(self.dHA[self.mid_dec], self.freqs)
            img = ax.pcolormesh(X, Y, self.beam_pattern_tab[tab][:, self.mid_dec, :], **kwargs)
            # fig.colorbar(img, ax=ax)
            ax.set_xlabel('dHA [arcmin]')
            ax.set_ylabel('Frequency [MHz]')
            ax.set_title('Constant DEC slice through center of CB')

            fig.tight_layout()
            fig.suptitle(f'TAB{tab:02d}')
        else:
            logger.info("TAB data not available - not plotting")

        # SB
        if self.beam_pattern_sb_int is not None:
            sbs = [0, 35, 70]
            fig, axes = plt.subplots(ncols=len(sbs), figsize=(12.8, 6), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
            axes = axes.flatten()

            # SB on sky
            for i, sb in enumerate(sbs):
                ax = axes[i]
                X, Y = self.dHA, self.dDEC
                img = ax.pcolormesh(X, Y, self.beam_pattern_sb_int[sb], **kwargs)
                # fig.colorbar(img, ax=ax)
                ax.set_aspect('equal')
                ax.set_xlabel('dHA [arcmin]')
                ax.set_ylabel('dDec [arcmin]')
                ax.set_title(f'SB{sb:02d}')

            fig.tight_layout()
            fig.suptitle('On-sky pattern summed over frequency')
        else:
            logger.info("SB data not available - not plotting")

        if show:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ha', required=True, type=float, help="Hour angle in degrees")
    parser.add_argument('--dec', required=True, type=float, help="Declination in degrees")
    parser.add_argument('--cb_model', required=False, type=str, default='gauss', help="CB model type to use "
                        "(Default: %(default)s)")
    parser.add_argument('--cb', required=False, type=int, default=0, help="Which CB to use when using 'real' CB model "
                        "(Default: %(default)s)")
    parser.add_argument('--fmin', type=int, default=1220,
                        help="Minimum frequency in MHz (default: %(default)s)")
    parser.add_argument('--fmax', type=int, default=1520,
                        help="Maximum frequency in MHz (default: %(default)s)")
    parser.add_argument('--nfreq', type=int, default=64,
                        help="Number of frequency channels (default: %(default)s)")
    parser.add_argument('--memmap_file', type=str, help="If present, use this file for numpy memmap")
    parser.add_argument('--plot', action='store_true', help="Create and show plots")

    args = parser.parse_args()

    # create HA, Dec arrays
    maxdist = 30  # arcmin
    npoint = 100
    ddec = np.linspace(-maxdist, maxdist, npoint) * u.arcmin
    dha = (np.linspace(-maxdist, maxdist, npoint) * u.arcmin)
    dHA, dDEC = np.meshgrid(dha, ddec)

    if args.cb is not None:
        out_prefix = "{}_cb{:02d}_HA{:.2f}_Dec{:.2f}_{}-{}".format(args.cb_model, args.cb, args.ha, args.dec,
                                                                   args.fmin, args.fmax)
    else:
        out_prefix = f"{args.cb_model}_cb_HA{args.ha:.2f}_Dec{args.dec:.2f}_{args.fmin}-{args.fmax}"

    # convert args to dict and remove unused params
    kwargs = vars(args).copy()
    del kwargs['plot']
    # rename cb to cbnum
    # keep user arg cb for simplicity
    kwargs['cbnum'] = kwargs['cb']
    del kwargs['cb']
    # add units
    kwargs['ha'] *= u.deg
    kwargs['dec'] *= u.deg
    kwargs['fmin'] *= u.MHz
    kwargs['fmax'] *= u.MHz
    # add ha dec offsets
    kwargs['dha'] = dHA
    kwargs['ddec'] = dDEC

    # generate and store full beam pattern
    beam_pattern = SBPattern(**kwargs)
    beam_pattern.save(out_prefix)
    # or load a beam pattern from disk
    # beam_pattern = SBPattern(load=True, fname='models/synthesized_beam_pattern_gauss_cb_PA10.600506.npy')

    # plot
    if args.plot:
        beam_pattern.plot(show=True)
