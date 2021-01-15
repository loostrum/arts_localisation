#!/usr/bin/env python

import numpy as np
import astropy.units as u
import astropy.constants as const

import arts_localisation.constants as arts_const
from arts_localisation.beam_models import SBGenerator

sbgen = SBGenerator.from_science_case(4)


def itrf_to_xyz(dish_pos, lon, ref_pos):
    """
    Convert ITRF to local XYZ coordinates

    :param array dish_pos: dish positions with shape (ndish, 3) (m)
    :param Quantity lon: longitude of observatory (rad)
    :param Quantity ref_pos: Observatory reference position relative to dish_pos (m)

    :return: XYZ positions of dishes
    """
    # rotate by -longitude to get reference to local meridian instead of Greenwich mean meridian
    rot_matrix = np.array([[np.cos(-lon), -np.sin(-lon), 0],
                           [np.sin(-lon), np.cos(-lon), 0],
                           [0, 0, 1]])
    dish_xyz = np.matmul(rot_matrix, dish_pos.T).T
    ref_xyz = np.matmul(rot_matrix, ref_pos)
    return dish_xyz - ref_xyz


def hadec_to_uvw(ha, dec, freqs, dish_pos):
    """
    Calculate uvw coordinates for given hadec pointing

    :param array ha: hour angle (rad)
    :param array dec: declination (rad)
    :param array freqs: frequencies (MHz)
    :param array dish_pos: relative dish positions (m)
    :return: uvw (array), shape (ndish, nfreq, 3)
    """
    rot_matrix = np.array([[np.sin(ha), np.cos(ha), 0],
                           [-np.sin(dec) * np.cos(ha), np.sin(dec) * np.sin(ha), np.cos(dec)],
                           [np.cos(dec) * np.cos(ha), -np.cos(dec) * np.sin(ha), np.sin(dec)]])
    uvw = np.zeros((len(dish_pos), 3))  # frequency axis will be added later
    for i, dish in enumerate(dish_pos):
        uvw[i] = np.tensordot(rot_matrix, dish, axes=1)
    # add scaling by wavelength
    scaling = freqs / const.c.to(u.m * u.MHz).value
    # return scaled uvw with shape (ndish, nfreq, 3)
    return uvw[:, None, :] * scaling[None, :, None]


class SBPatternSingle:
    def __init__(self, ha0, dec0, fmin, min_freq, nfreq):
        # fixed values
        self.nfreq = nfreq
        self.dec0 = dec0
        self.ha0 = ha0

        freqs = np.arange(self.nfreq) * arts_const.BANDWIDTH.to(u.MHz).value / self.nfreq + min_freq
        # get last channel that should be masked
        self.chan_mask_last = np.where(freqs < fmin)[0][-1]

        dish_pos = itrf_to_xyz(arts_const.DISH_ITRF['a8'].to(u.m).value, arts_const.WSRT_LON.to(u.rad).value,
                               arts_const.ARRAY_ITRF.to(u.m).value)
        self.ndish = len(dish_pos)

        self.sigmas = arts_const.CB_HPBW.to(u.rad).value * arts_const.REF_FREQ.to(u.MHz).value / freqs / \
            (2. * np.sqrt(2 * np.log(2)))
        self.uvw = hadec_to_uvw(ha0, dec0, freqs, dish_pos).astype(np.float32)
        self.dphi_tab = np.arange(self.ndish)[None, :] * np.arange(arts_const.NTAB)[:, None] / arts_const.NTAB

    def get_sb_model(self, dhacosdec, ddec):
        dec = self.dec0 + ddec
        dha = dhacosdec / np.cos(dec)

        ll = np.sin(-dha) * np.cos(ddec)
        mm = np.cos(self.dec0) * np.sin(dec) - np.sin(self.dec0) * np.cos(dec) * np.cos(dha)
        dphi_geom = self.uvw[..., 0] * ll + self.uvw[..., 1] * mm

        dphi = dphi_geom[None, ...] + self.dphi_tab[..., None]
        dphi_complex = np.exp(1j * 2 * np.pi * dphi)
        tabbeam = np.abs((dphi_complex.sum(axis=1) / self.ndish))**2
        primarybeam = np.exp(-.5 * (dhacosdec**2 + ddec**2) / self.sigmas**2)

        pbeam = tabbeam * primarybeam[None, ...]

        # convert TABs to frequency-integrated SBs
        nsubband = 32
        assert self.nfreq % nsubband == 0
        nfreq_per_subband = int(self.nfreq / nsubband)

        sbs = np.zeros(arts_const.NSB)
        for sb in range(arts_const.NSB):
            mapping = sbgen.get_map(sb)
            for subband, tab in enumerate(mapping):
                chan_start = subband * nfreq_per_subband
                chan_end = (subband + 1) * nfreq_per_subband
                # avoid channels that should be masked
                if chan_end < self.chan_mask_last:
                    continue
                elif chan_start < self.chan_mask_last:
                    chan_start = self.chan_mask_last
                sbs[sb] += pbeam[tab][chan_start:chan_end].sum(axis=0)
        # scale by number of used channels
        sbs /= (self.nfreq - self.chan_mask_last)
        return sbs
