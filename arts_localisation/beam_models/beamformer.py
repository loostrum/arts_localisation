#!/usr/bin/env python3
#
# beamformer tool to create TAB pattern

import numpy as np
import numba as nb
import astropy.units as u
import astropy.constants as const

from arts_localisation.constants import NTAB, WSRT_LON


class BeamFormer:

    def __init__(self, dish_pos, freqs, ntab=NTAB, lon=WSRT_LON, ref_pos=None, itrf=True):
        """
        ARTS tied-array beamformer simulation

        :param array dish_pos: dish positions with shape (ndish, 3), with unit
        :param array freqs: Observing frequencies (with unit)
        :param int ntab: Number of TABs (Default: NTAB from constants)
        :param Quantity lon: longitude of observatory (Default: WSRT)
        :param Quantity ref_pos: Observatory reference position relative to dish_pos (Default: None)
        :param bool itrf: Dish positions and ref position are ITRF coordinates (Default: True)
        """
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit
        self.freqs = freqs
        self.ntab = ntab

        # dish positions relative to reference position
        if itrf:
            self.dish_pos = self._itrf_to_xyz(dish_pos, lon, ref_pos)
        elif ref_pos is None:
            self.dish_pos = dish_pos
        else:
            self.dish_pos = dish_pos - ref_pos
        self.ndish = len(self.dish_pos)

        self.dphi_g = None
        self.shape = None

    @staticmethod
    def _itrf_to_xyz(dish_pos, lon, ref_pos=None):
        """
        Convert ITRF to local XYZ coordinates

        :param array dish_pos: dish positions with shape (ndish, 3), with unit
        :param Quantity lon: longitude of observatory (Default: WSRT)
        :param Quantity ref_pos: Observatory reference position relative to dish_pos (Default: None)

        :return: XYZ positions of dishes
        """
        # rotate by -longitude to get reference to local meridian instead of Greenwich mean meridian
        rot_matrix = np.array([[np.cos(-lon), -np.sin(-lon), 0],
                               [np.sin(-lon), np.cos(-lon), 0],
                               [0, 0, 1]])
        dish_xyz = np.matmul(rot_matrix, dish_pos.T).T
        if ref_pos is not None:
            ref_xyz = np.matmul(rot_matrix, ref_pos)
        else:
            ref_xyz = 0
        return dish_xyz - ref_xyz

    def _get_uvw(self, ha, dec):
        """
        Convert sky coordinates to uvw

        :param Quantity ha: hour angle
        :param Quantity dec: declination
        :return: array of uvw with shape (ndish, nfreq, 3)
        """
        rot_matrix = np.array([[np.sin(ha), np.cos(ha), 0],
                              [-np.sin(dec) * np.cos(ha), np.sin(dec) * np.sin(ha), np.cos(dec)],
                              [np.cos(dec) * np.cos(ha), -np.cos(dec) * np.sin(ha), np.sin(dec)]])

        uvw = np.zeros((self.ndish, 3))
        for i, dish in enumerate(self.dish_pos):
            uvw[i] = np.inner(rot_matrix, dish.value)
        # scale by wavelength
        scaling = (self.freqs / const.c).to(1 / dish.unit).value
        uvw = uvw[:, None, :] * scaling[None, :, None]
        # shape is now (ndish, nfreq, 3)
        return uvw

    def set_coordinates_and_phases(self, grid_ha, grid_dec, ha0, dec0):
        """
        Calculate the geometric phases at each point in the coordinate grid, given a phase centre

        :param array grid_ha: hour angle grid
        :param array grid_dec: declination grid
        :param ha0: hour angle of phase centre
        :param dec0: declination of phase centre
        """
        assert grid_ha.shape == grid_dec.shape
        self.shape = (self.freqs.shape[0], grid_ha.shape[0], grid_ha.shape[1])

        dha = grid_ha - ha0
        dra = -dha

        # calculate uww of baselines
        # shape is (ndish, nfreq, 3)
        uvw = self._get_uvw(ha0, dec0)

        # init geometric phases per dish
        dphi_shape = (self.ndish, self.freqs.shape[0], grid_ha.shape[0], grid_ha.shape[1])
        self.dphi_g = np.zeros(dphi_shape)

        # get l,m coordinates
        ll = np.sin(dra) * np.cos(grid_dec)
        mm = np.cos(dec0) * np.sin(grid_dec) - np.sin(dec0) * np.cos(grid_dec) * np.cos(dra)
        # loop over baselines
        for i, baseline in enumerate(uvw):
            uu, vv, ww = baseline.T
            # u,v,w have shape (nfreq)
            # store phase offset with respect to phase center
            self.dphi_g[i] = (uu[:, None, None] * ll + vv[:, None, None] * mm).to(1).value

    @staticmethod
    @nb.njit('float64[:, :, :](float64[:, :, :, :])', parallel=True)
    def _phase_to_pbeam(phases):
        """
        Convert an array of phase offsets to a power beam

        :param array phases: complex phase offsets
        :return: power beam
        """
        vbeam = np.exp(1j * 2 * np.pi * phases).sum(axis=0)
        return np.abs(vbeam) ** 2

    def beamform(self, tab=0):
        """
        Beamform a tied-array beam

        :param int tab: tied-array beam index
        :return: power beam
        """
        # define TAB phase offsets
        dphi_tab = np.arange(self.ndish) * tab / self.ntab

        # get total phase offset
        dphi = self.dphi_g + dphi_tab[:, None, None, None]

        # create voltage beam and sum over dishes
        pbeam = self._phase_to_pbeam(dphi)
        # scale by number of dished
        # squared because this is working on intensity beam
        pbeam /= self.ndish ** 2

        # return beamformed power
        return pbeam


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arts_localisation import tools
    from arts_localisation.constants import DISH_ITRF, ARRAY_ITRF, WSRT_LON

    print("Testing beamformer")

    # init coordinates
    print("Initializing coordinates")
    # due East, 45 deg above horizon
    # expect TABs parallel to horizon
    ha0, dec0 = tools.altaz_to_hadec(45 * u.deg, 90 * u.deg)

    ddec = np.linspace(-50, 50, 1000) * u.arcmin
    dha = np.linspace(-50, 50, 2000) * u.arcmin

    dHA, dDEC = np.meshgrid(dha, ddec)
    DEC = dec0 + dDEC
    dHA /= np.cos(DEC)
    HA = ha0 + dHA

    tab = 0
    freq = 1370 * u.MHz

    # get dish positions
    dish_pos_itrf = DISH_ITRF['a8']
    ref_pos_itrf = ARRAY_ITRF

    # init beamformer
    print("Initializing beamformer")
    bf = BeamFormer(dish_pos_itrf, freq, ref_pos=ref_pos_itrf)
    bf.set_coordinates_and_phases(HA, DEC, ha0, dec0)

    # run beamformer
    print(f"Forming TAB{tab:02d}")
    power = bf.beamform(tab)
    # remove freq axis
    power = power[0]

    # plot
    print("Plotting")

    def formatter(x, y):
        col = int(np.round(x))
        row = int(np.round(y))
        try:
            val = power[row, col]
            txt = f'x={x:.4f}    y={y:.4f}    [{val:.4f}]'
        except Exception:
            txt = f'x={x:.4f}    y={y:.4f}'
        return txt

    ALT, AZ = tools.hadec_to_altaz(HA, DEC)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(HA.to(u.deg).value, DEC.to(u.deg).value, power)
    ax.set_xlabel('HA (deg)')
    ax.set_ylabel('DEC (deg)')
    ax.format_coord = formatter

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(AZ.to(u.deg).value, ALT.to(u.deg).value, power)
    ax.set_xlabel('AZ (deg)')
    ax.set_ylabel('ALT (deg)')
    ax.format_coord = formatter

    plt.show()
