#!/usr/bin/env python3
#
# beamformer tool to create TAB pattern

import numpy as np
import numba as nb
import astropy.units as u
import astropy.constants as const

from constants import NTAB, WSRT_LON


class BeamFormer(object):
    
    def __init__(self, dish_pos, freqs, ntab=NTAB, lon=WSRT_LON, ref_pos=None):
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit
        self.freqs = freqs
        self.ntab = ntab
        
        # dish positions relative to reference position
        self.dish_pos = self._itrf_to_xyz(dish_pos, lon, ref_pos)
        self.ndish = len(self.dish_pos)
        
        self.dphi_g = None
        self.shape = None


    @staticmethod
    def _itrf_to_xyz(dish_pos, lon, ref_pos=None):
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
        rot_matrix = np.array([[np.sin(ha), np.cos(ha), 0], 
                              [-np.sin(dec)*np.cos(ha), np.sin(dec)*np.sin(ha), np.cos(dec)], 
                              [np.cos(dec)*np.cos(ha), -np.cos(dec)*np.sin(ha), np.sin(dec)]])
        
        uvw = np.zeros((self.ndish, 3))
        for i, dish in enumerate(self.dish_pos):
            uvw[i] = np.inner(rot_matrix, dish.value)
        # scale by wavelength
        scaling = (self.freqs / const.c).to(1/dish.unit).value
        uvw = uvw[:, None, :] * scaling[None, :, None]
        # shape is now (ndish, nfreq, 3)
        return uvw
    
    def set_coordinates_and_phases(self, grid_ha, grid_dec, ha0, dec0):
        assert grid_ha.shape == grid_dec.shape
        self.shape = (self.freqs.shape[0], grid_ha.shape[0], grid_ha.shape[1])        

        dha = grid_ha - ha0
        dra = -dha
        ddec = grid_dec - dec0

        # calculate uww of baselines
        # shape is (ndish, nfreq, 3)
        uvw = self._get_uvw(ha0, dec0)
        
        # init geometric phases per dish
        dphi_shape = (self.ndish, self.freqs.shape[0], grid_ha.shape[0], grid_ha.shape[1])
        self.dphi_g = np.zeros(dphi_shape)
        # loop over baselines
        for i, baseline in enumerate(uvw):
            uu, vv, ww = baseline.T
            # u,v,w have shape (nfreq)
            # store phase offset with respect to phase center
            self.dphi_g[i] = (uu[:, None, None] * np.sin(dra) * np.cos(grid_dec) + vv[:, None, None] * np.sin(ddec)).to(1).value


    @staticmethod
    @nb.njit('float64[:, :, :](float64[:, :, :, :])', parallel=True)
    def phase_to_pbeam(phases):
        vbeam = np.exp(1j*2*np.pi*phases).sum(axis=0)
        return np.abs(vbeam)**2
        
    
    def beamform(self, tab=0):
        # define TAB phase offsets
        dphi_tab = np.arange(self.ndish) * tab/self.ntab
        
        # get total phase offset
        dphi = self.dphi_g + dphi_tab[:, None, None, None]
        
        # create voltage beam and sum over dishes
        pbeam = self.phase_to_pbeam(dphi)
        # scale by number of dished
        # squared because this is working on intensity beam
        pbeam /= self.ndish**2

        # return beamformed power
        return pbeam
      
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import convert
    from constants import DISH_ITRF, ARRAY_ITRF, WSRT_LON, WSRT_LAT

    print("Testing beamformer")
    
    # init coordinates
    print("Initializing coordinates")   
    # due East, 45 deg above horizon
    # expect TABs parallel to horizon
    ha0, dec0 = convert.altaz_to_hadec(45*u.deg, 90*u.deg)
    
    ddec = np.linspace(-50, 50, 1000)*u.arcmin
    dha = np.linspace(-50, 50, 2000)*u.arcmin
    
    dHA, dDEC = np.meshgrid(dha, ddec)
    DEC = dec0 +dDEC
    dHA /= np.cos(DEC)
    HA = ha0 + dHA
    
    tab = 0
    freq = 1370*u.MHz

    # get dish positions
    dish_pos_itrf = DISH_ITRF['a8']
    ref_pos_itrf = ARRAY_ITRF
    
    # init beamformer
    print("Initializing beamformer")
    bf = BeamFormer(dish_pos_itrf, freq, ref_pos=ref_pos_itrf)
    bf.set_coordinates_and_phases(HA, DEC, ha0, dec0)
    
    # run beamformer
    print("Forming TAB{:02d}".format(tab))
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
            txt = 'x={:.4f}    y={:.4f}    [{:.4f}]'.format(x, y, val)
        except:
            txt = 'x={:.4f}    y={:.4f}'.format(x, y)
        return txt 


    ALT, AZ = convert.hadec_to_altaz(HA, DEC)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(HA.to(u.deg), DEC.to(u.deg), power)
    ax.set_xlabel('HA (deg)')
    ax.set_ylabel('DEC (deg)')
    ax.format_coord = formatter
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(AZ.to(u.deg), ALT.to(u.deg), power)
    ax.set_xlabel('AZ (deg)')
    ax.set_ylabel('ALT (deg)')
    ax.format_coord = formatter
    
    plt.show()
