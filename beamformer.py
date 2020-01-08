#!/usr/bin/env python3
#
# beamformer tool to create TAB pattern

import itertools

import numpy as np
import astropy.units as u
import astropy.constants as const

from constants import NTAB


class BeamFormer(object):
    
    def __init__(self, dish_pos, freqs, ntab=NTAB, ref_pos=0):
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit
        self.dish_pos = dish_pos - ref_pos
        self.freqs = freqs
        self.ntab = ntab
        
    def _get_uvw(self, ha, dec):
        rot_matrix = np.array([[np.sin(ha), np.cos(ha), 0], 
                              [-np.sin(dec)*np.cos(ha), np.sin(dec)*np.sin(ha), np.cos(dec)], 
                              [np.cos(dec)*np.cos(ha), -np.cos(dec)*np.sin(ha), np.sin(dec)]])
        
#         uvw = np.zeros_like(self.dish_pos)
        uvw = np.zeros((8, 3))
        for i, dish in enumerate(self.dish_pos):
#             dish = np.array([0, (dish/lamb).to(1), 0])
            uvw[i] = np.inner(rot_matrix, dish.value)
        # scale by wavelength
        uvw = uvw[:, None, :] * (self.freqs / const.c).to(1/dish.unit).value

        # shape is now (ndish, nfreq, 3)
        return uvw
    
    def beamform(self, grid_ha, grid_dec, ha0, dec0, tab=0):
        assert grid_ha.shape == grid_dec.shape

        # calculate the direction cosines l, m, n in terms of offset from phase center
        # see http://math_research.uct.ac.za/~siphelo/admin/interferometry/
        # 3_Positional_Astronomy/3_4_Direction_Cosine_Coordinates.html
        dha = grid_ha - ha0
        dra = -dha
        ddec = grid_dec - dec0
        
#         # l: check sign with orientation of tab in hadec frame
#         l = -np.cos(grid_dec)*np.sin(dha)
#         m = np.sin(grid_dec)*np.sin(dec0) - np.cos(grid_dec)*np.cos(dec0)*np.cos(dha)
#         n = np.sqrt(1 - l**2 - m**2)

#         # l // RA
#         l = dra.to(u.rad).value
#         # m // Dec
#         m = ddec.to(u.rad).value
#         n = np.sqrt(1 - l**2 - m**2)
        
#         lmn = np.transpose([l, m, n])
    
        # calculate uww of baselines
        # shape is (ndish, nfreq, 3)
        uvw = self._get_uvw(ha0, dec0)
        
        # init output voltage beam
        vbeam = np.zeros((self.freqs.shape[0], grid_ha.shape[0], grid_ha.shape[1]), dtype=complex)
        
        # loop over baselines
        for baseline in uvw:
            uu, vv, ww = baseline.T
            # u,v,w have shape (nfreq)
            # phase offset with respect to phase center
#             dphi = (uu[:, None, None] * dra.to(u.rad).value * np.cos(grid_dec) + vv[:, None, None] * ddec.to(u.rad).value).to(1).value
            dphi = (uu[:, None, None] * np.sin(dra) * np.cos(grid_dec) + vv[:, None, None] * np.sin(ddec)).to(1).value
            # add to voltage beam
            np.exp(1j*2*np.pi*dphi)
            vbeam += np.exp(1j*2*np.pi*dphi)

        # scale by number of baselines
        vbeam /= len(uvw)
        
        
        
#         # phase
#         phase = np.inner(uvw, lmn)
#         # subtract phase of reference pos (delay = w coordinate)
#         for i in range(len(phase)):
#             phase[i] -= uvw[i, 2]
        
        # calculate output E field, normalized by number of signals
#         e_field = np.exp(1j*2*np.pi*phase).mean(axis=0)

        # return beamformed power
        return np.abs(vbeam)**2
        
        
class BeamFormer_1D(object):

    def __init__(self, dish_pos, theta_proj=0*u.deg, freqs=1500*u.MHz, ntab=12):
        self.theta_proj = theta_proj
        if not isinstance(freqs.value, np.ndarray):
            freqs = np.array([freqs.value]) * freqs.unit
        self.dish_pos = dish_pos
        self.freqs = freqs
        self.ntab = ntab

    def __dphi(self, dtheta, baseline, grid=False):
        """
        Compute phase difference for given offset and baseline
        dphi = freq * dx /c
        :param dtheta: offsets
        :param baseline: baseline length
        :param grid: whether or not dheta is a 2D grid
        :return: phases
        """

        if grid:
            dphi = self.freqs[..., None, None] * baseline / const.c * \
                   (np.sin(self.theta_proj + dtheta) -
                    np.sin(self.theta_proj))
        else:
            dphi = self.freqs[..., None] * baseline / const.c * \
                   (np.sin(self.theta_proj + dtheta) -
                    np.sin(self.theta_proj))

        return dphi.to(1).value

    def beamform(self, dtheta, ref_dish=0, tab=0):
        """
        Compute total power for given offsets and dish positions
        :param dtheta: E-W offsets
        :param ref_dish: reference dish
        :param tab: TAB index
        :return: beamformed power, same shape as dtheta
        """
        # check number of dimensions in dtheta
        if len(dtheta.shape) == 2:
            grid = True
        else:
            assert len(dtheta.shape) == 1
            grid = False

        # initalize output E field
        if grid:
            e_tot = np.zeros((self.freqs.shape[0], dtheta.shape[0], dtheta.shape[1]), dtype=complex)
        else:
            e_tot = np.zeros((self.freqs.shape[0], dtheta.shape[0]), dtype=complex)

        baselines = self.dish_pos - self.dish_pos[ref_dish]
        for i, baseline in enumerate(baselines):
            # calculate tab phase offset
            tab_dphi = float(i) * tab/self.ntab
            # calculate geometric phase offset
            geometric_dphi = self.__dphi(dtheta, baseline, grid=grid)
            dphi = geometric_dphi + tab_dphi
            # store as complex value (amplitude of E field = 1)
            e_tot += np.exp(1j*2*np.pi*dphi)

        # normalize by number of signals
        e_tot /= len(baselines)

        # convert E field to intensity
        i_tot = np.abs(e_tot)**2
        return i_tot

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import convert
    from constants import DISH_ITRF, ARRAY_POS_ITRF, WSRT_LON, WSRT_LAT

    print("Testing beamformer")
    
    # init coordinates
    print("Initializing coordinates")
#     ha0 = 0*15*u.deg
#     dec0 = 88*u.deg
    
    
    ha0, dec0 = convert.altaz_to_hadec(45*u.deg, 90*u.deg)
    
    ddec = np.linspace(-50, 50, 1000)*u.arcmin
    dra = np.linspace(-50, 50, 1000)*u.arcmin
    dha = -dra
    
    HA, DEC = np.meshgrid(dha, ddec)
    DEC += dec0
    HA = ha0 + HA/np.cos(DEC)
    # cos(dec) correction
    
    
    tab = 0
    freq = 1370*u.MHz

    # get dish positions
    dish_pos_itrf = DISH_ITRF['a8']
    # rotate by -latitude to get reference to local meridian instead of Greenwhich mean meridian
    rot_matrix = np.array([[np.cos(-WSRT_LON), -np.sin(-WSRT_LON), 0],
                           [np.sin(-WSRT_LON), np.cos(-WSRT_LON), 0],
                           [0, 0, 1]])
    dish_xyz = np.matmul(rot_matrix, dish_pos_itrf.T).T
    array_xyz = np.matmul(rot_matrix, ARRAY_POS_ITRF)
    # dish_xyz is very close to [0, 144*i, 0]
    # array pos = approx. RT8
    
    # init beamformer
    print("Initializing beamformer")
    bf = BeamFormer(dish_xyz, freq, ref_pos=array_xyz)
    
    # run beamformer
    print("Forming TAB{:02d}".format(tab))
    power = bf.beamform(HA, DEC, ha0, dec0, tab)
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