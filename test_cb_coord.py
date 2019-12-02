#!/usr/bin/env python3
#
# Verify CB coordinate conversion:
# 1. Load Gaussian CB model
# 2. Pick pointing RA, Dec
# 3. Calculate RA, Dec grid 
# 4. Get peak sensitivity pixel of each CB
# 5. Get corresponding RA, Dec
# 6. Get offset from pointing center
# 7. Compare to CB offset definition file

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import convert
from constants import THETAMAX_CB, PHIMAX_CB, NTHETA_CB, NPHI_CB


if __name__ == '__main__':
    ra_src = 83.6332208333*u.deg
    dec_src = 22.01446111111111*u.deg
    fmodel = 'models/all_cb_gauss.npy'
    flayout = 'square_39p1.cb_offsets'

    print("cos(Dec) = {:.2f}".format(np.cos(dec_src)))

    print("Loading CB model")
    cb_model = np.load(fmodel)
    ncb, nphi, ntheta = cb_model.shape
    assert ntheta == NTHETA_CB
    assert nphi == NPHI_CB

    print("Generating coordinate grid")
    theta = np.linspace(-THETAMAX_CB, THETAMAX_CB, NTHETA_CB) * u.arcmin
    phi = np.linspace(-PHIMAX_CB, PHIMAX_CB, NPHI_CB) * u.arcmin
    theta = theta.to(u.deg)
    phi = phi.to(u.deg)
    # central point = pointing center
    # phi = dec: apply offset
    dec = dec_src + phi
    # theta = ra: apply offset scaled by cos dec (avg dec)
    ra = ra_src + theta/np.cos(dec_src)
    X, Y = np.meshgrid(ra, dec)
    # resolution elements
    res_theta = np.diff(theta)[0].to(u.deg).value
    res_phi = np.diff(phi)[0].to(u.deg).value


    print("Getting CB peaks")
    # RA, Dec peak of each CB
    peaks = np.zeros((ncb, 2))
    for cb in range(ncb):
        # get peak location
        peak = np.argmax(cb_model[cb])
        # get proper indices, instead of flattened
        peak = np.unravel_index(peak, cb_model[cb].shape)
        ra_peak = X[peak].to(u.deg).value
        dec_peak = Y[peak].to(u.deg).value
        peaks[cb] = (ra_peak, dec_peak)

    print("Calculating offsets")
    # load offset definition file
    cb_offsets = np.loadtxt('square_39p1.cb_offsets', usecols=[1, 2], delimiter=',')
    # RA, Dec offset in deg
    print("CB  dRA  dDec")
    for cb in range(ncb):
        # RA offset from CB00, scale by cos dec
        ra_off = (peaks[cb][0] - peaks[0][0]) * np.cos(dec_src)
        # Dec offset from CB00
        dec_off = (peaks[cb][1] - peaks[0][1])
        # compare to layout definition
        dra = ra_off - cb_offsets[cb][0]
        ddec = dec_off - cb_offsets[cb][1]
        # difference in number of resolution elements
        print(f"{cb:02d}  {dra/res_theta:.0f}  {ddec/res_phi:.0f}")
