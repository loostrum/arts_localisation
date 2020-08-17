#!/usr/bin/env python3
# Calculate parallactic angle from ha, dec

import argparse
import astropy.units as u

from arts_localisation.tools import hadec_to_proj, hadec_to_par

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ha', required=True, type=float, help="Hour angle in degrees")
    parser.add_argument('--dec', required=True, type=float, help="Declination in degrees")

    args = parser.parse_args()

    ha = args.ha * u.deg
    dec = args.dec * u.deg

    proj = hadec_to_proj(ha, dec)
    parang = hadec_to_par(ha, dec)

    print(f"Projection angle (E-W baseline projection; also SB rotation in AltAz): {proj:.2f}")
    print(f"Parallactic angle (SB rotation): {parang:.2f}")
