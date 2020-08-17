#!/usr/bin/env python

import logging
import numpy as np
from blimpy import Waterfall
from tqdm import tqdm

from .spectra import Spectra
from beam_models import SBGenerator


# init logger
logger = logging.getLogger(__name__)
# set blimpy to only log warnings
logging.getLogger("blimpy").setLevel(logging.WARNING)


class ARTSFilterbankReaderError(Exception):
    pass


class ARTSFilterbankReader:
    def __init__(self, fname, cb, ntab=12):
        self.ntab = ntab
        self.fnames = [fname.format(cb=cb, tab=tab) for tab in range(ntab)]
        self.nfreq, self.freqs, self.nsamp, self.tsamp = self.get_fil_params(tab=0)

        self.tab_data = None
        self.startbin = None
        self.chunksize = None
        self.times = None

        # initialize the SB Generator for SC4
        self.sb_generator = SBGenerator.from_science_case(4)

    def get_fil_params(self, tab):
        fil = Waterfall(self.fnames[tab], load_data=False)
        # read data shape
        nsamp, _, nfreq = fil.file_shape
        # construct frequency axis
        freqs = np.arange(fil.header['nchans']) * fil.header['foff'] + fil.header['fch1']

        return nfreq, freqs, nsamp, fil.header['tsamp']

    def read_filterbank(self, tab, startbin, chunksize):
        fil = Waterfall(self.fnames[tab], load_data=False)
        # read chunk of data
        fil.read_data(None, None, startbin, startbin + chunksize)
        # keep only time and freq axes, transpose to have frequency first
        return fil.data[:, 0, :].T

    def read_tabs(self, startbin, chunksize, tabs=None):
        tab_data = np.zeros((self.ntab, self.nfreq, chunksize))
        if tabs is None:
            tabs = range(self.ntab)
        for tab in tqdm(tabs, desc="Loading TAB data"):
            tab_data[tab] = self.read_filterbank(tab, startbin, chunksize)
        self.tab_data = tab_data
        self.startbin = startbin
        self.chunksize = chunksize
        self.times = np.arange(chunksize) * self.tsamp

    def get_sb(self, sb):
        if self.tab_data is None:
            raise ARTSFilterbankReaderError(f"No TAB data available, run {__class__.__name__}.read_tabs first")
        # synthesize the beam
        sb_data = self.sb_generator.synthesize_beam(self.tab_data, sb)
        # return as spectra object
        return Spectra(self.freqs, self.tsamp, sb_data, starttime=self.startbin * self.tsamp, dm=0)

    def load_single_sb(self, sb, startbin, chunksize):
        # load the data of the required TABs
        tabs = set(self.sb_generator.get_map(sb))
        self.read_tabs(startbin, chunksize, tabs)
        # generate the SB
        sb = self.get_sb(sb)
        # remove the TAB data to avoid issues when changing startbin/chunksize in other methods
        self.tab_data = None
        return sb
