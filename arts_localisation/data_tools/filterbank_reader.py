#!/usr/bin/env python
import logging
import numpy as np
from blimpy import Waterfall
from tqdm import tqdm

from arts_localisation.data_tools.spectra import Spectra
from arts_localisation.beam_models.sb_generator import SBGenerator
from arts_localisation.constants import NTAB

# init logger
logger = logging.getLogger(__name__)
# set blimpy to only log warnings
logging.getLogger("blimpy").setLevel(logging.WARNING)


class ARTSFilterbankReaderError(Exception):
    pass


class ARTSFilterbankReader:
    def __init__(self, fname, cb, ntab=NTAB, median_filter=True):
        """
        Filterbank reader for ARTS data, one file per TAB

        :param str fname: path to filterbank files, with {cb:02d} and {tab:02d} for CB and TAB indices
        :param int cb: CB index
        :param int ntab: Number of TABs (Default: NTAB from constants)
        :param bool median_filter: Enable median removal (Default: True)
        """
        self.ntab = ntab
        self.median_filter = median_filter
        self.fnames = [fname.format(cb=cb, tab=tab) for tab in range(ntab)]
        self.nfreq, self.freqs, self.nsamp, self.tsamp = self.get_fil_params(tab=0)

        self.tab_data = None
        self.startbin = None
        self.chunksize = None
        self.times = None

        # initialize the SB Generator for SC4
        self.sb_generator = SBGenerator.from_science_case(4)
        # set freq order for SB mapping
        if np.all(np.diff(self.freqs) < 0):
            logger.debug("Detected descending frequencies, reversing SB mapping")
            self.sb_generator.reversed = True

    def get_fil_params(self, tab=0):
        """
        Read filterbank parameters

        :param int tab: TAB index (Default: 0)
        :return: nfreq (int), freqs (array), nsamp (int), tsamp (float)
        """
        fil = Waterfall(self.fnames[tab], load_data=False)
        # read data shape
        nsamp, _, nfreq = fil.file_shape
        # construct frequency axis
        freqs = np.arange(fil.header['nchans']) * fil.header['foff'] + fil.header['fch1']

        return nfreq, freqs, nsamp, fil.header['tsamp']

    def read_filterbank(self, tab, startbin, chunksize):
        """
        Read a chunk of filterbank data

        :param int tab: TAB index
        :param int startbin: Index of first time sample to read
        :param int chunksize: Number of time samples to read
        :return: chunk of data with shape (nfreq, chunksize)
        """
        fil = Waterfall(self.fnames[tab], load_data=False)
        # read chunk of data
        fil.read_data(None, None, startbin, startbin + chunksize)
        # keep only time and freq axes, transpose to have frequency first
        data = fil.data[:, 0, :].T.astype(float)
        if self.median_filter:
            data -= np.median(data, axis=1, keepdims=True)
        return data

    def read_tabs(self, startbin, chunksize, tabs=None):
        """
        Read TAB data

        :param int startbin: Index of first time sample to read
        :param int chunksize: Number of time samples to read
        :param list tabs: which TABs to read (Default: all)
        """
        tab_data = np.zeros((self.ntab, self.nfreq, chunksize))
        if tabs is None:
            tabs = range(self.ntab)
        for tab in tqdm(tabs, desc="Loading TAB data"):
            tab_data[tab] = self.read_filterbank(tab, startbin, chunksize)
            if self.median_filter:
                tab_data[tab] -= np.median(tab_data[tab], axis=1, keepdims=True)
        self.tab_data = tab_data
        self.startbin = startbin
        self.chunksize = chunksize
        self.times = np.arange(chunksize) * self.tsamp

    def get_sb(self, sb):
        """
        Construct an SB. TAB data must be read before calling this method

        :param int sb: SB index
        :return: Spectra object with SB data
        """
        if self.tab_data is None:
            raise ARTSFilterbankReaderError(f"No TAB data available, run {__class__.__name__}.read_tabs first")
        # synthesize the beam
        sb_data = self.sb_generator.synthesize_beam(self.tab_data, sb)
        # return as spectra object
        return Spectra(self.freqs, self.tsamp, sb_data, starttime=self.startbin * self.tsamp, dm=0)

    def load_single_sb(self, sb, startbin, chunksize):
        """
        Convenience tool to read only a single SB and its associated TABs.
        *Note*: Any internal TAB data is cleared after calling this method

        :param int sb: SB index
        :param int startbin: Index of first time sample to read
        :param int chunksize: Number of time samples to read
        :return: Spectra object with SB data
        """
        # load the data of the required TABs
        tabs = set(self.sb_generator.get_map(sb))
        self.read_tabs(startbin, chunksize, tabs)
        # generate the SB
        sb = self.get_sb(sb)
        # remove the TAB data to avoid issues when changing startbin/chunksize in other methods
        self.tab_data = None
        return sb
