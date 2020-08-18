#!/usr/bin/env python
#
# synthesized beam generator

import os
import numpy as np


class SBGeneratorException(Exception):
    pass


class SBGenerator:

    def __init__(self, fname=None, science_case=None):
        """
        Synthesised beam layout and generator

        :param str fname: path to SB table
        :param science_case: ARTS science case (3 or 4)
        """
        self.sb_table = None
        self.nsub = None
        self.numtab = None
        self.numsb = None
        self.__reversed = None

        # set config
        self.table_folder = os.path.dirname(os.path.abspath(__file__))
        self.table = {'sc3': "sbtable-9tabs-114sbs-f1370.txt",
                      'sc4': "sbtable-sc4-12tabs-71sbs.txt"}

        self.numtab = {'sc3': 9, 'sc4': 12}
        self.numsb = {'sc3': 114, 'sc4': 71}

        # Get full path to SB table
        if fname and not os.path.isfile(fname):
            fname = os.path.join(self.table_folder, fname)
        elif science_case:
            fname = os.path.join(self.table_folder, self.table[f'sc{science_case:.0f}'])
        self.science_case = science_case
        self.fname = fname

        # load the table
        self._load_table()

    @property
    def reversed(self):
        """
        Whether or not the SB table is reversed for use on filterbank data

        :return: reversed (bool)
        """
        return self.__reversed

    @reversed.setter
    def reversed(self, state):
        """
        Reverse the SB table for use on filterbank data

        :param bool state: whether or not to reverse the table
        """
        if self.__reversed == state:
            # already in desired state
            return
        else:
            # reverse the table
            self.sb_mapping = self.sb_mapping[:, ::-1]
            # store state
            self.__reversed = state

    @classmethod
    def from_table(cls, fname):
        """
        Initalize with provided SB table

        :param str fname: Path to SB table
        :return: SBGenerator object
        """
        return cls(fname=fname)

    @classmethod
    def from_science_case(cls, science_case):
        """
        Initalize default table for given science cases

        :param int science_case: science case (3 or 4)
        :return: SBGenerator object
        """
        if science_case not in (3, 4):
            raise SBGeneratorException(f'Invalid science case: {science_case}')
        return cls(science_case=science_case)

    def _load_table(self):
        """
        Load the SB table
        """
        self.sb_mapping = np.loadtxt(self.fname, dtype=int)
        numsb, self.nsub = self.sb_mapping.shape
        # do some extra checks if table is loaded based on science case
        # otherwise this is the users's responsibility
        if self.science_case:
            # check that the number of SBs is what we expect and TABs are not out of range
            if self.science_case == 3:
                expected_numtab = self.numtab['sc3']
                expected_numsb = self.numsb['sc3']
            else:
                expected_numtab = self.numtab['sc4']
                expected_numsb = self.numsb['sc4']
            # number of SBs and TABs

            # verify number of SBs
            if not expected_numsb == numsb:
                raise SBGeneratorException("Number of SBs ({}) not equal to expected value ({})".format(numsb,
                                                                                                        expected_numsb))
            # verify max TAB index, might be less than maximum if not all SBs are generated
            if not np.amax(self.sb_mapping) < expected_numtab:
                raise SBGeneratorException("Maximum TAB ({}) higher than maximum for this science case ({})".format(
                                           max(self.sb_mapping), expected_numtab))
            self.numsb = numsb
            self.numtab = expected_numtab
        else:
            self.numsb = numsb
            # assume the highest TAB index is used in the table
            self.numtab = np.amax(self.sb_mapping) + 1
        self.__reversed = False

    def get_map(self, sb):
        """
        Return mapping of requested SB

        :param int sb: beam to return mapping for
        :return: SB mapping for requested beam
        """
        return self.sb_mapping[sb]

    def synthesize_beam(self, data, sb):
        """
        Synthesise beam

        :param array data: TAB data with shape [TAB, freq, others(s)]
        :param int sb: SB index
        :return: SB data with shape [freq, other(s)]
        """
        ntab, nfreq, = data.shape[:2]
        # verify that SB index is ok
        if not sb < self.numsb:
            raise SBGeneratorException("SB index too high: {}; maximum is {}".format(sb, self.numsb - 1))
        if not sb >= 0:
            raise SBGeneratorException("SB index cannot be negative")
        # verify that number of TABs is ok
        if not ntab == self.numtab:
            raise SBGeneratorException("Number of TABs ({}) not equal to expected number of TABs ({})".format(
                                       ntab, self.numtab))
        # verify number of channels
        if nfreq % self.nsub:
            raise SBGeneratorException("Error: Number of subbands ({}) is not a factor of "
                                       "number of channels ({})".format(self.nsub, nfreq))

        nchan_per_subband = int(nfreq / self.nsub)
        # the output beam has the same shape as input apart from the missing TAB column
        beam = np.zeros(data.shape[1:])
        for subband, tab in enumerate(self.sb_mapping[sb]):
            # get correct subband of correct tab and add it to raw SB
            # after vsplit, shape is (nsub, nfreq/nsub, ntime) -> simply [subband] gets correct subband
            # assign to subband of sb
            beam[subband * nchan_per_subband:(subband + 1) * nchan_per_subband] = np.vsplit(data[tab], self.nsub)[subband]
        return beam
