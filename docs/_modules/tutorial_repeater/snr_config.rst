Configuration file
^^^^^^^^^^^^^^^^^^
The first step is to create a YAML-format configuration file containing the relevant observation parameters and settings for
S/N determination.

The ``global`` section of the configuration file contains general information about the data:

#. ``fmin_data``: lowest frequency of the data (MHz).
#. ``bandwidth``: bandwidth of the data (MHz).
#. ``fmin``: lowest frequency to use (MHz).
#. ``fmax``: highest frequency to use (MHz).
#. ``snrmin``: S/N threshold.

The data values can be read from the filterbank files, using e.g. PRESTO's `readfile`.
All data below/above ``fmin``/``fmax`` will be ignored.
These values can also be set per burst. In this example we will do the latter, so the ``fmin`` and
``fmax`` settings are ignored.
``snrmin`` sets the S/N threshold to actually use in the localisation; any beams where the measured S/N is lower
are ignored. It is a global setting, because the S/N calculation also uses it (but only for visualization).
Here the S/N threshold is set to 6, which was chosen based on the output diagnostic plots of the S/N calculation.

The ``snr`` section contains the global settings specific to S/N determination:

#. ``window_load``: chunk of data to load, must be at least twice the dispersion delay (s).
#. ``window_zoom``: chunk of data to keep after dedispersion (s).
#. ``dm``: dispersion measure (pc/cc).
#. ``width_max``: maximum boxcar width used in S/N calculation.

The DM of R3 is roughly 350 pc/cc. This corresponds to a ~700 ms delay across the Apertif band. Therefore we
set the load window to 2 seconds. The zoom window is set to 0.5 seconds. The value of the latter doesn't matter
that much, as long as it is much wider than the (dedispsersed) pulse itself. The maximum boxcar width is set
to 100, corresponding to 8 ms. The bursts used in this example are all more narrow than that.

Now we need to create a section for each burst. The names of these sections can be anything.
Here I use the burst arrival date and arrival
time within the observation as name.

.. note:: If the burst name contains only digits and underscores, it must be quoted

For each burst, we then need to add these settings:

#. ``filterbank``: path to filterbank file, replace CB index by {cb} and TAB index by {tab}.
#. ``cbs``: list of CBs to determine S/N for.
#. ``neighbours``: Whether or not to calculate the S/N for the neighbours of the CBs in the CB list as well.
#. ``toa_filterbank``: Rough arrival time of the burst within the filterbank file (s). Optional; If not present, the burst is assumed to be in the centre of the file.
#. ``main_cb``: CB of main detection (used to determine exact burst arrival time in filterbank data).
#. ``main_sb``: SB of main detection (used to determine exact burst arrival time in filterbank data).
#. ``fmin``: as in global section, but only for this burst.
#. ``fmax``: as in global section, but only for this burst.

As only CB00 data are available, the CB list is simply ``[0]`` and neighbours should be set to ``False``.
We are using filterbank snippets, where the burst is always near the centre, so we do not have to set ``toa_filterbank``.
The bursts were detected in the pointing centre, i.e. in SB35 of CB00. These values are set for ``main_sb`` and ``main_cb``.
The ``fmin`` and ``fmax`` values were not set globally, so if we need to select a limited frequency range we should do
so here. Each burst was inspected by eye to select the frequency range.

.. note:: If ``fmin`` and ``fmax`` are never set, all frequencies are used.


After these steps, the configuration file looks like this:

.. code-block:: yaml

    global:
        fmin_data: 1219.70092773
        bandwidth: 300
        snrmin: 6

    snr:
        window_load: 2.0
        window_zoom: 0.5
        dm: 349.2
        width_max: 100

    '20200511_3610':
        fmin: 1350
        filterbank: 'data/CB{cb}_10.0sec_dm0_t03610_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35

    '20200528_2063':
        fmin: 1350
        filterbank: 'data/CB{cb}_10.0sec_dm0_t02063_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35

    '20200527_3437':
        fmax: 1450
        filterbank: 'data/CB{cb}_10.0sec_dm0_t03437_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35
