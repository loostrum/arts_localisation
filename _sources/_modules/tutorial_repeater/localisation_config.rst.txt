Update configuration file for localisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A few sections need to be add to the configuration file to be able to run the localisation.

First, a new ``localisation`` section should be added containing:

#. ``ra``: right ascension of centre of localisation grid (decimal degrees).
#. ``dec``: declination of centre of localisation grid (decimal degrees).
#. ``size``: size of one side of the (square) localisation grid (arcmin).
#. ``resolution``: resolution of the localisation grid (arcsec).
#. ``cb_model``: type of CB model to use, options are gauss, airy, or real.
#. ``source_ra``: right ascension of the source (decimal degrees). Optional; only used for visualization
#. ``source_dec``: declination of the source (decimal degrees). Optional; only used for visualization

Initially, we centre the localisation grid on the centre of the CB, which in this example happens to coincide
with the real source position.
The grid size is chosen to be 35 arcminutes, which covers the CB roughly out to its
half-power width. The resolution is set to 10 arcseconds. This is roughly half the half-power with of a single SB.
A higher resolution is needed for accurate localisation, but cannot be used in combination with a large grid size.
The limitation here is memory usage, and potentially speed: With the settings chosen here, the beam model
is calculated for roughly 270 million points, requiring roughly 2GB of memory just to hold those data.
Clearly, increasing the resolution to 1 arcsecond is not feasible: that would require 200GB of memory.
Instead, the localisation is first run at lower resolution, then again after zooming in on the are of interest.
The lowered grid size then allows for a higher resolution.
Finally, we set the ``cb_model`` to Gaussian (the used model has little effect on the result), and put in the known
coordinates of FRB 180916.J0158+65.

Next, more parameters need to be added to the section of each burst:

#. ``tstart`` and ``toa`` **or** ``tarr``: start time of observation (ISOT format) plus arrival time within observation (s), or just burst arrival time (ISOT).
#. ``reference_cb``: subsection name of the CB to use as reference point (actually, the SB with highest S/N within
   that CB is used as reference). This is typically the same CB as the ``main_cb``.
#. ``pointing_ra``: right ascension of the telescope pointing (decimal degrees). Optional; Instead, the pointing of each CB can be given separately
#. ``pointing_dec``: declination of the telescope pointing (decimal degrees). Optional; Instead, the pointing of each CB can be given separately

For this example, the ``tstart`` + ``toa`` combination is used. The reference CB is always CB00. Only one CB is used,
so we do not use the pointing settings, but put in the CB pointing in the specific subsections (see below).

.. warning::
    Do **not** use the start time as noted in the filterbank header and/or parameterset: The real start time is synced to a
    1.024 s interval upon observation start, slightly shifting the intended start time as needed. This change is not
    reflected in the filterbank header nor parameterset. To get the real start time, look at the start packet value
    in the observation log, e.g.::

        oostrum@arts041:~$ grep 'Start packet' /home/arts/observations/log/20200511/2020-05-11-07:36:22.R3/fill_ringbuffer_i.00
        Start packet = 1241548892456250

    The start packet is the unix time stamp of the start time in units of native Apertif time samples, i.e. one second
    equals 781250 samples.
    The value can be converted to ISOT format with e.g. astropy::

        >>> from astropy.time import Time
        >>> Time(1241548892456250 / 781250., format='unix').isot
        2020-05-11T07:36:22.344


Then, a subsection needs to be added for each CB. The names of these sections do not matter, but do make sure that
the ``reference_cb`` value matches the subsection name of the CB with the main detection. The subsections should contain:

#. ``ra``: right ascension of this CB (not required if ``pointing_ra`` is given)
#. ``dec``: declination of this CB (not required if ``pointing_dec`` is given)
#. ``snr_array``: path to file containing S/N in each SB, as output by `arts_calc_snr`
#. ``sefd``: system-equivalent flux density of this CB (Jy). One value is used to scale the sensitivity of the
   entire CB (i.e. no frequency dependence). Typically this value is extracted from drift scan results. If not given, a value of 85 Jy is used.

We put in the CB coordinates as given in the filterbank header. The S/N array files contain the name of the burst and CB,
so they can easily be linked to the correct burst/CB within the configuration file. The SEFDs were taken from the drift scans taken closest
in time to the burst detections.

After adding these sections, the complete configuration file looks like this:

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

    localisation:
        source_ra: 29.50333
        source_dec: 65.71675

        ra: 29.50333
        dec: 65.71675

        size: 35
        resolution: 10
        cb_model: gauss

    '20200511_3610':
        fmin: 1350
        filterbank: 'data/CB{cb}_10.0sec_dm0_t03610_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35

        tstart: "2020-05-11T07:36:22.0"
        toa: 3610.84
        reference_cb: 'CB00'

        CB00:
            ra: 29.50333
            dec: 65.71675
            snr_array: 'snr/R3_20200511_3610_CB00_SNR.txt'
            sefd: 80

    '20200528_2063':
        fmin: 1350
        filterbank: 'data/CB{cb}_10.0sec_dm0_t02063_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35

        tstart: "2020-05-28T05:13:48"
        toa: 2063.73
        reference_cb: 'CB00'

        CB00:
            ra: 29.50333
            dec: 65.71675
            snr_array: 'snr/R3_20200528_2063_CB00_SNR.txt'
            sefd: 80

    '20200527_3437':
        fmax: 1450
        filterbank: 'data/CB{cb}_10.0sec_dm0_t03437_sb-1_tab{tab}.fil'
        cbs: [0]
        neighbours: False
        main_cb: 0
        main_sb: 35

        tstart: "2020-05-27T13:37:55"
        toa: 3437.12
        reference_cb: 'CB00'

        CB00:
            ra: 29.50333
            dec: 65.71675
            snr_array: 'snr/R3_20200527_3437_CB00_SNR.txt'
            sefd: 80
