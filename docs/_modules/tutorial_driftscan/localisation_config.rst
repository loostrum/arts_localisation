Update configuration file for localisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Updating the localisation file for localisation is mostly the same as for the
:ref:`_modules/tutorial_repeater/intro:Repeater localisation` tutorial. The main difference is that
FRB 190709 was found in a drift scan, so we need to specify the HA and Dec, instead of RA and Dec.

.. note::
    The filterbank header always lists the pointing as RA and Dec. The RA field really contains the HA in case
    of a drift scan. The Dec is then the *local* Dec.

The scripts automatically convert HA and Dec into J2000 RA and Dec. Both the localisation grid and
CB/telescope pointing coordinates can be given in the HADEC frame. For CB/telescope pointing this works out of the box:
the burst arrival time is used to convert HADEC to RADEC. In the case of the localisation grid, however, this
is not the case. There might be several bursts with different arrival times, so the code does not know which time to use.
Therefore, the user has to manually put in the timestamp to use for converting the localisation grid HADEC to RADEC.
In this case, we of course want this timestamp to coincide with the burst arrival time.

.. note::
    To calculate the burst arrival time from the observation start time and arrival time of the burst within the observation,
    use something like this::

        >>> from astropy.time import Time, TimeDelta
        >>> (Time('2019-07-09T05:02:24.064') + TimeDelta(657.977, format='sec')).isot
        '2019-07-09T05:13:22.041'



The final configuration file looks like this:

.. code-block:: yaml

    global:
        fmin_data: 1219.70092773
        bandwidth: 300
        snrmin: 5

    snr:
        window_load: 2.0
        window_zoom: 0.5
        dm: 663.1
        width_max: 100

    localisation:
        ha: -12.52167
        dec: 31.95253
        time: '2019-07-09T05:13:22.041'

        size: 35
        resolution: 10
        cb_model: gauss

    burst00:
        fmin: 1300
        filterbank: 'data/CB{cb}_10.0sec_dm0_t0657_sb-1_tab{tab}.fil'
        cbs: [10]
        neighbours: False
        main_cb: 10
        main_sb: 36

        tstart: '2019-07-09T05:02:24.064'
        toa: 657.977
        reference_cb: CB10

        CB10:
            ha: -12.52167
            dec: 31.95253
            snr_array: 'snr/FRB190709_burst00_CB10_SNR.txt'
