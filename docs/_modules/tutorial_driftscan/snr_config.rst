Configuration file
^^^^^^^^^^^^^^^^^^
The first step is to create a YAML-format configuration file containing the relevant observation parameters and settings for
S/N determination. This process is identical to the :ref:`_modules/tutorial_repeater/intro:Repeater localisation` tutorial.
FRB 190709 was detected in SB36 of CB10 at a DM of 663.1 pc/cc. After a first run with a S/N threshold of 6, the
threshold was lowered to 5 based on the output figure (see next step)

The configuration file looks like this:

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

    burst00:
        fmin: 1300
        filterbank: 'data/CB{cb}_10.0sec_dm0_t0657_sb-1_tab{tab}.fil'
        cbs: [10]
        neighbours: False
        main_cb: 10
        main_sb: 36
