Tutorial
========

In short, the localisation consists of the following steps:

#. Note in which CBs the FRB was detected.
#. Save a snippet for the detection CBs, as well as for all surrounding CBs.
#. Create a configuration file with the settings to use to S/N determination
#. Run ``arts_calc_snr``
#. Add localisation settings to the configuration file, using coarse resolution
#. Run ``arts_run_localisation``
#. Repeat the previous step with different settings until the entire uncertainty region is covered
#. Re-run the localisation at high resolution

Two examples are provided:

.. toctree::
    :maxdepth: 1

    tutorial_repeater/intro
    tutorial_oneoff/intro





