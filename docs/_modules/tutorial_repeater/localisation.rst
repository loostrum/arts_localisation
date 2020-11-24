Localisation
^^^^^^^^^^^^

Finally, we can run the actual localisation::

    arts_run_localisation --config R3.yaml --output_folder snr --save_plots --verbose

.. note::
    Any burst/CB combination for which an output file already exists is skipped

    It is possible to overwrite settings from the config file. Run ``arts_run_localisation -h`` for an overview of all options
