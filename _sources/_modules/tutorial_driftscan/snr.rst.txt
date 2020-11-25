S/N measurement
^^^^^^^^^^^^^^^
The configuration file now contains all information needed to measure the S/N in each synthesised beam (SB)
of each CB of each burst. The following command executes the S/N calculation and creates some diagnostic plots::

    arts_calc_snr --config FRB190709.yaml --output_folder snr --save_plots --verbose

The following files are produced::

    (py36) leon@zeus:FRB190709$ ls snr
    FRB190709_burst00_CB10_SNR.txt  FRB190709_burst00_SNR.pdf

As there is only one burst in one CB, there are just two output files: the S/N array of CB10, and a figure showing
the S/N in each SB:

.. image:: ../../_images/FRB190709_burst00_SNR.png
    :width: 600
    :alt: S/N of FRB 190709

