One-off drift scan localisation
--------------------------------
This tutorial assumes the user is familiar with the localisation process. For a tutorial covering all steps,
refer to :ref:`_modules/tutorial_repeater/intro:Repeater localisation`.

In this tutorial, we will localize the first-ever FRB detected by ARTS, FRB 190709. This burst was detected during a
drift scan of a calibration source. This means the telescope were not tracking one (RA, Dec) point, but stationary and
effectively "tracking" a single (HA, Dec) point, where HA and Dec are the *local* hour angle and declination, respectively.

FRB 190709 was detected in CB10. The available data are thus the 12 TABs of CB10::

    (py36) leon@zeus:FRB190709$ ls data
    CB10_10.0sec_dm0_t0657_sb-1_tab00.fil  CB10_10.0sec_dm0_t0657_sb-1_tab06.fil
    CB10_10.0sec_dm0_t0657_sb-1_tab01.fil  CB10_10.0sec_dm0_t0657_sb-1_tab07.fil
    CB10_10.0sec_dm0_t0657_sb-1_tab02.fil  CB10_10.0sec_dm0_t0657_sb-1_tab08.fil
    CB10_10.0sec_dm0_t0657_sb-1_tab03.fil  CB10_10.0sec_dm0_t0657_sb-1_tab09.fil
    CB10_10.0sec_dm0_t0657_sb-1_tab04.fil  CB10_10.0sec_dm0_t0657_sb-1_tab10.fil
    CB10_10.0sec_dm0_t0657_sb-1_tab05.fil  CB10_10.0sec_dm0_t0657_sb-1_tab11.fil

Click on one of the below links to go directly to that step, or simply click "Next" in the bottom right
corner to go to the next step.

.. toctree::
   :maxdepth: 1

   snr_config
   snr
   localisation_config
   localisation
