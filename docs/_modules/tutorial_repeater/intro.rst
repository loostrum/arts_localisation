Repeater localisation
--------------------------------
We are going to localize the repeating FRB 180916.J0158+65 (also known as R3) using 3 bursts.
The telescope was directly pointed at the source, so the detections happened in compound beam zero (CB00).
For this example, we are not going to use the neighbouring beams. Let's first look at the available data::

    (py36) leon@zeus:R3$ ls data
    CB00_10.0sec_dm0_t02063_sb-1_tab00.fil  CB00_10.0sec_dm0_t03437_sb-1_tab00.fil  CB00_10.0sec_dm0_t03610_sb-1_tab00.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab01.fil  CB00_10.0sec_dm0_t03437_sb-1_tab01.fil  CB00_10.0sec_dm0_t03610_sb-1_tab01.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab02.fil  CB00_10.0sec_dm0_t03437_sb-1_tab02.fil  CB00_10.0sec_dm0_t03610_sb-1_tab02.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab03.fil  CB00_10.0sec_dm0_t03437_sb-1_tab03.fil  CB00_10.0sec_dm0_t03610_sb-1_tab03.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab04.fil  CB00_10.0sec_dm0_t03437_sb-1_tab04.fil  CB00_10.0sec_dm0_t03610_sb-1_tab04.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab05.fil  CB00_10.0sec_dm0_t03437_sb-1_tab05.fil  CB00_10.0sec_dm0_t03610_sb-1_tab05.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab06.fil  CB00_10.0sec_dm0_t03437_sb-1_tab06.fil  CB00_10.0sec_dm0_t03610_sb-1_tab06.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab07.fil  CB00_10.0sec_dm0_t03437_sb-1_tab07.fil  CB00_10.0sec_dm0_t03610_sb-1_tab07.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab08.fil  CB00_10.0sec_dm0_t03437_sb-1_tab08.fil  CB00_10.0sec_dm0_t03610_sb-1_tab08.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab09.fil  CB00_10.0sec_dm0_t03437_sb-1_tab09.fil  CB00_10.0sec_dm0_t03610_sb-1_tab09.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab10.fil  CB00_10.0sec_dm0_t03437_sb-1_tab10.fil  CB00_10.0sec_dm0_t03610_sb-1_tab10.fil
    CB00_10.0sec_dm0_t02063_sb-1_tab11.fil  CB00_10.0sec_dm0_t03437_sb-1_tab11.fil  CB00_10.0sec_dm0_t03610_sb-1_tab11.fil

There are 36 files in total: 12 tied-array beams (TABs) for each of the 3 bursts. Indeed, we only have data for CB00.

Click on one of the below links to go directly to that step, or simply click "Next" in the bottom right
corner to go to the next step.

.. toctree::
   :maxdepth: 1

   snr_config
   snr
   localisation_config
   localisation
