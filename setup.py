#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='arts_localisation',
      version='0.1',
      description='Localisation of ARTS transients',
      url='http://github.com/loostrum/arts_localisation',
      author='Leon Oostrum',
      author_email='oostrum@astron.nl',
      license='GPLv3',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy',
                        'astropy',
                        'blimpy',
                        'pyyaml',
                        'scipy',
                        'matplotlib',
                        'tqdm'],
      include_package_data=True,
      entry_points={'console_scripts': ['arts_calc_snr=arts_localisation.calc_snr:main',
                                        'arts_run_localisation=arts_localisation.run_localisation:main']},
      # scripts=['bin/darc_start_all_services',
      #          'bin/darc_stop_all_services',
      #          'bin/darc_start_master',
      #          'bin/darc_stop_master',
      #          'bin/darc_kill_all'])
      )
