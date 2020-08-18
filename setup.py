#!/usr/bin/env python

from setuptools import setup, find_packages
from arts_localisation.__version__ import version


setup(name='arts_localisation',
      version=version,
      description='Localisation of ARTS transients',
      url='http://github.com/loostrum/arts_localisation',
      author='Leon Oostrum',
      author_email='oostrum@astron.nl',
      license='GPLv3',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy',
                        'astropy',
                        'numba',
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
