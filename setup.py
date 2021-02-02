#!/usr/bin/env python

import os
from setuptools import setup, find_packages


with open(os.path.join('arts_localisation', '__version__.py')) as version_file:
    version = {}
    exec(version_file.read(), version)
    project_version = version['__version__']


setup(name='arts_localisation',
      version=project_version,
      description='Localisation of ARTS transients',
      url='http://github.com/loostrum/arts_localisation',
      author='Leon Oostrum',
      author_email='l.oostrum@esciencecenter.nl',
      license="Apache Software License 2.0",
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
      )
