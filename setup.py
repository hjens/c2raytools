'''
Created on Jun 20, 2013

@author: Hannes Jensen

Setup script
'''

from distutils.core import setup
setup(name='c2raytools',
      version='1.0',
      author='Hannes Jensen',
      author_email='hjens@astro.su.se',
      package_dir = {'c2raytools' : 'src/c2raytools'},
      packages=['c2raytools'],
      )