import os
from setuptools import setup, find_packages

VERSION = '0.1.17' 
DESCRIPTION = 'Rectvision package'

with open("README.md", "r") as fh:
        LONG_DESCRIPTION = fh.read()

# Setting up
setup(
        name="rectvision", 
        version=VERSION,
        author="Rectlabs",
        author_email="<team@rectlabs.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. 
        
        keywords=['python']
)