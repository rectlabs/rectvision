from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Rectvision package'
LONG_DESCRIPTION = 'This package allows the user to download and upload data to rectvision and train models'

# Setting up
setup(
        name="rectvision", 
        version=VERSION,
        author="Rectlabs",
        author_email="<team@rectlabs.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. 
        
        keywords=['python']
)