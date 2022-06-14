from setuptools import setup, find_packages

VERSION = '0.1.10' 
DESCRIPTION = 'Rectvision package'
LONG_DESCRIPTION = 'This package allows the user to interact with the rectvision platform via code'

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