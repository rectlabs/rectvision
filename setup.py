from setuptools import setup, find_packages
import codecs
import os, sys, subprocess


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '3.0.43'
DESCRIPTION = 'A low-code tool to help create your own AI'
LONG_DESCRIPTION = 'A package that allows to build your own computer visions and NLP systems'


requirement_path = here + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Setting up
setup(
    name="rectvision",
    version=VERSION,
    author="Rectlabs Inc",
    author_email="rectanglenet@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=install_requires,
    keywords=['python', 'NLP', 'computer vision', 'deep learning', 'AI'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)