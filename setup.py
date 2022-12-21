from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '3.0.29'
DESCRIPTION = 'A low-code tool to help create your own AI'
LONG_DESCRIPTION = 'A package that allows to build your own computer visions and NLP systems'

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
    # install_requires=["requests", "importlib-metadata", "boto3==1.22.9", "dtw==1.4.0", "labelme==5.0.1","moviepy==1.0.3", "numpy==1.20.0",
    # "pyJWT==2.4.0", "python-dotenv==0.19.2", "PyYAML==5.4.1", "opencv-contrib-python==4.6.0.66", "sklearn"],
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