# RectVision

##  Welcome to the RectVision Package!
**[RectVision](https://rectvision.com/)** is a web-based tool that helps you organize your AI data process, end to end on a single platform. It allows users to create training reproducible pipelines or workflow from data annotation to model training, deployment and inferencing.

This package allows registered RectVision users to interact with the web-based platform via simple python commands from a local machine or on Google Colab. It is built to work seamlessly with the features of the [RectVision webapp](https://app.rectvision.com/auth/sign-in?returnTo=%252Fd), allowing users to 

- download and convert annotations made in the webapp
- modify or process images and data
- train and evaluate models
- perform inference with trained models

### Installation
This package supports an installation of [Python 3.6 and above](https://www.python.org/downloads/). It is available on Windows, MacOS and Linux Systems.

#### Requirements
- [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
    - When installing detectron2, ensure torch and torchvision versions are compatible with the version of detectron2. These versions have been shown to be compatible with both the above detectron2 installation and rectivision's installation.
    ```bash
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```
```bash
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

#### pip (Recommended)
To install the current `rectvision`:

```bash
pip install --upgrade rectvision
```
Alternatively, you may download the wheels from **[PyPI](https://pypi.org/project/rectvision/#files)**

### Quickstart
To familiarize yourself with this package, follow this [colab tutorial](https://colab.research.google.com/drive/1nnh39TVMKEd61IWxY7HT-REAGvdrZNIk?usp=sharing). See the [documentation](https://docs.rectvision.com/) for more details on how to use this package.