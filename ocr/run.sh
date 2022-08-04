#!/bin/sh

sudo apt update

sudo rm Anaconda3-5.3.1-Linux-x86_64.sh
rm -rf ~/Anaconda

cd ~
mkdir tmp
cd tmp

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

bash Anaconda3-2022.05-Linux-x86_64.sh
cd ~
source .bashrc

conda create -n ocr-env-p39 python=3.9
conda activate ocr-env-p39

pip3 install numpy==1.23.1 opencv-python==4.6.0.66 packaging==21.3 pdf2image==1.16.0 Pillow==9.2.0 pyparsing==3.0.9 pytesseract==0.3.9
apt-get update && apt-get install -y python3-opencv

sudo apt install tesseract-ocr -y
apt-get install poppler-utils
