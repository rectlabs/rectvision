# Optical Character Recognition from PDFs and Images
Reading texts in PDF or Images.

## Steps:
1. Clone this repo.

2. Install dependencies by following these steps:
- `cd ocr`
    - `pip install -r requirements-exp.txt` in your virtual environment.

- Install tesseract: 
    - For macOS: `brew install tesseract`
    - For Linux: `sudo apt install tesseract-ocr`

    Additional info on how to install tesseract on Windows can be found [here](https://tesseract-ocr.github.io/tessdoc/Compiling.html).

- Open `recognize.py` and edit the following variables with the full path to downloaded _/bin/tesseract_ and _/share/tessdata_, respectively. An example is in `recognize.py`.
    - `pytesseract.pytesseract.tesseract_cmd` and 
    - `tessdata_dir_config`.


## How to use:
After following the installation guide, run the following command line: 
`python recognize.py --file_path <path/to/file>`

__Please Note that expected file format is either .pdf, .jpeg, .jpg or .png.__

To learn more about `pytesseract`, please refer to this [repository](https://github.com/madmaze/pytesseract).
