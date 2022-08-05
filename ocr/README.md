# Optical Character Recognition from PDFs and Images
Reading texts in PDF or Images.

## Run with a CLI:
1. Clone this repo.

2. cd `rectvision/ocr`

3. Run `bash run.sh`

4. Run `python recognize.py --file_path <path/to/file>`, where __<path/to/file>__ is to be replaced with the path to the file for text extraction.

5. To test the codes, skip #3 and #4 and run `python test.py`.

### Output
A _.txt_ file containing the extracted texts. It will be stored in the same directory and with the same name as the input file.

__Please note that expected file format is either .pdf, .jpeg, .jpg or .png.__

## Credit:
This was built using [pytesseract](https://github.com/madmaze/pytesseract), a Python wrapper for tesseract.
