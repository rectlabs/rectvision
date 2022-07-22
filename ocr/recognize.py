# from distutils.command.config import config
import pytesseract
from preprocess import pdf2img

## INSTALLATION STEPS
# brew install tesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/5.2.0/bin/tesseract"

tessdata_dir_config = r'--tessdata-dir "/usr/local/Cellar/tesseract/5.2.0/share/tessdata"'


def recognize_pdf(pdf_file):
    img = pdf2img(pdf_file)
    extractedInformation = pytesseract.image_to_string(
        img, lang="eng", config=tessdata_dir_config)
    
    return extractedInformation


file = 'test_data/ocr_test2pages.pdf'

# EXPORT THE TEXT AS A .TXT
outfilename = file.split('.')[0]+'.txt'
with open(outfilename, 'w', encoding="utf-8") as f:
    f.write(recognize_pdf(file))
    f.close()