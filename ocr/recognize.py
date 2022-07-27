import pytesseract
import numpy as np
import cv2
import argparse
from preprocess import pdf2img


pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/5.2.0/bin/tesseract"

tessdata_dir_config = r'--tessdata-dir "/usr/local/Cellar/tesseract/5.2.0/share/tessdata"'

class TextRecognizer():
    '''
    Recognize and extract texts in pdfs and Images
    '''
    def __init__(self, pdf_path:str=None, img_path:str=None):
        self.pdf_path = pdf_path
        self.img_path = img_path


    def recognize_pdf(self):
        self.image_of_pdf = pdf2img(self.pdf_path)
        extractedInformation = pytesseract.image_to_string(
            self.image_of_pdf, lang="eng", config=tessdata_dir_config)
        
        return extractedInformation

    def recognize_img(self):
        self.image = cv2.imread(self.img_path)
        extractedInformation = pytesseract.image_to_string(
            self.image, lang="eng", config=tessdata_dir_config)

        return extractedInformation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        help="the location where the .pdf, .jpg, .jpeg, or .png file is saved"
    )

    args = parser.parse_args()

    file = args.file_path

    if file==None:
        print("This command is not recognised. Run this instead:\
        \n\npython recognize.py --file_path <path to file>\n")

    else:
        outfilename = file.split('.')[0]+'.txt'

        with open(outfilename, 'w', encoding="utf-8") as f:
            if file.endswith(".pdf"):
                recognise = TextRecognizer(pdf_path=file)
                f.write(recognise.recognize_pdf())
                f.close()

            elif file.endswith((".jpg", ".jpeg", ".png")):
                recognise = TextRecognizer(img_path=file)
                f.write(recognise.recognize_img())
                f.close()

    
            
