from pdf2image import convert_from_path
import numpy as np

# CONVERT PDF TO IMAGE: IT STACKS IMAGES OF EACH PDF PAGE
def pdf2img(pdf_path):
    image_of_pdf = np.concatenate(
        tuple(convert_from_path(pdf_path)), axis=0)

    return image_of_pdf