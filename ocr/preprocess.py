from pdf2image import convert_from_path
import numpy as np

# CONVERT PDF TO IMAGE: IT STACKS IMAGES OF EACH PDF PAGE
def pdf2img(path2pdf):
    image_of_pdf = np.concatenate(
        tuple(convert_from_path(path2pdf)), axis=0)

    return image_of_pdf




# def grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# def threshold(gray_image):
#     thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
#     im_bw = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#     return im_bw

# def noise_removal(image):
#     kernel = np.ones((1, 1), np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     kernel = np.ones((1, 1), np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     image = cv2.medianBlur(image, 3)
#     return (image)

# def thin_font(image):
#     import numpy as np
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2,2),np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)

# def thick_font(image):
#     import numpy as np
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2,2),np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)

# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     # gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(newImage, (9, 9), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
#     dilate = cv2.dilate(thresh, kernel, iterations=2)

#     # Find all contours
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     for c in contours:
#         rect = cv2.boundingRect(c)
#         x,y,w,h = rect
#         cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

#     # Find largest contour and surround in min area box
#     largestContour = contours[0]
#     print (len(contours))
#     minAreaRect = cv2.minAreaRect(largestContour)
#     cv2.imwrite("temp/boxes.jpg", newImage)
#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     if angle < -45:
#         angle = 90 + angle
#     return -1.0 * angle

# Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage

# Deskew image
# def deskew(cvImage):
#     angle = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle)

# def remove_borders(image):
#     contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
#     cnt = cntsSorted[-1]
#     x, y, w, h = cv2.boundingRect(cnt)
#     crop = image[y:y+h, x:x+w]
#     return (crop)
