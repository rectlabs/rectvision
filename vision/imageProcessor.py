import numpy as np 
import cv2
import PIL
from PIL import Image
from sklearn.metrics import pairwise
from statistics import mean

class ImageProcessor():
    def __init__(self, image_path=None):
        if image_path is not None:
            self.image_pil = Image.open(image_path)
            self.image_array = cv2.imread(image_path)
            self.image_height, self.image_width = self.image_array.shape[0:2]

    def imageFlipping(self, flip_dir='horizontal', img=None):
        '''
        This method flips image in the direction specified by flip_dir.

        The parameter passed is:
        img: Image to flip.
        flip_dir: The direction to flip in ('horizontal' or 'vertical').

        The parameter returned is:
        img_flip: Flipped image as numpy ndarray 
        '''
        if img is None:
            img = self.image_array
        if flip_dir=='horizontal':
            img_flip = cv2.flip(img, 1)
        elif flip_dir=='vertical':
            img_flip = cv2.flip(img, 0)
            
        return img_flip

    def imageRotationWithoutAngle(self, rotation_dir='clockwise', img=None):
        '''
        This method rotates image in the direction specified by rotation_dir.

        The parameter passed is:
        rotation_dir: The direction to rotate ('clockwise' or 'anticlockwise').
        img: Image to rotate.

        The parameter returned is:
        img_rot: Rotated image as numpy ndarray 
        '''
        if img is None:
            img = self.image_array
        if rotation_dir=='clockwise':
            img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_dir=='anticlockwise':
            img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        return img_rot

    def imageRotationWithAngle(self, rotation_angle=0, img=None):
        '''
        This method rotates image in the direction specified by rotation_angle.

        The parameter passed is:
        rotation_angle: The direction to rotate (0≤rotation_angle≤360).
        img: Image to rotate

        The parameter returned is:
        img_rot: Rotated image as pil image
        '''
        if img is None:
            img = self.image_pil
        img_rot = img.rotate(rotation_angle, expand=True)
        return img_rot

    def imageResizing(self, new_dim=None, img=None):
        '''
        This method resizes image to dimension specified by new_dim.

        The parameter passed is:
        new_dim: A tuple containing resized image dimension in the form (h,w).
        img: Image to resize.

        The parameter returned is:
        img_resized: Resized image as numpy ndarray
        '''
        if img is None:
            img = self.image_array
        if new_dim is None:
            img_height, img_width = img.shape[0:2]
            new_dim = (img_height, img_width)
               
        img_resized = cv2.resize(img, new_dim[::-1])
        return img_resized

    def imageScaling(self, scale=1, img=None):
        '''
        This method scales image based on the specified scale value.

        The parameter passed is:
        scale: The scale factor for which the image should be scaled. 
               If 0≤scale≤1, image is reduced, if scale>1, image is enlarged.

        The parameter returned is:
        img_scaled: Scaled image as numpy ndarray
        '''
        if img is None:
            img = self.image_array
        img_height, img_width = img.shape[0:2]
        if scale:            
            height = int(img_height*scale)
            width = int(img_width*scale)
            new_dim = (height, width)
        
        img_scaled = cv2.resize(img, new_dim[::-1])
        return img_scaled

    def imageCropping(self, crop_start = (0,0), crop_end=None, img=None):
        '''
        This method crops out a portion of an image based on the specified 
        crop_start and crop_end values.

        The parameters passed are:
        crop_start: Tuple containing the starting coordinates of the crop in the form (w,h).
        crop_end: Tuple containing the stop coordinates of the crop in the form (w,h)

        The parameter returned is:
        img_crop: Cropped image as numpy ndarray
        '''
        if img is None:
            img = self.image_array
        img_height, img_width = img.shape[0:2]
        if crop_end is None:
            crop_end = (img_width, img_height)
        img_crop = img[crop_start[1]:crop_end[1], crop_start[0]:crop_end[0]]
        return img_crop

    def faceDetection(self, img=None):
        '''
        This method detects and crops out human faces present in an image.
        
        The parameters passed is:
        img: Image from which to detect faces.

        The parameter returned is:
        faces: List of all detected images as numpy ndarrays
        '''
        if img is None:
            img = self.image_array
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_coordinates = face_cascade.detectMultiScale(img_gray, 1.05, 4)
        faces = []
        faces_w = []
        faces_h = []
        for (x, y, w, h) in face_coordinates:
            faces.append(self.imageCropping((x, y), (x+w, y+h)))  
            faces_w.append(w)
            faces_h.append(h)  

        #get average width and height of images
        out_face_size = (mean(faces_h), mean(faces_w))
        #resize all faces to average sizes
        for idx, face in enumerate(faces):
            if face.shape[:2] != out_face_size:
                faces[idx] = cv2.resize(face, out_face_size[::-1])
        #create a vertical collage of all detected images
        face_collage = np.vstack(faces)    
        return face_collage, faces, face_coordinates

    def insertLabel(self, label_path, label_start=(0,0), label_size=None, img=None):
        '''
        This method inserts a label to an image.

        The parameters passed are:
        img: Image to add label to.
        label_path: path to the label to be inserted.
        label_start: Tuple containing the starting coordinates of the label 
                     on the original image in the form(h,w).
        label_size: Tuple containing the size of the label in the form (h,w)

        The parameter returned is:
        added_image: Image with label added as numpy ndarray
        '''
        if img is None:
            img = self.image_array 
                  
        label = cv2.imread(label_path)
        
        if label_size is None:
            label_size = label.shape[:2] #result is in h,w,c
        else:
            label = cv2.resize(label, label_size[::-1])
        
        added_image = img
        label_end = (label_start[0]+label_size[0], label_start[1]+label_size[1])
        roi = img[label_start[0]:label_end[0], label_start[1]:label_end[1]]
        label = cv2.addWeighted(roi, 0, label, 1, 0.0)
        added_image[label_start[0]:label_end[0], label_start[1]:label_end[1]] = label
        
        return added_image
    
    def findFace(self, ref_face_path, thresh=0.9, img=None):
        '''
        This method finds a face in an image.

        The parameters passed are:
        ref_face_path: Path to the reference face to be found.
        thresh: Minimum simiilarity value. Must be a value between 0 and 1.
        img: Image to look for faces in.

        The parameters returned are:
        found_face: Face found as numpy ndarray
        similarity_percentage: Percentage similarity between reference face and found face. 
        '''
        if img is None:
            img=self.image_array

        ref_face = cv2.imread(ref_face_path)
        _, ref_face, _ = self.faceDetection(ref_face)
        #convert to greyscale to remove colour bias in comparison
        ref_face = cv2.cvtColor(ref_face[0], cv2.COLOR_BGR2GRAY)
        ref_face_size = ref_face.shape
        #flatten ref_face
        ref_face = ref_face.reshape(1, -1)
        _, faces, face_coordinates = self.faceDetection()
        similarities = []
        #check faces for ref_image and return true based on threshold
        for face in faces:
            #convert face to grayscale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #resize face to ref_face size.
            face = cv2.resize(face, ref_face_size[:2])
            #flatten face
            face = face.reshape(1, -1)
            #compute cosine similarity
            similarity = pairwise.cosine_similarity(face, ref_face)
            similarities.append(similarity)
        
        max_similarity_idx = np.argmax(similarities)
        if similarities[max_similarity_idx] >= thresh:
            similarity_percentage = similarities[max_similarity_idx] * 100
            found_face = faces[max_similarity_idx]
            #get face coordinates
            found_face_coordinates = face_coordinates[max_similarity_idx]
            #define individual coordinates for found face
            found_face_x, found_face_y = found_face_coordinates[:2]
            found_face_w, found_face_h = found_face_coordinates[2:]
            #draw red bounding box around found face in original image
            cv2.rectangle(img, (found_face_x, found_face_y), (found_face_x+found_face_w, found_face_y+found_face_h), (0,0,255))
            
            return img, similarity_percentage
        print('FACE NOT FOUND')
        
        
    