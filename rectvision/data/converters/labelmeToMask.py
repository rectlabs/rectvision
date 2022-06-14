import numpy as np
import json
import glob
import os
import argparse
import cv2
import ast
import PIL.Image

class LabelmeToMask():   
    def __init__(self, label_to_colour_file_path, ann_dir, out_mask_dir):
        self.label_to_colour = self.read_dictionary(label_to_colour_file_path)
        self.ann_dir = ann_dir
        self.out_mask_dir = out_mask_dir
        self.all_points = []
        self.current_labels = []
        self.current_img_path = None

        self.json_to_mask()

    def read_dictionary(self, dict_file_path):
        with open(dict_file_path, "r") as data:
            dictionary = ast.literal_eval(data.read())  
        return dictionary     
    
    def extract_info_from_json(self, ann_path):
        self.all_points = []
        self.current_labels = []
        with open(ann_path, 'r') as fp:
            data = json.load(fp)
            #get path to original image
            self.current_img_path = data['imagePath']
            #get annotation coordinates and labels
            for shapes in data['shapes']:
                #get boundind box coordinates
                points = shapes['points']
                self.all_points.append(np.array(points, np.int32))
                #get label for each set of points
                label = shapes['label']
                self.current_labels.append(label)

    def draw_mask(self, image, points, colour):
        #reshape array points
        points = points.reshape((-1, 1, 2))
        #draw polygon around point and colour it colour
        image = cv2.fillPoly(image, [points], color=colour)
        return image

    def mask_background(self, image, points, bkg_colour):
        mask_value = 255
        stencil = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(stencil, points, mask_value)
        sel = stencil != mask_value
        image[sel] = bkg_colour
        return image
    
    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def json_to_mask(self):
        print('Starting conversion...')
        #check validity of out_mask_dir and create it if it doesn't exist
        self.out_mask_dir = self.valid_path(self.out_mask_dir)
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
            self.extract_info_from_json(ann_path)
            image = cv2.imread(os.path.join(self.ann_dir, self.current_img_path))

            for idx, points in enumerate(self.all_points):
                label = self.current_labels[idx]
                colour = self.label_to_colour[label]
                image = self.draw_mask(image, points, colour)
            #mask out background with the colour black
            image = self.mask_background(image, self.all_points, (0,0,0))
            out_path = os.path.join(self.out_mask_dir, self.current_img_path)
            cv2.imwrite(out_path, image)
        print('All done!')




    

    