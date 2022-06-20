import numpy as np
import json
import glob
import shutil
import os
import argparse
import ast
from labelme import utils


class LabelmeToDarknetTxt():   
    def __init__(self, label_to_id_file_path, ann_dir, out_txt_dir):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_txt_dir = out_txt_dir
        self.ppts = []
        self.current_img_path = None
        self.current_img_width = 0
        self.current_img_height = 0
        
        self.json_to_txt()

    def read_dictionary(self, dict_file_path):
        with open(dict_file_path, "r") as data:
            dictionary = ast.literal_eval(data.read())  
        return dictionary     
    
    def extract_info_from_json(self, ann_path):
        self.ppts = []
        with open(ann_path, 'r') as fp:
            data = json.load(fp)
            #get path to original image
            self.current_img_path = data['imagePath']
            #get  image dimension from imageData
            self.current_img_height, self.current_img_width = utils.img_b64_to_arr(data["imageData"]).shape[:2]
            #get annotation coordinates and labels
            for shapes in data['shapes']:
                points = shapes['points']
                #get label for each set of points
                label = shapes['label']
                #convert label to ID from label_to_id dictionary
                label_id = self.label_to_id[label]
                #get properties of the bounding box
                x_center, y_center, width, height = self.pointsTobbox(points)
                self.ppts.append([label_id, 
                                  np.abs(x_center)/self.current_img_width, 
                                  np.abs(y_center)/self.current_img_height, 
                                  np.abs(width)/self.current_img_width, 
                                  np.abs(height)/self.current_img_height, '\n'])

    def pointsTobbox(self, points):
        '''This method converts the points array to the properties of the object's bounding box'''
        x1,y1 = points[0][:]
        x2,y2 = points[1][:]        
        width = x2 - x1
        height = y2-y1
        x_center = x1 + (width/2)
        y_center = y1 + (height/2)
        return x_center, y_center, width, height

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def json_to_txt(self):
        print('Starting conversion...')
        #checkk validity of output text dir and create one if it doesn't exist
        self.out_txt_dir = self.valid_path(self.out_txt_dir)
        #iterate over all json files in self.ann_dir and extract required info from each
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
            self.extract_info_from_json(ann_path)

            #copy image from ann_dir and save to out_txt_dir
            shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_txt_dir)
            #write to text file
            out_txt_path = os.path.splitext(os.path.join(self.out_txt_dir, self.current_img_path))[0] + '.txt'
            with open(out_txt_path, 'w') as f:
                #write each object per line
                for point in self.ppts:
                    f.write(' '.join(str(ppt) for ppt in point))
                
        print('All done!')
        
  

    