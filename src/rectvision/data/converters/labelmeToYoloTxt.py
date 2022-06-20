import numpy as np
import json
import glob
import os
import shutil
import argparse
import ast


class LabelmeToYoloTxt():   
    def __init__(self, label_to_id_file_path, ann_dir, out_txt_dir):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_txt_dir = out_txt_dir
        self.ppts = []
        self.current_img_path = None

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
            # img_path = os.path.join(self.ann_dir, data['imagePath'])
            self.current_img_path = data['imagePath']
            #get annotation coordinates and labels
            for shapes in data['shapes']:
                points = shapes['points']
                #get label for each set of points
                label = shapes['label']
                label_id = self.label_to_id[label]
                x1, y1, width, height = self.pointsTobbox(points)
                self.ppts.append([label_id, x1, y1, width, height, '\n'])

    def pointsTobbox(self, points):
        '''This method converts the points array to x_min, x_max,
        width and height of the boounding box'''
        x1,y1 = points[0][:]
        x2,y2 = points[1][:]
        width = x2 - x1
        height = y2-y1
        return x1, y1, width, height

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path


    def json_to_txt(self):
        print('Starting conversion...')
        #check validity of out_txt_dir and create it if it doesn't exist
        self.out_txt_dir = self.valid_path(self.out_txt_dir)
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
        




    

    