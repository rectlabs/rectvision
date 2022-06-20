import numpy as np
import json
import glob
import os
import argparse
import ast
import csv
import shutil
from labelme import utils


class LabelmeToTfCsv():   
    def __init__(self, ann_dir, out_csv_dir):
        self.ann_dir = ann_dir
        self.out_csv_dir = out_csv_dir
        self.ppts = []
        self.current_img_path = None
        self.current_img_width = 0
        self.current_img_height = 0

        self.json_to_csv()
    
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
                x_min, y_min, x_max, y_max = self.pointsTobbox(points)
                self.ppts.append([self.current_img_path, label, self.current_img_width, self.current_img_height, x_min, y_min, x_max, y_max])
        

    def pointsTobbox(self, points):
        x_min,y_min = points[0][:]
        x_max,y_max = points[1][:]        
        return x_min, y_min, x_max, y_max

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path


    def json_to_csv(self):
        print('Starting conversion...')
        self.out_csv_dir = self.valid_path(self.out_csv_dir)
        out_csv_path = os.path.join(self.out_csv_dir, 'annotations.csv')
        with open(out_csv_path, 'w') as f:
            writer = csv.writer(f)

            for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
                self.extract_info_from_json(ann_path)
                writer.writerows(self.ppts)    
                #copy image from ann_dir and save to out_csv_dir
                shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_csv_dir)   
                
        print('All done!')
        



    

    