import numpy as np
import json
import glob
import os
import argparse
import ast
from labelme import utils


class generate_json():   
    def __init__(self, project_id, ann_file_path, out_json_dir, shape_type='rectangle'):
        self.ann_file_path = ann_file_path
        self.out_json_dir = out_json_dir
        self.project_id = project_id
        self.shape_type = shape_type
        self.ppts = []

        self.rectjson_to_json()

        
    def extract_info_from_json(self):
        with open(self.ann_file_path, 'r') as fp:
            data = json.load(fp)
            project_data = data['projects'][self.project_id]
            #iterate over all images in project_data and obtain properties for each image
            for image_data in project_data:
                current_img_path = image_data['image_url']
                annotations = image_data['annotation']
                image_ppts = []
                for label in annotations:
                    #get list of annotations for label
                    label_annotations = annotations[label]
                    for points in label_annotations:
                        image_ppts.append([current_img_path, label, self.shape_type, self.pointsTobbox(points)])
                self.ppts.append(image_ppts)


    def pointsTobbox(self, points):
        '''This method converts the points array to the properties of the object's bounding box'''
        x_min, y_min = points[0], points[1] - points[3]
        x_max, y_max = points[0] + points[3], points[1]
        return [[x_min, y_min], [x_max, y_max]]

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def rectjson_to_json(self):
        print('Starting conversion...')
        self.out_json_dir = self.valid_path(self.out_json_dir)
        for image_ppt in self.ppts:
            
            output_json_dict = {
                'shapes': [],
                'imagePath': ''
            }
            #get all image info
            self.extract_info_from_json()
            for ppt in image_ppt:
                img_path, label, shape_type, points = ppt
                #update imagePath
                output_json_dict['imagePath']=img_path
                #shapes info
                shapes_info = {
                    'label':'',
                    'points': [],
                    'shape_type': ''
                }
                #update shapes_info
                shapes_info['label'] = label
                shapes_info['points'] = points
                shapes_info['shape_type'] = shape_type
                output_json_dict['shapes'].append(shapes_info)
            #write to json file
            out_json_path = os.path.splitext(os.path.join(self.out_json_dir, image_ppt[0][0]))[0] + '.json'
            with open(out_json_path, 'w') as f:
                output_json = json.dumps(output_json_dict)
                f.write(output_json)
        print('All done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='rectvision json annotation to labelme json annotation'
    )
    parser.add_argument('project_id',
        help='ID of project to generate annotation files for',
        type=str
    )
    parser.add_argument('annotation_file_path',
        help='Path to rectvision annotation file',
        type=str
    )
    parser.add_argument('output_json_dir',
        help='Path to directory to store the labelme json files',
        type=str
    )
    parser.add_argument('shape_type', 
        help='Type of annotation shape. Either rectangle or polygon',
        type=str 
        )
    args = parser.parse_args()
    generate_json(args.project_id, args.annotation_file_path, args.output_json_dir, args.shape_type)


    

    