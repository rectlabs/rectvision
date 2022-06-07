import numpy as np
import json
import glob
import os
import argparse
import ast


class generate_txt():   
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
                x_min, y_min, x_max, y_max = self.pointsTobbox(points)
                self.ppts.append([x_min, y_min, x_max, y_max, label_id])
        

    def pointsTobbox(self, points):
        x_min,y_min = points[0][:]
        x_max,y_max = points[1][:]        
        return x_min, y_min, x_max, y_max

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path


    def json_to_txt(self):
        print('Starting conversion...')
        self.out_txt_dir = self.valid_path(self.out_txt_dir)
        out_txt_path = os.path.join(self.out_txt_dir, 'annotations.txt')
        with open(out_txt_path, 'w') as f:
            for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
                self.extract_info_from_json(ann_path)
                f.writelines([self.current_img_path, ' ', ' '.join(str(','.join(str(ppt) for ppt in ppts))
                                                                        for ppts in self.ppts), '\n'])       
                
        print('All done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='labelme json annotation to keras yolo v3 text annotation files'
    )
    parser.add_argument('label_to_id_file_path',
        help='Path to file containing dictionary mapping labels to IDs',
        type=str
    )
    parser.add_argument('annotation_dir',
        help='Path to directory storing the labelme annotation files',
        type=str
    )
    parser.add_argument('output_txt_dir',
        help='Path to directory to store the text files',
        type=str
    )
    args = parser.parse_args()
    generate_txt(args.label_to_id_file_path, args.annotation_dir, args.output_txt_dir)


    

    