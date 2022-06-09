import argparse
import xml.etree.ElementTree as ET
import ast
import os
import glob
import numpy as np


class xml_to_darknetTxt():
    def __init__(self, label_to_id_file_path, ann_dir, out_txt_dir):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_txt_dir = out_txt_dir
        self.ppts = []
        self.current_img_path = None
        self.current_img_width = 0
        self.current_img_height = 0

        self.xml_to_txt()

    def read_dictionary(self, dict_file_path):
        with open(dict_file_path, "r") as data:
            dictionary = ast.literal_eval(data.read())  
        return dictionary     
    
    def extract_info_from_xml(self, ann_path):
        self.ppts = []
        annotation_root = ET.parse(ann_path).getroot()
        current_img_path = annotation_root.findtext('path')
        if current_img_path is None:
            current_img_path = annotation_root.findtext('filename')
        self.current_img_path = current_img_path
        #get width and height of image
        size = annotation_root.find('size')
        self.current_img_width = int(size.findtext('width'))
        self.current_img_height = int(size.findtext('height'))
        
        for obj in annotation_root.findall('object'):
            #get label
            label = obj.findtext('name')
            label_id =  self.label_to_id[label]
            bndbox = obj.find('bndbox')
            x1 = int(float(bndbox.findtext('xmin'))) - 1
            y1 = int(float(bndbox.findtext('ymin'))) - 1
            x2 = int(float(bndbox.findtext('xmax')))
            y2 = int(float(bndbox.findtext('ymax')))
            width = x2 - x1
            height = y2-y1
            x_center = x1 + (width/2)
            y_center = y1 + (height/2)
            self.ppts.append([label_id, 
                              np.abs(x_center)/self.current_img_width, 
                              np.abs(y_center)/self.current_img_height, 
                              np.abs(width)/self.current_img_width, 
                              np.abs(height)/self.current_img_height, '\n'])

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def xml_to_txt(self):
        print('Starting conversion...')
        self.out_txt_dir = self.valid_path(self.out_txt_dir)
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.xml')):
            self.extract_info_from_xml(ann_path)

            #write to text file
            out_txt_path = os.path.splitext(os.path.join(self.out_txt_dir, self.current_img_path))[0] + '.txt'
            with open(out_txt_path, 'w') as f:
                #write each object per line
                for point in self.ppts:
                    f.write(' '.join(str(ppt) for ppt in point))
                
        print('All done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='XML annotation to darknet text annotation files'
    )
    parser.add_argument('label_to_id_file_path',
        help='Path to file containing dictionary mapping labels to IDs',
        type=str
    )
    parser.add_argument('annotation_dir',
        help='Path to directory storing the XML annotation files',
        type=str
    )
    parser.add_argument('output_txt_dir',
        help='Path to directory to store the text files',
        type=str
    )
    args = parser.parse_args()
    xml_to_darknetTxt(args.label_to_id_file_path, args.annotation_dir, args.output_txt_dir)


    

    

