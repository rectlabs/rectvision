import xml.etree.ElementTree as ET
import json
import glob
import os
import argparse
import ast
from labelme import utils


class generate_xml():   
    def __init__(self, label_to_id_file_path, ann_dir, out_xml_dir, xml_template='templates\xmlTemplate.xml', database='Open Images'):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_xml_dir = out_xml_dir
        self.xml_template = xml_template
        self.database = database
        self.ppts = []
        self.current_img_path = None
        self.current_img_width = 0
        self.current_img_height = 0
        self.current_img_depth = 0

        self.json_to_xml()

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
            self.current_img_height, self.current_img_width, self.current_img_depth = utils.img_b64_to_arr(data["imageData"]).shape
            #get annotation coordinates and labels
            for shapes in data['shapes']:
                points = shapes['points']
                #get label for each set of points
                label = shapes['label']
                #convert label to ID from label_to_id dictionary
                label_id = self.label_to_id[label]
                #get properties of the bounding box
                x_min, x_max, y_min, y_max = self.pointsTobbox(points)
                self.ppts.append([label_id,x_min, x_max, y_min, y_max])

    def pointsTobbox(self, points):
        '''This method converts the points array to the properties of the object's bounding box'''
        x_min,y_min = points[0][:]
        x_max,y_max = points[1][:]        
        return x_min, x_max, y_min, y_max

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def json_to_xml(self):
        print('Starting conversion...')
        tree = ET.parse(self.xml_template)
        root = tree.getroot()
        #checkk validity of output text dir and create one if it doesn't exist
        self.out_xml_dir = self.valid_path(self.out_xml_dir)
        #iterate over all json files in self.ann_dir and extract required info from each
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
            self.extract_info_from_json(ann_path)

            #get output xml file
            out_xml_path = os.path.splitext(os.path.join(self.out_xml_dir, self.current_img_path))[0] + '.xml'
            #modify xml template
            folder = root.find('folder')
            folder.txt = self.ann_dir

            fname = root.find('filemane')
            fname.txt = self.current_img_path

            src = root.find('source')
            database = src.find('database')
            database.text = self.database

            size = root.find('size')
            width = size.find('width')
            width.text = str(self.current_img_width)
            height = size.find('height')
            height.text = str(self.current_img_height)
            depth = size.find('depth')
            depth.text = str(self.current_img_depth)
            
            #write objects as specific annotations
            for ppt in self.ppts:
                #append new object
                obj = ET.SubElement(root, 'object')

                name = ET.SubElement(obj, 'name')
                name.text = str(ppt[0])

                pose = ET.SubElement(obj, 'pose')
                pose.txt = 'Unspecified'

                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(ppt[1]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(ppt[2]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(ppt[3]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(ppt[4]))
            
            #save annotation to out_xml_path
            tree.write(out_xml_path)
                
        print('All done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='labelme json annotation to pascal VOC xml annotation files'
    )
    parser.add_argument('label_to_id_file_path',
        help='Path to file containing dictionary mapping labels to IDs',
        type=str
    )
    parser.add_argument('annotation_dir',
        help='Path to directory storing the labelme json annotation files',
        type=str
    )
    parser.add_argument('output_xml_dir',
        help='Path to directory to store the xml files',
        type=str
    )
    parser.add_argument('xml_template_path',
        help='Path to xml template to use for conversion',
        type=str
    )
    parser.add_argument('database',
        help='Name of database from which images were obtained',
        type=str
    )
    
    args = parser.parse_args()
    generate_xml(args.label_to_id_file_path, args.annotation_dir, args.output_xml_dir, args.xml_template_path, args.database)


    

    