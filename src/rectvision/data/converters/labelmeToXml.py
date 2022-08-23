import xml.etree.ElementTree as ET
import json
import glob
import shutil
import requests
import os
import argparse
import ast
from labelme import utils


class LabelmeToXml():   
    def __init__(self, label_to_id_file_path, ann_dir, out_xml_dir, database='User Provided'):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_xml_dir = out_xml_dir
        self.database = database
        self.ppts = []
        self.current_img_path = None
        self.current_img_width = 0
        self.current_img_height = 0
        self.current_img_depth = 0

        self.json_to_xml()

    def get_xml_template(self):
        request_url = 'https://rectvision.s3.amazonaws.com/xmlTemplate.xml'
        response = requests.get(request_url)
        open('template.xml', 'wb').write(response.content)

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
        self.get_xml_template()
        print('Starting conversion...')
        tree = ET.parse('template.xml')
        root = tree.getroot()
        #checkk validity of output text dir and create one if it doesn't exist
        self.out_xml_dir = self.valid_path(self.out_xml_dir)
        #iterate over all json files in self.ann_dir and extract required info from each
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.json')):
            self.extract_info_from_json(ann_path)

            #copy image from ann_dir and save to out_xml_dir
            shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_xml_dir)

            #get output xml file
            out_xml_path = os.path.splitext(os.path.join(self.out_xml_dir, self.current_img_path))[0] + '.xml'
            #modify xml template
            folder = root.find('folder')
            folder.text = self.ann_dir

            fname = root.find('filename')
            fname.text = self.current_img_path

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
                pose.text = 'Unspecified'

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
        


    

    