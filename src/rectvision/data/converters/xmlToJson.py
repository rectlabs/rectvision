import argparse
import json
import xml.etree.ElementTree as ET
import os
import glob
import shutil


class XmlToJson():   
    def __init__(self, ann_dir, out_json_dir, shape_type='rectangle'):
        self.ann_dir = ann_dir
        self.out_json_dir = out_json_dir
        self.shape_type = shape_type
        self.ppts = []
        self.current_img_path = None

        self.xml_to_json()

    def extract_info_from_xml(self, ann_path):
        self.ppts = []
        annotation_root = ET.parse(ann_path).getroot()
        img_path = annotation_root.findtext('path')
        if img_path is None:
            img_path = annotation_root.findtext('filename')
        self.current_img_path = img_path
        for obj in annotation_root.findall('object'):
            #get label
            label = obj.findtext('name')
            #get coordinates
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.findtext('xmin'))) - 1
            ymin = int(float(bndbox.findtext('ymin'))) - 1
            xmax = int(float(bndbox.findtext('xmax')))
            ymax = int(float(bndbox.findtext('ymax')))
            points = [[xmin, ymin], [xmax, ymax]]
            self.ppts.append([self.current_img_path, label, self.shape_type, points])
       

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def xml_to_json(self):
        print('Starting conversion...')
        self.out_json_dir = self.valid_path(self.out_json_dir)
        for ann_path in glob.glob(os.path.join(self.ann_dir, '*.xml')):
            
            output_json_dict = {
                'shapes': [],
                'imagePath': ''
            }
            #get image info
            self.extract_info_from_xml(ann_path)
            #copy image from ann_dir and save to out_json_dir
            shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_json_dir)
            for ppt in self.ppts:
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
            out_json_path = os.path.splitext(os.path.join(self.out_json_dir, self.current_img_path))[0] + '.json'
            with open(out_json_path, 'w') as f:
                output_json = json.dumps(output_json_dict)
                f.write(output_json)
        print('All done!')
        

