import argparse
import xml.etree.ElementTree as ET
import ast
import os
import glob
import csv
import shutil


class XmlToKerasRetinanetCsv():
    def __init__(self, ann_dir, out_csv_dir):
        self.ann_dir = ann_dir
        self.out_csv_dir = out_csv_dir
        self.ppts = []
        self.current_img_path = None

        self.xml_to_csv()
    
    def extract_info_from_xml(self, ann_path):
        self.ppts = []
        annotation_root = ET.parse(ann_path).getroot()
        current_img_path = annotation_root.findtext('path')
        if current_img_path is None:
            current_img_path = annotation_root.findtext('filename')
        self.current_img_path = current_img_path
        
        for obj in annotation_root.findall('object'):
            #get label
            label = obj.findtext('name')
            bndbox = obj.find('bndbox')
            x_min = int(float(bndbox.findtext('xmin'))) - 1
            y_min = int(float(bndbox.findtext('ymin'))) - 1
            x_max = int(float(bndbox.findtext('xmax')))
            y_max = int(float(bndbox.findtext('ymax')))
            self.ppts.append([self.current_img_path, x_min, y_min, x_max, y_max, label])

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def xml_to_csv(self):
        print('Starting conversion...')
        self.out_csv_dir = self.valid_path(self.out_csv_dir)
        out_csv_path = os.path.join(self.out_csv_dir, 'annotations.csv')
        with open(out_csv_path, 'w') as f:
            writer = csv.writer(f)
            for ann_path in glob.glob(os.path.join(self.ann_dir, '*.xml')):
                self.extract_info_from_xml(ann_path)
                writer.writerows(self.ppts)  

                #copy image from ann_dir and save to out_csv_dir
                shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_csv_dir)             
        print('All done!')
        



    

    


