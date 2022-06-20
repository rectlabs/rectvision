import argparse
import xml.etree.ElementTree as ET
import ast
import os
import glob
import shutil

class XmlToYolov3KerasTxt():
    def __init__(self, label_to_id_file_path, ann_dir, out_txt_dir):
        self.label_to_id = self.read_dictionary(label_to_id_file_path)
        self.ann_dir = ann_dir
        self.out_txt_dir = out_txt_dir
        self.ppts = []
        self.current_img_path = None

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
        
        for obj in annotation_root.findall('object'):
            #get label
            label = obj.findtext('name')
            label_id =  self.label_to_id[label]
            bndbox = obj.find('bndbox')
            x_min = int(float(bndbox.findtext('xmin'))) - 1
            y_min = int(float(bndbox.findtext('ymin'))) - 1
            x_max = int(float(bndbox.findtext('xmax')))
            y_max = int(float(bndbox.findtext('ymax')))
            self.ppts.append([x_min, y_min, x_max, y_max, label_id])

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def xml_to_txt(self):
        print('Starting conversion...')
        self.out_txt_dir = self.valid_path(self.out_txt_dir)
        out_txt_path = os.path.join(self.out_txt_dir, 'annotations.txt')
        with open(out_txt_path, 'w') as f:
            for ann_path in glob.glob(os.path.join(self.ann_dir, '*.xml')):
                self.extract_info_from_xml(ann_path)
                f.writelines([self.current_img_path, ' ', ' '.join(str(','.join(str(ppt) for ppt in ppts))
                                                                        for ppts in self.ppts), '\n'])     

                #copy image from ann_dir and save to out_txt_dir
                shutil.copy(os.path.join(self.ann_dir, self.current_img_path), self.out_txt_dir)  
                
        print('All done!')
        



    

    


