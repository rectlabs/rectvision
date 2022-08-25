import os
import argparse
import shutil
import xml.etree.ElementTree as ET
import json
from datetime import datetime
import numpy as np
import glob
from PIL import Image, ImageDraw


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return float(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class XmlToCoco(object):
    def __init__(self, ann_dir, out_coco_dir, project_desc=""):
        self.ann_dir = ann_dir
        self.xml_ann = glob.glob(os.path.join(self.ann_dir, "*.xml"))
        self.out_coco_dir = out_coco_dir
        self.save_json_path = os.path.join(self.valid_path(self.out_coco_dir), 'annotations.json')
        self.project_desc = project_desc
        self.info = {}
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()
    
    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def gen_info(self):
        self.info["description"] = self.project_desc
        date = datetime.now()
        self.info["version"] = "1.0"
        # self.info["year"] = date.year
        self.info["date_created"] = str(date)

    def image(self, data, num):
        '''This method gets the width, height, id and filename for an image and 
        stores it in the image dictionary'''
        image = {}
        #data is ET.parse(ann_path).getroot()
        #get image width and height
        size = data.find('size')
        height = int(size.findtext('width'))
        width = int(size.findtext('height'))
        size = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        #get image file name
        image["file_name"] = data.findtext('filename').split("/")[-1]

        self.height = height
        self.width = width

        return image

    def data_transfer(self):
        for num, xml_file in enumerate(self.xml_ann):
            #open xml ann file and get fileroot
            data = ET.parse(xml_file).getroot()
            self.images.append(self.image(data, num))
            for obj in data.findall('object'):
                label = obj.findtext('name')
                if label not in self.label:
                    self.label.append(label)
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.findtext('xmin')))
                ymin = int(float(bndbox.findtext('ymin'))) 
                xmax = int(float(bndbox.findtext('xmax')))
                ymax = int(float(bndbox.findtext('ymax')))
                
                points = [[xmin, ymin], [xmax, ymax]]
                # print(points)
                self.annotations.append(self.annotation(points, label, num))
                self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)
        category["name"] = label
        return category

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        #x and y coordinates
        x = contour[:, 0]
        y = contour[:, 1]
        #area of bounding polgon
        # area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        # annotation["area"] = area
        annotation["image_id"] = num
        [[xmin, ymin], [xmax, ymax]] = points
        o_width = xmax - xmin
        o_height = ymax - ymin
        annotation["bbox"] = [xmin, ymin, o_width, o_height]

        annotation["category_id"] = label  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def data2coco(self):
        data_coco = {}
        data_coco["info"] = self.info
        data_coco["images"] = self.images        
        data_coco["annotations"] = self.annotations
        data_coco["categories"] = self.categories
        return data_coco

    def copy_images(self):
        for path in os.listdir(self.ann_dir):
            if not path.endswith(".xml"):
                # Copy image to out_coco_dir
                shutil.copy(os.path.join(self.ann_dir, path), self.out_coco_dir)

    def save_json(self):
        print("save coco json")
        self.gen_info()
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4, cls=NpEncoder)
        self.copy_images()
