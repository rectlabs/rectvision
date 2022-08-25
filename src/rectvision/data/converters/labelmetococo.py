import os
import argparse
import json
import shutil

from labelme import utils
from datetime import datetime
import numpy as np
import glob
import PIL.Image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return float(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class LabelmeToCoco():
    def __init__(self, ann_dir, out_coco_dir, project_desc=""):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.ann_dir = ann_dir
        self.labelme_json = glob.glob(os.path.join(self.ann_dir, "*.json"))
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

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(json_file)
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        '''This method gets the width, height, id and filename for an image and 
        stores it in the image dictionary'''
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)
        category["name"] = label
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        #x and y coordinates
        x = contour[:, 0]
        y = contour[:, 1]
        #area of bounding polgon
        # area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # annotation["segmentation"] = [[list(np.asarray(points).flatten())]]
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

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["info"] = self.info
        data_coco["images"] = self.images        
        data_coco["annotations"] = self.annotations
        data_coco["categories"] = self.categories
        return data_coco

    def copy_images(self):
        for path in os.listdir(self.ann_dir):
            if not path.endswith(".json"):
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
