import tensorflow as tf
import numpy as np
import ast
import json
import os
import argparse
from PIL import Image

class CocoToTfrecords():
    def __init__(self, in_coco_path, out_tfrecord_dir):
        self.in_coco_path = in_coco_path
        self.out_tfrecord_dir = out_tfrecord_dir
        self.annotations = None
        self.image_ppts = None
        self.image_paths = []
        self.categories = None
        self.images = None

        self.coco_to_tfrecord()

    #methods to convert values to type compatible with tf.train.Example
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_feature_list(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _image_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def create_example(self, image, path, annotation):
        feature = {
            "image": self._image_feature(image),
            "path": self._bytes_feature(tf.io.serialize_tensor(path)),
            "area": self._float_feature(annotation["area"]),
            "bbox": self._float_feature_list(annotation["bbox"]),
            "category_id": self._int64_feature(annotation["category_id"]),
            "id": self._int64_feature(annotation["id"]),
            "image_id": self._int64_feature(annotation["image_id"]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def get_coco_info(self, ann_path):
        with open(ann_path, 'r') as f:
            info = json.load(f)
            
        self.annotations = info['annotations']
        self.image_ppts = info['images']
        self.categories = info['categories']

        for id in range(len(self.image_ppts)):
            self.image_paths.append(self.image_ppts[id]['file_name'])
    
    def coco_to_tfrecord(self):
        print('Starting conversion...')
        self.out_tfrecord_dir = self.valid_path(self.out_tfrecord_dir)
        out_tfrecord_path = os.path.join(self.out_tfrecord_dir, 'annotations.tfrecord')
        #get root directory of annotation file
        coco_root_dir = os.path.dirname(self.in_coco_path)
        #get info from coco file
        self.get_coco_info(self.in_coco_path)
        for annotation in self.annotations:
            with tf.io.TFRecordWriter(out_tfrecord_path) as writer:
                image_id = annotation['image_id']
                image_path = os.path.join(coco_root_dir, self.image_paths[image_id])
                # image_path = f'{coco_root_dir}\{self.image_paths[image_id]}'
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = self.create_example(image, image_path, annotation)
                writer.write(example.SerializeToString())  
        
        print('Writing to label map file')
        label_map_path = os.path.join(self.out_tfrecord_dir, 'label_map.txt')
        self.generate_labelmap(label_map_path)
        print('All done!')

    
    def generate_labelmap(self, label_map_path):
        with open(label_map_path, 'w') as f:
            self.categories = [str(cat) for cat in self.categories]
            f.writelines(self.categories)
            
