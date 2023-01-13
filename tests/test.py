from src.rectvision.data.converters.rectvisConverter import *
from src.rectvision import models
import os, sys

token = "6a2884f385d006c51bc22c6242f153c5a250e7b1235f6ea10031140daa93287a316f4c06a419f96d89f6725029e904c851a857a2b80def791b34f7d981894da7e1864f9282662cfa72ac003bc32f58f7fd0cd3b0a7278e0dec765a826bc17f975b6fd9eefe9e7ffbbae1eccff6176bbdb895bc862a01f7272c239d58b43cd2219c47f5ecd80eaf4170c97407f309d234"
rv = RectvisionConverter(0.5, 0.3, 0.2, 'yolo-txt', token = token)

current_dir = os.getcwd()
project_dir = current_dir 
 

# num_classes, img_size, batch_size, num_epochs, labels, project_name, project_dir = 2, 512, 2, 2000, ["test", "cat", "dog"], "mask_detect", project_dir
# model_yolo = models.Yolov5(num_classes, img_size, batch_size, num_epochs, labels, project_name, project_dir)

# # training
# model_yolo.train()

# # get map
# model_yolo.get_map()

# # run inference
# model_yolo.inference(images=project_dir  + '/dataset/test/images', confidence=0.0, out_dir=project_dir + '/detections/')
