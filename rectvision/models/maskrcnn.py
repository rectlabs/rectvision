import json
import os
import numpy as np
import cv2
import random
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.catalog import MetadataCatalog
from detectron2.data import build_detection_test_loader

class MaskRCNN():
    def __init__(self, num_classes, batch_size, num_epochs, project_dir):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.project_dir = self.valid_path(project_dir)
        self.valid_annotations = os.path.join(self.project_dir,'valid/annotations.json')
        self.train_annotations = os.path.join(self.project_dir,'train/annotations.json')
        self.test_annotations = os.path.join(self.project_dir, 'test/annotations.json')
        self.valid_images = os.path.join(self.project_dir,'valid/images')
        self.train_images = os.path.join(self.project_dir,'train/images')
        self.test_images = os.path.join(self.project_dir, 'test/images')

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def register_data(self):
        for data_category, dataset in {'train_data':[self.train_annotations, self.train_images], 
                                  'test_data':[self.test_annotations, self.test_images], 
                                  'valid_data':[self.valid_annotations, self.valid_images]}.items():
            register_coco_instances(data_category, {}, dataset[0], dataset[1])
    
    def edit_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("train_data")
        cfg.DATASETS.TEST = ("valid_data")
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  
        cfg.SOLVER.MAX_ITER = self.num_epochs    
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.TEST.DETECTIONS_PER_IMAGE = 20
        cfg.OUTPUT_DIR = os.path.join(self.working_dir, 'output')

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        return cfg
        
    def train(self):
        self.register_data()
        self.model_cfg = self.edit_config()
        trainer = DefaultTrainer(self.model_cfg) 
        trainer.resume_or_load(resume=True)
        trainer.train() 

    def train_registered(self):
        self.model_cfg = self.edit_config()
        trainer = DefaultTrainer(self.model_cfg) 
        trainer.resume_or_load(resume=True)
        trainer.train() 

    def evaluate(self):
        self.model_cfg.MODEL.WEIGHTS = os.path.join(self.model_cfg.OUTPUT_DIR, "model_final.pth") 
        self.model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        predictor = DefaultPredictor(self.model_cfg)
        evaluator = COCOEvaluator("test_data", output_dir="./output")
        val_loader = build_detection_test_loader(self.model_cfg, "test_data")
        inference_on_dataset(predictor.model, val_loader, evaluator)
        #get metadata for id_to_label mapping
        test_metadata = MetadataCatalog.get("test_data")
        return predictor, test_metadata

    def predict(self, image, predictor, metadata):
        image = cv2.imread(image)
        outputs = predictor(image)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        labels = metadata.thing_classes
        pred_masks = outputs['instances'].pred_masks.cpu().numpy()
        bbox = []
        for (box, mask, class_id, score) in zip(boxes.astype(np.float64), pred_masks, classes, scores):
            # x0, y0, x1, y1 = box #top left, bottom right
            bbox.append([box, mask, score, class_id, labels[class_id]])

        return bbox



