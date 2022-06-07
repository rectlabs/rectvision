### code

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# set GPU usage
device = torch.device('cuda:0')

# Some basic setup:
# Setup detectron2 logger
import detectron2, cv2
from tqdm import tqdm
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")



predictor = DefaultPredictor(cfg)
# outputs = predictor(im)


print('video dir: ', sys.argv[1])
cap = cv2.VideoCapture(sys.argv[1])

TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#define the codec and create videoWriter object
fourcc= cv2.VideoWriter_fourcc(*'XVID')
print((width, height))
out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (1536, 864))


# def detectBox(outputs, frame):
#     color = (255, 0, 0)
#     instance = outputs['instances']
#     for count, (bbox, scores, class_name) in enumerate(zip(instance.pred_boxes, instance.scores, instance.pred_classes)):
#         bbox = bbox.to('cpu').numpy()
#         scores = scores.to('cpu').numpy()
#         class_name = class_name.to('cpu').numpy()
#         cv2.rectangle(frame, (int(bbox[0]), int(
#                 bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
#         cv2.putText(frame, str(class_name)+ '_'+ str(count), (int(
#                 bbox[0]), int(bbox[1]-10)), 0, 0.5, (255, 255, 255), 1)

#     return frame


def detectBox(outputs, frame):
    v = Visualizer(frame[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image() #[:, :, ::-1]


for _, i in enumerate(tqdm(range(TotalFrames))):
    ret, frame = cap.read()
    if ret== True:
        # frame = cv2.flip(frame, 1)
        detections = predictor(frame)

        appendFrame = detectBox(detections, frame)
        # print(appendFrame.shape)
        appendFrame = cv2.cvtColor(appendFrame, cv2.COLOR_BGR2RGB)
        out.write(appendFrame)

        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()