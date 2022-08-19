import subprocess
import os
import sys
import yaml
import shutil

class Yolov5():
    def __init__(self, num_classes, img_size, batch_size, num_epochs, labels, project_name, project_dir):
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.labels = labels
        self.project_name = project_name
        self.project_dir = self.valid_path(project_dir)

    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def setup(self):
        self.git_clone("https://github.com/ultralytics/yolov5")
        # copy to yolov5 in project_dir
        shutil.copytree('yolov5', os.path.join(self.project_dir, 'yolov5'))
        os.chdir(os.path.join(self.project_dir, "yolov5"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    def git_clone(self, url):
        subprocess.check_output(["git",  "clone", url])
    
    def create_data_config(self):
        configs = {"train": "../dataset/train/images",
                   "val": "../dataset/valid/images",
                   "test": "../dataset/test/images",
                   "nc": self.num_classes,
                   "names": self.labels
                  }
        self.data_yaml = os.path.join(self.project_dir, "yolov5/data", "data.yaml")
        with open(self.data_yaml, "w") as file:
            yaml.dump(configs, file, default_flow_style=None)    

    def train(self):
        self.setup()
        self.create_data_config()
        self.train_model = os.path.join(self.project_dir, "yolov5/train.py")
        subprocess.call(["python", self.train_model, 
                        "--img", str(self.img_size),
                        "--cfg", "yolov5s.yaml",
                        "--hyp", "hyp.scratch-low.yaml",
                        "--batch", str(self.batch_size),
                        "--epochs", str(self.num_epochs),
                        "--data", self.data_yaml,
                        "--weights", "yolov5s.pt",
                        "--workers", "24",
                        "--name", self.project_name])
        # !python {self.train_model} --img {self.img_size} --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch {self.batch_size} --epochs {self.num_epochs} --data {self.data_yaml} --weights yolov5s.pt --workers 24 --name {self.project_name}
        print('Check {} for training logs'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name)))

    def get_map(self):
        self.test_model = os.path.join(self.project_dir, "yolov5/val.py")
        self.model_weights = os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, "weights/best.pt")
        subprocess.call(["python", self.test_model, 
                        "--weights", self.model_weights,
                        "--data", self.data_yaml,
                        "--task", "test",
                        "--name", self.project_name + '_performance'])
        # !python {self.test_model} --weights {self.model_weights} --data {self.data_yaml} --task test --name {self.project_name + '_performance'}
        print('Check {} for results'.format(os.path.join(self.project_dir, "yolov5/runs/test", self.project_name+'_performance')))

    def inference(self, images, confidence, out_dir):
      self.detect_model = os.path.join(self.project_dir, "yolov5/detect.py")
      self.model_weights = os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, "weights/best.pt")
      subprocess.call(["python", self.detect_model, 
                        "--source", images,
                        "--weights", self.model_weights,
                        "--conf", confidence,
                        "--name", self.project_name + '_detections'])
    #   !python {self.detect_model} --source {images} --weights {self.model_weights} --conf {confidence} --name {self.project_name + '_detections'}
      
      shutil.copytree(os.path.join(self.project_dir, "yolov5/runs/detect", self.project_name+'_detections'), out_dir)
      print('Check {} and {} for detections'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name+'_detections'), out_dir))

    
# project_dir = r"C:\Users\sanni\Documents\rectangleai\rectvision\test_yolo"
# num_classes, img_size, batch_size, num_epochs, labels, project_name, project_dir = 3, 512, 2, 2, ["displeased", "laughing", "neutral"], "test_project", project_dir
# model = Yolov5(num_classes, img_size, batch_size, num_epochs, labels, project_name, project_dir)

# model.train()
        

