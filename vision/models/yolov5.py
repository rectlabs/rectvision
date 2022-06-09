import subprocess
import os
import sys
import yaml

class Yolov5():
    def __init__(self, num_classes, img_size, batch_size, num_epochs, labels, project_name):
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.labels = labels
        self.project_name = project_name
        self.working_dir = os.path.dirname(__file__)

    def setup(self):
        self.git_clone("https://github.com/ultralytics/yolov5")
        os.chdir(os.path.join(self.working_dir, "yolov5"))
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
        self.data_yaml = os.path.join(self.working_dir, "yolov5/data", "data.yaml")
        with open(self.data_yaml, "w") as file:
            yaml.dump(configs, file, default_flow_style=None)    

    def train(self):
        self.train_model = os.path.join(self.working_dir,"yolov5/train.py")
        
        subprocess.call(["Python", self.train_model, 
                        "--img", str(self.img_size),
                        "--cfg", "yolov5s.yaml",
                        "--hyp", "hyp.scratch-low.yaml",
                        "--batch", str(self.batch_size),
                        "--epochs", str(self.num_epochs),
                        "--data", self.data_yaml,
                        "--weights", "yolov5s.pt",
                        "--workers", "24",
                        "--name", self.project_name])
    def get_map(self):
        self.test_model = os.path.join(self.working_dir,"yolov5/test.py")
        self.model_weights = os.path.join(self.working_dir,"yolov5/runs/train", self.project_name, "weights/best.pt")
        subprocess.call(["Python", self.test_model,
                        "--weights", self.model_weights,
                        "--data", self.data_yaml,
                        "--task", "test",
                        "--workers", "24",
                        "--name", "yolo_det"])

    
# num_classes, img_size, batch_size, num_epochs, labels, project_name = 3, 512, 2, 2, ["displeased", "laughing", "neutral"], "test_project"
# model = Model(num_classes, img_size, batch_size, num_epochs, labels, project_name)
# model.setup()
# model.create_data_config()
# model.train()
    
        

