import subprocess, os, random, sys, math, yaml, shutil


def otp(length = 5) :
    # Declare a digits variable 
    # which stores all digits
    digits = "0123456789"
    OTP = ""
   # length of password can be changed
   # by changing value in range
    for i in range(length) :
        OTP += digits[math.floor(random.random() * 10)]
 
    return OTP

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
        try:
          shutil.copytree('yolov5', os.path.join(self.project_dir, 'yolov5'))
          os.chdir(os.path.join(self.project_dir, "yolov5"))
          subprocess.run([sys.executable, "-m", "pip", "install", "-qr", "requirements.txt"])
        except Exception as err:
          pass

    def no_setup(self):
        os.chdir(os.path.join(self.project_dir, "yolov5"))
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    def git_clone(self, url):
        process = subprocess.run(["git", "clone", url], capture_output=True, text=True)
        # if process.returncode == 0:
        #   print('success!!!')
        # else:
        #   print('Something went wrong')
        #   print(process.stderr)
    
    def create_data_config(self):
        train_path = os.path.join(self.project_dir, "dataset/images/train")
        test_path = os.path.join(self.project_dir, "dataset/images/test")
        val_path = os.path.join(self.project_dir, "dataset/images/val")
        configs = {"train": train_path,
                   "val": val_path,
                   "test": test_path,
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
        print('Training YOLOv5 Model...')
        process = subprocess.run(["python3", self.train_model, 
                        "--img", str(self.img_size),
                        "--batch", str(self.batch_size),
                        "--epochs", str(self.num_epochs),
                        "--data", self.data_yaml,
                        "--weights", "yolov5s.pt",
                        "--name", self.project_name,
                        "--cache"], capture_output=True, text=True)
        if process.returncode == 0:
          print('Trained successfully!') 
          print(process.stdout)         
          print('Check {} for training logs'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name)))
          print('More importantly, Check {} for progression of training performance'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, '/results.csv')))
        else:
          print('Training could not be completed. Check error below for more details')
          print(process.stderr)
      
    def train_file_exists(self):
        self.no_setup()
        self.create_data_config()
        self.train_model = os.path.join(self.project_dir, "yolov5/train.py")
        print('Training YOLOv5 Model...')
        process = subprocess.run(["python", self.train_model, 
                        "--img", str(self.img_size),
                        "--batch", str(self.batch_size),
                        "--epochs", str(self.num_epochs),
                        "--data", self.data_yaml,
                        "--weights", "yolov5s.pt",
                        "--name", self.project_name,
                        "--cache"], capture_output=True, text=True)
        # !python {self.train_model} --img {self.img_size} --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch {self.batch_size} --epochs {self.num_epochs} --data {self.data_yaml} --weights yolov5s.pt --workers 24 --name {self.project_name}
        if process.returncode == 0:
          print('Trained successfully!') 
          print(process.stdout)         
          print('Check {} for training logs'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name)))
          print('More importantly, Check {} for progression of training performance'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, '/results.csv')))
        else:
          print('Training could not be completed. Check error below for more details')
          print(process.stderr)

    def get_map(self):
        self.test_model = os.path.join(self.project_dir, "yolov5/val.py")
        self.model_weights = os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, "weights/best.pt")
        print('Evaluation in progress...')
        process = subprocess.run(["python", self.test_model, 
                        "--weights", self.model_weights,
                        "--data", self.data_yaml,
                        "--task", "test",
                        "--name", self.project_name + '_performance'], capture_output=True, text=True)
        # !python {self.test_model} --weights {self.model_weights} --data {self.data_yaml} --task test --name {self.project_name + '_performance'}
        if process.returncode == 0:
          print('Evaluated successfully!') 
          print(process.stdout)         
          print('Check {} for results'.format(os.path.join(self.project_dir, "yolov5/runs/val", self.project_name+'_performance')))
        else:
          print('Evaluation could not be completed. Check error below for more details')
          print(process.stderr)
        

    def inference(self, images, out_dir, img_size = 416, confidence= 0.3, iou_thres = 0.3):
      self.outdir =out_dir + '/' + otp()
      self.detect_model = os.path.join(self.project_dir, "yolov5/detect.py")
      self.model_weights = os.path.join(self.project_dir, "yolov5/runs/train", self.project_name, "weights/best.pt")
      print('Inference in progress...')
      process = subprocess.run(["python", self.detect_model, 
                        "--source", images,
                        "--img",  img_size,
                        "--weights", self.model_weights,
                        "--conf", str(confidence),
                        "--iou-thres", str(iou_thres),
                        "--name", self.project_name + '_detections',
                        "--save-txt"], capture_output=True, text=True)
      if process.returncode == 0:
          print('Inference completed successfully!') 
          print(process.stdout)   
          shutil.copytree(os.path.join(self.project_dir, "yolov5/runs/detect", self.project_name+'_detections'), self.out_dir)
          print('Check {} and {} for detections and annotations generated'.format(os.path.join(self.project_dir, "yolov5/runs/train", self.project_name+'_detections'), self.out_dir))

      else:
          print('Inference could not be completed. Check error below for more details')
          print(process.stderr)
      

    

        

