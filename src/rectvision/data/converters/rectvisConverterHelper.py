import numpy as np
from zipfile import ZipFile
from pathlib import Path
import json
import random
import base64
import glob
import requests
import labelme
import shutil
import os
import csv
import ast
import xml.etree.ElementTree as ET
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class GenerateAnnotation():   
    def __init__(self, export_format, annotations, labels=[], shape_type='rectangle', 
                database = 'User Provided', train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
        self.endpoint = "https://test.backend.app.rectvision.com/api/v1/"
        self.annotations = annotations
        self.export_format = export_format
        self.shape_type = shape_type
        self.database = database
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.ppts = []
        self.labels = labels
        #create results directory
        self.valid_path('dataset')
        #generate labelmap
        # TODO
        self.label_to_id_file_path = os.path.join('dataset', 'labelmap.txt')
        self.train_path = self.valid_path(os.path.join('dataset', 'train'))
        self.test_path = self.valid_path(os.path.join('dataset', 'test'))
        self.validation_path = self.valid_path(os.path.join('dataset', 'validation'))
        self.images_folder = self.valid_path(os.path.join('dataset', 'images'))
        self.labels_folder = self.valid_path(os.path.join('dataset', 'labels'))
        self.generate_labelmap(self.labels, self.label_to_id_file_path)
        self.label_to_id = self.read_dictionary(self.label_to_id_file_path)


        #insert which annotation conversion method should be called
        if self.export_format == 'labelme-json':
            self.rectjson_to_json()
        elif self.export_format == 'darknet-txt':
            self.rectjson_to_darknetTxt()
        elif self.export_format == 'yolo-txt':
            self.rectjson_to_yoloTxt()
        elif self.export_format == 'xml':
            self.rectjson_to_xml()
        elif self.export_format == 'yolov3-keras-txt':
            self.rectjson_to_yolov3kerasTxt()
        elif self.export_format == 'keras-retinanet-csv':
            self.rectjson_to_kerasRetinanetCsv()
        elif self.export_format == 'tf-csv':
            self.rectjson_to_tfCsv()

        
    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def replace_extension(self, path, new_extension):
        p = Path(path)
        extensions = ''.join(p.suffixes)
        return str(p).replace(extensions, new_extension)

    def download_image(self, image_url, save_dir):
        filename = image_url.split("/")[-1]
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            out_dir = self.valid_path(save_dir)
            with open(os.path.join(out_dir, filename),'wb') as f:
                shutil.copyfileobj(r.raw, f)
      
        else:
            print('Image Couldn\'t be retreived')

    def compress_annotations(self, zip_path):
        #get files in folder 
        ann_directories = [self.train_path, self.test_path, self.validation_path]
        self.ann_file_paths = [self.label_to_id_file_path]
        for directory in ann_directories:
            for root, directories, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    self.ann_file_paths.append(filepath)

        #write files in file_paths to a zipfile
        with ZipFile(zip_path, 'w') as zip:
            for file in self.ann_file_paths:
                zip.write(file)
        return zip_path

    def upload_annotation(self):
        #this method zips, uploads zipped annotation file to s3 and updates db url
        #zip projects directory
        zip_path = self.compress_annotations(zip_path=os.path.join('results', 'annotations.zip'))
        #post annotation to endpoint   
        base_url = self.endpoint + 'projects/'
        request_url = base_url + self.project_id + '/zip-file'
        headers={'Authorization':self.user_token}
        fo = open(zip_path, 'rb')
        files = {'file': ('zip', fo, 'application/zip')}
        response = requests.post(request_url, files=files, headers=headers)
        if response.ok:
            url = json.loads(response.text)['data']['upload']['Location']
            # print("Upload completed successfully!")
            # print(response.text)
            fo.close()
            shutil.rmtree('results', ignore_errors=False, onerror=None)
            return url
        else:
            print("Something went wrong!")
            print(response.text)       
        

    
    def generate_labelmap(self, labels, label_to_id_file_path='label2id.txt'):
        label_ids = list(range(0, len(labels)))
        label_to_id = dict(zip(labels, label_ids))

        with open(label_to_id_file_path, 'w') as data:
            data.write(str(label_to_id))
    
    def read_dictionary(self, dict_file_path):
        with open(dict_file_path, "r") as data:
            dictionary = ast.literal_eval(data.read())  
        return dictionary 

    def extract_info_from_json(self):
        self.ppts = []
        for image_name, image_data in tqdm(self.annotations.items(), desc = 'converting data'):
            current_img_path = image_name
            current_img_width, current_img_height, current_img_depth = image_data['image_width'], image_data['image_height'], image_data['image_channels']
            points = image_data['points']
            labels = image_data['labels']
            image_ppts = []

            if self.export_format == 'labelme-json':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    image_ppts.append([current_img_path, label, self.shape_type, [[x_min, y_min], [x_max, y_max]]])
                self.ppts.append(image_ppts)
                
            
            elif self.export_format == 'darknet-txt':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    label_id = self.label_to_id[label]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    width, height = x_max - x_min, y_max - y_min
                    x_center, y_center,  = x_min + (width/2), y_min + (height/2)
                    image_ppts.append([current_img_path, label_id, 
                                        np.abs(x_center)/current_img_width, 
                                        np.abs(y_center)/current_img_height, 
                                        np.abs(width)/current_img_width, 
                                        np.abs(height)/current_img_height, '\n' ])
                self.ppts.append(image_ppts)
                

            elif self.export_format == 'yolo-txt':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    label_id = self.label_to_id[label]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    width, height = x_max - x_min, y_max - y_min
                    x_center, y_center,  = (x_min + x_max)/2, (y_min + y_max)/2
                    image_ppts.append([current_img_path, label_id, x_center/current_img_width, 
                                       y_center/current_img_height, width/current_img_width, height/current_img_height, '\n' ])
                self.ppts.append(image_ppts)

            elif self.export_format == 'yolov3-keras-txt':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    label_id = self.label_to_id[label]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    image_ppts.append([current_img_path, x_min, y_min, x_max, y_max, label_id])
                self.ppts.append(image_ppts)
            
            elif self.export_format == 'xml':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    image_ppts.append([current_img_path, current_img_width, 
                                        current_img_height, current_img_depth,
                                        label, self.shape_type, x_min, x_max, y_min, y_max])
                self.ppts.append(image_ppts)

            elif self.export_format == 'keras-retinanet-csv':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    image_ppts.append([current_img_path, x_min, y_min, x_max, y_max, label])
                self.ppts.append(image_ppts)

            elif self.export_format == 'tf-csv':
                for idx, point in enumerate(points):
                    label = labels[idx]
                    label_id = self.label_to_id[label]
                    point_x = [coord[0] for coord in point]
                    point_y = [coord[1] for coord in point]
                    x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                    image_ppts.append([current_img_path, label,
                                        current_img_width, current_img_height,
                                        x_min, y_min, x_max, y_max])
                self.ppts.append(image_ppts)

       
    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def split(self, arr, split1, split2, split3):
        random.shuffle(arr)
        arr_length = len(arr)
        arr1 = arr[:round(split1*arr_length)]
        arr2 = arr[round(split1*arr_length):round((split1+split2)*arr_length)]
        arr3 = arr[round((split1+split2)*arr_length):]
        return arr1, arr2, arr3

    def rectjson_to_json(self):
        print('Starting conversion...')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        for image_ppt in self.ppts_train:
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.train_path, 'images'))
            output_json_dict = {
                'shapes': [],
                'imagePath': ''
            }
            for ppt in image_ppt:
                img_path, label, shape_type, points = ppt
                #update imagePath
                output_json_dict['imagePath']=img_path
                data = labelme.LabelFile.load_image_file(os.path.join(self.train_path, 'images', img_path))
                output_json_dict['imageData']=base64.b64encode(data).decode('utf-8')
                #shapes info
                shapes_info = {
                    'label':'',
                    'points': [],
                    'shape_type': ''
                }
                #update shapes_info
                shapes_info['label'] = label
                shapes_info['points'] = points
                shapes_info['shape_type'] = shape_type
                output_json_dict['shapes'].append(shapes_info)
            #write to json file
            train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
            out_annotation_path = os.path.join(train_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.json'))
            with open(out_annotation_path, 'w') as f:
                output_json = json.dumps(output_json_dict, indent=4, cls=NpEncoder)
                f.write(output_json)
        for image_ppt in self.ppts_test:
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.test_path, 'images'))
            output_json_dict = {
                'shapes': [],
                'imagePath': ''
            }
            for ppt in image_ppt:
                img_path, label, shape_type, points = ppt
                #update imagePath
                output_json_dict['imagePath']=img_path
                data = labelme.LabelFile.load_image_file(os.path.join(self.test_path, 'images', img_path))
                output_json_dict['imageData']=base64.b64encode(data).decode('utf-8')
                #shapes info
                shapes_info = {
                    'label':'',
                    'points': [],
                    'shape_type': ''
                }
                #update shapes_info
                shapes_info['label'] = label
                shapes_info['points'] = points
                shapes_info['shape_type'] = shape_type
                output_json_dict['shapes'].append(shapes_info)
            #write to json file
            test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
            out_annotation_path = os.path.join(test_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.json'))
            with open(out_annotation_path, 'w') as f:
                output_json = json.dumps(output_json_dict, indent=4, cls=NpEncoder)
                f.write(output_json)
        for image_ppt in self.ppts_valid:
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.validation_path, 'images'))
            output_json_dict = {
                'shapes': [],
                'imagePath': ''
            }
            for ppt in image_ppt:
                img_path, label, shape_type, points = ppt
                #update imagePath
                output_json_dict['imagePath']=img_path
                data = labelme.LabelFile.load_image_file(os.path.join(self.validation_path, 'images', img_path))
                output_json_dict['imageData']=base64.b64encode(data).decode('utf-8')
                #shapes info
                shapes_info = {
                    'label':'',
                    'points': [],
                    'shape_type': ''
                }
                #update shapes_info
                shapes_info['label'] = label
                shapes_info['points'] = points
                shapes_info['shape_type'] = shape_type
                output_json_dict['shapes'].append(shapes_info)
            #write to json file
            validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
            out_annotation_path = os.path.join(validation_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.json'))
            with open(out_annotation_path, 'w') as f:
                output_json = json.dumps(output_json_dict, indent=4, cls=NpEncoder)
                f.write(output_json)
        print('All done!')

    def rectjson_to_darknetTxt(self):
        print('Starting conversion...')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        for image_ppt in self.ppts_train:   
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.train_path, 'images'))         
            #write to txt file
            train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
            out_annotation_path = os.path.join(train_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.txt'))
            with open(out_annotation_path, 'w') as f:
                for ppts in image_ppt:
                    f.write(' '.join(str(ppt) for ppt in ppts[1:]))
        for image_ppt in self.ppts_test: 
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.test_path, 'images'))           
            #write to txt file
            test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
            out_annotation_path = os.path.join(test_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.txt'))
            with open(out_annotation_path, 'w') as f:
                for ppts in image_ppt:
                    f.write(' '.join(str(ppt) for ppt in ppts[1:]))
        for image_ppt in self.ppts_valid:        
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.validation_path, 'images'))    
            #write to txt file
            validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
            out_annotation_path = os.path.join(validation_out_annotation_dir, self.replace_extension(image_ppt[0][0], '.txt'))
            with open(out_annotation_path, 'w') as f:
                for ppts in image_ppt:
                    f.write(' '.join(str(ppt) for ppt in ppts[1:]))
        print('All done!') 

    def rectjson_to_yoloTxt(self):
        self.ppts = []
        for image_name, image_data in tqdm(self.annotations.items(), desc = 'converting data'):
            current_img_path = image_name
            current_img_width, current_img_height, current_img_depth = image_data['image_width'], image_data['image_height'], image_data['image_channels']
            points = image_data['points']
            labels = image_data['labels']
            image_ppts = []

            for idx, point in enumerate(points):
                label = labels[idx]
                label_id = self.label_to_id[label]
                point_x = [coord[0] for coord in point]
                point_y = [coord[1] for coord in point]
                x_min, y_min, x_max, y_max = min(point_x), min(point_y), max(point_x), max(point_y)
                width, height = x_max - x_min, y_max - y_min
                x_center, y_center,  = x_min + (width/2), y_min + (height/2)
                image_ppts.append([current_img_path, label_id, 
                                    np.abs(x_center)/current_img_width, 
                                    np.abs(y_center)/current_img_height, 
                                    np.abs(width)/current_img_width, 
                                    np.abs(height)/current_img_height, '\n' ])

            # #split to train, test, validation
            decision = self.annotations[image_name]['data_tag']
            
            if decision == 'train':
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.images_folder, 'train'))       
                #write to txt file
                train_out_annotation_dir = self.valid_path(os.path.join(self.labels_folder, 'train'))
                out_annotation_path = os.path.join(train_out_annotation_dir, self.replace_extension(image_ppts[0][0], '.txt'))
                with open(out_annotation_path, 'w') as f:
                    for ppts in image_ppts:
                        f.write(' '.join(str(ppt) for ppt in ppts[1:]))

            elif decision == 'test':
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.images_folder, 'test'))        
                #write to txt file
                test_out_annotation_dir = self.valid_path(os.path.join(self.labels_folder, 'test'))
                out_annotation_path = os.path.join(test_out_annotation_dir, self.replace_extension(image_ppts[0][0], '.txt'))
                with open(out_annotation_path, 'w') as f:
                    for ppts in image_ppts:
                        f.write(' '.join(str(ppt) for ppt in ppts[1:]))

            else:
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.images_folder, 'val'))    
                #write to txt file
                validation_out_annotation_dir = self.valid_path(os.path.join(self.labels_folder, 'val'))
                out_annotation_path = os.path.join(validation_out_annotation_dir, self.replace_extension(image_ppts[0][0], '.txt'))
                with open(out_annotation_path, 'w') as f:
                    for ppts in image_ppts:
                        f.write(' '.join(str(ppt) for ppt in ppts[1:]))
        print('All done!')  

    def rectjson_to_yolov3kerasTxt(self):
        print('Starting conversion...')
        #write to txt file
        train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
        out_annotation_path = os.path.join(train_out_annotation_dir, 'annotations.txt')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        with open(out_annotation_path, 'w') as f:
            for image_ppt in self.ppts_train:    
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.train_path, 'images'))        
                f.writelines([image_name, ' ', ' '.join(str(','.join(str(ppt) for ppt in ppts[1:])) 
                                                                for ppts in image_ppt), '\n'])
        #write to txt file
        test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
        out_annotation_path = os.path.join(test_out_annotation_dir, 'annotations.txt')
        with open(out_annotation_path, 'w') as f:
            for image_ppt in self.ppts_test:   
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.test_path, 'images'))         
                f.writelines([image_name, ' ', ' '.join(str(','.join(str(ppt) for ppt in ppts[1:])) 
                                                                for ppts in image_ppt), '\n'])
        #write to txt file
        validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
        out_annotation_path = os.path.join(validation_out_annotation_dir, 'annotations.txt')
        with open(out_annotation_path, 'w') as f:
            for image_ppt in self.ppts_valid:   
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.validation_path, 'images'))          
                f.writelines([image_name, ' ', ' '.join(str(','.join(str(ppt) for ppt in ppts[1:])) 
                                                                for ppts in image_ppt), '\n'])
        print('All done!') 

    def get_xml_template(self):
        request_url = 'https://rectvision.s3.amazonaws.com/xmlTemplate.xml'
        response = requests.get(request_url)
        open('template.xml', 'wb').write(response.content)

    def rectjson_to_xml(self):
        #get template
        self.get_xml_template()
        print('Starting conversion...')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        tree = ET.parse('template.xml')
        root = tree.getroot()
        for image_ppt in self.ppts_train:
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.train_path, 'images'))    
            img_path, img_width, img_height, img_depth = image_ppt[0][:4]         
            #write to xml file
            train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
            out_annotation_path = os.path.join(train_out_annotation_dir, self.replace_extension(img_path, '.xml'))
            #modify xml template
            folder = root.find('folder')
            folder.text = train_out_annotation_dir

            fname = root.find('filename')
            fname.text = img_path

            src = root.find('source')
            database = src.find('database')
            database.text = self.database

            size = root.find('size')
            width = size.find('width')
            width.text = str(img_width)
            height = size.find('height')
            height.text = str(img_height)
            depth = size.find('depth')
            depth.text = str(img_depth)

            for ppt in image_ppt:
                #append new object
                obj = ET.SubElement(root, 'object')

                name = ET.SubElement(obj, 'name')
                name.text = str(ppt[4])

                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'

                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(ppt[6]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(ppt[7]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(ppt[8]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(ppt[9]))

            #save annotation to xml file
            tree.write(out_annotation_path)
        
        for image_ppt in self.ppts_test:   
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.test_path, 'images'))
            img_path, img_width, img_height, img_depth = image_ppt[0][:4]         
            #write to xml file
            test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
            out_annotation_path = os.path.join(test_out_annotation_dir, self.replace_extension(img_path, '.xml'))
            #modify xml template
            folder = root.find('folder')
            folder.text = test_out_annotation_dir

            fname = root.find('filename')
            fname.text = img_path

            src = root.find('source')
            database = src.find('database')
            database.text = self.database

            size = root.find('size')
            width = size.find('width')
            width.text = str(img_width)
            height = size.find('height')
            height.text = str(img_height)
            depth = size.find('depth')
            depth.text = str(img_depth)

            for ppt in image_ppt:
                #append new object
                obj = ET.SubElement(root, 'object')

                name = ET.SubElement(obj, 'name')
                name.text = str(ppt[4])

                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'

                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(ppt[6]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(ppt[7]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(ppt[8]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(ppt[9]))

            #save annotation to xml file
            tree.write(out_annotation_path)
        
        for image_ppt in self.ppts_valid:   
            #download image
            image_name = image_ppt[0][0]
            image_url = self.annotations[image_name]['image_url']
            self.download_image(image_url, os.path.join(self.validation_path, 'images'))
            img_path, img_width, img_height, img_depth = image_ppt[0][:4]         
            #write to xml file
            validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
            out_annotation_path = os.path.join(validation_out_annotation_dir, self.replace_extension(img_path, '.xml'))
            #modify xml template
            folder = root.find('folder')
            folder.text = validation_out_annotation_dir

            fname = root.find('filename')
            fname.text = img_path

            src = root.find('source')
            database = src.find('database')
            database.text = self.database

            size = root.find('size')
            width = size.find('width')
            width.text = str(img_width)
            height = size.find('height')
            height.text = str(img_height)
            depth = size.find('depth')
            depth.text = str(img_depth)

            for ppt in image_ppt:
                #append new object
                obj = ET.SubElement(root, 'object')

                name = ET.SubElement(obj, 'name')
                name.text = str(ppt[4])

                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'

                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(ppt[6]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(ppt[7]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(ppt[8]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(ppt[9]))

            #save annotation to xml file
            tree.write(out_annotation_path)
        print('All done!')

    def rectjson_to_kerasRetinanetCsv(self):
        print('Starting conversion...')
        #write to csv file
        train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
        test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
        validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
        out_annotation_path_train = os.path.join(train_out_annotation_dir, 'annotations.csv')
        out_annotation_path_test = os.path.join(test_out_annotation_dir, 'annotations.csv')
        out_annotation_path_valid = os.path.join(validation_out_annotation_dir, 'annotations.csv')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        with open(out_annotation_path_train, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_train:  
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.train_path, 'images'))          
                writer.writerows(image_ppt)
        with open(out_annotation_path_test, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_test:
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.test_path, 'images'))            
                writer.writerows(image_ppt)
        with open(out_annotation_path_valid, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_valid:  
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.validation_path, 'images'))          
                writer.writerows(image_ppt)

        print('All done!')

    def rectjson_to_tfCsv(self):
        print('Starting conversion...')
        #write to csv file
        train_out_annotation_dir = self.valid_path(os.path.join(self.train_path, 'annotations'))
        test_out_annotation_dir = self.valid_path(os.path.join(self.test_path, 'annotations'))
        validation_out_annotation_dir = self.valid_path(os.path.join(self.validation_path, 'annotations'))
        out_annotation_path_train = os.path.join(train_out_annotation_dir, 'annotations.csv')
        out_annotation_path_test = os.path.join(test_out_annotation_dir, 'annotations.csv')
        out_annotation_path_valid = os.path.join(validation_out_annotation_dir, 'annotations.csv')
        #get all image info
        self.extract_info_from_json()
        #split to train, test, validation
        self.ppts_train, self.ppts_test, self.ppts_valid = self.split(self.ppts, self.train_ratio, self.test_ratio, self.valid_ratio)
        with open(out_annotation_path_train, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_train:  
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.train_path, 'images'))
                writer.writerows(image_ppt)
        with open(out_annotation_path_test, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_test:   
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.test_path, 'images'))         
                writer.writerows(image_ppt)
        with open(out_annotation_path_valid, 'w') as f:
            writer = csv.writer(f)

            for image_ppt in self.ppts_valid:    
                #download image
                image_name = image_ppt[0][0]
                image_url = self.annotations[image_name]['image_url']
                self.download_image(image_url, os.path.join(self.validation_path, 'images'))        
                writer.writerows(image_ppt)
        print('All done!')
        

    