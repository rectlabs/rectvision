#this script gets project's annotation and call conversion script to convert to specified format.
from json import loads
import os, uuid, shutil
import getpass
import jwt, cv2, random
import requests
from tqdm import tqdm
from .rectvisConverterHelper import GenerateAnnotation

def download_url_to_disk(image_url):
    try:
        os.makedir('tmp')
    except Exception as err:
        pass
    file_disk_path = './tmp/' + str(uuid.uuid4()) + ".jpg"
    r = requests.get(image_url, stream = True)

    # Check if the csv was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        with open(os.path.join(file_disk_path),'wb') as f:
            shutil.copyfileobj(r.raw, f)
    
    else:
        pass 
    
    return file_disk_path

def get_image_widith_height(image_url):
    file_path = download_url_to_disk(image_url)
    img = cv2.imread(file_path)
    height, width, channel = img.shape
    os.remove(file_path)
    return height, width, channel

class RectvisionConverter():
    def __init__(self, train_split=0.6, test_split=0.2, validation_split=0.2, export_format='yolo-txt', token = None): 
        if token == None:       
            self.token = getpass.getpass('Enter Token: ')
        else:
            self.token = token
        self.endpoint = "https://test.backend.app.rectvision.com/api/v1/"
        base_url = self.endpoint + 'projects/decode-token?token='
        request_url = base_url + self.token
        self.login_response = requests.get(request_url)
        while not self.login_response.ok:
            print("Something went wrong! Ensure the right token was entered!")
            self.token = getpass.getpass('Enter Token: ')
            request_url = base_url + self.token
            self.login_response = requests.get(request_url)
        self.export_format = export_format
        self.train_split = train_split
        self.test_split = test_split
        self.validation_split = validation_split
        self.user_id, self.project_id, self.user_token = self.get_creds()
        self.rectvision_converter()
    
    def get_creds(self):
        creds = loads(self.login_response.text)['data']['decoded']
        user_id, project_id, user_token = creds['user_id'], creds['project_id'], creds['token']
        return user_id, project_id, user_token

    def get_db_info(self):
        #get project details from get-a-project endpoint
        base_url = self.endpoint + 'projects/'
        request_url = base_url + self.project_id
        headers={'Authorization':self.user_token}
        response = requests.get(request_url, headers=headers)
        current_project = loads(response.text)['data']['project']

        # annotations = current_project['annotations']
        labels = [label['value'] for label in current_project['labels']]
        if current_project['annotation_choice'] == 'Object_detection':
            shape_type = 'rectangle'

        return labels, shape_type 

    def get_annotations(self):
        #get annotations from endpoint
        base_url = self.endpoint + 'projects/'
        request_url_files = base_url + self.project_id + '/files?limit=1000000000'
        request_url_annotations = base_url + self.project_id + '/annotations?limit=1000000000'
        headers={'Authorization':self.user_token}
        file_response = requests.get(request_url_files, headers=headers)
        annotation_response = requests.get(request_url_annotations, headers=headers)
        file_properties = loads(file_response.text)
        annotation_properties = loads(annotation_response.text)
        file_prop = file_properties['data']['files']
        annotation_prop = annotation_properties['data']['annotations']

        rearranged_annotations = {}
        total_data = len(annotation_prop)
        data_split = random.choices(['train', 'test', 'val'], weights = [self.train_split, self.test_split, self.validation_split], k=total_data)
        for data_tag, annon_prop in tqdm(zip(data_split, annotation_prop), desc = 'getting annotations'):
            image_name = annon_prop['file']['name']
            if image_name not in rearranged_annotations:
                rearranged_annotations[image_name] = {}
            else:
                continue
            if 'points' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['points'] = []
            if 'labels' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['labels'] = []
            if 'data_tag' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['data_tag'] = None

            rearranged_annotations[image_name]['image_id'] = annon_prop['file_id']
            rearranged_annotations[image_name]['image_url'] = annon_prop['file']['url']
            rearranged_annotations[image_name]['image_height'] =annon_prop['image_height']
            rearranged_annotations[image_name]['image_width'] = annon_prop['image_width']
            rearranged_annotations[image_name]['image_channels'] = 3
            rearranged_annotations[image_name]['points'].append(annon_prop['points'])
            rearranged_annotations[image_name]['labels'].append(annon_prop['label']['value'])
            rearranged_annotations[image_name]['data_tag'] = data_tag
        return rearranged_annotations
      



    def rectvision_converter(self):
        labels, shape_type = self.get_db_info()
        rearranged_annotations = self.get_annotations()        
        #convert data
        GenerateAnnotation(export_format=self.export_format, annotations=rearranged_annotations, labels=labels, shape_type=shape_type,
                        database = 'User Provided', train_ratio=self.train_split, test_ratio=self.test_split, valid_ratio=self.validation_split)

        print('Annotations saved to results directory in the current working directory')

