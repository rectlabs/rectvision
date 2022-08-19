#this script gets project's annotation and call conversion script to convert to specified format.
from json import loads
import os
import jwt
from dotenv import load_dotenv
import requests
from .rectvisConverterHelper import GenerateAnnotation

class RectvisionConverter():
    def __init__(self, token, train_split, test_split, validation_split, export_format):
        load_dotenv()
        self.token = token
        self.export_format = export_format
        self.train_split = train_split
        self.test_split = test_split
        self.validation_split = validation_split
        self.user_id, self.project_id, self.user_token = self.get_creds()
        self.rectvision_converter()
    
    def get_creds(self):
        key = os.getenv('JWT_KEY')
        creds = jwt.decode(self.token, key, algorithms = "HS256")['payload']
        user_id, project_id, user_token = creds['user_id'], creds['projectId'], creds['token']
        return user_id, project_id, user_token

    def get_db_info(self):
        #get project details from get-a-project endpoint
        base_url = 'https://backend.app.rectvision.com/api/v1/projects/'
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
        base_url = 'https://backend.app.rectvision.com/api/v1/projects/'
        request_url = base_url + self.project_id + '/annotations?limit=1000000000'
        headers={'Authorization':self.user_token}
        response = requests.get(request_url, headers=headers)
        annotations = loads(response.text)['data']['annotations']

        rearranged_annotations = {}
        for annotation in annotations:
            image_name = annotation['file']['name']
            if image_name not in rearranged_annotations:
                rearranged_annotations[image_name] = {}
            if 'points' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['points'] = []
            if 'labels' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['labels'] = []
            rearranged_annotations[image_name]['image_id'] = annotation['file_id']
            rearranged_annotations[image_name]['image_url'] = annotation['file']['url']
            rearranged_annotations[image_name]['image_height'] = annotation['file']['metadata']['height']
            rearranged_annotations[image_name]['image_width'] = annotation['file']['metadata']['width']
            rearranged_annotations[image_name]['image_channels'] = annotation['file']['metadata']['channels']
            rearranged_annotations[image_name]['points'].append(annotation['points'])
            rearranged_annotations[image_name]['labels'].append(annotation['label']['value'])

        return rearranged_annotations

    def rectvision_converter(self):
        labels, shape_type = self.get_db_info()
        rearranged_annotations = self.get_annotations()        
        #convert data
        GenerateAnnotation(export_format=self.export_format, annotations=rearranged_annotations, labels=labels, shape_type=shape_type,
                        database = 'User Provided', train_ratio=self.train_split, test_ratio=self.test_split, valid_ratio=self.validation_split)

        print('Annotations saved to results directory in the current working directory')

