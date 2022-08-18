#this script gets project's annotation and call conversion script to convert to specified format.
from json import loads
import os
import requests
from .rectvisConverterHelper import GenerateAnnotation

class RectvisionConverter():
    def __init__(self, project_id, user_token, export_format):
        self.project_id = project_id
        self.user_token = user_token
        self.export_format = export_format

        self.rectvision_converter()

    def get_db_info(self, project_id, user_token):
        #get project details from get-a-project endpoint
        base_url = 'https://backend.app.rectvision.com/api/v1/projects/'
        request_url = base_url + project_id
        headers={'Authorization':user_token}
        response = requests.get(request_url, headers=headers)
        current_project = loads(response.text)['data']['project']

        # annotations = current_project['annotations']
        labels = [label['value'] for label in current_project['labels']]
        if current_project['annotation_choice'] == 'Object_detection':
            shape_type = 'rectangle'

        return labels, shape_type       

    def get_annotations(self, project_id, user_token):
        #get annotations from endpoint
        base_url = 'https://backend.app.rectvision.com/api/v1/projects/'
        request_url = base_url + project_id + '/annotations?limit=1000000000'
        headers={'Authorization':user_token}
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
        #get db info
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + self.project_id
        headers={'Authorization': self.user_token}
        response = requests.get(request_url, headers=headers)
        current_project = loads(response.text)['data']
        annotations = current_project['annotations']
        labels = current_project['labels']
        if current_project['annotation_choice'] == 'Object_detection':
            shape_type = 'rectangle'
        train_ratio = current_project['training']['train_split']
        test_ratio = current_project['training']['test_split']
        valid_ratio = current_project['training']['validation_split']
        
        #convert data
        GenerateAnnotation(export_format=self.export_format, annotations=annotations, labels=labels, shape_type=shape_type,
                        database = 'User Provided', train_ratio=train_ratio, test_ratio=test_ratio, valid_ratio=valid_ratio)

        print('Annotations saved in train, test and validation folders!')

