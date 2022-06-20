#this script pulls from kafka and calls rectvisconverter
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

