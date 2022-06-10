#this script pulls from kafka and calls rectvisconverter
from json import loads
import os
import requests
from kafka import KafkaConsumer
from pymongo import MongoClient
from rectvisConverterHelper import GenerateAnnotation

class rectvision_converter():
    def __init__(self, user_id, project_id, user_token):
        self.user_id = user_id
        self.project_id = project_id
        self.user_token = user_token
        self.convert_data()
        
    def get_db_info(self):
        #get project details from endpoint
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + self.project_id
        headers={'Authorization': self.user_token}
        response = requests.get(request_url, headers=headers)
        current_project = loads(response.text)['data']

        export_format = current_project['export_format']
        annotations = current_project['annotations']
        labels = current_project['labels']
        if current_project['annotation_choice'] == 'Object_detection':
            shape_type = 'rectangle'
        train_ratio = current_project['training']['train_split']
        test_ratio = current_project['training']['test_split']
        valid_ratio = current_project['training']['validation_split']
        
        return export_format, annotations, labels, shape_type, train_ratio, test_ratio, valid_ratio        

    def convert_data(self):
        export_format, annotations, labels, shape_type, train_ratio, test_ratio, valid_ratio = self.get_db_info()
        GenerateAnnotation(export_format=export_format, annotations=annotations, labels=labels, shape_type=shape_type,
                                        xml_template = 'templates/xmlTemplate.xml', database = 'User Provided',
                                        train_ratio=train_ratio, test_ratio=test_ratio, valid_ratio=valid_ratio)  

        


            
    
         