#this script gets project's annotation and call conversion script to convert to specified format.
from json import loads
import os
import getpass
import jwt
import requests
from .rectvisConverterHelper import GenerateAnnotation

class RectvisionConverter():
    def __init__(self, train_split, test_split, validation_split, export_format):        
        self.token = getpass.getpass('Enter Token: ')
        self.endpoint = "https://test.backend.app.rectvision.com/api/v1/"
        base_url = self.endpoint + 'projects/decode-token?token='
        request_url = base_url + self.token
        self.login_response = requests.get(request_url)
        while not self.login_response.ok:
            print("Something went wrong! Ensure the right token was entered!")
            self.token = getpass.getpass('Enter Token: ')
            request_url = base_url + self.token
            self.login_response = requests.get(request_url)
        print('User Authenticated!')
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
        for annon_prop, f_prop in zip(annotation_prop, file_prop):
            image_name = f_prop['name']
            if image_name not in rearranged_annotations and len(f_prop['meta'].keys()) == 13:
                rearranged_annotations[image_name] = {}
            else:
                continue
            if 'points' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['points'] = []
            if 'labels' not in rearranged_annotations[image_name]:
                rearranged_annotations[image_name]['labels'] = []
            rearranged_annotations[image_name]['image_id'] = f_prop['project_id']
            rearranged_annotations[image_name]['image_url'] = f_prop['url']
            # some files have missing heights and widths, pls investigate why #TODO
            rearranged_annotations[image_name]['image_height'] = f_prop['meta']['height']
            rearranged_annotations[image_name]['image_width'] = f_prop['meta']['width']
            rearranged_annotations[image_name]['image_channels'] = f_prop['meta']['channels']
            rearranged_annotations[image_name]['points'].append(annon_prop['points'])
            rearranged_annotations[image_name]['labels'].append(annon_prop['label']['value'])

        return rearranged_annotations
      

    # def get_annotations(self):
    #     #get annotations from endpoint
    #     base_url = self.endpoint + 'projects/'
    #     request_url = base_url + self.project_id + '/annotations?limit=1000000000'
    #     headers={'Authorization':self.user_token}
    #     response = requests.get(request_url, headers=headers)
    #     annotations = loads(response.text)['data']['annotations']

    #     rearranged_annotations = {}
    #     for annotation in annotations:
    #         image_name = annotation['file']['name']
    #         if image_name not in rearranged_annotations:
    #             rearranged_annotations[image_name] = {}
    #         if 'points' not in rearranged_annotations[image_name]:
    #             rearranged_annotations[image_name]['points'] = []
    #         if 'labels' not in rearranged_annotations[image_name]:
    #             rearranged_annotations[image_name]['labels'] = []
    #         rearranged_annotations[image_name]['image_id'] = annotation['file_id']
    #         rearranged_annotations[image_name]['image_url'] = annotation['file']['url']
    #         rearranged_annotations[image_name]['image_height'] = annotation['file']['metadata']['height']
    #         rearranged_annotations[image_name]['image_width'] = annotation['file']['metadata']['width']
    #         rearranged_annotations[image_name]['image_channels'] = annotation['file']['metadata']['channels']
    #         rearranged_annotations[image_name]['points'].append(annotation['points'])
    #         rearranged_annotations[image_name]['labels'].append(annotation['label']['value'])

    #     return rearranged_annotations

    def rectvision_converter(self):
        labels, shape_type = self.get_db_info()
        rearranged_annotations = self.get_annotations()        
        #convert data
        GenerateAnnotation(export_format=self.export_format, annotations=rearranged_annotations, labels=labels, shape_type=shape_type,
                        database = 'User Provided', train_ratio=self.train_split, test_ratio=self.test_split, valid_ratio=self.validation_split)

        print('Annotations saved to results directory in the current working directory')

