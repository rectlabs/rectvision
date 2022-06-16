# from .s3API import s3API
import requests
from json import loads, dump

# class Download():
#     def __init__(self, user_id, project_id, filename, output_filepath):
#         self.user_id = user_id
#         self.project_id = project_id
#         self.filename = filename #name of file stored in aws bucket
#         self.output_filepath = output_filepath #name to be used to save downloaded file

#         #download the data
#         self.download()

#     def download(self):
#         """This method downloads data from the aws storage
#         of this specified project_id"""
         
#         s3API.downloader(self.filename, self.output_filepath)
#         print('{} downloaded successfully and saved to {}'.format(self.filename, self.output_filepath))

class Download():
    def __init__(self, project_id, user_token):
        self.project_id = project_id
        self.user_token = user_token
        
    def download(self):
        #get project details from endpoint
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + self.project_id
        headers={'Authorization': self.user_token}
        response = requests.get(request_url, headers=headers)
        if response.ok:            
            current_project = loads(response.text)['data']
            annotations = current_project['annotations']
            with open('annotations.json', 'w') as f:
                dump(annotations, f)
            labels = current_project['labels']    
            with open('labels.txt', 'w') as f:
                dump(labels, f)
            print('Annotations saved to annotations.json and labels saved to labels.txt')
            
        else:
            print("Something went wrong! Make sure project_id and token entered are correct!")
            print(response.text)

