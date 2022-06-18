#this module uploads user annotation folder (test, validation and train) as a zip file
# from .s3API import s3API
from zipfile import ZipFile
import requests
from json import loads
# class Upload():
#     def __init__(self, user_id, project_id, input_filepath, filename):
#         self.user_id = user_id
#         self.project_id = project_id
#         self.input_filepath = input_filepath
#         self.filename = filename

#         #upload the data
#         self.upload()

#     def upload(self):
#         """This method uploads data to the aws storage
#         of this specified project_id"""
#         url = s3API.uploader(self.input_filepath, self.filename)['url'] + str(self.filename)
#         print("presigned url of uploaded file: ", url)

class Upload():
    def __init__(self, user_id, project_id, user_token, train_path, test_path, validation_path, label_to_id_path):
        self.user_id = user_id
        self.project_id = project_id
        self.user_token = user_token
        self.ann_file_paths = [train_path, test_path, validation_path, label_to_id_path]

    def compress_annotations(self, zip_path):
        #write files in ann_file_paths to a zipfile
        with ZipFile(zip_path, 'w') as zip:
            for file in self.ann_file_paths:
                zip.write(file)
        return zip_path

    def upload(self):
        #this method zips, uploads zipped annotation file to s3 and updates db url
        #zip projects directory
        zip_path = self.compress_annotations(zip_path='annotations.zip')
        #post annotation to endpoint   
        url = 'http://164.92.64.23/api/v1/upload/project/export'
        data={'userId':self.user_id, 'projectId':self.project_id, 'fileType':'files'}
        file = {'file': ('annotation', open(zip_path, 'rb'))}        
        headers={'Authorization':self.user_token}
        response = requests.post(url, files=file, data=data, headers=headers)
        if response.ok:
            print("Upload completed successfully!")
            response_details = loads(response.text)['data']
            #extract url from response_details
            url = response_details['imagePath']
            print('Uploaded successfully to {}'.format(url))
            return url
            
        else:
            print("Something went wrong!")
            print(response.text)



        