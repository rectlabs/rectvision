###this script returns all credentials linked to a particular project
import requests
from json import loads

class get_project_id():
    def __init__(self, project_id, token):
        self.project_id = project_id
        self.token = token

    def get_project_ids(self):
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + self.project_id
        headers={'Authorization':self.token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Project {} credentials: \n".format(self.project_id))            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            
        else:
            print("Something went wrong! Make sure project_id and token entered are correct!")
            print(response.text)