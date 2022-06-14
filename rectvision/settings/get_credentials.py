import getpass
import requests
from json import loads

class GetCredentials():
    def __init__(self, username = input('Enter Your Email: '), password = getpass.getpass('Enter Your Password: ')):
        self.username = username
        self.password = password

    def get_user_id(self):
        request_url = 'http://164.92.64.23/api/v1/auth/login'
        data={'email':self.username, 'password':self.password}
        response = requests.post(request_url, data=data)
        if response.ok:
            print("Successful login!")            
            response_details = loads(response.text)['data']
            user_id = response_details["user"]["_id"]
            token = response_details["token"]
            print('Use this user_id and token pair to validate this session: {}, {}'.format(user_id, token))
        else:
            print("Something went wrong! Make sure email and password entered are correct!")
            print(response.text)

    def get_project_ids(self):
        base_url = 'http://164.92.64.23/api/v1/project/all/'
        request_url = base_url + self.user_id
        headers={'Authorization':self.token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Projects and their respective ids: \n")            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            for project in response_details:
                project_name = project['name']
                project_id = project['_id']
                print('Project Name: {}    Project ID: {}')            
        else:
            print("Something went wrong! Make sure user_id and token entered are correct!")
            print(response.text)

    def get_project_credential(self):
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + self.project_id
        headers={'Authorization':self.token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Project {} credentials: \n".format(self.project_id))            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            print(response_details)            
        else:
            print("Something went wrong! Make sure project_id and token entered are correct!")
            print(response.text)