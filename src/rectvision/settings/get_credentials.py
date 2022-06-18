import getpass
import requests
from json import loads

class GetCredentials():
    def __init__(self, username = input('Enter Your Email: '), password = getpass.getpass('Enter Your Password: ')):
        self.username = username
        self.password = password

        #login
        request_url = 'http://164.92.64.23/api/v1/auth/login'
        data={'email':self.username, 'password':self.password}
        self.login_response = requests.post(request_url, data=data)

        while not self.login_response.ok:
            print("Something went wrong! Make sure email and password entered are correct!")
            self.username = input('Enter Your Email: ')
            self.password = getpass.getpass('Enter Your Password: ')
            self.login_response = requests.post(request_url, data=data)
        print("Successful login!") 

    def get_user_id(self):                           
        response_details = loads(self.login_response.text)['data']
        user_id = response_details["user"]["_id"]
        token = response_details["token"]
        print('Use this user_id and token pair to validate this session: {}, {}'.format(user_id, token))
        return user_id, token        

    def get_project_ids(self, user_id, token):
        base_url = 'http://164.92.64.23/api/v1/project/all/'
        request_url = base_url + user_id
        headers={'Authorization':token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Projects and their respective ids: \n")            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            for project in response_details:
                project_name = project['name']
                project_id = project['_id']
                print('Project Name: {}    Project ID: {}'.format(project_name, project_id))            
        else:
            print("Something went wrong! Make sure user_id and token entered are correct!")
            print(response.text)

    def get_project_credential(self, project_id, token):
        base_url = 'http://164.92.64.23/api/v1/project/'
        request_url = base_url + project_id
        headers={'Authorization':token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Project {} credentials: \n".format(project_id))            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            return response_details           
        else:
            print("Something went wrong! Make sure project_id and token entered are correct!")
            print(response.text)