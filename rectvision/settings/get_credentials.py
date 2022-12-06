import getpass
import requests
import jwt
import os
from dotenv import load_dotenv
from json import loads

class GetCredentials():
    def __init__(self, token):
        load_dotenv()
        self.token = token
        self.user_id, self.project_id, self.user_token = self.get_creds()

    def get_creds(self):
        key = os.getenv('JWT_KEY')
        creds = jwt.decode(self.token, key, algorithm = "HS256")['payload']
        user_id, project_id, user_token = creds['user_id'], creds['project_id'], creds['token']
        return user_id, project_id, user_token
   
    # def get_project_credential(self):
    #     base_url = 'https://backend.app.rectvision.com/api/v1/projects/'
    #     request_url = base_url + self.project_id
    #     headers={'Authorization':self.user_token}
    #     response = requests.get(request_url, headers=headers)
    #     if response.ok:
    #         print("Project {} credentials: \n".format(project_id))            
    #         response_details = loads(response.text)['data']['project']
    #         #extract project details from response_details
    #         return response_details           
    #     else:
    #         print("Something went wrong! Make sure project_id and token entered are correct!")
    #         print(response.text)