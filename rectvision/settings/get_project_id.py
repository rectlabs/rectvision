###this script returns all projects (names and respective project_id) for the user to choose from
import requests
from json import loads

class GetProjectIds():
    def __init__(self, user_id, token):
        self.user_id = user_id
        self.token = token

    def get_project_ids(self):
        base_url = 'http://164.92.64.23/api/v1/project/all/'
        request_url = base_url + self.user_id
        headers={'Authorization':self.token}
        response = requests.get(request_url, headers=headers)
        if response.ok:
            print("Projects and their respective ids: \n")            
            response_details = loads(response.text)['data']
            #extract project details from response_details
            
        else:
            print("Something went wrong! Make sure user_id and token entered are correct!")
            print(response.text)

# def get_user_id(user_id, token):
#     user = GetProjectIds(user_id, token)
#     user.get_project_ids()