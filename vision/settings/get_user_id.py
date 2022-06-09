###this script takes in user id and password, and returns token
###to be used in the other modules (data download, model training)
import getpass
import requests
from json import loads

class get_user_id():
    def __init__(self):
        self.login()

    def get_credentials(self):
        username = input('Enter Your Email: ')
        password = getpass.getpass('Enter Your Password: ')

        return username, password

    def login(self):
        username, password = self.get_credentials()
        request_url = 'http://164.92.64.23/api/v1/auth/login'
        data={'email':username, 'password':password}
        response = requests.post(request_url, data=data)
        if response.ok:
            print("Successful login!")            
            response_details = loads(response.text)['data']
            user_id = response_details["user"]["_id"]
            token = response_details["token"]
            print('Use this user_id and token pair to validate this session: {}'.format(user_id, token))
        else:
            print("Something went wrong! Make sure email and password entered are correct!")
            print(response.text)
        
        