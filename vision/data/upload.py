from s3API import s3API
class Upload():
    def __init__(self, user_id, project_id, input_filepath, filename):
        self.user_id = user_id
        self.project_id = project_id
        self.input_filepath = input_filepath
        self.filename = filename

        #upload the data
        self.upload()

    def upload(self):
        """This method uploads data to the aws storage
        of this specified project_id"""
        url = s3API.uploader(self.input_filepath, self.filename)['url'] + str(self.filename)
        print("presigned url of uploaded file: ", url)
        