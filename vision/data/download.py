from s3API import s3API

class Download():
    def __init__(self, user_id, project_id, filename, output_filepath):
        self.user_id = user_id
        self.project_id = project_id
        self.filename = filename #name of file stored in aws bucket
        self.output_filepath = output_filepath #name to be used to save downloaded file

        #download the data
        self.download()

    def download(self):
        """This method downloads data from the aws storage
        of this specified project_id"""
         
        s3API.downloader(self.filename, self.output_filepath)
        print('{} downloaded successfully and saved to {}'.format(self.filename, self.output_filepath))